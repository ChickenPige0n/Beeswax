use anyhow::{Context, Result, bail, ensure};
use beeswax::i18n::{Language, language, tr};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::Duration,
};

/// Models pinned in models.json at the repository root. The release workflow
/// vendors the same files, so the download path is only used by source builds
/// or when the models folder was removed.
const MODEL_MANIFEST: &str = include_str!("../models.json");

#[derive(Deserialize)]
struct Manifest {
    repository: String,
    revision: String,
    models: Vec<ManifestModel>,
}

#[derive(Deserialize)]
struct ManifestModel {
    name: String,
    path: String,
    bytes: u64,
    sha256: String,
}

fn manifest() -> &'static Manifest {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    MANIFEST
        .get_or_init(|| serde_json::from_str(MODEL_MANIFEST).expect("models.json is valid JSON"))
}

#[derive(Clone, Debug, Default)]
pub struct State {
    pub ready: bool,
    pub active: bool,
    pub message: Option<String>,
    pub error: Option<String>,
}

impl State {
    pub fn for_directory(folder: &Path) -> Self {
        Self {
            ready: required_models_present(folder),
            active: false,
            message: None,
            error: None,
        }
    }
}

pub fn required_models_present(folder: &Path) -> bool {
    manifest().models.iter().all(|model| {
        folder
            .join(&model.name)
            .metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == model.bytes)
    })
}

pub fn start(folder: PathBuf, state: Arc<Mutex<State>>) {
    {
        let mut current = state.lock().unwrap();
        let ready = required_models_present(&folder);
        if current.active || ready {
            current.ready = ready;
            current.error = None;
            current.message = None;
            return;
        }
        current.active = true;
        current.ready = false;
        current.error = None;
        current.message = Some(
            tr(
                "Preparing required voice models…",
                "正在准备所需的语音模型…",
            )
            .to_owned(),
        );
    }

    thread::spawn(move || {
        let result = download_required(&folder, |index, file, bytes| {
            let models = &manifest().models;
            let total = models
                .iter()
                .find(|model| model.name.as_str() == file)
                .map_or(0, |model| model.bytes);
            let mb = bytes as f64 / 1_000_000.;
            let total_mb = total as f64 / 1_000_000.;
            let count = models.len();
            state.lock().unwrap().message = Some(match language() {
                Language::En => {
                    format!(
                        "Downloading voice model {index}/{count}: {file} ({mb:.1} / {total_mb:.1} MB)"
                    )
                }
                Language::ZhCn => {
                    format!(
                        "正在下载语音模型 {index}/{count}：{file}（{mb:.1} / {total_mb:.1} MB）"
                    )
                }
            });
        });
        let mut current = state.lock().unwrap();
        current.active = false;
        current.ready = result.is_ok() && required_models_present(&folder);
        current.error = result.err().map(|error| format!("{error:#}"));
        current.message = if current.ready {
            Some(tr("Required voice models are ready.", "所需的语音模型已就绪。").to_owned())
        } else {
            None
        };
    });
}

fn download_required(folder: &Path, mut progress: impl FnMut(usize, &str, u64)) -> Result<()> {
    std::fs::create_dir_all(folder).with_context(|| {
        tr(
            "Could not create the models directory",
            "无法创建 models 目录",
        )
    })?;
    let manifest = manifest();
    for (index, model) in manifest.models.iter().enumerate() {
        let destination = folder.join(&model.name);
        if destination
            .metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == model.bytes)
        {
            continue;
        }
        progress(index + 1, &model.name, 0);
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            manifest.repository, manifest.revision, model.path
        );
        download_file(&url, &destination, &model.sha256, model.bytes, |bytes| {
            progress(index + 1, &model.name, bytes)
        })?;
    }
    ensure!(
        required_models_present(folder),
        "{}",
        tr(
            "The downloaded VoxLock files are incomplete. Retry the download.",
            "下载的 VoxLock 模型文件不完整，请重试下载。"
        )
    );
    Ok(())
}

fn download_file(
    url: &str,
    destination: &Path,
    expected_sha256: &str,
    expected_bytes: u64,
    mut progress: impl FnMut(u64),
) -> Result<()> {
    let partial = destination.with_extension("onnx.download");
    let _ = std::fs::remove_file(&partial);
    let mut cmd = Command::new(curl_program());
    cmd.args([
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "2",
        "--connect-timeout",
        "20",
        "--output",
    ])
    .arg(&partial)
    .arg(url)
    .stdout(Stdio::null())
    .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().with_context(|| {
        tr(
            "Could not start curl to download the voice models",
            "无法启动 curl 下载语音模型",
        )
    })?;

    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        progress(partial.metadata().map_or(0, |metadata| metadata.len()));
        thread::sleep(Duration::from_millis(250));
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let _ = std::fs::remove_file(&partial);
        bail!(
            "{}",
            match (language(), detail.is_empty()) {
                (Language::En, true) => "Model download failed".to_owned(),
                (Language::En, false) => format!("Model download failed: {detail}"),
                (Language::ZhCn, true) => "模型下载失败".to_owned(),
                (Language::ZhCn, false) => format!("模型下载失败：{detail}"),
            }
        );
    }

    let length = partial
        .metadata()
        .with_context(|| tr("Downloaded model file is missing", "已下载的模型文件不存在"))?
        .len();
    ensure!(
        length == expected_bytes,
        "{}",
        match language() {
            Language::En => format!(
                "Downloaded model file has the wrong size ({length} bytes, expected {expected_bytes})"
            ),
            Language::ZhCn => {
                format!(
                    "已下载的模型文件大小不正确（实际 {length} 字节，预期 {expected_bytes} 字节）"
                )
            }
        }
    );
    let actual_sha256 = sha256_file(&partial)?;
    if actual_sha256 != expected_sha256 {
        let _ = std::fs::remove_file(&partial);
        let display = destination.display();
        bail!(
            "{}",
            match language() {
                Language::En => format!("SHA-256 verification failed for {display}"),
                Language::ZhCn => format!("{display} 的 SHA-256 校验失败"),
            }
        );
    }
    if destination.exists() {
        std::fs::remove_file(destination).with_context(|| {
            tr(
                "Could not replace the incomplete model file",
                "无法替换未完成的模型文件",
            )
        })?;
    }
    std::fs::rename(&partial, destination).with_context(|| {
        tr(
            "Could not install the downloaded model",
            "无法安装已下载的模型",
        )
    })?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| {
        tr(
            "Could not read the downloaded model",
            "无法读取已下载的模型文件",
        )
    })?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn curl_program() -> &'static str {
    if cfg!(windows) { "curl.exe" } else { "curl" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_pins_complete_models() {
        let manifest = manifest();
        assert!(!manifest.repository.is_empty());
        assert_eq!(manifest.revision.len(), 40);
        assert!(!manifest.models.is_empty());
        for model in &manifest.models {
            assert!(model.name.ends_with(".onnx"));
            assert!(model.path.starts_with("onnx/"));
            assert!(model.bytes > 1_000_000);
            assert_eq!(model.sha256.len(), 64);
            assert!(model.sha256.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}
