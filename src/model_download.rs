use anyhow::{Context, Result, bail, ensure};
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
        current.message = Some("Preparing required voice models…".to_owned());
    }

    thread::spawn(move || {
        let result = download_required(&folder, |index, file, bytes| {
            let models = &manifest().models;
            let total = models
                .iter()
                .find(|model| model.name.as_str() == file)
                .map_or(0, |model| model.bytes);
            state.lock().unwrap().message = Some(format!(
                "Downloading voice model {index}/{}: {file} · {:.1} / {:.1} MB",
                models.len(),
                bytes as f64 / 1_000_000.,
                total as f64 / 1_000_000.
            ));
        });
        let mut current = state.lock().unwrap();
        current.active = false;
        current.ready = result.is_ok() && required_models_present(&folder);
        current.error = result.err().map(|error| format!("{error:#}"));
        current.message = if current.ready {
            Some("Required voice models are ready.".to_owned())
        } else {
            None
        };
    });
}

fn download_required(folder: &Path, mut progress: impl FnMut(usize, &str, u64)) -> Result<()> {
    std::fs::create_dir_all(folder).context("Could not create the models directory")?;
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
        "The downloaded VoxLock files are incomplete. Retry the download."
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
    let mut child = Command::new(curl_program())
        .args([
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
        .stderr(Stdio::piped())
        .spawn()
        .context("Could not start curl to download the voice models")?;

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
            "Model download failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        );
    }

    let length = partial
        .metadata()
        .context("Downloaded model file is missing")?
        .len();
    ensure!(
        length == expected_bytes,
        "Downloaded model file has the wrong size ({length} bytes, expected {expected_bytes})"
    );
    let actual_sha256 = sha256_file(&partial)?;
    if actual_sha256 != expected_sha256 {
        let _ = std::fs::remove_file(&partial);
        bail!("SHA-256 verification failed for {}", destination.display());
    }
    if destination.exists() {
        std::fs::remove_file(destination).context("Could not replace the incomplete model file")?;
    }
    std::fs::rename(&partial, destination).context("Could not install the downloaded model")?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).context("Could not read the downloaded model")?;
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
