use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

const MINIMUM_MODEL_BYTES: u64 = 1_000_000;
const VOXLOCK_MODEL_REVISION: &str = "a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03";
const VOXLOCK_MODEL_URL: &str = "https://huggingface.co/arbazsiddiqui/voxlock/resolve";
const REQUIRED_MODELS: [(&str, &str, &str); 2] = [
    (
        "enroll.onnx",
        "onnx/enroll.onnx",
        "86645cba23f207def0b31a6e74a615d3b7a813b4fdf957c53b3af2905ba2ffd3",
    ),
    (
        "chunk_int8.onnx",
        "onnx/chunk_int8.onnx",
        "f0d2b0aaf95affe8894ea008d0105be4f5f5354f7244835534c31b1a7dd4699a",
    ),
];

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
    REQUIRED_MODELS.iter().all(|(name, _, _)| {
        folder
            .join(name)
            .metadata()
            .is_ok_and(|metadata| metadata.len() >= MINIMUM_MODEL_BYTES)
    })
}

pub fn start(folder: PathBuf, state: Arc<Mutex<State>>) {
    {
        let mut current = state.lock().unwrap();
        if current.active || required_models_present(&folder) {
            current.ready = required_models_present(&folder);
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
            state.lock().unwrap().message = Some(format!(
                "Downloading model {index}/{}: {file} · {:.1} MB",
                REQUIRED_MODELS.len(),
                bytes as f64 / 1_000_000.
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
    for (index, (name, file, checksum)) in REQUIRED_MODELS.iter().enumerate() {
        let destination = folder.join(name);
        if destination
            .metadata()
            .is_ok_and(|metadata| metadata.len() >= MINIMUM_MODEL_BYTES)
        {
            continue;
        }
        progress(index + 1, name, 0);
        let url = format!("{VOXLOCK_MODEL_URL}/{VOXLOCK_MODEL_REVISION}/{file}");
        download_file(&url, &destination, checksum, |bytes| {
            progress(index + 1, name, bytes)
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
        length >= MINIMUM_MODEL_BYTES,
        "Downloaded model file is unexpectedly small ({length} bytes)"
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
