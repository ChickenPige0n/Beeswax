use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{io::Write, path::Path};

#[derive(Serialize, Deserialize)]
struct Profile {
    version: u32,
    sample_rate: u32,
    samples: Vec<f32>,
}
pub fn load(dir: &Path) -> Result<Vec<f32>> {
    let p: Profile = serde_json::from_slice(&std::fs::read(dir.join("voice.json"))?)?;
    ensure!(
        p.version == 1 && p.sample_rate == 16000 && p.samples.len() == 48000,
        "Unsupported enrollment profile"
    );
    beeswax::dsp::enrollment(&p.samples)
}
pub fn save(dir: &Path, samples: &[f32]) -> Result<()> {
    std::fs::create_dir_all(dir).context("Cannot create local voice profile directory")?;
    let bytes = serde_json::to_vec(&Profile {
        version: 1,
        sample_rate: 16000,
        samples: samples.to_vec(),
    })?;
    let tmp = dir.join("voice.json.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, dir.join("voice.json"))?;
    // Portable PCM16 copy so the enrollment clip can be listened to in any player.
    let mut file = std::fs::File::create(dir.join("enrollment.wav"))?;
    let length = (samples.len() * 2) as u32;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + length).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&16000u32.to_le_bytes())?;
    file.write_all(&32000u32.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&length.to_le_bytes())?;
    for &v in samples {
        file.write_all(&((v.clamp(-1., 1.) * 32767.) as i16).to_le_bytes())?;
    }
    Ok(())
}
/// The backend chosen in the last session, if any.
pub fn load_backend(dir: &Path) -> Option<beeswax::Backend> {
    let text = std::fs::read_to_string(dir.join("settings.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    match value.get("backend")?.as_str()? {
        "cuda" => Some(beeswax::Backend::Cuda),
        "cpu" => Some(beeswax::Backend::Cpu),
        _ => None,
    }
}

/// Remember the backend so a GPU machine does not have to be reconfigured each launch.
pub fn save_backend(dir: &Path, backend: beeswax::Backend) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let value = serde_json::json!({
        "backend": match backend {
            beeswax::Backend::Cpu => "cpu",
            beeswax::Backend::Cuda => "cuda",
        }
    });
    let _ = std::fs::write(dir.join("settings.json"), value.to_string());
}

pub fn forget(dir: &Path) -> Result<()> {
    for name in ["voice.json", "enrollment.wav", "voice.json.tmp"] {
        match std::fs::remove_file(dir.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn saved_voice_round_trips_and_forget_removes_audio() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("profile-test-{}", std::process::id()));
        let voice = vec![0.05; 48000];
        save(&dir, &voice).unwrap();
        assert_eq!(load(&dir).unwrap(), voice);
        assert_eq!(
            std::fs::metadata(dir.join("enrollment.wav")).unwrap().len(),
            96044
        );
        // Replacement is supported on Windows as well as first-time enrollment.
        save(&dir, &vec![0.1; 48000]).unwrap();
        assert_eq!(load(&dir).unwrap(), vec![0.1; 48000]);
        forget(&dir).unwrap();
        assert!(!dir.join("voice.json").exists());
        assert!(!dir.join("enrollment.wav").exists());
        forget(&dir).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
