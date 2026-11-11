use anyhow::{Context, Result, ensure};
use beeswax::i18n::{Language, tr};
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
        tr("Unsupported enrollment profile", "不支持的声纹配置格式")
    );
    beeswax::dsp::enrollment(&p.samples)
}
pub fn save(dir: &Path, samples: &[f32]) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| {
        tr(
            "Cannot create local voice profile directory",
            "无法创建本地声纹目录",
        )
    })?;
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
#[derive(Default, Serialize, Deserialize)]
struct StoredSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

fn read_settings(dir: &Path) -> StoredSettings {
    std::fs::read_to_string(dir.join("settings.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn write_settings(dir: &Path, settings: &StoredSettings) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    if let Ok(bytes) = serde_json::to_vec(settings) {
        let _ = std::fs::write(dir.join("settings.json"), bytes);
    }
}

/// The backend chosen in the last session, if any.
pub fn load_backend(dir: &Path) -> Option<beeswax::Backend> {
    match read_settings(dir).backend.as_deref()? {
        "cuda" => Some(beeswax::Backend::Cuda),
        "cpu" => Some(beeswax::Backend::Cpu),
        _ => None,
    }
}

/// Remember the backend so a GPU machine does not have to be reconfigured each launch.
pub fn save_backend(dir: &Path, backend: beeswax::Backend) {
    let mut settings = read_settings(dir);
    settings.backend = Some(
        match backend {
            beeswax::Backend::Cpu => "cpu",
            beeswax::Backend::Cuda => "cuda",
        }
        .to_owned(),
    );
    write_settings(dir, &settings);
}

/// The UI language chosen in the last session, if any.
pub fn load_language(dir: &Path) -> Option<Language> {
    Language::from_code(read_settings(dir).language.as_deref()?)
}

/// Persist the chosen UI language alongside other local settings.
pub fn save_language(dir: &Path, language: Language) {
    let mut settings = read_settings(dir);
    settings.language = Some(language.code().to_owned());
    write_settings(dir, &settings);
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

    #[test]
    fn settings_preserve_backend_and_language_together() {
        let dir = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!("settings-test-{}", std::process::id()));
        save_backend(&dir, beeswax::Backend::Cuda);
        save_language(&dir, Language::ZhCn);
        assert_eq!(load_backend(&dir), Some(beeswax::Backend::Cuda));
        assert_eq!(load_language(&dir), Some(Language::ZhCn));
        save_backend(&dir, beeswax::Backend::Cpu);
        assert_eq!(load_backend(&dir), Some(beeswax::Backend::Cpu));
        assert_eq!(load_language(&dir), Some(Language::ZhCn));
        let _ = std::fs::remove_file(dir.join("settings.json"));
        let _ = std::fs::remove_dir(dir);
    }
}
