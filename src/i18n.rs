use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Language {
    #[default]
    En,
    ZhCn,
}

static CURRENT_LANG: AtomicU8 = AtomicU8::new(0);

impl Language {
    pub fn from_code(code: &str) -> Option<Self> {
        let lower = code.trim().to_ascii_lowercase();
        if lower == "en" || lower.starts_with("en-") || lower.starts_with("en_") {
            Some(Self::En)
        } else if lower == "zh"
            || lower == "cn"
            || lower.starts_with("zh-")
            || lower.starts_with("zh_")
        {
            Some(Self::ZhCn)
        } else {
            None
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::ZhCn => "zh-CN",
        }
    }

    pub fn detect() -> Self {
        for var in ["BEESWAX_LANG", "LC_ALL", "LC_MESSAGES", "LANG"] {
            if let Ok(value) = std::env::var(var)
                && let Some(lang) = Self::from_code(&value)
            {
                return lang;
            }
        }
        detect_system_language().unwrap_or(Self::En)
    }

    pub fn is_zh(self) -> bool {
        matches!(self, Self::ZhCn)
    }
}

pub fn set_language(lang: Language) {
    let raw = match lang {
        Language::En => 0,
        Language::ZhCn => 1,
    };
    CURRENT_LANG.store(raw, Ordering::Relaxed);
}

pub fn language() -> Language {
    match CURRENT_LANG.load(Ordering::Relaxed) {
        1 => Language::ZhCn,
        _ => Language::En,
    }
}

pub fn pick<'a>(lang: Language, en: &'a str, zh: &'a str) -> &'a str {
    match lang {
        Language::En => en,
        Language::ZhCn => zh,
    }
}

pub fn tr<'a>(en: &'a str, zh: &'a str) -> &'a str {
    pick(language(), en, zh)
}

#[cfg(windows)]
fn detect_system_language() -> Option<Language> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetUserDefaultUILanguage() -> u16;
    }
    // LANG_CHINESE = 0x04 (primary language mask 0x03ff).
    let lang_id = unsafe { GetUserDefaultUILanguage() };
    if (lang_id & 0x03ff) == 0x04 {
        Some(Language::ZhCn)
    } else {
        Some(Language::En)
    }
}

#[cfg(not(windows))]
fn detect_system_language() -> Option<Language> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_english_and_chinese_codes() {
        assert_eq!(Language::from_code("en"), Some(Language::En));
        assert_eq!(Language::from_code("en-US"), Some(Language::En));
        assert_eq!(Language::from_code("zh"), Some(Language::ZhCn));
        assert_eq!(Language::from_code("zh-CN"), Some(Language::ZhCn));
        assert_eq!(
            Language::from_code("zh_Hans_CN.UTF-8"),
            Some(Language::ZhCn)
        );
        assert_eq!(Language::from_code("cn"), Some(Language::ZhCn));
        assert_eq!(Language::from_code("fr"), None);
    }

    #[test]
    fn round_trips_atomic_language_state() {
        set_language(Language::ZhCn);
        assert_eq!(language(), Language::ZhCn);
        assert_eq!(tr("Stop", "停止"), "停止");
        set_language(Language::En);
        assert_eq!(language(), Language::En);
        assert_eq!(tr("Stop", "停止"), "Stop");
    }
}
