use anyhow::{Context, Result, bail};
use semver::Version;
use serde::Deserialize;
use std::{
    path::Path,
    process::Command,
    sync::{Arc, Mutex},
    thread,
};

#[derive(Clone, Debug, Default)]
pub struct State {
    pub message: Option<String>,
    pub available: Option<(String, String)>,
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
}

pub fn repository(root: &Path) -> Option<String> {
    std::env::var("BEESWAX_GITHUB_REPOSITORY")
        .ok()
        .and_then(|value| normalize_repository(&value))
        .or_else(|| option_env!("BEESWAX_GITHUB_REPOSITORY").and_then(normalize_repository))
        .or_else(|| remote_repository(root))
}

pub fn start(repository: Option<String>, state: Arc<Mutex<State>>) {
    let Some(repository) = repository else {
        return;
    };
    state.lock().unwrap().message = Some("Checking for updates…".to_owned());
    thread::spawn(move || {
        let result = check_latest(&repository);
        let mut current = state.lock().unwrap();
        match result {
            Ok(Some((version, url))) => {
                current.message = Some(format!("Beeswax {version} is available"));
                current.available = Some((version, url));
            }
            Ok(None) => {
                current.message = Some(format!(
                    "Beeswax {} is up to date",
                    env!("CARGO_PKG_VERSION")
                ));
            }
            Err(error) => current.message = Some(format!("Update check failed: {error:#}")),
        }
    });
}

fn check_latest(repository: &str) -> Result<Option<(String, String)>> {
    let url = format!("https://api.github.com/repos/{repository}/releases/latest");
    let output = Command::new(curl_program())
        .args([
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "10",
            "--max-time",
            "20",
            "--user-agent",
            concat!("Beeswax/", env!("CARGO_PKG_VERSION")),
            "--header",
            "Accept: application/vnd.github+json",
        ])
        .arg(url)
        .output()
        .context("Could not start curl to check GitHub releases")?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        bail!(
            "GitHub release request failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        );
    }

    let release: Release = serde_json::from_slice(&output.stdout)
        .context("GitHub returned an invalid release response")?;
    let version = release
        .tag_name
        .strip_prefix('v')
        .or_else(|| release.tag_name.strip_prefix('V'))
        .unwrap_or(&release.tag_name);
    let latest = Version::parse(version).with_context(|| {
        format!(
            "Release tag is not a semantic version: {}",
            release.tag_name
        )
    })?;
    let current = Version::parse(env!("CARGO_PKG_VERSION"))?;
    if latest > current {
        let expected_prefix = format!("https://github.com/{repository}/releases/tag/");
        if !release.html_url.starts_with(&expected_prefix) {
            bail!("GitHub returned an unexpected release URL");
        }
        Ok(Some((latest.to_string(), release.html_url)))
    } else {
        Ok(None)
    }
}

fn normalize_repository(value: &str) -> Option<String> {
    let value = value.trim();
    let repository = value
        .strip_prefix("https://github.com/")
        .or_else(|| value.strip_prefix("git@github.com:"))
        .or_else(|| value.strip_prefix("ssh://git@github.com/"))
        .unwrap_or(value)
        .trim_end_matches('/');
    let repository = repository.strip_suffix(".git").unwrap_or(repository);
    let mut parts = repository.split('/');
    let (Some(owner), Some(repo)) = (parts.next(), parts.next()) else {
        return None;
    };
    if parts.next().is_some()
        || owner.is_empty()
        || repo.is_empty()
        || !owner.chars().all(valid_repo_char)
        || !repo.chars().all(valid_repo_char)
    {
        return None;
    }
    Some(format!("{owner}/{repo}"))
}

fn valid_repo_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
}

fn remote_repository(root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["-C", root.to_str()?, "config", "--get", "remote.origin.url"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .and_then(|value| normalize_repository(&value))
}

pub fn open_release(url: &str) {
    if url.starts_with("https://github.com/") {
        let _ = Command::new("explorer.exe").arg(url).spawn();
    }
}

fn curl_program() -> &'static str {
    if cfg!(windows) { "curl.exe" } else { "curl" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_urls_normalize_to_owner_and_repo() {
        assert_eq!(
            normalize_repository("https://github.com/example/beeswax.git"),
            Some("example/beeswax".to_owned())
        );
        assert_eq!(
            normalize_repository("git@github.com:example/beeswax.git"),
            Some("example/beeswax".to_owned())
        );
        assert_eq!(normalize_repository("https://example.com/other/repo"), None);
    }

    #[test]
    fn semantic_version_orders_prereleases_and_releases() {
        let stable = Version::parse("1.2.0").unwrap();
        let preview = Version::parse("1.2.0-rc.1").unwrap();
        assert!(stable > preview);
        assert!(Version::parse("1.3.0").unwrap() > stable);
    }
}
