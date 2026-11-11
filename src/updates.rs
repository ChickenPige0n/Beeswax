use anyhow::{Context, Result, bail};
use beeswax::i18n::{Language, language, tr};
use semver::Version;
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
    state.lock().unwrap().message = Some(tr("Checking for updates…", "正在检查更新…").to_owned());
    thread::spawn(move || {
        let result = check_latest(&repository);
        let mut current = state.lock().unwrap();
        match result {
            Ok(Some((version, url))) => {
                current.message = Some(match language() {
                    Language::En => format!("Beeswax {version} is available"),
                    Language::ZhCn => format!("发现新版本 Beeswax {version}"),
                });
                current.available = Some((version, url));
            }
            Ok(None) => {
                let ver = env!("CARGO_PKG_VERSION");
                current.message = Some(match language() {
                    Language::En => format!("Beeswax {ver} is up to date"),
                    Language::ZhCn => format!("当前已是最新版本 Beeswax {ver}"),
                });
            }
            Err(error) => {
                current.message = Some(match language() {
                    Language::En => format!("Update check failed: {error:#}"),
                    Language::ZhCn => format!("检查更新失败：{error:#}"),
                })
            }
        }
    });
}

/// Ask github.com which release is current instead of using the REST API.
/// The releases page redirects to /releases/tag/vX.Y.Z and is not subject to
/// the unauthenticated API's hourly rate limit (which shows up as HTTP 403).
fn check_latest(repository: &str) -> Result<Option<(String, String)>> {
    let url = format!("https://github.com/{repository}/releases/latest");
    let mut cmd = Command::new(curl_program());
    cmd.args([
        "--location",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "10",
        "--max-time",
        "20",
        "--user-agent",
        concat!("Beeswax/", env!("CARGO_PKG_VERSION")),
        "--header",
        "Accept: text/html",
        "--output",
        if cfg!(windows) { "NUL" } else { "/dev/null" },
        "--write-out",
        "%{http_code}\n%{url_effective}",
    ])
    .arg(url);
    hide_console(&mut cmd);
    let output = cmd.output().with_context(|| {
        tr(
            "Could not start curl to check GitHub releases",
            "无法启动 curl 检查 GitHub 更新",
        )
    })?;
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

    let response = String::from_utf8_lossy(&output.stdout);
    let mut lines = response.lines();
    let status = lines.next().unwrap_or_default().trim();
    let effective_url = lines.next().unwrap_or_default().trim();
    release_from_redirect(repository, status, effective_url)
}

fn release_from_redirect(
    repository: &str,
    status: &str,
    effective_url: &str,
) -> Result<Option<(String, String)>> {
    match status {
        "200" => {}
        "404" => return Ok(None),
        other => bail!("GitHub returned HTTP {other} for {repository}"),
    }
    let prefix = format!("https://github.com/{repository}/releases/tag/");
    let Some(tag) = effective_url.strip_prefix(&prefix) else {
        bail!("GitHub returned an unexpected release URL");
    };
    if tag.is_empty() || tag.contains('/') || tag.contains('?') {
        bail!("GitHub returned an unexpected release tag");
    }
    let version = tag
        .strip_prefix('v')
        .or_else(|| tag.strip_prefix('V'))
        .unwrap_or(tag);
    let latest = Version::parse(version)
        .with_context(|| format!("Release tag is not a semantic version: {tag}"))?;
    let current = Version::parse(env!("CARGO_PKG_VERSION"))?;
    if latest > current {
        Ok(Some((latest.to_string(), effective_url.to_owned())))
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
    let mut cmd = Command::new("git");
    cmd.args(["-C", root.to_str()?, "config", "--get", "remote.origin.url"]);
    hide_console(&mut cmd);
    let output = cmd.output().ok()?;
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

fn hide_console(#[allow(unused_variables)] cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

fn curl_program() -> &'static str {
    if cfg!(windows) { "curl.exe" } else { "curl" }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASE: &str = "https://github.com/example/beeswax/releases/tag/v9.9.9";

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
    fn github_release_redirect_is_parsed() {
        let (version, url) = release_from_redirect("example/beeswax", "200", RELEASE)
            .unwrap()
            .unwrap();
        assert_eq!(version, "9.9.9");
        assert_eq!(url, RELEASE);
    }

    #[test]
    fn missing_or_older_releases_are_not_updates() {
        assert!(
            release_from_redirect(
                "example/beeswax",
                "404",
                "https://github.com/example/beeswax/releases"
            )
            .unwrap()
            .is_none()
        );
        assert!(
            release_from_redirect(
                "example/beeswax",
                "200",
                "https://github.com/example/beeswax/releases/tag/v0.0.1"
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn unexpected_redirects_and_statuses_are_errors() {
        assert!(
            release_from_redirect(
                "example/beeswax",
                "200",
                "https://github.com/example/other/releases/tag/v9.9.9"
            )
            .is_err()
        );
        assert!(
            release_from_redirect(
                "example/beeswax",
                "200",
                "https://github.com/example/beeswax/releases"
            )
            .is_err()
        );
        assert!(release_from_redirect("example/beeswax", "403", "").is_err());
    }

    #[test]
    fn semantic_version_orders_prereleases_and_releases() {
        let stable = Version::parse("1.2.0").unwrap();
        let preview = Version::parse("1.2.0-rc.1").unwrap();
        assert!(stable > preview);
        assert!(Version::parse("1.3.0").unwrap() > stable);
    }
}
