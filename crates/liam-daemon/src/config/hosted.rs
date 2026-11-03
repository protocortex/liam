// SPDX-License-Identifier: Apache-2.0
//! The `[hosted]` section: calls to hosted model providers. Off by default.
//! Keys never live here; they come from a key file or the Keychain.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 60;
/// Four MiB, far above a model answer and far below a memory problem.
const DEFAULT_MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct HostedConfig {
    pub enabled: bool,
    /// The only hosts a request may go to, each written as the URL shows it:
    /// lowercase, no scheme, no port, no path.
    pub allowed_hosts: Vec<String>,
    /// How long one request may take, answer included.
    pub request_timeout_secs: u64,
    /// The most bytes of a response that are read.
    pub max_response_bytes: usize,
}

impl Default for HostedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_hosts: Vec::new(),
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
        }
    }
}

impl HostedConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.request_timeout_secs > 0,
            "hosted.request_timeout_secs must be at least 1"
        );
        anyhow::ensure!(
            self.max_response_bytes > 0,
            "hosted.max_response_bytes must be at least 1"
        );
        anyhow::ensure!(
            !self.enabled || !self.allowed_hosts.is_empty(),
            "hosted.allowed_hosts must name at least one host when hosted.enabled is true"
        );
        for host in &self.allowed_hosts {
            anyhow::ensure!(
                is_bare_host(host),
                "hosted.allowed_hosts entry {host:?} must be a lowercase host name with no \
                 scheme, port, path or user"
            );
        }
        Ok(())
    }
}

fn is_bare_host(host: &str) -> bool {
    !host.is_empty()
        && host
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn load(toml: &str) -> anyhow::Result<Config> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liam.toml");
        std::fs::write(&path, toml).unwrap();
        Config::load(&path)
    }

    #[test]
    fn a_config_without_the_section_calls_no_provider() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.hosted, HostedConfig::default());
        assert!(!config.hosted.enabled);
    }

    #[test]
    fn enabling_hosted_calls_needs_a_host_to_allow() {
        // Act
        let empty = load("[hosted]\nenabled = true\n").unwrap_err();
        let listed =
            load("[hosted]\nenabled = true\nallowed_hosts = [\"api.anthropic.com\"]\n").unwrap();

        // Assert
        assert!(format!("{empty:#}").contains("hosted.allowed_hosts"));
        assert_eq!(listed.hosted.allowed_hosts, ["api.anthropic.com"]);
    }

    #[test]
    fn a_host_written_as_a_url_or_in_capitals_is_refused() {
        for host in [
            "https://api.anthropic.com",
            "API.anthropic.com",
            "api.anthropic.com:443",
            "api.anthropic.com/v1",
            "evil.com@api.anthropic.com",
            "",
        ] {
            // Act
            let error = load(&format!("[hosted]\nallowed_hosts = [{host:?}]\n")).unwrap_err();

            // Assert
            assert!(
                format!("{error:#}").contains("hosted.allowed_hosts"),
                "{host}"
            );
        }
    }

    #[test]
    fn the_limits_are_checked_and_unknown_keys_fail() {
        // Act
        let no_timeout = load("[hosted]\nrequest_timeout_secs = 0\n").unwrap_err();
        let no_size = load("[hosted]\nmax_response_bytes = 0\n").unwrap_err();
        let unknown = load("[hosted]\napi_key = \"secret\"\n").unwrap_err();

        // Assert
        assert!(format!("{no_timeout:#}").contains("hosted.request_timeout_secs"));
        assert!(format!("{no_size:#}").contains("hosted.max_response_bytes"));
        assert!(format!("{unknown:#}").contains("api_key"));
    }
}
