// SPDX-License-Identifier: Apache-2.0
//! The `[hosted]` section: calls to hosted model providers. Off by default.
//! Keys never live here; they come from a key file or the Keychain.

use std::collections::BTreeMap;

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

use super::pins::{ModelPin, PIN_PROVIDERS};

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
    /// What may leave the machine. Nothing does by default.
    pub privacy: PrivacyConfig,
    /// The dated model id each provider is pinned to.
    pub models: BTreeMap<String, String>,
}

/// `[hosted.privacy]`: the gate every item passes before it goes into a hosted
/// prompt.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct PrivacyConfig {
    /// Scopes whose items may be sent. A scope covers its descendants:
    /// `work` covers `work/liam`. The unscoped group is the empty string.
    pub allowed_scopes: Vec<String>,
    /// Producers whose items are never sent, in any scope.
    pub local_only_sources: Vec<String>,
    /// The region each provider processes requests in, as the operator
    /// declares it.
    pub provider_regions: BTreeMap<String, String>,
    /// For a scope (and its descendants), the regions it may be sent to.
    pub attested_regions: BTreeMap<String, Vec<String>>,
}

impl Default for HostedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_hosts: Vec::new(),
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            privacy: PrivacyConfig::default(),
            models: BTreeMap::new(),
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
        self.privacy.validate()?;
        for (provider, model_id) in &self.models {
            ModelPin::new(provider, model_id)?;
        }
        Ok(())
    }

    /// The pins, in provider order. `validate` has checked each one.
    pub fn pins(&self) -> Vec<ModelPin> {
        self.models
            .iter()
            .filter_map(|(provider, model_id)| ModelPin::new(provider, model_id).ok())
            .collect()
    }
}

impl PrivacyConfig {
    fn validate(&self) -> anyhow::Result<()> {
        for provider in self.provider_regions.keys() {
            anyhow::ensure!(
                PIN_PROVIDERS.contains(&provider.as_str()),
                "hosted.privacy.provider_regions.{provider} is not a known provider; use one of {}",
                PIN_PROVIDERS.join(", ")
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
    fn an_alias_in_the_models_table_fails_the_load_and_a_dated_id_loads() {
        // Act
        let alias = load("[hosted.models]\nopenai = \"gpt-4o\"\n").unwrap_err();
        let unknown = load("[hosted.models]\nmistral = \"m-20250101\"\n").unwrap_err();
        let dated = load("[hosted.models]\nopenai = \"gpt-4o-2024-08-06\"\n").unwrap();

        // Assert
        assert!(format!("{alias:#}").contains("alias"));
        assert!(format!("{unknown:#}").contains("not a known provider"));
        assert_eq!(dated.hosted.pins().len(), 1);
    }

    #[test]
    fn the_privacy_table_loads_and_names_only_known_providers() {
        // Act
        let loaded = load(
            "[hosted.privacy]\nallowed_scopes = [\"work\"]\nlocal_only_sources = [\"client:diary\"]\n\
             [hosted.privacy.provider_regions]\nanthropic = \"us\"\n\
             [hosted.privacy.attested_regions]\nwork = [\"us\"]\n",
        )
        .unwrap();
        let unknown = load("[hosted.privacy.provider_regions]\nmistral = \"eu\"\n").unwrap_err();

        // Assert
        assert_eq!(loaded.hosted.privacy.allowed_scopes, ["work"]);
        assert_eq!(loaded.hosted.privacy.attested_regions["work"], ["us"]);
        assert!(format!("{unknown:#}").contains("provider_regions.mistral"));
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
