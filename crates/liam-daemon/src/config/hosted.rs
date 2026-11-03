// SPDX-License-Identifier: Apache-2.0
//! The `[hosted]` section: calls to hosted model providers. Off by default.
//! Keys never live here; they come from a key file or the Keychain.

use std::collections::BTreeMap;

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

use super::pins::{ModelPin, PIN_PROVIDERS};
use super::pricing::{Price, Provider};

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
    /// What each pinned model costs.
    pub prices: PricesConfig,
    /// A provider's key file, an absolute path to a file only the owner can
    /// read. A provider without one is read from the Keychain, under the
    /// service `liam-<provider>`.
    pub key_files: BTreeMap<String, String>,
}

/// `[hosted.prices]`: a versioned price table, so a recorded cost can be tied
/// to the rates it was computed with.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct PricesConfig {
    /// Names this table, for example the month the rates were read.
    pub version: String,
    /// Rates by pinned model id, in millionths of a dollar per million tokens.
    pub models: BTreeMap<String, Price>,
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
            prices: PricesConfig::default(),
            key_files: BTreeMap::new(),
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
        for provider in self.key_files.keys() {
            anyhow::ensure!(
                PIN_PROVIDERS.contains(&provider.as_str()),
                "hosted.key_files.{provider} is not a known provider; use one of {}",
                PIN_PROVIDERS.join(", ")
            );
        }
        anyhow::ensure!(
            self.prices.models.is_empty() || !self.prices.version.trim().is_empty(),
            "hosted.prices.version must name the price table"
        );
        Ok(())
    }

    /// The pin of `provider`, if one is set.
    pub fn pin_for(&self, provider: Provider) -> Option<ModelPin> {
        let model_id = self.models.get(provider.name())?;
        ModelPin::new(provider.name(), model_id).ok()
    }

    /// The rates of a pinned model id.
    pub fn price_for(&self, model_id: &str) -> Option<Price> {
        self.prices.models.get(model_id).copied()
    }

    /// Checks that `provider`, chosen for `key`, can be called: hosted calls
    /// are on, the model is pinned, and its price is known.
    pub fn check_provider(&self, key: &str, provider: Provider) -> anyhow::Result<()> {
        if provider == Provider::Local {
            return Ok(());
        }
        let name = provider.name();
        anyhow::ensure!(self.enabled, "{key} = {name:?} needs hosted.enabled = true");
        let pin = self.pin_for(provider);
        anyhow::ensure!(
            pin.is_some(),
            "{key} = {name:?} needs hosted.models.{name} to be set"
        );
        let pinned = pin.map(|pin| pin.model_id).unwrap_or_default();
        anyhow::ensure!(
            self.price_for(&pinned).is_some(),
            "{key} = {name:?} needs hosted.prices.models.{pinned}"
        );
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

    const HOSTED_ANTHROPIC: &str =
        "[hosted]\nenabled = true\nallowed_hosts = [\"api.anthropic.com\"]\n\
        [hosted.models]\nanthropic = \"claude-sonnet-4-5-20250929\"\n";
    const ANTHROPIC_PRICE: &str = "[hosted.prices]\nversion = \"2026-11\"\n\
        [hosted.prices.models.\"claude-sonnet-4-5-20250929\"]\n\
        input_micro_per_mtok = 3000000\noutput_micro_per_mtok = 15000000\n";

    #[test]
    fn a_hosted_provider_for_a_stage_needs_hosted_on_a_pin_a_price_and_metrics() {
        // Arrange
        let stage = "[extraction]\nprovider = \"anthropic\"\n";
        let off = format!("{stage}[hosted]\nenabled = false\n");
        let unpinned =
            format!("{stage}[hosted]\nenabled = true\nallowed_hosts = [\"api.anthropic.com\"]\n");
        let unpriced = format!("{stage}{HOSTED_ANTHROPIC}");
        let blind =
            format!("{stage}{HOSTED_ANTHROPIC}{ANTHROPIC_PRICE}[metrics]\nenabled = false\n");

        // Act
        let errors: Vec<String> = [off, unpinned, unpriced, blind]
            .iter()
            .map(|toml| format!("{:#}", load(toml).unwrap_err()))
            .collect();
        let ready = load(&format!("{stage}{HOSTED_ANTHROPIC}{ANTHROPIC_PRICE}")).unwrap();

        // Assert
        assert!(errors[0].contains("extraction.provider") && errors[0].contains("hosted.enabled"));
        assert!(errors[1].contains("hosted.models.anthropic"));
        assert!(errors[2].contains("hosted.prices.models.claude-sonnet-4-5-20250929"));
        assert!(errors[3].contains("metrics.enabled"));
        assert_eq!(ready.extraction.provider, Provider::Anthropic);
        assert_eq!(ready.summary.provider, Provider::Local);
    }

    #[test]
    fn the_summary_stage_is_checked_the_same_way() {
        // Act
        let error = load("[summary]\nprovider = \"openai\"\n").unwrap_err();

        // Assert
        assert!(format!("{error:#}").contains("summary.provider"));
    }

    #[test]
    fn a_price_table_needs_a_version_and_a_key_file_needs_a_known_provider() {
        // Act
        let unversioned =
            load("[hosted.prices.models.\"gpt-4o-2024-08-06\"]\ninput_micro_per_mtok = 1\n")
                .unwrap_err();
        let stranger = load("[hosted.key_files]\nmistral = \"/keys/m\"\n").unwrap_err();
        let known = load("[hosted.key_files]\nopenai = \"/keys/o\"\n").unwrap();

        // Assert
        assert!(format!("{unversioned:#}").contains("hosted.prices.version"));
        assert!(format!("{stranger:#}").contains("hosted.key_files.mistral"));
        assert_eq!(known.hosted.key_files["openai"], "/keys/o");
    }
}
