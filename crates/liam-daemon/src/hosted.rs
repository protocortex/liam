// SPDX-License-Identifier: Apache-2.0
//! Calls to hosted model providers. The policy layer (`client`) sits over any
//! transport, so the checks that keep a request on an allowed host, off
//! redirects and inside its limits are tested without a socket.

pub mod anthropic;
pub mod client;
pub mod cost;
pub mod dialect;
pub mod gemini;
pub mod model;
pub mod openai;
pub mod secrets;
pub mod summariser;
pub mod transport;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use liam_log::crypto::KeychainItems;

use self::client::HostedClient;
use self::cost::CostRecorder;
use self::dialect::Dialect;
use self::model::{hosted_identity, HostedModel};
use self::secrets::SecretSource;
use self::summariser::{HostedExtractorProvider, HostedSummariser};
use self::transport::UreqTransport;
use crate::config::{Config, Provider};
use crate::extraction::client::ModelIdentity;
use crate::extraction::gate::ProviderGate;
use crate::extraction::identity::IdentitySource;
use crate::extraction::setup::Backend;
use crate::extraction::sidecar::FreeRam;
use crate::metrics::recorder::Metrics;
use crate::summary::summariser::Summariser;

const KEYCHAIN_SERVICE_PREFIX: &str = "liam-";

#[cfg(target_os = "macos")]
fn keychain() -> Arc<dyn KeychainItems> {
    Arc::new(liam_log::crypto::SystemKeychain)
}

/// Only macOS has a Keychain; elsewhere the key must come from a file.
#[cfg(not(target_os = "macos"))]
fn keychain() -> Arc<dyn KeychainItems> {
    Arc::new(liam_log::crypto::MemoryKeychain::default())
}

fn dialect_for(provider: Provider) -> anyhow::Result<Box<dyn Dialect>> {
    Ok(match provider {
        Provider::Anthropic => Box::new(anthropic::Anthropic),
        Provider::OpenAi => Box::new(openai::OpenAi),
        Provider::Gemini => Box::new(gemini::Gemini),
        Provider::Local => anyhow::bail!("the local model is not a hosted provider"),
    })
}

fn key_source(config: &Config, provider: Provider) -> SecretSource {
    match config.hosted.key_files.get(provider.name()) {
        Some(path) => SecretSource::File(path.into()),
        None => SecretSource::Keychain {
            service: format!("{KEYCHAIN_SERVICE_PREFIX}{}", provider.name()),
        },
    }
}

/// The pinned model of `provider`, with its key read and its calls counted in
/// `metrics`.
///
/// # Errors
///
/// Fails when the provider has no pin or its key cannot be read.
pub fn open_model(
    config: &Config,
    provider: Provider,
    metrics: Arc<Metrics>,
) -> anyhow::Result<Arc<HostedModel<UreqTransport>>> {
    let pin = config
        .hosted
        .pin_for(provider)
        .with_context(|| format!("hosted.models.{} is not set", provider.name()))?;
    let key = key_source(config, provider)
        .read(keychain().as_ref())
        .with_context(|| {
            format!(
                "could not read the {name} key; set hosted.key_files.{name} or, on macOS, add the Keychain item {KEYCHAIN_SERVICE_PREFIX}{name}",
                name = provider.name()
            )
        })?;
    let hosted = &config.hosted;
    let transport = UreqTransport::new(Duration::from_secs(hosted.request_timeout_secs));
    Ok(Arc::new(HostedModel::new(
        HostedClient::new(transport, hosted),
        dialect_for(provider)?,
        pin,
        key,
        CostRecorder::new(metrics, hosted.prices.clone()),
    )))
}

/// The summariser for a hosted `provider`.
///
/// # Errors
///
/// Fails as [`open_model`] does.
pub fn summariser(
    config: &Config,
    provider: Provider,
    metrics: Arc<Metrics>,
) -> anyhow::Result<Arc<dyn Summariser>> {
    let model = open_model(config, provider, metrics)?;
    Ok(Arc::new(HostedSummariser::new(model)))
}

/// The extraction backend for a hosted `provider`. The key is read when a run
/// first needs the provider, not here.
///
/// # Errors
///
/// Fails when the provider has no pin.
pub fn extraction_backend(
    config: &Config,
    provider: Provider,
    metrics: Arc<Metrics>,
) -> anyhow::Result<Backend> {
    let pin = config
        .hosted
        .pin_for(provider)
        .with_context(|| format!("hosted.models.{} is not set", provider.name()))?;
    let identity = hosted_identity(&pin);
    let opener = {
        let config = config.clone();
        move || open_model(&config, provider, metrics.clone())
    };
    Ok(Backend {
        provider: Arc::new(HostedExtractorProvider::new(opener)),
        identity: Arc::new(KnownIdentity(identity)),
        free_ram: Arc::new(NoRamLimit),
        gate: Some(Arc::new(ProviderGate::new(
            config.hosted.privacy.clone(),
            provider,
        ))),
    })
}

/// A hosted model is named by its pin, so there is no file to hash.
struct KnownIdentity(ModelIdentity);

#[async_trait]
impl IdentitySource for KnownIdentity {
    async fn identity(&self) -> anyhow::Result<ModelIdentity> {
        Ok(self.0.clone())
    }
}

/// The host's free memory says nothing about a model that runs elsewhere.
struct NoRamLimit;

impl FreeRam for NoRamLimit {
    fn free_mb(&self) -> u64 {
        u64::MAX
    }
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;
    use crate::config::{HostedConfig, PricesConfig};
    use crate::hosted::test_support::{metrics, price};

    const PIN: &str = "claude-sonnet-4-5-20250929";

    fn config_with_key(dir: &std::path::Path, mode: u32) -> Config {
        let key = dir.join("anthropic.key");
        std::fs::write(&key, "sk-test-key\n").unwrap();
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(mode)).unwrap();
        Config {
            hosted: HostedConfig {
                enabled: true,
                allowed_hosts: vec![anthropic::HOST.to_owned()],
                models: [("anthropic".to_owned(), PIN.to_owned())].into(),
                prices: PricesConfig {
                    version: "test".to_owned(),
                    models: [(PIN.to_owned(), price())].into(),
                },
                key_files: [("anthropic".to_owned(), key.to_str().unwrap().to_owned())].into(),
                ..HostedConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn a_pinned_provider_with_an_owner_only_key_file_opens_under_its_pin() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = config_with_key(dir.path(), 0o600);
        let (metrics, _) = metrics();

        // Act
        let model = open_model(&config, Provider::Anthropic, metrics.clone()).unwrap();
        let summariser = summariser(&config, Provider::Anthropic, metrics).unwrap();

        // Assert
        assert_eq!(model.identity().model_id, format!("anthropic:{PIN}"));
        assert_eq!(summariser.identity(), model.identity());
    }

    #[test]
    fn a_key_file_open_to_others_or_a_missing_pin_refuses_to_open() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let loose = config_with_key(dir.path(), 0o644);
        let (metrics, _) = metrics();
        let unpinned = Config::default();

        // Act
        let loose_error = open_model(&loose, Provider::Anthropic, metrics.clone())
            .err()
            .unwrap();
        let pin_error = open_model(&unpinned, Provider::Anthropic, metrics)
            .err()
            .unwrap();
        let local = dialect_for(Provider::Local).err().unwrap();

        // Assert
        assert!(format!("{loose_error:#}").contains("anthropic key"));
        assert!(format!("{pin_error:#}").contains("hosted.models.anthropic"));
        assert!(local.to_string().contains("not a hosted provider"));
    }

    #[tokio::test]
    async fn the_extraction_backend_names_the_pin_reads_no_key_yet_and_carries_a_gate() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let mut config = config_with_key(dir.path(), 0o644);
        let (metrics, _) = metrics();

        // Act
        let backend = extraction_backend(&config, Provider::Anthropic, metrics.clone()).unwrap();
        let identity = backend.identity.identity().await.unwrap();
        let opened = backend
            .provider
            .open(&crate::config::ExtractionConfig::default())
            .await;
        config.hosted.models.clear();
        let unpinned = extraction_backend(&config, Provider::Anthropic, metrics);

        // Assert: the loose key file is only noticed when the provider opens
        assert_eq!(identity.model_id, format!("anthropic:{PIN}"));
        assert!(opened.is_err());
        assert!(backend.gate.is_some());
        assert_eq!(backend.free_ram.free_mb(), u64::MAX);
        assert!(unpinned.is_err());
    }
}
