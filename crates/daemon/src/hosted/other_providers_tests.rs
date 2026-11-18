// SPDX-License-Identifier: Apache-2.0
//! The OpenAI and Gemini providers open the same way the Anthropic one does.

use std::os::unix::fs::PermissionsExt;

use super::*;
use crate::config::{HostedConfig, PricesConfig};
use crate::hosted::test_support::{metrics, price};

const OWNER_ONLY: u32 = 0o600;

/// A provider, its pinned model and the host a call to it may reach.
const PROVIDERS: [(Provider, &str, &str); 2] = [
    (Provider::OpenAi, "gpt-4o-2024-08-06", openai::HOST),
    (Provider::Gemini, "gemini-2.0-flash-001", gemini::HOST),
];

fn config_for(dir: &std::path::Path, provider: Provider, pin: &str, host: &str) -> Config {
    let key = dir.join(format!("{}.key", provider.name()));
    std::fs::write(&key, "sk-test-key\n").unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(OWNER_ONLY)).unwrap();
    Config {
        hosted: HostedConfig {
            enabled: true,
            allowed_hosts: vec![host.to_owned()],
            models: [(provider.name().to_owned(), pin.to_owned())].into(),
            prices: PricesConfig {
                version: "test".to_owned(),
                models: [(pin.to_owned(), price())].into(),
            },
            key_files: [(provider.name().to_owned(), key.to_str().unwrap().to_owned())].into(),
            ..HostedConfig::default()
        },
        ..Config::default()
    }
}

#[test]
fn each_provider_opens_a_model_and_a_summariser_under_its_own_pin() {
    for (provider, pin, host) in PROVIDERS {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dir.path(), provider, pin, host);
        let (metrics, _) = metrics();

        // Act
        let model = open_model(&config, provider, metrics.clone()).unwrap();
        let summariser = summariser(&config, provider, metrics).unwrap();

        // Assert
        let expected = format!("{}:{pin}", provider.name());
        assert_eq!(model.identity().model_id, expected);
        assert_eq!(summariser.identity(), model.identity());
    }
}

#[test]
fn each_provider_opens_a_batch_client() {
    for (provider, pin, host) in PROVIDERS {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = config_for(dir.path(), provider, pin, host);
        let (metrics, _) = metrics();

        // Act
        let batch = open_batch(&config, provider, metrics);

        // Assert
        assert!(batch.is_ok(), "{}: {:?}", provider.name(), batch.err());
    }
}

#[test]
fn without_a_key_file_the_key_comes_from_the_keychain_item_named_for_the_provider() {
    for (provider, _, _) in PROVIDERS {
        // Act
        let source = key_source(&Config::default(), provider);

        // Assert
        assert_eq!(
            source,
            SecretSource::Keychain {
                service: format!("liam-{}", provider.name())
            }
        );
    }
}
