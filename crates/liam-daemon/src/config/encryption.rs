// SPDX-License-Identifier: Apache-2.0
//! The `[encryption]` section: which master key wraps the log's data keys.
//! Unset keeps a plaintext log.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

/// Where the master key lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(rename_all = "lowercase")]
pub enum MasterKeySource {
    /// The macOS login keychain.
    Keychain,
    /// A key file only its owner may read, named by `key_file`.
    File,
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct EncryptionConfig {
    /// Setting this turns on sealing for new log events. Unset keeps the log
    /// in plaintext, as before.
    pub master_key: Option<MasterKeySource>,
    /// The key file for `master_key = "file"`. Kept off by the keychain.
    pub key_file: Option<String>,
}

impl EncryptionConfig {
    /// Whether new log events are sealed.
    pub fn enabled(&self) -> bool {
        self.master_key.is_some()
    }

    /// Expands `~` in `key_file` once, so every reader sees the same path.
    pub(super) fn expand_paths(&mut self, home: &str) -> anyhow::Result<()> {
        if let Some(text) = self.key_file.as_deref() {
            self.key_file = Some(crate::models::resolve_path_with_home(
                "encryption.key_file",
                text,
                home,
            )?);
        }
        Ok(())
    }

    /// Rejects a source without what it needs, naming the key to fix.
    pub fn validate(&self) -> anyhow::Result<()> {
        match (self.master_key, self.key_file.as_deref()) {
            (Some(MasterKeySource::File), None) => {
                anyhow::bail!(
                    "encryption.key_file is required when encryption.master_key = \"file\""
                )
            }
            (Some(MasterKeySource::File), Some(path)) => {
                anyhow::ensure!(
                    !path.trim().is_empty(),
                    "encryption.key_file must not be empty"
                );
            }
            (Some(MasterKeySource::Keychain), Some(_)) => {
                anyhow::bail!(
                    "encryption.key_file only applies when encryption.master_key = \"file\""
                )
            }
            (None, Some(_)) => {
                anyhow::bail!("encryption.key_file is set but encryption.master_key is not")
            }
            (Some(MasterKeySource::Keychain), None) | (None, None) => {}
        }
        Ok(())
    }
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
    fn a_config_without_the_section_keeps_the_log_plaintext() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.encryption, EncryptionConfig::default());
        assert!(!config.encryption.enabled());
    }

    #[test]
    fn keychain_and_file_sources_parse() {
        // Act
        let keychain = load("[encryption]\nmaster_key = \"keychain\"\n").unwrap();
        let file =
            load("[encryption]\nmaster_key = \"file\"\nkey_file = \"/keys/master\"\n").unwrap();

        // Assert
        assert_eq!(
            keychain.encryption.master_key,
            Some(MasterKeySource::Keychain)
        );
        assert!(keychain.encryption.enabled());
        assert_eq!(file.encryption.master_key, Some(MasterKeySource::File));
        assert_eq!(file.encryption.key_file.as_deref(), Some("/keys/master"));
    }

    #[test]
    fn a_tilde_in_the_key_file_is_expanded() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liam.toml");
        std::fs::write(
            &path,
            "[encryption]\nmaster_key = \"file\"\nkey_file = \"~/m.key\"\n",
        )
        .unwrap();

        // Act
        let config = Config::load_with_home(&path, "/home/alice").unwrap();

        // Assert
        assert_eq!(
            config.encryption.key_file.as_deref(),
            Some("/home/alice/m.key")
        );
    }

    #[test]
    fn invalid_sections_fail_naming_the_key() {
        for (toml, expected) in [
            (
                "[encryption]\nmaster_key = \"file\"\n",
                "encryption.key_file is required",
            ),
            (
                "[encryption]\nmaster_key = \"file\"\nkey_file = \" \"\n",
                "must not be empty",
            ),
            (
                "[encryption]\nmaster_key = \"keychain\"\nkey_file = \"/k\"\n",
                "only applies",
            ),
            ("[encryption]\nkey_file = \"/k\"\n", "master_key is not"),
        ] {
            // Act
            let error = load(toml).unwrap_err().to_string();

            // Assert
            assert!(error.contains(expected), "{toml}: {error}");
        }
    }

    #[test]
    fn an_unknown_provider_or_key_is_rejected() {
        // Act
        let provider = load("[encryption]\nmaster_key = \"vault\"\n");
        let key = load("[encryption]\nmaster_ky = \"file\"\n");

        // Assert
        assert!(provider.is_err());
        assert!(key.is_err());
    }
}
