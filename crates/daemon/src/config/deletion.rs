// SPDX-License-Identifier: Apache-2.0
//! The `[deletion]` section: how long a delete may take, and whether the
//! database zeroes the pages a delete frees.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

const DEFAULT_DEADLINE_DAYS: u32 = 30;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct DeletionConfig {
    /// Days a delete job has to finish before it is reported as overdue.
    pub deadline_days: u32,
    /// Zero the pages a delete frees, so removed text does not stay in the
    /// database file. It costs a write per freed page, and an SSD or a
    /// copy-on-write file system can still keep an older copy of a block, so
    /// this lowers the risk and does not remove it.
    pub secure_delete: bool,
}

impl Default for DeletionConfig {
    fn default() -> Self {
        Self {
            deadline_days: DEFAULT_DEADLINE_DAYS,
            secure_delete: true,
        }
    }
}

impl DeletionConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.deadline_days > 0,
            "deletion.deadline_days must be at least 1"
        );
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
    fn a_config_without_the_section_deletes_securely_with_a_thirty_day_deadline() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.deletion, DeletionConfig::default());
        assert_eq!(config.deletion.deadline_days, DEFAULT_DEADLINE_DAYS);
        assert!(config.deletion.secure_delete);
    }

    #[test]
    fn the_section_overrides_each_key() {
        // Act
        let config = load("[deletion]\ndeadline_days = 7\nsecure_delete = false\n").unwrap();

        // Assert
        assert_eq!(config.deletion.deadline_days, 7);
        assert!(!config.deletion.secure_delete);
    }

    #[test]
    fn an_unknown_key_or_a_zero_deadline_fails_naming_the_key() {
        for (toml, expected) in [
            ("[deletion]\nsecure_deletes = true\n", "secure_deletes"),
            ("[deletion]\ndeadline_days = 0\n", "deletion.deadline_days"),
        ] {
            // Act
            let error = format!("{:#}", load(toml).unwrap_err());

            // Assert
            assert!(error.contains(expected), "{error}");
        }
    }
}
