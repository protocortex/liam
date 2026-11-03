// SPDX-License-Identifier: Apache-2.0
//! The `[admin]` section: which producers may call the `delete` and
//! `consolidate` tools.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

/// A producer name is a claim a client makes, not an authenticated identity, so
/// this list guards against honest mistakes and not against abuse.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct AdminConfig {
    /// Empty by default, so the admin tools refuse everyone until a producer is
    /// listed.
    pub producers: Vec<String>,
}

impl AdminConfig {
    pub fn allows(&self, producer: &str) -> bool {
        self.producers.iter().any(|allowed| allowed == producer)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.producers.iter().all(|name| !name.trim().is_empty()),
            "admin.producers must not hold an empty name"
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
    fn a_config_without_the_section_admits_no_producer() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.admin, AdminConfig::default());
        assert!(!config.admin.allows("anyone"));
    }

    #[test]
    fn a_listed_producer_is_admitted_and_others_are_not() {
        // Act
        let config = load("[admin]\nproducers = [\"ops\"]\n").unwrap();

        // Assert
        assert!(config.admin.allows("ops"));
        assert!(!config.admin.allows("ops-two"));
    }

    #[test]
    fn an_empty_name_or_an_unknown_key_is_refused() {
        // Act
        let empty = load("[admin]\nproducers = [\" \"]\n").unwrap_err();
        let unknown = load("[admin]\nowners = []\n").unwrap_err();

        // Assert
        assert!(format!("{empty:#}").contains("admin.producers"));
        assert!(format!("{unknown:#}").contains("owners"));
    }
}
