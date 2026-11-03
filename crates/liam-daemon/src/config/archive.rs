// SPDX-License-Identifier: Apache-2.0
//! The `[archive]` section: moving closed log partitions to another directory.
//! Nothing is ever deleted. With the section absent or `enabled = false`, no
//! partition moves.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

const DEFAULT_AFTER_DAYS: u32 = 90;
const DEFAULT_ZSTD_LEVEL: i32 = 3;
/// The levels zstd accepts.
const ZSTD_LEVELS: std::ops::RangeInclusive<i32> = 1..=22;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct ArchiveConfig {
    pub enabled: bool,
    /// Archive a closed partition once its newest event is this many days old.
    pub after_days: u32,
    /// Where the archived partitions go. It must exist before the first run.
    pub dir: Option<String>,
    /// The zstd level of the archived copy. Sealed bodies barely compress, so a
    /// higher level mostly costs time.
    pub zstd_level: i32,
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            after_days: DEFAULT_AFTER_DAYS,
            dir: None,
            zstd_level: DEFAULT_ZSTD_LEVEL,
        }
    }
}

impl ArchiveConfig {
    /// Expands `~` in `dir` once, so every reader sees the same path.
    pub(super) fn expand_paths(&mut self, home: &str) -> anyhow::Result<()> {
        if let Some(text) = self.dir.as_deref() {
            self.dir = Some(crate::models::resolve_path_with_home(
                "archive.dir",
                text,
                home,
            )?);
        }
        Ok(())
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.after_days > 0, "archive.after_days must be at least 1");
        anyhow::ensure!(
            ZSTD_LEVELS.contains(&self.zstd_level),
            "archive.zstd_level must be between {} and {}",
            ZSTD_LEVELS.start(),
            ZSTD_LEVELS.end()
        );
        anyhow::ensure!(
            !self.enabled
                || self
                    .dir
                    .as_deref()
                    .is_some_and(|dir| !dir.trim().is_empty()),
            "archive.dir is required when archive.enabled is true"
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
    fn a_config_without_the_section_archives_nothing() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.archive, ArchiveConfig::default());
        assert!(!config.archive.enabled);
    }

    #[test]
    fn an_enabled_archive_needs_a_directory() {
        // Act
        let missing = load("[archive]\nenabled = true\n").unwrap_err();
        let blank = load("[archive]\nenabled = true\ndir = \" \"\n").unwrap_err();
        let set = load("[archive]\nenabled = true\ndir = \"/var/liam-archive\"\n").unwrap();

        // Assert
        assert!(format!("{missing:#}").contains("archive.dir"));
        assert!(format!("{blank:#}").contains("archive.dir"));
        assert_eq!(set.archive.dir.as_deref(), Some("/var/liam-archive"));
    }

    #[test]
    fn the_age_and_the_level_are_checked_and_unknown_keys_fail() {
        // Act
        let no_age = load("[archive]\nafter_days = 0\n").unwrap_err();
        let low = load("[archive]\nzstd_level = 0\n").unwrap_err();
        let high = load("[archive]\nzstd_level = 23\n").unwrap_err();
        let unknown = load("[archive]\nretain_days = 3\n").unwrap_err();

        // Assert
        assert!(format!("{no_age:#}").contains("archive.after_days"));
        assert!(format!("{low:#}").contains("archive.zstd_level"));
        assert!(format!("{high:#}").contains("archive.zstd_level"));
        assert!(format!("{unknown:#}").contains("retain_days"));
    }
}
