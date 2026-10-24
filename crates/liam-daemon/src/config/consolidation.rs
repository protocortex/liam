// SPDX-License-Identifier: Apache-2.0
//! The `[consolidation]` section: the nightly-style job that snapshots the
//! store, merges near-duplicates, runs maintenance and commits only if a probe
//! set still recalls what it did before.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// A probe whose recall falls by this much fails the run.
const DEFAULT_PROBE_RECALL_MARGIN: f64 = 0.02;
/// Cosine similarity at or above which two nodes are near-duplicates. High,
/// since a wrong merge costs a link between unrelated facts.
const DEFAULT_DEDUP_SIMILARITY_THRESHOLD: f32 = 0.95;
const DEFAULT_PROBE_COUNT: usize = 20;
/// Each probe is a retrieval query run twice per run, so a typo like an extra
/// zero would make every run slow.
const MAX_PROBE_COUNT: usize = 1_000;
/// Suffix of the snapshot directory beside the database.
const SNAPSHOT_DIR_SUFFIX: &str = ".snapshots";

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConsolidationConfig {
    /// Recall drop, in absolute terms, at which a probe fails the run.
    pub probe_recall_margin: f64,
    /// Cosine similarity at or above which two nodes are merged.
    pub dedup_similarity_threshold: f32,
    /// Where runs keep their snapshots. Unset means `<database stem>.snapshots`
    /// beside the database, see `default_snapshot_dir`.
    pub snapshot_dir: Option<String>,
    /// Most probes drawn from the store to gate a run.
    pub probe_count: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            probe_recall_margin: DEFAULT_PROBE_RECALL_MARGIN,
            dedup_similarity_threshold: DEFAULT_DEDUP_SIMILARITY_THRESHOLD,
            snapshot_dir: None,
            probe_count: DEFAULT_PROBE_COUNT,
        }
    }
}

impl ConsolidationConfig {
    /// Rejects values the job cannot run with, naming the key to fix.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.probe_recall_margin > 0.0 && self.probe_recall_margin <= 1.0,
            "consolidation.probe_recall_margin must be above 0 and at most 1, got {}",
            self.probe_recall_margin
        );
        anyhow::ensure!(
            self.dedup_similarity_threshold > 0.0 && self.dedup_similarity_threshold <= 1.0,
            "consolidation.dedup_similarity_threshold must be above 0 and at most 1, got {}",
            self.dedup_similarity_threshold
        );
        // Without probes nothing would gate a run, so it is refused up front.
        anyhow::ensure!(
            (1..=MAX_PROBE_COUNT).contains(&self.probe_count),
            "consolidation.probe_count must be between 1 and {MAX_PROBE_COUNT}, got {}",
            self.probe_count
        );
        anyhow::ensure!(
            self.snapshot_dir
                .as_deref()
                .is_none_or(|dir| !dir.trim().is_empty()),
            "consolidation.snapshot_dir must not be empty; remove it to keep the snapshots \
             beside the database"
        );
        Ok(())
    }
}

/// `<database stem>.snapshots` beside the database: the snapshot directory
/// used when the config names none.
pub(crate) fn default_snapshot_dir(database: &Path) -> PathBuf {
    let stem = database.file_stem().unwrap_or(database.as_os_str());
    let mut name = stem.to_owned();
    name.push(SNAPSHOT_DIR_SUFFIX);
    database.with_file_name(name)
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
    fn a_config_without_the_section_gets_every_default() {
        // Arrange
        let toml = "[gc]\n";

        // Act
        let config = load(toml).expect("a config without [consolidation] must parse");

        // Assert
        assert_eq!(config.consolidation, ConsolidationConfig::default());
        assert_eq!(config.consolidation.probe_recall_margin, 0.02);
        assert_eq!(config.consolidation.dedup_similarity_threshold, 0.95);
        assert_eq!(config.consolidation.probe_count, 20);
        assert_eq!(config.consolidation.snapshot_dir, None);
    }

    #[test]
    fn a_section_with_one_key_keeps_the_defaults_of_the_others() {
        // Arrange
        let toml = "[consolidation]\nprobe_count = 5\n";

        // Act
        let config = load(toml).unwrap();

        // Assert
        assert_eq!(config.consolidation.probe_count, 5);
        assert_eq!(config.consolidation.probe_recall_margin, 0.02);
    }

    #[test]
    fn an_unknown_key_fails_loudly_naming_the_key() {
        // Arrange
        let toml = "[consolidation]\nprobe_recal_margin = 0.1\n";

        // Act
        let error = load(toml).expect_err("a typo must be rejected");

        // Assert
        assert!(error.to_string().contains("probe_recal_margin"), "{error}");
    }

    #[test]
    fn a_value_the_job_cannot_run_with_is_rejected_naming_the_key() {
        for (toml, key) in [
            (
                "probe_recall_margin = 0.0",
                "consolidation.probe_recall_margin",
            ),
            (
                "probe_recall_margin = 1.5",
                "consolidation.probe_recall_margin",
            ),
            (
                "probe_recall_margin = nan",
                "consolidation.probe_recall_margin",
            ),
            (
                "dedup_similarity_threshold = 0.0",
                "consolidation.dedup_similarity_threshold",
            ),
            (
                "dedup_similarity_threshold = 1.1",
                "consolidation.dedup_similarity_threshold",
            ),
            (
                "dedup_similarity_threshold = nan",
                "consolidation.dedup_similarity_threshold",
            ),
            ("probe_count = 0", "consolidation.probe_count"),
            ("probe_count = 1001", "consolidation.probe_count"),
            ("snapshot_dir = \"  \"", "consolidation.snapshot_dir"),
        ] {
            // Arrange
            let toml = format!("[consolidation]\n{toml}\n");

            // Act
            let error = load(&toml).expect_err(&toml);

            // Assert
            assert!(error.to_string().contains(key), "{toml}: {error}");
        }
    }

    #[test]
    fn the_boundary_values_are_accepted() {
        // Arrange
        let toml = "[consolidation]\nprobe_recall_margin = 1.0\ndedup_similarity_threshold = 1.0\nprobe_count = 1\n";

        // Act
        let result = load(toml);

        // Assert
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn the_largest_probe_count_is_accepted() {
        let toml = format!("[consolidation]\nprobe_count = {MAX_PROBE_COUNT}\n");

        assert!(load(&toml).is_ok());
    }

    #[test]
    fn the_default_snapshot_directory_is_a_sibling_named_after_the_database_stem() {
        for (database, dir) in [
            ("/data/work.db", "/data/work.snapshots"),
            ("liam.db", "liam.snapshots"),
            ("/data/store", "/data/store.snapshots"),
        ] {
            assert_eq!(
                default_snapshot_dir(Path::new(database)),
                PathBuf::from(dir),
                "database {database}"
            );
        }
    }

    #[test]
    fn the_shipped_liam_toml_carries_the_code_defaults() {
        // Arrange
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../liam.toml"));
        let text = std::fs::read_to_string(path).unwrap();

        // Act
        let config = Config::load(path).expect("shipped liam.toml must parse");

        // Assert
        assert!(text.contains("[consolidation]"));
        for key in [
            "probe_recall_margin",
            "dedup_similarity_threshold",
            "snapshot_dir",
            "probe_count",
        ] {
            assert!(
                text.lines()
                    .any(|line| line.trim_start_matches('#').trim_start().starts_with(key)),
                "liam.toml must document consolidation.{key}"
            );
        }
        assert_eq!(config.consolidation, ConsolidationConfig::default());
    }
}
