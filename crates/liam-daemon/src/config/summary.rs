// SPDX-License-Identifier: Apache-2.0
//! The `[summary]` section: the summary tree the consolidation job builds over
//! recorded nodes. Off until enabled.

use serde::Deserialize;

use super::pricing::Provider;
#[cfg(test)]
use serde::Serialize;

/// The deepest summary tree the builder plans.
pub const MAX_SUMMARY_DEPTH: u8 = 3;
const DEFAULT_KNN: usize = 8;
const DEFAULT_MIN_SIMILARITY: f32 = 0.5;
/// A month: items further apart in time are not linked.
const DEFAULT_TIME_WINDOW_HOURS: u64 = 720;
const DEFAULT_MIN_CLUSTER_SIZE: usize = 3;
const DEFAULT_MAX_CLUSTER_SIZE: usize = 30;
const DEFAULT_MAX_DEPTH: u8 = 3;
const DEFAULT_RECLUSTER_CHANGE_RATIO: f64 = 0.2;
const DEFAULT_RECLUSTER_INTERVAL_DAYS: u64 = 7;
const DEFAULT_MIN_ATTACH_SIMILARITY: f32 = 0.6;
const DEFAULT_MAX_UNATTACHED: u32 = 20;
const DEFAULT_MAX_SUMMARIES_PER_RUN: usize = 20;
const DEFAULT_RUN_DEADLINE_SECS: u64 = 600;
const DEFAULT_MIN_GROUNDED_SHARE: f64 = 0.5;
const DEFAULT_MAX_SUMMARY_FAILURES: u32 = 3;
/// Cosine similarity of a question to a summary below which `ask` leaves the
/// summary out.
pub const DEFAULT_MIN_SUMMARY_SIMILARITY: f32 = 0.5;
const DEFAULT_DISSOLVE_GRACE_RUNS: u32 = 3;
const DEFAULT_DISSOLVE_OVERLAP: f64 = 0.5;
/// Event ids shown on one provenance line before the rest are counted.
pub const DEFAULT_MAX_BASIS_IDS_RENDERED: usize = 8;
/// A typo of a few extra zeros would stall a run or overflow the arithmetic on
/// the duration it turns into.
const MAX_RUN_DEADLINE_SECS: u64 = 86_400;
const MAX_TIME_WINDOW_HOURS: u64 = 24 * 365 * 10;
const MAX_RECLUSTER_INTERVAL_DAYS: u64 = 365 * 10;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct SummaryConfig {
    /// False makes the whole hook a no-op.
    pub enabled: bool,
    /// Who writes the summaries: the local model, or a hosted provider (which
    /// also needs `[hosted]`). `ask` and entity synthesis always stay local.
    pub provider: Provider,
    /// Nearest neighbours considered for each item.
    pub knn: usize,
    /// Least cosine similarity at which two items are linked.
    pub min_similarity: f32,
    /// Items further apart than this are not linked.
    pub time_window_hours: u64,
    /// Fewest members of a cluster that gets a summary.
    pub min_cluster_size: usize,
    /// Most members of one cluster; a larger one is split.
    pub max_cluster_size: usize,
    /// Levels of summaries above the nodes, at most 3.
    pub max_depth: u8,
    /// Share of a level's items that may change before it is clustered again.
    pub recluster_change_ratio: f64,
    /// Age of the last full build after which a level is clustered again.
    pub recluster_interval_days: u64,
    /// Least similarity to a cluster's centroid at which a new item joins it.
    pub min_attach_similarity: f32,
    /// New items that may wait unattached before a level is clustered again.
    pub max_unattached: u32,
    /// Most summaries one run generates.
    pub max_summaries_per_run: usize,
    /// Bounds the generation of one run.
    pub run_deadline_secs: u64,
    /// Least share of a summary's content words that its items must hold.
    pub min_grounded_share: f64,
    /// Failures of one child set after which it is skipped.
    pub max_summary_failures: u32,
    /// Least cosine similarity of a question to a summary for `ask` to use it.
    pub min_summary_similarity: f32,
    /// Runs a summary whose cluster is gone waits for a replacement before it is closed.
    pub dissolve_grace_runs: u32,
    /// Share of a dissolved summary's items that a stored summary must hold to replace it.
    pub dissolve_overlap: f64,
    /// Event ids a provenance line lists before it counts the rest.
    pub max_basis_ids_rendered: usize,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: Provider::Local,
            knn: DEFAULT_KNN,
            min_similarity: DEFAULT_MIN_SIMILARITY,
            time_window_hours: DEFAULT_TIME_WINDOW_HOURS,
            min_cluster_size: DEFAULT_MIN_CLUSTER_SIZE,
            max_cluster_size: DEFAULT_MAX_CLUSTER_SIZE,
            max_depth: DEFAULT_MAX_DEPTH,
            recluster_change_ratio: DEFAULT_RECLUSTER_CHANGE_RATIO,
            recluster_interval_days: DEFAULT_RECLUSTER_INTERVAL_DAYS,
            min_attach_similarity: DEFAULT_MIN_ATTACH_SIMILARITY,
            max_unattached: DEFAULT_MAX_UNATTACHED,
            max_summaries_per_run: DEFAULT_MAX_SUMMARIES_PER_RUN,
            run_deadline_secs: DEFAULT_RUN_DEADLINE_SECS,
            min_grounded_share: DEFAULT_MIN_GROUNDED_SHARE,
            max_summary_failures: DEFAULT_MAX_SUMMARY_FAILURES,
            min_summary_similarity: DEFAULT_MIN_SUMMARY_SIMILARITY,
            dissolve_grace_runs: DEFAULT_DISSOLVE_GRACE_RUNS,
            dissolve_overlap: DEFAULT_DISSOLVE_OVERLAP,
            max_basis_ids_rendered: DEFAULT_MAX_BASIS_IDS_RENDERED,
        }
    }
}

impl SummaryConfig {
    /// Rejects values the tree cannot be built with, naming the key to fix.
    pub fn validate(&self) -> anyhow::Result<()> {
        for (key, value) in [
            ("knn", self.knn as u64),
            ("max_summaries_per_run", self.max_summaries_per_run as u64),
            ("max_summary_failures", u64::from(self.max_summary_failures)),
            ("dissolve_grace_runs", u64::from(self.dissolve_grace_runs)),
            ("max_basis_ids_rendered", self.max_basis_ids_rendered as u64),
            ("run_deadline_secs", self.run_deadline_secs),
            ("time_window_hours", self.time_window_hours),
            ("recluster_interval_days", self.recluster_interval_days),
        ] {
            anyhow::ensure!(value > 0, "summary.{key} must be above 0");
        }
        anyhow::ensure!(
            self.run_deadline_secs <= MAX_RUN_DEADLINE_SECS,
            "summary.run_deadline_secs must be at most {MAX_RUN_DEADLINE_SECS}"
        );
        anyhow::ensure!(
            self.time_window_hours <= MAX_TIME_WINDOW_HOURS,
            "summary.time_window_hours must be at most {MAX_TIME_WINDOW_HOURS}"
        );
        anyhow::ensure!(
            self.recluster_interval_days <= MAX_RECLUSTER_INTERVAL_DAYS,
            "summary.recluster_interval_days must be at most {MAX_RECLUSTER_INTERVAL_DAYS}"
        );
        anyhow::ensure!(
            self.min_cluster_size >= 2,
            "summary.min_cluster_size must be at least 2, got {}",
            self.min_cluster_size
        );
        anyhow::ensure!(
            self.max_cluster_size >= self.min_cluster_size,
            "summary.max_cluster_size must be at least summary.min_cluster_size, got {}",
            self.max_cluster_size
        );
        anyhow::ensure!(
            (1..=MAX_SUMMARY_DEPTH).contains(&self.max_depth),
            "summary.max_depth must be between 1 and {MAX_SUMMARY_DEPTH}, got {}",
            self.max_depth
        );
        for (key, value) in [
            ("min_similarity", f64::from(self.min_similarity)),
            (
                "min_attach_similarity",
                f64::from(self.min_attach_similarity),
            ),
            (
                "min_summary_similarity",
                f64::from(self.min_summary_similarity),
            ),
        ] {
            anyhow::ensure!(
                (-1.0..=1.0).contains(&value),
                "summary.{key} must be between -1 and 1, got {value}"
            );
        }
        for (key, value) in [
            ("recluster_change_ratio", self.recluster_change_ratio),
            ("min_grounded_share", self.min_grounded_share),
            ("dissolve_overlap", self.dissolve_overlap),
        ] {
            anyhow::ensure!(
                value > 0.0 && value <= 1.0,
                "summary.{key} must be above 0 and at most 1, got {value}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::config::Config;

    fn load(toml: &str) -> anyhow::Result<Config> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liam.toml");
        std::fs::write(&path, toml).unwrap();
        Config::load(&path)
    }

    fn section(line: &str) -> String {
        format!("[summary]\n{line}\n")
    }

    #[test]
    fn a_config_without_the_section_gets_every_default_and_is_off() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.summary, SummaryConfig::default());
        assert!(!config.summary.enabled);
    }

    #[test]
    fn an_unknown_key_fails_loudly_naming_the_key() {
        // Act
        let error = load(&section("knm = 3")).expect_err("a typo must be rejected");

        // Assert
        assert!(error.to_string().contains("knm"), "{error}");
    }

    #[test]
    fn a_value_the_tree_cannot_be_built_with_is_rejected_naming_the_key() {
        for (line, key) in [
            ("knn = 0", "summary.knn"),
            ("max_summaries_per_run = 0", "summary.max_summaries_per_run"),
            ("max_summary_failures = 0", "summary.max_summary_failures"),
            ("dissolve_grace_runs = 0", "summary.dissolve_grace_runs"),
            (
                "max_basis_ids_rendered = 0",
                "summary.max_basis_ids_rendered",
            ),
            ("dissolve_overlap = 0.0", "summary.dissolve_overlap"),
            ("dissolve_overlap = 1.5", "summary.dissolve_overlap"),
            ("dissolve_overlap = nan", "summary.dissolve_overlap"),
            ("run_deadline_secs = 0", "summary.run_deadline_secs"),
            ("run_deadline_secs = 86401", "summary.run_deadline_secs"),
            ("time_window_hours = 0", "summary.time_window_hours"),
            ("time_window_hours = 87601", "summary.time_window_hours"),
            (
                "recluster_interval_days = 0",
                "summary.recluster_interval_days",
            ),
            (
                "recluster_interval_days = 3651",
                "summary.recluster_interval_days",
            ),
            ("min_cluster_size = 1", "summary.min_cluster_size"),
            ("max_cluster_size = 2", "summary.max_cluster_size"),
            ("max_depth = 0", "summary.max_depth"),
            ("max_depth = 4", "summary.max_depth"),
            ("min_similarity = 1.1", "summary.min_similarity"),
            ("min_similarity = nan", "summary.min_similarity"),
            (
                "min_attach_similarity = -1.1",
                "summary.min_attach_similarity",
            ),
            (
                "min_summary_similarity = 1.1",
                "summary.min_summary_similarity",
            ),
            (
                "min_summary_similarity = nan",
                "summary.min_summary_similarity",
            ),
            (
                "recluster_change_ratio = 0.0",
                "summary.recluster_change_ratio",
            ),
            (
                "recluster_change_ratio = 1.5",
                "summary.recluster_change_ratio",
            ),
            ("min_grounded_share = 0.0", "summary.min_grounded_share"),
            ("min_grounded_share = nan", "summary.min_grounded_share"),
        ] {
            // Act
            let error = load(&section(line)).expect_err(line);

            // Assert
            assert!(error.to_string().contains(key), "{line}: {error}");
        }
    }

    #[test]
    fn the_boundary_values_are_accepted() {
        for line in [
            "knn = 1",
            "run_deadline_secs = 86400",
            "min_cluster_size = 2\nmax_cluster_size = 2",
            "max_depth = 1",
            "max_depth = 3",
            "min_similarity = -1.0",
            "min_similarity = 1.0",
            "min_attach_similarity = -1.0",
            "min_attach_similarity = 1.0",
            "min_summary_similarity = -1.0",
            "min_summary_similarity = 1.0",
            "recluster_change_ratio = 1.0",
            "min_grounded_share = 1.0",
            "dissolve_grace_runs = 1",
            "dissolve_overlap = 1.0",
            "max_basis_ids_rendered = 1",
            "max_unattached = 0",
        ] {
            // Act
            let result = load(&section(line));

            // Assert
            assert!(result.is_ok(), "{line}: {result:?}");
        }
    }

    const README_SECTION_HEADING: &str = "## Summaries";
    const SECTION_MARK: &str = "## ";

    /// Every key of the section, read off the serialized default so a new field
    /// cannot be left out of the check.
    fn summary_keys() -> Vec<String> {
        let value = serde_json::to_value(SummaryConfig::default()).unwrap();
        value.as_object().unwrap().keys().cloned().collect()
    }

    /// Whether `line` sets `key`, live or commented out.
    fn assigns(line: &str, key: &str) -> bool {
        line.trim_start()
            .trim_start_matches('#')
            .trim_start()
            .strip_prefix(key)
            .is_some_and(|rest| rest.trim_start().starts_with('='))
    }

    fn read_root(file: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(file);
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn the_shipped_liam_toml_carries_the_code_defaults() {
        // Arrange
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../liam.toml"));

        // Act
        let config = Config::load(path).expect("shipped liam.toml must parse");

        // Assert
        assert_eq!(config.summary, SummaryConfig::default());
    }

    #[test]
    fn liam_toml_documents_every_summary_key() {
        // Arrange
        let text = read_root("liam.toml");
        let documented: Vec<&str> = text
            .lines()
            .skip_while(|line| !line.starts_with("[summary]"))
            .skip(1)
            .take_while(|line| !line.starts_with('['))
            .collect();

        // Act
        let missing: Vec<String> = summary_keys()
            .into_iter()
            .filter(|key| !documented.iter().any(|line| assigns(line, key)))
            .collect();

        // Assert
        assert!(missing.is_empty(), "liam.toml must document {missing:?}");
    }

    #[test]
    fn the_readme_summaries_section_documents_every_summary_key() {
        // Arrange
        let text = read_root("README.md");
        let section: Vec<&str> = text
            .lines()
            .skip_while(|line| *line != README_SECTION_HEADING)
            .skip(1)
            .take_while(|line| !line.starts_with(SECTION_MARK))
            .collect();
        assert!(
            !section.is_empty(),
            "README.md needs a {README_SECTION_HEADING} section"
        );

        // Act
        let missing: Vec<String> = summary_keys()
            .into_iter()
            .filter(|key| {
                let row = format!("| `{key}` |");
                !section.iter().any(|line| line.starts_with(&row))
            })
            .collect();

        // Assert
        assert!(missing.is_empty(), "README.md must document {missing:?}");
    }
}
