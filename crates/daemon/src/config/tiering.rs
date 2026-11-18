// SPDX-License-Identifier: Apache-2.0
//! The `[tiering]` section: which items keep their vectors. Off until enabled.
//! The numeric defaults are starting values, not measured on real data.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

use super::extraction::MAX_RUN_DEADLINE_SECS;

const DEFAULT_HITS_WEIGHT: f64 = 0.4;
const DEFAULT_RECENCY_WEIGHT: f64 = 0.3;
const DEFAULT_FEEDBACK_WEIGHT: f64 = 0.2;
const DEFAULT_TRUST_WEIGHT: f64 = 0.1;
const DEFAULT_TAU_DAYS: f64 = 30.0;
const DEFAULT_HITS_CAP: u32 = 20;
const DEFAULT_HITS_PER_PRODUCER_ITEM_DAY_CAP: u32 = 10;
const DEFAULT_HITS_PER_PRODUCER_DAY_CAP: u32 = 100;
const DEFAULT_UNKNOWN_TRUST_CEILING: f64 = 0.3;
const DEFAULT_PROMOTE_THRESHOLD: f64 = 0.5;
const DEFAULT_DEMOTE_THRESHOLD: f64 = 0.15;
const DEFAULT_MIN_RESIDENCY_DAYS: u32 = 7;
const DEFAULT_MIN_ITEM_AGE_DAYS: u32 = 14;
const DEFAULT_MAX_PROMOTIONS_PER_RUN: usize = 50;
const DEFAULT_MAX_DEMOTIONS_PER_RUN: usize = 50;
const DEFAULT_WINDOW_DAYS: u32 = 30;
const DEFAULT_USAGE_FLUSH_SECS: u64 = 60;
const DEFAULT_MAX_TRACKED_ITEMS: usize = 10_000;
const DEFAULT_FEEDBACK_PER_PRODUCER_PER_HOUR: u32 = 60;
const DEFAULT_FEEDBACK_ITEM_CAP: usize = 20;
const DEFAULT_FEEDBACK_PRIOR: f64 = 1.0;
const DEFAULT_COOLDOWN_RUNS: u32 = 3;
const DEFAULT_COLD_PROBE_POOL: usize = 24;
const DEFAULT_COLD_OVERSAMPLE: usize = 4;
const DEFAULT_COLD_MAX_HAMMING_RATIO: f64 = 0.35;
const DEFAULT_COLD_ARM_WEIGHT: f64 = 0.5;
const DEFAULT_RUN_DEADLINE_SECS: u64 = 600;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct TieringConfig {
    /// False makes the tier step of a consolidation run a no-op.
    pub enabled: bool,
    /// Weight of the recent hits term of the utility score.
    pub w1: f64,
    /// Weight of the recency term.
    pub w2: f64,
    /// Weight of the answer contribution term, left out for an item with no feedback.
    pub w3: f64,
    /// Weight of the producer trust term.
    pub w4: f64,
    /// Days over which the recency term falls to 1/e.
    pub tau_days: f64,
    /// Hits in the window at which the hits term reaches 1.
    pub hits_cap: u32,
    /// Most hits one producer can add to one item in a day.
    pub hits_per_producer_item_day_cap: u32,
    /// Most hits one producer can add across all items in a day. Past it the
    /// largest contributions are cut first.
    pub hits_per_producer_day_cap: u32,
    /// The most trust the fallback producer (`producers.unknown_id`) can have,
    /// as an author and as a giver of verdicts.
    pub unknown_trust_ceiling: f64,
    /// Utility at or above which a cold item is promoted.
    pub promote_threshold: f64,
    /// Utility below which a hot item is demoted.
    pub demote_threshold: f64,
    /// Days a promoted item is protected from demotion.
    pub min_residency_days: u32,
    /// Least age, from the newest version's write, for an item to be demoted.
    pub min_item_age_days: u32,
    pub max_promotions_per_run: usize,
    pub max_demotions_per_run: usize,
    /// Days of usage that count, and of observation needed before any demotion.
    pub window_days: u32,
    /// Seconds between flushes of the in-memory access counts to the log.
    pub usage_flush_secs: u64,
    /// Items the in-memory counts hold before the oldest is dropped.
    pub max_tracked_items: usize,
    /// Verdicts one producer may give per hour.
    pub feedback_per_producer_per_hour: u32,
    /// Most recent verdicts on one item that the score reads.
    pub feedback_item_cap: usize,
    /// Pseudo-verdicts of "correct" that the contribution rate starts from.
    pub feedback_prior_correct: f64,
    /// Pseudo-verdicts of "incorrect" that the contribution rate starts from.
    pub feedback_prior_incorrect: f64,
    /// Runs a lineage moved by a rolled back run is left alone.
    pub cooldown_runs: u32,
    /// Cold items the probe contributes to one query.
    pub cold_probe_pool: usize,
    /// Candidates asked of the signature index per item wanted.
    pub cold_oversample: usize,
    /// Share of signature bits that may differ for a cold item to be a candidate.
    pub cold_max_hamming_ratio: f64,
    /// Factor on the cold probe's contribution to a fused score.
    pub cold_arm_weight: f64,
    /// Bounds the tier step of one run.
    pub run_deadline_secs: u64,
}

impl Default for TieringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            w1: DEFAULT_HITS_WEIGHT,
            w2: DEFAULT_RECENCY_WEIGHT,
            w3: DEFAULT_FEEDBACK_WEIGHT,
            w4: DEFAULT_TRUST_WEIGHT,
            tau_days: DEFAULT_TAU_DAYS,
            hits_cap: DEFAULT_HITS_CAP,
            hits_per_producer_item_day_cap: DEFAULT_HITS_PER_PRODUCER_ITEM_DAY_CAP,
            hits_per_producer_day_cap: DEFAULT_HITS_PER_PRODUCER_DAY_CAP,
            unknown_trust_ceiling: DEFAULT_UNKNOWN_TRUST_CEILING,
            promote_threshold: DEFAULT_PROMOTE_THRESHOLD,
            demote_threshold: DEFAULT_DEMOTE_THRESHOLD,
            min_residency_days: DEFAULT_MIN_RESIDENCY_DAYS,
            min_item_age_days: DEFAULT_MIN_ITEM_AGE_DAYS,
            max_promotions_per_run: DEFAULT_MAX_PROMOTIONS_PER_RUN,
            max_demotions_per_run: DEFAULT_MAX_DEMOTIONS_PER_RUN,
            window_days: DEFAULT_WINDOW_DAYS,
            usage_flush_secs: DEFAULT_USAGE_FLUSH_SECS,
            max_tracked_items: DEFAULT_MAX_TRACKED_ITEMS,
            feedback_per_producer_per_hour: DEFAULT_FEEDBACK_PER_PRODUCER_PER_HOUR,
            feedback_item_cap: DEFAULT_FEEDBACK_ITEM_CAP,
            feedback_prior_correct: DEFAULT_FEEDBACK_PRIOR,
            feedback_prior_incorrect: DEFAULT_FEEDBACK_PRIOR,
            cooldown_runs: DEFAULT_COOLDOWN_RUNS,
            cold_probe_pool: DEFAULT_COLD_PROBE_POOL,
            cold_oversample: DEFAULT_COLD_OVERSAMPLE,
            cold_max_hamming_ratio: DEFAULT_COLD_MAX_HAMMING_RATIO,
            cold_arm_weight: DEFAULT_COLD_ARM_WEIGHT,
            run_deadline_secs: DEFAULT_RUN_DEADLINE_SECS,
        }
    }
}

impl TieringConfig {
    /// Rejects values the score and the planner cannot work with, naming the
    /// key to fix.
    pub fn validate(&self) -> anyhow::Result<()> {
        let weights = [self.w1, self.w2, self.w3, self.w4];
        for (key, value) in ["w1", "w2", "w3", "w4"].into_iter().zip(weights) {
            ensure_non_negative(key, value)?;
        }
        anyhow::ensure!(
            weights.iter().sum::<f64>() > 0.0,
            "tiering.w1 to tiering.w4 must not all be 0"
        );
        anyhow::ensure!(
            self.tau_days.is_finite() && self.tau_days > 0.0,
            "tiering.tau_days must be above 0, got {}",
            self.tau_days
        );
        for (key, value) in [
            ("promote_threshold", self.promote_threshold),
            ("demote_threshold", self.demote_threshold),
        ] {
            anyhow::ensure!(
                (0.0..=1.0).contains(&value),
                "tiering.{key} must be between 0 and 1, got {value}"
            );
        }
        anyhow::ensure!(
            self.demote_threshold < self.promote_threshold,
            "tiering.demote_threshold must be below tiering.promote_threshold, got {} and {}",
            self.demote_threshold,
            self.promote_threshold
        );
        for (key, value) in [
            ("cold_max_hamming_ratio", self.cold_max_hamming_ratio),
            ("cold_arm_weight", self.cold_arm_weight),
        ] {
            anyhow::ensure!(
                value > 0.0 && value <= 1.0,
                "tiering.{key} must be above 0 and at most 1, got {value}"
            );
        }
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.unknown_trust_ceiling),
            "tiering.unknown_trust_ceiling must be between 0 and 1, got {}",
            self.unknown_trust_ceiling
        );
        ensure_non_negative("feedback_prior_correct", self.feedback_prior_correct)?;
        ensure_non_negative("feedback_prior_incorrect", self.feedback_prior_incorrect)?;
        for (key, value) in [
            ("hits_cap", u64::from(self.hits_cap)),
            (
                "hits_per_producer_item_day_cap",
                u64::from(self.hits_per_producer_item_day_cap),
            ),
            (
                "hits_per_producer_day_cap",
                u64::from(self.hits_per_producer_day_cap),
            ),
            ("window_days", u64::from(self.window_days)),
            ("usage_flush_secs", self.usage_flush_secs),
            ("max_tracked_items", self.max_tracked_items as u64),
            (
                "feedback_per_producer_per_hour",
                u64::from(self.feedback_per_producer_per_hour),
            ),
            ("feedback_item_cap", self.feedback_item_cap as u64),
            ("max_promotions_per_run", self.max_promotions_per_run as u64),
            ("max_demotions_per_run", self.max_demotions_per_run as u64),
            ("cold_probe_pool", self.cold_probe_pool as u64),
            ("cold_oversample", self.cold_oversample as u64),
            ("run_deadline_secs", self.run_deadline_secs),
        ] {
            anyhow::ensure!(value > 0, "tiering.{key} must be above 0");
        }
        anyhow::ensure!(
            self.run_deadline_secs <= MAX_RUN_DEADLINE_SECS,
            "tiering.run_deadline_secs must be at most {MAX_RUN_DEADLINE_SECS}"
        );
        Ok(())
    }
}

fn ensure_non_negative(key: &str, value: f64) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.is_finite() && value >= 0.0,
        "tiering.{key} must be 0 or more, got {value}"
    );
    Ok(())
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

    fn section(line: &str) -> String {
        format!("[tiering]\n{line}\n")
    }

    #[test]
    fn a_config_without_the_section_gets_every_default_and_is_off() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.tiering, TieringConfig::default());
        assert!(!config.tiering.enabled);
    }

    #[test]
    fn the_defaults_pass_their_own_validation() {
        // Act
        let result = TieringConfig::default().validate();

        // Assert
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn an_unknown_key_fails_loudly_naming_the_key() {
        // Act
        let error = load(&section("w9 = 1.0")).expect_err("a typo must be rejected");

        // Assert
        assert!(error.to_string().contains("w9"), "{error}");
    }

    #[test]
    fn a_value_the_planner_cannot_work_with_is_rejected_naming_the_key() {
        for (line, key) in [
            ("w1 = -0.1", "tiering.w1"),
            ("w2 = nan", "tiering.w2"),
            ("w3 = inf", "tiering.w3"),
            ("w4 = -1.0", "tiering.w4"),
            ("w1 = 0.0\nw2 = 0.0\nw3 = 0.0\nw4 = 0.0", "tiering.w1 to"),
            ("tau_days = 0.0", "tiering.tau_days"),
            ("tau_days = nan", "tiering.tau_days"),
            ("promote_threshold = 1.1", "tiering.promote_threshold"),
            ("demote_threshold = -0.1", "tiering.demote_threshold"),
            (
                "promote_threshold = 0.3\ndemote_threshold = 0.3",
                "tiering.demote_threshold must be below",
            ),
            (
                "promote_threshold = 0.2\ndemote_threshold = 0.4",
                "tiering.demote_threshold must be below",
            ),
            (
                "cold_max_hamming_ratio = 0.0",
                "tiering.cold_max_hamming_ratio",
            ),
            (
                "cold_max_hamming_ratio = 1.5",
                "tiering.cold_max_hamming_ratio",
            ),
            ("cold_arm_weight = 0.0", "tiering.cold_arm_weight"),
            ("cold_arm_weight = 1.1", "tiering.cold_arm_weight"),
            (
                "feedback_prior_correct = -1.0",
                "tiering.feedback_prior_correct",
            ),
            (
                "feedback_prior_incorrect = nan",
                "tiering.feedback_prior_incorrect",
            ),
            ("hits_cap = 0", "tiering.hits_cap"),
            (
                "hits_per_producer_item_day_cap = 0",
                "tiering.hits_per_producer_item_day_cap",
            ),
            (
                "hits_per_producer_day_cap = 0",
                "tiering.hits_per_producer_day_cap",
            ),
            (
                "unknown_trust_ceiling = 1.5",
                "tiering.unknown_trust_ceiling",
            ),
            (
                "unknown_trust_ceiling = nan",
                "tiering.unknown_trust_ceiling",
            ),
            ("window_days = 0", "tiering.window_days"),
            ("usage_flush_secs = 0", "tiering.usage_flush_secs"),
            ("max_tracked_items = 0", "tiering.max_tracked_items"),
            (
                "feedback_per_producer_per_hour = 0",
                "tiering.feedback_per_producer_per_hour",
            ),
            ("feedback_item_cap = 0", "tiering.feedback_item_cap"),
            (
                "max_promotions_per_run = 0",
                "tiering.max_promotions_per_run",
            ),
            ("max_demotions_per_run = 0", "tiering.max_demotions_per_run"),
            ("cold_probe_pool = 0", "tiering.cold_probe_pool"),
            ("cold_oversample = 0", "tiering.cold_oversample"),
            ("run_deadline_secs = 0", "tiering.run_deadline_secs"),
            ("run_deadline_secs = 86401", "tiering.run_deadline_secs"),
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
            "w1 = 0.0\nw2 = 0.0\nw3 = 0.0\nw4 = 1.0",
            "demote_threshold = 0.0",
            "promote_threshold = 1.0",
            "cold_max_hamming_ratio = 1.0",
            "cold_arm_weight = 1.0",
            "feedback_prior_correct = 0.0\nfeedback_prior_incorrect = 0.0",
            "cooldown_runs = 0",
            "hits_cap = 1",
            "run_deadline_secs = 86400",
        ] {
            // Act
            let result = load(&section(line));

            // Assert
            assert!(result.is_ok(), "{line}: {result:?}");
        }
    }

    use crate::config::docs::{page_body, read_root, TIERING_PAGE};

    const PAGE_HEADING: &str = "# Tiering";

    /// Every key of the section, read off the serialized default so a new field
    /// cannot be left out of the check.
    fn tiering_keys() -> Vec<String> {
        let value = serde_json::to_value(TieringConfig::default()).unwrap();
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

    /// The lines of the `[tiering]` table of a `liam.toml`.
    fn toml_section(text: &str) -> Vec<&str> {
        text.lines()
            .skip_while(|line| !line.starts_with("[tiering]"))
            .skip(1)
            .take_while(|line| !line.starts_with('['))
            .collect()
    }

    /// The lines of the tiering page under its title.
    fn page_section(text: &str) -> Vec<String> {
        page_body(text, PAGE_HEADING)
    }

    fn missing_from_toml(section: &[&str]) -> Vec<String> {
        tiering_keys()
            .into_iter()
            .filter(|key| !section.iter().any(|line| assigns(line, key)))
            .collect()
    }

    fn missing_from_page(section: &[String]) -> Vec<String> {
        tiering_keys()
            .into_iter()
            .filter(|key| {
                let row = format!("| `{key}` |");
                !section.iter().any(|line| line.starts_with(&row))
            })
            .collect()
    }

    #[test]
    fn the_shipped_liam_toml_carries_the_code_defaults() {
        // Arrange
        let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../liam.toml"));

        // Act
        let config = Config::load(path).expect("shipped liam.toml must parse");

        // Assert
        assert_eq!(config.tiering, TieringConfig::default());
    }

    #[test]
    fn liam_toml_documents_every_tiering_key() {
        // Arrange
        let text = read_root("liam.toml");

        // Act
        let missing = missing_from_toml(&toml_section(&text));

        // Assert
        assert!(missing.is_empty(), "liam.toml must document {missing:?}");
    }

    #[test]
    fn the_tiering_page_documents_every_tiering_key() {
        // Arrange
        let text = read_root(TIERING_PAGE);
        let section = page_section(&text);
        assert!(
            !section.is_empty(),
            "{TIERING_PAGE} needs a {PAGE_HEADING} title"
        );

        // Act
        let missing = missing_from_page(&section);

        // Assert
        assert!(
            missing.is_empty(),
            "{TIERING_PAGE} must document {missing:?}"
        );
    }

    #[test]
    fn a_section_with_a_key_removed_fails_the_check() {
        // Arrange
        let toml = read_root("liam.toml");
        let page = read_root(TIERING_PAGE);
        let toml_lines: Vec<&str> = toml_section(&toml)
            .into_iter()
            .filter(|line| !assigns(line, "cooldown_runs"))
            .collect();
        let page_lines: Vec<String> = page_section(&page)
            .into_iter()
            .filter(|line| !line.starts_with("| `cooldown_runs` |"))
            .collect();

        // Act
        let (from_toml, from_page) = (
            missing_from_toml(&toml_lines),
            missing_from_page(&page_lines),
        );

        // Assert
        assert_eq!(from_toml, vec!["cooldown_runs".to_string()]);
        assert_eq!(from_page, vec!["cooldown_runs".to_string()]);
    }
}
