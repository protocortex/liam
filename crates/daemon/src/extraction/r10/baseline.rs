// SPDX-License-Identifier: Apache-2.0
//! The committed result of the real-model run and the gate derived from it.
//! The gate holds only for the prompt and model it was measured with, so a
//! change to either needs a new baseline from a reviewed run.

use std::collections::BTreeMap;

use serde::Deserialize;

/// A first run that scores below this cannot become the gate, whatever it
/// measured.
pub(super) const SANITY_FLOOR_F1: f64 = 0.5;

const COMMITTED: &str = include_str!("baseline.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Baseline {
    pub prompt_version: String,
    pub model_sha256: String,
    pub seed: u64,
    /// `None` until a reviewed run on a machine with enough memory has been
    /// committed; the gate then sits at the sanity floor.
    pub measured: Option<Measured>,
    pub gate: Gate,
}

/// What the reviewed run produced, kept as the record the gate was set from.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Measured {
    pub f1: f64,
    pub per_predicate_f1: BTreeMap<String, f64>,
    pub peak_rss_mb: u64,
    pub facts_dropped_ungrounded: usize,
}

/// Limits a later run must stay inside. Peak memory is gated by the
/// configured `ram_budget_mb`, not by a measured figure.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Gate {
    pub min_f1: f64,
    /// Grounding that rejects honest paraphrase shows up here as a rise.
    pub max_facts_dropped_ungrounded: usize,
}

impl Baseline {
    pub(super) fn committed() -> Self {
        serde_json::from_str(COMMITTED).expect("baseline.json is valid")
    }

    /// Fails with what to do when the run is not the one the baseline was
    /// measured on.
    pub(super) fn check_bound(
        &self,
        prompt_version: &str,
        model_sha256: &str,
    ) -> Result<(), String> {
        let mut stale = Vec::new();
        if self.prompt_version != prompt_version {
            stale.push(format!(
                "prompt_version is {prompt_version}, the baseline has {}",
                self.prompt_version
            ));
        }
        if self.model_sha256 != model_sha256 {
            stale.push(format!(
                "model_sha256 is {model_sha256}, the baseline has {}",
                self.model_sha256
            ));
        }
        if stale.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "the R10 baseline is out of date ({}): run the real-model tier and commit a \
                 new baseline.json",
                stale.join("; ")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExtractionConfig;
    use crate::extraction::prompt::PROMPT_VERSION;
    use crate::extraction::r10::corpus::{CORPUS_SEED, EVENT_COUNT, HAS_TITLE, LIVES_IN, WORKS_AT};
    use liam_daemon::fetch::MODEL_ASSET;

    const OTHER_SHA: &str = "ab";

    #[test]
    fn the_committed_baseline_is_bound_to_this_prompt_and_model() {
        // Arrange
        let baseline = Baseline::committed();

        // Act
        let bound = baseline.check_bound(PROMPT_VERSION, MODEL_ASSET.sha256);

        // Assert
        assert_eq!(bound, Ok(()));
        assert_eq!(baseline.seed, CORPUS_SEED);
    }

    #[test]
    fn a_changed_prompt_version_fails_with_a_clear_message() {
        // Arrange
        let baseline = Baseline::committed();

        // Act
        let message = baseline
            .check_bound("next", MODEL_ASSET.sha256)
            .unwrap_err();

        // Assert
        assert!(message.contains("prompt_version is next"), "{message}");
        assert!(message.contains("commit a new baseline.json"), "{message}");
        assert!(!message.contains("model_sha256"), "{message}");
    }

    #[test]
    fn a_changed_model_fails_with_a_clear_message() {
        // Arrange
        let baseline = Baseline::committed();

        // Act
        let message = baseline.check_bound(PROMPT_VERSION, OTHER_SHA).unwrap_err();

        // Assert
        assert!(
            message.contains(&format!("model_sha256 is {OTHER_SHA}")),
            "{message}"
        );
        assert!(!message.contains("prompt_version"), "{message}");
    }

    #[test]
    fn the_gate_sits_between_the_sanity_floor_and_the_measurement() {
        // Arrange
        let Baseline { measured, gate, .. } = Baseline::committed();

        // Assert
        assert!(gate.min_f1 >= SANITY_FLOOR_F1);
        match measured {
            Some(measured) => {
                assert!(gate.min_f1 <= measured.f1);
                assert!(measured.facts_dropped_ungrounded <= gate.max_facts_dropped_ungrounded);
            }
            None => assert_eq!(gate.min_f1, SANITY_FLOOR_F1),
        }
    }

    #[test]
    fn the_ungrounded_ceiling_is_below_the_corpus_size() {
        // Arrange
        let Baseline { gate, .. } = Baseline::committed();

        // Assert
        assert!(
            gate.max_facts_dropped_ungrounded < EVENT_COUNT,
            "a ceiling of {} cannot fail a run over {EVENT_COUNT} events",
            gate.max_facts_dropped_ungrounded
        );
    }

    #[test]
    fn the_measured_predicates_are_the_ones_the_corpus_labels() {
        // Arrange
        let baseline = Baseline::committed();
        let labelled = [LIVES_IN, WORKS_AT, HAS_TITLE];
        let measured = baseline.measured.unwrap_or_default();

        // Act
        let unknown: Vec<&String> = measured
            .per_predicate_f1
            .keys()
            .filter(|predicate| !labelled.contains(&predicate.as_str()))
            .collect();

        // Assert
        assert!(unknown.is_empty(), "{unknown:?}");
    }

    #[test]
    fn the_measured_memory_fits_the_default_budget() {
        // Arrange
        let baseline = Baseline::committed();

        // Assert
        let measured = baseline.measured.unwrap_or_default();
        assert!(measured.peak_rss_mb < ExtractionConfig::default().ram_budget_mb);
    }
}
