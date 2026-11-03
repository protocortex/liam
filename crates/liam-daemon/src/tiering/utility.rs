// SPDX-License-Identifier: Apache-2.0
//! The utility score of one lineage: a weighted mean of four terms, each in
//! [0, 1].

use std::collections::HashMap;

use liam_store::{ItemView, Millis};

use crate::config::{ExtractionConfig, TieringConfig};

/// What the score and the planner read from the configuration.
#[derive(Clone, Debug)]
pub struct TierParams {
    /// The weights of the hits, recency, feedback and trust terms.
    pub weights: [f64; 4],
    pub tau_days: f64,
    pub hits_cap: u32,
    pub window_days: u32,
    pub feedback_prior_correct: f64,
    pub feedback_prior_incorrect: f64,
    pub feedback_item_cap: usize,
    pub promote_threshold: f64,
    pub demote_threshold: f64,
    pub min_item_age_days: u32,
    pub max_promotions: usize,
    pub max_demotions: usize,
    producer_trust: HashMap<String, f64>,
    default_trust: f64,
}

impl TierParams {
    /// Takes producer trust from `[extraction.trust]`, the one place it is set.
    pub fn new(tiering: &TieringConfig, extraction: &ExtractionConfig) -> Self {
        Self {
            weights: [tiering.w1, tiering.w2, tiering.w3, tiering.w4],
            tau_days: tiering.tau_days,
            hits_cap: tiering.hits_cap,
            window_days: tiering.window_days,
            feedback_prior_correct: tiering.feedback_prior_correct,
            feedback_prior_incorrect: tiering.feedback_prior_incorrect,
            feedback_item_cap: tiering.feedback_item_cap,
            promote_threshold: tiering.promote_threshold,
            demote_threshold: tiering.demote_threshold,
            min_item_age_days: tiering.min_item_age_days,
            max_promotions: tiering.max_promotions_per_run,
            max_demotions: tiering.max_demotions_per_run,
            producer_trust: extraction.trust.clone(),
            default_trust: extraction.default_trust,
        }
    }

    fn trust_of(&self, producer: &str) -> f64 {
        self.producer_trust
            .get(producer)
            .copied()
            .unwrap_or(self.default_trust)
    }
}

/// A score and how many of the four terms it was computed from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Utility {
    pub value: f64,
    pub terms_used: usize,
}

/// The utility of `item` at `now`. Only the feedback term can be missing, and
/// then the weights of the others are scaled to sum to what they did. An item
/// nobody used scores 0 on hits, which is a measurement and not a gap.
pub fn utility(item: &ItemView, params: &TierParams, now: Millis) -> Utility {
    let terms = [
        Some(hits_term(item, params)),
        Some(recency_term(item, params, now)),
        feedback_term(item, params),
        Some(params.trust_of(&item.producer).clamp(0.0, 1.0)),
    ];
    let mut weighted = 0.0;
    let mut weight_sum = 0.0;
    let mut terms_used = 0;
    for (term, weight) in terms.into_iter().zip(params.weights) {
        if let Some(term) = term {
            weighted += weight * term;
            weight_sum += weight;
            terms_used += 1;
        }
    }
    let value = if weight_sum > 0.0 {
        weighted / weight_sum
    } else {
        0.0
    };
    Utility { value, terms_used }
}

/// Hits in the window on a log scale that reaches 1 at `hits_cap`. The
/// snapshot holds only the days inside the window.
fn hits_term(item: &ItemView, params: &TierParams) -> f64 {
    let hits: i64 = item.hits_by_day.iter().map(|(_, hits)| hits).sum();
    let reach = (1.0 + f64::from(params.hits_cap)).ln();
    ((1.0 + hits.max(0) as f64).ln() / reach).min(1.0)
}

/// Decay since the last access, or since the newest version was written when
/// nothing accessed it. Write time is transaction time, which a caller cannot
/// backdate as it can `valid_from`. An access time ahead of `now` counts as now.
fn recency_term(item: &ItemView, params: &TierParams, now: Millis) -> f64 {
    let last = item.last_access_at.unwrap_or(item.newest_tx_from);
    let elapsed_ms = (now.0 - last.0).max(0);
    let days = elapsed_ms as f64 / Millis::days(1).0 as f64;
    (-days / params.tau_days).exp()
}

/// The trust-weighted share of correct verdicts among the latest ones, over a
/// prior so that a single verdict cannot swing it to 0 or 1. `None` when there
/// is nothing to weigh.
fn feedback_term(item: &ItemView, params: &TierParams) -> Option<f64> {
    if item.feedback.is_empty() {
        return None;
    }
    let (mut correct, mut total) = (params.feedback_prior_correct, 0.0);
    for verdict in item.feedback.iter().take(params.feedback_item_cap) {
        let trust = params.trust_of(&verdict.producer);
        total += trust;
        if verdict.correct {
            correct += trust;
        }
    }
    let denominator = params.feedback_prior_correct + params.feedback_prior_incorrect + total;
    (denominator > 0.0).then(|| correct / denominator)
}

#[cfg(test)]
pub(super) mod fixtures {
    use liam_store::{CountedFeedback, ItemView, LineageKey, Millis, Tier};

    use super::*;

    pub const NOW: Millis = Millis(100 * 86_400_000);
    pub const PRODUCER: &str = "agent-a";

    pub fn params_with(weights: [f64; 4], trust: f64) -> TierParams {
        let tiering = TieringConfig {
            w1: weights[0],
            w2: weights[1],
            w3: weights[2],
            w4: weights[3],
            ..TieringConfig::default()
        };
        let mut extraction = ExtractionConfig::default();
        extraction.trust.insert(PRODUCER.to_string(), trust);
        TierParams::new(&tiering, &extraction)
    }

    /// Sets the trust of `producer` without a configuration round trip.
    pub fn trusting(params: &mut TierParams, producer: &str, trust: f64) {
        params.producer_trust.insert(producer.to_string(), trust);
    }

    pub fn item(key: &str) -> ItemView {
        ItemView {
            lineage_key: LineageKey::from_raw(key),
            newest_tx_from: Millis(NOW.0 - 60 * 86_400_000),
            hits_by_day: Vec::new(),
            last_access_at: None,
            feedback: Vec::new(),
            tier: Tier::Hot,
            changed_at: None,
            residency_until: None,
            has_vector: true,
            has_signature: true,
            producer: PRODUCER.to_string(),
        }
    }

    pub fn verdict(producer: &str, correct: bool) -> CountedFeedback {
        CountedFeedback {
            producer: producer.to_string(),
            at: NOW,
            correct,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    const EPSILON: f64 = 1e-9;
    const ONLY_HITS: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
    const ONLY_RECENCY: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
    const ONLY_FEEDBACK: [f64; 4] = [0.0, 0.0, 1.0, 0.0];
    const ONLY_TRUST: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
    const DEFAULT_CAP: f64 = 20.0;
    const DEFAULT_TAU: f64 = 30.0;

    /// An item with 5 hits, a last access 10 days ago, one correct verdict from
    /// a producer trusted 0.8 and one incorrect from one trusted 0.4, written
    /// by a producer trusted 0.6.
    fn used_item() -> ItemView {
        ItemView {
            hits_by_day: vec![(95, 3), (98, 2)],
            last_access_at: Some(Millis(NOW.0 - 10 * 86_400_000)),
            feedback: vec![verdict("high", true), verdict("low", false)],
            ..item("used")
        }
    }

    fn params(weights: [f64; 4]) -> TierParams {
        let mut params = params_with(weights, 0.6);
        params.producer_trust.insert("high".into(), 0.8);
        params.producer_trust.insert("low".into(), 0.4);
        params
    }

    fn hits_value() -> f64 {
        6.0_f64.ln() / (1.0 + DEFAULT_CAP).ln()
    }

    fn recency_value() -> f64 {
        (-10.0 / DEFAULT_TAU).exp()
    }

    /// (1 + 0.8) / (1 + 1 + 0.8 + 0.4)
    const FEEDBACK_VALUE: f64 = 1.8 / 3.2;

    #[test]
    fn each_term_alone_and_all_four_match_the_hand_computed_value() {
        for (weights, expected, terms) in [
            (ONLY_HITS, hits_value(), 4),
            (ONLY_RECENCY, recency_value(), 4),
            (ONLY_FEEDBACK, FEEDBACK_VALUE, 4),
            (ONLY_TRUST, 0.6, 4),
            (
                [0.4, 0.3, 0.2, 0.1],
                0.4 * hits_value() + 0.3 * recency_value() + 0.2 * FEEDBACK_VALUE + 0.1 * 0.6,
                4,
            ),
        ] {
            // Arrange
            let params = params(weights);

            // Act
            let score = utility(&used_item(), &params, NOW);

            // Assert
            assert!(
                (score.value - expected).abs() < EPSILON,
                "{weights:?}: {} != {expected}",
                score.value
            );
            assert_eq!(score.terms_used, terms);
        }
    }

    #[test]
    fn utility_does_not_fall_as_hits_grow() {
        // Arrange
        let params = params([0.4, 0.3, 0.2, 0.1]);
        let mut previous = f64::MIN;

        for hits in 0..=2 * DEFAULT_CAP as i64 {
            let item = ItemView {
                hits_by_day: vec![(99, hits)],
                ..used_item()
            };

            // Act
            let value = utility(&item, &params, NOW).value;

            // Assert
            assert!(value >= previous, "{hits} hits: {value} < {previous}");
            previous = value;
        }
    }

    #[test]
    fn hits_stop_counting_at_the_cap() {
        // Arrange
        let params = params(ONLY_HITS);
        let at_cap = ItemView {
            hits_by_day: vec![(99, 20)],
            ..used_item()
        };
        let past_cap = ItemView {
            hits_by_day: vec![(99, 500)],
            ..used_item()
        };

        // Act
        let (at, past) = (
            utility(&at_cap, &params, NOW).value,
            utility(&past_cap, &params, NOW).value,
        );

        // Assert
        assert!((at - 1.0).abs() < EPSILON);
        assert_eq!(past, 1.0);
    }

    #[test]
    fn an_item_with_no_feedback_renormalizes_over_the_three_other_terms() {
        // Arrange
        let weights = [0.4, 0.3, 0.2, 0.1];
        let item = ItemView {
            feedback: Vec::new(),
            ..used_item()
        };
        let expected = (0.4 * hits_value() + 0.3 * recency_value() + 0.1 * 0.6) / (0.4 + 0.3 + 0.1);

        // Act
        let score = utility(&item, &params(weights), NOW);

        // Assert
        assert!((score.value - expected).abs() < EPSILON);
        assert_eq!(score.terms_used, 3);
    }

    #[test]
    fn zero_hits_is_a_real_zero_and_not_a_missing_term() {
        // Arrange: with hits missing the score would be the recency alone
        let item = ItemView {
            hits_by_day: Vec::new(),
            ..used_item()
        };

        // Act
        let score = utility(&item, &params([1.0, 1.0, 0.0, 0.0]), NOW);

        // Assert
        assert!((score.value - recency_value() / 2.0).abs() < EPSILON);
    }

    #[test]
    fn with_only_the_feedback_weight_and_no_feedback_the_score_is_zero() {
        // Arrange
        let item = ItemView {
            feedback: Vec::new(),
            ..used_item()
        };

        // Act
        let score = utility(&item, &params(ONLY_FEEDBACK), NOW);

        // Assert
        assert_eq!(score.value, 0.0);
    }

    #[test]
    fn an_item_nobody_accessed_decays_from_its_write_time() {
        // Arrange: written 60 days before now
        let item = ItemView {
            last_access_at: None,
            ..used_item()
        };

        // Act
        let score = utility(&item, &params(ONLY_RECENCY), NOW);

        // Assert
        assert!((score.value - (-60.0 / DEFAULT_TAU).exp()).abs() < EPSILON);
    }

    #[test]
    fn an_access_time_ahead_of_now_counts_as_now() {
        // Arrange
        let item = ItemView {
            last_access_at: Some(Millis(NOW.0 + 5 * 86_400_000)),
            ..used_item()
        };

        // Act
        let score = utility(&item, &params(ONLY_RECENCY), NOW);

        // Assert
        assert_eq!(score.value, 1.0);
    }

    #[test]
    fn only_the_latest_verdicts_up_to_the_cap_are_read() {
        // Arrange: newest first, so the third is the one that falls off
        let two = ItemView {
            feedback: vec![verdict("high", true), verdict("high", true)],
            ..used_item()
        };
        let three = ItemView {
            feedback: vec![
                verdict("high", true),
                verdict("high", true),
                verdict("high", false),
            ],
            ..used_item()
        };
        let mut params = params(ONLY_FEEDBACK);

        // Act
        params.feedback_item_cap = 2;
        let (cut, whole) = (
            utility(&three, &params, NOW).value,
            utility(&two, &params, NOW).value,
        );
        params.feedback_item_cap = 3;
        let uncut = utility(&three, &params, NOW).value;

        // Assert
        assert_eq!(cut, whole);
        assert!(uncut < cut);
    }

    #[test]
    fn a_producer_the_table_does_not_name_gets_the_default_trust() {
        // Arrange
        let item = ItemView {
            producer: "stranger".into(),
            ..used_item()
        };

        // Act
        let score = utility(&item, &params(ONLY_TRUST), NOW);

        // Assert
        assert_eq!(score.value, ExtractionConfig::default().default_trust);
    }
}
