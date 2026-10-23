// SPDX-License-Identifier: Apache-2.0
//! The probe suite: fixed queries with known-correct answers, scored before and
//! after a consolidation run so a run that hurts retrieval is not committed.

use std::collections::{BTreeMap, HashSet};

use liam_store::{DefaultGraph, NodeId, Query, Result};

use crate::retrieval_eval::{ndcg_at_10, recall_at_k};

/// How many hits each probe retrieves, the cutoff of the headline metrics.
const PROBE_K: usize = 10;

/// Float noise allowed when testing a drop against the margin, so a drop that
/// is exactly the margin on paper is not missed by a rounding error.
const MARGIN_EPSILON: f64 = 1e-9;

/// One query and the nodes a correct answer must contain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub name: String,
    pub query: String,
    pub relevant: Vec<NodeId>,
}

/// How one probe scored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeScore {
    pub recall: f64,
    pub ndcg: f64,
}

/// Every probe's score, by probe name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProbeReport {
    pub scores: BTreeMap<String, ProbeScore>,
}

/// Runs each probe against the store's current contents.
///
/// # Errors
///
/// Fails when the store cannot answer a query.
pub async fn run_probes(store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport> {
    let mut report = ProbeReport::default();
    for probe in probes {
        let hits = store
            .query(&Query::text(probe.query.as_str()).with_k(PROBE_K))
            .await?;
        let retrieved: Vec<String> = hits.iter().map(|hit| hit.id.as_str().to_string()).collect();
        let relevant: HashSet<&str> = probe.relevant.iter().map(NodeId::as_str).collect();
        let score = ProbeScore {
            recall: recall_at_k(&retrieved, &relevant, PROBE_K),
            ndcg: ndcg_at_10(&retrieved, &relevant),
        };
        report.scores.insert(probe.name.clone(), score);
    }
    Ok(report)
}

/// True when no probe present in both reports lost `margin` or more of its
/// recall. A drop of exactly `margin` counts as a regression, so the check
/// fails closed. A probe with no baseline has nothing to regress from.
pub fn compare(baseline: &ProbeReport, current: &ProbeReport, margin: f64) -> bool {
    !current.scores.iter().any(|(name, now)| {
        baseline.scores.get(name).is_some_and(|before| {
            let drop = before.recall - now.recall;
            drop > 0.0 && drop + MARGIN_EPSILON >= margin
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_store::{GraphConfig, NewNode};

    const DIMS: usize = 4;
    const MARGIN: f64 = 0.02;

    fn report(entries: &[(&str, f64)]) -> ProbeReport {
        ProbeReport {
            scores: entries
                .iter()
                .map(|(name, recall)| {
                    (
                        (*name).to_string(),
                        ProbeScore {
                            recall: *recall,
                            ndcg: 0.0,
                        },
                    )
                })
                .collect(),
        }
    }

    async fn open_file_store(dir: &tempfile::TempDir) -> DefaultGraph {
        let path = dir.path().join("probes.db");
        DefaultGraph::open(path.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .unwrap()
    }

    #[test]
    fn compare_flags_a_drop_larger_than_the_margin() {
        // Arrange
        let baseline = report(&[("a", 1.0), ("b", 0.5)]);
        let current = report(&[("a", 1.0), ("b", 0.4)]);

        // Act
        let ok = compare(&baseline, &current, MARGIN);

        // Assert
        assert!(!ok);
    }

    #[test]
    fn compare_passes_when_every_probe_is_unchanged_or_better() {
        // Arrange
        let baseline = report(&[("a", 0.5), ("b", 0.5)]);
        let current = report(&[("a", 0.5), ("b", 1.0)]);

        // Act
        let ok = compare(&baseline, &current, MARGIN);

        // Assert
        assert!(ok);
    }

    #[test]
    fn compare_passes_a_drop_smaller_than_the_margin() {
        // Arrange
        let baseline = report(&[("a", 0.5)]);
        let current = report(&[("a", 0.49)]);

        // Act
        let ok = compare(&baseline, &current, MARGIN);

        // Assert
        assert!(ok);
    }

    #[test]
    fn compare_treats_a_drop_of_exactly_the_margin_as_a_regression() {
        // Arrange
        let baseline = report(&[("a", 1.0)]);
        let current = report(&[("a", 0.5)]);

        // Act
        let ok = compare(&baseline, &current, 0.5);

        // Assert
        assert!(!ok);
    }

    #[test]
    fn compare_treats_a_decimal_drop_of_exactly_the_margin_as_a_regression() {
        // Arrange: 1.0 - 0.8 is 0.19999999999999996 in floating point
        let baseline = report(&[("a", 1.0)]);
        let current = report(&[("a", 0.8)]);

        // Act
        let ok = compare(&baseline, &current, 0.2);

        // Assert
        assert!(!ok);
    }

    #[test]
    fn compare_ignores_a_probe_with_no_baseline() {
        // Arrange
        let baseline = report(&[("a", 0.5)]);
        let current = report(&[("a", 0.5), ("new", 0.0)]);

        // Act
        let ok = compare(&baseline, &current, MARGIN);

        // Assert
        assert!(ok);
    }

    #[test]
    fn compare_ignores_a_probe_dropped_from_the_current_report() {
        // Arrange
        let baseline = report(&[("a", 0.5), ("gone", 1.0)]);
        let current = report(&[("a", 0.5)]);

        // Act
        let ok = compare(&baseline, &current, MARGIN);

        // Assert
        assert!(ok);
    }

    #[test]
    fn compare_passes_unchanged_scores_even_with_a_zero_margin() {
        // Arrange
        let baseline = report(&[("a", 0.5)]);

        // Act
        let ok = compare(&baseline, &baseline, 0.0);

        // Assert
        assert!(ok);
    }

    #[tokio::test]
    async fn run_probes_scores_a_real_store_against_hand_computed_values() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = open_file_store(&dir).await;
        let found = store
            .insert(NewNode::now("fact", "found", "quartz crystals grow slowly"))
            .await
            .unwrap();
        let missed = store
            .insert(NewNode::now("fact", "missed", "granite is an igneous rock"))
            .await
            .unwrap();
        let probes = [Probe {
            name: "quartz".into(),
            query: "quartz".into(),
            relevant: vec![found, missed],
        }];

        // Act
        let report = run_probes(&store, &probes).await.unwrap();

        // Assert: one of two relevant nodes at rank 1
        let score = report.scores["quartz"];
        let ideal = 1.0 + 1.0 / 3f64.log2();
        assert!((score.recall - 0.5).abs() < 1e-12);
        assert!((score.ndcg - 1.0 / ideal).abs() < 1e-12);
    }

    #[tokio::test]
    async fn run_probes_then_compare_detects_a_store_that_lost_the_relevant_nodes() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let before = open_file_store(&dir).await;
        let a = before
            .insert(NewNode::now("fact", "a", "quartz crystals grow slowly"))
            .await
            .unwrap();
        let b = before
            .insert(NewNode::now("fact", "b", "quartz veins cut the granite"))
            .await
            .unwrap();
        let probes = [Probe {
            name: "quartz".into(),
            query: "quartz".into(),
            relevant: vec![a, b],
        }];
        let baseline = run_probes(&before, &probes).await.unwrap();
        let other_dir = tempfile::tempdir().unwrap();
        let after = open_file_store(&other_dir).await;
        after
            .insert(NewNode::now("fact", "a", "quartz crystals grow slowly"))
            .await
            .unwrap();

        // Act
        let current = run_probes(&after, &probes).await.unwrap();

        // Assert
        assert!((baseline.scores["quartz"].recall - 1.0).abs() < 1e-12);
        assert!(!compare(&baseline, &current, MARGIN));
    }
}
