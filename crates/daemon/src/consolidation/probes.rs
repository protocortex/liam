// SPDX-License-Identifier: Apache-2.0
//! The probe suite: fixed queries with known-correct answers, scored before and
//! after a consolidation run so a run that hurts retrieval is not committed.

use std::collections::{BTreeMap, HashSet};

use liam_store::{DefaultGraph, NodeId, Query};
use serde::{Deserialize, Serialize};

use crate::retrieval_eval::{ndcg_at_10, recall_at_k};

/// How many hits each probe retrieves, the cutoff of the headline metrics.
const PROBE_K: usize = 10;

/// Float noise allowed when testing a drop against the margin, so a drop that
/// is exactly the margin on paper is not missed by a rounding error.
const MARGIN_EPSILON: f64 = 1e-9;

/// One query and the nodes a correct answer must contain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Probe {
    pub name: String,
    pub query: String,
    pub relevant: Vec<NodeId>,
    /// When set, the vector arm runs too; otherwise the probe is lexical only.
    pub embedding: Option<Vec<f32>>,
}

/// Why a probe run failed.
#[derive(Debug)]
pub enum ProbeError {
    /// Two probes share a name, so one score would silently replace the other.
    DuplicateName(String),
    Store(liam_store::Error),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateName(name) => write!(f, "probe name {name:?} is used more than once"),
            Self::Store(err) => write!(f, "the store could not answer a probe: {err}"),
        }
    }
}

impl std::error::Error for ProbeError {}

impl From<liam_store::Error> for ProbeError {
    fn from(err: liam_store::Error) -> Self {
        Self::Store(err)
    }
}

/// How one probe scored.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProbeScore {
    pub recall: f64,
    pub ndcg: f64,
}

/// Every probe's score, by probe name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub scores: BTreeMap<String, ProbeScore>,
}

/// Runs each probe against the store's current contents.
///
/// # Errors
///
/// Fails on a duplicate probe name, or when the store cannot answer a query.
pub async fn run_probes(store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
    let mut report = ProbeReport::default();
    for probe in probes {
        if report.scores.contains_key(&probe.name) {
            return Err(ProbeError::DuplicateName(probe.name.clone()));
        }
        let mut query = Query::text(probe.query.as_str()).with_k(PROBE_K);
        if let Some(embedding) = &probe.embedding {
            query = query.with_embedding(embedding.clone());
        }
        let hits = store.query(&query).await?;
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

/// True when the run kept every baseline probe within `margin` of its recall.
/// A drop of exactly `margin` counts as a regression, so the check fails
/// closed. So does anything that leaves nothing to trust: an empty baseline or
/// current report, a baseline probe missing from the current one, or a
/// non-finite score or margin. A probe with no baseline is not checked.
pub fn compare(baseline: &ProbeReport, current: &ProbeReport, margin: f64) -> bool {
    if baseline.scores.is_empty() || current.scores.is_empty() || !margin.is_finite() {
        return false;
    }
    baseline.scores.iter().all(|(name, before)| {
        current.scores.get(name).is_some_and(|now| {
            let drop = before.recall - now.recall;
            drop.is_finite() && !(drop > 0.0 && drop + MARGIN_EPSILON >= margin)
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
    fn compare_flags_a_baseline_probe_missing_from_the_current_report() {
        // Arrange
        let baseline = report(&[("a", 0.5), ("gone", 1.0)]);
        let current = report(&[("a", 0.5)]);

        // Act
        let ok = compare(&baseline, &current, MARGIN);

        // Assert
        assert!(!ok);
    }

    #[test]
    fn compare_flags_an_empty_baseline() {
        // Arrange
        let current = report(&[("a", 1.0)]);

        // Act
        let ok = compare(&ProbeReport::default(), &current, MARGIN);

        // Assert
        assert!(!ok);
    }

    #[test]
    fn compare_flags_an_empty_current_report() {
        // Arrange
        let baseline = report(&[("a", 1.0)]);

        // Act
        let ok = compare(&baseline, &ProbeReport::default(), MARGIN);

        // Assert
        assert!(!ok);
    }

    #[test]
    fn compare_flags_a_non_finite_score() {
        // Arrange
        let baseline = report(&[("a", 0.5)]);
        let current = report(&[("a", f64::NAN)]);

        // Act
        let ok = compare(&baseline, &current, MARGIN);

        // Assert
        assert!(!ok);
    }

    #[test]
    fn compare_flags_a_non_finite_margin() {
        // Arrange
        let baseline = report(&[("a", 0.5)]);

        // Act
        let ok = compare(&baseline, &baseline, f64::NAN);

        // Assert
        assert!(!ok);
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
            embedding: None,
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
            embedding: None,
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

    #[tokio::test]
    async fn run_probes_rejects_a_duplicate_probe_name() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = open_file_store(&dir).await;
        let probe = Probe {
            name: "same".into(),
            query: "quartz".into(),
            relevant: Vec::new(),
            embedding: None,
        };
        let probes = [probe.clone(), probe];

        // Act
        let result = run_probes(&store, &probes).await;

        // Assert
        assert!(matches!(result, Err(ProbeError::DuplicateName(name)) if name == "same"));
    }

    #[tokio::test]
    async fn run_probes_uses_the_embedding_when_present() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = open_file_store(&dir).await;
        let near = store
            .insert(NewNode::now("fact", "near", "alpha").with_embedding(vec![1.0, 0.0, 0.0, 0.0]))
            .await
            .unwrap();
        let far = store
            .insert(NewNode::now("fact", "far", "beta").with_embedding(vec![0.0, 1.0, 0.0, 0.0]))
            .await
            .unwrap();
        let probe = |name: &str, relevant: &NodeId, embedding: Option<Vec<f32>>| Probe {
            name: name.into(),
            query: "no lexical match".into(),
            relevant: vec![relevant.clone()],
            embedding,
        };
        let probes = [
            probe("near", &near, Some(vec![1.0, 0.0, 0.0, 0.0])),
            probe("far", &far, Some(vec![1.0, 0.0, 0.0, 0.0])),
            probe("lexical", &near, None),
        ];

        // Act
        let report = run_probes(&store, &probes).await.unwrap();

        // Assert
        assert!((report.scores["near"].ndcg - 1.0).abs() < 1e-12);
        assert!(report.scores["far"].ndcg < 1.0);
        assert!(report.scores["lexical"].recall.abs() < 1e-12);
    }
}
