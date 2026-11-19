// SPDX-License-Identifier: Apache-2.0
//! The rules of a `recall` call: how many hits it returns, how many
//! candidates the store fuses to find them, and which kinds of row it shows.

use liam_store::{ExplainedHit, Retrieved};

/// Default and upper bound on the hits `recall` returns. WHY the cap: an
/// unbounded `k` would make one call arbitrarily slow.
const DEFAULT_RECALL_K: usize = 8;
const MAX_RECALL_K: usize = 32;

/// Fewest candidates the store fuses for `recall`, however small `k` is.
const CANDIDATE_POOL_FLOOR: usize = 16;

/// Resolve the caller's `k` into `1..=MAX_RECALL_K`, defaulting to
/// `DEFAULT_RECALL_K`. `k = 0` would return nothing for a query the store can
/// answer.
pub fn clamp_recall_k(k: Option<usize>) -> usize {
    k.unwrap_or(DEFAULT_RECALL_K).clamp(1, MAX_RECALL_K)
}

/// Candidates the store fuses before the cut to `k`: twice `k`, within the
/// floor and `MAX_RECALL_K`. The fused order depends on the pool size, so
/// narrowing it changes which hits come first.
pub(crate) fn candidate_pool(k: usize) -> usize {
    (k * 2).clamp(CANDIDATE_POOL_FLOOR, MAX_RECALL_K)
}

/// The hits `recall` shows: recorded memory and entity pages. The store hands
/// back every inferred row once asked for them, so this leaves out a summary,
/// which `recall` does not carry, and an entity whose page was never written,
/// which is a row with no text.
pub(crate) fn only_recalled(hits: Vec<ExplainedHit>) -> Vec<ExplainedHit> {
    hits.into_iter()
        .filter(|hit| match &hit.hit.retrieved {
            Retrieved::Recorded(_) => true,
            Retrieved::Inferred(page) => {
                page.summary_level.is_none() && !page.text.as_str().is_empty()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_is_twice_k_within_the_floor_and_the_recall_maximum() {
        // Arrange: k and its pool
        let cases = [(1, 16), (8, 16), (9, 18), (16, 32), (20, 32), (32, 32)];

        for (k, expected) in cases {
            // Act
            let pool = candidate_pool(k);

            // Assert
            assert_eq!(pool, expected, "pool for k={k}");
        }
    }

    #[test]
    fn clamp_recall_k_defaults_and_bounds_the_hit_count() {
        // Arrange / Act / Assert
        assert_eq!(clamp_recall_k(None), DEFAULT_RECALL_K, "absent k defaults");
        assert_eq!(clamp_recall_k(Some(4)), 4, "in-range k passes through");
        assert_eq!(
            clamp_recall_k(Some(MAX_RECALL_K)),
            MAX_RECALL_K,
            "the cap itself is allowed"
        );
        assert_eq!(
            clamp_recall_k(Some(10_000)),
            MAX_RECALL_K,
            "oversized k is capped"
        );
        assert_eq!(clamp_recall_k(Some(0)), 1, "k=0 returns one hit, not none");
    }

    use liam_store::{
        Basis, Derivation, InferredHit, InferredText, Millis, NodeId, RecordedHit, RecordedText,
    };

    fn hit(retrieved: Retrieved) -> ExplainedHit {
        ExplainedHit {
            hit: liam_store::Hit {
                id: NodeId::from_raw("h"),
                kind: "k".to_string(),
                label: "l".to_string(),
                retrieved,
                attributes: serde_json::Value::Null,
                score: 1.0,
            },
            lexical_rank: None,
            vector_rank: None,
            cold_rank: None,
            rrf: 1.0,
            confidence: 1.0,
            decay: 1.0,
            valid_from: Millis(0),
            expanded: false,
        }
    }

    fn recorded() -> Retrieved {
        Retrieved::Recorded(RecordedHit {
            content: RecordedText::for_test("said so"),
            events: Vec::new(),
            derivation: Derivation::Recorded,
        })
    }

    fn inferred(text: &str, summary_level: Option<u8>) -> Retrieved {
        Retrieved::Inferred(InferredHit {
            text: InferredText::for_test(text),
            basis: Basis::Unknown,
            derivation: Derivation::Inferred,
            summary_level,
        })
    }

    #[test]
    fn recall_keeps_memories_and_written_pages_and_drops_summaries_and_empty_rows() {
        // Arrange
        let hits = vec![
            hit(recorded()),
            hit(inferred("Kim (person)\n1970-01-01: Kim likes tea.", None)),
            hit(inferred("a summary", Some(1))),
            hit(inferred("", None)),
        ];

        // Act
        let shown = only_recalled(hits);

        // Assert
        let texts: Vec<&str> = shown.iter().map(|h| h.hit.retrieved.as_text()).collect();
        assert_eq!(
            texts,
            ["said so", "Kim (person)\n1970-01-01: Kim likes tea."]
        );
    }
}
