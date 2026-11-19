// SPDX-License-Identifier: Apache-2.0
//! The rules of a `recall` call: how many hits it returns, how many
//! candidates the store fuses to find them, and which kinds of row it shows.

use liam_store::{ExplainedHit, Retrieved};
use serde_json::Value;

use crate::entity_page;

/// Default and upper bound on the hits `recall` returns. WHY the cap: an
/// unbounded `k` would make one call arbitrarily slow.
const DEFAULT_RECALL_K: usize = 8;
const MAX_RECALL_K: usize = 32;

/// Fewest candidates the store fuses for `recall`, however small `k` is.
const CANDIDATE_POOL_FLOOR: usize = 16;

/// Most entity pages one reply adds to its recorded memories. A page never
/// takes the place of a recorded memory, so a reply holds up to `k` memories
/// plus this many pages.
pub(crate) const MAX_PAGES_IN_RECALL: usize = 2;

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

/// Whether `hit` is a page `recall` may show. The store hands back every
/// inferred row, so this leaves out a summary, which `recall` does not carry,
/// an entity whose page was never written, which is a row with no text, and a
/// page an earlier version wrote, which a model composed.
fn is_current_page(hit: &ExplainedHit) -> bool {
    match &hit.hit.retrieved {
        Retrieved::Recorded(_) => false,
        Retrieved::Inferred(page) => {
            page.summary_level.is_none()
                && !page.text.as_str().is_empty()
                && entity_page::is_stamped(&hit.hit.attributes)
        }
    }
}

/// The first `k` recorded memories in the order given, and up to
/// `MAX_PAGES_IN_RECALL` pages from `joint`, a ranking of recorded and
/// inferred rows together. A page goes after as many recorded memories as
/// `joint` puts above it, so its place depends on how it ranks against
/// memories and never on a score from a ranking of pages alone, which gives
/// the best page of any query the top score. A page is added to the memories
/// and never replaces one, so the memories are those `recall` returns without
/// pages.
pub(crate) fn with_pages(
    recorded: Vec<ExplainedHit>,
    joint: Vec<ExplainedHit>,
    k: usize,
) -> Vec<ExplainedHit> {
    let mut shown: Vec<ExplainedHit> = recorded.into_iter().take(k).collect();
    let mut recorded_above = 0;
    let mut added = 0;
    for hit in joint {
        if added == MAX_PAGES_IN_RECALL {
            break;
        }
        if is_current_page(&hit) {
            let at = (recorded_above + added).min(shown.len());
            shown.insert(at, hit);
            added += 1;
        } else if matches!(hit.hit.retrieved, Retrieved::Recorded(_)) {
            recorded_above += 1;
        }
    }
    shown
}

/// The attributes a reader sees on `hit`: a page's without the builder's marker.
pub(crate) fn shown_attributes(hit: &ExplainedHit) -> Value {
    match &hit.hit.retrieved {
        Retrieved::Inferred(_) => entity_page::unstamped(&hit.hit.attributes),
        Retrieved::Recorded(_) => hit.hit.attributes.clone(),
    }
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
                attributes: entity_page::stamped(Value::Null),
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

    fn scored(mut hit: ExplainedHit, id: &str) -> ExplainedHit {
        hit.hit.id = NodeId::from_raw(id);
        hit
    }

    fn page(id: &str) -> ExplainedHit {
        scored(hit(inferred("Kim (person)\n1970-01-01: tea.", None)), id)
    }

    fn memory(id: &str) -> ExplainedHit {
        scored(hit(recorded()), id)
    }

    fn ids(hits: &[ExplainedHit]) -> Vec<&str> {
        hits.iter().map(|h| h.hit.id.as_str()).collect()
    }

    #[test]
    fn only_current_pages_are_added_never_summaries_empty_rows_legacy_pages_or_memories() {
        // Arrange
        let mut legacy = scored(hit(inferred("old profile", None)), "legacy");
        legacy.hit.attributes = serde_json::json!({});
        let joint = vec![
            legacy,
            scored(hit(inferred("a summary", Some(1))), "summary"),
            scored(hit(inferred("", None)), "empty"),
            page("page"),
        ];

        // Act
        let shown = with_pages(vec![memory("m")], joint, 8);

        // Assert
        assert_eq!(ids(&shown), ["page", "m"]);
    }

    #[test]
    fn a_page_goes_after_the_memories_the_joint_ranking_puts_above_it() {
        // Arrange
        let recorded = vec![memory("a"), memory("b"), memory("c")];
        let joint = vec![memory("a"), page("p0"), memory("b"), memory("c")];

        // Act
        let shown = with_pages(recorded, joint, 8);

        // Assert
        assert_eq!(ids(&shown), ["a", "p0", "b", "c"]);
    }

    #[test]
    fn a_page_the_joint_ranking_puts_first_goes_first_and_one_ranked_last_goes_last() {
        // Arrange
        let recorded = vec![memory("a"), memory("b")];
        let joint = vec![page("hi"), memory("a"), memory("b"), page("lo")];

        // Act
        let shown = with_pages(recorded, joint, 8);

        // Assert
        assert_eq!(ids(&shown), ["hi", "a", "b", "lo"]);
    }

    #[test]
    fn at_most_the_page_cap_is_added_and_the_best_pages_are_the_ones_kept() {
        // Arrange
        let joint: Vec<ExplainedHit> = (0..=MAX_PAGES_IN_RECALL)
            .map(|n| page(&format!("p{n}")))
            .collect();

        // Act
        let shown = with_pages(vec![memory("m")], joint, 8);

        // Assert
        assert_eq!(ids(&shown), ["p0", "p1", "m"]);
    }

    #[test]
    fn pages_never_push_a_memory_out_of_the_first_k_and_the_memories_keep_their_order() {
        // Arrange
        let recorded = vec![memory("a"), memory("b"), memory("c")];
        let joint = vec![
            page("p0"),
            page("p1"),
            memory("a"),
            memory("b"),
            memory("c"),
        ];

        // Act
        let shown = with_pages(recorded, joint, 2);

        // Assert
        assert_eq!(ids(&shown), ["p0", "p1", "a", "b"]);
    }

    #[test]
    fn a_page_the_joint_ranking_puts_below_every_memory_is_still_added_after_them() {
        // Arrange: the joint ranking holds more memories than the first k shown
        let recorded = vec![memory("a")];
        let joint = vec![memory("a"), memory("b"), memory("c"), page("p")];

        // Act
        let shown = with_pages(recorded, joint, 8);

        // Assert
        assert_eq!(ids(&shown), ["a", "p"]);
    }

    #[test]
    fn a_page_is_shown_without_the_builder_marker_and_a_memory_keeps_its_attributes() {
        // Arrange
        let mut recorded = memory("m");
        recorded.hit.attributes = entity_page::stamped(serde_json::json!({"k": 1}));
        let mut stamped = page("p");
        stamped.hit.attributes = entity_page::stamped(serde_json::json!({"k": 2}));

        // Act
        let (memory_view, page_view) = (shown_attributes(&recorded), shown_attributes(&stamped));

        // Assert
        assert_eq!(
            memory_view,
            entity_page::stamped(serde_json::json!({"k": 1}))
        );
        assert_eq!(page_view, serde_json::json!({"k": 2}));
    }
}
