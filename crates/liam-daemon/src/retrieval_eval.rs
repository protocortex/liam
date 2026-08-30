// SPDX-License-Identifier: Apache-2.0
//! Retrieval-quality benchmark for `Graph::query`/`query_core`.
//!
//! Where `eval.rs` scores whether the LLM's ANSWER stays grounded, this module
//! scores whether RETRIEVAL surfaces the right memories in the first place:
//! precision, recall, and reciprocal rank over a fixed corpus and question
//! set, broken down by the eight query features the pipeline supports:
//! `Lexical`, `VectorSemantic`, `GraphExpansion`, `KindFilter`, `ScopeFilter`,
//! `Decay`, `Confidence`, `AsOf` (see `Category`).
//!
//! # Two tiers
//!
//! - **Text-only, always-on:** seeds the corpus with no embedding at all and
//!   runs every category except `VectorSemantic` (there is nothing for a
//!   vector arm to rank without one). This is a non-`#[ignore]`d
//!   `#[tokio::test]`, so it runs under plain `cargo test -p liam-daemon` and
//!   the existing CI `test` job, with no model download. Added in WU-3.
//! - **Real-embedder, gated + `#[ignore]`d:** loads the actual local embedder
//!   behind `#[cfg(feature = "local")]` and scores `VectorSemantic` too. Added
//!   in WU-4, which documents the exact run command in that module's doc
//!   comment; TODO(WU-4): the full command belongs here once that module
//!   exists (see `eval.rs`'s `# llama.cpp baseline` style for the convention
//!   to follow).
//!
//! # Metrics
//!
//! `precision_at_k` and `recall_at_k` at k=4 and k=8 (the pipeline's typical
//! result-set sizes), plus `reciprocal_rank` (unbounded: the first hit's rank
//! is what matters, not whether it falls inside a fixed window). All three
//! are pure functions over a retrieved-id list and a relevant-id set, with no
//! `Graph`/`Query` dependency, so they are unit-tested directly below without
//! opening a store.
//!
//! This is a **scored baseline, not a CI gate**: there are no
//! quality-threshold assertions on the metric values themselves. What IS
//! asserted (in WU-3/WU-4's harness tests) is structural: the harness does
//! not panic, every metric value lands in `[0, 1]`, every non-excluded
//! question produces a result, and each category's fixtures are internally
//! consistent. The numbers themselves are printed via `format_report` for a
//! maintainer to read and, when they change meaningfully, hand-transcribe
//! into a dated table here, the same convention `eval.rs` uses for its own
//! baselines.

use std::collections::HashSet;

/// The unique ids among the first `k` entries of `retrieved`. WHY a shared
/// helper: both `precision_at_k` and `recall_at_k` need this exact
/// clamp-then-dedup step (a repeated id in the raw list must count once, not
/// once per occurrence), and duplicating it would let the two drift apart.
/// `reciprocal_rank` has no `k` to clamp to, so it does not use this.
fn top_k_unique(retrieved: &[String], k: usize) -> HashSet<&str> {
    retrieved.iter().take(k).map(String::as_str).collect()
}

/// Fraction of the top-`k` retrieved ids that are relevant. Denominator is
/// `k` itself (not the number of ids actually retrieved), the standard
/// precision@k convention: a short retrieved list is not rewarded for having
/// fewer slots to be wrong in. `k == 0` returns `0.0` rather than dividing by
/// zero.
pub fn precision_at_k(retrieved: &[String], relevant: &HashSet<&str>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let top = top_k_unique(retrieved, k);
    let hits = top.iter().filter(|id| relevant.contains(*id)).count();
    hits as f64 / k as f64
}

/// Fraction of `relevant` that appears anywhere in the top-`k` retrieved ids.
/// Denominator is `relevant.len()`, so an empty `relevant` set (nothing to
/// find) and `k == 0` (nowhere to look) both short-circuit to `0.0` instead
/// of dividing by zero.
pub fn recall_at_k(retrieved: &[String], relevant: &HashSet<&str>, k: usize) -> f64 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }
    let top = top_k_unique(retrieved, k);
    let hits = relevant.iter().filter(|id| top.contains(*id)).count();
    hits as f64 / relevant.len() as f64
}

/// `1 / rank` of the first relevant id in `retrieved` (1-indexed), or `0.0`
/// if none of `relevant` appears anywhere in `retrieved`. Unbounded by `k`:
/// what matters is how far down the full list the first hit sits.
pub fn reciprocal_rank(retrieved: &[String], relevant: &HashSet<&str>) -> f64 {
    retrieved
        .iter()
        .position(|id| relevant.contains(id.as_str()))
        .map_or(0.0, |idx| 1.0 / (idx + 1) as f64)
}

/// The eight query features the retrieval benchmark scores. WU-2 extends this
/// module with `Fact`/`Question`/`CORPUS`/`QUESTIONS`, which reference this
/// enum; this WU only needs the enum itself and a stable iteration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Lexical,
    VectorSemantic,
    GraphExpansion,
    KindFilter,
    ScopeFilter,
    Decay,
    Confidence,
    AsOf,
}

impl Category {
    /// All eight variants, in the order `format_report` prints them. A plain
    /// array keeps this dependency-free; there is no need for an external
    /// enum-iteration crate over eight fixed, known-at-compile-time variants.
    pub const ALL: [Category; 8] = [
        Category::Lexical,
        Category::VectorSemantic,
        Category::GraphExpansion,
        Category::KindFilter,
        Category::ScopeFilter,
        Category::Decay,
        Category::Confidence,
        Category::AsOf,
    ];

    /// The name printed in the report; matches the variant's own name so a
    /// reader can grep this file for what a row means.
    fn name(self) -> &'static str {
        match self {
            Category::Lexical => "Lexical",
            Category::VectorSemantic => "VectorSemantic",
            Category::GraphExpansion => "GraphExpansion",
            Category::KindFilter => "KindFilter",
            Category::ScopeFilter => "ScopeFilter",
            Category::Decay => "Decay",
            Category::Confidence => "Confidence",
            Category::AsOf => "AsOf",
        }
    }
}

/// Precision/recall/MRR for one category, once at least one question in it
/// was actually scored.
#[derive(Debug, Clone, Copy)]
pub struct CategoryMetrics {
    pub precision_at_4: f64,
    pub precision_at_8: f64,
    pub recall_at_4: f64,
    pub recall_at_8: f64,
    pub mrr: f64,
}

/// One category's row in the report. `metrics` is `None` when `scored == 0`:
/// the text-only tier never runs `VectorSemantic`, and printing `0.0` for a
/// metric that never ran would read as "the pipeline scored zero" instead of
/// "this category was not exercised this run", which is a different fact.
#[derive(Debug, Clone, Copy)]
pub struct CategoryScore {
    pub category: Category,
    pub scored: usize,
    pub metrics: Option<CategoryMetrics>,
}

/// Renders one row per `CategoryScore`, in the order given, with column
/// headers for the count and each metric. Pure and synchronous: it takes
/// pre-computed scores, so the harness tests can assert on the exact text
/// without opening a store.
pub fn format_report(scores: &[CategoryScore]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    // `write!` into a `String` never fails, so these `let _ =` discard a
    // `Result` that is never actually an error here.
    let _ = writeln!(
        out,
        "{:<16}{:>4}  {:>11}{:>13}{:>10}{:>10}{:>7}",
        "category", "n", "precision@4", "precision@8", "recall@4", "recall@8", "MRR"
    );
    for score in scores {
        match score.metrics {
            Some(m) => {
                let _ = writeln!(
                    out,
                    "{:<16}{:>4}  {:>11.3}{:>13.3}{:>10.3}{:>10.3}{:>7.3}",
                    score.category.name(),
                    score.scored,
                    m.precision_at_4,
                    m.precision_at_8,
                    m.recall_at_4,
                    m.recall_at_8,
                    m.mrr,
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "{:<16}{:>4}  not run this tier",
                    score.category.name(),
                    score.scored,
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set<'a>(items: &[&'a str]) -> HashSet<&'a str> {
        items.iter().copied().collect()
    }

    #[test]
    fn no_relevant_items_scores_zero_without_panicking() {
        // Arrange
        let retrieved = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let relevant = set(&[]);

        // Act / Assert
        assert_eq!(precision_at_k(&retrieved, &relevant, 4), 0.0);
        assert_eq!(recall_at_k(&retrieved, &relevant, 4), 0.0);
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 0.0);
    }

    #[test]
    fn empty_retrieved_list_scores_zero_without_panicking() {
        // Arrange
        let retrieved: Vec<String> = vec![];
        let relevant = set(&["a"]);

        // Act / Assert
        assert_eq!(precision_at_k(&retrieved, &relevant, 4), 0.0);
        assert_eq!(recall_at_k(&retrieved, &relevant, 4), 0.0);
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 0.0);
    }

    #[test]
    fn single_relevant_item_ranked_first_scores_perfectly() {
        // Arrange
        let retrieved = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let relevant = set(&["a"]);

        // Act / Assert: recall@k is 1.0 for any k >= 1 once the only
        // relevant item is first.
        assert_eq!(recall_at_k(&retrieved, &relevant, 1), 1.0);
        assert_eq!(recall_at_k(&retrieved, &relevant, 4), 1.0);
        assert_eq!(reciprocal_rank(&retrieved, &relevant), 1.0);
    }

    #[test]
    fn relevant_item_at_exactly_position_k_counts_inclusive() {
        // Arrange: relevant item at 1-indexed position 3.
        let retrieved = vec!["x".to_string(), "y".to_string(), "a".to_string()];
        let relevant = set(&["a"]);

        // Act
        let recall = recall_at_k(&retrieved, &relevant, 3);

        // Assert
        assert_eq!(recall, 1.0, "position k must count as inside the window");
    }

    #[test]
    fn relevant_item_just_past_position_k_does_not_count() {
        // Arrange: relevant item at 1-indexed position 4, k stays at 3.
        let retrieved = vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
            "a".to_string(),
        ];
        let relevant = set(&["a"]);

        // Act
        let recall = recall_at_k(&retrieved, &relevant, 3);

        // Assert
        assert_eq!(recall, 0.0, "position k+1 must fall outside the window");
    }

    #[test]
    fn precision_at_k_reflects_the_exact_fraction_retrieved() {
        // Arrange: 2 of 4 relevant items retrieved within the first k=4.
        let retrieved = vec![
            "a".to_string(),
            "x".to_string(),
            "b".to_string(),
            "y".to_string(),
        ];
        let relevant = set(&["a", "b", "c", "d"]);

        // Act
        let precision = precision_at_k(&retrieved, &relevant, 4);

        // Assert
        assert_eq!(precision, 0.5);
    }

    #[test]
    fn reciprocal_rank_pins_the_fractional_formula_at_rank_three() {
        // Arrange: the only relevant item sits at 1-indexed rank 3.
        let retrieved = vec!["x".to_string(), "y".to_string(), "a".to_string()];
        let relevant = set(&["a"]);

        // Act
        let rr = reciprocal_rank(&retrieved, &relevant);

        // Assert
        assert_eq!(rr, 1.0 / 3.0);
    }

    #[test]
    fn reciprocal_rank_is_zero_when_lists_never_overlap() {
        // Arrange: both lists non-empty, but no id appears in both.
        let retrieved = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let relevant = set(&["a", "b"]);

        // Act
        let rr = reciprocal_rank(&retrieved, &relevant);

        // Assert
        assert_eq!(rr, 0.0);
    }

    #[test]
    fn k_zero_scores_zero_without_panicking_or_nan() {
        // Arrange
        let retrieved = vec!["a".to_string(), "b".to_string()];
        let relevant = set(&["a"]);

        // Act
        let precision = precision_at_k(&retrieved, &relevant, 0);
        let recall = recall_at_k(&retrieved, &relevant, 0);

        // Assert
        assert_eq!(precision, 0.0);
        assert!(precision.is_finite());
        assert_eq!(recall, 0.0);
        assert!(recall.is_finite());
    }

    #[test]
    fn precision_at_k_does_not_double_count_a_duplicate_id() {
        // Arrange: "a" appears twice within the first k=2 raw entries.
        let retrieved = vec!["a".to_string(), "a".to_string(), "b".to_string()];
        let relevant = set(&["a"]);

        // Act
        let precision = precision_at_k(&retrieved, &relevant, 2);

        // Assert: one unique relevant hit over a denominator of k=2, not the
        // 1.0 a naive per-occurrence count would produce.
        assert_eq!(precision, 0.5);
    }

    fn all_category_scores() -> Vec<CategoryScore> {
        Category::ALL
            .iter()
            .map(|&category| CategoryScore {
                category,
                scored: 5,
                metrics: Some(CategoryMetrics {
                    precision_at_4: 0.8,
                    precision_at_8: 0.75,
                    recall_at_4: 0.9,
                    recall_at_8: 0.95,
                    mrr: 0.833,
                }),
            })
            .collect()
    }

    #[test]
    fn format_report_names_every_category_and_every_metric_column() {
        // Arrange
        let scores = all_category_scores();

        // Act
        let report = format_report(&scores);

        // Assert
        for category in Category::ALL {
            assert!(
                report.contains(category.name()),
                "report missing category {category:?}: {report}"
            );
        }
        assert!(report.contains("precision"), "{report}");
        assert!(report.contains("recall"), "{report}");
        assert!(report.contains("MRR"), "{report}");
    }

    #[test]
    fn format_report_marks_a_zero_question_category_as_not_run() {
        // Arrange
        let scores = vec![CategoryScore {
            category: Category::VectorSemantic,
            scored: 0,
            metrics: None,
        }];

        // Act
        let report = format_report(&scores);

        // Assert
        assert!(
            report.to_lowercase().contains("not run"),
            "a zero-question category must read as not-run, not a misleading 0.0: {report}"
        );
        assert!(!report.contains("0.000"), "{report}");
    }
}
