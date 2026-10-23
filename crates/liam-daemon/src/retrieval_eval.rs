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
//!   the existing CI `test` job, with no model download.
//! - **Real-embedder, `#[ignore]`d:** `run_retrieval_eval` loads the actual
//!   local embedder (needs `--features local`; other builds return an error)
//!   and scores `VectorSemantic` too. `evaluate_embedder` is the same run
//!   over any `Embedder`, which the mock-embedder test uses. Downloads model
//!   weights on first run and is slow, like `eval.rs`'s own gated tier, so
//!   its test is `#[ignore]`d rather than part of plain
//!   `cargo test -p liam-daemon`:
//!
//!   ```text
//!   cargo test -p liam-daemon --features local -- --ignored --nocapture retrieval
//!   ```
//!
//!   Env override: `LIAM_RETRIEVAL_EVAL_MODEL`, mirroring `eval.rs`'s
//!   `LIAM_EVAL_MODEL` for local A/B runs against a different embedding
//!   model.
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
//! Two headline metrics at k=10 sit beside them: `recall_at_k` at 10
//! (`recall@10`) and `ndcg_at_10` (`nDCG@10`, binary relevance). The
//! real-embedder tier also prints an embed timing line (call count, item
//! count, mean call latency, throughput). One untimed warm-up `embed` call
//! runs first, so model initialisation is excluded from those timings.
//!
//! The text-only tier IS a CI gate on quality: its per-category recall@10
//! floors (see "Text-only baseline" below) fail the build if retrieval
//! regresses, which is safe because that tier is deterministic. The
//! real-embedder tier is a scored baseline only, with no assertions on its
//! metric values. Both tiers also assert structure: the harness does not
//! panic, every metric value lands in `[0, 1]`, every non-excluded question
//! produces a result, and each category's fixtures are internally consistent
//! (labels exist, every answer appears in its fact, Decay and AsOf pairs are
//! ordered in time). The numbers are printed via `format_report` for a
//! maintainer to read and, when they change meaningfully, hand-transcribe
//! into a dated table here, the same convention `eval.rs` uses for its own
//! baselines.
//!
//! # Golden set
//!
//! 80 questions, 10 per category, over a synthetic corpus of one fictional
//! company, kept in the `golden_set` submodule. LIAM is pre-release and has
//! no real usage or query logs, so this set is a synthetic stand-in for real
//! usage, to be replaced or extended once actual users exist. It stays small
//! because every fixture is hand-written and hand-checked. Each question
//! carries an `answer` phrase that a test asserts appears in its expected
//! fact, so a relevance label is checked against the text. Question names
//! and corpus labels are stable ids that reports quote: add new ones, never
//! rename.
//!
//! # Text-only baseline (2026-10-23, 70 questions, the always-on tier)
//!
//! | category       | n  | precision@4 | precision@8 | recall@4 | recall@8 | recall@10 | nDCG@10 | MRR   |
//! |----------------|----|-------------|-------------|----------|----------|-----------|---------|-------|
//! | Lexical        | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | GraphExpansion | 10 | 0.350       | 0.212       | 0.700    | 0.850    | 0.900     | 0.831   | 1.000 |
//! | KindFilter     | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | ScopeFilter    | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | Decay          | 10 | 0.175       | 0.125       | 0.700    | 1.000    | 1.000     | 0.527   | 0.376 |
//! | Confidence     | 10 | 0.075       | 0.087       | 0.300    | 0.700    | 0.900     | 0.377   | 0.228 |
//! | AsOf           | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//!
//! The same tier on the pre-expansion 40-question set (2026-10-23, 35
//! questions scored, re-run against the earlier corpus):
//!
//! | category       | n | precision@4 | precision@8 | recall@4 | recall@8 | recall@10 | nDCG@10 | MRR   |
//! |----------------|---|-------------|-------------|----------|----------|-----------|---------|-------|
//! | Lexical        | 5 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | GraphExpansion | 5 | 0.450       | 0.225       | 0.900    | 0.900    | 0.900     | 0.882   | 1.000 |
//! | KindFilter     | 5 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | ScopeFilter    | 5 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | Decay          | 5 | 0.100       | 0.125       | 0.400    | 1.000    | 1.000     | 0.468   | 0.302 |
//! | Confidence     | 5 | 0.050       | 0.050       | 0.200    | 0.400    | 0.800     | 0.289   | 0.150 |
//! | AsOf           | 5 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//!
//! The text-only tier is deterministic, so its recall@10 per category is
//! asserted against floors (`TEXT_ONLY_RECALL_AT_10_FLOORS`) set to the
//! measured values of the 80-question run above; every floor except
//! `Confidence` also equals the 40-question value, and `Confidence` rose from
//! 0.8 to 0.9 as the set grew. A drop in any category fails the build.
//!
//! # Real-embedder baseline (2026-08-30, 40-question set, Apple M1 Pro,
//! # macOS 15.7.5, Qwen/Qwen3-Embedding-0.6B, 768 dims, `cargo test --release
//! # ... --ignored --nocapture retrieval`)
//!
//! This table predates the 80-question golden set and is NOT recomputed: it
//! needs a model download and a manual run. Treat it as historical until the
//! next real-embedder run replaces it.
//!
//! | category       | n | precision@4 | precision@8 | recall@4 | recall@8 | recall@10 | nDCG@10 | MRR   |
//! |----------------|---|-------------|-------------|----------|----------|-----------|---------|-------|
//! | Lexical        | 5 | 0.250       | 0.125       | 1.000    | 1.000    | -         | -       | 1.000 |
//! | VectorSemantic | 5 | 0.250       | 0.125       | 1.000    | 1.000    | -         | -       | 1.000 |
//! | GraphExpansion | 5 | 0.450       | 0.225       | 0.900    | 0.900    | -         | -       | 1.000 |
//! | KindFilter     | 5 | 0.250       | 0.125       | 1.000    | 1.000    | -         | -       | 1.000 |
//! | ScopeFilter    | 5 | 0.250       | 0.125       | 1.000    | 1.000    | -         | -       | 1.000 |
//! | Decay          | 5 | 0.100       | 0.125       | 0.400    | 1.000    | -         | -       | 0.302 |
//! | Confidence     | 5 | 0.050       | 0.050       | 0.200    | 0.400    | -         | -       | 0.150 |
//! | AsOf           | 5 | 0.250       | 0.125       | 1.000    | 1.000    | -         | -       | 1.000 |
//!
//! The recall@10 and nDCG@10 columns were added after this baseline was
//! taken, so they read `-` until the next real-embedder run fills them in.
//!
//! `VectorSemantic` (the category this tier adds over the text-only one)
//! lands at a perfect 1.000 recall/MRR on this small, single-narrative
//! corpus: every paraphrase question's expected fact is the sole real
//! semantic match in the corpus for that paraphrase, so once the real
//! embedder is in the loop there is no other candidate for it to lose rank
//! to. `Confidence` is the harness's weakest category here, which is a
//! property of the small fixed corpus and `HARNESS_K` (every fact is
//! observable, so a competing lower-confidence version of the same subject
//! can outrank the expected one), not a regression to chase.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use liam_model::{Embedder, TextKind};
use liam_store::{DefaultGraph, ExplainedHit, Millis, NewEdge, NodeId, Query, Result};

use serde::Serialize;

use crate::config::Config;
#[cfg(feature = "local")]
use crate::models;

mod golden_set;
#[cfg(test)]
mod scale_probe;
use golden_set::{CORPUS, QUESTIONS};

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

/// The cutoff for the recall@10 and `ndcg_at_10` headline metrics.
const HEADLINE_K: usize = 10;

/// Rank discount for a 0-indexed position: `1 / log2(rank + 1)` with rank
/// 1-indexed.
fn rank_discount(index: usize) -> f64 {
    1.0 / ((index + 2) as f64).log2()
}

/// Normalised discounted cumulative gain over the top 10, with binary
/// relevance. A repeated id earns its gain once, at its first position, but
/// still occupies a rank, matching how the sibling metrics dedupe. The ideal
/// ranking puts every relevant id (up to 10) first; an empty `relevant` set
/// has no ideal ranking, so it returns `0.0` instead of dividing by zero.
pub fn ndcg_at_10(retrieved: &[String], relevant: &HashSet<&str>) -> f64 {
    let ideal_hits = relevant.len().min(HEADLINE_K);
    if ideal_hits == 0 {
        return 0.0;
    }
    let mut seen: HashSet<&str> = HashSet::new();
    let dcg: f64 = retrieved
        .iter()
        .take(HEADLINE_K)
        .enumerate()
        .filter(|(_, id)| relevant.contains(id.as_str()) && seen.insert(id.as_str()))
        .map(|(index, _)| rank_discount(index))
        .sum();
    let ideal_dcg: f64 = (0..ideal_hits).map(rank_discount).sum();
    dcg / ideal_dcg
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

/// The eight query features the retrieval benchmark scores, with a stable
/// iteration order for the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CategoryMetrics {
    pub precision_at_4: f64,
    pub precision_at_8: f64,
    pub recall_at_4: f64,
    pub recall_at_8: f64,
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
}

/// One question's metrics, kept so a regression can be traced to the
/// question rather than hidden inside a category mean.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct QuestionScore {
    pub name: &'static str,
    pub category: Category,
    pub metrics: CategoryMetrics,
}

/// One category's row in the report. `metrics` is `None` when `scored == 0`:
/// the text-only tier never runs `VectorSemantic`, and printing `0.0` for a
/// metric that never ran would read as "the pipeline scored zero" instead of
/// "this category was not exercised this run", which is a different fact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
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
        "{:<16}{:>4}  {:>11}{:>13}{:>10}{:>10}{:>11}{:>10}{:>7}",
        "category",
        "n",
        "precision@4",
        "precision@8",
        "recall@4",
        "recall@8",
        "recall@10",
        "nDCG@10",
        "MRR"
    );
    for score in scores {
        match score.metrics {
            Some(m) => {
                let _ = writeln!(
                    out,
                    "{:<16}{:>4}  {:>11.3}{:>13.3}{:>10.3}{:>10.3}{:>11.3}{:>10.3}{:>7.3}",
                    score.category.name(),
                    score.scored,
                    m.precision_at_4,
                    m.precision_at_8,
                    m.recall_at_4,
                    m.recall_at_8,
                    m.recall_at_10,
                    m.ndcg_at_10,
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

/// Wall-clock cost of embedder calls, one entry per `embed`/`embed_batch`
/// call. Durations are passed in rather than measured here so the arithmetic
/// is testable without depending on real elapsed time.
#[derive(Debug, Default)]
pub struct EmbedTimings {
    calls: Vec<EmbedCall>,
}

#[derive(Debug, Clone, Copy)]
struct EmbedCall {
    items: usize,
    elapsed: Duration,
}

/// Aggregate latency and throughput over every recorded embedder call.
/// Latencies are `f64` milliseconds so the JSON form is a plain number, not
/// `Duration`'s `{secs, nanos}` object.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TimingSummary {
    pub calls: usize,
    pub items: usize,
    pub total_ms: f64,
    pub mean_latency_ms: f64,
    pub throughput_per_sec: f64,
}

impl EmbedTimings {
    pub fn record(&mut self, items: usize, elapsed: Duration) {
        self.calls.push(EmbedCall { items, elapsed });
    }

    /// `None` when nothing was recorded. Throughput is `0.0` when the total
    /// elapsed time is zero, since a rate over no time is undefined.
    pub fn summary(&self) -> Option<TimingSummary> {
        let calls = u32::try_from(self.calls.len()).ok().filter(|&n| n > 0)?;
        let items: usize = self.calls.iter().map(|c| c.items).sum();
        let total: Duration = self.calls.iter().map(|c| c.elapsed).sum();
        let throughput_per_sec = if total.is_zero() {
            0.0
        } else {
            items as f64 / total.as_secs_f64()
        };
        Some(TimingSummary {
            calls: calls as usize,
            items,
            total_ms: total.as_secs_f64() * MS_PER_SEC,
            mean_latency_ms: (total / calls).as_secs_f64() * MS_PER_SEC,
            throughput_per_sec,
        })
    }
}

/// One-line rendering of a `TimingSummary` for the report.
pub fn format_timing(summary: &TimingSummary) -> String {
    format!(
        "embedder: {} calls, {} items, mean latency {:.1} ms, throughput {:.1} items/s\n",
        summary.calls, summary.items, summary.mean_latency_ms, summary.throughput_per_sec,
    )
}

/// Milliseconds per second, for rendering latencies.
const MS_PER_SEC: f64 = 1_000.0;

/// One day in milliseconds, so `CORPUS`/`QUESTIONS` express ages, half-lives,
/// and as-of windows in days rather than raw millisecond counts. Convention
/// shared by `Fact::valid_from_offset_ms` and `Knob::AsOf`: negative is in the
/// past relative to harness run time (more negative = further back);
/// `Knob::HalfLife` is a duration, so it stays positive.
const DAY_MS: i64 = 24 * 60 * 60 * 1_000;

/// One seeded memory for the retrieval benchmark. Every category needs a
/// subset of these fields (see the module doc's text-plus-knob rule); unused
/// fields default to `None` via `FACT_DEFAULTS`, so most entries only name
/// the two or three fields that matter for their category.
///
/// `kind`/`content`/`scope`/`valid_from_offset_ms` are read by the harness
/// below to seed each `NewNode`; the fixture-consistency checks in `mod
/// tests` only need `label`/`edge_target`/`confidence`/`competes_with`.
#[derive(Debug, Clone, Copy)]
struct Fact {
    kind: &'static str,
    label: &'static str,
    content: &'static str,
    /// Retrieval partition; set only for `ScopeFilter` facts.
    scope: Option<&'static str>,
    /// `None` means `NewNode`'s own default of `1.0`; set only for
    /// `Confidence` facts, which need two competing, genuinely different
    /// values.
    confidence: Option<f64>,
    /// Relative offset in milliseconds (see `DAY_MS`), applied at harness
    /// seed time; set only for `Decay`/`AsOf` facts, which each need an older
    /// and a newer version.
    valid_from_offset_ms: Option<i64>,
    /// Another fact's `label` this fact links to via `Graph::link`; set only
    /// on the "linked" half of a `GraphExpansion` pair (the seed half has
    /// none, mirroring how only one direction of a relation is authored).
    edge_target: Option<&'static str>,
    /// Another fact's `label` this fact's confidence is meant to be compared
    /// against; set only on the higher-confidence ("expected") half of a
    /// `Confidence` pair, read back by the fixture-consistency check below.
    #[cfg_attr(not(test), allow(dead_code))]
    competes_with: Option<&'static str>,
}

/// The category-specific extra a `Question` layers on top of its text query,
/// per the module doc's text-plus-knob rule. `Lexical`/`GraphExpansion`/
/// `Confidence` need nothing beyond `query_text` itself, since their "knob"
/// is structural (an edge, or a `competes_with` pair) rather than a `Query`
/// field, so they carry `Knob::None`.
///
/// The fixture-consistency check in `mod tests` only matches each variant's
/// discriminant (`Kind(_)`, not the carried value); `build_query` below reads
/// the carried value itself to build a `Query`.
#[derive(Debug, Clone, Copy)]
enum Knob {
    None,
    Kind(&'static str),
    Scope(&'static str),
    /// A duration in milliseconds (see `DAY_MS`), always positive.
    HalfLife(i64),
    /// An offset in milliseconds from harness run time, same convention as
    /// `Fact::valid_from_offset_ms` (negative = in the past).
    AsOf(i64),
}

/// One retrieval question: a text query plus its category's knob, and the
/// `CORPUS` labels a correct run must surface.
#[derive(Debug, Clone, Copy)]
struct Question {
    name: &'static str,
    category: Category,
    query_text: &'static str,
    knob: Knob,
    expected_relevant: &'static [&'static str],
    /// A phrase the answering fact's `content` must contain, so a relevance
    /// label is checked against the text rather than trusted. For a
    /// `GraphExpansion` question the answer lives in the linked fact, not the
    /// seed.
    #[cfg_attr(not(test), allow(dead_code))]
    answer: &'static str,
}

/// Build the `Query` a `Question` describes: text plus its knob layered on
/// top, using `base` as the shared time origin for `Knob::AsOf` (the module
/// doc's "two tiers" section and `Fact::valid_from_offset_ms`'s doc comment
/// share this convention). Module-level and outside every cfg gate, not just
/// `mod tests`'s: both the text-only tier and the real-embedder tier build
/// queries the same way, and duplicating this per tier would let the two
/// drift apart silently.
///
/// `embedding` is `None` for the text-only tier (no vector arm to feed) and
/// `Some(...)` for `Category::VectorSemantic` questions in the real-embedder
/// tier, which embeds `question.query_text` itself before calling this and
/// passes the result through rather than this function owning any embedder.
fn build_query(question: &Question, base: Millis, k: usize, embedding: Option<Vec<f32>>) -> Query {
    let mut query = Query::text(question.query_text).with_k(k);
    if let Some(embedding) = embedding {
        query = query.with_embedding(embedding);
    }
    match question.knob {
        Knob::None => query,
        Knob::Kind(kind) => query.with_kind(kind),
        Knob::Scope(scope) => query.with_scope(scope),
        Knob::HalfLife(half_life) => query.with_half_life(Millis(half_life)),
        Knob::AsOf(offset) => query.with_as_of(Millis(base.0 + offset)),
    }
}

/// Score one question's already-run retrieval against the metric functions
/// above. Shared by both tiers for the same reason as `build_query`.
fn score_question(retrieved: &[String], relevant: &HashSet<&str>) -> CategoryMetrics {
    CategoryMetrics {
        precision_at_4: precision_at_k(retrieved, relevant, 4),
        precision_at_8: precision_at_k(retrieved, relevant, 8),
        recall_at_4: recall_at_k(retrieved, relevant, 4),
        recall_at_8: recall_at_k(retrieved, relevant, 8),
        recall_at_10: recall_at_k(retrieved, relevant, HEADLINE_K),
        ndcg_at_10: ndcg_at_10(retrieved, relevant),
        mrr: reciprocal_rank(retrieved, relevant),
    }
}

/// Average a category's per-question metrics into its one report row, or
/// `None` for an empty slice (the category was not scored this run, which
/// `format_report` renders as "not run this tier" rather than a misleading
/// `0.0`). Shared by both tiers for the same reason as `build_query`.
fn average_category_metrics(per_question: &[CategoryMetrics]) -> Option<CategoryMetrics> {
    if per_question.is_empty() {
        return None;
    }
    let n = per_question.len() as f64;
    let mut sum = CategoryMetrics {
        precision_at_4: 0.0,
        precision_at_8: 0.0,
        recall_at_4: 0.0,
        recall_at_8: 0.0,
        recall_at_10: 0.0,
        ndcg_at_10: 0.0,
        mrr: 0.0,
    };
    for m in per_question {
        sum.precision_at_4 += m.precision_at_4;
        sum.precision_at_8 += m.precision_at_8;
        sum.recall_at_4 += m.recall_at_4;
        sum.recall_at_8 += m.recall_at_8;
        sum.recall_at_10 += m.recall_at_10;
        sum.ndcg_at_10 += m.ndcg_at_10;
        sum.mrr += m.mrr;
    }
    Some(CategoryMetrics {
        precision_at_4: sum.precision_at_4 / n,
        precision_at_8: sum.precision_at_8 / n,
        recall_at_4: sum.recall_at_4 / n,
        recall_at_8: sum.recall_at_8 / n,
        recall_at_10: sum.recall_at_10 / n,
        ndcg_at_10: sum.ndcg_at_10 / n,
        mrr: sum.mrr / n,
    })
}

/// Retrieval width for every `Query` both harness tiers below build.
/// Deliberately as wide as the whole corpus rather than the pipeline's
/// usual `k=8` default: `fts5_query` (`graph.rs:1495`) ORs every
/// whitespace-split query word, including stopwords like "the", so on this
/// small, single-narrative corpus (every fact mentions "Kestrel Robotics"
/// or shares other common words) most of `CORPUS` weakly lexically matches
/// most queries. `query_core`'s RRF fusion (`graph.rs:739-846`) always
/// ranks a real, even weak, lexical/vector match above a
/// graph-expansion-only one (the expansion floor score is strictly below
/// the lowest possible real-match score), so at the pipeline's normal
/// `k=8` those weak matches alone fill every output slot and a
/// `GraphExpansion` pair's linked-only-by-edge fact never surfaces, even
/// though the edge was seeded and consulted. Widening `k` to the full
/// corpus does not change any category's top-ranked results
/// (`precision_at_k`/`recall_at_k` only ever look at their own `k=4`/`k=8`
/// window of `retrieved`, and relative rank order among real matches is
/// unaffected by how many extra low-ranked candidates follow); it only
/// stops legitimate expansion hits from being cut off before they can be
/// observed. Module-level and outside every cfg gate, not just `mod
/// tests`'s, for the same reason as `build_query` above: both tiers need
/// the same value, and duplicating it would let the two drift apart.
const HARNESS_K: usize = CORPUS.len();

/// Seed every `GraphExpansion` pair's edge into `store`: `seed` is the fact
/// a `Question` matches lexically (the corpus comment's "seed", named by
/// `edge_target`); `linked` is the fact that carries `edge_target` (the
/// corpus comment's "linked" half, reachable only through the edge). `seed
/// -> linked` mirrors the existing
/// `kind_filter_holds_for_graph_expanded_neighbours` test in
/// `liam-store/src/graph.rs` (`NewEdge::new(&seed, &neighbour, ...)`) and
/// the plan's own `seed_id`/`neighbor_id` naming. `Graph::neighbors` is a
/// bidirectional UNION either way (`graph.rs:848-858`), so this choice is
/// for readability, not correctness. Shared by both tiers for the same reason
/// as `build_query`: both seed the exact same edges the exact same way.
async fn seed_graph_expansion_edges(
    store: &DefaultGraph,
    ids: &HashMap<&str, NodeId>,
) -> Result<()> {
    for linked in CORPUS.iter().filter(|f| f.edge_target.is_some()) {
        let seed_label = linked.edge_target.expect("filtered to Some above");
        let seed_id = ids
            .get(seed_label)
            .unwrap_or_else(|| panic!("edge_target {seed_label:?} not seeded"));
        let linked_id = ids
            .get(linked.label)
            .unwrap_or_else(|| panic!("fact {:?} not seeded", linked.label));
        store
            .link(NewEdge::new(seed_id, linked_id, "mentions"))
            .await?;
    }
    Ok(())
}

/// Turn one question's raw `query_explained` hits into the
/// `retrieved`/`relevant` pair the metric functions (`precision_at_k`,
/// `recall_at_k`, `reciprocal_rank`) expect. Shared by both tiers for the
/// same reason as `build_query`.
fn retrieved_and_relevant<'a>(
    hits: &[ExplainedHit],
    question: &Question,
    ids: &'a HashMap<&'a str, NodeId>,
) -> (Vec<String>, HashSet<&'a str>) {
    let retrieved: Vec<String> = hits.iter().map(|h| h.hit.id.as_str().to_string()).collect();
    let relevant: HashSet<&str> = question
        .expected_relevant
        .iter()
        .map(|label| {
            ids.get(label)
                .unwrap_or_else(|| panic!("expected_relevant label {label:?} not seeded"))
                .as_str()
        })
        .collect();
    (retrieved, relevant)
}

/// Assert every scored metric across every category is a valid, finite
/// fraction in `[0.0, 1.0]`. Shared by both tiers for the same reason as
/// `build_query`.
#[cfg(test)]
fn assert_metrics_in_unit_range(per_category: &HashMap<Category, Vec<CategoryMetrics>>) {
    for metrics in per_category.values().flatten() {
        for value in [
            metrics.precision_at_4,
            metrics.precision_at_8,
            metrics.recall_at_4,
            metrics.recall_at_8,
            metrics.recall_at_10,
            metrics.ndcg_at_10,
            metrics.mrr,
        ] {
            assert!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "metric value {value} outside [0.0, 1.0]"
            );
        }
    }
}

/// The result of one embedder-backed run over `CORPUS`/`QUESTIONS`.
///
/// `per_category` always holds one row per `Category`, in `Category::ALL`
/// order, so two reports line up row by row and serialize identically for
/// identical metrics.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub model_id: String,
    pub dims: usize,
    pub corpus_facts: usize,
    pub questions: usize,
    pub per_category: Vec<CategoryScore>,
    /// Every scored question, in `QUESTIONS` order.
    pub per_question: Vec<QuestionScore>,
    /// Mean over every scored question, not over the category means, so a
    /// category with more questions weighs more.
    pub overall: Option<CategoryMetrics>,
    /// `None` only if nothing was embedded.
    pub timings: Option<TimingSummary>,
    /// Names of `GraphExpansion` questions whose hits showed no expansion.
    /// Non-empty means an edge was not seeded or not consulted.
    pub graph_expansion_unexpanded: Vec<String>,
}

impl EvalReport {
    /// The report as printable text: a header, the per-category table, the
    /// overall row, and the embed timing line.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;

        let mut out = String::new();
        // `write!` into a `String` never fails.
        let _ = writeln!(
            out,
            "model: {} ({} dims), {} facts, {} questions",
            self.model_id, self.dims, self.corpus_facts, self.questions
        );
        out.push_str(&format_report(&self.per_category));
        if let Some(m) = self.overall {
            let _ = writeln!(
                out,
                "overall: recall@10 {:.3}, nDCG@10 {:.3}, MRR {:.3}",
                m.recall_at_10, m.ndcg_at_10, m.mrr
            );
        }
        if let Some(summary) = &self.timings {
            out.push_str(&format_timing(summary));
        }
        out
    }
}

/// Text embedded once, untimed, before the corpus so one-off model
/// initialisation does not inflate the first timed call.
const WARM_UP_TEXT: &str = "warm-up";

/// The embedding models the harness can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalModel {
    Qwen3,
    BgeM3,
}

impl EvalModel {
    pub const ALL: [EvalModel; 2] = [EvalModel::Qwen3, EvalModel::BgeM3];

    /// The Hugging Face id the model loads under.
    pub fn id(self) -> &'static str {
        match self {
            EvalModel::Qwen3 => liam_model::QWEN3_EMBEDDING_MODEL_ID,
            EvalModel::BgeM3 => liam_model::BGE_M3_MODEL_ID,
        }
    }

    /// The embedding width to run at. BGE-M3 has one fixed width; Qwen3 runs
    /// at the configured one (Matryoshka truncation).
    pub fn dims(self, configured: usize) -> usize {
        match self {
            EvalModel::Qwen3 => configured,
            EvalModel::BgeM3 => liam_model::BGE_M3_DIMS,
        }
    }
}

impl std::str::FromStr for EvalModel {
    type Err = String;

    /// Hugging Face ids are case-insensitive, so the match is too.
    fn from_str(id: &str) -> std::result::Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|model| model.id().eq_ignore_ascii_case(id))
            .ok_or_else(|| {
                let supported: Vec<&str> = Self::ALL.iter().map(|m| m.id()).collect();
                format!(
                    "unknown embedding model {id:?}; supported models: {}",
                    supported.join(", ")
                )
            })
    }
}

/// Loads `model` and runs the benchmark with it. This is the entry point for
/// callers that only have a model choice.
///
/// Reads the Qwen3 width and the cache directory from `config`, so the
/// harness measures what the daemon with that config would run.
///
/// # Errors
///
/// Fails when the model cannot be loaded, including in a build without the
/// `local` feature, which has no real embedder to load.
pub async fn run_retrieval_eval(model: EvalModel, config: &Config) -> anyhow::Result<EvalReport> {
    let embedder = load_embedder(model, config)?;
    evaluate_embedder(embedder.as_ref(), model.id()).await
}

#[cfg(feature = "local")]
fn load_embedder(model: EvalModel, config: &Config) -> anyhow::Result<Arc<dyn Embedder>> {
    let mut config = config.clone();
    config.embedder.provider = LOCAL_PROVIDER.to_string();
    config.embedder.model = model.id().to_string();
    config.embedding_dims = model.dims(config.embedding_dims);
    models::build_embedder(&config)
}

#[cfg(not(feature = "local"))]
fn load_embedder(model: EvalModel, _config: &Config) -> anyhow::Result<Arc<dyn Embedder>> {
    anyhow::bail!(
        "cannot load embedder {}: this build has no local models. \
         Rebuild with `--features local`.",
        model.id()
    )
}

/// The `embedder.provider` value that selects the in-process models.
#[cfg(feature = "local")]
const LOCAL_PROVIDER: &str = "local";

/// Inserts every `CORPUS` fact into `store` and returns label to id.
///
/// With `embedder`, each fact's content is embedded and the call timed into
/// `timings`; with `None` the facts are text-only. `clock` is moved to each
/// fact's backdated instant so its transaction time matches its `valid_from`
/// (otherwise an `AsOf` query in the past would see nothing), then back to
/// `base` so edges and queries resolve "now" to `base`.
async fn seed_corpus(
    store: &DefaultGraph,
    clock: &liam_store::FixedClock,
    base: Millis,
    embedder: Option<&dyn Embedder>,
    timings: &mut EmbedTimings,
) -> anyhow::Result<HashMap<&'static str, NodeId>> {
    use std::time::Instant;

    use anyhow::Context as _;
    use liam_store::NewNode;

    let mut ids: HashMap<&'static str, NodeId> = HashMap::new();
    for fact in CORPUS {
        let valid_from = fact
            .valid_from_offset_ms
            .map(|offset| Millis(base.0 + offset));
        clock.set(valid_from.unwrap_or(base));

        let mut node = NewNode::now(fact.kind, fact.label, fact.content);
        if let Some(embedder) = embedder {
            let started = Instant::now();
            let embedding = embedder
                .embed(fact.content, TextKind::Document)
                .await
                .with_context(|| format!("embedding corpus fact {}", fact.label))?;
            timings.record(1, started.elapsed());
            node = node.with_embedding(embedding);
        }
        if let Some(scope) = fact.scope {
            node = node.with_scope(scope);
        }
        if let Some(confidence) = fact.confidence {
            node = node.with_confidence(confidence);
        }
        if let Some(valid_from) = valid_from {
            node = node.with_valid_from(valid_from);
        }
        let id = store
            .insert(node)
            .await
            .with_context(|| format!("seeding corpus fact {}", fact.label))?;
        ids.insert(fact.label, id);
    }
    clock.set(base);
    Ok(ids)
}

/// Seeds `CORPUS` into a fresh in-memory store with embeddings from
/// `embedder`, runs every question (embedding the `VectorSemantic` query
/// text too), and scores the results.
///
/// Takes the embedder as a parameter so tests can run the whole pipeline with
/// the mock. Metric values are only meaningful for a real model.
///
/// # Errors
///
/// Fails on an embedder or store error. An empty result for a question is a
/// valid zero score, not an error.
pub async fn evaluate_embedder(
    embedder: &dyn Embedder,
    model_id: &str,
) -> anyhow::Result<EvalReport> {
    use std::time::Instant;

    use anyhow::Context as _;
    use liam_store::{FixedClock, GraphConfig};

    let dims = embedder.dims();
    embedder
        .embed(WARM_UP_TEXT, TextKind::Document)
        .await
        .context("warm-up embed")?;

    let base = Millis::now();
    let clock = Arc::new(FixedClock::new(base));
    let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(dims), clock.clone())
        .await
        .context("opening in-memory store")?;

    let mut timings = EmbedTimings::default();
    let ids = seed_corpus(&store, &clock, base, Some(embedder), &mut timings).await?;
    seed_graph_expansion_edges(&store, &ids)
        .await
        .context("linking GraphExpansion pairs")?;

    let mut per_question: Vec<QuestionScore> = Vec::new();
    let mut graph_expansion_unexpanded: Vec<String> = Vec::new();
    for question in QUESTIONS {
        let embedding = if question.category == Category::VectorSemantic {
            let started = Instant::now();
            let embedding = embedder
                .embed(question.query_text, TextKind::Query)
                .await
                .with_context(|| format!("embedding query for {}", question.name))?;
            timings.record(1, started.elapsed());
            Some(embedding)
        } else {
            None
        };
        let query = build_query(question, base, HARNESS_K, embedding);
        let hits = store
            .query_explained(&query)
            .await
            .with_context(|| format!("running question {}", question.name))?;

        if question.category == Category::GraphExpansion && !hits.iter().any(|h| h.expanded) {
            graph_expansion_unexpanded.push(question.name.to_string());
        }

        let (retrieved, relevant) = retrieved_and_relevant(&hits, question, &ids);
        per_question.push(QuestionScore {
            name: question.name,
            category: question.category,
            metrics: score_question(&retrieved, &relevant),
        });
    }

    let per_category: Vec<CategoryScore> = Category::ALL
        .iter()
        .map(|&category| {
            let metrics: Vec<CategoryMetrics> = per_question
                .iter()
                .filter(|q| q.category == category)
                .map(|q| q.metrics)
                .collect();
            CategoryScore {
                category,
                scored: metrics.len(),
                metrics: average_category_metrics(&metrics),
            }
        })
        .collect();
    let all_metrics: Vec<CategoryMetrics> = per_question.iter().map(|q| q.metrics).collect();

    Ok(EvalReport {
        model_id: model_id.to_string(),
        dims,
        corpus_facts: CORPUS.len(),
        questions: QUESTIONS.len(),
        per_category,
        per_question,
        overall: average_category_metrics(&all_metrics),
        timings: timings.summary(),
        graph_expansion_unexpanded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact_by_label(label: &str) -> Option<&'static Fact> {
        CORPUS.iter().find(|f| f.label == label)
    }

    #[test]
    fn every_question_expected_relevant_label_exists_in_corpus() {
        // Given the full QUESTIONS table, when every question's
        // expected_relevant labels are checked against CORPUS, then every
        // one exists as a real corpus label.
        for q in QUESTIONS {
            for label in q.expected_relevant {
                assert!(
                    fact_by_label(label).is_some(),
                    "question {} references unknown corpus label {label:?}",
                    q.name
                );
            }
        }
    }

    #[test]
    fn every_category_has_at_least_one_question() {
        // Given QUESTIONS grouped by category, when checked, then every one
        // of the 8 Category::ALL variants has at least one question.
        for category in Category::ALL {
            assert!(
                QUESTIONS.iter().any(|q| q.category == category),
                "category {category:?} has no question"
            );
        }
    }

    #[test]
    fn question_names_are_unique() {
        // Given QUESTIONS, when checked for uniqueness by name, then no two
        // questions share a name.
        let mut names: Vec<&str> = QUESTIONS.iter().map(|q| q.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate question names");
    }

    #[test]
    fn every_non_vector_semantic_question_carries_its_declared_knob() {
        // Given every Question other than VectorSemantic, when its
        // knob/shape is inspected, then it always carries non-empty
        // query_text, and for KindFilter/ScopeFilter/Decay/AsOf specifically,
        // the matching category-specific knob field is actually set.
        for q in QUESTIONS {
            if q.category == Category::VectorSemantic {
                continue;
            }
            assert!(
                !q.query_text.is_empty(),
                "question {} has empty query_text",
                q.name
            );
            let knob_matches_category = match q.category {
                Category::KindFilter => matches!(q.knob, Knob::Kind(_)),
                Category::ScopeFilter => matches!(q.knob, Knob::Scope(_)),
                Category::Decay => matches!(q.knob, Knob::HalfLife(_)),
                Category::AsOf => matches!(q.knob, Knob::AsOf(_)),
                // Lexical/GraphExpansion/Confidence have no Query-level knob;
                // their "knob" is structural (an edge or a competes_with
                // pair), checked by the two tests below instead. Require
                // Knob::None explicitly here rather than accepting any
                // value, so a copy-paste error that leaves a stray
                // non-None knob on one of these three categories is
                // caught instead of silently passing.
                _ => matches!(q.knob, Knob::None),
            };
            assert!(
                knob_matches_category,
                "question {} (category {:?}) does not carry its declared knob: {:?}",
                q.name, q.category, q.knob
            );
        }
    }

    #[test]
    fn confidence_questions_name_a_real_competitor_with_a_different_confidence() {
        // Given every Confidence question's expected-relevant fact, when its
        // competes_with field is read from CORPUS, then the named fact
        // exists in CORPUS AND its confidence value actually differs from
        // the expected-relevant fact's confidence.
        for q in QUESTIONS
            .iter()
            .filter(|q| q.category == Category::Confidence)
        {
            for label in q.expected_relevant {
                let fact = fact_by_label(label).unwrap_or_else(|| {
                    panic!(
                        "question {} references unknown corpus label {label:?}",
                        q.name
                    )
                });
                let competitor_label = fact.competes_with.unwrap_or_else(|| {
                    panic!(
                        "confidence fact {label:?} (question {}) has no competes_with",
                        q.name
                    )
                });
                let competitor = fact_by_label(competitor_label).unwrap_or_else(|| {
                    panic!(
                        "fact {label:?}'s competes_with names unknown label {competitor_label:?}"
                    )
                });
                let fact_confidence = fact.confidence.unwrap_or(1.0);
                let competitor_confidence = competitor.confidence.unwrap_or(1.0);
                assert_ne!(
                    fact_confidence, competitor_confidence,
                    "fact {label:?} and its named competitor {competitor_label:?} must differ \
                     in confidence"
                );
            }
        }
    }

    #[test]
    fn graph_expansion_edge_targets_exist_in_corpus() {
        // Given QUESTIONS and CORPUS, when a fact carries an edge_target for
        // a GraphExpansion-relevant fact, then the target label exists in
        // CORPUS.
        for fact in CORPUS.iter().filter(|f| f.edge_target.is_some()) {
            let target = fact.edge_target.expect("filtered to Some above");
            assert!(
                fact_by_label(target).is_some(),
                "fact {:?} has edge_target {target:?}, which is not a corpus label",
                fact.label
            );
        }
    }

    #[test]
    fn graph_expansion_questions_expected_relevant_matches_seeded_edge() {
        // Given each GraphExpansion question's expected_relevant pair
        // (authored as [seed, linked fact]), when the linked fact's
        // edge_target is read back from CORPUS, then it names the seed
        // label, tying the question's ground truth to the specific edge
        // the harness seeds via Graph::link, not just to some valid corpus
        // label.
        for q in QUESTIONS
            .iter()
            .filter(|q| q.category == Category::GraphExpansion)
        {
            assert_eq!(
                q.expected_relevant.len(),
                2,
                "GraphExpansion question {} must name exactly a seed and its linked fact",
                q.name
            );
            let seed_label = q.expected_relevant[0];
            let linked_label = q.expected_relevant[1];
            let linked_fact = fact_by_label(linked_label).unwrap_or_else(|| {
                panic!(
                    "question {} references unknown corpus label {linked_label:?}",
                    q.name
                )
            });
            assert_eq!(
                linked_fact.edge_target,
                Some(seed_label),
                "question {}'s linked fact {linked_label:?} must have edge_target pointing at \
                 seed {seed_label:?}",
                q.name
            );
        }
    }

    /// Documented size of the golden set: every category carries the same
    /// number of questions so no category's average is noisier than another's.
    const QUESTIONS_PER_CATEGORY: usize = 10;
    const GOLDEN_SET_SIZE: usize = QUESTIONS_PER_CATEGORY * Category::ALL.len();

    /// Names of the questions that predate the expansion. Reports quote
    /// questions by name, so renaming or dropping one would silently break
    /// comparisons against earlier runs.
    const ORIGINAL_QUESTION_NAMES: [&str; 40] = [
        "lexical_hq_location",
        "lexical_founders",
        "lexical_titanium_frame",
        "lexical_falconer_battery",
        "lexical_demo_day_flight",
        "vector_solar_charging",
        "vector_remote_work",
        "vector_drone_swarm",
        "vector_onboarding_speed",
        "vector_quieter_rotors",
        "graph_motor_recall_reason",
        "graph_weatherproof_casing",
        "graph_logistics_partnership",
        "graph_firmware_rollback",
        "graph_motor_bearing_defect",
        "kind_warehouse_decision",
        "kind_data_retention_policy",
        "kind_battery_supplier_fact",
        "kind_certification_flight_episode",
        "kind_insurance_provider_decision",
        "scope_engineering_sprint",
        "scope_legal_compliance",
        "scope_finance_budget",
        "scope_operations_maintenance",
        "scope_marketing_launch",
        "decay_camera_sensor",
        "decay_flight_range",
        "decay_support_hours",
        "decay_pricing_tier",
        "decay_team_size",
        "asof_product_name",
        "asof_headquarters",
        "asof_ceo",
        "asof_product_line",
        "asof_motor_vendor",
        "confidence_top_speed_terms",
        "confidence_top_speed_phrase",
        "confidence_payload_terms",
        "confidence_payload_phrase",
        "confidence_battery_life",
    ];

    /// Per-category recall@10 the text-only tier must keep meeting. Each
    /// equals the value measured on the 80-question set. Exact equality is
    /// safe because this tier is deterministic: no embedder, an in-memory
    /// store, and a fixed clock.
    const TEXT_ONLY_RECALL_AT_10_FLOORS: [(Category, f64); 7] = [
        (Category::Lexical, 1.0),
        (Category::GraphExpansion, 0.9),
        (Category::KindFilter, 1.0),
        (Category::ScopeFilter, 1.0),
        (Category::Decay, 1.0),
        (Category::Confidence, 0.9),
        (Category::AsOf, 1.0),
    ];

    #[test]
    fn every_category_carries_the_documented_question_count() {
        // Given QUESTIONS grouped by category, when each group is counted,
        // then every category has exactly QUESTIONS_PER_CATEGORY questions.
        for category in Category::ALL {
            let count = QUESTIONS.iter().filter(|q| q.category == category).count();
            assert_eq!(
                count, QUESTIONS_PER_CATEGORY,
                "category {category:?} has {count} questions"
            );
        }
    }

    #[test]
    fn golden_set_has_the_documented_total_size() {
        // Given QUESTIONS, when counted, then the total matches the
        // documented golden set size.
        assert_eq!(QUESTIONS.len(), GOLDEN_SET_SIZE);
    }

    #[test]
    fn corpus_labels_are_unique() {
        // Given CORPUS, when checked for uniqueness by label, then no two
        // facts share a label (labels are the fixture's stable ids).
        let mut labels: Vec<&str> = CORPUS.iter().map(|f| f.label).collect();
        let total = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), total, "duplicate corpus labels");
    }

    #[test]
    fn no_question_lists_the_same_relevant_label_twice() {
        // Given every question, when its expected_relevant labels are
        // deduplicated, then nothing is lost, so recall's denominator is the
        // true number of relevant facts.
        for q in QUESTIONS {
            let unique: HashSet<&str> = q.expected_relevant.iter().copied().collect();
            assert_eq!(
                unique.len(),
                q.expected_relevant.len(),
                "question {} repeats a relevant label",
                q.name
            );
            assert!(
                !q.expected_relevant.is_empty(),
                "question {} has no relevant label",
                q.name
            );
        }
    }

    #[test]
    fn original_question_names_are_still_present() {
        // Given the names of the pre-expansion questions, when looked up in
        // QUESTIONS, then every one is still there.
        for name in ORIGINAL_QUESTION_NAMES {
            assert!(
                QUESTIONS.iter().any(|q| q.name == name),
                "original question {name:?} was renamed or removed"
            );
        }
    }

    #[test]
    fn kind_filter_questions_expect_a_fact_of_the_filtered_kind() {
        // Given each KindFilter question, when its expected fact is read from
        // CORPUS, then the fact's kind equals the question's kind filter.
        for q in QUESTIONS
            .iter()
            .filter(|q| q.category == Category::KindFilter)
        {
            let Knob::Kind(kind) = q.knob else {
                panic!("question {} has no kind knob", q.name);
            };
            for label in q.expected_relevant {
                let fact = fact_by_label(label).expect("label checked elsewhere");
                assert_eq!(fact.kind, kind, "question {} expects {label:?}", q.name);
            }
        }
    }

    #[test]
    fn scope_filter_questions_expect_a_fact_in_the_filtered_scope() {
        // Given each ScopeFilter question, when its expected fact is read
        // from CORPUS, then the fact's scope equals the question's scope.
        for q in QUESTIONS
            .iter()
            .filter(|q| q.category == Category::ScopeFilter)
        {
            let Knob::Scope(scope) = q.knob else {
                panic!("question {} has no scope knob", q.name);
            };
            for label in q.expected_relevant {
                let fact = fact_by_label(label).expect("label checked elsewhere");
                assert_eq!(
                    fact.scope,
                    Some(scope),
                    "question {} expects {label:?}",
                    q.name
                );
            }
        }
    }

    #[test]
    fn as_of_questions_expect_a_fact_already_valid_at_the_as_of_instant() {
        // Given each AsOf question, when its expected fact's valid-from
        // offset is compared with the as-of offset, then the fact is already
        // valid at that instant.
        for q in QUESTIONS.iter().filter(|q| q.category == Category::AsOf) {
            let Knob::AsOf(as_of) = q.knob else {
                panic!("question {} has no as-of knob", q.name);
            };
            for label in q.expected_relevant {
                let fact = fact_by_label(label).expect("label checked elsewhere");
                let valid_from = fact
                    .valid_from_offset_ms
                    .unwrap_or_else(|| panic!("fact {label:?} has no valid_from offset"));
                assert!(
                    valid_from <= as_of,
                    "question {} expects {label:?}, not yet valid at its as-of instant",
                    q.name
                );
                // The superseding version must be invalid at that instant,
                // else the as-of knob would not be what picks the original.
                let superseding = fact_by_label(&format!("{} (Current)", label_stem(label)))
                    .or_else(|| fact_by_label(&format!("{} (Renamed)", label_stem(label))))
                    .unwrap_or_else(|| panic!("fact {label:?} has no superseding version"));
                let superseding_valid_from =
                    superseding.valid_from_offset_ms.unwrap_or_else(|| {
                        panic!("fact {:?} has no valid_from offset", superseding.label)
                    });
                assert!(
                    superseding_valid_from > as_of,
                    "question {}: superseding fact {:?} is already valid at the as-of instant",
                    q.name,
                    superseding.label
                );
            }
        }
    }

    #[test]
    fn every_question_answer_appears_in_its_expected_fact() {
        // Given each question's answer phrase, when its expected facts are
        // read from CORPUS, then the answer is in the answering fact's
        // content: the last expected label, which for a GraphExpansion
        // question is the linked fact rather than the lexical seed.
        for q in QUESTIONS {
            assert!(!q.answer.is_empty(), "question {} has no answer", q.name);
            let answering = q
                .expected_relevant
                .last()
                .expect("checked non-empty elsewhere");
            let fact = fact_by_label(answering).expect("label checked elsewhere");
            assert!(
                fact.content.contains(q.answer),
                "question {}: fact {answering:?} does not contain answer {:?}",
                q.name,
                q.answer
            );
        }
    }

    /// The label with its trailing parenthesised version tag removed, so
    /// `Support Hours (Old)` and `Support Hours (New)` share a stem.
    fn label_stem(label: &str) -> &str {
        label.rsplit_once(" (").map_or(label, |(stem, _)| stem)
    }

    #[test]
    fn decay_questions_expect_the_newest_version_of_their_fact() {
        // Given each Decay question, when its expected fact is compared with
        // the other facts sharing its label stem, then it was valid later
        // than every one of them, so decay favours it.
        for q in QUESTIONS.iter().filter(|q| q.category == Category::Decay) {
            for label in q.expected_relevant {
                let fact = fact_by_label(label).expect("label checked elsewhere");
                let valid_from = fact
                    .valid_from_offset_ms
                    .unwrap_or_else(|| panic!("fact {label:?} has no valid_from offset"));
                let older_versions: Vec<&Fact> = CORPUS
                    .iter()
                    .filter(|f| f.label != *label && label_stem(f.label) == label_stem(label))
                    .collect();
                assert!(
                    !older_versions.is_empty(),
                    "question {} expects {label:?}, which has no other version to decay",
                    q.name
                );
                for other in older_versions {
                    let other_valid_from = other.valid_from_offset_ms.unwrap_or_else(|| {
                        panic!("fact {:?} has no valid_from offset", other.label)
                    });
                    assert!(
                        valid_from > other_valid_from,
                        "question {} expects {label:?}, which is not newer than {:?}",
                        q.name,
                        other.label
                    );
                }
            }
        }
    }

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
    fn precision_at_k_denominator_is_k_not_retrieved_len() {
        // Arrange: a short retrieved list, well under k=4, with its one
        // entry relevant.
        let retrieved = vec!["a".to_string()];
        let relevant = set(&["a"]);

        // Act
        let precision = precision_at_k(&retrieved, &relevant, 4);

        // Assert: divided by k=4 (0.25), not by retrieved.len()=1, which
        // would wrongly give 1.0 and reward a short list for having fewer
        // slots to be wrong in.
        assert_eq!(precision, 0.25);
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

    #[test]
    fn average_category_metrics_computes_the_mean_of_each_field() {
        // Arrange: three hand-built CategoryMetrics whose fields sum to a
        // clean mean, so a copy-paste bug that averages the wrong field is
        // caught by a mismatched value rather than hidden by a coincidence.
        let a = CategoryMetrics {
            precision_at_4: 1.0,
            precision_at_8: 0.5,
            recall_at_4: 0.0,
            recall_at_8: 1.0,
            recall_at_10: 0.25,
            ndcg_at_10: 0.25,
            mrr: 0.5,
        };
        let b = CategoryMetrics {
            precision_at_4: 0.0,
            precision_at_8: 1.0,
            recall_at_4: 1.0,
            recall_at_8: 0.0,
            recall_at_10: 0.75,
            ndcg_at_10: 0.5,
            mrr: 1.0,
        };
        let c = CategoryMetrics {
            precision_at_4: 0.5,
            precision_at_8: 0.0,
            recall_at_4: 0.5,
            recall_at_8: 0.5,
            recall_at_10: 1.0,
            ndcg_at_10: 1.0,
            mrr: 0.0,
        };

        // Act
        let average = average_category_metrics(&[a, b, c]).expect("non-empty slice returns Some");

        // Assert: each field is the hand-computed mean of that same field
        // across a, b, c.
        assert_eq!(average.precision_at_4, 0.5);
        assert_eq!(average.precision_at_8, 0.5);
        assert_eq!(average.recall_at_4, 0.5);
        assert_eq!(average.recall_at_8, 0.5);
        // recall@10 and nDCG@10 means (2/3 and 7/12) differ from the recall@8
        // and MRR means (0.5), so a swapped field fails here.
        assert!((average.recall_at_10 - 2.0 / 3.0).abs() < 1e-9);
        assert!((average.ndcg_at_10 - 7.0 / 12.0).abs() < 1e-9);
        assert_eq!(average.mrr, 0.5);
    }

    #[test]
    fn average_category_metrics_returns_none_for_empty_slice() {
        // Given an empty slice, when averaged, then None, not a
        // divide-by-zero NaN or a misleading 0.0.
        assert!(average_category_metrics(&[]).is_none());
    }

    /// Tolerance for comparing logarithm-based scores against hand-computed
    /// decimals.
    const SCORE_EPSILON: f64 = 1e-9;

    fn ids(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_at_k_ten_counts_hits_inside_the_first_ten_only() {
        // Arrange: relevant ids sit at ranks 2, 10 and 11 of 12 results.
        let retrieved = ids(&[
            "x1", "r1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "r2", "r3", "x9",
        ]);
        let relevant = set(&["r1", "r2", "r3"]);

        // Act
        let recall = recall_at_k(&retrieved, &relevant, 10);

        // Assert: rank 10 is inside the window, rank 11 is not, so 2 of 3.
        assert_eq!(recall, 2.0 / 3.0);
    }

    #[test]
    fn recall_at_k_ten_with_a_list_shorter_than_ten_uses_the_whole_list() {
        // Arrange
        let retrieved = ids(&["a", "x", "b"]);
        let relevant = set(&["a", "b"]);

        // Act
        let recall = recall_at_k(&retrieved, &relevant, 10);

        // Assert
        assert_eq!(recall, 1.0);
    }

    #[test]
    fn recall_at_k_ten_is_zero_for_an_empty_relevant_set() {
        // Arrange
        let retrieved = ids(&["a", "b"]);
        let relevant = set(&[]);

        // Act
        let recall = recall_at_k(&retrieved, &relevant, 10);

        // Assert
        assert_eq!(recall, 0.0);
    }

    #[test]
    fn ndcg_at_10_matches_a_hand_computed_worked_example() {
        // Arrange: relevant a, b, c; a is ranked 1 and b is ranked 3.
        // DCG  = 1/log2(2) + 1/log2(4)               = 1 + 0.5          = 1.5
        // IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)   = 2.1309297535714573
        let retrieved = ids(&["a", "x", "b", "y"]);
        let relevant = set(&["a", "b", "c"]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert
        assert!((ndcg - 0.7039180890341347).abs() < SCORE_EPSILON, "{ndcg}");
    }

    #[test]
    fn ndcg_at_10_is_zero_for_an_empty_relevant_set() {
        // Arrange
        let retrieved = ids(&["a", "b"]);
        let relevant = set(&[]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert: an empty ideal ranking must not divide by zero.
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn ndcg_at_10_is_zero_for_an_empty_retrieved_list() {
        // Arrange
        let retrieved: Vec<String> = vec![];
        let relevant = set(&["a"]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn ndcg_at_10_is_one_for_a_single_relevant_item_at_rank_one() {
        // Arrange
        let retrieved = ids(&["a", "x", "y"]);
        let relevant = set(&["a"]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert
        assert!((ndcg - 1.0).abs() < SCORE_EPSILON, "{ndcg}");
    }

    #[test]
    fn ndcg_at_10_is_one_when_every_retrieved_item_is_relevant() {
        // Arrange
        let retrieved = ids(&["a", "b", "c"]);
        let relevant = set(&["a", "b", "c"]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert
        assert!((ndcg - 1.0).abs() < SCORE_EPSILON, "{ndcg}");
    }

    #[test]
    fn ndcg_at_10_with_a_list_shorter_than_ten_penalises_the_missing_item() {
        // Arrange: only a is returned; the ideal ranking also has b at rank 2.
        // DCG = 1, IDCG = 1 + 1/log2(3) = 1.6309297535714573
        let retrieved = ids(&["a"]);
        let relevant = set(&["a", "b"]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert
        assert!((ndcg - 0.6131471927654584).abs() < SCORE_EPSILON, "{ndcg}");
    }

    #[test]
    fn ndcg_at_10_ignores_a_relevant_item_ranked_eleventh() {
        // Arrange
        let retrieved = ids(&[
            "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "a",
        ]);
        let relevant = set(&["a"]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert
        assert_eq!(ndcg, 0.0);
    }

    #[test]
    fn ndcg_at_10_counts_a_repeated_id_once_but_it_still_occupies_a_rank() {
        // Arrange: a appears at ranks 1 and 2, b at rank 3.
        // DCG  = 1/log2(2) + 1/log2(4)   = 1.5 (the second a earns nothing)
        // IDCG = 1/log2(2) + 1/log2(3)   = 1.6309297535714573
        let retrieved = ids(&["a", "a", "b"]);
        let relevant = set(&["a", "b"]);

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);

        // Assert
        assert!((ndcg - 0.9197207891481876).abs() < SCORE_EPSILON, "{ndcg}");
    }

    #[test]
    fn score_question_fills_recall_at_10_and_ndcg_at_10() {
        // Arrange
        // The only relevant id sits at rank 9: outside recall@8, inside @10.
        let retrieved = ids(&["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "a"]);
        let relevant = set(&["a"]);

        // Act
        let metrics = score_question(&retrieved, &relevant);

        // Assert: recall@10 and nDCG@10 differ from recall@8 and MRR.
        assert_eq!(metrics.recall_at_8, 0.0);
        assert_eq!(metrics.recall_at_10, 1.0);
        assert!((metrics.ndcg_at_10 - 1.0 / 10f64.log2()).abs() < SCORE_EPSILON);
        assert!((metrics.mrr - 1.0 / 9.0).abs() < SCORE_EPSILON);
    }

    #[test]
    fn a_perfect_top_ten_over_eleven_relevant_ids_has_full_ndcg_and_partial_recall() {
        // Arrange: eleven relevant ids, the first ten retrieved in order.
        let all: Vec<String> = (0..11).map(|i| format!("r{i}")).collect();
        let relevant: HashSet<&str> = all.iter().map(String::as_str).collect();
        let retrieved: Vec<String> = all[..10].to_vec();

        // Act
        let ndcg = ndcg_at_10(&retrieved, &relevant);
        let recall = recall_at_k(&retrieved, &relevant, 10);

        // Assert: the ideal ranking is capped at ten, so nDCG@10 is 1.0.
        assert!((ndcg - 1.0).abs() < SCORE_EPSILON, "{ndcg}");
        assert!((recall - 10.0 / 11.0).abs() < SCORE_EPSILON, "{recall}");
    }

    #[test]
    fn embed_timings_summarise_latency_and_throughput_from_recorded_durations() {
        // Arrange: 1 item in 100ms, then 3 items in 300ms.
        let mut timings = EmbedTimings::default();
        timings.record(1, Duration::from_millis(100));
        timings.record(3, Duration::from_millis(300));

        // Act
        let summary = timings.summary().expect("two calls were recorded");

        // Assert: 4 items over 0.4s is 10 items/s; mean call latency is 200ms.
        assert_eq!(summary.calls, 2);
        assert_eq!(summary.items, 4);
        assert!((summary.total_ms - 400.0).abs() < SCORE_EPSILON);
        assert!((summary.mean_latency_ms - 200.0).abs() < SCORE_EPSILON);
        assert!((summary.throughput_per_sec - 10.0).abs() < SCORE_EPSILON);
    }

    #[test]
    fn embed_timings_with_no_calls_have_no_summary() {
        // Arrange
        let timings = EmbedTimings::default();

        // Act / Assert
        assert!(timings.summary().is_none());
    }

    #[test]
    fn embed_timings_with_zero_elapsed_report_zero_throughput_not_infinity() {
        // Arrange
        let mut timings = EmbedTimings::default();
        timings.record(5, Duration::ZERO);

        // Act
        let summary = timings.summary().expect("one call was recorded");

        // Assert
        assert_eq!(summary.throughput_per_sec, 0.0);
    }

    #[test]
    fn format_timing_reports_calls_latency_and_throughput() {
        // Arrange
        let mut timings = EmbedTimings::default();
        timings.record(4, Duration::from_millis(400));
        let summary = timings.summary().expect("one call was recorded");

        // Act
        let text = format_timing(&summary);

        // Assert
        assert_eq!(
            text,
            "embedder: 1 calls, 4 items, mean latency 400.0 ms, throughput 10.0 items/s\n"
        );
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
                    recall_at_10: 0.97,
                    ndcg_at_10: 0.88,
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
        assert!(report.contains("recall@10"), "{report}");
        assert!(report.contains("nDCG@10"), "{report}");
        assert!(report.contains("MRR"), "{report}");
    }

    #[test]
    fn format_report_places_recall_at_10_and_ndcg_at_10_in_their_columns() {
        // Arrange
        let scores = all_category_scores();

        // Act
        let report = format_report(&scores);

        // Assert: the row carries the values in header order, between
        // recall@8 and MRR.
        let row = report
            .lines()
            .find(|l| l.starts_with(Category::Lexical.name()))
            .expect("Lexical row");
        let values: Vec<&str> = row.split_whitespace().collect();
        let n = values.len();
        assert_eq!(
            values[n - 4..],
            ["0.950", "0.970", "0.880", "0.833"],
            "{row}"
        );
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

    /// Embedding width for the whole-pipeline runs below. The text-only tier
    /// seeds no vectors, but `GraphConfig` still needs a vector-table width.
    const DIMS: usize = 8;

    fn hand_built_report() -> EvalReport {
        let metrics = CategoryMetrics {
            precision_at_4: 0.25,
            precision_at_8: 0.125,
            recall_at_4: 0.5,
            recall_at_8: 0.75,
            recall_at_10: 0.9,
            ndcg_at_10: 0.8,
            mrr: 0.7,
        };
        EvalReport {
            model_id: "test/model".to_string(),
            dims: 4,
            corpus_facts: 3,
            questions: 2,
            per_category: vec![
                CategoryScore {
                    category: Category::Lexical,
                    scored: 2,
                    metrics: Some(metrics),
                },
                CategoryScore {
                    category: Category::VectorSemantic,
                    scored: 0,
                    metrics: None,
                },
            ],
            per_question: vec![QuestionScore {
                name: "lexical-1",
                category: Category::Lexical,
                metrics,
            }],
            overall: Some(metrics),
            timings: Some(TimingSummary {
                calls: 2,
                items: 2,
                total_ms: 20.0,
                mean_latency_ms: 10.0,
                throughput_per_sec: 100.0,
            }),
            graph_expansion_unexpanded: Vec::new(),
        }
    }

    #[test]
    fn render_lists_header_table_overall_and_timing_in_order() {
        // Arrange
        let report = hand_built_report();

        // Act
        let text = report.render();

        // Assert
        let positions: Vec<usize> = [
            "model: test/model (4 dims), 3 facts, 2 questions",
            "Lexical",
            "VectorSemantic",
            "overall: recall@10 0.900, nDCG@10 0.800, MRR 0.700",
            "embedder: 2 calls",
        ]
        .iter()
        .map(|needle| {
            text.find(needle)
                .unwrap_or_else(|| panic!("missing {needle:?} in: {text}"))
        })
        .collect();
        assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");
    }

    #[test]
    fn report_serializes_categories_in_order_with_null_for_unscored() {
        // Arrange
        let report = hand_built_report();

        // Act
        let json = serde_json::to_value(&report).expect("serialize report");

        // Assert
        let metrics = serde_json::json!({
            "precision_at_4": 0.25,
            "precision_at_8": 0.125,
            "recall_at_4": 0.5,
            "recall_at_8": 0.75,
            "recall_at_10": 0.9,
            "ndcg_at_10": 0.8,
            "mrr": 0.7,
        });
        let expected = serde_json::json!({
            "model_id": "test/model",
            "dims": 4,
            "corpus_facts": 3,
            "questions": 2,
            "per_category": [
                { "category": "Lexical", "scored": 2, "metrics": metrics },
                { "category": "VectorSemantic", "scored": 0, "metrics": null },
            ],
            "per_question": [
                { "name": "lexical-1", "category": "Lexical", "metrics": metrics },
            ],
            "overall": metrics,
            "timings": {
                "calls": 2,
                "items": 2,
                "total_ms": 20.0,
                "mean_latency_ms": 10.0,
                "throughput_per_sec": 100.0,
            },
            "graph_expansion_unexpanded": [],
        });
        assert_eq!(json, expected);
    }

    #[tokio::test]
    async fn evaluate_embedder_with_the_mock_scores_every_category() {
        // Arrange
        let embedder = liam_model::MockEmbedder::new(DIMS);

        // Act
        let report = evaluate_embedder(&embedder, "mock")
            .await
            .expect("mock run succeeds");

        // Assert
        let categories: Vec<Category> = report.per_category.iter().map(|s| s.category).collect();
        assert_eq!(categories, Category::ALL);
        assert!(report.per_category.iter().all(|s| s.metrics.is_some()));
        for score in &report.per_category {
            let expected = QUESTIONS
                .iter()
                .filter(|q| q.category == score.category)
                .count();
            assert_eq!(score.scored, expected, "{:?}", score.category);
        }
        let scored_total: usize = report.per_category.iter().map(|s| s.scored).sum();
        assert_eq!(scored_total, QUESTIONS.len());
        assert_eq!(report.per_question.len(), QUESTIONS.len());
        let overall = report.overall.expect("every question was scored");
        assert!((0.0..=1.0).contains(&overall.recall_at_10));
        assert_eq!(report.dims, DIMS);
        assert_eq!(report.corpus_facts, CORPUS.len());
        assert_eq!(report.questions, QUESTIONS.len());
        assert!(report.graph_expansion_unexpanded.is_empty());
        let vector_questions = QUESTIONS
            .iter()
            .filter(|q| q.category == Category::VectorSemantic)
            .count();
        let timings = report.timings.expect("the mock run embeds the corpus");
        assert_eq!(timings.calls, CORPUS.len() + vector_questions);
    }

    /// Wraps the mock, logs every text embedded, and sleeps per call so the
    /// timing span has a measurable floor.
    struct RecordingEmbedder {
        inner: liam_model::MockEmbedder,
        seen: std::sync::Mutex<Vec<String>>,
    }

    const PER_CALL_DELAY: Duration = Duration::from_millis(2);

    #[async_trait::async_trait]
    impl Embedder for RecordingEmbedder {
        fn dims(&self) -> usize {
            self.inner.dims()
        }

        async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
            self.seen.lock().expect("lock").push(text.to_string());
            tokio::time::sleep(PER_CALL_DELAY).await;
            self.inner.embed(text, kind).await
        }
    }

    #[tokio::test]
    async fn evaluate_embedder_warms_up_first_and_times_each_embed_call() {
        // Arrange
        let embedder = RecordingEmbedder {
            inner: liam_model::MockEmbedder::new(DIMS),
            seen: std::sync::Mutex::new(Vec::new()),
        };

        // Act
        let report = evaluate_embedder(&embedder, "mock")
            .await
            .expect("mock run succeeds");

        // Assert
        let seen = embedder.seen.lock().expect("lock");
        assert_eq!(seen.first().map(String::as_str), Some(WARM_UP_TEXT));
        let timings = report.timings.expect("the run embeds the corpus");
        assert_eq!(timings.calls, seen.len() - 1);
        let floor_ms = PER_CALL_DELAY.as_secs_f64() * MS_PER_SEC;
        assert!(
            timings.mean_latency_ms >= floor_ms,
            "mean {} ms is below the {floor_ms} ms every call sleeps",
            timings.mean_latency_ms
        );
    }

    #[cfg(not(feature = "local"))]
    #[tokio::test]
    async fn run_retrieval_eval_without_the_local_feature_explains_how_to_fix_it() {
        // Act
        let err = run_retrieval_eval(EvalModel::Qwen3, &Config::default())
            .await
            .expect_err("no real embedder in this build");

        // Assert
        assert!(err.to_string().contains("--features local"), "{err}");
    }

    #[test]
    fn eval_model_parses_each_supported_id_case_insensitively() {
        // Arrange
        let cases = [
            (liam_model::QWEN3_EMBEDDING_MODEL_ID, EvalModel::Qwen3),
            ("qwen/qwen3-embedding-0.6b", EvalModel::Qwen3),
            (liam_model::BGE_M3_MODEL_ID, EvalModel::BgeM3),
            ("baai/BGE-M3", EvalModel::BgeM3),
        ];

        for (id, expected) in cases {
            // Act
            let parsed: EvalModel = id.parse().expect("a supported id parses");

            // Assert
            assert_eq!(parsed, expected, "{id}");
        }
    }

    #[test]
    fn eval_model_round_trips_through_its_canonical_id() {
        for model in EvalModel::ALL {
            // Act
            let parsed: EvalModel = model.id().parse().expect("a canonical id parses");

            // Assert
            assert_eq!(parsed, model);
        }
    }

    #[test]
    fn an_unknown_model_id_is_refused_naming_it_and_the_supported_ids() {
        // Act
        let err = "BAAI/bge-m3-unsupervised"
            .parse::<EvalModel>()
            .expect_err("an unknown id must not parse");

        // Assert
        assert!(err.contains("BAAI/bge-m3-unsupervised"), "{err}");
        assert!(err.contains(liam_model::QWEN3_EMBEDDING_MODEL_ID), "{err}");
        assert!(err.contains(liam_model::BGE_M3_MODEL_ID), "{err}");
    }

    #[test]
    fn bge_m3_runs_at_its_native_width_and_qwen3_at_the_configured_one() {
        // Arrange
        let configured = 768;

        // Act / Assert
        assert_eq!(EvalModel::Qwen3.dims(configured), configured);
        assert_eq!(EvalModel::BgeM3.dims(configured), liam_model::BGE_M3_DIMS);
    }

    /// The report fields that depend on wall-clock time: every key of
    /// `timings` except the call and item counts.
    const TIMING_FIELDS: [&str; 3] = ["total_ms", "mean_latency_ms", "throughput_per_sec"];

    fn json_without_timing(report: &EvalReport) -> String {
        let mut value = serde_json::to_value(report).expect("report serializes");
        let timings = value["timings"]
            .as_object_mut()
            .expect("the mock run embeds the corpus");
        for field in TIMING_FIELDS {
            timings.remove(field);
        }
        serde_json::to_string(&value).expect("value serializes")
    }

    #[tokio::test]
    async fn two_mock_runs_produce_byte_identical_json_apart_from_timing() {
        // Arrange
        let embedder = liam_model::MockEmbedder::new(DIMS);

        // Act
        let first = evaluate_embedder(&embedder, "mock").await.expect("run 1");
        let second = evaluate_embedder(&embedder, "mock").await.expect("run 2");

        // Assert
        assert_eq!(json_without_timing(&first), json_without_timing(&second));
    }

    /// Given the seeded `CORPUS`/`QUESTIONS` fixtures, when every
    /// non-`VectorSemantic` question runs against a real in-memory store,
    /// then the harness completes without panicking, every scored metric
    /// lands in `[0.0, 1.0]`, every `GraphExpansion` question individually
    /// shows graph expansion (not just a lexical match) at work, and the
    /// printed report names all 8 categories with `VectorSemantic` marked
    /// not-run.
    #[tokio::test]
    async fn text_only_tier_scores_every_non_vector_category() {
        use std::sync::Arc;

        use liam_store::{DefaultGraph, FixedClock, GraphConfig};

        // Arrange: one shared time origin for both corpus seeding and every
        // AsOf/HalfLife knob, so they agree on "now".
        let base = Millis::now();
        // A fixed, harness-driven clock, not `DefaultGraph::open`'s real
        // system clock. `Graph::insert`'s bitemporal write also stamps each
        // row's transaction time (`tx_from`) from this clock
        // (`graph.rs:126`'s "live at T" predicate checks `tx_from <= t` in
        // addition to `valid_from <= t`); if every row's `tx_from` were the
        // real "now" (today), any `Knob::AsOf` query pointing further into
        // the past than today would find nothing live, regardless of
        // `valid_from`, since the row would not yet exist as far as the
        // store's own transaction time is concerned. Setting the clock to
        // each fact's own backdated instant before inserting it (below)
        // makes `tx_from` agree with `valid_from`, exactly as if the fact
        // had genuinely been recorded on the day it became true.
        let clock = Arc::new(FixedClock::new(base));
        let store =
            DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
                .await
                .expect("open in-memory store");

        // Text-only: no embedder, so no fact carries a vector (see the
        // module doc).
        let ids = seed_corpus(&store, &clock, base, None, &mut EmbedTimings::default())
            .await
            .expect("seed corpus");

        // Seed GraphExpansion edges.
        seed_graph_expansion_edges(&store, &ids)
            .await
            .expect("link GraphExpansion pair");

        // Act: run every non-VectorSemantic question, scoring each against
        // the metric functions.
        let mut per_category: HashMap<Category, Vec<CategoryMetrics>> = HashMap::new();
        let mut graph_expansion_expanded: Vec<(&str, bool)> = Vec::new();

        for question in QUESTIONS
            .iter()
            .filter(|q| q.category != Category::VectorSemantic)
        {
            let query = build_query(question, base, HARNESS_K, None);
            // An empty result is a valid 0.0-scoring outcome, not a harness
            // bug; only a genuine store error panics here.
            let hits = store
                .query_explained(&query)
                .await
                .expect("query_explained should succeed");

            if question.category == Category::GraphExpansion {
                graph_expansion_expanded.push((question.name, hits.iter().any(|h| h.expanded)));
            }

            let (retrieved, relevant) = retrieved_and_relevant(&hits, question, &ids);

            per_category
                .entry(question.category)
                .or_default()
                .push(score_question(&retrieved, &relevant));
        }

        // Assert: every scored metric is a valid, finite fraction.
        assert_metrics_in_unit_range(&per_category);

        // Assert: EVERY GraphExpansion question individually observed
        // expanded == true, not just any one of them: questions can share a
        // corpus pair, and a single OR'd bool would stay green even if
        // expansion broke for all but one of them.
        let graph_expansion_not_expanded: Vec<&str> = graph_expansion_expanded
            .iter()
            .filter(|(_, expanded)| !expanded)
            .map(|(name, _)| *name)
            .collect();
        assert!(
            graph_expansion_not_expanded.is_empty(),
            "expected every GraphExpansion question to have expanded == true, proving its edge \
             was seeded and consulted, but {graph_expansion_not_expanded:?} did not \
             (saw: {graph_expansion_expanded:?})"
        );

        // Aggregate per-category into the report; VectorSemantic is always
        // present but never scored this tier.
        let scores: Vec<CategoryScore> = Category::ALL
            .iter()
            .map(|&category| {
                if category == Category::VectorSemantic {
                    return CategoryScore {
                        category,
                        scored: 0,
                        metrics: None,
                    };
                }
                let empty = Vec::new();
                let results = per_category.get(&category).unwrap_or(&empty);
                CategoryScore {
                    category,
                    scored: results.len(),
                    metrics: average_category_metrics(results),
                }
            })
            .collect();

        for (category, floor) in TEXT_ONLY_RECALL_AT_10_FLOORS {
            let recall = scores
                .iter()
                .find(|score| score.category == category)
                .and_then(|score| score.metrics)
                .map(|metrics| metrics.recall_at_10)
                .unwrap_or_else(|| panic!("category {category:?} was not scored"));
            assert!(
                recall >= floor,
                "{category:?} recall@10 {recall:.3} fell below its baseline floor {floor:.3}"
            );
        }

        let report = format_report(&scores);
        println!("{report}");

        for category in Category::ALL {
            assert!(
                report.contains(category.name()),
                "report missing category {category:?}: {report}"
            );
        }
        let vector_semantic_line = report
            .lines()
            .find(|line| line.contains(Category::VectorSemantic.name()))
            .unwrap_or_else(|| panic!("report has no VectorSemantic row: {report}"));
        assert!(
            vector_semantic_line.contains("not run this tier"),
            "VectorSemantic row must read as not-run, not scored: {vector_semantic_line}"
        );
    }
}

/// Real-embedder tier: `#[ignore]`d since it downloads model weights on first
/// run (see the module doc's "Two tiers" section for the run command).
#[cfg(all(test, feature = "local"))]
mod real_embedder_run {
    use super::*;

    /// `LIAM_RETRIEVAL_EVAL_MODEL` overrides the model for a local A/B run,
    /// mirroring `eval.rs`'s `LIAM_EVAL_MODEL`.
    fn model() -> EvalModel {
        std::env::var("LIAM_RETRIEVAL_EVAL_MODEL")
            .unwrap_or_else(|_| Config::default().embedder.model)
            .parse()
            .expect("LIAM_RETRIEVAL_EVAL_MODEL names a supported model")
    }

    /// Given the seeded fixtures and a real local embedder, when every
    /// question runs including `VectorSemantic`, then every metric lands in
    /// `[0.0, 1.0]`, every `GraphExpansion` question shows expansion, and the
    /// printed report names all 8 categories.
    #[tokio::test]
    #[ignore = "downloads embedder weights; see module doc for the run command"]
    async fn real_embedder_tier_scores_every_category_including_vector_semantic() {
        // Act
        let report = run_retrieval_eval(model(), &Config::default())
            .await
            .expect("real embedder run");
        let text = report.render();
        println!("{text}");

        // Assert
        let mut per_category: HashMap<Category, Vec<CategoryMetrics>> = HashMap::new();
        for question in &report.per_question {
            per_category
                .entry(question.category)
                .or_default()
                .push(question.metrics);
        }
        assert_metrics_in_unit_range(&per_category);
        assert!(
            report.graph_expansion_unexpanded.is_empty(),
            "GraphExpansion questions without expansion: {:?}",
            report.graph_expansion_unexpanded
        );
        for category in Category::ALL {
            assert!(
                text.contains(category.name()),
                "report missing category {category:?}: {text}"
            );
        }
        let vector_questions = QUESTIONS
            .iter()
            .filter(|q| q.category == Category::VectorSemantic)
            .count();
        let timings = report.timings.expect("the real tier embeds every fact");
        assert_eq!(timings.calls, CORPUS.len() + vector_questions);
    }

    /// Given the same real model and corpus, when the harness runs twice,
    /// then the ranking-derived metrics match to within `EPSILON`. Not exact
    /// equality: multi-threaded matrix kernels may reduce in a different
    /// order across runs and flip a near-tied ranking.
    #[tokio::test]
    #[ignore = "downloads embedder weights; see module doc for the run command"]
    async fn two_real_runs_agree_on_every_ranking_metric() {
        // Arrange
        const EPSILON: f64 = 1e-9;
        let config = Config::default();

        // Act
        let first = run_retrieval_eval(model(), &config).await.expect("run 1");
        let second = run_retrieval_eval(model(), &config).await.expect("run 2");

        // Assert
        let (a, b) = (
            first.overall.expect("scored"),
            second.overall.expect("scored"),
        );
        for (x, y) in [
            (a.recall_at_10, b.recall_at_10),
            (a.ndcg_at_10, b.ndcg_at_10),
            (a.mrr, b.mrr),
            (a.precision_at_4, b.precision_at_4),
            (a.precision_at_8, b.precision_at_8),
            (a.recall_at_4, b.recall_at_4),
            (a.recall_at_8, b.recall_at_8),
        ] {
            assert!((x - y).abs() < EPSILON, "{x} vs {y}");
        }
    }
}
