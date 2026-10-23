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
//!   behind `#[cfg(feature = "local")]` and scores `VectorSemantic` too.
//!   Downloads model weights on first run and is slow, like `eval.rs`'s own
//!   gated tier, so it is `#[ignore]`d rather than part of plain
//!   `cargo test -p liam-daemon`:
//!
//!   ```text
//!   cargo test -p liam-daemon --features local -- --ignored --nocapture retrieval
//!   ```
//!
//!   Env override: `LIAM_RETRIEVAL_EVAL_MODEL`, mirroring `eval.rs`'s
//!   `LIAM_EVAL_MODEL` for local A/B runs against a different embedding
//!   model. Added in WU-4; see `mod real_embedder_run` below for the harness
//!   itself.
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
//! This is a **scored baseline, not a CI gate**: the only quality-threshold
//! assertions are the deterministic text-only tier's per-category recall@10
//! floors (see "Text-only baseline" below); the real-embedder tier has none
//! on its metric values. What IS
//! asserted (in WU-3/WU-4's harness tests) is structural: the harness does
//! not panic, every metric value lands in `[0, 1]`, every non-excluded
//! question produces a result, and each category's fixtures are internally
//! consistent. The numbers themselves are printed via `format_report` for a
//! maintainer to read and, when they change meaningfully, hand-transcribe
//! into a dated table here, the same convention `eval.rs` uses for its own
//! baselines.
//!
//! # Golden set
//!
//! 80 questions, 10 per category, over a synthetic corpus of one fictional
//! company. LIAM is pre-release and has no real usage or query logs, so this
//! set substitutes for the "real usage" query set the evaluation brief asks
//! for: it is a deliberate deviation, to be replaced or extended once actual
//! users exist. The target stays at 80, well short of the brief's 200 to 500,
//! because every fixture is hand-written and hand-checked. Each fact is
//! authored to contain the answer to the question that names it, so a
//! relevance label is correct by construction rather than judged after the
//! fact. Question names and corpus labels are stable ids that reports quote:
//! add new ones, never rename.
//!
//! # Text-only baseline (2026-10-23, 80 questions, the always-on tier)
//!
//! | category       | n  | precision@4 | precision@8 | recall@4 | recall@8 | recall@10 | nDCG@10 | MRR   |
//! |----------------|----|-------------|-------------|----------|----------|-----------|---------|-------|
//! | Lexical        | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | GraphExpansion | 10 | 0.375       | 0.225       | 0.750    | 0.900    | 0.900     | 0.851   | 1.000 |
//! | KindFilter     | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | ScopeFilter    | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//! | Decay          | 10 | 0.175       | 0.125       | 0.700    | 1.000    | 1.000     | 0.527   | 0.376 |
//! | Confidence     | 10 | 0.075       | 0.087       | 0.300    | 0.700    | 0.900     | 0.377   | 0.228 |
//! | AsOf           | 10 | 0.250       | 0.125       | 1.000    | 1.000    | 1.000     | 1.000   | 1.000 |
//!
//! Unlike the real-embedder tier, the text-only tier is deterministic, so
//! its recall@10 per category is asserted against the floors recorded in the
//! earlier 40-question run (see `TEXT_ONLY_RECALL_AT_10_FLOORS`): growing the
//! set must not regress what that run already measured.
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
//! | Lexical        | 5 | 0.250       | 0.125       | 1.000    | 1.000    || 1.000    | -         | -       | 1.000 |
//! | VectorSemantic | 5 | 0.250       | 0.125       | 1.000    | 1.000    || 1.000    | -         | -       | 1.000 |
//! | GraphExpansion | 5 | 0.450       | 0.225       | 0.900    | 0.900    || 1.000    | -         | -       | 1.000 |
//! | KindFilter     | 5 | 0.250       | 0.125       | 1.000    | 1.000    || 1.000    | -         | -       | 1.000 |
//! | ScopeFilter    | 5 | 0.250       | 0.125       | 1.000    | 1.000    || 1.000    | -         | -       | 1.000 |
//! | Decay          | 5 | 0.100       | 0.125       | 0.400    | 1.000    || 0.302    | -         | -       | 0.302 |
//! | Confidence     | 5 | 0.050       | 0.050       | 0.200    | 0.400    || 0.150    | -         | -       | 0.150 |
//! | AsOf           | 5 | 0.250       | 0.125       | 1.000    | 1.000    || 1.000    | -         | -       | 1.000 |
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
use std::time::Duration;

use liam_store::{DefaultGraph, ExplainedHit, Millis, NewEdge, NodeId, Query, Result};

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
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
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
#[derive(Debug, Clone, Copy)]
pub struct TimingSummary {
    pub calls: usize,
    pub items: usize,
    pub total: Duration,
    pub mean_latency: Duration,
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
            total,
            mean_latency: total / calls,
            throughput_per_sec,
        })
    }
}

/// One-line rendering of a `TimingSummary` for the report.
pub fn format_timing(summary: &TimingSummary) -> String {
    format!(
        "embedder: {} calls, {} items, mean latency {:.1} ms, throughput {:.1} items/s\n",
        summary.calls,
        summary.items,
        summary.mean_latency.as_secs_f64() * MS_PER_SEC,
        summary.throughput_per_sec,
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
    competes_with: Option<&'static str>,
}

const FACT_DEFAULTS: Fact = Fact {
    kind: "fact",
    label: "",
    content: "",
    scope: None,
    confidence: None,
    valid_from_offset_ms: None,
    edge_target: None,
    competes_with: None,
};

/// Fictional domain: Kestrel Robotics, a drone company, and its flagship
/// Falconer drone. Consistent across every entry so a reviewer can follow the
/// story; nothing here refers to a real person or company. Grouped under a
/// heading naming the category each block primarily serves; a handful of
/// facts (the `GraphExpansion` seeds) are also referenced by more than one
/// `Question`.
const CORPUS: &[Fact] = &[
    // Lexical: exact/near-term-matching facts with no other knob involved.
    Fact {
        kind: "fact",
        label: "Kestrel HQ",
        content: "Kestrel Robotics headquarters is located in Boulder, Colorado.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Kestrel Founding",
        content: "Kestrel Robotics was founded in 2019 by Priya Anand and Tomas Reyes.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Titanium Frame Decision",
        content: "The engineering team chose a titanium frame for the Falconer drone to cut \
                   weight by 20 percent.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Battery",
        content: "The Falconer drone uses a 6400 mAh lithium battery pack.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Falconer Demo Day",
        content: "On demo day the Falconer drone completed a 45 minute autonomous flight \
                   without a single course correction.",
        ..FACT_DEFAULTS
    },
    // VectorSemantic: real-embedder tier only (WU-4); this WU just authors
    // the fact and its paraphrase-target `Question.query_text`.
    Fact {
        kind: "fact",
        label: "Solar Charging Pilot",
        content: "Kestrel Robotics is piloting solar panel charging docks for its warehouse \
                   drones.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Remote Work Policy",
        content: "Kestrel Robotics allows engineers to work from home three days a week.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Drone Swarm Research",
        content: "Kestrel Robotics researchers are studying how small drone swarms coordinate \
                   obstacle avoidance.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Customer Onboarding Redesign",
        content: "The customer success team redesigned onboarding to cut new customer setup \
                   time in half.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Noise Reduction Rotor",
        content: "The newest Falconer rotor blades reduce audible flight noise by 12 decibels.",
        ..FACT_DEFAULTS
    },
    // GraphExpansion: each seed is matched lexically by its Question; each
    // linked fact carries edge_target naming its seed, so the harness can
    // Graph::link them and prove expansion, not just lexical match, found it.
    Fact {
        kind: "decision",
        label: "Falconer Motor Recall",
        content: "Kestrel Robotics issued a recall for the Falconer's left motor after bearing \
                   wear was reported.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Motor Vendor Notice",
        content: "The motor vendor notified Kestrel Robotics about a batch of bearings that \
                   wear faster than spec.",
        edge_target: Some("Falconer Motor Recall"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Weatherproof Casing",
        content: "The engineering team weatherproofed the Falconer casing to withstand heavy \
                   rain.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Field Test Storm Report",
        content: "A field test flew the Falconer through a thunderstorm and the weatherproof \
                   casing kept all electronics dry.",
        edge_target: Some("Weatherproof Casing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Logistics Partnership Signed",
        content: "Kestrel Robotics signed a last mile delivery partnership with SwiftHaul.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "SwiftHaul Pilot Results",
        content: "The SwiftHaul delivery pilot completed 500 drone deliveries with a 98 \
                   percent on time rate.",
        edge_target: Some("Logistics Partnership Signed"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Software Update Rollback",
        content: "Kestrel Robotics rolled back the v3.2 flight control firmware update after \
                   reports of mid flight instability.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Firmware Bug Report",
        content: "A pilot reported the Falconer drifting sideways during hover after \
                   installing firmware v3.2.",
        edge_target: Some("Software Update Rollback"),
        ..FACT_DEFAULTS
    },
    // KindFilter: each pair shares the same topic terms and differs only in
    // kind, so a wrong kind filter would let the distractor rank instead.
    Fact {
        kind: "decision",
        label: "Warehouse Automation Decision",
        content: "Kestrel Robotics decided to move forward with warehouse automation using \
                   robotic picking arms.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Warehouse Automation Kickoff",
        content: "The warehouse automation project kicked off with a training session for \
                   warehouse staff.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Data Retention Policy",
        content: "Kestrel Robotics retains flight log data for two years before deletion.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Data Retention Incident",
        content: "An audit found flight log data retention exceeded two years before the \
                   policy was enforced.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Supplier Fact",
        content: "Kestrel Robotics sources lithium battery cells from a supplier based in \
                   Nevada.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Battery Supplier Switch Decision",
        content: "Kestrel Robotics decided to switch lithium battery cell suppliers to one \
                   based in Nevada.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Certification Flight Episode",
        content: "The certification flight for the Falconer drone passed all FAA test points \
                   on the first attempt.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Certification Flight Scheduling",
        content: "Kestrel Robotics scheduled the certification flight for the Falconer drone \
                   after final firmware review.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Insurance Provider Decision",
        content: "Kestrel Robotics chose a new insurance provider for its commercial drone \
                   fleet.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Insurance Provider Coverage",
        content: "The new insurance provider covers up to two million dollars per drone \
                   incident for the commercial fleet.",
        ..FACT_DEFAULTS
    },
    // ScopeFilter: same pattern, scope is the only thing that differs
    // between each pair.
    Fact {
        kind: "fact",
        label: "Engineering Sprint Cadence",
        content: "The engineering team runs two week sprint cycles for the Falconer firmware \
                   roadmap.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Marketing Sprint Cadence",
        content: "The marketing team also runs two week sprint cycles for campaign planning.",
        scope: Some("marketing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Legal Compliance Review",
        content: "Legal completed a compliance review of the Falconer's export control \
                   classification.",
        scope: Some("legal"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operations Compliance Review",
        content: "Operations completed a separate compliance review of warehouse safety \
                   procedures.",
        scope: Some("operations"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Finance Budget Approval",
        content: "Finance approved the annual budget for the Falconer production line.",
        scope: Some("finance"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Engineering Budget Request",
        content: "Engineering submitted a budget request for the Falconer production line \
                   expansion.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Operations Fleet Maintenance",
        content: "Operations scheduled fleet wide maintenance for all active Falconer drones.",
        scope: Some("operations"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Engineering Maintenance Design",
        content: "Engineering redesigned the maintenance access panel on the Falconer drone.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Marketing Launch Campaign",
        content: "Marketing launched a campaign announcing the Falconer drone to retail \
                   customers.",
        scope: Some("marketing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Legal Launch Review",
        content: "Legal reviewed the marketing materials before the Falconer drone launch \
                   campaign.",
        scope: Some("legal"),
        ..FACT_DEFAULTS
    },
    // Decay: each pair states the same fact, with the newer half restating
    // it plus one clause, so decay (not a lexical difference) is what should
    // separate them.
    Fact {
        kind: "fact",
        label: "Camera Sensor Spec (2024)",
        content: "The Falconer drone camera sensor captures 12 megapixel stills at 30 frames \
                   per second.",
        valid_from_offset_ms: Some(-500 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Camera Sensor Spec (2026)",
        content: "The Falconer drone camera sensor captures 12 megapixel stills at 30 frames \
                   per second, now with improved low light performance.",
        valid_from_offset_ms: Some(-2 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Range Spec (Early)",
        content: "The Falconer drone has a maximum flight range of 8 kilometers on a single \
                   charge.",
        valid_from_offset_ms: Some(-400 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Range Spec (Current)",
        content: "The Falconer drone has a maximum flight range of 8 kilometers on a single \
                   charge, extended by the new battery pack.",
        valid_from_offset_ms: Some(-3 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Support Hours (Old)",
        content: "Kestrel Robotics customer support operates from 9 AM to 5 PM Mountain Time.",
        valid_from_offset_ms: Some(-300 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Support Hours (New)",
        content: "Kestrel Robotics customer support operates from 9 AM to 5 PM Mountain Time, \
                   now with weekend coverage added.",
        valid_from_offset_ms: Some(-5 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Pricing Tier (Old)",
        content: "The Falconer drone starter package is priced at four thousand dollars.",
        valid_from_offset_ms: Some(-200 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Pricing Tier (New)",
        content: "The Falconer drone starter package is priced at four thousand dollars, \
                   unchanged from last year.",
        valid_from_offset_ms: Some(-DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Team Size (Old)",
        content: "The Kestrel Robotics engineering team has thirty employees.",
        valid_from_offset_ms: Some(-600 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Team Size (New)",
        content: "The Kestrel Robotics engineering team has thirty employees, following a \
                   recent hiring freeze.",
        valid_from_offset_ms: Some(-4 * DAY_MS),
        ..FACT_DEFAULTS
    },
    // AsOf: each pair is an original fact later superseded by a changed one;
    // the Question's Knob::AsOf offset sits between the two, so only the
    // original should be visible at that instant.
    Fact {
        kind: "fact",
        label: "Product Name (Original)",
        content: "The Falconer drone was originally named the SkyHawk before its public \
                   release.",
        valid_from_offset_ms: Some(-500 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Product Name (Renamed)",
        content: "The drone formerly called SkyHawk was renamed Falconer ahead of its public \
                   release.",
        valid_from_offset_ms: Some(-490 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Headquarters (Original)",
        content: "Kestrel Robotics' original headquarters was a shared workspace in Denver.",
        valid_from_offset_ms: Some(-800 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Headquarters (Current)",
        content: "Kestrel Robotics moved its headquarters from the shared Denver workspace to \
                   a dedicated site in Boulder.",
        valid_from_offset_ms: Some(-700 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "CEO (Original)",
        content: "Priya Anand served as Kestrel Robotics' first CEO.",
        valid_from_offset_ms: Some(-900 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "CEO (Current)",
        content: "Tomas Reyes became Kestrel Robotics' CEO after Priya Anand stepped down.",
        valid_from_offset_ms: Some(-100 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Product Line (Original)",
        content: "Kestrel Robotics originally built only agricultural survey drones.",
        valid_from_offset_ms: Some(-1000 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Product Line (Current)",
        content: "Kestrel Robotics expanded from agricultural survey drones into the Falconer \
                   commercial delivery line.",
        valid_from_offset_ms: Some(-300 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Motor Vendor (Original)",
        content: "The Falconer's motor was originally sourced from a supplier called Torque \
                   Dynamics.",
        valid_from_offset_ms: Some(-450 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Motor Vendor (Current)",
        content: "Kestrel Robotics switched the Falconer's motor supplier from Torque Dynamics \
                   to a new vendor after the bearing recall.",
        valid_from_offset_ms: Some(-50 * DAY_MS),
        ..FACT_DEFAULTS
    },
    // Confidence: each pair states the same fact from a verified and an
    // unverified source; competes_with on the higher-confidence half names
    // the one it must outrank.
    Fact {
        kind: "fact",
        label: "Falconer Top Speed (Verified)",
        content: "Verified wind tunnel testing measured the Falconer drone's top speed at 65 \
                   kilometers per hour.",
        confidence: Some(0.95),
        competes_with: Some("Falconer Top Speed (Field Report)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Top Speed (Field Report)",
        content: "An early field report estimated the Falconer drone's top speed at around 60 \
                   kilometers per hour.",
        confidence: Some(0.4),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Payload Capacity (Verified)",
        content: "Verified payload testing confirmed the Falconer drone carries up to 3 \
                   kilograms.",
        confidence: Some(0.9),
        competes_with: Some("Payload Capacity (Rumor)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Payload Capacity (Rumor)",
        content: "An unconfirmed rumor claimed the Falconer drone could carry up to 5 \
                   kilograms.",
        confidence: Some(0.3),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Life (Verified)",
        content: "Verified bench testing measured the Falconer drone's battery life at 40 \
                   minutes of flight.",
        confidence: Some(0.85),
        competes_with: Some("Battery Life (Marketing Estimate)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Life (Marketing Estimate)",
        content: "An early marketing estimate claimed the Falconer drone's battery life was 50 \
                   minutes of flight.",
        confidence: Some(0.5),
        ..FACT_DEFAULTS
    },
    // Expansion block: five more facts (or pairs) per category, appended after
    // the original corpus so existing labels never move. Every fact's content
    // is written to contain the answer to the question that names it, which is
    // what makes its relevance label correct by construction. Like the rest of
    // the corpus this is synthetic: LIAM is pre-release with no real usage or
    // query logs, so these questions stand in for the "real usage" set the
    // evaluation brief asks for, to be replaced once actual users exist.
    //
    // Lexical
    Fact {
        kind: "fact",
        label: "Delivery Network Size",
        content: "Kestrel Robotics operates 120 Falconer drones in its delivery network.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Harrier Survey Drone",
        content: "The Harrier survey drone maps crop fields using a multispectral camera.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Charging Dock Standard",
        content: "Kestrel Robotics adopted the Qi2 inductive standard for every charging dock.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Aerospace Expo Booth",
        content: "At the Aerospace Expo in Denver the company booth drew 3000 visitors in two \
                   days.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Operator Certificate Policy",
        content: "Every Falconer operator must hold a Part 107 remote pilot certificate.",
        ..FACT_DEFAULTS
    },
    // VectorSemantic: each fact is the sole real semantic match for its
    // paraphrase and shares almost no vocabulary with it.
    Fact {
        kind: "fact",
        label: "Night Flight Waiver",
        content: "Kestrel Robotics received regulatory approval to operate its drones after \
                   sunset.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Employee Equity Plan",
        content: "Kestrel Robotics grants equity options to every full time employee after one \
                   year.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Cybersecurity Audit",
        content: "An external firm tested the drone control software for hacking \
                   vulnerabilities and found none critical.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Recycled Packaging",
        content: "Shipping boxes for Falconer spare parts are now made entirely from recycled \
                   cardboard.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Wildlife Disturbance Study",
        content: "A study found the drone flight paths rarely disturbed nesting birds near the \
                   Boulder test field.",
        ..FACT_DEFAULTS
    },
    // GraphExpansion: five more seed and linked pairs, same shape as above.
    Fact {
        kind: "decision",
        label: "Antenna Upgrade Decision",
        content: "Engineers replaced the telemetry antenna with a longer range model.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Ridge Signal Trial",
        content: "A ridge top trial confirmed the signal stayed solid at twelve kilometers.",
        edge_target: Some("Antenna Upgrade Decision"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Gimbal Redesign Decision",
        content: "The team redesigned the camera gimbal to stop vibration blur in footage.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Crisp Inspection Images",
        content: "Customers reported crisp inspection images after the stabilizer update \
                   shipped.",
        edge_target: Some("Gimbal Redesign Decision"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Training Academy Launch",
        content: "Kestrel Robotics opened a pilot training academy in Austin.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Academy First Cohort",
        content: "Twenty four students graduated the inaugural class with perfect safety scores.",
        edge_target: Some("Training Academy Launch"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Cloud Telemetry Migration",
        content: "Flight telemetry storage moved from on premise servers to a managed cloud \
                   database.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Hosting Cost Savings",
        content: "Hosting expenses dropped by 35 percent within one quarter.",
        edge_target: Some("Cloud Telemetry Migration"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Propeller Order Pause",
        content: "Kestrel Robotics paused orders from its propeller manufacturer over missed \
                   delivery dates.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Backup Blade Vendor",
        content: "A second manufacturer in Ohio agreed to supply carbon fiber blades within \
                   six weeks.",
        edge_target: Some("Propeller Order Pause"),
        ..FACT_DEFAULTS
    },
    // KindFilter: five more pairs sharing topic terms and differing in kind.
    Fact {
        kind: "decision",
        label: "Hangar Lease Decision",
        content: "Kestrel Robotics decided to lease a second hangar near the Boulder airfield.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Hangar Lease Walkthrough",
        content: "Leadership toured the second hangar near the Boulder airfield last week.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Battery Disposal Policy",
        content: "Retired batteries must be recycled through a certified hazardous waste handler.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Disposal Incident",
        content: "A retired battery sent to a standard waste handler overheated in transit.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Hospital Escalation Episode",
        content: "A hospital customer escalated a delayed delivery complaint to the executive \
                   team.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Hospital Escalation Decision",
        content: "The executive team decided to assign a dedicated manager to hospital customer \
                   delivery complaints.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Gust Tolerance Fact",
        content: "The Falconer safely operates in sustained gusts up to forty kilometers per \
                   hour.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Gust Grounding Policy",
        content: "Operators must ground the Falconer when sustained gusts exceed thirty \
                   kilometers per hour.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Open Source Firmware Decision",
        content: "Kestrel Robotics decided to publish the ground station firmware as open \
                   source.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Open Source Firmware Announcement",
        content: "The ground station firmware repository went public at a community meetup, \
                   open source at last.",
        ..FACT_DEFAULTS
    },
    // ScopeFilter: five more pairs differing only in scope.
    Fact {
        kind: "fact",
        label: "Support Ticket Backlog",
        content: "Support cleared the ticket backlog after adding two weekend shifts.",
        scope: Some("support"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Sales Ticket Backlog",
        content: "Sales reviewed the ticket backlog impact on renewal conversations.",
        scope: Some("sales"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "HR Hiring Plan",
        content: "HR opened twelve new roles for the firmware team this year.",
        scope: Some("hr"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Engineering Hiring Needs",
        content: "Engineering listed the firmware skills missing from the current team roles.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Security Access Audit",
        content: "Security audited badge access to the drone assembly hall.",
        scope: Some("security"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operations Badge Procedure",
        content: "Operations updated the badge access procedure for assembly hall visitors.",
        scope: Some("operations"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Sales Pipeline Review",
        content: "Sales reviewed the enterprise pipeline for delivery drone contracts.",
        scope: Some("sales"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Marketing Case Studies",
        content: "Marketing drafted case studies from the enterprise delivery drone contracts.",
        scope: Some("marketing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Finance Tax Credit",
        content: "Finance filed for a research tax credit covering drone development costs.",
        scope: Some("finance"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Legal Tax Credit Deadline",
        content: "Legal confirmed the research tax credit filing met the statutory deadline.",
        scope: Some("legal"),
        ..FACT_DEFAULTS
    },
    // Decay: five more pairs, the newer half restating the older plus a
    // clause.
    Fact {
        kind: "fact",
        label: "Warranty Term (Old)",
        content: "Falconer drones carry a one year warranty on all flight components.",
        valid_from_offset_ms: Some(-450 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Warranty Term (New)",
        content: "Falconer drones carry a one year warranty on all flight components, now \
                   extended to cover battery packs.",
        valid_from_offset_ms: Some(-4 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Charge Time (Old)",
        content: "A depleted Falconer battery reaches full charge in ninety minutes.",
        valid_from_offset_ms: Some(-380 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Charge Time (New)",
        content: "A depleted Falconer battery reaches full charge in ninety minutes, helped by \
                   the faster charging dock.",
        valid_from_offset_ms: Some(-3 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Depot Count (Old)",
        content: "Kestrel Robotics runs three regional depots for drone maintenance.",
        valid_from_offset_ms: Some(-350 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Depot Count (New)",
        content: "Kestrel Robotics runs three regional depots for drone maintenance, with \
                   another site under construction.",
        valid_from_offset_ms: Some(-2 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "App Rating (Old)",
        content: "The Kestrel mobile app holds a four point five star store rating.",
        valid_from_offset_ms: Some(-280 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "App Rating (New)",
        content: "The Kestrel mobile app holds a four point five star store rating, with over \
                   ten thousand reviews.",
        valid_from_offset_ms: Some(-4 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Lead Time (Old)",
        content: "New Falconer orders ship within six weeks of purchase.",
        valid_from_offset_ms: Some(-420 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Lead Time (New)",
        content: "New Falconer orders ship within six weeks of purchase, now tracked in the \
                   customer portal.",
        valid_from_offset_ms: Some(-DAY_MS),
        ..FACT_DEFAULTS
    },
    // AsOf: five more pairs, an original fact later superseded.
    Fact {
        kind: "fact",
        label: "Support Channel (Original)",
        content: "Customers originally reached support only by phone.",
        valid_from_offset_ms: Some(-600 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Support Channel (Current)",
        content: "Kestrel Robotics replaced phone only support with a chat portal.",
        valid_from_offset_ms: Some(-200 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Assembly Site (Original)",
        content: "The first Falconer assembly line ran from a leased warehouse in Longmont.",
        valid_from_offset_ms: Some(-700 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Assembly Site (Current)",
        content: "Assembly moved from the leased Longmont warehouse into the new Boulder \
                   factory.",
        valid_from_offset_ms: Some(-250 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Funding Round (Original)",
        content: "Kestrel Robotics closed a two million dollar seed round.",
        valid_from_offset_ms: Some(-650 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Funding Round (Current)",
        content: "Kestrel Robotics followed its seed round with a twenty million dollar \
                   Series A.",
        valid_from_offset_ms: Some(-300 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Flight Controller (Original)",
        content: "Early Falconer prototypes flew on an off the shelf open source autopilot \
                   board.",
        valid_from_offset_ms: Some(-550 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Flight Controller (Current)",
        content: "Kestrel Robotics replaced the open source autopilot board with an in house \
                   flight controller.",
        valid_from_offset_ms: Some(-150 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Delivery Radius (Original)",
        content: "Falconer deliveries were limited to a five kilometer radius around each \
                   depot.",
        valid_from_offset_ms: Some(-480 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Delivery Radius (Current)",
        content: "The delivery radius around each depot grew from five kilometers to eight.",
        valid_from_offset_ms: Some(-90 * DAY_MS),
        ..FACT_DEFAULTS
    },
    // Confidence: five more pairs; the verified half names the unverified
    // one it must outrank.
    Fact {
        kind: "fact",
        label: "Charge Cycles (Verified)",
        content: "Verified lab cycling showed the Falconer battery lasts 800 charge cycles.",
        confidence: Some(0.92),
        competes_with: Some("Charge Cycles (Forum Post)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Charge Cycles (Forum Post)",
        content: "A forum post claimed the Falconer battery lasts 1500 charge cycles.",
        confidence: Some(0.35),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operating Temperature (Verified)",
        content: "Chamber testing verified Falconer operation from minus ten to forty degrees \
                   Celsius.",
        confidence: Some(0.9),
        competes_with: Some("Operating Temperature (Blog)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operating Temperature (Blog)",
        content: "A blog post said the Falconer works down to minus thirty degrees Celsius.",
        confidence: Some(0.3),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Open Orders (Verified)",
        content: "The finance ledger shows 340 open Falconer orders.",
        confidence: Some(0.88),
        competes_with: Some("Open Orders (Hallway Rumor)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Open Orders (Hallway Rumor)",
        content: "Hallway chatter suggested 900 open Falconer orders.",
        confidence: Some(0.25),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Crash Rate (Verified)",
        content: "Flight logs verified a crash rate of one per ten thousand flights.",
        confidence: Some(0.93),
        competes_with: Some("Crash Rate (Anecdote)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Crash Rate (Anecdote)",
        content: "An anecdote from one customer implied a crash rate of one per thousand \
                   flights.",
        confidence: Some(0.4),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Ingress Rating (Verified)",
        content: "The certification report verified an IP55 ingress protection rating.",
        confidence: Some(0.95),
        competes_with: Some("Ingress Rating (Sales Slide)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Ingress Rating (Sales Slide)",
        content: "An old sales slide advertised an IP67 ingress protection rating.",
        confidence: Some(0.45),
        ..FACT_DEFAULTS
    },
];

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
}

const QUESTIONS: &[Question] = &[
    // Lexical
    Question {
        name: "lexical_hq_location",
        category: Category::Lexical,
        query_text: "Where is Kestrel Robotics headquarters located?",
        knob: Knob::None,
        expected_relevant: &["Kestrel HQ"],
    },
    Question {
        name: "lexical_founders",
        category: Category::Lexical,
        query_text: "Who founded Kestrel Robotics?",
        knob: Knob::None,
        expected_relevant: &["Kestrel Founding"],
    },
    Question {
        name: "lexical_titanium_frame",
        category: Category::Lexical,
        query_text: "What frame material did the engineering team choose for the Falconer \
                      drone?",
        knob: Knob::None,
        expected_relevant: &["Titanium Frame Decision"],
    },
    Question {
        name: "lexical_falconer_battery",
        category: Category::Lexical,
        query_text: "What battery does the Falconer drone use?",
        knob: Knob::None,
        expected_relevant: &["Falconer Battery"],
    },
    Question {
        name: "lexical_demo_day_flight",
        category: Category::Lexical,
        query_text: "How long did the Falconer drone fly autonomously on demo day?",
        knob: Knob::None,
        expected_relevant: &["Falconer Demo Day"],
    },
    // VectorSemantic (real-embedder tier only, WU-4): each query_text is a
    // paraphrase with weak lexical overlap against its expected fact.
    Question {
        name: "vector_solar_charging",
        category: Category::VectorSemantic,
        query_text: "Is the company experimenting with any renewable energy for recharging \
                      its fleet?",
        knob: Knob::None,
        expected_relevant: &["Solar Charging Pilot"],
    },
    Question {
        name: "vector_remote_work",
        category: Category::VectorSemantic,
        query_text: "Can staff do their jobs from a home office part of the week?",
        knob: Knob::None,
        expected_relevant: &["Remote Work Policy"],
    },
    Question {
        name: "vector_drone_swarm",
        category: Category::VectorSemantic,
        query_text: "What is being studied about groups of small flying robots avoiding \
                      obstacles together?",
        knob: Knob::None,
        expected_relevant: &["Drone Swarm Research"],
    },
    Question {
        name: "vector_onboarding_speed",
        category: Category::VectorSemantic,
        query_text: "Did the company make it quicker for new clients to get started?",
        knob: Knob::None,
        expected_relevant: &["Customer Onboarding Redesign"],
    },
    Question {
        name: "vector_quieter_rotors",
        category: Category::VectorSemantic,
        query_text: "Are the blades on the newest aircraft quieter than before?",
        knob: Knob::None,
        expected_relevant: &["Noise Reduction Rotor"],
    },
    // GraphExpansion: query_text surfaces the seed via lexical match;
    // expected_relevant names both the seed and its linked fact.
    Question {
        name: "graph_motor_recall_reason",
        category: Category::GraphExpansion,
        // Deliberately avoids "Kestrel Robotics": that phrase appears in
        // most of CORPUS (company-history/org facts), and at HARNESS_K's
        // widened k, fts5_query's stopword-OR would then lexically match
        // the entire corpus for this query, leaving nothing for
        // GraphExpansion to discover that wasn't already found lexically
        // (see the strengthened per-question assertion below, which
        // caught exactly this before this comment was added).
        query_text: "Why was the Falconer's motor recalled?",
        knob: Knob::None,
        expected_relevant: &["Falconer Motor Recall", "Motor Vendor Notice"],
    },
    Question {
        name: "graph_weatherproof_casing",
        category: Category::GraphExpansion,
        query_text: "How did the engineering team weatherproof the Falconer casing?",
        knob: Knob::None,
        expected_relevant: &["Weatherproof Casing", "Field Test Storm Report"],
    },
    Question {
        name: "graph_logistics_partnership",
        category: Category::GraphExpansion,
        query_text: "What logistics partnership did Kestrel Robotics sign?",
        knob: Knob::None,
        expected_relevant: &["Logistics Partnership Signed", "SwiftHaul Pilot Results"],
    },
    Question {
        name: "graph_firmware_rollback",
        category: Category::GraphExpansion,
        query_text: "Why was the v3.2 firmware update rolled back?",
        knob: Knob::None,
        expected_relevant: &["Software Update Rollback", "Firmware Bug Report"],
    },
    Question {
        name: "graph_motor_bearing_defect",
        category: Category::GraphExpansion,
        query_text: "What bearing defect caused the Falconer motor recall?",
        knob: Knob::None,
        expected_relevant: &["Falconer Motor Recall", "Motor Vendor Notice"],
    },
    // KindFilter
    Question {
        name: "kind_warehouse_decision",
        category: Category::KindFilter,
        query_text: "warehouse automation",
        knob: Knob::Kind("decision"),
        expected_relevant: &["Warehouse Automation Decision"],
    },
    Question {
        name: "kind_data_retention_policy",
        category: Category::KindFilter,
        query_text: "flight log data retention",
        knob: Knob::Kind("policy"),
        expected_relevant: &["Data Retention Policy"],
    },
    Question {
        name: "kind_battery_supplier_fact",
        category: Category::KindFilter,
        query_text: "lithium battery cell supplier Nevada",
        knob: Knob::Kind("fact"),
        expected_relevant: &["Battery Supplier Fact"],
    },
    Question {
        name: "kind_certification_flight_episode",
        category: Category::KindFilter,
        query_text: "certification flight Falconer drone",
        knob: Knob::Kind("episode"),
        expected_relevant: &["Certification Flight Episode"],
    },
    Question {
        name: "kind_insurance_provider_decision",
        category: Category::KindFilter,
        query_text: "insurance provider commercial drone fleet",
        knob: Knob::Kind("decision"),
        expected_relevant: &["Insurance Provider Decision"],
    },
    // ScopeFilter
    Question {
        name: "scope_engineering_sprint",
        category: Category::ScopeFilter,
        query_text: "two week sprint cycles",
        knob: Knob::Scope("engineering"),
        expected_relevant: &["Engineering Sprint Cadence"],
    },
    Question {
        name: "scope_legal_compliance",
        category: Category::ScopeFilter,
        query_text: "compliance review",
        knob: Knob::Scope("legal"),
        expected_relevant: &["Legal Compliance Review"],
    },
    Question {
        name: "scope_finance_budget",
        category: Category::ScopeFilter,
        query_text: "Falconer production line budget",
        knob: Knob::Scope("finance"),
        expected_relevant: &["Finance Budget Approval"],
    },
    Question {
        name: "scope_operations_maintenance",
        category: Category::ScopeFilter,
        query_text: "Falconer maintenance",
        knob: Knob::Scope("operations"),
        expected_relevant: &["Operations Fleet Maintenance"],
    },
    Question {
        name: "scope_marketing_launch",
        category: Category::ScopeFilter,
        query_text: "Falconer drone launch campaign",
        knob: Knob::Scope("marketing"),
        expected_relevant: &["Marketing Launch Campaign"],
    },
    // Decay
    Question {
        name: "decay_camera_sensor",
        category: Category::Decay,
        query_text: "Falconer drone camera sensor megapixel stills frames per second",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Camera Sensor Spec (2026)"],
    },
    Question {
        name: "decay_flight_range",
        category: Category::Decay,
        query_text: "Falconer drone maximum flight range single charge",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Range Spec (Current)"],
    },
    Question {
        name: "decay_support_hours",
        category: Category::Decay,
        query_text: "Kestrel Robotics customer support operating hours Mountain Time",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Support Hours (New)"],
    },
    Question {
        name: "decay_pricing_tier",
        category: Category::Decay,
        query_text: "Falconer drone starter package price",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Pricing Tier (New)"],
    },
    Question {
        name: "decay_team_size",
        category: Category::Decay,
        query_text: "Kestrel Robotics engineering team employees",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Team Size (New)"],
    },
    // AsOf
    Question {
        name: "asof_product_name",
        category: Category::AsOf,
        query_text: "What was the Falconer drone originally called before its public \
                      release?",
        knob: Knob::AsOf(-495 * DAY_MS),
        expected_relevant: &["Product Name (Original)"],
    },
    Question {
        name: "asof_headquarters",
        category: Category::AsOf,
        query_text: "Where was Kestrel Robotics' original headquarters before the move to \
                      Boulder?",
        knob: Knob::AsOf(-750 * DAY_MS),
        expected_relevant: &["Headquarters (Original)"],
    },
    Question {
        name: "asof_ceo",
        category: Category::AsOf,
        query_text: "Who was Kestrel Robotics' first CEO?",
        knob: Knob::AsOf(-500 * DAY_MS),
        expected_relevant: &["CEO (Original)"],
    },
    Question {
        name: "asof_product_line",
        category: Category::AsOf,
        query_text: "What kind of drones did Kestrel Robotics originally build?",
        knob: Knob::AsOf(-600 * DAY_MS),
        expected_relevant: &["Product Line (Original)"],
    },
    Question {
        name: "asof_motor_vendor",
        category: Category::AsOf,
        query_text: "Which vendor originally supplied the Falconer's motor?",
        knob: Knob::AsOf(-250 * DAY_MS),
        expected_relevant: &["Falconer Motor Vendor (Original)"],
    },
    // Confidence: query_text matches both halves of a pair; expected_relevant
    // names the higher-confidence half, whose competes_with names the loser
    // (checked below).
    Question {
        name: "confidence_top_speed_terms",
        category: Category::Confidence,
        query_text: "Falconer drone top speed kilometers per hour",
        knob: Knob::None,
        expected_relevant: &["Falconer Top Speed (Verified)"],
    },
    Question {
        name: "confidence_top_speed_phrase",
        category: Category::Confidence,
        query_text: "How fast can the Falconer drone fly?",
        knob: Knob::None,
        expected_relevant: &["Falconer Top Speed (Verified)"],
    },
    Question {
        name: "confidence_payload_terms",
        category: Category::Confidence,
        query_text: "Falconer drone payload capacity kilograms",
        knob: Knob::None,
        expected_relevant: &["Payload Capacity (Verified)"],
    },
    Question {
        name: "confidence_payload_phrase",
        category: Category::Confidence,
        query_text: "How much weight can the Falconer drone carry?",
        knob: Knob::None,
        expected_relevant: &["Payload Capacity (Verified)"],
    },
    Question {
        name: "confidence_battery_life",
        category: Category::Confidence,
        query_text: "Falconer drone battery life minutes of flight",
        knob: Knob::None,
        expected_relevant: &["Battery Life (Verified)"],
    },
    // Expansion block: five more questions per category, each naming the
    // facts the matching expansion block in CORPUS wrote to answer it.
    //
    // Lexical
    Question {
        name: "lexical_delivery_network_size",
        category: Category::Lexical,
        query_text: "How many drones are in the delivery network?",
        knob: Knob::None,
        expected_relevant: &["Delivery Network Size"],
    },
    Question {
        name: "lexical_harrier_survey",
        category: Category::Lexical,
        query_text: "Which drone maps crop fields with a multispectral camera?",
        knob: Knob::None,
        expected_relevant: &["Harrier Survey Drone"],
    },
    Question {
        name: "lexical_charging_dock_standard",
        category: Category::Lexical,
        query_text: "Which inductive standard do the charging docks use?",
        knob: Knob::None,
        expected_relevant: &["Charging Dock Standard"],
    },
    Question {
        name: "lexical_expo_visitors",
        category: Category::Lexical,
        query_text: "How many visitors came to the company booth at the Aerospace Expo?",
        knob: Knob::None,
        expected_relevant: &["Aerospace Expo Booth"],
    },
    Question {
        name: "lexical_operator_certificate",
        category: Category::Lexical,
        query_text: "What certificate must a Falconer operator hold?",
        knob: Knob::None,
        expected_relevant: &["Operator Certificate Policy"],
    },
    // VectorSemantic
    Question {
        name: "vector_night_flight",
        category: Category::VectorSemantic,
        query_text: "Is the firm allowed to fly its aircraft in the dark?",
        knob: Knob::None,
        expected_relevant: &["Night Flight Waiver"],
    },
    Question {
        name: "vector_equity_plan",
        category: Category::VectorSemantic,
        query_text: "Do workers get a share of ownership in the business after a while?",
        knob: Knob::None,
        expected_relevant: &["Employee Equity Plan"],
    },
    Question {
        name: "vector_security_audit",
        category: Category::VectorSemantic,
        query_text: "Did outsiders probe the flight software for ways attackers could break in?",
        knob: Knob::None,
        expected_relevant: &["Cybersecurity Audit"],
    },
    Question {
        name: "vector_recycled_packaging",
        category: Category::VectorSemantic,
        query_text: "Has the packaging for replacement components become more eco friendly?",
        knob: Knob::None,
        expected_relevant: &["Recycled Packaging"],
    },
    Question {
        name: "vector_wildlife_impact",
        category: Category::VectorSemantic,
        query_text: "Does flying the machines upset animals that live near the testing site?",
        knob: Knob::None,
        expected_relevant: &["Wildlife Disturbance Study"],
    },
    // GraphExpansion
    Question {
        name: "graph_antenna_upgrade",
        category: Category::GraphExpansion,
        query_text: "telemetry antenna replaced",
        knob: Knob::None,
        expected_relevant: &["Antenna Upgrade Decision", "Ridge Signal Trial"],
    },
    Question {
        name: "graph_gimbal_redesign",
        category: Category::GraphExpansion,
        query_text: "camera gimbal redesigned",
        knob: Knob::None,
        expected_relevant: &["Gimbal Redesign Decision", "Crisp Inspection Images"],
    },
    Question {
        name: "graph_training_academy",
        category: Category::GraphExpansion,
        query_text: "What came of opening the Austin training academy?",
        knob: Knob::None,
        expected_relevant: &["Training Academy Launch", "Academy First Cohort"],
    },
    Question {
        name: "graph_cloud_migration",
        category: Category::GraphExpansion,
        query_text: "flight telemetry storage managed cloud database",
        knob: Knob::None,
        expected_relevant: &["Cloud Telemetry Migration", "Hosting Cost Savings"],
    },
    Question {
        name: "graph_propeller_pause",
        category: Category::GraphExpansion,
        query_text: "Why were propeller orders paused?",
        knob: Knob::None,
        expected_relevant: &["Propeller Order Pause", "Backup Blade Vendor"],
    },
    // KindFilter
    Question {
        name: "kind_hangar_decision",
        category: Category::KindFilter,
        query_text: "second hangar Boulder airfield",
        knob: Knob::Kind("decision"),
        expected_relevant: &["Hangar Lease Decision"],
    },
    Question {
        name: "kind_battery_disposal_policy",
        category: Category::KindFilter,
        query_text: "retired batteries hazardous waste handler",
        knob: Knob::Kind("policy"),
        expected_relevant: &["Battery Disposal Policy"],
    },
    Question {
        name: "kind_hospital_escalation_episode",
        category: Category::KindFilter,
        query_text: "hospital customer delivery complaint",
        knob: Knob::Kind("episode"),
        expected_relevant: &["Hospital Escalation Episode"],
    },
    Question {
        name: "kind_gust_tolerance_fact",
        category: Category::KindFilter,
        query_text: "sustained gusts Falconer",
        knob: Knob::Kind("fact"),
        expected_relevant: &["Gust Tolerance Fact"],
    },
    Question {
        name: "kind_open_source_episode",
        category: Category::KindFilter,
        query_text: "ground station firmware open source",
        knob: Knob::Kind("episode"),
        expected_relevant: &["Open Source Firmware Announcement"],
    },
    // ScopeFilter
    Question {
        name: "scope_support_backlog",
        category: Category::ScopeFilter,
        query_text: "ticket backlog",
        knob: Knob::Scope("support"),
        expected_relevant: &["Support Ticket Backlog"],
    },
    Question {
        name: "scope_hr_hiring",
        category: Category::ScopeFilter,
        query_text: "firmware team roles",
        knob: Knob::Scope("hr"),
        expected_relevant: &["HR Hiring Plan"],
    },
    Question {
        name: "scope_security_badge",
        category: Category::ScopeFilter,
        query_text: "badge access assembly hall",
        knob: Knob::Scope("security"),
        expected_relevant: &["Security Access Audit"],
    },
    Question {
        name: "scope_sales_contracts",
        category: Category::ScopeFilter,
        query_text: "enterprise delivery drone contracts",
        knob: Knob::Scope("sales"),
        expected_relevant: &["Sales Pipeline Review"],
    },
    Question {
        name: "scope_legal_tax_credit",
        category: Category::ScopeFilter,
        query_text: "research tax credit",
        knob: Knob::Scope("legal"),
        expected_relevant: &["Legal Tax Credit Deadline"],
    },
    // Decay
    Question {
        name: "decay_warranty_term",
        category: Category::Decay,
        query_text: "Falconer warranty flight components",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Warranty Term (New)"],
    },
    Question {
        name: "decay_charge_time",
        category: Category::Decay,
        query_text: "Falconer battery full charge ninety minutes",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Charge Time (New)"],
    },
    Question {
        name: "decay_depot_count",
        category: Category::Decay,
        query_text: "Kestrel Robotics three regional depots for drone maintenance",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Depot Count (New)"],
    },
    Question {
        name: "decay_app_rating",
        category: Category::Decay,
        query_text: "Kestrel mobile app store rating",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["App Rating (New)"],
    },
    Question {
        name: "decay_lead_time",
        category: Category::Decay,
        query_text: "New Falconer orders ship within six weeks of purchase",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Lead Time (New)"],
    },
    // AsOf
    Question {
        name: "asof_support_channel",
        category: Category::AsOf,
        query_text: "How did customers originally reach support?",
        knob: Knob::AsOf(-400 * DAY_MS),
        expected_relevant: &["Support Channel (Original)"],
    },
    Question {
        name: "asof_assembly_site",
        category: Category::AsOf,
        query_text: "Where did the first Falconer assembly line run?",
        knob: Knob::AsOf(-500 * DAY_MS),
        expected_relevant: &["Assembly Site (Original)"],
    },
    Question {
        name: "asof_funding_round",
        category: Category::AsOf,
        query_text: "How large was the seed round Kestrel Robotics closed?",
        knob: Knob::AsOf(-500 * DAY_MS),
        expected_relevant: &["Funding Round (Original)"],
    },
    Question {
        name: "asof_flight_controller",
        category: Category::AsOf,
        query_text: "Which autopilot board did early Falconer prototypes use?",
        knob: Knob::AsOf(-350 * DAY_MS),
        expected_relevant: &["Flight Controller (Original)"],
    },
    Question {
        name: "asof_delivery_radius",
        category: Category::AsOf,
        query_text: "What was the delivery radius around each depot originally?",
        knob: Knob::AsOf(-300 * DAY_MS),
        expected_relevant: &["Delivery Radius (Original)"],
    },
    // Confidence
    Question {
        name: "confidence_charge_cycles",
        category: Category::Confidence,
        query_text: "Falconer battery charge cycles",
        knob: Knob::None,
        expected_relevant: &["Charge Cycles (Verified)"],
    },
    Question {
        name: "confidence_operating_temperature",
        category: Category::Confidence,
        query_text: "Falconer operating temperature degrees Celsius",
        knob: Knob::None,
        expected_relevant: &["Operating Temperature (Verified)"],
    },
    Question {
        name: "confidence_open_orders",
        category: Category::Confidence,
        query_text: "How many open Falconer orders are there?",
        knob: Knob::None,
        expected_relevant: &["Open Orders (Verified)"],
    },
    Question {
        name: "confidence_crash_rate",
        category: Category::Confidence,
        query_text: "What is the Falconer crash rate per flights?",
        knob: Knob::None,
        expected_relevant: &["Crash Rate (Verified)"],
    },
    Question {
        name: "confidence_ingress_rating",
        category: Category::Confidence,
        query_text: "Falconer ingress protection rating",
        knob: Knob::None,
        expected_relevant: &["Ingress Rating (Verified)"],
    },
];

/// Build the `Query` a `Question` describes: text plus its knob layered on
/// top, using `base` as the shared time origin for `Knob::AsOf` (the module
/// doc's "two tiers" section and `Fact::valid_from_offset_ms`'s doc comment
/// share this convention). Module-level and outside every cfg gate, not just
/// `mod tests`'s: both the text-only tier below and WU-4's real-embedder
/// tier build queries the same way, and duplicating this per tier would let
/// the two drift apart silently.
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

/// Score one question's already-run retrieval against WU-1's metric
/// functions. Shared with WU-4 for the same reason as `build_query` above.
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
/// `0.0`). Shared with WU-4 for the same reason as `build_query` above.
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
/// for readability, not correctness. Shared with WU-4 for the same reason
/// as `build_query` above: both tiers seed the exact same edges the exact
/// same way.
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
/// `retrieved`/`relevant` pair WU-1's metric functions (`precision_at_k`,
/// `recall_at_k`, `reciprocal_rank`) expect. Shared with WU-4 for the same
/// reason as `build_query` above.
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
/// fraction in `[0.0, 1.0]`. Shared with WU-4 for the same reason as
/// `build_query` above.
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

    /// Per-category recall@10 the text-only tier must keep meeting. Taken
    /// from the 40-question baseline, so growing the corpus may not regress
    /// retrieval quality on the categories that baseline already covered.
    const TEXT_ONLY_RECALL_AT_10_FLOORS: [(Category, f64); 7] = [
        (Category::Lexical, 1.0),
        (Category::GraphExpansion, 0.9),
        (Category::KindFilter, 1.0),
        (Category::ScopeFilter, 1.0),
        (Category::Decay, 1.0),
        (Category::Confidence, 0.8),
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
        assert_eq!(summary.total, Duration::from_millis(400));
        assert_eq!(summary.mean_latency, Duration::from_millis(200));
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

    /// Mock embedding width: irrelevant to this tier, since no fact is ever
    /// seeded with an embedding (see the module doc's "text-only" tier), but
    /// `GraphConfig` still needs a configured vector-table dimension to open.
    /// Matches `eval.rs`'s own placeholder (`crates/liam-daemon/src/eval.rs`).
    const DIMS: usize = 8;

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

        use liam_store::{DefaultGraph, FixedClock, GraphConfig, NewNode};

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

        // Seed every fact, text-only: no `.with_embedding(...)` call, since
        // every question this tier runs is text-only and there is nothing
        // for a seeded vector to be read by (see the module doc).
        let mut ids: HashMap<&str, NodeId> = HashMap::new();
        for fact in CORPUS {
            let valid_from = fact
                .valid_from_offset_ms
                .map(|offset| Millis(base.0 + offset));
            clock.set(valid_from.unwrap_or(base));

            let mut node = NewNode::now(fact.kind, fact.label, fact.content);
            if let Some(scope) = fact.scope {
                node = node.with_scope(scope);
            }
            if let Some(confidence) = fact.confidence {
                node = node.with_confidence(confidence);
            }
            if let Some(valid_from) = valid_from {
                node = node.with_valid_from(valid_from);
            }
            let id = store.insert(node).await.expect("seed corpus fact");
            ids.insert(fact.label, id);
        }

        // Back to "today" for edges and every query below: GraphExpansion
        // facts carry no offset of their own, and every non-`AsOf`
        // question's implicit "now" (its `Query.as_of` stays `None`, so
        // `query_core` falls back to `self.clock.now()`) must resolve to
        // `base`, not to whichever backdated instant the corpus loop above
        // last left the clock at.
        clock.set(base);

        // Seed GraphExpansion edges.
        seed_graph_expansion_edges(&store, &ids)
            .await
            .expect("link GraphExpansion pair");

        // Act: run every non-VectorSemantic question, scoring each against
        // WU-1's metrics.
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
        // expanded == true, not just any one of them (two of the five
        // target the same corpus pair, so at most 4 distinct edges are
        // represented; a single OR'd bool would stay green even if
        // expansion broke for 4 of the 5 questions).
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

/// Real-embedder tier: gated behind `feature = "local"` and `#[ignore]`d,
/// since it downloads model weights on first run and takes real time (see
/// the module doc's "Two tiers" section for the exact run command). A
/// sibling of `mod tests`, not nested inside it, specifically so it can
/// `use super::*;` and reuse `build_query`/`score_question`/
/// `average_category_metrics`/`HARNESS_K`/`seed_graph_expansion_edges`/
/// `retrieved_and_relevant`/`assert_metrics_in_unit_range` the same way
/// `mod tests` does. The corpus-seeding loop itself stays duplicated
/// between the two tiers below (this one embeds each fact, `mod tests`'s
/// does not); that is the one piece of the harness that genuinely differs
/// by tier.
#[cfg(feature = "local")]
mod real_embedder_run {
    use std::time::Instant;

    use super::*;

    /// Read an env override, falling back to `default` when unset. Same
    /// small shape as `eval.rs`'s own `env_or`; redefined locally rather
    /// than shared across modules since it is a two-line function with no
    /// state to keep in sync.
    fn env_or(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }

    /// Given the seeded `CORPUS`/`QUESTIONS` fixtures and a real local
    /// embedder, when every question runs (including `VectorSemantic`, which
    /// the text-only tier above skips) against a real in-memory store, then
    /// the harness completes without panicking, every scored metric lands in
    /// `[0.0, 1.0]`, every `GraphExpansion` question individually shows
    /// graph expansion at work, and the printed report names all 8
    /// categories.
    #[tokio::test]
    #[ignore = "downloads embedder weights; see module doc for the run command"]
    async fn real_embedder_tier_scores_every_category_including_vector_semantic() {
        use std::sync::Arc;

        use liam_model::{Embedder, TextKind};
        use liam_store::{DefaultGraph, FixedClock, GraphConfig, NewNode};

        // Arrange: load the real embedder first, since its output width
        // sizes the store's vector table below. `model_id`/`dims` read the
        // daemon's own shipped defaults (`Config::default()`), not a
        // hardcoded copy, so this harness keeps measuring whatever the
        // daemon actually ships; `LIAM_RETRIEVAL_EVAL_MODEL` overrides the
        // model for a local A/B run, mirroring `eval.rs`'s `LIAM_EVAL_MODEL`.
        let model_id = env_or(
            "LIAM_RETRIEVAL_EVAL_MODEL",
            &crate::config::Config::default().embedder.model,
        );
        let dims = crate::config::Config::default().embedding_dims;
        let cache_dir = liam_daemon::models::resolve_path_with_home(
            "embedder.cache_dir",
            &crate::config::Config::default().embedder.cache_dir,
            &std::env::var("HOME").unwrap_or_default(),
        )
        .expect("resolve embedder.cache_dir");
        let embedder = liam_model::FastEmbedEmbedder::load(&model_id, dims, &cache_dir)
            .expect("load real embedder");

        // Untimed warm-up so one-off model initialisation does not inflate
        // the first timed call.
        embedder
            .embed("warm-up", TextKind::Document)
            .await
            .expect("warm-up embed");

        // Same bitemporal-clock pattern as the text-only tier's test above
        // (see its comment for why `DefaultGraph::open`'s real clock would
        // break `Knob::AsOf`), but this tier's own store instance,
        // independent of that test's.
        let base = Millis::now();
        let clock = Arc::new(FixedClock::new(base));
        let store =
            DefaultGraph::open_with_clock(":memory:", GraphConfig::new(dims), clock.clone())
                .await
                .expect("open in-memory store");

        // Seed every fact with a real embedding this time, in addition to
        // everything the text-only tier already seeds.
        let mut ids: HashMap<&str, NodeId> = HashMap::new();
        let mut timings = EmbedTimings::default();
        for fact in CORPUS {
            let valid_from = fact
                .valid_from_offset_ms
                .map(|offset| Millis(base.0 + offset));
            clock.set(valid_from.unwrap_or(base));

            let started = Instant::now();
            let embedding = embedder
                .embed(fact.content, TextKind::Document)
                .await
                .expect("embed corpus fact");
            timings.record(1, started.elapsed());
            let mut node =
                NewNode::now(fact.kind, fact.label, fact.content).with_embedding(embedding);
            if let Some(scope) = fact.scope {
                node = node.with_scope(scope);
            }
            if let Some(confidence) = fact.confidence {
                node = node.with_confidence(confidence);
            }
            if let Some(valid_from) = valid_from {
                node = node.with_valid_from(valid_from);
            }
            let id = store.insert(node).await.expect("seed corpus fact");
            ids.insert(fact.label, id);
        }

        // Back to "today" for edges and every query below, same reason as
        // the text-only tier's test.
        clock.set(base);

        // Seed GraphExpansion edges, same direction convention as the
        // text-only tier's test.
        seed_graph_expansion_edges(&store, &ids)
            .await
            .expect("link GraphExpansion pair");

        // Act: run every question, no category filter this time, scoring
        // each against WU-1's metrics. VectorSemantic questions get their
        // query text embedded too, then passed through `build_query`.
        let mut per_category: HashMap<Category, Vec<CategoryMetrics>> = HashMap::new();
        let mut graph_expansion_expanded: Vec<(&str, bool)> = Vec::new();

        for question in QUESTIONS {
            let embedding = if question.category == Category::VectorSemantic {
                let started = Instant::now();
                let embedding = embedder
                    .embed(question.query_text, TextKind::Query)
                    .await
                    .expect("embed VectorSemantic query");
                timings.record(1, started.elapsed());
                Some(embedding)
            } else {
                None
            };
            let query = build_query(question, base, HARNESS_K, embedding);
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
        // expanded == true, same strengthened check as the text-only
        // tier's test above (see its comment for why a single OR'd bool is
        // too weak here).
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

        // Aggregate per-category into the report; every category is scored
        // this tier, including VectorSemantic (no not-run special case).
        let scores: Vec<CategoryScore> = Category::ALL
            .iter()
            .map(|&category| {
                let empty = Vec::new();
                let results = per_category.get(&category).unwrap_or(&empty);
                CategoryScore {
                    category,
                    scored: results.len(),
                    metrics: average_category_metrics(results),
                }
            })
            .collect();

        let report = format_report(&scores);
        println!("{report}");

        let summary = timings
            .summary()
            .expect("the real tier embeds every corpus fact");
        println!("{}", format_timing(&summary));
        println!("(timings exclude the untimed warm-up embed call)");
        let vector_semantic_questions = QUESTIONS
            .iter()
            .filter(|q| q.category == Category::VectorSemantic)
            .count();
        assert_eq!(summary.calls, CORPUS.len() + vector_semantic_questions);

        for category in Category::ALL {
            assert!(
                report.contains(category.name()),
                "report missing category {category:?}: {report}"
            );
        }
    }
}
