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
/// `kind`/`content`/`scope`/`valid_from_offset_ms` are read only by the
/// harness this WU does not add (WU-3/WU-4 seed `NewNode` from them); the
/// fixture-consistency checks below only need `label`/`edge_target`/
/// `confidence`/`competes_with`. `allow(dead_code)` here is temporary and
/// removed once WU-3 reads the rest.
#[allow(dead_code)]
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
];

/// The category-specific extra a `Question` layers on top of its text query,
/// per the module doc's text-plus-knob rule. `Lexical`/`GraphExpansion`/
/// `Confidence` need nothing beyond `query_text` itself, since their "knob"
/// is structural (an edge, or a `competes_with` pair) rather than a `Query`
/// field, so they carry `Knob::None`.
///
/// The check below only matches each variant's discriminant (`Kind(_)`, not
/// the carried value); WU-3/WU-4 read the value itself to build a `Query`.
/// `allow(dead_code)` here is temporary and removed once WU-3 reads it.
#[allow(dead_code)]
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
        query_text: "Why did Kestrel Robotics recall the Falconer's motor?",
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
];

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
                // pair), checked by the two tests below instead.
                _ => true,
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
