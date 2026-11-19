// SPDX-License-Identifier: Apache-2.0
//! The graph handle, generic over a `Backend`. Shared logic (write, read, GC,
//! clustering) routes through the backend trait; only vector storage and search
//! differ per backend.
//!
//! Visibility uses a full bitemporal predicate: a node is visible "as of" an
//! instant T when it was recorded before T, not yet superseded at T, and true
//! in the world at T. Passing a past T yields point-in-time recall.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use liam_log::event::{BasisRef, NodeRow};
use liam_log::LogOffset;

pub use self::backfill::BackfillReport;
pub use self::deletion_log::DeletionRecord;
pub use self::fact_read::{FactPage, MAX_SOURCES_PER_FACT};
pub use self::fact_write::MAX_FACTS_PER_BATCH;
pub use self::facts::batch_key;
pub use self::lineage::{LineageStore, GRAPH_STORE};
use self::logged_plan::{refuse_supersedes, Collision, STORE_SOURCE};
use self::logged_write::WriteOutcome;
pub use self::logged_write::{EventLog, PoisonHook, SharedLog};
use self::node_events::ListedEvents;
pub use self::rebuild::{RebuildMode, RebuildReport};
pub use self::reembed::{ContentEmbedder, EmbedError, ReembedReport};
pub use self::replay::CatchUpReport;
pub use self::rolled_back::RUN_SOURCE_PREFIX;
pub use self::scrub::ScrubReport;
pub use self::snapshot::{
    recover_interrupted_restore, verify_snapshot, Snapshot, SnapshotLog, SnapshotMeta,
};
pub use self::stats::{QueryTimings, TierBytes};
pub use self::summaries::child_hash;
pub use self::tiers::{
    dequantize, hamming, quantize, IndexedEntry, QuantIndex, Quantized, BITS_PER_WORD,
};
use self::tiers::{ColdArm, ColdQuery};
use crate::backend::Backend;
use crate::clock::{Clock, SystemClock};
use crate::error::{Error, Result};
use crate::ids::{EdgeId, Millis, NodeId, FOREVER};
use crate::schema::schema;
use crate::types::{
    Basis, Change, ClusterMember, ClusterState, Derivation, EpisodeEdge, EpisodeRef, EpisodeResult,
    ExplainedHit, Fingerprint, GraphConfig, Hit, InferredHit, InferredRows, InferredText, NewEdge,
    NewNode, Query, RecordedHit, RecordedText, Retrieved,
};
use crate::value::{Row, Value};

mod arbitration;
mod backfill;
mod deletion_log;
mod fact_projection;
mod fact_read;
mod fact_write;
mod facts;
mod gc;
mod lineage;
mod log_cursor;
mod log_open;
mod logged_plan;
mod logged_write;
mod node_events;
mod projection;
mod rebuild;
mod reembed;
mod replay;
mod rolled_back;
mod row_keys;
mod run_marker;
mod scrub;
mod snapshot;
mod stats;
mod summaries;
mod swept_by;
mod tiers;

/// How many candidates an ambiguous handle reports back. Bounded so a
/// one-character handle answers with something a caller can act on instead of
/// every live node in the store.
const HANDLE_MATCH_LIMIT: usize = 8;

/// The combined guard an edge INSERT conditions on, and the diagnostic that
/// names which of the three failed: source live, target live, edge already
/// exists. Shared as one literal so the guard and its diagnosis can never drift
/// apart on the SQL itself. Bind order: src, dst, `FOREVER`, kind.
const EDGE_REFUSAL_DIAGNOSTIC_SQL: &str = "SELECT
   EXISTS(SELECT 1 FROM nodes WHERE id = ?1 AND tx_to = ?3),
   EXISTS(SELECT 1 FROM nodes WHERE id = ?2 AND tx_to = ?3),
   EXISTS(SELECT 1 FROM edges
          WHERE src = ?1 AND dst = ?2 AND type = ?4 AND tx_to = ?3)";

/// The conditional edge INSERT: insert the row only if the source is live, the
/// target is live, and no identical edge already exists for the ordered triple
/// (src, dst, type). Shared as one literal, matching
/// `EDGE_REFUSAL_DIAGNOSTIC_SQL`, so the guard and its diagnosis can never drift
/// apart. Bind order: id, src, dst, type, attributes, tx_from, tx_to, FOREVER.
const EDGE_INSERT_SQL: &str = "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
     SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7
     WHERE EXISTS     (SELECT 1 FROM nodes WHERE id = ?2 AND tx_to = ?8)
       AND EXISTS     (SELECT 1 FROM nodes WHERE id = ?3 AND tx_to = ?8)
       AND NOT EXISTS (SELECT 1 FROM edges
                       WHERE src = ?2 AND dst = ?3 AND type = ?4 AND tx_to = ?8)";

/// The attributes of an edge no caller described, such as one `relate` writes.
const EMPTY_ATTRIBUTES: &str = "{}";

/// The specific refusal message for a failed edge write, given the three
/// flags `EDGE_REFUSAL_DIAGNOSTIC_SQL` produces.
fn edge_refusal(
    source_live: bool,
    target_live: bool,
    edge_exists: bool,
    src: &NodeId,
    dst: &NodeId,
    kind: &str,
) -> Error {
    if !source_live {
        return Error::RelateRefused(format!("source node {} is not live", src.as_str()));
    }
    if !target_live {
        return Error::RelateRefused(format!("target node {} is not live", dst.as_str()));
    }
    if edge_exists {
        return Error::RelateRefused(edge_exists_message(src, dst, kind));
    }
    // Every guard passes now, so one of them flipped between the insert and
    // this read. A retry would land.
    Error::ConcurrentWrite
}

/// The refusal text for a triple that already has a live edge. Retry classifiers
/// match on its wording, so every refusal over a twin starts with it.
fn edge_exists_message(src: &NodeId, dst: &NodeId, kind: &str) -> String {
    format!(
        "{} already relates to {} as '{kind}'",
        src.as_str(),
        dst.as_str()
    )
}

/// The three flags of `EDGE_REFUSAL_DIAGNOSTIC_SQL`'s row: source live, target
/// live, edge already exists.
fn edge_guard_flags(row: &Row) -> Result<[bool; 3]> {
    let flag = |i| row.get_i64(i).map(|value| value != 0);
    Ok([flag(0)?, flag(1)?, flag(2)?])
}

/// What a single `relate` attempt resolved to, once a caller replaying a
/// batch of edges needs to tell a safe no-op from a real problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeOutcome {
    /// The edge row was written.
    Inserted,
    /// `relate` refused because the identical triple already exists: the
    /// edge is already there, so replaying this attempt again is a no-op.
    AlreadyPresent,
    /// `relate` refused because the source or target node is not live: not
    /// this call's fault, and retrying it will not change that.
    NotLive,
    /// Anything else, including a non-`RelateRefused` error: worth surfacing
    /// rather than silently treated as a safe no-op.
    RealFailure,
}

/// Classifies a `relate` result into the four outcomes a batch replay cares
/// about, reading `RelateRefused`'s message the same way `edge_refusal`
/// built it.
pub fn classify_relate_outcome(result: &Result<EdgeId>) -> EdgeOutcome {
    match result {
        Ok(_) => EdgeOutcome::Inserted,
        Err(Error::RelateRefused(message)) if message.contains("already relates to") => {
            EdgeOutcome::AlreadyPresent
        }
        Err(Error::RelateRefused(message)) if message.contains("is not live") => {
            EdgeOutcome::NotLive
        }
        Err(_) => EdgeOutcome::RealFailure,
    }
}

/// `classify_relate_outcome` for a write that reports what it did. A relate that
/// deduplicated against an indexed twin returns the first edge's id but writes
/// nothing, so it is not an insert.
fn repair_outcome(result: Result<(EdgeId, WriteOutcome)>) -> EdgeOutcome {
    match result {
        Ok((_, WriteOutcome::Duplicate(_))) => EdgeOutcome::AlreadyPresent,
        other => classify_relate_outcome(&other.map(|(id, _)| id)),
    }
}

/// Folds one `relate` outcome into a per-`SUPERSEDES`-edge repair tally, so
/// two independent repair branches (mentions pointing at the superseded node,
/// and the superseded node's own outgoing mentions) can share one combined
/// outcome for the watermark.
fn fold_repair_outcome(
    outcome: EdgeOutcome,
    inserted: &mut usize,
    saw_insert: &mut bool,
    saw_failure: &mut bool,
) {
    match outcome {
        EdgeOutcome::Inserted => {
            *inserted += 1;
            *saw_insert = true;
        }
        EdgeOutcome::RealFailure => *saw_failure = true,
        EdgeOutcome::AlreadyPresent | EdgeOutcome::NotLive => {}
    }
}

/// The newest `tx_from` a batch replay can safely resume from, given the
/// outcomes it has processed so far. Bounded on the VALUE of the first
/// failure's `tx_from`, not its position in `processed`: an `ingest_episode`
/// call can stamp several supersessions with one shared `tx_from`, so two
/// entries can tie, and whichever of them is a `RealFailure` has to exclude
/// every entry at or after that same timestamp, regardless of which one
/// happened to be processed first. Returns `None` when there is no safe
/// entry to resume from, either because `processed` is empty or because the
/// earliest failure's `tx_from` is also the smallest timestamp present.
pub fn next_watermark(processed: &[(Millis, EdgeOutcome)]) -> Option<Millis> {
    let failure_bound = processed
        .iter()
        .find(|(_, outcome)| *outcome == EdgeOutcome::RealFailure)
        .map(|(tx_from, _)| *tx_from);
    match failure_bound {
        Some(bound) => processed
            .iter()
            .filter(|(tx_from, _)| *tx_from < bound)
            .map(|(tx_from, _)| *tx_from)
            .max(),
        None => processed.iter().map(|(tx_from, _)| *tx_from).max(),
    }
}

/// Resolve an `EpisodeRef` to the node id it names inside one
/// `ingest_episode` call: the pre-generated id at its index, for `New`, or
/// the given id, for `Existing`. Every `New` index is validated in bounds
/// before `ingest_episode` opens a transaction, so indexing here never
/// panics.
fn resolve_episode_ref<'a>(r: &'a EpisodeRef, ids: &'a [NodeId]) -> &'a NodeId {
    match r {
        EpisodeRef::New(i) => &ids[*i],
        EpisodeRef::Existing(id) => id,
    }
}

/// Bitemporal "live at T" predicate over an aliased nodes table. `?{t}` is T.
fn live_at(alias: &str, t: usize) -> String {
    format!(
        "{alias}.tx_from <= ?{t} AND {alias}.tx_to > ?{t} \
         AND {alias}.valid_from <= ?{t} AND {alias}.valid_until > ?{t}"
    )
}

/// Scope-hierarchy read predicate: `col` matches the scope bound at `?{n}`
/// exactly, or is a descendant separated by `/` (e.g. `work` also matches
/// `work/liam`, but not `work-archive`). SQLite lets a numbered placeholder
/// repeat, so this needs only the one bound param at `?{n}`. Shared by
/// `lexical`, `fetch_candidates`, and `backends::libsql::vector_search`.
pub(crate) fn scope_within(col: &str, n: usize) -> String {
    format!("({col} = ?{n} OR substr({col}, 1, length(?{n})+1) = ?{n} || '/')")
}

/// SQL + params for "the live node with this subject (and scope, and
/// producer if given), if any" as of `as_of`. For reads: a write replaces the
/// row `open_by_subject_query` finds, whatever its clock reads.
fn live_by_subject_query(
    subject: &str,
    as_of: Millis,
    scope: Option<&str>,
    producer: Option<&str>,
) -> (String, Vec<Value>) {
    by_subject_query(subject, &live_at("nodes", 2), as_of.into(), scope, producer)
}

/// SQL + params for "the open node with this subject (and scope, and producer
/// if given), if any", as `(id, tx_from)`. Run against a write's open
/// transaction by every write that replaces a competitor. Open means
/// `tx_to = FOREVER`, the row the close step can end, so it is a fact about the
/// store and never about the caller's clock. Valid time is not tested, so a
/// write replaces an open row even when its `valid_from` is in the future. A
/// `valid_from <= now` guard cannot be added: a row written at clock 3000 has
/// `valid_from` 3000 by default, so it would hide that row from a write at 1000
/// and leave two open rows for the subject, the backwards-clock bug again.
fn open_by_subject_query(
    subject: &str,
    scope: Option<&str>,
    producer: Option<&str>,
) -> (String, Vec<Value>) {
    by_subject_query(subject, "nodes.tx_to = ?2", FOREVER.into(), scope, producer)
}

/// The shared collision key of both subject queries: `liveness` is the
/// predicate over `nodes` bound to `?2`, which is `liveness_param`.
fn by_subject_query(
    subject: &str,
    liveness: &str,
    liveness_param: Value,
    scope: Option<&str>,
    producer: Option<&str>,
) -> (String, Vec<Value>) {
    let mut params: Vec<Value> = vec![subject.into(), liveness_param];
    let mut filters = String::new();
    let mut next = 3;
    if let Some(s) = scope {
        params.push(s.into());
        filters.push_str(&format!(" AND scope = ?{next}"));
        next += 1;
    } else {
        filters.push_str(" AND scope IS NULL");
    }
    if let Some(p) = producer {
        params.push(p.into());
        filters.push_str(&format!(" AND producer = ?{next}"));
    } else {
        // Producer-blind collision: only an entity page may collide here, never
        // a fact that happens to share the same subject and scope. A page is
        // told by its derivation and not by empty content, because a page that
        // has been written holds text.
        filters.push_str(&format!(
            " AND derivation = '{}'",
            Derivation::Inferred.as_str()
        ));
    }
    // If two nodes ever share a subject+scope, pick the newest
    // deterministically (tie-break by id) rather than an arbitrary row.
    let sql = format!(
        "SELECT id, tx_from FROM nodes WHERE subject = ?1 AND {liveness}{filters}
         ORDER BY tx_from DESC, id DESC",
    );
    (sql, params)
}

/// SQL + params for "is this node live at `as_of`".
#[cfg(test)]
fn live_as_of_query(id: &str, as_of: Millis) -> (String, Vec<Value>) {
    let sql = format!(
        "SELECT 1 FROM nodes WHERE id = ?1 AND {live}",
        live = live_at("nodes", 2)
    );
    (sql, vec![id.into(), as_of.into()])
}

/// SQL + params for the `tx_from` of the node `id` if it is open, which is the
/// row a supersede of `id` can close.
fn open_node_query(id: &str) -> (&'static str, Vec<Value>) {
    (
        "SELECT tx_from FROM nodes WHERE id = ?1 AND tx_to = ?2",
        vec![id.into(), FOREVER.into()],
    )
}

/// A node's own collision key for the subject queries: entity pages collide
/// on `(subject, scope)` alone, everything else also keys on `producer`.
fn collision_producer(node: &NewNode) -> Option<&str> {
    if node.entity_page {
        None
    } else {
        Some(node.producer.as_str())
    }
}

/// Stated default from the scope-field-validation design (2026-08-27), not
/// measured from any production data (there is none yet): a starting
/// point, matching how `MAX_EPISODE_ITEMS`/`MAX_ATTRIBUTES_CHARS` are
/// treated in `liam-daemon`.
const MAX_SCOPE_CHARS: usize = 200;

/// Normalize and validate an optional `scope` (the retrieval-partition
/// string on `NewNode` and `Query`) before it reaches storage or a filter.
/// `None` passes through unchanged. Otherwise: trims whitespace, rejects
/// an empty-after-trim value, a value over `MAX_SCOPE_CHARS`, any
/// character outside ASCII alphanumeric/`-`/`_`/`/`, and a leading or
/// trailing `/` or an empty segment (`//`). `/` separates hierarchy
/// segments: reads (`lexical`/`fetch_candidates`/`vector_search`) match a
/// scope plus every descendant, while write-side collision detection
/// (`open_by_subject_query`) stays exact-match. Called at every
/// write entry point that accepts a `scope` (`insert`, `upsert_by`,
/// `supersede`, `ingest_episode`), from the read path (`query_core`), and
/// from `migrate::normalize_scope_column` to flag data written before this
/// existed, so a malformed value can never mean "matches nothing" on one
/// side and "a real partition" on the other.
pub fn validate_scope(scope: &Option<String>) -> Result<Option<String>> {
    let Some(raw) = scope else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(Error::InvalidScope("scope must not be empty".to_string()));
    }
    if trimmed.chars().count() > MAX_SCOPE_CHARS {
        return Err(Error::InvalidScope(format!(
            "scope exceeds {MAX_SCOPE_CHARS} characters"
        )));
    }
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/'))
    {
        return Err(Error::InvalidScope(
            "scope must contain only ASCII letters, digits, '-', '_', '/'".to_string(),
        ));
    }
    if trimmed.starts_with('/') || trimmed.ends_with('/') || trimmed.contains("//") {
        return Err(Error::InvalidScope(
            "scope must not start or end with '/', or contain an empty segment ('//')".to_string(),
        ));
    }
    Ok(Some(trimmed.to_string()))
}

fn opt_text(v: Option<String>) -> Value {
    v.map(Value::Text).unwrap_or(Value::Null)
}

/// The row a `NewNode` becomes once ids and times are minted and its scope and
/// attributes are in stored form. The log hashes and stores this, never the
/// raw request.
fn resolve_node_row(id: &NodeId, node: &NewNode, now: Millis) -> Result<NodeRow> {
    Ok(NodeRow {
        id: id.as_str().to_string(),
        kind: node.kind.clone(),
        label: node.label.clone(),
        content: node.content.clone(),
        producer: node.producer.clone(),
        attributes: serde_json::to_string(&node.attributes)?,
        scope: node.scope.clone(),
        subject: node.subject.clone(),
        confidence: node.confidence,
        valid_from: node.valid_from.unwrap_or(now).0,
        valid_from_supplied: node.valid_from.is_some(),
        valid_until: FOREVER.0,
        tx_from: now.0,
        tx_to: FOREVER.0,
        derivation: Some(node.derivation.as_str().to_string()),
        basis: node
            .basis
            .iter()
            .map(|event| BasisRef {
                event_id: event.event_id.clone(),
                synthetic: event.synthetic,
            })
            .collect(),
    })
}

/// The statement that stores `row`. A row from an event written before
/// derivations were logged is stored `recorded`, and a replay then works out
/// which of those were pages. A derivation that is none of the three is an
/// error, so a bad value never reaches a column that reads back as a type.
fn node_row_insert(row: &NodeRow) -> Result<(String, Vec<Value>)> {
    let derivation = match &row.derivation {
        Some(text) => Derivation::parse(text)?,
        None => Derivation::Recorded,
    };
    let sql = "INSERT INTO nodes
         (id, kind, label, content, producer, attributes, scope, subject, confidence,
          valid_from, valid_until, tx_from, tx_to, derivation)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)"
        .to_string();
    let params = vec![
        row.id.clone().into(),
        row.kind.clone().into(),
        row.label.clone().into(),
        row.content.clone().into(),
        row.producer.clone().into(),
        row.attributes.clone().into(),
        opt_text(row.scope.clone()),
        opt_text(row.subject.clone()),
        Value::Real(row.confidence),
        row.valid_from.into(),
        row.valid_until.into(),
        row.tx_from.into(),
        row.tx_to.into(),
        derivation.as_str().into(),
    ];
    Ok((sql, params))
}

fn decay_factor(valid_from: Millis, now: Millis, half_life: Option<Millis>) -> f64 {
    match half_life {
        Some(h) if h.0 > 0 => {
            let age = (now.0 - valid_from.0).max(0) as f64;
            0.5f64.powf(age / h.0 as f64)
        }
        _ => 1.0,
    }
}

/// A raw node row as stored: unlike `Hit`/`ExplainedHit`, carries no
/// retrieval score, so it fits a direct by-id lookup as well as scoring.
#[non_exhaustive]
pub struct Candidate {
    /// Insertion order of the row, the stable tie-break between equal scores.
    /// Ids carry a random ULID suffix, so they order differently per store.
    pub(crate) rowid: i64,
    pub id: NodeId,
    pub kind: String,
    pub label: String,
    pub retrieved: Retrieved,
    pub scope: Option<String>,
    pub attributes: serde_json::Value,
    pub confidence: f64,
    pub valid_from: Millis,
    pub producer: String,
}

impl Candidate {
    /// A recorded row with the given identity and text, no scope or attributes,
    /// full confidence, valid from the epoch; set the public fields to refine it.
    pub fn new(
        id: NodeId,
        kind: impl Into<String>,
        label: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            rowid: 0,
            id,
            kind: kind.into(),
            label: label.into(),
            retrieved: Retrieved::Recorded(RecordedHit {
                content: RecordedText::new(content),
                events: Vec::new(),
                derivation: Derivation::Recorded,
            }),
            scope: None,
            attributes: serde_json::Value::Null,
            confidence: 1.0,
            valid_from: Millis(0),
            producer: "unknown".to_string(),
        }
    }
}

/// The one ranking order: score descending (a total order, so a NaN score
/// cannot make the sort inconsistent), then insertion order. Ranking ties
/// follow the rowid, so a store rebuilt from a natively written log (replayed
/// in write order) ranks ties as the original did; one rebuilt from a
/// backfill gets rowids in id order, so its ties can order differently.
fn rank_order(a_score: f64, a_rowid: i64, b_score: f64, b_rowid: i64) -> std::cmp::Ordering {
    b_score.total_cmp(&a_score).then(a_rowid.cmp(&b_rowid))
}

/// Turns the lexical, vector and cold rank lists into one fused RRF score per
/// candidate id, using `weight / (rrf_k + rank + 1)` summed across every list a
/// candidate appears in. The first two lists weigh 1; `cold_weight` scales the
/// third, so a store with no cold items scores exactly as it did without it.
fn rrf_fuse(
    lexical: &[NodeId],
    vector: &[NodeId],
    cold: &[NodeId],
    rrf_k: f64,
    cold_weight: f64,
) -> HashMap<String, f64> {
    let mut rrf: HashMap<String, f64> = HashMap::new();
    for (list, weight) in [(lexical, 1.0), (vector, 1.0), (cold, cold_weight)] {
        for (rank, id) in list.iter().enumerate() {
            *rrf.entry(id.as_str().to_string()).or_insert(0.0) +=
                weight / (rrf_k + rank as f64 + 1.0);
        }
    }
    rrf
}

/// The ids that seed graph expansion: the top `k` by fused score, ties broken
/// by `rowid` as in the final ranking. An id with no rowid (its row vanished
/// since the search) sorts last among its ties.
fn rank_seeds(rrf: &HashMap<String, f64>, rowids: &HashMap<String, i64>, k: usize) -> Vec<String> {
    let rowid = |id: &str| rowids.get(id).copied().unwrap_or(i64::MAX);
    let mut seeds: Vec<(&str, f64)> = rrf.iter().map(|(id, s)| (id.as_str(), *s)).collect();
    seeds.sort_by(|a, b| rank_order(a.1, rowid(a.0), b.1, rowid(b.0)));
    seeds.truncate(k);
    seeds.into_iter().map(|(id, _)| id.to_string()).collect()
}

/// The place each arm of a query gave an id, by id.
#[derive(Default)]
struct Ranks {
    lexical: HashMap<String, usize>,
    vector: HashMap<String, usize>,
    cold: HashMap<String, usize>,
}

impl Ranks {
    fn of(lexical: &[NodeId], vector: &[NodeId], cold: &[NodeId]) -> Self {
        let rank = |list: &[NodeId]| {
            list.iter()
                .enumerate()
                .map(|(i, id)| (id.as_str().to_string(), i))
                .collect()
        };
        Self {
            lexical: rank(lexical),
            vector: rank(vector),
            cold: rank(cold),
        }
    }
}

/// Scores hydrated candidate rows by RRF base score, confidence, time decay,
/// and expansion weighting, then returns the top `k` by score descending.
/// Pure and DB-free: every input is already-fetched data, so callers can
/// exercise it directly with hand-built rows, no `Graph` or backend needed.
#[allow(clippy::too_many_arguments)] // each argument is a distinct value, no natural grouping
fn score_and_rank(
    rows: Vec<Candidate>,
    candidates: &HashMap<String, (f64, bool)>,
    ranks: &Ranks,
    now: Millis,
    half_life: Option<Millis>,
    expansion_weight: f64,
    floor: f64,
    k: usize,
) -> Vec<ExplainedHit> {
    let mut out = Vec::with_capacity(rows.len());
    for c in rows {
        let (base, is_expanded) = candidates
            .get(c.id.as_str())
            .copied()
            .unwrap_or((floor, true));
        let decay = decay_factor(c.valid_from, now, half_life);
        let weight = if is_expanded { expansion_weight } else { 1.0 };
        let score = base * c.confidence * decay * weight;
        out.push((
            c.rowid,
            ExplainedHit {
                hit: Hit {
                    id: c.id.clone(),
                    kind: c.kind,
                    label: c.label,
                    retrieved: c.retrieved,
                    attributes: c.attributes,
                    score,
                },
                lexical_rank: ranks.lexical.get(c.id.as_str()).copied(),
                vector_rank: ranks.vector.get(c.id.as_str()).copied(),
                cold_rank: ranks.cold.get(c.id.as_str()).copied(),
                rrf: base,
                confidence: c.confidence,
                decay,
                valid_from: c.valid_from,
                expanded: is_expanded,
            },
        ));
    }
    out.sort_by(|(a_rowid, a), (b_rowid, b)| {
        rank_order(a.hit.score, *a_rowid, b.hit.score, *b_rowid)
    });
    out.truncate(k);
    out.into_iter().map(|(_, hit)| hit).collect()
}

pub struct Graph<B: Backend> {
    backend: B,
    clock: Arc<dyn Clock>,
    dims: usize,
    rrf_k: f64,
    expansion_weight: f64,
    log: Option<SharedLog>,
    embedder: Option<Arc<dyn ContentEmbedder>>,
    /// The binary signature of every live node that has one.
    quant: QuantIndex,
    cold_arm: ColdArm,
    /// Orders each update of `quant` after the commit it reports, and a reload
    /// of it after the tables it reads.
    quant_gate: tiers::IndexGate,
    /// Lets one tier move at a time read the lineage it moves and then move it.
    tier_gate: tokio::sync::Mutex<()>,
    /// How often the dedup filter was wrong, for the metrics.
    bloom_counters: liam_log::dedup::BloomCounters,
}

impl<B: Backend> Graph<B> {
    pub async fn open(path: &str, config: GraphConfig) -> Result<Self> {
        Self::open_with_clock(path, config, Arc::new(SystemClock)).await
    }

    pub async fn open_with_clock(
        path: &str,
        config: GraphConfig,
        clock: Arc<dyn Clock>,
    ) -> Result<Self> {
        // A crash during a restore leaves the database without its WAL; put
        // it back before the file is opened.
        snapshot::recover_interrupted_restore(std::path::Path::new(path))?;
        let backend = B::open(path, config.read_pool_size).await?;
        let mut ddl = schema(&config);
        ddl.push_str(&backend.vector_ddl(config.embedding_dims));
        backend.execute_batch(&ddl).await?;
        if config.secure_delete {
            backend.set_secure_delete(true).await?;
        }
        // `schema()` above is entirely `CREATE TABLE IF NOT EXISTS`, so a
        // database that already existed before `producer` was added never ran
        // it and does not have the column. This guarded ALTER TABLE is what
        // gives it one; a fresh database already has it from the DDL and this
        // call then no-ops. Both paths end in the same shape.
        crate::migrate::add_column_if_missing(
            &backend,
            "nodes",
            "producer",
            "TEXT NOT NULL DEFAULT 'unknown'",
        )
        .await?;
        // Same shape of fix for `derivation`; the backfill then marks the
        // entity pages that were written before the column existed.
        crate::migrate::add_column_if_missing(
            &backend,
            "nodes",
            "derivation",
            "TEXT NOT NULL DEFAULT 'recorded'",
        )
        .await?;
        crate::migrate::backfill_derivation(&backend, clock.now()).await?;
        // Gives an existing database the ON DELETE CASCADE that the DDL above
        // only grants a fresh one, by rebuilding the referencing tables
        // (ADR-0003). Detects and no-ops on a database that already has it.
        crate::migrate::ensure_cascade(&backend).await?;
        crate::migrate::ensure_usage_key(&backend).await?;
        crate::migrate::add_backfill_node_rowid_cursor(&backend).await?;
        // Trims whitespace from any scope value stored before `validate_scope`
        // existed, so it still matches a scope-filtered query under the new
        // trim-then-compare rules. Logs (does not rewrite) anything still
        // invalid after the trim.
        crate::migrate::normalize_scope_column(&backend).await?;
        let graph = Self {
            backend,
            clock,
            dims: config.embedding_dims,
            rrf_k: config.rrf_k,
            expansion_weight: config.expansion_weight,
            log: None,
            embedder: None,
            quant: QuantIndex::default(),
            cold_arm: ColdArm::of(&config),
            quant_gate: tiers::IndexGate::default(),
            tier_gate: tokio::sync::Mutex::new(()),
            bloom_counters: liam_log::dedup::BloomCounters::default(),
        };
        // Lineage first: it is what the index reload joins the tiers through.
        graph.backfill_lineage().await?;
        graph.reload_quant_index().await?;
        Ok(graph)
    }

    /// Routes `insert`, `upsert_by`, `supersede`, `relate`, `link`,
    /// `ingest_episode`, and `gc` through `log`: each is appended to it before it
    /// is applied, a sweep as chunked tombstones. Without a log, every write is
    /// unlogged.
    ///
    /// Refuses a log the store does not belong to (`LogIdMismatch`) and a store
    /// ahead of the log (`CursorBeyondLog`), then rebuilds the log's dedup
    /// filter from `log_hash_index` so a hash written before this process
    /// started is deduplicated.
    ///
    /// A store that holds rows and has not completed its backfill refuses every
    /// logged write with `BackfillRequired`, until `backfill_log_from_projection`
    /// completes.
    pub async fn with_log(mut self, log: SharedLog) -> Result<Self> {
        log_open::check_and_prime(&self.backend, &log).await?;
        self.log = Some(log);
        Ok(self)
    }

    /// The id of the log this store has been attached to, `None` for a store
    /// that never was. It lets a caller refuse a log directory that cannot be
    /// the store's before creating anything in it.
    pub async fn stored_log_id(&self) -> Result<Option<uuid::Uuid>> {
        Ok(log_cursor::read(&self.backend)
            .await?
            .map(|cursor| cursor.log_id))
    }

    /// The last log record the store has accounted for, applied or voided, or
    /// `None` before the first. A scan up to here sees only records the store
    /// has settled.
    pub async fn last_applied_offset(&self) -> Result<Option<LogOffset>> {
        Ok(log_cursor::read(&self.backend)
            .await?
            .and_then(|cursor| cursor.last))
    }

    // ---- write ----

    pub async fn insert(&self, mut node: NewNode) -> Result<NodeId> {
        node.scope = validate_scope(&node.scope)?;
        let id = self.project_insert(&NodeId::new(), &node).await?;
        self.store_vector(&id, &node).await?;
        Ok(id)
    }

    /// The vector write takes the write lock a transaction holds, so it follows
    /// the commit. It is idempotent, which lets a retry repair a vector an
    /// earlier attempt committed the node without.
    async fn store_vector(&self, id: &NodeId, node: &NewNode) -> Result<()> {
        if let Some(embedding) = node.embedding.as_deref() {
            self.put_vector(id, embedding).await?;
        }
        Ok(())
    }

    /// The signature is stored first, so a vector never exists without one even
    /// when the node was written by an earlier call, as a duplicate's is.
    async fn put_vector(&self, id: &NodeId, embedding: &[f32]) -> Result<()> {
        self.check_dims(embedding)?;
        self.ensure_quant(id, embedding).await?;
        self.backend.vector_upsert(id.as_str(), embedding).await
    }

    /// Insert, or supersede a competing live node with the same collision
    /// key: `(subject, scope, producer)` for a fact, `(subject, scope)` alone
    /// for an entity page. This makes contradiction handling automatic: the
    /// caller sets a subject, the library closes the prior version and links
    /// to it.
    pub async fn upsert_by(&self, mut node: NewNode) -> Result<NodeId> {
        node.scope = validate_scope(&node.scope)?;
        let Some(collision) = Collision::of(&node) else {
            return self.insert(node).await;
        };
        // The competitor is looked up inside the write's transaction, so
        // concurrent upserts of one subject form a chain.
        let id = self.project_upsert(&node, collision).await?;
        self.store_vector(&id, &node).await?;
        Ok(id)
    }

    /// Close the old node in transaction time, insert the new one, link them
    /// with a reserved `supersedes` edge.
    pub async fn supersede(&self, old: &NodeId, mut node: NewNode) -> Result<NodeId> {
        node.scope = validate_scope(&node.scope)?;
        let new_id = self.project_supersede(old, &node).await?;
        self.store_vector(&new_id, &node).await?;
        Ok(new_id)
    }

    /// Writes an edge with attributes through the same guarded, logged path as
    /// `relate`, so it is refused, deduplicated and replayed the same way. The
    /// reserved `supersedes` relation is always refused, and so is a live twin
    /// that carries other attributes, which a duplicate answer would drop.
    pub async fn link(&self, edge: NewEdge) -> Result<EdgeId> {
        refuse_supersedes([edge.kind.as_str()])?;
        let attributes = serde_json::to_string(&edge.attributes)?;
        let written = self
            .project_edge(&edge.src, &edge.dst, &edge.kind, &attributes)
            .await?;
        Ok(written.0)
    }

    /// Assert a semantic edge between two live nodes, idempotently.
    ///
    /// Liveness and idempotency are conditions ON the insert rather than checks
    /// before it, which is what makes this safe without a new backend
    /// capability (ADR-0001). Checking first and inserting after would let a
    /// concurrent `supersede` close an endpoint in the gap, and moving both
    /// into `execute_atomic` cannot help: that method takes a statement list
    /// built before the call and runs each with `execute`, never `query`
    /// (`backend.rs:53`, `backends/libsql.rs:228`), so nothing inside the
    /// transaction can read a row and branch on it. A single statement is
    /// atomic by itself, so the guards ride along in its `WHERE`.
    ///
    /// The idempotency guard keys on the ORDERED triple, so `relate(a, b, t)`
    /// and `relate(b, a, t)` both insert. That is correct for a directed edge
    /// and deliberately does not stop the pair reaching weight 2.0 in
    /// clustering; that guard lives on the read side, in
    /// `recompute_communities` (ADR-0001 Amendment 1, ADR-0002 Amendment 2).
    ///
    /// Endpoints are tested with `tx_to = FOREVER` alone, not the four-column
    /// `live_at` predicate this file uses everywhere else. Deliberate, not an
    /// oversight: see ADR-0001 Amendment 2 before "fixing" it, because widening
    /// it changes the guarantee rather than tightening it.
    pub async fn relate(&self, src: &NodeId, dst: &NodeId, kind: &str) -> Result<EdgeId> {
        Ok(self.project_edge(src, dst, kind, EMPTY_ATTRIBUTES).await?.0)
    }

    /// `relate`, with the log record stamped by `source` instead of the store,
    /// so a replay can tell who wrote the edge.
    pub async fn relate_from(
        &self,
        src: &NodeId,
        dst: &NodeId,
        kind: &str,
        source: &str,
    ) -> Result<EdgeId> {
        let written = self
            .project_edge_from(src, dst, kind, EMPTY_ATTRIBUTES, source)
            .await?;
        Ok(written.0)
    }

    /// Write a batch of nodes and the edges between them as one atomic unit:
    /// a producer submitting several facts, the entities they mention, and
    /// the edges linking them all lands together or not at all. Edges
    /// reference their endpoints by `EpisodeRef`, either a fresh node's
    /// index into `nodes` or an already-existing id, so a caller never needs
    /// to read back a freshly-inserted node's id before linking it.
    ///
    /// A node carrying a `subject` supersedes a same-subject competitor,
    /// whether it went live before this call or earlier in this same
    /// episode: ordinary check-then-write per node, inside the same loop, in
    /// order.
    pub async fn ingest_episode(
        &self,
        mut nodes: Vec<NewNode>,
        edges: Vec<EpisodeEdge>,
    ) -> Result<EpisodeResult> {
        // A bad `EpisodeRef::New` index must fail before any transaction
        // opens, not mid-write.
        for edge in &edges {
            for r in [&edge.from, &edge.to] {
                if let EpisodeRef::New(i) = r {
                    if *i >= nodes.len() {
                        return Err(Error::InvalidReference(format!(
                            "new node index {i} is out of bounds: this episode has {} nodes",
                            nodes.len()
                        )));
                    }
                }
            }
        }

        for node in &mut nodes {
            node.scope = validate_scope(&node.scope)?;
        }

        // Pre-generate every node's id (a ULID, no DB round-trip) so edges
        // can resolve `EpisodeRef::New(i)` to a real id before any row lands.
        let ids: Vec<NodeId> = nodes.iter().map(|_| NodeId::new()).collect();
        let edge_ids = self.project_episode(&ids, &nodes, &edges).await?;

        // Vector writes stay outside the transaction, exactly where `insert`
        // and `supersede` already put them: `BackendTx` never grows a
        // `vector_upsert` method, so nothing about "having a transaction"
        // moves this call inside one.
        //
        // Unlike `insert`/`supersede`'s single-node `?` shape, a failure here
        // does not abort the loop or fail the call. By this point the
        // transaction has already committed, so the meaningful
        // success/failure boundary for `ingest_episode` is behind us: the row
        // data is correct and durable, the vector just won't surface in
        // search until it's retried or backfilled. Aborting on the first
        // failure would silently orphan every embedding after it; returning
        // `Err` here would tell a caller the call failed when the durable
        // write actually succeeded, inviting a retry that mints duplicate
        // nodes for episodes with no `subject` to dedup against. A dims
        // mismatch is handled the same way as a `vector_upsert` failure: it's
        // a real config bug, not transient I/O, but nodes in one call
        // typically share an embedder config, so continuing just logs the
        // same problem a few more times rather than compounding any new
        // risk.
        for (id, node) in ids.iter().zip(&nodes) {
            if let Some(embedding) = node.embedding.as_deref() {
                if let Err(e) = self.put_vector(id, embedding).await {
                    tracing::error!(?id, error = %e, "episode vector write failed, node row committed without a vector");
                }
            }
        }

        Ok(EpisodeResult {
            node_ids: ids,
            edge_ids,
        })
    }

    /// Resolve a client-supplied handle to a full node id. A handle is any
    /// prefix of a ULID, including the whole 26 characters; `recall` renders 13
    /// (ADR-0001 Amendment 3).
    ///
    /// Two matches is an error, never a pick. That is the property that makes
    /// prefixes acceptable where ADR-0001 rejected addressing by label: a label
    /// collision is invisible and silently writes a plausible wrong edge, while
    /// a prefix collision is visible right here. The error carries the
    /// candidates in full because the caller was only ever shown 13 characters
    /// and cannot lengthen the handle on its own.
    pub async fn resolve_handle(&self, handle: &str) -> Result<NodeId> {
        // Errors quote what the caller actually sent. Echoing the normalised
        // form back shows a model a string it never wrote, which invites it to
        // retry against the transformation instead of against its own input.
        let sent = handle.trim().to_string();
        // Crockford base32 is case-insensitive by definition, GLOB is not.
        let handle = sent.to_ascii_uppercase();
        // `*`, `?` and `[` are GLOB wildcards, so an unfiltered handle could
        // match far more than its prefix. With a single live node in the store
        // a bare `*` would even resolve successfully, to an arbitrary node.
        // Restricting to alphanumerics excludes every metacharacter; it is
        // wider than the real ULID alphabet, which costs nothing because a
        // non-ULID character simply matches no row.
        if handle.is_empty() || !handle.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(Error::HandleNotFound(sent));
        }
        // GLOB, not LIKE. SQLite's LIKE is case-insensitive for ASCII by
        // default, so it cannot use the unique index on `nodes(id)`: measured
        // with EXPLAIN QUERY PLAN, LIKE plans as SCAN and GLOB as SEARCH.
        let rows = self
            .backend
            .query(
                "SELECT id FROM nodes WHERE id GLOB ?1 AND tx_to = ?2 ORDER BY id LIMIT ?3",
                &[
                    format!("{handle}*").into(),
                    FOREVER.into(),
                    (HANDLE_MATCH_LIMIT as i64).into(),
                ],
            )
            .await?;
        match rows.len() {
            0 => Err(Error::HandleNotFound(sent)),
            1 => Ok(NodeId::from_raw(rows[0].get_string(0)?)),
            // The list is capped, so the message names no count: a truncated
            // list with a claimed total would be a lie the caller cannot check.
            _ => Err(Error::AmbiguousHandle {
                handle: sent,
                candidates: rows
                    .iter()
                    .filter_map(|r| r.get_string(0).ok())
                    .collect::<Vec<_>>(),
            }),
        }
    }

    // ---- read ----

    pub async fn query(&self, q: &Query) -> Result<Vec<Hit>> {
        Ok(self
            .query_core(q)
            .await?
            .0
            .into_iter()
            .map(|e| e.hit)
            .collect())
    }

    /// Like `query`, but each result carries the score components that produced
    /// it: lexical rank, vector rank, fused RRF, confidence, decay, expansion.
    pub async fn query_explained(&self, q: &Query) -> Result<Vec<ExplainedHit>> {
        Ok(self.query_core(q).await?.0)
    }

    /// Like `query_explained`, with how long each retrieval arm took.
    pub async fn query_timed(&self, q: &Query) -> Result<(Vec<ExplainedHit>, QueryTimings)> {
        self.query_core(q).await
    }

    /// Fetch one node by id, live at `as_of`. `None` if it doesn't exist or
    /// isn't live at that instant.
    pub async fn get(&self, id: &NodeId, as_of: Millis) -> Result<Option<Candidate>> {
        Ok(self
            .fetch_candidates(std::slice::from_ref(id), as_of, None, None)
            .await?
            .into_iter()
            .next())
    }

    /// The vector stored for `id`, or `None` when the node has none. A closed
    /// node keeps its vector.
    pub async fn embedding(&self, id: &NodeId) -> Result<Option<Vec<f32>>> {
        self.backend.vector_get(id.as_str(), self.dims).await
    }

    /// Nodes `entity_id` mentions, live at `as_of`, most-recently mentioned
    /// first (`fetch_candidates`'s own order is not meaningful, restored after).
    pub async fn mentions(
        &self,
        entity_id: &NodeId,
        as_of: Millis,
        limit: usize,
    ) -> Result<Vec<Candidate>> {
        // Over-fetch, matching `query_core`'s pool: the edge predicate alone
        // doesn't know a `dst` node was later superseded, so hydration below
        // can drop candidates and must not do so out of a bare `limit`-sized
        // pool.
        let pool = limit.max(1) * 3;
        let rows = self
            .backend
            .query(
                "SELECT dst FROM edges WHERE src = ?1 AND type = ?2
                 AND tx_from <= ?3 AND tx_to > ?3
                 ORDER BY tx_from DESC, id DESC LIMIT ?4",
                &[
                    entity_id.as_str().into(),
                    crate::types::relation::MENTIONS.into(),
                    as_of.into(),
                    (pool as i64).into(),
                ],
            )
            .await?;
        let ordered_ids = ids_from(&rows)?;
        let candidates = self
            .fetch_candidates(&ordered_ids, as_of, None, None)
            .await?;
        let mut reordered = reorder_by_recency(candidates, &ordered_ids);
        reordered.truncate(limit);
        Ok(reordered)
    }

    async fn query_core(&self, q: &Query) -> Result<(Vec<ExplainedHit>, QueryTimings)> {
        let now = q.as_of.unwrap_or_else(|| self.clock.now());
        let pool = q.k.max(1) * 3;
        let scope = validate_scope(&q.scope)?;
        let scope = scope.as_deref();
        let kind = q.kind.as_deref();

        let mut timings = QueryTimings::default();
        let started = Instant::now();
        let lexical = match q.text.as_deref() {
            Some(text) => {
                let filter = Filter {
                    kind,
                    scope,
                    inferred: q.inferred,
                };
                self.lexical(text, filter, now, pool).await?
            }
            None => Vec::new(),
        };
        timings.lexical = q.text.is_some().then(|| started.elapsed());
        let started = Instant::now();
        let vector = match q.embedding.as_deref() {
            Some(embedding) => {
                self.backend
                    .vector_search(embedding, pool, kind, scope, now, q.inferred)
                    .await?
            }
            None => Vec::new(),
        };
        timings.vector = q.embedding.is_some().then(|| started.elapsed());

        let started = Instant::now();
        let cold = match q.embedding.as_deref() {
            Some(embedding) => {
                let arm = ColdQuery {
                    pool,
                    kind,
                    scope,
                    as_of: now,
                    inferred: q.inferred,
                };
                self.cold_probe(embedding, arm).await?
            }
            None => Vec::new(),
        };
        timings.cold = q.embedding.is_some().then(|| started.elapsed());

        let ranks = Ranks::of(&lexical, &vector, &cold);
        let rrf = rrf_fuse(&lexical, &vector, &cold, self.rrf_k, self.cold_arm.weight);

        // Seeds: top of the fused list. Expand their neighbours, down-weighted.
        let rowids = self.rowids_of(rrf.keys().map(String::as_str)).await?;
        let seeds = rank_seeds(&rrf, &rowids, q.k);

        let floor = 1.0 / (self.rrf_k + pool as f64 + 1.0);
        let mut candidates: HashMap<String, (f64, bool)> = HashMap::new();
        for (id, s) in &rrf {
            candidates.insert(id.clone(), (*s, false));
        }
        let mut expanded: HashSet<String> = HashSet::new();
        let seed_ids: Vec<NodeId> = seeds
            .iter()
            .map(|id| NodeId::from_raw(id.clone()))
            .collect();
        for seed_id in &seed_ids {
            for n in self.neighbors_where(seed_id, now, q.inferred).await? {
                expanded.insert(n.as_str().to_string());
            }
        }
        let same_as_ids: HashSet<String> = self
            .same_as_closure(&seed_ids, now, q.inferred)
            .await?
            .into_iter()
            .map(|n| n.as_str().to_string())
            .collect();
        for id in expanded {
            candidates.entry(id).or_insert((floor, true));
        }

        let ids: Vec<NodeId> = candidates
            .keys()
            .map(|s| NodeId::from_raw(s.clone()))
            .collect();
        let mut rows = self.fetch_candidates(&ids, now, scope, kind).await?;

        // same_as bridges are exempt from the query's own scope filter: they
        // link across scope domains by definition, so any id not already
        // hydrated in-scope by the call above needs a second, unscoped fetch.
        let hydrated: HashSet<&str> = rows.iter().map(|c| c.id.as_str()).collect();
        let remainder: Vec<NodeId> = same_as_ids
            .into_iter()
            .filter(|id| !hydrated.contains(id.as_str()))
            .map(NodeId::from_raw)
            .collect();
        if !remainder.is_empty() {
            let bridged = self.fetch_candidates(&remainder, now, None, kind).await?;
            for c in &bridged {
                candidates
                    .entry(c.id.as_str().to_string())
                    .or_insert((floor, true));
            }
            rows.extend(bridged);
        }
        // Backstop for the filters above: an inferred row must not take a ranked
        // slot, so it goes before the cut to `k`.
        match q.inferred {
            InferredRows::Exclude => rows.retain(|c| !c.retrieved.is_inferred()),
            InferredRows::Only => rows.retain(|c| c.retrieved.is_inferred()),
            InferredRows::Include => {}
        }

        let out = score_and_rank(
            rows,
            &candidates,
            &ranks,
            now,
            q.half_life,
            self.expansion_weight,
            floor,
            q.k,
        );
        Ok((out, timings))
    }

    pub async fn neighbors(&self, seed: &NodeId, as_of: Millis) -> Result<Vec<NodeId>> {
        self.neighbors_where(seed, as_of, InferredRows::Include)
            .await
    }

    /// `neighbors`, stepping only onto the derivations `inferred` allows.
    async fn neighbors_where(
        &self,
        seed: &NodeId,
        as_of: Millis,
        inferred: InferredRows,
    ) -> Result<Vec<NodeId>> {
        let (dst_ok, src_ok) = (not_inferred("dst", inferred), not_inferred("src", inferred));
        let rows = self
            .backend
            .query(
                &format!(
                    "SELECT dst FROM edges WHERE src = ?1 AND tx_from <= ?2 AND tx_to > ?2{dst_ok}
                     UNION SELECT src FROM edges WHERE dst = ?1 AND tx_from <= ?2 AND tx_to > ?2{src_ok}"
                ),
                &[seed.as_str().into(), as_of.into()],
            )
            .await?;
        ids_from(&rows)
    }

    /// Every node reachable from any of `seeds` by a chain of `same_as` edges
    /// of any length, in either direction, as of `as_of`. Always includes
    /// every seed. Runs one query per BFS layer across the whole frontier
    /// (a multi-source BFS), not one per seed, so a caller checking many
    /// seeds at once still pays for at most a handful of round trips rather
    /// than one per seed.
    ///
    /// Steps only onto the derivations `inferred` allows.
    pub(crate) async fn same_as_closure(
        &self,
        seeds: &[NodeId],
        as_of: Millis,
        inferred: InferredRows,
    ) -> Result<HashSet<NodeId>> {
        let (dst_ok, src_ok) = (not_inferred("dst", inferred), not_inferred("src", inferred));
        let mut visited: HashSet<NodeId> = seeds.iter().cloned().collect();
        let mut frontier: Vec<NodeId> = seeds.to_vec();
        while !frontier.is_empty() {
            let placeholders = (1..=frontier.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(",");
            let type_idx = frontier.len() + 1;
            let as_of_idx = frontier.len() + 2;
            let sql = format!(
                "SELECT dst FROM edges WHERE src IN ({placeholders}) AND type = ?{type_idx} AND tx_from <= ?{as_of_idx} AND tx_to > ?{as_of_idx}{dst_ok}
                 UNION SELECT src FROM edges WHERE dst IN ({placeholders}) AND type = ?{type_idx} AND tx_from <= ?{as_of_idx} AND tx_to > ?{as_of_idx}{src_ok}"
            );
            let mut params: Vec<Value> = frontier.iter().map(|id| id.as_str().into()).collect();
            params.push(crate::types::relation::SAME_AS.into());
            params.push(as_of.into());
            let rows = self.backend.query(&sql, &params).await?;
            let mut next_frontier = Vec::new();
            for id in ids_from(&rows)? {
                if visited.insert(id.clone()) {
                    next_frontier.push(id);
                }
            }
            frontier = next_frontier;
        }
        Ok(visited)
    }

    async fn lexical(
        &self,
        text: &str,
        filter: Filter<'_>,
        now: Millis,
        k: usize,
    ) -> Result<Vec<NodeId>> {
        let Filter {
            kind,
            scope,
            inferred,
        } = filter;
        // FTS5 parses the MATCH value as its own query expression, so a raw
        // question (`?`, apostrophes, bare operators) is a syntax error that
        // aborts the whole hybrid query. Sanitize to a term-quoted literal.
        let match_query = fts5_query(text);
        if !match_query.chars().any(|c| c.is_alphanumeric()) {
            // Nothing searchable (empty or all-punctuation); skip the lexical
            // leg so the vector leg, if present, still runs.
            return Ok(Vec::new());
        }
        let mut params: Vec<Value> = vec![match_query.into(), now.into(), (k as i64).into()];
        let mut filters = String::new();
        let mut next = 4;
        if let Some(kind) = kind {
            filters.push_str(&format!(" AND n.kind = ?{next}"));
            params.push(kind.into());
            next += 1;
        }
        if let Some(scope) = scope {
            filters.push_str(&format!(" AND {}", scope_within("n.scope", next)));
            params.push(scope.into());
        }
        filters.push_str(&row_derivation("n", inferred));
        let sql = format!(
            "SELECT n.id FROM nodes_fts
             JOIN nodes n ON n.rowid = nodes_fts.rowid
             WHERE nodes_fts MATCH ?1 AND {live}{filters}
             ORDER BY bm25(nodes_fts), n.rowid LIMIT ?3",
            live = live_at("n", 2),
        );
        let rows = self.backend.query(&sql, &params).await?;
        ids_from(&rows)
    }

    /// Insertion order of each id's row, in one query per chunk. An id with no
    /// row is absent from the result.
    async fn rowids_of<'a>(
        &self,
        ids: impl Iterator<Item = &'a str>,
    ) -> Result<HashMap<String, i64>> {
        const CHUNK: usize = 500;
        let ids: Vec<&str> = ids.collect();
        let mut out = HashMap::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            let placeholders = (1..=chunk.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(",");
            let params: Vec<Value> = chunk.iter().map(|id| (*id).into()).collect();
            let rows = self
                .backend
                .query(
                    &format!("SELECT id, rowid FROM nodes WHERE id IN ({placeholders})"),
                    &params,
                )
                .await?;
            for row in &rows {
                out.insert(row.get_string(0)?, row.get_i64(1)?);
            }
        }
        Ok(out)
    }

    /// Fetch the live rows for a set of candidate ids in one query per chunk
    /// (missing or expired ids simply don't come back). Chunked to stay under
    /// SQLite's 999 bound-parameter limit. Each chunk is in insertion order;
    /// across chunks the order follows `ids`, so callers needing a total order
    /// sort on `Candidate::rowid`.
    async fn fetch_candidates(
        &self,
        ids: &[NodeId],
        now: Millis,
        scope: Option<&str>,
        kind: Option<&str>,
    ) -> Result<Vec<Candidate>> {
        const CHUNK: usize = 512;
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            // `?1` is the live-at instant; `?2..` are the ids; the scope and
            // kind filters, if any, take the parameters after those.
            let mut params: Vec<Value> = vec![now.into()];
            let placeholders = chunk
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    params.push(id.as_str().into());
                    format!("?{}", i + 2)
                })
                .collect::<Vec<_>>()
                .join(",");
            // Both filters are applied HERE, not only on the retrieval paths:
            // graph expansion adds a seed's neighbours to the candidate set
            // without consulting the query at all, so a neighbour of another
            // kind or scope would otherwise ride along and break the filter the
            // caller asked for.
            let mut filters = String::new();
            if let Some(v) = scope {
                params.push(v.into());
                filters.push_str(&format!(" AND {}", scope_within("scope", params.len())));
            }
            if let Some(v) = kind {
                params.push(v.into());
                filters.push_str(&format!(" AND kind = ?{}", params.len()));
            }
            let sql = format!(
                "SELECT id, kind, label, content, scope, attributes, confidence, valid_from, rowid,
                        producer, derivation
                 FROM nodes WHERE id IN ({placeholders}) AND {live}{filters}
                 ORDER BY rowid",
                live = live_at("nodes", 1),
            );
            let rows = self.backend.query(&sql, &params).await?;
            let row_ids: Vec<String> = rows
                .iter()
                .map(|row| row.get_string(0))
                .collect::<Result<_>>()?;
            let mut listed = self
                .listed_events(&row_ids.iter().map(String::as_str).collect::<Vec<_>>())
                .await?;
            for row in &rows {
                let events = listed.remove(&row.get_string(0)?).unwrap_or_default();
                out.push(Candidate {
                    rowid: row.get_i64(8)?,
                    id: NodeId::from_raw(row.get_string(0)?),
                    kind: row.get_string(1)?,
                    label: row.get_string(2)?,
                    retrieved: retrieved(
                        Derivation::parse(&row.get_string(10)?)?,
                        row.get_string(3)?,
                        events,
                    ),
                    scope: opt_string(row, 4)?,
                    attributes: serde_json::from_str(&row.get_string(5)?)?,
                    confidence: row_f64(row, 6),
                    valid_from: Millis(row.get_i64(7)?),
                    producer: row.get_string(9)?,
                });
            }
        }
        Ok(out)
    }

    #[cfg(test)]
    async fn exists_as_of(&self, id: &NodeId, as_of: Millis) -> Result<bool> {
        let (sql, params) = live_as_of_query(id.as_str(), as_of);
        let rows = self.backend.query(&sql, &params).await?;
        Ok(!rows.is_empty())
    }

    fn check_dims(&self, embedding: &[f32]) -> Result<()> {
        if embedding.len() == self.dims {
            return Ok(());
        }
        Err(Error::Dimension {
            expected: self.dims,
            got: embedding.len(),
        })
    }

    // ---- change cursor ----

    /// Nodes recorded or closed strictly after `cursor`, for rebuilding only
    /// the vectors that changed.
    ///
    /// **This cursor cannot see an edge write, so it is useless for clustering.**
    /// It queries `nodes` only, and `Change` carries a node id, a timestamp and
    /// a closed flag. A `relate` inserts into `edges` and touches no node row,
    /// so this returns nothing for exactly the event that changes communities.
    /// An earlier version of this comment promised "recompute only changed
    /// communities", which the query has never supported.
    ///
    /// Clustering staleness is detected by the edge fingerprint in
    /// `cluster_state` instead (ADR-0002). Making this cursor able to answer the
    /// question would take the append-only change ledger that ADR-0002 weighs
    /// and defers.
    pub async fn changes_since(&self, cursor: Millis) -> Result<Vec<Change>> {
        let rows = self
            .backend
            .query(
                "SELECT id, tx_from, tx_to, rowid FROM nodes
                 WHERE tx_from > ?1 OR (tx_to > ?1 AND tx_to < ?2)
                 ORDER BY tx_from, rowid",
                &[cursor.into(), FOREVER.into()],
            )
            .await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let tx_to = row.get_i64(2)?;
            out.push(Change {
                id: NodeId::from_raw(row.get_string(0)?),
                tx_from: Millis(row.get_i64(1)?),
                rowid: row.get_i64(3)?,
                closed: tx_to != FOREVER.0 && tx_to > cursor.0,
            });
        }
        Ok(out)
    }
}

/// The live semantic edge set, as a `WHERE` clause shared by the fingerprint
/// and the graph read so the two can never drift apart on what they count.
const LIVE_SEMANTIC_EDGES: &str = "WHERE tx_to = ?1 AND type != ?2";

impl<B: Backend> Graph<B> {
    /// The cheap half of the staleness check: answer "did anything change"
    /// without loading the edge set.
    ///
    /// The recompute does NOT use this. It takes its fingerprint from the same
    /// statement that reads the edges, because two separate reads can be
    /// straddled by a `relate` and leave the stored fingerprint ahead of the
    /// graph it describes. See `edges_with_fingerprint`.
    pub async fn edge_fingerprint(&self) -> Result<Fingerprint> {
        let sql = format!("SELECT COUNT(*), MAX(tx_from) FROM edges {LIVE_SEMANTIC_EDGES}");
        let rows = self
            .backend
            .query(
                &sql,
                &[FOREVER.into(), crate::types::relation::SUPERSEDES.into()],
            )
            .await?;
        let Some(row) = rows.first() else {
            // An aggregate without GROUP BY always returns one row, so this is
            // unreachable rather than an empty-store case.
            return Err(Error::Backend("fingerprint query returned no row".into()));
        };
        Ok(Fingerprint {
            edge_count: row.get_i64(0)?,
            max_tx_from: max_tx_from_of(row, 1)?,
        })
    }

    /// The mentions fingerprint of one entity, keyed by `(subject, scope)`
    /// rather than `NodeId`, so a check made before a supersession still finds
    /// the entity after one.
    ///
    /// Resolves the live node for `(subject, scope)` as of `as_of` first, then
    /// aggregates `COUNT(*)`/`MAX(tx_from)` over that node's own live
    /// `mentions` edges, mirroring `edge_fingerprint`'s query shape at entity
    /// scope instead of store scope. A subject with no live node under that
    /// scope (wrong scope, or never observed) fingerprints as empty rather
    /// than an error: "nothing to be stale yet" is a valid state here.
    pub async fn entity_mentions_fingerprint(
        &self,
        subject: &str,
        scope: Option<&str>,
        as_of: Millis,
    ) -> Result<Fingerprint> {
        let (sql, params) = live_by_subject_query(subject, as_of, scope, None);
        let rows = self.backend.query(&sql, &params).await?;
        let Some(node_row) = rows.first() else {
            return Ok(Fingerprint {
                edge_count: 0,
                max_tx_from: Millis(0),
            });
        };
        let node_id = NodeId::from_raw(node_row.get_string(0)?);

        let rows = self
            .backend
            .query(
                "SELECT COUNT(*), MAX(tx_from) FROM edges
                 WHERE src = ?1 AND type = ?2 AND tx_from <= ?3 AND tx_to > ?3",
                &[
                    node_id.as_str().into(),
                    crate::types::relation::MENTIONS.into(),
                    as_of.into(),
                ],
            )
            .await?;
        let Some(row) = rows.first() else {
            // An aggregate without GROUP BY always returns one row, so this is
            // unreachable rather than an empty-edge-set case.
            return Err(Error::Backend("fingerprint query returned no row".into()));
        };
        Ok(Fingerprint {
            edge_count: row.get_i64(0)?,
            max_tx_from: max_tx_from_of(row, 1)?,
        })
    }

    /// The subjects of live entity pages, sorted, up to `limit`. An entity page
    /// is a live node that is the source of a live `mentions` edge.
    pub async fn live_entity_subjects(&self, limit: usize) -> Result<Vec<String>> {
        let rows = self
            .backend
            .query(
                "SELECT DISTINCT nodes.subject FROM nodes
                 JOIN edges ON edges.src = nodes.id
                 WHERE nodes.tx_to = ?1 AND edges.tx_to = ?1 AND edges.type = ?2
                   AND nodes.subject IS NOT NULL
                 ORDER BY nodes.subject LIMIT ?3",
                &[
                    FOREVER.into(),
                    crate::types::relation::MENTIONS.into(),
                    i64::try_from(limit).unwrap_or(i64::MAX).into(),
                ],
            )
            .await?;
        rows.iter().map(|row| row.get_string(0)).collect()
    }

    /// Record the mentions fingerprint an entity was just synthesized from,
    /// keyed by `(subject, scope)` so a later `stale_entities` check survives
    /// the entity's `NodeId` changing under a supersede.
    ///
    /// Delete-then-insert, like `cluster_state_writes`: `entity_mention_state`
    /// has a real `(subject, scope)` primary key, but the pair of statements
    /// keeps both tables' write paths in one idiom rather than introducing a
    /// second one for this table alone.
    pub async fn mark_entity_synthesized(
        &self,
        subject: &str,
        scope: Option<&str>,
        fingerprint: Fingerprint,
        as_of: Millis,
    ) -> Result<()> {
        let scope = scope.unwrap_or("");
        let statements = vec![
            (
                "DELETE FROM entity_mention_state WHERE subject = ?1 AND scope = ?2".to_string(),
                vec![subject.into(), scope.into()],
            ),
            (
                "INSERT INTO entity_mention_state
                   (subject, scope, mentions_count, mentions_max_tx_from, last_synthesized_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)"
                    .to_string(),
                vec![
                    subject.into(),
                    scope.into(),
                    fingerprint.edge_count.into(),
                    fingerprint.max_tx_from.into(),
                    as_of.into(),
                ],
            ),
        ];
        self.backend.execute_atomic(&statements).await?;
        Ok(())
    }

    /// Live `(NodeId, subject, scope)` triples due for resynthesis, oldest
    /// `last_synthesized_at` first, up to `limit`.
    ///
    /// Two candidate sources feed the same list: a stored
    /// `entity_mention_state` row whose live fingerprint has drifted from what
    /// it recorded, and a live entity with no stored row at all (never
    /// synthesized, sorted first via `Millis(0)`). A stored row whose subject
    /// no longer resolves to a live node is skipped rather than reported:
    /// there is nothing left to resynthesize.
    pub async fn stale_entities(
        &self,
        limit: usize,
        as_of: Millis,
    ) -> Result<Vec<(NodeId, String, Option<String>)>> {
        struct Candidate {
            id: NodeId,
            subject: String,
            scope: Option<String>,
            last_synthesized_at: Millis,
            rowid: i64,
        }

        let mut candidates = Vec::new();

        let state_rows = self
            .backend
            .query(
                "SELECT subject, scope, mentions_count, mentions_max_tx_from, last_synthesized_at
                 FROM entity_mention_state",
                &[],
            )
            .await?;
        for row in &state_rows {
            let subject = row.get_string(0)?;
            let raw_scope = row.get_string(1)?;
            let scope = if raw_scope.is_empty() {
                None
            } else {
                Some(raw_scope)
            };
            let stored_count = row.get_i64(2)?;
            let stored_max_tx_from = Millis(row.get_i64(3)?);
            let last_synthesized_at = Millis(row.get_i64(4)?);

            let (live_sql, live_params) =
                live_by_subject_query(&subject, as_of, scope.as_deref(), None);
            let live_rows = self.backend.query(&live_sql, &live_params).await?;
            let Some(live_row) = live_rows.first() else {
                continue;
            };
            let id = NodeId::from_raw(live_row.get_string(0)?);

            let live_fingerprint = self
                .entity_mentions_fingerprint(&subject, scope.as_deref(), as_of)
                .await?;
            if live_fingerprint.edge_count != stored_count
                || live_fingerprint.max_tx_from != stored_max_tx_from
            {
                let rowid = self
                    .backend
                    .query(
                        "SELECT rowid FROM nodes WHERE id = ?1",
                        &[id.as_str().into()],
                    )
                    .await?
                    .first()
                    .map(|r| r.get_i64(0))
                    .transpose()?
                    .unwrap_or(i64::MAX);
                candidates.push(Candidate {
                    id,
                    subject,
                    scope,
                    last_synthesized_at,
                    rowid,
                });
            }
        }

        let live = live_at("n", 1);
        let never_synthesized_rows = self
            .backend
            .query(
                &format!(
                    "SELECT n.id, n.subject, n.scope, n.rowid
                     FROM nodes n
                     LEFT JOIN entity_mention_state s
                       ON s.subject = n.subject AND s.scope = COALESCE(n.scope, '')
                     WHERE n.subject IS NOT NULL AND n.content = '' AND {live} AND s.subject IS NULL"
                ),
                &[as_of.into()],
            )
            .await?;
        for row in &never_synthesized_rows {
            candidates.push(Candidate {
                id: NodeId::from_raw(row.get_string(0)?),
                subject: row.get_string(1)?,
                scope: opt_string(row, 2)?,
                last_synthesized_at: Millis(0),
                rowid: row.get_i64(3)?,
            });
        }

        candidates.sort_by(|a, b| {
            a.last_synthesized_at
                .cmp(&b.last_synthesized_at)
                .then_with(|| a.rowid.cmp(&b.rowid))
        });
        Ok(candidates
            .into_iter()
            .take(limit)
            .map(|c| (c.id, c.subject, c.scope))
            .collect())
    }

    /// The live entity pages whose attributes lack the key `marker`, one per
    /// `(subject, scope)` and the newest of its rows, oldest first, up to
    /// `limit`. A page with no text is not listed: it was never written.
    pub async fn pages_without_marker(
        &self,
        marker: &str,
        limit: usize,
        as_of: Millis,
    ) -> Result<Vec<(NodeId, String, Option<String>)>> {
        let sql = format!(
            "SELECT n.id, n.subject, n.scope FROM nodes n
             WHERE n.derivation = '{inferred}' AND n.subject IS NOT NULL AND n.content != ''
               AND json_type(n.attributes, ?2) IS NULL AND {live}
               AND n.id = (SELECT m.id FROM nodes m
                           WHERE m.subject = n.subject AND m.scope IS n.scope
                             AND m.derivation = '{inferred}' AND {live_m}
                           ORDER BY m.tx_from DESC, m.id DESC LIMIT 1)
             ORDER BY n.rowid LIMIT ?3",
            inferred = Derivation::Inferred.as_str(),
            live = live_at("n", 1),
            live_m = live_at("m", 1),
        );
        let path = format!("$.{marker}");
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let rows = self
            .backend
            .query(&sql, &[as_of.into(), path.into(), limit.into()])
            .await?;
        rows.iter()
            .map(|row| {
                Ok((
                    NodeId::from_raw(row.get_string(0)?),
                    row.get_string(1)?,
                    opt_string(row, 2)?,
                ))
            })
            .collect()
    }

    /// Reattach mentions still pointing at a superseded node, and the
    /// superseded node's own outgoing mentions, onto the node that now
    /// stands in for it, walking a multi-hop chain to the terminal live node
    /// when more than one supersession landed before this ran, then advance
    /// a persisted watermark so a later run only scans supersessions this
    /// pass has not already seen. Returns the count of `mentions` edges
    /// actually inserted.
    ///
    /// The repair insert goes through `relate`: both the liveness guard and
    /// the duplicate-triple guard on `(src, dst, type)` have to hold, since
    /// two `mentions` edges seeded outside `relate` can otherwise race each
    /// other onto the same repaired triple within one pass. Each `relate` result is classified via `classify_relate_outcome`
    /// and folded per `SUPERSEDES` edge: a real failure on any one repair
    /// keeps that edge's `tx_from` from advancing the watermark, even when a
    /// sibling repair on the same edge succeeded, so a later run revisits
    /// exactly what this one could not finish.
    pub async fn repair_superseded_mentions(&self) -> Result<usize> {
        self.repair_superseded_mentions_from(STORE_SOURCE).await
    }

    /// `repair_superseded_mentions` with the log records stamped by `source`, so
    /// a rollback of the consolidation run named in it cancels the repair.
    pub async fn repair_superseded_mentions_from(&self, source: &str) -> Result<usize> {
        let watermark = self.repair_watermark().await?;
        let supersede_rows = self
            .backend
            .query(
                "SELECT dst, tx_from FROM edges WHERE type = ?1 AND tx_from > ?2
                 ORDER BY tx_from ASC",
                &[crate::types::relation::SUPERSEDES.into(), watermark.into()],
            )
            .await?;

        let mut inserted = 0usize;
        let mut processed: Vec<(Millis, EdgeOutcome)> = Vec::with_capacity(supersede_rows.len());

        for row in &supersede_rows {
            let original = NodeId::from_raw(row.get_string(0)?);
            let tx_from = Millis(row.get_i64(1)?);
            let terminal = self.terminal_supersession(&original).await?;

            let mention_rows = self
                .backend
                .query(
                    "SELECT src FROM edges WHERE dst = ?1 AND type = ?2 AND tx_to = ?3",
                    &[
                        original.as_str().into(),
                        crate::types::relation::MENTIONS.into(),
                        FOREVER.into(),
                    ],
                )
                .await?;

            let mut saw_failure = false;
            let mut saw_insert = false;
            for mention_row in &mention_rows {
                let mentioning = NodeId::from_raw(mention_row.get_string(0)?);
                let result = self
                    .project_edge_from(
                        &mentioning,
                        &terminal,
                        crate::types::relation::MENTIONS,
                        EMPTY_ATTRIBUTES,
                        source,
                    )
                    .await;
                fold_repair_outcome(
                    repair_outcome(result),
                    &mut inserted,
                    &mut saw_insert,
                    &mut saw_failure,
                );
            }

            // The original node's own outgoing mentions, not just mentions
            // pointing at it: `supersede` never moves a node's own edges, so
            // an entity superseded by resynthesis keeps these on the dead id
            // forever unless repair moves them to the terminal node too. A
            // no-op whenever `original` is a plain fact, since a `mentions`
            // edge's `src` is always the entity end, never the fact end, so
            // a fact has no live outgoing `mentions` edges to find here.
            let own_mention_rows = self
                .backend
                .query(
                    "SELECT dst FROM edges WHERE src = ?1 AND type = ?2 AND tx_to = ?3",
                    &[
                        original.as_str().into(),
                        crate::types::relation::MENTIONS.into(),
                        FOREVER.into(),
                    ],
                )
                .await?;
            for mention_row in &own_mention_rows {
                let mentioned = NodeId::from_raw(mention_row.get_string(0)?);
                let result = self
                    .project_edge_from(
                        &terminal,
                        &mentioned,
                        crate::types::relation::MENTIONS,
                        EMPTY_ATTRIBUTES,
                        source,
                    )
                    .await;
                fold_repair_outcome(
                    repair_outcome(result),
                    &mut inserted,
                    &mut saw_insert,
                    &mut saw_failure,
                );
            }

            // A `SUPERSEDES` edge with nothing to repair, or nothing but safe
            // no-ops, is fully handled: only an actual failure blocks the
            // watermark from moving past it.
            let outcome = if saw_failure {
                EdgeOutcome::RealFailure
            } else if saw_insert {
                EdgeOutcome::Inserted
            } else {
                EdgeOutcome::AlreadyPresent
            };
            processed.push((tx_from, outcome));
        }

        if let Some(new_watermark) = next_watermark(&processed) {
            // A `supersedes` edge clamped to its predecessor's start can be later
            // than the clock. Persisting that would skip every supersession
            // until the clock caught up, so such edges are rescanned instead,
            // which is safe because a repeated repair is a no-op.
            let new_watermark = new_watermark.min(self.clock.now());
            if new_watermark > watermark {
                self.persist_repair_watermark(new_watermark).await?;
            }
        }

        Ok(inserted)
    }

    /// The persisted repair watermark, or `Millis(0)` when repair has never
    /// run, so the first pass scans every `SUPERSEDES` edge in the store.
    async fn repair_watermark(&self) -> Result<Millis> {
        let rows = self
            .backend
            .query(
                "SELECT last_repaired_at FROM provenance_repair_state WHERE id = 1",
                &[],
            )
            .await?;
        match rows.first() {
            Some(row) => Ok(Millis(row.get_i64(0)?)),
            None => Ok(Millis(0)),
        }
    }

    /// Delete-then-insert into the single-row `provenance_repair_state`
    /// table, matching `mark_entity_synthesized`'s idiom for a table with no
    /// natural key to `UPDATE` against across backends.
    async fn persist_repair_watermark(&self, watermark: Millis) -> Result<()> {
        let statements = vec![
            (
                "DELETE FROM provenance_repair_state WHERE id = 1".to_string(),
                Vec::new(),
            ),
            (
                "INSERT INTO provenance_repair_state (id, last_repaired_at) VALUES (1, ?1)"
                    .to_string(),
                vec![watermark.into()],
            ),
        ];
        self.backend.execute_atomic(&statements).await?;
        Ok(())
    }

    /// The terminal live node reached by following `SUPERSEDES` edges forward
    /// from `original`: each step asks "who superseded the current node" (an
    /// edge with `dst` equal to it, since `supersede` inserts `new -> old`),
    /// moving to that edge's `src`, until no further supersession is found.
    /// A node superseded only once returns after a single step; a node never
    /// superseded again returns `original` itself unchanged.
    async fn terminal_supersession(&self, original: &NodeId) -> Result<NodeId> {
        let mut current = original.clone();
        loop {
            let rows = self
                .backend
                .query(
                    "SELECT src FROM edges WHERE dst = ?1 AND type = ?2 AND tx_to = ?3",
                    &[
                        current.as_str().into(),
                        crate::types::relation::SUPERSEDES.into(),
                        FOREVER.into(),
                    ],
                )
                .await?;
            match rows.first() {
                Some(row) => current = NodeId::from_raw(row.get_string(0)?),
                None => return Ok(current),
            }
        }
    }

    /// The stored run state, or `None` when this store has never clustered.
    ///
    /// `None` is NOT the same as a zero fingerprint and must never be defaulted
    /// into one. A database written by a build that predates `cluster_state`
    /// arrives with a populated `node_community` and no state row; if `None`
    /// read as `(0, 0)` it would match the live fingerprint of an edgeless
    /// store, and that stale assignment would be served forever.
    pub(crate) async fn read_cluster_state(&self) -> Result<Option<ClusterState>> {
        let rows = self
            .backend
            .query(
                "SELECT edge_count, max_tx_from, computed_at, last_cold_start_at
                 FROM cluster_state LIMIT 1",
                &[],
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        Ok(Some(ClusterState {
            fingerprint: Fingerprint {
                edge_count: row.get_i64(0)?,
                max_tx_from: Millis(row.get_i64(1)?),
            },
            computed_at: Millis(row.get_i64(2)?),
            last_cold_start_at: Millis(row.get_i64(3)?),
        }))
    }

    /// The statement pair that replaces the single `cluster_state` row, for
    /// appending to the same `execute_atomic` list as the assignment.
    ///
    /// The `DELETE` is load-bearing: `cluster_state` declares no PRIMARY KEY or
    /// UNIQUE, so nothing in the schema stops a second row accumulating.
    fn cluster_state_writes(state: &ClusterState) -> Vec<(String, Vec<Value>)> {
        vec![
            ("DELETE FROM cluster_state".to_string(), Vec::new()),
            (
                "INSERT INTO cluster_state
                   (edge_count, max_tx_from, computed_at, last_cold_start_at)
                 VALUES (?1, ?2, ?3, ?4)"
                    .to_string(),
                vec![
                    state.fingerprint.edge_count.into(),
                    state.fingerprint.max_tx_from.into(),
                    state.computed_at.into(),
                    state.last_cold_start_at.into(),
                ],
            ),
        ]
    }

    /// The live semantic edge set AND the fingerprint of that exact set, from
    /// one statement.
    ///
    /// One statement, not two, and that is the whole point. ADR-0002 mandates a
    /// separate `SELECT COUNT(*), MAX(tx_from)` and argues the guarantee as
    /// read time versus commit time, but reads never serialize with writes
    /// (`backend.rs`), so a `relate` can land between two reads. If the
    /// fingerprint were taken second it would describe an edge the graph never
    /// saw, the next staleness check would MATCH, and a stale assignment would
    /// be served over a live edge, which is the precise failure the record
    /// exists to prevent. Window functions collapse both into one snapshot, so
    /// there is no window to get the order wrong in. ADR-0002 Amendment 5.
    ///
    /// An empty edge set returns zero rows, which is the fingerprint `(0, 0)`,
    /// so the NULL that Amendment 3 handles cannot arise on this path at all.
    ///
    /// `ORDER BY id` fixes the row order, which `build_cluster_input` turns into
    /// the dense vertex numbering. Without it the numbering varies run to run on
    /// identical data, and so can Leiden's seeded partition.
    async fn edges_with_fingerprint(&self) -> Result<(Vec<Row>, Fingerprint)> {
        let sql = format!(
            "SELECT src, dst, type, COUNT(*) OVER (), MAX(tx_from) OVER ()
             FROM edges {LIVE_SEMANTIC_EDGES} ORDER BY id"
        );
        let rows = self
            .backend
            .query(
                &sql,
                &[FOREVER.into(), crate::types::relation::SUPERSEDES.into()],
            )
            .await?;
        let fingerprint = match rows.first() {
            Some(row) => Fingerprint {
                edge_count: row.get_i64(3)?,
                max_tx_from: max_tx_from_of(row, 4)?,
            },
            None => Fingerprint {
                edge_count: 0,
                max_tx_from: Millis(0),
            },
        };
        Ok((rows, fingerprint))
    }

    /// Recompute the assignment unconditionally and persist it with the
    /// fingerprint of the graph it was built from.
    ///
    /// Callers wanting "recompute only if stale" want `refresh_communities`
    /// (or `communities`, which wraps it), the seam both the GC tick and the
    /// read side share.
    pub async fn recompute_communities(&self) -> Result<usize> {
        self.recompute_with(self.read_cluster_state().await?).await
    }

    /// The stored state and the live fingerprint, from ONE statement, so a
    /// `relate` landing between two separate reads can't make a stale
    /// assignment look current (ADR-0002 Amendment 6, same fix as
    /// `edges_with_fingerprint` applies to the recompute path).
    ///
    /// Scalar subqueries, not a join: an edgeless store still returns one row,
    /// with `cluster_state` empty showing as four NULLs on it.
    async fn staleness_snapshot(&self) -> Result<(Option<ClusterState>, Fingerprint)> {
        let sql = format!(
            "SELECT COUNT(*), MAX(tx_from),
               (SELECT edge_count         FROM cluster_state LIMIT 1),
               (SELECT max_tx_from        FROM cluster_state LIMIT 1),
               (SELECT computed_at        FROM cluster_state LIMIT 1),
               (SELECT last_cold_start_at FROM cluster_state LIMIT 1)
             FROM edges {LIVE_SEMANTIC_EDGES}"
        );
        let rows = self
            .backend
            .query(
                &sql,
                &[FOREVER.into(), crate::types::relation::SUPERSEDES.into()],
            )
            .await?;
        let Some(row) = rows.first() else {
            // The aggregate has no GROUP BY, so it always returns one row.
            return Err(Error::Backend(
                "staleness snapshot query returned no row".into(),
            ));
        };
        let live = Fingerprint {
            edge_count: row.get_i64(0)?,
            max_tx_from: max_tx_from_of(row, 1)?,
        };
        // `cluster_state`'s columns are NOT NULL, so column 2 alone tells
        // whether the row exists. Defaulting a NULL here to (0, 0) would match
        // an edgeless store's real fingerprint and serve a stale assignment.
        let state = match row.0.get(2) {
            Some(Value::Null) | None => None,
            _ => Some(ClusterState {
                fingerprint: Fingerprint {
                    edge_count: row.get_i64(2)?,
                    max_tx_from: Millis(row.get_i64(3)?),
                },
                computed_at: Millis(row.get_i64(4)?),
                last_cold_start_at: Millis(row.get_i64(5)?),
            }),
        };
        Ok((state, live))
    }

    /// Recompute only if the stored assignment no longer matches the live edge
    /// set; report whether it did. The shared seam for the GC tick and
    /// `communities`, so "assignment matches fingerprint" lives in one place.
    pub async fn refresh_communities(&self) -> Result<bool> {
        let (state, live) = self.staleness_snapshot().await?;
        if state.as_ref().map(|s| s.fingerprint) == Some(live) {
            return Ok(false);
        }
        self.recompute_with(state).await?;
        Ok(true)
    }

    async fn recompute_with(&self, state: Option<ClusterState>) -> Result<usize> {
        use crate::cluster::detect;

        let started = std::time::Instant::now();
        let (rows, fingerprint) = self.edges_with_fingerprint().await?;
        let (labels, edges) = build_cluster_input(&rows)?;
        let now = self.clock.now();

        // Cold whenever history cannot be trusted to seed from. No prior run at
        // all, an empty stored assignment (seeding singletons and calling it
        // warm would only lie in the logs), or the 24-hour escape hatch.
        let stored = self.stored_communities().await?;
        let cold = match state {
            None => true,
            Some(state) => stored.is_empty() || cold_start_due(state.last_cold_start_at, now),
        };
        let seed = if cold {
            None
        } else {
            Some(build_seed(&labels, &stored))
        };
        let assignment = detect(labels.len(), &edges, seed);

        let mut statements: Vec<(String, Vec<Value>)> =
            vec![("DELETE FROM node_community".to_string(), Vec::new())];
        for (i, community) in assignment.iter().enumerate() {
            statements.push((
                "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, ?2, ?3)"
                    .to_string(),
                vec![
                    labels[i].as_str().into(),
                    (*community as i64).into(),
                    now.into(),
                ],
            ));
        }
        // The CAPTURED fingerprint, never a fresh query here. A `relate` landing
        // during the Leiden run above must leave the stored value BEHIND the
        // real one, so the next check mismatches and recomputes. Re-reading it
        // at commit would record that edge as included when the assignment
        // never saw it.
        //
        // `last_cold_start_at` is carried forward on a warm run. Writing `now`
        // would recreate the defect ADR-0002 Amendment 1 fixed, at the write
        // site instead of the read site, with the same symptom: the escape
        // hatch never fires again. A warm run always has a prior state to carry
        // from, because `state == None` forces cold above.
        statements.extend(Self::cluster_state_writes(&ClusterState {
            fingerprint,
            computed_at: now,
            last_cold_start_at: if cold {
                now
            } else {
                state.map_or(now, |s| s.last_cold_start_at)
            },
        }));
        self.backend.execute_atomic(&statements).await?;

        let count = assignment.iter().collect::<HashSet<_>>().len();
        // ADR-0002 requires both paths to report size and cost, so the ceiling
        // is found from real use rather than guessed at.
        tracing::info!(
            nodes = labels.len(),
            edges = edges.len(),
            communities = count,
            cold,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "communities recomputed"
        );
        Ok(count)
    }

    /// Community assignments, recomputing first if the stored one no longer
    /// matches the live edge set.
    ///
    /// **The integers are not durable.** Leiden renumbers by first appearance
    /// over a dense index that shifts whenever the edge set changes, so the
    /// number labelling a group can differ between calls even when the grouping
    /// does not. Read them as "these nodes are together in this response", never
    /// as a handle to store and compare later.
    ///
    /// A failed recompute surfaces its error rather than quietly returning a
    /// stale answer, since serving stale data is what the fingerprint exists to
    /// prevent. `execute_atomic` is all or nothing, so the previous assignment
    /// survives intact.
    pub async fn communities(&self) -> Result<Vec<(NodeId, i64)>> {
        self.refresh_communities().await?;
        self.stored_communities().await
    }

    /// Live nodes grouped by community, largest group first, with no
    /// community id on any member: it isn't durable across runs, so it must
    /// never look like something a client can store.
    ///
    /// Filters on `live_at`, matching what `recall` would show, because
    /// `node_community` can legitimately hold a superseded node: clustering
    /// filters edge liveness, not node liveness (ADR-0001).
    ///
    /// Inferred members are left out unless `include_inferred`; the stored
    /// assignment is not changed, so a group may come out smaller.
    pub async fn community_groups(
        &self,
        include_inferred: bool,
    ) -> Result<Vec<Vec<ClusterMember>>> {
        self.refresh_communities().await?;
        let now = self.clock.now();
        let sql = format!(
            "SELECT nc.community, n.id, n.kind, n.label
               FROM node_community nc JOIN nodes n ON n.id = nc.node_id
              WHERE {live}{strict}
              ORDER BY nc.community, n.id",
            live = live_at("n", 1),
            strict = row_derivation("n", include_inferred.into()),
        );
        let rows = self.backend.query(&sql, &[now.into()]).await?;

        let mut members = Vec::with_capacity(rows.len());
        for row in &rows {
            members.push((
                row.get_i64(0)?,
                ClusterMember {
                    id: NodeId::from_raw(row.get_string(1)?),
                    kind: row.get_string(2)?,
                    label: row.get_string(3)?,
                },
            ));
        }
        Ok(group_and_sort_communities(members))
    }

    /// The stored assignment, with no staleness check. `pub(crate)` so the only
    /// way out of the crate is the checked path above.
    pub(crate) async fn stored_communities(&self) -> Result<Vec<(NodeId, i64)>> {
        let rows = self
            .backend
            .query(
                "SELECT node_id, community FROM node_community ORDER BY community, node_id",
                &[],
            )
            .await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            out.push((NodeId::from_raw(row.get_string(0)?), row.get_i64(1)?));
        }
        Ok(out)
    }
}

/// What a lexical search narrows by.
#[derive(Clone, Copy)]
struct Filter<'a> {
    kind: Option<&'a str>,
    scope: Option<&'a str>,
    inferred: InferredRows,
}

/// A predicate on the derivation of the node named by `column`, an edge end:
/// none when `inferred` allows both kinds. The derivation is a store constant,
/// so it is written into the statement rather than bound.
fn not_inferred(column: &str, inferred: InferredRows) -> String {
    let membership = match inferred {
        InferredRows::Include => return String::new(),
        InferredRows::Exclude => "NOT IN",
        InferredRows::Only => "IN",
    };
    format!(
        " AND {column} {membership} (SELECT id FROM nodes WHERE derivation = '{}')",
        Derivation::Inferred.as_str()
    )
}

/// `not_inferred` for a query that already reads the `nodes` row as `alias`.
fn row_derivation(alias: &str, inferred: InferredRows) -> String {
    let comparison = match inferred {
        InferredRows::Include => return String::new(),
        InferredRows::Exclude => "!=",
        InferredRows::Only => "=",
    };
    format!(
        " AND {alias}.derivation {comparison} '{}'",
        Derivation::Inferred.as_str()
    )
}

/// A stored row's text as the type its derivation says it is. An inferred row
/// with no listed basis reads as `Unknown`, not as an empty list.
fn retrieved(derivation: Derivation, content: String, events: ListedEvents) -> Retrieved {
    match derivation {
        Derivation::Inferred => Retrieved::Inferred(InferredHit {
            text: InferredText::new(content),
            basis: match events.basis.is_empty() {
                true => Basis::Unknown,
                false => Basis::Known(events.basis),
            },
            derivation,
            summary_level: None,
        }),
        Derivation::Recorded | Derivation::Extracted => Retrieved::Recorded(RecordedHit {
            content: RecordedText::new(content),
            events: events.origin,
            derivation,
        }),
    }
}

fn ids_from(rows: &[Row]) -> Result<Vec<NodeId>> {
    rows.iter()
        .map(|r| Ok(NodeId::from_raw(r.get_string(0)?)))
        .collect()
}

/// Restores `candidates` to the order `ordered_ids` names, dropping any id
/// with no matching candidate.
fn reorder_by_recency(candidates: Vec<Candidate>, ordered_ids: &[NodeId]) -> Vec<Candidate> {
    let mut by_id: HashMap<NodeId, Candidate> =
        candidates.into_iter().map(|c| (c.id.clone(), c)).collect();
    ordered_ids
        .iter()
        .filter_map(|id| by_id.remove(id))
        .collect()
}

/// Groups members by community id, largest group first, tied groups broken by
/// the first member's id. A free function (not inline in `community_groups`)
/// so the tie-break is testable against fixed input, independent of what
/// numeric id Leiden happens to assign.
fn group_and_sort_communities(members: Vec<(i64, ClusterMember)>) -> Vec<Vec<ClusterMember>> {
    let mut by_community: BTreeMap<i64, Vec<ClusterMember>> = BTreeMap::new();
    for (community, member) in members {
        by_community.entry(community).or_default().push(member);
    }
    // No entry in `by_community` is ever empty: it's only created by
    // `.or_default().push(...)`, which adds a member in the same step.
    let mut groups: Vec<Vec<ClusterMember>> = by_community.into_values().collect();
    groups.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| a[0].id.as_str().cmp(b[0].id.as_str()))
    });
    groups
}

fn row_f64(row: &Row, i: usize) -> f64 {
    match row.0.get(i) {
        Some(Value::Real(v)) => *v,
        Some(Value::Int(v)) => *v as f64,
        _ => 1.0,
    }
}

/// Read a nullable TEXT column as `Option<String>`. `Row::get_string` errors
/// on NULL, exactly the case a nullable column like `scope` needs to allow.
fn opt_string(row: &Row, i: usize) -> Result<Option<String>> {
    match row.0.get(i) {
        Some(Value::Text(v)) => Ok(Some(v.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(other) => Err(Error::Backend(format!(
            "column {i} is neither text nor null: {other:?}"
        ))),
    }
}

/// Turn arbitrary user text into a safe FTS5 MATCH string. FTS5 parses the
/// *value* of a MATCH parameter as its own query language, so a natural-language
/// question (`?`, `'`, `:`, leading `-`, bare `AND`/`OR`/`NOT`, parentheses)
/// raises a syntax error and aborts the whole hybrid query. Wrapping each
/// whitespace token in double quotes (embedded `"` doubled) makes every operator
/// character match literally; the tokenizer still stems inside each quoted
/// phrase, and single-token quoting is equivalent to a bare term under the
/// tokenizer, so well-formed single-word queries are unchanged.
///
/// The quoted terms are joined with `OR`, not a space. A space is implicit AND
/// in FTS5, which would require every word of the query to appear in a
/// document. A stored memory is a statement; a question shares only its
/// content words with the statement that answers it and adds interrogatives
/// and auxiliaries ("when", "does") the statement never has, so AND-joining
/// made the lexical arm return zero hits for any natural-language question.
/// ORing recovers that recall: BM25 still ranks by inverse document frequency,
/// so a rare shared term dominates the score while "the" or "when" contribute
/// almost nothing, and RRF re-sorts the fused result anyway.
///
/// Tokens are lowercased and only the first occurrence of each is kept, so a
/// long repetitive query cannot expand into thousands of redundant OR terms.
/// The consolidation probes share this function, so they get the same dedupe.
fn fts5_query(text: &str) -> String {
    let mut seen = HashSet::new();
    text.split_whitespace()
        .map(str::to_lowercase)
        .filter(|t| seen.insert(t.clone()))
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Turn live edge rows of `(src, dst, type)` into the dense node labels and the
/// edge list Leiden runs on. Pure, and separate from `recompute_communities` so
/// the dedup below can be asserted directly. Going through the partition
/// instead would test it via a heuristic: a count of communities does not
/// discriminate between a deduped graph and a doubled one, since modularity can
/// land on the same number of groups either way.
///
/// Collapses each unordered pair to one edge. `relate` is idempotent on the
/// ORDERED triple, so a client asserting both directions writes two rows, which
/// is right for a directed edge table. `leiden-rs` then merges only consecutive
/// exact duplicates in `build_undirected_csr`, so `(a,b)` and `(b,a)` both
/// survive and the pair reaches weight 2.0 (ADR-0002 Amendment 2).
///
/// `type` stays in the key on purpose. `relate(a, b, "mentions")` and
/// `relate(a, b, "causes")` are two independent things a client said about one
/// pair, and neither collides with the other on the ordered triple, so both are
/// in the table deliberately. Keying on the pair alone would erase that and
/// weigh a pair carrying two relations like one carrying a single relation.
fn build_cluster_input(rows: &[Row]) -> Result<(Vec<String>, Vec<crate::cluster::Edge>)> {
    use crate::cluster::Edge;

    let mut index: HashMap<String, usize> = HashMap::new();
    let mut labels: Vec<String> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut seen: HashSet<(usize, usize, String)> = HashSet::new();
    for row in rows {
        let u = intern(&mut index, &mut labels, row.get_string(0)?);
        let v = intern(&mut index, &mut labels, row.get_string(1)?);
        if seen.insert((u.min(v), u.max(v), row.get_string(2)?)) {
            edges.push(Edge(u, v));
        }
    }
    Ok((labels, edges))
}

/// Read a `MAX(tx_from)` column, treating SQL NULL as 0.
///
/// `MAX()` over zero rows is NULL, which `Row::get_i64` rejects, so every store
/// would error until its first `relate` if this were a plain `get_i64`. Handled
/// in Rust rather than with `COALESCE` in the SQL, per ADR-0002 Amendment 3:
/// the null is real and the code should say so, instead of the query hiding it.
fn max_tx_from_of(row: &Row, i: usize) -> Result<Millis> {
    match row.0.get(i) {
        Some(Value::Int(v)) => Ok(Millis(*v)),
        Some(Value::Null) | None => Ok(Millis(0)),
        Some(other) => Err(Error::Backend(format!(
            "max_tx_from is neither an integer nor null: {other:?}"
        ))),
    }
}

/// Whether the next run must start from singletons rather than from the stored
/// assignment.
///
/// Leiden's local-moving phase is a greedy hill-climb, so seeding every run
/// from the previous one can hold a partition in a local optimum a cold start
/// would escape. This is the escape hatch, on a schedule.
///
/// Strictly greater, deliberately. On the default six-hour tick a cold run at
/// t=0 finds `24 > 24` false at t=24 and fires at t=30 instead. That is
/// inherent to sampling a threshold on an interval, it is fine for an escape
/// hatch, and ADR-0002 Amendment 1 records it so nobody files it as an
/// off-by-one and tightens the schedule.
fn cold_start_due(last_cold_start_at: Millis, now: Millis) -> bool {
    now.0 - last_cold_start_at.0 > Millis::days(1).0
}

/// Map the stored assignment onto the new dense index space, so Leiden can warm
/// start from it.
///
/// Iterates `labels`, never the stored rows, for two reasons that are both
/// requirements rather than style. The output must be exactly `labels.len()`
/// long or `run_with_initial_partition` rejects it as an invalid partition. And
/// a node that was an edge endpoint last run but is not one now simply has no
/// index this run, so walking the stored rows would produce a seed of the wrong
/// shape built around nodes that no longer exist.
///
/// New nodes take ids strictly above every stored id. Leiden groups by raw
/// integer equality and has no notion of a reserved id, so reusing one that an
/// existing community already holds would silently seed an unrelated new node
/// into that community and bias the run toward a merge with no basis in the
/// graph. The ids stay dense on purpose: `Partition::renumber` allocates a
/// vector sized by the LARGEST id, so a hash or a big constant offset would
/// turn a handful of communities into a huge allocation.
fn build_seed(labels: &[String], stored: &[(NodeId, i64)]) -> Vec<usize> {
    let by_id: HashMap<&str, usize> = stored
        .iter()
        .map(|(id, community)| (id.as_str(), usize::try_from(*community).unwrap_or(0)))
        .collect();
    // Stored ids are non-negative and this is their maximum, so every fresh id
    // below is above all of them.
    let mut next_fresh = by_id.values().copied().max().map_or(0, |max| max + 1);
    labels
        .iter()
        .map(|label| match by_id.get(label.as_str()) {
            Some(&community) => community,
            None => {
                let fresh = next_fresh;
                next_fresh += 1;
                fresh
            }
        })
        .collect()
}

fn intern(index: &mut HashMap<String, usize>, labels: &mut Vec<String>, id: String) -> usize {
    if let Some(&i) = index.get(&id) {
        return i;
    }
    let i = labels.len();
    labels.push(id.clone());
    index.insert(id, i);
    i
}

#[cfg(test)]
mod tests;
