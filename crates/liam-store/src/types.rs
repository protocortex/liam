// SPDX-License-Identifier: Apache-2.0
//! Public value types. Domain-agnostic: `kind`, edge `type`, `scope`, and
//! `subject` are opaque strings; `attributes` is a JSON bag the library stores
//! and returns without interpreting.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::ids::{EdgeId, FactId, Millis, NodeId};

fn empty_attributes() -> Value {
    Value::Object(Map::new())
}

/// Reserved edge `type` values that carry library meaning. Use these instead of
/// string literals so provenance/versioning relations stay centralized.
pub mod relation {
    /// Links a new node to the prior version it replaced (contradiction handling).
    pub const SUPERSEDES: &str = "supersedes";
    /// Provenance: an entity node references a source fact/episode that mentions it.
    pub const MENTIONS: &str = "mentions";
    /// Links two nodes representing the same real-world entity.
    pub const SAME_AS: &str = "same_as";
    /// Links a node to the earlier node whose content it nearly repeats.
    pub const NEAR_DUPLICATE_OF: &str = "near_duplicate_of";
}

/// Construction parameters. `embedding_dims` sets the dimension the backend's
/// vector storage is created with; `rrf_k` tunes reciprocal rank fusion.
#[derive(Clone, Copy, Debug)]
pub struct GraphConfig {
    pub embedding_dims: usize,
    pub rrf_k: f64,
    /// Score multiplier applied to graph-expanded-only hits (0..=1). Keeps an
    /// inferred neighbour from outranking a direct match.
    pub expansion_weight: f64,
    /// Independent connections a pooling backend holds open for reads;
    /// passed straight through to `Backend::open`. A backend that cannot
    /// safely pool the configured path (an in-memory database, for
    /// `LibsqlBackend`) ignores this and falls back to a single shared
    /// connection regardless of what is configured here: that fallback is a
    /// correctness guard against handing concurrent readers an empty,
    /// private in-memory database, not a tuning knob a config value should
    /// ever be allowed to override.
    pub read_pool_size: usize,
}

impl GraphConfig {
    pub fn new(embedding_dims: usize) -> Self {
        Self {
            embedding_dims,
            rrf_k: 60.0,
            expansion_weight: 0.5,
            read_pool_size: 4,
        }
    }
    /// Sets the RRF constant. A negative value is clamped to zero and a
    /// non-finite one is ignored, so fused scores stay finite and comparable.
    pub fn with_rrf_k(mut self, k: f64) -> Self {
        if k.is_finite() {
            self.rrf_k = k.max(0.0);
        }
        self
    }
    pub fn with_expansion_weight(mut self, w: f64) -> Self {
        self.expansion_weight = w;
        self
    }
    pub fn with_read_pool_size(mut self, read_pool_size: usize) -> Self {
        self.read_pool_size = read_pool_size;
        self
    }
}

/// Names the line of versions one item's tier, usage and feedback follow. A new
/// version of a node keeps the key of the one it replaced, and two live nodes
/// that no write would make versions of each other never share one.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LineageKey(String);

impl LineageKey {
    /// A key read back from the store or the log. It is not checked, because
    /// only the store makes new ones.
    pub fn from_raw(raw: impl Into<String>) -> Self {
        Self(raw.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Where an item's embeddings are kept. A hot item has its f32 vector, int8 form
/// and signature; a cold one keeps its row, text index, edges and signature only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    Hot,
    Cold,
}

impl Tier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Cold => "cold",
        }
    }

    /// The tier `text` names. Anything else is an error, not a default.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "hot" => Ok(Self::Hot),
            "cold" => Ok(Self::Cold),
            other => Err(Error::UnknownTier(other.to_string())),
        }
    }
}

/// A change of tier for one lineage. `run_id` and `reason` say who moved it and
/// why, for the record that logs the move.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierMove {
    pub lineage_key: LineageKey,
    pub to: Tier,
    pub run_id: String,
    pub reason: String,
    pub changed_at: Millis,
    /// Until when the lineage stays where it was put, for a promotion.
    pub residency_until: Option<Millis>,
}

/// The accesses of one lineage in one logged batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageEntry {
    pub lineage_key: LineageKey,
    pub hits: u32,
    pub last_access_at: Millis,
}

/// A verdict on one result, as a producer gave it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedbackEvent {
    pub event_id: String,
    pub lineage_key: LineageKey,
    pub producer: String,
    pub at: Millis,
    pub correct: bool,
}

/// A feedback event that counted, as the utility reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountedFeedback {
    pub producer: String,
    pub at: Millis,
    pub correct: bool,
}

/// Everything the planner reads about one lineage that has a live node.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemView {
    pub lineage_key: LineageKey,
    /// When the newest live version was written, by transaction time.
    pub newest_tx_from: Millis,
    /// Hits per day as (day, hits), for the days inside the window, oldest first.
    pub hits_by_day: Vec<(i64, i64)>,
    pub last_access_at: Option<Millis>,
    /// Counted feedback, newest first.
    pub feedback: Vec<CountedFeedback>,
    pub tier: Tier,
    pub changed_at: Option<Millis>,
    pub residency_until: Option<Millis>,
    /// Some live node of the lineage has a vector.
    pub has_vector: bool,
    /// Every live node of the lineage that has a vector also has a signature.
    pub has_signature: bool,
    /// The producer of the newest live node.
    pub producer: String,
}

/// A consistent read of usage, feedback and tiers, taken once per run.
#[derive(Clone, Debug, PartialEq)]
pub struct UsageSnapshot {
    pub now: Millis,
    /// The start of the day of the earliest logged usage batch, `None` before
    /// the first.
    pub observation_since: Option<Millis>,
    /// Ordered by lineage key.
    pub items: Vec<ItemView>,
}

/// What `reconcile_tier_vectors` did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReconcileReport {
    /// Vectors deleted from nodes of a cold lineage.
    pub deleted: usize,
    /// Nodes written after the lineage went cold, which keep their vector.
    pub skipped_late: usize,
    /// Nodes with a vector and no signature, which keep it.
    pub skipped_unsigned: usize,
}

/// A live node of a lineage and what it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineageNode {
    pub node_id: NodeId,
    pub tx_from: Millis,
    pub has_vector: bool,
    pub has_signature: bool,
}

/// How a stored row came to be. Strict reads leave out `Inferred` rows, which
/// a model composed, so they never stand in for something a client recorded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Derivation {
    /// Written by a client as it was given.
    #[default]
    Recorded,
    /// Taken out of recorded content by an extractor.
    Extracted,
    /// Composed by a model, such as an entity page.
    Inferred,
}

impl Derivation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recorded => "recorded",
            Self::Extracted => "extracted",
            Self::Inferred => "inferred",
        }
    }

    /// The derivation `text` names. Anything else is an error, not a default.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "recorded" => Ok(Self::Recorded),
            "extracted" => Ok(Self::Extracted),
            "inferred" => Ok(Self::Inferred),
            other => Err(Error::UnknownDerivation(other.to_string())),
        }
    }
}

/// How a log event relates to the node it is listed for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventRole {
    /// The event that wrote the node.
    Origin,
    /// An event a composed node rests on, such as a mention an entity page
    /// was written from.
    Basis,
}

impl EventRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Origin => "origin",
            Self::Basis => "basis",
        }
    }
}

/// A log event id kept beside a node. `synthetic` is set when no log record
/// backs the id, so a caller never reads it as a real source.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventRef {
    pub event_id: String,
    pub synthetic: bool,
}

/// What a summary was made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildKind {
    Node,
    Summary,
}

impl ChildKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Summary => "summary",
        }
    }

    /// The kind `text` names. Anything else is an error, not a default.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "node" => Ok(Self::Node),
            "summary" => Ok(Self::Summary),
            other => Err(Error::Backend(format!(
                "summary_children holds an unknown child kind {other}"
            ))),
        }
    }
}

/// One child of a summary: a node at level 1, a lower summary above that.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryChild {
    pub id: String,
    pub kind: ChildKind,
}

/// A summary to store. Its identity in the tree is the child set, so the
/// children are given beside it and the store derives the `child_hash`.
#[derive(Clone, Debug)]
pub struct NewSummary {
    pub level: u8,
    /// '' for the unscoped tree.
    pub scope: String,
    pub text: String,
    /// How many of the summary's source events had their item text in the prompt.
    pub events_fed: u32,
    pub valid_min: Millis,
    pub valid_max: Millis,
    pub model_id: String,
    pub prompt_version: String,
}

/// What `put_summary` did. Both carry the id of the live row for the child set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PutOutcome {
    Inserted(String),
    /// A live row already covers this child set; nothing was written.
    Unchanged(String),
}

/// A stored summary version.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub id: String,
    pub level: u8,
    pub scope: String,
    pub child_hash: String,
    pub text: InferredText,
    pub derivation: Derivation,
    pub events_fed: u32,
    pub valid_min: Millis,
    pub valid_max: Millis,
    pub model_id: String,
    pub prompt_version: String,
    pub tx_from: Millis,
    pub tx_to: Millis,
}

/// A summary and its vector, as a read that ranks summaries needs them. The
/// vector is `None` when the summary has none, or one of another width than the
/// store's, as after an embedder change.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryEntry {
    pub summary: Summary,
    pub vector: Option<Vec<f32>>,
}

/// Whether a summary's events cover its children's, and how many of them are
/// real log events. A closure of synthetic events only proves nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosureReport {
    /// Children with an event the summary's own events lack.
    pub uncovered_children: Vec<String>,
    pub real_events: usize,
    pub synthetic_events: usize,
}

impl ClosureReport {
    pub fn is_closed(&self) -> bool {
        self.uncovered_children.is_empty()
    }
}

/// How often generation for one child set has failed, under the prompt and model
/// that failed. A different prompt or model starts the count again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SummaryFailure {
    pub failures: u32,
    pub last_error: String,
    pub prompt_version: String,
    pub model_id: String,
}

/// Where the last full build of one tree level left off, for the re-cluster policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeState {
    pub scope: String,
    pub level: u8,
    pub last_full_build_at: Millis,
    pub baseline_item_count: u32,
    pub unattached_count: u32,
}

/// A live recorded or extracted node, as the tree builder reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryInput {
    pub id: NodeId,
    pub kind: String,
    pub label: String,
    pub content: RecordedText,
    pub valid_from: Millis,
    pub scope: Option<String>,
    pub origin_events: Vec<EventRef>,
    pub vector: Vec<f32>,
}

/// The items of one scope that can be summarized.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryInputs {
    pub items: Vec<SummaryInput>,
    /// Live nodes left out because they have no stored vector.
    pub skipped_no_vector: usize,
}

/// A node to insert. Embedding is optional and supplied by the caller.
#[derive(Clone, Debug)]
pub struct NewNode {
    pub kind: String,
    pub label: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub attributes: Value,
    /// When the fact becomes true in the world. `None` means "resolve to the
    /// store's clock at insert time" so valid time stays deterministic under an
    /// injected clock; set it explicitly to backdate.
    pub valid_from: Option<Millis>,
    /// Optional retrieval partition (project, agent, namespace).
    pub scope: Option<String>,
    /// Optional identity for contradiction handling: a fact collides on
    /// `(subject, scope, producer)`; an entity page collides on `(subject, scope)` alone.
    pub subject: Option<String>,
    pub confidence: f64,
    /// Who wrote this node: an MCP client identity, a job name, or
    /// `"unknown"`. Written and stored only; it is deliberately absent from
    /// `Hit`, `Query`, and every daemon tool surface, since exposing
    /// provenance on read is M2.6 (tool surface) and M3.5 (scope/identity
    /// semantics), not this change. It is part of a fact's collision key
    /// (`subject`, `scope`, `producer`); an entity page ignores it, colliding
    /// on `(subject, scope)` alone.
    pub producer: String,
    /// Set only by `NewNode::entity`; any path rebuilding a `NewNode` from a stored row must go
    /// through `NewNode::entity`, never a struct literal, or the marker silently resets to false.
    pub(crate) entity_page: bool,
    /// Set only through `with_derivation`, which `NewNode::entity` alone calls.
    pub(crate) derivation: Derivation,
    /// Events the node is composed from, written with the node as its `Basis`
    /// events. Empty for a node that is not composed from other nodes.
    pub basis: Vec<EventRef>,
}

impl NewNode {
    pub fn now(
        kind: impl Into<String>,
        label: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            kind: kind.into(),
            label: label.into(),
            content: content.into(),
            embedding: None,
            attributes: empty_attributes(),
            valid_from: None,
            scope: None,
            subject: None,
            confidence: 1.0,
            producer: "unknown".to_string(),
            entity_page: false,
            derivation: Derivation::Recorded,
            basis: Vec::new(),
        }
    }
    /// An entity page node. `entity_type` becomes the `kind` (e.g. "person",
    /// "company", "concept"); `name` is the label and, normalized (trimmed +
    /// lowercased), the `subject` so re-observing the same entity supersedes
    /// via `upsert_by` rather than duplicating. Content is empty until M2
    /// synthesizes the compiled truth.
    pub fn entity(entity_type: impl Into<String>, name: impl Into<String>) -> Self {
        let name = name.into();
        let subject = name.trim().to_lowercase();
        let mut node = Self::now(entity_type, name, String::new()).with_subject(subject);
        node.entity_page = true;
        node.with_derivation(Derivation::Inferred)
    }
    fn with_derivation(mut self, derivation: Derivation) -> Self {
        self.derivation = derivation;
        self
    }
    pub fn with_basis(mut self, basis: Vec<EventRef>) -> Self {
        self.basis = basis;
        self
    }
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
    pub fn with_attributes(mut self, attributes: Value) -> Self {
        self.attributes = attributes;
        self
    }
    pub fn with_valid_from(mut self, valid_from: Millis) -> Self {
        self.valid_from = Some(valid_from);
        self
    }
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }
    pub fn with_producer(mut self, producer: impl Into<String>) -> Self {
        self.producer = producer.into();
        self
    }
}

/// How an `EpisodeEdge` names one of its endpoints inside one
/// `Graph::ingest_episode` call: either a node freshly created by that same
/// call, by its index into the `nodes` list passed alongside, or a node that
/// already exists in the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EpisodeRef {
    /// Index into the `nodes` list passed to `ingest_episode`.
    New(usize),
    /// A node id that already exists in the store.
    Existing(NodeId),
}

/// One edge inside an `ingest_episode` call, referencing its endpoints by
/// `EpisodeRef` so it can link nodes that do not have real ids yet.
#[derive(Clone, Debug)]
pub struct EpisodeEdge {
    pub from: EpisodeRef,
    pub to: EpisodeRef,
    pub kind: String,
    pub attributes: Value,
}

/// What `Graph::ingest_episode` wrote: the ids it assigned, in the same
/// order and length as the `nodes`/`edges` lists the caller passed in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpisodeResult {
    pub node_ids: Vec<NodeId>,
    pub edge_ids: Vec<EdgeId>,
}

/// A typed edge to insert.
#[derive(Clone, Debug)]
pub struct NewEdge {
    pub src: NodeId,
    pub dst: NodeId,
    pub kind: String,
    pub attributes: Value,
}

impl NewEdge {
    pub fn new(src: &NodeId, dst: &NodeId, kind: impl Into<String>) -> Self {
        Self {
            src: src.clone(),
            dst: dst.clone(),
            kind: kind.into(),
            attributes: empty_attributes(),
        }
    }
    pub fn with_attributes(mut self, attributes: Value) -> Self {
        self.attributes = attributes;
        self
    }
}

/// A retrieval request. Present signals are fused; absent ones are skipped.
#[derive(Clone, Debug)]
pub struct Query {
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub kind: Option<String>,
    pub scope: Option<String>,
    pub k: usize,
    /// When set, retrieve the live set as of this instant instead of now.
    pub as_of: Option<Millis>,
    /// When set, older facts are down-weighted with this half-life.
    pub half_life: Option<Millis>,
    /// Whether inferred rows, such as entity pages, may be retrieved. Off by
    /// default: a strict read returns only what a client recorded.
    pub include_inferred: bool,
}

impl Query {
    pub fn text(text: impl Into<String>) -> Self {
        Self::empty().with_text(text)
    }
    pub fn vector(embedding: Vec<f32>) -> Self {
        Self::empty().with_embedding(embedding)
    }
    fn empty() -> Self {
        Self {
            text: None,
            embedding: None,
            kind: None,
            scope: None,
            k: 8,
            as_of: None,
            half_life: None,
            include_inferred: false,
        }
    }
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    pub fn with_as_of(mut self, as_of: Millis) -> Self {
        self.as_of = Some(as_of);
        self
    }
    pub fn with_half_life(mut self, half_life: Millis) -> Self {
        self.half_life = Some(half_life);
        self
    }
    pub fn with_inferred(mut self, include_inferred: bool) -> Self {
        self.include_inferred = include_inferred;
        self
    }
}

/// Text a client recorded. It has no conversion from [`InferredText`], so code
/// cannot hand model-written text on as recorded:
///
/// ```compile_fail,E0308
/// use liam_store::{InferredText, RecordedText};
/// fn f(inferred: InferredText) -> RecordedText { inferred }
/// ```
///
/// The same function over one type compiles, so the failure above is the type
/// swap and not a bad import:
///
/// ```
/// use liam_store::{InferredText, RecordedText};
/// fn f(recorded: RecordedText) -> RecordedText { recorded }
/// fn g(inferred: InferredText) -> InferredText { inferred }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedText(String);

/// Text a model composed, such as an entity page. See [`RecordedText`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferredText(String);

impl RecordedText {
    pub(crate) fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// For fixtures in tests that build typed values without a store.
    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(text: impl Into<String>) -> Self {
        Self::new(text)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl InferredText {
    pub(crate) fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// For fixtures in tests that build typed values without a store.
    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(text: impl Into<String>) -> Self {
        Self::new(text)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The events an inferred row rests on. A row written before events were
/// listed has none, which reads as `Unknown` and not as an empty list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Basis {
    Known(Vec<EventRef>),
    Unknown,
}

/// A retrieved row that was recorded, with the events that wrote it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedHit {
    pub content: RecordedText,
    pub events: Vec<EventRef>,
    /// `Recorded` or `Extracted`.
    pub derivation: Derivation,
}

/// A retrieved row a model composed, with what it rests on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferredHit {
    pub text: InferredText,
    pub basis: Basis,
    pub derivation: Derivation,
    /// The level of a summary; `None` for an entity page.
    pub summary_level: Option<u8>,
}

/// The text of a retrieved row, typed by where it came from. Every reader gets
/// the text through this, so recorded and inferred content stay apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Retrieved {
    Recorded(RecordedHit),
    Inferred(InferredHit),
}

impl Retrieved {
    /// The text without its type, for a caller that only measures or matches it.
    /// Rendering must go through the variant instead, so its label follows.
    pub fn as_text(&self) -> &str {
        match self {
            Self::Recorded(hit) => hit.content.as_str(),
            Self::Inferred(hit) => hit.text.as_str(),
        }
    }

    pub fn is_inferred(&self) -> bool {
        matches!(self, Self::Inferred(_))
    }
}

/// A retrieved node with its final fused score.
#[derive(Clone, Debug)]
pub struct Hit {
    pub id: NodeId,
    pub kind: String,
    pub label: String,
    pub retrieved: Retrieved,
    pub attributes: Value,
    pub score: f64,
}

/// A hit plus the components that produced its score, for relevance debugging.
#[derive(Clone, Debug)]
pub struct ExplainedHit {
    pub hit: Hit,
    pub lexical_rank: Option<usize>,
    pub vector_rank: Option<usize>,
    pub rrf: f64,
    pub confidence: f64,
    pub decay: f64,
    /// The raw "known since" instant `decay` was computed from, so callers
    /// (e.g. the daemon's `ask` tool) can render a date instead of a factor.
    pub valid_from: Millis,
    pub expanded: bool,
}

/// A node that changed at or after a cursor, from `changes_since`.
#[derive(Clone, Debug)]
pub struct Change {
    pub id: NodeId,
    pub tx_from: Millis,
    /// Insertion order of the row, the tie-break between changes that share
    /// a `tx_from`.
    pub rowid: i64,
    /// True when the change is a closure (the node left the live set).
    pub closed: bool,
}

/// One retention rule: remove nodes of `kind` older than `max_age`.
#[derive(Clone, Debug)]
pub struct RetentionRule {
    pub kind: String,
    pub max_age: Millis,
}

/// What GC should sweep. The caller owns the policy; the library executes it.
#[derive(Clone, Debug)]
pub struct RetentionPolicy {
    pub rules: Vec<RetentionRule>,
    pub reclaim: bool,
}

impl RetentionPolicy {
    pub fn keep(kind: impl Into<String>, max_age: Millis) -> Self {
        Self {
            rules: vec![RetentionRule {
                kind: kind.into(),
                max_age,
            }],
            reclaim: true,
        }
    }
    pub fn and_keep(mut self, kind: impl Into<String>, max_age: Millis) -> Self {
        self.rules.push(RetentionRule {
            kind: kind.into(),
            max_age,
        });
        self
    }
    pub fn without_reclaim(mut self) -> Self {
        self.reclaim = false;
        self
    }
}

#[derive(Debug, Default)]
pub struct GcReport {
    pub nodes_removed: u64,
    pub edges_removed: u64,
}

/// The change signal clustering uses to decide whether a stored assignment is
/// still current: how many live semantic edges exist, and the newest
/// transaction time among them (ADR-0002).
///
/// Both halves are needed. `max_tx_from` catches an insertion, which is what
/// `relate` does. `edge_count` catches a deletion, which a timestamp cannot,
/// because `gc` hard-deletes edges and removing a row never advances a maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    pub edge_count: i64,
    pub max_tx_from: Millis,
}

/// The single `cluster_state` row: the fingerprint the stored assignment was
/// computed from, plus the two timestamps.
///
/// `computed_at` is for logs and is read by no rule. `last_cold_start_at`
/// advances ONLY on a from-singletons run, and is what the 24-hour rule reads.
/// Advancing it on a warm run makes the rule dead code on exactly the stores it
/// exists for; see ADR-0002 Amendment 1 before changing that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterState {
    pub fingerprint: Fingerprint,
    pub computed_at: Millis,
    pub last_cold_start_at: Millis,
}

/// One node inside a rendered community group. No community id: the integer
/// Leiden assigns is not durable across runs, so it must never reach a
/// client as something to store or compare (ADR-0002).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterMember {
    pub id: NodeId,
    pub kind: String,
    pub label: String,
}

/// How many objects a predicate may hold for one subject at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Cardinality {
    /// One live object: a newer or more trusted fact replaces or outranks it.
    One,
    /// Any number of live objects, none of which closes another.
    Many,
}

/// How two objects of a predicate are judged to be the same value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchMode {
    /// Equal normalized objects only.
    Exact,
    /// Also when one object's tokens contain the other's ("Sydney" and
    /// "Sydney, Australia").
    Contains,
}

/// What a fact's object must be found in, in the text it cites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Grounding {
    Text,
    Date,
}

/// The rules for one predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredicateSpec {
    pub cardinality: Cardinality,
    pub match_mode: MatchMode,
    pub grounding: Grounding,
}

impl Default for PredicateSpec {
    /// The non-destructive spec an undeclared predicate gets.
    fn default() -> Self {
        Self {
            cardinality: Cardinality::Many,
            match_mode: MatchMode::Exact,
            grounding: Grounding::Text,
        }
    }
}

/// The closed set of predicates and their rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vocabulary {
    specs: BTreeMap<String, PredicateSpec>,
}

impl Vocabulary {
    pub fn new(specs: BTreeMap<String, PredicateSpec>) -> Self {
        Self { specs }
    }

    /// The spec of `predicate`, or the default for one that was never declared.
    pub fn spec(&self, predicate: &str) -> PredicateSpec {
        self.specs.get(predicate).copied().unwrap_or_default()
    }
}

/// Bounds on how arbitration treats a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArbitrationCaps {
    /// Live competing lineages one key may hold before the weakest is retracted.
    pub max_competing_per_key: usize,
}

impl ArbitrationCaps {
    pub const DEFAULT_MAX_COMPETING_PER_KEY: usize = 5;
}

impl Default for ArbitrationCaps {
    fn default() -> Self {
        Self {
            max_competing_per_key: Self::DEFAULT_MAX_COMPETING_PER_KEY,
        }
    }
}

/// Which extractor produced a fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractorName {
    Rules,
    Llm,
}

impl ExtractorName {
    /// The value stored in `extractor_id` and the `extraction_cursor` name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rules => "rules",
            Self::Llm => "llm",
        }
    }

    /// The extractor stored under `stored`, or `None` for a value no write produces.
    pub(crate) fn from_stored(stored: &str) -> Option<Self> {
        [Self::Rules, Self::Llm]
            .into_iter()
            .find(|name| name.as_str() == stored)
    }
}

/// A fact to assert. `scope` is `None` for none; it is stored as ''.
/// `evidence_at` is the latest `min(observed_at, ingested_at)` among the cited
/// events, which decides recency; it is separate from `valid_from`.
#[derive(Clone, Debug)]
pub struct NewFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub scope: Option<String>,
    pub valid_from: Millis,
    pub valid_until: Millis,
    pub evidence_at: Millis,
    pub confidence: f64,
    pub trust_score: f64,
    pub extractor: ExtractorName,
    pub extractor_version: String,
    /// Log event ids the fact came from. A fact with none is refused.
    pub source_event_ids: Vec<String>,
}

/// One stored version of a fact.
#[derive(Clone, Debug, PartialEq)]
pub struct Fact {
    pub id: FactId,
    /// The id of the first version, shared by every version of this fact.
    pub lineage_id: FactId,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// The normalized object every equality check uses.
    pub object_key: String,
    pub scope: Option<String>,
    pub evidence_at: Millis,
    pub valid_from: Millis,
    pub valid_until: Millis,
    pub tx_from: Millis,
    pub tx_to: Millis,
    /// The version whose insertion closed this one, when a replacement did.
    pub closed_by: Option<FactId>,
    pub confidence: f64,
    pub trust_score: f64,
    pub extractor_id: ExtractorName,
    pub extractor_version: String,
    pub source_event_ids: Vec<String>,
}

/// What `assert_fact` did with a fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArbitrationOutcome {
    /// A retraction blocks this object for evidence this old.
    Suppressed,
    Inserted {
        id: FactId,
    },
    /// A live fact already says this; its sources were linked to it.
    Corroborated {
        lineage_id: FactId,
        changed: bool,
    },
    Replaced {
        closed: Vec<FactId>,
        inserted: FactId,
    },
    /// Stored beside a more trusted live fact, which stays open.
    Competing {
        id: FactId,
    },
    /// Stored as past history, because it rests on older evidence.
    Historical {
        id: FactId,
    },
}

/// What `end_fact` did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EndOutcome {
    Ended {
        closed: Vec<FactId>,
    },
    NoMatch,
    /// The live fact is more trusted than the statement ending it.
    Outranked,
    /// The end would not fall after the live fact's start.
    Rejected,
}

/// A statement that a fact has stopped being true. It closes a matching live
/// fact and never inserts a live one. `scope` is `None` for none.
#[derive(Clone, Debug)]
pub struct EndFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub scope: Option<String>,
    pub valid_until: Millis,
    /// The latest `min(observed_at, ingested_at)` among the cited events.
    pub evidence_at: Millis,
    pub trust_score: f64,
    /// Log event ids the statement came from. One with none is refused.
    pub source_event_ids: Vec<String>,
}

/// What `retract_fact` did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetractOutcome {
    /// The lineage is no longer asserted. `restored` lists the live versions
    /// brought back for the facts it had replaced.
    Retracted { restored: Vec<FactId> },
    /// The lineage has no current version.
    NotFound,
    /// Restoring was asked for, but another live fact holds the key; nothing
    /// changed.
    RestoreBlocked,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_undeclared_predicate_is_many_valued_with_exact_matching() {
        // Arrange
        let vocabulary = Vocabulary::new(BTreeMap::from([(
            "works_at".to_string(),
            PredicateSpec {
                cardinality: Cardinality::One,
                ..PredicateSpec::default()
            },
        )]));

        // Act
        let declared = vocabulary.spec("works_at");
        let undeclared = vocabulary.spec("resides_in");

        // Assert
        assert_eq!(declared.cardinality, Cardinality::One);
        assert_eq!(undeclared, PredicateSpec::default());
        assert_eq!(undeclared.cardinality, Cardinality::Many);
        assert_eq!(undeclared.match_mode, MatchMode::Exact);
    }

    #[test]
    fn extractor_names_are_the_stored_ids() {
        assert_eq!(ExtractorName::Rules.as_str(), "rules");
        assert_eq!(ExtractorName::Llm.as_str(), "llm");
    }

    #[test]
    fn entity_page_flag_set_only_by_entity_constructor() {
        let fact = NewNode::now("fact", "l", "c").with_subject("x");
        assert!(!fact.entity_page);

        let page = NewNode::entity("person", "X");
        assert!(page.entity_page);
    }

    #[test]
    fn only_the_entity_constructor_sets_an_inferred_derivation() {
        // Arrange / Act
        let page = NewNode::entity("person", "X");
        let note = NewNode::now("fact", "l", "c");

        // Assert
        assert_eq!(page.derivation, Derivation::Inferred);
        assert_eq!(note.derivation, Derivation::Recorded);
    }

    #[test]
    fn every_derivation_parses_back_from_its_stored_text() {
        for derivation in [
            Derivation::Recorded,
            Derivation::Extracted,
            Derivation::Inferred,
        ] {
            // Act
            let parsed = Derivation::parse(derivation.as_str());

            // Assert
            assert_eq!(parsed.unwrap(), derivation);
        }
    }

    #[test]
    fn an_unknown_derivation_text_is_a_typed_error_not_a_default() {
        // Act
        let parsed = Derivation::parse("weird");

        // Assert
        assert!(matches!(parsed, Err(Error::UnknownDerivation(text)) if text == "weird"));
    }
}
