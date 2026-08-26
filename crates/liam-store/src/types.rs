// SPDX-License-Identifier: Apache-2.0
//! Public value types. Domain-agnostic: `kind`, edge `type`, `scope`, and
//! `subject` are opaque strings; `attributes` is a JSON bag the library stores
//! and returns without interpreting.

use serde_json::{Map, Value};

use crate::ids::{EdgeId, Millis, NodeId};

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
    pub fn with_rrf_k(mut self, k: f64) -> Self {
        self.rrf_k = k;
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
    /// Optional identity for contradiction handling; two live nodes with the
    /// same subject in the same scope are treated as competing versions.
    pub subject: Option<String>,
    pub confidence: f64,
    /// Who wrote this node: an MCP client identity, a job name, or
    /// `"unknown"`. Written and stored only; it is deliberately absent from
    /// `Hit`, `Query`, and every daemon tool surface, since exposing
    /// provenance on read is M2.6 (tool surface) and M3.5 (scope/identity
    /// semantics), not this change. It also plays no part in the
    /// `upsert_by`/`supersede` competitor key: two live nodes for one
    /// subject differing only by producer is the conflict M3.5 owns, so
    /// last writer still wins here regardless of who wrote it.
    pub producer: String,
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
        Self::now(entity_type, name, String::new()).with_subject(subject)
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
}

/// A retrieved node with its final fused score.
#[derive(Clone, Debug)]
pub struct Hit {
    pub id: NodeId,
    pub kind: String,
    pub label: String,
    pub content: String,
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
