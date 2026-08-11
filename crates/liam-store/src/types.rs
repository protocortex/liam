// SPDX-License-Identifier: MIT OR Apache-2.0
//! Public value types. Domain-agnostic: `kind`, edge `type`, `scope`, and
//! `subject` are opaque strings; `attributes` is a JSON bag the library stores
//! and returns without interpreting.

use serde_json::{Map, Value};

use crate::ids::{Millis, NodeId};

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
}

impl GraphConfig {
    pub fn new(embedding_dims: usize) -> Self {
        Self {
            embedding_dims,
            rrf_k: 60.0,
            expansion_weight: 0.5,
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
