// SPDX-License-Identifier: Apache-2.0
//! liam-store: a bitemporal graph store with hybrid retrieval.
//!
//! Nothing is overwritten: superseded facts stay readable as history. The store
//! is generic over its storage engine (libSQL today; see `backends` for why
//! the trait stays with one implementation). It owns structure, time, and
//! retrieval; it does not own your domain model or your embedding model.

mod graph;
mod migrate;
mod schema;

pub mod backend;
pub mod backends;
pub mod clock;
pub mod cluster;
pub mod error;
pub mod ids;
pub mod types;
pub mod value;

pub use backend::Backend;
pub use backends::DefaultBackend;
pub use clock::{Clock, FixedClock, SystemClock};
pub use error::{Error, MismatchSource, Result};
pub use graph::{
    batch_key, child_hash, dequantize, hamming, quantize, recover_interrupted_restore,
    validate_scope, verify_snapshot, BackfillReport, Candidate, CatchUpReport, ContentEmbedder,
    DeletionRecord, EmbedError, EventLog, FactPage, Graph, IndexedEntry, LineageStore, PoisonHook,
    QuantIndex, Quantized, QueryTimings, RebuildMode, RebuildReport, ReembedReport, ScrubReport,
    SharedLog, Snapshot, SnapshotLog, SnapshotMeta, TierBytes, BITS_PER_WORD, GRAPH_STORE,
    MAX_FACTS_PER_BATCH, MAX_SOURCES_PER_FACT, RUN_SOURCE_PREFIX,
};
pub use ids::{EdgeId, FactId, Millis, NodeId, BEGINNING, FOREVER, HANDLE_LEN};
pub use types::{
    relation, ApplyOutcome, ArbitrationCaps, ArbitrationOutcome, Basis, Cardinality, Change,
    ChildKind, ClosureReport, ClusterMember, ClusterState, CountedFeedback, Derivation, EndFact,
    EndOutcome, EventRef, EventRole, ExplainedHit, ExtractorName, Fact, FeedbackEvent, Fingerprint,
    GcReport, GraphConfig, Grounding, Hit, InferredHit, InferredText, ItemView, LineageHits,
    LineageKey, LineageNode, LineageTarget, MatchMode, NewEdge, NewFact, NewNode, NewSummary,
    NodeText, OriginClaim, OriginVerdict, PredicateSpec, ProducerUsage, PurgeReport, PutOutcome,
    Query, ReconcileReport, RecordedHit, RecordedText, Residual, RetentionPolicy, RetentionRule,
    RetractOutcome, Retrieved, Summary, SummaryChild, SummaryEntry, SummaryFailure, SummaryInput,
    SummaryInputs, Tier, TierMove, TreeState, UsageEntry, UsageSnapshot, Vocabulary,
};
pub use value::{Row, Value};

/// The graph over whichever backend feature is enabled. Most consumers use this.
pub type DefaultGraph = Graph<DefaultBackend>;
