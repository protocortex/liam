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
    batch_key, child_hash, recover_interrupted_restore, validate_scope, verify_snapshot,
    BackfillReport, Candidate, CatchUpReport, ContentEmbedder, EmbedError, EventLog, FactPage,
    Graph, PoisonHook, RebuildMode, RebuildReport, ReembedReport, SharedLog, Snapshot, SnapshotLog,
    SnapshotMeta, MAX_FACTS_PER_BATCH, MAX_SOURCES_PER_FACT, RUN_SOURCE_PREFIX,
};
pub use ids::{EdgeId, FactId, Millis, NodeId, BEGINNING, FOREVER, HANDLE_LEN};
pub use types::{
    relation, ArbitrationCaps, ArbitrationOutcome, Basis, Cardinality, Change, ChildKind,
    ClosureReport, ClusterMember, ClusterState, Derivation, EndFact, EndOutcome, EventRef,
    EventRole, ExplainedHit, ExtractorName, Fact, Fingerprint, GcReport, GraphConfig, Grounding,
    Hit, InferredHit, InferredText, MatchMode, NewEdge, NewFact, NewNode, NewSummary,
    PredicateSpec, PutOutcome, Query, RecordedHit, RecordedText, RetentionPolicy, RetentionRule,
    RetractOutcome, Retrieved, Summary, SummaryChild, SummaryEntry, SummaryFailure, SummaryInput,
    SummaryInputs, TreeState, Vocabulary,
};
pub use value::{Row, Value};

/// The graph over whichever backend feature is enabled. Most consumers use this.
pub type DefaultGraph = Graph<DefaultBackend>;
