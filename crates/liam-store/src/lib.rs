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
    batch_key, recover_interrupted_restore, BackfillReport, Candidate, CatchUpReport,
    ContentEmbedder, EmbedError, EventLog, Graph, PoisonHook, RebuildMode, RebuildReport,
    ReembedReport, SharedLog, Snapshot, SnapshotLog, SnapshotMeta,
};
pub use ids::{EdgeId, FactId, Millis, NodeId, BEGINNING, FOREVER, HANDLE_LEN};
pub use types::{
    relation, ArbitrationCaps, ArbitrationOutcome, Cardinality, Change, ClusterMember,
    ClusterState, EndOutcome, ExplainedHit, ExtractorName, Fact, Fingerprint, GcReport,
    GraphConfig, Grounding, Hit, MatchMode, NewEdge, NewFact, NewNode, PredicateSpec, Query,
    RetentionPolicy, RetentionRule, Vocabulary,
};
pub use value::{Row, Value};

/// The graph over whichever backend feature is enabled. Most consumers use this.
pub type DefaultGraph = Graph<DefaultBackend>;
