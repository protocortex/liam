// SPDX-License-Identifier: AGPL-3.0-only
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
pub use error::{Error, Result};
pub use graph::Graph;
pub use ids::{EdgeId, Millis, NodeId, FOREVER, HANDLE_LEN};
pub use types::{
    relation, Change, ClusterState, ExplainedHit, Fingerprint, GcReport, GraphConfig, Hit, NewEdge,
    NewNode, Query, RetentionPolicy, RetentionRule,
};
pub use value::{Row, Value};

/// The graph over whichever backend feature is enabled. Most consumers use this.
pub type DefaultGraph = Graph<DefaultBackend>;
