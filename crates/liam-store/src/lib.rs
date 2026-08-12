// SPDX-License-Identifier: MIT OR Apache-2.0
//! liam-store: a bitemporal graph store with hybrid retrieval.
//!
//! Nothing is overwritten: superseded facts stay readable as history. The store
//! is generic over its storage engine (libSQL or, via the `backend-rusqlite`
//! feature, stock SQLite with sqlite-vec). It owns structure, time, and
//! retrieval; it does not own your domain model or your embedding model.

mod graph;
mod migrate;
mod schema;

pub mod backend;
pub mod backends;
pub mod clock;
#[cfg(feature = "cluster")]
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
pub use ids::{EdgeId, Millis, NodeId, FOREVER};
pub use types::{
    relation, Change, ExplainedHit, GcReport, GraphConfig, Hit, NewEdge, NewNode, Query,
    RetentionPolicy, RetentionRule,
};
pub use value::{Row, Value};

/// The graph over whichever backend feature is enabled. Most consumers use this.
pub type DefaultGraph = Graph<DefaultBackend>;
