// SPDX-License-Identifier: Apache-2.0
//! The backend seam. Everything the shared graph logic needs from storage.
//!
//! The design insight: full-text, graph, and CRUD SQL are identical across
//! libSQL and stock SQLite, so they run through `execute`/`query`. Only vector
//! storage and search diverge (libSQL native functions versus the sqlite-vec
//! extension), so those are their own methods and each backend owns its dialect
//! and its physical layout.

use async_trait::async_trait;

use crate::error::Result;
use crate::ids::{Millis, NodeId};
use crate::value::{Row, Value};

/// **Concurrency contract.** Writes serialize; reads do not.
///
/// `execute`, `execute_batch`, `execute_atomic`, and the vector-writing
/// methods (`vector_upsert`, `vector_delete`, `vector_sweep_orphans`) are
/// writes: a backend MUST ensure at most one runs at a time, however it
/// chooses to enforce that (`LibsqlBackend` uses one write connection behind
/// an async mutex).
///
/// `query` and `vector_search` are reads: a backend MUST let them run
/// concurrently with an in-flight write rather than queuing behind it.
/// Routing every connection, read or write, through a single shared
/// connection would make "concurrent clients" true at the transport layer
/// and false at the store, which is the whole reason this store gained a
/// separate read pool. Any second backend must honour the same contract.
#[async_trait]
pub trait Backend: Send + Sync + Sized {
    /// Open (and if needed create) a local database at `path`.
    ///
    /// `read_pool_size` requests that many independent connections be held
    /// open for reads, when this backend pools them at all; a backend that
    /// does not pool reads (or, like `LibsqlBackend` on an unpoolable path,
    /// cannot safely pool the given `path`) MAY ignore it.
    async fn open(path: &str, read_pool_size: usize) -> Result<Self>;

    /// Write. Serializes with every other write; see the trait's
    /// concurrency contract above.
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64>;
    /// Read. Does not serialize with writes or with other reads; see the
    /// trait's concurrency contract above.
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>>;
    /// Write. Serializes with every other write; see the trait's
    /// concurrency contract above.
    async fn execute_batch(&self, sql: &str) -> Result<()>;

    /// Write. Run several statements in one transaction, all or nothing.
    /// Serializes with every other write; see the trait's concurrency
    /// contract above.
    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()>;

    // ---- vector capability: the one backend-divergent piece ----

    /// DDL for this backend's vector storage, appended to the core schema.
    fn vector_ddl(&self, dims: usize) -> String;

    /// Write. Serializes with every other write; see the trait's
    /// concurrency contract above.
    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()>;
    /// Write. Serializes with every other write; see the trait's
    /// concurrency contract above.
    async fn vector_delete(&self, node_id: &str) -> Result<()>;

    /// Read. Nearest node ids to `query`, restricted to the live set at
    /// `as_of` and, when given, to `kind` and `scope`. Each backend applies
    /// the filter in its own layout. Does not serialize with writes or with
    /// other reads; see the trait's concurrency contract above.
    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        kind: Option<&str>,
        scope: Option<&str>,
        as_of: Millis,
    ) -> Result<Vec<NodeId>>;

    /// Write. Remove stored vectors whose node no longer exists (post-GC
    /// cleanup). Serializes with every other write; see the trait's
    /// concurrency contract above.
    async fn vector_sweep_orphans(&self) -> Result<u64>;
}
