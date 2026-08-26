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
///
/// `begin` is a write too, but a coarser one than any other method here.
/// Every other write method holds the write lock for exactly one statement
/// (or, for `execute_atomic`, one fixed, small, library-built list); a
/// `BackendTx` returned by `begin` holds it for its entire open lifetime,
/// however long the caller takes between `begin()` and its
/// `commit()`/`rollback()`. A caller that keeps a `BackendTx` open across an
/// unbounded amount of work blocks every other write for that whole span,
/// so this capability is deliberately narrow: it exists for callers that
/// can bound how long they hold it open, not as a general substitute for
/// `execute`/`execute_atomic`.
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

    /// Open a transaction. Write. Serializes with every other write, but
    /// for as long as the returned `BackendTx` stays open, not for one
    /// statement; see the trait's concurrency contract above.
    async fn begin(&self) -> Result<Box<dyn BackendTx + '_>>;
}

/// An open transaction on a `Backend`. Holds whatever lock or connection
/// state its backend needs for the duration between `begin()` and this
/// being consumed by `commit()`/`rollback()` (or dropped without either,
/// which each implementor must define a safe fallback for).
#[async_trait]
pub trait BackendTx: Send {
    /// Write, inside this transaction. Not yet durable until `commit()`.
    async fn execute(&mut self, sql: &str, params: &[Value]) -> Result<u64>;
    /// Read, inside this transaction: sees this transaction's own writes so
    /// far, not yet visible to any other connection.
    async fn query(&mut self, sql: &str, params: &[Value]) -> Result<Vec<Row>>;
    /// Consume this transaction, making its writes durable.
    async fn commit(self: Box<Self>) -> Result<()>;
    /// Consume this transaction, discarding its writes.
    async fn rollback(self: Box<Self>) -> Result<()>;
}
