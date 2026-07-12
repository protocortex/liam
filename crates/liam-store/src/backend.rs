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

#[async_trait]
pub trait Backend: Send + Sync + Sized {
    /// Open (and if needed create) a local database at `path`.
    async fn open(path: &str) -> Result<Self>;

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64>;
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>>;
    async fn execute_batch(&self, sql: &str) -> Result<()>;

    /// Run several statements in one transaction, all or nothing.
    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()>;

    // ---- vector capability: the one backend-divergent piece ----

    /// DDL for this backend's vector storage, appended to the core schema.
    fn vector_ddl(&self, dims: usize) -> String;

    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()>;
    async fn vector_delete(&self, node_id: &str) -> Result<()>;

    /// Nearest node ids to `query`, restricted to the live set at `as_of` and,
    /// when given, to `kind` and `scope`. Each backend applies the filter in its
    /// own layout.
    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        kind: Option<&str>,
        scope: Option<&str>,
        as_of: Millis,
    ) -> Result<Vec<NodeId>>;

    /// Remove stored vectors whose node no longer exists (post-GC cleanup).
    async fn vector_sweep_orphans(&self) -> Result<u64>;
}
