// SPDX-License-Identifier: AGPL-3.0-only
//! rusqlite backend with sqlite-vec for vector search. STATUS: scaffold. The
//! shape is fixed; the bodies land next, because two things need care and a
//! compile to get right:
//!
//! 1. Sync/async bridge. rusqlite is synchronous, so the connection lives behind
//!    `Arc<Mutex<Connection>>` and every method runs its work inside
//!    `tokio::task::spawn_blocking`. That is why this backend pulls in tokio.
//! 2. sqlite-vec dialect. Vectors live in a `vec0` virtual table, not a column.
//!    Registration is `sqlite3_auto_extension(sqlite3_vec_init)` before opening;
//!    search is `WHERE embedding MATCH ? ORDER BY distance LIMIT ?`. Confirm the
//!    exact `vec0` DDL, insert, and match syntax against the sqlite-vec version.
//!
//! Filtering by validity/kind: sqlite-vec supports metadata columns on `vec0`,
//! or the search post-filters by joining back to `nodes`. The join approach
//! matches this crate's exact-recall stance and is the intended implementation.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::Connection;

use crate::backend::Backend;
use crate::error::Result;
use crate::ids::{Millis, NodeId};
use crate::value::{Row, Value};

pub struct RusqliteBackend {
    #[allow(dead_code)]
    conn: Arc<Mutex<Connection>>,
}

#[async_trait]
impl Backend for RusqliteBackend {
    async fn open(_path: &str) -> Result<Self> {
        // register sqlite-vec via sqlite3_auto_extension, then Connection::open.
        todo!("rusqlite backend: open + sqlite-vec registration")
    }

    async fn execute(&self, _sql: &str, _params: &[Value]) -> Result<u64> {
        todo!("spawn_blocking: conn.execute with mapped params")
    }

    async fn query(&self, _sql: &str, _params: &[Value]) -> Result<Vec<Row>> {
        todo!("spawn_blocking: conn.prepare + materialize rows into Vec<Row>")
    }

    async fn execute_batch(&self, _sql: &str) -> Result<()> {
        todo!("spawn_blocking: conn.execute_batch")
    }

    async fn execute_atomic(&self, _statements: &[(String, Vec<Value>)]) -> Result<()> {
        todo!("spawn_blocking: transaction over the statements")
    }

    fn vector_ddl(&self, _dims: usize) -> String {
        // e.g. CREATE VIRTUAL TABLE node_vectors USING vec0(
        //   node_id TEXT PRIMARY KEY, embedding float[dims]);
        todo!("sqlite-vec vec0 DDL")
    }

    async fn vector_upsert(&self, _node_id: &str, _embedding: &[f32]) -> Result<()> {
        todo!("insert into vec0 (embedding passed as bytes via zerocopy)")
    }

    async fn vector_delete(&self, _node_id: &str) -> Result<()> {
        todo!("delete from vec0 by node_id")
    }

    async fn vector_search(
        &self,
        _query: &[f32],
        _k: usize,
        _kind: Option<&str>,
        _scope: Option<&str>,
        _as_of: Millis,
    ) -> Result<Vec<NodeId>> {
        todo!("vec0 MATCH query, join nodes to filter by validity/kind/scope")
    }

    async fn vector_sweep_orphans(&self) -> Result<u64> {
        todo!("delete vec0 rows whose node_id is absent from nodes")
    }
}
