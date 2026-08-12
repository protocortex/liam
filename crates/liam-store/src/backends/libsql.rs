// SPDX-License-Identifier: MIT OR Apache-2.0
//! libSQL backend. Native vector search via `F32_BLOB` and `vector_distance_cos`,
//! with embeddings in a `node_vectors` table so the search can prefilter against
//! the live node set in one query.
//!
//! VERSION CHECK: the libSQL parameter-binding call (`params_from_iter`) and the
//! row accessors (`column_count`, `get_value`) are the surface to confirm against
//! the version you pin.

use async_trait::async_trait;
use libsql::{Builder, Connection, Database};

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::ids::{Millis, NodeId};
use crate::value::{Row, Value};

/// Milliseconds a connection waits on `SQLITE_BUSY` before giving up.
/// Config plumbing for this arrives in WU-5; until then a named constant is
/// correct, not a parameter nobody can set yet.
const BUSY_TIMEOUT_MS: i64 = 5000;

/// Retains the `Database` handle alongside the connection it built, so more
/// connections can be opened against the same file later. Dropping the
/// `Database` (as the previous version of this backend did) would leave the
/// read pool WU-2 adds with no way to open further connections.
pub struct LibsqlBackend {
    // Not read by production code yet: the read pool WU-2 adds is its first
    // production reader. Until then this module's own tests read it directly
    // to prove a second connection can be opened and configured, which is
    // this Work Unit's Done When.
    #[allow(dead_code)]
    db: Database,
    conn: Connection,
}

fn err(e: libsql::Error) -> Error {
    Error::Backend(e.to_string())
}

/// Applies the pragmas that describe connection state rather than database
/// file state. SQLite resets these to their defaults on every new
/// connection, unlike `journal_mode`, which is persisted in the database
/// file itself, so this cannot be a one-time call at `open`: every
/// connection this backend hands out, including the read pool WU-2 adds,
/// must go through this helper.
async fn configure_connection(conn: &Connection) -> Result<()> {
    conn.query(&format!("PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}"), ())
        .await
        .map_err(err)?;
    // NORMAL is the standard companion to WAL: it syncs at checkpoints
    // rather than on every commit. FULL costs a sync per commit for
    // durability this workload does not need, since WAL already protects
    // against corruption on a crash; at worst the last few commits are lost.
    conn.query("PRAGMA synchronous = NORMAL", ())
        .await
        .map_err(err)?;
    Ok(())
}

fn to_libsql(v: &Value) -> libsql::Value {
    match v {
        Value::Null => libsql::Value::Null,
        Value::Int(i) => libsql::Value::Integer(*i),
        Value::Real(r) => libsql::Value::Real(*r),
        Value::Text(s) => libsql::Value::Text(s.clone()),
        Value::Blob(b) => libsql::Value::Blob(b.clone()),
    }
}

fn from_libsql(v: libsql::Value) -> Value {
    match v {
        libsql::Value::Null => Value::Null,
        libsql::Value::Integer(i) => Value::Int(i),
        libsql::Value::Real(r) => Value::Real(r),
        libsql::Value::Text(s) => Value::Text(s),
        libsql::Value::Blob(b) => Value::Blob(b),
    }
}

fn bind(params: &[Value]) -> Vec<libsql::Value> {
    params.iter().map(to_libsql).collect()
}

async fn read_rows(mut rows: libsql::Rows) -> Result<Vec<Row>> {
    let cols = rows.column_count();
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(err)? {
        let mut values = Vec::with_capacity(cols as usize);
        for i in 0..cols {
            values.push(from_libsql(row.get_value(i).map_err(err)?));
        }
        out.push(Row(values));
    }
    Ok(out)
}

fn le_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for x in embedding {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

fn vector_literal(embedding: &[f32]) -> String {
    let mut out = String::with_capacity(embedding.len() * 8 + 2);
    out.push('[');
    for (i, x) in embedding.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&x.to_string());
    }
    out.push(']');
    out
}

#[async_trait]
impl Backend for LibsqlBackend {
    async fn open(path: &str) -> Result<Self> {
        let db = Builder::new_local(path).build().await.map_err(err)?;
        let conn = db.connect().map_err(err)?;
        // WAL is persistent in the database file, not the connection, so it
        // is set once here rather than in `configure_connection`. Verified
        // empirically against libsql 0.9.30: issuing `PRAGMA journal_mode =
        // WAL` as the very first statement on the first connection sticks,
        // confirmed by querying `PRAGMA journal_mode` back on a file-backed
        // database (see `wal_is_enabled_on_a_file_backed_database` below),
        // which reports `wal` rather than the default `delete`. It is a
        // no-op on `:memory:` (SQLite keeps in-memory databases on their own
        // journal mode regardless of what is requested), so it runs
        // unconditionally here and its result is asserted only for
        // file-backed paths.
        conn.query("PRAGMA journal_mode = WAL", ())
            .await
            .map_err(err)?;
        configure_connection(&conn).await?;
        Ok(Self { db, conn })
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        self.conn
            .execute(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)
    }

    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let rows = self
            .conn
            .query(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)?;
        read_rows(rows).await
    }

    async fn execute_batch(&self, sql: &str) -> Result<()> {
        self.conn.execute_batch(sql).await.map(|_| ()).map_err(err)
    }

    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()> {
        let tx = self.conn.transaction().await.map_err(err)?;
        for (sql, params) in statements {
            tx.execute(sql, libsql::params_from_iter(bind(params)))
                .await
                .map_err(err)?;
        }
        tx.commit().await.map_err(err)
    }

    fn vector_ddl(&self, dims: usize) -> String {
        format!(
            "CREATE TABLE IF NOT EXISTS node_vectors (
  node_id   TEXT NOT NULL PRIMARY KEY REFERENCES nodes(id),
  embedding F32_BLOB({dims}) NOT NULL
);"
        )
    }

    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO node_vectors (node_id, embedding) VALUES (?1, ?2)
                 ON CONFLICT(node_id) DO UPDATE SET embedding = excluded.embedding",
                libsql::params_from_iter(vec![
                    libsql::Value::Text(node_id.to_string()),
                    libsql::Value::Blob(le_bytes(embedding)),
                ]),
            )
            .await
            .map_err(err)?;
        Ok(())
    }

    async fn vector_delete(&self, node_id: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM node_vectors WHERE node_id = ?1",
                libsql::params_from_iter(vec![libsql::Value::Text(node_id.to_string())]),
            )
            .await
            .map_err(err)?;
        Ok(())
    }

    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        kind: Option<&str>,
        scope: Option<&str>,
        as_of: Millis,
    ) -> Result<Vec<NodeId>> {
        let mut params = vec![
            libsql::Value::Text(vector_literal(query)),
            libsql::Value::Integer(as_of.0),
            libsql::Value::Integer(k as i64),
        ];
        let mut filters = String::new();
        let mut next = 4;
        if let Some(kind) = kind {
            filters.push_str(&format!(" AND n.kind = ?{next}"));
            params.push(libsql::Value::Text(kind.to_string()));
            next += 1;
        }
        if let Some(scope) = scope {
            filters.push_str(&format!(" AND n.scope = ?{next}"));
            params.push(libsql::Value::Text(scope.to_string()));
        }
        // Same four-bound "live at T" predicate the lexical path enforces
        // (see `live_at` in graph.rs): recorded before T, not yet superseded
        // at T, and true in the world at T. `?2` is `as_of`, reused for all
        // four bounds.
        let sql = format!(
            "SELECT v.node_id FROM node_vectors v
             JOIN nodes n ON n.id = v.node_id
             WHERE n.tx_from <= ?2 AND n.tx_to > ?2
               AND n.valid_from <= ?2 AND n.valid_until > ?2{filters}
             ORDER BY vector_distance_cos(v.embedding, vector(?1)) LIMIT ?3"
        );
        let rows = self
            .conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(err)?;
        let rows = read_rows(rows).await?;
        rows.iter()
            .map(|r| Ok(NodeId::from_raw(r.get_string(0)?)))
            .collect()
    }

    async fn vector_sweep_orphans(&self) -> Result<u64> {
        self.conn
            .execute(
                "DELETE FROM node_vectors WHERE node_id NOT IN (SELECT id FROM nodes)",
                libsql::params_from_iter(Vec::<libsql::Value>::new()),
            )
            .await
            .map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// `:memory:` cannot stand in for this: WAL is a no-op on an in-memory
    /// database, so the assertion below would be vacuous there. See
    /// `memory_backend_opens_and_executes_without_a_wal_assertion` for the
    /// in-memory case this deliberately does not claim.
    #[tokio::test]
    async fn wal_is_enabled_on_a_file_backed_database() {
        // Arrange
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("wal.db");

        // Act
        let backend = LibsqlBackend::open(path.to_str().expect("utf8 path"))
            .await
            .expect("open file-backed backend");
        let rows = backend
            .query("PRAGMA journal_mode", &[])
            .await
            .expect("query journal_mode");

        // Assert: queried back, not assumed.
        assert_eq!(rows[0].get_string(0).unwrap(), "wal");
    }

    #[tokio::test]
    async fn a_connection_created_after_open_gets_the_configured_busy_timeout() {
        // Arrange
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("busy.db");
        let backend = LibsqlBackend::open(path.to_str().expect("utf8 path"))
            .await
            .expect("open file-backed backend");

        // Act: open a further connection the way the read pool (WU-2) will,
        // routed through the same per-connection pragma helper, and query
        // its busy_timeout back rather than assuming the call landed.
        let second = backend
            .db
            .connect()
            .map_err(err)
            .expect("second connection");
        configure_connection(&second)
            .await
            .expect("configure second connection");
        let rows = read_rows(
            second
                .query("PRAGMA busy_timeout", ())
                .await
                .expect("query busy_timeout"),
        )
        .await
        .expect("read busy_timeout rows");

        // Assert
        assert_eq!(rows[0].get_i64(0).unwrap(), BUSY_TIMEOUT_MS);
    }

    #[tokio::test]
    async fn memory_backend_opens_and_executes_without_a_wal_assertion() {
        // Arrange & Act
        let backend = LibsqlBackend::open(":memory:")
            .await
            .expect("open in-memory backend");
        let rows = backend.query("SELECT 1", &[]).await.expect("query");

        // Assert: opening and querying succeed. `journal_mode` is
        // deliberately not checked here: WAL does not apply to in-memory
        // databases, so that assertion would be vacuous or wrong.
        assert_eq!(rows[0].get_i64(0).unwrap(), 1);
    }
}
