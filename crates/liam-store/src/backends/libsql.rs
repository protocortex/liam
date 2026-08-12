// SPDX-License-Identifier: MIT OR Apache-2.0
//! libSQL backend. Native vector search via `F32_BLOB` and `vector_distance_cos`,
//! with embeddings in a `node_vectors` table so the search can prefilter against
//! the live node set in one query.
//!
//! VERSION CHECK: the libSQL parameter-binding call (`params_from_iter`) and the
//! row accessors (`column_count`, `get_value`) are the surface to confirm against
//! the version you pin.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use libsql::{Builder, Connection, Database};
use tokio::sync::Mutex;

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::ids::{Millis, NodeId};
use crate::value::{Row, Value};

/// Milliseconds a connection waits on `SQLITE_BUSY` before giving up.
/// Config plumbing for this arrives in WU-5; until then a named constant is
/// correct, not a parameter nobody can set yet.
const BUSY_TIMEOUT_MS: i64 = 5000;

/// Number of independent connections held for reads on a file-backed
/// database. Config plumbing for this arrives in WU-5 (daemon side); until
/// then a named constant is correct here, not a setter nobody calls yet.
const READ_POOL_SIZE: usize = 4;

/// Whether `path` is one of the in-memory database spellings SQLite
/// accepts: the bare `:memory:`, a `file:` URI whose file part is
/// `:memory:` (for example `file::memory:?cache=shared`), or any `file:`
/// URI carrying a `mode=memory` query parameter (for example
/// `file:name?mode=memory&cache=shared`). `database_path` comes from user
/// config (`liam.toml`), so a caller can spell "in-memory" more than one
/// way; missing a spelling here would open `READ_POOL_SIZE` separate,
/// empty in-memory databases and silently lose every write. Forcing the
/// pool down to one connection is safe even for `cache=shared`, where
/// several connections would actually share the same database, so this
/// errs toward treating a path as in-memory rather than not.
fn is_in_memory(path: &str) -> bool {
    if path == ":memory:" {
        return true;
    }
    let Some(rest) = path.strip_prefix("file:") else {
        return false;
    };
    let (file_part, query) = rest.split_once('?').unwrap_or((rest, ""));
    file_part == ":memory:" || query.split('&').any(|param| param == "mode=memory")
}

pub struct LibsqlBackend {
    /// The single write connection, taken by `execute`, `execute_batch`,
    /// `execute_atomic`, and every vector-writing method. Guarding it with a
    /// mutex serializes writes at the application level instead of letting
    /// them race on `SQLITE_BUSY` and hoping `busy_timeout` sorts it out.
    write: Mutex<Connection>,
    /// Connections used for reads (`query`, `vector_search`), picked in
    /// round robin. Reads never take `write`'s lock, so a read completes
    /// even while a write is in flight; that is the entire reason this pool
    /// exists separately from `write`.
    ///
    /// For an in-memory database this holds exactly one entry: a CLONE of
    /// the write connection (see `open`), never a second `db.connect()`.
    /// Each connection to `:memory:` is its own private database, so a
    /// second `connect()` would silently hand reads an empty store; cloning
    /// the `Connection` handle (cheap: it wraps an `Arc`) reuses the exact
    /// same underlying database instead.
    read_pool: Vec<Connection>,
    next_reader: AtomicUsize,
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

/// Opens `READ_POOL_SIZE` fresh connections against `db`, each configured
/// with the same per-connection pragmas as every other connection this
/// backend hands out. Only called for file-backed databases: `open`'s
/// `:memory:` branch never reaches this, because a second `db.connect()`
/// there would open an unrelated, empty in-memory database rather than
/// another handle onto the same one.
async fn open_read_pool(db: &Database) -> Result<Vec<Connection>> {
    let mut pool = Vec::with_capacity(READ_POOL_SIZE);
    for _ in 0..READ_POOL_SIZE {
        let conn = db.connect().map_err(err)?;
        configure_connection(&conn).await?;
        pool.push(conn);
    }
    Ok(pool)
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
        let write_conn = db.connect().map_err(err)?;
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
        write_conn
            .query("PRAGMA journal_mode = WAL", ())
            .await
            .map_err(err)?;
        configure_connection(&write_conn).await?;

        // Guard `:memory:` explicitly, before anything opens a second
        // connection: each connection to an in-memory database is its own
        // private database, so a pool built the normal way would hand out
        // several empty stores.
        let memory_read_conn = is_in_memory(path).then(|| write_conn.clone());

        let read_pool = match memory_read_conn {
            Some(shared) => vec![shared],
            None => open_read_pool(&db).await?,
        };

        Ok(Self {
            write: Mutex::new(write_conn),
            read_pool,
            next_reader: AtomicUsize::new(0),
        })
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        let conn = self.write.lock().await;
        conn.execute(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)
    }

    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let rows = self
            .reader()
            .query(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)?;
        read_rows(rows).await
    }

    async fn execute_batch(&self, sql: &str) -> Result<()> {
        let conn = self.write.lock().await;
        conn.execute_batch(sql).await.map(|_| ()).map_err(err)
    }

    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()> {
        let conn = self.write.lock().await;
        let tx = conn.transaction().await.map_err(err)?;
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
        let conn = self.write.lock().await;
        conn.execute(
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
        let conn = self.write.lock().await;
        conn.execute(
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
            .reader()
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(err)?;
        let rows = read_rows(rows).await?;
        rows.iter()
            .map(|r| Ok(NodeId::from_raw(r.get_string(0)?)))
            .collect()
    }

    async fn vector_sweep_orphans(&self) -> Result<u64> {
        let conn = self.write.lock().await;
        conn.execute(
            "DELETE FROM node_vectors WHERE node_id NOT IN (SELECT id FROM nodes)",
            libsql::params_from_iter(Vec::<libsql::Value>::new()),
        )
        .await
        .map_err(err)
    }
}

impl LibsqlBackend {
    /// Picks the next read connection in round robin. For a file-backed
    /// database this spreads reads across `READ_POOL_SIZE` independent
    /// connections, none of which is `write`, so reads never queue behind a
    /// write. For `:memory:` it always returns the single shared connection
    /// `open` built.
    fn reader(&self) -> &Connection {
        let idx = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.read_pool.len();
        &self.read_pool[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn is_in_memory_matches_every_in_memory_spelling_and_rejects_file_paths() {
        // Plain spelling, used throughout this codebase.
        assert!(is_in_memory(":memory:"));
        // `file:` URI whose file part is `:memory:`, with and without a
        // trailing query string.
        assert!(is_in_memory("file::memory:"));
        assert!(is_in_memory("file::memory:?cache=shared"));
        // `file:` URI naming a database but requesting `mode=memory`,
        // regardless of where that parameter falls among others.
        assert!(is_in_memory("file:memdb1?mode=memory&cache=shared"));
        assert!(is_in_memory("file:memdb1?cache=shared&mode=memory"));
        // Ordinary file paths, bare or as a `file:` URI, are not in-memory.
        assert!(!is_in_memory("liam.db"));
        assert!(!is_in_memory("file:liam.db"));
        assert!(!is_in_memory("file:liam.db?mode=rwc"));
    }

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
    async fn a_read_pool_connection_has_the_configured_busy_timeout() {
        // Arrange
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("busy.db");
        let backend = LibsqlBackend::open(path.to_str().expect("utf8 path"))
            .await
            .expect("open file-backed backend");

        // Act: query the pragma on an actual read pool connection, the one
        // `query` and `vector_search` hand every read to, rather than a
        // connection this test builds and configures itself. That pins
        // `open_read_pool` calling `configure_connection`, not merely that
        // the helper works in isolation.
        let rows = read_rows(
            backend.read_pool[0]
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

    async fn file_backend_at(name: &str) -> (TempDir, LibsqlBackend) {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join(name);
        let backend = LibsqlBackend::open(path.to_str().expect("utf8 path"))
            .await
            .expect("open file-backed backend");
        (dir, backend)
    }

    /// The read-during-write test must prove OVERLAP, not merely that the
    /// read finished: a read queued behind a single shared connection would
    /// also "finish" eventually, and that would pin nothing. So the test
    /// takes the exact same lock `execute` would, on the same task, and
    /// keeps the guard alive across the read. If `query` routed through
    /// `write` too, awaiting it here (same task, so the guard can never be
    /// released) would hang forever; the timeout turns that failure mode
    /// into a clean test failure instead of a stuck CI job.
    #[tokio::test]
    async fn a_read_completes_while_the_caller_holds_the_write_lock() {
        // Arrange
        let (_dir, backend) = file_backend_at("overlap.db").await;

        // Act: hold the write mutex ourselves and run a read while it is
        // still held.
        let guard = backend.write.lock().await;
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            backend.query("SELECT 1", &[]),
        )
        .await;

        // Assert: the read completed, and the guard is still in scope right
        // here, proving the read did not wait on it.
        let rows = result
            .expect("read did not complete while the write lock was held")
            .expect("query succeeded");
        assert_eq!(rows[0].get_i64(0).unwrap(), 1);
        drop(guard);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_writes_on_a_file_database_all_land_with_no_busy_error() {
        // Arrange
        let (_dir, backend) = file_backend_at("concurrent.db").await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");
        let backend = std::sync::Arc::new(backend);
        const WRITES: i64 = 20;

        // Act: fire N writes concurrently from separate tasks on separate
        // threads.
        let handles: Vec<_> = (0..WRITES)
            .map(|i| {
                let backend = backend.clone();
                tokio::spawn(async move {
                    backend
                        .execute("INSERT INTO t (id) VALUES (?1)", &[Value::Int(i)])
                        .await
                })
            })
            .collect();

        // Assert: none surfaced an error (SQLITE_BUSY or otherwise), and
        // every row landed.
        for handle in handles {
            handle
                .await
                .expect("write task did not panic")
                .expect("write succeeded");
        }
        let rows = backend
            .query("SELECT COUNT(*) FROM t", &[])
            .await
            .expect("count rows");
        assert_eq!(rows[0].get_i64(0).unwrap(), WRITES);
    }

    /// Pins the `:memory:` guard: without it, this write and this read would
    /// land on two SEPARATE, private in-memory databases (a fresh
    /// `db.connect()` per pool slot each opens its own empty store), and the
    /// read would come back empty.
    #[tokio::test]
    async fn a_memory_backed_write_is_visible_to_a_subsequent_read() {
        // Arrange
        let backend = LibsqlBackend::open(":memory:")
            .await
            .expect("open in-memory backend");
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");

        // Act
        backend
            .execute("INSERT INTO t (id) VALUES (?1)", &[Value::Int(1)])
            .await
            .expect("insert");
        let rows = backend.query("SELECT id FROM t", &[]).await.expect("query");

        // Assert
        assert_eq!(
            backend.read_pool.len(),
            1,
            "the :memory: pool must be size 1"
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get_i64(0).unwrap(), 1);
    }
}
