// SPDX-License-Identifier: Apache-2.0
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
use tokio::sync::{Mutex, MutexGuard};

use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{Millis, NodeId};
use crate::value::{Row, Value};

/// Milliseconds a connection waits on `SQLITE_BUSY` before giving up.
/// Config plumbing for this arrives in WU-5; until then a named constant is
/// correct, not a parameter nobody can set yet.
const BUSY_TIMEOUT_MS: i64 = 5000;

/// The smallest read pool `open` will ever build. `reader()` computes `%
/// self.read_pool.len()`, an integer division whose divisor must never be
/// zero, so a configured `read_pool_size` of 0 is floored to this instead
/// of producing an empty pool.
const MIN_READ_POOL_SIZE: usize = 1;

/// Whether `path` can safely back a multi-connection read pool: true only
/// for a plain filesystem path, one that does not start with `file:` and
/// is not the bare `:memory:` spelling. `database_path` comes from user
/// config (`liam.toml`), and libSQL accepts several in-memory spellings
/// through a `file:` URI: `file::memory:`, a `mode=memory` query
/// parameter, `vfs=memdb`, and possibly others this crate does not know
/// about. Each connection to an in-memory database is its own private,
/// empty database, so pooling one of these paths would hand every read an
/// empty store while every write lands somewhere no read ever looks.
///
/// Rather than enumerate every in-memory spelling and risk missing the
/// next one (which is exactly how this predicate's predecessor broke on
/// `vfs=memdb`), this inverts the check: every `file:` URI is treated as
/// unsafe to pool, full stop, because a `file:` URI can carry `vfs=`,
/// `mode=`, `cache=`, and other query parameters that change sharing
/// semantics in ways a string match cannot reliably decide. That includes
/// a `file:` URI naming an ordinary on-disk file, which could safely pool
/// but falls back to a single connection anyway. Getting that case wrong
/// costs a little read concurrency for an exotic-ish path; an
/// unrecognised spelling must cost performance, never correctness.
fn can_pool_reads(path: &str) -> bool {
    path != ":memory:" && !path.starts_with("file:")
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
    /// For any path `can_pool_reads` does not deem safe to pool (every
    /// in-memory spelling, and, conservatively, every `file:` URI) this
    /// holds exactly one entry: a CLONE of the write connection (see
    /// `open`), never a second `db.connect()`. Each connection to an
    /// in-memory database is its own private database, so a second
    /// `connect()` would silently hand reads an empty store; cloning the
    /// `Connection` handle (cheap: it wraps an `Arc`) reuses the exact same
    /// underlying database instead.
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

/// Opens `read_pool_size` fresh connections against `db`, each configured
/// with the same per-connection pragmas as every other connection this
/// backend hands out. Only called when `can_pool_reads` judges `path` safe
/// to pool: `open`'s single-connection branch never reaches this, because
/// a second `db.connect()` against an in-memory database, or any path this
/// crate cannot be sure is not one, would open an unrelated, empty
/// database rather than another handle onto the same one.
///
/// Callers must pass a `read_pool_size` of at least 1: `reader()` picks a
/// connection with `% self.read_pool.len()`, which panics on an empty pool.
/// `open` enforces that floor before calling this, so it is not repeated
/// here.
async fn open_read_pool(db: &Database, read_pool_size: usize) -> Result<Vec<Connection>> {
    let mut pool = Vec::with_capacity(read_pool_size);
    for _ in 0..read_pool_size {
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
    async fn open(path: &str, read_pool_size: usize) -> Result<Self> {
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

        // Guard every path that is not unambiguously poolable, before
        // anything opens a second connection: each connection to an
        // in-memory database is its own private database, so a pool built
        // the normal way would hand out several empty stores for any path
        // this crate fails to recognise as such.
        let single_read_conn = (!can_pool_reads(path)).then(|| write_conn.clone());

        let read_pool = match single_read_conn {
            Some(shared) => vec![shared],
            // Floored here, the only place a pool is actually built: see
            // `MIN_READ_POOL_SIZE` for why.
            None => open_read_pool(&db, read_pool_size.max(MIN_READ_POOL_SIZE)).await?,
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

    async fn begin(&self) -> Result<Box<dyn BackendTx + '_>> {
        let guard = self.write.lock().await;
        let transaction = guard.transaction().await.map_err(err)?;
        Ok(Box::new(LibsqlTx { transaction, guard }))
    }
}

/// An open transaction on `LibsqlBackend`'s write connection.
///
/// Field order matters for Drop safety and MUST stay `transaction` before
/// `guard`: Rust drops struct fields in declaration order. An implicit drop
/// (an early `?` return from a caller, or a bug that never calls
/// `commit`/`rollback`) must run the transaction's own rollback-on-drop
/// BEFORE the write lock releases, so no other writer can touch the
/// connection while that rollback is still in flight. Declaring `guard`
/// first would let the lock release while the transaction is still rolling
/// back underneath it.
struct LibsqlTx<'a> {
    transaction: libsql::Transaction,
    // Never read directly: held purely so the write lock stays taken for
    // this transaction's whole lifetime and releases, via `Drop`, exactly
    // when `transaction` (declared above, so it drops first) is gone.
    #[allow(dead_code)]
    guard: MutexGuard<'a, Connection>,
}

#[async_trait]
impl<'a> BackendTx for LibsqlTx<'a> {
    async fn execute(&mut self, sql: &str, params: &[Value]) -> Result<u64> {
        // `Transaction` derefs to `Connection`, so this is the same call
        // `LibsqlBackend::execute` makes, just against the transaction's
        // connection instead of the shared write connection directly.
        self.transaction
            .execute(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)
    }

    async fn query(&mut self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let rows = self
            .transaction
            .query(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)?;
        read_rows(rows).await
    }

    async fn commit(self: Box<Self>) -> Result<()> {
        // `self` (and with it `self.guard`) drops when this returns,
        // releasing the write lock exactly once, on completion.
        self.transaction.commit().await.map_err(err)
    }

    async fn rollback(self: Box<Self>) -> Result<()> {
        self.transaction.rollback().await.map_err(err)
    }
}

impl LibsqlBackend {
    /// Picks the next read connection in round robin. For a path
    /// `can_pool_reads` judges safe this spreads reads across the
    /// configured number of independent connections, none of which is
    /// `write`, so reads never queue behind a write. Otherwise it always
    /// returns the single shared connection `open` built.
    fn reader(&self) -> &Connection {
        let idx = self.next_reader.fetch_add(1, Ordering::Relaxed) % self.read_pool.len();
        &self.read_pool[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Pool size for tests that only need a working backend and do not care
    /// how many read connections it holds; the dedicated pool-size tests
    /// below (`a_read_pool_size_of_zero_is_floored_to_one_connection`,
    /// `a_configured_read_pool_size_is_honoured_for_a_file_backed_database`)
    /// use their own explicit, meaningful values instead.
    const ARBITRARY_POOL_SIZE: usize = 4;

    #[test]
    fn can_pool_reads_is_true_only_for_a_plain_filesystem_path() {
        // Plain `:memory:`, used throughout this codebase.
        assert!(!can_pool_reads(":memory:"));
        // `file:` URI whose file part is `:memory:`, with and without a
        // trailing query string.
        assert!(!can_pool_reads("file::memory:"));
        assert!(!can_pool_reads("file::memory:?cache=shared"));
        // `file:` URI naming a database but requesting `mode=memory`.
        assert!(!can_pool_reads("file:x?mode=memory"));
        // `vfs=memdb`: libSQL's other in-memory spelling, the one the old
        // string-matching `is_in_memory` predicate missed, which is the
        // exact bug this predicate exists to make impossible to repeat.
        assert!(!can_pool_reads("file:x?vfs=memdb"));
        assert!(!can_pool_reads("file:x?cache=shared&vfs=memdb"));

        // Plain filesystem paths, relative or absolute, are the only case
        // unambiguous enough to pool.
        assert!(can_pool_reads("liam.db"));
        assert!(can_pool_reads("/var/lib/liam/liam.db"));

        // A `file:` URI naming an ordinary on-disk file also takes the
        // single-connection path, even though it could safely pool. This
        // is deliberate, safe conservatism, not a bug: a `file:` URI can
        // carry query parameters this predicate does not try to parse, so
        // every `file:` URI is treated as unsafe to pool rather than
        // guessing at which parameters matter.
        assert!(!can_pool_reads("file:liam.db?mode=rwc"));
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
        let backend = LibsqlBackend::open(path.to_str().expect("utf8 path"), ARBITRARY_POOL_SIZE)
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
        let backend = LibsqlBackend::open(path.to_str().expect("utf8 path"), ARBITRARY_POOL_SIZE)
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
        let backend = LibsqlBackend::open(":memory:", ARBITRARY_POOL_SIZE)
            .await
            .expect("open in-memory backend");
        let rows = backend.query("SELECT 1", &[]).await.expect("query");

        // Assert: opening and querying succeed. `journal_mode` is
        // deliberately not checked here: WAL does not apply to in-memory
        // databases, so that assertion would be vacuous or wrong.
        assert_eq!(rows[0].get_i64(0).unwrap(), 1);
    }

    async fn file_backend_at(name: &str, read_pool_size: usize) -> (TempDir, LibsqlBackend) {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join(name);
        let backend = LibsqlBackend::open(path.to_str().expect("utf8 path"), read_pool_size)
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
        let (_dir, backend) = file_backend_at("overlap.db", ARBITRARY_POOL_SIZE).await;

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
        let (_dir, backend) = file_backend_at("concurrent.db", ARBITRARY_POOL_SIZE).await;
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
        let backend = LibsqlBackend::open(":memory:", ARBITRARY_POOL_SIZE)
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

    /// Regression test for a bug caught in review: the old `is_in_memory`
    /// predicate matched `:memory:` and `mode=memory` but missed
    /// `vfs=memdb`, libSQL's other in-memory spelling. `database_path`
    /// comes from user config, so `file:x?vfs=memdb` is a legitimate
    /// `liam.toml` value, and with the old predicate it fell through to
    /// `open_read_pool`, which opened four separate, empty memdb
    /// databases; every read then missed every write. Confirmed
    /// empirically before the fix: opening this same URI through the full
    /// `Graph` API and querying after an insert failed with `no such
    /// table: nodes_fts`, because the pooled read connection never saw the
    /// schema the write connection created. `can_pool_reads` closes this
    /// by treating every `file:` URI as unsafe to pool, so this must keep
    /// falling back to a single shared connection no matter what other
    /// in-memory spelling libSQL adds in the future.
    #[tokio::test]
    async fn a_vfs_memdb_backed_write_is_visible_to_a_subsequent_read() {
        // Arrange
        let backend =
            LibsqlBackend::open("file:vfs_memdb_regression?vfs=memdb", ARBITRARY_POOL_SIZE)
                .await
                .expect("open vfs=memdb backend");
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
            "the vfs=memdb pool must fall back to a single shared connection"
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get_i64(0).unwrap(), 1);
    }

    /// A configured `read_pool_size` of 0 must not produce an empty pool:
    /// `reader()` computes `% self.read_pool.len()`, which panics on divide
    /// by zero the moment any query runs. `open` floors the size at 1, so
    /// this proves the floor rather than merely that `open` did not panic.
    #[tokio::test]
    async fn a_read_pool_size_of_zero_is_floored_to_one_connection() {
        // Given a configured read_pool_size of 0
        let (_dir, backend) = file_backend_at("zero_pool.db", 0).await;

        // Then the pool has at least one connection
        assert_eq!(backend.read_pool.len(), 1);

        // When a read runs (the operation that divides by the pool length)
        let rows = backend.query("SELECT 1", &[]).await;

        // Then it succeeds rather than panicking
        assert_eq!(rows.expect("query must succeed")[0].get_i64(0).unwrap(), 1);
    }

    /// Pins the plumbing this Work Unit exists for: a configured
    /// `read_pool_size` must reach `LibsqlBackend::open` and actually size
    /// the pool, not just parse in `liam.toml` and go nowhere. 7 is
    /// deliberately not the old hardcoded default (4), so this cannot pass
    /// by coincidence.
    #[tokio::test]
    async fn a_configured_read_pool_size_is_honoured_for_a_file_backed_database() {
        // Given a configured read_pool_size of 7
        let (_dir, backend) = file_backend_at("seven_pool.db", 7).await;

        // Then the pool actually has that many connections
        assert_eq!(backend.read_pool.len(), 7);
    }

    #[tokio::test]
    async fn begin_then_write_then_read_sees_own_write_then_commit_then_visible_outside() {
        // Arrange
        let (_dir, backend) = file_backend_at("tx_commit.db", ARBITRARY_POOL_SIZE).await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");

        // Act: write and, still inside the same open tx, read it back
        // (proves read-your-own-write), then commit.
        let mut tx = backend.begin().await.expect("begin transaction");
        tx.execute("INSERT INTO t (id) VALUES (?1)", &[Value::Int(1)])
            .await
            .expect("insert inside tx");
        let seen_inside = tx
            .query("SELECT id FROM t", &[])
            .await
            .expect("query inside tx");
        assert_eq!(
            seen_inside.len(),
            1,
            "the write must be visible inside its own still-open tx"
        );
        tx.commit().await.expect("commit");

        // Assert: visible through the ordinary Backend::query too, now that
        // the tx has committed.
        let seen_outside = backend
            .query("SELECT id FROM t", &[])
            .await
            .expect("query outside tx");
        assert_eq!(seen_outside.len(), 1);
        assert_eq!(seen_outside[0].get_i64(0).unwrap(), 1);
    }

    #[tokio::test]
    async fn rollback_leaves_no_trace() {
        // Arrange
        let (_dir, backend) = file_backend_at("tx_rollback.db", ARBITRARY_POOL_SIZE).await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");

        // Act
        let mut tx = backend.begin().await.expect("begin transaction");
        tx.execute("INSERT INTO t (id) VALUES (?1)", &[Value::Int(1)])
            .await
            .expect("insert inside tx");
        tx.rollback().await.expect("rollback");

        // Assert
        let rows = backend.query("SELECT id FROM t", &[]).await.expect("query");
        assert_eq!(rows.len(), 0);
    }

    /// Distinct from `rollback_leaves_no_trace`: that test calls `rollback`
    /// explicitly and would pass even if `Drop` did nothing at all. This one
    /// never calls `commit` or `rollback`, so it is the only test that
    /// actually exercises the `Drop`-triggered rollback path.
    #[tokio::test]
    async fn dropping_without_commit_leaves_no_trace() {
        // Arrange
        let (_dir, backend) = file_backend_at("tx_drop.db", ARBITRARY_POOL_SIZE).await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");

        // Act: `tx` goes out of scope here with no call to `commit` or
        // `rollback`.
        {
            let mut tx = backend.begin().await.expect("begin transaction");
            tx.execute("INSERT INTO t (id) VALUES (?1)", &[Value::Int(1)])
                .await
                .expect("insert inside tx");
        }

        // Assert
        let rows = backend.query("SELECT id FROM t", &[]).await.expect("query");
        assert_eq!(rows.len(), 0);
    }

    /// Proves `BackendTx` holds the write lock for its whole open lifetime,
    /// not just per statement (the coarser contract `begin`'s doc comment on
    /// `Backend` describes). No sleeps or timing polls: a `Notify` orders
    /// "the spawned task has started and is about to contend for the write
    /// lock" strictly before "the main task checks whether it finished", so
    /// the `is_finished` check right after can't race the spawn itself.
    #[tokio::test]
    async fn begin_serializes_with_other_writes() {
        // Arrange
        let (_dir, backend) = file_backend_at("tx_serializes.db", ARBITRARY_POOL_SIZE).await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");
        let backend = std::sync::Arc::new(backend);
        let notify = std::sync::Arc::new(tokio::sync::Notify::new());

        // Act: open a tx and write on it, without committing yet, so the
        // write lock stays held.
        let mut tx = backend.begin().await.expect("begin transaction");
        tx.execute("INSERT INTO t (id) VALUES (?1)", &[Value::Int(1)])
            .await
            .expect("insert inside tx");

        // Spawn a second task that announces (before anything else, in
        // particular before any `.await`) that it is about to contend for
        // the write lock, then immediately awaits an ordinary write through
        // the same backend.
        let spawned_backend = backend.clone();
        let spawned_notify = notify.clone();
        let handle = tokio::spawn(async move {
            spawned_notify.notify_one();
            spawned_backend
                .execute("INSERT INTO t (id) VALUES (?1)", &[Value::Int(2)])
                .await
        });

        // Assert: once the spawned task has started, it must still be
        // blocked on the write lock, since the main task's open tx still
        // holds it.
        notify.notified().await;
        assert!(
            !handle.is_finished(),
            "the spawned write must still be blocked while the tx is open"
        );

        // Act: release the write lock by committing.
        tx.commit().await.expect("commit");

        // Assert: the spawned write completes now that the lock is free.
        handle
            .await
            .expect("spawned task did not panic")
            .expect("spawned write succeeded");
    }
}
