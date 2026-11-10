// SPDX-License-Identifier: Apache-2.0
//! libSQL backend. Native vector search via `F32_BLOB` and `vector_distance_cos`,
//! with embeddings in a `node_vectors` table so the search can prefilter against
//! the live node set in one query.
//!
//! VERSION CHECK: the libSQL parameter-binding call (`params_from_iter`) and the
//! row accessors (`column_count`, `get_value`) are the surface to confirm against
//! the version you pin.

use std::sync::{Arc, Mutex as StdMutex, MutexGuard as StdMutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use libsql::{Builder, Connection, Database};
use tokio::runtime::Handle;
use tokio::sync::{Mutex, MutexGuard, OwnedSemaphorePermit, Semaphore};

use crate::backend::{Backend, BackendTx, CheckpointMode, CheckpointResult};
use crate::error::{Error, Result};
use crate::graph::scope_within;
use crate::ids::{Millis, NodeId, FOREVER};
use crate::types::Derivation;
use crate::value::{Row, Value};

mod read_hooks;
#[cfg(test)]
use read_hooks::ReadEvent;
use read_hooks::ReadHooks;

/// Milliseconds a connection waits on `SQLITE_BUSY` before giving up.
/// Config plumbing for this arrives in WU-5; until then a named constant is
/// correct, not a parameter nobody can set yet.
const BUSY_TIMEOUT_MS: i64 = 5000;

/// The smallest read pool `open` will ever build. `reader()` computes `%
/// self.read_pool.len()`, an integer division whose divisor must never be
/// zero, so a configured `read_pool_size` of 0 is floored to this instead
/// of producing an empty pool.
const MIN_READ_POOL_SIZE: usize = 1;

/// The vector table has no cascade from `nodes`, so a node's vector is deleted
/// on its own, also by a removal that runs inside a caller's transaction.
const VECTOR_DELETE_SQL: &str = "DELETE FROM node_vectors WHERE node_id = ?1";

const VECTOR_CLEAR_SQL: &str = "DELETE FROM node_vectors";

/// libSQL fixes the `F32_BLOB` width when the table is created and accepts a
/// vector of any width on insert, so a changed width means a new table.
const VECTOR_DROP_SQL: &str = "DROP TABLE IF EXISTS node_vectors";

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

/// Locks a std mutex, recovering the data if a panicking read poisoned it.
fn lock<T>(mutex: &StdMutex<T>) -> StdMutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Read connections, one semaphore permit per connection: a read waits for
/// a permit, then takes an idle connection, so a slow read never holds up a
/// read on another connection.
struct ReadPool {
    #[cfg(test)]
    size: usize,
    permits: Arc<Semaphore>,
    idle: Arc<StdMutex<Vec<Connection>>>,
    /// False when the connection is shared with the write path, where an
    /// interrupt would abort a write.
    interruptible: bool,
}

impl ReadPool {
    fn new(connections: Vec<Connection>, interruptible: bool) -> Self {
        let size = connections.len();
        Self {
            #[cfg(test)]
            size,
            permits: Arc::new(Semaphore::new(size)),
            idle: Arc::new(StdMutex::new(connections)),
            interruptible,
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.size
    }

    #[cfg(test)]
    fn idle_len(&self) -> usize {
        lock(&self.idle).len()
    }

    async fn checkout(&self) -> Result<ReadLease> {
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::Backend("the read pool is closed".to_owned()))?;
        let conn = lock(&self.idle)
            .pop()
            .expect("a free permit always has an idle connection behind it");
        Ok(ReadLease {
            conn: Some(conn),
            permit: Some(permit),
            state: Arc::default(),
            idle: self.idle.clone(),
            interruptible: self.interruptible,
        })
    }
}

#[derive(Default)]
struct ReadState {
    /// The lease is over: the connection is back, or about to be.
    finished: bool,
    /// The read was dropped before it finished.
    cancelled: bool,
}

/// A checked out read connection. Dropping it, on return, error or panic,
/// marks the read finished, puts the connection back, then frees the permit.
struct ReadLease {
    conn: Option<Connection>,
    permit: Option<OwnedSemaphorePermit>,
    state: Arc<StdMutex<ReadState>>,
    idle: Arc<StdMutex<Vec<Connection>>>,
    interruptible: bool,
}

impl ReadLease {
    fn conn(&self) -> &Connection {
        self.conn
            .as_ref()
            .expect("a live lease holds its connection")
    }

    fn cancel_guard(&self, hooks: ReadHooks) -> CancelGuard {
        CancelGuard {
            conn: self.conn().clone(),
            state: self.state.clone(),
            interruptible: self.interruptible,
            hooks,
        }
    }

    fn is_cancelled(&self) -> bool {
        lock(&self.state).cancelled
    }
}

impl Drop for ReadLease {
    fn drop(&mut self) {
        lock(&self.state).finished = true;
        if let Some(conn) = self.conn.take() {
            lock(&self.idle).push(conn);
        }
        self.permit.take();
    }
}

/// Held by the awaiting read. If it drops before the lease finished, the
/// read was cancelled: it stops the running statement.
struct CancelGuard {
    conn: Connection,
    state: Arc<StdMutex<ReadState>>,
    interruptible: bool,
    hooks: ReadHooks,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        // The interrupt runs under the lock so the lease cannot finish, and
        // hand the connection to another read, in between.
        let mut state = lock(&self.state);
        if state.finished {
            return;
        }
        state.cancelled = true;
        if self.interruptible {
            self.hooks.count_interrupt();
            // The only failure is a connection that is already closed.
            let _ = self.conn.interrupt();
        }
    }
}

/// The blocking half of a read: runs on a blocking thread and owns the lease
/// until the rows are collected.
fn read_blocking(
    lease: ReadLease,
    runtime: &Handle,
    sql: &str,
    params: Vec<libsql::Value>,
    hooks: &ReadHooks,
) -> Result<Vec<Row>> {
    hooks.at_start();
    if lease.is_cancelled() {
        hooks.skipped();
        return Err(Error::Backend(
            "the read was cancelled before it started".to_owned(),
        ));
    }
    assert!(!hooks.take_panic(), "read panic requested by a test hook");
    runtime.block_on(async {
        let rows = lease
            .conn()
            .query(sql, libsql::params_from_iter(params))
            .await
            .map_err(err);
        match rows {
            Ok(rows) => read_rows(rows, hooks).await,
            Err(e) => {
                let failed = Err(e);
                hooks.outcome(&failed);
                failed
            }
        }
    })
}

pub struct LibsqlBackend {
    /// The single write connection, taken by `execute`, `execute_batch`,
    /// `execute_atomic`, and every vector-writing method. Guarding it with a
    /// mutex serializes writes at the application level instead of letting
    /// them race on `SQLITE_BUSY` and hoping `busy_timeout` sorts it out.
    write: Mutex<Connection>,
    /// Connections used for reads (`query`, `vector_search`). Reads never
    /// take `write`'s lock, so a read completes even while a write is in
    /// flight; that is the entire reason this pool exists separately from
    /// `write`.
    ///
    /// For any path `can_pool_reads` does not deem safe to pool (every
    /// in-memory spelling, and, conservatively, every `file:` URI) this
    /// holds exactly one entry: a CLONE of the write connection (see
    /// `open`), never a second `db.connect()`. Each connection to an
    /// in-memory database is its own private database, so a second
    /// `connect()` would silently hand reads an empty store; cloning the
    /// `Connection` handle (cheap: it wraps an `Arc`) reuses the exact same
    /// underlying database instead.
    read_pool: ReadPool,
    #[cfg(test)]
    hooks: ReadHooks,
}

/// What `nodes_missing_vectors` trims from a node's content to decide whether
/// anything is left to embed: SQLite's own `TRIM` strips only spaces.
const CONTENT_WHITESPACE: &str = " \t\n\x0b\x0c\r";

/// The low byte of an extended result code is the primary code.
const PRIMARY_CODE_MASK: i32 = 0xff;

fn err(e: libsql::Error) -> Error {
    match e {
        libsql::Error::SqliteFailure(code, message)
            if code & PRIMARY_CODE_MASK == libsql::ffi::SQLITE_CONSTRAINT =>
        {
            Error::Constraint(message)
        }
        other => Error::Backend(other.to_string()),
    }
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

async fn read_rows(mut rows: libsql::Rows, hooks: &ReadHooks) -> Result<Vec<Row>> {
    let result = collect_rows(&mut rows, hooks).await;
    hooks.statement_ended();
    hooks.outcome(&result);
    result
}

async fn collect_rows(rows: &mut libsql::Rows, hooks: &ReadHooks) -> Result<Vec<Row>> {
    let cols = rows.column_count();
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(err)? {
        let mut values = Vec::with_capacity(cols as usize);
        for i in 0..cols {
            values.push(from_libsql(row.get_value(i).map_err(err)?));
        }
        out.push(Row(values));
        hooks.row_fetched(out.len())?;
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

/// The `dims` floats in `bytes`, little endian as `le_bytes` writes them. The
/// column accepts a vector of any width, so a blob of another length is an
/// error here rather than a vector of the wrong size.
fn decode_vector(bytes: &[u8], dims: usize) -> Result<Vec<f32>> {
    let (chunks, rest) = bytes.as_chunks::<{ std::mem::size_of::<f32>() }>();
    if !rest.is_empty() {
        return Err(Error::Backend(format!(
            "a stored vector of {} bytes is not whole f32 values",
            bytes.len()
        )));
    }
    if chunks.len() != dims {
        return Err(Error::Dimension {
            expected: dims,
            got: chunks.len(),
        });
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
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
        // empirically against libsql: issuing `PRAGMA journal_mode =
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
            Some(shared) => ReadPool::new(vec![shared], false),
            // Floored here, the only place a pool is actually built: see
            // `MIN_READ_POOL_SIZE` for why.
            None => ReadPool::new(
                open_read_pool(&db, read_pool_size.max(MIN_READ_POOL_SIZE)).await?,
                true,
            ),
        };

        Ok(Self {
            write: Mutex::new(write_conn),
            read_pool,
            #[cfg(test)]
            hooks: ReadHooks::none(),
        })
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        let conn = self.write.lock().await;
        conn.execute(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)
    }

    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        self.read(sql.to_owned(), bind(params)).await
    }

    async fn execute_batch(&self, sql: &str) -> Result<()> {
        let conn = self.write.lock().await;
        conn.execute_batch(sql).await.map(|_| ()).map_err(err)
    }

    async fn wal_checkpoint(
        &self,
        mode: CheckpointMode,
        wait: Duration,
    ) -> Result<CheckpointResult> {
        let conn = self.write.lock().await;
        conn.query(&format!("PRAGMA busy_timeout = {}", wait.as_millis()), ())
            .await
            .map_err(err)?;
        let checkpoint = conn.query(mode.pragma(), ()).await.map_err(err);
        // The wait is restored whether or not the checkpoint ran.
        let restored = conn
            .query(&format!("PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}"), ())
            .await
            .map_err(err);
        let rows = read_rows(checkpoint?, &ReadHooks::none()).await?;
        restored?;
        let row = rows
            .first()
            .ok_or_else(|| Error::Backend("wal_checkpoint returned no row".to_owned()))?;
        Ok(CheckpointResult {
            busy: row.get_i64(0)? != 0,
            log_frames: row.get_i64(1)?,
            checkpointed: row.get_i64(2)?,
        })
    }

    async fn set_secure_delete(&self, on: bool) -> Result<bool> {
        let conn = self.write.lock().await;
        let set = if on { "ON" } else { "OFF" };
        conn.query(&format!("PRAGMA secure_delete = {set}"), ())
            .await
            .map_err(err)?;
        let rows = conn.query("PRAGMA secure_delete", ()).await.map_err(err)?;
        let rows = read_rows(rows, &ReadHooks::none()).await?;
        let value = rows
            .first()
            .ok_or_else(|| Error::Backend("secure_delete returned no row".to_owned()))?;
        Ok(value.get_i64(0)? != 0)
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

    async fn vector_insert_if_absent(&self, node_id: &str, embedding: &[f32]) -> Result<bool> {
        let conn = self.write.lock().await;
        let inserted = conn
            .execute(
                "INSERT INTO node_vectors (node_id, embedding) VALUES (?1, ?2)
                 ON CONFLICT(node_id) DO NOTHING",
                libsql::params_from_iter(vec![
                    libsql::Value::Text(node_id.to_string()),
                    libsql::Value::Blob(le_bytes(embedding)),
                ]),
            )
            .await
            .map_err(err)?;
        Ok(inserted > 0)
    }

    fn vector_delete_sql(&self) -> Option<&'static str> {
        Some(VECTOR_DELETE_SQL)
    }

    fn vector_clear_sql(&self) -> Option<&'static str> {
        Some(VECTOR_CLEAR_SQL)
    }

    fn vector_reset_statements(&self, dims: usize) -> Vec<String> {
        vec![VECTOR_DROP_SQL.to_owned(), self.vector_ddl(dims)]
    }

    async fn vector_delete(&self, node_id: &str) -> Result<()> {
        let conn = self.write.lock().await;
        conn.execute(
            VECTOR_DELETE_SQL,
            libsql::params_from_iter(vec![libsql::Value::Text(node_id.to_string())]),
        )
        .await
        .map_err(err)?;
        Ok(())
    }

    async fn vector_get(&self, node_id: &str, dims: usize) -> Result<Option<Vec<f32>>> {
        let rows = self
            .query(
                "SELECT embedding FROM node_vectors WHERE node_id = ?1",
                &[node_id.into()],
            )
            .await?;
        rows.first()
            .map(|row| decode_vector(row.get_blob(0)?, dims))
            .transpose()
    }

    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        kind: Option<&str>,
        scope: Option<&str>,
        as_of: Millis,
        include_inferred: bool,
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
            filters.push_str(&format!(" AND {}", scope_within("n.scope", next)));
            params.push(libsql::Value::Text(scope.to_string()));
            next += 1;
        }
        if !include_inferred {
            filters.push_str(&format!(" AND n.derivation != ?{next}"));
            params.push(libsql::Value::Text(
                Derivation::Inferred.as_str().to_string(),
            ));
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
             ORDER BY vector_distance_cos(v.embedding, vector(?1)), n.rowid LIMIT ?3"
        );
        let rows = self.read(sql, params).await?;
        rows.iter()
            .map(|r| Ok(NodeId::from_raw(r.get_string(0)?)))
            .collect()
    }

    async fn nodes_missing_vectors(&self) -> Result<Vec<NodeId>> {
        let rows = self
            .query(
                "SELECT n.id FROM nodes n
                 LEFT JOIN node_vectors v ON v.node_id = n.id
                 WHERE n.tx_to = ?1 AND v.node_id IS NULL
                   AND TRIM(n.content, ?2) != ''
                 ORDER BY n.id",
                &[FOREVER.into(), CONTENT_WHITESPACE.into()],
            )
            .await?;
        rows.iter()
            .map(|row| Ok(NodeId::from_raw(row.get_string(0)?)))
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
        read_rows(rows, &ReadHooks::none()).await
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
    /// Runs one read on a pooled connection, on a blocking thread. If the
    /// caller drops this future, the guard interrupts the running statement
    /// and the connection goes back to the pool once the thread unwinds.
    async fn read(&self, sql: String, params: Vec<libsql::Value>) -> Result<Vec<Row>> {
        let lease = self.read_pool.checkout().await?;
        let hooks = self.read_hooks();
        let _guard = lease.cancel_guard(hooks.clone());
        let runtime = Handle::current();
        tokio::task::spawn_blocking(move || read_blocking(lease, &runtime, &sql, params, &hooks))
            .await
            .map_err(|e| Error::Backend(format!("the read task failed: {e}")))?
    }

    #[cfg(test)]
    fn read_hooks(&self) -> ReadHooks {
        self.hooks.clone()
    }

    #[cfg(not(test))]
    fn read_hooks(&self) -> ReadHooks {
        ReadHooks::none()
    }

    #[cfg(test)]
    fn idle_len(&self) -> usize {
        self.read_pool.idle_len()
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
    fn decode_vector_reads_back_what_le_bytes_wrote() {
        // Arrange
        let floats = [0.25, -1.5, f32::MIN_POSITIVE];

        // Act
        let decoded = decode_vector(&le_bytes(&floats), floats.len());

        // Assert
        assert_eq!(decoded.unwrap(), floats);
    }

    #[test]
    fn decode_vector_refuses_a_blob_that_does_not_match_the_dims() {
        for (bytes, dims, expected) in [
            (
                12,
                4,
                Error::Dimension {
                    expected: 4,
                    got: 3,
                },
            ),
            (
                12,
                2,
                Error::Dimension {
                    expected: 2,
                    got: 3,
                },
            ),
            (
                0,
                1,
                Error::Dimension {
                    expected: 1,
                    got: 0,
                },
            ),
        ] {
            // Act
            let decoded = decode_vector(&vec![0; bytes], dims);

            // Assert
            assert_eq!(decoded.unwrap_err().to_string(), expected.to_string());
        }
    }

    #[test]
    fn decode_vector_refuses_a_blob_that_is_not_whole_floats() {
        // Act
        let decoded = decode_vector(&[0; 13], 3);

        // Assert
        assert!(matches!(decoded, Err(Error::Backend(_))));
    }

    #[test]
    fn a_constraint_failure_is_told_apart_from_a_busy_or_io_failure() {
        // Arrange: the extended code of a foreign key violation keeps the
        // primary constraint code in its low byte.
        let foreign_key = libsql::ffi::SQLITE_CONSTRAINT | (3 << 8);
        let cases = [
            (libsql::ffi::SQLITE_CONSTRAINT, true),
            (foreign_key, true),
            (libsql::ffi::SQLITE_BUSY, false),
            (libsql::ffi::SQLITE_IOERR, false),
        ];

        for (code, is_constraint) in cases {
            // Act
            let mapped = err(libsql::Error::SqliteFailure(code, "failed".into()));

            // Assert
            assert_eq!(
                matches!(mapped, Error::Constraint(_)),
                is_constraint,
                "code {code}: {mapped:?}"
            );
        }
    }

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
        let lease = backend.read_pool.checkout().await.expect("check out");
        let rows = read_rows(
            lease
                .conn()
                .query("PRAGMA busy_timeout", ())
                .await
                .expect("query busy_timeout"),
            &ReadHooks::none(),
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

    /// Tests for the read lease, the blocking read and the cancel guard. A
    /// wall-clock limit appears only as a hang guard around an awaited
    /// result; every assertion is on rows, counts or reports.
    mod read_lease {
        use std::sync::mpsc::Sender;
        use std::sync::{Arc, Barrier};
        use std::time::Duration;

        use tokio::sync::mpsc::{error::TryRecvError, UnboundedReceiver};
        use tokio::task::JoinHandle;

        use super::*;

        const HANG_GUARD: Duration = Duration::from_secs(2);

        /// Emits rows without end. Only ever run behind the hold gate.
        const UNBOUNDED_SQL: &str =
            "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT n FROM c";
        const BOUNDED_ROWS: usize = 1000;
        const BOUNDED_SQL: &str =
            "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n < 1000) SELECT n FROM c";

        type ReadHandle = JoinHandle<Result<Vec<Row>>>;

        async fn guarded<T>(what: &str, fut: impl std::future::Future<Output = T>) -> T {
            tokio::time::timeout(HANG_GUARD, fut)
                .await
                .unwrap_or_else(|_| panic!("timed out waiting for {what}"))
        }

        async fn next_event(events: &mut UnboundedReceiver<ReadEvent>) -> ReadEvent {
            guarded("a read report", events.recv())
                .await
                .expect("the report channel stays open")
        }

        fn assert_no_report(events: &mut UnboundedReceiver<ReadEvent>) {
            assert_eq!(events.try_recv(), Err(TryRecvError::Empty));
        }

        fn spawn_read(backend: &Arc<LibsqlBackend>, sql: &'static str) -> ReadHandle {
            let backend = backend.clone();
            tokio::spawn(async move { backend.query(sql, &[]).await })
        }

        async fn read_one(backend: &Arc<LibsqlBackend>, sql: &'static str) -> Vec<Row> {
            guarded("a read", spawn_read(backend, sql))
                .await
                .expect("read task did not panic")
                .expect("read succeeded")
        }

        async fn wait_for_idle(backend: &LibsqlBackend, expected: usize) {
            guarded("the pool to refill", async {
                while backend.idle_len() != expected {
                    tokio::task::yield_now().await;
                }
            })
            .await;
        }

        /// Drops the read's future and reports whether it is gone. It does
        /// not panic, so the caller can free what the read still holds
        /// before it asserts.
        async fn drop_read(read: &mut ReadHandle) -> bool {
            read.abort();
            tokio::time::timeout(HANG_GUARD, async {
                while !read.is_finished() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok()
        }

        /// Takes the test's part in a two party barrier from a thread of its
        /// own, so a read that never reaches the barrier cannot hang the test.
        fn release(barrier: &Arc<Barrier>) {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
            });
        }

        /// A pooled store with one unbounded read parked after its first row.
        struct HeldRead {
            _dir: TempDir,
            backend: Arc<LibsqlBackend>,
            events: UnboundedReceiver<ReadEvent>,
            gate: Sender<()>,
            read: ReadHandle,
        }

        impl HeldRead {
            async fn start(pool_size: usize) -> Self {
                let (dir, mut backend) = file_backend_at("held.db", pool_size).await;
                let mut events = backend.hooks.events();
                let gate = backend.hooks.arm_hold();
                let backend = Arc::new(backend);
                let read = spawn_read(&backend, UNBOUNDED_SQL);
                assert_eq!(next_event(&mut events).await, ReadEvent::Started);
                Self {
                    _dir: dir,
                    backend,
                    events,
                    gate,
                    read,
                }
            }

            /// Drops the read's future, then lets the paused statement go on.
            async fn drop_then_release(&mut self) -> bool {
                let dropped = drop_read(&mut self.read).await;
                let _ = self.gate.send(());
                dropped
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_paused_read_does_not_hold_up_a_read_on_another_connection() {
            // Arrange
            let mut held = HeldRead::start(2).await;
            assert_no_report(&mut held.events);

            // Act
            let rows = read_one(&held.backend, "SELECT 2").await;

            // Assert
            assert_eq!(rows[0].get_i64(0).unwrap(), 2);
            // The only report so far is the second read's own.
            assert_eq!(next_event(&mut held.events).await, ReadEvent::Ran);
            assert_no_report(&mut held.events);
            held.drop_then_release().await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn dropping_the_read_future_interrupts_its_running_statement() {
            // Arrange
            let mut held = HeldRead::start(2).await;

            // Act
            let dropped = held.drop_then_release().await;

            // Assert
            assert_eq!(next_event(&mut held.events).await, ReadEvent::Interrupted);
            assert!(dropped, "the read future was not dropped");
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn dropping_the_read_future_returns_its_connection_to_the_pool() {
            // Arrange
            let mut held = HeldRead::start(2).await;

            // Act
            held.drop_then_release().await;

            // Assert
            wait_for_idle(&held.backend, 2).await;
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn every_connection_serves_a_read_after_an_interrupt() {
            // Arrange
            let mut held = HeldRead::start(2).await;
            held.drop_then_release().await;
            assert_eq!(next_event(&mut held.events).await, ReadEvent::Interrupted);
            wait_for_idle(&held.backend, 2).await;

            // Act
            let (first, second) = tokio::join!(
                read_one(&held.backend, "SELECT 1"),
                read_one(&held.backend, "SELECT 2"),
            );

            // Assert
            assert_eq!(first[0].get_i64(0).unwrap(), 1);
            assert_eq!(second[0].get_i64(0).unwrap(), 2);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_guard_dropped_after_a_finished_read_does_not_interrupt() {
            // Arrange
            let (_dir, backend) = file_backend_at("late.db", 1).await;
            let backend = Arc::new(backend);

            // Act
            let first = read_one(&backend, BOUNDED_SQL).await;
            wait_for_idle(&backend, 1).await;

            // Assert
            assert_eq!(first.len(), BOUNDED_ROWS);
            assert_eq!(backend.hooks.interrupt_count(), 0);
            let second = read_one(&backend, "SELECT 2").await;
            assert_eq!(second[0].get_i64(0).unwrap(), 2);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_guard_dropped_after_the_statement_ends_interrupts_once_and_leaks_nothing() {
            // Arrange
            let (_dir, mut backend) = file_backend_at("window.db", 1).await;
            let mut events = backend.hooks.events();
            let finish = backend.hooks.arm_finish_barrier();
            let backend = Arc::new(backend);
            let mut read = spawn_read(&backend, BOUNDED_SQL);
            assert_eq!(next_event(&mut events).await, ReadEvent::StatementEnded);

            // Act
            let dropped = drop_read(&mut read).await;
            let interrupts = backend.hooks.interrupt_count();
            release(&finish);

            // Assert
            assert!(dropped, "the read future was not dropped");
            assert_eq!(interrupts, 1);
            let second = read_one(&backend, "SELECT 2").await;
            assert_eq!(second[0].get_i64(0).unwrap(), 2);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn dropping_the_guard_while_the_statement_runs_interrupts_it_once() {
            // Arrange
            let mut held = HeldRead::start(2).await;

            // Act
            held.drop_then_release().await;

            // Assert
            assert_eq!(next_event(&mut held.events).await, ReadEvent::Interrupted);
            assert_eq!(held.backend.hooks.interrupt_count(), 1);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_read_dropped_before_it_starts_is_skipped_and_interrupts_once() {
            // Arrange
            let (_dir, mut backend) = file_backend_at("early.db", 1).await;
            let mut events = backend.hooks.events();
            let start = backend.hooks.arm_start_barrier();
            let backend = Arc::new(backend);
            let mut read = spawn_read(&backend, BOUNDED_SQL);
            assert_eq!(next_event(&mut events).await, ReadEvent::AtStart);

            // Act
            let dropped = drop_read(&mut read).await;
            release(&start);

            // Assert
            assert!(dropped, "the read future was not dropped");
            assert_eq!(next_event(&mut events).await, ReadEvent::Skipped);
            assert_eq!(backend.hooks.interrupt_count(), 1);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_panicking_read_returns_an_error_and_its_connection() {
            // Arrange
            let (_dir, backend) = file_backend_at("panic.db", 2).await;
            backend.hooks.arm_panic();
            let backend = Arc::new(backend);

            // Act
            let result = guarded("the read", spawn_read(&backend, "SELECT 1"))
                .await
                .expect("the panic does not escape the read");

            // Assert
            assert!(result.is_err(), "the panicking read returned rows");
            wait_for_idle(&backend, 2).await;
            let next = read_one(&backend, "SELECT 2").await;
            assert_eq!(next[0].get_i64(0).unwrap(), 2);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_guard_dropped_after_a_panicking_read_does_not_interrupt() {
            // Arrange
            let (_dir, backend) = file_backend_at("panic_guard.db", 1).await;
            backend.hooks.arm_panic();
            let backend = Arc::new(backend);

            // Act
            let result = guarded("the read", backend.query("SELECT 1", &[])).await;

            // Assert
            assert!(result.is_err(), "the panicking read returned rows");
            assert_eq!(backend.hooks.interrupt_count(), 0);
            let next = read_one(&backend, "SELECT 2").await;
            assert_eq!(next[0].get_i64(0).unwrap(), 2);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_read_waits_for_a_connection_when_every_permit_is_held() {
            // Arrange
            let mut held = HeldRead::start(1).await;
            let mut waiting = spawn_read(&held.backend, "SELECT 2");

            // Act
            let timed_out = tokio::time::timeout(Duration::from_millis(200), &mut waiting)
                .await
                .is_err();
            waiting.abort();
            held.drop_then_release().await;

            // Assert
            assert!(timed_out, "the read finished without a free connection");
            wait_for_idle(&held.backend, 1).await;
        }

        struct WindowRun {
            interrupts: usize,
            outcome: ReadEvent,
            rows_written: i64,
        }

        /// A bounded read whose statement has ended but whose lease has not
        /// dropped; its guard drops there, then a write runs.
        async fn drop_guard_in_the_statement_end_window(single_connection: bool) -> WindowRun {
            let dir = TempDir::new().expect("create temp dir");
            let file = dir.path().join("window_write.db");
            let file = file.to_str().expect("utf8 path");
            let path = if single_connection {
                format!("file:{file}")
            } else {
                file.to_owned()
            };
            let mut backend = LibsqlBackend::open(&path, 2).await.expect("open backend");
            backend
                .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
                .await
                .expect("create table");
            let mut events = backend.hooks.events();
            let finish = backend.hooks.arm_finish_barrier();
            let backend = Arc::new(backend);
            let mut read = spawn_read(&backend, BOUNDED_SQL);
            assert_eq!(next_event(&mut events).await, ReadEvent::StatementEnded);

            drop_read(&mut read).await;
            let interrupts = backend.hooks.interrupt_count();
            release(&finish);
            let outcome = next_event(&mut events).await;
            let writer = backend.clone();
            guarded(
                "the write",
                tokio::spawn(
                    async move { writer.execute("INSERT INTO t (id) VALUES (1)", &[]).await },
                ),
            )
            .await
            .expect("write task did not panic")
            .expect("write succeeded");
            let rows = read_one(&backend, "SELECT COUNT(*) FROM t").await;

            WindowRun {
                interrupts,
                outcome,
                rows_written: rows[0].get_i64(0).unwrap(),
            }
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_single_connection_store_never_interrupts_a_read() {
            // Arrange and Act
            let run = drop_guard_in_the_statement_end_window(true).await;

            // Assert
            assert_eq!(run.interrupts, 0);
            assert_eq!(run.outcome, ReadEvent::Ran);
            assert_eq!(run.rows_written, 1);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn a_pooled_store_interrupts_in_the_same_window_as_a_control() {
            // Arrange and Act
            let run = drop_guard_in_the_statement_end_window(false).await;

            // Assert
            assert_eq!(run.interrupts, 1);
            assert_eq!(run.rows_written, 1);
        }
    }
}
