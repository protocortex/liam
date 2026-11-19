// SPDX-License-Identifier: Apache-2.0
//! Tests of the graph. Shared helpers live here; the tests are split by topic
//! into the modules below.

use super::*;
use crate::clock::FixedClock;
use crate::types::{relation, RetentionPolicy};
use crate::DefaultGraph;
use tempfile::TempDir;

mod archive;
mod backfill;
mod backwards_clock;
mod catch_up;
mod cold_probe;
mod communities;
mod deletion_log;
mod derivation;
mod fact_arbitration;
mod fact_batch;
mod fact_log;
mod fact_read;
mod fingerprint_and_staleness;
mod fts5_and_lexical;
mod gc;
mod gc_log;
mod ingest_episode_core;
mod ingest_episode_scope_and_dedup;
mod lineage;
mod log_open;
mod log_write;
mod log_write_faults;
mod mentions_and_ranking;
mod node_events;
mod quant;
mod ranking_ties;
mod reads_by_id_and_time;
mod rebuild;
mod recompute_refresh;
mod reembed;
mod relate_and_handles;
mod repair_watermark;
mod run_marker;
mod same_as_bridge;
mod scrub;
mod sealed;
mod snapshot;
mod snapshot_meta;
mod soak;
mod stats;
mod summaries;
mod summary_log;
mod support;
mod swept_by;
mod tier_apply;
mod tier_log;
mod tier_race;
mod tier_verify;
mod tiers;
mod typed_retrieval;
mod upsert_and_scope;
mod vector_reads;

/// A graph whose clock starts at 1000, with a handle to move it.
async fn gc_graph() -> (DefaultGraph, Arc<FixedClock>) {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    (g, clock)
}

async fn graph_at(t: Millis) -> DefaultGraph {
    let clock = Arc::new(FixedClock::new(t));
    DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock)
        .await
        .unwrap()
}

/// A live edge row written behind the guards of `relate` and `link`, for a
/// test that needs a state those refuse to create.
async fn insert_edge_row(g: &DefaultGraph, src: &NodeId, dst: &NodeId, kind: &str) {
    g.backend
        .execute(
            "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
             VALUES (?1, ?2, ?3, ?4, '{}', ?5, ?6)",
            &[
                EdgeId::new().as_str().into(),
                src.as_str().into(),
                dst.as_str().into(),
                kind.into(),
                g.clock.now().into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
}

/// Runs `f`'s future to completion on a dedicated OS thread with its own
/// single-threaded runtime, waiting up to `timeout` from the calling
/// thread. Unlike `tokio::time::timeout`, whose deadline is only checked
/// when the wrapped future yields, this catches a hang that never yields
/// at all (a tight loop of synchronous DB calls), since the deadline is
/// enforced by a thread the hung code can never block.
fn run_with_real_timeout<T, F, Fut>(
    timeout: std::time::Duration,
    f: F,
) -> std::result::Result<T, std::sync::mpsc::RecvTimeoutError>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = T>,
{
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let _ = tx.send(rt.block_on(f()));
    });
    rx.recv_timeout(timeout)
}

/// `:memory:` cannot stand in for the rest of S1: WAL is a no-op on an
/// in-memory database, and every `:memory:` connection is its own private
/// database, so a connection pool over it would silently fan out to
/// separate empty stores. This opens the same graph on a real file so
/// later tests exercise real single-file database semantics. The
/// `TempDir` guard is returned alongside the graph, not dropped here,
/// because dropping it deletes the database file out from under a graph
/// that still holds it open.
async fn file_graph_at(t: Millis) -> (TempDir, DefaultGraph) {
    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("graph.db");
    let clock = Arc::new(FixedClock::new(t));
    let g = DefaultGraph::open_with_clock(
        path.to_str().expect("temp path is valid utf-8"),
        GraphConfig::new(8),
        clock,
    )
    .await
    .unwrap();
    (dir, g)
}

/// Two nodes joined by one edge, on a clock the test controls.
async fn seeded_pair(t0: Millis) -> (DefaultGraph, Arc<FixedClock>, NodeId, NodeId) {
    let clock = Arc::new(FixedClock::new(t0));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();
    g.relate(&a, &b, "mentions").await.unwrap();
    (g, clock, a, b)
}

/// An edge to insert the first time a query matching `trigger` returns, so
/// a test can land a write in the middle of a recompute.
static INTERPOSE: std::sync::Mutex<Option<(String, String, String)>> = std::sync::Mutex::new(None);

/// How many issued queries contained a watched substring. Set the
/// substring, run the call under test, read the count: proves a read is
/// ONE statement, which `INTERPOSE` (below) can't show.
static QUERY_MATCH_COUNT: std::sync::Mutex<(String, usize)> =
    std::sync::Mutex::new((String::new(), 0));

/// `INTERPOSE` and `QUERY_MATCH_COUNT` are process-wide state shared by
/// every test that touches them; `cargo test`'s default parallelism would
/// otherwise let two such tests race on the same counter. `tokio::sync::Mutex`
/// because the guard must stay held across `.await` for the whole test body.
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Delegates everything to the real backend, except that one query fires a
/// single injected write after it returns.
///
/// This exists for exactly one test: the guarantee that the stored
/// fingerprint is the one captured WITH the graph is a property about
/// concurrent writes, and no ordinary single-threaded test can observe it.
struct Interposing(crate::backends::DefaultBackend);

#[async_trait::async_trait]
impl Backend for Interposing {
    async fn open(path: &str, read_pool_size: usize) -> Result<Self> {
        Ok(Self(
            crate::backends::DefaultBackend::open(path, read_pool_size).await?,
        ))
    }
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let rows = self.0.query(sql, params).await?;
        {
            let mut watch = QUERY_MATCH_COUNT.lock().unwrap();
            if !watch.0.is_empty() && sql.contains(watch.0.as_str()) {
                watch.1 += 1;
            }
        }
        let fire = {
            let mut slot = INTERPOSE.lock().unwrap();
            match slot.as_ref() {
                Some((trigger, _, _)) if sql.contains(trigger.as_str()) => slot.take(),
                _ => None,
            }
        };
        if let Some((_, src, dst)) = fire {
            self.0
                .execute(
                    "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
                     VALUES (?1, ?2, ?3, 'mentions', '{}', ?4, ?5)",
                    &[
                        EdgeId::new().as_str().into(),
                        src.into(),
                        dst.into(),
                        Millis(9_000_000).into(),
                        FOREVER.into(),
                    ],
                )
                .await?;
        }
        Ok(rows)
    }
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        self.0.execute(sql, params).await
    }
    async fn execute_batch(&self, sql: &str) -> Result<()> {
        self.0.execute_batch(sql).await
    }
    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()> {
        self.0.execute_atomic(statements).await
    }
    fn vector_ddl(&self, dims: usize) -> String {
        self.0.vector_ddl(dims)
    }
    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()> {
        self.0.vector_upsert(node_id, embedding).await
    }
    async fn vector_insert_if_absent(&self, node_id: &str, embedding: &[f32]) -> Result<bool> {
        self.0.vector_insert_if_absent(node_id, embedding).await
    }
    fn vector_delete_sql(&self) -> Option<&'static str> {
        self.0.vector_delete_sql()
    }
    fn vector_clear_sql(&self) -> Option<&'static str> {
        self.0.vector_clear_sql()
    }
    async fn vector_delete(&self, node_id: &str) -> Result<()> {
        self.0.vector_delete(node_id).await
    }
    async fn vector_get(&self, node_id: &str, dims: usize) -> Result<Option<Vec<f32>>> {
        self.0.vector_get(node_id, dims).await
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
        self.0
            .vector_search(query, k, kind, scope, as_of, include_inferred)
            .await
    }
    async fn nodes_missing_vectors(&self) -> Result<Vec<NodeId>> {
        self.0.nodes_missing_vectors().await
    }
    async fn vector_sweep_orphans(&self) -> Result<u64> {
        self.0.vector_sweep_orphans().await
    }
    async fn wal_checkpoint(
        &self,
        mode: crate::backend::CheckpointMode,
        wait: std::time::Duration,
    ) -> Result<crate::backend::CheckpointResult> {
        self.0.wal_checkpoint(mode, wait).await
    }
    async fn set_secure_delete(&self, on: bool) -> Result<bool> {
        self.0.set_secure_delete(on).await
    }
    async fn begin(&self) -> Result<Box<dyn crate::backend::BackendTx + '_>> {
        self.0.begin().await
    }
}

/// Wraps a real `BackendTx`, failing exactly one targeted `execute` call
/// while delegating every other call straight through.
///
/// `Interposing` (above) wraps `Backend`, not `BackendTx`, and its own
/// `begin` just delegates straight through (WU-1), so it never sees a
/// transaction-scoped call; this is a separate, transaction-scoped fault
/// injector, purpose-built for `a_mid_transaction_db_failure_leaves_zero_rows`.
struct FailingTx<'a> {
    inner: Box<dyn crate::backend::BackendTx + 'a>,
    /// 0-based count of `execute` calls seen so far.
    execute_calls: usize,
    /// The `execute_calls` value (before incrementing) that fails.
    fail_on_execute: usize,
    fail_on_row: Option<String>,
    probe: &'a TxProbe,
}

#[async_trait::async_trait]
impl<'a> crate::backend::BackendTx for FailingTx<'a> {
    async fn execute(&mut self, sql: &str, params: &[Value]) -> Result<u64> {
        let call = self.execute_calls;
        self.execute_calls += 1;
        self.probe.executed.lock().unwrap().push(sql.to_string());
        let touches_row = self.fail_on_row.as_deref().is_some_and(|row| {
            params
                .iter()
                .any(|param| matches!(param, Value::Text(text) if text == row))
        });
        if call == self.fail_on_execute || touches_row {
            return Err(Error::Backend(
                "injected mid-transaction failure".to_string(),
            ));
        }
        self.inner.execute(sql, params).await
    }
    async fn query(&mut self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let rows = self.inner.query(sql, params).await;
        self.probe.after_query().await;
        rows
    }
    async fn commit(self: Box<Self>) -> Result<()> {
        self.probe.before_commit().await;
        if self
            .probe
            .fail_commit
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            // The failed commit still has to give back the write lock.
            self.inner.rollback().await?;
            return Err(Error::Backend("injected commit failure".to_string()));
        }
        self.inner.commit().await
    }
    async fn rollback(self: Box<Self>) -> Result<()> {
        self.inner.rollback().await
    }
}

/// Timing hooks for the transactions of a `FailingBackend`, for tests that
/// hold or observe a transaction rather than fail it.
#[derive(Default)]
struct TxProbe {
    begins: std::sync::atomic::AtomicUsize,
    delay_odd_begins: std::sync::atomic::AtomicBool,
    /// Delays the first `begin` of each pair, counting from `begins` = 0.
    delay_even_begins: std::sync::atomic::AtomicBool,
    hang_commit: std::sync::atomic::AtomicBool,
    fail_commit: std::sync::atomic::AtomicBool,
    /// A lock whose held-or-free state is recorded at every commit.
    watched: std::sync::Mutex<Option<SharedLog>>,
    commit_lock_held: std::sync::Mutex<Vec<bool>>,
    /// Every statement a transaction was asked to execute, the one that
    /// was made to fail included.
    executed: std::sync::Mutex<Vec<String>>,
    /// Holds the first transaction that reads, right after its first read.
    query_gate: std::sync::Mutex<Option<Arc<QueryGate>>>,
}

/// Stops one transaction between its first read and the statements that
/// follow, so a test can start a second writer while the first holds the
/// write lock, then let the first go.
struct QueryGate {
    armed: std::sync::atomic::AtomicBool,
    reached: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}

impl QueryGate {
    fn new() -> Self {
        Self {
            armed: std::sync::atomic::AtomicBool::new(true),
            reached: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        }
    }

    /// Returns once a transaction is held at the gate.
    async fn wait_until_held(&self) {
        self.reached.acquire().await.unwrap().forget();
    }

    fn let_go(&self) {
        self.release.add_permits(1);
    }
}

impl TxProbe {
    async fn after_query(&self) {
        let gate = self.query_gate.lock().unwrap().clone();
        if let Some(gate) = gate {
            if gate.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
                gate.reached.add_permits(1);
                gate.release.acquire().await.unwrap().forget();
            }
        }
    }

    async fn before_begin(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        let nth = self.begins.fetch_add(1, SeqCst);
        let slow = match nth % 2 {
            0 => &self.delay_even_begins,
            _ => &self.delay_odd_begins,
        };
        if slow.load(SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    async fn before_commit(&self) {
        if self.hang_commit.load(std::sync::atomic::Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        let watched = self.watched.lock().unwrap().clone();
        if let Some(log) = watched {
            let held = log.try_lock().is_err();
            self.commit_lock_held.lock().unwrap().push(held);
        }
    }
}

/// Wraps a real `Backend`; its own `begin` returns a `FailingTx`, unlike
/// `Interposing::begin`'s plain pass-through. `fail_on_execute` defaults
/// to "never" (`usize::MAX`), so a test that needs a fault sets it
/// explicitly after `open` and any other test sees ordinary behavior.
struct FailingBackend {
    inner: crate::backends::DefaultBackend,
    fail_on_execute: std::sync::atomic::AtomicUsize,
    /// Fails the first statement of a transaction that binds this row id,
    /// whatever its position in the transaction.
    fail_on_row: std::sync::Mutex<Option<String>>,
    /// The `vector_delete` call, counted from `set_fail_on_vector_delete`,
    /// that fails.
    fail_on_vector_delete: std::sync::atomic::AtomicUsize,
    vector_deletes: std::sync::atomic::AtomicUsize,
    /// Fails every plain `query`, `execute`, `execute_batch` and
    /// `wal_checkpoint` whose SQL starts with this text, and counts them.
    fail_on_sql: std::sync::Mutex<Option<String>>,
    sql_failures: std::sync::atomic::AtomicUsize,
    probe: TxProbe,
}

impl FailingBackend {
    /// Fails statements that start with `prefix` until it is set to `None`.
    fn set_fail_on_sql(&self, prefix: Option<&str>) {
        *self.fail_on_sql.lock().unwrap() = prefix.map(str::to_owned);
        self.sql_failures
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// How many statements the injector has failed.
    fn sql_failures(&self) -> usize {
        self.sql_failures.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn check_sql(&self, sql: &str) -> Result<()> {
        let armed = self.fail_on_sql.lock().unwrap();
        match armed.as_deref() {
            Some(prefix) if sql.starts_with(prefix) => {
                self.sql_failures
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err(Error::Backend("injected sql failure".to_string()))
            }
            _ => Ok(()),
        }
    }

    fn set_fail_on_execute(&self, n: usize) {
        self.fail_on_execute
            .store(n, std::sync::atomic::Ordering::SeqCst);
    }

    /// Fails the `n`th (1-based) `vector_delete` from now on.
    fn set_fail_on_vector_delete(&self, n: usize) {
        self.vector_deletes
            .store(0, std::sync::atomic::Ordering::SeqCst);
        self.fail_on_vector_delete
            .store(n, std::sync::atomic::Ordering::SeqCst);
    }

    fn set_fail_on_row(&self, row_id: Option<&str>) {
        *self.fail_on_row.lock().unwrap() = row_id.map(str::to_owned);
    }

    fn set_fail_commit(&self, fail: bool) {
        self.probe
            .fail_commit
            .store(fail, std::sync::atomic::Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl Backend for FailingBackend {
    async fn open(path: &str, read_pool_size: usize) -> Result<Self> {
        Ok(Self {
            inner: crate::backends::DefaultBackend::open(path, read_pool_size).await?,
            fail_on_execute: std::sync::atomic::AtomicUsize::new(usize::MAX),
            fail_on_row: std::sync::Mutex::new(None),
            fail_on_vector_delete: std::sync::atomic::AtomicUsize::new(usize::MAX),
            vector_deletes: std::sync::atomic::AtomicUsize::new(0),
            fail_on_sql: std::sync::Mutex::new(None),
            sql_failures: std::sync::atomic::AtomicUsize::new(0),
            probe: TxProbe::default(),
        })
    }
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        self.check_sql(sql)?;
        self.inner.query(sql, params).await
    }
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        self.check_sql(sql)?;
        self.inner.execute(sql, params).await
    }
    async fn execute_batch(&self, sql: &str) -> Result<()> {
        self.check_sql(sql)?;
        self.inner.execute_batch(sql).await
    }
    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()> {
        self.inner.execute_atomic(statements).await
    }
    fn vector_ddl(&self, dims: usize) -> String {
        self.inner.vector_ddl(dims)
    }
    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()> {
        self.inner.vector_upsert(node_id, embedding).await
    }
    async fn vector_insert_if_absent(&self, node_id: &str, embedding: &[f32]) -> Result<bool> {
        self.inner.vector_insert_if_absent(node_id, embedding).await
    }
    fn vector_delete_sql(&self) -> Option<&'static str> {
        self.inner.vector_delete_sql()
    }
    fn vector_clear_sql(&self) -> Option<&'static str> {
        self.inner.vector_clear_sql()
    }
    async fn vector_delete(&self, node_id: &str) -> Result<()> {
        let call = self
            .vector_deletes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if call
            == self
                .fail_on_vector_delete
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(Error::Backend("injected vector_delete failure".to_string()));
        }
        self.inner.vector_delete(node_id).await
    }
    async fn vector_get(&self, node_id: &str, dims: usize) -> Result<Option<Vec<f32>>> {
        self.inner.vector_get(node_id, dims).await
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
        self.inner
            .vector_search(query, k, kind, scope, as_of, include_inferred)
            .await
    }
    async fn nodes_missing_vectors(&self) -> Result<Vec<NodeId>> {
        self.inner.nodes_missing_vectors().await
    }
    async fn vector_sweep_orphans(&self) -> Result<u64> {
        self.inner.vector_sweep_orphans().await
    }
    async fn wal_checkpoint(
        &self,
        mode: crate::backend::CheckpointMode,
        wait: std::time::Duration,
    ) -> Result<crate::backend::CheckpointResult> {
        self.check_sql("PRAGMA wal_checkpoint")?;
        self.inner.wal_checkpoint(mode, wait).await
    }
    async fn set_secure_delete(&self, on: bool) -> Result<bool> {
        self.inner.set_secure_delete(on).await
    }
    async fn begin(&self) -> Result<Box<dyn crate::backend::BackendTx + '_>> {
        self.probe.before_begin().await;
        let inner = self.inner.begin().await?;
        let fail_on_execute = self
            .fail_on_execute
            .load(std::sync::atomic::Ordering::SeqCst);
        Ok(Box::new(FailingTx {
            inner,
            execute_calls: 0,
            fail_on_execute,
            fail_on_row: self.fail_on_row.lock().unwrap().clone(),
            probe: &self.probe,
        }))
    }
}

/// Wraps a real `Backend`, failing exactly one targeted `vector_upsert`
/// call while delegating every other call straight through.
///
/// Simpler than `FailingTx`/`FailingBackend` above: `vector_upsert` is a
/// plain `Backend` method, not a `BackendTx` one, so this needs no fake
/// transaction, just one intercepted method. `fail_on_vector_upsert`
/// defaults to "never" (`usize::MAX`) so any other test using this type
/// would see ordinary behavior; the one test that needs a fault sets it
/// explicitly after `open`.
struct FailingVectorBackend {
    inner: crate::backends::DefaultBackend,
    /// 0-based count of `vector_upsert` calls seen so far.
    vector_upsert_calls: std::sync::atomic::AtomicUsize,
    /// The `vector_upsert_calls` value (before incrementing) that fails.
    fail_on_vector_upsert: std::sync::atomic::AtomicUsize,
}

impl FailingVectorBackend {
    fn set_fail_on_vector_upsert(&self, n: usize) {
        self.fail_on_vector_upsert
            .store(n, std::sync::atomic::Ordering::SeqCst);
    }
}

#[async_trait::async_trait]
impl Backend for FailingVectorBackend {
    async fn open(path: &str, read_pool_size: usize) -> Result<Self> {
        Ok(Self {
            inner: crate::backends::DefaultBackend::open(path, read_pool_size).await?,
            vector_upsert_calls: std::sync::atomic::AtomicUsize::new(0),
            fail_on_vector_upsert: std::sync::atomic::AtomicUsize::new(usize::MAX),
        })
    }
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        self.inner.query(sql, params).await
    }
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        self.inner.execute(sql, params).await
    }
    async fn execute_batch(&self, sql: &str) -> Result<()> {
        self.inner.execute_batch(sql).await
    }
    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()> {
        self.inner.execute_atomic(statements).await
    }
    fn vector_ddl(&self, dims: usize) -> String {
        self.inner.vector_ddl(dims)
    }
    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()> {
        let call = self
            .vector_upsert_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call
            == self
                .fail_on_vector_upsert
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Err(Error::Backend("injected vector_upsert failure".to_string()));
        }
        self.inner.vector_upsert(node_id, embedding).await
    }
    async fn vector_insert_if_absent(&self, node_id: &str, embedding: &[f32]) -> Result<bool> {
        self.inner.vector_insert_if_absent(node_id, embedding).await
    }
    fn vector_delete_sql(&self) -> Option<&'static str> {
        self.inner.vector_delete_sql()
    }
    fn vector_clear_sql(&self) -> Option<&'static str> {
        self.inner.vector_clear_sql()
    }
    async fn vector_delete(&self, node_id: &str) -> Result<()> {
        self.inner.vector_delete(node_id).await
    }
    async fn vector_get(&self, node_id: &str, dims: usize) -> Result<Option<Vec<f32>>> {
        self.inner.vector_get(node_id, dims).await
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
        self.inner
            .vector_search(query, k, kind, scope, as_of, include_inferred)
            .await
    }
    async fn nodes_missing_vectors(&self) -> Result<Vec<NodeId>> {
        self.inner.nodes_missing_vectors().await
    }
    async fn vector_sweep_orphans(&self) -> Result<u64> {
        self.inner.vector_sweep_orphans().await
    }
    async fn wal_checkpoint(
        &self,
        mode: crate::backend::CheckpointMode,
        wait: std::time::Duration,
    ) -> Result<crate::backend::CheckpointResult> {
        self.inner.wal_checkpoint(mode, wait).await
    }
    async fn set_secure_delete(&self, on: bool) -> Result<bool> {
        self.inner.set_secure_delete(on).await
    }
    async fn begin(&self) -> Result<Box<dyn crate::backend::BackendTx + '_>> {
        self.inner.begin().await
    }
}
