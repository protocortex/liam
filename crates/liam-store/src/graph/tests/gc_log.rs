// SPDX-License-Identifier: Apache-2.0
//! `gc` with an event log attached: every swept row is tombstoned in the log
//! before it is deleted, in chunks, under the log lock. Tests run against a real
//! WAL and reader in a temp directory, and judge a sweep by what the log holds
//! and by what a store rebuilt from it holds.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Mutex as StdMutex};

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::event::{LogEvent, LogPayload, TombstoneTable, TombstoneTarget};
use liam_log::reader::{LogRecord, SequentialScanReader};
use liam_log::wal::{SystemClock, WalError, WalWriter};
use liam_log::{LogOffset, LogWriter};
use tokio::sync::Notify;

use super::log_write::within_deadline;
use super::support::{
    count, cursor, fact, fact_at, has_node, offset_pair, shared, Env, ONE_SEGMENT,
};
use super::*;
use crate::DefaultBackend;

// ---- a real log a test can make fail or hold ----

/// What a test does to the log writer while a sweep runs.
#[derive(Default)]
struct Control {
    seen: AtomicU64,
    /// The append, counted from `fail_nth_from_now`, that fails. Zero never.
    fail_on: AtomicU64,
    /// Held by the next append until the test releases it.
    gate: StdMutex<Option<(Arc<Notify>, mpsc::Receiver<()>)>>,
}

impl Control {
    /// Fails the `n`th append from now on, `n` counting from 1.
    fn fail_nth_from_now(&self, n: u64) {
        self.seen.store(0, Ordering::SeqCst);
        self.fail_on.store(n, Ordering::SeqCst);
    }

    /// Makes the next append announce itself and wait. The notification says the
    /// append is running; sending on the returned sender lets it go on.
    fn hold_next_append(&self) -> (Arc<Notify>, mpsc::Sender<()>) {
        let (release, wait) = mpsc::channel();
        let reached = Arc::new(Notify::new());
        *self.gate.lock().unwrap() = Some((Arc::clone(&reached), wait));
        (reached, release)
    }
}

/// A WAL writer that fails or holds an append when its `Control` says so.
struct Controlled {
    inner: WalWriter<SystemClock>,
    control: Arc<Control>,
}

impl LogWriter for Controlled {
    fn append(&mut self, event: &LogEvent) -> std::result::Result<LogOffset, WalError> {
        let held = self.control.gate.lock().unwrap().take();
        if let Some((reached, release)) = held {
            reached.notify_one();
            let _ = release.recv();
        }
        let nth = self.control.seen.fetch_add(1, Ordering::SeqCst) + 1;
        if nth == self.control.fail_on.load(Ordering::SeqCst) {
            return Err(WalError::Io(io::Error::other("injected append failure")));
        }
        self.inner.append(event)
    }

    fn log_id(&self) -> uuid::Uuid {
        self.inner.log_id()
    }

    fn head(&self) -> Option<LogOffset> {
        self.inner.head()
    }
}

fn controlled_log(env: &Env) -> (SharedLog, Arc<Control>) {
    let control = Arc::new(Control::default());
    let inner = WalWriter::open_with_system_clock(&env.wal_dir(), ONE_SEGMENT).expect("open wal");
    let reader = SequentialScanReader::local(&env.wal_dir()).expect("open reader");
    let writer = Controlled {
        inner,
        control: Arc::clone(&control),
    };
    let log = EventLog::new(
        Box::new(writer),
        Arc::new(reader),
        HashBloom::new(BloomConfig::default()),
    );
    (shared(log), control)
}

// ---- a store over that log ----

struct Store<B: Backend> {
    graph: Graph<B>,
    log: SharedLog,
    control: Arc<Control>,
}

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

async fn open_store<B: Backend>(env: &Env, db: &str) -> Store<B> {
    open_store_at(env, db, clock()).await
}

async fn open_store_at<B: Backend>(env: &Env, db: &str, clock: Arc<FixedClock>) -> Store<B> {
    let (log, control) = controlled_log(env);
    let (graph, log) = env.open_over::<B>(log, db, clock).await;
    Store {
        graph,
        log,
        control,
    }
}

/// A store over a new writer on the same log directory, as after a restart.
/// The store that wrote the log must be gone first.
async fn fresh_store(env: &Env) -> (DefaultGraph, SharedLog) {
    env.open::<DefaultBackend>("fresh.db", clock()).await
}

/// Sweeps everything of kind `fact` that began before 999, which is every
/// `fact_at` node and no `fact` node written now.
fn policy() -> RetentionPolicy {
    RetentionPolicy::keep("fact", Millis(1)).without_reclaim()
}

/// The rows a sweep of `policy` is about.
struct Seeded {
    aged: Vec<NodeId>,
    kept: Vec<NodeId>,
    /// Edges with an aged endpoint, which the sweep removes.
    swept_edges: Vec<EdgeId>,
    kept_edge: EdgeId,
}

/// `aged` nodes past retention and three that are not. The first three aged
/// nodes carry two edges that go with them: one between two aged nodes and one
/// to a kept node. A third edge joins two kept nodes. Needs `aged >= 3`.
async fn seed<B: Backend>(g: &Graph<B>, aged: usize) -> Seeded {
    let mut old = Vec::new();
    for i in 0..aged {
        old.push(g.insert(fact_at(&format!("aged-{i}"))).await.unwrap());
    }
    let mut kept = Vec::new();
    for i in 0..3 {
        kept.push(g.insert(fact(&format!("kept-{i}"))).await.unwrap());
    }
    let swept_edges = vec![
        g.relate(&old[0], &old[1], "mentions").await.unwrap(),
        g.relate(&old[2], &kept[0], "mentions").await.unwrap(),
    ];
    let kept_edge = g.relate(&kept[0], &kept[1], "mentions").await.unwrap();
    Seeded {
        aged: old,
        kept,
        swept_edges,
        kept_edge,
    }
}

// ---- reading the log and the store back ----

fn batches(records: &[LogRecord]) -> Vec<Vec<TombstoneTarget>> {
    records
        .iter()
        .filter_map(|record| match &record.event.payload {
            LogPayload::Tombstone(targets) => Some(targets.clone()),
            _ => None,
        })
        .collect()
}

fn targeted(batches: &[Vec<TombstoneTarget>], table: TombstoneTable) -> Vec<String> {
    let mut ids: Vec<String> = batches
        .iter()
        .flatten()
        .filter(|target| target.table == table)
        .map(|target| target.id.clone())
        .collect();
    ids.sort();
    ids
}

fn sorted(ids: &[NodeId]) -> Vec<String> {
    let mut ids: Vec<String> = ids.iter().map(|id| id.as_str().to_owned()).collect();
    ids.sort();
    ids
}

/// Every tombstone target names a swept row, never a kept one, and the swept
/// nodes are each tombstoned exactly once. Whether an edge is tombstoned itself
/// or goes with its node is the replay's business, so either is accepted.
fn assert_only_swept_rows(batches: &[Vec<TombstoneTarget>], seeded: &Seeded) {
    assert_eq!(
        targeted(batches, TombstoneTable::Nodes),
        sorted(&seeded.aged)
    );
    for target in batches.iter().flatten() {
        let swept = match target.table {
            TombstoneTable::Nodes | TombstoneTable::NodeCommunity => {
                seeded.aged.iter().any(|id| id.as_str() == target.id)
            }
            TombstoneTable::Edges => seeded.swept_edges.iter().any(|id| id.as_str() == target.id),
        };
        assert!(swept, "a kept or unknown row was tombstoned: {target:?}");
    }
    let kept_edge = seeded.kept_edge.as_str();
    assert!(
        batches
            .iter()
            .flatten()
            .all(|target| target.id != kept_edge),
        "the edge between kept nodes was tombstoned"
    );
    for id in &seeded.kept {
        assert!(
            batches
                .iter()
                .flatten()
                .all(|target| target.id != id.as_str()),
            "a kept node was tombstoned: {id:?}"
        );
    }
}

const NODE_COLUMNS: &str = "id, kind, label, content, producer, attributes, scope, subject, \
     confidence, valid_from, valid_until, tx_from, tx_to";
const EDGE_COLUMNS: &str = "id, src, dst, type, attributes, tx_from, tx_to";

/// The nodes and edges in full, ordered by id, so two stores compare exactly.
async fn dump<B: Backend>(g: &Graph<B>) -> String {
    let mut state = String::new();
    for (table, columns) in [("nodes", NODE_COLUMNS), ("edges", EDGE_COLUMNS)] {
        let rows = g
            .backend
            .query(&format!("SELECT {columns} FROM {table} ORDER BY id"), &[])
            .await
            .unwrap();
        state.push_str(&format!("{table}: {rows:?}\n"));
    }
    state
}

fn injected(error: &Error) -> bool {
    matches!(error, Error::Backend(message) if message.contains("injected"))
}

/// The offset the store's cursor sits on.
async fn cursor_offset<B: Backend>(g: &Graph<B>) -> Option<(i64, i64)> {
    cursor(g).await.and_then(|(_, offset)| offset)
}

// ---- what a sweep logs ----

#[tokio::test]
async fn gc_log_tombstones_every_swept_node_and_nothing_else() {
    // Arrange
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let seeded = seed(&graph, 5).await;
    let before = env.records().await.len();

    // Act
    let report = graph.gc(&policy()).await.unwrap();

    // Assert: five node targets, only tombstones appended, the store applied them
    let records = env.records().await;
    let appended = &records[before..];
    assert!(
        appended
            .iter()
            .all(|record| matches!(record.event.payload, LogPayload::Tombstone(_))),
        "{appended:?}"
    );
    assert_only_swept_rows(&batches(appended), &seeded);
    assert_eq!(report.nodes_removed, 5);
    assert_eq!(
        cursor_offset(&graph).await,
        Some(offset_pair(records.last().unwrap().offset))
    );
}

#[tokio::test]
async fn gc_log_splits_a_large_sweep_into_chunked_events() {
    // Arrange: ten aged nodes and a chunk of three targets
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let graph = graph.with_gc_chunk(3);
    let seeded = seed(&graph, 10).await;
    let before = env.records().await.len();

    // Act
    let report = graph.gc(&policy()).await.unwrap();

    // Assert: no event is over the chunk, there are several, and together they
    // tombstone exactly the swept rows
    let records = env.records().await;
    let appended = batches(&records[before..]);
    assert!(appended.len() >= 4, "{} events", appended.len());
    assert!(
        appended.iter().all(|targets| targets.len() <= 3),
        "{appended:?}"
    );
    assert_only_swept_rows(&appended, &seeded);
    assert_eq!(report.nodes_removed, 10);
    assert_eq!(count(&graph, "nodes").await, 3);
    assert_eq!(
        cursor_offset(&graph).await,
        Some(offset_pair(records.last().unwrap().offset))
    );
}

#[tokio::test]
async fn gc_log_reports_the_same_counts_as_a_sweep_without_a_log() {
    // Arrange: the same store twice, one of them logged
    let env = Env::new();
    let Store { graph: logged, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let plain = graph_at(Millis(1000)).await;
    seed(&logged, 5).await;
    seed(&plain, 5).await;
    let before = env.records().await.len();

    // Act
    let logged_report = logged.gc(&policy()).await.unwrap();
    let plain_report = plain.gc(&policy()).await.unwrap();

    // Assert: the counts are rows deleted, and the logged sweep did log them
    assert_eq!(
        (logged_report.nodes_removed, logged_report.edges_removed),
        (5, 2)
    );
    assert_eq!(
        (logged_report.nodes_removed, logged_report.edges_removed),
        (plain_report.nodes_removed, plain_report.edges_removed)
    );
    assert!(!batches(&env.records().await[before..]).is_empty());
}

#[tokio::test]
async fn gc_log_with_nothing_to_sweep_appends_nothing() {
    // Arrange: every node is current
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    graph.insert(fact("current")).await.unwrap();
    let before = (env.records().await.len(), cursor_offset(&graph).await);

    // Act
    let report = graph.gc(&policy()).await.unwrap();

    // Assert
    assert_eq!((report.nodes_removed, report.edges_removed), (0, 0));
    assert_eq!(
        (env.records().await.len(), cursor_offset(&graph).await),
        before
    );
}

// ---- a node with a vector ----

fn embedding() -> Vec<f32> {
    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

#[tokio::test]
async fn gc_log_sweeps_a_node_that_has_a_vector() {
    // Arrange
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let id = graph
        .insert(fact_at("embedded").with_embedding(embedding()))
        .await
        .unwrap();
    assert_eq!(count(&graph, "node_vectors").await, 1);

    // Act
    let swept = graph.gc(&policy()).await;

    // Assert: no foreign key error, and the vector went with the node
    let report = swept.expect("a node with a vector must be swept");
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(count(&graph, "nodes").await, 0);
    assert_eq!(count(&graph, "node_vectors").await, 0);
    let tombstoned = batches(&env.records().await);
    assert_eq!(targeted(&tombstoned, TombstoneTable::Nodes), sorted(&[id]));
}

#[tokio::test]
async fn gc_log_sweeps_a_node_that_has_a_vector_without_a_log() {
    // Arrange
    let graph = graph_at(Millis(1000)).await;
    graph
        .insert(fact_at("embedded").with_embedding(embedding()))
        .await
        .unwrap();
    assert_eq!(count(&graph, "node_vectors").await, 1);

    // Act
    let swept = graph.gc(&policy()).await;

    // Assert
    let report = swept.expect("a node with a vector must be swept");
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(count(&graph, "nodes").await, 0);
    assert_eq!(count(&graph, "node_vectors").await, 0);
}

// ---- a sweep is in the log for good ----

/// A store that swept with a small chunk, as a dump of what it holds after, and
/// the rows it swept. Its writer is gone, so another store can open the log.
async fn swept_store(env: &Env) -> (String, Seeded) {
    let Store { graph, log, .. } = open_store::<DefaultBackend>(env, "live.db").await;
    let graph = graph.with_gc_chunk(2);
    let seeded = seed(&graph, 6).await;
    graph.gc(&policy()).await.unwrap();
    let live = dump(&graph).await;
    drop(graph);
    drop(log);
    (live, seeded)
}

#[tokio::test]
async fn gc_log_rebuild_from_the_log_does_not_resurrect_swept_rows() {
    // Arrange
    let env = Env::new();
    let (live, seeded) = swept_store(&env).await;
    let (fresh, log) = fresh_store(&env).await;

    // Act
    let (rebuilt, _) = fresh
        .rebuild_from_log(log, RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert
    assert_eq!(dump(&rebuilt).await, live);
    for id in &seeded.aged {
        assert!(!has_node(&rebuilt, id.as_str()).await, "{id:?} came back");
    }
    assert_eq!(count(&rebuilt, "nodes").await, 3);
    assert_eq!(count(&rebuilt, "edges").await, 1);
}

#[tokio::test]
async fn gc_log_catch_up_on_a_fresh_store_replays_the_tombstones() {
    // Arrange
    let env = Env::new();
    let (live, seeded) = swept_store(&env).await;
    let (fresh, _log) = fresh_store(&env).await;

    // Act
    fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(dump(&fresh).await, live);
    for id in &seeded.aged {
        assert!(!has_node(&fresh, id.as_str()).await, "{id:?} came back");
    }
}

// ---- the log lock is held across the sweep ----

#[tokio::test]
async fn gc_log_a_concurrent_insert_waits_for_the_sweep_and_is_not_swept() {
    within_deadline(async {
        // Arrange: the sweep's tombstone append is held, so its ids are chosen
        // and it holds the log lock
        let env = Env::new();
        let Store {
            graph,
            log,
            control,
        } = open_store::<DefaultBackend>(&env, "live.db").await;
        let graph = Arc::new(graph);
        let seeded = seed(&*graph, 5).await;
        let before = env.records().await.len();
        let (reached, release) = control.hold_next_append();
        let mut sweep = tokio::spawn({
            let graph = Arc::clone(&graph);
            async move { graph.gc(&policy()).await }
        });
        tokio::select! {
            () = reached.notified() => {}
            finished = &mut sweep => panic!("the sweep finished without logging: {finished:?}"),
        }

        // Act: a row that is itself past retention is inserted meanwhile
        let late = tokio::spawn({
            let graph = Arc::clone(&graph);
            async move { graph.insert(fact_at("late arrival")).await }
        });
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }

        // Assert: it is waiting on the log lock the sweep holds
        assert!(log.try_lock().is_err(), "the sweep must hold the log lock");
        assert!(!late.is_finished(), "the insert must wait for the sweep");

        // Act: the sweep goes on
        release.send(()).unwrap();
        let report = sweep.await.unwrap().unwrap();
        let late = late.await.unwrap().unwrap();

        // Assert: the late row was not swept, and the log agrees with the store
        assert_eq!(report.nodes_removed, 5);
        assert!(has_node(&*graph, late.as_str()).await);
        let records = env.records().await;
        let appended = &records[before..];
        assert_only_swept_rows(&batches(appended), &seeded);
        assert!(
            batches(appended)
                .iter()
                .flatten()
                .all(|target| target.id != late.as_str()),
            "the late row was tombstoned"
        );
        let last = appended.last().expect("records");
        assert!(
            matches!(&last.event.payload, LogPayload::NodeWrite(row) if row.id == late.as_str()),
            "the insert is logged after the sweep: {last:?}"
        );
        let live = dump(&*graph).await;
        drop(graph);
        drop(log);
        let (fresh, _log) = fresh_store(&env).await;
        fresh.catch_up().await.unwrap();
        assert_eq!(dump(&fresh).await, live);
    })
    .await;
}

// ---- failures ----

#[tokio::test]
async fn gc_log_a_failed_tombstone_append_deletes_nothing_and_returns_the_error() {
    // Arrange
    let env = Env::new();
    let Store {
        graph,
        log,
        control,
    } = open_store::<DefaultBackend>(&env, "live.db").await;
    seed(&graph, 5).await;
    let before = (
        env.records().await.len(),
        dump(&graph).await,
        cursor_offset(&graph).await,
    );
    control.fail_nth_from_now(1);

    // Act
    let failed = graph.gc(&policy()).await;

    // Assert: nothing logged, nothing deleted, the cursor where it was, and the
    // log still takes writes
    assert!(matches!(failed, Err(Error::LogAppend(_))), "{failed:?}");
    assert_eq!(
        (
            env.records().await.len(),
            dump(&graph).await,
            cursor_offset(&graph).await
        ),
        before
    );
    assert!(!log.lock().await.is_poisoned());
    assert!(graph.insert(fact("after")).await.is_ok());
}

#[tokio::test]
async fn gc_log_a_projection_failure_voids_the_tombstone_and_keeps_the_rows() {
    // Arrange
    let env = Env::new();
    let Store { graph, .. } = open_store::<FailingBackend>(&env, "live.db").await;
    seed(&graph, 5).await;
    let before = (env.records().await.len(), dump(&graph).await);
    graph.backend.set_fail_on_execute(0);

    // Act
    let failed = graph.gc(&policy()).await;

    // Assert: the error is the projection's, the rows are all there, and the log
    // holds the tombstone followed by the record that cancels it
    let error = failed.expect_err("the sweep must fail");
    assert!(injected(&error), "{error:?}");
    let records = env.records().await;
    assert_eq!(dump(&graph).await, before.1);
    let appended: Vec<&LogEvent> = records[before.0..]
        .iter()
        .map(|record| &record.event)
        .collect();
    let [tombstone, void] = appended[..] else {
        panic!("a tombstone then its void: {appended:?}");
    };
    assert!(matches!(tombstone.payload, LogPayload::Tombstone(_)));
    assert_eq!(
        void.payload,
        LogPayload::Voided {
            target_event_id: tombstone.event_id.clone()
        }
    );
    assert_eq!(
        cursor_offset(&graph).await,
        Some(offset_pair(records.last().unwrap().offset))
    );

    // Act: a replay on a fresh store skips the cancelled tombstone
    let live = dump(&graph).await;
    drop(graph);
    let (fresh, _log) = fresh_store(&env).await;
    let replayed = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(replayed.skipped_voided, 1, "{replayed:?}");
    assert_eq!(dump(&fresh).await, live);
}

#[tokio::test]
async fn gc_log_a_sweep_retried_after_a_voided_one_sweeps_the_rows() {
    // Arrange: the first sweep fails after its tombstone is appended
    let env = Env::new();
    let Store { graph, .. } = open_store::<FailingBackend>(&env, "live.db").await;
    seed(&graph, 5).await;
    graph.backend.set_fail_on_execute(0);
    graph.gc(&policy()).await.expect_err("the sweep must fail");
    graph.backend.set_fail_on_execute(usize::MAX);

    // Act
    let report = graph.gc(&policy()).await.unwrap();

    // Assert
    assert_eq!(report.nodes_removed, 5);
    assert_eq!(count(&graph, "nodes").await, 3);
}

#[tokio::test]
async fn gc_log_a_failed_void_poisons_the_log() {
    // Arrange: the projection fails, and so does the second append, the void
    let env = Env::new();
    let Store {
        graph,
        log,
        control,
    } = open_store::<FailingBackend>(&env, "live.db").await;
    seed(&graph, 5).await;
    let before = (env.records().await.len(), dump(&graph).await);
    graph.backend.set_fail_on_execute(0);
    control.fail_nth_from_now(2);

    // Act
    let failed = graph.gc(&policy()).await;
    let refused = graph.insert(fact("after")).await;

    // Assert: the projection's error surfaces, the tombstone is the last record,
    // and the log refuses writes until it is reopened
    let error = failed.expect_err("the sweep must fail");
    assert!(injected(&error), "{error:?}");
    assert!(log.lock().await.is_poisoned());
    assert!(matches!(refused, Err(Error::LogPoisoned)), "{refused:?}");
    let records = env.records().await;
    let appended = batches(&records[before.0..]);
    assert_eq!(appended.len(), 1);
    assert_eq!(records.len(), before.0 + 1, "no void was logged");
    assert_eq!(dump(&graph).await, before.1);
}

// ---- the existing gc tests, against a logged graph ----

#[tokio::test]
async fn gc_log_ages_out_by_kind() {
    // Arrange
    const DAY: i64 = 86_400_000;
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(Millis(100 * DAY)));
    let Store { graph, .. } = open_store_at::<DefaultBackend>(&env, "live.db", clock).await;
    let episode = graph
        .insert(NewNode::now("episode", "old", "x").with_valid_from(Millis(10 * DAY)))
        .await
        .unwrap();
    graph
        .insert(NewNode::now("decision", "keep", "y").with_valid_from(Millis(10 * DAY)))
        .await
        .unwrap();

    // Act
    let report = graph
        .gc(&RetentionPolicy::keep("episode", Millis::days(30)).without_reclaim())
        .await
        .unwrap();

    // Assert
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(
        targeted(&batches(&env.records().await), TombstoneTable::Nodes),
        sorted(&[episode])
    );
}

#[tokio::test]
async fn gc_log_sweeps_a_node_that_still_has_an_edge_pointing_at_it() {
    // Arrange
    let env = Env::new();
    let clock = clock();
    let Store { graph, .. } =
        open_store_at::<DefaultBackend>(&env, "live.db", Arc::clone(&clock)).await;
    let a = graph.insert(NewNode::now("fact", "a", "x1")).await.unwrap();
    let b = graph.insert(NewNode::now("fact", "b", "x2")).await.unwrap();
    graph.link(NewEdge::new(&a, &b, "mentions")).await.unwrap();
    clock.set(Millis(11_000));

    // Act
    let report = graph
        .gc(&RetentionPolicy::keep("fact", Millis(1)))
        .await
        .expect("gc must not fail on a store that holds an edge");

    // Assert
    assert_eq!(report.nodes_removed, 2);
    assert_eq!(report.edges_removed, 1, "the edge goes with its endpoints");
    assert_eq!(count(&graph, "edges").await, 0);
    assert_eq!(
        targeted(&batches(&env.records().await), TombstoneTable::Nodes),
        sorted(&[a, b])
    );
}

#[tokio::test]
async fn gc_log_sweeps_a_node_that_still_has_a_community_assignment() {
    // Arrange
    let env = Env::new();
    let clock = clock();
    let Store { graph, .. } =
        open_store_at::<DefaultBackend>(&env, "live.db", Arc::clone(&clock)).await;
    let a = graph.insert(NewNode::now("fact", "a", "x1")).await.unwrap();
    let b = graph.insert(NewNode::now("fact", "b", "x2")).await.unwrap();
    graph.link(NewEdge::new(&a, &b, "mentions")).await.unwrap();
    graph.recompute_communities().await.unwrap();
    assert!(
        !graph.communities().await.unwrap().is_empty(),
        "assignment seeded"
    );
    clock.set(Millis(11_000));

    // Act
    graph
        .gc(&RetentionPolicy::keep("fact", Millis(1)))
        .await
        .expect("gc must not fail on an assigned node");

    // Assert
    assert!(graph.communities().await.unwrap().is_empty());
    assert_eq!(
        targeted(&batches(&env.records().await), TombstoneTable::Nodes),
        sorted(&[a, b])
    );
}

#[tokio::test]
async fn gc_log_leaves_an_edge_whose_endpoints_both_survive() {
    // Arrange
    let env = Env::new();
    let clock = clock();
    let Store { graph, .. } =
        open_store_at::<DefaultBackend>(&env, "live.db", Arc::clone(&clock)).await;
    let a = graph.insert(NewNode::now("keep", "a", "x1")).await.unwrap();
    let b = graph.insert(NewNode::now("keep", "b", "x2")).await.unwrap();
    graph.link(NewEdge::new(&a, &b, "mentions")).await.unwrap();
    clock.set(Millis(11_000));

    // Act: a rule for a different kind, so nothing is swept at all
    let report = graph
        .gc(&RetentionPolicy::keep("fact", Millis(1)))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.nodes_removed, 0);
    assert_eq!(
        report.edges_removed, 0,
        "an unrelated rule swept a live edge"
    );
    assert!(batches(&env.records().await).is_empty());
}
