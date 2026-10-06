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
    count, cursor_offset, embedding, fact, fact_at, gc_ages_out_by_kind,
    gc_leaves_an_edge_whose_endpoints_both_survive,
    gc_sweeps_a_node_that_still_has_a_community_assignment,
    gc_sweeps_a_node_that_still_has_an_edge_pointing_at_it, has_node, injected, insert_orphan_edge,
    node, offset_pair, shared, snapshot, Env, ReembedProbe, StubEmbedder, ONE_SEGMENT,
};
use super::*;
use crate::DefaultBackend;

// ---- a real log a test can make fail or hold ----

/// What a test does to the log writer while a sweep runs.
#[derive(Default)]
pub(super) struct Control {
    seen: AtomicU64,
    /// The append, counted from `fail_nth_from_now`, that fails. Zero never.
    fail_on: AtomicU64,
    /// Held by the next append until the test releases it.
    gate: StdMutex<Option<(Arc<Notify>, mpsc::Receiver<()>)>>,
}

impl Control {
    /// Fails the `n`th append from now on, `n` counting from 1.
    pub(super) fn fail_nth_from_now(&self, n: u64) {
        self.seen.store(0, Ordering::SeqCst);
        self.fail_on.store(n, Ordering::SeqCst);
    }

    /// Makes the next append announce itself and wait. The notification says the
    /// append is running; sending on the returned sender lets it go on.
    pub(super) fn hold_next_append(&self) -> (Arc<Notify>, mpsc::Sender<()>) {
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

pub(super) fn controlled_log(env: &Env) -> (SharedLog, Arc<Control>) {
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

const DIMS: usize = 8;

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
        graph: graph.with_embedder(StubEmbedder::new(DIMS)),
        log,
        control,
    }
}

/// A store over a new writer on the same log directory, as after a restart.
/// The store that wrote the log must be gone first.
async fn fresh_store(env: &Env, db: &str) -> (DefaultGraph, SharedLog) {
    let (graph, log) = env.open::<DefaultBackend>(db, clock()).await;
    (graph.with_embedder(StubEmbedder::new(DIMS)), log)
}

/// A new store that has caught up with the log.
async fn caught_up(env: &Env) -> DefaultGraph {
    let (fresh, _log) = fresh_store(env, "caught-up.db").await;
    fresh.catch_up().await.unwrap();
    fresh
}

/// A new store rebuilt from the log.
async fn rebuilt(env: &Env) -> DefaultGraph {
    let (fresh, log) = fresh_store(env, "rebuilt.db").await;
    let (rebuilt, _) = fresh
        .rebuild_from_log(log, RebuildMode::RequireEmpty)
        .await
        .unwrap();
    rebuilt
}

/// Sweeps everything of kind `fact` that began before 999, which is every
/// `fact_at` node and no `fact` node written now.
fn policy() -> RetentionPolicy {
    RetentionPolicy::keep("fact", Millis(1)).without_reclaim()
}

/// `node` with the vector the stub embedder gives its content, so a store that
/// replays the node and embeds it ends up with the same vector.
fn embedded(node: NewNode) -> NewNode {
    let vector = StubEmbedder::vector_for(DIMS, &node.content);
    node.with_embedding(vector)
}

/// The rows a sweep of `policy` is about.
struct Seeded {
    aged: Vec<NodeId>,
    kept: Vec<NodeId>,
    /// Edges with an aged endpoint, which the sweep removes.
    swept_edges: Vec<EdgeId>,
    kept_edge: EdgeId,
}

/// `aged` nodes past retention and three that are not, every one with a vector.
/// The first three aged nodes carry two edges that go with them: one between two
/// aged nodes and one to a kept node. A third edge joins two kept nodes. Needs
/// `aged >= 3`.
async fn seed<B: Backend>(g: &Graph<B>, aged: usize) -> Seeded {
    let mut old = Vec::new();
    for i in 0..aged {
        let node = embedded(fact_at(&format!("aged-{i}")));
        old.push(g.insert(node).await.unwrap());
    }
    let mut kept = Vec::new();
    for i in 0..3 {
        let node = embedded(fact(&format!("kept-{i}")));
        kept.push(g.insert(node).await.unwrap());
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

/// Gives `ids` a community assignment, which the log never records.
async fn assign_communities<B: Backend>(g: &Graph<B>, ids: &[NodeId]) {
    for id in ids {
        g.backend
            .execute(
                "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, 1, 1)",
                &[id.as_str().into()],
            )
            .await
            .unwrap();
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

/// How many targets each logged event names, in log order.
fn sizes(batches: &[Vec<TombstoneTarget>]) -> Vec<usize> {
    batches.iter().map(Vec::len).collect()
}

/// The log tombstones exactly `swept` and names no other row.
async fn assert_tombstoned(env: &Env, swept: &[NodeId]) {
    let logged = batches(&env.records().await);
    assert_eq!(targeted(&logged, TombstoneTable::Nodes), sorted(swept));
    assert_eq!(logged.iter().flatten().count(), swept.len());
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

/// `snapshot` and the tables a sweep must also leave right: the vectors, and the
/// community assignments the log never records, so a replay agrees on them only
/// when every node that had one was swept.
async fn dump<B: Backend>(g: &Graph<B>) -> String {
    let mut state = format!("{:?}\n", snapshot(g).await);
    for table in ["node_vectors", "node_community"] {
        let rows = g
            .backend
            .query(&format!("SELECT * FROM {table} ORDER BY 1"), &[])
            .await
            .unwrap();
        state.push_str(&format!("{table}: {rows:?}\n"));
    }
    state
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
async fn gc_log_splits_a_large_sweep_into_chunks_of_the_target_size() {
    // Arrange: ten aged nodes in a chain of edges, and a chunk of three targets
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let seeded = seed(&graph, 10).await;
    let mut chain = Vec::new();
    for ends in seeded.aged[2..].windows(2) {
        graph.relate(&ends[0], &ends[1], "mentions").await.unwrap();
        chain.push((ends[0].as_str().to_owned(), ends[1].as_str().to_owned()));
    }
    let before = env.records().await.len();

    // Act
    let report = graph.sweep(&policy(), 3).await.unwrap();

    // Assert: full events and a short last one, and together exactly the swept rows
    let records = env.records().await;
    let appended = batches(&records[before..]);
    assert_eq!(sizes(&appended), [3, 3, 3, 1]);
    assert_only_swept_rows(&appended, &seeded);
    assert_eq!(count(&graph, "nodes").await, 3);
    assert_eq!(
        cursor_offset(&graph).await,
        Some(offset_pair(records.last().unwrap().offset))
    );

    // Assert: an edge is counted once, whether its ends share a chunk or not,
    // and the chain has both kinds
    let chunk_of = |id: &str| {
        appended
            .iter()
            .position(|targets| targets.iter().any(|target| target.id == id))
            .expect("a chain node is tombstoned")
    };
    let spans = |same: bool| {
        chain
            .iter()
            .any(|(src, dst)| (chunk_of(src) == chunk_of(dst)) == same)
    };
    assert!(spans(true) && spans(false), "{chain:?}");
    assert_eq!(report.nodes_removed, 10);
    assert_eq!(report.edges_removed, 2 + chain.len() as u64);
}

#[tokio::test]
async fn gc_log_sweeps_every_rule_in_chunks_and_adds_up_the_reports() {
    // Arrange: five aged facts and three aged episodes, an edge from an episode
    // to a fact the first rule sweeps, and a chunk of two
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let seeded = seed(&graph, 5).await;
    assign_communities(&graph, &seeded.aged[..2]).await;
    let mut episodes = Vec::new();
    for i in 0..3 {
        let node = NewNode::now("episode", "label", format!("episode-{i}"));
        episodes.push(
            graph
                .insert(embedded(node.with_valid_from(Millis(500))))
                .await
                .unwrap(),
        );
    }
    graph
        .relate(&episodes[0], &seeded.aged[3], "mentions")
        .await
        .unwrap();
    let before = env.records().await.len();
    let both = RetentionPolicy::keep("fact", Millis(1))
        .and_keep("episode", Millis(1))
        .without_reclaim();

    // Act
    let report = graph.sweep(&both, 2).await.unwrap();

    // Assert: each rule's nodes in chunks of their own, summed in the report
    let appended = batches(&env.records().await[before..]);
    assert_eq!(sizes(&appended), [2, 2, 1, 2, 1]);
    for chunk in &appended {
        let facts = chunk
            .iter()
            .filter(|target| seeded.aged.iter().any(|id| id.as_str() == target.id))
            .count();
        assert!(facts == 0 || facts == chunk.len(), "mixed kinds: {chunk:?}");
    }
    let swept: Vec<NodeId> = seeded.aged.iter().chain(&episodes).cloned().collect();
    assert_eq!(targeted(&appended, TombstoneTable::Nodes), sorted(&swept));
    assert_eq!(report.nodes_removed, 8);
    assert_eq!(report.edges_removed, 3);

    // Assert: a store that replays the log holds the same
    let live = dump(&graph).await;
    drop(graph);
    assert_eq!(dump(&caught_up(&env).await).await, live);
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
async fn gc_log_snapshot_sees_a_change_that_only_touches_edges() {
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "edges.db").await;
    let a = graph.insert(fact("edge-a")).await.unwrap();
    let b = graph.insert(fact("edge-b")).await.unwrap();
    let before = snapshot(&graph).await;

    graph.relate(&a, &b, "mentions").await.unwrap();

    assert_ne!(snapshot(&graph).await, before);
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

#[tokio::test]
async fn gc_log_keeps_a_node_that_began_exactly_at_the_cutoff() {
    // Arrange: a retention of 500 at 1000 puts the cutoff at 500
    let env = Env::new();
    let Store { graph, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let at_cutoff = graph
        .insert(fact("at the cutoff").with_valid_from(Millis(500)))
        .await
        .unwrap();
    let older = graph
        .insert(fact("older").with_valid_from(Millis(499)))
        .await
        .unwrap();

    // Act
    let report = graph
        .gc(&RetentionPolicy::keep("fact", Millis(500)).without_reclaim())
        .await
        .unwrap();

    // Assert: only a node that began before the cutoff is swept
    assert_eq!(report.nodes_removed, 1);
    assert!(has_node(&graph, at_cutoff.as_str()).await);
    assert!(!has_node(&graph, older.as_str()).await);
    assert_tombstoned(&env, &[older]).await;
}

#[tokio::test]
async fn gc_log_sweeps_a_superseded_chain_with_its_edge() {
    // Arrange: the first version is closed, and a supersedes edge joins the two
    let env = Env::new();
    let Store { graph, log, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    let first = graph
        .insert(fact_at("first").with_subject("subject"))
        .await
        .unwrap();
    let second = graph
        .supersede(&first, fact_at("second").with_subject("subject"))
        .await
        .unwrap();
    assert_eq!(count(&graph, "edges").await, 1);

    // Act
    let report = graph.gc(&policy()).await.unwrap();

    // Assert: both versions and the edge are gone, and the log says so
    assert_eq!((report.nodes_removed, report.edges_removed), (2, 1));
    assert_eq!(count(&graph, "nodes").await, 0);
    assert_eq!(count(&graph, "edges").await, 0);
    assert_tombstoned(&env, &[first, second]).await;

    // Assert: a rebuild and a catch up both agree with the store
    let live = dump(&graph).await;
    drop(graph);
    drop(log);
    assert_eq!(dump(&rebuilt(&env).await).await, live, "rebuild");
    assert_eq!(dump(&caught_up(&env).await).await, live, "catch_up");
}

// ---- an orphaned edge ----

#[tokio::test]
async fn gc_log_tombstones_an_edge_whose_endpoint_is_gone() {
    // Arrange: three edges that no node holds up, and a chunk of two
    let env = Env::new();
    let Store { graph, log, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    graph.insert(embedded(fact("current"))).await.unwrap();
    for id in ["orphan-a", "orphan-b", "orphan-c"] {
        insert_orphan_edge(&graph, id).await;
    }
    let before = env.records().await.len();

    // Act
    let report = graph.sweep(&policy(), 2).await.unwrap();

    // Assert: the edges are counted, tombstoned in chunks, and deleted
    assert_eq!((report.nodes_removed, report.edges_removed), (0, 3));
    let appended = batches(&env.records().await[before..]);
    assert_eq!(sizes(&appended), [2, 1]);
    assert_eq!(
        targeted(&appended, TombstoneTable::Edges),
        ["orphan-a", "orphan-b", "orphan-c"]
    );
    assert_eq!(count(&graph, "edges").await, 0);

    // Assert: a replay of that tombstone on a store that never held the edges
    // agrees with this one
    let live = dump(&graph).await;
    drop(graph);
    drop(log);
    assert_eq!(dump(&caught_up(&env).await).await, live);
}

// ---- a node with a vector ----

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
    assert_tombstoned(&env, &[id]).await;
}

// ---- a sweep is in the log for good ----

/// A store that swept with a small chunk, as a dump of what it holds after, and
/// the rows it swept. Its writer is gone, so another store can open the log.
async fn swept_store(env: &Env) -> (String, Seeded) {
    let Store { graph, log, .. } = open_store::<DefaultBackend>(env, "live.db").await;
    let seeded = seed(&graph, 6).await;
    assign_communities(&graph, &seeded.aged[..2]).await;
    graph.sweep(&policy(), 2).await.unwrap();
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

    // Act
    let rebuilt = rebuilt(&env).await;

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

    // Act
    let fresh = caught_up(&env).await;

    // Assert
    assert_eq!(dump(&fresh).await, live);
    for id in &seeded.aged {
        assert!(!has_node(&fresh, id.as_str()).await, "{id:?} came back");
    }
}

// ---- the log lock is held across the sweep ----

/// Seeds aged nodes of each kind in `kinds`, sweeps them in chunks of two with
/// the first append held, and inserts a row that is itself past retention while
/// it waits. Returns what the log holds after that insert. The sweep takes
/// `tombstones` events, and none of them may let the insert in.
async fn insert_during_a_held_sweep(kinds: &[(&str, usize)], tombstones: usize) {
    within_deadline(async {
        // Arrange: the first tombstone append is held, so the sweep has chosen
        // its ids and holds the log lock
        let env = Env::new();
        let Store {
            graph,
            log,
            control,
        } = open_store::<DefaultBackend>(&env, "live.db").await;
        let graph = Arc::new(graph);
        let mut swept = Vec::new();
        let mut policy = RetentionPolicy::keep(kinds[0].0, Millis(1)).without_reclaim();
        for (n, (kind, aged)) in kinds.iter().enumerate() {
            if n > 0 {
                policy = policy.and_keep(*kind, Millis(1));
            }
            for i in 0..*aged {
                let node = NewNode::now(*kind, "label", format!("{kind}-{i}"));
                swept.push(
                    graph
                        .insert(embedded(node.with_valid_from(Millis(500))))
                        .await
                        .unwrap(),
                );
            }
        }
        let before = env.records().await.len();
        let (reached, release) = control.hold_next_append();
        let mut sweep = tokio::spawn({
            let graph = Arc::clone(&graph);
            async move { graph.sweep(&policy, 2).await }
        });
        tokio::select! {
            () = reached.notified() => {}
            finished = &mut sweep => panic!("the sweep finished without logging: {finished:?}"),
        }

        // Act: a row that is itself past retention is inserted meanwhile
        let late = tokio::spawn({
            let graph = Arc::clone(&graph);
            async move { graph.insert(embedded(fact_at("late arrival"))).await }
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

        // Assert: the late row was not swept, and it is logged after every tombstone
        assert_eq!(report.nodes_removed, swept.len() as u64);
        assert!(has_node(&*graph, late.as_str()).await);
        let records = env.records().await;
        let (last, tombstoned) = records[before..].split_last().expect("records");
        assert_eq!(batches(tombstoned).len(), tombstones);
        assert_eq!(
            tombstoned.len(),
            tombstones,
            "only tombstones before the insert"
        );
        assert_eq!(
            targeted(&batches(tombstoned), TombstoneTable::Nodes),
            sorted(&swept)
        );
        assert!(
            matches!(&last.event.payload, LogPayload::NodeWrite(row) if row.id == late.as_str()),
            "the insert is logged after the sweep: {last:?}"
        );
        let live = dump(&*graph).await;
        drop(graph);
        drop(log);
        assert_eq!(dump(&caught_up(&env).await).await, live);
    })
    .await;
}

#[tokio::test]
async fn gc_log_a_concurrent_insert_waits_out_every_chunk_of_a_rule() {
    // Arrange: five aged nodes in chunks of two make three tombstones
    insert_during_a_held_sweep(&[("fact", 5)], 3).await;
}

#[tokio::test]
async fn gc_log_a_concurrent_insert_waits_out_every_rule() {
    // Arrange: one chunk a rule, so only the lock can keep the insert behind both
    insert_during_a_held_sweep(&[("fact", 2), ("episode", 2)], 2).await;
}

#[tokio::test]
async fn gc_log_frees_the_log_before_the_vector_cleanup() {
    // Arrange
    let env = Env::new();
    let Store { graph, log, .. } = open_store::<ReembedProbe>(&env, "live.db").await;
    seed(&graph, 5).await;
    graph.backend.watch_orphan_sweep(&log);

    // Act
    graph.gc(&policy()).await.unwrap();

    // Assert: what follows the last log write does not make writers wait
    assert_eq!(graph.backend.log_free_at_orphan_sweep(), Some(true));
}

// ---- a node that lands after the doomed set is chosen ----

/// Two aged nodes, then a third that is written straight to the store right
/// after the sweep selects its doomed ones. Returns the two the sweep took.
async fn sweep_over_a_row_that_lands_mid_selection(g: &Graph<ReembedProbe>) -> Vec<NodeId> {
    let first = g.insert(fact_at("first")).await.unwrap();
    let second = g.insert(fact_at("second")).await.unwrap();
    g.backend
        .insert_after_query("SELECT id FROM nodes WHERE kind", node("landed", "landed"));

    let report = g.gc(&policy()).await.unwrap();

    assert_eq!(report.nodes_removed, 2);
    assert!(has_node(g, "landed").await, "the late row was swept");
    assert_eq!(count(g, "nodes").await, 1);
    vec![first, second]
}

#[tokio::test]
async fn gc_log_removes_only_the_nodes_it_selected() {
    // Arrange
    let env = Env::new();
    let Store { graph, .. } = open_store::<ReembedProbe>(&env, "live.db").await;

    // Act
    let swept = sweep_over_a_row_that_lands_mid_selection(&graph).await;

    // Assert: the late row is neither deleted nor tombstoned
    assert_tombstoned(&env, &swept).await;
}

#[tokio::test]
async fn gc_removes_only_the_nodes_it_selected_without_a_log() {
    // Arrange
    let graph = Graph::<ReembedProbe>::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
        .await
        .unwrap();

    // Act and Assert
    sweep_over_a_row_that_lands_mid_selection(&graph).await;
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
async fn gc_log_a_poisoned_log_refuses_the_sweep() {
    // Arrange
    let env = Env::new();
    let Store { graph, log, .. } = open_store::<DefaultBackend>(&env, "live.db").await;
    seed(&graph, 5).await;
    let before = (env.records().await.len(), dump(&graph).await);
    log.lock().await.poison();

    // Act
    let refused = graph.gc(&policy()).await;

    // Assert: nothing was logged or deleted
    assert!(matches!(refused, Err(Error::LogPoisoned)), "{refused:?}");
    assert_eq!((env.records().await.len(), dump(&graph).await), before);
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
    let (fresh, _log) = fresh_store(&env, "fresh.db").await;
    let replayed = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(replayed.skipped_voided, 1, "{replayed:?}");
    assert_eq!(dump(&fresh).await, live);
}

#[tokio::test]
async fn gc_log_a_failure_in_a_later_chunk_keeps_the_chunks_before_it() {
    // Arrange: five aged nodes in chunks of two, and a row in the second chunk
    // that cannot be removed
    let env = Env::new();
    let Store { graph, .. } = open_store::<FailingBackend>(&env, "live.db").await;
    seed(&graph, 5).await;
    let selected = graph
        .backend
        .query(
            "SELECT id FROM nodes WHERE kind = ?1 AND valid_from < ?2",
            &["fact".into(), 999_i64.into()],
        )
        .await
        .unwrap();
    let second_chunk = selected[2].get_string(0).unwrap();
    graph.backend.set_fail_on_row(Some(&second_chunk));
    let before = env.records().await.len();

    // Act
    let failed = graph.sweep(&policy(), 2).await;

    // Assert: the error surfaces, the first chunk is deleted, and the log holds
    // it, then the second chunk's tombstone and the record that cancels it
    let error = failed.expect_err("the sweep must fail");
    assert!(injected(&error), "{error:?}");
    assert_eq!(count(&graph, "nodes").await, 6);
    let records = env.records().await;
    let appended = &records[before..];
    let [first, second, void] = appended else {
        panic!("two tombstones then a void: {appended:?}");
    };
    assert_eq!(sizes(&batches(&appended[..2])), [2, 2]);
    assert_eq!(
        void.event.payload,
        LogPayload::Voided {
            target_event_id: second.event.event_id.clone()
        }
    );
    assert!(first.event.event_id != second.event.event_id);
    assert_eq!(cursor_offset(&graph).await, Some(offset_pair(void.offset)));

    // Assert: a replay skips the cancelled chunk and holds what the store does
    let live = dump(&graph).await;
    drop(graph);
    let (fresh, _log) = fresh_store(&env, "fresh.db").await;
    let replayed = fresh.catch_up().await.unwrap();
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
    let env = Env::new();
    let clock = clock();
    let Store { graph, .. } =
        open_store_at::<DefaultBackend>(&env, "live.db", Arc::clone(&clock)).await;

    // Act
    let swept = gc_ages_out_by_kind(&graph, &clock).await;

    // Assert
    assert_tombstoned(&env, &swept).await;
}

#[tokio::test]
async fn gc_log_sweeps_a_node_that_still_has_an_edge_pointing_at_it() {
    // Arrange
    let env = Env::new();
    let clock = clock();
    let Store { graph, .. } =
        open_store_at::<DefaultBackend>(&env, "live.db", Arc::clone(&clock)).await;

    // Act
    let swept = gc_sweeps_a_node_that_still_has_an_edge_pointing_at_it(&graph, &clock).await;

    // Assert
    assert_tombstoned(&env, &swept).await;
}

#[tokio::test]
async fn gc_log_sweeps_a_node_that_still_has_a_community_assignment() {
    // Arrange
    let env = Env::new();
    let clock = clock();
    let Store { graph, .. } =
        open_store_at::<DefaultBackend>(&env, "live.db", Arc::clone(&clock)).await;

    // Act
    let swept = gc_sweeps_a_node_that_still_has_a_community_assignment(&graph, &clock).await;

    // Assert
    assert_tombstoned(&env, &swept).await;
}

#[tokio::test]
async fn gc_log_leaves_an_edge_whose_endpoints_both_survive() {
    // Arrange
    let env = Env::new();
    let clock = clock();
    let Store { graph, .. } =
        open_store_at::<DefaultBackend>(&env, "live.db", Arc::clone(&clock)).await;

    // Act
    let swept = gc_leaves_an_edge_whose_endpoints_both_survive(&graph, &clock).await;

    // Assert: nothing was tombstoned
    assert_tombstoned(&env, &swept).await;
}
