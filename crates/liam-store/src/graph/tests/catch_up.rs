// SPDX-License-Identifier: Apache-2.0
//! Replay of the event log into the projection. Tests run against a real WAL
//! and reader in a temp directory: an event that was logged but never projected
//! is appended with a standalone writer, as a crash between the append and the
//! projection leaves it.

use std::path::PathBuf;
use std::time::Duration;

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::event::{
    EdgeRow, LogEvent, LogPayload, NodeRow, RowEffect, TombstoneTable, TombstoneTarget,
    CURRENT_SCHEMA_VERSION,
};
use liam_log::hash::{edge_row_hash, node_row_hash};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{WalConfig, WalWriter};
use liam_log::{LogOffset, LogWriter};

use super::log_write::{count, cursor, fact_at, offset_pair};
use super::*;
use crate::DefaultBackend;

const ONE_SEGMENT: WalConfig = WalConfig {
    segment_max_bytes: 1 << 20,
    rotate_interval_secs: 3600,
};

/// Rotates after every append, so each event sits in a segment of its own.
const SEGMENT_PER_EVENT: WalConfig = WalConfig {
    segment_max_bytes: 1,
    rotate_interval_secs: 3600,
};

/// A log directory and the databases that project it.
struct Env {
    dir: TempDir,
}

impl Env {
    fn new() -> Self {
        Self {
            dir: TempDir::new().unwrap(),
        }
    }

    fn wal_dir(&self) -> PathBuf {
        self.dir.path().join("wal")
    }

    fn db(&self, name: &str) -> String {
        self.dir.path().join(name).to_str().unwrap().to_owned()
    }

    /// Appends `events` with a writer of their own that is gone afterwards, so
    /// no store has projected them.
    fn append(&self, events: &[LogEvent]) -> Vec<LogOffset> {
        self.append_with(ONE_SEGMENT, events)
    }

    fn append_with(&self, config: WalConfig, events: &[LogEvent]) -> Vec<LogOffset> {
        let mut writer =
            WalWriter::open_with_system_clock(&self.wal_dir(), config).expect("open wal");
        events
            .iter()
            .map(|event| writer.append(event).expect("append event"))
            .collect()
    }

    /// A log over the directory that can be replayed from.
    fn log(&self, config: WalConfig) -> SharedLog {
        let writer = WalWriter::open_with_system_clock(&self.wal_dir(), config).expect("open wal");
        let reader = SequentialScanReader::local(&self.wal_dir()).expect("open reader");
        let bloom = HashBloom::new(BloomConfig::default());
        let log = EventLog::new(Box::new(writer), bloom).with_reader(Arc::new(reader));
        Arc::new(tokio::sync::Mutex::new(log))
    }

    async fn open<B: Backend>(&self, db: &str, clock: Arc<FixedClock>) -> (Graph<B>, SharedLog) {
        self.open_over(self.log(ONE_SEGMENT), db, clock).await
    }

    async fn open_over<B: Backend>(
        &self,
        log: SharedLog,
        db: &str,
        clock: Arc<FixedClock>,
    ) -> (Graph<B>, SharedLog) {
        let graph = Graph::<B>::open_with_clock(&self.db(db), GraphConfig::new(8), clock)
            .await
            .expect("open graph")
            .with_log(Arc::clone(&log))
            .await
            .expect("attach log");
        (graph, log)
    }
}

fn clock_at(t: i64) -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(t)))
}

// ---- events, built the way the write path logs them ----

fn node(id: &str, content: &str) -> NodeRow {
    NodeRow {
        id: id.into(),
        kind: "fact".into(),
        label: "label".into(),
        content: content.into(),
        producer: "agent-a".into(),
        attributes: "{}".into(),
        scope: Some("proj/a".into()),
        subject: None,
        confidence: 0.75,
        valid_from: 500,
        valid_from_supplied: true,
        valid_until: FOREVER.0,
        tx_from: 1000,
        tx_to: FOREVER.0,
    }
}

fn edge(id: &str, src: &str, dst: &str) -> EdgeRow {
    EdgeRow {
        id: id.into(),
        src: src.into(),
        dst: dst.into(),
        edge_type: "mentions".into(),
        attributes: "{}".into(),
        tx_from: 1000,
        tx_to: FOREVER.0,
    }
}

fn event(event_id: &str, content_hash: [u8; 32], payload: LogPayload) -> LogEvent {
    LogEvent {
        event_id: event_id.into(),
        content_hash,
        source: "agent-a".into(),
        trust_score: 0.75,
        observed_at: 500,
        ingested_at: 1000,
        encryption_key_id: None,
        schema_version: CURRENT_SCHEMA_VERSION,
        payload,
    }
}

fn node_write(id: &str, content: &str) -> LogEvent {
    let row = node(id, content);
    event(
        &format!("event-{id}"),
        node_row_hash(&row),
        LogPayload::NodeWrite(row),
    )
}

fn edge_write(id: &str, src: &str, dst: &str) -> LogEvent {
    let row = edge(id, src, dst);
    event(
        &format!("event-{id}"),
        edge_row_hash(&row),
        LogPayload::EdgeWrite(row),
    )
}

fn episode(id: &str, nodes: &[NodeRow], edges: &[EdgeRow]) -> LogEvent {
    let effects: Vec<RowEffect> = nodes
        .iter()
        .cloned()
        .map(RowEffect::Node)
        .chain(edges.iter().cloned().map(RowEffect::Edge))
        .collect();
    let hash = node_row_hash(&nodes[0]);
    event(
        &format!("event-{id}"),
        hash,
        LogPayload::EpisodeBatch(effects),
    )
}

fn follow_up(event_id: &str, like: &LogEvent, payload: LogPayload) -> LogEvent {
    event(event_id, like.content_hash, payload)
}

// ---- reading the store back ----

async fn dump<B: Backend>(g: &Graph<B>, table: &str) -> String {
    let rows = g
        .backend
        .query(&format!("SELECT * FROM {table} ORDER BY 1"), &[])
        .await
        .unwrap();
    format!("{rows:?}")
}

async fn tx_to<B: Backend>(g: &Graph<B>, id: &str) -> i64 {
    let rows = g
        .backend
        .query("SELECT tx_to FROM nodes WHERE id = ?1", &[id.into()])
        .await
        .unwrap();
    rows.first().expect("node row").get_i64(0).unwrap()
}

async fn has_node<B: Backend>(g: &Graph<B>, id: &str) -> bool {
    let rows = g
        .backend
        .query("SELECT 1 FROM nodes WHERE id = ?1", &[id.into()])
        .await
        .unwrap();
    !rows.is_empty()
}

async fn cursor_offset<B: Backend>(g: &Graph<B>) -> Option<(i64, i64)> {
    cursor(g).await.and_then(|(_, offset)| offset)
}

async fn quarantined<B: Backend>(g: &Graph<B>) -> Vec<(String, i64, i64, String)> {
    g.backend
        .query(
            "SELECT event_id, segment, seg_index, reason FROM log_quarantine ORDER BY event_id",
            &[],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| {
            (
                row.get_string(0).unwrap(),
                row.get_i64(1).unwrap(),
                row.get_i64(2).unwrap(),
                row.get_string(3).unwrap(),
            )
        })
        .collect()
}

fn applied(applied: usize) -> CatchUp {
    CatchUp {
        applied,
        ..CatchUp::default()
    }
}

// ---- tests ----

#[tokio::test]
async fn catch_up_applies_logged_node_edge_and_episode_events_once_after_a_crash() {
    // Arrange: events that reached the log but never the projection
    let env = Env::new();
    let (c, d) = (node("node-c", "gamma"), node("node-d", "delta"));
    let offsets = env.append(&[
        node_write("node-a", "alpha"),
        node_write("node-b", "beta"),
        edge_write("edge-ab", "node-a", "node-b"),
        episode("batch", &[c, d], &[edge("edge-cd", "node-c", "node-d")]),
    ]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(report, applied(4));
    assert_eq!(count(&g, "nodes").await, 4);
    assert_eq!(count(&g, "edges").await, 2);
    assert_eq!(count(&g, "log_hash_index").await, 6);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_run_twice_applies_nothing_the_second_time_and_keeps_the_cursor() {
    // Arrange
    let env = Env::new();
    let offsets = env.append(&[node_write("node-a", "alpha"), node_write("node-b", "beta")]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    let first = g.catch_up().await.unwrap();
    let before = (
        dump(&g, "nodes").await,
        dump(&g, "log_cursor").await,
        dump(&g, "log_hash_index").await,
    );

    // Act
    let second = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(first, applied(2));
    assert_eq!(second, CatchUp::default());
    let after = (
        dump(&g, "nodes").await,
        dump(&g, "log_cursor").await,
        dump(&g, "log_hash_index").await,
    );
    assert_eq!(after, before);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_skips_a_voided_event_and_the_cursor_moves_past_it() {
    // Arrange
    let env = Env::new();
    let doomed = node_write("node-a", "alpha");
    let void = follow_up(
        "event-void",
        &doomed,
        LogPayload::Voided {
            target_event_id: doomed.event_id.clone(),
        },
    );
    let offsets = env.append(&[doomed, void, node_write("node-b", "beta")]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.skipped_voided, 1);
    assert_eq!(report.quarantined, 0);
    assert!(
        !has_node(&g, "node-a").await,
        "a voided write is never applied"
    );
    assert!(has_node(&g, "node-b").await);
    assert_eq!(count(&g, "log_hash_index").await, 1);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_passes_duplicate_of_and_tombstone_records_without_projecting_them() {
    // Arrange
    let env = Env::new();
    let first = node_write("node-a", "alpha");
    let duplicate = follow_up(
        "event-duplicate",
        &first,
        LogPayload::DuplicateOf {
            first_event_id: first.event_id.clone(),
        },
    );
    let tombstone = follow_up(
        "event-tombstone",
        &first,
        LogPayload::Tombstone(vec![TombstoneTarget {
            table: TombstoneTable::Nodes,
            id: "node-a".into(),
        }]),
    );
    let offsets = env.append(&[first, duplicate, tombstone]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.quarantined, 0);
    assert_eq!(report.skipped_voided, 0);
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(count(&g, "log_hash_index").await, 1);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_replays_a_supersede_into_the_same_rows_as_the_live_write() {
    // Arrange: a live store writes a node and supersedes it, so the log holds an
    // insert and an episode batch with the `supersedes` edge.
    let env = Env::new();
    let clock = clock_at(1000);
    let (live, _live_log) = env
        .open::<DefaultBackend>("live.db", Arc::clone(&clock))
        .await;
    let old = live
        .insert(fact_at("v1").with_subject("subject"))
        .await
        .unwrap();
    clock.set(Millis(2000));
    let new = live
        .upsert_by(fact_at("v2").with_subject("subject"))
        .await
        .unwrap();
    let (replayed, _log) = env
        .open::<DefaultBackend>("replayed.db", clock_at(9000))
        .await;

    // Act
    let report = replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(report, applied(2));
    assert_eq!(tx_to(&replayed, old.as_str()).await, 2000);
    assert_eq!(tx_to(&replayed, new.as_str()).await, FOREVER.0);
    assert_eq!(dump(&replayed, "nodes").await, dump(&live, "nodes").await);
    assert_eq!(dump(&replayed, "edges").await, dump(&live, "edges").await);
    assert_eq!(
        dump(&replayed, "log_hash_index").await,
        dump(&live, "log_hash_index").await
    );
}

#[tokio::test]
async fn catch_up_does_not_insert_rows_the_store_already_holds() {
    // Arrange: a live store that projected an insert and a supersede, with its
    // cursor put back as if the cursor write had been lost.
    let env = Env::new();
    let clock = clock_at(1000);
    let (g, _log) = env
        .open::<DefaultBackend>("graph.db", Arc::clone(&clock))
        .await;
    g.insert(fact_at("v1").with_subject("subject"))
        .await
        .unwrap();
    clock.set(Millis(2000));
    g.upsert_by(fact_at("v2").with_subject("subject"))
        .await
        .unwrap();
    let at_head = cursor_offset(&g).await;
    g.backend
        .execute(
            "UPDATE log_cursor SET last_segment = NULL, last_index = NULL",
            &[],
        )
        .await
        .unwrap();
    let before = (
        dump(&g, "nodes").await,
        dump(&g, "edges").await,
        dump(&g, "log_hash_index").await,
    );

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUp {
        skipped_applied: 2,
        ..CatchUp::default()
    };
    assert_eq!(report, expected);
    let after = (
        dump(&g, "nodes").await,
        dump(&g, "edges").await,
        dump(&g, "log_hash_index").await,
    );
    assert_eq!(after, before);
    assert_eq!(cursor_offset(&g).await, at_head);
}

#[tokio::test]
async fn catch_up_quarantines_a_refused_event_and_applies_the_ones_after_it() {
    // Arrange: the edge points at a node the log never wrote
    let env = Env::new();
    let refused = edge_write("edge-ghost", "node-a", "node-ghost");
    let refused_id = refused.event_id.clone();
    let offsets = env.append(&[
        node_write("node-a", "alpha"),
        refused,
        node_write("node-b", "beta"),
    ]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUp {
        applied: 2,
        quarantined: 1,
        ..CatchUp::default()
    };
    assert_eq!(report, expected);
    assert!(
        has_node(&g, "node-b").await,
        "events after the refused one still apply"
    );
    assert_eq!(count(&g, "edges").await, 0);
    let held = quarantined(&g).await;
    assert_eq!(held.len(), 1);
    let (event_id, segment, index, reason) = &held[0];
    assert_eq!(event_id, &refused_id);
    assert_eq!((*segment, *index), offset_pair(offsets[1]));
    assert!(
        !reason.is_empty(),
        "the quarantine names why the event was refused"
    );
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_returns_a_transient_backend_error_and_resumes_from_the_last_applied_event() {
    // Arrange: a node write, then a four node episode. The fault fires at each
    // transaction's fourth statement, which only the episode reaches.
    let env = Env::new();
    let nodes: Vec<NodeRow> = (0..4)
        .map(|n| node(&format!("batch-{n}"), &format!("content-{n}")))
        .collect();
    let offsets = env.append(&[node_write("node-a", "alpha"), episode("batch", &nodes, &[])]);
    let (g, _log) = env.open::<FailingBackend>("graph.db", clock_at(9000)).await;
    g.backend.set_fail_on_execute(3);

    // Act
    let failed = g.catch_up().await;

    // Assert: the first event is applied and stays so, the second left no trace
    let error = failed.expect_err("a backend failure is not swallowed");
    assert!(matches!(error, Error::Backend(_)), "{error:?}");
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(offsets[0])));
    assert!(
        quarantined(&g).await.is_empty(),
        "a transient error is not a refusal"
    );

    // Act: the backend recovers
    g.backend.set_fail_on_execute(usize::MAX);
    let retried = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(retried, applied(1));
    assert_eq!(count(&g, "nodes").await, 5);
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(offsets[1])));
}

#[tokio::test]
async fn catch_up_indexes_replayed_rows_so_a_later_identical_insert_deduplicates() {
    // Arrange: a live store writes the node, a fresh one replays it
    let env = Env::new();
    let (live, _live_log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
    let first = live.insert(fact_at("same")).await.unwrap();
    let hash = live
        .backend
        .query("SELECT content_hash FROM log_hash_index", &[])
        .await
        .unwrap()[0]
        .get_blob(0)
        .unwrap()
        .to_vec();
    let hash: [u8; 32] = hash.try_into().expect("a 32 byte hash");
    let (replayed, log) = env
        .open::<DefaultBackend>("replayed.db", clock_at(9000))
        .await;

    // Act
    replayed.catch_up().await.unwrap();
    let again = replayed.insert(fact_at("same")).await.unwrap();

    // Assert
    assert!(log.lock().await.bloom_might_contain(&hash));
    assert_eq!(
        dump(&replayed, "log_hash_index").await,
        dump(&live, "log_hash_index").await
    );
    assert_eq!(
        again, first,
        "the replayed row is the live carrier of the hash"
    );
    assert_eq!(count(&replayed, "nodes").await, 1);
}

#[tokio::test]
async fn catch_up_does_not_read_past_the_writer_head() {
    // Arrange: the store's writer acknowledged two events, then a third landed
    // in the same segment behind its back.
    let env = Env::new();
    let offsets = env.append(&[node_write("node-a", "alpha"), node_write("node-b", "beta")]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    env.append(&[node_write("node-c", "gamma")]);

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(report, applied(2));
    assert!(!has_node(&g, "node-c").await);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_without_a_log_is_a_no_op() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(report, CatchUp::default());
    assert_eq!(count(&g, "log_cursor").await, 0);
    assert_eq!(count(&g, "nodes").await, 0);
}

#[tokio::test]
async fn catch_up_replays_a_whole_multi_segment_log_into_a_fresh_store() {
    // Arrange: three events in three segments, and a store whose cursor has no
    // offset yet
    let env = Env::new();
    let offsets = env.append_with(
        SEGMENT_PER_EVENT,
        &[
            node_write("node-a", "alpha"),
            node_write("node-b", "beta"),
            node_write("node-c", "gamma"),
        ],
    );
    let log = env.log(SEGMENT_PER_EVENT);
    let (g, _log) = env
        .open_over::<DefaultBackend>(log, "graph.db", clock_at(9000))
        .await;
    let fresh = cursor_offset(&g).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(fresh, None);
    assert_eq!(report, applied(3));
    assert_eq!(count(&g, "nodes").await, 3);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_waits_for_the_log_lock_that_writers_take() {
    // Arrange
    let env = Env::new();
    env.append(&[node_write("node-a", "alpha")]);
    let (g, log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    let held = log.lock().await;

    // Act
    let blocked = tokio::time::timeout(Duration::from_millis(200), g.catch_up()).await;
    drop(held);
    let report = g.catch_up().await.unwrap();

    // Assert
    assert!(blocked.is_err(), "catch_up ran while a writer held the log");
    assert_eq!(report, applied(1));
}
