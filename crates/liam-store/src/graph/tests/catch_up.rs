// SPDX-License-Identifier: Apache-2.0
//! Replay of the event log into the projection. Tests run against a real WAL
//! and reader in a temp directory: an event that was logged but never projected
//! is appended with a standalone writer, as a crash between the append and the
//! projection leaves it.

use std::io;
use std::time::Duration;

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::event::{
    EdgeRow, LogEvent, LogPayload, NodeRow, RowEffect, TombstoneTable, TombstoneTarget,
};
use liam_log::hash::{edge_row_hash, node_row_hash};
use liam_log::reader::{LogRecord, SequentialScanReader};
use liam_log::wal::{WalConfig, WalError, WalWriter};
use liam_log::{LogOffset, LogWriter};

use super::support::{
    count, cursor, event, fact, fact_at, has_node, offset_pair, record_counts, shared, Env,
    ReembedProbe, StubEmbedder, ONE_SEGMENT,
};
use super::*;
use crate::DefaultBackend;

/// Rotates after every append, so each event sits in a segment of its own.
const SEGMENT_PER_EVENT: WalConfig = WalConfig {
    segment_max_bytes: 1,
    rotate_interval_secs: 3600,
};

/// A writer that reports the log's head but fails every append, as a full disk
/// would leave it.
struct FullDisk(WalWriter<liam_log::wal::SystemClock>);

impl LogWriter for FullDisk {
    fn append(&mut self, _event: &LogEvent) -> std::result::Result<LogOffset, WalError> {
        Err(WalError::Io(io::Error::other("disk full")))
    }

    fn log_id(&self) -> uuid::Uuid {
        self.0.log_id()
    }

    fn head(&self) -> Option<LogOffset> {
        self.0.head()
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

/// A `supersedes` edge written on its own: it closes `dst`, then inserts the
/// edge with no guard on `src`.
fn supersedes_write(id: &str, src: &str, dst: &str) -> LogEvent {
    let mut row = edge(id, src, dst);
    row.edge_type = relation::SUPERSEDES.into();
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

fn tombstone(event_id: &str, targets: &[(TombstoneTable, &str)]) -> LogEvent {
    let targets = targets
        .iter()
        .map(|(table, id)| TombstoneTarget {
            table: *table,
            id: (*id).into(),
        })
        .collect();
    event(event_id, [0; 32], LogPayload::Tombstone(targets))
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

async fn lose_cursor<B: Backend>(g: &Graph<B>) {
    g.backend
        .execute(
            "UPDATE log_cursor SET last_segment = NULL, last_index = NULL",
            &[],
        )
        .await
        .unwrap();
}

async fn set_community<B: Backend>(g: &Graph<B>, node_id: &NodeId) {
    g.backend
        .execute(
            "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, 1, 1)",
            &[node_id.as_str().into()],
        )
        .await
        .unwrap();
}

/// The tables a replay writes, as one comparable value.
async fn snapshot<B: Backend>(g: &Graph<B>) -> [String; 3] {
    [
        dump(g, "nodes").await,
        dump(g, "edges").await,
        dump(g, "log_hash_index").await,
    ]
}

/// Appends `tombstone` behind a store that has logged its writes, lets the store
/// apply it, then puts the cursor back to the start, as a lost cursor write
/// would. Returns the store, what it held before the replay, and the replay's
/// report.
async fn replay_over_an_applied_tombstone(
    env: &Env,
    tombstone: LogEvent,
) -> (Graph<DefaultBackend>, [String; 3], CatchUpReport) {
    env.append(&[tombstone]);
    let (g, _log) = env.open::<DefaultBackend>("live.db", clock_at(9000)).await;
    g.catch_up().await.unwrap();
    let settled = snapshot(&g).await;
    lose_cursor(&g).await;
    let report = g.catch_up().await.unwrap();
    (g, settled, report)
}

/// Asserts that a replay over an applied tombstone changed nothing and
/// refused nothing.
async fn assert_replayed_without_loss(
    env: &Env,
    g: &Graph<DefaultBackend>,
    settled: &[String; 3],
    report: CatchUpReport,
    expected: CatchUpReport,
) {
    assert_eq!(record_counts(report), expected);
    assert!(quarantined(g).await.is_empty(), "nothing is refused");
    assert!(
        voided_targets_of(&env.records().await).is_empty(),
        "nothing is voided in the log"
    );
    assert_eq!(&snapshot(g).await, settled);
}
async fn set_cursor<B: Backend>(g: &Graph<B>, offset: LogOffset) {
    let (segment, index) = offset_pair(offset);
    g.backend
        .execute(
            "UPDATE log_cursor SET last_segment = ?1, last_index = ?2",
            &[segment.into(), index.into()],
        )
        .await
        .unwrap();
}

/// The events the `Voided` records among `records` cancel.
fn voided_targets_of(records: &[LogRecord]) -> Vec<String> {
    records
        .iter()
        .filter_map(|record| match &record.event.payload {
            LogPayload::Voided { target_event_id } => Some(target_event_id.clone()),
            _ => None,
        })
        .collect()
}

fn applied(applied: usize) -> CatchUpReport {
    CatchUpReport {
        applied,
        ..CatchUpReport::default()
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
    assert_eq!(record_counts(report), applied(4));
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
    assert_eq!(record_counts(first), applied(2));
    assert_eq!(record_counts(second), CatchUpReport::default());
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
    let expected = CatchUpReport {
        applied: 1,
        skipped_voided: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
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
async fn catch_up_passes_a_duplicate_of_record_and_applies_a_tombstone() {
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
    assert_eq!(record_counts(report), applied(2));
    assert_eq!(count(&g, "nodes").await, 0);
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
    assert_eq!(record_counts(report), applied(2));
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
    let expected = CatchUpReport {
        already_applied: 2,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
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
    let expected = CatchUpReport {
        applied: 2,
        quarantined: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
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
        reason.contains("node-ghost"),
        "the quarantine names what the event was refused over: {reason}"
    );
    let records = env.records().await;
    assert_eq!(
        records.len(),
        4,
        "the three events and the void behind them"
    );
    assert_eq!(
        voided_targets_of(&records),
        vec![refused_id],
        "the log records the refusal as a void, as a live failure does"
    );
    let void_at = records[3].offset;
    assert_eq!(void_at.index, offsets[2].index + 1);
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(void_at)));

    // Act
    let again = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(again), CatchUpReport::default());
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(void_at)));

    // Act: a crash left the cursor behind the void
    set_cursor(&g, offsets[2]).await;
    let after_crash = g.catch_up().await.unwrap();

    // Assert: the void is passed, not read as work and not counted
    assert_eq!(record_counts(after_crash), CatchUpReport::default());
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(void_at)));
}

#[tokio::test]
async fn catch_up_returns_a_transient_backend_error_and_resumes_from_the_last_applied_event() {
    // Arrange: a node write, then a four node episode whose last node insert
    // is the statement that fails.
    let env = Env::new();
    let nodes: Vec<NodeRow> = (0..4)
        .map(|n| node(&format!("batch-{n}"), &format!("content-{n}")))
        .collect();
    let offsets = env.append(&[node_write("node-a", "alpha"), episode("batch", &nodes, &[])]);
    let (g, _log) = env.open::<FailingBackend>("graph.db", clock_at(9000)).await;
    g.backend.set_fail_on_row(Some("batch-3"));

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
    g.backend.set_fail_on_row(None);
    let retried = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(retried), applied(1));
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
    assert_eq!(record_counts(report), applied(2));
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
    assert_eq!(report, CatchUpReport::default());
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
    assert_eq!(record_counts(report), applied(3));
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
    assert_eq!(record_counts(report), applied(1));
}

#[tokio::test]
async fn catch_up_refuses_a_poisoned_log_and_applies_nothing() {
    // Arrange: a log with an unprojected event whose tail is no longer known
    let env = Env::new();
    env.append(&[node_write("node-a", "alpha")]);
    let (g, log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    log.lock().await.poison();

    // Act
    let refused = g.catch_up().await;

    // Assert
    assert!(matches!(refused, Err(Error::LogPoisoned)), "{refused:?}");
    assert_eq!(count(&g, "nodes").await, 0);
    assert_eq!(cursor_offset(&g).await, None);
}

#[tokio::test]
async fn catch_up_voids_a_quarantined_event_so_a_store_rebuilt_from_the_log_skips_it_too() {
    // Arrange
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        edge_write("edge-ghost", "node-a", "node-ghost"),
        node_write("node-b", "beta"),
    ]);
    let (first, _log) = env.open::<DefaultBackend>("first.db", clock_at(9000)).await;
    first.catch_up().await.unwrap();
    let (rebuilt, _log) = env
        .open::<DefaultBackend>("rebuilt.db", clock_at(9000))
        .await;

    // Act
    let report = rebuilt.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 2,
        skipped_voided: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
    assert_eq!(dump(&rebuilt, "nodes").await, dump(&first, "nodes").await);
    assert_eq!(dump(&rebuilt, "edges").await, dump(&first, "edges").await);
    assert!(
        quarantined(&rebuilt).await.is_empty(),
        "only the store that refused the event holds the refusal"
    );
}

#[tokio::test]
async fn catch_up_moves_the_cursor_onto_the_void_when_the_refused_event_is_the_last_record() {
    // Arrange
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        edge_write("edge-ghost", "node-a", "node-ghost"),
    ]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let first = g.catch_up().await.unwrap();
    let second = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 1,
        quarantined: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(first), expected);
    assert_eq!(record_counts(second), CatchUpReport::default());
    let records = env.records().await;
    let void_at = records.last().expect("the void").offset;
    assert!(matches!(
        records.last().map(|r| &r.event.payload),
        Some(LogPayload::Voided { .. })
    ));
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(void_at)));
}

#[tokio::test]
async fn catch_up_quarantines_a_foreign_key_violation_and_leaves_no_trace_of_the_event() {
    // Arrange: the edge closes node-a, then points at a node the log never wrote
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        supersedes_write("edge-fk", "node-ghost", "node-a"),
        node_write("node-b", "beta"),
    ]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 2,
        quarantined: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
    assert_eq!(
        tx_to(&g, "node-a").await,
        FOREVER.0,
        "the close rolled back with the refused edge"
    );
    assert_eq!(count(&g, "edges").await, 0);
    let held = quarantined(&g).await;
    assert_eq!(held.len(), 1);
    assert!(held[0].3.contains("constraint"), "{}", held[0].3);
    assert!(has_node(&g, "node-b").await);
}

#[tokio::test]
async fn catch_up_quarantines_a_supersede_of_a_node_the_log_never_wrote() {
    // Arrange
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        supersedes_write("edge-gone", "node-a", "node-ghost"),
        node_write("node-b", "beta"),
    ]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 2,
        quarantined: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
    let held = quarantined(&g).await;
    assert_eq!(held.len(), 1);
    assert!(held[0].3.contains("node-ghost"), "{}", held[0].3);
}

#[tokio::test]
async fn catch_up_quarantines_an_event_the_store_holds_only_part_of() {
    // Arrange: an episode of two nodes was replayed, then one node was lost and
    // the cursor with it
    let env = Env::new();
    let nodes = [node("batch-0", "zero"), node("batch-1", "one")];
    env.append(&[episode("batch", &nodes, &[]), node_write("node-b", "beta")]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    g.catch_up().await.unwrap();
    g.backend
        .execute("DELETE FROM nodes WHERE id = 'batch-1'", &[])
        .await
        .unwrap();
    g.backend
        .execute(
            "UPDATE log_cursor SET last_segment = NULL, last_index = NULL",
            &[],
        )
        .await
        .unwrap();

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        already_applied: 1,
        quarantined: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
    let held = quarantined(&g).await;
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].0, "event-batch");
    assert!(held[0].3.contains("event-batch"), "{}", held[0].3);
    assert!(has_node(&g, "batch-0").await, "the rows it holds stay");
    assert!(
        !has_node(&g, "batch-1").await,
        "nothing re-inserts the lost row"
    );
}

#[tokio::test]
async fn catch_up_leaves_the_cursor_on_a_voided_event_when_the_next_one_fails() {
    // Arrange: the target, an event that fails to project, then the target's void
    let env = Env::new();
    let target = node_write("node-a", "alpha");
    let void = follow_up(
        "event-void",
        &target,
        LogPayload::Voided {
            target_event_id: target.event_id.clone(),
        },
    );
    let offsets = env.append(&[target, node_write("node-x", "chi"), void]);
    let (g, _log) = env.open::<FailingBackend>("graph.db", clock_at(9000)).await;
    g.backend.set_fail_on_row(Some("node-x"));

    // Act
    let failed = g.catch_up().await;

    // Assert
    let error = failed.expect_err("the failing event stops replay");
    assert!(matches!(error, Error::Backend(_)), "{error:?}");
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(offsets[0])));
    assert_eq!(count(&g, "nodes").await, 0);

    // Act: the backend recovers
    g.backend.set_fail_on_row(None);
    let retried = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(retried), applied(1));
    assert!(has_node(&g, "node-x").await);
    assert!(!has_node(&g, "node-a").await);
}

#[tokio::test]
async fn catch_up_skips_an_event_whose_void_is_in_a_later_segment() {
    // Arrange: three segments, the void two segments after its target
    let env = Env::new();
    let target = node_write("node-a", "alpha");
    let void = follow_up(
        "event-void",
        &target,
        LogPayload::Voided {
            target_event_id: target.event_id.clone(),
        },
    );
    let offsets = env.append_with(
        SEGMENT_PER_EVENT,
        &[target, node_write("node-b", "beta"), void],
    );
    let log = env.log(SEGMENT_PER_EVENT);
    let (g, _log) = env
        .open_over::<DefaultBackend>(log, "graph.db", clock_at(9000))
        .await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 1,
        skipped_voided: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
    assert!(!has_node(&g, "node-a").await);
    assert!(has_node(&g, "node-b").await);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_replays_a_live_episode_relate_and_link_into_the_same_rows() {
    // Arrange
    let env = Env::new();
    let (live, _live_log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
    let mentions = EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::New(1),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({"weight": 2}),
    };
    let episode = live
        .ingest_episode(vec![fact_at("first"), fact_at("second")], vec![mentions])
        .await
        .unwrap();
    live.relate(&episode.node_ids[1], &episode.node_ids[0], "supports")
        .await
        .unwrap();
    let cites = NewEdge::new(&episode.node_ids[0], &episode.node_ids[1], "cites")
        .with_attributes(serde_json::json!({"page": 4, "quote": "same"}));
    live.link(cites).await.unwrap();
    let (replayed, _log) = env
        .open::<DefaultBackend>("replayed.db", clock_at(9000))
        .await;

    // Act
    let report = replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(report), applied(3));
    assert_eq!(count(&replayed, "nodes").await, 2);
    assert_eq!(count(&replayed, "edges").await, 3);
    for table in ["nodes", "edges", "log_hash_index"] {
        assert_eq!(dump(&replayed, table).await, dump(&live, table).await);
    }
}

#[tokio::test]
async fn catch_up_replays_a_node_written_without_a_valid_time_as_the_live_write_stored_it() {
    // Arrange
    let env = Env::new();
    let (live, _live_log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
    live.insert(fact("no valid time")).await.unwrap();
    let (replayed, _log) = env
        .open::<DefaultBackend>("replayed.db", clock_at(9000))
        .await;

    // Act
    let report = replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(report), applied(1));
    assert_eq!(dump(&replayed, "nodes").await, dump(&live, "nodes").await);
    let valid_from = replayed
        .backend
        .query("SELECT valid_from FROM nodes", &[])
        .await
        .unwrap()[0]
        .get_i64(0)
        .unwrap();
    assert_eq!(valid_from, 1000, "the live write's time, not the replay's");
}

#[tokio::test]
async fn catch_up_poisons_the_log_when_the_void_of_a_refused_event_cannot_be_appended() {
    // Arrange
    let env = Env::new();
    let offsets = env.append(&[
        node_write("node-a", "alpha"),
        edge_write("edge-ghost", "node-a", "node-ghost"),
    ]);
    let wal = WalWriter::open_with_system_clock(&env.wal_dir(), ONE_SEGMENT).expect("open wal");
    let reader = SequentialScanReader::local(&env.wal_dir()).expect("open reader");
    let bloom = HashBloom::new(BloomConfig::default());
    let log = shared(EventLog::new(
        Box::new(FullDisk(wal)),
        Arc::new(reader),
        bloom,
    ));
    let (g, log) = env
        .open_over::<DefaultBackend>(log, "graph.db", clock_at(9000))
        .await;

    // Act
    let failed = g.catch_up().await;
    let retried = g.catch_up().await;

    // Assert
    assert!(matches!(failed, Err(Error::LogAppend(_))), "{failed:?}");
    assert!(log.lock().await.is_poisoned());
    assert!(matches!(retried, Err(Error::LogPoisoned)), "{retried:?}");
    assert!(
        quarantined(&g).await.is_empty(),
        "an event is not recorded as refused while its void is missing"
    );
    assert_eq!(cursor_offset(&g).await, Some(offset_pair(offsets[0])));
}

#[tokio::test]
async fn catch_up_teaches_the_dedup_filter_the_hashes_of_rows_the_store_already_holds() {
    // Arrange: a store that holds the row but whose hash index does not
    let env = Env::new();
    let event = node_write("node-a", "alpha");
    let hash = event.content_hash;
    env.append(&[event]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    g.catch_up().await.unwrap();
    g.backend
        .execute_batch(
            "DELETE FROM log_hash_index;
             UPDATE log_cursor SET last_segment = NULL, last_index = NULL;",
        )
        .await
        .unwrap();
    drop(g);
    let (g, log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    assert!(!log.lock().await.bloom_might_contain(&hash));

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        already_applied: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
    assert!(log.lock().await.bloom_might_contain(&hash));
}

#[tokio::test]
async fn catch_up_embeds_a_replayed_node_and_does_not_embed_it_again() {
    // Arrange: a node the log carries without a vector
    let env = Env::new();
    env.append(&[node_write("node-a", "alpha")]);
    let embedder = StubEmbedder::new(8);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    let g = g.with_embedder(embedder.clone());

    // Act
    let first = g.catch_up().await.unwrap();
    let second = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 1,
        reembedded: ReembedReport {
            re_embedded: 1,
            ..ReembedReport::default()
        },
        ..CatchUpReport::default()
    };
    assert_eq!(first, expected);
    assert_eq!(second, CatchUpReport::default());
    assert_eq!(count(&g, "node_vectors").await, 1);
    assert_eq!(embedder.calls(), ["alpha"]);
}

#[tokio::test]
async fn catch_up_reports_a_node_it_could_not_embed_and_still_replays_every_event() {
    // Arrange
    let env = Env::new();
    let offsets = env.append(&[node_write("node-a", "alpha"), node_write("node-b", "beta")]);
    let embedder = StubEmbedder::new(8);
    embedder.fail_on(Some("alpha"));
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    let g = g.with_embedder(embedder.clone());

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 2,
        reembedded: ReembedReport {
            re_embedded: 1,
            failed: 1,
            ..ReembedReport::default()
        },
        ..CatchUpReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(
        cursor_offset(&g).await,
        offsets.last().map(|o| offset_pair(*o))
    );
}

#[tokio::test]
async fn catch_up_embeds_nodes_replayed_by_an_earlier_run_that_never_got_to_it() {
    // Arrange: a run that replayed the node with no embedder at hand
    let env = Env::new();
    env.append(&[node_write("node-a", "alpha")]);
    let (first, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    first.catch_up().await.unwrap();
    drop(first);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    let g = g.with_embedder(StubEmbedder::new(8));

    // Act: nothing is left to replay
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        reembedded: ReembedReport {
            re_embedded: 1,
            ..ReembedReport::default()
        },
        ..CatchUpReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(count(&g, "node_vectors").await, 1);
}

#[tokio::test]
async fn catch_up_without_an_embedder_reports_the_nodes_left_without_a_vector() {
    // Arrange
    let env = Env::new();
    env.append(&[node_write("node-a", "alpha")]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        applied: 1,
        reembedded: ReembedReport {
            pending: 1,
            ..ReembedReport::default()
        },
        ..CatchUpReport::default()
    };
    assert_eq!(report, expected);
}

#[tokio::test]
async fn catch_up_returns_the_replay_report_when_the_nodes_cannot_be_listed() {
    // Arrange
    let env = Env::new();
    env.append(&[node_write("node-a", "alpha")]);
    let (g, _log) = env.open::<ReembedProbe>("graph.db", clock_at(9000)).await;
    let g = g.with_embedder(StubEmbedder::new(8));
    g.backend.fail_listing();

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert: the replayed event is reported and no embedding is claimed
    let expected = CatchUpReport {
        applied: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(count(&g, "nodes").await, 1);
}

#[tokio::test]
async fn catch_up_applies_a_repeated_node_tombstone_and_takes_its_edges_vector_and_community_row_with_it(
) {
    // Arrange: a store with a vector and a community row, then a log that
    // tombstones the node twice behind its back
    let env = Env::new();
    let (a, b) = {
        let (first, _log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
        let first = first.with_embedder(StubEmbedder::new(8));
        let a = first.insert(fact_at("a")).await.unwrap();
        let b = first.insert(fact_at("b")).await.unwrap();
        first.relate(&a, &b, "mentions").await.unwrap();
        first.reembed_missing().await.unwrap();
        set_community(&first, &a).await;
        (a, b)
    };
    let gone = [(TombstoneTable::Nodes, a.as_str())];
    env.append(&[
        tombstone("event-tombstone", &gone),
        tombstone("event-tombstone-again", &gone),
    ]);
    let (live, _log) = env.open::<DefaultBackend>("live.db", clock_at(9000)).await;
    let live = live.with_embedder(StubEmbedder::new(8));

    // Act
    let report = live.catch_up().await.unwrap();

    // Assert: the second tombstone finds nothing left and is not an error
    assert_eq!(record_counts(report), applied(2));
    assert!(!has_node(&live, a.as_str()).await);
    assert!(has_node(&live, b.as_str()).await);
    assert_eq!(count(&live, "edges").await, 0);
    assert_eq!(count(&live, "node_community").await, 0);
    assert_eq!(count(&live, "node_vectors").await, 1);
}

#[tokio::test]
async fn catch_up_applies_an_edge_tombstone_and_a_community_tombstone_and_keeps_every_other_row() {
    // Arrange: three nodes, two edges, two community rows, then a log that
    // tombstones one edge and one community row behind the store's back
    let env = Env::new();
    let (a, b, first_edge) = {
        let (first, _log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
        let a = first.insert(fact_at("a")).await.unwrap();
        let b = first.insert(fact_at("b")).await.unwrap();
        let c = first.insert(fact_at("c")).await.unwrap();
        let first_edge = first.relate(&a, &b, "mentions").await.unwrap();
        first.relate(&b, &c, "mentions").await.unwrap();
        set_community(&first, &a).await;
        set_community(&first, &b).await;
        (a, b, first_edge)
    };
    env.append(&[
        tombstone(
            "event-edge",
            &[(TombstoneTable::Edges, first_edge.as_str())],
        ),
        tombstone(
            "event-community",
            &[(TombstoneTable::NodeCommunity, a.as_str())],
        ),
    ]);
    let (g, _log) = env.open::<DefaultBackend>("live.db", clock_at(9000)).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(report), applied(2));
    assert_eq!(count(&g, "nodes").await, 3);
    let edges = g
        .backend
        .query("SELECT src, dst FROM edges", &[])
        .await
        .unwrap();
    assert_eq!(edges.len(), 1, "only the tombstoned edge is gone");
    assert_eq!(edges[0].get_string(0).unwrap(), b.as_str());
    let communities = g
        .backend
        .query("SELECT node_id FROM node_community", &[])
        .await
        .unwrap();
    assert_eq!(communities.len(), 1, "only the tombstoned row is gone");
    assert_eq!(communities[0].get_string(0).unwrap(), b.as_str());
}

#[tokio::test]
async fn catch_up_replays_an_insert_a_relate_and_a_node_tombstone_into_an_empty_store_as_the_live_store_holds_them(
) {
    // Arrange: the live store applies the tombstone it did not write itself
    let env = Env::new();
    let a = {
        let (live, _log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
        let a = live.insert(fact_at("a")).await.unwrap();
        let b = live.insert(fact_at("b")).await.unwrap();
        live.relate(&a, &b, "mentions").await.unwrap();
        a
    };
    env.append(&[tombstone(
        "event-tombstone",
        &[(TombstoneTable::Nodes, a.as_str())],
    )]);
    let (live, _log) = env.open::<DefaultBackend>("live.db", clock_at(9000)).await;
    live.catch_up().await.unwrap();
    let (replayed, _log) = env
        .open::<DefaultBackend>("replayed.db", clock_at(9000))
        .await;

    // Act
    let report = replayed.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(report), applied(4));
    assert!(quarantined(&replayed).await.is_empty());
    assert_eq!(count(&replayed, "nodes").await, 1);
    assert_eq!(count(&replayed, "edges").await, 0);
    for table in ["nodes", "edges"] {
        assert_eq!(dump(&replayed, table).await, dump(&live, table).await);
    }
}

#[tokio::test]
async fn catch_up_does_not_quarantine_an_event_whose_supersedes_edge_a_later_tombstone_took_when_the_cursor_lags(
) {
    // Arrange: an upsert pair, then a tombstone of the old version, which takes
    // the `supersedes` edge with it
    let env = Env::new();
    let clock = clock_at(1000);
    let old = {
        let (live, _log) = env
            .open::<DefaultBackend>("live.db", Arc::clone(&clock))
            .await;
        let old = live
            .insert(fact_at("v1").with_subject("subject"))
            .await
            .unwrap();
        clock.set(Millis(2000));
        live.upsert_by(fact_at("v2").with_subject("subject"))
            .await
            .unwrap();
        old
    };
    let gone = tombstone("event-tombstone", &[(TombstoneTable::Nodes, old.as_str())]);

    // Act
    let (g, settled, report) = replay_over_an_applied_tombstone(&env, gone).await;

    // Assert: the node write and the tombstone run again, the batch is found held
    let expected = CatchUpReport {
        applied: 2,
        already_applied: 1,
        ..CatchUpReport::default()
    };
    assert_replayed_without_loss(&env, &g, &settled, report, expected).await;
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(count(&g, "edges").await, 0);
}

#[tokio::test]
async fn catch_up_does_not_quarantine_an_episode_whose_edge_a_later_tombstone_took_by_id() {
    // Arrange
    let env = Env::new();
    let edge = {
        let (live, _log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
        let mentions = EpisodeEdge {
            from: EpisodeRef::New(0),
            to: EpisodeRef::New(1),
            kind: "mentions".to_string(),
            attributes: serde_json::json!({}),
        };
        let episode = live
            .ingest_episode(vec![fact_at("x"), fact_at("y")], vec![mentions])
            .await
            .unwrap();
        episode.edge_ids[0].clone()
    };
    let gone = tombstone("event-tombstone", &[(TombstoneTable::Edges, edge.as_str())]);

    // Act
    let (g, settled, report) = replay_over_an_applied_tombstone(&env, gone).await;

    // Assert
    let expected = CatchUpReport {
        applied: 1,
        already_applied: 1,
        ..CatchUpReport::default()
    };
    assert_replayed_without_loss(&env, &g, &settled, report, expected).await;
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(count(&g, "edges").await, 0);
}

#[tokio::test]
async fn catch_up_does_not_quarantine_an_episode_whose_edge_starts_at_a_node_a_later_tombstone_took(
) {
    // Arrange: the edge runs from a node the log wrote earlier to a node of the
    // episode, and that earlier node is then tombstoned
    let env = Env::new();
    let tail = {
        let (live, _log) = env.open::<DefaultBackend>("live.db", clock_at(1000)).await;
        let tail = live.insert(fact_at("tail")).await.unwrap();
        let from_tail = EpisodeEdge {
            from: EpisodeRef::Existing(tail.clone()),
            to: EpisodeRef::New(0),
            kind: "mentions".to_string(),
            attributes: serde_json::json!({}),
        };
        live.ingest_episode(vec![fact_at("head")], vec![from_tail])
            .await
            .unwrap();
        tail
    };
    let gone = tombstone("event-tombstone", &[(TombstoneTable::Nodes, tail.as_str())]);

    // Act
    let (g, settled, report) = replay_over_an_applied_tombstone(&env, gone).await;

    // Assert
    let expected = CatchUpReport {
        applied: 2,
        already_applied: 1,
        ..CatchUpReport::default()
    };
    assert_replayed_without_loss(&env, &g, &settled, report, expected).await;
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(count(&g, "edges").await, 0);
}

#[tokio::test]
async fn catch_up_still_quarantines_a_half_written_event_when_the_tombstone_that_could_explain_it_was_voided(
) {
    // Arrange: a batch that supersedes node-a, replayed once and then left
    // without its edge, with a tombstone of node-a in the log that never ran
    let env = Env::new();
    let mut supersedes = edge("edge-s", "node-b", "node-a");
    supersedes.edge_type = relation::SUPERSEDES.into();
    let doomed = tombstone("event-tombstone", &[(TombstoneTable::Nodes, "node-a")]);
    let void = follow_up(
        "event-void",
        &doomed,
        LogPayload::Voided {
            target_event_id: doomed.event_id.clone(),
        },
    );
    env.append(&[
        node_write("node-a", "alpha"),
        episode("batch", &[node("node-b", "beta")], &[supersedes]),
        doomed,
        void,
    ]);
    let (g, _log) = env.open::<DefaultBackend>("graph.db", clock_at(9000)).await;
    g.catch_up().await.unwrap();
    g.backend
        .execute("DELETE FROM edges WHERE id = 'edge-s'", &[])
        .await
        .unwrap();
    lose_cursor(&g).await;

    // Act
    let report = g.catch_up().await.unwrap();

    // Assert
    let expected = CatchUpReport {
        already_applied: 1,
        skipped_voided: 1,
        quarantined: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report), expected);
    let held = quarantined(&g).await;
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].0, "event-batch");
    assert!(
        has_node(&g, "node-a").await,
        "the voided tombstone never ran"
    );
}
