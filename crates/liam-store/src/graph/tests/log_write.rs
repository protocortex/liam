// SPDX-License-Identifier: Apache-2.0
//! Log-first write path: every write appends to the injected log inside one
//! transaction. Tests name the log through `RecordingLog`, a double that keeps
//! what it accepted so a test can compare the log against the projection.

use std::io;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::event::{EdgeRow, LogEvent, LogPayload, NodeRow, RowEffect, CURRENT_SCHEMA_VERSION};
use liam_log::hash::{content_hashes, edge_row_hash, node_row_hash};
use liam_log::test_support::FailingLogWriter;
use liam_log::wal::WalError;
use liam_log::{LogOffset, LogWriter};
use uuid::Uuid;

use super::*;
use crate::graph::projection::{apply_steps, steps_for};
use crate::DefaultBackend;

pub(super) type Appended = Arc<StdMutex<Vec<(LogOffset, LogEvent)>>>;

/// A log that records every event it accepted, over a writer that can be told
/// to fail, so a test can compare the log against the projection.
pub(super) struct RecordingLog {
    inner: FailingLogWriter,
    appended: Appended,
    fail_after_recording: bool,
}

impl RecordingLog {
    pub(super) fn new() -> (Self, Appended) {
        Self::over(FailingLogWriter::fail_after(u64::MAX))
    }

    fn over(inner: FailingLogWriter) -> (Self, Appended) {
        let appended = Appended::default();
        let log = Self {
            inner,
            appended: Arc::clone(&appended),
            fail_after_recording: false,
        };
        (log, appended)
    }

    /// Models a writer that reports a failure for a record that did land.
    fn failing_after_recording(mut self) -> Self {
        self.fail_after_recording = true;
        self
    }
}

impl LogWriter for RecordingLog {
    fn append(&mut self, event: &LogEvent) -> std::result::Result<LogOffset, WalError> {
        let offset = self.inner.append(event)?;
        self.appended.lock().unwrap().push((offset, event.clone()));
        if self.fail_after_recording {
            return Err(WalError::Io(io::Error::other("failed after recording")));
        }
        Ok(offset)
    }

    fn log_id(&self) -> Uuid {
        self.inner.log_id()
    }
}

pub(super) fn share(writer: impl LogWriter + 'static) -> SharedLog {
    let bloom = HashBloom::new(BloomConfig::default());
    Arc::new(tokio::sync::Mutex::new(EventLog::new(
        Box::new(writer),
        bloom,
    )))
}

async fn open_clocked<B: Backend>(
    path: &str,
    clock: Arc<impl Clock + 'static>,
    log: SharedLog,
) -> Graph<B> {
    let graph = Graph::<B>::open_with_clock(path, GraphConfig::new(8), clock)
        .await
        .expect("open graph");
    graph.with_log(log).await.expect("attach log")
}

pub(super) async fn open_with<B: Backend>(path: &str, t: Millis, log: SharedLog) -> Graph<B> {
    open_clocked(path, Arc::new(FixedClock::new(t)), log).await
}

/// Like `logged_graph`, with the clock a test moves between writes so a time
/// the store reads twice for one write shows up as two different times.
async fn clocked_graph<B: Backend>(t: Millis) -> (Graph<B>, Appended, String, Arc<FixedClock>) {
    let (log, appended) = RecordingLog::new();
    let log_id = log.log_id().to_string();
    let clock = Arc::new(FixedClock::new(t));
    let graph = open_clocked(":memory:", Arc::clone(&clock), share(log)).await;
    (graph, appended, log_id, clock)
}

/// A logged in-memory graph, the log's recorder, and the log's id.
pub(super) async fn logged_graph<B: Backend>(t: Millis) -> (Graph<B>, Appended, String) {
    let (graph, appended, log_id, _) = clocked_graph(t).await;
    (graph, appended, log_id)
}

/// A clock that reads one millisecond later every time it is read.
struct TickingClock(AtomicI64);

impl Clock for TickingClock {
    fn now(&self) -> Millis {
        Millis(self.0.fetch_add(1, Ordering::SeqCst))
    }
}

pub(super) fn fact(content: &str) -> NewNode {
    NewNode::now("fact", "label", content)
        .with_producer("agent-a")
        .with_confidence(0.75)
}

/// `fact` with a supplied valid time, so source, trust, valid time, and ingest
/// time are four different values on the log record.
pub(super) fn fact_at(content: &str) -> NewNode {
    fact(content).with_valid_from(Millis(500))
}

fn assert_envelope(event: &LogEvent) {
    assert_envelope_at(event, 1000);
}

fn assert_envelope_at(event: &LogEvent, ingested_at: i64) {
    let envelope = (
        event.source.as_str(),
        event.trust_score,
        event.observed_at,
        event.ingested_at,
    );
    assert_eq!(envelope, ("agent-a", 0.75, 500, ingested_at), "{event:?}");
}

pub(super) fn events(appended: &Appended) -> Vec<LogEvent> {
    appended
        .lock()
        .unwrap()
        .iter()
        .map(|(_, event)| event.clone())
        .collect()
}

fn node_writes(appended: &Appended) -> Vec<NodeRow> {
    events(appended)
        .into_iter()
        .filter_map(|event| match event.payload {
            LogPayload::NodeWrite(row) => Some(row),
            _ => None,
        })
        .collect()
}

pub(super) async fn count<B: Backend>(g: &Graph<B>, table: &str) -> i64 {
    let rows = g
        .backend
        .query(&format!("SELECT COUNT(*) FROM {table}"), &[])
        .await
        .unwrap();
    rows[0].get_i64(0).unwrap()
}

/// `None` when the cursor row does not exist yet; otherwise its log id and
/// last applied offset (`None` while the offsets are still NULL).
pub(super) async fn cursor<B: Backend>(g: &Graph<B>) -> Option<(String, Option<(i64, i64)>)> {
    let rows = g
        .backend
        .query(
            "SELECT log_id, last_segment, last_index FROM log_cursor",
            &[],
        )
        .await
        .unwrap();
    let row = rows.first()?;
    let offset = match (&row.0[1], &row.0[2]) {
        (Value::Int(segment), Value::Int(index)) => Some((*segment, *index)),
        _ => None,
    };
    Some((row.get_string(0).unwrap(), offset))
}

pub(super) fn offset_pair(offset: LogOffset) -> (i64, i64) {
    (offset.segment as i64, offset.index as i64)
}

fn last_offset(appended: &Appended) -> LogOffset {
    appended.lock().unwrap().last().map_or(
        LogOffset {
            segment: u64::MAX,
            index: u64::MAX,
        },
        |(offset, _)| *offset,
    )
}

/// The stored `nodes` row as the log would carry it. `valid_from_supplied` is
/// not a column, so the caller states what it passed.
async fn stored_row<B: Backend>(g: &Graph<B>, id: &NodeId, valid_from_supplied: bool) -> NodeRow {
    let rows = g
        .backend
        .query(
            "SELECT id, kind, label, content, producer, attributes, scope, subject,
                    confidence, valid_from, valid_until, tx_from, tx_to
             FROM nodes WHERE id = ?1",
            &[id.as_str().into()],
        )
        .await
        .unwrap();
    let row = rows.first().expect("stored node row");
    let optional = |i: usize| match &row.0[i] {
        Value::Text(text) => Some(text.clone()),
        _ => None,
    };
    let confidence = match &row.0[8] {
        Value::Real(value) => *value,
        other => panic!("confidence is not a real: {other:?}"),
    };
    NodeRow {
        id: row.get_string(0).unwrap(),
        kind: row.get_string(1).unwrap(),
        label: row.get_string(2).unwrap(),
        content: row.get_string(3).unwrap(),
        producer: row.get_string(4).unwrap(),
        attributes: row.get_string(5).unwrap(),
        scope: optional(6),
        subject: optional(7),
        confidence,
        valid_from: row.get_i64(9).unwrap(),
        valid_from_supplied,
        valid_until: row.get_i64(10).unwrap(),
        tx_from: row.get_i64(11).unwrap(),
        tx_to: row.get_i64(12).unwrap(),
    }
}

#[tokio::test]
async fn log_write_insert_appends_one_node_write_equal_to_the_stored_row() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let node = fact_at("content")
        .with_scope(" proj/a ")
        .with_subject("subject-1")
        .with_attributes(serde_json::json!({"k": "v"}));

    // Act
    let id = g.insert(node).await.unwrap();

    // Assert
    let logged = events(&appended);
    assert_eq!(logged.len(), 1, "one insert appends exactly one event");
    let event = &logged[0];
    let stored = stored_row(&g, &id, true).await;
    assert_eq!(event.payload, LogPayload::NodeWrite(stored.clone()));
    assert_eq!(event.content_hash, node_row_hash(&stored));
    assert_eq!(event.schema_version, CURRENT_SCHEMA_VERSION);
    assert_eq!(event.encryption_key_id, None);
    assert!(!event.event_id.is_empty());
    assert_envelope(event);
    assert_eq!(
        cursor(&g).await,
        Some((log_id, Some(offset_pair(last_offset(&appended)))))
    );
    let indexed = g
        .backend
        .query(
            "SELECT content_hash, first_event_id, row_ids FROM log_hash_index",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(indexed.len(), 1);
    assert!(
        matches!(&indexed[0].0[0], Value::Blob(hash) if hash.as_slice() == event.content_hash),
        "index key must be the event's content hash"
    );
    assert_eq!(indexed[0].get_string(1).unwrap(), event.event_id);
    let row_ids: Vec<String> = serde_json::from_str(&indexed[0].get_string(2).unwrap()).unwrap();
    assert_eq!(row_ids, vec![id.as_str().to_string()]);
}

#[tokio::test]
async fn log_write_insert_without_a_log_behaves_as_before() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let first = g.insert(fact("same")).await.unwrap();
    let second = g.insert(fact("same")).await.unwrap();

    // Assert: no dedup, and none of the log tables is touched.
    assert_ne!(first, second);
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(count(&g, "log_cursor").await, 0);
    assert_eq!(count(&g, "log_hash_index").await, 0);
    assert_eq!(count(&g, "log_quarantine").await, 0);
}

#[tokio::test]
async fn log_write_duplicate_insert_returns_the_first_id_and_appends_duplicate_of() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let first = g.insert(fact_at("same")).await.unwrap();

    // Act
    let second = g.insert(fact_at("same")).await.unwrap();

    // Assert
    assert_eq!(second, first, "a duplicate returns the first node's id");
    assert_eq!(count(&g, "nodes").await, 1, "a duplicate writes no row");
    let logged = events(&appended);
    assert_eq!(logged.len(), 2);
    assert!(matches!(logged[0].payload, LogPayload::NodeWrite(_)));
    assert_eq!(
        logged[1].payload,
        LogPayload::DuplicateOf {
            first_event_id: logged[0].event_id.clone()
        }
    );
    assert_eq!(logged[1].content_hash, logged[0].content_hash);
    assert_envelope(&logged[1]);
    assert_eq!(
        cursor(&g).await,
        Some((log_id, Some(offset_pair(last_offset(&appended)))))
    );
}

#[tokio::test]
async fn log_write_content_that_differs_only_in_unstored_form_is_a_duplicate() {
    // Arrange: the hash covers the resolved row, so the scope's surrounding
    // whitespace and the attributes' key order must not make a distinct write.
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let cases = [
        (
            fact("scoped").with_scope("proj/a"),
            fact("scoped").with_scope("  proj/a "),
        ),
        (
            fact("attrs").with_attributes(serde_json::json!({"a": 1, "b": 2})),
            fact("attrs").with_attributes(serde_json::json!({"b": 2, "a": 1})),
        ),
    ];

    for (first, second) in cases {
        // Act
        let first = g.insert(first).await.unwrap();
        let second = g.insert(second).await.unwrap();

        // Assert
        assert_eq!(second, first);
    }
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(node_writes(&appended).len(), 2);
}

#[tokio::test]
async fn log_write_a_caller_supplied_valid_from_is_flagged_and_changes_the_hash() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let unset = g.insert(fact("same")).await.unwrap();

    // Act
    let supplied = g
        .insert(fact("same").with_valid_from(Millis(500)))
        .await
        .unwrap();

    // Assert: stored, not deduplicated, and the flag reflects what the caller sent.
    assert_ne!(supplied, unset);
    let rows = node_writes(&appended);
    assert_eq!(rows.len(), 2);
    assert!(!rows[0].valid_from_supplied);
    assert!(rows[1].valid_from_supplied);
    assert_eq!(rows[1].valid_from, 500);
}

#[tokio::test]
async fn log_write_hash_index_hit_on_a_superseded_row_is_not_a_duplicate() {
    // Arrange: write X, then supersede it with Y, which closes X.
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let x = g.insert(fact("X").with_subject("subject-1")).await.unwrap();
    g.supersede(&x, fact("Y").with_subject("subject-1"))
        .await
        .unwrap();

    // Act
    let third = g.insert(fact("X").with_subject("subject-1")).await.unwrap();

    // Assert: stored live, logged as a write, and the index points at it.
    assert_ne!(third, x);
    let live = g
        .backend
        .query(
            "SELECT COUNT(*) FROM nodes WHERE id = ?1 AND tx_to = ?2",
            &[third.as_str().into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(live[0].get_i64(0).unwrap(), 1);
    let logged = events(&appended);
    assert!(
        logged
            .iter()
            .all(|event| !matches!(event.payload, LogPayload::DuplicateOf { .. })),
        "a superseded carrier must not deduplicate: {logged:?}"
    );
    assert!(!logged.is_empty(), "the writes must be logged");
    let last = logged.last().unwrap();
    let LogPayload::NodeWrite(row) = &last.payload else {
        panic!("the third write must be logged as a NodeWrite: {last:?}");
    };
    assert_eq!(row.id, third.as_str());
    let indexed = g
        .backend
        .query(
            "SELECT first_event_id FROM log_hash_index WHERE content_hash = ?1",
            &[node_row_hash(row).to_vec().into()],
        )
        .await
        .unwrap();
    assert_eq!(indexed.len(), 1);
    assert_eq!(indexed[0].get_string(0).unwrap(), last.event_id);
}

#[tokio::test]
async fn log_write_append_failure_fails_the_insert_and_leaves_no_trace() {
    // Arrange: the first append fails, every later one succeeds.
    let (log, appended) = RecordingLog::over(FailingLogWriter::fail_on_nth(1));
    let g = open_with::<DefaultBackend>(":memory:", Millis(1000), share(log)).await;
    let before = cursor(&g).await;

    // Act
    let failed = g.insert(fact("same")).await;

    // Assert: no record, no row, no index entry, no cursor change.
    assert!(matches!(failed, Err(Error::LogAppend(_))), "{failed:?}");
    assert!(events(&appended).is_empty());
    assert_eq!(count(&g, "nodes").await, 0);
    assert_eq!(count(&g, "log_hash_index").await, 0);
    assert_eq!(cursor(&g).await, before);

    // Act: the same content again is a first write, not a duplicate.
    let retried = g.insert(fact("same")).await;

    // Assert
    assert!(retried.is_ok(), "{retried:?}");
    assert_eq!(count(&g, "nodes").await, 1);
}

#[tokio::test]
async fn log_write_a_writer_that_always_fails_stores_nothing() {
    // Arrange
    let (log, appended) = RecordingLog::over(FailingLogWriter::fail_after(0));
    let g = open_with::<DefaultBackend>(":memory:", Millis(1000), share(log)).await;

    // Act
    let first = g.insert(fact("one")).await;
    let second = g.insert(fact("two")).await;

    // Assert
    assert!(matches!(first, Err(Error::LogAppend(_))), "{first:?}");
    assert!(matches!(second, Err(Error::LogAppend(_))), "{second:?}");
    assert_eq!(count(&g, "nodes").await, 0);
    assert!(events(&appended).is_empty());
}

#[tokio::test]
async fn log_write_a_failure_reported_for_a_record_that_landed_stores_no_row() {
    // Arrange: the writer keeps the record but reports an error, so the log
    // holds a write the store did not apply.
    let (log, appended) = RecordingLog::new();
    let log = log.failing_after_recording();
    let g = open_with::<DefaultBackend>(":memory:", Millis(1000), share(log)).await;
    let before = cursor(&g).await;

    // Act
    let failed = g.insert(fact("same")).await;

    // Assert: the store's outcome is pinned: no row, cursor unmoved.
    assert!(matches!(failed, Err(Error::LogAppend(_))), "{failed:?}");
    assert_eq!(node_writes(&appended).len(), 1);
    assert_eq!(count(&g, "nodes").await, 0);
    assert_eq!(cursor(&g).await, before);
}

#[tokio::test]
async fn log_write_a_failed_duplicate_append_leaves_the_cursor_where_it_was() {
    // Arrange: the second append, the DuplicateOf, fails.
    let (log, appended) = RecordingLog::over(FailingLogWriter::fail_on_nth(2));
    let g = open_with::<DefaultBackend>(":memory:", Millis(1000), share(log)).await;
    g.insert(fact("same")).await.unwrap();
    let before = cursor(&g).await;

    // Act
    let failed = g.insert(fact("same")).await;

    // Assert
    assert!(matches!(failed, Err(Error::LogAppend(_))), "{failed:?}");
    assert_eq!(cursor(&g).await, before);
    assert_eq!(node_writes(&appended).len(), 1);
    assert_eq!(events(&appended).len(), 1, "only the NodeWrite is logged");
}

#[tokio::test]
async fn log_write_projection_failure_voids_the_write_and_leaves_nothing_behind() {
    // The transaction's executes are the node insert, the hash index, and the
    // cursor: a failure at any of them must leave one atomic unit undone.
    for failing_execute in 0..=2 {
        // Arrange
        let (log, appended) = RecordingLog::new();
        let log_id = log.log_id().to_string();
        let shared = share(log);
        let g = open_with::<FailingBackend>(":memory:", Millis(1000), Arc::clone(&shared)).await;
        g.backend.set_fail_on_execute(failing_execute);

        // Act
        let failed = g.insert(fact_at("same")).await;

        // Assert
        let context = format!("failing execute {failing_execute}");
        assert!(
            matches!(failed, Err(Error::Backend(_))),
            "{context}: {failed:?}"
        );
        assert_eq!(count(&g, "nodes").await, 0, "{context}");
        assert_eq!(count(&g, "log_hash_index").await, 0, "{context}");
        let logged = events(&appended);
        assert_eq!(logged.len(), 2, "{context}: the write, then its void");
        assert!(matches!(logged[0].payload, LogPayload::NodeWrite(_)));
        assert_eq!(
            logged[1].payload,
            LogPayload::Voided {
                target_event_id: logged[0].event_id.clone()
            }
        );
        assert_envelope(&logged[1]);
        assert_eq!(
            cursor(&g).await,
            Some((log_id, Some(offset_pair(last_offset(&appended))))),
            "{context}: the cursor sits on the void"
        );
        assert!(!shared
            .lock()
            .await
            .bloom_might_contain(&logged[0].content_hash));

        // Act: a retry after the fault clears is a first write, not a duplicate.
        g.backend.set_fail_on_execute(usize::MAX);
        let retried = g.insert(fact_at("same")).await;

        // Assert
        assert!(retried.is_ok(), "{context}: {retried:?}");
        assert_eq!(count(&g, "nodes").await, 1, "{context}");
        assert!(matches!(
            events(&appended).last().unwrap().payload,
            LogPayload::NodeWrite(_)
        ));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn log_write_concurrent_inserts_keep_wal_order_equal_to_commit_order() {
    // Arrange: every second `begin` is slow, so a write that let go of the log
    // lock before its commit would be overtaken by the next one.
    const PAIRS: usize = 20;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let (log, appended) = RecordingLog::new();
    let shared = share(log);
    let g = Arc::new(
        open_with::<FailingBackend>(path.to_str().unwrap(), Millis(1000), Arc::clone(&shared))
            .await,
    );
    g.backend
        .probe
        .delay_odd_begins
        .store(true, Ordering::SeqCst);
    *g.backend.probe.watched.lock().unwrap() = Some(shared);

    // Act
    for pair in 0..PAIRS {
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let tasks: Vec<_> = (0..2)
            .map(|side| {
                let (g, barrier) = (Arc::clone(&g), Arc::clone(&barrier));
                tokio::spawn(async move {
                    barrier.wait().await;
                    g.insert(fact(&format!("pair-{pair}-side-{side}"))).await
                })
            })
            .collect();
        for task in tasks {
            task.await.unwrap().unwrap();
        }
    }

    // Assert: node rowids are assigned in commit order, and the log lock was
    // held at every commit.
    let wal_order: Vec<String> = node_writes(&appended).into_iter().map(|r| r.id).collect();
    let commit_order: Vec<String> = g
        .backend
        .query("SELECT id FROM nodes ORDER BY rowid", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect();
    assert_eq!(wal_order.len(), PAIRS * 2);
    assert_eq!(wal_order, commit_order);
    let lock_held = g.backend.probe.commit_lock_held.lock().unwrap().clone();
    assert_eq!(lock_held, vec![true; PAIRS * 2]);
}

#[tokio::test]
async fn log_write_a_cancelled_write_poisons_the_log_until_reopen() {
    // Arrange: the write is appended, then its commit never resolves.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let path = path.to_str().unwrap();
    let (log, appended) = RecordingLog::new();
    let g = open_with::<FailingBackend>(path, Millis(1000), share(log)).await;
    g.backend.probe.hang_commit.store(true, Ordering::SeqCst);

    // Act: drop the insert future while it waits on the commit.
    let cancelled =
        tokio::time::timeout(Duration::from_millis(200), g.insert(fact("orphan"))).await;
    g.backend.probe.hang_commit.store(false, Ordering::SeqCst);
    let refused = g.insert(fact("next")).await;

    // Assert: the orphan is in the log only, and logged writes are refused.
    assert!(cancelled.is_err(), "the insert must still be pending");
    assert_eq!(node_writes(&appended).len(), 1);
    assert!(matches!(refused, Err(Error::LogPoisoned)), "{refused:?}");
    assert_eq!(
        node_writes(&appended).len(),
        1,
        "a refused write appends nothing"
    );

    // Act: reopen the same database on a fresh log.
    drop(g);
    let (fresh, _) = RecordingLog::new();
    let reopened = open_with::<DefaultBackend>(path, Millis(1000), share(fresh)).await;

    // Assert
    assert!(reopened.insert(fact("next")).await.is_ok());
}

#[tokio::test]
async fn log_write_graphs_on_one_log_share_the_dedup_filter_and_the_poison_flag() {
    // Arrange: two handles on one database and one log. The log's fourth append,
    // the void of a failed write, fails.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let path = path.to_str().unwrap();
    let (log, _) = RecordingLog::over(FailingLogWriter::fail_on_nth(4));
    let shared = share(log);
    let a = open_with::<FailingBackend>(path, Millis(1000), Arc::clone(&shared)).await;
    let b = open_with::<DefaultBackend>(path, Millis(1000), shared).await;

    // Act: A writes H, then B writes the same content.
    let first = a.insert(fact("H")).await.unwrap();
    let second = b.insert(fact("H")).await.unwrap();

    // Assert: B's filter already knows H, so it is a duplicate, not a second row.
    assert_eq!(second, first);
    assert_eq!(count(&a, "nodes").await, 1);

    // Act: A's write fails after its append and the void cannot be appended.
    a.backend.set_fail_on_execute(0);
    let failed = a.insert(fact("other")).await;
    let refused = b.insert(fact("another")).await;

    // Assert: the poison A set refuses B.
    assert!(matches!(failed, Err(Error::Backend(_))), "{failed:?}");
    assert!(matches!(refused, Err(Error::LogPoisoned)), "{refused:?}");
}

#[tokio::test]
async fn log_write_vector_upsert_runs_after_commit_not_inside_the_transaction() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let embedding = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Act: the vector write takes the lock `begin()` holds, so running it
    // inside the transaction would hang.
    let inserted = tokio::time::timeout(
        Duration::from_secs(10),
        g.insert(fact("embedded").with_embedding(embedding.clone())),
    )
    .await
    .expect("insert must not deadlock on the write lock");

    // Assert
    let id = inserted.unwrap();
    assert_eq!(node_writes(&appended).len(), 1);
    let hits = g
        .backend
        .vector_search(&embedding, 5, None, None, Millis(1000))
        .await
        .unwrap();
    assert!(hits.contains(&id));
}

#[tokio::test]
async fn log_write_a_retry_after_a_vector_failure_repairs_the_vector_on_the_first_node() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<FailingVectorBackend>(Millis(1000)).await;
    g.backend.set_fail_on_vector_upsert(0);
    let embedding = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Act
    let result = g
        .insert(fact("embedded").with_embedding(embedding.clone()))
        .await;

    // Assert: the error surfaces, but the write is durable in both places.
    assert!(result.is_err(), "the vector failure must be reported");
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(node_writes(&appended).len(), 1);
    assert_eq!(
        cursor(&g).await,
        Some((log_id, Some(offset_pair(last_offset(&appended)))))
    );
    let first = NodeId::from_raw(node_writes(&appended)[0].id.clone());

    // Act: a retry with the wrong dimensions, then with the right ones.
    let wrong = g
        .insert(fact("embedded").with_embedding(vec![1.0, 0.0, 0.0]))
        .await;
    let retried = g
        .insert(fact("embedded").with_embedding(embedding.clone()))
        .await;

    // Assert: the duplicate is checked and repaired against the first node.
    assert!(
        matches!(
            wrong,
            Err(Error::Dimension {
                expected: 8,
                got: 3
            })
        ),
        "{wrong:?}"
    );
    assert_eq!(retried.unwrap(), first);
    let hits = g
        .backend
        .vector_search(&embedding, 5, None, None, Millis(1000))
        .await
        .unwrap();
    assert!(hits.contains(&first));
}

#[tokio::test]
async fn log_write_a_void_double_fault_poisons_the_log_until_reopen() {
    // Arrange: the write appends, its projection fails, and the void append
    // (the log's second) fails too.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let path = path.to_str().unwrap();
    let (log, appended) = RecordingLog::over(FailingLogWriter::fail_on_nth(2));
    let g = open_with::<FailingBackend>(path, Millis(1000), share(log)).await;
    g.backend.set_fail_on_execute(0);

    // Act
    let first = g.insert(fact("first")).await;
    g.backend.set_fail_on_execute(usize::MAX);
    let logged_before = events(&appended);
    let second = g.insert(fact("second")).await;

    // Assert: the original error surfaces, then every logged write is refused
    // without appending anything.
    assert!(matches!(first, Err(Error::Backend(_))), "{first:?}");
    assert!(matches!(second, Err(Error::LogPoisoned)), "{second:?}");
    assert!(matches!(
        logged_before[..],
        [LogEvent {
            payload: LogPayload::NodeWrite(_),
            ..
        }]
    ));
    assert_eq!(events(&appended), logged_before);
    assert_eq!(count(&g, "nodes").await, 0);
    assert_eq!(count(&g, "log_hash_index").await, 0);

    // Act: reopen the same database on a fresh writer.
    drop(g);
    let (fresh, _) = RecordingLog::new();
    let reopened = open_with::<DefaultBackend>(path, Millis(1000), share(fresh)).await;

    // Assert
    assert!(reopened.insert(fact("second")).await.is_ok());
}

// ---- the remaining write paths: upsert_by, supersede, relate, ingest_episode ----
//
// Payload shapes these tests pin, so a replay can rebuild the store from the
// log alone:
// - insert, and an upsert_by with no live competitor: `NodeWrite`.
// - relate: `EdgeWrite`.
// - supersede, and an upsert_by that finds a live competitor: one
//   `EpisodeBatch` of `[Node(new row), Edge(supersedes new -> old)]`. The batch
//   has no row-close effect, so the close is implied: replaying a `supersedes`
//   edge closes its `dst` at the edge's `tx_from`.
// - ingest_episode: one `EpisodeBatch` of every row in write order: for each
//   node its row and, when it superseded a competitor, that `supersedes` edge,
//   then the episode's own edges.

async fn stored_edge<B: Backend>(g: &Graph<B>, id: &str) -> EdgeRow {
    let rows = g
        .backend
        .query(
            "SELECT id, src, dst, type, attributes, tx_from, tx_to FROM edges WHERE id = ?1",
            &[id.into()],
        )
        .await
        .unwrap();
    let row = rows.first().expect("stored edge row");
    EdgeRow {
        id: row.get_string(0).unwrap(),
        src: row.get_string(1).unwrap(),
        dst: row.get_string(2).unwrap(),
        edge_type: row.get_string(3).unwrap(),
        attributes: row.get_string(4).unwrap(),
        tx_from: row.get_i64(5).unwrap(),
        tx_to: row.get_i64(6).unwrap(),
    }
}

async fn node_effect<B: Backend>(g: &Graph<B>, id: &NodeId) -> RowEffect {
    RowEffect::Node(stored_row(g, id, true).await)
}

async fn edge_between<B: Backend>(g: &Graph<B>, src: &str, dst: &str, kind: &str) -> EdgeRow {
    let rows = g
        .backend
        .query(
            "SELECT id FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3",
            &[src.into(), dst.into(), kind.into()],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 1, "one {kind} edge from {src} to {dst}");
    stored_edge(g, &rows[0].get_string(0).unwrap()).await
}

async fn index_entry<B: Backend>(g: &Graph<B>, hash: &[u8; 32]) -> Option<(String, Vec<String>)> {
    let rows = g
        .backend
        .query(
            "SELECT first_event_id, row_ids FROM log_hash_index WHERE content_hash = ?1",
            &[hash.to_vec().into()],
        )
        .await
        .unwrap();
    let row = rows.first()?;
    let row_ids = serde_json::from_str(&row.get_string(1).unwrap()).unwrap();
    Some((row.get_string(0).unwrap(), row_ids))
}

/// Every row the event carries is in the hash index under the event's id.
async fn assert_indexed_with<B: Backend>(g: &Graph<B>, event: &LogEvent) {
    let carried = content_hashes(event);
    assert!(!carried.is_empty(), "the event carries rows: {event:?}");
    for (hash, row_id) in carried {
        let entry = index_entry(g, &hash).await;
        assert!(entry.is_some(), "row {row_id} is not in the hash index");
        let (first_event_id, row_ids) = entry.unwrap();
        assert_eq!(first_event_id, event.event_id, "row {row_id}");
        assert!(row_ids.contains(&row_id), "row {row_id} in {row_ids:?}");
    }
}

pub(super) async fn assert_cursor_at_last_event<B: Backend>(
    g: &Graph<B>,
    appended: &Appended,
    log_id: &str,
) {
    assert_eq!(
        cursor(g).await,
        Some((log_id.to_string(), Some(offset_pair(last_offset(appended)))))
    );
}

pub(super) fn events_since(appended: &Appended, since: usize) -> Vec<LogEvent> {
    events(appended).split_off(since)
}

fn has_duplicate_of(logged: &[LogEvent]) -> bool {
    logged
        .iter()
        .any(|event| matches!(event.payload, LogPayload::DuplicateOf { .. }))
}

fn batch_effects(event: &LogEvent) -> Vec<RowEffect> {
    match &event.payload {
        LogPayload::EpisodeBatch(effects) => effects.clone(),
        other => panic!("expected an EpisodeBatch, found {other:?}"),
    }
}

fn batch_node(event: &LogEvent) -> NodeRow {
    match batch_effects(event).into_iter().next() {
        Some(RowEffect::Node(row)) => row,
        other => panic!("expected the batch to start with a node row, found {other:?}"),
    }
}

pub(super) async fn live_ids_with_subject<B: Backend>(g: &Graph<B>, subject: &str) -> Vec<String> {
    g.backend
        .query(
            "SELECT id FROM nodes WHERE subject = ?1 AND tx_to = ?2 ORDER BY rowid",
            &[subject.into(), FOREVER.into()],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect()
}

async fn assert_log_tables_untouched<B: Backend>(g: &Graph<B>) {
    assert_eq!(count(g, "log_cursor").await, 0);
    assert_eq!(count(g, "log_hash_index").await, 0);
    assert_eq!(count(g, "log_quarantine").await, 0);
}

pub(super) fn mentions(from: usize, to: usize) -> EpisodeEdge {
    EpisodeEdge {
        from: EpisodeRef::New(from),
        to: EpisodeRef::New(to),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({"w": 1}),
    }
}

/// One supersede at `at` is one event: the new row and the `supersedes` edge,
/// with the old row closed at the edge's start.
async fn assert_supersede_logged<B: Backend>(
    g: &Graph<B>,
    appended: &Appended,
    since: usize,
    (old, new): (&NodeId, &NodeId),
    (log_id, at): (&str, i64),
) {
    let logged = events_since(appended, since);
    assert_eq!(logged.len(), 1, "one event for the supersede: {logged:?}");
    let event = &logged[0];
    let new_row = stored_row(g, new, true).await;
    let edge = edge_between(g, new.as_str(), old.as_str(), relation::SUPERSEDES).await;
    assert_eq!(
        event.payload,
        LogPayload::EpisodeBatch(vec![
            RowEffect::Node(new_row.clone()),
            RowEffect::Edge(edge.clone())
        ])
    );
    assert_eq!(event.content_hash, node_row_hash(&new_row));
    assert_envelope_at(event, at);
    assert_eq!(edge.tx_from, at);
    assert_eq!(new_row.tx_from, at);
    assert_eq!(
        stored_row(g, old, true).await.tx_to,
        edge.tx_from,
        "the old row closes where the supersedes edge starts"
    );
    assert_cursor_at_last_event(g, appended, log_id).await;
    assert_indexed_with(g, event).await;
}

#[tokio::test]
async fn log_write_upsert_by_without_a_live_competitor_appends_one_node_write_equal_to_the_stored_row(
) {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let node = fact_at("content")
        .with_scope(" proj/a ")
        .with_subject("subject-1")
        .with_attributes(serde_json::json!({"k": "v"}));

    // Act
    let id = g.upsert_by(node).await.unwrap();

    // Assert
    let logged = events(&appended);
    assert_eq!(logged.len(), 1, "{logged:?}");
    let stored = stored_row(&g, &id, true).await;
    assert_eq!(logged[0].payload, LogPayload::NodeWrite(stored.clone()));
    assert_eq!(logged[0].content_hash, node_row_hash(&stored));
    assert_envelope(&logged[0]);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
    assert_indexed_with(&g, &logged[0]).await;
}

#[tokio::test]
async fn log_write_supersede_appends_one_batch_of_the_new_row_and_its_supersedes_edge() {
    // Arrange: the clock moves after the setup write, so the write under test
    // cannot take a time from anything but its own read.
    let (g, appended, log_id, clock) = clocked_graph::<DefaultBackend>(Millis(1000)).await;
    let old = g.insert(fact_at("old").with_subject("s")).await.unwrap();
    let since = events(&appended).len();
    clock.set(Millis(2000));

    // Act
    let new = g
        .supersede(&old, fact_at("new").with_subject("s"))
        .await
        .unwrap();

    // Assert
    assert_supersede_logged(&g, &appended, since, (&old, &new), (&log_id, 2000)).await;
}

#[tokio::test]
async fn log_write_upsert_by_over_a_live_subject_logs_the_same_batch_as_supersede() {
    // Arrange
    let (g, appended, log_id, clock) = clocked_graph::<DefaultBackend>(Millis(1000)).await;
    let old = g.upsert_by(fact_at("old").with_subject("s")).await.unwrap();
    let since = events(&appended).len();
    clock.set(Millis(2000));

    // Act
    let new = g.upsert_by(fact_at("new").with_subject("s")).await.unwrap();

    // Assert
    assert_ne!(new, old);
    assert_supersede_logged(&g, &appended, since, (&old, &new), (&log_id, 2000)).await;
}

#[tokio::test]
async fn log_write_relate_appends_one_edge_write_equal_to_the_stored_row() {
    // Arrange
    let (g, appended, log_id, clock) = clocked_graph::<DefaultBackend>(Millis(1000)).await;
    let src = g.insert(fact_at("src")).await.unwrap();
    let dst = g.insert(fact_at("dst")).await.unwrap();
    let since = events(&appended).len();
    clock.set(Millis(2000));

    // Act
    let id = g.relate(&src, &dst, "mentions").await.unwrap();

    // Assert
    let logged = events_since(&appended, since);
    assert_eq!(logged.len(), 1, "{logged:?}");
    let event = &logged[0];
    let stored = stored_edge(&g, id.as_str()).await;
    assert_eq!(event.payload, LogPayload::EdgeWrite(stored.clone()));
    assert_eq!(event.content_hash, edge_row_hash(&stored));
    assert_eq!((event.observed_at, event.ingested_at), (2000, 2000));
    assert_eq!(stored.tx_from, 2000);
    assert_eq!(
        (event.source.as_str(), event.trust_score),
        ("liam-store", 1.0)
    );
    assert_eq!(event.schema_version, CURRENT_SCHEMA_VERSION);
    assert_eq!(event.encryption_key_id, None);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
    assert_indexed_with(&g, event).await;
}

#[tokio::test]
async fn log_write_ingest_episode_appends_one_batch_equal_to_every_stored_row() {
    // Arrange: n0 supersedes the live row x, and n2 supersedes n0 inside the
    // same episode, so the episode's edge starts at n2, the node left live.
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let x = g.insert(fact_at("v1").with_subject("s")).await.unwrap();
    let peer = g.insert(fact_at("peer")).await.unwrap();
    let since = events(&appended).len();
    let cites = EpisodeEdge {
        from: EpisodeRef::New(1),
        to: EpisodeRef::Existing(peer.clone()),
        kind: "cites".to_string(),
        attributes: serde_json::json!({}),
    };

    // Act
    let result = g
        .ingest_episode(
            vec![
                fact_at("v2").with_subject("s"),
                fact_at("ep-peer"),
                fact_at("v3").with_subject("s"),
            ],
            vec![mentions(2, 1), cites],
        )
        .await
        .unwrap();

    // Assert: one event, rows in write order.
    let logged = events_since(&appended, since);
    assert_eq!(logged.len(), 1, "one event per episode: {logged:?}");
    let [n0, n1, n2] = [0, 1, 2].map(|i| result.node_ids[i].clone());
    // The log carries n0 as written, still open: the supersedes edge after it
    // is what closes it, on replay as on the live write.
    let n0_as_written = NodeRow {
        tx_to: FOREVER.0,
        ..stored_row(&g, &n0, true).await
    };
    let expected = vec![
        RowEffect::Node(n0_as_written),
        RowEffect::Edge(edge_between(&g, n0.as_str(), x.as_str(), relation::SUPERSEDES).await),
        node_effect(&g, &n1).await,
        node_effect(&g, &n2).await,
        RowEffect::Edge(edge_between(&g, n2.as_str(), n0.as_str(), relation::SUPERSEDES).await),
        RowEffect::Edge(stored_edge(&g, result.edge_ids[0].as_str()).await),
        RowEffect::Edge(stored_edge(&g, result.edge_ids[1].as_str()).await),
    ];
    assert_eq!(logged[0].payload, LogPayload::EpisodeBatch(expected));
    assert_eq!(logged[0].ingested_at, 1000);
    assert_eq!(stored_row(&g, &x, true).await.tx_to, 1000);
    assert_eq!(stored_row(&g, &n0, true).await.tx_to, 1000);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
}

#[tokio::test]
async fn log_write_every_episode_row_registers_in_the_hash_index_under_the_batch_event() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;

    // Act
    g.ingest_episode(vec![fact_at("a"), fact_at("b")], vec![mentions(0, 1)])
        .await
        .unwrap();

    // Assert
    let logged = events(&appended);
    assert_eq!(logged.len(), 1, "{logged:?}");
    assert_eq!(batch_effects(&logged[0]).len(), 3);
    assert_indexed_with(&g, &logged[0]).await;
    assert_eq!(count(&g, "log_hash_index").await, 3);
}

#[tokio::test]
async fn log_write_a_duplicate_inside_one_episode_keeps_the_first_carrier_without_a_duplicate_of() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;

    // Act: the same content twice in one batch.
    let result = g
        .ingest_episode(vec![fact_at("twin"), fact_at("twin")], vec![])
        .await
        .unwrap();

    // Assert: the batch is all or nothing, so both rows are stored and nothing
    // is logged as a duplicate. The index keeps the first carrier first.
    let logged = events(&appended);
    assert_eq!(logged.len(), 1, "{logged:?}");
    assert_eq!(count(&g, "nodes").await, 2);
    let first = &result.node_ids[0];
    let hash = node_row_hash(&stored_row(&g, first, true).await);
    let entry = index_entry(&g, &hash).await;
    assert!(entry.is_some(), "the twin hash is indexed");
    let (first_event_id, row_ids) = entry.unwrap();
    assert_eq!(first_event_id, logged[0].event_id);
    assert_eq!(row_ids.first().map(String::as_str), Some(first.as_str()));

    // Act: later identical content resolves to the first carrier.
    let later = g.insert(fact_at("twin")).await.unwrap();

    // Assert
    assert_eq!(&later, first);
}

#[tokio::test]
async fn log_write_each_episode_is_one_batch_event_of_the_stored_rows() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;

    // Act
    let first = g
        .ingest_episode(vec![fact_at("a"), fact_at("b")], vec![mentions(0, 1)])
        .await
        .unwrap();
    g.ingest_episode(vec![fact_at("c")], vec![]).await.unwrap();

    // Assert: the first batch is the stored rows in write order.
    let logged = events(&appended);
    assert_eq!(logged.len(), 2, "{logged:?}");
    let a = stored_row(&g, &first.node_ids[0], true).await;
    let b = stored_row(&g, &first.node_ids[1], true).await;
    let edge = stored_edge(&g, first.edge_ids[0].as_str()).await;
    assert_eq!(
        logged[0].payload,
        LogPayload::EpisodeBatch(vec![
            RowEffect::Node(a.clone()),
            RowEffect::Node(b),
            RowEffect::Edge(edge)
        ])
    );
    assert_eq!(logged[0].content_hash, node_row_hash(&a));
    let stamp = (&logged[0].source, logged[0].trust_score);
    assert_eq!(stamp, (&"agent-a".to_string(), 0.75));
    assert_eq!((logged[0].observed_at, logged[0].ingested_at), (1000, 1000));
    assert_eq!(batch_effects(&logged[1]).len(), 1);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
}

#[tokio::test]
async fn log_write_an_episode_is_not_deduplicated_against_live_content() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let live = g.insert(fact_at("same")).await.unwrap();

    // Act
    let result = g
        .ingest_episode(vec![fact_at("same")], vec![])
        .await
        .unwrap();

    // Assert: a batch applies as a whole, so it never becomes a partial duplicate.
    let logged = events(&appended);
    assert_eq!(logged.len(), 2, "{logged:?}");
    assert!(matches!(logged[1].payload, LogPayload::EpisodeBatch(_)));
    assert!(!has_duplicate_of(&logged));
    assert_ne!(result.node_ids[0], live);
    assert_eq!(count(&g, "nodes").await, 2);
}

// ---- live-row rule for upsert_by and supersede ----

#[tokio::test]
async fn log_write_upsert_by_with_the_content_of_a_live_row_returns_it_and_logs_duplicate_of() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let first = g
        .upsert_by(fact_at("same").with_subject("s"))
        .await
        .unwrap();

    // Act
    let second = g
        .upsert_by(fact_at("same").with_subject("s"))
        .await
        .unwrap();

    // Assert
    assert_eq!(second, first);
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(count(&g, "edges").await, 0);
    let logged = events(&appended);
    assert_eq!(logged.len(), 2, "{logged:?}");
    assert_eq!(
        logged[1].payload,
        LogPayload::DuplicateOf {
            first_event_id: logged[0].event_id.clone()
        }
    );
    assert_envelope(&logged[1]);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
}

/// `new` was written as the last of `expected_events` events: stored live,
/// logged as a write, and the index points at it.
async fn assert_stored_as_a_new_write<B: Backend>(
    g: &Graph<B>,
    appended: &Appended,
    new: &NodeId,
    expected_events: usize,
) {
    let logged = events(appended);
    assert_eq!(logged.len(), expected_events, "{logged:?}");
    assert!(!has_duplicate_of(&logged), "{logged:?}");
    let last = logged.last().unwrap();
    let row = batch_node(last);
    assert_eq!(row.id, new.as_str());
    assert_eq!(
        index_entry(g, &node_row_hash(&row)).await,
        Some((last.event_id.clone(), vec![new.as_str().to_string()]))
    );
    assert_eq!(live_ids_with_subject(g, "s").await, vec![new.as_str()]);
}

#[tokio::test]
async fn log_write_upsert_by_with_content_whose_carrier_was_superseded_is_a_new_write() {
    // Arrange: X, then Y over it, so X's carrier is closed.
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let x = g.upsert_by(fact_at("X").with_subject("s")).await.unwrap();
    g.upsert_by(fact_at("Y").with_subject("s")).await.unwrap();

    // Act
    let third = g.upsert_by(fact_at("X").with_subject("s")).await.unwrap();

    // Assert
    assert_ne!(third, x);
    assert_stored_as_a_new_write(&g, &appended, &third, 3).await;
}

#[tokio::test]
async fn log_write_supersede_with_content_already_live_still_closes_old_and_repoints_the_index() {
    // Arrange
    let (g, appended, log_id, clock) = clocked_graph::<DefaultBackend>(Millis(1000)).await;
    let x = g.insert(fact_at("X").with_subject("s")).await.unwrap();
    let y = g.insert(fact_at("Y").with_subject("t")).await.unwrap();
    let since = events(&appended).len();
    clock.set(Millis(2000));

    // Act: the replacement's content already has a live carrier, but the caller
    // asked for a state change, so it is never deduplicated.
    let z = g
        .supersede(&x, fact_at("Y").with_subject("t"))
        .await
        .unwrap();

    // Assert: x is closed, z is a new live row, y is untouched, and the index
    // follows the newest write.
    assert_ne!(z, y);
    assert_supersede_logged(&g, &appended, since, (&x, &z), (&log_id, 2000)).await;
    assert_eq!(stored_row(&g, &y, true).await.tx_to, FOREVER.0);
    assert_eq!(
        live_ids_with_subject(&g, "t").await,
        vec![y.as_str(), z.as_str()]
    );
    let logged = events(&appended);
    assert!(!has_duplicate_of(&logged), "{logged:?}");
    let hash = node_row_hash(&stored_row(&g, &z, true).await);
    assert_eq!(
        index_entry(&g, &hash).await,
        Some((logged[2].event_id.clone(), vec![z.as_str().to_string()]))
    );
}

#[tokio::test]
async fn log_write_supersede_with_content_whose_carrier_was_superseded_is_a_new_write() {
    // Arrange: X, then Y over it, so X's carrier is closed.
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let x = g.insert(fact_at("X").with_subject("s")).await.unwrap();
    let y = g
        .supersede(&x, fact_at("Y").with_subject("s"))
        .await
        .unwrap();

    // Act
    let third = g
        .supersede(&y, fact_at("X").with_subject("s"))
        .await
        .unwrap();

    // Assert
    assert_ne!(third, x);
    assert_stored_as_a_new_write(&g, &appended, &third, 3).await;
}

#[tokio::test]
async fn log_write_supersede_of_a_row_that_is_not_live_fails_without_appending() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let old = g.insert(fact_at("old").with_subject("s")).await.unwrap();
    g.supersede(&old, fact_at("new").with_subject("s"))
        .await
        .unwrap();
    let logged_before = events(&appended);

    // Act
    let refused = g.supersede(&old, fact_at("again").with_subject("s")).await;

    // Assert: the insert and the first supersede are logged, the refusal is not.
    assert_eq!(logged_before.len(), 2, "{logged_before:?}");
    assert!(
        matches!(refused, Err(Error::NodeNotFound(_))),
        "{refused:?}"
    );
    assert_eq!(events(&appended), logged_before);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
    assert!(g.insert(fact_at("after")).await.is_ok(), "log not poisoned");
}

// ---- relate dedup ----

#[tokio::test]
async fn log_write_a_second_identical_relate_returns_the_first_edge_and_logs_duplicate_of() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let src = g.insert(fact_at("src")).await.unwrap();
    let dst = g.insert(fact_at("dst")).await.unwrap();
    let since = events(&appended).len();
    let first = g.relate(&src, &dst, "mentions").await;

    // Act
    let second = g.relate(&src, &dst, "mentions").await;

    // Assert
    assert!(first.is_ok(), "{first:?}");
    assert_eq!(second.as_ref().ok(), first.as_ref().ok(), "{second:?}");
    assert_eq!(count(&g, "edges").await, 1);
    let logged = events_since(&appended, since);
    assert_eq!(logged.len(), 2, "{logged:?}");
    assert!(matches!(logged[0].payload, LogPayload::EdgeWrite(_)));
    assert_eq!(
        logged[1].payload,
        LogPayload::DuplicateOf {
            first_event_id: logged[0].event_id.clone()
        }
    );
    assert_eq!(logged[1].content_hash, logged[0].content_hash);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
}

// ---- write time, the close safety net, and the reserved relation ----

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn log_write_the_write_time_is_read_after_the_log_lock_and_the_transaction() {
    // Arrange: the write lock is held, so the insert holds the log lock and
    // waits in `begin`.
    let (log, appended) = RecordingLog::new();
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let shared = share(log);
    let g = Arc::new(open_clocked::<DefaultBackend>(":memory:", Arc::clone(&clock), shared).await);
    let held = g.backend.begin().await.unwrap();
    let insert = tokio::spawn({
        let g = Arc::clone(&g);
        async move { g.insert(fact("late")).await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Act: the clock moves while the write waits, then the write goes ahead.
    clock.set(Millis(5000));
    held.rollback().await.unwrap();
    let id = insert.await.unwrap().unwrap();

    // Assert
    assert_eq!(stored_row(&g, &id, false).await.tx_from, 5000);
    assert_eq!(events(&appended)[0].ingested_at, 5000);
}

/// `old` was superseded by `current` at 3000, and the clock has since stepped
/// back to 2500, a time at which `old` still reads as live although it is closed.
async fn graph_with_a_stepped_back_clock() -> (DefaultGraph, Appended, String, NodeId, NodeId) {
    let (g, appended, log_id, clock) = clocked_graph(Millis(1000)).await;
    let old = g.insert(fact_at("v1").with_subject("s")).await.unwrap();
    clock.set(Millis(3000));
    let current = g.upsert_by(fact_at("v2").with_subject("s")).await.unwrap();
    clock.set(Millis(2500));
    (g, appended, log_id, old, current)
}

async fn assert_voided_with_one_live_row(
    (g, appended, log_id): (&DefaultGraph, &Appended, &str),
    current: &NodeId,
    failed: Result<NodeId>,
) {
    assert!(matches!(failed, Err(Error::NodeNotFound(_))), "{failed:?}");
    assert_eq!(live_ids_with_subject(g, "s").await, vec![current.as_str()]);
    let logged = events(appended);
    let [.., write, void] = logged.as_slice() else {
        panic!("the write and its void are logged: {logged:?}");
    };
    assert!(matches!(write.payload, LogPayload::EpisodeBatch(_)));
    assert_eq!(
        void.payload,
        LogPayload::Voided {
            target_event_id: write.event_id.clone()
        }
    );
    assert_cursor_at_last_event(g, appended, log_id).await;
}

#[tokio::test]
async fn log_write_a_clock_stepping_back_voids_an_upsert_instead_of_leaving_two_live_rows() {
    // Arrange
    let (g, appended, log_id, _, current) = graph_with_a_stepped_back_clock().await;

    // Act: the competitor the earlier time finds is closed already.
    let failed = g.upsert_by(fact_at("v3").with_subject("s")).await;

    // Assert
    assert_voided_with_one_live_row((&g, &appended, &log_id), &current, failed).await;
}

#[tokio::test]
async fn log_write_a_supersede_of_a_row_closed_since_is_voided_not_a_silent_no_op() {
    // Arrange
    let (g, appended, log_id, old, current) = graph_with_a_stepped_back_clock().await;

    // Act
    let failed = g.supersede(&old, fact_at("v3").with_subject("s")).await;

    // Assert
    assert_voided_with_one_live_row((&g, &appended, &log_id), &current, failed).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn log_write_concurrent_upserts_of_one_subject_never_leave_two_live_rows() {
    // Arrange: every second `begin` is slow, and every clock read is later than
    // the one before, so a write stamped before it holds the log lock would be
    // blind to a row the other committed first.
    const SUBJECTS: usize = 20;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let (log, _) = RecordingLog::new();
    let clock = Arc::new(TickingClock(AtomicI64::new(1000)));
    let g =
        Arc::new(open_clocked::<FailingBackend>(path.to_str().unwrap(), clock, share(log)).await);
    g.backend
        .probe
        .delay_odd_begins
        .store(true, Ordering::SeqCst);

    for subject in 0..SUBJECTS {
        // Act
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let tasks: Vec<_> = (0..2)
            .map(|side| {
                let (g, barrier) = (Arc::clone(&g), Arc::clone(&barrier));
                tokio::spawn(async move {
                    barrier.wait().await;
                    let node = fact(&format!("side-{side}")).with_subject(format!("s-{subject}"));
                    g.upsert_by(node).await
                })
            })
            .collect();
        for task in tasks {
            task.await.unwrap().unwrap();
        }

        // Assert
        let live = live_ids_with_subject(&g, &format!("s-{subject}")).await;
        assert_eq!(live.len(), 1, "subject {subject}: {live:?}");
    }
    assert_eq!(count(&g, "nodes").await, (SUBJECTS * 2) as i64);
}

fn assert_reserved_refused(refused: &Result<impl std::fmt::Debug>) {
    assert!(
        matches!(refused, Err(Error::RelateRefused(message)) if message.contains("reserved")),
        "{refused:?}"
    );
}

#[tokio::test]
async fn log_write_relate_refuses_the_supersedes_relation_before_anything_is_appended() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let src = g.insert(fact_at("src")).await.unwrap();
    let dst = g.insert(fact_at("dst")).await.unwrap();
    let since = events(&appended).len();

    // Act
    let refused = g.relate(&src, &dst, relation::SUPERSEDES).await;

    // Assert: refused up front, so the log is untouched and not poisoned.
    assert_reserved_refused(&refused);
    assert_eq!(events(&appended).len(), since);
    assert_eq!(count(&g, "edges").await, 0);
    assert!(g.insert(fact_at("next")).await.is_ok());
    let unlogged = graph_at(Millis(1000)).await;
    let a = unlogged.insert(fact_at("a")).await.unwrap();
    let b = unlogged.insert(fact_at("b")).await.unwrap();
    assert!(unlogged.relate(&a, &b, relation::SUPERSEDES).await.is_ok());
}

#[tokio::test]
async fn log_write_an_episode_refuses_a_supersedes_edge_before_anything_is_appended() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let edge = EpisodeEdge {
        kind: relation::SUPERSEDES.to_string(),
        ..mentions(0, 1)
    };

    // Act
    let refused = g
        .ingest_episode(vec![fact_at("a"), fact_at("b")], vec![edge])
        .await;

    // Assert
    assert_reserved_refused(&refused);
    assert!(events(&appended).is_empty());
    assert_eq!(count(&g, "nodes").await, 0);
    assert!(g.insert(fact_at("next")).await.is_ok());
}

// ---- duplicates, repair, and replay ----

#[tokio::test]
async fn log_write_a_duplicate_is_the_live_row_among_those_an_episode_wrote_under_one_hash() {
    // Arrange: the episode's second node supersedes its identical first one, so
    // the hash index lists a closed row before the live one.
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let node = || fact_at("same").with_subject("s");
    let episode = g
        .ingest_episode(vec![node(), node()], vec![])
        .await
        .unwrap();
    let live = &episode.node_ids[1];
    assert_eq!(live_ids_with_subject(&g, "s").await, vec![live.as_str()]);

    // Act
    let again = g.insert(node()).await.unwrap();

    // Assert
    assert_eq!(&again, live);
    assert!(has_duplicate_of(&events(&appended)));
    assert_eq!(count(&g, "nodes").await, 2);
}

#[tokio::test]
async fn log_write_repair_counts_only_the_mentions_it_wrote() {
    // Arrange: m mentions both a and b, which c replaced in turn, so repairing
    // the two supersedes edges relates m to c twice and writes it once.
    let (g, _, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let m = g.insert(fact_at("m")).await.unwrap();
    let a = g.insert(fact_at("a").with_subject("s")).await.unwrap();
    g.relate(&m, &a, relation::MENTIONS).await.unwrap();
    let b = g
        .supersede(&a, fact_at("b").with_subject("s"))
        .await
        .unwrap();
    g.relate(&m, &b, relation::MENTIONS).await.unwrap();
    let c = g
        .supersede(&b, fact_at("c").with_subject("s"))
        .await
        .unwrap();

    // Act
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert
    assert_eq!(repaired, 1);
    let live = edge_between(&g, m.as_str(), c.as_str(), relation::MENTIONS).await;
    assert_eq!(live.tx_to, FOREVER.0);
}

/// Every stored column of the store's two bitemporal tables, in write order.
async fn dump_rows<B: Backend>(g: &Graph<B>) -> String {
    let nodes = "SELECT id, kind, label, content, producer, attributes, scope, subject,
                        confidence, valid_from, valid_until, tx_from, tx_to
                 FROM nodes ORDER BY rowid";
    let edges = "SELECT id, src, dst, type, attributes, tx_from, tx_to FROM edges ORDER BY rowid";
    let mut dump = String::new();
    for sql in [nodes, edges] {
        dump += &format!("{:?}\n", g.backend.query(sql, &[]).await.unwrap());
    }
    dump
}

#[tokio::test]
async fn log_write_replaying_the_logged_payloads_reproduces_the_live_rows() {
    // Arrange: one history with every write shape, including an episode whose
    // second node supersedes its first and a third that supersedes a stored row.
    let (g, appended, _, clock) = clocked_graph::<DefaultBackend>(Millis(1000)).await;
    let a = g.insert(fact_at("a").with_subject("s1")).await.unwrap();
    clock.set(Millis(2000));
    let a2 = g
        .supersede(&a, fact_at("a2").with_subject("s1"))
        .await
        .unwrap();
    clock.set(Millis(3000));
    let b = g.insert(fact_at("b")).await.unwrap();
    g.relate(&a2, &b, relation::MENTIONS).await.unwrap();
    clock.set(Millis(4000));
    let nodes = vec![
        fact_at("e0").with_subject("s2"),
        fact_at("e1").with_subject("s2"),
        fact_at("e2").with_subject("s1"),
    ];
    g.ingest_episode(nodes, vec![mentions(1, 2)]).await.unwrap();

    // Act: apply each payload's own statements to an empty store.
    let replica = graph_at(Millis(1000)).await;
    for event in events(&appended) {
        let mut tx = replica.backend.begin().await.unwrap();
        apply_steps(&mut *tx, &steps_for(&event.payload))
            .await
            .unwrap();
        tx.commit().await.unwrap();
    }

    // Assert
    assert_eq!(count(&replica, "edges").await, 5);
    assert_eq!(dump_rows(&replica).await, dump_rows(&g).await);
}

#[tokio::test]
async fn log_write_relate_refuses_a_live_twin_the_log_never_recorded() {
    // Arrange: the edge was written before a log was attached, so it is live
    // but not in the hash index.
    let unlogged = graph_at(Millis(1000)).await;
    let src = unlogged.insert(fact_at("src")).await.unwrap();
    let dst = unlogged.insert(fact_at("dst")).await.unwrap();
    unlogged.relate(&src, &dst, "mentions").await.unwrap();
    let (log, appended) = RecordingLog::new();
    let g = unlogged.with_log(share(log)).await.unwrap();

    // Act
    let refused = g.relate(&src, &dst, "mentions").await;

    // Assert: only an indexed twin can be answered as a duplicate.
    assert!(
        matches!(refused, Err(Error::RelateRefused(_))),
        "{refused:?}"
    );
    assert!(events(&appended).is_empty());
    assert_eq!(count(&g, "edges").await, 1);
}

#[tokio::test]
async fn log_write_relate_after_the_first_edge_was_closed_or_deleted_is_a_new_write() {
    for how in ["closed", "deleted"] {
        // Arrange
        let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
        let src = g.insert(fact_at("src")).await.unwrap();
        let dst = g.insert(fact_at("dst")).await.unwrap();
        let since = events(&appended).len();
        let first = g.relate(&src, &dst, "mentions").await.unwrap();
        if how == "closed" {
            g.backend
                .execute(
                    "UPDATE edges SET tx_to = ?1 WHERE id = ?2",
                    &[Millis(1000).into(), first.as_str().into()],
                )
                .await
                .unwrap();
        } else {
            g.backend
                .execute("DELETE FROM edges WHERE id = ?1", &[first.as_str().into()])
                .await
                .unwrap();
        }

        // Act
        let second = g.relate(&src, &dst, "mentions").await;

        // Assert: stored and logged as a write, and the index follows it.
        assert!(second.is_ok(), "{how}: {second:?}");
        let second = second.unwrap();
        assert_ne!(second, first, "{how}");
        let live = g
            .backend
            .query("SELECT id FROM edges WHERE tx_to = ?1", &[FOREVER.into()])
            .await
            .unwrap();
        assert_eq!(live.len(), 1, "{how}");
        assert_eq!(live[0].get_string(0).unwrap(), second.as_str(), "{how}");
        let logged = events_since(&appended, since);
        assert_eq!(logged.len(), 2, "{how}: {logged:?}");
        assert!(!has_duplicate_of(&logged), "{how}: {logged:?}");
        let stored = stored_edge(&g, second.as_str()).await;
        assert_eq!(logged[1].payload, LogPayload::EdgeWrite(stored.clone()));
        assert_eq!(
            index_entry(&g, &edge_row_hash(&stored)).await,
            Some((
                logged[1].event_id.clone(),
                vec![second.as_str().to_string()]
            )),
            "{how}"
        );
    }
}

#[tokio::test]
async fn log_write_a_refused_relate_appends_nothing_and_leaves_the_log_usable() {
    // Arrange: an edge exists, then its source is superseded.
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let src = g.insert(fact_at("src").with_subject("s")).await.unwrap();
    let dst = g.insert(fact_at("dst")).await.unwrap();
    g.relate(&src, &dst, "mentions").await.unwrap();
    g.supersede(&src, fact_at("src2").with_subject("s"))
        .await
        .unwrap();
    let logged_before = events(&appended);
    let cursor_before = cursor(&g).await;

    // Act: the source is no longer live, though the edge's hash is indexed.
    let refused = g.relate(&src, &dst, "mentions").await;

    // Assert: a refusal is not a duplicate and is not logged.
    assert_eq!(logged_before.len(), 4, "{logged_before:?}");
    assert!(
        matches!(&refused, Err(Error::RelateRefused(m)) if m.contains("not live")),
        "{refused:?}"
    );
    assert_eq!(events(&appended), logged_before);
    assert_eq!(cursor(&g).await, cursor_before);
    assert_cursor_at_last_event(&g, &appended, &log_id).await;
    assert!(g.insert(fact_at("after")).await.is_ok(), "log not poisoned");
}

// ---- no log injected ----

#[tokio::test]
async fn log_write_upsert_by_without_a_log_behaves_as_before() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let first = g.upsert_by(fact("same").with_subject("s")).await.unwrap();

    // Act
    let second = g.upsert_by(fact("same").with_subject("s")).await.unwrap();

    // Assert: no dedup, the older version is closed and linked.
    assert_ne!(second, first);
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(live_ids_with_subject(&g, "s").await, vec![second.as_str()]);
    edge_between(&g, second.as_str(), first.as_str(), relation::SUPERSEDES).await;
    assert_log_tables_untouched(&g).await;
}

#[tokio::test]
async fn log_write_supersede_without_a_log_behaves_as_before() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let old = g.insert(fact("old")).await.unwrap();

    // Act
    let new = g.supersede(&old, fact("new")).await.unwrap();
    let refused = g.supersede(&old, fact("again")).await;

    // Assert
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(stored_row(&g, &old, false).await.tx_to, 1000);
    edge_between(&g, new.as_str(), old.as_str(), relation::SUPERSEDES).await;
    assert!(
        matches!(refused, Err(Error::NodeNotFound(_))),
        "{refused:?}"
    );
    assert_log_tables_untouched(&g).await;
}

#[tokio::test]
async fn log_write_relate_without_a_log_behaves_as_before() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let src = g.insert(fact("src")).await.unwrap();
    let dst = g.insert(fact("dst")).await.unwrap();

    // Act
    let first = g.relate(&src, &dst, "mentions").await;
    let second = g.relate(&src, &dst, "mentions").await;

    // Assert: the repeat is refused, not deduplicated.
    assert!(first.is_ok(), "{first:?}");
    assert!(
        matches!(&second, Err(Error::RelateRefused(m)) if m.contains("already relates")),
        "{second:?}"
    );
    assert_eq!(count(&g, "edges").await, 1);
    assert_log_tables_untouched(&g).await;
}

#[tokio::test]
async fn log_write_ingest_episode_without_a_log_behaves_as_before() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let result = g
        .ingest_episode(vec![fact("twin"), fact("twin")], vec![mentions(0, 1)])
        .await
        .unwrap();

    // Assert: both rows are stored and nothing is deduplicated or logged.
    assert_eq!(result.node_ids.len(), 2);
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(count(&g, "edges").await, 1);
    assert_log_tables_untouched(&g).await;
}
