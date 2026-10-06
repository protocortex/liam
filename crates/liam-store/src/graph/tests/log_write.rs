// SPDX-License-Identifier: Apache-2.0
//! Log-first write path: every write appends to the injected log inside one
//! transaction. Tests name the log through `RecordingLog`, a double that keeps
//! what it accepted so a test can compare the log against the projection.

use std::io;
use std::sync::atomic::Ordering;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::event::{LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
use liam_log::hash::node_row_hash;
use liam_log::test_support::FailingLogWriter;
use liam_log::wal::WalError;
use liam_log::{LogOffset, LogWriter};
use uuid::Uuid;

use super::*;
use crate::DefaultBackend;

type Appended = Arc<StdMutex<Vec<(LogOffset, LogEvent)>>>;

/// A log that records every event it accepted, over a writer that can be told
/// to fail, so a test can compare the log against the projection.
struct RecordingLog {
    inner: FailingLogWriter,
    appended: Appended,
    fail_after_recording: bool,
}

impl RecordingLog {
    fn new() -> (Self, Appended) {
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

fn share(writer: impl LogWriter + 'static) -> SharedLog {
    let bloom = HashBloom::new(BloomConfig::default());
    Arc::new(tokio::sync::Mutex::new(EventLog::new(
        Box::new(writer),
        bloom,
    )))
}

async fn open_with<B: Backend>(path: &str, t: Millis, log: SharedLog) -> Graph<B> {
    let clock = Arc::new(FixedClock::new(t));
    let graph = Graph::<B>::open_with_clock(path, GraphConfig::new(8), clock)
        .await
        .expect("open graph");
    graph.with_log(log).await.expect("attach log")
}

/// A logged in-memory graph, the log's recorder, and the log's id.
async fn logged_graph<B: Backend>(t: Millis) -> (Graph<B>, Appended, String) {
    let (log, appended) = RecordingLog::new();
    let log_id = log.log_id().to_string();
    (open_with(":memory:", t, share(log)).await, appended, log_id)
}

fn fact(content: &str) -> NewNode {
    NewNode::now("fact", "label", content)
        .with_producer("agent-a")
        .with_confidence(0.75)
}

/// `fact` with a supplied valid time, so source, trust, valid time, and ingest
/// time are four different values on the log record.
fn fact_at(content: &str) -> NewNode {
    fact(content).with_valid_from(Millis(500))
}

fn assert_envelope(event: &LogEvent) {
    let envelope = (
        event.source.as_str(),
        event.trust_score,
        event.observed_at,
        event.ingested_at,
    );
    assert_eq!(envelope, ("agent-a", 0.75, 500, 1000), "{event:?}");
}

fn events(appended: &Appended) -> Vec<LogEvent> {
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

async fn count<B: Backend>(g: &Graph<B>, table: &str) -> i64 {
    let rows = g
        .backend
        .query(&format!("SELECT COUNT(*) FROM {table}"), &[])
        .await
        .unwrap();
    rows[0].get_i64(0).unwrap()
}

/// `None` when the cursor row does not exist yet; otherwise its log id and
/// last applied offset (`None` while the offsets are still NULL).
async fn cursor<B: Backend>(g: &Graph<B>) -> Option<(String, Option<(i64, i64)>)> {
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

fn offset_pair(offset: LogOffset) -> (i64, i64) {
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
    // Arrange: write X, then supersede it with Y. The supersede path is the
    // existing unlogged one, which is enough to close X.
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
    // Arrange: two handles on one database and one log. The log's third append,
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
