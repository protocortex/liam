// SPDX-License-Identifier: Apache-2.0
//! Log-first write path: every write appends to the injected log inside one
//! transaction. Tests name the log through `RecordingLog`, a double that keeps
//! what it accepted so a test can compare the log against the projection.

use std::sync::Mutex as StdMutex;
use std::time::Duration;

use liam_log::event::{LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
use liam_log::hash::node_row_hash;
use liam_log::test_support::FailingLogWriter;
use liam_log::wal::WalError;
use liam_log::{LogOffset, LogWriter};
use uuid::Uuid;

use super::*;
use crate::DefaultBackend;

type Appended = Arc<StdMutex<Vec<(LogOffset, LogEvent)>>>;

/// A log that records every accepted event and can be told to stall some
/// appends, so a concurrency test can make the slow append lose a race.
struct RecordingLog {
    inner: FailingLogWriter,
    appended: Appended,
    stall_odd_appends: bool,
    attempts: u64,
}

impl RecordingLog {
    fn new() -> (Self, Appended) {
        let appended = Appended::default();
        let log = Self {
            inner: FailingLogWriter::fail_after(u64::MAX),
            appended: Arc::clone(&appended),
            stall_odd_appends: false,
            attempts: 0,
        };
        (log, appended)
    }

    fn stalling_odd_appends(mut self) -> Self {
        self.stall_odd_appends = true;
        self
    }
}

impl LogWriter for RecordingLog {
    fn append(&mut self, event: &LogEvent) -> std::result::Result<LogOffset, WalError> {
        self.attempts += 1;
        if self.stall_odd_appends && self.attempts % 2 == 1 {
            std::thread::sleep(Duration::from_millis(10));
        }
        let offset = self.inner.append(event)?;
        self.appended.lock().unwrap().push((offset, event.clone()));
        Ok(offset)
    }

    fn log_id(&self) -> Uuid {
        self.inner.log_id()
    }
}

fn share(writer: impl LogWriter + 'static) -> SharedLog {
    Arc::new(tokio::sync::Mutex::new(Box::new(writer)))
}

async fn open_with<B: Backend>(path: &str, t: Millis, log: SharedLog) -> Graph<B> {
    let clock = Arc::new(FixedClock::new(t));
    Graph::<B>::open_with_clock(path, GraphConfig::new(8), clock)
        .await
        .expect("open graph")
        .with_log(log)
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
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let node = fact("content")
        .with_scope(" proj/a ")
        .with_subject("subject-1")
        .with_attributes(serde_json::json!({"k": "v"}));

    // Act
    let id = g.insert(node).await.unwrap();

    // Assert
    let logged = events(&appended);
    assert_eq!(logged.len(), 1, "one insert appends exactly one event");
    let event = &logged[0];
    let stored = stored_row(&g, &id, false).await;
    assert_eq!(event.payload, LogPayload::NodeWrite(stored.clone()));
    assert_eq!(event.content_hash, node_row_hash(&stored));
    assert_eq!(event.schema_version, CURRENT_SCHEMA_VERSION);
    assert_eq!(event.encryption_key_id, None);
    assert_eq!(event.ingested_at, 1000);
    assert!(!event.event_id.is_empty());
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
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let first = g.insert(fact("same")).await.unwrap();

    // Act
    let second = g.insert(fact("same")).await.unwrap();

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
}

#[tokio::test]
async fn log_write_content_that_differs_only_by_scope_whitespace_is_a_duplicate() {
    // Arrange: the hash covers the resolved row, so a trimmed scope matches.
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let first = g.insert(fact("same").with_scope("proj/a")).await.unwrap();

    // Act
    let second = g
        .insert(fact("same").with_scope("  proj/a "))
        .await
        .unwrap();

    // Assert
    assert_eq!(second, first);
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(node_writes(&appended).len(), 1);
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
async fn log_write_a_new_hash_is_indexed_against_its_event_and_row() {
    // Arrange
    let (g, appended, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;

    // Act
    let id = g.insert(fact("content")).await.unwrap();

    // Assert
    let logged = events(&appended);
    assert_eq!(logged.len(), 1, "one insert appends exactly one event");
    let event = &logged[0];
    let rows = g
        .backend
        .query(
            "SELECT content_hash, first_event_id, row_ids FROM log_hash_index",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(
        matches!(&rows[0].0[0], Value::Blob(hash) if hash.as_slice() == event.content_hash),
        "index key must be the event's content hash"
    );
    assert_eq!(rows[0].get_string(1).unwrap(), event.event_id);
    let row_ids: Vec<String> = serde_json::from_str(&rows[0].get_string(2).unwrap()).unwrap();
    assert_eq!(row_ids, vec![id.as_str().to_string()]);
}

#[tokio::test]
async fn log_write_cursor_advances_to_the_appended_offset_in_the_same_commit() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;

    // Act
    g.insert(fact("one")).await.unwrap();
    let after_first = cursor(&g).await;
    g.insert(fact("two")).await.unwrap();
    let after_second = cursor(&g).await;

    // Assert
    let offsets: Vec<_> = appended.lock().unwrap().iter().map(|(o, _)| *o).collect();
    assert_eq!(offsets.len(), 2);
    assert_eq!(
        after_first,
        Some((log_id.clone(), Some(offset_pair(offsets[0]))))
    );
    assert_eq!(after_second, Some((log_id, Some(offset_pair(offsets[1])))));
}

#[tokio::test]
async fn log_write_append_failure_fails_the_insert_and_leaves_no_trace() {
    // Arrange: the first append fails, every later one succeeds.
    let log = share(FailingLogWriter::fail_on_nth(1));
    let g = open_with::<DefaultBackend>(":memory:", Millis(1000), log).await;
    let before = cursor(&g).await;

    // Act
    let failed = g.insert(fact("same")).await;

    // Assert: no row, no index entry, no cursor change.
    assert!(failed.is_err(), "the insert must report the append failure");
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
async fn log_write_projection_failure_after_append_appends_voided_and_returns_the_error() {
    // Arrange: the node INSERT is the transaction's first execute.
    let (g, appended, _) = logged_graph::<FailingBackend>(Millis(1000)).await;
    g.backend.set_fail_on_execute(0);

    // Act
    let failed = g.insert(fact("same")).await;

    // Assert
    assert!(matches!(failed, Err(Error::Backend(_))), "{failed:?}");
    assert_eq!(count(&g, "nodes").await, 0);
    assert_eq!(count(&g, "log_hash_index").await, 0);
    let logged = events(&appended);
    assert_eq!(logged.len(), 2, "the write, then its void: {logged:?}");
    assert!(matches!(logged[0].payload, LogPayload::NodeWrite(_)));
    assert_eq!(
        logged[1].payload,
        LogPayload::Voided {
            target_event_id: logged[0].event_id.clone()
        }
    );

    // Act: a retry after the fault clears is a first write, not a duplicate.
    g.backend.set_fail_on_execute(usize::MAX);
    let retried = g.insert(fact("same")).await;

    // Assert
    assert!(retried.is_ok(), "{retried:?}");
    assert_eq!(count(&g, "nodes").await, 1);
    assert!(matches!(
        events(&appended).last().unwrap().payload,
        LogPayload::NodeWrite(_)
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn log_write_concurrent_inserts_keep_wal_order_equal_to_commit_order() {
    // Arrange: a stalled append per pair gives a writer that appends outside
    // the transaction's critical section room to be overtaken.
    const PAIRS: usize = 20;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let (log, appended) = RecordingLog::new();
    let g = Arc::new(
        open_with::<DefaultBackend>(
            path.to_str().unwrap(),
            Millis(1000),
            share(log.stalling_odd_appends()),
        )
        .await,
    );

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

    // Assert: node rowids are assigned in commit order.
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
async fn log_write_vector_failure_leaves_the_node_committed_and_logged() {
    // Arrange
    let (g, appended, log_id) = logged_graph::<FailingVectorBackend>(Millis(1000)).await;
    g.backend.set_fail_on_vector_upsert(0);
    let embedding = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Act
    let result = g.insert(fact("embedded").with_embedding(embedding)).await;

    // Assert: the error surfaces, but the write is durable in both places.
    assert!(result.is_err(), "the vector failure must be reported");
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(node_writes(&appended).len(), 1);
    assert_eq!(
        cursor(&g).await,
        Some((log_id, Some(offset_pair(last_offset(&appended)))))
    );
}

#[tokio::test]
async fn log_write_voided_append_moves_the_cursor_past_the_failed_write() {
    // Arrange: the write is appended, its projection fails, its void is appended.
    let (g, appended, log_id) = logged_graph::<FailingBackend>(Millis(1000)).await;
    g.backend.set_fail_on_execute(0);

    // Act
    let failed = g.insert(fact("same")).await;

    // Assert: the cursor tracks the last record the store accounted for, so it
    // sits on the void, not on the write it cancelled.
    assert!(failed.is_err());
    let logged = events(&appended);
    assert!(matches!(logged[1].payload, LogPayload::Voided { .. }));
    assert_eq!(
        cursor(&g).await,
        Some((log_id, Some(offset_pair(last_offset(&appended)))))
    );
}

#[tokio::test]
async fn log_write_a_void_double_fault_poisons_the_log_until_reopen() {
    // Arrange: the write appends, its projection fails, and the void append
    // (the log's second) fails too.
    let log = share(FailingLogWriter::fail_on_nth(2));
    let g = open_with::<FailingBackend>(":memory:", Millis(1000), log).await;
    g.backend.set_fail_on_execute(0);

    // Act
    let first = g.insert(fact("first")).await;
    g.backend.set_fail_on_execute(usize::MAX);
    let second = g.insert(fact("second")).await;

    // Assert: the original error surfaces, then every logged write is refused.
    assert!(matches!(first, Err(Error::Backend(_))), "{first:?}");
    assert!(matches!(second, Err(Error::LogPoisoned)), "{second:?}");
    assert_eq!(count(&g, "nodes").await, 0);
    assert_eq!(count(&g, "log_hash_index").await, 0);
}
