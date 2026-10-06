// SPDX-License-Identifier: Apache-2.0
//! Log-first write path under faults and concurrency: a failure at every
//! transaction statement of each write path is voided and leaves the projection
//! untouched, and concurrent writes keep the log in commit order.

use std::sync::atomic::Ordering;

use liam_log::event::LogPayload;
use liam_log::hash::content_hashes;
use liam_log::LogWriter;

use super::log_write::{
    assert_cursor_at_last_event, count, cursor, events, events_since, fact, fact_at,
    live_ids_with_subject, logged_graph, mentions, offset_pair, open_with, share, Appended,
    RecordingLog,
};
use super::*;
use crate::DefaultBackend;

// ---- projection and cursor faults, per write path ----

/// A snapshot of what the projection and the hash index hold.
#[derive(Debug, PartialEq)]
struct Snapshot {
    nodes: Vec<(String, i64)>,
    edges: Vec<(String, i64)>,
    index: Vec<(Vec<u8>, String, String)>,
}

async fn id_and_tx_to<B: Backend>(g: &Graph<B>, table: &str) -> Vec<(String, i64)> {
    g.backend
        .query(&format!("SELECT id, tx_to FROM {table} ORDER BY id"), &[])
        .await
        .unwrap()
        .iter()
        .map(|row| (row.get_string(0).unwrap(), row.get_i64(1).unwrap()))
        .collect()
}

async fn snapshot<B: Backend>(g: &Graph<B>) -> Snapshot {
    let index = g
        .backend
        .query(
            "SELECT content_hash, first_event_id, row_ids FROM log_hash_index
             ORDER BY content_hash",
            &[],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| match &row.0[0] {
            Value::Blob(hash) => (
                hash.clone(),
                row.get_string(1).unwrap(),
                row.get_string(2).unwrap(),
            ),
            other => panic!("content_hash is not a blob: {other:?}"),
        })
        .collect();
    Snapshot {
        nodes: id_and_tx_to(g, "nodes").await,
        edges: id_and_tx_to(g, "edges").await,
        index,
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    UpsertInsert,
    UpsertSupersede,
    Supersede,
    Relate,
    Episode,
}

impl Op {
    /// The fewest statements the path's transaction runs: its rows, one hash
    /// index upsert per row it carries, and the cursor. A count below this means
    /// the harness stopped before the end of the write.
    fn min_statements(self) -> usize {
        match self {
            Op::UpsertInsert | Op::Relate => 3,
            Op::UpsertSupersede | Op::Supersede => 6,
            Op::Episode => 12,
        }
    }
}

/// A live node `x` (subject "s") and a live peer `y`, written before a fault
/// is armed.
struct Seeded {
    x: NodeId,
    y: NodeId,
}

async fn seed<B: Backend>(g: &Graph<B>) -> Seeded {
    Seeded {
        x: g.insert(fact_at("v1").with_subject("s")).await.unwrap(),
        y: g.insert(fact_at("peer")).await.unwrap(),
    }
}

async fn run_op<B: Backend>(g: &Graph<B>, op: Op, seeded: &Seeded) -> Result<()> {
    match op {
        Op::UpsertInsert => g
            .upsert_by(fact_at("new").with_subject("other"))
            .await
            .map(drop),
        Op::UpsertSupersede => g.upsert_by(fact_at("v2").with_subject("s")).await.map(drop),
        Op::Supersede => g
            .supersede(&seeded.x, fact_at("v2").with_subject("s"))
            .await
            .map(drop),
        Op::Relate => g.relate(&seeded.x, &seeded.y, "mentions").await.map(drop),
        Op::Episode => {
            let cites = EpisodeEdge {
                from: EpisodeRef::New(1),
                to: EpisodeRef::Existing(seeded.y.clone()),
                kind: "cites".to_string(),
                attributes: serde_json::json!({}),
            };
            g.ingest_episode(
                vec![fact_at("v2").with_subject("s"), fact_at("ep-peer")],
                vec![mentions(0, 1), cites],
            )
            .await
            .map(drop)
        }
    }
}

/// What fails during the write under test.
#[derive(Clone, Copy, Debug)]
enum Fault {
    /// The transaction statement at this 0-based index.
    Statement(usize),
    /// The commit that follows the last statement.
    Commit,
}

/// One run of `op` on a fresh logged graph with a `Fault` armed.
struct Faulted {
    g: Graph<FailingBackend>,
    appended: Appended,
    log_id: String,
    shared: SharedLog,
    seeded: Seeded,
    before: Snapshot,
    seeded_events: usize,
    result: Result<()>,
}

async fn run_with_fault(op: Op, fault: Fault) -> Faulted {
    let (log, appended) = RecordingLog::new();
    let log_id = log.log_id().to_string();
    let shared = share(log);
    let g = open_with::<FailingBackend>(":memory:", Millis(1000), Arc::clone(&shared)).await;
    let seeded = seed(&g).await;
    let before = snapshot(&g).await;
    let seeded_events = events(&appended).len();
    match fault {
        Fault::Statement(index) => g.backend.set_fail_on_execute(index),
        Fault::Commit => g.backend.set_fail_commit(true),
    }
    let result = run_op(&g, op, &seeded).await;
    Faulted {
        g,
        appended,
        log_id,
        shared,
        seeded,
        before,
        seeded_events,
        result,
    }
}

/// How many statements `op` runs in its transaction: the first failing index
/// that no longer fires. Every index below it must have failed with the
/// injector's own error, so a fault that went unnoticed, or a write that failed
/// for another reason, cannot shorten the count.
async fn statement_count(op: Op) -> usize {
    for index in 0..40 {
        let result = run_with_fault(op, Fault::Statement(index)).await.result;
        match result {
            Ok(()) => {
                let at_least = op.min_statements();
                assert!(
                    index >= at_least,
                    "{op:?} ran {index} statements, expected {at_least}"
                );
                return index;
            }
            Err(_) => assert!(injected(&result), "{op:?}, statement {index}: {result:?}"),
        }
    }
    panic!("{op:?} still fails after 40 statements");
}

fn injected(result: &Result<()>) -> bool {
    matches!(result, Err(Error::Backend(message)) if message.contains("injected"))
}

/// The fault fired, nothing committed, and the log holds the write followed by
/// its void with the cursor on the void.
async fn assert_voided(faulted: &Faulted, context: &str) {
    assert!(injected(&faulted.result), "{context}: {:?}", faulted.result);
    assert_eq!(snapshot(&faulted.g).await, faulted.before, "{context}");
    let logged = events_since(&faulted.appended, faulted.seeded_events);
    assert_eq!(logged.len(), 2, "{context}: the write, then its void");
    assert_eq!(
        logged[1].payload,
        LogPayload::Voided {
            target_event_id: logged[0].event_id.clone()
        },
        "{context}"
    );
    assert_cursor_at_last_event(&faulted.g, &faulted.appended, &faulted.log_id).await;
    let bloom = faulted.shared.lock().await;
    for (hash, row_id) in content_hashes(&logged[0]) {
        assert!(
            !bloom.bloom_might_contain(&hash),
            "{context}: voided row {row_id} must not stay in the filter"
        );
    }
}

/// A retry once the fault clears is a first write, not a duplicate.
async fn assert_retry_is_a_first_write(faulted: &Faulted, op: Op, context: &str) {
    faulted.g.backend.set_fail_on_execute(usize::MAX);
    faulted.g.backend.set_fail_commit(false);

    let retried = run_op(&faulted.g, op, &faulted.seeded).await;

    assert!(retried.is_ok(), "{context}: {retried:?}");
    let last = events(&faulted.appended).pop().unwrap();
    assert!(
        !matches!(
            last.payload,
            LogPayload::DuplicateOf { .. } | LogPayload::Voided { .. }
        ),
        "{context}: {last:?}"
    );
    assert_ne!(snapshot(&faulted.g).await, faulted.before, "{context}");
}

async fn assert_failure_at_every_statement_is_voided(op: Op) {
    for index in 0..statement_count(op).await {
        // Arrange and Act
        let faulted = run_with_fault(op, Fault::Statement(index)).await;

        // Assert
        let context = format!("{op:?}, failing execute {index}");
        assert_voided(&faulted, &context).await;
        assert_retry_is_a_first_write(&faulted, op, &context).await;
    }
}

async fn assert_commit_failure_is_voided(op: Op) {
    // Arrange and Act
    let faulted = run_with_fault(op, Fault::Commit).await;

    // Assert
    let context = format!("{op:?}, failing the commit");
    assert_voided(&faulted, &context).await;
    assert_retry_is_a_first_write(&faulted, op, &context).await;
}

/// The cursor is the last statement of the write's transaction, so failing it
/// must undo the projection with it.
async fn assert_cursor_failure_commits_nothing(op: Op) {
    // Arrange
    let statements = statement_count(op).await;

    // Act
    let faulted = run_with_fault(op, Fault::Statement(statements - 1)).await;

    // Assert
    assert_voided(&faulted, &format!("{op:?}, failing the cursor statement")).await;
    let write_offset = faulted.appended.lock().unwrap()[faulted.seeded_events].0;
    assert_ne!(
        cursor(&faulted.g).await.and_then(|(_, offset)| offset),
        Some(offset_pair(write_offset)),
        "the cursor must not sit on the voided write"
    );
}

#[tokio::test]
async fn log_write_upsert_by_projection_failure_voids_the_insert() {
    assert_failure_at_every_statement_is_voided(Op::UpsertInsert).await;
}

#[tokio::test]
async fn log_write_upsert_by_projection_failure_voids_the_supersede() {
    assert_failure_at_every_statement_is_voided(Op::UpsertSupersede).await;
}

#[tokio::test]
async fn log_write_supersede_projection_failure_voids_the_batch() {
    assert_failure_at_every_statement_is_voided(Op::Supersede).await;
}

#[tokio::test]
async fn log_write_relate_projection_failure_voids_the_edge_write() {
    assert_failure_at_every_statement_is_voided(Op::Relate).await;
}

#[tokio::test]
async fn log_write_ingest_episode_projection_failure_voids_the_whole_batch() {
    assert_failure_at_every_statement_is_voided(Op::Episode).await;
}

#[tokio::test]
async fn log_write_upsert_by_insert_cursor_failure_commits_nothing() {
    assert_cursor_failure_commits_nothing(Op::UpsertInsert).await;
}

#[tokio::test]
async fn log_write_upsert_by_supersede_cursor_failure_commits_nothing() {
    assert_cursor_failure_commits_nothing(Op::UpsertSupersede).await;
}

#[tokio::test]
async fn log_write_supersede_cursor_failure_commits_nothing() {
    assert_cursor_failure_commits_nothing(Op::Supersede).await;
}

#[tokio::test]
async fn log_write_relate_cursor_failure_commits_nothing() {
    assert_cursor_failure_commits_nothing(Op::Relate).await;
}

#[tokio::test]
async fn log_write_ingest_episode_cursor_failure_commits_nothing() {
    assert_cursor_failure_commits_nothing(Op::Episode).await;
}

#[tokio::test]
async fn log_write_upsert_by_insert_commit_failure_is_voided() {
    assert_commit_failure_is_voided(Op::UpsertInsert).await;
}

#[tokio::test]
async fn log_write_upsert_by_supersede_commit_failure_is_voided() {
    assert_commit_failure_is_voided(Op::UpsertSupersede).await;
}

#[tokio::test]
async fn log_write_supersede_commit_failure_is_voided() {
    assert_commit_failure_is_voided(Op::Supersede).await;
}

#[tokio::test]
async fn log_write_relate_commit_failure_is_voided() {
    assert_commit_failure_is_voided(Op::Relate).await;
}

#[tokio::test]
async fn log_write_ingest_episode_commit_failure_is_voided() {
    assert_commit_failure_is_voided(Op::Episode).await;
}

// ---- concurrency ----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn log_write_concurrent_upsert_by_calls_look_up_the_live_row_inside_the_transaction() {
    // Arrange: one live row, then writers that all upsert its subject at once.
    const WRITERS: usize = 8;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let (log, appended) = RecordingLog::new();
    let g = Arc::new(
        open_with::<FailingBackend>(path.to_str().unwrap(), Millis(1000), share(log)).await,
    );
    g.upsert_by(fact("v-seed").with_subject("s")).await.unwrap();
    let since = events(&appended).len();
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    // Act
    let tasks: Vec<_> = (0..WRITERS)
        .map(|writer| {
            let (g, barrier) = (Arc::clone(&g), Arc::clone(&barrier));
            tokio::spawn(async move {
                barrier.wait().await;
                g.upsert_by(fact(&format!("v-{writer}")).with_subject("s"))
                    .await
            })
        })
        .collect();
    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap());
    }

    // Assert: each write superseded the row the one before it left live, so
    // the versions form one chain with one live end.
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let logged = events_since(&appended, since);
    assert_eq!(logged.len(), WRITERS, "{logged:?}");
    assert!(logged
        .iter()
        .all(|event| matches!(event.payload, LogPayload::EpisodeBatch(_))));
    assert_eq!(live_ids_with_subject(&g, "s").await.len(), 1);
    let closed = g
        .backend
        .query(
            "SELECT COUNT(DISTINCT dst) FROM edges WHERE type = ?1",
            &[relation::SUPERSEDES.into()],
        )
        .await
        .unwrap();
    assert_eq!(closed[0].get_i64(0).unwrap(), WRITERS as i64);
}

#[derive(Clone, Copy)]
enum Kind {
    Upsert,
    Supersede,
    Relate,
    Episode,
}

const KINDS: [Kind; 4] = [Kind::Upsert, Kind::Supersede, Kind::Relate, Kind::Episode];

async fn mixed_write(
    g: &Graph<FailingBackend>,
    kind: Kind,
    tag: usize,
    (target, a, b): (&NodeId, &NodeId, &NodeId),
) -> Result<()> {
    match kind {
        Kind::Upsert => g
            .upsert_by(fact(&format!("up-{tag}")).with_subject(format!("subject-{tag}")))
            .await
            .map(drop),
        Kind::Supersede => g
            .supersede(target, fact(&format!("sup-{tag}")))
            .await
            .map(drop),
        Kind::Relate => g.relate(a, b, &format!("rel-{tag}")).await.map(drop),
        Kind::Episode => g
            .ingest_episode(
                vec![fact(&format!("ep-{tag}-0")), fact(&format!("ep-{tag}-1"))],
                vec![mentions(0, 1)],
            )
            .await
            .map(drop),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn log_write_mixed_concurrent_writes_keep_wal_order_equal_to_commit_order() {
    // Arrange: the first `begin` of each pair is slow, so a write that appended
    // before its `begin` and let go of the log lock would be overtaken by the
    // second, and the log order would no longer be the commit order. Each write
    // type runs against a row of its own.
    const PAIRS: usize = 12;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("graph.db");
    let (log, appended) = RecordingLog::new();
    let shared = share(log);
    let g = Arc::new(
        open_with::<FailingBackend>(path.to_str().unwrap(), Millis(1000), Arc::clone(&shared))
            .await,
    );
    let mut targets = Vec::new();
    for tag in 0..PAIRS * 2 {
        let seed = fact(&format!("seed-{tag}")).with_subject(format!("subject-{tag}"));
        targets.push(g.upsert_by(seed).await.unwrap());
    }
    let targets = Arc::new(targets);
    let a = g.insert(fact("relate-a")).await.unwrap();
    let b = g.insert(fact("relate-b")).await.unwrap();
    let since = events(&appended).len();
    g.backend.probe.begins.store(0, Ordering::SeqCst);
    g.backend
        .probe
        .delay_even_begins
        .store(true, Ordering::SeqCst);
    *g.backend.probe.watched.lock().unwrap() = Some(shared);

    // Act
    for pair in 0..PAIRS {
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let tasks: Vec<_> = (0..2)
            .map(|side| {
                let tag = pair * 2 + side;
                let kind = KINDS[(pair + side) % KINDS.len()];
                let (g, barrier, targets) =
                    (Arc::clone(&g), Arc::clone(&barrier), Arc::clone(&targets));
                let (a, b) = (a.clone(), b.clone());
                tokio::spawn(async move {
                    barrier.wait().await;
                    mixed_write(&g, kind, tag, (&targets[tag], &a, &b)).await
                })
            })
            .collect();
        for task in tasks {
            let written = task.await.unwrap();
            assert!(written.is_ok(), "{written:?}");
        }
    }

    // Assert: each write registers its rows in the hash index inside its own
    // transaction, so the index's row order is the commit order.
    let wal_order: Vec<String> = events_since(&appended, since)
        .into_iter()
        .map(|event| event.event_id)
        .collect();
    assert_eq!(wal_order.len(), PAIRS * 2);
    let mut commit_order: Vec<String> = g
        .backend
        .query(
            "SELECT first_event_id FROM log_hash_index ORDER BY rowid",
            &[],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .filter(|event_id| wal_order.contains(event_id))
        .collect();
    commit_order.dedup();
    assert_eq!(wal_order, commit_order);
    let lock_held = g.backend.probe.commit_lock_held.lock().unwrap().clone();
    assert_eq!(lock_held, vec![true; PAIRS * 2]);
}

// ---- ingest_episode atomicity ----

#[tokio::test]
async fn log_write_a_failing_episode_edge_voids_the_whole_batch_and_keeps_no_node() {
    // Arrange: the same edge twice, so the second insert is refused after both
    // nodes and the first edge were written.
    let (g, appended, log_id) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    g.insert(fact_at("seed")).await.unwrap();
    let before = snapshot(&g).await;
    let since = events(&appended).len();

    // Act
    let failed = g
        .ingest_episode(
            vec![fact_at("a"), fact_at("b")],
            vec![mentions(0, 1), mentions(0, 1)],
        )
        .await;

    // Assert
    assert!(matches!(failed, Err(Error::RelateRefused(_))), "{failed:?}");
    assert_eq!(snapshot(&g).await, before, "no node, edge, or index entry");
    let logged = events_since(&appended, since);
    assert_eq!(logged.len(), 2, "the batch, then its void: {logged:?}");
    assert!(matches!(logged[0].payload, LogPayload::EpisodeBatch(_)));
    assert_eq!(
        logged[1].payload,
        LogPayload::Voided {
            target_event_id: logged[0].event_id.clone()
        }
    );
    assert_cursor_at_last_event(&g, &appended, &log_id).await;

    // Act: a valid episode afterwards is a first write.
    let retried = g
        .ingest_episode(vec![fact_at("a"), fact_at("b")], vec![mentions(0, 1)])
        .await;

    // Assert
    assert!(retried.is_ok(), "{retried:?}");
    assert_eq!(count(&g, "nodes").await, 3);
}
