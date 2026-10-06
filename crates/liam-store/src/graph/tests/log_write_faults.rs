// SPDX-License-Identifier: Apache-2.0
//! Log-first write path under faults and concurrency: a failure at every
//! transaction statement of each write path is voided and leaves the projection
//! untouched, and concurrent writes keep the log in commit order.

use std::sync::atomic::Ordering;

use liam_log::event::{LogEvent, LogPayload};
use liam_log::hash::content_hashes;
use liam_log::LogWriter;

use super::log_write::{
    assert_cursor_at_last_event, events, events_since, logged_graph, mentions, open_with, race,
    within_deadline, Appended, RecordingLog,
};
use super::support::{count, fact_at, share};
use super::*;
use crate::DefaultBackend;

// ---- projection and cursor faults, per write path ----

/// A snapshot of what the projection and the hash index hold.
#[derive(Debug, PartialEq)]
struct Snapshot {
    nodes: Vec<(String, i64)>,
    edges: Vec<(String, i64)>,
    index: Vec<(String, String, String)>,
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
            "SELECT hex(content_hash), first_event_id, row_ids FROM log_hash_index
             ORDER BY content_hash",
            &[],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| {
            let column = |i| row.get_string(i).unwrap();
            (column(0), column(1), column(2))
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
    Insert,
    UpsertInsert,
    UpsertSupersede,
    Supersede,
    Relate,
    Episode,
}

const OPS: [Op; 6] = [
    Op::Insert,
    Op::UpsertInsert,
    Op::UpsertSupersede,
    Op::Supersede,
    Op::Relate,
    Op::Episode,
];

impl Op {
    /// How many statements the path's transaction runs: its rows, one hash
    /// index upsert per distinct row hash it carries, and the cursor last.
    fn statements(self) -> usize {
        match self {
            Op::Insert | Op::UpsertInsert | Op::Relate => 3,
            Op::UpsertSupersede | Op::Supersede => 6,
            Op::Episode => 12,
        }
    }
}

/// A live node `x` (subject "subject-{tag}") and a live peer `y`, written before
/// a fault is armed. Writes with different tags touch different rows.
struct Seeded {
    tag: usize,
    x: NodeId,
    y: NodeId,
}

async fn seed<B: Backend>(g: &Graph<B>, tag: usize) -> Seeded {
    Seeded {
        tag,
        x: g.insert(fact_at(&format!("v1-{tag}")).with_subject(format!("subject-{tag}")))
            .await
            .unwrap(),
        y: g.insert(fact_at(&format!("peer-{tag}"))).await.unwrap(),
    }
}

async fn run_op<B: Backend>(g: &Graph<B>, op: Op, seeded: &Seeded) -> Result<()> {
    let tag = seeded.tag;
    let subject = format!("subject-{tag}");
    match op {
        Op::Insert => g.insert(fact_at(&format!("new-{tag}"))).await.map(drop),
        Op::UpsertInsert => g
            .upsert_by(fact_at(&format!("new-{tag}")).with_subject(format!("other-{tag}")))
            .await
            .map(drop),
        Op::UpsertSupersede => g
            .upsert_by(fact_at(&format!("v2-{tag}")).with_subject(subject))
            .await
            .map(drop),
        Op::Supersede => g
            .supersede(
                &seeded.x,
                fact_at(&format!("v2-{tag}")).with_subject(subject),
            )
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
            let nodes = vec![
                fact_at(&format!("v2-{tag}")).with_subject(subject),
                fact_at(&format!("ep-peer-{tag}")),
            ];
            g.ingest_episode(nodes, vec![mentions(0, 1), cites])
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
    let seeded = seed(&g, 0).await;
    let before = snapshot(&g).await;
    let seeded_events = events(&appended).len();
    g.backend.probe.executed.lock().unwrap().clear();
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

/// Finds how many statements `op` runs in its transaction: the first failing
/// index that no longer fires. Every index below it must have failed with the
/// injector's own error, so a fault that went unnoticed, or a write that failed
/// for another reason, cannot shorten the count. The count must be exactly
/// `Op::statements`, so a write that skips a statement fails here.
async fn statement_count(op: Op) -> usize {
    for index in 0..40 {
        let result = run_with_fault(op, Fault::Statement(index)).await.result;
        match result {
            Ok(()) => {
                assert_eq!(index, op.statements(), "{op:?}: statements it ran");
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
    let envelope = |event: &LogEvent| {
        (
            event.source.clone(),
            event.trust_score,
            event.observed_at,
            event.ingested_at,
        )
    };
    assert_eq!(envelope(&logged[1]), envelope(&logged[0]), "{context}");
    assert_cursor_at_last_event(&faulted.g, &faulted.appended, &faulted.log_id).await;
    let bloom = faulted.shared.lock().await;
    for (hash, row_id) in content_hashes(&logged[0]) {
        assert!(
            !bloom.bloom_might_contain(&hash),
            "{context}: voided row {row_id} must not stay in the filter"
        );
    }
}

/// A retry once the fault clears is a first write, not a duplicate: the log
/// holds the write, its void, and the same kind of write again, with the cursor
/// on the last of them.
async fn assert_retry_is_a_first_write(faulted: &Faulted, op: Op, context: &str) {
    faulted.g.backend.set_fail_on_execute(usize::MAX);
    faulted.g.backend.set_fail_commit(false);

    let retried = run_op(&faulted.g, op, &faulted.seeded).await;

    assert!(retried.is_ok(), "{context}: {retried:?}");
    let logged = events_since(&faulted.appended, faulted.seeded_events);
    assert_eq!(logged.len(), 3, "{context}: write, void, retry: {logged:?}");
    assert!(
        matches!(logged[1].payload, LogPayload::Voided { .. }),
        "{context}: {logged:?}"
    );
    assert_eq!(
        std::mem::discriminant(&logged[2].payload),
        std::mem::discriminant(&logged[0].payload),
        "{context}: {logged:?}"
    );
    assert_cursor_at_last_event(&faulted.g, &faulted.appended, &faulted.log_id).await;
    assert_ne!(snapshot(&faulted.g).await, faulted.before, "{context}");
}

async fn assert_failure_at_every_statement_is_voided(op: Op) {
    let statements = statement_count(op).await;
    for index in 0..statements {
        // Arrange and Act
        let faulted = run_with_fault(op, Fault::Statement(index)).await;

        // Assert: the cursor is the last statement, so failing it undoes the
        // projection with it.
        let context = format!("{op:?}, failing execute {index}");
        let executed = faulted.g.backend.probe.executed.lock().unwrap().clone();
        assert_eq!(executed.len(), index + 1, "{context}: {executed:?}");
        assert_eq!(
            executed[index].contains("log_cursor"),
            index == statements - 1,
            "{context}: {}",
            executed[index]
        );
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

/// The fault tests every write path gets, in a module named for it.
macro_rules! write_path_fault_tests {
    ($($module:ident => $op:expr),* $(,)?) => {$(
        mod $module {
            use super::*;

            #[tokio::test]
            async fn log_write_projection_failure_voids_the_write() {
                within_deadline(assert_failure_at_every_statement_is_voided($op)).await;
            }

            #[tokio::test]
            async fn log_write_commit_failure_is_voided() {
                within_deadline(assert_commit_failure_is_voided($op)).await;
            }
        }
    )*};
}

write_path_fault_tests! {
    insert => Op::Insert,
    upsert_by_insert => Op::UpsertInsert,
    upsert_by_supersede => Op::UpsertSupersede,
    supersede => Op::Supersede,
    relate => Op::Relate,
    ingest_episode => Op::Episode,
}

// ---- concurrency ----

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn log_write_mixed_concurrent_writes_keep_wal_order_equal_to_commit_order() {
    within_deadline(async {
        // Arrange: the first `begin` of each pair is slow, so a write that appended
        // before its `begin` and let go of the log lock would be overtaken by the
        // second, and the log order would no longer be the commit order. Each write
        // path runs against rows of its own.
        const PAIRS: usize = 12;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("graph.db");
        let (log, appended) = RecordingLog::new();
        let shared = share(log);
        let g = Arc::new(
            open_with::<FailingBackend>(path.to_str().unwrap(), Millis(1000), Arc::clone(&shared))
                .await,
        );
        let mut seeded = Vec::new();
        for tag in 0..PAIRS * 2 {
            seeded.push(seed(&*g, tag).await);
        }
        let seeded = Arc::new(seeded);
        let since = events(&appended).len();
        g.backend.probe.begins.store(0, Ordering::SeqCst);
        g.backend
            .probe
            .delay_even_begins
            .store(true, Ordering::SeqCst);
        *g.backend.probe.watched.lock().unwrap() = Some(shared);

        // Act
        for pair in 0..PAIRS {
            let written = race(2, |side| {
                let tag = pair * 2 + side;
                let op = OPS[(pair + side) % OPS.len()];
                let (g, seeded) = (Arc::clone(&g), Arc::clone(&seeded));
                async move { run_op(&g, op, &seeded[tag]).await }
            })
            .await;
            assert!(written.iter().all(Result::is_ok), "{written:?}");
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
    })
    .await;
}

// ---- ingest_episode atomicity ----

#[tokio::test]
async fn log_write_a_failing_episode_edge_voids_the_whole_batch_and_keeps_no_node() {
    within_deadline(async {
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
    })
    .await;
}
