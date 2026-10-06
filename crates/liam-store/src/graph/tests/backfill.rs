// SPDX-License-Identifier: Apache-2.0
//! Backfilling a store that predates its log: every row it holds is written
//! into the log once, a run that stops partway resumes where it left off, and a
//! rebuild from the backfilled log recreates the live rows. Tests run against a
//! real WAL and reader in a temp directory. A store with no log is written
//! through the public API, then reopened on a log.

use liam_log::event::{EdgeRow, LogEvent, LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
use liam_log::hash::{edge_row_hash, node_row_hash};

use super::gc_log::{controlled_log, Control};
use super::log_write::within_deadline;
use super::support::{
    count, cursor_offset, event, fact, fact_at, offset_pair, Env, EDGE_COLUMNS, NODE_COLUMNS,
};
use super::*;
use crate::DefaultBackend;

const DIMS: usize = 8;

/// Rows `seed_standard` writes: six nodes and three edges.
const STANDARD_NODES: usize = 6;
const STANDARD_ROWS: usize = 9;

/// The batch the interruption tests run with, small enough that the six
/// standard nodes span three batches.
const BATCH: usize = 2;

fn clock_at(at: i64) -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(at)))
}

/// A store with no log, as a database that predates the log is opened.
async fn pre_log(env: &Env, db: &str, clock: &Arc<FixedClock>) -> DefaultGraph {
    DefaultGraph::open_with_clock(&env.db(db), GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap()
}

/// The same database opened on a log, with the clock at 7000 and a log writer a
/// test can make fail or hold. Whatever opened the log before must be gone.
struct Logged {
    graph: DefaultGraph,
    log: SharedLog,
    control: Arc<Control>,
    clock: Arc<FixedClock>,
}

async fn open_logged(env: &Env, db: &str) -> Logged {
    let clock = clock_at(7000);
    let (log, control) = controlled_log(env);
    let (graph, log) = env
        .open_over::<DefaultBackend>(log, db, Arc::clone(&clock))
        .await;
    Logged {
        graph,
        log,
        control,
        clock,
    }
}

// ---- stores to backfill ----

/// Six nodes, the last written without a valid time, and three edges.
async fn seed_standard(g: &DefaultGraph, clock: &FixedClock) {
    let mut ids = Vec::new();
    for i in 0..STANDARD_NODES {
        clock.set(Millis(1000 + 100 * i as i64));
        let content = format!("node-{i}");
        let node = if i == STANDARD_NODES - 1 {
            fact(&content)
        } else {
            fact_at(&content)
        };
        ids.push(g.insert(node).await.unwrap());
    }
    clock.set(Millis(2000));
    for (src, dst, kind) in [(0, 1, "mentions"), (1, 2, "mentions"), (2, 3, "cites")] {
        g.relate(&ids[src], &ids[dst], kind).await.unwrap();
    }
}

async fn seed_nodes(g: &DefaultGraph, clock: &FixedClock, nodes: usize) {
    for i in 0..nodes {
        clock.set(Millis(1000 + 100 * i as i64));
        g.insert(fact_at(&format!("node-{i}"))).await.unwrap();
    }
}

/// Superseded nodes that live edges point at or leave from, and two versions of
/// one subject replaced in turn: seven nodes of which three are closed, and six
/// edges of which three are `supersedes`.
async fn seed_chain(g: &DefaultGraph, clock: &FixedClock) {
    clock.set(Millis(1000));
    let alpha = g.insert(fact_at("alpha")).await.unwrap();
    clock.set(Millis(1100));
    let beta = g.insert(fact_at("beta")).await.unwrap();
    clock.set(Millis(1200));
    let gamma = g.insert(fact_at("gamma")).await.unwrap();
    clock.set(Millis(1300));
    g.relate(&alpha, &gamma, "mentions").await.unwrap();
    g.relate(&gamma, &beta, "cites").await.unwrap();
    clock.set(Millis(1400));
    g.supersede(&gamma, fact_at("gamma2")).await.unwrap();
    clock.set(Millis(1500));
    g.upsert_by(fact_at("v1").with_subject("s")).await.unwrap();
    clock.set(Millis(1600));
    g.upsert_by(fact_at("v2").with_subject("s")).await.unwrap();
    clock.set(Millis(1700));
    let v3 = g.upsert_by(fact_at("v3").with_subject("s")).await.unwrap();
    clock.set(Millis(1800));
    g.relate(&beta, &v3, "mentions").await.unwrap();
}

/// Every node and edge row, closed ones included.
async fn all_rows(g: &DefaultGraph) -> usize {
    (count(g, "nodes").await + count(g, "edges").await) as usize
}

// ---- reading the store and the log back ----

/// What the live rows hold, read with plain SQL, and their hash multiset.
struct Live {
    dump: (String, String),
    multiset: Vec<[u8; 32]>,
}

async fn live_of(g: &DefaultGraph) -> Live {
    let live = format!("tx_to = {}", FOREVER.0);
    let nodes = format!("SELECT {NODE_COLUMNS} FROM nodes WHERE {live} ORDER BY id");
    let edges = format!("SELECT {EDGE_COLUMNS} FROM edges WHERE {live} ORDER BY id");
    Live {
        dump: (
            format!("{:?}", g.backend.query(&nodes, &[]).await.unwrap()),
            format!("{:?}", g.backend.query(&edges, &[]).await.unwrap()),
        ),
        multiset: g.projection_hash_multiset().await.unwrap(),
    }
}

/// Rebuilds a new store from the log in `env` and checks its live rows are
/// exactly `expected`. Every store that wrote to the log must be gone.
async fn assert_rebuild_equals(env: &Env, expected: &Live) {
    let (fresh, log) = env
        .open::<DefaultBackend>("rebuilt.db", clock_at(1000))
        .await;
    let (rebuilt, _) = fresh
        .rebuild_from_log(log, RebuildMode::RequireEmpty)
        .await
        .expect("rebuild from the backfilled log");
    let found = live_of(&rebuilt).await;
    assert_eq!(found.dump, expected.dump, "live rows differ after rebuild");
    assert_eq!(found.multiset, expected.multiset);
}

#[derive(Debug, PartialEq)]
struct State {
    phase: String,
    last_id: Option<String>,
    complete: bool,
}

async fn state(g: &DefaultGraph) -> Option<State> {
    let rows = g
        .backend
        .query(
            "SELECT phase, last_id, completed_at FROM backfill_state",
            &[],
        )
        .await
        .unwrap();
    let row = rows.first()?;
    Some(State {
        phase: row.get_string(0).unwrap(),
        last_id: opt_string(row, 1).unwrap(),
        complete: !matches!(row.0[2], Value::Null),
    })
}

/// The state row exactly as stored, so a test can tell an untouched row from
/// one rewritten with the same meaning.
async fn raw_state(g: &DefaultGraph) -> String {
    let rows = g
        .backend
        .query("SELECT * FROM backfill_state", &[])
        .await
        .unwrap();
    format!("{rows:?}")
}

async fn node_ids(g: &DefaultGraph) -> Vec<String> {
    g.backend
        .query("SELECT id FROM nodes ORDER BY id", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect()
}

async fn stored_nodes(g: &DefaultGraph) -> Vec<NodeRow> {
    let sql = format!("SELECT {NODE_COLUMNS} FROM nodes ORDER BY id");
    let rows = g.backend.query(&sql, &[]).await.unwrap();
    rows.iter()
        .map(|row| NodeRow {
            id: row.get_string(0).unwrap(),
            kind: row.get_string(1).unwrap(),
            label: row.get_string(2).unwrap(),
            content: row.get_string(3).unwrap(),
            producer: row.get_string(4).unwrap(),
            attributes: row.get_string(5).unwrap(),
            scope: opt_string(row, 6).unwrap(),
            subject: opt_string(row, 7).unwrap(),
            confidence: row_f64(row, 8),
            valid_from: row.get_i64(9).unwrap(),
            valid_from_supplied: true,
            valid_until: row.get_i64(10).unwrap(),
            tx_from: row.get_i64(11).unwrap(),
            tx_to: row.get_i64(12).unwrap(),
        })
        .collect()
}

async fn stored_edges(g: &DefaultGraph) -> Vec<EdgeRow> {
    let sql = format!("SELECT {EDGE_COLUMNS} FROM edges ORDER BY id");
    let rows = g.backend.query(&sql, &[]).await.unwrap();
    rows.iter()
        .map(|row| EdgeRow {
            id: row.get_string(0).unwrap(),
            src: row.get_string(1).unwrap(),
            dst: row.get_string(2).unwrap(),
            edge_type: row.get_string(3).unwrap(),
            attributes: row.get_string(4).unwrap(),
            tx_from: row.get_i64(5).unwrap(),
            tx_to: row.get_i64(6).unwrap(),
        })
        .collect()
}

async fn logged_events(env: &Env) -> Vec<LogEvent> {
    env.records()
        .await
        .into_iter()
        .map(|record| record.event)
        .collect()
}

fn node_writes(events: &[LogEvent]) -> Vec<&NodeRow> {
    events
        .iter()
        .filter_map(|event| match &event.payload {
            LogPayload::NodeWrite(row) => Some(row),
            _ => None,
        })
        .collect()
}

fn edge_writes(events: &[LogEvent]) -> Vec<&EdgeRow> {
    events
        .iter()
        .filter_map(|event| match &event.payload {
            LogPayload::EdgeWrite(row) => Some(row),
            _ => None,
        })
        .collect()
}

/// The log holds one event for each of `rows` rows and nothing else, and no row
/// is in it twice.
fn assert_each_row_logged_once(events: &[LogEvent], rows: usize) {
    let mut ids: Vec<&str> = node_writes(events)
        .iter()
        .map(|row| row.id.as_str())
        .chain(edge_writes(events).iter().map(|row| row.id.as_str()))
        .collect();
    ids.sort_unstable();
    let written = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), written, "a row is in the log twice");
    assert_eq!(written, rows, "rows logged");
    assert_eq!(events.len(), rows, "the log holds events besides the rows");
}

async fn head_offset(env: &Env) -> Option<(i64, i64)> {
    env.records()
        .await
        .last()
        .map(|record| offset_pair(record.offset))
}

fn is_log_append(result: &Result<BackfillReport>) -> bool {
    matches!(result, Err(Error::LogAppend(_)))
}

// ---- a populated store ----

#[tokio::test]
async fn backfill_logs_every_live_row_once_and_marks_the_store_complete() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    seed_standard(&pre_log(&env, "live.db", &clock).await, &clock).await;
    let logged = open_logged(&env, "live.db").await;
    assert!(logged.graph.needs_backfill().await.unwrap());

    // Act
    let report = logged.graph.backfill_log_from_projection().await.unwrap();

    // Assert: one event per row, by kind, and the state says every row is in.
    let events = logged_events(&env).await;
    assert_eq!(
        report,
        BackfillReport {
            nodes: STANDARD_NODES,
            edges: 3,
            resumed: false
        }
    );
    assert_each_row_logged_once(&events, STANDARD_ROWS);
    assert_eq!(node_writes(&events).len(), STANDARD_NODES);
    assert_eq!(edge_writes(&events).len(), 3);
    assert!(state(&logged.graph).await.expect("a state row").complete);
    assert!(!logged.graph.needs_backfill().await.unwrap());
}

#[tokio::test]
async fn backfill_then_rebuild_matches_the_state_before_the_backfill() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    let pre = pre_log(&env, "live.db", &clock).await;
    seed_standard(&pre, &clock).await;
    let before = live_of(&pre).await;
    drop(pre);
    let logged = open_logged(&env, "live.db").await;

    // Act
    logged.graph.backfill_log_from_projection().await.unwrap();
    drop(logged);

    // Assert
    assert_each_row_logged_once(&logged_events(&env).await, STANDARD_ROWS);
    assert_rebuild_equals(&env, &before).await;
}

#[tokio::test]
async fn backfill_on_a_completed_store_appends_nothing_and_leaves_the_state_alone() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    seed_standard(&pre_log(&env, "live.db", &clock).await, &clock).await;
    let logged = open_logged(&env, "live.db").await;
    logged.graph.backfill_log_from_projection().await.unwrap();
    let (state_before, cursor_before) = (
        raw_state(&logged.graph).await,
        cursor_offset(&logged.graph).await,
    );
    logged.clock.set(Millis(9000));

    // Act
    let again = logged.graph.backfill_log_from_projection().await.unwrap();

    // Assert
    assert_eq!(again, BackfillReport::default());
    assert_each_row_logged_once(&logged_events(&env).await, STANDARD_ROWS);
    assert_eq!(raw_state(&logged.graph).await, state_before);
    assert_eq!(cursor_offset(&logged.graph).await, cursor_before);
}

#[tokio::test]
async fn backfill_of_an_empty_store_completes_at_once_with_no_events() {
    // Arrange
    let env = Env::new();
    let logged = open_logged(&env, "live.db").await;
    assert!(logged.graph.needs_backfill().await.unwrap());

    // Act
    let report = logged.graph.backfill_log_from_projection().await.unwrap();

    // Assert
    assert_eq!(report, BackfillReport::default());
    assert!(logged_events(&env).await.is_empty());
    assert!(state(&logged.graph).await.expect("a state row").complete);
    assert!(!logged.graph.needs_backfill().await.unwrap());
}

#[tokio::test]
async fn backfill_covers_a_store_one_row_short_of_a_batch_exactly_and_one_over() {
    // Arrange: nodes only, so the batch boundary is the only variable.
    for nodes in [0, 1, BATCH, BATCH + 1] {
        let env = Env::new();
        let clock = clock_at(1000);
        let pre = pre_log(&env, "live.db", &clock).await;
        seed_nodes(&pre, &clock, nodes).await;
        let before = live_of(&pre).await;
        drop(pre);
        let logged = open_logged(&env, "live.db").await;

        // Act
        let report = logged.graph.backfill_in_batches(BATCH).await.unwrap();

        // Assert
        assert_eq!(report.nodes, nodes, "{nodes} nodes");
        assert_eq!(report.edges, 0, "{nodes} nodes");
        assert_each_row_logged_once(&logged_events(&env).await, nodes);
        assert!(
            state(&logged.graph).await.expect("a state row").complete,
            "{nodes} nodes"
        );
        drop(logged);
        assert_rebuild_equals(&env, &before).await;
    }
}

// ---- what is logged ----

#[tokio::test]
async fn backfill_stamps_each_event_from_the_stored_row() {
    // Arrange: nodes with a supplied valid time, with none, and with another
    // confidence, so every field of the envelope has a distinct source.
    let env = Env::new();
    let clock = clock_at(1000);
    let pre = pre_log(&env, "live.db", &clock).await;
    let a = pre.insert(fact_at("a")).await.unwrap();
    clock.set(Millis(1100));
    let b = pre.insert(fact("b")).await.unwrap();
    clock.set(Millis(1200));
    pre.insert(fact_at("c").with_confidence(0.4)).await.unwrap();
    clock.set(Millis(1300));
    pre.relate(&a, &b, "mentions").await.unwrap();
    let nodes = stored_nodes(&pre).await;
    let edges = stored_edges(&pre).await;
    drop(pre);
    let logged = open_logged(&env, "live.db").await;

    // Act
    logged.graph.backfill_log_from_projection().await.unwrap();

    // Assert
    let events = logged_events(&env).await;
    assert_each_row_logged_once(&events, nodes.len() + edges.len());
    let mut event_ids: Vec<&str> = events.iter().map(|e| e.event_id.as_str()).collect();
    event_ids.sort_unstable();
    event_ids.dedup();
    assert_eq!(event_ids.len(), events.len(), "event ids repeat");
    for event in &events {
        assert_eq!(event.source, "backfill");
        assert_eq!(event.ingested_at, 7000, "ingested when it was backfilled");
        assert_eq!(event.encryption_key_id, None);
        assert_eq!(event.schema_version, CURRENT_SCHEMA_VERSION);
    }
    for event in &events {
        match &event.payload {
            LogPayload::NodeWrite(row) => {
                let stored = nodes
                    .iter()
                    .find(|n| n.id == row.id)
                    .expect("a stored node");
                assert_eq!(row, stored, "the row is logged as stored");
                assert_eq!(event.content_hash, node_row_hash(stored));
                assert_eq!(event.trust_score, stored.confidence);
                assert_eq!(event.observed_at, stored.valid_from);
            }
            LogPayload::EdgeWrite(row) => {
                let stored = edges
                    .iter()
                    .find(|e| e.id == row.id)
                    .expect("a stored edge");
                assert_eq!(row, stored, "the row is logged as stored");
                assert_eq!(event.content_hash, edge_row_hash(stored));
                assert_eq!(event.trust_score, 1.0);
                assert_eq!(event.observed_at, stored.tx_from);
            }
            other => panic!("backfill logged a record that is not a row: {other:?}"),
        }
    }
}

#[tokio::test]
async fn backfill_leaves_the_cursor_on_the_log_head_and_a_reopen_accepts_it() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    seed_standard(&pre_log(&env, "live.db", &clock).await, &clock).await;
    let logged = open_logged(&env, "live.db").await;

    // Act
    logged.graph.backfill_log_from_projection().await.unwrap();

    // Assert
    let head = head_offset(&env).await;
    assert!(head.is_some(), "the backfill logged nothing");
    assert_eq!(cursor_offset(&logged.graph).await, head);
    drop(logged);
    let reopened = open_logged(&env, "live.db").await;
    assert_eq!(cursor_offset(&reopened.graph).await, head);
    assert!(!reopened.graph.needs_backfill().await.unwrap());
}

#[tokio::test]
async fn backfill_indexes_every_row_so_a_later_identical_insert_is_a_duplicate() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    let pre = pre_log(&env, "live.db", &clock).await;
    seed_standard(&pre, &clock).await;
    let first = node_ids(&pre).await;
    let nodes = stored_nodes(&pre).await;
    let original = nodes
        .iter()
        .find(|node| node.content == "node-0")
        .expect("node-0");
    drop(pre);
    let logged = open_logged(&env, "live.db").await;
    logged.graph.backfill_log_from_projection().await.unwrap();
    let backfilled = logged_events(&env).await;

    // Assert: the index and the filter know every hash the log now holds.
    assert_eq!(
        count(&logged.graph, "log_hash_index").await as usize,
        STANDARD_ROWS
    );
    for event in &backfilled {
        assert!(
            logged
                .log
                .lock()
                .await
                .bloom_might_contain(&event.content_hash),
            "the dedup filter does not know {}",
            event.event_id
        );
    }

    // Act: the content of a backfilled node, written again.
    let again = logged.graph.insert(fact_at("node-0")).await.unwrap();

    // Assert: the first carrier answers, and the log says so.
    assert_eq!(again.as_str(), original.id);
    assert_eq!(count(&logged.graph, "nodes").await as usize, first.len());
    let events = logged_events(&env).await;
    assert_eq!(events.len(), STANDARD_ROWS + 1);
    let first_event = backfilled
        .iter()
        .find(|event| matches!(&event.payload, LogPayload::NodeWrite(row) if row.id == original.id))
        .expect("the backfill event of node-0");
    match &events.last().expect("a record").payload {
        LogPayload::DuplicateOf { first_event_id } => {
            assert_eq!(*first_event_id, first_event.event_id)
        }
        other => panic!("expected a DuplicateOf, got {other:?}"),
    }
}

#[tokio::test]
async fn backfill_logs_rows_that_share_a_content_hash_once_each_even_after_an_interruption() {
    // Arrange: a store from before dedup holds the same content three times.
    let env = Env::new();
    let clock = clock_at(1000);
    let pre = pre_log(&env, "live.db", &clock).await;
    for _ in 0..3 {
        pre.insert(fact_at("same")).await.unwrap();
    }
    let before = live_of(&pre).await;
    drop(pre);
    let logged = open_logged(&env, "live.db").await;
    logged.control.fail_nth_from_now(3);

    // Act: the third append fails with two rows logged, then the run resumes.
    let failed = logged.graph.backfill_log_from_projection().await;
    assert!(is_log_append(&failed), "{failed:?}");
    assert_eq!(logged_events(&env).await.len(), 2);
    drop(logged);
    let resumed = open_logged(&env, "live.db").await;
    let report = resumed.graph.backfill_log_from_projection().await.unwrap();

    // Assert
    assert_eq!(report.nodes, 1);
    assert_each_row_logged_once(&logged_events(&env).await, 3);
    drop(resumed);
    assert_rebuild_equals(&env, &before).await;
}

// ---- superseded rows ----

#[tokio::test]
async fn backfill_of_a_supersede_chain_rebuilds_the_live_rows_and_closes_the_rest() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    let pre = pre_log(&env, "live.db", &clock).await;
    seed_chain(&pre, &clock).await;
    let before = live_of(&pre).await;
    let all_before = (stored_nodes(&pre).await, stored_edges(&pre).await);
    let rows = all_rows(&pre).await;
    drop(pre);
    let logged = open_logged(&env, "live.db").await;

    // Act
    let report = logged.graph.backfill_log_from_projection().await.unwrap();
    drop(logged);

    // Assert: a superseded node a live edge or `supersedes` edge names is
    // logged too, so replay finds it to close.
    assert_eq!(report.nodes + report.edges, rows);
    assert_each_row_logged_once(&logged_events(&env).await, rows);
    assert_rebuild_equals(&env, &before).await;
    let (fresh, _log) = env
        .open::<DefaultBackend>("rebuilt.db", clock_at(1000))
        .await;
    assert_eq!(count(&fresh, "log_quarantine").await, 0);
    assert_eq!(
        (stored_nodes(&fresh).await, stored_edges(&fresh).await),
        all_before,
        "the superseded nodes are closed again at the same time"
    );
}

#[tokio::test]
async fn backfill_resumes_to_the_same_log_wherever_a_chain_store_is_interrupted() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    let pre = pre_log(&env, "probe.db", &clock).await;
    seed_chain(&pre, &clock).await;
    let rows = all_rows(&pre).await;
    drop(pre);

    for appended in 0..rows {
        let env = Env::new();
        let clock = clock_at(1000);
        let pre = pre_log(&env, "live.db", &clock).await;
        seed_chain(&pre, &clock).await;
        let before = live_of(&pre).await;
        drop(pre);
        let logged = open_logged(&env, "live.db").await;
        logged.control.fail_nth_from_now(appended as u64 + 1);

        // Act: stop after `appended` events, then run again on a new log writer.
        let failed = logged.graph.backfill_in_batches(BATCH).await;
        assert!(is_log_append(&failed), "{appended}: {failed:?}");
        assert_eq!(logged_events(&env).await.len(), appended);
        drop(logged);
        let resumed = open_logged(&env, "live.db").await;
        let report = resumed.graph.backfill_in_batches(BATCH).await.unwrap();

        // Assert
        assert_eq!(report.nodes + report.edges, rows - appended, "{appended}");
        assert_each_row_logged_once(&logged_events(&env).await, rows);
        assert!(state(&resumed.graph).await.expect("state").complete);
        drop(resumed);
        assert_rebuild_equals(&env, &before).await;
    }
}

// ---- interruption and resume ----

#[tokio::test]
async fn backfill_state_trails_the_log_by_less_than_a_batch_and_a_resume_appends_only_the_rest() {
    // Each case is how many rows were logged before the append failed, and the
    // index, among the nodes by id, of the last row the state records.
    let cases = [
        (1, None),
        (2, Some(1)),
        (3, Some(1)),
        (4, Some(3)),
        (5, Some(3)),
    ];
    for (appended, recorded) in cases {
        // Arrange
        let env = Env::new();
        let clock = clock_at(1000);
        let pre = pre_log(&env, "live.db", &clock).await;
        seed_standard(&pre, &clock).await;
        let before = live_of(&pre).await;
        let ids = node_ids(&pre).await;
        drop(pre);
        let logged = open_logged(&env, "live.db").await;
        logged.control.fail_nth_from_now(appended as u64 + 1);

        // Act
        let failed = logged.graph.backfill_in_batches(BATCH).await;

        // Assert: the run stopped, and the state is at the last full batch.
        assert!(is_log_append(&failed), "{appended}: {failed:?}");
        assert_eq!(logged_events(&env).await.len(), appended, "{appended}");
        let expected = recorded.map(|index| State {
            phase: "nodes".into(),
            last_id: Some(ids[index].clone()),
            complete: false,
        });
        assert_eq!(state(&logged.graph).await, expected, "{appended}");
        assert!(logged.graph.needs_backfill().await.unwrap(), "{appended}");
        drop(logged);

        // Act: a reopened store carries on.
        let resumed = open_logged(&env, "live.db").await;
        let report = resumed.graph.backfill_in_batches(BATCH).await.unwrap();

        // Assert: only the rows still missing are appended.
        assert_eq!(report.resumed, recorded.is_some(), "{appended}");
        assert_eq!(report.nodes, STANDARD_NODES - appended, "{appended}");
        assert_eq!(report.edges, 3, "{appended}");
        assert_each_row_logged_once(&logged_events(&env).await, STANDARD_ROWS);
        assert!(state(&resumed.graph).await.expect("state").complete);
        assert!(!resumed.graph.needs_backfill().await.unwrap());
        drop(resumed);
        assert_rebuild_equals(&env, &before).await;
    }
}

#[tokio::test]
async fn backfill_does_not_log_a_row_twice_that_the_log_holds_ahead_of_the_state() {
    // Each case is how many rows the interrupted run logged, and how many nodes
    // after them were appended by a run that died before it recorded them.
    for (appended, ahead) in [(0, 1), (3, 2), (5, 1)] {
        // Arrange
        let env = Env::new();
        let clock = clock_at(1000);
        let pre = pre_log(&env, "live.db", &clock).await;
        seed_standard(&pre, &clock).await;
        let before = live_of(&pre).await;
        drop(pre);
        if appended > 0 {
            let logged = open_logged(&env, "live.db").await;
            logged.control.fail_nth_from_now(appended as u64 + 1);
            let failed = logged.graph.backfill_in_batches(BATCH).await;
            assert!(is_log_append(&failed), "{appended}: {failed:?}");
        }
        let nodes = stored_nodes(&pre_log(&env, "live.db", &clock).await).await;
        let orphaned: Vec<LogEvent> = nodes[appended..appended + ahead]
            .iter()
            .map(|row| {
                let mut event = event(
                    &format!("ahead-{}", row.id),
                    node_row_hash(row),
                    LogPayload::NodeWrite(row.clone()),
                );
                event.source = "backfill".into();
                event
            })
            .collect();
        env.append(&orphaned);

        // Act
        let resumed = open_logged(&env, "live.db").await;
        let report = resumed.graph.backfill_in_batches(BATCH).await.unwrap();

        // Assert
        let case = format!("{appended} logged, {ahead} ahead");
        assert_eq!(
            report.nodes + report.edges,
            STANDARD_ROWS - appended - ahead,
            "{case}"
        );
        assert_each_row_logged_once(&logged_events(&env).await, STANDARD_ROWS);
        assert_eq!(
            count(&resumed.graph, "log_hash_index").await as usize,
            STANDARD_ROWS,
            "{case}: every row is indexed"
        );
        assert_eq!(cursor_offset(&resumed.graph).await, head_offset(&env).await);
        assert!(state(&resumed.graph).await.expect("state").complete);
        drop(resumed);
        assert_rebuild_equals(&env, &before).await;
    }
}

// ---- when to backfill ----

#[tokio::test]
async fn backfill_is_not_needed_for_a_store_with_a_log_and_a_completed_state() {
    // Arrange: a store that was on its log from the start, with the marker set.
    let env = Env::new();
    let logged = open_logged(&env, "live.db").await;
    logged
        .graph
        .insert(fact_at("written with a log"))
        .await
        .unwrap();
    logged
        .graph
        .backend
        .execute(
            "INSERT INTO backfill_state (id, phase, last_id, completed_at)
             VALUES (1, 'supersedes', NULL, 1000)",
            &[],
        )
        .await
        .unwrap();
    let events_before = logged_events(&env).await.len();
    let state_before = raw_state(&logged.graph).await;

    // Act
    let needed = logged.graph.needs_backfill().await.unwrap();
    let report = logged.graph.backfill_log_from_projection().await.unwrap();

    // Assert
    assert!(!needed);
    assert_eq!(report, BackfillReport::default());
    assert_eq!(logged_events(&env).await.len(), events_before);
    assert_eq!(raw_state(&logged.graph).await, state_before);
}

#[tokio::test]
async fn backfill_is_needed_for_a_populated_store_that_has_no_state() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    seed_standard(&pre_log(&env, "live.db", &clock).await, &clock).await;
    let logged = open_logged(&env, "live.db").await;

    // Act
    let needed = logged.graph.needs_backfill().await.unwrap();

    // Assert: asking changes nothing.
    assert!(needed);
    assert_eq!(state(&logged.graph).await, None);
    assert!(logged_events(&env).await.is_empty());
}

#[tokio::test]
async fn backfill_needs_a_log_to_write_to() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    let pre = pre_log(&env, "live.db", &clock).await;
    seed_standard(&pre, &clock).await;
    let before = live_of(&pre).await;

    // Act
    let needed = pre.needs_backfill().await;
    let backfilled = pre.backfill_log_from_projection().await;

    // Assert
    assert!(matches!(needed, Err(Error::NoEventLog)), "{needed:?}");
    assert!(
        matches!(backfilled, Err(Error::NoEventLog)),
        "{backfilled:?}"
    );
    assert_eq!(live_of(&pre).await.dump, before.dump);
    assert_eq!(state(&pre).await, None);
}

// ---- the log lock ----

#[tokio::test]
async fn backfill_refuses_a_poisoned_log_and_writes_nothing() {
    // Arrange
    let env = Env::new();
    let clock = clock_at(1000);
    seed_standard(&pre_log(&env, "live.db", &clock).await, &clock).await;
    let logged = open_logged(&env, "live.db").await;
    logged.log.lock().await.poison();

    // Act
    let refused = logged.graph.backfill_log_from_projection().await;

    // Assert
    assert!(matches!(refused, Err(Error::LogPoisoned)), "{refused:?}");
    assert!(logged_events(&env).await.is_empty());
    assert_eq!(state(&logged.graph).await, None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backfill_holds_the_log_lock_for_the_whole_run_so_a_concurrent_insert_waits() {
    within_deadline(async {
        // Arrange: the first append is held, so the backfill is mid-run
        let env = Env::new();
        let clock = clock_at(1000);
        seed_standard(&pre_log(&env, "live.db", &clock).await, &clock).await;
        let logged = open_logged(&env, "live.db").await;
        let (log, control) = (logged.log, logged.control);
        let graph = Arc::new(logged.graph);
        let (reached, release) = control.hold_next_append();
        let mut backfill = tokio::spawn({
            let graph = Arc::clone(&graph);
            async move { graph.backfill_in_batches(BATCH).await }
        });
        tokio::select! {
            () = reached.notified() => {}
            finished = &mut backfill => panic!("the backfill finished without appending: {finished:?}"),
        }

        // Act: a write arrives meanwhile
        let late = tokio::spawn({
            let graph = Arc::clone(&graph);
            async move { graph.insert(fact_at("late arrival")).await }
        });
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }

        // Assert: it waits on the lock the backfill holds
        assert!(log.try_lock().is_err(), "the backfill must hold the log lock");
        assert!(!late.is_finished(), "the insert must wait for the backfill");

        // Act: the backfill goes on
        release.send(()).unwrap();
        let report = backfill.await.unwrap().unwrap();
        late.await.unwrap().unwrap();

        // Assert: the late write is logged after every backfilled row
        assert_eq!(report.nodes + report.edges, STANDARD_ROWS);
        let events = logged_events(&env).await;
        let (last, backfilled) = events.split_last().expect("events");
        assert_each_row_logged_once(backfilled, STANDARD_ROWS);
        assert!(backfilled.iter().all(|event| event.source == "backfill"));
        assert!(
            matches!(&last.payload, LogPayload::NodeWrite(row) if row.content == "late arrival"),
            "{last:?}"
        );
    })
    .await;
}
