// SPDX-License-Identifier: Apache-2.0
//! Rebuilding the projection from the log alone. Tests run against a real WAL
//! and reader in a temp directory. What a store should hold is judged against
//! an oracle that reads the rows back with plain SQL and hashes them with the
//! log crate, never against the store's own multiset method.

use liam_log::event::{EdgeRow, LogEvent, LogPayload, NodeRow, TombstoneTable};
use liam_log::hash::{edge_row_hash, node_row_hash};

use super::support::{
    count, cursor, edge, edge_write, event, fact, fact_at, has_node, node, node_write, offset_pair,
    quarantined, record_counts, tombstone, Env, StubEmbedder, ONE_SEGMENT, SEGMENT_PER_EVENT,
};
use super::*;
use crate::error::MismatchSource;
use crate::graph::logged_write::HeldLog;
use crate::schema::{Clear, DERIVED_TABLES};
use crate::types::{EpisodeEdge, EpisodeRef};
use crate::DefaultBackend;

const DIMS: usize = 8;

/// A store opened over the log, with the handles a test needs to steer it.
struct Opened {
    graph: DefaultGraph,
    log: SharedLog,
    clock: Arc<FixedClock>,
    embedder: Option<Arc<StubEmbedder>>,
}

async fn open(env: &Env, db: &str) -> Opened {
    open_with(env, db, Some(StubEmbedder::new(DIMS))).await
}

async fn open_without_embedder(env: &Env, db: &str) -> Opened {
    open_with(env, db, None).await
}

async fn open_with(env: &Env, db: &str, embedder: Option<Arc<StubEmbedder>>) -> Opened {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (mut graph, log) = env.open::<DefaultBackend>(db, Arc::clone(&clock)).await;
    if let Some(embedder) = &embedder {
        graph = graph.with_embedder(embedder.clone());
    }
    Opened {
        graph,
        log,
        clock,
        embedder,
    }
}

impl Opened {
    /// The texts the embedder was asked for, sorted.
    fn embedded(&self) -> Vec<String> {
        self.embedder.as_ref().expect("an embedder").calls()
    }
}

/// A store that is not attached to any log, to read what a refused rebuild
/// left in the database file.
async fn inspect(env: &Env, db: &str) -> DefaultGraph {
    DefaultGraph::open(&env.db(db), GraphConfig::new(DIMS))
        .await
        .unwrap()
}

/// Rebuilds a store over the log it is attached to.
async fn rebuild(opened: Opened, mode: RebuildMode) -> Result<(Opened, RebuildReport)> {
    let log = Arc::clone(&opened.log);
    let (graph, report) = opened.graph.rebuild_from_log(log, mode).await?;
    Ok((Opened { graph, ..opened }, report))
}

async fn rebuilt(opened: Opened, mode: RebuildMode) -> (Opened, RebuildReport) {
    rebuild(opened, mode).await.unwrap()
}

/// The error a rebuild fails with. The store is gone with it.
async fn failed(opened: Opened, mode: RebuildMode) -> Error {
    match rebuild(opened, mode).await {
        Err(error) => error,
        Ok((_, report)) => panic!("the rebuild went ahead: {report:?}"),
    }
}

/// The error a rebuild refuses with, after checking the database file is as it
/// was, which a fresh handle reads because the refused store is gone.
async fn refused(env: &Env, db: &str, opened: Opened, mode: RebuildMode) -> Error {
    let before = full_state(&opened.graph).await;
    let error = failed(opened, mode).await;
    assert_eq!(
        full_state(&inspect(env, db).await).await,
        before,
        "{error:?}"
    );
    error
}

/// What a mismatch compared and its counts: expected, found, missing, unexpected.
fn mismatch(error: &Error) -> (MismatchSource, [usize; 4]) {
    match error {
        Error::RebuildMismatch {
            against,
            expected,
            found,
            missing,
            unexpected,
        } => (*against, [*expected, *found, *missing, *unexpected]),
        other => panic!("not a mismatch: {other:?}"),
    }
}

/// The record that cancels `target`, as a replay-time quarantine appends it.
fn voided(target: &LogEvent) -> LogEvent {
    event(
        &format!("void-{}", target.event_id),
        target.content_hash,
        LogPayload::Voided {
            target_event_id: target.event_id.clone(),
        },
    )
}

// ---- a populated store ----

/// The rows a test names after `populate`.
struct Populated {
    lonely: NodeId,
    mentions: EdgeId,
}

/// Live nodes and edges after `populate`.
const LIVE_NODES: usize = 8;
const LIVE_EDGES: usize = 4;

/// Every logged write path: inserts, upsert_by over and without a live
/// competitor, supersede, relate, and an episode, with `v1` and `gamma`
/// superseded, `lonely` a live node with no edge, and `open ended` a node
/// written without a valid time.
async fn populate(opened: &Opened) -> Populated {
    let (g, clock) = (&opened.graph, &opened.clock);
    clock.set(Millis(1000));
    let alpha = g.insert(fact_at("alpha")).await.unwrap();
    clock.set(Millis(1100));
    let beta = g.insert(fact_at("beta")).await.unwrap();
    clock.set(Millis(1200));
    g.upsert_by(fact_at("v1").with_subject("s")).await.unwrap();
    clock.set(Millis(1300));
    g.upsert_by(fact_at("v2").with_subject("s")).await.unwrap();
    clock.set(Millis(1400));
    let gamma = g.insert(fact_at("gamma")).await.unwrap();
    clock.set(Millis(1500));
    g.supersede(&gamma, fact_at("gamma2")).await.unwrap();
    clock.set(Millis(1600));
    let mentions = g.relate(&alpha, &beta, "mentions").await.unwrap();
    clock.set(Millis(1700));
    let episode_edge = EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::New(1),
        kind: "mentions".to_string(),
        attributes: serde_json::json!({}),
    };
    g.ingest_episode(vec![fact_at("ep-1"), fact_at("ep-2")], vec![episode_edge])
        .await
        .unwrap();
    clock.set(Millis(1800));
    let lonely = g.insert(fact_at("lonely")).await.unwrap();
    clock.set(Millis(1900));
    g.insert(fact("open ended")).await.unwrap();
    Populated { lonely, mentions }
}

// ---- reading the store back ----

const NODE_COLUMNS: &str = "id, kind, label, content, producer, attributes, scope, subject, \
     confidence, valid_from, valid_until, tx_from, tx_to";
const EDGE_COLUMNS: &str = "id, src, dst, type, attributes, tx_from, tx_to";

/// Every row of `nodes` and `edges` in full, ordered by id, so two stores can
/// be compared exactly.
async fn snapshot<B: Backend>(g: &Graph<B>) -> (String, String) {
    let nodes = format!("SELECT {NODE_COLUMNS} FROM nodes ORDER BY id");
    let edges = format!("SELECT {EDGE_COLUMNS} FROM edges ORDER BY id");
    (
        format!("{:?}", g.backend.query(&nodes, &[]).await.unwrap()),
        format!("{:?}", g.backend.query(&edges, &[]).await.unwrap()),
    )
}

/// Every table a rebuild touches, in full, so two equal values mean a refused
/// rebuild changed nothing.
async fn full_state<B: Backend>(g: &Graph<B>) -> String {
    let tables = DERIVED_TABLES
        .iter()
        .map(|(table, _)| *table)
        .chain(["node_vectors", "log_cursor"]);
    let mut state = String::new();
    for table in tables {
        let rows = g
            .backend
            .query(&format!("SELECT * FROM {table} ORDER BY 1"), &[])
            .await
            .unwrap();
        state.push_str(&format!("{table}: {rows:?}\n"));
    }
    state
}

fn text(value: &Value) -> Option<String> {
    match value {
        Value::Text(text) => Some(text.clone()),
        _ => None,
    }
}

fn real(value: &Value) -> f64 {
    match value {
        Value::Real(real) => *real,
        Value::Int(int) => *int as f64,
        other => panic!("not a number: {other:?}"),
    }
}

/// Hashes the nodes selected by `filter` the way a rebuild must: from the row
/// as stored, which does not record whether `valid_from` was supplied.
async fn node_hashes<B: Backend>(g: &Graph<B>, filter: &str) -> Vec<[u8; 32]> {
    let rows = g
        .backend
        .query(
            &format!("SELECT {NODE_COLUMNS} FROM nodes WHERE {filter}"),
            &[],
        )
        .await
        .unwrap();
    rows.iter()
        .map(|row| {
            node_row_hash(&NodeRow {
                id: row.get_string(0).unwrap(),
                kind: row.get_string(1).unwrap(),
                label: row.get_string(2).unwrap(),
                content: row.get_string(3).unwrap(),
                producer: row.get_string(4).unwrap(),
                attributes: row.get_string(5).unwrap(),
                scope: text(&row.0[6]),
                subject: text(&row.0[7]),
                confidence: real(&row.0[8]),
                valid_from: row.get_i64(9).unwrap(),
                valid_from_supplied: true,
                valid_until: row.get_i64(10).unwrap(),
                tx_from: row.get_i64(11).unwrap(),
                tx_to: row.get_i64(12).unwrap(),
            })
        })
        .collect()
}

async fn edge_hashes<B: Backend>(g: &Graph<B>, filter: &str) -> Vec<[u8; 32]> {
    let rows = g
        .backend
        .query(
            &format!("SELECT {EDGE_COLUMNS} FROM edges WHERE {filter}"),
            &[],
        )
        .await
        .unwrap();
    rows.iter()
        .map(|row| {
            edge_row_hash(&EdgeRow {
                id: row.get_string(0).unwrap(),
                src: row.get_string(1).unwrap(),
                dst: row.get_string(2).unwrap(),
                edge_type: row.get_string(3).unwrap(),
                attributes: row.get_string(4).unwrap(),
                tx_from: row.get_i64(5).unwrap(),
                tx_to: row.get_i64(6).unwrap(),
            })
        })
        .collect()
}

/// The sorted multiset of hashes of every live node and edge, computed
/// independently of the store's own method.
async fn oracle_multiset<B: Backend>(g: &Graph<B>) -> Vec<[u8; 32]> {
    let live = format!("tx_to = {}", FOREVER.0);
    let mut hashes = node_hashes(g, &live).await;
    hashes.extend(edge_hashes(g, &live).await);
    hashes.sort();
    hashes
}

async fn node_hash<B: Backend>(g: &Graph<B>, id: &str) -> [u8; 32] {
    node_hashes(g, &format!("id = '{id}'")).await[0]
}

async fn has_edge<B: Backend>(g: &Graph<B>, id: &str) -> bool {
    let rows = g
        .backend
        .query("SELECT 1 FROM edges WHERE id = ?1", &[id.into()])
        .await
        .unwrap();
    !rows.is_empty()
}

/// `(id, content)` of every live node, sorted.
async fn live_contents<B: Backend>(g: &Graph<B>) -> Vec<(String, String)> {
    let mut rows: Vec<_> = g
        .backend
        .query(
            "SELECT id, content FROM nodes WHERE tx_to = ?1",
            &[FOREVER.into()],
        )
        .await
        .unwrap()
        .iter()
        .map(|row| (row.get_string(0).unwrap(), row.get_string(1).unwrap()))
        .collect();
    rows.sort();
    rows
}

async fn stored_vector<B: Backend>(g: &Graph<B>, id: &str) -> Vec<f32> {
    let rows = g
        .backend
        .query(
            "SELECT embedding FROM node_vectors WHERE node_id = ?1",
            &[id.into()],
        )
        .await
        .unwrap();
    rows[0]
        .get_blob(0)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// A live node written behind the log's back, as a store edited by hand holds.
/// Its fields match `fact_at`, so a node of the same content hashes alike.
async fn insert_unlogged<B: Backend>(g: &Graph<B>, id: &str, content: &str) {
    g.backend
        .execute(
            &format!(
                "INSERT INTO nodes ({NODE_COLUMNS}) VALUES
                 (?1, 'fact', 'label', ?2, 'agent-a', '{{}}', NULL, NULL, 0.75, 500, ?3, 1000, ?3)"
            ),
            &[id.into(), content.into(), FOREVER.into()],
        )
        .await
        .unwrap();
}

async fn delete_node<B: Backend>(g: &Graph<B>, id: &NodeId) {
    g.backend
        .execute("DELETE FROM nodes WHERE id = ?1", &[id.as_str().into()])
        .await
        .unwrap();
}

async fn cursor_offset<B: Backend>(g: &Graph<B>) -> Option<(i64, i64)> {
    cursor(g).await.and_then(|(_, offset)| offset)
}

/// A store that applied the first two records of a log that has since grown by
/// two more, which is what an interrupted rebuild leaves behind.
async fn behind_the_log(env: &Env) -> Opened {
    env.append(&[node_write("node-a", "alpha"), node_write("node-b", "beta")]);
    let first = open(env, "behind.db").await;
    first.graph.catch_up().await.unwrap();
    drop(first);
    env.append(&[
        edge_write("edge-ab", "node-a", "node-b"),
        node_write("node-c", "gamma"),
    ]);
    open(env, "behind.db").await
}

// ---- tests ----

#[tokio::test]
async fn rebuild_reproduces_nodes_and_edges_exactly_after_the_database_is_deleted() {
    // Arrange: a populated store, plus a write the log voided, then its database
    // is gone and a new one points at the same log directory
    let env = Env::new();
    let (before, multiset_before) = {
        let first = open(&env, "first.db").await;
        populate(&first).await;
        (
            snapshot(&first.graph).await,
            oracle_multiset(&first.graph).await,
        )
    };
    let ghost = node_write("node-ghost", "never lands");
    env.append(&[ghost.clone(), voided(&ghost)]);
    let fresh = open(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert: the same rows, and the same multiset of hashes
    assert_eq!(snapshot(&fresh.graph).await, before);
    assert_eq!(multiset_before.len(), LIVE_NODES + LIVE_EDGES);
    assert_eq!(
        fresh.graph.projection_hash_multiset().await.unwrap(),
        multiset_before
    );
    assert_eq!(
        (report.live_nodes, report.live_edges),
        (LIVE_NODES, LIVE_EDGES)
    );
    assert_eq!(report.replayed.skipped_voided, 1, "{report:?}");
    assert!(!has_node(&fresh.graph, "node-ghost").await);
    let records = env.records().await;
    let last = records.last().expect("a log").offset;
    assert_eq!(cursor_offset(&fresh.graph).await, Some(offset_pair(last)));
}

#[tokio::test]
async fn rebuild_over_the_live_store_after_a_check_yields_identical_rows() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    populate(&live).await;
    let before = snapshot(&live.graph).await;
    let multiset_before = oracle_multiset(&live.graph).await;
    let events = env.records().await.len();

    // Act
    let (live, report) = rebuilt(live, RebuildMode::ResetChecked).await;

    // Assert
    assert_eq!(snapshot(&live.graph).await, before);
    assert_eq!(
        live.graph.projection_hash_multiset().await.unwrap(),
        multiset_before
    );
    assert_eq!(report.replayed.applied, events, "{report:?}");
    assert_eq!(
        (report.live_nodes, report.live_edges),
        (LIVE_NODES, LIVE_EDGES)
    );
    assert_eq!(live.graph.rows_unknown_to_log().await.unwrap(), 0);
}

#[tokio::test]
async fn rebuild_without_a_reset_refuses_a_store_that_holds_one_node_and_no_edge() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    live.graph.insert(fact_at("alone")).await.unwrap();

    // Act
    let error = refused(&env, "live.db", live, RebuildMode::RequireEmpty).await;

    // Assert
    assert!(matches!(error, Error::ProjectionNotEmpty), "{error:?}");
}

#[tokio::test]
async fn checked_rebuild_refuses_a_lost_record_even_when_the_row_count_is_unchanged() {
    // Arrange: the store lost a logged row and holds an unlogged one in its
    // place, so it still has two live rows
    let env = Env::new();
    let live = open(&env, "live.db").await;
    live.graph.insert(fact_at("kept")).await.unwrap();
    let lost = live.graph.insert(fact_at("lost")).await.unwrap();
    delete_node(&live.graph, &lost).await;
    insert_unlogged(&live.graph, "node-unlogged", "only the store knows").await;
    assert_eq!(count(&live.graph, "nodes").await, 2);

    // Act
    let error = refused(&env, "live.db", live, RebuildMode::ResetChecked).await;

    // Assert: the count is the same before and after, the rebuild still refuses
    // to call the stores alike, and the row only the store knew is still there
    assert_eq!(
        mismatch(&error),
        (MismatchSource::PriorProjection, [2, 2, 1, 1])
    );
    assert!(has_node(&inspect(&env, "live.db").await, "node-unlogged").await);
}

#[tokio::test]
async fn checked_rebuild_refuses_a_store_that_only_lost_or_only_gained_rows() {
    // Arrange: three logged nodes with two gone, and two logged nodes beside
    // three the log never saw
    let (lost_env, gained_env) = (Env::new(), Env::new());
    let lost = open(&lost_env, "live.db").await;
    lost.graph.insert(fact_at("kept")).await.unwrap();
    for content in ["gone-1", "gone-2"] {
        let id = lost.graph.insert(fact_at(content)).await.unwrap();
        delete_node(&lost.graph, &id).await;
    }
    let gained = open(&gained_env, "live.db").await;
    gained.graph.insert(fact_at("one")).await.unwrap();
    gained.graph.insert(fact_at("two")).await.unwrap();
    for id in ["node-x", "node-y", "node-z"] {
        insert_unlogged(&gained.graph, id, &format!("extra {id}")).await;
    }

    // Act
    let lost = refused(&lost_env, "live.db", lost, RebuildMode::ResetChecked).await;
    let gained = refused(&gained_env, "live.db", gained, RebuildMode::ResetChecked).await;

    // Assert
    let prior = MismatchSource::PriorProjection;
    assert_eq!(mismatch(&lost), (prior, [3, 1, 2, 0]));
    assert_eq!(mismatch(&gained), (prior, [2, 5, 0, 3]));
}

#[tokio::test]
async fn catch_up_finishes_a_store_that_is_behind_the_log() {
    // Arrange
    let env = Env::new();
    let behind = behind_the_log(&env).await;
    let records = env.records().await;
    assert_eq!(
        cursor_offset(&behind.graph).await,
        Some(offset_pair(records[1].offset))
    );

    // Act
    let report = behind.graph.catch_up().await.unwrap();

    // Assert
    assert_eq!(record_counts(report).applied, 2);
    assert_eq!(count(&behind.graph, "nodes").await, 3);
    assert_eq!(count(&behind.graph, "edges").await, 1);
    let last = records.last().unwrap().offset;
    assert_eq!(cursor_offset(&behind.graph).await, Some(offset_pair(last)));
}

#[tokio::test]
async fn rebuild_modes_leave_a_store_behind_the_log_alone_except_replace_which_starts_over() {
    // Arrange
    let env = Env::new();
    let behind = behind_the_log(&env).await;
    let records = env.records().await;
    let behind_at = Some(offset_pair(records[1].offset));
    assert_eq!(cursor_offset(&behind.graph).await, behind_at);

    // Act
    let required = refused(&env, "behind.db", behind, RebuildMode::RequireEmpty).await;
    let behind = open(&env, "behind.db").await;
    let checked = refused(&env, "behind.db", behind, RebuildMode::ResetChecked).await;
    let behind = open(&env, "behind.db").await;
    assert_eq!(cursor_offset(&behind.graph).await, behind_at);
    let (replaced, report) = rebuilt(behind, RebuildMode::Replace).await;

    // Assert: the log holds two rows the store has not applied yet
    assert!(
        matches!(required, Error::ProjectionNotEmpty),
        "{required:?}"
    );
    assert_eq!(
        mismatch(&checked),
        (MismatchSource::PriorProjection, [4, 2, 2, 0])
    );
    assert_eq!((report.live_nodes, report.live_edges), (3, 1));
    assert_eq!(report.replayed.applied, 4);
    let last = records.last().unwrap().offset;
    assert_eq!(
        cursor_offset(&replaced.graph).await,
        Some(offset_pair(last))
    );
}

#[tokio::test]
async fn rebuild_multiset_counts_live_rows_only_and_leaves_superseded_ones_out() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    populate(&live).await;
    let superseded = format!("tx_to != {}", FOREVER.0);
    let dead = node_hashes(&live.graph, &superseded).await;
    assert_eq!(dead.len(), 2, "v1 and gamma are superseded");

    // Act
    let multiset = live.graph.projection_hash_multiset().await.unwrap();

    // Assert
    assert_eq!(multiset, oracle_multiset(&live.graph).await);
    assert_eq!(multiset.len(), LIVE_NODES + LIVE_EDGES);
    assert!(dead.iter().all(|hash| !multiset.contains(hash)));
}

#[tokio::test]
async fn rebuild_multiset_keeps_a_hash_held_by_two_live_rows_twice() {
    // Arrange: a live twin of alpha, which the dedup path would never write
    let env = Env::new();
    let live = open(&env, "live.db").await;
    let alpha = live.graph.insert(fact_at("alpha")).await.unwrap();
    insert_unlogged(&live.graph, "node-twin", "alpha").await;
    let hash = node_hash(&live.graph, alpha.as_str()).await;
    assert_eq!(node_hash(&live.graph, "node-twin").await, hash);

    // Act
    let multiset = live.graph.projection_hash_multiset().await.unwrap();

    // Assert
    assert_eq!(multiset.iter().filter(|held| **held == hash).count(), 2);
    assert_eq!(multiset, oracle_multiset(&live.graph).await);
}

// A tombstone already in a log must stay applied when the log is replayed, so
// a rebuild is where replay learns what to do with one.
#[tokio::test]
async fn rebuild_keeps_tombstoned_rows_out_of_the_store() {
    // Arrange: the log ends in a tombstone for a node and an edge
    let env = Env::new();
    let (lonely, mentions) = {
        let first = open(&env, "first.db").await;
        let populated = populate(&first).await;
        (populated.lonely, populated.mentions)
    };
    env.append(&[tombstone(
        "event-tombstone",
        &[
            (TombstoneTable::Nodes, lonely.as_str()),
            (TombstoneTable::Edges, mentions.as_str()),
        ],
    )]);
    let fresh = open(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert
    assert!(!has_node(&fresh.graph, lonely.as_str()).await);
    assert!(!has_edge(&fresh.graph, mentions.as_str()).await);
    assert_eq!(
        (report.live_nodes, report.live_edges),
        (LIVE_NODES - 1, LIVE_EDGES - 1)
    );
    assert_eq!(
        fresh.graph.projection_hash_multiset().await.unwrap(),
        oracle_multiset(&fresh.graph).await
    );
}

#[tokio::test]
async fn rebuild_drops_the_edges_of_a_tombstoned_node_that_the_tombstone_does_not_name() {
    // Arrange: a is tombstoned alone, and edges end at it on both sides
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        node_write("node-b", "beta"),
        node_write("node-c", "gamma"),
        edge_write("edge-ab", "node-a", "node-b"),
        edge_write("edge-ca", "node-c", "node-a"),
        edge_write("edge-bc", "node-b", "node-c"),
        tombstone("event-tombstone", &[(TombstoneTable::Nodes, "node-a")]),
    ]);
    let fresh = open(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert: both edges of a are gone with it, and the one between the others stays
    assert!(!has_edge(&fresh.graph, "edge-ab").await);
    assert!(!has_edge(&fresh.graph, "edge-ca").await);
    assert!(has_edge(&fresh.graph, "edge-bc").await);
    assert_eq!((report.live_nodes, report.live_edges), (2, 1));
}

#[tokio::test]
async fn rebuild_leaves_rows_the_log_wrote_already_closed_out_of_the_live_rows() {
    // Arrange: a node and an edge logged with a transaction end time
    let env = Env::new();
    let mut closed_node = node("node-closed", "old");
    closed_node.tx_to = 2000;
    let mut closed_edge = edge("edge-closed", "node-a", "node-b");
    closed_edge.tx_to = 2000;
    env.append(&[
        node_write("node-a", "alpha"),
        node_write("node-b", "beta"),
        event(
            "event-node-closed",
            node_row_hash(&closed_node),
            LogPayload::NodeWrite(closed_node),
        ),
        event(
            "event-edge-closed",
            edge_row_hash(&closed_edge),
            LogPayload::EdgeWrite(closed_edge),
        ),
    ]);
    let fresh = open(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert: both closed rows are stored, and neither counts as live
    assert_eq!(count(&fresh.graph, "nodes").await, 3);
    assert_eq!(count(&fresh.graph, "edges").await, 1);
    assert_eq!((report.live_nodes, report.live_edges), (2, 0));
}

#[tokio::test]
async fn rebuild_embeds_the_content_of_every_live_node_again() {
    // Arrange
    let env = Env::new();
    {
        let first = open(&env, "first.db").await;
        populate(&first).await;
    }
    let fresh = open(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert: one vector per live node, from that node's content, and none
    // for the superseded ones
    let live = live_contents(&fresh.graph).await;
    assert_eq!(live.len(), LIVE_NODES);
    assert_eq!(count(&fresh.graph, "node_vectors").await, LIVE_NODES as i64);
    for (id, content) in &live {
        assert_eq!(
            stored_vector(&fresh.graph, id).await,
            StubEmbedder::vector_for(DIMS, content),
            "{content}"
        );
    }
    let mut contents: Vec<_> = live.into_iter().map(|(_, content)| content).collect();
    contents.sort();
    assert_eq!(fresh.embedded(), contents);
    assert_eq!(report.replayed.reembedded.re_embedded, LIVE_NODES);
    assert_eq!(report.replayed.reembedded.pending, 0);
}

#[tokio::test]
async fn rebuild_without_an_embedder_reports_the_nodes_pending_instead_of_failing() {
    // Arrange
    let env = Env::new();
    {
        let first = open(&env, "first.db").await;
        populate(&first).await;
    }
    let fresh = open_without_embedder(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert
    assert_eq!(report.replayed.reembedded.pending, LIVE_NODES);
    assert_eq!(report.replayed.reembedded.re_embedded, 0);
    assert_eq!(count(&fresh.graph, "node_vectors").await, 0);
    assert_eq!(report.live_nodes, LIVE_NODES);
}

/// Puts a row in every table a rebuild empties but `nodes`, `edges` and
/// `log_hash_index`, which `populate` fills, and names the tables it did.
async fn seed_derived_tables(live: &Opened) -> [&'static str; 5] {
    let g = &live.graph;
    let node_id = g
        .backend
        .query("SELECT id FROM nodes LIMIT 1", &[])
        .await
        .unwrap()[0]
        .get_string(0)
        .unwrap();
    let community =
        "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, 1, 1)";
    g.backend
        .execute(community, &[node_id.as_str().into()])
        .await
        .unwrap();
    for sql in [
        "INSERT INTO cluster_state (edge_count, max_tx_from, computed_at, last_cold_start_at)
         VALUES (1, 1, 1, 1)",
        "INSERT INTO provenance_repair_state (id, last_repaired_at) VALUES (1, 5)",
        "INSERT INTO entity_mention_state
         (subject, scope, mentions_count, mentions_max_tx_from, last_synthesized_at)
         VALUES ('s', '', 1, 1, 1)",
        "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
         VALUES ('event-x', 0, 0, 'refused', 1)",
    ] {
        g.backend.execute(sql, &[]).await.unwrap();
    }
    [
        "node_community",
        "cluster_state",
        "provenance_repair_state",
        "entity_mention_state",
        "log_quarantine",
    ]
}

#[tokio::test]
async fn reset_empties_every_derived_table_by_its_rule_and_keeps_the_log_id() {
    for adopting in [false, true] {
        // Arrange: rows in every table the log derives, a vector among them
        let env = Env::new();
        let live = open(&env, "live.db").await;
        populate(&live).await;
        live.graph
            .insert(fact_at("with a vector").with_embedding(vec![0.5; DIMS]))
            .await
            .unwrap();
        let mut seeded = seed_derived_tables(&live).await.to_vec();
        seeded.extend(["nodes", "edges", "log_hash_index"]);
        let mut listed: Vec<_> = DERIVED_TABLES.iter().map(|(table, _)| *table).collect();
        seeded.sort_unstable();
        listed.sort_unstable();
        assert_eq!(seeded, listed, "a derived table the test does not fill");
        for table in listed.iter().chain(&["node_vectors"]) {
            assert!(count(&live.graph, table).await > 0, "{table}");
        }
        let before = snapshot(&live.graph).await;
        let (log_id, _) = cursor(&live.graph).await.expect("a cursor");
        let indexed = live
            .graph
            .backend
            .query("SELECT content_hash FROM log_hash_index LIMIT 1", &[])
            .await
            .unwrap()[0]
            .get_blob(0)
            .unwrap()
            .to_vec();
        let indexed = <[u8; 32]>::try_from(indexed).unwrap();
        assert!(live.log.lock().await.bloom_might_contain(&indexed));

        // Act
        let mut held = HeldLog::acquire(&live.log).await.unwrap();
        live.graph.reset(&mut held, adopting).await.unwrap();
        drop(held);

        // Assert: each table is empty unless it is kept for the same log, the
        // cursor starts over for the same log, and the filter forgot the hashes
        for (table, clear) in DERIVED_TABLES {
            let kept = clear == Clear::OnAdoption && !adopting;
            assert_eq!(
                count(&live.graph, table).await,
                i64::from(kept),
                "{table}, adopting: {adopting}"
            );
        }
        assert_eq!(count(&live.graph, "node_vectors").await, 0);
        assert_eq!(cursor(&live.graph).await, Some((log_id, None)));
        assert!(!live.log.lock().await.bloom_might_contain(&indexed));
        let replayed = live.graph.catch_up().await.unwrap();
        assert!(replayed.applied > 0, "{replayed:?}");
        assert_eq!(snapshot(&live.graph).await, before);
    }
}

#[tokio::test]
async fn rebuild_on_the_same_log_keeps_the_operators_records_and_clears_the_repair_watermark() {
    // Arrange: a live store whose replay refused the edge to a missing node,
    // with a repair watermark and a mention state
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        edge_write("edge-ghost", "node-a", "node-ghost"),
        node_write("node-b", "beta"),
    ]);
    let live = open(&env, "live.db").await;
    live.graph.catch_up().await.unwrap();
    for sql in [
        "INSERT INTO provenance_repair_state (id, last_repaired_at) VALUES (1, 5)",
        "INSERT INTO entity_mention_state
         (subject, scope, mentions_count, mentions_max_tx_from, last_synthesized_at)
         VALUES ('s', '', 1, 1, 1)",
    ] {
        live.graph.backend.execute(sql, &[]).await.unwrap();
    }
    let before = snapshot(&live.graph).await;
    let quarantine = quarantined(&live.graph).await;
    assert_eq!(quarantine.len(), 1);
    let records = env.records().await.len();
    assert!(!has_edge(&live.graph, "edge-ghost").await);

    // Act
    let (live, report) = rebuilt(live, RebuildMode::ResetChecked).await;

    // Assert: the void already in the log keeps the event out, as it did live,
    // the reason it was refused is still on record, and no second void exists
    let expected = CatchUpReport {
        applied: 2,
        skipped_voided: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report.replayed), expected);
    assert_eq!(snapshot(&live.graph).await, before);
    assert_eq!((report.live_nodes, report.live_edges), (2, 0));
    assert_eq!(quarantined(&live.graph).await, quarantine);
    assert_eq!(env.records().await.len(), records);
    assert_eq!(count(&live.graph, "provenance_repair_state").await, 0);
    assert_eq!(count(&live.graph, "entity_mention_state").await, 1);
}

#[tokio::test]
async fn rebuild_quarantines_a_refused_event_the_log_has_not_voided_yet() {
    // Arrange: nothing has replayed this log, so no void exists
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        edge_write("edge-ghost", "node-a", "node-ghost"),
        node_write("node-b", "beta"),
    ]);
    let fresh = open(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert: the same outcome a live replay has, quarantine row and void included
    let expected = CatchUpReport {
        applied: 2,
        quarantined: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report.replayed), expected);
    assert!(has_node(&fresh.graph, "node-a").await);
    assert!(has_node(&fresh.graph, "node-b").await);
    assert!(!has_edge(&fresh.graph, "edge-ghost").await);
    assert_eq!(count(&fresh.graph, "log_quarantine").await, 1);
    assert!(matches!(
        env.records()
            .await
            .last()
            .map(|record| &record.event.payload),
        Some(LogPayload::Voided { .. })
    ));
}

#[tokio::test]
async fn rebuild_replays_a_linked_edge_with_its_attributes() {
    // Arrange
    let env = Env::new();
    let before = {
        let first = open(&env, "first.db").await;
        let a = first.graph.insert(fact_at("a")).await.unwrap();
        let b = first.graph.insert(fact_at("b")).await.unwrap();
        let edge = NewEdge::new(&a, &b, "mentions").with_attributes(serde_json::json!({"w": 2}));
        first.graph.link(edge).await.unwrap();
        snapshot(&first.graph).await
    };
    let fresh = open(&env, "fresh.db").await;

    // Act
    let (fresh, report) = rebuilt(fresh, RebuildMode::RequireEmpty).await;

    // Assert
    assert_eq!(snapshot(&fresh.graph).await, before);
    assert_eq!((report.live_nodes, report.live_edges), (2, 1));
}

#[tokio::test]
async fn rows_unknown_to_log_counts_what_the_log_never_recorded_by_hash_not_by_id() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    let a = live.graph.insert(fact_at("original")).await.unwrap();
    let b = live.graph.insert(fact_at("untouched")).await.unwrap();
    let unknown = || live.graph.rows_unknown_to_log();
    assert_eq!(unknown().await.unwrap(), 0);

    // Act and assert: an unlogged node, then an unlogged edge between logged
    // nodes, then a logged node edited in place under the id the log knows
    insert_unlogged(&live.graph, "node-unlogged", "only the store knows").await;
    assert_eq!(unknown().await.unwrap(), 1);
    live.graph
        .backend
        .execute(
            "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
             VALUES ('edge-unlogged', ?1, ?2, 'mentions', '{}', 1000, ?3)",
            &[a.as_str().into(), b.as_str().into(), FOREVER.into()],
        )
        .await
        .unwrap();
    assert_eq!(unknown().await.unwrap(), 2);
    live.graph
        .backend
        .execute(
            "UPDATE nodes SET content = 'edited' WHERE id = ?1",
            &[a.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(unknown().await.unwrap(), 3);
}

#[tokio::test]
async fn rows_unknown_to_log_needs_an_attached_log() {
    // Arrange
    let graph = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .unwrap();

    // Act
    let refused = graph.rows_unknown_to_log().await;

    // Assert
    assert!(matches!(refused, Err(Error::NoEventLog)), "{refused:?}");
}

#[tokio::test]
async fn rebuild_replace_drops_a_row_the_log_never_recorded_without_a_mismatch() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    live.graph.insert(fact_at("kept")).await.unwrap();
    insert_unlogged(&live.graph, "node-unlogged", "only the store knows").await;

    // Act
    let (live, report) = rebuilt(live, RebuildMode::Replace).await;

    // Assert
    assert!(!has_node(&live.graph, "node-unlogged").await);
    assert_eq!(report.live_nodes, 1);
}

/// A store holding two live rows that no log has a record of.
async fn unlogged_pair(env: &Env, db: &str) -> Opened {
    let live = open(env, db).await;
    insert_unlogged(&live.graph, "node-x", "extra x").await;
    insert_unlogged(&live.graph, "node-y", "extra y").await;
    live
}

#[tokio::test]
async fn rebuild_over_an_empty_log_in_each_mode_refuses_what_it_would_wipe() {
    // Arrange: a log with no record and a store holding two rows it lacks
    let env = Env::new();
    let stores = [
        unlogged_pair(&env, "required.db").await,
        unlogged_pair(&env, "checked.db").await,
        unlogged_pair(&env, "replaced.db").await,
    ];
    let [required, checked, replaced] = stores;

    // Act: each is checked untouched afterwards
    let required = refused(&env, "required.db", required, RebuildMode::RequireEmpty).await;
    let checked = refused(&env, "checked.db", checked, RebuildMode::ResetChecked).await;
    let replaced = refused(&env, "replaced.db", replaced, RebuildMode::Replace).await;

    // Assert
    assert!(
        matches!(required, Error::ProjectionNotEmpty),
        "{required:?}"
    );
    assert_eq!(
        mismatch(&checked),
        (MismatchSource::PriorProjection, [0, 2, 0, 2])
    );
    assert!(
        matches!(
            replaced,
            Error::EmptyLogWouldWipe {
                rows_unknown_to_log: 2
            }
        ),
        "{replaced:?}"
    );
}

#[tokio::test]
async fn rebuild_of_an_empty_store_from_an_empty_log_succeeds_in_each_mode() {
    for mode in [
        RebuildMode::RequireEmpty,
        RebuildMode::ResetChecked,
        RebuildMode::Replace,
    ] {
        // Arrange
        let env = Env::new();
        let empty = open(&env, "empty.db").await;
        let log_id = empty.log.lock().await.log_id().to_string();

        // Act
        let (empty, report) = rebuilt(empty, mode).await;

        // Assert
        assert_eq!(report.replayed.applied, 0, "{mode:?}");
        assert_eq!((report.live_nodes, report.live_edges), (0, 0), "{mode:?}");
        assert_eq!(cursor(&empty.graph).await, Some((log_id, None)), "{mode:?}");
    }
}

#[tokio::test]
async fn rebuild_reads_a_log_whose_every_record_has_a_segment_of_its_own() {
    // Arrange: three records in three segments
    let env = Env::new();
    env.append_with(
        SEGMENT_PER_EVENT,
        &[
            node_write("node-a", "alpha"),
            node_write("node-b", "beta"),
            edge_write("edge-ab", "node-a", "node-b"),
        ],
    );
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let log = env.log(SEGMENT_PER_EVENT);
    let (graph, log) = env
        .open_over::<DefaultBackend>(log, "fresh.db", clock)
        .await;

    // Act
    let (graph, report) = graph
        .rebuild_from_log(log, RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert: the cursor sits on the last record, which is in the last segment
    assert_eq!((report.live_nodes, report.live_edges), (2, 1));
    let last = env.records().await.last().unwrap().offset;
    assert_eq!(last.segment, 2);
    assert_eq!(cursor_offset(&graph).await, Some(offset_pair(last)));
}

/// Opens `db` in `env` over the log of `other`, the way a store is that does
/// not belong to it: the open is refused, so the store is attached to no log.
async fn open_over_another_log(env: &Env, other: &Env, db: &str) -> DefaultGraph {
    let path = env.db(db);
    let graph = || DefaultGraph::open(&path, GraphConfig::new(DIMS));
    let attached = graph()
        .await
        .unwrap()
        .with_log(other.log(ONE_SEGMENT))
        .await;
    assert!(
        matches!(attached, Err(Error::LogIdMismatch { .. })),
        "{:?}",
        attached.err()
    );
    graph().await.unwrap()
}

#[tokio::test]
async fn rebuild_replace_refuses_a_log_the_store_does_not_belong_to_while_it_holds_rows() {
    // Arrange: a store written through one log, holding three rows a second
    // log with one row of its own does not know
    let (first, second) = (Env::new(), Env::new());
    let live = open(&first, "live.db").await;
    live.graph.insert(fact_at("one")).await.unwrap();
    live.graph.insert(fact_at("two")).await.unwrap();
    insert_unlogged(&live.graph, "node-x", "only the store knows").await;
    let store_log = live.log.lock().await.log_id();
    let before = full_state(&live.graph).await;
    drop(live);
    second.append(&[node_write("node-a", "alpha")]);
    let log = second.log(ONE_SEGMENT);
    let other_log = log.lock().await.log_id();
    let graph = open_over_another_log(&first, &second, "live.db").await;

    // Act
    let error = graph
        .rebuild_from_log(log, RebuildMode::Replace)
        .await
        .err()
        .expect("the rebuild was refused");

    // Assert: none of the rows were dropped
    assert!(
        matches!(
            error,
            Error::ForeignLogWouldWipe { store, log, rows_unknown_to_log: 3 }
                if store == store_log && log == other_log
        ),
        "{error:?}"
    );
    assert_eq!(full_state(&inspect(&first, "live.db").await).await, before);
}

#[tokio::test]
async fn rebuild_adopts_the_log_of_an_empty_store_that_belonged_to_another() {
    // Arrange: a database written through one log, opened over a second one,
    // holding what the first log left in the tables a rebuild keeps otherwise
    let (first, second) = (Env::new(), Env::new());
    drop(open(&first, "shared.db").await);
    second.append(&[node_write("node-a", "alpha")]);
    let graph = open_over_another_log(&first, &second, "shared.db").await;
    for sql in [
        "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
         VALUES ('event-x', 0, 0, 'refused', 1)",
        "INSERT INTO entity_mention_state
         (subject, scope, mentions_count, mentions_max_tx_from, last_synthesized_at)
         VALUES ('s', '', 1, 1, 1)",
    ] {
        graph.backend.execute(sql, &[]).await.unwrap();
    }
    let log = second.log(ONE_SEGMENT);

    // Act
    let (graph, report) = graph
        .rebuild_from_log(Arc::clone(&log), RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert: the store now belongs to the second log and holds its rows only
    assert_eq!(report.live_nodes, 1);
    let adopted = log.lock().await.log_id().to_string();
    assert_eq!(cursor(&graph).await.map(|(id, _)| id), Some(adopted));
    assert!(has_node(&graph, "node-a").await);
    assert_eq!(count(&graph, "log_quarantine").await, 0);
    assert_eq!(count(&graph, "entity_mention_state").await, 0);
    // the returned store is attached to the log it adopted
    assert_eq!(graph.rows_unknown_to_log().await.unwrap(), 0);
}

/// A trigger that alters or drops a row as it is inserted, which survives the
/// reset because a delete never drops it. The replay then produces rows the log
/// does not describe.
async fn drift_on_insert<B: Backend>(g: &Graph<B>, body: &str) {
    let create = format!(
        "CREATE TRIGGER drift AFTER INSERT ON nodes WHEN new.content = 'drifting' BEGIN {body} END"
    );
    g.backend.execute(&create, &[]).await.unwrap();
}

#[tokio::test]
async fn rebuild_into_an_empty_store_fails_when_the_replay_drops_a_row_the_log_holds() {
    // Arrange
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        node_write("node-b", "drifting"),
        node_write("node-c", "gamma"),
    ]);
    let fresh = open(&env, "fresh.db").await;
    drift_on_insert(&fresh.graph, "DELETE FROM nodes WHERE id = new.id;").await;

    // Act
    let error = failed(fresh, RebuildMode::RequireEmpty).await;

    // Assert: the replay itself committed, and the check against the log is what
    // refuses to call the result a rebuild
    assert_eq!(mismatch(&error), (MismatchSource::Log, [3, 2, 1, 0]));
}

#[tokio::test]
async fn replace_fails_when_the_replay_alters_a_row_the_log_holds() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    live.graph.insert(fact_at("alpha")).await.unwrap();
    live.graph.insert(fact_at("drifting")).await.unwrap();
    let alter = "UPDATE nodes SET label = 'drifted' WHERE id = new.id;";
    drift_on_insert(&live.graph, alter).await;

    // Act
    let error = failed(live, RebuildMode::Replace).await;

    // Assert: one row replaced by another, so the counts agree
    assert_eq!(mismatch(&error), (MismatchSource::Log, [2, 2, 1, 1]));
}
