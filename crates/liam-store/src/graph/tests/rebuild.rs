// SPDX-License-Identifier: Apache-2.0
//! Rebuilding the projection from the log alone. Tests run against a real WAL
//! and reader in a temp directory. What a store should hold is judged against
//! an oracle that reads the rows back with plain SQL and hashes them with the
//! log crate, never against the store's own multiset method.

use liam_log::event::{EdgeRow, LogEvent, LogPayload, NodeRow, TombstoneTable, TombstoneTarget};
use liam_log::hash::{edge_row_hash, node_row_hash};

use super::support::{
    count, cursor, event, fact_at, has_node, offset_pair, record_counts, Env, StubEmbedder,
    ONE_SEGMENT,
};
use super::*;
use crate::types::{EpisodeEdge, EpisodeRef};
use crate::DefaultBackend;

const DIMS: usize = 8;

/// A store opened over the log, with the handles a test needs to steer it.
struct Opened {
    graph: DefaultGraph,
    log: SharedLog,
    clock: Arc<FixedClock>,
    embedder: Arc<StubEmbedder>,
}

/// A store over the log with a stub embedder attached.
async fn open(env: &Env, db: &str) -> Opened {
    let embedder = StubEmbedder::new(DIMS);
    let mut opened = open_without_embedder(env, db).await;
    opened.graph = opened.graph.with_embedder(embedder.clone());
    opened.embedder = embedder;
    opened
}

async fn open_without_embedder(env: &Env, db: &str) -> Opened {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (graph, log) = env.open::<DefaultBackend>(db, Arc::clone(&clock)).await;
    Opened {
        graph,
        log,
        clock,
        embedder: StubEmbedder::new(DIMS),
    }
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

fn node_write(id: &str, content: &str) -> LogEvent {
    let row = node(id, content);
    event(
        &format!("event-{id}"),
        node_row_hash(&row),
        LogPayload::NodeWrite(row),
    )
}

fn edge_write(id: &str, src: &str, dst: &str) -> LogEvent {
    let row = EdgeRow {
        id: id.into(),
        src: src.into(),
        dst: dst.into(),
        edge_type: "mentions".into(),
        attributes: "{}".into(),
        tx_from: 1000,
        tx_to: FOREVER.0,
    };
    event(
        &format!("event-{id}"),
        edge_row_hash(&row),
        LogPayload::EdgeWrite(row),
    )
}

fn voided(target: &LogEvent) -> LogEvent {
    event(
        &format!("void-{}", target.event_id),
        target.content_hash,
        LogPayload::Voided {
            target_event_id: target.event_id.clone(),
        },
    )
}

fn tombstone(targets: &[(TombstoneTable, &str)]) -> LogEvent {
    let targets = targets
        .iter()
        .map(|(table, id)| TombstoneTarget {
            table: *table,
            id: (*id).into(),
        })
        .collect();
    event("event-tombstone", [0; 32], LogPayload::Tombstone(targets))
}

// ---- a populated store ----

/// The rows a test names after `populate`.
struct Populated {
    lonely: NodeId,
    mentions: EdgeId,
}

/// Live nodes and edges after `populate`.
const LIVE_NODES: usize = 7;
const LIVE_EDGES: usize = 4;

/// Every logged write path: inserts, upsert_by over and without a live
/// competitor, supersede, relate, and an episode. That is eight events, with
/// `v1` and `gamma` superseded and `lonely` a live node with no edge.
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

async fn cursor_offset<B: Backend>(g: &Graph<B>) -> Option<(i64, i64)> {
    cursor(g).await.and_then(|(_, offset)| offset)
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
    let report = fresh
        .graph
        .rebuild_from_log(RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert: the same rows, and the same multiset of hashes
    assert_eq!(snapshot(&fresh.graph).await, before);
    assert_eq!(multiset_before.len(), LIVE_NODES + LIVE_EDGES);
    assert_eq!(
        fresh.graph.projection_hash_multiset().await.unwrap(),
        multiset_before
    );
    assert!(report.hash_multiset_ok);
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
async fn rebuild_over_the_live_store_after_a_reset_yields_identical_rows() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    populate(&live).await;
    let before = snapshot(&live.graph).await;
    let multiset_before = oracle_multiset(&live.graph).await;
    let events = env.records().await.len();

    // Act
    let report = live
        .graph
        .rebuild_from_log(RebuildMode::Reset)
        .await
        .unwrap();

    // Assert
    assert_eq!(snapshot(&live.graph).await, before);
    assert_eq!(
        live.graph.projection_hash_multiset().await.unwrap(),
        multiset_before
    );
    assert!(report.hash_multiset_ok);
    assert_eq!(report.replayed.applied, events, "{report:?}");
    assert_eq!(
        (report.live_nodes, report.live_edges),
        (LIVE_NODES, LIVE_EDGES)
    );
}

#[tokio::test]
async fn rebuild_without_a_reset_refuses_a_projection_that_holds_rows() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    populate(&live).await;
    let before = snapshot(&live.graph).await;

    // Act
    let refused = live.graph.rebuild_from_log(RebuildMode::RequireEmpty).await;

    // Assert: nothing was touched
    assert!(
        matches!(refused, Err(Error::ProjectionNotEmpty)),
        "{refused:?}"
    );
    assert_eq!(snapshot(&live.graph).await, before);
}

#[tokio::test]
async fn rebuild_detects_a_lost_record_even_when_the_row_count_is_unchanged() {
    // Arrange: the store lost a logged row and holds an unlogged one in its
    // place, so it still has two live rows
    let env = Env::new();
    let live = open(&env, "live.db").await;
    live.graph.insert(fact_at("kept")).await.unwrap();
    let lost = live.graph.insert(fact_at("lost")).await.unwrap();
    live.graph
        .backend
        .execute("DELETE FROM nodes WHERE id = ?1", &[lost.as_str().into()])
        .await
        .unwrap();
    insert_unlogged(&live.graph, "node-unlogged", "only the store knows").await;
    assert_eq!(count(&live.graph, "nodes").await, 2);

    // Act
    let outcome = live.graph.rebuild_from_log(RebuildMode::Reset).await;

    // Assert: the count is the same before and after, and the rebuild still
    // refuses to call the result a match
    assert!(
        matches!(
            outcome,
            Err(Error::RebuildMismatch {
                expected: 2,
                found: 2,
                missing: 1,
                unexpected: 1,
            })
        ),
        "{outcome:?}"
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
    env.append(&[tombstone(&[
        (TombstoneTable::Nodes, lonely.as_str()),
        (TombstoneTable::Edges, mentions.as_str()),
    ])]);
    let fresh = open(&env, "fresh.db").await;

    // Act
    let report = fresh
        .graph
        .rebuild_from_log(RebuildMode::RequireEmpty)
        .await
        .unwrap();

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
async fn rebuild_embeds_the_content_of_every_live_node_again() {
    // Arrange
    let env = Env::new();
    {
        let first = open(&env, "first.db").await;
        populate(&first).await;
    }
    let fresh = open(&env, "fresh.db").await;

    // Act
    let report = fresh
        .graph
        .rebuild_from_log(RebuildMode::RequireEmpty)
        .await
        .unwrap();

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
    assert_eq!(fresh.embedder.calls(), contents);
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
    let report = fresh
        .graph
        .rebuild_from_log(RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert
    assert_eq!(report.replayed.reembedded.pending, LIVE_NODES);
    assert_eq!(report.replayed.reembedded.re_embedded, 0);
    assert_eq!(count(&fresh.graph, "node_vectors").await, 0);
    assert_eq!(report.live_nodes, LIVE_NODES);
}

#[tokio::test]
async fn rebuild_reset_empties_every_projection_table_and_keeps_the_log_id() {
    // Arrange: rows in every table the log derives
    let env = Env::new();
    let live = open(&env, "live.db").await;
    populate(&live).await;
    live.graph
        .insert(fact_at("with a vector").with_embedding(vec![0.5; DIMS]))
        .await
        .unwrap();
    live.graph
        .backend
        .execute(
            "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
             VALUES ('event-x', 0, 0, 'refused', 1)",
            &[],
        )
        .await
        .unwrap();
    let before = snapshot(&live.graph).await;
    let (log_id, offset) = cursor(&live.graph).await.expect("a cursor");
    assert!(offset.is_some());
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
    live.graph.reset_projection().await.unwrap();

    // Assert: every table is empty, the cursor starts over for the same log, and
    // the filter forgot the hashes
    for table in [
        "nodes",
        "edges",
        "node_vectors",
        "log_hash_index",
        "log_quarantine",
    ] {
        assert_eq!(count(&live.graph, table).await, 0, "{table}");
    }
    assert_eq!(cursor(&live.graph).await, Some((log_id, None)));
    assert!(!live.log.lock().await.bloom_might_contain(&indexed));
    let replayed = live.graph.catch_up().await.unwrap();
    assert!(replayed.applied > 0, "{replayed:?}");
    assert_eq!(snapshot(&live.graph).await, before);
}

#[tokio::test]
async fn rebuild_leaves_a_quarantined_event_out_of_the_store() {
    // Arrange: a live store whose replay refused the edge to a missing node
    let env = Env::new();
    env.append(&[
        node_write("node-a", "alpha"),
        edge_write("edge-ghost", "node-a", "node-ghost"),
        node_write("node-b", "beta"),
    ]);
    let live = open(&env, "live.db").await;
    live.graph.catch_up().await.unwrap();
    let before = snapshot(&live.graph).await;
    assert!(!has_edge(&live.graph, "edge-ghost").await);

    // Act
    let report = live
        .graph
        .rebuild_from_log(RebuildMode::Reset)
        .await
        .unwrap();

    // Assert: the replay-time void keeps the event out, as it did live
    let expected = CatchUpReport {
        applied: 2,
        skipped_voided: 1,
        ..CatchUpReport::default()
    };
    assert_eq!(record_counts(report.replayed), expected);
    assert_eq!(snapshot(&live.graph).await, before);
    assert_eq!((report.live_nodes, report.live_edges), (2, 0));
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
    let report = fresh
        .graph
        .rebuild_from_log(RebuildMode::RequireEmpty)
        .await
        .unwrap();

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
    let report = fresh
        .graph
        .rebuild_from_log(RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert
    assert_eq!(snapshot(&fresh.graph).await, before);
    assert_eq!((report.live_nodes, report.live_edges), (2, 1));
}

#[tokio::test]
async fn rebuild_rows_unknown_to_log_counts_only_what_the_log_never_recorded() {
    // Arrange
    let env = Env::new();
    let live = open(&env, "live.db").await;
    live.graph.insert(fact_at("kept")).await.unwrap();
    assert_eq!(live.graph.rows_unknown_to_log().await.unwrap(), 0);
    insert_unlogged(&live.graph, "node-unlogged", "only the store knows").await;

    // Act
    let unknown = live.graph.rows_unknown_to_log().await.unwrap();

    // Assert
    assert_eq!(unknown, 1);
}

#[tokio::test]
async fn rebuild_without_a_log_is_refused() {
    // Arrange
    let graph = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .unwrap();

    // Act
    let refused = graph.rebuild_from_log(RebuildMode::Reset).await;

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
    let report = live
        .graph
        .rebuild_from_log(RebuildMode::Replace)
        .await
        .unwrap();

    // Assert
    assert!(!has_node(&live.graph, "node-unlogged").await);
    assert_eq!(report.live_nodes, 1);
}

#[tokio::test]
async fn rebuild_adopts_the_log_of_a_store_that_belonged_to_another() {
    // Arrange: a database written through one log, opened over a second one
    let (first, second) = (Env::new(), Env::new());
    drop(open(&first, "shared.db").await);
    second.append(&[node_write("node-a", "alpha")]);
    let database = first.db("shared.db");
    let open = || DefaultGraph::open(&database, GraphConfig::new(DIMS));
    let refused = open()
        .await
        .unwrap()
        .with_log(second.log(ONE_SEGMENT))
        .await;
    assert!(
        matches!(refused, Err(Error::LogIdMismatch { .. })),
        "{:?}",
        refused.err()
    );
    let log = second.log(ONE_SEGMENT);
    let graph = open().await.unwrap().with_log_for_rebuild(Arc::clone(&log));

    // Act
    let report = graph
        .rebuild_from_log(RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert: the store now belongs to the second log and holds its rows
    assert_eq!(report.live_nodes, 1);
    let adopted = log.lock().await.log_id().to_string();
    assert_eq!(cursor(&graph).await.map(|(id, _)| id), Some(adopted));
    assert!(has_node(&graph, "node-a").await);
}
