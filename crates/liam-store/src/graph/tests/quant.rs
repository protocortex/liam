// SPDX-License-Identifier: Apache-2.0
//! The binary signature of a node and the index over it, and the lineage key a
//! node is written with. The signature is stored in the transaction that
//! writes the node, and the index follows every commit, close and removal.

use std::collections::BTreeMap;

use super::log_write::{logged_graph, mentions, race, within_deadline};
use super::support::{count, fact, injected, Env, StubEmbedder};
use super::*;
use crate::types::{LineageKey, RetentionPolicy};
use crate::DefaultBackend;

pub(super) const DIMS: usize = 8;

/// A vector whose sign pattern is `pattern`: component `j` is positive when bit
/// `j` is set. Its signature is `pattern`, so distances are easy to state.
pub(super) fn signed(pattern: u8) -> Vec<f32> {
    (0..DIMS)
        .map(|bit| if pattern >> bit & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

pub(super) fn node(label: &str, pattern: u8) -> NewNode {
    NewNode::now("fact", label, format!("content of {label}")).with_embedding(signed(pattern))
}

pub(super) async fn open_at(t: Millis) -> DefaultGraph {
    DefaultGraph::open_with_clock(
        ":memory:",
        GraphConfig::new(DIMS),
        Arc::new(FixedClock::new(t)),
    )
    .await
    .unwrap()
}

pub(super) async fn quant_ids<B: Backend>(g: &Graph<B>) -> Vec<String> {
    column(g, "SELECT node_id FROM node_quant ORDER BY node_id").await
}

pub(super) async fn column<B: Backend>(g: &Graph<B>, sql: &str) -> Vec<String> {
    g.backend
        .query(sql, &[])
        .await
        .unwrap()
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect()
}

pub(super) fn indexed_ids<B: Backend>(g: &Graph<B>) -> Vec<String> {
    g.quant_index()
        .snapshot()
        .into_iter()
        .map(|(id, _, _)| id)
        .collect()
}

/// The ids of the live nodes that have a signature row, which the index must
/// hold exactly.
pub(super) async fn live_signed<B: Backend>(g: &Graph<B>) -> Vec<String> {
    column(
        g,
        "SELECT q.node_id FROM node_quant q JOIN nodes n ON n.id = q.node_id
         WHERE n.tx_to = 4102444800000 ORDER BY q.node_id",
    )
    .await
}

pub(super) async fn lineages<B: Backend>(g: &Graph<B>) -> BTreeMap<String, String> {
    g.backend
        .query("SELECT node_id, lineage_key FROM node_lineage", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| (row.get_string(0).unwrap(), row.get_string(1).unwrap()))
        .collect()
}

// ---- the signature is written with the node ----

#[tokio::test]
async fn every_write_path_stores_a_signature_and_indexes_it_and_a_closed_version_leaves() {
    // Arrange
    let g = open_at(Millis(1000)).await;

    // Act
    let inserted = g.insert(node("inserted", 0b0001)).await.unwrap();
    let upserted = g
        .upsert_by(node("upserted", 0b0010).with_subject("subject"))
        .await
        .unwrap();
    let replacing = g
        .upsert_by(node("replacing", 0b0100).with_subject("subject"))
        .await
        .unwrap();
    let superseding = g
        .supersede(&inserted, node("superseding", 0b1000))
        .await
        .unwrap();
    let episode = g
        .ingest_episode(
            vec![node("episode one", 0b0011), node("episode two", 0b1100)],
            vec![mentions(0, 1)],
        )
        .await
        .unwrap()
        .node_ids;

    // Assert: a row for every node written, an entry only for the live ones
    let mut written = vec![
        inserted.clone(),
        upserted.clone(),
        replacing.clone(),
        superseding.clone(),
    ];
    written.extend(episode.clone());
    let mut all: Vec<String> = written.iter().map(|id| id.as_str().to_string()).collect();
    all.sort();
    assert_eq!(quant_ids(&g).await, all);
    let mut live = all.clone();
    live.retain(|id| id != inserted.as_str() && id != upserted.as_str());
    assert_eq!(indexed_ids(&g), live);
    assert_eq!(indexed_ids(&g), live_signed(&g).await);
}

#[tokio::test]
async fn a_signature_is_the_sign_bits_and_the_int8_form_is_kept_while_the_node_is_hot() {
    // Arrange
    let g = open_at(Millis(1000)).await;

    // Act
    let id = g.insert(node("n", 0b1010_0101)).await.unwrap();

    // Assert
    let entry = g.quant_index().snapshot().pop().unwrap();
    assert_eq!(entry, (id.as_str().to_string(), false, vec![0b1010_0101]));
    let row = g
        .backend
        .query("SELECT i8, scale, quant_version FROM node_quant", &[])
        .await
        .unwrap();
    assert_eq!(row[0].get_blob(0).unwrap().len(), DIMS);
    assert_eq!(row[0].get_f64(1).unwrap(), f64::from(1.0_f32 / 127.0));
    assert_eq!(row[0].get_i64(2).unwrap(), 1);
}

#[tokio::test]
async fn a_node_without_an_embedding_has_no_signature() {
    // Arrange
    let g = open_at(Millis(1000)).await;

    // Act
    g.insert(NewNode::now("fact", "plain", "no vector"))
        .await
        .unwrap();

    // Assert
    assert_eq!(count(&g, "node_quant").await, 0);
    assert!(g.quant_index().is_empty());
}

/// Writes that carry an embedding, over a store that already holds `old`.
#[derive(Clone, Copy, Debug)]
enum Write {
    Insert,
    Supersede,
    Episode,
}

impl Write {
    /// The transaction statements the write runs, which the fault test must
    /// find: the node, its origin event, its lineage and its signature, and for
    /// a supersede the close of the old node and the edge to it.
    fn statements(self) -> usize {
        match self {
            Write::Insert | Write::Episode => 4,
            Write::Supersede => 6,
        }
    }

    async fn run(self, g: &Graph<FailingBackend>, old: &NodeId) -> Result<()> {
        match self {
            Write::Insert => g.insert(node("new", 0b0110)).await.map(drop),
            Write::Supersede => g.supersede(old, node("new", 0b0110)).await.map(drop),
            Write::Episode => g
                .ingest_episode(vec![node("new", 0b0110)], Vec::new())
                .await
                .map(drop),
        }
    }
}

#[derive(Debug, PartialEq)]
struct Held {
    nodes: Vec<(String, i64)>,
    quant: Vec<String>,
    lineage: BTreeMap<String, String>,
    indexed: Vec<IndexedEntry>,
}

async fn held(g: &Graph<FailingBackend>) -> Held {
    let nodes = g
        .backend
        .query("SELECT id, tx_to FROM nodes ORDER BY id", &[])
        .await
        .unwrap()
        .iter()
        .map(|row| (row.get_string(0).unwrap(), row.get_i64(1).unwrap()))
        .collect();
    Held {
        nodes,
        quant: quant_ids(g).await,
        lineage: lineages(g).await,
        indexed: g.quant_index().snapshot(),
    }
}

#[tokio::test]
async fn a_failure_at_any_statement_of_an_embedded_write_leaves_nothing_behind() {
    for write in [Write::Insert, Write::Supersede, Write::Episode] {
        // Arrange
        let g = Graph::<FailingBackend>::open_with_clock(
            ":memory:",
            GraphConfig::new(DIMS),
            Arc::new(FixedClock::new(Millis(1000))),
        )
        .await
        .unwrap();
        let old = g.insert(node("old", 0b1111)).await.unwrap();
        let before = held(&g).await;
        let mut failures = 0;

        // Act and assert: each statement in turn fails, and only the write after
        // the last one succeeds
        for statement in 0..20 {
            g.backend.set_fail_on_execute(statement);
            let result = write.run(&g, &old).await;
            match result {
                Err(ref error) => {
                    assert!(injected(error), "{write:?} at {statement}: {error:?}");
                    assert_eq!(held(&g).await, before, "{write:?} at {statement}");
                    failures += 1;
                }
                Ok(()) => break,
            }
        }
        assert_eq!(failures, write.statements(), "{write:?}: statements it ran");
        let after = held(&g).await;
        assert_eq!(after.indexed.len(), live_signed(&g).await.len());
        assert!(after.quant.len() > before.quant.len());
    }
}

#[tokio::test]
async fn a_committed_node_keeps_its_signature_when_the_vector_write_fails() {
    // Arrange
    let g = Graph::<FailingVectorBackend>::open_with_clock(
        ":memory:",
        GraphConfig::new(DIMS),
        Arc::new(FixedClock::new(Millis(1000))),
    )
    .await
    .unwrap();
    g.backend.set_fail_on_vector_upsert(0);

    // Act
    let result = g.insert(node("n", 0b0101)).await;

    // Assert
    assert!(result.is_err());
    assert_eq!(count(&g, "nodes").await, 1);
    assert_eq!(count(&g, "node_quant").await, 1);
    assert_eq!(count(&g, "node_vectors").await, 0);
    assert_eq!(g.quant_index().len(), 1);
}

#[tokio::test]
async fn an_unusable_embedding_fails_the_write_before_any_row_exists() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let old = g.insert(NewNode::now("fact", "old", "old")).await.unwrap();
    let mut non_finite = signed(0);
    non_finite[3] = f32::NAN;
    let cases = [(vec![1.0, 2.0, 3.0], "dimension"), (non_finite, "finite")];

    for (embedding, why) in cases {
        let bad = || NewNode::now("fact", "bad", "bad").with_embedding(embedding.clone());

        // Act
        let results = [
            g.insert(bad()).await.map(drop),
            g.upsert_by(bad().with_subject("s")).await.map(drop),
            g.supersede(&old, bad()).await.map(drop),
        ];

        // Assert
        for result in results {
            let message = result.expect_err(why).to_string();
            assert!(message.contains(why), "{message}");
        }
        assert_eq!(count(&g, "nodes").await, 1, "{why}");
        assert_eq!(count(&g, "node_lineage").await, 1, "{why}");
    }
}

#[tokio::test]
async fn an_episode_node_with_an_unusable_embedding_is_written_without_a_signature() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let bad = NewNode::now("fact", "bad", "bad").with_embedding(vec![1.0]);

    // Act
    let result = g
        .ingest_episode(vec![bad, node("good", 0b1)], Vec::new())
        .await;

    // Assert
    let ids = result.unwrap().node_ids;
    assert_eq!(count(&g, "nodes").await, 2);
    assert_eq!(quant_ids(&g).await, [ids[1].as_str().to_string()]);
}

#[tokio::test]
async fn a_duplicate_write_that_brings_an_embedding_gives_the_first_node_a_signature() {
    // Arrange
    let (g, _appended, _log) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    let first = g.insert(fact("same words")).await.unwrap();
    assert_eq!(count(&g, "node_quant").await, 0);

    // Act
    let second = g
        .insert(fact("same words").with_embedding(signed(0b11)))
        .await
        .unwrap();

    // Assert
    assert_eq!(second, first);
    assert_eq!(quant_ids(&g).await, [first.as_str().to_string()]);
    assert_eq!(indexed_ids(&g), [first.as_str().to_string()]);
}

// ---- the index follows the tables ----

#[tokio::test]
async fn backfill_writes_a_signature_for_each_vector_once() {
    // Arrange: a store from before signatures, which has vectors and no rows
    let g = open_at(Millis(1000)).await;
    let with_vector = g.insert(node("embedded", 0b1001)).await.unwrap();
    g.insert(NewNode::now("fact", "plain", "no vector"))
        .await
        .unwrap();
    for sql in ["DELETE FROM node_quant", "DELETE FROM quant_backfill_state"] {
        g.backend.execute(sql, &[]).await.unwrap();
    }
    g.reload_quant_index().await.unwrap();
    assert!(g.quant_index().is_empty());

    // Act
    let first = g.backfill_quant().await.unwrap();
    let again = g.backfill_quant().await.unwrap();

    // Assert
    assert_eq!((first, again), (1, 0));
    assert_eq!(quant_ids(&g).await, [with_vector.as_str().to_string()]);
    assert_eq!(
        g.quant_index().snapshot(),
        [(with_vector.as_str().to_string(), false, vec![0b1001])]
    );
    assert_eq!(count(&g, "quant_backfill_state").await, 1);
}

#[tokio::test]
async fn the_index_equals_the_table_after_the_store_is_reopened() {
    // Arrange
    let env = Env::new();
    let path = env.db("reopen.db");
    let open = || {
        DefaultGraph::open_with_clock(
            &path,
            GraphConfig::new(DIMS),
            Arc::new(FixedClock::new(Millis(1000))),
        )
    };
    let g = open().await.unwrap();
    let old = g.insert(node("old", 0b01)).await.unwrap();
    g.supersede(&old, node("new", 0b10)).await.unwrap();
    g.insert(node("other", 0b11)).await.unwrap();
    let before = g.quant_index().snapshot();
    drop(g);

    // Act
    let reopened = open().await.unwrap();

    // Assert
    assert_eq!(reopened.quant_index().snapshot(), before);
    assert_eq!(indexed_ids(&reopened), live_signed(&reopened).await);
}

#[tokio::test]
async fn a_reload_repairs_an_index_that_drifted_from_the_table() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let id = g.insert(node("n", 0b1)).await.unwrap();
    g.quant_index().remove([id.as_str()]);
    g.quant_index().insert("ghost", true, vec![7]);

    // Act
    g.reload_quant_index().await.unwrap();

    // Assert
    assert_eq!(indexed_ids(&g), [id.as_str().to_string()]);
}

#[tokio::test]
async fn gc_removes_the_signature_row_and_the_index_entry_of_a_deleted_node() {
    // Arrange
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let doomed = g
        .insert(NewNode::now("scratch", "old", "old").with_embedding(signed(1)))
        .await
        .unwrap();
    let kept = g.insert(node("kept", 0b10)).await.unwrap();
    clock.set(Millis(1000 + 10_000));

    // Act
    let report = g
        .gc(&RetentionPolicy::keep("scratch", Millis(1000)))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(quant_ids(&g).await, [kept.as_str().to_string()]);
    assert_eq!(indexed_ids(&g), [kept.as_str().to_string()]);
    assert_eq!(count(&g, "node_lineage").await, 1);
    assert!(!lineages(&g).await.contains_key(doomed.as_str()));
}

#[tokio::test]
async fn hamming_top_over_stored_signatures_is_exact_and_ties_go_by_id() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let near = g.insert(node("near", 0b0000_0001)).await.unwrap();
    let mid = g.insert(node("mid", 0b0000_0111)).await.unwrap();
    let far = g.insert(node("far", 0b1111_1111)).await.unwrap();
    let tie_a = g.insert(node("tie a", 0b0000_0110)).await.unwrap();
    let tie_b = g.insert(node("tie b", 0b0000_1010)).await.unwrap();
    let query = quantize(&signed(0)).unwrap().bin;

    // Act
    let found = g.quant_index().hamming_top(&query, 10, false).unwrap();

    // Assert: distances 1, 2, 2, 3, 8; the two ties by ascending id
    let mut ties = [tie_a, tie_b];
    ties.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    let expected = [
        (near, 1),
        (ties[0].clone(), 2),
        (ties[1].clone(), 2),
        (mid, 3),
        (far, 8),
    ];
    assert_eq!(found, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writers_racing_readers_leave_the_index_equal_to_the_table() {
    const WRITERS: usize = 4;
    const ROUNDS: usize = 25;
    // Arrange
    let g = Arc::new(open_at(Millis(1000)).await);
    let query = quantize(&signed(0)).unwrap().bin;

    // Act: each writer inserts and supersedes while readers search
    let observed = within_deadline(race(WRITERS * 2, |worker| {
        let (g, query) = (Arc::clone(&g), query.clone());
        async move {
            let mut seen = Vec::new();
            for round in 0..ROUNDS {
                if worker < WRITERS {
                    let id = g
                        .insert(node(&format!("w{worker} r{round}"), round as u8))
                        .await
                        .unwrap();
                    g.supersede(&id, node(&format!("w{worker} r{round} v2"), !(round as u8)))
                        .await
                        .unwrap();
                } else {
                    let found = g.quant_index().hamming_top(&query, 8, false).unwrap();
                    seen.extend(found.into_iter().map(|(id, _)| id.as_str().to_string()));
                }
            }
            seen
        }
    }))
    .await;

    // Assert: every id a reader saw was written, and the index ends equal
    let written: std::collections::HashSet<String> = column(&g, "SELECT id FROM nodes")
        .await
        .into_iter()
        .collect();
    for id in observed.iter().flatten() {
        assert!(written.contains(id), "{id} was never written");
    }
    assert_eq!(indexed_ids(&g), live_signed(&g).await);
    assert_eq!(g.quant_index().len(), WRITERS * ROUNDS);
}

// ---- lineage ----

#[tokio::test]
async fn lineage_keys_keep_the_collision_identity_of_a_subject() {
    // Arrange
    let g = open_at(Millis(1000)).await;

    // Act
    let page = g
        .upsert_by(NewNode::entity("person", "Ada").with_scope("work"))
        .await
        .unwrap();
    let from_a = g
        .upsert_by(
            NewNode::now("fact", "a", "text")
                .with_subject("ada")
                .with_scope("work")
                .with_producer("agent-a"),
        )
        .await
        .unwrap();
    let from_b = g
        .upsert_by(
            NewNode::now("fact", "b", "text")
                .with_subject("ada")
                .with_scope("work")
                .with_producer("agent-b"),
        )
        .await
        .unwrap();

    // Assert: three distinct keys, one for each thing a write can replace
    let keys = lineages(&g).await;
    let of = |id: &NodeId| LineageKey::from_raw(keys[id.as_str()].clone());
    assert_eq!(of(&page).as_str(), "e:4:work:3:ada");
    assert_eq!(of(&from_a).as_str(), "s:4:work:3:ada:7:agent-a");
    assert_eq!(of(&from_b).as_str(), "s:4:work:3:ada:7:agent-b");
}

#[tokio::test]
async fn a_new_version_keeps_the_key_of_the_node_it_replaced() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    let page = |label: &str| NewNode::entity("person", label).with_scope("work");

    // Act
    let first = g.upsert_by(page("Ada")).await.unwrap();
    let second = g.upsert_by(page("Ada")).await.unwrap();
    let root = g
        .insert(NewNode::now("scratch", "root", "x"))
        .await
        .unwrap();
    let middle = g
        .supersede(&root, NewNode::now("fact", "m", "y"))
        .await
        .unwrap();
    let last = g
        .supersede(&middle, NewNode::now("fact", "l", "z"))
        .await
        .unwrap();

    // Assert
    let keys = lineages(&g).await;
    assert_eq!(keys[first.as_str()], keys[second.as_str()]);
    let chain = [&root, &middle, &last].map(|id| keys[id.as_str()].clone());
    assert!(chain.iter().all(|key| *key == chain[0]), "{chain:?}");
    assert_eq!(
        chain[0],
        format!("n:{}:{}", root.as_str().len(), root.as_str())
    );
}

#[tokio::test]
async fn deleting_the_root_of_a_chain_does_not_change_the_key_of_the_survivors() {
    // Arrange
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let root = g
        .insert(NewNode::now("scratch", "root", "x"))
        .await
        .unwrap();
    let child = g
        .supersede(&root, NewNode::now("fact", "c", "y"))
        .await
        .unwrap();
    let before = lineages(&g).await[child.as_str()].clone();
    clock.set(Millis(1000 + 10_000));

    // Act
    g.gc(&RetentionPolicy::keep("scratch", Millis(1000)))
        .await
        .unwrap();

    // Assert
    let keys = lineages(&g).await;
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[child.as_str()], before);
}

#[tokio::test]
async fn a_rebuild_reproduces_lineage_and_signatures_from_the_log_and_the_embedder() {
    // Arrange
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (graph, log) = env
        .open::<DefaultBackend>("live.db", Arc::clone(&clock))
        .await;
    let graph = graph.with_embedder(StubEmbedder::new(DIMS));
    let root = graph
        .insert(NewNode::now("fact", "root", "first"))
        .await
        .unwrap();
    let child = graph
        .supersede(&root, NewNode::now("fact", "child", "second"))
        .await
        .unwrap();
    graph
        .upsert_by(NewNode::entity("person", "Ada").with_scope("work"))
        .await
        .unwrap();
    graph.reembed_missing().await.unwrap();
    let keys = lineages(&graph).await;
    let indexed = graph.quant_index().snapshot();

    // Act
    let (rebuilt, _) = graph
        .rebuild_from_log(log, RebuildMode::Replace)
        .await
        .unwrap();

    // Assert
    assert_eq!(lineages(&rebuilt).await, keys);
    assert_eq!(rebuilt.quant_index().snapshot(), indexed);
    assert!(!indexed_ids(&rebuilt).contains(&root.as_str().to_string()));
    assert!(indexed_ids(&rebuilt).contains(&child.as_str().to_string()));
}
