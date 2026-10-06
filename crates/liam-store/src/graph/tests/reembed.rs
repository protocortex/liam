// SPDX-License-Identifier: Apache-2.0
//! Re-embedding the live nodes that have no vector.

use std::collections::HashSet;

use super::support::{count, fact_at, ReembedProbe, StubEmbedder};
use super::*;

const DIMS: usize = 8;

fn vector() -> Vec<f32> {
    vec![0.5; DIMS]
}

async fn missing<B: Backend>(g: &Graph<B>) -> HashSet<NodeId> {
    g.backend
        .nodes_missing_vectors()
        .await
        .unwrap()
        .into_iter()
        .collect()
}

async fn probed_graph() -> Graph<ReembedProbe> {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    Graph::<ReembedProbe>::open_with_clock(":memory:", GraphConfig::new(DIMS), clock)
        .await
        .unwrap()
}

/// The node whose vector is nearest to `vector`.
async fn nearest<B: Backend>(g: &Graph<B>, vector: &[f32]) -> Vec<NodeId> {
    g.backend
        .vector_search(vector, 1, None, None, Millis(1000))
        .await
        .unwrap()
}

async fn stored_vector<B: Backend>(g: &Graph<B>, id: &NodeId) -> Vec<f32> {
    let rows = g
        .backend
        .query(
            "SELECT embedding FROM node_vectors WHERE node_id = ?1",
            &[id.as_str().into()],
        )
        .await
        .unwrap();
    let bytes = rows[0].get_blob(0).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

#[tokio::test]
async fn reembed_missing_embeds_exactly_the_live_nodes_that_have_no_vector() {
    // Arrange: two live nodes without a vector, one with, and one superseded
    let embedder = StubEmbedder::new(DIMS);
    let g = graph_at(Millis(1000)).await.with_embedder(embedder.clone());
    let a = g.insert(fact_at("a")).await.unwrap();
    g.insert(fact_at("b").with_embedding(vector()))
        .await
        .unwrap();
    let c = g.insert(fact_at("c")).await.unwrap();
    let d = g.supersede(&c, fact_at("d")).await.unwrap();
    let live_without = HashSet::from([a.clone(), d.clone()]);
    assert_eq!(missing(&g).await, live_without);

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert
    let expected = ReembedReport {
        re_embedded: 2,
        ..ReembedReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(embedder.calls(), ["a", "d"]);
    assert!(missing(&g).await.is_empty());
    assert_eq!(count(&g, "node_vectors").await, 3);
    assert_eq!(nearest(&g, &StubEmbedder::vector_for(DIMS, "a")).await, [a]);
    assert_eq!(nearest(&g, &StubEmbedder::vector_for(DIMS, "d")).await, [d]);
}

#[tokio::test]
async fn reembed_missing_leaves_a_node_that_has_a_vector_alone() {
    // Arrange
    let embedder = StubEmbedder::new(DIMS);
    let g = graph_at(Millis(1000)).await.with_embedder(embedder.clone());
    g.insert(fact_at("b").with_embedding(vector()))
        .await
        .unwrap();

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert
    assert_eq!(report, ReembedReport::default());
    assert!(embedder.calls().is_empty());
}

#[tokio::test]
async fn reembed_missing_counts_a_failed_embed_and_finds_the_node_on_the_next_pass() {
    // Arrange
    let embedder = StubEmbedder::new(DIMS);
    let g = graph_at(Millis(1000)).await.with_embedder(embedder.clone());
    let a = g.insert(fact_at("a")).await.unwrap();
    g.insert(fact_at("b")).await.unwrap();
    embedder.fail_on(Some("a"));

    // Act
    let failed = g.reembed_missing().await.unwrap();

    // Assert: the node stays, without a vector, and the other node is embedded
    let expected = ReembedReport {
        re_embedded: 1,
        failed: 1,
        ..ReembedReport::default()
    };
    assert_eq!(failed, expected);
    assert_eq!(missing(&g).await, HashSet::from([a]));
    assert_eq!(count(&g, "nodes").await, 2);

    // Act: the embedder recovers
    embedder.fail_on(None);
    let retried = g.reembed_missing().await.unwrap();

    // Assert
    let expected = ReembedReport {
        re_embedded: 1,
        ..ReembedReport::default()
    };
    assert_eq!(retried, expected);
    assert!(missing(&g).await.is_empty());
}

#[tokio::test]
async fn reembed_missing_counts_a_vector_of_the_wrong_size_as_a_failure() {
    // Arrange
    let g = graph_at(Millis(1000))
        .await
        .with_embedder(StubEmbedder::new(DIMS - 1));
    let a = g.insert(fact_at("a")).await.unwrap();

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert
    let expected = ReembedReport {
        failed: 1,
        ..ReembedReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(missing(&g).await, HashSet::from([a]));
}

#[tokio::test]
async fn reembed_missing_without_an_embedder_changes_nothing_and_counts_the_gap() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(fact_at("a")).await.unwrap();
    g.insert(fact_at("b").with_embedding(vector()))
        .await
        .unwrap();

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert
    let expected = ReembedReport {
        pending: 1,
        ..ReembedReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(missing(&g).await, HashSet::from([a]));
}

#[tokio::test]
async fn reembed_missing_skips_nodes_with_no_content_to_embed() {
    // Arrange: an entity page, a whitespace only fact, and a fact with content
    let embedder = StubEmbedder::new(DIMS);
    let g = graph_at(Millis(1000)).await.with_embedder(embedder.clone());
    g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    g.insert(fact_at(" \t\r\n\x0b\x0c ")).await.unwrap();
    let a = g.insert(fact_at("a")).await.unwrap();

    // Act
    let listed = missing(&g).await;
    let report = g.reembed_missing().await.unwrap();

    // Assert
    assert_eq!(listed, HashSet::from([a]));
    let expected = ReembedReport {
        re_embedded: 1,
        ..ReembedReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(embedder.calls(), ["a"]);
    assert_eq!(count(&g, "node_vectors").await, 1);
}

#[tokio::test]
async fn reembed_missing_keeps_a_vector_stored_after_the_listing() {
    // Arrange: a live write stores the vector between the listing and the repair
    let embedder = StubEmbedder::new(DIMS);
    let g = probed_graph().await.with_embedder(embedder);
    let a = g.insert(fact_at("a")).await.unwrap();
    g.insert(fact_at("b")).await.unwrap();
    let live_write = vector();
    g.backend.store_after_listing(&a, live_write.clone());

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert: the live vector survives and only the other node is counted
    assert_eq!(stored_vector(&g, &a).await, live_write);
    let expected = ReembedReport {
        re_embedded: 1,
        ..ReembedReport::default()
    };
    assert_eq!(report, expected);
    assert!(missing(&g).await.is_empty());
}

#[tokio::test]
async fn reembed_missing_counts_an_unreadable_node_as_failed_and_goes_on() {
    // Arrange
    let embedder = StubEmbedder::new(DIMS);
    let g = probed_graph().await.with_embedder(embedder.clone());
    let a = g.insert(fact_at("a")).await.unwrap();
    g.insert(fact_at("b")).await.unwrap();
    g.backend.fail_content_read(&a);

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert
    let expected = ReembedReport {
        re_embedded: 1,
        failed: 1,
        ..ReembedReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(embedder.calls(), ["b"]);
    assert_eq!(missing(&g).await, HashSet::from([a]));
}

#[tokio::test]
async fn reembed_missing_ignores_a_node_removed_after_it_was_listed() {
    // Arrange: the listing names a node that does not exist
    let embedder = StubEmbedder::new(DIMS);
    let g = probed_graph().await.with_embedder(embedder.clone());
    g.insert(fact_at("a")).await.unwrap();
    g.backend.list_also(NodeId::new());

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert
    let expected = ReembedReport {
        re_embedded: 1,
        ..ReembedReport::default()
    };
    assert_eq!(report, expected);
    assert_eq!(embedder.calls(), ["a"]);
}
