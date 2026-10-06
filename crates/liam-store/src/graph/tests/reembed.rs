// SPDX-License-Identifier: Apache-2.0
//! Re-embedding the live nodes that have no vector.

use std::collections::HashSet;

use super::support::{count, fact_at, StubEmbedder};
use super::*;

const DIMS: usize = 8;

fn vector() -> Vec<f32> {
    vec![0.5; DIMS]
}

async fn missing(g: &DefaultGraph) -> HashSet<NodeId> {
    g.nodes_missing_vectors()
        .await
        .unwrap()
        .into_iter()
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
        failed: 0,
    };
    assert_eq!(report, expected);
    assert_eq!(embedder.calls(), ["a", "d"]);
    assert!(missing(&g).await.is_empty());
    assert_eq!(count(&g, "node_vectors").await, 3);
    let found = g
        .backend
        .vector_search(&[1.0; DIMS], 10, None, None, Millis(1000))
        .await
        .unwrap();
    assert!(found.contains(&a) && found.contains(&d), "{found:?}");
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
        failed: 0,
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
        re_embedded: 0,
        failed: 1,
    };
    assert_eq!(report, expected);
    assert_eq!(missing(&g).await, HashSet::from([a]));
}

#[tokio::test]
async fn reembed_missing_without_an_embedder_changes_nothing() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(fact_at("a")).await.unwrap();

    // Act
    let report = g.reembed_missing().await.unwrap();

    // Assert
    assert_eq!(report, ReembedReport::default());
    assert_eq!(missing(&g).await, HashSet::from([a]));
}
