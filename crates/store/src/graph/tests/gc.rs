// SPDX-License-Identifier: Apache-2.0
//! Garbage collection of nodes, edges and vectors.

use super::*;

#[tokio::test]
async fn gc_ages_out_by_kind() {
    let (g, clock) = gc_graph().await;

    support::gc_ages_out_by_kind(&g, &clock).await;
}

#[tokio::test]
async fn gc_with_a_non_positive_max_age_deletes_nothing() {
    for max_age in [Millis(0), Millis(-1)] {
        // Arrange
        let (g, clock) = gc_graph().await;
        clock.set(Millis(100_000));
        g.insert(NewNode::now("episode", "old", "x").with_valid_from(Millis(10)))
            .await
            .unwrap();

        // Act
        let swept = g.gc(&RetentionPolicy::keep("episode", max_age)).await;

        // Assert
        assert!(swept.is_err(), "max_age {max_age:?} must be refused");
        assert_eq!(support::count(&g, "nodes").await, 1, "max_age {max_age:?}");
    }
}

#[tokio::test]
async fn gc_sweeps_a_node_that_still_has_an_edge_pointing_at_it() {
    let (g, clock) = gc_graph().await;

    support::gc_sweeps_a_node_that_still_has_an_edge_pointing_at_it(&g, &clock).await;
}

#[tokio::test]
async fn gc_sweeps_a_node_that_still_has_a_community_assignment() {
    let (g, clock) = gc_graph().await;

    support::gc_sweeps_a_node_that_still_has_a_community_assignment(&g, &clock).await;
}

#[tokio::test]
async fn gc_leaves_an_edge_whose_endpoints_both_survive() {
    let (g, clock) = gc_graph().await;

    support::gc_leaves_an_edge_whose_endpoints_both_survive(&g, &clock).await;
}

#[tokio::test]
async fn gc_sweeps_a_node_that_has_a_vector() {
    // Arrange
    let (g, _clock) = gc_graph().await;
    g.insert(support::fact_at("embedded").with_embedding(support::embedding()))
        .await
        .unwrap();
    assert_eq!(support::count(&g, "node_vectors").await, 1);

    // Act
    let swept = g
        .gc(&RetentionPolicy::keep("fact", Millis(1)).without_reclaim())
        .await;

    // Assert: the vector went with the node
    assert_eq!(swept.unwrap().nodes_removed, 1);
    assert_eq!(support::count(&g, "nodes").await, 0);
    assert_eq!(support::count(&g, "node_vectors").await, 0);
}

#[tokio::test]
async fn gc_removes_an_edge_whose_endpoint_is_gone() {
    // Arrange
    let (g, _clock) = gc_graph().await;
    support::insert_orphan_edge(&g, "orphan").await;

    // Act
    let report = g
        .gc(&RetentionPolicy::keep("fact", Millis(1)))
        .await
        .unwrap();

    // Assert
    assert_eq!((report.nodes_removed, report.edges_removed), (0, 1));
    assert_eq!(support::count(&g, "edges").await, 0);
}
