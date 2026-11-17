// SPDX-License-Identifier: Apache-2.0
//! The sizes the store reports about its tiers.

use super::log_write::logged_graph;
use super::quant::{node, open_at, DIMS};
use super::tiers::{cold_at, drop_embeddings, lineage_of_node};
use super::*;
use crate::DefaultBackend;

const F32_BYTES: u64 = 4;
/// One binary word per 64 dimensions, eight bytes each.
const SIGNATURE_BYTES: u64 = 8;

#[tokio::test]
async fn storage_bytes_splits_the_embeddings_and_signatures_by_the_lineage_tier() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    g.insert(node("hot", 0b1111_0000)).await.unwrap();
    let cold = g.insert(node("cold", 0b0000_1111)).await.unwrap();
    let lineage = lineage_of_node(&g, &cold).await;
    g.set_tier(&cold_at(&lineage, 1), None).await.unwrap();
    drop_embeddings(&g, &cold).await;

    // Act
    let bytes = g.storage_bytes().await.unwrap();

    // Assert
    let vector = DIMS as u64 * F32_BYTES;
    let int8 = DIMS as u64;
    assert_eq!(bytes.hot, vector + SIGNATURE_BYTES + int8);
    assert_eq!(bytes.cold, SIGNATURE_BYTES);
}

#[tokio::test]
async fn an_empty_store_holds_no_bytes() {
    // Arrange
    let g = open_at(Millis(1000)).await;

    // Act
    let bytes = g.storage_bytes().await.unwrap();

    // Assert
    assert_eq!(bytes, TierBytes::default());
}

#[tokio::test]
async fn the_filter_counts_a_new_hash_as_a_true_negative_and_a_repeat_as_neither_miss() {
    // Arrange
    let (g, _, _) = logged_graph::<DefaultBackend>(Millis(1000)).await;
    g.insert(NewNode::now("fact", "same", "same"))
        .await
        .unwrap();

    // Act
    g.insert(NewNode::now("fact", "same", "same"))
        .await
        .unwrap();

    // Assert
    let stats = g.bloom_stats();
    assert_eq!((stats.confirmed_false, stats.true_negatives), (0, 1));
}

#[tokio::test]
async fn a_timed_query_times_only_the_arms_it_ran() {
    // Arrange
    let g = open_at(Millis(1000)).await;
    g.insert(node("one", 0b1111_0000)).await.unwrap();

    // Act
    let (hits, timings) = g.query_timed(&Query::text("content")).await.unwrap();

    // Assert
    assert_eq!(hits.len(), 1);
    assert!(timings.lexical.is_some());
    assert_eq!((timings.vector, timings.cold), (None, None));
}
