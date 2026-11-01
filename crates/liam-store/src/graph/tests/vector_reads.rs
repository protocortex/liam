// SPDX-License-Identifier: Apache-2.0
//! Reading a stored vector back, and leaving inferred nodes out of a vector
//! search. The libSQL column keeps the bytes it was given, so a round trip is
//! compared bit for bit.

use super::*;
use crate::types::Derivation;

const DIMS: usize = 8;
const AS_OF: Millis = Millis(1000);

fn vector(seed: f32) -> Vec<f32> {
    (0..DIMS).map(|i| seed + i as f32 * 0.1).collect()
}

/// A store in a file of its own, as the daemon opens one.
async fn open(dir: &TempDir) -> DefaultGraph {
    let path = dir.path().join("vectors.db");
    let clock = Arc::new(FixedClock::new(AS_OF));
    DefaultGraph::open_with_clock(path.to_str().unwrap(), GraphConfig::new(DIMS), clock)
        .await
        .unwrap()
}

fn embedded(derivation: Derivation, label: &str, embedding: Vec<f32>) -> NewNode {
    let node = match derivation {
        Derivation::Inferred => NewNode::entity("person", label),
        _ => NewNode::now("fact", label, "text"),
    };
    node.with_embedding(embedding)
}

#[tokio::test]
async fn a_stored_vector_reads_back_bit_for_bit() {
    // Arrange: values a text round trip would change
    let dir = TempDir::new().unwrap();
    let g = open(&dir).await;
    let mut stored = vector(0.1);
    stored[0] = f32::MIN_POSITIVE;
    stored[1] = -0.0;
    stored[2] = 1e-30;
    stored[3] = f32::MAX;
    let id = g
        .insert(embedded(Derivation::Recorded, "n", stored.clone()))
        .await
        .unwrap();

    // Act
    let read = g.backend.vector_get(id.as_str(), DIMS).await.unwrap();

    // Assert
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&read.expect("a vector")), bits(&stored));
}

#[tokio::test]
async fn an_unknown_or_deleted_node_has_no_vector_and_a_closed_one_keeps_it() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let g = open(&dir).await;
    let stored = vector(0.5);
    let closed = g
        .insert(embedded(Derivation::Recorded, "closed", stored.clone()))
        .await
        .unwrap();
    g.supersede(&closed, NewNode::now("fact", "newer", "text"))
        .await
        .unwrap();
    let deleted = g
        .insert(embedded(Derivation::Recorded, "deleted", vector(0.9)))
        .await
        .unwrap();
    for sql in [
        "DELETE FROM node_vectors WHERE node_id = ?1",
        "DELETE FROM nodes WHERE id = ?1",
    ] {
        g.backend
            .execute(sql, &[deleted.as_str().into()])
            .await
            .unwrap();
    }

    // Act
    let unknown = g.backend.vector_get("no-such-node", DIMS).await.unwrap();
    let after_delete = g.backend.vector_get(deleted.as_str(), DIMS).await.unwrap();
    let after_close = g.backend.vector_get(closed.as_str(), DIMS).await.unwrap();

    // Assert
    assert_eq!(unknown, None);
    assert_eq!(after_delete, None);
    assert_eq!(after_close, Some(stored));
}

#[tokio::test]
async fn a_stored_vector_of_another_width_is_a_typed_error() {
    // Arrange: the column accepts any width, so write three floats by hand
    let dir = TempDir::new().unwrap();
    let g = open(&dir).await;
    let id = g.insert(NewNode::now("fact", "n", "text")).await.unwrap();
    let three_floats: Vec<u8> = [1.0f32, 2.0, 3.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    g.backend
        .execute(
            "INSERT INTO node_vectors (node_id, embedding) VALUES (?1, ?2)",
            &[id.as_str().into(), three_floats.into()],
        )
        .await
        .unwrap();

    // Act
    let read = g.backend.vector_get(id.as_str(), DIMS).await;

    // Assert
    assert!(matches!(
        read,
        Err(Error::Dimension {
            expected: DIMS,
            got: 3
        })
    ));
}

#[tokio::test]
async fn a_vector_search_leaves_inferred_nodes_out_unless_asked() {
    // Arrange: a page and a fact equally near the query
    let dir = TempDir::new().unwrap();
    let g = open(&dir).await;
    let query = vector(0.3);
    let recorded = g
        .insert(embedded(Derivation::Recorded, "fact", query.clone()))
        .await
        .unwrap();
    let inferred = g
        .insert(embedded(Derivation::Inferred, "Page", query.clone()))
        .await
        .unwrap();

    // Act
    let strict = g
        .backend
        .vector_search(&query, 5, None, None, AS_OF, false)
        .await
        .unwrap();
    let all = g
        .backend
        .vector_search(&query, 5, None, None, AS_OF, true)
        .await
        .unwrap();

    // Assert
    assert_eq!(strict, vec![recorded.clone()]);
    let all: HashSet<NodeId> = all.into_iter().collect();
    assert_eq!(all, HashSet::from([recorded, inferred]));
}

#[tokio::test]
async fn a_node_removed_by_gc_with_reclaim_has_no_vector() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("gc.db");
    let clock = Arc::new(FixedClock::new(AS_OF));
    let g = DefaultGraph::open_with_clock(
        path.to_str().unwrap(),
        GraphConfig::new(DIMS),
        clock.clone(),
    )
    .await
    .unwrap();
    let id = g
        .insert(embedded(Derivation::Recorded, "old", vector(0.2)))
        .await
        .unwrap();
    assert!(g
        .backend
        .vector_get(id.as_str(), DIMS)
        .await
        .unwrap()
        .is_some());
    clock.set(Millis(AS_OF.0 + 5000));

    // Act
    let report = g
        .gc(&RetentionPolicy::keep("fact", Millis(1)))
        .await
        .unwrap();
    let read = g.backend.vector_get(id.as_str(), DIMS).await;

    // Assert
    assert_eq!(report.nodes_removed, 1);
    assert_eq!(read.unwrap(), None);
}

#[tokio::test]
async fn a_scoped_vector_search_still_leaves_inferred_nodes_out() {
    // Arrange: the scope clause comes first, so the derivation parameter must
    // be numbered after it
    let dir = TempDir::new().unwrap();
    let g = open(&dir).await;
    let query = vector(0.3);
    let recorded = g
        .insert(embedded(Derivation::Recorded, "fact", query.clone()).with_scope("work/liam"))
        .await
        .unwrap();
    let inferred = g
        .insert(embedded(Derivation::Inferred, "Page", query.clone()).with_scope("work/liam"))
        .await
        .unwrap();
    let elsewhere = g
        .insert(embedded(Derivation::Recorded, "other", query.clone()).with_scope("personal"))
        .await
        .unwrap();

    // Act
    let strict = g
        .backend
        .vector_search(&query, 5, None, Some("work"), AS_OF, false)
        .await
        .unwrap();
    let strict_kind = g
        .backend
        .vector_search(&query, 5, Some("fact"), Some("work"), AS_OF, false)
        .await
        .unwrap();
    let all = g
        .backend
        .vector_search(&query, 5, None, Some("work"), AS_OF, true)
        .await
        .unwrap();

    // Assert
    assert_eq!(strict, vec![recorded.clone()]);
    assert_eq!(strict_kind, vec![recorded.clone()]);
    let all: HashSet<NodeId> = all.into_iter().collect();
    assert_eq!(all, HashSet::from([recorded, inferred]));
    assert!(!all.contains(&elsewhere));
}
