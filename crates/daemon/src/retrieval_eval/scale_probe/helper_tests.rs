// SPDX-License-Identifier: Apache-2.0
//! The probe's own helpers, at a size that runs in a blink, so the ignored
//! full run does not find them broken hours in.

use super::*;

const DIMS: usize = 8;
const ROWS: usize = 5;

#[test]
fn a_generated_vector_has_the_asked_width_and_unit_length() {
    // Arrange
    let mut source = VectorSource(SEED);

    // Act
    let vector = source.unit_vector(DIMS);

    // Assert
    assert_eq!(vector.len(), DIMS);
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "{norm}");
}

#[test]
fn the_same_seed_gives_the_same_vectors_and_another_seed_does_not() {
    // Arrange
    let (mut first, mut second, mut other) = (
        VectorSource(SEED),
        VectorSource(SEED),
        VectorSource(SEED ^ 1),
    );

    // Act
    let (a, b, c) = (
        first.unit_vector(DIMS),
        second.unit_vector(DIMS),
        other.unit_vector(DIMS),
    );

    // Assert
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[tokio::test]
async fn a_filled_store_holds_the_asked_number_of_rows_with_or_without_vectors() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    for with_vectors in [true, false] {
        let database = dir.path().join(format!("probe-{with_vectors}.db"));

        // Act
        let store = filled_store(&database, DIMS, ROWS, with_vectors).await;
        let hits = store
            .query_explained(&Query::text("synthetic").with_k(ROWS * 2))
            .await
            .unwrap();

        // Assert
        assert_eq!(hits.len(), ROWS, "with_vectors = {with_vectors}");
    }
}

#[tokio::test]
async fn timing_queries_reports_a_positive_mean_and_tail() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let store = filled_store(&dir.path().join("probe.db"), DIMS, ROWS, true).await;

    // Act
    let latency = time_queries(&store, DIMS).await;

    // Assert
    assert!(latency.mean > 0.0, "{}", latency.mean);
    assert!(latency.tail > 0.0, "{}", latency.tail);
}

#[tokio::test]
async fn a_checkpoint_folds_the_log_into_the_database_file() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let database = dir.path().join("probe.db");
    let store = filled_store(&database, DIMS, ROWS, true).await;
    let before = on_disk_bytes(&database);

    // Act
    checkpoint(&database).await;
    let after = on_disk_bytes(&database);

    // Assert
    assert!(before > 0);
    assert_eq!(
        after,
        std::fs::metadata(&database).unwrap().len(),
        "nothing is left in the write-ahead log"
    );
    drop(store);
}
