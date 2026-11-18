// SPDX-License-Identifier: Apache-2.0
//! The similarity measure and the error text of a dedup pass.

use super::*;

const THRESHOLD: f32 = 1.5;

#[test]
fn vectors_of_different_widths_are_not_similar_at_all() {
    // Act
    let similarity = cosine(&[1.0, 0.0], &[1.0, 0.0, 0.0]);

    // Assert
    assert_eq!(similarity, 0.0);
}

#[test]
fn a_zero_vector_is_not_similar_to_anything() {
    // Act
    let similarity = cosine(&[0.0, 0.0], &[1.0, 1.0]);

    // Assert
    assert_eq!(similarity, 0.0);
}

#[test]
fn parallel_vectors_are_fully_similar_whatever_their_length() {
    // Act
    let similarity = cosine(&[1.0, 2.0], &[2.0, 4.0]);

    // Assert
    assert!((similarity - 1.0).abs() < 1e-6, "{similarity}");
}

#[test]
fn each_dedup_error_carries_what_was_wrong() {
    // Arrange
    let threshold = DedupError::InvalidThreshold(THRESHOLD);
    let store = DedupError::from(liam_store::Error::Backend("disk full".into()));

    // Act
    let (threshold_text, store_text) = (threshold.to_string(), store.to_string());

    // Assert
    assert!(threshold_text.contains("1.5"), "{threshold_text}");
    assert!(store_text.contains("disk full"), "{store_text}");
    assert!(matches!(store, DedupError::Store(_)));
}
