// SPDX-License-Identifier: Apache-2.0
//! What a refused or failed fact write says.

use super::*;

#[test]
fn a_refusal_and_a_store_failure_read_differently() {
    // Arrange
    let rejected = ApplyError::Rejected("a confidence above one".into());
    let failed = ApplyError::Store(Error::Backend("disk full".into()));

    // Act
    let (rejected_text, failed_text) = (rejected.to_string(), failed.to_string());

    // Assert
    assert!(
        rejected_text.contains("a confidence above one"),
        "{rejected_text}"
    );
    assert!(failed_text.contains("disk full"), "{failed_text}");
    assert_ne!(rejected_text, failed_text);
}

#[test]
fn a_fact_without_a_source_is_rejected_and_a_backend_failure_is_a_store_failure() {
    // Act
    let rejected = ApplyError::from(Error::FactWithoutSource);
    let failed = ApplyError::from(Error::Backend("disk full".into()));

    // Assert
    assert!(matches!(rejected, ApplyError::Rejected(_)), "{rejected:?}");
    assert!(matches!(failed, ApplyError::Store(_)), "{failed:?}");
}
