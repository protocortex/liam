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
    assert_eq!(
        rejected_text,
        "the store refused the facts: a confidence above one"
    );
    assert_eq!(failed_text, "the store failed: backend: disk full");
}
