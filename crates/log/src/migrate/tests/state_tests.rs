// SPDX-License-Identifier: Apache-2.0
//! The progress marker a crashed migration leaves behind.

use super::*;

#[tokio::test]
async fn a_progress_marker_that_is_not_a_number_stops_the_run_and_leaves_the_log_alone() {
    // Arrange
    let fixture = plaintext_log().await;
    let before = raw(&fixture).await;
    let state = fixture.dir.path().join(STATE_FILE);
    fs::write(&state, b"not a sequence").unwrap();

    // Act
    let result = run(&fixture, &NoFault);

    // Assert
    assert!(
        matches!(&result, Err(MigrateError::StateCorrupt { path }) if *path == state),
        "{result:?}"
    );
    assert_eq!(raw(&fixture).await, before, "nothing was rewritten");
}

#[tokio::test]
async fn a_progress_marker_that_cannot_be_read_is_an_io_error() {
    // Arrange
    let fixture = plaintext_log().await;
    fs::create_dir(fixture.dir.path().join(STATE_FILE)).unwrap();

    // Act
    let result = run(&fixture, &NoFault);

    // Assert
    assert!(matches!(result, Err(MigrateError::Io(_))), "{result:?}");
}
