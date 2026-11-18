// SPDX-License-Identifier: Apache-2.0
//! The commands that attach a log without serving: a log that belongs to
//! another database is refused with both paths named.

use std::path::PathBuf;

use super::test_support::{
    config_in, database_in, database_whose_log_was_deleted, log_dir_in, standalone_log,
};
use super::*;

/// A database written through a log that is gone, and a log of another
/// database in its place.
async fn database_beside_a_foreign_log() -> (tempfile::TempDir, tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    database_whose_log_was_deleted(dir.path()).await;
    let other_dir = tempfile::tempdir().unwrap();
    let foreign_log = log_dir_in(other_dir.path());
    drop(standalone_log(&foreign_log));
    (dir, other_dir, foreign_log)
}

#[tokio::test]
async fn a_delete_over_a_foreign_log_names_the_log_and_the_database() {
    // Arrange
    let (dir, _other, foreign_log) = database_beside_a_foreign_log().await;
    let database = database_in(dir.path());

    // Act
    let result = open_store_for_delete(&config_in(dir.path()), &database, &foreign_log).await;

    // Assert
    let error = result.map(|_| ()).expect_err("a foreign log is refused");
    assert_eq!(
        error.to_string(),
        format!(
            "the event log in {} cannot be used with the database {}",
            foreign_log.display(),
            database.display()
        )
    );
}

#[tokio::test]
async fn migrating_over_a_foreign_log_names_the_log_and_the_database() {
    // Arrange
    let (dir, _other, foreign_log) = database_beside_a_foreign_log().await;
    let database = database_in(dir.path());

    // Act
    let result = migrate_unlogged_rows(&config_in(dir.path()), &database, &foreign_log).await;

    // Assert
    let error = result.expect_err("a foreign log is refused");
    assert_eq!(
        error.to_string(),
        format!(
            "the event log in {} cannot be used with the database {}; the cause is below",
            foreign_log.display(),
            database.display()
        )
    );
}
