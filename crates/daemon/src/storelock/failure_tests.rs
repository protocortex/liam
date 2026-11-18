// SPDX-License-Identifier: Apache-2.0
//! Why the store lock could not be taken, when the cause is the file system.

use super::*;

#[test]
fn a_database_directory_that_cannot_be_created_is_named() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"").unwrap();
    let database = blocker.join("state").join("liam.db");

    // Act
    let error = StoreLock::acquire(&database).expect_err("the directory cannot exist");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "failed to create the database directory {}",
            blocker.join("state").display()
        )),
        "{error}"
    );
}

#[test]
fn a_lock_file_that_cannot_be_opened_is_named() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let database = dir.path().join("liam.db");
    let lock_path = lock_path_for(&database);
    std::fs::create_dir(&lock_path).unwrap();

    // Act
    let error = StoreLock::acquire(&database).expect_err("a directory is not a lock file");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "failed to open store lock file {}",
            lock_path.display()
        )),
        "{error}"
    );
}
