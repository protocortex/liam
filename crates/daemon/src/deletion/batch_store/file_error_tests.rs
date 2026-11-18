// SPDX-License-Identifier: Apache-2.0
//! A batch table that cannot be read or saved says which file it was.

use super::*;

#[test]
fn a_table_path_that_is_a_directory_cannot_be_read() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("batches.json");
    std::fs::create_dir(&path).unwrap();
    let store = BatchStore::new(path.clone());

    // Act
    let error = store.load().expect_err("a directory is not a table");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "the batch table {} could not be read",
            path.display()
        )),
        "{error}"
    );
}

#[test]
fn a_table_that_cannot_be_written_says_which_file_it_was() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"").unwrap();
    let path = blocker.join("batches.json");
    let store = BatchStore::new(path.clone());

    // Act
    let error = store
        .replace(Vec::new())
        .expect_err("no file can sit under a file");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "could not save the batch table {}",
            path.display()
        )),
        "{error}"
    );
}
