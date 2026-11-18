// SPDX-License-Identifier: Apache-2.0
//! A record or a database that cannot be read is an error, never "absent":
//! treating it as absent would hide a model change.

use super::*;

fn mock_fingerprint() -> EmbedderFingerprint {
    EmbedderFingerprint::from_config(&Config::default())
}

#[test]
fn a_record_that_cannot_be_read_is_an_error_naming_it() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let database = dir.path().join("liam.db");
    let record = fingerprint_path_for(&database);
    std::fs::create_dir(&record).unwrap();

    // Act
    let error = verify(&database, &mock_fingerprint()).expect_err("an unreadable record");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "the embedder fingerprint {} could not be read",
            record.display()
        )),
        "{error}"
    );
}

#[test]
fn a_record_that_cannot_be_written_is_an_error_naming_it() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let database = dir.path().join("liam.db");
    let record = fingerprint_path_for(&database);
    std::fs::create_dir(&record).unwrap();

    // Act
    let error =
        super::record(&database, &mock_fingerprint()).expect_err("no file can replace a directory");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "could not record the embedder fingerprint at {}",
            record.display()
        )),
        "{error}"
    );
}

#[test]
fn a_database_under_a_plain_file_cannot_be_inspected() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"").unwrap();
    let database = blocker.join("liam.db");

    // Act
    let error = predates_fingerprints(&database).expect_err("the path cannot be inspected");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "the database {} could not be inspected",
            database.display()
        )),
        "{error}"
    );
}
