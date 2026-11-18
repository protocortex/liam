// SPDX-License-Identifier: Apache-2.0
//! The saved alert state: a file that cannot be trusted is dropped for a
//! fresh state and counted, and one that cannot be written is an error.

use super::*;

const OTHER_VERSION: u32 = FORMAT_VERSION + 1;

#[test]
fn a_state_path_that_cannot_be_read_gives_a_fresh_state_and_one_warning() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");
    std::fs::create_dir(&path).unwrap();

    // Act
    let loaded = AlertState::load(&path);

    // Assert
    assert_eq!(loaded.warnings, 1);
    assert_eq!(loaded.state.version, FORMAT_VERSION);
    assert!(loaded.state.values.is_empty());
}

#[test]
fn a_state_of_another_version_gives_a_fresh_state_and_one_warning() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");
    std::fs::write(
        &path,
        format!(r#"{{"version":{OTHER_VERSION},"values":{{}}}}"#),
    )
    .unwrap();

    // Act
    let loaded = AlertState::load(&path);

    // Assert
    assert_eq!(loaded.warnings, 1);
    assert_eq!(loaded.state.version, FORMAT_VERSION);
}

#[test]
fn a_state_that_cannot_be_saved_names_its_path() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"").unwrap();
    let path = blocker.join("state.json");

    // Act
    let error = AlertState::default()
        .save(&path)
        .expect_err("no file can sit under a file");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "could not save the metrics state {}",
            path.display()
        )),
        "{error}"
    );
}
