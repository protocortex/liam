// SPDX-License-Identifier: Apache-2.0
//! The probe set a run saves at its baseline and reads back later.

use super::*;

const OTHER_VERSION: u32 = FORMAT_VERSION + 1;

fn empty_set() -> ProbeSet {
    ProbeSet {
        probes: Vec::new(),
        baseline: ProbeReport::default(),
    }
}

#[test]
fn a_set_that_is_not_json_cannot_be_loaded_and_names_its_file() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("probes.json");
    std::fs::write(&path, b"not json").unwrap();

    // Act
    let error = ProbeSet::load(&path).expect_err("not a probe set");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "the consolidation probe set {} could not be parsed",
            path.display()
        )),
        "{error}"
    );
}

#[test]
fn a_set_written_by_another_version_is_refused_with_both_versions() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("probes.json");
    std::fs::write(
        &path,
        format!(r#"{{"version":{OTHER_VERSION},"probes":[],"baseline":{{"scores":{{}}}}}}"#),
    )
    .unwrap();

    // Act
    let error = ProbeSet::load(&path).expect_err("another version");

    // Assert
    assert_eq!(
        error.to_string(),
        format!(
            "the consolidation probe set {} has version {OTHER_VERSION}, but this build reads \
             version {FORMAT_VERSION}",
            path.display()
        )
    );
}

#[test]
fn a_set_that_cannot_be_saved_names_its_path() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"").unwrap();
    let path = blocker.join("probes.json");

    // Act
    let error = empty_set()
        .save(&path)
        .expect_err("no file can sit under a file");

    // Assert
    assert!(
        error.to_string().starts_with(&format!(
            "could not save the consolidation probe set at {}",
            path.display()
        )),
        "{error}"
    );
}
