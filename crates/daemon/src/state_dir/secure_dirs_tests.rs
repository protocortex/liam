// SPDX-License-Identifier: Apache-2.0
//! `secure_dirs` over the cases it reports: what it changes and what it leaves.

use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};

use super::*;

const WIDE: u32 = 0o755;
const OWNER_ONLY: u32 = 0o700;
const RISK: &str = "the socket can be reached";

fn set_mode(path: &Path, mode: u32) {
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}

fn mode_of(path: &Path) -> u32 {
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

#[test]
fn a_wide_directory_in_the_state_root_is_restricted_to_its_owner() {
    // Arrange
    let home = tempfile::tempdir().unwrap();
    let root = default_state_dir(home.path());
    fs::create_dir(&root).unwrap();
    set_mode(&root, WIDE);

    // Act
    secure_dirs(&[root.as_path()], &root, RISK);

    // Assert
    assert_eq!(mode_of(&root), OWNER_ONLY);
}

#[test]
fn a_wide_directory_outside_the_state_root_keeps_its_mode() {
    // Arrange
    let home = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    set_mode(elsewhere.path(), WIDE);

    // Act
    secure_dirs(&[elsewhere.path()], &default_state_dir(home.path()), RISK);

    // Assert
    assert_eq!(mode_of(elsewhere.path()), WIDE);
}

#[test]
fn a_directory_behind_a_symlink_keeps_its_mode() {
    // Arrange
    let home = tempfile::tempdir().unwrap();
    let root = default_state_dir(home.path());
    let real = home.path().join("real");
    fs::create_dir_all(&root).unwrap();
    fs::create_dir(&real).unwrap();
    set_mode(&real, WIDE);
    let link = root.join("link");
    symlink(&real, &link).unwrap();

    // Act
    secure_dirs(&[link.as_path()], &root, RISK);

    // Assert
    assert_eq!(mode_of(&real), WIDE);
}

#[test]
fn a_relative_directory_keeps_its_mode_and_the_others_are_still_secured() {
    // Arrange
    let home = tempfile::tempdir().unwrap();
    let root = default_state_dir(home.path());
    fs::create_dir(&root).unwrap();
    set_mode(&root, WIDE);
    let relative = Path::new("relative-state-dir");

    // Act
    secure_dirs(&[relative, root.as_path()], &root, RISK);

    // Assert
    assert_eq!(mode_of(&root), OWNER_ONLY);
    assert!(!relative.exists());
}

#[test]
fn a_directory_that_cannot_be_inspected_does_not_stop_the_daemon() {
    // Arrange: a file where the directory's parent should be
    let home = tempfile::tempdir().unwrap();
    let root = default_state_dir(home.path());
    fs::create_dir(&root).unwrap();
    let file = root.join("file");
    fs::write(&file, b"").unwrap();
    let under_a_file = file.join("child");

    // Act
    secure_dirs(&[under_a_file.as_path()], &root, RISK);

    // Assert
    assert!(file.is_file());
}
