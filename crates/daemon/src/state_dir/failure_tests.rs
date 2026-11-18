// SPDX-License-Identifier: Apache-2.0
//! Directories and files the check cannot look at: the reason says why, and
//! nothing is changed on a guess.

use std::os::unix::fs::PermissionsExt;

use super::*;

const NO_ACCESS: u32 = 0o000;
const OWNER_ONLY: u32 = 0o700;

fn running_as_root() -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

/// A directory under a state root, set to `mode` and restored on drop so the
/// temporary directory can be removed.
struct Narrowed(PathBuf);

impl Narrowed {
    fn new(path: &Path, mode: u32) -> Self {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
        Self(path.to_path_buf())
    }
}

impl Drop for Narrowed {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(OWNER_ONLY));
    }
}

#[test]
fn each_skip_says_why_the_directory_was_left_alone() {
    // Act / Assert
    assert_eq!(Skip::NotAbsolute.reason(), "the path is not absolute");
    assert_eq!(
        Skip::EscapesRoot.reason(),
        "the path climbs above the filesystem root"
    );
    assert!(Skip::Symlink.reason().contains("symbolic link"));
    assert!(Skip::Symlink.reason().contains("chmod 700"));
}

#[test]
fn a_step_that_failed_keeps_the_error_kind_and_names_the_step() {
    // Arrange
    let cause = std::io::Error::from(std::io::ErrorKind::PermissionDenied);

    // Act
    let described = failed("check for symbolic links in", cause);

    // Assert
    assert_eq!(described.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(
        described
            .to_string()
            .starts_with("could not check for symbolic links in the path: "),
        "{described}"
    );
}

#[test]
fn a_path_that_cannot_be_searched_is_an_error_not_a_pass() {
    if running_as_root() {
        return;
    }
    // Arrange
    let root = tempfile::tempdir().unwrap();
    let outer = root.path().join("outer");
    std::fs::create_dir_all(outer.join("inner")).unwrap();
    let _closed = Narrowed::new(&outer, NO_ACCESS);

    // Act
    let result = secure_dir(&outer.join("inner"), root.path());

    // Assert
    let error = result.expect_err("a directory that cannot be searched");
    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(
        error
            .to_string()
            .starts_with("could not check for symbolic links in the path"),
        "{error}"
    );
}

#[test]
fn a_directory_that_cannot_be_listed_is_an_error() {
    if running_as_root() {
        return;
    }
    // Arrange
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("log");
    std::fs::create_dir(&dir).unwrap();
    let _closed = Narrowed::new(&dir, NO_ACCESS);

    // Act
    let result = secure_files_in(&dir, root.path(), |_| true);

    // Assert
    let error = result.expect_err("a directory that cannot be listed");
    assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
}
