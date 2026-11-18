// SPDX-License-Identifier: Apache-2.0
//! `bind` refusals that come from the file system under the socket path.

use std::os::unix::fs::PermissionsExt;

use super::*;

const READ_ONLY_DIR: u32 = 0o500;
const NO_ACCESS_DIR: u32 = 0o000;
const OWNER_ONLY_DIR: u32 = 0o700;
/// Short, under a short base: a socket path inside the system temp directory
/// can pass the address length limit on macOS.
const SCRATCH_PREFIX: &str = "liam-bind-";
const SCRATCH_BASE: &str = "/tmp";

/// A scratch directory. Its mode goes back to owner-only before it is
/// removed, so a test that narrowed it cannot leak it.
struct ScratchDir(tempfile::TempDir);

impl ScratchDir {
    fn new() -> Self {
        Self(
            tempfile::Builder::new()
                .prefix(SCRATCH_PREFIX)
                .tempdir_in(SCRATCH_BASE)
                .unwrap(),
        )
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    fn set_mode(&self, mode: u32) {
        std::fs::set_permissions(self.path(), std::fs::Permissions::from_mode(mode)).unwrap();
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ =
            std::fs::set_permissions(self.path(), std::fs::Permissions::from_mode(OWNER_ONLY_DIR));
    }
}

/// True, with a line saying so, when the test cannot show a refusal because
/// root passes every permission check.
fn skipped_as_root(test: &str) -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    let root = unsafe { libc::geteuid() == 0 };
    if root {
        eprintln!("SKIPPED {test}: running as root, which ignores directory permission bits");
    }
    root
}

async fn refusal(path: &Path) -> String {
    match bind(path).await {
        Err(error) => error.to_string(),
        Ok(_) => panic!("bind should have refused {}", path.display()),
    }
}

#[tokio::test]
async fn a_socket_directory_that_is_a_file_is_reported_with_its_path() {
    // Arrange
    let scratch = ScratchDir::new();
    let not_a_dir = scratch.path().join("file");
    std::fs::write(&not_a_dir, b"").unwrap();

    // Act
    let message = refusal(&not_a_dir.join("liamd.sock")).await;

    // Assert
    assert!(
        message.starts_with(&format!(
            "failed to create socket directory {}",
            not_a_dir.display()
        )),
        "{message}"
    );
}

#[tokio::test]
async fn a_directory_that_cannot_be_searched_is_reported_as_unreadable() {
    if skipped_as_root("a_directory_that_cannot_be_searched_is_reported_as_unreadable") {
        return;
    }
    // Arrange
    let scratch = ScratchDir::new();
    let path = scratch.path().join("liamd.sock");
    scratch.set_mode(NO_ACCESS_DIR);

    // Act
    let message = refusal(&path).await;

    // Assert
    assert!(
        message.starts_with(&format!(
            "failed to inspect existing path {}",
            path.display()
        )),
        "{message}"
    );
}

#[tokio::test]
async fn a_stale_socket_that_cannot_be_removed_is_reported_and_left_in_place() {
    if skipped_as_root("a_stale_socket_that_cannot_be_removed_is_reported_and_left_in_place") {
        return;
    }
    // Arrange
    let scratch = ScratchDir::new();
    let path = scratch.path().join("liamd.sock");
    drop(UnixListener::bind(&path).unwrap());
    scratch.set_mode(READ_ONLY_DIR);

    // Act
    let message = refusal(&path).await;

    // Assert
    assert!(
        message.starts_with(&format!("failed to remove stale socket {}", path.display())),
        "{message}"
    );
    assert!(path.exists());
}

#[tokio::test]
async fn a_parent_directory_that_cannot_be_written_is_named_as_the_thing_to_check() {
    if skipped_as_root("a_parent_directory_that_cannot_be_written_is_named_as_the_thing_to_check") {
        return;
    }
    // Arrange
    let scratch = ScratchDir::new();
    let path = scratch.path().join("liamd.sock");
    scratch.set_mode(READ_ONLY_DIR);

    // Act
    let message = refusal(&path).await;

    // Assert
    assert!(
        message.starts_with(&format!("failed to bind socket at {}", path.display())),
        "{message}"
    );
    assert!(
        message.contains("parent directory is writable"),
        "{message}"
    );
    assert!(!path.exists());
}
