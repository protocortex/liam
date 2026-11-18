// SPDX-License-Identifier: Apache-2.0
//! `bind` refusals that come from the file system under the socket path.

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

const READ_ONLY_DIR: u32 = 0o500;
const NO_ACCESS_DIR: u32 = 0o000;
const OWNER_ONLY_DIR: u32 = 0o700;

/// A short directory under `/tmp`: a socket path inside the system temp
/// directory can pass the address length limit on macOS. Its mode goes back to
/// owner-only before it is removed, so a test that narrowed it cannot leak it.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(format!("/tmp/liam-bind-{}-{unique}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }

    fn set_mode(&self, mode: u32) {
        std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(mode)).unwrap();
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(OWNER_ONLY_DIR));
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn running_as_root() -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() == 0 }
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
    let not_a_dir = scratch.0.join("file");
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
    if running_as_root() {
        return;
    }
    // Arrange
    let scratch = ScratchDir::new();
    let path = scratch.0.join("liamd.sock");
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
    if running_as_root() {
        return;
    }
    // Arrange
    let scratch = ScratchDir::new();
    let path = scratch.0.join("liamd.sock");
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
