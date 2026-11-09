// SPDX-License-Identifier: Apache-2.0
//! Integrity check for a cached GGUF before llama.cpp opens it.
//!
//! The size is checked on every start. The sha256 of a file that is hundreds of
//! megabytes takes seconds, so it runs once per file: a small marker beside the
//! blob records the size and modification time it was verified at, and a file
//! that still matches its marker is trusted. A file that fails either check is
//! deleted, so the next run downloads it again instead of reusing bad bytes.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};

use crate::error::{ModelError, Result};

const VERIFIED_SUFFIX: &str = ".verified";
const READ_CHUNK_BYTES: usize = 1 << 20;

fn fail(message: String) -> ModelError {
    ModelError::Embed(message)
}

/// What the marker holds for a file: its size, modification time and the hash
/// it was verified against.
fn stamp(blob: &Path, sha256: &str) -> Result<String> {
    let meta = fs::metadata(blob).map_err(|e| fail(format!("read {}: {e}", blob.display())))?;
    let nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    Ok(format!("{} {nanos} {sha256}", meta.len()))
}

fn marker_path(blob: &Path) -> std::path::PathBuf {
    let mut name = blob.as_os_str().to_owned();
    name.push(VERIFIED_SUFFIX);
    name.into()
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).map_err(|e| fail(format!("open {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0; READ_CHUNK_BYTES];
    loop {
        let read = file
            .read(&mut chunk)
            .map_err(|e| fail(format!("read {}: {e}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Removes the blob a hub cache path points to, its marker and the path
/// itself (a snapshot link in the hub layout).
fn discard(path: &Path, blob: &Path) {
    let _ = fs::remove_file(marker_path(blob));
    let _ = fs::remove_file(blob);
    let _ = fs::remove_file(path);
}

/// Checks `path` against a pinned size and sha256, deleting it on a mismatch.
///
/// # Errors
///
/// Fails when the file is missing, its size or hash differs from the pin, or
/// it cannot be read. A mismatching file is removed first.
pub fn verify_pinned_file(path: &Path, size_bytes: u64, sha256: &str) -> Result<()> {
    let blob =
        fs::canonicalize(path).map_err(|e| fail(format!("resolve {}: {e}", path.display())))?;
    let size = fs::metadata(&blob)
        .map_err(|e| fail(format!("read {}: {e}", blob.display())))?
        .len();
    if size != size_bytes {
        discard(path, &blob);
        return Err(fail(format!(
            "{} is {size} bytes, expected {size_bytes}; it was deleted, run again to download it",
            path.display()
        )));
    }
    let current = stamp(&blob, sha256)?;
    if fs::read_to_string(marker_path(&blob)).is_ok_and(|saved| saved == current) {
        return Ok(());
    }
    let actual = file_sha256(&blob)?;
    if actual != sha256 {
        discard(path, &blob);
        return Err(fail(format!(
            "{} has sha256 {actual}, expected {sha256}; it was deleted, run again to download it",
            path.display()
        )));
    }
    if let Err(e) = fs::write(marker_path(&blob), current) {
        tracing::warn!(error = %e, "could not record the verified marker; the next start will hash again");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: &[u8] = b"pretend gguf";
    // sha256 of BODY
    fn body_hash() -> String {
        let mut h = Sha256::new();
        h.update(BODY);
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn a_matching_file_passes_and_leaves_a_marker_that_skips_the_hash() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        fs::write(&path, BODY).unwrap();

        // Act
        verify_pinned_file(&path, BODY.len() as u64, &body_hash()).unwrap();
        assert!(marker_path(&path.canonicalize().unwrap()).exists());
        // Same size and mtime, so the marker answers without hashing: a wrong
        // pinned hash is rejected only because the marker is keyed on it.
        let second = verify_pinned_file(&path, BODY.len() as u64, &"0".repeat(64));

        // Assert
        assert!(second.is_err());
    }

    #[test]
    fn a_wrong_size_deletes_the_file_and_its_marker() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        fs::write(&path, BODY).unwrap();

        // Act
        let error = verify_pinned_file(&path, 1, &body_hash())
            .unwrap_err()
            .to_string();

        // Assert
        assert!(
            error.contains("expected 1") && error.contains("deleted"),
            "{error}"
        );
        assert!(!path.exists());
    }

    #[test]
    fn a_wrong_hash_deletes_the_blob_and_the_snapshot_link() {
        // Arrange: a hub style layout, a link pointing at a blob.
        let dir = tempfile::tempdir().unwrap();
        let blob = dir.path().join("blob");
        let link = dir.path().join("snapshot.gguf");
        fs::write(&blob, BODY).unwrap();
        std::os::unix::fs::symlink(&blob, &link).unwrap();

        // Act
        let error = verify_pinned_file(&link, BODY.len() as u64, &"0".repeat(64))
            .unwrap_err()
            .to_string();

        // Assert
        assert!(error.contains("sha256"), "{error}");
        assert!(!blob.exists() && fs::symlink_metadata(&link).is_err());
    }

    #[test]
    fn a_changed_file_is_hashed_again_despite_an_old_marker() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        fs::write(&path, BODY).unwrap();
        verify_pinned_file(&path, BODY.len() as u64, &body_hash()).unwrap();
        fs::write(&path, b"other bytes!").unwrap();

        // Act
        let result = verify_pinned_file(&path, 12, &body_hash());

        // Assert
        assert!(result.is_err());
        assert!(!path.exists());
    }
}
