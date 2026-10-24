// SPDX-License-Identifier: Apache-2.0
//! Which model a GGUF file is, by the hash of its bytes, so the response cache
//! and the failure counters follow the weights and not the file name.
//!
//! Hashing a multi-gigabyte file takes seconds, so the hash is kept in a file
//! beside the model and trusted while the file's size and modification time
//! are unchanged.

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use super::client::ModelIdentity;

const HASH_FILE_SUFFIX: &str = ".sha256.json";
const TEMP_SUFFIX: &str = ".tmp";
const READ_CHUNK_BYTES: usize = 1 << 20;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct StoredHash {
    size: u64,
    modified_millis: u64,
    sha256: String,
}

/// The identity of the model in the file at `gguf`: its file name and the
/// SHA-256 of its contents.
///
/// # Errors
///
/// Fails when the file cannot be read. A hash file that cannot be read or
/// written only costs a rehash.
pub fn model_identity(gguf: &Path) -> io::Result<ModelIdentity> {
    let meta = fs::metadata(gguf)?;
    let modified_millis = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        });
    let stamp = (meta.len(), modified_millis);
    let hash_path = hash_file_for(gguf);
    let sha256 = match read_stored(&hash_path, stamp) {
        Some(sha256) => sha256,
        None => {
            let sha256 = hash_file(gguf)?;
            store(&hash_path, stamp, &sha256);
            sha256
        }
    };
    let model_id = gguf
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    Ok(ModelIdentity {
        model_id,
        model_sha256: sha256,
    })
}

fn hash_file_for(gguf: &Path) -> PathBuf {
    let mut name = gguf.as_os_str().to_owned();
    name.push(HASH_FILE_SUFFIX);
    PathBuf::from(name)
}

fn read_stored(path: &Path, (size, modified_millis): (u64, u64)) -> Option<String> {
    let stored: StoredHash = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    (stored.size == size && stored.modified_millis == modified_millis).then_some(stored.sha256)
}

fn store(path: &Path, (size, modified_millis): (u64, u64), sha256: &str) {
    let stored = StoredHash {
        size,
        modified_millis,
        sha256: sha256.to_owned(),
    };
    let mut temp = path.as_os_str().to_owned();
    temp.push(TEMP_SUFFIX);
    let temp = PathBuf::from(temp);
    let written = serde_json::to_vec(&stored)
        .map_err(io::Error::other)
        .and_then(|bytes| fs::write(&temp, bytes))
        .and_then(|()| fs::rename(&temp, path));
    match written {
        Ok(()) => debug!("saved the model hash beside the model file"),
        Err(error) => {
            warn!(kind = ?error.kind(), "could not save the model hash, it is recomputed next time")
        }
    }
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut chunk = vec![0; READ_CHUNK_BYTES];
    loop {
        let read = file.read(&mut chunk)?;
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

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    /// SHA-256 of the three bytes "abc".
    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn model_in(dir: &TempDir, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join("model.gguf");
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn the_identity_is_the_file_name_and_the_hash_of_its_bytes() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = model_in(&dir, b"abc");

        // Act
        let identity = model_identity(&path).unwrap();

        // Assert
        assert_eq!(identity.model_id, "model.gguf");
        assert_eq!(identity.model_sha256, ABC_SHA256);
    }

    #[test]
    fn a_second_call_trusts_the_hash_file_while_size_and_time_hold() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = model_in(&dir, b"abc");
        model_identity(&path).unwrap();
        let hash_file = hash_file_for(&path);
        let mut stored: StoredHash =
            serde_json::from_slice(&fs::read(&hash_file).unwrap()).unwrap();
        stored.sha256 = "planted".into();
        fs::write(&hash_file, serde_json::to_vec(&stored).unwrap()).unwrap();

        // Act
        let identity = model_identity(&path).unwrap();

        // Assert
        assert_eq!(identity.model_sha256, "planted");
    }

    #[test]
    fn a_changed_file_is_hashed_again() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = model_in(&dir, b"abc");
        model_identity(&path).unwrap();
        fs::write(&path, b"abcd").unwrap();

        // Act
        let identity = model_identity(&path).unwrap();

        // Assert
        assert_ne!(identity.model_sha256, ABC_SHA256);
    }

    #[test]
    fn a_missing_file_is_an_error() {
        // Arrange
        let dir = TempDir::new().unwrap();

        // Act
        let result = model_identity(&dir.path().join("absent.gguf"));

        // Assert
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn a_hash_file_that_is_not_json_costs_a_rehash_only() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = model_in(&dir, b"abc");
        fs::write(hash_file_for(&path), b"not json").unwrap();

        // Act
        let identity = model_identity(&path).unwrap();

        // Assert
        assert_eq!(identity.model_sha256, ABC_SHA256);
    }
}
