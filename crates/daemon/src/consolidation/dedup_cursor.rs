// SPDX-License-Identifier: Apache-2.0
//! Where the last committed near-duplicate pass stopped. A pass scans a bounded
//! number of nodes, so a larger store is covered over successive runs, each
//! starting where the last committed one ended. A run that rolls back never
//! saves its position, so its nodes are scanned again.

use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use serde::{Deserialize, Serialize};

use super::dedup::ScanPosition;

/// Appended to the database path to name the cursor file.
const CURSOR_SUFFIX: &str = ".consolidation.dedup.json";

/// Layout version of the file, written as `version`. Any other is refused.
const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Sidecar {
    version: u32,
    #[serde(flatten)]
    position: ScanPosition,
}

/// Where the cursor for the database at `database_path` lives.
pub fn cursor_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(CURSOR_SUFFIX);
    PathBuf::from(path)
}

/// Writes `position` to `path`, replacing any earlier one.
pub fn save(path: &Path, position: ScanPosition) -> anyhow::Result<()> {
    let sidecar = Sidecar {
        version: FORMAT_VERSION,
        position,
    };
    let bytes = serde_json::to_vec(&sidecar)?;
    write_atomically(path, &bytes, &mut OsFileOps).map_err(|source| {
        anyhow::anyhow!(
            "could not save the dedup cursor at {}: {source}",
            path.display()
        )
    })
}

/// Reads the cursor at `path`, or `None` when no pass has committed yet. A file
/// that cannot be read or parsed is an error: starting over would rescan, but
/// guessing a position could skip nodes for good.
pub fn load(path: &Path) -> anyhow::Result<Option<ScanPosition>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => anyhow::bail!(
            "the dedup cursor {} could not be read: {source}",
            path.display()
        ),
    };
    let sidecar: Sidecar = serde_json::from_str(&text).map_err(|source| {
        anyhow::anyhow!(
            "the dedup cursor {} could not be parsed: {source}",
            path.display()
        )
    })?;
    anyhow::ensure!(
        sidecar.version == FORMAT_VERSION,
        "the dedup cursor {} has version {}, but this build reads version {FORMAT_VERSION}",
        path.display(),
        sidecar.version
    );
    Ok(Some(sidecar.position))
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_store::Millis;

    #[test]
    fn a_saved_cursor_loads_back_equal() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = cursor_path_for(&dir.path().join("liam.db"));
        let position = ScanPosition {
            tx_from: Millis(42),
            rowid: 7,
        };

        // Act
        save(&path, position).unwrap();

        // Assert
        assert_eq!(load(&path).unwrap(), Some(position));
    }

    #[test]
    fn no_file_means_no_pass_has_committed() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = cursor_path_for(&dir.path().join("liam.db"));

        // Act and Assert
        assert_eq!(load(&path).unwrap(), None);
    }

    #[test]
    fn a_corrupt_cursor_is_an_error_not_a_fresh_start() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = cursor_path_for(&dir.path().join("liam.db"));
        std::fs::write(&path, b"{not json").unwrap();

        // Act and Assert
        assert!(load(&path).is_err());
    }
}
