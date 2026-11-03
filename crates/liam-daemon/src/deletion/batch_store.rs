// SPDX-License-Identifier: Apache-2.0
//! The table of hosted batches. It lives in a file beside the database, outside
//! the snapshot, so a rolled back consolidation leaves it as it was and a
//! restart still knows which requests a provider already holds.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use serde::{Deserialize, Serialize};

/// Appended to the database path to name the table file.
const BATCH_FILE_SUFFIX: &str = ".hosted-batches.json";
/// Layout version of the file. Any other is refused.
const FORMAT_VERSION: u32 = 1;

/// Where the batch table of the database at `database_path` lives.
pub fn batch_store_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(BATCH_FILE_SUFFIX);
    PathBuf::from(path)
}

/// Where a request is in its batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchState {
    /// The provider holds the request and has not finished it.
    Submitted,
    /// The provider finished; the result is not in the cache yet.
    Ended,
    /// The result is in the cache.
    Collected,
    /// The provider or a validation refused the result.
    Failed,
    /// The provider let the batch lapse before it ran.
    Expired,
    /// The batch was cancelled; the request is sent again when it is needed.
    Cancelled,
}

/// Why a request in the `Failed` state failed, which decides whether the
/// failure counts against the events in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryFailure {
    /// The provider answered, and the answer did not validate.
    Invalid,
    /// The provider refused the request or answered as another model, or the
    /// request expired twice. Nothing about the events is to blame.
    Refused,
}

/// The first time a request is sent.
pub const FIRST_ATTEMPT: u32 = 1;

fn first_attempt() -> u32 {
    FIRST_ATTEMPT
}

impl BatchState {
    /// A request in a live state is the provider's to answer, so it is not sent
    /// again.
    pub fn is_live(self) -> bool {
        matches!(self, Self::Submitted | Self::Ended)
    }
}

/// One request sent in a batch. The cache key is also the request id the
/// provider sees, and the entry holds no prompt text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchEntry {
    pub cache_key: String,
    /// The provider's name, as `[hosted.models]` spells it.
    pub provider: String,
    pub batch_id: String,
    pub state: BatchState,
    /// Epoch milliseconds.
    pub submitted_at: i64,
    pub input_hash: String,
    pub source_event_ids: Vec<String>,
    /// How many times the request has been sent, the first send being 1. A
    /// request that expires is sent once more.
    #[serde(default = "first_attempt")]
    pub attempts: u32,
    /// Set while the state is `Failed`.
    #[serde(default)]
    pub failure: Option<EntryFailure>,
}

impl BatchEntry {
    /// Whether the request was built from one of `event_ids`.
    pub fn cites(&self, event_ids: &HashSet<String>) -> bool {
        self.source_event_ids
            .iter()
            .any(|id| event_ids.contains(id))
    }
}

#[derive(Serialize, Deserialize)]
struct BatchFile {
    version: u32,
    entries: Vec<BatchEntry>,
}

/// The entries of one database, kept in a file that is replaced whole after
/// every change.
pub struct BatchStore {
    path: PathBuf,
}

impl BatchStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Every entry, in the order they were written. A file that does not exist
    /// holds none.
    pub fn load(&self) -> anyhow::Result<Vec<BatchEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&self.path).map_err(|source| {
            anyhow::anyhow!(
                "the batch table {} could not be read: {source}",
                self.path.display()
            )
        })?;
        let file: BatchFile = serde_json::from_str(&text).map_err(|source| {
            anyhow::anyhow!(
                "the batch table {} could not be parsed: {source}",
                self.path.display()
            )
        })?;
        anyhow::ensure!(
            file.version == FORMAT_VERSION,
            "the batch table {} has version {}, but this build reads version {FORMAT_VERSION}",
            self.path.display(),
            file.version
        );
        Ok(file.entries)
    }

    /// The entries built from one of `event_ids`.
    pub fn citing(&self, event_ids: &HashSet<String>) -> anyhow::Result<Vec<BatchEntry>> {
        Ok(self
            .load()?
            .into_iter()
            .filter(|entry| entry.cites(event_ids))
            .collect())
    }

    /// Replaces the table with `entries`.
    pub fn replace(&self, entries: Vec<BatchEntry>) -> anyhow::Result<()> {
        let bytes = serde_json::to_vec_pretty(&BatchFile {
            version: FORMAT_VERSION,
            entries,
        })?;
        write_atomically(&self.path, &bytes, &mut OsFileOps).map_err(|source| {
            anyhow::anyhow!(
                "could not save the batch table {}: {source}",
                self.path.display()
            )
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn entry(cache_key: &str, batch_id: &str, events: &[&str]) -> BatchEntry {
        BatchEntry {
            cache_key: cache_key.to_owned(),
            provider: "anthropic".to_owned(),
            batch_id: batch_id.to_owned(),
            state: BatchState::Submitted,
            submitted_at: 1,
            input_hash: "hash".to_owned(),
            source_event_ids: events.iter().map(|id| (*id).to_owned()).collect(),
            attempts: FIRST_ATTEMPT,
            failure: None,
        }
    }

    fn ids(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    #[test]
    fn a_missing_file_holds_nothing_and_a_saved_table_reads_back_in_order() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = BatchStore::new(batch_store_path_for(&dir.path().join("liam.db")));
        let entries = vec![entry("k2", "b1", &["e2"]), entry("k1", "b1", &["e1"])];

        // Act
        let empty = store.load().unwrap();
        store.replace(entries.clone()).unwrap();
        let loaded = store.load().unwrap();

        // Assert
        assert!(empty.is_empty());
        assert_eq!(loaded, entries);
    }

    #[test]
    fn citing_returns_only_the_entries_built_from_the_given_events() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches.json"));
        store
            .replace(vec![
                entry("k1", "b1", &["e1", "e2"]),
                entry("k2", "b1", &["e3"]),
            ])
            .unwrap();

        // Act
        let found = store.citing(&ids(&["e2"])).unwrap();

        // Assert
        assert_eq!(found, vec![entry("k1", "b1", &["e1", "e2"])]);
    }

    #[test]
    fn a_table_of_another_version_or_with_garbage_is_refused() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batches.json");
        let store = BatchStore::new(path.clone());

        // Act
        std::fs::write(&path, r#"{"version":9,"entries":[]}"#).unwrap();
        let version = store.load().unwrap_err().to_string();
        std::fs::write(&path, "not json").unwrap();
        let garbage = store.load().unwrap_err().to_string();

        // Assert
        assert!(version.contains("version 9"), "{version}");
        assert!(garbage.contains("could not be parsed"), "{garbage}");
    }

    #[test]
    fn only_a_submitted_or_ended_request_is_live() {
        // Arrange
        let live = [BatchState::Submitted, BatchState::Ended];
        let settled = [
            BatchState::Collected,
            BatchState::Failed,
            BatchState::Expired,
            BatchState::Cancelled,
        ];

        // Act and Assert
        assert!(live.iter().all(|state| state.is_live()));
        assert!(settled.iter().all(|state| !state.is_live()));
    }
}
