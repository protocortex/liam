// SPDX-License-Identifier: Apache-2.0
//! The summary cache as a file beside the database, so a snapshot restore or a
//! rebuild neither reverts nor deletes it.
//!
//! It is a file of its own and not the extraction cache's: two stores over one
//! file would each rewrite it from their own copy when they trimmed it.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::generate::{SummaryCache, SUMMARY_PROMPT_VERSION};
use crate::extraction::cache::{CacheStore, NewEntry};
use crate::extraction::client::ModelIdentity;

const FILE_SUFFIX: &str = ".summary-cache.jsonl";

/// Where the summary cache for the database at `database_path` lives.
pub fn summary_cache_path(database_path: &Path) -> PathBuf {
    let mut name = database_path.as_os_str().to_owned();
    name.push(FILE_SUFFIX);
    PathBuf::from(name)
}

/// Generated text kept in a [`CacheStore`], stamped with the model that wrote it.
pub struct FileSummaryCache {
    store: Arc<CacheStore>,
    identity: ModelIdentity,
}

impl FileSummaryCache {
    /// # Errors
    ///
    /// Fails when the file exists and cannot be read.
    pub fn open(path: PathBuf, identity: ModelIdentity) -> io::Result<Self> {
        Ok(Self {
            store: Arc::new(CacheStore::open(path)?),
            identity,
        })
    }
}

impl SummaryCache for FileSummaryCache {
    fn get(&self, key: &str) -> Option<String> {
        self.store.get(key)
    }

    fn put(&self, key: &str, text: &str, source_event_ids: &[String]) -> io::Result<()> {
        self.store.put(NewEntry {
            key,
            response: text,
            model_id: &self.identity.model_id,
            model_sha256: &self.identity.model_sha256,
            prompt_version: SUMMARY_PROMPT_VERSION,
            source_event_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> ModelIdentity {
        ModelIdentity {
            model_id: "model".to_string(),
            model_sha256: String::new(),
        }
    }

    #[test]
    fn the_path_sits_beside_the_database_and_apart_from_the_extraction_cache() {
        // Act
        let path = summary_cache_path(Path::new("/data/liam.db"));

        // Assert
        assert_eq!(path, PathBuf::from("/data/liam.db.summary-cache.jsonl"));
        assert_ne!(
            path,
            crate::extraction::cache::cache_path(Path::new("/data/liam.db"))
        );
    }

    #[test]
    fn a_stored_text_is_served_again_after_the_file_is_reopened() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = summary_cache_path(&dir.path().join("liam.db"));
        FileSummaryCache::open(path.clone(), identity())
            .unwrap()
            .put("key", "the text", &["event-1".to_string()])
            .unwrap();

        // Act
        let reopened = FileSummaryCache::open(path, identity()).unwrap();

        // Assert
        assert_eq!(reopened.get("key").as_deref(), Some("the text"));
        assert_eq!(reopened.get("other"), None);
    }
}
