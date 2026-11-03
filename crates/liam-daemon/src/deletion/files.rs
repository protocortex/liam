// SPDX-License-Identifier: Apache-2.0
//! Lineage stores that live in files beside the database: the extraction
//! response cache, the dedup filter file, the consolidation snapshots, and the
//! manifest and cooldown file that name lineages.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use liam_store::{
    Backend, Error, Graph, LineageHits, LineageStore, LineageTarget, PurgeReport, Residual, Result,
};

use liam_daemon::consolidation::manifest::RunManifest;
use liam_daemon::consolidation::tier_history::{self, TierHistory};

use crate::extraction::cache::CacheStore;

const CACHE_STORE: &str = "extraction_cache";
const CACHE_ENTRY: &str = "cache_entry";
const FILTER_STORE: &str = "dedup_filter";
const FILTER_KIND: &str = "filter";
const SNAPSHOT_STORE: &str = "snapshots";
const SNAPSHOT_FILE: &str = "snapshot_file";
const SNAPSHOT_UNREADABLE: &str = "snapshot_unreadable";
const STATE_STORE: &str = "state_files";
const MANIFEST_ENTRY: &str = "manifest_tier_moved_lineages";
const COOLDOWN_ENTRY: &str = "tier_history_cooldown";
const ZERO_BLOCK: [u8; 8192] = [0; 8192];

fn failed(store: &str, reason: impl std::fmt::Display) -> Error {
    Error::LineageStore {
        store: store.to_owned(),
        reason: reason.to_string(),
    }
}

fn residual(store: &str, kind: &str, id: impl Into<String>) -> Residual {
    Residual {
        store: store.to_owned(),
        kind: kind.to_owned(),
        id: id.into(),
    }
}

fn hits(kind: &str, count: usize) -> LineageHits {
    let mut found = LineageHits::default();
    if count > 0 {
        found.rows.insert(kind.to_owned(), count);
    }
    found
}

/// The response cache, minus the entries that cite a deleted event.
pub struct CacheStoreLineage {
    cache: Arc<CacheStore>,
}

impl CacheStoreLineage {
    pub fn new(cache: Arc<CacheStore>) -> Self {
        Self { cache }
    }

    fn citing(&self, target: &LineageTarget) -> Vec<String> {
        let events: HashSet<String> = target.event_ids.iter().cloned().collect();
        self.cache.keys_citing(&events)
    }
}

#[async_trait]
impl LineageStore for CacheStoreLineage {
    fn name(&self) -> &str {
        CACHE_STORE
    }

    async fn find(&self, target: &LineageTarget) -> Result<LineageHits> {
        Ok(hits(CACHE_ENTRY, self.citing(target).len()))
    }

    async fn purge(&self, target: &LineageTarget) -> Result<PurgeReport> {
        let events: HashSet<String> = target.event_ids.iter().cloned().collect();
        let removed = self
            .cache
            .remove_by_event_ids(&events)
            .map_err(|error| failed(CACHE_STORE, error))?;
        Ok(PurgeReport { removed })
    }

    async fn verify(&self, target: &LineageTarget) -> Result<Vec<Residual>> {
        Ok(self
            .citing(target)
            .into_iter()
            .map(|key| residual(CACHE_STORE, CACHE_ENTRY, key))
            .collect())
    }
}

/// The dedup filter, rebuilt from the database once a purge has emptied it of
/// the deleted hashes, and its file when there is one. Add it after the
/// database store.
pub struct BloomLineage<B: Backend> {
    graph: Arc<Graph<B>>,
    filter_file: Option<PathBuf>,
}

impl<B: Backend> BloomLineage<B> {
    pub fn new(graph: Arc<Graph<B>>, filter_file: Option<PathBuf>) -> Self {
        Self { graph, filter_file }
    }

    async fn is_current(&self) -> Result<bool> {
        if !self.graph.dedup_filter_matches_index().await? {
            return Ok(false);
        }
        match &self.filter_file {
            Some(path) => self.graph.dedup_filter_file_matches_index(path).await,
            None => Ok(true),
        }
    }
}

#[async_trait]
impl<B: Backend> LineageStore for BloomLineage<B> {
    fn name(&self) -> &str {
        FILTER_STORE
    }

    async fn find(&self, _target: &LineageTarget) -> Result<LineageHits> {
        Ok(hits(FILTER_KIND, usize::from(!self.is_current().await?)))
    }

    async fn purge(&self, _target: &LineageTarget) -> Result<PurgeReport> {
        if self.is_current().await? {
            return Ok(PurgeReport::default());
        }
        self.graph
            .rebuild_dedup_filter(self.filter_file.as_deref())
            .await?;
        Ok(PurgeReport { removed: 1 })
    }

    async fn verify(&self, _target: &LineageTarget) -> Result<Vec<Residual>> {
        if self.is_current().await? {
            return Ok(Vec::new());
        }
        Ok(vec![residual(FILTER_STORE, FILTER_KIND, FILTER_KIND)])
    }
}

/// Snapshots of the database. Every one predates a delete, which runs with the
/// daemon stopped, so a delete zeroes and unlinks them all.
pub struct SnapshotLineage {
    dir: PathBuf,
}

impl SnapshotLineage {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn files(&self) -> Result<Vec<PathBuf>> {
        let mut found = Vec::new();
        collect_files(&self.dir, &mut found).map_err(|error| failed(SNAPSHOT_STORE, error))?;
        found.sort();
        Ok(found)
    }

    fn relative(&self, file: &Path) -> String {
        file.strip_prefix(&self.dir)
            .unwrap_or(file)
            .display()
            .to_string()
    }
}

fn collect_files(dir: &Path, found: &mut Vec<PathBuf>) -> io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, found)?;
        } else {
            found.push(path);
        }
    }
    Ok(())
}

/// Overwrites the file with zeros, syncs it, and unlinks it, so its bytes are
/// not left in a block the file system hands back.
fn zero_and_unlink(path: &Path) -> io::Result<()> {
    let length = fs::metadata(path)?.len();
    let mut file = OpenOptions::new().write(true).open(path)?;
    let mut left = length;
    while left > 0 {
        let block = left.min(ZERO_BLOCK.len() as u64) as usize;
        file.write_all(&ZERO_BLOCK[..block])?;
        left -= block as u64;
    }
    file.sync_all()?;
    fs::remove_file(path)
}

fn remove_empty_dirs(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            remove_empty_dirs(&path);
            // Fails while the directory still holds a file, which is the point.
            let _ = fs::remove_dir(&path);
        }
    }
}

#[async_trait]
impl LineageStore for SnapshotLineage {
    fn name(&self) -> &str {
        SNAPSHOT_STORE
    }

    async fn find(&self, _target: &LineageTarget) -> Result<LineageHits> {
        Ok(hits(SNAPSHOT_FILE, self.files()?.len()))
    }

    async fn purge(&self, _target: &LineageTarget) -> Result<PurgeReport> {
        let mut removed = 0;
        for file in self.files()? {
            match zero_and_unlink(&file) {
                Ok(()) => removed += 1,
                Err(error) => tracing::warn!(
                    file = %self.relative(&file),
                    %error,
                    "a snapshot could not be removed, verify will report it"
                ),
            }
        }
        remove_empty_dirs(&self.dir);
        Ok(PurgeReport { removed })
    }

    async fn verify(&self, _target: &LineageTarget) -> Result<Vec<Residual>> {
        Ok(self
            .files()?
            .iter()
            .map(|file| {
                let kind = match fs::File::open(file) {
                    Ok(_) => SNAPSHOT_FILE,
                    Err(_) => SNAPSHOT_UNREADABLE,
                };
                residual(SNAPSHOT_STORE, kind, self.relative(file))
            })
            .collect())
    }
}

/// The last run's manifest and the cooldown file, both of which name lineages
/// and so carry subject text. Takes the lineage keys from the resolved target.
pub struct StateFileLineage {
    manifest: PathBuf,
    history: PathBuf,
}

impl StateFileLineage {
    pub fn new(manifest: PathBuf, history: PathBuf) -> Self {
        Self { manifest, history }
    }

    fn read(&self) -> Result<(Option<RunManifest>, TierHistory)> {
        let manifest = RunManifest::load(&self.manifest).map_err(|e| failed(STATE_STORE, e))?;
        let history = tier_history::load(&self.history).map_err(|e| failed(STATE_STORE, e))?;
        Ok((manifest, history))
    }
}

fn named(keys: &[String], key: &str) -> bool {
    keys.iter().any(|named| named == key)
}

#[async_trait]
impl LineageStore for StateFileLineage {
    fn name(&self) -> &str {
        STATE_STORE
    }

    async fn find(&self, target: &LineageTarget) -> Result<LineageHits> {
        let (manifest, history) = self.read()?;
        let in_manifest = manifest.map_or(0, |manifest| {
            let moved = &manifest.tier_moved_lineages;
            moved
                .iter()
                .filter(|key| named(&target.lineage_keys, key))
                .count()
        });
        let cooling = history
            .cooldown
            .iter()
            .filter(|entry| named(&target.lineage_keys, &entry.lineage_key))
            .count();
        let mut found = hits(MANIFEST_ENTRY, in_manifest);
        found.rows.extend(hits(COOLDOWN_ENTRY, cooling).rows);
        Ok(found)
    }

    async fn purge(&self, target: &LineageTarget) -> Result<PurgeReport> {
        let (manifest, mut history) = self.read()?;
        let mut removed = 0;
        if let Some(mut manifest) = manifest {
            let before = manifest.tier_moved_lineages.len();
            manifest
                .tier_moved_lineages
                .retain(|key| !named(&target.lineage_keys, key));
            let dropped = before - manifest.tier_moved_lineages.len();
            if dropped > 0 {
                manifest
                    .save(&self.manifest)
                    .map_err(|e| failed(STATE_STORE, e))?;
                removed += dropped;
            }
        }
        let before = history.cooldown.len();
        history
            .cooldown
            .retain(|entry| !named(&target.lineage_keys, &entry.lineage_key));
        let dropped = before - history.cooldown.len();
        if dropped > 0 {
            tier_history::save(&self.history, &history).map_err(|e| failed(STATE_STORE, e))?;
            removed += dropped;
        }
        Ok(PurgeReport { removed })
    }

    async fn verify(&self, target: &LineageTarget) -> Result<Vec<Residual>> {
        let (manifest, history) = self.read()?;
        let mut left = Vec::new();
        if let Some(manifest) = manifest {
            for (index, key) in manifest.tier_moved_lineages.iter().enumerate() {
                if named(&target.lineage_keys, key) {
                    left.push(residual(STATE_STORE, MANIFEST_ENTRY, index.to_string()));
                }
            }
        }
        for (index, entry) in history.cooldown.iter().enumerate() {
            if named(&target.lineage_keys, &entry.lineage_key) {
                left.push(residual(STATE_STORE, COOLDOWN_ENTRY, index.to_string()));
            }
        }
        Ok(left)
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use liam_daemon::consolidation::manifest::RunManifest;
    use liam_daemon::embedder_fingerprint::EmbedderFingerprint;
    use liam_store::Millis;
    use tempfile::TempDir;

    use super::*;
    use crate::deletion::test_support::sealed_subjects;
    use crate::extraction::cache::{cache_path, NewEntry};

    const OWNER_ONLY: u32 = 0o600;

    fn target(events: &[&str], lineages: &[&str]) -> LineageTarget {
        LineageTarget {
            job_id: "job".into(),
            event_ids: events.iter().map(|id| (*id).into()).collect(),
            lineage_keys: lineages.iter().map(|key| (*key).into()).collect(),
            ..LineageTarget::default()
        }
    }

    fn put(cache: &CacheStore, key: &str, events: &[&str]) {
        let ids: Vec<String> = events.iter().map(|id| (*id).into()).collect();
        cache
            .put(NewEntry {
                key,
                response: "{}",
                model_id: "model-a",
                model_sha256: "sha-a",
                prompt_version: "prompt-1",
                source_event_ids: &ids,
            })
            .unwrap();
    }

    #[tokio::test]
    async fn the_cache_purge_removes_exactly_the_entries_that_cite_a_deleted_event() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("liam.db"));
        let cache = Arc::new(CacheStore::open(path.clone()).unwrap());
        put(&cache, "a", &["e1", "e2"]);
        put(&cache, "b", &["e3"]);
        put(&cache, "c", &["e2"]);
        let store = CacheStoreLineage::new(Arc::clone(&cache));
        let doomed = target(&["e2"], &[]);

        // Act
        let found = store.find(&doomed).await.unwrap();
        let report = store.purge(&doomed).await.unwrap();
        let again = store.purge(&doomed).await.unwrap();

        // Assert
        assert_eq!(found.total(), 2);
        assert_eq!((report.removed, again.removed), (2, 0));
        assert!(store.verify(&doomed).await.unwrap().is_empty());
        assert_eq!(cache.get("b"), Some("{}".into()));
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, OWNER_ONLY);
    }

    #[tokio::test]
    async fn a_failed_cache_rewrite_keeps_the_file_and_verify_still_reports_the_entries() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_path(&dir.path().join("liam.db"));
        let cache = Arc::new(CacheStore::open_with_refused_rename(path.clone()).unwrap());
        put(&cache, "a", &["e1"]);
        let before = fs::read(&path).unwrap();
        let store = CacheStoreLineage::new(cache);
        let doomed = target(&["e1"], &[]);

        // Act
        let result = store.purge(&doomed).await;

        // Assert
        assert!(
            matches!(result, Err(Error::LineageStore { .. })),
            "{result:?}"
        );
        assert_eq!(fs::read(&path).unwrap(), before);
        let left = store.verify(&doomed).await.unwrap();
        assert_eq!(left, vec![residual(CACHE_STORE, CACHE_ENTRY, "a")]);
    }

    #[tokio::test]
    async fn a_snapshot_is_zeroed_and_unlinked_and_an_unreadable_one_is_reported() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let run = dir.path().join("run-1");
        fs::create_dir_all(&run).unwrap();
        fs::write(run.join("snapshot.db"), vec![7u8; 20_000]).unwrap();
        fs::write(run.join("snapshot.json"), b"{}").unwrap();
        let locked = dir.path().join("run-2");
        fs::create_dir_all(&locked).unwrap();
        fs::write(locked.join("snapshot.db"), b"secret").unwrap();
        fs::set_permissions(
            locked.join("snapshot.db"),
            fs::Permissions::from_mode(0o000),
        )
        .unwrap();
        let store = SnapshotLineage::new(dir.path().to_path_buf());

        // Act
        let found = store.find(&target(&[], &[])).await.unwrap();
        let report = store.purge(&target(&[], &[])).await.unwrap();
        let left = store.verify(&target(&[], &[])).await.unwrap();

        // Assert
        assert_eq!(found.total(), 3);
        assert_eq!(report.removed, 2);
        assert!(!run.exists(), "the emptied run directory goes too");
        assert_eq!(
            left,
            vec![residual(
                SNAPSHOT_STORE,
                SNAPSHOT_UNREADABLE,
                "run-2/snapshot.db"
            )]
        );
    }

    #[tokio::test]
    async fn a_snapshot_directory_that_does_not_exist_holds_nothing() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let store = SnapshotLineage::new(dir.path().join("none"));

        // Act
        let report = store.purge(&target(&[], &[])).await.unwrap();

        // Assert
        assert_eq!(report.removed, 0);
        assert!(store.verify(&target(&[], &[])).await.unwrap().is_empty());
    }

    fn manifest_naming(lineages: &[&str]) -> RunManifest {
        let embedder = EmbedderFingerprint {
            provider: "mock".into(),
            model: "mock-model".into(),
            dims: 8,
            prompt_scheme: 1,
        };
        let mut manifest = RunManifest::new("run".into(), 1, embedder, "llm".into(), Millis(1));
        manifest.tier_moved_lineages = lineages.iter().map(|key| (*key).into()).collect();
        manifest
    }

    #[tokio::test]
    async fn the_manifest_and_the_cooldown_file_are_rewritten_without_a_deleted_lineage() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let manifest_path = dir.path().join("liam.db.consolidation.json");
        let history_path = dir.path().join("liam.db.tier-history.json");
        manifest_naming(&["e:alice", "e:bob"])
            .save(&manifest_path)
            .unwrap();
        let mut history = TierHistory::default();
        history.after_rollback(&["e:alice".to_owned(), "e:bob".to_owned()], 3);
        tier_history::save(&history_path, &history).unwrap();
        let store = StateFileLineage::new(manifest_path.clone(), history_path.clone());
        let doomed = target(&[], &["e:alice"]);

        // Act
        let found = store.find(&doomed).await.unwrap();
        let before = store.verify(&doomed).await.unwrap();
        let report = store.purge(&doomed).await.unwrap();
        let again = store.purge(&doomed).await.unwrap();

        // Assert
        assert_eq!(found.total(), 2);
        assert_eq!(before.len(), 2);
        assert_eq!((report.removed, again.removed), (2, 0));
        assert!(store.verify(&doomed).await.unwrap().is_empty());
        let manifest = RunManifest::load(&manifest_path).unwrap().unwrap();
        assert_eq!(manifest.tier_moved_lineages, vec!["e:bob".to_owned()]);
        let history = tier_history::load(&history_path).unwrap();
        assert_eq!(history.cooldown.len(), 1);
        assert_eq!(history.cooldown[0].lineage_key, "e:bob");
    }

    #[tokio::test]
    async fn absent_state_files_hold_nothing() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let store = StateFileLineage::new(dir.path().join("m.json"), dir.path().join("h.json"));

        // Act
        let report = store.purge(&target(&[], &["e:alice"])).await.unwrap();

        // Assert
        assert_eq!(report.removed, 0);
    }

    #[tokio::test]
    async fn the_dedup_filter_is_rebuilt_once_the_database_purge_has_emptied_the_index() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let subjects = sealed_subjects(dir.path()).await;
        let filter_file = dir.path().join("dedup.filter");
        let store = Arc::new(subjects.store);
        let filter = BloomLineage::new(Arc::clone(&store), Some(filter_file.clone()));
        let before = filter.find(&subjects.target).await.unwrap();
        store.purge(&subjects.target).await.unwrap();

        // Act
        let stale = filter.verify(&subjects.target).await.unwrap();
        let report = filter.purge(&subjects.target).await.unwrap();
        let again = filter.purge(&subjects.target).await.unwrap();

        // Assert
        assert_eq!(
            before.total(),
            0,
            "a filter built from the index is current"
        );
        assert_eq!(
            stale,
            vec![residual(FILTER_STORE, FILTER_KIND, FILTER_KIND)]
        );
        assert_eq!((report.removed, again.removed), (1, 0));
        assert!(filter.verify(&subjects.target).await.unwrap().is_empty());
        assert!(filter_file.exists(), "the filter file is rewritten");
    }
}
