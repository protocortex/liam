// SPDX-License-Identifier: Apache-2.0
//! Every store a delete job purges, run in the order they were added: the
//! database first, because the stores that read it (the dedup filter) must see
//! it purged.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_daemon::consolidation::manifest::manifest_path_for;
use liam_daemon::consolidation::tier_history::history_path_for;
use liam_store::{DefaultGraph, LineageHits, LineageStore, LineageTarget, PurgeReport, Residual};

use super::files::{BloomLineage, CacheStoreLineage, SnapshotLineage, StateFileLineage};
use super::memory::{InMemoryLineage, NodeState};
use crate::extraction::cache::{cache_path, CacheStore};

#[derive(Default)]
pub struct LineageRegistry {
    stores: Vec<Box<dyn LineageStore>>,
}

impl LineageRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, store: impl LineageStore + 'static) -> Self {
        self.stores.push(Box::new(store));
        self
    }

    /// Every store a delete of the database at `database_path` reaches, the
    /// database first. `states` are the in-memory holders of node ids that are
    /// alive in this process.
    pub fn for_database(
        database_path: &Path,
        snapshot_dir: &Path,
        graph: Arc<DefaultGraph>,
        states: Vec<Arc<dyn NodeState>>,
    ) -> anyhow::Result<Self> {
        let mut registry = Self::new()
            .with(Arc::clone(&graph))
            .with(BloomLineage::new(graph, None));
        // Opening the cache would make a file for a database that never used one.
        let cache_file = cache_path(database_path);
        if cache_file.exists() {
            let cache = CacheStore::open(cache_file.clone())
                .with_context(|| format!("could not open the cache {}", cache_file.display()))?;
            registry = registry.with(CacheStoreLineage::new(Arc::new(cache)));
        }
        Ok(registry
            .with(SnapshotLineage::new(snapshot_dir.to_path_buf()))
            .with(StateFileLineage::new(
                manifest_path_for(database_path),
                history_path_for(database_path),
            ))
            .with(InMemoryLineage::new(states)))
    }

    pub fn store_count(&self) -> usize {
        self.stores.len()
    }

    // The operator command lists these in the change that adds it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn names(&self) -> Vec<String> {
        self.stores
            .iter()
            .map(|store| store.name().to_owned())
            .collect()
    }

    /// What each store would remove, by store name.
    #[cfg_attr(not(test), allow(dead_code))]
    pub async fn find_all(
        &self,
        target: &LineageTarget,
    ) -> anyhow::Result<Vec<(String, LineageHits)>> {
        let mut found = Vec::new();
        for store in &self.stores {
            let hits = store
                .find(target)
                .await
                .with_context(|| format!("looking in {} failed", store.name()))?;
            found.push((store.name().to_owned(), hits));
        }
        Ok(found)
    }

    /// Purges the store at `index`. A job runs the stores in turn and stops at
    /// the first failure, which names the store; a purge is safe to run again,
    /// so the job resumes from there.
    pub async fn purge_at(
        &self,
        index: usize,
        target: &LineageTarget,
    ) -> anyhow::Result<(String, PurgeReport)> {
        let store = &self.stores[index];
        let report = store
            .purge(target)
            .await
            .with_context(|| format!("purging {} failed", store.name()))?;
        tracing::info!(
            store = store.name(),
            removed = report.removed,
            "store purged"
        );
        Ok((store.name().to_owned(), report))
    }

    /// Every residual of every store, each carrying its store's name.
    pub async fn verify_all(&self, target: &LineageTarget) -> anyhow::Result<Vec<Residual>> {
        let mut residuals = Vec::new();
        for store in &self.stores {
            residuals.extend(
                store
                    .verify(target)
                    .await
                    .with_context(|| format!("verifying {} failed", store.name()))?,
            );
        }
        Ok(residuals)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use liam_daemon::consolidation::manifest::RunManifest;
    use liam_daemon::embedder_fingerprint::EmbedderFingerprint;
    use liam_store::Millis;
    use tempfile::TempDir;

    use super::*;
    use crate::deletion::files::{
        BloomLineage, CacheStoreLineage, SnapshotLineage, StateFileLineage,
    };
    use crate::deletion::memory::{InMemoryLineage, NodeState};
    use crate::deletion::test_support::sealed_subjects;
    use crate::extraction::cache::{cache_path, CacheStore, NewEntry};
    use crate::tiering::usage::UsageRecorder;

    const CAPACITY: usize = 16;

    #[tokio::test]
    async fn verify_all_reports_a_left_residual_of_every_store_under_its_name() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let subjects = sealed_subjects(dir.path()).await;
        let mut target = subjects.target.clone();
        target.lineage_keys = vec!["e:alice".to_owned()];
        let cache = Arc::new(CacheStore::open(cache_path(&dir.path().join("cache.db"))).unwrap());
        cache
            .put(NewEntry {
                key: "k",
                response: "{}",
                model_id: "m",
                model_sha256: "s",
                prompt_version: "p",
                source_event_ids: &target.event_ids,
            })
            .unwrap();
        let snapshots = dir.path().join("snapshots");
        std::fs::create_dir_all(snapshots.join("run-1")).unwrap();
        std::fs::write(snapshots.join("run-1/snapshot.db"), b"plain").unwrap();
        let manifest_path = dir.path().join("manifest.json");
        let embedder = EmbedderFingerprint {
            provider: "mock".into(),
            model: "mock-model".into(),
            dims: 8,
            prompt_scheme: 1,
        };
        let mut manifest = RunManifest::new("run".into(), 1, embedder, "llm".into(), Millis(1));
        manifest.tier_moved_lineages = vec!["e:alice".to_owned()];
        manifest.save(&manifest_path).unwrap();
        let recorder = Arc::new(UsageRecorder::new(CAPACITY));
        recorder.record([&subjects.alice], Millis(1));
        let states: Vec<Arc<dyn NodeState>> = vec![recorder];
        let registry = LineageRegistry::new()
            .with(Arc::clone(&subjects.store))
            .with(BloomLineage::new(Arc::clone(&subjects.store), None))
            .with(CacheStoreLineage::new(cache))
            .with(SnapshotLineage::new(snapshots))
            .with(StateFileLineage::new(
                manifest_path,
                dir.path().join("history.json"),
            ))
            .with(InMemoryLineage::new(states));

        // Act
        let before = registry.verify_all(&target).await.unwrap();
        subjects.store.purge(&target).await.unwrap();
        let after_database = registry.verify_all(&target).await.unwrap();
        let mut purged = Vec::new();
        for index in 0..registry.store_count() {
            purged.push(registry.purge_at(index, &target).await.unwrap());
        }
        let after = registry.verify_all(&target).await.unwrap();
        let found = registry.find_all(&target).await.unwrap();

        // Assert
        let stores = |residuals: &[liam_store::Residual]| -> BTreeSet<String> {
            residuals.iter().map(|r| r.store.clone()).collect()
        };
        let every_store: BTreeSet<String> = registry.names().into_iter().collect();
        assert!(stores(&before).contains("graph"));
        let after_database_names = stores(&after_database);
        let expected: BTreeSet<String> = every_store
            .iter()
            .filter(|name| name.as_str() != "graph")
            .cloned()
            .collect();
        assert_eq!(after_database_names, expected);
        assert_eq!(purged.len(), every_store.len());
        assert!(after.is_empty(), "{after:?}");
        assert!(found.iter().all(|(_, hits)| hits.total() == 0), "{found:?}");
    }
}
