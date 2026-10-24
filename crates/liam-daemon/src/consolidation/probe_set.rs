// SPDX-License-Identifier: Apache-2.0
//! The probe set of one run, built from the live store before the run changes
//! anything. A probe asks for a node by its own content, so a correct answer
//! needs no hand-labelled data. The set is kept with its baseline scores beside
//! the manifest, so a run resumed after a crash still compares against the store
//! as it was before the run, not as the crash left it.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use liam_store::{DefaultGraph, Millis, NodeId, BEGINNING};
use rand_chacha::rand_core::RngCore;
use serde::{Deserialize, Serialize};

use super::probes::{Probe, ProbeReport};

/// Appended to the database path to name the probe set file.
const PROBE_SET_SUFFIX: &str = ".consolidation.probes.json";

/// Layout version of the file, written as `version`. Any other is refused.
const FORMAT_VERSION: u32 = 1;

/// The probes of a run and how the store scored on them before the run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeSet {
    pub probes: Vec<Probe>,
    pub baseline: ProbeReport,
}

#[derive(Serialize, Deserialize)]
struct Sidecar {
    version: u32,
    #[serde(flatten)]
    set: ProbeSet,
}

/// Where the probe set for the database at `database_path` lives.
pub fn probe_set_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(PROBE_SET_SUFFIX);
    PathBuf::from(path)
}

/// Picks up to `count` live nodes with content, in an order set by `rng` alone,
/// and makes a probe of each. Candidates are ordered by content before the draw
/// so that stores holding the same content pick the same probes whatever ids
/// their nodes were given.
///
/// # Errors
///
/// Fails when the store cannot list or read its nodes.
pub async fn pick_probes(
    store: &DefaultGraph,
    rng: &mut impl RngCore,
    as_of: Millis,
    count: usize,
) -> liam_store::Result<Vec<Probe>> {
    let ids: HashSet<NodeId> = store
        .changes_since(BEGINNING)
        .await?
        .into_iter()
        .filter(|change| !change.closed)
        .map(|change| change.id)
        .collect();

    let mut candidates = Vec::new();
    for id in &ids {
        if let Some(node) = store.get(id, as_of).await? {
            if !node.content.is_empty() {
                candidates.push((node.content, node.id));
            }
        }
    }
    candidates
        .sort_by(|(a, a_id), (b, b_id)| a.cmp(b).then_with(|| a_id.as_str().cmp(b_id.as_str())));

    let drawn = count.min(candidates.len());
    for i in 0..drawn {
        let j = i + (rng.next_u64() % (candidates.len() - i) as u64) as usize;
        candidates.swap(i, j);
    }
    candidates.truncate(drawn);
    Ok(candidates
        .into_iter()
        .map(|(content, id)| Probe {
            name: format!("node:{}", id.as_str()),
            query: content,
            relevant: vec![id],
            embedding: None,
        })
        .collect())
}

impl ProbeSet {
    /// The probes that still have a live node to find, with their baseline. A
    /// node the run's sweep removed on purpose cannot be recalled, so it must
    /// not count as a regression.
    ///
    /// # Errors
    ///
    /// Fails when the store cannot read a node.
    pub async fn retain_live(
        mut self,
        store: &DefaultGraph,
        as_of: Millis,
    ) -> liam_store::Result<Self> {
        let mut kept = Vec::new();
        for probe in self.probes {
            let mut live = false;
            for id in &probe.relevant {
                live |= store.get(id, as_of).await?.is_some();
            }
            if live {
                kept.push(probe);
            }
        }
        self.baseline
            .scores
            .retain(|name, _| kept.iter().any(|probe| &probe.name == name));
        self.probes = kept;
        Ok(self)
    }

    /// Writes the set to `path`, replacing any earlier one.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let sidecar = Sidecar {
            version: FORMAT_VERSION,
            set: self.clone(),
        };
        let bytes = serde_json::to_vec_pretty(&sidecar)?;
        write_atomically(path, &bytes, &mut OsFileOps).map_err(|source| {
            anyhow::anyhow!(
                "could not save the consolidation probe set at {}: {source}",
                path.display()
            )
        })
    }

    /// Reads the set at `path`. A missing or unreadable file is an error: a run
    /// past its baseline step has nothing to compare against without it.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| {
            anyhow::anyhow!(
                "the consolidation probe set {} could not be read: {source}",
                path.display()
            )
        })?;
        let sidecar: Sidecar = serde_json::from_str(&text).map_err(|source| {
            anyhow::anyhow!(
                "the consolidation probe set {} could not be parsed: {source}",
                path.display()
            )
        })?;
        anyhow::ensure!(
            sidecar.version == FORMAT_VERSION,
            "the consolidation probe set {} has version {}, but this build reads version \
             {FORMAT_VERSION}",
            path.display(),
            sidecar.version
        );
        Ok(sidecar.set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidation::probes::ProbeScore;
    use liam_store::{GraphConfig, NewNode, FOREVER};
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    const DIMS: usize = 4;
    /// Just before the end of time, so every live row is visible.
    const AS_OF: Millis = Millis(FOREVER.0 - 1);
    const NODES: usize = 12;
    const PICKED: usize = 5;
    const SEED: u64 = 7;
    const OTHER_SEED: u64 = 8;

    async fn store_with(dir: &tempfile::TempDir, name: &str, contents: &[String]) -> DefaultGraph {
        let path = dir.path().join(name);
        let store = DefaultGraph::open(path.to_str().unwrap(), GraphConfig::new(DIMS))
            .await
            .unwrap();
        for content in contents {
            store
                .insert(NewNode::now("fact", content, content))
                .await
                .unwrap();
        }
        store
    }

    fn contents(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("topic{i} detail{i}")).collect()
    }

    async fn queries(store: &DefaultGraph, seed: u64, count: usize) -> Vec<String> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        pick_probes(store, &mut rng, AS_OF, count)
            .await
            .unwrap()
            .into_iter()
            .map(|probe| probe.query)
            .collect()
    }

    #[tokio::test]
    async fn the_same_seed_picks_the_same_content_whatever_the_ids_and_insert_order() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let forward = store_with(&dir, "forward.db", &contents(NODES)).await;
        let mut reversed = contents(NODES);
        reversed.reverse();
        let backward = store_with(&dir, "backward.db", &reversed).await;

        // Act
        let first = queries(&forward, SEED, PICKED).await;
        let second = queries(&backward, SEED, PICKED).await;

        // Assert
        assert_eq!(first.len(), PICKED);
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn another_seed_picks_other_content() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = store_with(&dir, "seeds.db", &contents(NODES)).await;

        // Act
        let first = queries(&store, SEED, PICKED).await;
        let second = queries(&store, OTHER_SEED, PICKED).await;

        // Assert
        assert_ne!(first, second);
    }

    #[tokio::test]
    async fn a_store_smaller_than_the_count_gives_one_probe_per_node() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = store_with(&dir, "small.db", &contents(2)).await;

        // Act
        let picked = queries(&store, SEED, PICKED).await;

        // Assert
        assert_eq!(picked.len(), 2);
    }

    #[tokio::test]
    async fn an_empty_store_gives_no_probes() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = store_with(&dir, "empty.db", &[]).await;

        // Act
        let picked = queries(&store, SEED, PICKED).await;

        // Assert
        assert!(picked.is_empty());
    }

    #[tokio::test]
    async fn each_probe_asks_for_its_own_node() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = store_with(&dir, "own.db", &contents(NODES)).await;
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);

        // Act
        let probes = pick_probes(&store, &mut rng, AS_OF, PICKED).await.unwrap();

        // Assert
        for probe in &probes {
            let node = store.get(&probe.relevant[0], AS_OF).await.unwrap().unwrap();
            assert_eq!(node.content, probe.query);
            assert_eq!(probe.relevant.len(), 1);
        }
    }

    #[test]
    fn a_saved_set_loads_back_equal() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = probe_set_path_for(&dir.path().join("liam.db"));
        let mut baseline = ProbeReport::default();
        baseline.scores.insert(
            "node:a".into(),
            ProbeScore {
                recall: 1.0,
                ndcg: 0.5,
            },
        );
        let set = ProbeSet {
            probes: vec![Probe {
                name: "node:a".into(),
                query: "alpha".into(),
                relevant: vec![NodeId::from_raw("a")],
                embedding: None,
            }],
            baseline,
        };

        // Act
        set.save(&path).unwrap();
        let loaded = ProbeSet::load(&path).unwrap();

        // Assert
        assert_eq!(loaded, set);
    }

    #[test]
    fn loading_a_missing_set_is_an_error() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = probe_set_path_for(&dir.path().join("liam.db"));

        // Act
        let result = ProbeSet::load(&path);

        // Assert
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn retain_live_drops_a_probe_whose_node_is_gone_and_its_baseline() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = DefaultGraph::open(
            dir.path().join("p.db").to_str().unwrap(),
            liam_store::GraphConfig::new(DIMS),
        )
        .await
        .unwrap();
        let kept = store
            .insert(liam_store::NewNode::now("keep", "a", "alpha"))
            .await
            .unwrap();
        let gone = store
            .insert(liam_store::NewNode::now("fact", "b", "beta"))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        store
            .gc(&liam_store::RetentionPolicy::keep("fact", Millis(0)).without_reclaim())
            .await
            .unwrap();
        let probe = |name: &str, id: &NodeId| Probe {
            name: name.into(),
            query: name.into(),
            relevant: vec![id.clone()],
            embedding: None,
        };
        let score = ProbeScore {
            recall: 1.0,
            ndcg: 1.0,
        };
        let set = ProbeSet {
            probes: vec![probe("kept", &kept), probe("gone", &gone)],
            baseline: ProbeReport {
                scores: [("kept".to_owned(), score), ("gone".to_owned(), score)].into(),
            },
        };

        // Act
        let live = set.retain_live(&store, AS_OF).await.unwrap();

        // Assert
        let names: Vec<_> = live.probes.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["kept"]);
        assert_eq!(live.baseline.scores.keys().collect::<Vec<_>>(), ["kept"]);
    }
}
