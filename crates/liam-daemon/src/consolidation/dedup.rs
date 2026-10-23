// SPDX-License-Identifier: Apache-2.0
//! Near-duplicate merge: nodes whose content embeds almost the same are linked
//! by a `near_duplicate_of` edge. Both rows stay live, so the merge only adds
//! edges, each written through the logged `relate` path: a rebuild from the log
//! restores them, and a snapshot restore discards them.

use liam_store::{ContentEmbedder, DefaultGraph, Millis, NodeId};

/// The edge kind from a duplicate to the earlier node it repeats.
pub const NEAR_DUPLICATE_OF: &str = "near_duplicate_of";

/// Most nodes one pass embeds and compares. The comparison is quadratic and
/// each node costs an embedder call, so recent nodes are taken oldest first and
/// the rest wait for the next pass.
const MAX_CANDIDATES: usize = 500;

/// Most edges one pass writes. Each is its own write-lock hold, so this bounds
/// how long a pass can keep other writers waiting in total.
const MAX_MERGES_PER_PASS: usize = 200;

/// A node and the embedding of its content.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedded {
    pub id: NodeId,
    pub embedding: Vec<f32>,
}

/// One duplicate and the earlier node it repeats.
#[derive(Debug, Clone, PartialEq)]
pub struct MergePair {
    pub duplicate: NodeId,
    pub canonical: NodeId,
    pub similarity: f32,
}

/// What one pass did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MergeReport {
    pub candidates: usize,
    /// Nodes left out because the embedder failed on them.
    pub unembedded: usize,
    /// Pairs now linked, including ones a log dedup found already linked.
    pub linked: usize,
    /// Pairs the store refused: already linked, or a node is no longer live.
    pub refused: usize,
    /// Pairs found beyond `MAX_MERGES_PER_PASS`, left for the next pass.
    pub deferred: usize,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denominator = norm(a) * norm(b);
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

/// Pairs each item with the earlier kept item it matches best, at or above
/// `threshold`. An item that matches nothing is kept, so a duplicate always
/// points at a node that is not itself a duplicate. A tie goes to the earlier
/// item, so the same input in the same order always gives the same pairs.
pub fn plan_merges(items: &[Embedded], threshold: f32) -> Vec<MergePair> {
    let mut kept: Vec<&Embedded> = Vec::new();
    let mut pairs = Vec::new();
    for item in items {
        let mut best: Option<(&Embedded, f32)> = None;
        for candidate in &kept {
            let similarity = cosine(&candidate.embedding, &item.embedding);
            if similarity >= threshold && best.is_none_or(|(_, top)| similarity > top) {
                best = Some((candidate, similarity));
            }
        }
        match best {
            Some((canonical, similarity)) => pairs.push(MergePair {
                duplicate: item.id.clone(),
                canonical: canonical.id.clone(),
                similarity,
            }),
            None => kept.push(item),
        }
    }
    pairs
}

/// Links near-duplicates among the live nodes written at or after `since`.
///
/// # Errors
///
/// Fails when the store cannot list or read the nodes, or a link fails for a
/// reason other than the store refusing it.
pub async fn merge_near_duplicates(
    store: &DefaultGraph,
    embedder: &dyn ContentEmbedder,
    since: Millis,
    as_of: Millis,
    threshold: f32,
) -> liam_store::Result<MergeReport> {
    let mut changes = store.changes_since(since).await?;
    changes.retain(|change| !change.closed);
    changes.sort_by(|a, b| (a.tx_from, a.id.as_str()).cmp(&(b.tx_from, b.id.as_str())));
    changes.truncate(MAX_CANDIDATES);

    let mut report = MergeReport::default();
    let mut items = Vec::new();
    for change in &changes {
        let Some(node) = store.get(&change.id, as_of).await? else {
            continue;
        };
        // Entity pages carry no content of their own, so there is nothing to compare.
        if node.content.is_empty() {
            continue;
        }
        report.candidates += 1;
        match embedder.embed(&node.content).await {
            Ok(embedding) => items.push(Embedded {
                id: node.id,
                embedding,
            }),
            Err(reason) => {
                tracing::warn!(node = node.id.as_str(), %reason, "dedup: skipping a node the embedder failed on");
                report.unembedded += 1;
            }
        }
    }

    let pairs = plan_merges(&items, threshold);
    report.deferred = pairs.len().saturating_sub(MAX_MERGES_PER_PASS);
    tracing::info!(
        candidates = report.candidates,
        planned = pairs.len(),
        "dedup: merges planned"
    );
    for pair in pairs.iter().take(MAX_MERGES_PER_PASS) {
        match store
            .relate(&pair.duplicate, &pair.canonical, NEAR_DUPLICATE_OF)
            .await
        {
            Ok(_) => report.linked += 1,
            Err(liam_store::Error::RelateRefused(_)) => report.refused += 1,
            Err(err) => return Err(err),
        }
    }
    tracing::info!(
        linked = report.linked,
        refused = report.refused,
        deferred = report.deferred,
        "dedup: merges applied"
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event_log;
    use async_trait::async_trait;
    use liam_log::dedup::BloomConfig;
    use liam_log::wal::WalConfig;
    use liam_store::{EmbedError, GraphConfig, NewNode, RebuildMode, FOREVER};
    use std::collections::HashMap;
    use std::path::Path;

    const DIMS: usize = 2;
    /// Just before the end of time, so every live row is visible.
    const AS_OF: Millis = Millis(FOREVER.0 - 1);
    const WAL: WalConfig = WalConfig {
        segment_max_bytes: 1 << 20,
        rotate_interval_secs: 3600,
    };

    fn log_in(dir: &Path) -> liam_store::SharedLog {
        let log_dir = dir.join("events");
        let writer = event_log::open_writer(&log_dir, WAL).unwrap();
        event_log::shared_log(writer, &log_dir, BloomConfig::default(), None).unwrap()
    }

    /// Maps content to a fixed vector, so a test chooses the exact similarity.
    struct FixedEmbedder(HashMap<&'static str, Vec<f32>>);

    #[async_trait]
    impl ContentEmbedder for FixedEmbedder {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            self.0
                .get(text)
                .cloned()
                .ok_or_else(|| "no vector for this text".into())
        }
    }

    fn embedder(entries: &[(&'static str, [f32; DIMS])]) -> FixedEmbedder {
        FixedEmbedder(entries.iter().map(|(k, v)| (*k, v.to_vec())).collect())
    }

    fn item(id: &str, embedding: [f32; DIMS]) -> Embedded {
        Embedded {
            id: NodeId::from_raw(id.to_string()),
            embedding: embedding.to_vec(),
        }
    }

    async fn open(dir: &tempfile::TempDir) -> DefaultGraph {
        DefaultGraph::open(
            dir.path().join("dedup.db").to_str().unwrap(),
            GraphConfig::new(DIMS),
        )
        .await
        .unwrap()
        .with_log(log_in(dir.path()))
        .await
        .unwrap()
    }

    async fn insert(store: &DefaultGraph, content: &str) -> NodeId {
        store
            .insert(NewNode::now("fact", content, content))
            .await
            .unwrap()
    }

    async fn run(store: &DefaultGraph, embedder: &FixedEmbedder, threshold: f32) -> MergeReport {
        merge_near_duplicates(store, embedder, Millis(0), AS_OF, threshold)
            .await
            .unwrap()
    }

    async fn linked_count(store: &DefaultGraph) -> i64 {
        store.edge_fingerprint().await.unwrap().edge_count
    }

    #[test]
    fn plan_merges_pairs_items_above_the_threshold() {
        // Arrange
        let items = [
            item("a", [1.0, 0.0]),
            item("b", [1.0, 0.1]),
            item("c", [0.0, 1.0]),
        ];

        // Act
        let pairs = plan_merges(&items, 0.9);

        // Assert
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].duplicate.as_str(), "b");
        assert_eq!(pairs[0].canonical.as_str(), "a");
    }

    #[test]
    fn plan_merges_leaves_items_below_the_threshold_apart() {
        // Arrange
        let items = [item("a", [1.0, 0.0]), item("b", [0.0, 1.0])];

        // Act
        let pairs = plan_merges(&items, 0.5);

        // Assert
        assert!(pairs.is_empty());
    }

    #[test]
    fn plan_merges_merges_at_exactly_the_threshold() {
        // Arrange: the cosine of these two is exactly 0.6
        let items = [item("a", [1.0, 0.0]), item("b", [3.0, 4.0])];

        // Act
        let pairs = plan_merges(&items, 0.6);

        // Assert
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].similarity, 0.6);
    }

    #[test]
    fn plan_merges_breaks_a_tie_toward_the_earlier_item_every_time() {
        // Arrange: c is equally close to a and b, which are far apart
        let items = [
            item("a", [1.0, 0.0]),
            item("b", [0.0, 1.0]),
            item("c", [1.0, 1.0]),
        ];

        // Act
        let first = plan_merges(&items, 0.5);
        let second = plan_merges(&items, 0.5);

        // Assert
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].canonical.as_str(), "a");
    }

    #[test]
    fn plan_merges_points_a_chain_at_the_kept_node_not_at_a_duplicate() {
        // Arrange: b repeats a, c repeats b
        let items = [
            item("a", [1.0, 0.0]),
            item("b", [1.0, 0.05]),
            item("c", [1.0, 0.1]),
        ];

        // Act
        let pairs = plan_merges(&items, 0.9);

        // Assert
        assert!(pairs.iter().all(|pair| pair.canonical.as_str() == "a"));
        assert_eq!(pairs.len(), 2);
    }

    #[tokio::test]
    async fn merge_links_near_duplicates_and_keeps_both_rows() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = open(&dir).await;
        let first = insert(&store, "met Ana at noon").await;
        let second = insert(&store, "met Ana around noon").await;
        let unrelated = insert(&store, "tax filing is due").await;
        let embedder = embedder(&[
            ("met Ana at noon", [1.0, 0.0]),
            ("met Ana around noon", [1.0, 0.05]),
            ("tax filing is due", [0.0, 1.0]),
        ]);

        // Act
        let report = run(&store, &embedder, 0.9).await;

        // Assert
        assert_eq!(report.linked, 1);
        assert_eq!(
            store.neighbors(&second, AS_OF).await.unwrap(),
            vec![first.clone()]
        );
        assert!(store.neighbors(&unrelated, AS_OF).await.unwrap().is_empty());
        assert!(store.get(&first, AS_OF).await.unwrap().is_some());
        assert!(store.get(&second, AS_OF).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn merge_run_twice_adds_no_second_edge() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = open(&dir).await;
        insert(&store, "one").await;
        insert(&store, "one again").await;
        let embedder = embedder(&[("one", [1.0, 0.0]), ("one again", [1.0, 0.0])]);
        run(&store, &embedder, 0.9).await;

        // Act
        run(&store, &embedder, 0.9).await;

        // Assert
        assert_eq!(linked_count(&store).await, 1);
    }

    #[tokio::test]
    async fn merge_skips_a_node_the_embedder_fails_on() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = open(&dir).await;
        insert(&store, "known").await;
        insert(&store, "unknown").await;
        let embedder = embedder(&[("known", [1.0, 0.0])]);

        // Act
        let report = run(&store, &embedder, 0.9).await;

        // Assert
        assert_eq!(
            (report.candidates, report.unembedded, report.linked),
            (2, 1, 0)
        );
    }

    #[tokio::test]
    async fn merge_edges_come_back_after_a_rebuild_from_the_log() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = open(&dir).await;
        insert(&store, "one").await;
        insert(&store, "one again").await;
        let embedder = embedder(&[("one", [1.0, 0.0]), ("one again", [1.0, 0.0])]);
        run(&store, &embedder, 0.9).await;
        drop(store);
        let fresh = DefaultGraph::open(
            dir.path().join("rebuilt.db").to_str().unwrap(),
            GraphConfig::new(DIMS),
        )
        .await
        .unwrap();

        // Act
        let (_, report) = fresh
            .rebuild_from_log(log_in(dir.path()), RebuildMode::RequireEmpty)
            .await
            .unwrap();

        // Assert
        assert_eq!((report.live_nodes, report.live_edges), (2, 1));
    }
}
