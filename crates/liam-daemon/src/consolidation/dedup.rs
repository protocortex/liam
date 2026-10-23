// SPDX-License-Identifier: Apache-2.0
//! Near-duplicate merge: nodes whose content embeds almost the same are linked
//! by a `near_duplicate_of` edge. Both rows stay live, so the merge only adds
//! edges. Each edge goes through the logged `relate_from` path, stamped with the
//! run's source: a rebuild from the log replays it, and a rollback's
//! replay-forward skips it along with the rest of the run's own writes.

use std::collections::{BTreeMap, HashMap, HashSet};

use liam_store::relation::NEAR_DUPLICATE_OF;
use liam_store::{ContentEmbedder, DefaultGraph, Millis, NodeId, Query, BEGINNING};
use serde::{Deserialize, Serialize};

/// Most nodes one pass scans. The comparison is quadratic and each node costs
/// an embedder call, so the rest wait for the next pass.
const MAX_CANDIDATES: usize = 500;

/// Most edges one pass writes. Each is its own write-lock hold, so this bounds
/// how long a pass can keep other writers waiting in total.
const MAX_MERGES_PER_PASS: usize = 200;

/// Older nodes a vector search offers each scanned node to compare against.
const OLDER_NEIGHBOURS: usize = 3;

/// Why a pass failed.
#[derive(Debug)]
pub enum DedupError {
    /// A cosine threshold must be a number in (0, 1].
    InvalidThreshold(f32),
    Store(liam_store::Error),
}

impl std::fmt::Display for DedupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidThreshold(value) => {
                write!(f, "dedup threshold {value} is not a number in (0, 1]")
            }
            Self::Store(err) => write!(f, "dedup could not use the store: {err}"),
        }
    }
}

impl std::error::Error for DedupError {}

impl From<liam_store::Error> for DedupError {
    fn from(err: liam_store::Error) -> Self {
        Self::Store(err)
    }
}

/// Where a scan stopped. A later pass passes it back to continue after it, and
/// the row id breaks ties between nodes that share a `tx_from`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanPosition {
    pub tx_from: Millis,
    pub rowid: i64,
}

/// A node and the embedding of its content. Only nodes of the same scope and
/// producer are compared, since a merge across either would leak one party's
/// memory into another's.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedded {
    pub id: NodeId,
    pub scope: Option<String>,
    pub producer: String,
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
    /// Nodes the scan covered, whether or not they had content to compare.
    pub scanned: usize,
    /// Scanned nodes that were embedded.
    pub candidates: usize,
    /// Nodes left out because the embedder failed on them.
    pub unembedded: usize,
    /// Pairs now linked, including ones a log dedup found already linked.
    pub linked: usize,
    /// Pairs the store refused: a node is no longer live, or the edge differs.
    pub refused: usize,
    /// Pairs found beyond `MAX_MERGES_PER_PASS`, left for the next pass.
    pub deferred: usize,
    /// Pass this back as `after` to continue without skipping or repeating a
    /// node. `None` when the scan found nothing new.
    pub resume_from: Option<ScanPosition>,
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

fn check_threshold(threshold: f32) -> Result<(), DedupError> {
    if threshold.is_finite() && threshold > 0.0 && threshold <= 1.0 {
        Ok(())
    } else {
        Err(DedupError::InvalidThreshold(threshold))
    }
}

/// Pairs each item with the earlier kept item of its own scope and producer
/// that it matches best, at or above `threshold`. An item that matches nothing
/// is kept, so a duplicate always points at a node that is not itself a
/// duplicate. A tie goes to the earlier item, so the same input in the same
/// order always gives the same pairs. Pairs come out grouped, not in item order.
pub fn plan_merges(items: &[Embedded], threshold: f32) -> Vec<MergePair> {
    let mut groups: BTreeMap<(Option<&str>, &str), Vec<&Embedded>> = BTreeMap::new();
    for item in items {
        groups
            .entry((item.scope.as_deref(), item.producer.as_str()))
            .or_default()
            .push(item);
    }
    let mut pairs = Vec::new();
    for group in groups.values() {
        let mut kept: Vec<&Embedded> = Vec::new();
        for item in group {
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
    }
    pairs
}

/// The scanned nodes that have content, in scan order, each with where it sat.
struct Scan {
    items: Vec<Embedded>,
    positions: Vec<ScanPosition>,
    scanned: usize,
    unembedded: usize,
    last: Option<ScanPosition>,
}

async fn scan(
    store: &DefaultGraph,
    embedder: &dyn ContentEmbedder,
    after: Option<ScanPosition>,
    as_of: Millis,
) -> liam_store::Result<Scan> {
    // `changes_since` is exclusive on `tx_from`, so back up one to also see the
    // rows that share the position's `tx_from` and sort after it.
    let from = after.map_or(BEGINNING, |p| Millis(p.tx_from.0 - 1));
    let mut changes = store.changes_since(from).await?;
    changes.retain(|change| {
        !change.closed
            && after.is_none_or(|p| (change.tx_from, change.rowid) > (p.tx_from, p.rowid))
    });
    changes.truncate(MAX_CANDIDATES);

    let mut out = Scan {
        items: Vec::new(),
        positions: Vec::new(),
        scanned: changes.len(),
        unembedded: 0,
        last: changes.last().map(|c| ScanPosition {
            tx_from: c.tx_from,
            rowid: c.rowid,
        }),
    };
    for change in &changes {
        let Some(node) = store.get(&change.id, as_of).await? else {
            continue;
        };
        // Entity pages carry no content of their own, so there is nothing to compare.
        if node.content.is_empty() {
            continue;
        }
        match embedder.embed(&node.content).await {
            Ok(embedding) => {
                out.items.push(Embedded {
                    id: node.id,
                    scope: node.scope,
                    producer: node.producer,
                    embedding,
                });
                out.positions.push(ScanPosition {
                    tx_from: change.tx_from,
                    rowid: change.rowid,
                });
            }
            Err(reason) => {
                tracing::warn!(node = node.id.as_str(), %reason, "dedup: skipping a node the embedder failed on");
                out.unembedded += 1;
            }
        }
    }
    Ok(out)
}

/// The best older node of `item`'s own scope and producer, found by a bounded
/// vector search over the whole store. `batch` holds the ids compared in-batch,
/// and `embedded` caches embeddings across items.
async fn older_match(
    store: &DefaultGraph,
    embedder: &dyn ContentEmbedder,
    item: &Embedded,
    batch: &HashSet<&str>,
    as_of: Millis,
    threshold: f32,
    embedded: &mut HashMap<String, Option<Vec<f32>>>,
) -> liam_store::Result<Option<MergePair>> {
    let mut query = Query::vector(item.embedding.clone())
        .with_k(OLDER_NEIGHBOURS)
        .with_as_of(as_of);
    if let Some(scope) = &item.scope {
        query = query.with_scope(scope.clone());
    }
    let mut best: Option<MergePair> = None;
    for hit in store.query(&query).await? {
        if batch.contains(hit.id.as_str()) {
            continue;
        }
        let Some(node) = store.get(&hit.id, as_of).await? else {
            continue;
        };
        // The search reads a scope and its children; only the same scope merges.
        if node.scope != item.scope || node.producer != item.producer || node.content.is_empty() {
            continue;
        }
        if !embedded.contains_key(node.id.as_str()) {
            let vector = embedder.embed(&node.content).await.ok();
            embedded.insert(node.id.as_str().to_string(), vector);
        }
        let Some(Some(vector)) = embedded.get(node.id.as_str()) else {
            continue;
        };
        let similarity = cosine(vector, &item.embedding);
        if similarity >= threshold && best.as_ref().is_none_or(|top| similarity > top.similarity) {
            best = Some(MergePair {
                duplicate: item.id.clone(),
                canonical: node.id,
                similarity,
            });
        }
    }
    Ok(best)
}

/// Links near-duplicates among the live nodes written after `after`, and
/// between those and older nodes of the same scope and producer. `as_of` must
/// be an instant at which every live row is visible. `source` stamps each edge's
/// log record.
///
/// # Errors
///
/// Fails on a threshold outside (0, 1], when the store cannot list or read the
/// nodes, or when a link fails for a reason other than the store refusing it.
pub async fn merge_near_duplicates(
    store: &DefaultGraph,
    embedder: &dyn ContentEmbedder,
    after: Option<ScanPosition>,
    as_of: Millis,
    threshold: f32,
    source: &str,
) -> Result<MergeReport, DedupError> {
    check_threshold(threshold)?;
    let scan = scan(store, embedder, after, as_of).await?;
    let mut report = MergeReport {
        scanned: scan.scanned,
        candidates: scan.items.len(),
        unembedded: scan.unembedded,
        resume_from: scan.last,
        ..MergeReport::default()
    };

    let batch: HashSet<&str> = scan.items.iter().map(|item| item.id.as_str()).collect();
    let index: HashMap<&str, usize> = scan
        .items
        .iter()
        .enumerate()
        .map(|(i, item)| (item.id.as_str(), i))
        .collect();
    let mut pairs = plan_merges(&scan.items, threshold);
    let paired: HashSet<&str> = pairs.iter().map(|p| p.duplicate.as_str()).collect();
    let mut embedded = HashMap::new();
    let mut older = Vec::new();
    for item in scan
        .items
        .iter()
        .filter(|i| !paired.contains(i.id.as_str()))
    {
        let found = older_match(
            store,
            embedder,
            item,
            &batch,
            as_of,
            threshold,
            &mut embedded,
        )
        .await?;
        older.extend(found);
    }
    pairs.extend(older);
    pairs.sort_by_key(|pair| index[pair.duplicate.as_str()]);
    tracing::info!(
        scanned = report.scanned,
        candidates = report.candidates,
        planned = pairs.len(),
        "dedup: merges planned"
    );

    report.deferred = pairs.len().saturating_sub(MAX_MERGES_PER_PASS);
    pairs.truncate(MAX_MERGES_PER_PASS);
    for pair in &pairs {
        let written = store
            .relate_from(&pair.duplicate, &pair.canonical, NEAR_DUPLICATE_OF, source)
            .await;
        match written {
            Ok(_) => report.linked += 1,
            Err(liam_store::Error::RelateRefused(_)) => report.refused += 1,
            Err(err) => return Err(err.into()),
        }
    }
    if report.deferred > 0 {
        // Resume right after the last applied pair so the deferred ones are seen again.
        report.resume_from = pairs
            .last()
            .map(|pair| scan.positions[index[pair.duplicate.as_str()]]);
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

    const SOURCE: &str = "consolidation:test-run";
    use futures_util::StreamExt;
    use liam_log::dedup::BloomConfig;
    use liam_log::event::LogPayload;
    use liam_log::reader::{LogReader, SequentialScanReader};
    use liam_log::wal::WalConfig;
    use liam_store::{
        EmbedError, FixedClock, GraphConfig, NewNode, RebuildMode, SharedLog, FOREVER,
    };
    use std::path::Path;
    use std::sync::Arc;

    const DIMS: usize = 2;
    const START: Millis = Millis(1000);
    /// Just before the end of time, so every live row is visible.
    const AS_OF: Millis = Millis(FOREVER.0 - 1);
    const WAL: WalConfig = WalConfig {
        segment_max_bytes: 1 << 20,
        rotate_interval_secs: 3600,
    };
    const THRESHOLD: f32 = 0.9;
    const EXACT: f32 = 1.0;
    const ONE: [f32; DIMS] = [1.0, 0.0];
    const ORTHOGONAL: [f32; DIMS] = [0.0, 1.0];
    /// Radians between neighbours on the circle the cap tests spread nodes over.
    const ANGLE_STEP: f32 = 0.005;

    /// Maps content to a vector, so a test chooses the exact similarity.
    type VectorFor = Box<dyn Fn(&str) -> Option<Vec<f32>> + Send + Sync>;
    struct Embed(VectorFor);

    #[async_trait]
    impl ContentEmbedder for Embed {
        async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            (self.0)(text).ok_or_else(|| "no vector for this text".into())
        }
    }

    fn table(entries: &[(&'static str, [f32; DIMS])]) -> Embed {
        let map: HashMap<&str, Vec<f32>> = entries.iter().map(|(k, v)| (*k, v.to_vec())).collect();
        Embed(Box::new(move |text| map.get(text).cloned()))
    }

    fn constant(vector: [f32; DIMS]) -> Embed {
        Embed(Box::new(move |_| Some(vector.to_vec())))
    }

    /// "n7" sits at 7 steps round the circle, so no two nodes match at 1.0.
    fn spread() -> Embed {
        Embed(Box::new(|text| {
            let step: f32 = text.trim_start_matches('n').parse().ok()?;
            let angle = step * ANGLE_STEP;
            Some(vec![angle.cos(), angle.sin()])
        }))
    }

    fn item(id: &str, embedding: [f32; DIMS]) -> Embedded {
        in_scope(id, None, "p", embedding)
    }

    fn in_scope(id: &str, scope: Option<&str>, producer: &str, embedding: [f32; DIMS]) -> Embedded {
        Embedded {
            id: NodeId::from_raw(id.to_string()),
            scope: scope.map(str::to_string),
            producer: producer.to_string(),
            embedding: embedding.to_vec(),
        }
    }

    fn log_in(dir: &Path) -> SharedLog {
        let log_dir = dir.join("events");
        let writer = event_log::open_writer(&log_dir, WAL).unwrap();
        event_log::shared_log(writer, &log_dir, BloomConfig::default(), None).unwrap()
    }

    async fn open(dir: &tempfile::TempDir) -> (DefaultGraph, Arc<FixedClock>) {
        let clock = Arc::new(FixedClock::new(START));
        let store = DefaultGraph::open_with_clock(
            dir.path().join("dedup.db").to_str().unwrap(),
            GraphConfig::new(DIMS),
            clock.clone(),
        )
        .await
        .unwrap()
        .with_log(log_in(dir.path()))
        .await
        .unwrap();
        (store, clock)
    }

    async fn insert(store: &DefaultGraph, content: &str) -> NodeId {
        store.insert(fact(content)).await.unwrap()
    }

    fn fact(content: &str) -> NewNode {
        NewNode::now("fact", content, content)
    }

    async fn run(store: &DefaultGraph, embed: &Embed, threshold: f32) -> MergeReport {
        merge_near_duplicates(store, embed, None, AS_OF, threshold, SOURCE)
            .await
            .unwrap()
    }

    async fn edge_count(store: &DefaultGraph) -> i64 {
        store.edge_fingerprint().await.unwrap().edge_count
    }

    #[test]
    fn plan_merges_pairs_items_above_the_threshold() {
        // Arrange
        let items = [item("a", ONE), item("b", [1.0, 0.1]), item("c", ORTHOGONAL)];

        // Act
        let pairs = plan_merges(&items, THRESHOLD);

        // Assert
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].duplicate.as_str(), "b");
        assert_eq!(pairs[0].canonical.as_str(), "a");
    }

    #[test]
    fn plan_merges_leaves_items_below_the_threshold_apart() {
        // Arrange
        let items = [item("a", ONE), item("b", ORTHOGONAL)];

        // Act
        let pairs = plan_merges(&items, 0.5);

        // Assert
        assert!(pairs.is_empty());
    }

    #[test]
    fn plan_merges_merges_at_exactly_the_threshold() {
        // Arrange: the cosine of these two is exactly 0.6
        let items = [item("a", ONE), item("b", [3.0, 4.0])];

        // Act
        let pairs = plan_merges(&items, 0.6);

        // Assert
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].similarity, 0.6);
    }

    #[test]
    fn plan_merges_breaks_a_tie_toward_the_earlier_item_every_time() {
        // Arrange: c is equally close to a and b, which are far apart
        let items = [item("a", ONE), item("b", ORTHOGONAL), item("c", [1.0, 1.0])];

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
            item("a", ONE),
            item("b", [1.0, 0.05]),
            item("c", [1.0, 0.1]),
        ];

        // Act
        let pairs = plan_merges(&items, THRESHOLD);

        // Assert
        assert!(pairs.iter().all(|pair| pair.canonical.as_str() == "a"));
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn plan_merges_never_pairs_across_scopes_or_producers() {
        // Arrange: identical embeddings, but each pair differs in scope or producer
        let items = [
            in_scope("a", Some("one"), "p", ONE),
            in_scope("b", Some("two"), "p", ONE),
            in_scope("c", Some("one"), "q", ONE),
            in_scope("d", None, "p", ONE),
        ];

        // Act
        let pairs = plan_merges(&items, EXACT);

        // Assert
        assert!(pairs.is_empty());
    }

    #[test]
    fn plan_merges_pairs_within_one_scope_and_producer() {
        // Arrange
        let items = [
            in_scope("a", Some("one"), "p", ONE),
            in_scope("b", Some("two"), "p", ONE),
            in_scope("c", Some("one"), "p", ONE),
        ];

        // Act
        let pairs = plan_merges(&items, EXACT);

        // Assert
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].duplicate.as_str(), "c");
        assert_eq!(pairs[0].canonical.as_str(), "a");
    }

    #[tokio::test]
    async fn merge_links_near_duplicates_and_keeps_both_rows() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        let first = insert(&store, "met Ana at noon").await;
        let second = insert(&store, "met Ana around noon").await;
        let unrelated = insert(&store, "tax filing is due").await;
        let embed = table(&[
            ("met Ana at noon", ONE),
            ("met Ana around noon", [1.0, 0.05]),
            ("tax filing is due", ORTHOGONAL),
        ]);

        // Act
        let report = run(&store, &embed, THRESHOLD).await;

        // Assert
        assert_eq!(report.linked, 1);
        let neighbours = store.neighbors(&second, AS_OF).await.unwrap();
        assert_eq!(neighbours, vec![first.clone()]);
        assert!(store.neighbors(&unrelated, AS_OF).await.unwrap().is_empty());
        assert!(store.get(&first, AS_OF).await.unwrap().is_some());
        assert!(store.get(&second, AS_OF).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn merge_points_every_duplicate_at_the_earliest_node_when_clocks_tie() {
        // Arrange: three rows share one tx_from, so only the row order separates them
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        let first = insert(&store, "a").await;
        let second = insert(&store, "b").await;
        let third = insert(&store, "c").await;
        let embed = constant(ONE);

        // Act
        let report = run(&store, &embed, EXACT).await;

        // Assert
        assert_eq!(report.linked, 2);
        let mut linked = store.neighbors(&first, AS_OF).await.unwrap();
        linked.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        let mut expected = vec![second, third];
        expected.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        assert_eq!(linked, expected);
    }

    #[tokio::test]
    async fn merge_run_twice_adds_no_second_edge() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        insert(&store, "one").await;
        insert(&store, "one again").await;
        let embed = constant(ONE);
        run(&store, &embed, THRESHOLD).await;

        // Act
        run(&store, &embed, THRESHOLD).await;

        // Assert
        assert_eq!(edge_count(&store).await, 1);
    }

    #[tokio::test]
    async fn merge_skips_a_node_the_embedder_fails_on() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        insert(&store, "known").await;
        insert(&store, "unknown").await;
        let embed = table(&[("known", ONE)]);

        // Act
        let report = run(&store, &embed, THRESHOLD).await;

        // Assert
        assert_eq!(
            (report.candidates, report.unembedded, report.linked),
            (1, 1, 0)
        );
    }

    #[tokio::test]
    async fn merge_skips_entity_pages_that_have_no_content() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        store
            .insert(NewNode::entity("person", "Ana"))
            .await
            .unwrap();
        insert(&store, "one").await;
        insert(&store, "one again").await;

        // Act
        let report = run(&store, &constant(ONE), THRESHOLD).await;

        // Assert
        assert_eq!(
            (report.scanned, report.candidates, report.linked),
            (3, 2, 1)
        );
    }

    #[tokio::test]
    async fn merge_does_not_link_identical_content_across_scopes() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        store.insert(fact("same").with_scope("one")).await.unwrap();
        store.insert(fact("same").with_scope("two")).await.unwrap();

        // Act
        let report = run(&store, &constant(ONE), EXACT).await;

        // Assert
        assert_eq!(report.linked, 0);
        assert_eq!(edge_count(&store).await, 0);
    }

    #[tokio::test]
    async fn merge_links_a_new_node_to_an_older_one_scanned_in_an_earlier_pass() {
        // Arrange: the older node is past the scan position by the second pass
        let dir = tempfile::tempdir().unwrap();
        let (store, clock) = open(&dir).await;
        let old = store
            .insert(fact("old").with_scope("s").with_embedding(ONE.to_vec()))
            .await
            .unwrap();
        let embed = constant(ONE);
        let first = run(&store, &embed, THRESHOLD).await;
        clock.set(Millis(START.0 + 1));
        let new = store.insert(fact("new").with_scope("s")).await.unwrap();

        // Act
        let second =
            merge_near_duplicates(&store, &embed, first.resume_from, AS_OF, THRESHOLD, SOURCE)
                .await
                .unwrap();

        // Assert
        assert_eq!((second.scanned, second.linked), (1, 1));
        assert_eq!(store.neighbors(&new, AS_OF).await.unwrap(), vec![old]);
    }

    #[tokio::test]
    async fn merge_does_not_link_to_an_older_node_in_a_child_scope() {
        // Arrange: a search for scope "s" also reads "s/child", which must not merge
        let dir = tempfile::tempdir().unwrap();
        let (store, clock) = open(&dir).await;
        store
            .insert(
                fact("old")
                    .with_scope("s/child")
                    .with_embedding(ONE.to_vec()),
            )
            .await
            .unwrap();
        let embed = constant(ONE);
        let first = run(&store, &embed, THRESHOLD).await;
        clock.set(Millis(START.0 + 1));
        store.insert(fact("new").with_scope("s")).await.unwrap();

        // Act
        let second =
            merge_near_duplicates(&store, &embed, first.resume_from, AS_OF, THRESHOLD, SOURCE)
                .await
                .unwrap();

        // Assert
        assert_eq!((second.scanned, second.linked), (1, 0));
    }

    #[tokio::test]
    async fn merge_scans_every_node_across_capped_passes_with_no_gap() {
        // Arrange: every row shares one tx_from, so the cut falls inside it
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        let total = MAX_CANDIDATES * 2 + 100;
        for step in 0..total {
            insert(&store, &format!("n{step}")).await;
        }
        let embed = spread();

        // Act
        let mut after = None;
        let mut scanned = Vec::new();
        loop {
            let report = merge_near_duplicates(&store, &embed, after, AS_OF, EXACT, SOURCE)
                .await
                .unwrap();
            if report.scanned == 0 {
                break;
            }
            scanned.push(report.scanned);
            after = report.resume_from;
        }

        // Assert
        assert_eq!(scanned, vec![MAX_CANDIDATES, MAX_CANDIDATES, 100]);
        assert_eq!(edge_count(&store).await, 0);
    }

    #[tokio::test]
    async fn merge_caps_edges_per_pass_and_the_next_pass_finishes_the_rest() {
        // Arrange: every node repeats the first. Nodes past the cap are stored
        // with an orthogonal vector so the later search finds only the older ones.
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        let total = MAX_MERGES_PER_PASS + 50;
        for step in 0..total {
            let stored = if step <= MAX_MERGES_PER_PASS {
                ONE
            } else {
                ORTHOGONAL
            };
            let node = fact(&format!("m{step}")).with_embedding(stored.to_vec());
            store.insert(node).await.unwrap();
        }
        let embed = constant(ONE);

        // Act
        let first = run(&store, &embed, EXACT).await;
        let second = merge_near_duplicates(&store, &embed, first.resume_from, AS_OF, EXACT, SOURCE)
            .await
            .unwrap();

        // Assert
        assert_eq!(
            (first.linked, first.deferred),
            (MAX_MERGES_PER_PASS, total - 1 - MAX_MERGES_PER_PASS)
        );
        assert_eq!(
            (second.scanned, second.deferred),
            (total - 1 - MAX_MERGES_PER_PASS, 0)
        );
        assert_eq!(edge_count(&store).await, i64::try_from(total - 1).unwrap());
    }

    #[tokio::test]
    async fn merge_rejects_a_threshold_outside_zero_to_one() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        let embed = constant(ONE);

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.5, 1.5] {
            // Act
            let result = merge_near_duplicates(&store, &embed, None, AS_OF, bad, SOURCE).await;

            // Assert
            assert!(
                matches!(result, Err(DedupError::InvalidThreshold(_))),
                "threshold {bad} was accepted"
            );
        }
    }

    #[tokio::test]
    async fn merge_accepts_a_threshold_of_exactly_one() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;

        // Act
        let result = merge_near_duplicates(&store, &constant(ONE), None, AS_OF, 1.0, SOURCE).await;

        // Assert
        assert!(result.is_ok());
    }

    /// The (duplicate content, canonical content) of every edge written to the
    /// log, sorted, so two stores with different ids compare by what they hold.
    async fn decisions(store: DefaultGraph, dir: &Path, ids: &[NodeId]) -> Vec<(String, String)> {
        let mut content_of = HashMap::new();
        for id in ids {
            let node = store.get(id, AS_OF).await.unwrap().unwrap();
            content_of.insert(id.as_str().to_string(), node.content);
        }
        drop(store);
        let reader = SequentialScanReader::local(&dir.join("events")).unwrap();
        let mut stream = reader.scan(None);
        let mut out = Vec::new();
        while let Some(record) = stream.next().await {
            if let LogPayload::EdgeWrite(edge) = record.unwrap().event.payload {
                out.push((content_of[&edge.src].clone(), content_of[&edge.dst].clone()));
            }
        }
        out.sort();
        out
    }

    async fn tied_store(dir: &tempfile::TempDir) -> (DefaultGraph, Vec<NodeId>) {
        let (store, _) = open(dir).await;
        let mut ids = Vec::new();
        for content in ["a", "b", "c"] {
            ids.push(insert(&store, content).await);
        }
        (store, ids)
    }

    #[tokio::test]
    async fn merge_makes_the_same_decisions_in_two_stores_given_a_forced_tie() {
        // Arrange: c is equally close to a and b, which are far apart
        let embed = table(&[("a", ONE), ("b", ORTHOGONAL), ("c", [1.0, 1.0])]);
        let (first_dir, second_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (first, first_ids) = tied_store(&first_dir).await;
        let (second, second_ids) = tied_store(&second_dir).await;

        // Act
        run(&first, &embed, 0.5).await;
        run(&second, &embed, 0.5).await;

        // Assert
        let decided = decisions(first, first_dir.path(), &first_ids).await;
        assert_eq!(
            decided,
            decisions(second, second_dir.path(), &second_ids).await
        );
        assert_eq!(decided, vec![("c".to_string(), "a".to_string())]);
    }

    #[test]
    fn plan_merges_gives_the_same_ordered_pairs_however_the_groups_interleave() {
        // Arrange: four producers, each with an original then a repeat
        let grouped: Vec<Embedded> = ["a", "b", "c", "d"]
            .iter()
            .flat_map(|p| {
                [
                    in_scope(&format!("{p}1"), None, p, ONE),
                    in_scope(&format!("{p}2"), None, p, ONE),
                ]
            })
            .collect();
        // Each producer's own order is kept; only the interleaving changes
        let woven: Vec<Embedded> = [6, 0, 4, 2, 1, 7, 5, 3]
            .iter()
            .map(|&i| grouped[i].clone())
            .collect();

        // Act
        let from_grouped = plan_merges(&grouped, THRESHOLD);
        let from_woven = plan_merges(&woven, THRESHOLD);

        // Assert: producers come out in key order, each repeat pointing at its original
        let pairs: Vec<(&str, &str)> = from_grouped
            .iter()
            .map(|p| (p.duplicate.as_str(), p.canonical.as_str()))
            .collect();
        assert_eq!(
            pairs,
            [("a2", "a1"), ("b2", "b1"), ("c2", "c1"), ("d2", "d1")]
        );
        assert_eq!(from_grouped, from_woven);
    }

    #[tokio::test]
    async fn merge_edges_are_logged_with_the_consolidation_source() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        insert(&store, "one").await;
        insert(&store, "one again").await;

        // Act
        run(&store, &constant(ONE), THRESHOLD).await;
        drop(store);

        // Assert
        let reader = SequentialScanReader::local(&dir.path().join("events")).unwrap();
        let mut stream = reader.scan(None);
        let mut sources = Vec::new();
        while let Some(record) = stream.next().await {
            let event = record.unwrap().event;
            if matches!(event.payload, LogPayload::EdgeWrite(_)) {
                sources.push(event.source);
            }
        }
        assert_eq!(sources, vec![SOURCE.to_string()]);
    }

    #[tokio::test]
    async fn merge_edges_come_back_after_a_rebuild_from_the_log() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = open(&dir).await;
        insert(&store, "one").await;
        insert(&store, "one again").await;
        run(&store, &constant(ONE), THRESHOLD).await;
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
