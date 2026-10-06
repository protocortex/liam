// SPDX-License-Identifier: Apache-2.0

//! Bloom pre-check and exact hash index for write deduplication.
//!
//! The filter answers "possibly seen" cheaply; the index decides, and a rebuild
//! from the log restores both after a restart.

use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;

use async_trait::async_trait;
use fastbloom::BloomFilter;

use crate::event::{LogEvent, LogPayload};
use crate::hash::content_hashes;

/// Sizing for the bloom filter: capacity and the false-positive rate it is tuned for.
#[derive(Debug, Clone, PartialEq)]
pub struct BloomConfig {
    expected_items: usize,
    false_positive_rate: f64,
}

/// Why a bloom filter sizing was rejected.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum BloomConfigError {
    #[error("false positive rate {0} must be finite, above 0 and below 1")]
    FalsePositiveRate(f64),
    #[error("expected items must be at least 1")]
    NoExpectedItems,
}

impl BloomConfig {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Result<Self, BloomConfigError> {
        if !(false_positive_rate.is_finite()
            && false_positive_rate > 0.0
            && false_positive_rate < 1.0)
        {
            return Err(BloomConfigError::FalsePositiveRate(false_positive_rate));
        }
        if expected_items == 0 {
            return Err(BloomConfigError::NoExpectedItems);
        }
        Ok(Self {
            expected_items,
            false_positive_rate,
        })
    }

    pub fn expected_items(&self) -> usize {
        self.expected_items
    }

    pub fn false_positive_rate(&self) -> f64 {
        self.false_positive_rate
    }
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            expected_items: 100_000,
            false_positive_rate: 0.01,
        }
    }
}

/// Fixed seed so the same hashes always produce the same filter, across runs.
const BLOOM_SEED: u128 = 0x6c69_616d_626c_6f6f_6d73_6565_6430_3031;

/// Cheap pre-check in front of the exact lookup. May answer true for a hash
/// that was never inserted, never false for one that was.
///
/// `Sync` so a future borrowing a pre-check can move between threads.
pub trait PreCheck: Sync {
    fn might_contain(&self, hash: &[u8; 32]) -> bool;
}

/// The write a hash was first stored by, and the rows that write created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedWrite {
    pub first_event_id: String,
    pub row_ids: Vec<String>,
}

/// Exact lookup against the log's hash index; decides the final answer.
///
/// A store-backed index does I/O, so the lookup is async and may fail.
#[async_trait]
pub trait HashIndex {
    type Error;

    async fn lookup(&mut self, hash: &[u8; 32]) -> Result<Option<IndexedWrite>, Self::Error>;
}

/// The stored write for `hash`, or `None` when it is new: the pre-check
/// short-circuits a miss without touching the index, and the exact index
/// decides every pre-check hit.
pub async fn is_duplicate<I: HashIndex>(
    pre_check: &dyn PreCheck,
    index: &mut I,
    hash: &[u8; 32],
) -> Result<Option<IndexedWrite>, I::Error> {
    if !pre_check.might_contain(hash) {
        return Ok(None);
    }
    index.lookup(hash).await
}

/// Bloom filter over content hashes with a pinned seed.
#[derive(Debug, Clone, PartialEq)]
pub struct HashBloom {
    config: BloomConfig,
    filter: BloomFilter,
}

impl HashBloom {
    pub fn new(config: BloomConfig) -> Self {
        let filter = BloomFilter::with_false_pos(config.false_positive_rate)
            .seed(&BLOOM_SEED)
            .expected_items(config.expected_items);
        Self { config, filter }
    }

    pub fn from_hashes<'a>(
        config: BloomConfig,
        hashes: impl IntoIterator<Item = &'a [u8; 32]>,
    ) -> Self {
        let mut bloom = Self::new(config);
        for hash in hashes {
            bloom.insert(hash);
        }
        bloom
    }

    pub fn insert(&mut self, hash: &[u8; 32]) {
        self.filter.insert(hash);
    }
}

impl PreCheck for HashBloom {
    fn might_contain(&self, hash: &[u8; 32]) -> bool {
        self.filter.contains(hash)
    }
}

/// One hash with the write that first carried it and that write's rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub hash: [u8; 32],
    pub first_event_id: String,
    pub row_ids: Vec<String>,
}

/// In-memory `HashIndex`: maps a content hash to the first write that carried it.
#[derive(Debug, Default)]
pub struct InMemoryHashIndex {
    writes: HashMap<[u8; 32], IndexedWrite>,
}

impl InMemoryHashIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_entries(entries: impl IntoIterator<Item = IndexEntry>) -> Self {
        let mut index = Self::new();
        for entry in entries {
            index.insert(entry);
        }
        index
    }

    /// Keeps the first write for a hash, so repeats resolve to the original.
    pub fn insert(&mut self, entry: IndexEntry) {
        self.writes.entry(entry.hash).or_insert(IndexedWrite {
            first_event_id: entry.first_event_id,
            row_ids: entry.row_ids,
        });
    }

    pub fn first_event_id(&self, hash: &[u8; 32]) -> Option<&str> {
        self.writes
            .get(hash)
            .map(|write| write.first_event_id.as_str())
    }

    pub fn len(&self) -> usize {
        self.writes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

#[async_trait]
impl HashIndex for InMemoryHashIndex {
    type Error = Infallible;

    async fn lookup(&mut self, hash: &[u8; 32]) -> Result<Option<IndexedWrite>, Infallible> {
        Ok(self.writes.get(hash).cloned())
    }
}

/// What a rebuild yields: the bloom filter every inserted hash went into, and
/// the index entries for hashes whose first carrier was not voided.
#[derive(Debug)]
pub struct Rebuilt {
    pub bloom: HashBloom,
    pub entries: Vec<IndexEntry>,
}

/// Scans every event once, in log order, and returns the bloom filter and
/// index entries that incremental appends would have produced.
///
/// A `Voided` event drops the index entry of the write it voids, so a retry of
/// that write becomes the first carrier. The bloom keeps the hash: it cannot
/// delete, and a stale hit is overruled by the exact index. Tombstone and
/// duplicate events carry no content, so they add no hash.
pub fn rebuild_from_log(events: impl Iterator<Item = LogEvent>, config: BloomConfig) -> Rebuilt {
    let mut bloom = HashBloom::new(config);
    let mut carriers = FirstCarriers::default();
    for event in events {
        if let LogPayload::Voided { target_event_id } = &event.payload {
            carriers.void(target_event_id);
            continue;
        }
        for (hash, row_id) in content_hashes(&event) {
            bloom.insert(&hash);
            carriers.record(hash, &event.event_id, row_id);
        }
    }
    Rebuilt {
        bloom,
        entries: carriers.into_entries(),
    }
}

/// Index entries keyed by first-carrier order, with the lookups a void needs.
#[derive(Default)]
struct FirstCarriers {
    next_position: u64,
    by_position: BTreeMap<u64, IndexEntry>,
    position_by_hash: HashMap<[u8; 32], u64>,
    hashes_by_event: HashMap<String, Vec<[u8; 32]>>,
}

impl FirstCarriers {
    fn record(&mut self, hash: [u8; 32], event_id: &str, row_id: String) {
        if let Some(position) = self.position_by_hash.get(&hash) {
            if let Some(entry) = self.by_position.get_mut(position) {
                if entry.first_event_id == event_id {
                    entry.row_ids.push(row_id);
                }
            }
            return;
        }
        let position = self.next_position;
        self.next_position += 1;
        self.position_by_hash.insert(hash, position);
        self.hashes_by_event
            .entry(event_id.to_owned())
            .or_default()
            .push(hash);
        self.by_position.insert(
            position,
            IndexEntry {
                hash,
                first_event_id: event_id.to_owned(),
                row_ids: vec![row_id],
            },
        );
    }

    fn void(&mut self, target_event_id: &str) {
        for hash in self
            .hashes_by_event
            .remove(target_event_id)
            .unwrap_or_default()
        {
            if let Some(position) = self.position_by_hash.remove(&hash) {
                self.by_position.remove(&position);
            }
        }
    }

    fn into_entries(self) -> Vec<IndexEntry> {
        self.by_position.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{
        EdgeRow, NodeRow, RowEffect, TombstoneTable, TombstoneTarget, CURRENT_SCHEMA_VERSION,
    };
    use crate::fixtures::{filter_with, sample_hash, sample_hashes, small_config};
    use crate::hash::{hash_edge, hash_node, EdgeContent, NodeContent};

    struct AlwaysTrue;

    impl PreCheck for AlwaysTrue {
        fn might_contain(&self, _hash: &[u8; 32]) -> bool {
            true
        }
    }

    struct AlwaysFalse;

    impl PreCheck for AlwaysFalse {
        fn might_contain(&self, _hash: &[u8; 32]) -> bool {
            false
        }
    }

    #[derive(Default)]
    struct CountingIndex {
        stored: Vec<([u8; 32], IndexedWrite)>,
        lookups: usize,
    }

    #[async_trait]
    impl HashIndex for CountingIndex {
        type Error = Infallible;

        async fn lookup(&mut self, hash: &[u8; 32]) -> Result<Option<IndexedWrite>, Infallible> {
            self.lookups += 1;
            Ok(self
                .stored
                .iter()
                .find(|(stored, _)| stored == hash)
                .map(|(_, write)| write.clone()))
        }
    }

    struct FailingIndex;

    #[async_trait]
    impl HashIndex for FailingIndex {
        type Error = &'static str;

        async fn lookup(&mut self, _hash: &[u8; 32]) -> Result<Option<IndexedWrite>, &'static str> {
            Err("index unavailable")
        }
    }

    fn indexed_write(event_id: &str, row_id: &str) -> IndexedWrite {
        IndexedWrite {
            first_event_id: event_id.into(),
            row_ids: vec![row_id.into()],
        }
    }

    #[test]
    fn bloom_config_accepts_a_rate_strictly_between_zero_and_one() {
        // Arrange / Act
        let config = BloomConfig::new(1, 0.5);

        // Assert
        assert_eq!(
            config.map(|config| (config.expected_items(), config.false_positive_rate())),
            Ok((1, 0.5))
        );
    }

    #[test]
    fn bloom_config_rejects_a_rate_outside_the_open_unit_interval() {
        // Arrange
        let rates = [
            0.0,
            1.0,
            f64::NAN,
            -0.01,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.5,
        ];

        for rate in rates {
            // Act
            let result = BloomConfig::new(100, rate);

            // Assert
            assert!(
                matches!(result, Err(BloomConfigError::FalsePositiveRate(_))),
                "rate {rate} should be rejected, got {result:?}"
            );
        }
    }

    #[test]
    fn bloom_config_rejects_zero_expected_items() {
        // Arrange / Act
        let result = BloomConfig::new(0, 0.01);

        // Assert
        assert_eq!(result, Err(BloomConfigError::NoExpectedItems));
    }

    #[test]
    fn a_filter_sized_for_a_single_item_builds_and_works() {
        // Arrange
        let config = BloomConfig::new(1, 0.01).expect("valid config");
        let mut filter = HashBloom::new(config);
        let hash = sample_hash(1);

        // Act
        filter.insert(&hash);

        // Assert
        assert!(filter.might_contain(&hash));
    }

    #[test]
    fn default_config_targets_a_one_percent_false_positive_rate() {
        // Arrange / Act
        let config = BloomConfig::default();

        // Assert
        assert_eq!(config.false_positive_rate(), 0.01);
    }

    #[test]
    fn filters_built_from_the_same_hashes_are_equal_in_any_insertion_order() {
        // Arrange
        let hashes = sample_hashes(0..2_000);
        let reversed: Vec<[u8; 32]> = hashes.iter().rev().copied().collect();

        // Act
        let forward_filter = filter_with(small_config(), &hashes);
        let reversed_filter = filter_with(small_config(), &reversed);
        let separate_instance = filter_with(small_config(), &hashes);

        // Assert
        assert_eq!(forward_filter, reversed_filter);
        assert_eq!(forward_filter, separate_instance);
    }

    #[test]
    fn filters_built_from_different_hashes_are_not_equal() {
        // Arrange
        let first = filter_with(small_config(), &sample_hashes(0..500));
        let second = filter_with(small_config(), &sample_hashes(500..1_000));

        // Act
        let equal = first == second;

        // Assert
        assert!(!equal, "different content must not produce equal filters");
        assert_eq!(first, first.clone(), "a filter equals its own copy");
    }

    #[test]
    fn every_inserted_hash_is_reported_as_possibly_present() {
        // Arrange
        let hashes = sample_hashes(0..3_000);
        let filter = filter_with(small_config(), &hashes);

        // Act
        let missed: Vec<usize> = hashes
            .iter()
            .enumerate()
            .filter(|(_, hash)| !filter.might_contain(hash))
            .map(|(position, _)| position)
            .collect();

        // Assert
        assert!(
            missed.is_empty(),
            "{} false negatives, first at {:?}",
            missed.len(),
            missed.first()
        );
    }

    #[test]
    fn false_positive_rate_on_absent_hashes_stays_near_the_configured_rate() {
        // Arrange
        let filter = filter_with(small_config(), &sample_hashes(0..5_000));
        let absent = sample_hashes(1_000_000..1_020_000);

        // Act
        let false_positives = absent
            .iter()
            .filter(|hash| filter.might_contain(hash))
            .count();
        let rate = false_positives as f64 / absent.len() as f64;

        // Assert
        assert!(rate > 0.0, "a bloom filter at 1% must show some collisions");
        assert!(rate < 0.03, "rate {rate} far above the configured 1%");
    }

    #[test]
    fn rebuilt_filter_equals_the_incrementally_built_one_on_a_held_out_probe_set() {
        // Arrange
        let stored = sample_hashes(0..2_000);
        let probes: Vec<[u8; 32]> = sample_hashes(0..100)
            .into_iter()
            .chain(sample_hashes(2_000_000..2_001_000))
            .collect();
        let incremental = filter_with(small_config(), &stored);

        // Act
        let rebuilt = HashBloom::from_hashes(small_config(), &stored);

        // Assert
        let incremental_answers: Vec<bool> = probes
            .iter()
            .map(|h| incremental.might_contain(h))
            .collect();
        let rebuilt_answers: Vec<bool> = probes.iter().map(|h| rebuilt.might_contain(h)).collect();
        assert!(
            incremental_answers[..100].iter().all(|present| *present),
            "known-present probes must be found"
        );
        assert_eq!(rebuilt_answers, incremental_answers);
        assert_eq!(rebuilt, incremental);
    }

    #[tokio::test]
    async fn a_pre_check_hit_for_a_never_stored_hash_is_decided_by_the_exact_index() {
        // Arrange
        let mut index = CountingIndex::default();
        let never_stored = sample_hash(42);

        // Act
        let found = is_duplicate(&AlwaysTrue, &mut index, &never_stored).await;

        // Assert
        assert_eq!(
            found,
            Ok(None),
            "the exact lookup must overrule a false positive"
        );
        assert_eq!(index.lookups, 1, "a pre-check hit must consult the index");
    }

    #[tokio::test]
    async fn a_pre_check_hit_for_a_stored_hash_returns_the_stored_write() {
        // Arrange
        let stored = sample_hash(7);
        let write = indexed_write("event-7", "node-7");
        let mut index = CountingIndex {
            stored: vec![(stored, write.clone())],
            ..CountingIndex::default()
        };

        // Act
        let found = is_duplicate(&AlwaysTrue, &mut index, &stored).await;

        // Assert
        assert_eq!(found, Ok(Some(write)));
    }

    #[tokio::test]
    async fn a_pre_check_miss_skips_the_exact_index() {
        // Arrange
        let mut index = CountingIndex::default();

        // Act
        let found = is_duplicate(&AlwaysFalse, &mut index, &sample_hash(1)).await;

        // Assert
        assert_eq!(found, Ok(None));
        assert_eq!(index.lookups, 0, "a definite miss needs no index lookup");
    }

    #[tokio::test]
    async fn an_index_failure_is_reported_not_treated_as_a_miss() {
        // Arrange
        let mut index = FailingIndex;

        // Act
        let found = is_duplicate(&AlwaysTrue, &mut index, &sample_hash(1)).await;

        // Assert
        assert_eq!(found, Err("index unavailable"));
    }

    fn node_row(id: &str, content: &str) -> NodeRow {
        NodeRow {
            id: id.into(),
            kind: "fact".into(),
            label: "label".into(),
            content: content.into(),
            producer: "agent-a".into(),
            attributes: "{}".into(),
            scope: Some("proj/a".into()),
            subject: None,
            confidence: 0.75,
            valid_from: 1_000,
            valid_from_supplied: true,
            valid_until: 4_102_444_800_000,
            tx_from: 2_000,
            tx_to: 4_102_444_800_000,
        }
    }

    fn edge_row(id: &str, dst: &str) -> EdgeRow {
        EdgeRow {
            id: id.into(),
            src: "node-1".into(),
            dst: dst.into(),
            edge_type: "relates_to".into(),
            attributes: "{}".into(),
            tx_from: 2_000,
            tx_to: 4_102_444_800_000,
        }
    }

    // Sentinel content_hash: rebuild must derive hashes from the row content,
    // so this value must never end up in the filter or the index.
    const EVENT_HASH_SENTINEL: [u8; 32] = [0xEE; 32];

    fn log_event(event_id: &str, payload: LogPayload) -> LogEvent {
        LogEvent {
            event_id: event_id.into(),
            content_hash: EVENT_HASH_SENTINEL,
            source: "agent-a".into(),
            trust_score: 0.9,
            observed_at: 1_000,
            ingested_at: 2_000,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload,
        }
    }

    fn voided(event_id: &str, target: &str) -> LogEvent {
        log_event(
            event_id,
            LogPayload::Voided {
                target_event_id: target.into(),
            },
        )
    }

    // A row's hash is what the write path computed for the same request:
    // valid_from counts only when the caller supplied it.
    fn expected_node_hash(row: &NodeRow) -> [u8; 32] {
        hash_node(&NodeContent {
            kind: row.kind.clone(),
            label: row.label.clone(),
            content: row.content.clone(),
            producer: row.producer.clone(),
            scope: row.scope.clone(),
            subject: row.subject.clone(),
            attributes: row.attributes.clone(),
            valid_from: row.valid_from_supplied.then_some(row.valid_from),
            confidence: row.confidence,
        })
    }

    fn expected_edge_hash(row: &EdgeRow) -> [u8; 32] {
        hash_edge(&EdgeContent {
            src: row.src.clone(),
            dst: row.dst.clone(),
            edge_type: row.edge_type.clone(),
        })
    }

    fn entry_for<'a>(rebuilt: &'a Rebuilt, hash: &[u8; 32]) -> Option<&'a IndexEntry> {
        rebuilt.entries.iter().find(|entry| &entry.hash == hash)
    }

    #[test]
    fn an_inserted_entry_is_found_with_its_event_id() {
        // Arrange
        let mut index = InMemoryHashIndex::new();
        let hash = sample_hash(1);

        // Act
        index.insert(IndexEntry {
            hash,
            first_event_id: "event-1".into(),
            row_ids: vec!["node-1".into()],
        });

        // Assert
        assert_eq!(index.first_event_id(&hash), Some("event-1"));
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[tokio::test]
    async fn the_in_memory_index_looks_up_the_stored_write() {
        // Arrange
        let mut index = InMemoryHashIndex::from_entries([IndexEntry {
            hash: sample_hash(1),
            first_event_id: "event-1".into(),
            row_ids: vec!["node-1".into(), "edge-1".into()],
        }]);

        // Act
        let hit = index.lookup(&sample_hash(1)).await;
        let miss = index.lookup(&sample_hash(2)).await;

        // Assert
        assert_eq!(
            hit,
            Ok(Some(IndexedWrite {
                first_event_id: "event-1".into(),
                row_ids: vec!["node-1".into(), "edge-1".into()],
            }))
        );
        assert_eq!(miss, Ok(None));
    }

    #[test]
    fn a_second_insert_of_the_same_hash_keeps_the_first_event_id() {
        // Arrange
        let mut index = InMemoryHashIndex::new();
        let hash = sample_hash(1);
        let entry = |event_id: &str| IndexEntry {
            hash,
            first_event_id: event_id.into(),
            row_ids: Vec::new(),
        };
        index.insert(entry("event-1"));

        // Act
        index.insert(entry("event-2"));

        // Assert
        assert_eq!(index.first_event_id(&hash), Some("event-1"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn an_index_built_from_entries_holds_each_hash() {
        // Arrange
        let entries = (1..=2).map(|n| IndexEntry {
            hash: sample_hash(n),
            first_event_id: format!("event-{n}"),
            row_ids: Vec::new(),
        });

        // Act
        let index = InMemoryHashIndex::from_entries(entries);

        // Assert
        assert_eq!(index.first_event_id(&sample_hash(1)), Some("event-1"));
        assert_eq!(index.first_event_id(&sample_hash(2)), Some("event-2"));
        assert_eq!(index.first_event_id(&sample_hash(3)), None);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn an_empty_index_reports_empty() {
        // Arrange / Act
        let index = InMemoryHashIndex::new();

        // Assert
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn rebuilding_from_an_empty_log_gives_an_empty_filter_and_no_entries() {
        // Arrange
        let events: Vec<LogEvent> = Vec::new();

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(rebuilt.bloom, HashBloom::new(small_config()));
        assert!(rebuilt.entries.is_empty());
        assert!(!rebuilt.bloom.might_contain(&sample_hash(1)));
    }

    #[test]
    fn rebuild_indexes_a_node_write_and_an_edge_write_by_their_row_hashes() {
        // Arrange
        let node = node_row("node-1", "alpha");
        let edge = edge_row("edge-1", "node-2");
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(node.clone())),
            log_event("event-2", LogPayload::EdgeWrite(edge.clone())),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        let (node_hash, edge_hash) = (expected_node_hash(&node), expected_edge_hash(&edge));
        assert_eq!(
            rebuilt.entries,
            vec![
                IndexEntry {
                    hash: node_hash,
                    first_event_id: "event-1".into(),
                    row_ids: vec!["node-1".into()],
                },
                IndexEntry {
                    hash: edge_hash,
                    first_event_id: "event-2".into(),
                    row_ids: vec!["edge-1".into()],
                },
            ]
        );
        assert!(rebuilt.bloom.might_contain(&node_hash));
        assert!(rebuilt.bloom.might_contain(&edge_hash));
        assert!(entry_for(&rebuilt, &EVENT_HASH_SENTINEL).is_none());
    }

    #[test]
    fn each_row_of_an_episode_batch_hashes_independently() {
        // Arrange
        let (first, second) = (node_row("node-1", "alpha"), node_row("node-2", "beta"));
        let edge = edge_row("edge-1", "node-2");
        let batch = log_event(
            "event-1",
            LogPayload::EpisodeBatch(vec![
                RowEffect::Node(first.clone()),
                RowEffect::Node(second.clone()),
                RowEffect::Edge(edge.clone()),
            ]),
        );

        // Act
        let rebuilt = rebuild_from_log(vec![batch].into_iter(), small_config());

        // Assert
        for (hash, row_id) in [
            (expected_node_hash(&first), "node-1"),
            (expected_node_hash(&second), "node-2"),
            (expected_edge_hash(&edge), "edge-1"),
        ] {
            let entry = entry_for(&rebuilt, &hash).expect("entry for batch row");
            assert_eq!(entry.first_event_id, "event-1");
            assert_eq!(entry.row_ids, vec![row_id.to_owned()]);
            assert!(rebuilt.bloom.might_contain(&hash));
        }
        assert_eq!(rebuilt.entries.len(), 3);
    }

    #[test]
    fn identical_rows_in_one_batch_share_an_entry_listing_both_row_ids() {
        // Arrange
        let (first, second) = (node_row("node-1", "alpha"), node_row("node-2", "alpha"));
        let batch = log_event(
            "event-1",
            LogPayload::EpisodeBatch(vec![
                RowEffect::Node(first.clone()),
                RowEffect::Node(second),
            ]),
        );

        // Act
        let rebuilt = rebuild_from_log(vec![batch].into_iter(), small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![IndexEntry {
                hash: expected_node_hash(&first),
                first_event_id: "event-1".into(),
                row_ids: vec!["node-1".into(), "node-2".into()],
            }]
        );
    }

    #[test]
    fn an_empty_episode_batch_adds_nothing() {
        // Arrange
        let batch = log_event("event-1", LogPayload::EpisodeBatch(Vec::new()));

        // Act
        let rebuilt = rebuild_from_log(vec![batch].into_iter(), small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
        assert_eq!(rebuilt.bloom, HashBloom::new(small_config()));
    }

    #[test]
    fn rebuild_maps_a_repeated_hash_to_the_first_event_that_carried_it() {
        // Arrange
        let first = node_row("node-1", "alpha");
        let repeat = node_row("node-2", "alpha");
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(first.clone())),
            log_event("event-2", LogPayload::NodeWrite(repeat.clone())),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(expected_node_hash(&first), expected_node_hash(&repeat));
        assert_eq!(
            rebuilt.entries,
            vec![IndexEntry {
                hash: expected_node_hash(&first),
                first_event_id: "event-1".into(),
                row_ids: vec!["node-1".into()],
            }]
        );
    }

    #[test]
    fn edges_differing_only_in_attributes_share_one_entry() {
        // Arrange
        let first = edge_row("edge-1", "node-2");
        let second = EdgeRow {
            attributes: "{\"weight\":2}".into(),
            ..edge_row("edge-2", "node-2")
        };
        let events = vec![
            log_event("event-1", LogPayload::EdgeWrite(first)),
            log_event("event-2", LogPayload::EdgeWrite(second)),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(rebuilt.entries.len(), 1);
        assert_eq!(rebuilt.entries[0].first_event_id, "event-1");
    }

    #[test]
    fn entries_follow_the_order_their_first_carriers_were_written() {
        // Arrange
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(node_row("node-1", "c"))),
            log_event("event-2", LogPayload::NodeWrite(node_row("node-2", "a"))),
            log_event("event-3", LogPayload::NodeWrite(node_row("node-3", "b"))),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        let order: Vec<&str> = rebuilt
            .entries
            .iter()
            .map(|entry| entry.first_event_id.as_str())
            .collect();
        assert_eq!(order, ["event-1", "event-2", "event-3"]);
    }

    #[test]
    fn tombstone_and_duplicate_events_contribute_no_hash() {
        // Arrange
        let target = TombstoneTarget {
            table: TombstoneTable::Nodes,
            id: "node-1".into(),
        };
        let events = vec![
            log_event("event-1", LogPayload::Tombstone(vec![target])),
            log_event(
                "event-2",
                LogPayload::DuplicateOf {
                    first_event_id: "event-0".into(),
                },
            ),
            voided("event-3", "event-0"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
        assert_eq!(rebuilt.bloom, HashBloom::new(small_config()));
    }

    #[test]
    fn a_voided_event_removes_its_index_entry_but_the_bloom_keeps_the_hash() {
        // Arrange
        let node = node_row("node-1", "alpha");
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(node.clone())),
            voided("event-2", "event-1"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
        assert!(rebuilt.bloom.might_contain(&expected_node_hash(&node)));
    }

    #[test]
    fn a_retry_after_a_void_becomes_the_first_carrier() {
        // Arrange
        let first = node_row("node-1", "alpha");
        let retry = node_row("node-2", "alpha");
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(first.clone())),
            voided("event-2", "event-1"),
            log_event("event-3", LogPayload::NodeWrite(retry)),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![IndexEntry {
                hash: expected_node_hash(&first),
                first_event_id: "event-3".into(),
                row_ids: vec!["node-2".into()],
            }]
        );
    }

    #[test]
    fn voiding_a_batch_removes_every_entry_it_was_first_to_carry() {
        // Arrange
        let (first, second) = (node_row("node-1", "alpha"), node_row("node-2", "beta"));
        let events = vec![
            log_event(
                "event-1",
                LogPayload::EpisodeBatch(vec![RowEffect::Node(first), RowEffect::Node(second)]),
            ),
            voided("event-2", "event-1"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
    }

    #[test]
    fn voiding_a_later_carrier_leaves_the_first_carriers_entry_alone() {
        // Arrange
        let node = node_row("node-1", "alpha");
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(node.clone())),
            log_event(
                "event-2",
                LogPayload::NodeWrite(node_row("node-2", "alpha")),
            ),
            voided("event-3", "event-2"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![IndexEntry {
                hash: expected_node_hash(&node),
                first_event_id: "event-1".into(),
                row_ids: vec!["node-1".into()],
            }]
        );
    }

    #[test]
    fn a_void_for_an_unknown_event_changes_nothing() {
        // Arrange
        let events = vec![
            log_event(
                "event-1",
                LogPayload::NodeWrite(node_row("node-1", "alpha")),
            ),
            voided("event-2", "event-404"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(rebuilt.entries.len(), 1);
    }

    #[test]
    fn an_unsupplied_valid_from_is_hashed_as_absent_so_rebuild_matches_the_write_path() {
        // Arrange: the write path hashed the request without a valid_from, and
        // the stored row carries the default the store filled in.
        let row = NodeRow {
            valid_from: 1_700_000_000_000,
            valid_from_supplied: false,
            ..node_row("node-1", "alpha")
        };
        let write_path_hash = hash_node(&NodeContent {
            kind: row.kind.clone(),
            label: row.label.clone(),
            content: row.content.clone(),
            producer: row.producer.clone(),
            scope: row.scope.clone(),
            subject: row.subject.clone(),
            attributes: row.attributes.clone(),
            valid_from: None,
            confidence: row.confidence,
        });
        let mut incremental = HashBloom::new(small_config());
        incremental.insert(&write_path_hash);
        let events = vec![log_event("event-1", LogPayload::NodeWrite(row))];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(rebuilt.bloom, incremental);
        assert_eq!(
            rebuilt.entries,
            vec![IndexEntry {
                hash: write_path_hash,
                first_event_id: "event-1".into(),
                row_ids: vec!["node-1".into()],
            }]
        );
    }

    #[test]
    fn a_supplied_valid_from_is_part_of_the_rebuilt_hash() {
        // Arrange
        let supplied = node_row("node-1", "alpha");
        let unsupplied = NodeRow {
            valid_from_supplied: false,
            ..node_row("node-2", "alpha")
        };
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(supplied)),
            log_event("event-2", LogPayload::NodeWrite(unsupplied)),
        ];

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(rebuilt.entries.len(), 2);
    }

    fn mixed_events(count: usize) -> Vec<LogEvent> {
        (0..count)
            .map(|position| {
                let id = format!("event-{position}");
                let payload = match position % 3 {
                    0 => LogPayload::NodeWrite(node_row(&format!("node-{position}"), &id)),
                    1 => LogPayload::EdgeWrite(edge_row(&format!("edge-{position}"), &id)),
                    _ => LogPayload::EpisodeBatch(vec![
                        RowEffect::Node(node_row(&format!("node-{position}"), &id)),
                        RowEffect::Edge(edge_row(&format!("edge-{position}"), &format!("{id}b"))),
                    ]),
                };
                log_event(&id, payload)
            })
            .collect()
    }

    fn appended_hashes(event: &LogEvent) -> Vec<[u8; 32]> {
        match &event.payload {
            LogPayload::NodeWrite(row) => vec![expected_node_hash(row)],
            LogPayload::EdgeWrite(row) => vec![expected_edge_hash(row)],
            LogPayload::EpisodeBatch(effects) => effects
                .iter()
                .map(|effect| match effect {
                    RowEffect::Node(row) => expected_node_hash(row),
                    RowEffect::Edge(row) => expected_edge_hash(row),
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn rebuilt_filter_equals_the_one_built_incrementally_during_the_same_appends() {
        // Arrange
        let events = mixed_events(300);
        let mut incremental = HashBloom::new(small_config());
        for event in &events {
            for hash in appended_hashes(event) {
                incremental.insert(&hash);
            }
        }
        let probes: Vec<[u8; 32]> = sample_hashes(3_000_000..3_001_000);

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config()).bloom;

        // Assert
        assert_eq!(rebuilt, incremental);
        let rebuilt_answers: Vec<bool> = probes.iter().map(|h| rebuilt.might_contain(h)).collect();
        let incremental_answers: Vec<bool> = probes
            .iter()
            .map(|h| incremental.might_contain(h))
            .collect();
        assert_eq!(rebuilt_answers, incremental_answers);
    }

    #[test]
    fn rebuild_never_loses_a_hash_that_an_append_inserted() {
        // Arrange
        let events = mixed_events(300);
        let expected: Vec<([u8; 32], String)> = events
            .iter()
            .flat_map(|event| {
                appended_hashes(event)
                    .into_iter()
                    .map(|hash| (hash, event.event_id.clone()))
            })
            .collect();

        // Act
        let rebuilt = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(expected.len(), 300 + 100);
        let lost: Vec<&String> = expected
            .iter()
            .filter(|(hash, _)| {
                !rebuilt.bloom.might_contain(hash) || entry_for(&rebuilt, hash).is_none()
            })
            .map(|(_, event_id)| event_id)
            .collect();
        assert!(lost.is_empty(), "hashes lost for events {lost:?}");
        assert_eq!(rebuilt.entries.len(), expected.len());
    }
}
