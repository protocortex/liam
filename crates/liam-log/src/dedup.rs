// SPDX-License-Identifier: Apache-2.0

//! Bloom pre-check and exact hash index for write deduplication.
//!
//! The filter answers "possibly seen" cheaply; the index decides, and a rebuild
//! from the log restores both after a restart.

use std::collections::{HashMap, HashSet};
#[cfg(any(test, feature = "test-support"))]
use std::convert::Infallible;

use async_trait::async_trait;
use fastbloom::BloomFilter;

use crate::event::{LogEvent, LogPayload};
use crate::hash::content_hashes;

/// A 100k-hash filter takes about 120 KB, so an empty store costs almost nothing.
const DEFAULT_EXPECTED_ITEMS: usize = 100_000;

/// At 1% about 99 of 100 new hashes skip the index lookup, for roughly 10 bits per hash.
const DEFAULT_FALSE_POSITIVE_RATE: f64 = 0.01;

/// About 600 MB at the default rate, so a mistyped capacity cannot ask for tens of GiB.
const MAX_EXPECTED_ITEMS: usize = 500_000_000;

/// Below this the filter needs over 43 bits per hash, which defeats a cheap pre-check.
const MIN_FALSE_POSITIVE_RATE: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq)]
pub struct BloomConfig {
    expected_items: usize,
    false_positive_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum BloomConfigError {
    #[error("false positive rate {0} must be finite, above 0 and below 1")]
    FalsePositiveRate(f64),
    #[error(
        "false positive rate {0} is below the minimum of {min}",
        min = MIN_FALSE_POSITIVE_RATE
    )]
    FalsePositiveRateTooLow(f64),
    #[error("expected items must be at least 1")]
    NoExpectedItems,
    #[error("expected items {0} exceed the maximum of {max}", max = MAX_EXPECTED_ITEMS)]
    TooManyExpectedItems(usize),
}

impl BloomConfig {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Result<Self, BloomConfigError> {
        if !(false_positive_rate.is_finite()
            && false_positive_rate > 0.0
            && false_positive_rate < 1.0)
        {
            return Err(BloomConfigError::FalsePositiveRate(false_positive_rate));
        }
        if false_positive_rate < MIN_FALSE_POSITIVE_RATE {
            return Err(BloomConfigError::FalsePositiveRateTooLow(
                false_positive_rate,
            ));
        }
        if expected_items == 0 {
            return Err(BloomConfigError::NoExpectedItems);
        }
        if expected_items > MAX_EXPECTED_ITEMS {
            return Err(BloomConfigError::TooManyExpectedItems(expected_items));
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
            expected_items: DEFAULT_EXPECTED_ITEMS,
            false_positive_rate: DEFAULT_FALSE_POSITIVE_RATE,
        }
    }
}

/// Fixed seed so the same hashes always produce the same filter, across runs.
const BLOOM_SEED: u128 = 0x6c69_616d_626c_6f6f_6d73_6565_6430_3031;

/// Cheap pre-check in front of the exact lookup. May answer true for a hash
/// that was never inserted, never false for one that was.
pub trait PreCheck {
    fn might_contain(&self, hash: &[u8; 32]) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedWrite {
    pub first_event_id: String,
    /// A batch can carry identical rows, so the store keeps every row id of
    /// the first carrier.
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

/// The first write recorded for `hash`, or `None` when it is new. The pre-check
/// short-circuits a miss without touching the index, and the exact index
/// decides every pre-check hit.
///
/// The caller decides liveness: a hit counts as a duplicate only if the indexed
/// row is still live.
pub async fn find_first_write<P: PreCheck + ?Sized, I: HashIndex + ?Sized>(
    pre_check: &P,
    index: &mut I,
    hash: &[u8; 32],
) -> Result<Option<IndexedWrite>, I::Error> {
    if !pre_check.might_contain(hash) {
        return Ok(None);
    }
    index.lookup(hash).await
}

#[derive(Debug, Clone, PartialEq)]
pub struct HashBloom {
    filter: BloomFilter,
    config: BloomConfig,
}

impl HashBloom {
    pub fn new(config: BloomConfig) -> Self {
        let filter = BloomFilter::with_false_pos(config.false_positive_rate)
            .seed(&BLOOM_SEED)
            .expected_items(config.expected_items);
        Self { filter, config }
    }

    /// Rebuilds a filter from persisted bits, under the same pinned seed.
    /// `bits` must be non-empty and `num_hashes` at least 1.
    pub(crate) fn from_parts(bits: Vec<u64>, num_hashes: u32, config: BloomConfig) -> Self {
        let filter = BloomFilter::from_vec(bits)
            .seed(&BLOOM_SEED)
            .hashes(num_hashes);
        Self { filter, config }
    }

    /// The bit words and hash count, the inverse of `from_parts`.
    pub(crate) fn to_parts(&self) -> (Vec<u64>, u32) {
        (self.filter.iter().collect(), self.filter.num_hashes())
    }

    /// The sizing this filter was built with.
    pub fn config(&self) -> &BloomConfig {
        &self.config
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub hash: [u8; 32],
    pub write: IndexedWrite,
}

/// Maps a content hash to the first write that carried it.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct InMemoryHashIndex {
    writes: HashMap<[u8; 32], IndexedWrite>,
}

#[cfg(any(test, feature = "test-support"))]
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
        self.writes.entry(entry.hash).or_insert(entry.write);
    }
}

#[cfg(any(test, feature = "test-support"))]
#[async_trait]
impl HashIndex for InMemoryHashIndex {
    type Error = Infallible;

    async fn lookup(&mut self, hash: &[u8; 32]) -> Result<Option<IndexedWrite>, Infallible> {
        Ok(self.writes.get(hash).cloned())
    }
}

/// The bloom filter every inserted hash went into, and one index entry per
/// hash that still has a live carrier.
#[derive(Debug)]
pub struct Rebuilt {
    pub bloom: HashBloom,
    pub entries: Vec<IndexEntry>,
}

/// Scans every event once, in log order, and returns the bloom filter and
/// index entries that incremental appends would have produced.
///
/// Each hash maps to its last surviving carrier. A write whose hash already
/// had a live carrier is logged as a `DuplicateOf` and registers nothing, so a
/// later write of the same hash only exists because every earlier carrier was
/// dead or voided, and the later write is the one the live index points at.
/// A `Voided` event removes only its own target, so a retry of a voided write
/// survives however often the void repeats. Entries come out in the log
/// order of the surviving carrier.
///
/// An event whose id was already seen is a replayed copy and is skipped, as
/// is one whose id was voided earlier in the scan. The bloom keeps the hash of
/// a voided write: it cannot delete, and a stale hit is overruled by the exact
/// index. Tombstone and duplicate events carry no content, so add no hash.
pub fn rebuild_from_log(
    events: impl IntoIterator<Item = LogEvent>,
    config: BloomConfig,
) -> Rebuilt {
    let mut bloom = HashBloom::new(config);
    let mut carriers = Carriers::default();
    let mut seen = HashSet::new();
    let mut voided = HashSet::new();
    for event in events {
        if !seen.insert(event.event_id.clone()) {
            continue;
        }
        if let LogPayload::Voided { target_event_id } = event.payload {
            voided.insert(target_event_id);
            continue;
        }
        if voided.contains(&event.event_id) {
            continue;
        }
        for (hash, row_id) in content_hashes(&event) {
            bloom.insert(&hash);
            carriers.record(hash, &event.event_id, row_id);
        }
    }
    drop(seen);
    Rebuilt {
        bloom,
        entries: carriers.into_entries(&voided),
    }
}

struct Carrier {
    event_id: String,
    row_ids: Vec<String>,
    position: u64,
}

/// Every event that carried each hash, oldest first, with the log position
/// that orders the final entries.
#[derive(Default)]
struct Carriers {
    next_position: u64,
    by_hash: HashMap<[u8; 32], Vec<Carrier>>,
}

impl Carriers {
    fn record(&mut self, hash: [u8; 32], event_id: &str, row_id: String) {
        let carriers = self.by_hash.entry(hash).or_default();
        match carriers.last_mut() {
            Some(last) if last.event_id == event_id => last.row_ids.push(row_id),
            _ => {
                carriers.push(Carrier {
                    event_id: event_id.to_owned(),
                    row_ids: vec![row_id],
                    position: self.next_position,
                });
                self.next_position += 1;
            }
        }
    }

    fn into_entries(self, voided: &HashSet<String>) -> Vec<IndexEntry> {
        let mut surviving: Vec<(u64, IndexEntry)> = self
            .by_hash
            .into_iter()
            .filter_map(|(hash, carriers)| {
                let carrier = carriers
                    .into_iter()
                    .rev()
                    .find(|carrier| !voided.contains(&carrier.event_id))?;
                let write = IndexedWrite {
                    first_event_id: carrier.event_id,
                    row_ids: carrier.row_ids,
                };
                Some((carrier.position, IndexEntry { hash, write }))
            })
            .collect();
        surviving.sort_unstable_by_key(|(position, _)| *position);
        surviving.into_iter().map(|(_, entry)| entry).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EdgeRow, NodeRow, RowEffect, TombstoneTable, TombstoneTarget};
    use crate::fixtures::{
        edge_row, filter_with, log_event, node_row, sample_hash, sample_hashes, small_config,
        EVENT_HASH_SENTINEL,
    };
    use crate::hash::{content_hashes, hash_edge, hash_node, EdgeContent, NodeContent};

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

    fn assert_send<T: Send>(value: T) -> T {
        value
    }

    fn indexed_write(event_id: &str, row_ids: &[&str]) -> IndexedWrite {
        IndexedWrite {
            first_event_id: event_id.into(),
            row_ids: row_ids.iter().map(|id| (*id).to_owned()).collect(),
        }
    }

    fn entry(hash: [u8; 32], event_id: &str, row_ids: &[&str]) -> IndexEntry {
        IndexEntry {
            hash,
            write: indexed_write(event_id, row_ids),
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
    fn bloom_config_enforces_the_item_and_rate_limits_at_their_boundaries() {
        // Arrange
        let below_floor = MIN_FALSE_POSITIVE_RATE * 0.9;
        let cases = [
            (1, 0.01, Ok(())),
            (MAX_EXPECTED_ITEMS, 0.01, Ok(())),
            (100, MIN_FALSE_POSITIVE_RATE, Ok(())),
            (0, 0.01, Err(BloomConfigError::NoExpectedItems)),
            (
                MAX_EXPECTED_ITEMS + 1,
                0.01,
                Err(BloomConfigError::TooManyExpectedItems(
                    MAX_EXPECTED_ITEMS + 1,
                )),
            ),
            (
                usize::MAX,
                0.01,
                Err(BloomConfigError::TooManyExpectedItems(usize::MAX)),
            ),
            (
                100,
                below_floor,
                Err(BloomConfigError::FalsePositiveRateTooLow(below_floor)),
            ),
            (
                100,
                f64::MIN_POSITIVE,
                Err(BloomConfigError::FalsePositiveRateTooLow(f64::MIN_POSITIVE)),
            ),
        ];

        for (items, rate, expected) in cases {
            // Act
            let result = BloomConfig::new(items, rate).map(|_| ());

            // Assert
            assert_eq!(result, expected, "items {items}, rate {rate}");
        }
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
    fn the_default_config_is_the_named_defaults_and_passes_validation() {
        // Arrange / Act
        let validated = BloomConfig::new(DEFAULT_EXPECTED_ITEMS, DEFAULT_FALSE_POSITIVE_RATE);

        // Assert
        assert_eq!(validated, Ok(BloomConfig::default()));
    }

    #[test]
    fn filters_built_from_the_same_hashes_are_equal_in_any_insertion_order() {
        // Arrange: 7919 is prime and shares no factor with 2000, so the stride
        // visits every hash once in an order unlike forward or reverse.
        let hashes = sample_hashes(0..2_000);
        let strided: Vec<[u8; 32]> = (0..hashes.len())
            .map(|position| hashes[position * 7_919 % hashes.len()])
            .collect();
        let reversed: Vec<[u8; 32]> = hashes.iter().rev().copied().collect();
        assert_ne!(strided, hashes, "the stride must reorder the hashes");
        assert_ne!(strided, reversed, "the stride must differ from reversing");

        // Act
        let forward_filter = filter_with(small_config(), &hashes);
        let strided_filter = filter_with(small_config(), &strided);
        let reversed_filter = filter_with(small_config(), &reversed);

        // Assert
        assert_eq!(forward_filter, strided_filter);
        assert_eq!(forward_filter, reversed_filter);
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
    fn the_pinned_seed_gives_the_recorded_false_positive_count() {
        // Arrange: the seed and the probe set are fixed, so the count is one
        // repeatable value, 207 of 20000 (about 0.0104 for a configured 0.01).
        // A changed seed, hasher, or sizing moves it.
        let filter = filter_with(small_config(), &sample_hashes(0..5_000));
        let absent = sample_hashes(1_000_000..1_020_000);

        // Act
        let false_positives = absent
            .iter()
            .filter(|hash| filter.might_contain(hash))
            .count();

        // Assert
        assert_eq!(false_positives, 207);
    }

    #[test]
    fn the_pinned_seed_gives_the_recorded_answers_for_a_fixed_absent_set() {
        // Arrange: 1 marks an absent hash the filter wrongly reports present.
        let filter = filter_with(small_config(), &sample_hashes(0..5_000));
        let absent = sample_hashes(1_000_000..1_000_200);

        // Act
        let answers: String = absent
            .iter()
            .map(|hash| if filter.might_contain(hash) { '1' } else { '0' })
            .collect();

        // Assert
        assert_eq!(
            answers,
            concat!(
                "00000000000000000000000000000000000000000000000000",
                "00000000000000000000000000000000000000000000000001",
                "00000000000000100000000000000000000000000000000000",
                "00000000000100000000000000000000000000000000000000",
            )
        );
    }

    #[tokio::test]
    async fn a_pre_check_hit_for_a_never_stored_hash_is_decided_by_the_exact_index() {
        // Arrange
        let mut index = CountingIndex::default();
        let never_stored = sample_hash(42);

        // Act
        let found = find_first_write(&AlwaysTrue, &mut index, &never_stored).await;

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
        let write = indexed_write("event-7", &["node-7"]);
        let mut index = CountingIndex {
            stored: vec![(stored, write.clone())],
            ..CountingIndex::default()
        };

        // Act
        let found = find_first_write(&AlwaysTrue, &mut index, &stored).await;

        // Assert
        assert_eq!(found, Ok(Some(write)));
    }

    #[tokio::test]
    async fn a_pre_check_miss_skips_the_exact_index() {
        // Arrange
        let mut index = CountingIndex::default();

        // Act
        let found = find_first_write(&AlwaysFalse, &mut index, &sample_hash(1)).await;

        // Assert
        assert_eq!(found, Ok(None));
        assert_eq!(index.lookups, 0, "a definite miss needs no index lookup");
    }

    #[tokio::test]
    async fn an_index_failure_is_reported_not_treated_as_a_miss() {
        // Arrange
        let mut index = FailingIndex;

        // Act
        let found = find_first_write(&AlwaysTrue, &mut index, &sample_hash(1)).await;

        // Assert
        assert_eq!(found, Err("index unavailable"));
    }

    #[tokio::test]
    async fn the_lookup_future_is_send_and_accepts_trait_objects() {
        // Arrange
        let stored = sample_hash(7);
        let write = indexed_write("event-7", &["node-7"]);
        let mut index = CountingIndex {
            stored: vec![(stored, write.clone())],
            ..CountingIndex::default()
        };
        let dyn_pre_check: &dyn PreCheck = &AlwaysTrue;
        let dyn_index: &mut dyn HashIndex<Error = Infallible> = &mut index;

        // Act
        let via_objects = find_first_write(dyn_pre_check, dyn_index, &stored).await;
        let send_future = assert_send(find_first_write(&AlwaysTrue, &mut index, &stored));

        // Assert
        assert_eq!(via_objects, Ok(Some(write.clone())));
        assert_eq!(send_future.await, Ok(Some(write)));
    }

    fn voided(event_id: &str, target: &str) -> LogEvent {
        log_event(
            event_id,
            LogPayload::Voided {
                target_event_id: target.into(),
            },
        )
    }

    fn node_write(event_id: &str, row_id: &str, content: &str) -> LogEvent {
        log_event(event_id, LogPayload::NodeWrite(node_row(row_id, content)))
    }

    fn expected_node_hash(row: &NodeRow) -> [u8; 32] {
        content_hashes(&log_event("probe", LogPayload::NodeWrite(row.clone())))[0].0
    }

    fn expected_edge_hash(row: &EdgeRow) -> [u8; 32] {
        content_hashes(&log_event("probe", LogPayload::EdgeWrite(row.clone())))[0].0
    }

    fn alpha_hash() -> [u8; 32] {
        expected_node_hash(&node_row("any", "alpha"))
    }

    fn entry_for<'a>(rebuilt: &'a Rebuilt, hash: &[u8; 32]) -> Option<&'a IndexEntry> {
        rebuilt.entries.iter().find(|entry| &entry.hash == hash)
    }

    #[tokio::test]
    async fn the_in_memory_index_looks_up_the_full_stored_write() {
        // Arrange
        let mut index = InMemoryHashIndex::from_entries([entry(
            sample_hash(1),
            "event-1",
            &["node-1", "edge-1"],
        )]);

        // Act
        let hit = index.lookup(&sample_hash(1)).await;
        let miss = index.lookup(&sample_hash(2)).await;

        // Assert
        assert_eq!(
            hit,
            Ok(Some(indexed_write("event-1", &["node-1", "edge-1"])))
        );
        assert_eq!(miss, Ok(None));
    }

    #[tokio::test]
    async fn a_second_insert_of_the_same_hash_keeps_the_first_write_and_its_row_ids() {
        // Arrange
        let mut index = InMemoryHashIndex::new();
        let hash = sample_hash(1);
        index.insert(entry(hash, "event-1", &["node-1", "node-1b"]));

        // Act
        index.insert(entry(hash, "event-2", &["node-2"]));

        // Assert
        assert_eq!(
            index.lookup(&hash).await,
            Ok(Some(indexed_write("event-1", &["node-1", "node-1b"])))
        );
    }

    #[tokio::test]
    async fn from_entries_keeps_the_first_entry_of_a_repeated_hash() {
        // Arrange
        let entries = [
            entry(sample_hash(1), "event-1", &["node-1"]),
            entry(sample_hash(2), "event-2", &["node-2"]),
            entry(sample_hash(1), "event-3", &["node-3"]),
        ];

        // Act
        let mut index = InMemoryHashIndex::from_entries(entries);

        // Assert
        assert_eq!(
            index.lookup(&sample_hash(1)).await,
            Ok(Some(indexed_write("event-1", &["node-1"])))
        );
        assert_eq!(
            index.lookup(&sample_hash(2)).await,
            Ok(Some(indexed_write("event-2", &["node-2"])))
        );
        assert_eq!(index.lookup(&sample_hash(3)).await, Ok(None));
    }

    #[test]
    fn rebuilding_from_an_empty_log_gives_an_empty_filter_and_no_entries() {
        // Arrange
        let events: Vec<LogEvent> = Vec::new();

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

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
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        let node_hash = hash_node(&NodeContent {
            kind: "fact".into(),
            label: "label".into(),
            content: "alpha".into(),
            producer: "agent-a".into(),
            scope: Some("proj/a".into()),
            subject: None,
            attributes: "{}".into(),
            valid_from: Some(1_000),
            confidence: 0.75,
        });
        let edge_hash = hash_edge(&EdgeContent {
            src: "node-1".into(),
            dst: "node-2".into(),
            edge_type: "relates_to".into(),
        });
        assert_eq!(
            rebuilt.entries,
            vec![
                entry(node_hash, "event-1", &["node-1"]),
                entry(edge_hash, "event-2", &["edge-1"]),
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
        let rebuilt = rebuild_from_log(vec![batch], small_config());

        // Assert
        for (hash, row_id) in [
            (expected_node_hash(&first), "node-1"),
            (expected_node_hash(&second), "node-2"),
            (expected_edge_hash(&edge), "edge-1"),
        ] {
            assert_eq!(
                entry_for(&rebuilt, &hash),
                Some(&entry(hash, "event-1", &[row_id]))
            );
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
        let rebuilt = rebuild_from_log(vec![batch], small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![entry(
                expected_node_hash(&first),
                "event-1",
                &["node-1", "node-2"]
            )]
        );
    }

    #[test]
    fn an_empty_episode_batch_adds_nothing() {
        // Arrange
        let batch = log_event("event-1", LogPayload::EpisodeBatch(Vec::new()));

        // Act
        let rebuilt = rebuild_from_log(vec![batch], small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
        assert_eq!(rebuilt.bloom, HashBloom::new(small_config()));
    }

    #[test]
    fn a_hash_rewritten_after_a_supersede_maps_to_its_last_carrier() {
        // Arrange: X, then Y superseding it, then X again. The third write
        // exists only because the first carrier of X was no longer live.
        let first = node_row("node-1", "alpha");
        let events = vec![
            node_write("event-1", "node-1", "alpha"),
            node_write("event-2", "node-2", "beta"),
            node_write("event-3", "node-3", "alpha"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![
                entry(
                    expected_node_hash(&node_row("node-2", "beta")),
                    "event-2",
                    &["node-2"]
                ),
                entry(expected_node_hash(&first), "event-3", &["node-3"]),
            ]
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
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert_eq!(rebuilt.entries.len(), 1);
        assert_eq!(rebuilt.entries[0].write.first_event_id, "event-2");
    }

    #[test]
    fn entries_follow_the_log_order_of_their_carriers_not_the_order_of_their_ids() {
        // Arrange: the ids sort differently as text (event-10 before event-2
        // before event-9) than they appear in the log.
        let ids = [
            "event-9",
            "event-10",
            "event-2",
            "event-33",
            "event-4",
            "event-100",
        ];
        let events: Vec<LogEvent> = ids
            .iter()
            .enumerate()
            .map(|(position, id)| node_write(id, &format!("node-{position}"), id))
            .collect();

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        let order: Vec<&str> = rebuilt
            .entries
            .iter()
            .map(|entry| entry.write.first_event_id.as_str())
            .collect();
        assert_eq!(order, ids);
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
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
        assert_eq!(rebuilt.bloom, HashBloom::new(small_config()));
    }

    #[test]
    fn a_voided_event_removes_its_index_entry_but_the_bloom_keeps_the_hash() {
        // Arrange
        let events = vec![
            node_write("event-1", "node-1", "alpha"),
            voided("event-2", "event-1"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
        assert!(rebuilt.bloom.might_contain(&alpha_hash()));
    }

    #[test]
    fn a_retry_after_a_void_becomes_the_carrier() {
        // Arrange
        let events = vec![
            node_write("event-1", "node-1", "alpha"),
            voided("event-2", "event-1"),
            node_write("event-3", "node-2", "alpha"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![entry(alpha_hash(), "event-3", &["node-2"])]
        );
    }

    #[test]
    fn a_repeated_void_of_the_same_target_keeps_the_retry() {
        // Arrange
        let events = vec![
            node_write("event-1", "node-1", "alpha"),
            voided("event-2", "event-1"),
            node_write("event-3", "node-2", "alpha"),
            voided("event-4", "event-1"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![entry(alpha_hash(), "event-3", &["node-2"])]
        );
    }

    #[test]
    fn a_replayed_event_id_is_applied_once() {
        // Arrange
        let batch = log_event(
            "event-1",
            LogPayload::EpisodeBatch(vec![
                RowEffect::Node(node_row("node-1", "alpha")),
                RowEffect::Node(node_row("node-2", "beta")),
            ]),
        );
        let events = vec![batch.clone(), batch];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![
                entry(alpha_hash(), "event-1", &["node-1"]),
                entry(
                    expected_node_hash(&node_row("node-2", "beta")),
                    "event-1",
                    &["node-2"]
                ),
            ]
        );
    }

    #[test]
    fn an_event_voided_earlier_in_the_scan_is_skipped() {
        // Arrange
        let events = vec![
            voided("event-1", "event-2"),
            node_write("event-2", "node-1", "alpha"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
        assert_eq!(rebuilt.bloom, HashBloom::new(small_config()));
    }

    #[test]
    fn voiding_a_batch_removes_every_entry_it_carried() {
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
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert!(rebuilt.entries.is_empty());
    }

    #[test]
    fn voiding_a_later_carrier_leaves_the_earlier_carriers_entry() {
        // Arrange
        let events = vec![
            node_write("event-1", "node-1", "alpha"),
            node_write("event-2", "node-2", "alpha"),
            voided("event-3", "event-2"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert_eq!(
            rebuilt.entries,
            vec![entry(alpha_hash(), "event-1", &["node-1"])]
        );
    }

    #[test]
    fn a_void_for_an_unknown_event_changes_nothing() {
        // Arrange
        let events = vec![
            node_write("event-1", "node-1", "alpha"),
            voided("event-2", "event-404"),
        ];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

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
            kind: "fact".into(),
            label: "label".into(),
            content: "alpha".into(),
            producer: "agent-a".into(),
            scope: Some("proj/a".into()),
            subject: None,
            attributes: "{}".into(),
            valid_from: None,
            confidence: 0.75,
        });
        let mut incremental = HashBloom::new(small_config());
        incremental.insert(&write_path_hash);
        let events = vec![log_event("event-1", LogPayload::NodeWrite(row))];

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

        // Assert
        assert_eq!(rebuilt.bloom, incremental);
        assert_eq!(
            rebuilt.entries,
            vec![entry(write_path_hash, "event-1", &["node-1"])]
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
        let rebuilt = rebuild_from_log(events, small_config());

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

    #[test]
    fn rebuilt_filter_equals_the_one_built_incrementally_during_the_same_appends() {
        // Arrange
        let events = mixed_events(300);
        let mut incremental = HashBloom::new(small_config());
        for event in &events {
            for (hash, _) in content_hashes(event) {
                incremental.insert(&hash);
            }
        }

        // Act
        let rebuilt = rebuild_from_log(events, small_config()).bloom;

        // Assert
        assert_eq!(rebuilt, incremental);
    }

    #[test]
    fn rebuild_never_loses_a_hash_that_an_append_inserted() {
        // Arrange
        let events = mixed_events(300);
        let expected: Vec<([u8; 32], String)> = events
            .iter()
            .flat_map(|event| {
                content_hashes(event)
                    .into_iter()
                    .map(|(hash, _)| (hash, event.event_id.clone()))
            })
            .collect();

        // Act
        let rebuilt = rebuild_from_log(events, small_config());

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
