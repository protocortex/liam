// SPDX-License-Identifier: Apache-2.0

//! Content hashing for write deduplication.
//!
//! The hash covers what the caller asked to store, never server-minted ids or
//! timestamps, so the same request always hashes the same.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};

use fastbloom::BloomFilter;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::event::{EdgeRow, LogEvent, LogPayload, NodeRow, RowEffect};

/// What a node write asks to store. `valid_from` is `None` when the caller did
/// not supply one, so a server-chosen default never perturbs the hash.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeContent {
    pub kind: String,
    pub label: String,
    pub content: String,
    pub producer: String,
    pub scope: Option<String>,
    pub subject: Option<String>,
    pub attributes: String,
    pub valid_from: Option<i64>,
}

/// What an edge write asks to store.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeContent {
    pub src: String,
    pub dst: String,
    pub edge_type: String,
    pub producer: String,
    pub attributes: String,
}

const NODE_TAG: u8 = 0x01;
const EDGE_TAG: u8 = 0x02;

/// SHA-256 over the canonical encoding of a node write.
pub fn hash_node(node: &NodeContent) -> [u8; 32] {
    let mut buf = vec![NODE_TAG];
    put_str(&mut buf, &node.kind);
    put_str(&mut buf, &node.label);
    put_str(&mut buf, &node.content);
    put_str(&mut buf, &node.producer);
    put_opt_str(&mut buf, node.scope.as_deref());
    put_opt_str(&mut buf, node.subject.as_deref());
    put_str(&mut buf, &node.attributes);
    match node.valid_from {
        None => buf.push(0),
        Some(millis) => {
            buf.push(1);
            buf.extend(millis.to_le_bytes());
        }
    }
    Sha256::digest(&buf).into()
}

/// SHA-256 over the canonical encoding of an edge write.
pub fn hash_edge(edge: &EdgeContent) -> [u8; 32] {
    let mut buf = vec![EDGE_TAG];
    put_str(&mut buf, &edge.src);
    put_str(&mut buf, &edge.dst);
    put_str(&mut buf, &edge.edge_type);
    put_str(&mut buf, &edge.producer);
    put_str(&mut buf, &edge.attributes);
    Sha256::digest(&buf).into()
}

// Length prefixes keep field boundaries unambiguous, so moving bytes between
// adjacent fields changes the hash.
fn put_str(buf: &mut Vec<u8>, value: &str) {
    buf.extend((value.len() as u64).to_le_bytes());
    buf.extend(value.as_bytes());
}

fn put_opt_str(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        None => buf.push(0),
        Some(text) => {
            buf.push(1);
            put_str(buf, text);
        }
    }
}

/// Sizing for the bloom filter: capacity and the false-positive rate it is tuned for.
#[derive(Debug, Clone, PartialEq)]
pub struct BloomConfig {
    pub expected_items: usize,
    pub false_positive_rate: f64,
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
pub const BLOOM_SEED: u128 = 0x6c69_616d_626c_6f6f_6d73_6565_6430_3031;

/// Cheap pre-check in front of the exact lookup. May answer true for a hash
/// that was never inserted, never false for one that was.
pub trait PreCheck {
    fn might_contain(&self, hash: &[u8; 32]) -> bool;
}

/// Exact lookup against the log's hash index; decides the final answer.
pub trait HashIndex {
    fn contains_hash(&self, hash: &[u8; 32]) -> bool;
}

/// True when `hash` is already stored: the pre-check short-circuits a miss,
/// and the exact index decides every pre-check hit.
pub fn is_duplicate(pre_check: &dyn PreCheck, index: &dyn HashIndex, hash: &[u8; 32]) -> bool {
    pre_check.might_contain(hash) && index.contains_hash(hash)
}

/// Bloom filter over content hashes with a pinned seed.
#[derive(Debug, Clone, PartialEq)]
pub struct HashBloom {
    config: BloomConfig,
    filter: BloomFilter,
}

/// What lands on disk: the config the filter was sized with travels alongside
/// it, so a changed config is detected instead of silently misread.
#[derive(Serialize, Deserialize)]
struct PersistedFilter {
    expected_items: u64,
    false_positive_rate: f64,
    filter: BloomFilter,
}

const FILE_MAGIC: [u8; 4] = *b"LBF1";
const HEADER_LEN: usize = FILE_MAGIC.len() + size_of::<u32>();

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

    /// Writes magic, checksum, then the encoded filter to a temp file beside
    /// `path` and renames it into place, so a crash never leaves a partial file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let persisted = PersistedFilter {
            expected_items: self.config.expected_items as u64,
            false_positive_rate: self.config.false_positive_rate,
            filter: self.filter.clone(),
        };
        let payload = postcard::to_stdvec(&persisted).map_err(io::Error::other)?;
        let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
        bytes.extend(FILE_MAGIC);
        bytes.extend(crc32fast::hash(&payload).to_le_bytes());
        bytes.extend(payload);

        let mut staged = path.as_os_str().to_owned();
        staged.push(".tmp");
        let staged = PathBuf::from(staged);
        let mut file = File::create(&staged)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&staged, path)?;
        let dir = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        File::open(dir)?.sync_all()
    }

    /// Never panics: any unreadable, damaged, or differently configured file
    /// comes back as `Stale` so the caller rebuilds from the log.
    pub fn load(path: &Path, config: &BloomConfig) -> FilterLoad {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return FilterLoad::Stale(StaleReason::Missing)
            }
            Err(_) => return FilterLoad::Stale(StaleReason::Corrupt),
        };
        let Some(persisted) = decode(&bytes) else {
            return FilterLoad::Stale(StaleReason::Corrupt);
        };
        if persisted.expected_items != config.expected_items as u64
            || persisted.false_positive_rate != config.false_positive_rate
        {
            return FilterLoad::Stale(StaleReason::ConfigMismatch);
        }
        FilterLoad::Fresh(Self {
            config: config.clone(),
            filter: persisted.filter,
        })
    }
}

fn decode(bytes: &[u8]) -> Option<PersistedFilter> {
    let (magic, rest) = bytes.split_at_checked(FILE_MAGIC.len())?;
    let (checksum, payload) = rest.split_at_checked(size_of::<u32>())?;
    if magic != FILE_MAGIC || checksum != crc32fast::hash(payload).to_le_bytes() {
        return None;
    }
    postcard::from_bytes(payload).ok()
}

impl PreCheck for HashBloom {
    fn might_contain(&self, hash: &[u8; 32]) -> bool {
        self.filter.contains(hash)
    }
}

/// Why a persisted filter cannot be used as is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleReason {
    Missing,
    Corrupt,
    ConfigMismatch,
}

/// Result of loading a persisted filter: usable, or stale so the caller
/// rebuilds from the log before serving writes.
#[derive(Debug)]
pub enum FilterLoad {
    Fresh(HashBloom),
    Stale(StaleReason),
}

/// In-memory `HashIndex`: maps a content hash to the first event that carried it.
#[derive(Debug, Default)]
pub struct InMemoryHashIndex {
    first_events: HashMap<[u8; 32], String>,
}

impl InMemoryHashIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keeps the first event id for a hash, so repeats resolve to the original.
    pub fn insert(&mut self, hash: [u8; 32], event_id: &str) {
        self.first_events
            .entry(hash)
            .or_insert_with(|| event_id.to_owned());
    }

    pub fn first_event_id(&self, hash: &[u8; 32]) -> Option<&str> {
        self.first_events.get(hash).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.first_events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.first_events.is_empty()
    }
}

impl HashIndex for InMemoryHashIndex {
    fn contains_hash(&self, hash: &[u8; 32]) -> bool {
        self.first_events.contains_key(hash)
    }
}

/// Scans every event once and returns the bloom filter and hash index that
/// incremental appends would have produced. Tombstone, duplicate and voided
/// events carry no content, so they add no hash.
pub fn rebuild_from_log(
    events: impl Iterator<Item = LogEvent>,
    config: BloomConfig,
) -> (HashBloom, InMemoryHashIndex) {
    let mut bloom = HashBloom::new(config);
    let mut index = InMemoryHashIndex::new();
    for event in events {
        for hash in content_hashes(&event) {
            bloom.insert(&hash);
            index.insert(hash, &event.event_id);
        }
    }
    (bloom, index)
}

// A resolved row cannot say whether the caller supplied `valid_from`, so it is
// treated as supplied: that can only miss a dedup, never merge distinct writes.
// Edge rows carry no producer, so the event's source stands in for it.
fn content_hashes(event: &LogEvent) -> Vec<[u8; 32]> {
    match &event.payload {
        LogPayload::NodeWrite(row) => vec![node_row_hash(row)],
        LogPayload::EdgeWrite(row) => vec![edge_row_hash(row, &event.source)],
        LogPayload::EpisodeBatch(effects) => effects
            .iter()
            .map(|effect| match effect {
                RowEffect::Node(row) => node_row_hash(row),
                RowEffect::Edge(row) => edge_row_hash(row, &event.source),
            })
            .collect(),
        LogPayload::Tombstone(_) | LogPayload::DuplicateOf { .. } | LogPayload::Voided { .. } => {
            Vec::new()
        }
    }
}

fn node_row_hash(row: &NodeRow) -> [u8; 32] {
    hash_node(&NodeContent {
        kind: row.kind.clone(),
        label: row.label.clone(),
        content: row.content.clone(),
        producer: row.producer.clone(),
        scope: row.scope.clone(),
        subject: row.subject.clone(),
        attributes: row.attributes.clone(),
        valid_from: Some(row.valid_from),
    })
}

fn edge_row_hash(row: &EdgeRow, producer: &str) -> [u8; 32] {
    hash_edge(&EdgeContent {
        src: row.src.clone(),
        dst: row.dst.clone(),
        edge_type: row.edge_type.clone(),
        producer: producer.to_owned(),
        attributes: row.attributes.clone(),
    })
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::event::{
        EdgeRow, LogPayload, NodeRow, RowEffect, TombstoneTable, TombstoneTarget,
        CURRENT_SCHEMA_VERSION,
    };

    const NODE_TAG: u8 = 0x01;
    const EDGE_TAG: u8 = 0x02;

    fn node() -> NodeContent {
        NodeContent {
            kind: "fact".into(),
            label: "label".into(),
            content: "content".into(),
            producer: "agent-a".into(),
            scope: Some("proj/a".into()),
            subject: None,
            attributes: "{}".into(),
            valid_from: None,
        }
    }

    fn edge() -> EdgeContent {
        EdgeContent {
            src: "node-1".into(),
            dst: "node-2".into(),
            edge_type: "relates_to".into(),
            producer: "agent-a".into(),
            attributes: "{}".into(),
        }
    }

    fn hex(hash: &[u8; 32]) -> String {
        hash.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    // Independent of the production encoder on purpose: strings are a u64 LE
    // length then bytes, options are a 0/1 presence byte then the value.
    fn put_str(buf: &mut Vec<u8>, value: &str) {
        buf.extend((value.len() as u64).to_le_bytes());
        buf.extend(value.as_bytes());
    }

    fn put_opt_str(buf: &mut Vec<u8>, value: &Option<String>) {
        match value {
            None => buf.push(0),
            Some(text) => {
                buf.push(1);
                put_str(buf, text);
            }
        }
    }

    fn reference_node_hash(node: &NodeContent) -> [u8; 32] {
        let mut buf = vec![NODE_TAG];
        put_str(&mut buf, &node.kind);
        put_str(&mut buf, &node.label);
        put_str(&mut buf, &node.content);
        put_str(&mut buf, &node.producer);
        put_opt_str(&mut buf, &node.scope);
        put_opt_str(&mut buf, &node.subject);
        put_str(&mut buf, &node.attributes);
        match node.valid_from {
            None => buf.push(0),
            Some(millis) => {
                buf.push(1);
                buf.extend(millis.to_le_bytes());
            }
        }
        Sha256::digest(&buf).into()
    }

    fn reference_edge_hash(edge: &EdgeContent) -> [u8; 32] {
        let mut buf = vec![EDGE_TAG];
        put_str(&mut buf, &edge.src);
        put_str(&mut buf, &edge.dst);
        put_str(&mut buf, &edge.edge_type);
        put_str(&mut buf, &edge.producer);
        put_str(&mut buf, &edge.attributes);
        Sha256::digest(&buf).into()
    }

    #[test]
    fn identical_node_writes_without_valid_from_hash_equal() {
        // Arrange
        let (first, second) = (node(), node());

        // Act
        let (first_hash, second_hash) = (hash_node(&first), hash_node(&second));

        // Assert
        assert_eq!(first_hash, second_hash);
        assert_ne!(first_hash, [0; 32], "hash must not be the all-zero stub");
    }

    #[test]
    fn explicit_valid_from_changes_the_node_hash() {
        // Arrange
        let without = node();
        let with = NodeContent {
            valid_from: Some(1_000),
            ..node()
        };

        // Act
        let (without_hash, with_hash) = (hash_node(&without), hash_node(&with));

        // Assert
        assert_ne!(without_hash, with_hash);
    }

    #[test]
    fn different_explicit_valid_from_values_hash_differently() {
        // Arrange
        let early = NodeContent {
            valid_from: Some(1_000),
            ..node()
        };
        let late = NodeContent {
            valid_from: Some(2_000),
            ..node()
        };

        // Act
        let (early_hash, late_hash) = (hash_node(&early), hash_node(&late));

        // Assert
        assert_ne!(early_hash, late_hash);
    }

    #[test]
    fn node_hash_changes_when_any_hashed_field_changes() {
        // Arrange
        let cases: Vec<(&str, NodeContent)> = vec![
            (
                "kind",
                NodeContent {
                    kind: "event".into(),
                    ..node()
                },
            ),
            (
                "label",
                NodeContent {
                    label: "other".into(),
                    ..node()
                },
            ),
            (
                "content",
                NodeContent {
                    content: "other".into(),
                    ..node()
                },
            ),
            (
                "producer",
                NodeContent {
                    producer: "agent-b".into(),
                    ..node()
                },
            ),
            (
                "scope",
                NodeContent {
                    scope: Some("proj/b".into()),
                    ..node()
                },
            ),
            (
                "scope absent",
                NodeContent {
                    scope: None,
                    ..node()
                },
            ),
            (
                "subject",
                NodeContent {
                    subject: Some("user-1".into()),
                    ..node()
                },
            ),
            (
                "attributes",
                NodeContent {
                    attributes: "{\"a\":1}".into(),
                    ..node()
                },
            ),
        ];
        let baseline = hash_node(&node());

        for (field, changed) in cases {
            // Act
            let changed_hash = hash_node(&changed);

            // Assert
            assert_ne!(
                changed_hash, baseline,
                "changing {field} must change the hash"
            );
        }
    }

    #[test]
    fn swapping_values_between_fields_changes_the_node_hash() {
        // Arrange
        let original = NodeContent {
            label: "a".into(),
            content: "b".into(),
            ..node()
        };
        let swapped = NodeContent {
            label: "b".into(),
            content: "a".into(),
            ..node()
        };

        // Act
        let (original_hash, swapped_hash) = (hash_node(&original), hash_node(&swapped));

        // Assert
        assert_ne!(original_hash, swapped_hash);
    }

    #[test]
    fn moving_bytes_across_a_field_boundary_changes_the_node_hash() {
        // Arrange
        let split_early = NodeContent {
            label: "ab".into(),
            content: "c".into(),
            ..node()
        };
        let split_late = NodeContent {
            label: "a".into(),
            content: "bc".into(),
            ..node()
        };

        // Act
        let (early_hash, late_hash) = (hash_node(&split_early), hash_node(&split_late));

        // Assert
        assert_ne!(early_hash, late_hash);
    }

    #[test]
    fn node_hash_matches_the_independently_encoded_reference() {
        // Arrange
        let cases = [
            node(),
            NodeContent {
                valid_from: Some(1_000),
                ..node()
            },
            NodeContent {
                subject: Some("user-1".into()),
                scope: None,
                ..node()
            },
        ];

        for case in cases {
            // Act
            let actual = hash_node(&case);

            // Assert
            assert_eq!(hex(&actual), hex(&reference_node_hash(&case)));
        }
    }

    #[test]
    fn node_hash_golden_values_are_pinned() {
        // Arrange
        let without_valid_from = node();
        let with_valid_from = NodeContent {
            valid_from: Some(1_000),
            ..node()
        };

        // Act
        let (without_hash, with_hash) =
            (hash_node(&without_valid_from), hash_node(&with_valid_from));

        // Assert
        assert_eq!(
            hex(&without_hash),
            "0e3c88bf660ca305d4608bc04f319bc2438485edb4d4fd1f2f9b0b1f0573db3f"
        );
        assert_eq!(
            hex(&with_hash),
            "0af6cff7266fa614de625cb976cca8c45707052624be0ec2e085f8e38f6ac7cd"
        );
    }

    #[test]
    fn identical_edge_writes_hash_equal() {
        // Arrange
        let (first, second) = (edge(), edge());

        // Act
        let (first_hash, second_hash) = (hash_edge(&first), hash_edge(&second));

        // Assert
        assert_eq!(first_hash, second_hash);
        assert_ne!(first_hash, [0; 32], "hash must not be the all-zero stub");
    }

    #[test]
    fn edge_hash_changes_when_any_hashed_field_changes() {
        // Arrange
        let cases: Vec<(&str, EdgeContent)> = vec![
            (
                "src",
                EdgeContent {
                    src: "node-9".into(),
                    ..edge()
                },
            ),
            (
                "dst",
                EdgeContent {
                    dst: "node-9".into(),
                    ..edge()
                },
            ),
            (
                "edge_type",
                EdgeContent {
                    edge_type: "same_as".into(),
                    ..edge()
                },
            ),
            (
                "producer",
                EdgeContent {
                    producer: "agent-b".into(),
                    ..edge()
                },
            ),
            (
                "attributes",
                EdgeContent {
                    attributes: "{\"a\":1}".into(),
                    ..edge()
                },
            ),
        ];
        let baseline = hash_edge(&edge());

        for (field, changed) in cases {
            // Act
            let changed_hash = hash_edge(&changed);

            // Assert
            assert_ne!(
                changed_hash, baseline,
                "changing {field} must change the hash"
            );
        }
    }

    #[test]
    fn edge_hash_matches_the_independently_encoded_reference_and_golden() {
        // Arrange
        let content = edge();

        // Act
        let actual = hash_edge(&content);

        // Assert
        assert_eq!(hex(&actual), hex(&reference_edge_hash(&content)));
        assert_eq!(
            hex(&actual),
            "fc9898302f90dc88d9bb7f7dc756fa097c404536c501c0d004b2edc15db0048c"
        );
    }

    #[test]
    fn a_node_and_an_edge_never_share_a_hash() {
        // Arrange
        let (node_content, edge_content) = (node(), edge());

        // Act
        let (node_hash, edge_hash) = (hash_node(&node_content), hash_edge(&edge_content));

        // Assert
        assert_ne!(node_hash, edge_hash);
    }

    fn sample_hash(index: u32) -> [u8; 32] {
        Sha256::digest(index.to_le_bytes()).into()
    }

    fn sample_hashes(range: std::ops::Range<u32>) -> Vec<[u8; 32]> {
        range.map(sample_hash).collect()
    }

    fn small_config() -> BloomConfig {
        BloomConfig {
            expected_items: 5_000,
            false_positive_rate: 0.01,
        }
    }

    fn filter_with(config: BloomConfig, hashes: &[[u8; 32]]) -> HashBloom {
        let mut filter = HashBloom::new(config);
        for hash in hashes {
            filter.insert(hash);
        }
        filter
    }

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
        stored: Vec<[u8; 32]>,
        lookups: std::cell::Cell<usize>,
    }

    impl HashIndex for CountingIndex {
        fn contains_hash(&self, hash: &[u8; 32]) -> bool {
            self.lookups.set(self.lookups.get() + 1);
            self.stored.contains(hash)
        }
    }

    #[test]
    fn default_config_targets_a_one_percent_false_positive_rate() {
        // Arrange / Act
        let config = BloomConfig::default();

        // Assert
        assert_eq!(config.false_positive_rate, 0.01);
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

    #[test]
    fn a_pre_check_hit_for_a_never_stored_hash_is_decided_by_the_exact_index() {
        // Arrange
        let index = CountingIndex::default();
        let never_stored = sample_hash(42);

        // Act
        let duplicate = is_duplicate(&AlwaysTrue, &index, &never_stored);

        // Assert
        assert!(
            !duplicate,
            "the exact lookup must overrule a false positive"
        );
        assert_eq!(
            index.lookups.get(),
            1,
            "a pre-check hit must consult the index"
        );
    }

    #[test]
    fn a_pre_check_hit_for_a_stored_hash_is_confirmed_by_the_exact_index() {
        // Arrange
        let stored = sample_hash(7);
        let index = CountingIndex {
            stored: vec![stored],
            ..CountingIndex::default()
        };

        // Act
        let duplicate = is_duplicate(&AlwaysTrue, &index, &stored);

        // Assert
        assert!(duplicate);
    }

    #[test]
    fn a_pre_check_miss_skips_the_exact_index() {
        // Arrange
        let index = CountingIndex::default();

        // Act
        let duplicate = is_duplicate(&AlwaysFalse, &index, &sample_hash(1));

        // Assert
        assert!(!duplicate);
        assert_eq!(
            index.lookups.get(),
            0,
            "a definite miss needs no index lookup"
        );
    }

    #[test]
    fn a_saved_filter_loads_back_equal() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let hashes = sample_hashes(0..1_000);
        let original = filter_with(small_config(), &hashes);

        // Act
        original.save(&path).expect("save");
        let loaded = HashBloom::load(&path, &small_config());

        // Assert
        match loaded {
            FilterLoad::Fresh(filter) => {
                assert_eq!(filter, original);
                assert!(hashes.iter().all(|hash| filter.might_contain(hash)));
            }
            FilterLoad::Stale(reason) => panic!("expected a usable filter, got stale: {reason:?}"),
        }
    }

    #[test]
    fn a_missing_filter_file_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("absent.bin");

        // Act
        let loaded = HashBloom::load(&path, &small_config());

        // Assert
        assert!(
            matches!(loaded, FilterLoad::Stale(StaleReason::Missing)),
            "got {loaded:?}"
        );
    }

    #[test]
    fn a_corrupt_filter_file_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        std::fs::write(&path, b"this is not a bloom filter").expect("write garbage");

        // Act
        let loaded = HashBloom::load(&path, &small_config());

        // Assert
        assert!(
            matches!(loaded, FilterLoad::Stale(StaleReason::Corrupt)),
            "got {loaded:?}"
        );
    }

    #[test]
    fn a_truncated_filter_file_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path)
            .expect("save");
        let full = std::fs::read(&path).expect("read back");

        for kept in [0, 1, full.len() / 2, full.len() - 1] {
            std::fs::write(&path, &full[..kept]).expect("truncate");

            // Act
            let loaded = HashBloom::load(&path, &small_config());

            // Assert
            assert!(
                matches!(loaded, FilterLoad::Stale(StaleReason::Corrupt)),
                "truncated to {kept} of {} bytes: got {loaded:?}",
                full.len()
            );
        }
    }

    #[test]
    fn a_filter_saved_under_a_different_config_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path)
            .expect("save");
        let other_config = BloomConfig {
            expected_items: 50_000,
            false_positive_rate: 0.001,
        };

        // Act
        let loaded = HashBloom::load(&path, &other_config);

        // Assert
        assert!(
            matches!(loaded, FilterLoad::Stale(StaleReason::ConfigMismatch)),
            "got {loaded:?}"
        );
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
    const SOURCE: &str = "agent-a";

    fn log_event(event_id: &str, payload: LogPayload) -> LogEvent {
        LogEvent {
            event_id: event_id.into(),
            content_hash: EVENT_HASH_SENTINEL,
            source: SOURCE.into(),
            trust_score: 0.9,
            observed_at: 1_000,
            ingested_at: 2_000,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload,
        }
    }

    // A row's hash is what the write path would have computed for the same
    // request: the caller's valid_from counts as supplied, and an edge row,
    // which carries no producer, takes the event's source.
    fn expected_node_hash(row: &NodeRow) -> [u8; 32] {
        hash_node(&NodeContent {
            kind: row.kind.clone(),
            label: row.label.clone(),
            content: row.content.clone(),
            producer: row.producer.clone(),
            scope: row.scope.clone(),
            subject: row.subject.clone(),
            attributes: row.attributes.clone(),
            valid_from: Some(row.valid_from),
        })
    }

    fn expected_edge_hash(row: &EdgeRow) -> [u8; 32] {
        hash_edge(&EdgeContent {
            src: row.src.clone(),
            dst: row.dst.clone(),
            edge_type: row.edge_type.clone(),
            producer: SOURCE.into(),
            attributes: row.attributes.clone(),
        })
    }

    #[test]
    fn an_inserted_hash_is_found_with_its_event_id() {
        // Arrange
        let mut index = InMemoryHashIndex::new();
        let hash = sample_hash(1);

        // Act
        index.insert(hash, "event-1");

        // Assert
        assert_eq!(index.first_event_id(&hash), Some("event-1"));
        assert!(index.contains_hash(&hash));
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn an_absent_hash_is_not_found() {
        // Arrange
        let mut index = InMemoryHashIndex::new();
        index.insert(sample_hash(1), "event-1");

        // Act
        let absent = sample_hash(2);

        // Assert
        assert_eq!(index.first_event_id(&absent), None);
        assert!(!index.contains_hash(&absent));
    }

    #[test]
    fn a_second_insert_of_the_same_hash_keeps_the_first_event_id() {
        // Arrange
        let mut index = InMemoryHashIndex::new();
        let hash = sample_hash(1);
        index.insert(hash, "event-1");

        // Act
        index.insert(hash, "event-2");

        // Assert
        assert_eq!(index.first_event_id(&hash), Some("event-1"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn distinct_hashes_each_map_to_their_own_event() {
        // Arrange
        let mut index = InMemoryHashIndex::new();

        // Act
        index.insert(sample_hash(1), "event-1");
        index.insert(sample_hash(2), "event-2");

        // Assert
        assert_eq!(index.first_event_id(&sample_hash(1)), Some("event-1"));
        assert_eq!(index.first_event_id(&sample_hash(2)), Some("event-2"));
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
    fn rebuilding_from_an_empty_log_gives_an_empty_filter_and_index() {
        // Arrange
        let events: Vec<LogEvent> = Vec::new();

        // Act
        let (filter, index) = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(filter, HashBloom::new(small_config()));
        assert!(index.is_empty());
        assert!(!filter.might_contain(&sample_hash(1)));
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
        let (filter, index) = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        let (node_hash, edge_hash) = (expected_node_hash(&node), expected_edge_hash(&edge));
        assert_eq!(index.first_event_id(&node_hash), Some("event-1"));
        assert_eq!(index.first_event_id(&edge_hash), Some("event-2"));
        assert_eq!(index.len(), 2);
        assert!(filter.might_contain(&node_hash));
        assert!(filter.might_contain(&edge_hash));
        assert!(!index.contains_hash(&EVENT_HASH_SENTINEL));
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
        let (filter, index) = rebuild_from_log(vec![batch].into_iter(), small_config());

        // Assert
        for hash in [
            expected_node_hash(&first),
            expected_node_hash(&second),
            expected_edge_hash(&edge),
        ] {
            assert_eq!(index.first_event_id(&hash), Some("event-1"));
            assert!(filter.might_contain(&hash));
        }
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn an_empty_episode_batch_adds_nothing() {
        // Arrange
        let batch = log_event("event-1", LogPayload::EpisodeBatch(Vec::new()));

        // Act
        let (filter, index) = rebuild_from_log(vec![batch].into_iter(), small_config());

        // Assert
        assert!(index.is_empty());
        assert_eq!(filter, HashBloom::new(small_config()));
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
        let (_, index) = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(expected_node_hash(&first), expected_node_hash(&repeat));
        assert_eq!(
            index.first_event_id(&expected_node_hash(&first)),
            Some("event-1")
        );
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn tombstone_duplicate_and_voided_events_contribute_no_hash() {
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
            log_event(
                "event-3",
                LogPayload::Voided {
                    target_event_id: "event-0".into(),
                },
            ),
        ];

        // Act
        let (filter, index) = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert!(index.is_empty());
        assert!(!index.contains_hash(&EVENT_HASH_SENTINEL));
        assert_eq!(filter, HashBloom::new(small_config()));
    }

    // Voiding is applied by replay, which skips the voided event. Dedup keeps
    // the hash so a later identical write still resolves to the first event.
    #[test]
    fn a_voided_event_does_not_remove_its_targets_hash() {
        // Arrange
        let node = node_row("node-1", "alpha");
        let events = vec![
            log_event("event-1", LogPayload::NodeWrite(node.clone())),
            log_event(
                "event-2",
                LogPayload::Voided {
                    target_event_id: "event-1".into(),
                },
            ),
        ];

        // Act
        let (filter, index) = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        let hash = expected_node_hash(&node);
        assert_eq!(index.first_event_id(&hash), Some("event-1"));
        assert!(filter.might_contain(&hash));
        assert_eq!(index.len(), 1);
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
        let (rebuilt, _) = rebuild_from_log(events.into_iter(), small_config());

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
        let (filter, index) = rebuild_from_log(events.into_iter(), small_config());

        // Assert
        assert_eq!(expected.len(), 300 + 100);
        let lost: Vec<&String> = expected
            .iter()
            .filter(|(hash, _)| !filter.might_contain(hash) || !index.contains_hash(hash))
            .map(|(_, event_id)| event_id)
            .collect();
        assert!(lost.is_empty(), "hashes lost for events {lost:?}");
        assert_eq!(index.len(), expected.len());
    }
}
