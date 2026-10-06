// SPDX-License-Identifier: Apache-2.0

//! Content hashing for write deduplication.
//!
//! The hash covers what the caller asked to store, never server-minted ids or
//! timestamps, so the same request always hashes the same.
//!
//! A node is SHA-256 over `0x01`, then in this order: `kind`, `label`,
//! `content`, `producer` (each a u64 LE byte length then the UTF-8 bytes),
//! `scope` and `subject` (a 0/1 presence byte, then the string when present),
//! `attributes` (length-prefixed), `valid_from` (a 0/1 presence byte, then
//! the i64 LE value when present), and `confidence` (the f64 bits as u64 LE,
//! with -0.0 folded to 0.0 and every NaN folded to one canonical NaN). An edge
//! is SHA-256 over `0x02`, then `src`, `dst`, and `edge_type`, each
//! length-prefixed.

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
    pub confidence: f64,
}

/// What an edge write asks to store. An edge is identified by its endpoints
/// and type alone, so the producer and attributes do not take part in its hash.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeContent {
    pub src: String,
    pub dst: String,
    pub edge_type: String,
}

const NODE_TAG: u8 = 0x01;
const EDGE_TAG: u8 = 0x02;
// Quiet NaN with an empty payload, the one bit pattern every NaN folds to.
const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;

// The only place bytes become a hash, so a keyed digest replaces it here.
fn content_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Content hash over the canonical encoding of a node write.
///
/// The input must be the resolved, normalized row (scope trimmed, attributes
/// serialized), so the write path should hash through [`node_row_hash`] after
/// minting rather than call this on raw request values.
pub fn hash_node(node: &NodeContent) -> [u8; 32] {
    content_digest(&encode_node(node))
}

/// Content hash over the canonical encoding of an edge write.
///
/// The input must be the resolved row, so the write path should hash through
/// [`edge_row_hash`] after minting rather than call this on raw request values.
pub fn hash_edge(edge: &EdgeContent) -> [u8; 32] {
    content_digest(&encode_edge(edge))
}

pub(crate) fn encode_node(node: &NodeContent) -> Vec<u8> {
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
    buf.extend(canonical_f64_bits(node.confidence).to_le_bytes());
    buf
}

pub(crate) fn encode_edge(edge: &EdgeContent) -> Vec<u8> {
    let mut buf = vec![EDGE_TAG];
    put_str(&mut buf, &edge.src);
    put_str(&mut buf, &edge.dst);
    put_str(&mut buf, &edge.edge_type);
    buf
}

// Length prefixes keep field boundaries unambiguous, so moving bytes between
// adjacent fields changes the hash. The length counts bytes, not chars.
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

// Values that compare equal must hash equal, but -0.0 and 0.0 differ in bits
// and NaN payloads vary by producer.
fn canonical_f64_bits(value: f64) -> u64 {
    if value.is_nan() {
        CANONICAL_NAN_BITS
    } else if value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}

/// Each content hash an event carries, paired with the id of the row it covers.
pub fn content_hashes(event: &LogEvent) -> Vec<([u8; 32], String)> {
    match &event.payload {
        LogPayload::NodeWrite(row) => vec![(node_row_hash(row), row.id.clone())],
        LogPayload::EdgeWrite(row) => vec![(edge_row_hash(row), row.id.clone())],
        LogPayload::EpisodeBatch(effects) => effects
            .iter()
            .map(|effect| match effect {
                RowEffect::Node(row) => (node_row_hash(row), row.id.clone()),
                RowEffect::Edge(row) => (edge_row_hash(row), row.id.clone()),
            })
            .collect(),
        LogPayload::Tombstone(_) | LogPayload::DuplicateOf { .. } | LogPayload::Voided { .. } => {
            Vec::new()
        }
    }
}

/// Hash of the node a resolved row stores, ignoring ids and transaction times.
pub fn node_row_hash(row: &NodeRow) -> [u8; 32] {
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

/// Hash of the edge a resolved row stores, ignoring ids, attributes, and times.
pub fn edge_row_hash(row: &EdgeRow) -> [u8; 32] {
    hash_edge(&EdgeContent {
        src: row.src.clone(),
        dst: row.dst.clone(),
        edge_type: row.edge_type.clone(),
    })
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::event::{TombstoneTable, TombstoneTarget, CURRENT_SCHEMA_VERSION};

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
            confidence: 0.75,
        }
    }

    fn node_row() -> NodeRow {
        NodeRow {
            id: "node-1".into(),
            kind: "fact".into(),
            label: "label".into(),
            content: "content".into(),
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

    fn edge_row() -> EdgeRow {
        EdgeRow {
            id: "edge-1".into(),
            src: "node-1".into(),
            dst: "node-2".into(),
            edge_type: "relates_to".into(),
            attributes: "{}".into(),
            tx_from: 2_000,
            tx_to: 4_102_444_800_000,
        }
    }

    fn event_with(payload: LogPayload) -> LogEvent {
        LogEvent {
            event_id: "event-1".into(),
            content_hash: [0; 32],
            source: "agent-a".into(),
            trust_score: 0.9,
            observed_at: 1_000,
            ingested_at: 2_000,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload,
        }
    }

    fn edge() -> EdgeContent {
        EdgeContent {
            src: "node-1".into(),
            dst: "node-2".into(),
            edge_type: "relates_to".into(),
        }
    }

    fn hex(hash: &[u8; 32]) -> String {
        hash.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    // Independent of the production encoder on purpose: strings are a u64 LE
    // length then bytes, options are a 0/1 presence byte then the value, and a
    // float is its bits as u64 LE with -0.0 and every NaN folded to one pattern.
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

    fn untagged_node_bytes(node: &NodeContent) -> Vec<u8> {
        let mut buf = Vec::new();
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
        let bits = if node.confidence.is_nan() {
            0x7ff8_0000_0000_0000
        } else if node.confidence == 0.0 {
            0
        } else {
            node.confidence.to_bits()
        };
        buf.extend(bits.to_le_bytes());
        buf
    }

    fn untagged_edge_bytes(edge: &EdgeContent) -> Vec<u8> {
        let mut buf = Vec::new();
        put_str(&mut buf, &edge.src);
        put_str(&mut buf, &edge.dst);
        put_str(&mut buf, &edge.edge_type);
        buf
    }

    fn tagged_digest(tag: u8, untagged: &[u8]) -> [u8; 32] {
        let mut buf = vec![tag];
        buf.extend(untagged);
        Sha256::digest(&buf).into()
    }

    fn reference_node_hash(node: &NodeContent) -> [u8; 32] {
        tagged_digest(NODE_TAG, &untagged_node_bytes(node))
    }

    fn reference_edge_hash(edge: &EdgeContent) -> [u8; 32] {
        tagged_digest(EDGE_TAG, &untagged_edge_bytes(edge))
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
            (
                "confidence",
                NodeContent {
                    confidence: 0.9,
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
            NodeContent {
                scope: Some(String::new()),
                subject: Some(String::new()),
                ..node()
            },
            NodeContent {
                label: "café".into(),
                content: "日本".into(),
                ..node()
            },
        ]
        .into_iter()
        .chain(
            [i64::MIN, -1, 0, i64::MAX]
                .into_iter()
                .map(|millis| NodeContent {
                    valid_from: Some(millis),
                    ..node()
                }),
        )
        .chain(
            [
                0.0,
                -0.0,
                0.3,
                1.0,
                f64::NAN,
                f64::INFINITY,
                f64::MIN_POSITIVE,
            ]
            .into_iter()
            .map(|confidence| NodeContent {
                confidence,
                ..node()
            }),
        );

        for case in cases {
            // Act
            let actual = hash_node(&case);

            // Assert
            assert_eq!(hex(&actual), hex(&reference_node_hash(&case)));
        }
    }

    // Hand-written from the encoding in the module doc, not produced by any
    // encoder. The domain tag is prepended separately where it is checked.
    #[rustfmt::skip]
    const NODE_PREIMAGE: [u8; 90] = [
        0x04, 0, 0, 0, 0, 0, 0, 0, b'f', b'a', b'c', b't',                      // kind
        0x05, 0, 0, 0, 0, 0, 0, 0, b'l', b'a', b'b', b'e', b'l',                // label
        0x07, 0, 0, 0, 0, 0, 0, 0, b'c', b'o', b'n', b't', b'e', b'n', b't',    // content
        0x07, 0, 0, 0, 0, 0, 0, 0, b'a', b'g', b'e', b'n', b't', b'-', b'a',    // producer
        0x01, 0x06, 0, 0, 0, 0, 0, 0, 0, b'p', b'r', b'o', b'j', b'/', b'a',    // scope: present, then value
        0x00,                                                                   // subject: absent
        0x02, 0, 0, 0, 0, 0, 0, 0, b'{', b'}',                                  // attributes
        0x00,                                                                   // valid_from: absent
        0, 0, 0, 0, 0, 0, 0xe8, 0x3f,                                           // confidence 0.75 = 0x3fe8000000000000, LE
    ];

    #[rustfmt::skip]
    const EDGE_PREIMAGE: [u8; 46] = [
        0x06, 0, 0, 0, 0, 0, 0, 0, b'n', b'o', b'd', b'e', b'-', b'1',          // src
        0x06, 0, 0, 0, 0, 0, 0, 0, b'n', b'o', b'd', b'e', b'-', b'2',          // dst
        0x0a, 0, 0, 0, 0, 0, 0, 0, b'r', b'e', b'l', b'a', b't', b'e', b's', b'_', b't', b'o', // edge_type
    ];

    #[test]
    fn the_reference_encoder_produces_the_hand_written_preimages() {
        // Arrange
        let (node_content, edge_content) = (node(), edge());

        // Act
        let (node_bytes, edge_bytes) = (
            untagged_node_bytes(&node_content),
            untagged_edge_bytes(&edge_content),
        );

        // Assert
        assert_eq!(node_bytes, NODE_PREIMAGE);
        assert_eq!(edge_bytes, EDGE_PREIMAGE);
    }

    #[test]
    fn the_production_encoders_prepend_the_domain_tag_to_the_preimage() {
        // Arrange
        let (node_content, edge_content) = (node(), edge());

        // Act
        let (node_bytes, edge_bytes) = (encode_node(&node_content), encode_edge(&edge_content));

        // Assert
        assert_eq!(node_bytes[0], 0x01);
        assert_eq!(node_bytes[1..], NODE_PREIMAGE);
        assert_eq!(edge_bytes[0], 0x02);
        assert_eq!(edge_bytes[1..], EDGE_PREIMAGE);
    }

    #[test]
    fn hashes_are_the_digest_of_the_tagged_encoding() {
        // Arrange
        let (node_content, edge_content) = (node(), edge());

        // Act
        let (node_hash, edge_hash) = (hash_node(&node_content), hash_edge(&edge_content));

        // Assert
        assert_eq!(node_hash, content_digest(&encode_node(&node_content)));
        assert_eq!(edge_hash, content_digest(&encode_edge(&edge_content)));
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
        // printf '0104000000000000006661637405000000000000006c6162656c0700000000000000636f6e74656e7407000000000000006167656e742d6101060000000000000070726f6a2f610002000000000000007b7d00000000000000e83f' | xxd -r -p | shasum -a 256
        assert_eq!(
            hex(&without_hash),
            "162bc09dda6f46f80c0089a37fefb319033eb2e221e56e8861d0312768e65c28"
        );
        // printf '0104000000000000006661637405000000000000006c6162656c0700000000000000636f6e74656e7407000000000000006167656e742d6101060000000000000070726f6a2f610002000000000000007b7d01e803000000000000000000000000e83f' | xxd -r -p | shasum -a 256
        assert_eq!(
            hex(&with_hash),
            "b7fd1c61e0f751207bd9a17ee6d625feab2671a7937c3a23276de8e9ceae3c03"
        );
    }

    #[test]
    fn a_non_ascii_node_hash_counts_bytes_and_is_pinned() {
        // Arrange
        let content = NodeContent {
            label: "café".into(),
            content: "日本".into(),
            ..node()
        };

        // Act
        let actual = hash_node(&content);

        // Assert
        // "café" is 5 bytes and "日本" is 6, so the prefixes are 5 and 6, not 4 and 2.
        // printf '010400000000000000666163740500000000000000636166c3a90600000000000000e697a5e69cac07000000000000006167656e742d6101060000000000000070726f6a2f610002000000000000007b7d00000000000000e83f' | xxd -r -p | shasum -a 256
        assert_eq!(
            hex(&actual),
            "851c615a586d199524ea6f9179a2b9a3f9f1935145b6b7a534b79665b1b2a46b"
        );
        assert_eq!(hex(&actual), hex(&reference_node_hash(&content)));
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
                "src and dst swapped",
                EdgeContent {
                    src: "node-2".into(),
                    dst: "node-1".into(),
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
        // printf '0206000000000000006e6f64652d3106000000000000006e6f64652d320a0000000000000072656c617465735f746f' | xxd -r -p | shasum -a 256
        assert_eq!(
            hex(&actual),
            "c7de35063da038bb463a5130d216206caafb73b2fd2881b56d901a7c0374683b"
        );
    }

    #[test]
    fn a_node_and_an_edge_sharing_leading_fields_hash_differently() {
        // Arrange
        let node_content = node();
        let edge_content = EdgeContent {
            src: node_content.kind.clone(),
            dst: node_content.label.clone(),
            edge_type: node_content.content.clone(),
        };

        // Act
        let (node_hash, edge_hash) = (hash_node(&node_content), hash_edge(&edge_content));

        // Assert
        assert_ne!(node_hash, edge_hash);
    }

    // A node always encodes at least four length-prefixed strings and an edge
    // three, so their untagged bytes can never be equal. The closest they get
    // is an edge encoding that is a strict prefix of a node encoding.
    #[test]
    fn an_edge_encoding_can_only_be_a_strict_prefix_of_a_node_encoding() {
        // Arrange
        let node_content = node();
        let edge_content = EdgeContent {
            src: node_content.kind.clone(),
            dst: node_content.label.clone(),
            edge_type: node_content.content.clone(),
        };

        // Act
        let node_bytes = untagged_node_bytes(&node_content);
        let edge_bytes = untagged_edge_bytes(&edge_content);

        // Assert
        assert!(node_bytes.starts_with(&edge_bytes));
        assert!(node_bytes.len() > edge_bytes.len());
    }

    #[test]
    fn node_and_edge_domain_tags_are_distinct() {
        // Assert
        assert_ne!(NODE_TAG, EDGE_TAG);
    }

    #[test]
    fn valid_from_boundary_values_hash_distinctly() {
        // Arrange
        let values = [None, Some(i64::MIN), Some(-1), Some(0), Some(i64::MAX)];

        // Act
        let hashes: Vec<[u8; 32]> = values
            .iter()
            .map(|valid_from| {
                hash_node(&NodeContent {
                    valid_from: *valid_from,
                    ..node()
                })
            })
            .collect();

        // Assert
        for (i, first) in hashes.iter().enumerate() {
            for (j, second) in hashes.iter().enumerate().skip(i + 1) {
                assert_ne!(
                    first, second,
                    "valid_from {:?} and {:?} must hash differently",
                    values[i], values[j]
                );
            }
        }
    }

    #[test]
    fn an_explicit_zero_valid_from_differs_from_an_absent_one() {
        // Arrange
        let absent = node();
        let zero = NodeContent {
            valid_from: Some(0),
            ..node()
        };

        // Act
        let (absent_hash, zero_hash) = (hash_node(&absent), hash_node(&zero));

        // Assert
        assert_ne!(absent_hash, zero_hash);
    }

    #[test]
    fn valid_from_keeps_its_sign() {
        // Arrange
        let negative = NodeContent {
            valid_from: Some(-1_000),
            ..node()
        };
        let positive = NodeContent {
            valid_from: Some(1_000),
            ..node()
        };

        // Act
        let (negative_hash, positive_hash) = (hash_node(&negative), hash_node(&positive));

        // Assert
        assert_ne!(negative_hash, positive_hash);
    }

    #[test]
    fn absent_empty_and_present_optional_strings_hash_distinctly() {
        // Arrange
        let variants = [None, Some(String::new()), Some("x".to_string())];

        for field in ["scope", "subject"] {
            // Act
            let hashes: Vec<[u8; 32]> = variants
                .iter()
                .map(|value| {
                    let mut content = node();
                    match field {
                        "scope" => content.scope = value.clone(),
                        _ => content.subject = value.clone(),
                    }
                    hash_node(&content)
                })
                .collect();

            // Assert
            assert_ne!(hashes[0], hashes[1], "{field}: None vs empty");
            assert_ne!(hashes[0], hashes[2], "{field}: None vs value");
            assert_ne!(hashes[1], hashes[2], "{field}: empty vs value");
        }
    }

    #[test]
    fn moving_bytes_across_any_adjacent_node_field_boundary_changes_the_hash() {
        // Arrange
        let pairs: Vec<(&str, NodeContent, NodeContent)> = vec![
            (
                "kind/label",
                NodeContent {
                    kind: "ab".into(),
                    label: "c".into(),
                    ..node()
                },
                NodeContent {
                    kind: "a".into(),
                    label: "bc".into(),
                    ..node()
                },
            ),
            (
                "content/producer",
                NodeContent {
                    content: "ab".into(),
                    producer: "c".into(),
                    ..node()
                },
                NodeContent {
                    content: "a".into(),
                    producer: "bc".into(),
                    ..node()
                },
            ),
            (
                "producer/scope",
                NodeContent {
                    producer: "ab".into(),
                    scope: Some("c".into()),
                    ..node()
                },
                NodeContent {
                    producer: "a".into(),
                    scope: Some("bc".into()),
                    ..node()
                },
            ),
            (
                "scope/subject",
                NodeContent {
                    scope: Some("ab".into()),
                    subject: Some("c".into()),
                    ..node()
                },
                NodeContent {
                    scope: Some("a".into()),
                    subject: Some("bc".into()),
                    ..node()
                },
            ),
            (
                "subject/attributes",
                NodeContent {
                    subject: Some("ab".into()),
                    attributes: "c".into(),
                    ..node()
                },
                NodeContent {
                    subject: Some("a".into()),
                    attributes: "bc".into(),
                    ..node()
                },
            ),
        ];

        for (boundary, early, late) in pairs {
            // Act
            let (early_hash, late_hash) = (hash_node(&early), hash_node(&late));

            // Assert
            assert_ne!(early_hash, late_hash, "{boundary} boundary must be hashed");
        }
    }

    #[test]
    fn moving_bytes_across_an_edge_field_boundary_changes_the_hash() {
        // Arrange
        let early = EdgeContent {
            src: "ab".into(),
            dst: "c".into(),
            ..edge()
        };
        let late = EdgeContent {
            src: "a".into(),
            dst: "bc".into(),
            ..edge()
        };

        // Act
        let (early_hash, late_hash) = (hash_edge(&early), hash_edge(&late));

        // Assert
        assert_ne!(early_hash, late_hash);
    }

    #[test]
    fn confidence_changes_the_node_hash() {
        // Arrange
        let low = NodeContent {
            confidence: 0.3,
            ..node()
        };
        let high = NodeContent {
            confidence: 0.9,
            ..node()
        };

        // Act
        let (low_hash, high_hash) = (hash_node(&low), hash_node(&high));

        // Assert
        assert_ne!(low_hash, high_hash);
    }

    #[test]
    fn negative_and_positive_zero_confidence_hash_equal() {
        // Arrange
        let negative = NodeContent {
            confidence: -0.0,
            ..node()
        };
        let positive = NodeContent {
            confidence: 0.0,
            ..node()
        };

        // Act
        let (negative_hash, positive_hash) = (hash_node(&negative), hash_node(&positive));

        // Assert
        assert_eq!(negative_hash, positive_hash);
    }

    #[test]
    fn nan_confidences_with_different_payloads_hash_equal() {
        // Arrange
        let payloads = [
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0xfff8_0000_0000_0000),
            f64::from_bits(0x7ff0_0000_0000_0001),
        ];

        // Act
        let hashes: Vec<[u8; 32]> = payloads
            .iter()
            .map(|confidence| {
                assert!(confidence.is_nan());
                hash_node(&NodeContent {
                    confidence: *confidence,
                    ..node()
                })
            })
            .collect();

        // Assert
        assert!(hashes.windows(2).all(|pair| pair[0] == pair[1]));
        let zero = hash_node(&NodeContent {
            confidence: 0.0,
            ..node()
        });
        assert_ne!(hashes[0], zero);
    }

    #[test]
    fn edge_hash_matches_the_reference_for_a_non_ascii_source() {
        // Arrange
        let content = EdgeContent {
            src: "nœud-1".into(),
            ..edge()
        };

        // Act
        let actual = hash_edge(&content);

        // Assert
        // "nœud-1" is 7 bytes but 6 chars, so the prefix proves bytes are counted.
        // printf '0207000000000000006ec59375642d3106000000000000006e6f64652d320a0000000000000072656c617465735f746f' | xxd -r -p | shasum -a 256
        assert_eq!(
            hex(&actual),
            "9374fbc209f33bd94a53eff7050290533f4f9f5a42d3c32db776f3cccd0263e4"
        );
        assert_eq!(hex(&actual), hex(&reference_edge_hash(&content)));
    }

    #[test]
    fn node_row_hash_without_a_supplied_valid_from_ignores_it() {
        // Arrange
        let row = NodeRow {
            valid_from_supplied: false,
            valid_from: 12_345,
            ..node_row()
        };
        let expected = hash_node(&node());

        // Act
        let actual = node_row_hash(&row);

        // Assert
        assert_eq!(actual, expected);
    }

    #[test]
    fn node_row_hash_with_a_supplied_valid_from_covers_it() {
        // Arrange
        let expected = hash_node(&NodeContent {
            valid_from: Some(1_000),
            ..node()
        });

        // Act
        let actual = node_row_hash(&node_row());

        // Assert
        assert_eq!(actual, expected);
    }

    #[test]
    fn node_row_hash_covers_the_row_confidence() {
        // Arrange
        let row = NodeRow {
            confidence: 0.1,
            ..node_row()
        };
        let expected = hash_node(&NodeContent {
            valid_from: Some(1_000),
            confidence: 0.1,
            ..node()
        });

        // Act
        let actual = node_row_hash(&row);

        // Assert
        assert_eq!(actual, expected);
        assert_ne!(actual, node_row_hash(&node_row()));
    }

    #[test]
    fn node_row_hash_ignores_id_valid_until_and_transaction_times() {
        // Arrange
        let other = NodeRow {
            id: "node-99".into(),
            valid_until: 5_000,
            tx_from: 3_000,
            tx_to: 4_000,
            ..node_row()
        };

        // Act
        let (base_hash, other_hash) = (node_row_hash(&node_row()), node_row_hash(&other));

        // Assert
        assert_eq!(base_hash, other_hash);
    }

    #[test]
    fn node_row_hash_distinguishes_scope_from_subject() {
        // Arrange
        let scoped = NodeRow {
            scope: Some("x".into()),
            subject: None,
            ..node_row()
        };
        let subjected = NodeRow {
            scope: None,
            subject: Some("x".into()),
            ..node_row()
        };

        // Act
        let (scoped_hash, subjected_hash) = (node_row_hash(&scoped), node_row_hash(&subjected));

        // Assert
        assert_ne!(scoped_hash, subjected_hash);
    }

    #[test]
    fn node_row_hash_golden_value_is_pinned() {
        // Arrange
        let row = NodeRow {
            subject: Some("user-1".into()),
            valid_from_supplied: false,
            ..node_row()
        };

        // Act
        let actual = node_row_hash(&row);

        // Assert
        // printf '0104000000000000006661637405000000000000006c6162656c0700000000000000636f6e74656e7407000000000000006167656e742d6101060000000000000070726f6a2f61010600000000000000757365722d3102000000000000007b7d00000000000000e83f' | xxd -r -p | shasum -a 256
        assert_eq!(
            hex(&actual),
            "ae43f1b3aa5c0c698b9bf595271bd7b737c6bc58fa75e75405222c6713fc4d96"
        );
    }

    #[test]
    fn edge_row_hash_ignores_id_attributes_and_transaction_times() {
        // Arrange
        let other = EdgeRow {
            id: "edge-99".into(),
            attributes: "{\"w\":1}".into(),
            tx_from: 3_000,
            tx_to: 4_000,
            ..edge_row()
        };
        let expected = hash_edge(&edge());

        // Act
        let (base_hash, other_hash) = (edge_row_hash(&edge_row()), edge_row_hash(&other));

        // Assert
        assert_eq!(base_hash, other_hash);
        assert_eq!(base_hash, expected);
    }

    #[test]
    fn content_hashes_lists_each_episode_batch_row_in_order() {
        // Arrange
        let first = node_row();
        let second = EdgeRow {
            id: "edge-b".into(),
            ..edge_row()
        };
        let third = NodeRow {
            id: "node-c".into(),
            label: "other".into(),
            ..node_row()
        };
        let event = event_with(LogPayload::EpisodeBatch(vec![
            RowEffect::Node(first.clone()),
            RowEffect::Edge(second.clone()),
            RowEffect::Node(third.clone()),
        ]));

        // Act
        let actual = content_hashes(&event);

        // Assert
        assert_eq!(
            actual,
            vec![
                (node_row_hash(&first), "node-1".to_string()),
                (edge_row_hash(&second), "edge-b".to_string()),
                (node_row_hash(&third), "node-c".to_string()),
            ]
        );
    }

    #[test]
    fn content_hashes_pairs_a_single_row_write_with_its_id() {
        // Arrange
        let node_event = event_with(LogPayload::NodeWrite(node_row()));
        let edge_event = event_with(LogPayload::EdgeWrite(edge_row()));

        // Act
        let (node_hashes, edge_hashes) = (content_hashes(&node_event), content_hashes(&edge_event));

        // Assert
        assert_eq!(
            node_hashes,
            vec![(node_row_hash(&node_row()), "node-1".to_string())]
        );
        assert_eq!(
            edge_hashes,
            vec![(edge_row_hash(&edge_row()), "edge-1".to_string())]
        );
    }

    #[test]
    fn content_hashes_is_empty_for_payloads_that_store_no_row() {
        // Arrange
        let payloads = [
            LogPayload::Tombstone(vec![TombstoneTarget {
                table: TombstoneTable::Nodes,
                id: "node-1".into(),
            }]),
            LogPayload::DuplicateOf {
                first_event_id: "event-0".into(),
            },
            LogPayload::Voided {
                target_event_id: "event-0".into(),
            },
        ];

        for payload in payloads {
            // Act
            let hashes = content_hashes(&event_with(payload));

            // Assert
            assert!(hashes.is_empty());
        }
    }
}
