// SPDX-License-Identifier: Apache-2.0

//! Content hashing for write deduplication.
//!
//! The hash covers what the caller asked to store, never server-minted ids or
//! timestamps, so the same request always hashes the same.

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

fn content_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Content hash over the canonical encoding of a node write.
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
    content_digest(&buf)
}

/// Content hash over the canonical encoding of an edge write.
pub fn hash_edge(edge: &EdgeContent) -> [u8; 32] {
    let mut buf = vec![EDGE_TAG];
    put_str(&mut buf, &edge.src);
    put_str(&mut buf, &edge.dst);
    put_str(&mut buf, &edge.edge_type);
    content_digest(&buf)
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
    })
}

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
        ]
        .into_iter()
        .chain(
            [i64::MIN, -1, 0, i64::MAX]
                .into_iter()
                .map(|millis| NodeContent {
                    valid_from: Some(millis),
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
    fn the_domain_tag_is_part_of_what_each_hash_covers() {
        // Arrange
        let edge_content = edge();
        let untagged = untagged_edge_bytes(&edge_content);

        // Act
        let actual = hash_edge(&edge_content);

        // Assert
        assert_ne!(NODE_TAG, EDGE_TAG);
        assert_eq!(actual, tagged_digest(EDGE_TAG, &untagged));
        assert_ne!(actual, tagged_digest(NODE_TAG, &untagged));
        assert_ne!(actual, content_digest(&untagged));
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
}
