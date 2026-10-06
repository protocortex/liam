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
    fn a_node_and_an_edge_never_share_a_hash() {
        // Arrange
        let (node_content, edge_content) = (node(), edge());

        // Act
        let (node_hash, edge_hash) = (hash_node(&node_content), hash_edge(&edge_content));

        // Assert
        assert_ne!(node_hash, edge_hash);
    }
}
