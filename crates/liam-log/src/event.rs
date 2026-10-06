// SPDX-License-Identifier: Apache-2.0

//! The event record the log stores, and its postcard wire framing.
//!
//! An event holds the fully resolved row values, not the request that produced
//! them, so replaying the log re-applies recorded rows without re-minting ids
//! or re-reading a clock.

use serde::{Deserialize, Serialize};

/// Schema version this build writes and the newest it can read.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Why an encoded event could not be turned back into a [`LogEvent`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum EventError {
    /// The bytes were written by a newer build than this reader understands.
    #[error("unsupported schema version {found}, newest readable is {supported}")]
    UnsupportedSchemaVersion { found: u32, supported: u32 },
    /// The bytes are truncated or do not match the event layout.
    #[error("event bytes could not be decoded: {0}")]
    Decode(postcard::Error),
    /// The event could not be serialized.
    #[error("event could not be encoded: {0}")]
    Encode(postcard::Error),
    /// A complete event was decoded and bytes were left over, so the frame is corrupt.
    #[error("{count} unexpected bytes after the event")]
    TrailingBytes { count: usize },
}

/// A `nodes` row with every value already resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeRow {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub content: String,
    pub producer: String,
    pub attributes: String,
    pub scope: Option<String>,
    pub subject: Option<String>,
    pub confidence: f64,
    pub valid_from: i64,
    /// False when the caller left `valid_from` unset and the store filled in a
    /// default, so dedup hashing can ignore the minted value.
    pub valid_from_supplied: bool,
    pub valid_until: i64,
    pub tx_from: i64,
    pub tx_to: i64,
}

/// An `edges` row with every value already resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeRow {
    pub id: String,
    pub src: String,
    pub dst: String,
    pub edge_type: String,
    pub attributes: String,
    pub tx_from: i64,
    pub tx_to: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RowEffect {
    Node(NodeRow),
    Edge(EdgeRow),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LogPayload {
    NodeWrite(NodeRow),
    EdgeWrite(EdgeRow),
    EpisodeBatch(Vec<RowEffect>),
    /// One batched tombstone per GC chunk.
    Tombstone(Vec<TombstoneTarget>),
    DuplicateOf {
        first_event_id: String,
    },
    Voided {
        target_event_id: String,
    },
}

/// Table a tombstone targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TombstoneTable {
    Nodes,
    Edges,
    NodeCommunity,
}

/// One row removed by a tombstone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TombstoneTarget {
    pub table: TombstoneTable,
    pub id: String,
}

/// One durable log entry.
///
/// `schema_version` is not part of the serialized body: the frame prefix is its
/// only copy, so a frame can never carry two disagreeing versions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEvent {
    pub event_id: String,
    pub content_hash: [u8; 32],
    pub source: String,
    pub trust_score: f64,
    pub observed_at: i64,
    pub ingested_at: i64,
    /// Reserved for the encryption work; always `None` for now.
    pub encryption_key_id: Option<String>,
    #[serde(skip)]
    pub schema_version: u32,
    pub payload: LogPayload,
}

impl LogEvent {
    /// Encodes the event with its schema version ahead of the body.
    ///
    /// Fails for a version newer than this build can read, since the log must
    /// never hold a frame its own writer could not decode.
    pub fn encode(&self) -> Result<Vec<u8>, EventError> {
        if self.schema_version > CURRENT_SCHEMA_VERSION {
            return Err(EventError::UnsupportedSchemaVersion {
                found: self.schema_version,
                supported: CURRENT_SCHEMA_VERSION,
            });
        }
        let mut frame = postcard::to_stdvec(&self.schema_version).map_err(EventError::Encode)?;
        frame.extend(postcard::to_stdvec(self).map_err(EventError::Encode)?);
        Ok(frame)
    }

    /// Decodes bytes produced by [`LogEvent::encode`].
    ///
    /// The version prefix is checked before the body is touched, so a newer
    /// layout is reported as such instead of being misread.
    pub fn decode(bytes: &[u8]) -> Result<LogEvent, EventError> {
        let (version, body) =
            postcard::take_from_bytes::<u32>(bytes).map_err(EventError::Decode)?;
        if version > CURRENT_SCHEMA_VERSION {
            return Err(EventError::UnsupportedSchemaVersion {
                found: version,
                supported: CURRENT_SCHEMA_VERSION,
            });
        }
        let (mut event, rest) =
            postcard::take_from_bytes::<LogEvent>(body).map_err(EventError::Decode)?;
        if !rest.is_empty() {
            return Err(EventError::TrailingBytes { count: rest.len() });
        }
        event.schema_version = version;
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            content_hash: [7; 32],
            source: "agent-a".into(),
            trust_score: 0.9,
            observed_at: 1_000,
            ingested_at: 2_000,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload,
        }
    }

    fn assert_round_trips(event: LogEvent) {
        // Act
        let encoded = event.encode().expect("encode event");
        let decoded = LogEvent::decode(&encoded);

        // Assert
        assert_eq!(decoded, Ok(event));
    }

    #[test]
    fn node_write_round_trips() {
        // Arrange
        let event = event_with(LogPayload::NodeWrite(node_row()));

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn edge_write_round_trips() {
        // Arrange
        let event = event_with(LogPayload::EdgeWrite(edge_row()));

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn node_write_with_unsupplied_valid_from_round_trips() {
        // Arrange
        let mut node = node_row();
        node.valid_from_supplied = false;
        let event = event_with(LogPayload::NodeWrite(node));

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn episode_batch_with_effects_round_trips() {
        // Arrange
        let effects = vec![RowEffect::Node(node_row()), RowEffect::Edge(edge_row())];
        let event = event_with(LogPayload::EpisodeBatch(effects));

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn episode_batch_with_zero_effects_round_trips() {
        // Arrange
        let event = event_with(LogPayload::EpisodeBatch(Vec::new()));

        // Act and Assert
        assert_round_trips(event);
    }

    fn tombstone_targets() -> Vec<TombstoneTarget> {
        vec![
            TombstoneTarget {
                table: TombstoneTable::Nodes,
                id: "node-1".into(),
            },
            TombstoneTarget {
                table: TombstoneTable::Edges,
                id: "edge-1".into(),
            },
            TombstoneTarget {
                table: TombstoneTable::NodeCommunity,
                id: "node-2".into(),
            },
        ]
    }

    #[test]
    fn tombstone_round_trips() {
        // Arrange
        let event = event_with(LogPayload::Tombstone(tombstone_targets()));

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn duplicate_of_round_trips() {
        // Arrange
        let event = event_with(LogPayload::DuplicateOf {
            first_event_id: "event-0".into(),
        });

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn voided_round_trips() {
        // Arrange
        let event = event_with(LogPayload::Voided {
            target_event_id: "event-0".into(),
        });

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn empty_encryption_key_id_is_distinct_from_none() {
        // Arrange
        let none_event = event_with(LogPayload::NodeWrite(node_row()));
        let mut empty_event = none_event.clone();
        empty_event.encryption_key_id = Some(String::new());

        // Act
        let none_bytes = none_event.encode().expect("encode none");
        let empty_bytes = empty_event.encode().expect("encode empty");

        // Assert
        assert_ne!(none_bytes, empty_bytes);
        assert_eq!(LogEvent::decode(&empty_bytes), Ok(empty_event));
    }

    #[test]
    fn encryption_key_id_some_round_trips() {
        // Arrange
        let mut event = event_with(LogPayload::NodeWrite(node_row()));
        event.encryption_key_id = Some("key-1".into());

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn older_schema_version_round_trips() {
        // Arrange
        let mut event = event_with(LogPayload::NodeWrite(node_row()));
        event.schema_version = CURRENT_SCHEMA_VERSION - 1;

        // Act and Assert
        assert_round_trips(event);
    }

    fn versioned(version: u32, body: &[u8]) -> Vec<u8> {
        let mut bytes = postcard::to_stdvec(&version).expect("encode version prefix");
        bytes.extend_from_slice(body);
        bytes
    }

    #[test]
    fn future_version_with_undecodable_payload_is_unsupported_version() {
        // Arrange
        let future = CURRENT_SCHEMA_VERSION + 1;
        let garbage = [0xFF; 8];
        assert!(
            postcard::from_bytes::<LogEvent>(&garbage).is_err(),
            "fixture payload must not decode as an event"
        );
        let bytes = versioned(future, &garbage);

        // Act
        let result = LogEvent::decode(&bytes);

        // Assert
        assert_eq!(
            result,
            Err(EventError::UnsupportedSchemaVersion {
                found: future,
                supported: CURRENT_SCHEMA_VERSION,
            })
        );
    }

    #[test]
    fn future_version_with_empty_payload_is_unsupported_version() {
        // Arrange
        let future = CURRENT_SCHEMA_VERSION + 1;
        let bytes = versioned(future, &[]);

        // Act
        let result = LogEvent::decode(&bytes);

        // Assert
        assert_eq!(
            result,
            Err(EventError::UnsupportedSchemaVersion {
                found: future,
                supported: CURRENT_SCHEMA_VERSION,
            })
        );
    }

    #[test]
    fn largest_version_with_valid_current_body_is_unsupported_version() {
        // Arrange: a multi-byte version prefix in front of a body that would decode.
        let valid = event_with(LogPayload::NodeWrite(node_row()));
        let body = postcard::to_stdvec(&valid).expect("encode body");
        let bytes = versioned(u32::MAX, &body);

        // Act
        let result = LogEvent::decode(&bytes);

        // Assert
        assert_eq!(
            result,
            Err(EventError::UnsupportedSchemaVersion {
                found: u32::MAX,
                supported: CURRENT_SCHEMA_VERSION,
            })
        );
    }

    #[test]
    fn empty_input_is_a_decode_error() {
        // Arrange
        let bytes: &[u8] = &[];

        // Act
        let result = LogEvent::decode(bytes);

        // Assert
        assert_eq!(
            result,
            Err(EventError::Decode(
                postcard::Error::DeserializeUnexpectedEnd
            ))
        );
    }

    #[test]
    fn truncated_version_prefix_is_a_decode_error() {
        // Arrange: a continuation bit with no following byte cuts the varint prefix short.
        let bytes = [0x80];

        // Act
        let result = LogEvent::decode(&bytes);

        // Assert
        assert!(
            matches!(result, Err(EventError::Decode(_))),
            "expected a decode error, got {result:?}"
        );
    }

    #[test]
    fn truncated_encoding_is_a_decode_error() {
        // Arrange
        let encoded = event_with(LogPayload::NodeWrite(node_row()))
            .encode()
            .expect("encode event");
        let cut_lengths = [0, 1, 2, encoded.len() / 2, encoded.len() - 1];

        for cut in cut_lengths {
            // Act
            let result = LogEvent::decode(&encoded[..cut]);

            // Assert
            assert!(
                matches!(result, Err(EventError::Decode(_))),
                "cut at {cut} of {} bytes: expected a decode error, got {result:?}",
                encoded.len()
            );
        }
    }

    #[test]
    fn encode_rejects_a_version_newer_than_current() {
        // Arrange
        let mut event = event_with(LogPayload::NodeWrite(node_row()));
        event.schema_version = CURRENT_SCHEMA_VERSION + 1;

        // Act
        let result = event.encode();

        // Assert
        assert_eq!(
            result,
            Err(EventError::UnsupportedSchemaVersion {
                found: CURRENT_SCHEMA_VERSION + 1,
                supported: CURRENT_SCHEMA_VERSION,
            })
        );
    }

    #[test]
    fn schema_version_is_carried_only_by_the_frame_prefix() {
        // Arrange
        let current = event_with(LogPayload::NodeWrite(node_row()));
        let mut older = current.clone();
        older.schema_version = CURRENT_SCHEMA_VERSION - 1;

        // Act
        let current_bytes = current.encode().expect("encode current");
        let older_bytes = older.encode().expect("encode older");

        // Assert
        assert_eq!(current_bytes[1..], older_bytes[1..]);
        assert_ne!(current_bytes[0], older_bytes[0]);
    }

    #[test]
    fn trailing_bytes_after_a_valid_event_are_rejected() {
        // Arrange
        let mut bytes = event_with(LogPayload::NodeWrite(node_row()))
            .encode()
            .expect("encode event");
        bytes.push(0);

        // Act
        let result = LogEvent::decode(&bytes);

        // Assert
        assert_eq!(result, Err(EventError::TrailingBytes { count: 1 }));
    }

    #[test]
    fn extreme_integers_empty_strings_and_zero_hash_round_trip() {
        // Arrange
        let node = NodeRow {
            id: String::new(),
            kind: String::new(),
            label: String::new(),
            content: String::new(),
            producer: String::new(),
            attributes: String::new(),
            scope: Some(String::new()),
            subject: Some(String::new()),
            confidence: 0.0,
            valid_from: i64::MIN,
            valid_from_supplied: false,
            valid_until: i64::MAX,
            tx_from: i64::MIN,
            tx_to: i64::MAX,
        };
        let edge = EdgeRow {
            id: String::new(),
            src: String::new(),
            dst: String::new(),
            edge_type: String::new(),
            attributes: String::new(),
            tx_from: i64::MIN,
            tx_to: i64::MAX,
        };
        let payloads = [
            LogPayload::NodeWrite(node.clone()),
            LogPayload::EdgeWrite(edge.clone()),
            LogPayload::EpisodeBatch(vec![RowEffect::Node(node), RowEffect::Edge(edge)]),
            LogPayload::EpisodeBatch(Vec::new()),
        ];

        for payload in payloads {
            let mut event = event_with(payload);
            event.content_hash = [0; 32];
            event.observed_at = i64::MIN;
            event.ingested_at = i64::MAX;

            // Act and Assert
            assert_round_trips(event);
        }
    }

    #[test]
    fn special_floats_survive_a_round_trip_bit_for_bit() {
        // Arrange
        let specials = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0];

        for special in specials {
            let mut node = node_row();
            node.confidence = special;
            let mut event = event_with(LogPayload::NodeWrite(node));
            event.trust_score = special;
            let encoded = event.encode().expect("encode event");

            // Act
            let decoded = LogEvent::decode(&encoded).expect("decode event");

            // Assert
            let LogPayload::NodeWrite(decoded_node) = &decoded.payload else {
                panic!("expected a node write, got {:?}", decoded.payload);
            };
            assert_eq!(decoded_node.confidence.to_bits(), special.to_bits());
            assert_eq!(decoded.trust_score.to_bits(), special.to_bits());
            assert_eq!(decoded.encode().expect("re-encode"), encoded);
        }
    }

    fn from_hex(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex"))
            .collect()
    }

    #[test]
    fn v1_wire_format_is_pinned() {
        // Arrange: one fixed event per payload variant, in declaration order.
        let golden = [
            (LogPayload::NodeWrite(node_row()), "01076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f0000066e6f64652d310466616374056c6162656c07636f6e74656e74076167656e742d61027b7d010670726f6a2f6100000000000000e83fd00f0180e09ecce5ee01a01f80e09ecce5ee01"),
            (LogPayload::EdgeWrite(edge_row()), "01076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f000106656467652d31066e6f64652d31066e6f64652d320a72656c617465735f746f027b7da01f80e09ecce5ee01"),
            (
                LogPayload::EpisodeBatch(vec![
                    RowEffect::Node(node_row()),
                    RowEffect::Edge(edge_row()),
                ]),
                "01076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f00020200066e6f64652d310466616374056c6162656c07636f6e74656e74076167656e742d61027b7d010670726f6a2f6100000000000000e83fd00f0180e09ecce5ee01a01f80e09ecce5ee010106656467652d31066e6f64652d31066e6f64652d320a72656c617465735f746f027b7da01f80e09ecce5ee01",
            ),
            (
                LogPayload::Tombstone(tombstone_targets()),
                "01076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f00030300066e6f64652d310106656467652d3102066e6f64652d32",
            ),
            (
                LogPayload::DuplicateOf {
                    first_event_id: "event-0".into(),
                },
                "01076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f0004076576656e742d30",
            ),
            (
                LogPayload::Voided {
                    target_event_id: "event-0".into(),
                },
                "01076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f0005076576656e742d30",
            ),
        ];

        for (payload, hex) in golden {
            let event = event_with(payload);
            let expected_bytes = from_hex(hex);

            // Act
            let encoded = event.encode().expect("encode event");
            let decoded = LogEvent::decode(&expected_bytes);

            // Assert
            assert_eq!(encoded, expected_bytes, "encoding drifted for {event:?}");
            assert_eq!(decoded, Ok(event));
        }
    }
}
