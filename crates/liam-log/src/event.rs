// SPDX-License-Identifier: Apache-2.0

//! The event record the log stores, and its postcard wire framing.
//!
//! An event holds the fully resolved row values, not the request that produced
//! them, so replaying the log re-applies recorded rows without re-minting ids
//! or re-reading a clock.

use serde::{Deserialize, Serialize};

use crate::envelope::SealedRecord;

mod legacy;

use legacy::{LegacyEvent, LegacyPayload};

/// Schema version this build writes and the newest it can read. Version 2 adds
/// the fact payloads, version 4 the derived artifacts and version 5 the usage,
/// feedback and tier payloads and version 6 the sealed payload, which an older
/// build cannot replay. Version 3 adds the derivation and basis of a node row, which
/// changes its layout, so frames of versions before it are read through
/// [`legacy`]. Every frame comes back stamped with this version, since what
/// is returned is the current representation of the event.
pub const CURRENT_SCHEMA_VERSION: u32 = 6;

/// The first version whose node rows carry a derivation and a basis.
pub(crate) const NODE_DERIVATION_SCHEMA_VERSION: u32 = 3;

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
    /// The event is stamped with an older version than the layout it would be
    /// written in, so the frame's prefix would describe the wrong layout.
    #[error("event is stamped schema version {found}, but this build writes version {current}")]
    StaleSchemaVersion { found: u32, current: u32 },
}

/// A log event a node row rests on, as a node's basis lists it. `synthetic` is
/// set when no log record backs the id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasisRef {
    pub event_id: String,
    pub synthetic: bool,
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
    /// default, so dedup hashing can ignore the minted value. Backfilled rows
    /// default to true, because hashing the stored value never dedupes wrongly.
    pub valid_from_supplied: bool,
    pub valid_until: i64,
    pub tx_from: i64,
    pub tx_to: i64,
    /// How the row's text came about: `recorded`, `extracted` or `inferred`.
    /// `None` only on a row read from an event written before the field
    /// existed, whose derivation a replay works out from the rows around it.
    pub derivation: Option<String>,
    /// The events a synthesized row rests on, beyond the one that wrote it.
    pub basis: Vec<BasisRef>,
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

/// A `facts` row with every value already resolved. `scope` and `closed_by`
/// are empty strings for none, as the table stores them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactRowEffect {
    pub id: String,
    pub lineage_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_key: String,
    pub scope: String,
    pub evidence_at: i64,
    pub valid_from: i64,
    pub valid_until: i64,
    pub tx_from: i64,
    pub tx_to: i64,
    pub closed_by: String,
    pub confidence: f64,
    pub trust_score: f64,
    pub extractor_id: String,
    pub extractor_version: String,
}

/// A `fact_suppressions` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuppressionEffect {
    pub subject: String,
    pub predicate: String,
    pub scope: String,
    pub object_key: String,
    pub trust_score: f64,
    pub retracted_at: i64,
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
    /// What one fact write decided, resolved, so a replay applies it without
    /// arbitrating again. The versions in `closed_version_ids` get `tx_to = at`.
    FactWrite {
        at: i64,
        closed_version_ids: Vec<String>,
        inserted: Vec<FactRowEffect>,
        /// `(lineage_id, event_id)` pairs.
        source_links: Vec<(String, String)>,
        suppressions: Vec<SuppressionEffect>,
    },
    /// Cancels the fact writes whose source is the consolidation run `run_id`.
    RunRolledBack {
        run_id: String,
    },
    /// A text composed from other rows, with the rows and events it rests on.
    DerivedArtifact(DerivedArtifact),
    /// Ends a derived artifact that nothing replaces, such as a cluster that
    /// dissolved.
    DerivedArtifactClosed {
        artifact_id: String,
        at: i64,
    },
    /// The reads that returned each lineage in a period, counted in memory and
    /// flushed together. `day` is the UTC day of the flush, in days since the
    /// epoch. An empty batch is legal and marks when counting began.
    AccessBatch {
        day: i64,
        entries: Vec<UsageRow>,
    },
    /// A producer's verdict on a result, by the lineage the result came from.
    Feedback {
        lineage_key: String,
        correct: bool,
        producer: String,
        at: i64,
    },
    /// A lineage put in a tier, with the residency it carries so a replay needs
    /// no configuration.
    TierMove {
        lineage_key: String,
        to: TierName,
        run_id: String,
        reason: String,
        changed_at: i64,
        residency_until: Option<i64>,
    },
    /// Rows of another payload, each sealed under its own key group.
    Sealed(SealedRecord),
}

/// The reads of one lineage in an `AccessBatch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageRow {
    pub lineage_key: String,
    pub hits: u32,
    pub last_access_at: i64,
}

/// The tier a `TierMove` puts a lineage in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TierName {
    Hot,
    Cold,
}

/// The `kind` of a summary of a cluster of nodes.
pub const ARTIFACT_KIND_SUMMARY: &str = "summary";

/// A composed text, resolved so a replay stores it without composing again. The
/// artifacts in `closes` end at `at` as this one begins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedArtifact {
    pub artifact_id: String,
    pub kind: String,
    /// How many compositions deep: 1 summarizes nodes, 2 summarizes level 1.
    pub level: u8,
    /// The scope of the rows composed, empty for none.
    pub scope: String,
    pub text: String,
    /// The nodes (level 1) or lower artifacts the text was composed from.
    pub child_ids: Vec<String>,
    /// Every log event behind the children, so tracing needs no walk down.
    pub source_event_ids: Vec<BasisRef>,
    /// How many of those events had their item text in the prompt.
    pub events_fed: u32,
    pub derivation: String,
    pub model_id: String,
    pub prompt_version: String,
    /// Artifacts this one replaces.
    pub closes: Vec<String>,
    pub valid_min: i64,
    pub valid_max: i64,
    /// When the artifact begins, in transaction time.
    pub at: i64,
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
    /// The key group of the first sealed row, `None` for a plain event.
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
        if self.schema_version < CURRENT_SCHEMA_VERSION {
            return Err(EventError::StaleSchemaVersion {
                found: self.schema_version,
                current: CURRENT_SCHEMA_VERSION,
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
        let (event, rest) = if version < NODE_DERIVATION_SCHEMA_VERSION {
            let (legacy, rest) =
                postcard::take_from_bytes::<LegacyEvent>(body).map_err(EventError::Decode)?;
            (LogEvent::from(legacy), rest)
        } else {
            let (mut event, rest) =
                postcard::take_from_bytes::<LogEvent>(body).map_err(EventError::Decode)?;
            event.schema_version = CURRENT_SCHEMA_VERSION;
            (event, rest)
        };
        if !rest.is_empty() {
            return Err(EventError::TrailingBytes { count: rest.len() });
        }
        Ok(event)
    }
}

/// Decodes a payload written by a build that stamped `version`, in the layout
/// of that version, and returns it in the current one.
pub fn decode_payload(version: u32, bytes: &[u8]) -> Result<LogPayload, postcard::Error> {
    if version < NODE_DERIVATION_SCHEMA_VERSION {
        return postcard::from_bytes::<LegacyPayload>(bytes).map(LogPayload::from);
    }
    postcard::from_bytes(bytes)
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
            derivation: None,
            basis: Vec::new(),
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

    fn fact_write() -> LogPayload {
        let version = FactRowEffect {
            id: "fact-1".into(),
            lineage_id: "lineage-1".into(),
            subject: "alice".into(),
            predicate: "lives_in".into(),
            object: "Sydney".into(),
            object_key: "sydney".into(),
            scope: String::new(),
            evidence_at: 1_000,
            valid_from: 1_000,
            valid_until: 4_102_444_800_000,
            tx_from: 2_000,
            tx_to: 4_102_444_800_000,
            closed_by: String::new(),
            confidence: 1.0,
            trust_score: 0.75,
            extractor_id: "rules".into(),
            extractor_version: "v1".into(),
        };
        let suppression = SuppressionEffect {
            subject: "alice".into(),
            predicate: "lives_in".into(),
            scope: String::new(),
            object_key: "perth".into(),
            trust_score: 0.5,
            retracted_at: 3_000,
        };
        LogPayload::FactWrite {
            at: 2_000,
            closed_version_ids: vec!["fact-0".into()],
            inserted: vec![version],
            source_links: vec![("lineage-1".into(), "event-9".into())],
            suppressions: vec![suppression],
        }
    }

    #[test]
    fn fact_write_round_trips() {
        // Arrange
        let event = event_with(fact_write());

        // Act and Assert
        assert_round_trips(event);
    }

    #[test]
    fn run_rolled_back_round_trips() {
        // Arrange
        let event = event_with(LogPayload::RunRolledBack {
            run_id: "run-1".into(),
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
    fn an_event_stamped_with_an_older_version_is_not_encoded() {
        // Arrange
        let mut event = event_with(LogPayload::NodeWrite(node_row()));
        event.schema_version = CURRENT_SCHEMA_VERSION - 1;

        // Act
        let result = event.encode();

        // Assert
        assert_eq!(
            result,
            Err(EventError::StaleSchemaVersion {
                found: CURRENT_SCHEMA_VERSION - 1,
                current: CURRENT_SCHEMA_VERSION,
            })
        );
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
    fn the_frame_starts_with_the_current_version_and_the_body_does_not_repeat_it() {
        // Arrange
        let event = event_with(LogPayload::NodeWrite(node_row()));
        let prefix = postcard::to_stdvec(&CURRENT_SCHEMA_VERSION).expect("encode version prefix");
        let body = postcard::to_stdvec(&event).expect("encode body");

        // Act
        let encoded = event.encode().expect("encode event");

        // Assert
        assert_eq!(encoded, [prefix, body].concat());
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
            derivation: None,
            basis: Vec::new(),
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
    fn v1_frames_are_still_read() {
        // Arrange: one fixed frame per version 1 payload variant, in declaration order.
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
            let frame = from_hex(hex);

            // Act
            let decoded = LogEvent::decode(&frame);

            // Assert: a version 1 frame reads back in the current layout, with
            // no derivation on its rows, and is stamped with the current version.
            assert_eq!(decoded, Ok(event));
        }
    }

    #[test]
    fn v2_fact_payloads_wire_format_is_pinned() {
        // Arrange
        let golden = [
            (fact_write(), "02076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f0006a01f0106666163742d300106666163742d31096c696e656167652d3105616c696365086c697665735f696e065379646e6579067379646e657900d00fd00f80e09ecce5ee01a01f80e09ecce5ee0100000000000000f03f000000000000e83f0572756c657302763101096c696e656167652d31076576656e742d390105616c696365086c697665735f696e00057065727468000000000000e03ff02e"),
            (
                LogPayload::RunRolledBack {
                    run_id: "run-1".into(),
                },
                "02076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f00070572756e2d31",
            ),
        ];

        for (payload, hex) in golden {
            let event = event_with(payload);
            let frame = from_hex(hex);

            // Act
            let decoded = LogEvent::decode(&frame);

            // Assert
            assert_eq!(decoded, Ok(event));
        }
    }

    #[test]
    fn a_payload_is_read_in_the_layout_of_its_version() {
        // Arrange
        let legacy = LegacyPayload::EpisodeBatch(vec![legacy::LegacyRowEffect::Node(
            legacy::LegacyNodeRow {
                id: "node-1".into(),
                kind: "fact".into(),
                label: "label".into(),
                content: "content".into(),
                producer: "agent-a".into(),
                attributes: "{}".into(),
                scope: None,
                subject: Some("label".into()),
                confidence: 1.0,
                valid_from: 1,
                valid_from_supplied: false,
                valid_until: 2,
                tx_from: 3,
                tx_to: 4,
            },
        )]);
        let older = postcard::to_stdvec(&legacy).expect("encode legacy payload");
        let current = LogPayload::NodeWrite(node_row());
        let newer = postcard::to_stdvec(&current).expect("encode payload");

        // Act
        let from_older = decode_payload(NODE_DERIVATION_SCHEMA_VERSION - 1, &older);
        let from_current = decode_payload(CURRENT_SCHEMA_VERSION, &newer);

        // Assert
        let Ok(LogPayload::EpisodeBatch(effects)) = from_older else {
            panic!("expected an episode batch, got {from_older:?}");
        };
        let [RowEffect::Node(row)] = effects.as_slice() else {
            panic!("expected one node, got {effects:?}");
        };
        assert_eq!((row.id.as_str(), row.derivation.clone()), ("node-1", None));
        assert!(row.basis.is_empty());
        assert_eq!(from_current, Ok(current));
    }

    #[test]
    fn v3_node_rows_are_still_read() {
        // Arrange
        let mut inferred = node_row();
        inferred.derivation = Some("inferred".into());
        inferred.basis = vec![
            BasisRef {
                event_id: "event-7".into(),
                synthetic: false,
            },
            BasisRef {
                event_id: "event-8".into(),
                synthetic: true,
            },
        ];
        let event = event_with(LogPayload::NodeWrite(inferred));
        let golden = from_hex("03076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f0000066e6f64652d310466616374056c6162656c07636f6e74656e74076167656e742d61027b7d010670726f6a2f6100000000000000e83fd00f0180e09ecce5ee01a01f80e09ecce5ee010108696e66657272656402076576656e742d3700076576656e742d3801");

        // Act
        let decoded = LogEvent::decode(&golden);

        // Assert
        assert_eq!(decoded, Ok(event));
    }

    fn artifact() -> DerivedArtifact {
        DerivedArtifact {
            artifact_id: "artifact-1".into(),
            kind: ARTIFACT_KIND_SUMMARY.into(),
            level: 2,
            scope: "proj/a".into(),
            text: "A summary.".into(),
            child_ids: vec!["child-1".into(), "child-2".into()],
            source_event_ids: vec![
                BasisRef {
                    event_id: "event-7".into(),
                    synthetic: false,
                },
                BasisRef {
                    event_id: "event-8".into(),
                    synthetic: true,
                },
            ],
            events_fed: 1,
            derivation: "inferred".into(),
            model_id: "model-a".into(),
            prompt_version: "prompt-1".into(),
            closes: vec!["artifact-0".into()],
            valid_min: 1_000,
            valid_max: 2_000,
            at: 3_000,
        }
    }

    #[test]
    fn derived_artifact_payloads_round_trip() {
        // Arrange
        let payloads = [
            LogPayload::DerivedArtifact(artifact()),
            LogPayload::DerivedArtifactClosed {
                artifact_id: "artifact-1".into(),
                at: 4_000,
            },
        ];

        for payload in payloads {
            // Act and Assert
            assert_round_trips(event_with(payload));
        }
    }

    #[test]
    fn v4_artifact_frames_still_decode_and_v5_frames_carry_the_new_prefix() {
        // Arrange
        let golden = [
            (LogPayload::DerivedArtifact(artifact()), "04076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f00080a61727469666163742d310773756d6d617279020670726f6a2f610a412073756d6d6172792e02076368696c642d31076368696c642d3202076576656e742d3700076576656e742d38010108696e666572726564076d6f64656c2d610870726f6d70742d31010a61727469666163742d30d00fa01ff02e"),
            (
                LogPayload::DerivedArtifactClosed {
                    artifact_id: "artifact-1".into(),
                    at: 4_000,
                },
                "04076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f00090a61727469666163742d31c03e",
            ),
        ];

        for (payload, hex) in golden {
            let event = event_with(payload);
            let v4_bytes = from_hex(hex);

            // Act
            let decoded = LogEvent::decode(&v4_bytes);
            let encoded = event.encode().expect("encode event");

            // Assert: the old frame reads back as the current event, and the
            // new frame differs from it in the version prefix alone
            assert_eq!(decoded, Ok(event.clone()), "{event:?}");
            assert_eq!(encoded[0], CURRENT_SCHEMA_VERSION as u8);
            assert_eq!(encoded[1..], v4_bytes[1..], "layout drifted for {event:?}");
        }
    }

    fn tier_payloads() -> [LogPayload; 4] {
        [
            LogPayload::AccessBatch {
                day: 20_000,
                entries: vec![
                    UsageRow {
                        lineage_key: "n:3:abc".into(),
                        hits: 4,
                        last_access_at: 1_728_000_000_000,
                    },
                    UsageRow {
                        lineage_key: "s:!:1:x:0:".into(),
                        hits: 1,
                        last_access_at: 5,
                    },
                ],
            },
            LogPayload::AccessBatch {
                day: 20_001,
                entries: Vec::new(),
            },
            LogPayload::Feedback {
                lineage_key: "n:3:abc".into(),
                correct: false,
                producer: "agent-a".into(),
                at: 1_728_000_000_000,
            },
            LogPayload::TierMove {
                lineage_key: "n:3:abc".into(),
                to: TierName::Cold,
                run_id: "run-1".into(),
                reason: "unused".into(),
                changed_at: 1_728_000_000_000,
                residency_until: Some(1_730_000_000_000),
            },
        ]
    }

    #[test]
    fn usage_feedback_and_tier_payloads_round_trip() {
        for payload in tier_payloads() {
            // Act and Assert
            assert_round_trips(event_with(payload));
        }
        assert_round_trips(event_with(LogPayload::TierMove {
            lineage_key: "n:1:x".into(),
            to: TierName::Hot,
            run_id: "run-2".into(),
            reason: "used".into(),
            changed_at: 7,
            residency_until: None,
        }));
    }

    #[test]
    fn v5_usage_feedback_and_tier_payloads_wire_format_is_pinned() {
        // Arrange
        let golden = [
            "05076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f000ac0b80202076e3a333a6162630480809ccdca640a733a213a313a783a303a010a",
            "05076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f000ac2b80200",
            "05076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f000b076e3a333a61626300076167656e742d6180809ccdca64",
            "05076576656e742d310707070707070707070707070707070707070707070707070707070707070707076167656e742d61cdccccccccccec3fd00fa01f000c076e3a333a616263010572756e2d3106756e7573656480809ccdca640180d0c8c0d964",
        ];

        for (payload, hex) in tier_payloads().into_iter().zip(golden) {
            let event = event_with(payload);
            let mut encoded = event.encode().expect("encode event");
            let expected_bytes = from_hex(hex);

            // Act
            let decoded = LogEvent::decode(&expected_bytes);

            // Assert: a current frame differs from a v5 one only in its prefix.
            assert_eq!(encoded[0], CURRENT_SCHEMA_VERSION as u8);
            encoded[0] = expected_bytes[0];
            assert_eq!(encoded, expected_bytes, "encoding drifted for {event:?}");
            assert_eq!(decoded, Ok(event));
        }
    }

    #[test]
    fn payload_variant_indexes_are_pinned_so_only_appending_is_legal() {
        // Arrange
        let artifact = DerivedArtifact {
            artifact_id: "a".into(),
            kind: ARTIFACT_KIND_SUMMARY.into(),
            level: 1,
            scope: String::new(),
            text: String::new(),
            child_ids: Vec::new(),
            source_event_ids: Vec::new(),
            events_fed: 0,
            derivation: String::new(),
            model_id: String::new(),
            prompt_version: String::new(),
            closes: Vec::new(),
            valid_min: 0,
            valid_max: 0,
            at: 0,
        };
        let payloads = [
            LogPayload::NodeWrite(node_row()),
            LogPayload::EdgeWrite(edge_row()),
            LogPayload::EpisodeBatch(Vec::new()),
            LogPayload::Tombstone(Vec::new()),
            LogPayload::DuplicateOf {
                first_event_id: String::new(),
            },
            LogPayload::Voided {
                target_event_id: String::new(),
            },
            LogPayload::FactWrite {
                at: 0,
                closed_version_ids: Vec::new(),
                inserted: Vec::new(),
                source_links: Vec::new(),
                suppressions: Vec::new(),
            },
            LogPayload::RunRolledBack {
                run_id: String::new(),
            },
            LogPayload::DerivedArtifact(artifact),
            LogPayload::DerivedArtifactClosed {
                artifact_id: String::new(),
                at: 0,
            },
            LogPayload::AccessBatch {
                day: 0,
                entries: Vec::new(),
            },
            LogPayload::Feedback {
                lineage_key: String::new(),
                correct: true,
                producer: String::new(),
                at: 0,
            },
            LogPayload::TierMove {
                lineage_key: String::new(),
                to: TierName::Hot,
                run_id: String::new(),
                reason: String::new(),
                changed_at: 0,
                residency_until: None,
            },
            LogPayload::Sealed(SealedRecord {
                shape: crate::envelope::SealedShape::Node,
                rows: Vec::new(),
            }),
        ];

        for (index, payload) in payloads.iter().enumerate() {
            // Act
            let bytes = postcard::to_stdvec(payload).expect("encode payload");

            // Assert
            assert_eq!(usize::from(bytes[0]), index, "{payload:?}");
        }
    }

    #[test]
    fn this_build_writes_schema_version_six() {
        // Assert
        assert_eq!(CURRENT_SCHEMA_VERSION, 6);
    }
}
