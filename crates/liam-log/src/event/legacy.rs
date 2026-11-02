// SPDX-License-Identifier: Apache-2.0
//! The layout of events written before node rows carried a derivation and a
//! basis. The wire format is positional, so those frames can only be read with
//! the layout they were written in; they are converted to the current types
//! with the new fields left unset, which is how a replay tells them apart.

use serde::{Deserialize, Serialize};

use super::{
    EdgeRow, FactRowEffect, LogEvent, LogPayload, NodeRow, RowEffect, SuppressionEffect,
    TombstoneTarget, CURRENT_SCHEMA_VERSION,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct LegacyNodeRow {
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
    pub valid_from_supplied: bool,
    pub valid_until: i64,
    pub tx_from: i64,
    pub tx_to: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) enum LegacyRowEffect {
    Node(LegacyNodeRow),
    Edge(EdgeRow),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) enum LegacyPayload {
    NodeWrite(LegacyNodeRow),
    EdgeWrite(EdgeRow),
    EpisodeBatch(Vec<LegacyRowEffect>),
    Tombstone(Vec<TombstoneTarget>),
    DuplicateOf {
        first_event_id: String,
    },
    Voided {
        target_event_id: String,
    },
    FactWrite {
        at: i64,
        closed_version_ids: Vec<String>,
        inserted: Vec<FactRowEffect>,
        source_links: Vec<(String, String)>,
        suppressions: Vec<SuppressionEffect>,
    },
    RunRolledBack {
        run_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct LegacyEvent {
    pub event_id: String,
    pub content_hash: [u8; 32],
    pub source: String,
    pub trust_score: f64,
    pub observed_at: i64,
    pub ingested_at: i64,
    pub encryption_key_id: Option<String>,
    pub payload: LegacyPayload,
}

impl From<LegacyNodeRow> for NodeRow {
    fn from(row: LegacyNodeRow) -> Self {
        Self {
            id: row.id,
            kind: row.kind,
            label: row.label,
            content: row.content,
            producer: row.producer,
            attributes: row.attributes,
            scope: row.scope,
            subject: row.subject,
            confidence: row.confidence,
            valid_from: row.valid_from,
            valid_from_supplied: row.valid_from_supplied,
            valid_until: row.valid_until,
            tx_from: row.tx_from,
            tx_to: row.tx_to,
            derivation: None,
            basis: Vec::new(),
        }
    }
}

impl From<LegacyRowEffect> for RowEffect {
    fn from(effect: LegacyRowEffect) -> Self {
        match effect {
            LegacyRowEffect::Node(row) => Self::Node(row.into()),
            LegacyRowEffect::Edge(row) => Self::Edge(row),
        }
    }
}

impl From<LegacyPayload> for LogPayload {
    fn from(payload: LegacyPayload) -> Self {
        match payload {
            LegacyPayload::NodeWrite(row) => Self::NodeWrite(row.into()),
            LegacyPayload::EdgeWrite(row) => Self::EdgeWrite(row),
            LegacyPayload::EpisodeBatch(effects) => {
                Self::EpisodeBatch(effects.into_iter().map(Into::into).collect())
            }
            LegacyPayload::Tombstone(targets) => Self::Tombstone(targets),
            LegacyPayload::DuplicateOf { first_event_id } => Self::DuplicateOf { first_event_id },
            LegacyPayload::Voided { target_event_id } => Self::Voided { target_event_id },
            LegacyPayload::FactWrite {
                at,
                closed_version_ids,
                inserted,
                source_links,
                suppressions,
            } => Self::FactWrite {
                at,
                closed_version_ids,
                inserted,
                source_links,
                suppressions,
            },
            LegacyPayload::RunRolledBack { run_id } => Self::RunRolledBack { run_id },
        }
    }
}

impl From<LegacyEvent> for LogEvent {
    /// The event in the current layout and stamped with the current version, so
    /// writing it again produces a frame its version prefix describes.
    fn from(event: LegacyEvent) -> Self {
        Self {
            event_id: event.event_id,
            content_hash: event.content_hash,
            source: event.source,
            trust_score: event.trust_score,
            observed_at: event.observed_at,
            ingested_at: event.ingested_at,
            encryption_key_id: event.encryption_key_id,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload: event.payload.into(),
        }
    }
}
