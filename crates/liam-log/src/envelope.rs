// SPDX-License-Identifier: Apache-2.0

//! Sealed log records: a payload split into rows, each sealed under the key of
//! its own key group.
//!
//! A sealed event is still one ordinary [`LogEvent`], so the WAL framing and
//! the Parquet columns are unchanged. Destroying a key group makes its rows
//! unreadable while the rest of the record, and the header, stay intact, which
//! is how a delete reaches into an episode that holds several subjects.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use futures_util::future;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::crypto::{
    aead_open, aead_seal, hmac_sha256, CryptoError, KeyLookup, KeyMaterial, KeyRegistry, NONCE_LEN,
};
use crate::event::{LogEvent, LogPayload, RowEffect, UsageRow};
use crate::hash::{edge_row_hash, node_row_hash};
use crate::reader::{LogReader, LogRecord, LogStream, ReaderError};
use crate::{KeyId, LogOffset};

/// What a row of a sealed record holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowKind {
    Node,
    Edge,
    Fact,
    Artifact,
    Usage,
    Feedback,
    Tier,
}

/// How the rows of a sealed record go back together into a plain payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SealedShape {
    Node,
    Edge,
    Episode,
    Fact,
    Artifact,
    /// `day` stays readable, so a day's usage replays even when every entry is
    /// shredded.
    Usage {
        day: i64,
    },
    Feedback,
    Tier,
}

/// One row sealed under its key group. `row_id` is the plain id of the row
/// (a random UUID) and is empty for entries that have none, such as usage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedRow {
    pub row_id: String,
    pub kind: RowKind,
    pub key_id: KeyId,
    pub nonce: [u8; NONCE_LEN],
    pub body: Vec<u8>,
}

/// The payload of a sealed event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedRecord {
    pub shape: SealedShape,
    pub rows: Vec<SealedRow>,
}

/// What a sealed row decrypts to.
#[derive(Serialize, Deserialize)]
enum RowBody {
    Effect(RowEffect),
    Usage(UsageRow),
    /// A payload that is one row by itself.
    Whole(LogPayload),
}

/// Key ids the store read from `node_keys` inside its open transaction, for
/// rows whose group is not decided by the event alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyHints {
    /// The key group of an edge source that is not a row of the same event.
    pub edge_sources: HashMap<String, KeyId>,
    /// The key group of a lineage key an access, feedback or tier entry names.
    pub lineages: HashMap<String, KeyId>,
}

#[derive(Debug, thiserror::Error)]
pub enum SealError {
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    #[error("row of event {event_id} could not be encoded: {source}")]
    Encode {
        event_id: String,
        source: postcard::Error,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error("key {key_id} is not in the registry, the registry is lost or out of date")]
    KeyMissing { key_id: KeyId },
    #[error("sealed record {event_id} failed authentication: it was altered or moved")]
    Authentication { event_id: String },
    #[error("sealed record {event_id} holds a row that could not be decoded: {source}")]
    Decode {
        event_id: String,
        source: postcard::Error,
    },
    #[error("sealed record {event_id} has rows that do not fit its shape")]
    Shape { event_id: String },
    #[error(transparent)]
    Crypto(#[from] CryptoError),
}

/// The plain header of one sealed row: which row it is and whose key seals it.
/// A replay reads these to learn which key group each node belongs to, with no
/// key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowKeyRef {
    pub row_id: String,
    pub kind: RowKind,
    pub key_id: KeyId,
}

/// The headers of the rows of a sealed event, empty for a plain one.
pub fn row_keys(event: &LogEvent) -> Vec<RowKeyRef> {
    let LogPayload::Sealed(record) = &event.payload else {
        return Vec::new();
    };
    record
        .rows
        .iter()
        .map(|row| RowKeyRef {
            row_id: row.row_id.clone(),
            kind: row.kind,
            key_id: row.key_id.clone(),
        })
        .collect()
}

/// A record with the rows that could be opened, and the key groups whose rows
/// could not because the group was destroyed or is being destroyed.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenedRecord {
    pub event: LogEvent,
    pub shredded: Vec<KeyId>,
}

/// The header fields every row binds as associated data, with the shape, the
/// row's place, and the row itself. The schema version is left out: a decoded
/// event is stamped with the newest version, which a later build would change.
#[derive(Serialize)]
struct Aad<'a> {
    event_id: &'a str,
    source: &'a str,
    trust_score_bits: u64,
    observed_at: i64,
    ingested_at: i64,
    content_hash: &'a [u8; 32],
    encryption_key_id: Option<&'a str>,
    shape: &'a SealedShape,
    row_count: usize,
    index: usize,
    row_id: &'a str,
    kind: RowKind,
    key_id: &'a str,
}

fn aad_bytes(
    event: &LogEvent,
    shape: &SealedShape,
    row_count: usize,
    index: usize,
    row: AadRow<'_>,
) -> Vec<u8> {
    let aad = Aad {
        event_id: &event.event_id,
        source: &event.source,
        trust_score_bits: event.trust_score.to_bits(),
        observed_at: event.observed_at,
        ingested_at: event.ingested_at,
        content_hash: &event.content_hash,
        encryption_key_id: event.encryption_key_id.as_deref(),
        shape,
        row_count,
        index,
        row_id: row.row_id,
        kind: row.kind,
        key_id: row.key_id,
    };
    // Plain scalars and strings always encode.
    postcard::to_stdvec(&aad).expect("aad fields encode")
}

struct AadRow<'a> {
    row_id: &'a str,
    kind: RowKind,
    key_id: &'a str,
}

/// A row waiting to be sealed.
struct PlainRow {
    row_id: String,
    kind: RowKind,
    key_id: KeyId,
    body: RowBody,
}

/// Seals events with the keys of a registry.
#[derive(Clone)]
pub struct Sealer {
    registry: Arc<KeyRegistry>,
}

impl Sealer {
    pub fn new(registry: Arc<KeyRegistry>) -> Self {
        Self { registry }
    }

    /// Seals `event`. Events that carry no content stay plain, and so does
    /// one with no rows.
    pub fn seal(&self, event: &LogEvent, hints: &KeyHints) -> Result<LogEvent, SealError> {
        let mut keys = Keys::new(&self.registry);
        let Some((shape, rows)) = split_rows(event, hints, &mut keys)? else {
            return Ok(plain_header(event));
        };
        let mut sealed = LogEvent {
            content_hash: header_hash(&rows, &mut keys)?,
            encryption_key_id: rows.first().map(|row| row.key_id.clone()),
            ..event.clone()
        };
        let row_count = rows.len();
        let mut sealed_rows = Vec::with_capacity(row_count);
        for (index, row) in rows.into_iter().enumerate() {
            let bytes = postcard::to_stdvec(&row.body).map_err(|source| SealError::Encode {
                event_id: event.event_id.clone(),
                source,
            })?;
            let aad = aad_bytes(
                &sealed,
                &shape,
                row_count,
                index,
                AadRow {
                    row_id: &row.row_id,
                    kind: row.kind,
                    key_id: &row.key_id,
                },
            );
            let sealed_bytes = aead_seal(&keys.material(&row.key_id)?.data, &aad, &bytes)?;
            sealed_rows.push(SealedRow {
                row_id: row.row_id,
                kind: row.kind,
                key_id: row.key_id,
                nonce: sealed_bytes.nonce,
                body: sealed_bytes.body,
            });
        }
        sealed.payload = LogPayload::Sealed(SealedRecord {
            shape,
            rows: sealed_rows,
        });
        Ok(sealed)
    }

    /// Seals `event` and passes the result to `append` while holding the
    /// registry's write gate, so a key group marked for destruction can never
    /// be written to after the mark.
    pub fn seal_then<T>(
        &self,
        event: &LogEvent,
        hints: &KeyHints,
        append: impl FnOnce(&LogEvent) -> T,
    ) -> Result<T, SealError> {
        let _gate = self.registry.write_gate();
        let sealed = self.seal(event, hints)?;
        Ok(append(&sealed))
    }
}

/// A plain event keeps its content, but a `DuplicateOf` carries no content and
/// its hash would only confirm what was written, so it is zeroed.
fn plain_header(event: &LogEvent) -> LogEvent {
    let mut plain = event.clone();
    if matches!(plain.payload, LogPayload::DuplicateOf { .. }) {
        plain.content_hash = [0; 32];
    }
    plain
}

/// The HMAC, under the dedup key of the first hashable row's group, of that
/// row's plain content hash. All zero when no row is hashable.
fn header_hash(rows: &[PlainRow], keys: &mut Keys<'_>) -> Result<[u8; 32], SealError> {
    let hashable = rows.iter().find_map(|row| match &row.body {
        RowBody::Effect(RowEffect::Node(node)) => Some((&row.key_id, node_row_hash(node))),
        RowBody::Effect(RowEffect::Edge(edge)) => Some((&row.key_id, edge_row_hash(edge))),
        _ => None,
    });
    let Some((key_id, hash)) = hashable else {
        return Ok([0; 32]);
    };
    Ok(hmac_sha256(keys.material(key_id)?.dedup.as_bytes(), &hash))
}

/// Splits a payload into rows and chooses the key group of each. `None` means
/// the payload stays plain.
fn split_rows(
    event: &LogEvent,
    hints: &KeyHints,
    keys: &mut Keys<'_>,
) -> Result<Option<(SealedShape, Vec<PlainRow>)>, SealError> {
    let split = match &event.payload {
        LogPayload::NodeWrite(node) => {
            let effects = [RowEffect::Node(node.clone())];
            (SealedShape::Node, effect_rows(&effects, hints, keys)?)
        }
        LogPayload::EdgeWrite(edge) => {
            let effects = [RowEffect::Edge(edge.clone())];
            (SealedShape::Edge, effect_rows(&effects, hints, keys)?)
        }
        LogPayload::EpisodeBatch(effects) if !effects.is_empty() => {
            (SealedShape::Episode, effect_rows(effects, hints, keys)?)
        }
        LogPayload::FactWrite { inserted, .. } => {
            let row_id = inserted
                .first()
                .map(|fact| fact.id.clone())
                .unwrap_or_default();
            let row = whole_row(RowKind::Fact, row_id, keys.fresh()?, &event.payload);
            (SealedShape::Fact, vec![row])
        }
        LogPayload::DerivedArtifact(artifact) => {
            let row = whole_row(
                RowKind::Artifact,
                artifact.artifact_id.clone(),
                keys.fresh()?,
                &event.payload,
            );
            (SealedShape::Artifact, vec![row])
        }
        LogPayload::AccessBatch { day, entries } if !entries.is_empty() => {
            let mut rows = Vec::with_capacity(entries.len());
            for entry in entries {
                rows.push(PlainRow {
                    row_id: String::new(),
                    kind: RowKind::Usage,
                    key_id: keys.for_lineage(hints, &entry.lineage_key)?,
                    body: RowBody::Usage(entry.clone()),
                });
            }
            (SealedShape::Usage { day: *day }, rows)
        }
        LogPayload::Feedback { lineage_key, .. } => {
            let key_id = keys.for_lineage(hints, lineage_key)?;
            (
                SealedShape::Feedback,
                vec![whole_row(
                    RowKind::Feedback,
                    String::new(),
                    key_id,
                    &event.payload,
                )],
            )
        }
        LogPayload::TierMove { lineage_key, .. } => {
            let key_id = keys.for_lineage(hints, lineage_key)?;
            (
                SealedShape::Tier,
                vec![whole_row(
                    RowKind::Tier,
                    String::new(),
                    key_id,
                    &event.payload,
                )],
            )
        }
        // No content to protect, or nothing to seal.
        LogPayload::EpisodeBatch(_)
        | LogPayload::AccessBatch { .. }
        | LogPayload::Tombstone(_)
        | LogPayload::DuplicateOf { .. }
        | LogPayload::Voided { .. }
        | LogPayload::Quarantined { .. }
        | LogPayload::RunRolledBack { .. }
        | LogPayload::DerivedArtifactClosed { .. }
        | LogPayload::Sealed(_)
        | LogPayload::KeyDestroyed { .. }
        | LogPayload::DeletionRecorded { .. }
        | LogPayload::LineageRemoval { .. }
        | LogPayload::DeletionCancelled { .. } => return Ok(None),
    };
    Ok(Some(split))
}

fn whole_row(kind: RowKind, row_id: String, key_id: KeyId, payload: &LogPayload) -> PlainRow {
    PlainRow {
        row_id,
        kind,
        key_id,
        body: RowBody::Whole(payload.clone()),
    }
}

/// Nodes go under their subject's group, or the event's key when they have
/// none. An edge goes under its source's group, wherever that node is.
fn effect_rows(
    effects: &[RowEffect],
    hints: &KeyHints,
    keys: &mut Keys<'_>,
) -> Result<Vec<PlainRow>, SealError> {
    let mut node_keys: HashMap<&str, KeyId> = HashMap::new();
    for effect in effects {
        if let RowEffect::Node(node) = effect {
            let key_id = match node.subject.as_deref() {
                Some(subject) => keys.for_subject(node.scope.as_deref(), subject)?,
                None => keys.for_event()?,
            };
            node_keys.insert(&node.id, key_id);
        }
    }
    let mut rows = Vec::with_capacity(effects.len());
    for effect in effects {
        let (row_id, kind, key_id) = match effect {
            RowEffect::Node(node) => (
                node.id.clone(),
                RowKind::Node,
                node_keys[node.id.as_str()].clone(),
            ),
            RowEffect::Edge(edge) => {
                let key_id = match node_keys.get(edge.src.as_str()) {
                    Some(key_id) => key_id.clone(),
                    None => match hints.edge_sources.get(&edge.src) {
                        Some(hinted) => keys.hinted(hinted)?,
                        None => keys.for_event()?,
                    },
                };
                (edge.id.clone(), RowKind::Edge, key_id)
            }
        };
        rows.push(PlainRow {
            row_id,
            kind,
            key_id,
            body: RowBody::Effect(effect.clone()),
        });
    }
    Ok(rows)
}

/// The key groups one seal touches, with their unwrapped keys.
struct Keys<'a> {
    registry: &'a KeyRegistry,
    event_key: Option<KeyId>,
    material: HashMap<KeyId, KeyMaterial>,
}

impl<'a> Keys<'a> {
    fn new(registry: &'a KeyRegistry) -> Self {
        Self {
            registry,
            event_key: None,
            material: HashMap::new(),
        }
    }

    fn for_subject(&mut self, scope: Option<&str>, subject: &str) -> Result<KeyId, SealError> {
        let key_id = self.registry.key_for_subject(scope, subject)?;
        self.hinted(&key_id)
    }

    /// The one key all of an event's subjectless rows share.
    fn for_event(&mut self) -> Result<KeyId, SealError> {
        if let Some(key_id) = &self.event_key {
            return Ok(key_id.clone());
        }
        let key_id = self.fresh()?;
        self.event_key = Some(key_id.clone());
        Ok(key_id)
    }

    /// A key of its own, for a row that must be destroyable alone.
    fn fresh(&mut self) -> Result<KeyId, SealError> {
        let key_id = self.registry.new_event_key()?;
        self.hinted(&key_id)
    }

    fn for_lineage(&mut self, hints: &KeyHints, lineage_key: &str) -> Result<KeyId, SealError> {
        match hints.lineages.get(lineage_key) {
            Some(key_id) => self.hinted(key_id),
            None => self.for_event(),
        }
    }

    /// Confirms a key group still takes writes and loads its keys.
    fn hinted(&mut self, key_id: &str) -> Result<KeyId, SealError> {
        self.material(key_id)?;
        Ok(key_id.to_owned())
    }

    fn material(&mut self, key_id: &str) -> Result<&KeyMaterial, SealError> {
        if !self.material.contains_key(key_id) {
            let material = match self.registry.open_key(key_id)? {
                KeyLookup::Active(material) => material,
                KeyLookup::Destroying | KeyLookup::Destroyed => {
                    return Err(CryptoError::KeyGroupLocked {
                        key_id: key_id.to_owned(),
                    }
                    .into())
                }
                KeyLookup::Missing => {
                    return Err(CryptoError::KeyMissing {
                        key_id: key_id.to_owned(),
                    }
                    .into())
                }
            };
            self.material.insert(key_id.to_owned(), material);
        }
        Ok(&self.material[key_id])
    }
}

/// The sources of the edges `payload` writes whose source is not a node of the
/// same payload. Their key groups come from the hints.
pub fn outside_edge_sources(payload: &LogPayload) -> HashSet<String> {
    match payload {
        LogPayload::EdgeWrite(edge) => HashSet::from([edge.src.clone()]),
        LogPayload::EpisodeBatch(effects) => {
            let inside: HashSet<&str> = effects
                .iter()
                .filter_map(|effect| match effect {
                    RowEffect::Node(node) => Some(node.id.as_str()),
                    RowEffect::Edge(_) => None,
                })
                .collect();
            effects
                .iter()
                .filter_map(|effect| match effect {
                    RowEffect::Edge(edge) if !inside.contains(edge.src.as_str()) => {
                        Some(edge.src.clone())
                    }
                    _ => None,
                })
                .collect()
        }
        _ => HashSet::new(),
    }
}

/// The lineage keys an access, feedback or tier payload names. Their key
/// groups come from the hints.
pub fn named_lineages(payload: &LogPayload) -> HashSet<String> {
    match payload {
        LogPayload::AccessBatch { entries, .. } => {
            entries.iter().map(|row| row.lineage_key.clone()).collect()
        }
        LogPayload::Feedback { lineage_key, .. } | LogPayload::TierMove { lineage_key, .. } => {
            HashSet::from([lineage_key.clone()])
        }
        _ => HashSet::new(),
    }
}

/// Opens the sealed rows of `event` whose key groups are alive. A key the
/// registry has never heard of is an error, never a shredded row.
pub fn open_record(registry: &KeyRegistry, event: &LogEvent) -> Result<OpenedRecord, OpenError> {
    let LogPayload::Sealed(record) = &event.payload else {
        return Ok(OpenedRecord {
            event: event.clone(),
            shredded: Vec::new(),
        });
    };
    let mut shredded: Vec<KeyId> = Vec::new();
    let mut bodies = Vec::with_capacity(record.rows.len());
    for (index, row) in record.rows.iter().enumerate() {
        let material = match registry.open_key(&row.key_id)? {
            KeyLookup::Active(material) => material,
            KeyLookup::Destroying | KeyLookup::Destroyed => {
                if !shredded.contains(&row.key_id) {
                    shredded.push(row.key_id.clone());
                }
                continue;
            }
            KeyLookup::Missing => {
                return Err(OpenError::KeyMissing {
                    key_id: row.key_id.clone(),
                })
            }
        };
        let aad = aad_bytes(
            event,
            &record.shape,
            record.rows.len(),
            index,
            AadRow {
                row_id: &row.row_id,
                kind: row.kind,
                key_id: &row.key_id,
            },
        );
        let bytes =
            aead_open(&material.data, &aad, &row.nonce, &row.body).map_err(
                |error| match error {
                    CryptoError::Tampered => OpenError::Authentication {
                        event_id: event.event_id.clone(),
                    },
                    other => OpenError::Crypto(other),
                },
            )?;
        let body: RowBody = postcard::from_bytes(&bytes).map_err(|source| OpenError::Decode {
            event_id: event.event_id.clone(),
            source,
        })?;
        bodies.push(body);
    }
    let payload = assemble(&record.shape, bodies).ok_or_else(|| OpenError::Shape {
        event_id: event.event_id.clone(),
    })?;
    Ok(OpenedRecord {
        event: LogEvent {
            payload,
            ..event.clone()
        },
        shredded,
    })
}

/// Puts the opened rows back into the payload they came from. A shape with no
/// rows left becomes an empty batch, which a replay passes without projecting.
fn assemble(shape: &SealedShape, bodies: Vec<RowBody>) -> Option<LogPayload> {
    let empty = || LogPayload::EpisodeBatch(Vec::new());
    match shape {
        SealedShape::Usage { day } => {
            let entries = bodies
                .into_iter()
                .map(|body| match body {
                    RowBody::Usage(entry) => Some(entry),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(LogPayload::AccessBatch { day: *day, entries })
        }
        SealedShape::Node | SealedShape::Edge => match bodies.into_iter().next() {
            None => Some(empty()),
            Some(RowBody::Effect(RowEffect::Node(node))) if *shape == SealedShape::Node => {
                Some(LogPayload::NodeWrite(node))
            }
            Some(RowBody::Effect(RowEffect::Edge(edge))) if *shape == SealedShape::Edge => {
                Some(LogPayload::EdgeWrite(edge))
            }
            Some(_) => None,
        },
        SealedShape::Episode => {
            let effects = bodies
                .into_iter()
                .map(|body| match body {
                    RowBody::Effect(effect) => Some(effect),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(LogPayload::EpisodeBatch(effects))
        }
        SealedShape::Fact | SealedShape::Artifact | SealedShape::Feedback | SealedShape::Tier => {
            match bodies.into_iter().next() {
                None => Some(empty()),
                Some(RowBody::Whole(payload)) => Some(payload),
                Some(_) => None,
            }
        }
    }
}

/// Opens the sealed records of any reader.
pub struct OpeningReader<R> {
    inner: R,
    registry: Arc<KeyRegistry>,
}

impl<R: LogReader> OpeningReader<R> {
    pub fn new(inner: R, registry: Arc<KeyRegistry>) -> Self {
        Self { inner, registry }
    }

    fn opening(&self, records: LogStream) -> LogStream {
        let registry = Arc::clone(&self.registry);
        Box::pin(records.scan(false, move |failed, item| {
            if *failed {
                return future::ready(None);
            }
            let opened = item.and_then(|record| open_scanned(&registry, record));
            *failed = opened.is_err();
            future::ready(Some(opened))
        }))
    }
}

fn open_scanned(registry: &KeyRegistry, mut record: LogRecord) -> Result<LogRecord, ReaderError> {
    if !matches!(record.event.payload, LogPayload::Sealed(_)) {
        return Ok(record);
    }
    let opened = open_record(registry, &record.event).map_err(|error| match error {
        OpenError::KeyMissing { key_id } => ReaderError::KeyMissing { key_id },
        other => ReaderError::Sealed {
            event_id: record.event.event_id.clone(),
            source: other,
        },
    })?;
    record.row_keys = row_keys(&record.event);
    record.event = opened.event;
    record.shredded = opened.shredded;
    Ok(record)
}

impl<R: LogReader> LogReader for OpeningReader<R> {
    fn scan(&self, from: Option<LogOffset>) -> LogStream {
        self.opening(self.inner.scan(from))
    }

    fn scan_through(&self, from: Option<LogOffset>, through: LogOffset) -> LogStream {
        self.opening(self.inner.scan_through(from, through))
    }
}

#[cfg(test)]
mod tests;
