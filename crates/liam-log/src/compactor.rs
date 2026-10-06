// SPDX-License-Identifier: Apache-2.0

//! Compaction of closed WAL segments into Parquet files.
//!
//! Parquet files are keyed by the WAL segment sequence, so the file for a
//! segment is deterministic and a re-run overwrites it instead of duplicating.
//! A segment is removed only after its Parquet object has been read back and
//! checked, and the directory entry removal has reached disk.

use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeBinaryArray, Float64Array, Int64Array, LargeBinaryArray, RecordBatch,
    StringArray, UInt32Array,
};
use arrow_schema::{ArrowError, DataType, Field, Metadata, Schema, SchemaRef};
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;

use crate::event::{LogEvent, LogPayload, CURRENT_SCHEMA_VERSION};
use crate::wal::{
    scan_segment, segment_paths, segment_sequence, sequence_stem, sync_dir_for, SegmentSink,
    WalError, PARQUET_EXTENSION, SEGMENT_EXTENSION,
};

/// Parquet key-value metadata entry that carries the schema version of the file.
pub const SCHEMA_VERSION_KEY: &str = "schema_version";

// A follow-up stamps the log id here too, so a file from another log is refused.
const WAL_SEGMENT_SEQUENCE_KEY: &str = "wal_segment_sequence";
const EVENT_COUNT_KEY: &str = "event_count";

const HASH_LEN: i32 = 32;

mod column {
    pub(super) const EVENT_ID: &str = "event_id";
    pub(super) const CONTENT_HASH: &str = "content_hash";
    pub(super) const SOURCE: &str = "source";
    pub(super) const TRUST_SCORE: &str = "trust_score";
    pub(super) const OBSERVED_AT: &str = "observed_at";
    pub(super) const INGESTED_AT: &str = "ingested_at";
    pub(super) const ENCRYPTION_KEY_ID: &str = "encryption_key_id";
    pub(super) const SCHEMA_VERSION: &str = "schema_version";
    pub(super) const PAYLOAD: &str = "payload";
}

/// Why a segment could not be compacted, or a Parquet file could not be read.
#[derive(Debug, thiserror::Error)]
pub enum CompactError {
    #[error("arrow conversion failed: {0}")]
    Arrow(#[from] ArrowError),
    #[error("parquet encoding failed: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("object store operation failed: {0}")]
    Store(#[from] object_store::Error),
    #[error("payload could not be decoded: {0}")]
    Payload(postcard::Error),
    #[error("wal segment could not be read: {0}")]
    Wal(#[from] WalError),
    #[error("column {name} is missing or has an unexpected type")]
    Column { name: &'static str },
    #[error("content_hash of row {row} is not {} bytes", HASH_LEN)]
    HashLength { row: usize },
    #[error("event payloads exceed what the 64 bit payload column offsets can address")]
    PayloadTooLarge,
    #[error(
        "parquet schema version {found:?} is not supported, the newest known is {}",
        CURRENT_SCHEMA_VERSION
    )]
    UnsupportedSchemaVersion { found: Option<String> },
    #[error("parquet metadata {key} is {found:?}, expected {expected}")]
    MetadataMismatch {
        key: &'static str,
        expected: u64,
        found: Option<String>,
    },
    #[error("parquet for segment {sequence} holds {found} events, the segment has {expected}")]
    ReadBackMismatch {
        sequence: u64,
        expected: usize,
        found: usize,
    },
    #[error("wal segment {} is closed but has a damaged or truncated tail", segment.display())]
    IncompleteSegment { segment: PathBuf },
    #[error("wal segment {} is the open segment and is not compacted", segment.display())]
    OpenSegment { segment: PathBuf },
    #[error("wal segment {} could not be removed durably: {source}", segment.display())]
    Remove {
        segment: PathBuf,
        source: std::io::Error,
    },
    #[error("{} is not a wal segment name", segment.display())]
    NotASegment { segment: PathBuf },
}

/// Matches the WAL segment stem so a segment and its file pair up by name.
pub(crate) fn parquet_name(sequence: u64) -> String {
    format!("{}.{PARQUET_EXTENSION}", sequence_stem(sequence))
}

pub(crate) fn arrow_schema() -> SchemaRef {
    let fields = vec![
        Field::new(column::EVENT_ID, DataType::Utf8, false),
        Field::new(
            column::CONTENT_HASH,
            DataType::FixedSizeBinary(HASH_LEN),
            false,
        ),
        Field::new(column::SOURCE, DataType::Utf8, false),
        Field::new(column::TRUST_SCORE, DataType::Float64, false),
        Field::new(column::OBSERVED_AT, DataType::Int64, false),
        Field::new(column::INGESTED_AT, DataType::Int64, false),
        Field::new(column::ENCRYPTION_KEY_ID, DataType::Utf8, true),
        Field::new(column::SCHEMA_VERSION, DataType::UInt32, false),
        Field::new(column::PAYLOAD, DataType::LargeBinary, false),
    ];
    let metadata = [(SCHEMA_VERSION_KEY, CURRENT_SCHEMA_VERSION.to_string())];
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

/// Rejects payloads whose summed length the column's `i64` offsets cannot hold,
/// which would otherwise panic while the array is built.
fn ensure_offsets_fit(lengths: impl IntoIterator<Item = usize>) -> Result<(), CompactError> {
    lengths
        .into_iter()
        .try_fold(0_i64, |total, len| {
            total.checked_add(i64::try_from(len).ok()?)
        })
        .map(drop)
        .ok_or(CompactError::PayloadTooLarge)
}

pub(crate) fn events_to_batch(events: &[LogEvent]) -> Result<RecordBatch, CompactError> {
    let payloads = events
        .iter()
        .map(|event| postcard::to_stdvec(&event.payload).map_err(CompactError::Payload))
        .collect::<Result<Vec<_>, _>>()?;
    ensure_offsets_fit(payloads.iter().map(Vec::len))?;
    // `try_from_iter` rejects an empty input, which is a valid (empty) segment.
    let hashes = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
        events.iter().map(|e| Some(e.content_hash)),
        HASH_LEN,
    )?;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            events.iter().map(|e| e.event_id.as_str()),
        )),
        Arc::new(hashes),
        Arc::new(StringArray::from_iter_values(
            events.iter().map(|e| e.source.as_str()),
        )),
        Arc::new(Float64Array::from_iter_values(
            events.iter().map(|e| e.trust_score),
        )),
        Arc::new(Int64Array::from_iter_values(
            events.iter().map(|e| e.observed_at),
        )),
        Arc::new(Int64Array::from_iter_values(
            events.iter().map(|e| e.ingested_at),
        )),
        Arc::new(StringArray::from_iter(
            events.iter().map(|e| e.encryption_key_id.as_deref()),
        )),
        Arc::new(UInt32Array::from_iter_values(
            events.iter().map(|e| e.schema_version),
        )),
        Arc::new(LargeBinaryArray::from_iter_values(payloads)),
    ];
    Ok(RecordBatch::try_new(arrow_schema(), columns)?)
}

/// Rows come back in order; a row stamped with a newer schema is refused.
pub(crate) fn batch_to_events(batch: &RecordBatch) -> Result<Vec<LogEvent>, CompactError> {
    let event_ids = typed_column::<StringArray>(batch, column::EVENT_ID)?;
    let hashes = typed_column::<FixedSizeBinaryArray>(batch, column::CONTENT_HASH)?;
    let sources = typed_column::<StringArray>(batch, column::SOURCE)?;
    let trust_scores = typed_column::<Float64Array>(batch, column::TRUST_SCORE)?;
    let observed_at = typed_column::<Int64Array>(batch, column::OBSERVED_AT)?;
    let ingested_at = typed_column::<Int64Array>(batch, column::INGESTED_AT)?;
    let key_ids = typed_column::<StringArray>(batch, column::ENCRYPTION_KEY_ID)?;
    let versions = typed_column::<UInt32Array>(batch, column::SCHEMA_VERSION)?;
    let payloads = typed_column::<LargeBinaryArray>(batch, column::PAYLOAD)?;

    (0..batch.num_rows())
        .map(|row| {
            let schema_version = versions.value(row);
            if schema_version > CURRENT_SCHEMA_VERSION {
                return Err(CompactError::UnsupportedSchemaVersion {
                    found: Some(schema_version.to_string()),
                });
            }
            let content_hash = hashes
                .value(row)
                .try_into()
                .map_err(|_| CompactError::HashLength { row })?;
            let payload: LogPayload =
                postcard::from_bytes(payloads.value(row)).map_err(CompactError::Payload)?;
            Ok(LogEvent {
                event_id: event_ids.value(row).to_owned(),
                content_hash,
                source: sources.value(row).to_owned(),
                trust_score: trust_scores.value(row),
                observed_at: observed_at.value(row),
                ingested_at: ingested_at.value(row),
                encryption_key_id: key_ids.is_valid(row).then(|| key_ids.value(row).to_owned()),
                schema_version,
                payload,
            })
        })
        .collect()
}

fn typed_column<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    name: &'static str,
) -> Result<&'a T, CompactError> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<T>())
        .ok_or(CompactError::Column { name })
}

fn object_path(sequence: u64) -> StorePath {
    StorePath::from(parquet_name(sequence))
}

fn stamp(sequence: u64, event_count: usize) -> Vec<KeyValue> {
    vec![
        KeyValue::new(WAL_SEGMENT_SEQUENCE_KEY.to_string(), sequence.to_string()),
        KeyValue::new(EVENT_COUNT_KEY.to_string(), event_count.to_string()),
    ]
}

fn encode_parquet(batch: &RecordBatch, metadata: Vec<KeyValue>) -> Result<Vec<u8>, CompactError> {
    let properties = WriterProperties::builder()
        .set_key_value_metadata(Some(metadata))
        .build();
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, batch.schema(), Some(properties))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(buffer)
}

/// Writes the events of one segment as a single Parquet object, replacing any
/// earlier object for the same sequence.
///
/// The store's `put` publishes the whole object at once, so a reader never
/// sees a partial file.
pub(crate) async fn write_parquet(
    store: &dyn ObjectStore,
    sequence: u64,
    events: &[LogEvent],
) -> Result<(), CompactError> {
    let bytes = encode_parquet(&events_to_batch(events)?, stamp(sequence, events.len()))?;
    store
        .put(&object_path(sequence), PutPayload::from(bytes))
        .await?;
    Ok(())
}

/// Reads the object for `sequence`, refusing a file whose schema version or
/// stamped sequence and event count do not match what was decoded.
pub(crate) async fn read_parquet(
    store: &dyn ObjectStore,
    sequence: u64,
) -> Result<Vec<LogEvent>, CompactError> {
    let bytes = store.get(&object_path(sequence)).await?.bytes().await?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
    let metadata = builder.schema().metadata().clone();
    ensure_supported_version(&metadata)?;
    ensure_stamp(&metadata, WAL_SEGMENT_SEQUENCE_KEY, sequence)?;
    let mut events = Vec::new();
    for batch in builder.build()? {
        events.extend(batch_to_events(&batch?)?);
    }
    ensure_stamp(&metadata, EVENT_COUNT_KEY, events.len() as u64)?;
    Ok(events)
}

fn ensure_supported_version(metadata: &Metadata) -> Result<(), CompactError> {
    let found = metadata.get(SCHEMA_VERSION_KEY);
    match found.and_then(|version| version.parse::<u32>().ok()) {
        Some(version) if version <= CURRENT_SCHEMA_VERSION => Ok(()),
        _ => Err(CompactError::UnsupportedSchemaVersion {
            found: found.cloned(),
        }),
    }
}

fn ensure_stamp(metadata: &Metadata, key: &'static str, expected: u64) -> Result<(), CompactError> {
    let found = metadata.get(key);
    if found.and_then(|value| value.parse::<u64>().ok()) == Some(expected) {
        return Ok(());
    }
    Err(CompactError::MetadataMismatch {
        key,
        expected,
        found: found.cloned(),
    })
}

/// What compacting one notified segment did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compaction {
    /// The segment's events were written to Parquet and the segment removed.
    Compacted,
    /// The segment no longer exists, so an earlier run already compacted it.
    AlreadyCompacted,
}

/// Turns closed WAL segments into Parquet objects.
///
/// The store must be rooted at the WAL directory so a segment and its Parquet
/// object sit side by side.
pub struct Compactor {
    store: Arc<dyn ObjectStore>,
}

impl Compactor {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self { store }
    }

    /// A compactor over the WAL directory itself, fsyncing every object it writes.
    pub fn local(dir: &Path) -> Result<Self, CompactError> {
        let store = LocalFileSystem::new_with_prefix(dir)?.with_fsync(true);
        Ok(Self::new(Arc::new(store)))
    }

    /// Compacts one closed segment: Parquet first, then the segment is removed.
    ///
    /// The segment is removed only after the Parquet object is stored and read
    /// back with the same event count, so a crash or failure before that point
    /// leaves the segment in place and a re-run overwrites the object.
    pub async fn compact_segment(&self, segment: &Path) -> Result<Compaction, CompactError> {
        let sequence = segment_sequence(segment, SEGMENT_EXTENSION).ok_or_else(|| {
            CompactError::NotASegment {
                segment: segment.to_path_buf(),
            }
        })?;
        ensure_closed(segment, sequence)?;
        let Some(events) = read_closed_segment(segment)? else {
            return Ok(Compaction::AlreadyCompacted);
        };
        write_parquet(self.store.as_ref(), sequence, &events).await?;
        let stored = read_parquet(self.store.as_ref(), sequence).await?;
        if stored.len() != events.len() {
            return Err(CompactError::ReadBackMismatch {
                sequence,
                expected: events.len(),
                found: stored.len(),
            });
        }
        remove_durably(segment)?;
        Ok(Compaction::Compacted)
    }

    /// Compacts every notified segment until all senders are gone, then stops.
    ///
    /// Each segment runs in its own task, so a failure or panic is logged and
    /// leaves the segment on disk without stopping later notifications.
    pub fn spawn(self, mut notifications: UnboundedReceiver<PathBuf>) -> JoinHandle<()> {
        let compactor = Arc::new(self);
        tokio::spawn(async move {
            while let Some(segment) = notifications.recv().await {
                let compactor = Arc::clone(&compactor);
                let path = segment.clone();
                match tokio::spawn(async move { compactor.compact_segment(&path).await }).await {
                    Ok(Ok(outcome)) => {
                        tracing::debug!(segment = %segment.display(), ?outcome, "segment processed");
                    }
                    Ok(Err(error)) => {
                        tracing::error!(segment = %segment.display(), %error, "segment compaction failed");
                    }
                    Err(error) => {
                        tracing::error!(segment = %segment.display(), %error, "segment compaction task failed");
                    }
                }
            }
        })
    }
}

/// The writer always has a successor open once it closes a segment, so the
/// highest-numbered segment is the one still being appended to.
fn ensure_closed(segment: &Path, sequence: u64) -> Result<(), CompactError> {
    let newest = segment_paths(sync_dir_for(segment))
        .map_err(WalError::Io)?
        .pop();
    match newest {
        Some((newest, _)) if newest == sequence => Err(CompactError::OpenSegment {
            segment: segment.to_path_buf(),
        }),
        _ => Ok(()),
    }
}

/// Reads every event of a closed segment; `None` when the segment is gone.
fn read_closed_segment(segment: &Path) -> Result<Option<Vec<LogEvent>>, CompactError> {
    let bytes = match fs::read(segment) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(WalError::Io(error).into()),
    };
    let scan = scan_segment(segment, &bytes)?;
    // Only an open segment can have a torn tail, so any here is damage.
    if scan.valid_len != bytes.len() as u64 {
        return Err(CompactError::IncompleteSegment {
            segment: segment.to_path_buf(),
        });
    }
    Ok(Some(scan.events))
}

fn remove_durably(segment: &Path) -> Result<(), CompactError> {
    fs::remove_file(segment)
        .and_then(|()| File::open(sync_dir_for(segment))?.sync_all())
        .map_err(|source| CompactError::Remove {
            segment: segment.to_path_buf(),
            source,
        })
}

/// A `SegmentSink` that forwards closed-segment paths to a compactor task.
///
/// The channel is unbounded because `segment_closed` runs inside the writer's
/// lock and must never block an append.
pub struct ChannelSink(UnboundedSender<PathBuf>);

impl ChannelSink {
    pub fn channel() -> (Self, UnboundedReceiver<PathBuf>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (Self(sender), receiver)
    }
}

impl SegmentSink for ChannelSink {
    fn segment_closed(&self, path: &Path) {
        if self.0.send(path.to_path_buf()).is_err() {
            tracing::warn!(segment = %path.display(), "compactor is gone, closed segment not queued");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type, UInt32Type};
    use async_trait::async_trait;
    use futures_core::stream::BoxStream;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutResult,
    };

    use super::*;
    use crate::event::{RowEffect, TombstoneTable, TombstoneTarget};
    use crate::fixtures::{edge_row, event, frame_len, log_event, node_row, sample_hash};
    use crate::wal::{segment_name, SystemClock, WalConfig, WalWriter};
    use crate::LogWriter;

    const ONE_MIB: usize = 1024 * 1024;
    const EVENTS_PER_SEGMENT: usize = 2;
    const TASK_TIMEOUT: Duration = Duration::from_secs(10);

    fn local_store(dir: &tempfile::TempDir) -> LocalFileSystem {
        LocalFileSystem::new_with_prefix(dir.path()).expect("local store over temp dir")
    }

    /// Compares field by field so a NaN trust score matches by bit pattern.
    fn assert_same_events(actual: &[LogEvent], expected: &[LogEvent]) {
        assert_eq!(actual.len(), expected.len(), "event count");
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                got.trust_score.to_bits(),
                want.trust_score.to_bits(),
                "trust_score bits of event {index}"
            );
            let (mut got, mut want) = (got.clone(), want.clone());
            got.trust_score = 0.0;
            want.trust_score = 0.0;
            assert_eq!(got, want, "event {index}");
        }
    }

    fn every_payload_variant() -> Vec<LogEvent> {
        let payloads = vec![
            LogPayload::NodeWrite(node_row("node-1", "content")),
            LogPayload::EdgeWrite(edge_row("edge-1", "node-2")),
            LogPayload::EpisodeBatch(vec![
                RowEffect::Node(node_row("node-2", "other")),
                RowEffect::Edge(edge_row("edge-2", "node-3")),
            ]),
            LogPayload::Tombstone(vec![TombstoneTarget {
                table: TombstoneTable::NodeCommunity,
                id: "node-4".into(),
            }]),
            LogPayload::DuplicateOf {
                first_event_id: "event-0".into(),
            },
            LogPayload::Voided {
                target_event_id: "event-0".into(),
            },
        ];
        payloads
            .into_iter()
            .enumerate()
            .map(|(index, payload)| log_event(&format!("variant-{index}"), payload))
            .collect()
    }

    /// Every field differs from every other, so a column wired to the wrong
    /// field cannot match by coincidence. The trust score is a non-canonical NaN.
    fn distinct_event() -> LogEvent {
        LogEvent {
            event_id: "id-column".into(),
            content_hash: sample_hash(1),
            source: "source-column".into(),
            trust_score: f64::from_bits(0xfff8_0000_0000_0001),
            observed_at: 1_111,
            ingested_at: 2_222,
            encryption_key_id: Some("key-column".into()),
            schema_version: u32::MAX,
            payload: LogPayload::NodeWrite(node_row("node-distinct", "distinct")),
        }
    }

    fn with_key_id(key_id: Option<&str>) -> LogEvent {
        let mut event = event(1);
        event.encryption_key_id = key_id.map(String::from);
        event
    }

    fn with_timestamps(observed_at: i64, ingested_at: i64) -> LogEvent {
        let mut event = event(1);
        event.observed_at = observed_at;
        event.ingested_at = ingested_at;
        event
    }

    fn with_trust_score(trust_score: f64) -> LogEvent {
        let mut event = event(1);
        event.trust_score = trust_score;
        event
    }

    fn with_large_payload() -> LogEvent {
        let mut row = node_row("node-big", "");
        row.content = "x".repeat(ONE_MIB);
        log_event("large", LogPayload::NodeWrite(row))
    }

    async fn put_bytes(store: &LocalFileSystem, sequence: u64, bytes: Vec<u8>) {
        store
            .put(&object_path(sequence), PutPayload::from(bytes))
            .await
            .expect("put parquet bytes");
    }

    #[test]
    fn parquet_name_is_keyed_by_the_segment_sequence() {
        // Arrange
        let cases = [
            (0, "00000000000000000000.parquet"),
            (7, "00000000000000000007.parquet"),
            (u64::MAX, "18446744073709551615.parquet"),
        ];

        for (sequence, expected) in cases {
            // Act
            let name = parquet_name(sequence);

            // Assert
            assert_eq!(name, expected, "sequence {sequence}");
        }
    }

    #[test]
    fn schema_has_a_column_per_header_field_and_a_large_binary_payload() {
        // Arrange
        let expected = [
            (column::EVENT_ID, DataType::Utf8, false),
            (column::CONTENT_HASH, DataType::FixedSizeBinary(32), false),
            (column::SOURCE, DataType::Utf8, false),
            (column::TRUST_SCORE, DataType::Float64, false),
            (column::OBSERVED_AT, DataType::Int64, false),
            (column::INGESTED_AT, DataType::Int64, false),
            (column::ENCRYPTION_KEY_ID, DataType::Utf8, true),
            (column::SCHEMA_VERSION, DataType::UInt32, false),
            (column::PAYLOAD, DataType::LargeBinary, false),
        ];

        // Act
        let schema = arrow_schema();

        // Assert
        let actual: Vec<_> = schema
            .fields()
            .iter()
            .map(|field| {
                (
                    field.name().as_str(),
                    field.data_type().clone(),
                    field.is_nullable(),
                )
            })
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(
            schema.metadata().get(SCHEMA_VERSION_KEY),
            Some(&CURRENT_SCHEMA_VERSION.to_string()),
            "schema metadata"
        );
    }

    #[test]
    fn each_column_holds_its_own_event_field() {
        // Arrange
        let event = distinct_event();

        // Act
        let batch = events_to_batch(std::slice::from_ref(&event)).expect("convert to a batch");

        // Assert
        let col = |name: &str| batch.column_by_name(name).expect("column").clone();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.schema(), arrow_schema());
        let string = |name: &str| col(name).as_string::<i32>().value(0).to_owned();
        assert_eq!(string(column::EVENT_ID), event.event_id);
        assert_eq!(string(column::SOURCE), event.source);
        assert_eq!(
            Some(string(column::ENCRYPTION_KEY_ID)),
            event.encryption_key_id
        );
        let hash = col(column::CONTENT_HASH);
        assert_eq!(hash.as_fixed_size_binary().value(0), event.content_hash);
        let int = |name: &str| col(name).as_primitive::<Int64Type>().value(0);
        assert_eq!(int(column::OBSERVED_AT), event.observed_at);
        assert_eq!(int(column::INGESTED_AT), event.ingested_at);
        let trust = col(column::TRUST_SCORE);
        let trust = trust.as_primitive::<Float64Type>().value(0);
        assert_eq!(trust.to_bits(), event.trust_score.to_bits());
        let version = col(column::SCHEMA_VERSION);
        assert_eq!(
            version.as_primitive::<UInt32Type>().value(0),
            event.schema_version
        );
        let payload = postcard::to_stdvec(&event.payload).expect("encode payload");
        assert_eq!(col(column::PAYLOAD).as_binary::<i64>().value(0), payload);
    }

    #[test]
    fn payload_lengths_that_overflow_the_column_offsets_are_a_typed_error() {
        // Arrange
        let overflowing = [vec![i64::MAX as usize, 1], vec![usize::MAX]];

        // Act
        let accepted = ensure_offsets_fit([1, 2, 3]);

        // Assert
        assert!(accepted.is_ok());
        for lengths in overflowing {
            let result = ensure_offsets_fit(lengths.clone());
            assert!(
                matches!(result, Err(CompactError::PayloadTooLarge)),
                "{lengths:?}: {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn events_round_trip_through_parquet_exactly() {
        // Arrange
        let mut distinct = distinct_event();
        distinct.schema_version = CURRENT_SCHEMA_VERSION;
        let cases: Vec<(&str, Vec<LogEvent>)> = vec![
            ("empty list", Vec::new()),
            ("several in order", (0..5).map(event).collect()),
            ("every field distinct", vec![distinct]),
            ("key id none", vec![with_key_id(None)]),
            ("key id some", vec![with_key_id(Some("key-7"))]),
            ("key id empty", vec![with_key_id(Some(""))]),
            (
                "timestamp minimum",
                vec![with_timestamps(i64::MIN, i64::MIN)],
            ),
            (
                "timestamp maximum",
                vec![with_timestamps(i64::MAX, i64::MAX)],
            ),
            ("nan trust score", vec![with_trust_score(f64::NAN)]),
            ("negative zero trust", vec![with_trust_score(-0.0)]),
            (
                "infinite trust score",
                vec![with_trust_score(f64::INFINITY)],
            ),
            ("one mebibyte payload", vec![with_large_payload()]),
            ("every payload variant", every_payload_variant()),
        ];

        for (name, events) in cases {
            let dir = tempfile::tempdir().expect("temp dir");
            let store = local_store(&dir);

            // Act
            write_parquet(&store, 5, &events)
                .await
                .unwrap_or_else(|error| panic!("{name}: write failed: {error}"));
            let read = read_parquet(&store, 5)
                .await
                .unwrap_or_else(|error| panic!("{name}: read failed: {error}"));

            // Assert
            assert_same_events(&read, &events);
        }
    }

    #[tokio::test]
    async fn parquet_file_carries_its_version_sequence_and_event_count_as_metadata() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let store = local_store(&dir);
        let events: Vec<LogEvent> = (0..3).map(event).collect();
        write_parquet(&store, 2, &events).await.expect("write");
        let bytes = store
            .get(&object_path(2))
            .await
            .expect("get file")
            .bytes()
            .await
            .expect("file bytes");

        // Act
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).expect("open parquet");

        // Assert
        let metadata = builder.schema().metadata();
        let expected = [
            (SCHEMA_VERSION_KEY, CURRENT_SCHEMA_VERSION.to_string()),
            (WAL_SEGMENT_SEQUENCE_KEY, "2".to_string()),
            (EVENT_COUNT_KEY, "3".to_string()),
        ];
        for (key, value) in expected {
            assert_eq!(metadata.get(key), Some(&value), "metadata {key}");
        }
    }

    /// A file of `events` with the given schema version (none leaves it out)
    /// and stamps.
    fn forged_parquet(
        events: &[LogEvent],
        version: Option<String>,
        sequence: u64,
        count: usize,
    ) -> Vec<u8> {
        let batch = events_to_batch(events).expect("batch");
        let metadata: Metadata = version
            .into_iter()
            .map(|version| (SCHEMA_VERSION_KEY, version))
            .collect();
        let schema = Arc::new(Schema::new_with_metadata(
            batch.schema().fields().clone(),
            metadata,
        ));
        let batch = RecordBatch::try_new(schema, batch.columns().to_vec()).expect("re-schema");
        encode_parquet(&batch, stamp(sequence, count)).expect("encode")
    }

    #[tokio::test]
    async fn reading_refuses_files_and_rows_from_a_newer_or_unknown_schema() {
        // Arrange
        let current = Some(CURRENT_SCHEMA_VERSION.to_string());
        let newer = Some((CURRENT_SCHEMA_VERSION + 1).to_string());
        let cases = [
            ("newer file version", vec![event(0)], newer),
            (
                "unparseable file version",
                vec![event(0)],
                Some("one".into()),
            ),
            ("missing file version", vec![event(0)], None),
            ("newer row version", vec![distinct_event()], current),
        ];

        for (name, events, version) in cases {
            let dir = tempfile::tempdir().expect("temp dir");
            let store = local_store(&dir);
            put_bytes(&store, 5, forged_parquet(&events, version, 5, 1)).await;

            // Act
            let result = read_parquet(&store, 5).await;

            // Assert
            assert!(
                matches!(result, Err(CompactError::UnsupportedSchemaVersion { .. })),
                "{name}: {result:?}"
            );
        }
    }

    #[tokio::test]
    async fn reading_refuses_stamps_that_disagree_with_the_name_or_the_events() {
        // Arrange
        let current = Some(CURRENT_SCHEMA_VERSION.to_string());
        let cases = [
            (
                WAL_SEGMENT_SEQUENCE_KEY,
                forged_parquet(&[event(0)], current.clone(), 4, 1),
            ),
            (EVENT_COUNT_KEY, forged_parquet(&[event(0)], current, 5, 2)),
        ];

        for (key, bytes) in cases {
            let dir = tempfile::tempdir().expect("temp dir");
            let store = local_store(&dir);
            put_bytes(&store, 5, bytes).await;

            // Act
            let result = read_parquet(&store, 5).await;

            // Assert
            assert!(
                matches!(result, Err(CompactError::MetadataMismatch { key: found, .. }) if found == key),
                "{key}: {result:?}"
            );
        }
    }

    // Compactor task

    /// A local store whose `put` misbehaves as configured; every other call is
    /// delegated.
    #[derive(Debug)]
    struct FaultyStore {
        inner: LocalFileSystem,
        fault: PutFault,
    }

    #[derive(Debug)]
    enum PutFault {
        Fail,
        PanicOnce(AtomicBool),
        /// Stores these bytes instead of what was put.
        Substitute(PutPayload),
    }

    impl fmt::Display for FaultyStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "FaultyStore")
        }
    }

    #[async_trait]
    impl ObjectStore for FaultyStore {
        async fn put_opts(
            &self,
            location: &StorePath,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            let payload = match &self.fault {
                PutFault::Fail => {
                    return Err(object_store::Error::Generic {
                        store: "FaultyStore",
                        source: "injected put failure".into(),
                    })
                }
                PutFault::PanicOnce(armed) if armed.swap(false, Ordering::SeqCst) => {
                    panic!("injected put panic")
                }
                PutFault::PanicOnce(_) => payload,
                PutFault::Substitute(other) => other.clone(),
            };
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &StorePath,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &StorePath,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<StorePath>>,
        ) -> BoxStream<'static, object_store::Result<StorePath>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&StorePath>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&StorePath>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &StorePath,
            to: &StorePath,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    fn faulty(dir: &tempfile::TempDir, fault: PutFault) -> Compactor {
        let inner = local_store(dir);
        Compactor::new(Arc::new(FaultyStore { inner, fault }))
    }

    struct ClosedSegment {
        sequence: u64,
        path: PathBuf,
        events: Vec<LogEvent>,
    }

    struct Log {
        dir: tempfile::TempDir,
        closed: Vec<ClosedSegment>,
        open: PathBuf,
    }

    fn open_writer(dir: &Path) -> WalWriter<SystemClock> {
        let config = WalConfig {
            segment_max_bytes: EVENTS_PER_SEGMENT as u64 * frame_len(&event(0)),
            rotate_interval_secs: u64::MAX,
        };
        WalWriter::open_with_system_clock(dir, config).expect("open wal writer")
    }

    fn append_events(writer: &mut impl LogWriter, range: Range<usize>) {
        for index in range {
            writer.append(&event(index)).expect("append event");
        }
    }

    /// Segments `0..closed_segments` are closed with two events each, and the
    /// next segment is open with one event.
    fn write_log(closed_segments: u64) -> Log {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut writer = open_writer(dir.path());
        let total = closed_segments as usize * EVENTS_PER_SEGMENT + 1;
        append_events(&mut writer, 0..total);
        let closed = (0..closed_segments)
            .map(|sequence| {
                let first = sequence as usize * EVENTS_PER_SEGMENT;
                ClosedSegment {
                    sequence,
                    path: dir.path().join(segment_name(sequence)),
                    events: (first..first + EVENTS_PER_SEGMENT).map(event).collect(),
                }
            })
            .collect();
        let open = dir.path().join(segment_name(closed_segments));
        Log { dir, closed, open }
    }

    fn file_names(dir: &Path, extension: &str) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("list dir")
            .map(|entry| entry.expect("dir entry").file_name().into_string().unwrap())
            .filter(|name| name.ends_with(extension))
            .collect();
        names.sort();
        names
    }

    fn assert_no_parquet(dir: &Path) {
        assert_eq!(file_names(dir, ".parquet"), Vec::<String>::new());
    }

    fn expected_parquet_names(segments: &[&ClosedSegment]) -> Vec<String> {
        segments
            .iter()
            .map(|segment| parquet_name(segment.sequence))
            .collect()
    }

    async fn assert_parquet_holds(store: &LocalFileSystem, segment: &ClosedSegment) {
        let read = read_parquet(store, segment.sequence)
            .await
            .expect("read parquet");
        assert_same_events(&read, &segment.events);
    }

    fn assert_bytes(path: &Path, expected: &[u8], what: &str) {
        let actual = std::fs::read(path).unwrap_or_else(|_| panic!("{what} is still present"));
        assert_eq!(actual, expected, "{what} bytes");
    }

    /// Queues the segments, closes the channel and waits for the task to drain
    /// it and stop, so the files are final when this returns.
    async fn compact_all(compactor: Compactor, segments: &[&Path]) {
        let (sink, receiver) = ChannelSink::channel();
        for segment in segments {
            sink.segment_closed(segment);
        }
        drop(sink);
        let handle = compactor.spawn(receiver);
        tokio::time::timeout(TASK_TIMEOUT, handle)
            .await
            .expect("compactor task stops once the channel is closed")
            .expect("compactor task does not panic");
    }

    fn local_compactor(dir: &Path) -> Compactor {
        Compactor::local(dir).expect("compactor")
    }

    #[tokio::test]
    async fn notified_segment_becomes_a_parquet_file_and_the_wal_segment_is_removed() {
        // Arrange
        let log = write_log(1);
        let segment = &log.closed[0];

        // Act
        compact_all(local_compactor(log.dir.path()), &[&segment.path]).await;

        // Assert
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            expected_parquet_names(&[segment])
        );
        assert!(!segment.path.exists(), "compacted wal segment is removed");
        assert_parquet_holds(&local_store(&log.dir), segment).await;
    }

    #[tokio::test]
    async fn only_the_notified_segment_is_compacted() {
        // Arrange
        let log = write_log(2);
        let (untouched, notified) = (&log.closed[0], &log.closed[1]);
        let untouched_bytes = std::fs::read(&untouched.path).expect("read segment");
        let open_bytes = std::fs::read(&log.open).expect("read open segment");

        // Act
        compact_all(local_compactor(log.dir.path()), &[&notified.path]).await;

        // Assert
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            expected_parquet_names(&[notified])
        );
        assert!(!notified.path.exists(), "notified segment is removed");
        assert_parquet_holds(&local_store(&log.dir), notified).await;
        assert_bytes(
            &untouched.path,
            &untouched_bytes,
            "segment that was not notified",
        );
        assert_bytes(&log.open, &open_bytes, "open segment");
    }

    #[tokio::test]
    async fn segment_that_no_longer_exists_is_a_no_op() {
        // Arrange
        let log = write_log(1);
        let compactor = local_compactor(log.dir.path());
        let gone = log.dir.path().join(segment_name(7));
        let before = file_names(log.dir.path(), "");

        // Act
        let outcome = compactor.compact_segment(&gone).await;

        // Assert
        assert!(
            matches!(outcome, Ok(Compaction::AlreadyCompacted)),
            "outcome {outcome:?}"
        );
        assert_eq!(
            file_names(log.dir.path(), ""),
            before,
            "no empty parquet is created"
        );
    }

    #[tokio::test]
    async fn task_keeps_compacting_after_a_notification_that_fails() {
        // Arrange
        let log = write_log(1);
        let closed = &log.closed[0];
        let notes = log.dir.path().join("notes.txt");
        let compactor = local_compactor(log.dir.path());
        let failure = compactor.compact_segment(&notes).await;
        assert!(
            matches!(failure, Err(CompactError::NotASegment { .. })),
            "the notification must fail for this test to mean anything: {failure:?}"
        );

        // Act
        compact_all(compactor, &[&notes, &closed.path]).await;

        // Assert
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            expected_parquet_names(&[closed])
        );
        assert!(!closed.path.exists(), "segment after the failed one");
    }

    #[tokio::test]
    async fn task_keeps_compacting_after_a_compaction_panics() {
        // Arrange
        let log = write_log(2);
        let (panicking, healthy) = (&log.closed[0], &log.closed[1]);
        let panicking_bytes = std::fs::read(&panicking.path).expect("read segment");
        let compactor = faulty(&log.dir, PutFault::PanicOnce(AtomicBool::new(true)));

        // Act
        compact_all(compactor, &[&panicking.path, &healthy.path]).await;

        // Assert
        assert_bytes(
            &panicking.path,
            &panicking_bytes,
            "segment whose compaction panicked",
        );
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            expected_parquet_names(&[healthy])
        );
        assert!(!healthy.path.exists(), "segment after the panic");
    }

    #[tokio::test]
    async fn compacting_again_after_the_parquet_was_written_leaves_one_parquet() {
        // Arrange
        let log = write_log(1);
        let segment = &log.closed[0];
        let store = local_store(&log.dir);
        let stale: Vec<LogEvent> = (50..52).map(event).collect();
        write_parquet(&store, segment.sequence, &stale)
            .await
            .expect("earlier run wrote the parquet");

        // Act
        let outcome = local_compactor(log.dir.path())
            .compact_segment(&segment.path)
            .await;

        // Assert
        assert!(
            matches!(outcome, Ok(Compaction::Compacted)),
            "outcome {outcome:?}"
        );
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            expected_parquet_names(&[segment])
        );
        assert!(!segment.path.exists(), "wal segment is removed");
        assert_parquet_holds(&store, segment).await;
    }

    #[tokio::test]
    async fn task_drains_pending_notifications_and_stops_when_the_channel_closes() {
        // Arrange
        let log = write_log(3);
        let paths: Vec<&Path> = log.closed.iter().map(|s| s.path.as_path()).collect();
        let store = local_store(&log.dir);

        // Act
        compact_all(local_compactor(log.dir.path()), &paths).await;

        // Assert
        let segments: Vec<&ClosedSegment> = log.closed.iter().collect();
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            expected_parquet_names(&segments)
        );
        for segment in segments {
            assert!(!segment.path.exists(), "segment {}", segment.sequence);
            assert_parquet_holds(&store, segment).await;
        }
    }

    #[tokio::test]
    async fn rotation_by_the_writer_reaches_the_compactor_through_the_channel_sink() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let (sink, receiver) = ChannelSink::channel();
        let mut writer = open_writer(dir.path()).with_sink(Box::new(sink));
        let handle = local_compactor(dir.path()).spawn(receiver);
        let store = local_store(&dir);

        // Act
        append_events(&mut writer, 0..EVENTS_PER_SEGMENT + 1);
        drop(writer);
        tokio::time::timeout(TASK_TIMEOUT, handle)
            .await
            .expect("compactor task stops when the writer is dropped")
            .expect("compactor task does not panic");

        // Assert
        assert_eq!(file_names(dir.path(), ".parquet"), vec![parquet_name(0)]);
        assert_eq!(file_names(dir.path(), ".wal"), vec![segment_name(1)]);
        let read = read_parquet(&store, 0).await.expect("read parquet");
        assert_same_events(
            &read,
            &(0..EVENTS_PER_SEGMENT).map(event).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn appends_stay_ok_and_rotate_when_the_compactor_is_gone() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let (sink, receiver) = ChannelSink::channel();
        drop(receiver);
        let mut writer = open_writer(dir.path()).with_sink(Box::new(sink));

        // Act
        append_events(&mut writer, 0..EVENTS_PER_SEGMENT + 1);

        // Assert
        assert_eq!(
            file_names(dir.path(), ".wal"),
            vec![segment_name(0), segment_name(1)]
        );
    }

    #[tokio::test]
    async fn the_open_segment_is_refused_and_left_untouched() {
        // Arrange
        let log = write_log(1);
        let open_bytes = std::fs::read(&log.open).expect("read open segment");

        // Act
        let outcome = local_compactor(log.dir.path())
            .compact_segment(&log.open)
            .await;

        // Assert
        assert!(
            matches!(outcome, Err(CompactError::OpenSegment { .. })),
            "outcome {outcome:?}"
        );
        assert_bytes(&log.open, &open_bytes, "open segment");
        assert_no_parquet(log.dir.path());
    }

    #[tokio::test]
    async fn a_failed_put_keeps_the_wal_segment_byte_for_byte() {
        // Arrange
        let log = write_log(1);
        let segment = &log.closed[0];
        let before = std::fs::read(&segment.path).expect("read segment");
        let compactor = faulty(&log.dir, PutFault::Fail);

        // Act
        let outcome = compactor.compact_segment(&segment.path).await;

        // Assert
        assert!(
            matches!(outcome, Err(CompactError::Store(_))),
            "outcome {outcome:?}"
        );
        assert_bytes(&segment.path, &before, "wal segment");
        assert_no_parquet(log.dir.path());
    }

    #[tokio::test]
    async fn an_object_that_reads_back_with_other_events_keeps_the_wal_segment() {
        // Arrange
        let log = write_log(1);
        let segment = &log.closed[0];
        let before = std::fs::read(&segment.path).expect("read segment");
        let other = events_to_batch(&[event(99)]).expect("batch");
        let other = encode_parquet(&other, stamp(segment.sequence, 1)).expect("encode");
        let compactor = faulty(&log.dir, PutFault::Substitute(PutPayload::from(other)));

        // Act
        let outcome = compactor.compact_segment(&segment.path).await;

        // Assert
        assert!(
            matches!(
                outcome,
                Err(CompactError::ReadBackMismatch {
                    expected: 2,
                    found: 1,
                    ..
                })
            ),
            "outcome {outcome:?}"
        );
        assert_bytes(&segment.path, &before, "wal segment");
    }

    #[tokio::test]
    async fn a_closed_segment_with_a_damaged_tail_is_refused_and_kept() {
        // Arrange
        let log = write_log(1);
        let segment = &log.closed[0];
        let mut bytes = std::fs::read(&segment.path).expect("read segment");
        *bytes.last_mut().expect("segment is not empty") ^= 0xFF;
        std::fs::write(&segment.path, &bytes).expect("damage the last record");

        // Act
        let outcome = local_compactor(log.dir.path())
            .compact_segment(&segment.path)
            .await;

        // Assert
        assert!(
            matches!(outcome, Err(CompactError::IncompleteSegment { .. })),
            "outcome {outcome:?}"
        );
        assert_bytes(&segment.path, &bytes, "damaged segment");
        assert_no_parquet(log.dir.path());
    }
}
