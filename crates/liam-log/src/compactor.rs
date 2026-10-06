// SPDX-License-Identifier: Apache-2.0

//! Compaction of closed WAL segments into Parquet files.
//!
//! Parquet files are keyed by the WAL segment sequence, so the file for a
//! segment is deterministic and a re-run overwrites it instead of duplicating.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, FixedSizeBinaryArray, Float64Array, Int64Array, RecordBatch,
    StringArray, UInt32Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;

use crate::event::{LogEvent, LogPayload, CURRENT_SCHEMA_VERSION};
use crate::wal::{read_segment, segment_sequence, SegmentSink, WalError};

/// Parquet key-value metadata entry that carries the schema version of the file.
pub const SCHEMA_VERSION_KEY: &str = "schema_version";

const HASH_LEN: i32 = 32;

/// Why a set of events could not be written to or read from Parquet.
#[derive(Debug, thiserror::Error)]
pub enum CompactError {
    #[error("arrow conversion failed: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("parquet encoding failed: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("object store operation failed: {0}")]
    Store(#[from] object_store::Error),
    #[error("payload could not be decoded: {0}")]
    Payload(postcard::Error),
    #[error("wal segment could not be read: {0}")]
    Wal(#[from] WalError),
    #[error("wal segment {} could not be removed: {source}", segment.display())]
    Remove {
        segment: PathBuf,
        source: std::io::Error,
    },
    #[error("{} is not a wal segment name", segment.display())]
    NotASegment { segment: PathBuf },
}

/// Name of the Parquet file for the WAL segment with this sequence.
///
/// Matches the WAL segment stem so a segment and its file pair up by name.
pub fn parquet_name(sequence: u64) -> String {
    format!("{sequence:020}.parquet")
}

/// The Arrow schema every compacted file uses.
pub fn arrow_schema() -> SchemaRef {
    let fields = vec![
        Field::new("event_id", DataType::Utf8, false),
        Field::new("content_hash", DataType::FixedSizeBinary(HASH_LEN), false),
        Field::new("source", DataType::Utf8, false),
        Field::new("trust_score", DataType::Float64, false),
        Field::new("observed_at", DataType::Int64, false),
        Field::new("ingested_at", DataType::Int64, false),
        Field::new("encryption_key_id", DataType::Utf8, true),
        Field::new("schema_version", DataType::UInt32, false),
        Field::new("payload", DataType::Binary, false),
    ];
    let metadata = HashMap::from([(
        SCHEMA_VERSION_KEY.to_string(),
        CURRENT_SCHEMA_VERSION.to_string(),
    )]);
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

/// Converts events to one record batch, one row per event, in order.
pub fn events_to_batch(events: &[LogEvent]) -> Result<RecordBatch, CompactError> {
    let payloads = events
        .iter()
        .map(|event| postcard::to_stdvec(&event.payload).map_err(CompactError::Payload))
        .collect::<Result<Vec<_>, _>>()?;
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
        Arc::new(BinaryArray::from_iter_values(payloads)),
    ];
    Ok(RecordBatch::try_new(arrow_schema(), columns)?)
}

/// Converts a record batch back to events, in row order.
pub fn batch_to_events(batch: &RecordBatch) -> Result<Vec<LogEvent>, CompactError> {
    let event_ids = column::<StringArray>(batch, "event_id")?;
    let hashes = column::<FixedSizeBinaryArray>(batch, "content_hash")?;
    let sources = column::<StringArray>(batch, "source")?;
    let trust_scores = column::<Float64Array>(batch, "trust_score")?;
    let observed_at = column::<Int64Array>(batch, "observed_at")?;
    let ingested_at = column::<Int64Array>(batch, "ingested_at")?;
    let key_ids = column::<StringArray>(batch, "encryption_key_id")?;
    let versions = column::<UInt32Array>(batch, "schema_version")?;
    let payloads = column::<BinaryArray>(batch, "payload")?;

    (0..batch.num_rows())
        .map(|row| {
            let content_hash = hashes.value(row).try_into().map_err(|_| {
                arrow::error::ArrowError::InvalidArgumentError(format!(
                    "content_hash of row {row} is not {HASH_LEN} bytes"
                ))
            })?;
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
                schema_version: versions.value(row),
                payload,
            })
        })
        .collect()
}

fn column<'a, T: Array + 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a T, CompactError> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref::<T>())
        .ok_or_else(|| {
            arrow::error::ArrowError::SchemaError(format!(
                "column {name} is missing or has an unexpected type"
            ))
            .into()
        })
}

fn object_path(sequence: u64) -> StorePath {
    StorePath::from(parquet_name(sequence))
}

/// Writes the events of one segment as a single Parquet object, replacing any
/// earlier object for the same sequence.
///
/// The store's `put` publishes the whole object at once, so a reader never
/// sees a partial file.
pub async fn write_parquet(
    store: &dyn ObjectStore,
    sequence: u64,
    events: &[LogEvent],
) -> Result<(), CompactError> {
    let batch = events_to_batch(events)?;
    let mut buffer = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buffer, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    store
        .put(&object_path(sequence), PutPayload::from(buffer))
        .await?;
    Ok(())
}

/// Reads back every event of the Parquet object for this segment sequence.
pub async fn read_parquet(
    store: &dyn ObjectStore,
    sequence: u64,
) -> Result<Vec<LogEvent>, CompactError> {
    let bytes = store.get(&object_path(sequence)).await?.bytes().await?;
    let mut events = Vec::new();
    for batch in ParquetRecordBatchReaderBuilder::try_new(bytes)?.build()? {
        events.extend(batch_to_events(&batch?)?);
    }
    Ok(events)
}

/// What compacting one notified segment did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compaction {
    /// The segment's events were written to Parquet and the segment removed.
    Compacted,
    /// The segment no longer exists, so an earlier run already compacted it.
    AlreadyCompacted,
}

/// Turns closed WAL segments into Parquet files in the same directory.
pub struct Compactor {
    store: Arc<dyn ObjectStore>,
}

impl Compactor {
    pub fn new(dir: &Path) -> Result<Self, CompactError> {
        Ok(Self {
            store: Arc::new(LocalFileSystem::new_with_prefix(dir)?),
        })
    }

    /// Compacts one closed segment: Parquet first, then the segment is removed.
    ///
    /// The segment is removed only after the Parquet object is stored, so a
    /// crash in between leaves both files and a re-run overwrites the object.
    pub async fn compact_segment(&self, segment: &Path) -> Result<Compaction, CompactError> {
        let sequence = segment_sequence(segment).ok_or_else(|| CompactError::NotASegment {
            segment: segment.to_path_buf(),
        })?;
        let events = match read_segment(segment) {
            Ok(events) => events,
            Err(WalError::Io(error)) if error.kind() == ErrorKind::NotFound => {
                return Ok(Compaction::AlreadyCompacted)
            }
            Err(error) => return Err(error.into()),
        };
        write_parquet(self.store.as_ref(), sequence, &events).await?;
        std::fs::remove_file(segment).map_err(|source| CompactError::Remove {
            segment: segment.to_path_buf(),
            source,
        })?;
        Ok(Compaction::Compacted)
    }

    /// Compacts every notified segment until all senders are gone, then stops.
    ///
    /// A failed segment is logged and left on disk, so later notifications
    /// still run.
    pub fn spawn(self, mut notifications: UnboundedReceiver<PathBuf>) -> JoinHandle<()> {
        tokio::spawn(async move {
            while let Some(segment) = notifications.recv().await {
                match self.compact_segment(&segment).await {
                    Ok(outcome) => {
                        tracing::debug!(segment = %segment.display(), ?outcome, "segment processed");
                    }
                    Err(error) => {
                        tracing::error!(segment = %segment.display(), %error, "segment compaction failed");
                    }
                }
            }
        })
    }
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
    use arrow::datatypes::DataType;
    use object_store::local::LocalFileSystem;
    use object_store::path::Path as StorePath;
    use object_store::ObjectStoreExt;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::*;
    use crate::event::{LogPayload, RowEffect, TombstoneTable, TombstoneTarget};
    use crate::fixtures::{edge_row, event, log_event, node_row};

    const ONE_MIB: usize = 1024 * 1024;

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
    fn parquet_name_is_deterministic_and_distinct_per_sequence() {
        // Arrange
        let (first, second) = (3, 4);

        // Act
        let (repeat_a, repeat_b) = (parquet_name(first), parquet_name(first));
        let other = parquet_name(second);

        // Assert
        assert_eq!(repeat_a, repeat_b);
        assert_ne!(repeat_a, other);
    }

    #[test]
    fn schema_has_a_column_per_header_field_and_a_binary_payload() {
        // Arrange
        let expected = [
            ("event_id", DataType::Utf8, false),
            ("content_hash", DataType::FixedSizeBinary(32), false),
            ("source", DataType::Utf8, false),
            ("trust_score", DataType::Float64, false),
            ("observed_at", DataType::Int64, false),
            ("ingested_at", DataType::Int64, false),
            ("encryption_key_id", DataType::Utf8, true),
            ("schema_version", DataType::UInt32, false),
            ("payload", DataType::Binary, false),
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
    }

    #[test]
    fn schema_carries_the_schema_version_as_metadata() {
        // Arrange
        let expected = crate::event::CURRENT_SCHEMA_VERSION.to_string();

        // Act
        let schema = arrow_schema();

        // Assert
        assert_eq!(
            schema.metadata().get(SCHEMA_VERSION_KEY),
            Some(&expected),
            "schema metadata"
        );
    }

    #[test]
    fn batch_has_one_row_per_event() {
        // Arrange
        let events: Vec<LogEvent> = (0..3).map(event).collect();

        // Act
        let batch = events_to_batch(&events).expect("convert events to a batch");

        // Assert
        assert_eq!(batch.num_rows(), events.len());
        assert_eq!(batch.schema(), arrow_schema());
    }

    #[tokio::test]
    async fn events_round_trip_through_parquet_exactly() {
        // Arrange
        let cases: Vec<(&str, Vec<LogEvent>)> = vec![
            ("empty list", Vec::new()),
            ("several in order", (0..5).map(event).collect()),
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
    async fn written_file_is_named_after_the_segment_sequence() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let store = local_store(&dir);
        let events: Vec<LogEvent> = (0..2).map(event).collect();

        // Act
        write_parquet(&store, 12, &events).await.expect("write");

        // Assert
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .expect("list dir")
            .map(|entry| entry.expect("dir entry").file_name().into_string().unwrap())
            .collect();
        assert_eq!(names, vec![parquet_name(12)]);
    }

    #[tokio::test]
    async fn writing_the_same_sequence_again_overwrites_instead_of_duplicating() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let store = local_store(&dir);
        let first: Vec<LogEvent> = (0..3).map(event).collect();
        let second: Vec<LogEvent> = (10..12).map(event).collect();
        write_parquet(&store, 9, &first).await.expect("first write");

        // Act
        write_parquet(&store, 9, &second)
            .await
            .expect("second write");

        // Assert
        let read = read_parquet(&store, 9).await.expect("read back");
        assert_same_events(&read, &second);
        let file_count = std::fs::read_dir(dir.path()).expect("list dir").count();
        assert_eq!(file_count, 1, "one file per segment sequence");
    }

    #[tokio::test]
    async fn parquet_file_carries_the_schema_version_in_its_metadata() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let store = local_store(&dir);
        write_parquet(&store, 2, &[event(0)]).await.expect("write");
        let bytes = store
            .get(&StorePath::from(parquet_name(2)))
            .await
            .expect("get file")
            .bytes()
            .await
            .expect("file bytes");

        // Act
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).expect("open parquet");

        // Assert
        let expected = crate::event::CURRENT_SCHEMA_VERSION.to_string();
        assert_eq!(
            builder.schema().metadata().get(SCHEMA_VERSION_KEY),
            Some(&expected),
            "file schema metadata"
        );
    }

    // Compactor task

    use std::time::Duration;

    use crate::wal::{segment_name, SystemClock, WalConfig, WalWriter};
    use crate::LogWriter;

    /// Length of a record's frame header: length prefix plus two checksums.
    const FRAME_OVERHEAD: u64 = 12;
    const EVENTS_PER_SEGMENT: usize = 2;
    const TASK_TIMEOUT: Duration = Duration::from_secs(10);

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
        let frame_len = event(0).encode().expect("encode event").len() as u64 + FRAME_OVERHEAD;
        let config = WalConfig {
            segment_max_bytes: EVENTS_PER_SEGMENT as u64 * frame_len,
            rotate_interval_secs: u64::MAX,
        };
        WalWriter::open_with_system_clock(dir, config).expect("open wal writer")
    }

    fn append_events(writer: &mut impl LogWriter, range: std::ops::Range<usize>) {
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

    /// Queues the segments, closes the channel and waits for the task to drain
    /// it and stop, so the files are final when this returns.
    async fn compact_all(dir: &Path, segments: &[&Path]) {
        let (sink, receiver) = ChannelSink::channel();
        for segment in segments {
            sink.segment_closed(segment);
        }
        drop(sink);
        let handle = Compactor::new(dir).expect("compactor").spawn(receiver);
        tokio::time::timeout(TASK_TIMEOUT, handle)
            .await
            .expect("compactor task stops once the channel is closed")
            .expect("compactor task does not panic");
    }

    #[tokio::test]
    async fn notified_segment_becomes_a_parquet_file_and_the_wal_segment_is_removed() {
        // Arrange
        let log = write_log(1);
        let segment = &log.closed[0];
        let store = local_store(&log.dir);

        // Act
        compact_all(log.dir.path(), &[&segment.path]).await;

        // Assert
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            vec![parquet_name(segment.sequence)]
        );
        assert!(!segment.path.exists(), "compacted wal segment is removed");
        let read = read_parquet(&store, segment.sequence)
            .await
            .expect("read parquet");
        assert_same_events(&read, &segment.events);
    }

    #[tokio::test]
    async fn open_segment_is_untouched_after_a_real_notification_was_processed() {
        // Arrange
        let log = write_log(1);
        let open_bytes = std::fs::read(&log.open).expect("read open segment");
        let closed = &log.closed[0];

        // Act
        compact_all(log.dir.path(), &[&closed.path]).await;

        // Assert
        let processed = file_names(log.dir.path(), ".parquet");
        assert_eq!(
            processed,
            vec![parquet_name(closed.sequence)],
            "the notification was processed, so the check below is not vacuous"
        );
        assert_eq!(
            std::fs::read(&log.open).expect("open segment still present"),
            open_bytes,
            "open segment bytes"
        );
        let open_sequence = closed.sequence + 1;
        assert!(!processed.contains(&parquet_name(open_sequence)));
    }

    #[tokio::test]
    async fn only_the_notified_segment_is_compacted() {
        // Arrange
        let log = write_log(2);
        let (untouched, notified) = (&log.closed[0], &log.closed[1]);
        let untouched_bytes = std::fs::read(&untouched.path).expect("read segment");
        let open_bytes = std::fs::read(&log.open).expect("read open segment");

        // Act
        compact_all(log.dir.path(), &[&notified.path]).await;

        // Assert
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            vec![parquet_name(notified.sequence)]
        );
        assert!(!notified.path.exists(), "notified segment is removed");
        assert_eq!(
            std::fs::read(&untouched.path).expect("segment still present"),
            untouched_bytes,
            "closed segment that was not notified"
        );
        assert_eq!(
            std::fs::read(&log.open).expect("open segment still present"),
            open_bytes,
            "open segment bytes"
        );
    }

    #[tokio::test]
    async fn segment_that_no_longer_exists_is_a_no_op() {
        // Arrange
        let log = write_log(1);
        let compactor = Compactor::new(log.dir.path()).expect("compactor");
        let gone = log.dir.path().join(segment_name(7));
        let before = (
            file_names(log.dir.path(), ".wal"),
            file_names(log.dir.path(), ".parquet"),
        );

        // Act
        let outcome = compactor.compact_segment(&gone).await;

        // Assert
        assert!(
            matches!(outcome, Ok(Compaction::AlreadyCompacted)),
            "outcome {outcome:?}"
        );
        let after = (
            file_names(log.dir.path(), ".wal"),
            file_names(log.dir.path(), ".parquet"),
        );
        assert_eq!(after, before, "no empty parquet is created");
    }

    #[tokio::test]
    async fn task_keeps_compacting_after_a_notification_for_a_missing_segment() {
        // Arrange
        let log = write_log(1);
        let closed = &log.closed[0];
        let gone = log.dir.path().join(segment_name(7));

        // Act
        compact_all(log.dir.path(), &[&gone, &closed.path]).await;

        // Assert
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            vec![parquet_name(closed.sequence)]
        );
        assert!(!closed.path.exists(), "segment after the missing one");
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
        let compactor = Compactor::new(log.dir.path()).expect("compactor");

        // Act
        let outcome = compactor.compact_segment(&segment.path).await;

        // Assert
        assert!(
            matches!(outcome, Ok(Compaction::Compacted)),
            "outcome {outcome:?}"
        );
        assert_eq!(
            file_names(log.dir.path(), ".parquet"),
            vec![parquet_name(segment.sequence)]
        );
        assert!(!segment.path.exists(), "wal segment is removed");
        let read = read_parquet(&store, segment.sequence)
            .await
            .expect("read parquet");
        assert_same_events(&read, &segment.events);
    }

    #[tokio::test]
    async fn task_drains_pending_notifications_and_stops_when_the_channel_closes() {
        // Arrange
        let log = write_log(3);
        let paths: Vec<&Path> = log.closed.iter().map(|s| s.path.as_path()).collect();

        // Act
        compact_all(log.dir.path(), &paths).await;

        // Assert
        let expected: Vec<String> = log
            .closed
            .iter()
            .map(|segment| parquet_name(segment.sequence))
            .collect();
        assert_eq!(file_names(log.dir.path(), ".parquet"), expected);
        assert!(log.closed.iter().all(|segment| !segment.path.exists()));
    }

    #[tokio::test]
    async fn rotation_by_the_writer_reaches_the_compactor_through_the_channel_sink() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let (sink, receiver) = ChannelSink::channel();
        let mut writer = open_writer(dir.path()).with_sink(Box::new(sink));
        let handle = Compactor::new(dir.path())
            .expect("compactor")
            .spawn(receiver);
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
}
