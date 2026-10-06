// SPDX-License-Identifier: Apache-2.0

//! Read side of the log: a pluggable `LogReader` and a sequential scan over a
//! log directory.
//!
//! A scan is an async stream that loads one segment's events at a time, so a
//! very large log is never held in memory whole and a caller that stops early
//! never pays for the segments it did not reach. It runs on the caller's
//! runtime and owns none. A DuckDB-backed reader can implement `LogReader`
//! later without changing the writer or the on-disk format.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::vec;

use futures_core::Stream;
use futures_util::stream;
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;

use crate::compactor::{read_event_count, read_parquet, CompactError};
use crate::event::LogEvent;
use crate::wal::{
    numbered_files, open_segment_sequences, read_segment, segment_paths, WalError,
    PARQUET_EXTENSION,
};
use crate::LogOffset;

/// One event with the position it was written at.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LogRecord {
    pub offset: LogOffset,
    pub event: LogEvent,
}

impl LogRecord {
    pub fn new(offset: LogOffset, event: LogEvent) -> Self {
        Self { offset, event }
    }
}

/// Why a scan could not continue. The scan yields nothing after an error.
///
/// Deliberately exhaustive, so a caller that matches it sees every failure
/// mode. Encrypted and partitioned reads (ADR 0011) add `KeyMissing` and
/// `PartitionUnreachable` here.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[error("wal segment {sequence} could not be read: {source}")]
    Wal { sequence: u64, source: WalError },
    #[error("parquet segment {sequence} could not be read: {source}")]
    Parquet { sequence: u64, source: CompactError },
    #[error("object store failed: {0}")]
    Store(#[from] object_store::Error),
    #[error("log io failed: {0}")]
    Io(#[from] io::Error),
    #[error("segment {sequence} is missing from the log")]
    MissingSegment { sequence: u64 },
    #[error(
        "segment {sequence} holds {wal_events} events in its wal but its parquet is stamped with {parquet_events}"
    )]
    TwinMismatch {
        sequence: u64,
        wal_events: u64,
        parquet_events: u64,
    },
}

/// Records in write order, pulled lazily. After an error the stream ends.
pub type LogStream = Pin<Box<dyn Stream<Item = Result<LogRecord, ReaderError>> + Send>>;

/// Reads the log back in write order.
///
/// Records of the open segment may include an append the writer has not yet
/// acknowledged. A caller that applies records must use `scan_through` with the
/// writer's last acknowledged offset (ADR 0011).
pub trait LogReader: Send + Sync {
    /// Yields every record written after `from`, exclusive, or every record
    /// when `from` is `None`.
    fn scan(&self, from: Option<LogOffset>) -> LogStream;

    /// Like `scan`, but stops after the record at `through`, inclusive.
    fn scan_through(&self, from: Option<LogOffset>, through: LogOffset) -> LogStream;
}

/// Scans the Parquet segments and the WAL segments of one log directory.
#[derive(Debug, Clone)]
pub struct SequentialScanReader {
    dir: PathBuf,
    store: Arc<dyn ObjectStore>,
}

impl SequentialScanReader {
    /// A reader whose `store` is rooted at `dir`, so a segment and its Parquet
    /// object sit side by side.
    pub fn new(dir: &Path, store: Arc<dyn ObjectStore>) -> Self {
        Self {
            dir: dir.to_path_buf(),
            store,
        }
    }

    /// A reader over the local directory itself.
    pub fn local(dir: &Path) -> Result<Self, ReaderError> {
        let store = LocalFileSystem::new_with_prefix(dir)?;
        Ok(Self::new(dir, Arc::new(store)))
    }

    fn stream(&self, from: Option<LogOffset>, through: Option<LogOffset>) -> LogStream {
        let scan = Scan {
            reader: self.clone(),
            from,
            through,
            segments: None,
            records: Vec::new().into_iter(),
            finished: false,
        };
        Box::pin(stream::unfold(scan, |mut scan| async move {
            scan.pull().await.map(|item| (item, scan))
        }))
    }
}

impl LogReader for SequentialScanReader {
    fn scan(&self, from: Option<LogOffset>) -> LogStream {
        self.stream(from, None)
    }

    fn scan_through(&self, from: Option<LogOffset>, through: LogOffset) -> LogStream {
        self.stream(from, Some(through))
    }
}

/// One segment of the directory and where its events can be read from.
struct SegmentFiles {
    sequence: u64,
    /// The WAL file, if the segment has not been compacted away.
    wal: Option<PathBuf>,
    /// A Parquet file existed when the directory was listed.
    parquet: bool,
    /// The writer may still append to the WAL, so a torn tail there is an
    /// interrupted append rather than damage.
    open: bool,
}

/// The stream state behind `SequentialScanReader::scan`. The directory is
/// listed on the first pull and one segment's records are resident at a time.
struct Scan {
    reader: SequentialScanReader,
    from: Option<LogOffset>,
    through: Option<LogOffset>,
    segments: Option<vec::IntoIter<SegmentFiles>>,
    records: vec::IntoIter<LogRecord>,
    finished: bool,
}

impl Scan {
    async fn pull(&mut self) -> Option<Result<LogRecord, ReaderError>> {
        if self.finished {
            return None;
        }
        let item = self.next_record().await.transpose();
        self.finished = !matches!(item, Some(Ok(_)));
        item
    }

    async fn next_record(&mut self) -> Result<Option<LogRecord>, ReaderError> {
        loop {
            if let Some(record) = self.records.next() {
                return Ok(Some(record));
            }
            if self.segments.is_none() {
                let listed = self.reader.list(self.from, self.through).await?;
                self.segments = Some(listed.into_iter());
            }
            let Some(files) = self.segments.as_mut().and_then(Iterator::next) else {
                return Ok(None);
            };
            self.records = self.load(&files).await?.into_iter();
        }
    }

    /// The records of one segment inside the scan's bounds.
    async fn load(&self, files: &SegmentFiles) -> Result<Vec<LogRecord>, ReaderError> {
        let segment = files.sequence;
        let events = self.reader.read_events(files).await?;
        Ok(events
            .into_iter()
            .zip(0..)
            .map(|(event, index)| LogRecord::new(LogOffset { segment, index }, event))
            .filter(|record| {
                self.from.is_none_or(|from| record.offset > from)
                    && self.through.is_none_or(|through| record.offset <= through)
            })
            .collect())
    }
}

impl SequentialScanReader {
    async fn list(
        &self,
        from: Option<LogOffset>,
        through: Option<LogOffset>,
    ) -> Result<Vec<SegmentFiles>, ReaderError> {
        let dir = self.dir.clone();
        tokio::task::spawn_blocking(move || list_segments(&dir, from, through))
            .await
            .map_err(io::Error::other)?
    }

    /// The WAL wins while it exists: the compactor removes a WAL only after its
    /// Parquet verified, so while the WAL exists it is the only copy known
    /// good and a twin beside it may be partial or stale. The twin is a
    /// cross-check of the event count, and the fallback when the WAL is damaged.
    async fn read_events(&self, files: &SegmentFiles) -> Result<Vec<LogEvent>, ReaderError> {
        let sequence = files.sequence;
        if let Some(wal) = files.wal.clone() {
            let closed = !files.open;
            let read = tokio::task::spawn_blocking(move || read_segment(&wal, closed))
                .await
                .map_err(io::Error::other)?;
            match read {
                Ok(Some(events)) => {
                    if files.parquet {
                        self.ensure_twin_agrees(sequence, events.len()).await?;
                    }
                    return Ok(events);
                }
                Ok(None) => {} // Compacted after the directory was listed.
                Err(source) => return self.fall_back_to_twin(files, source).await,
            }
        }
        self.read_twin(sequence).await
    }

    async fn read_twin(&self, sequence: u64) -> Result<Vec<LogEvent>, ReaderError> {
        read_parquet(self.store.as_ref(), sequence)
            .await
            .map_err(|source| ReaderError::Parquet { sequence, source })
    }

    /// A twin whose footer cannot be read is ignored, since the WAL is the
    /// authority; one that counts other events than the WAL does means one of
    /// them lost records, so the scan stops instead of picking a side.
    async fn ensure_twin_agrees(
        &self,
        sequence: u64,
        wal_events: usize,
    ) -> Result<(), ReaderError> {
        let wal_events = wal_events as u64;
        match read_event_count(self.store.as_ref(), sequence).await {
            Ok(parquet_events) if parquet_events == wal_events => Ok(()),
            Ok(parquet_events) => Err(ReaderError::TwinMismatch {
                sequence,
                wal_events,
                parquet_events,
            }),
            Err(CompactError::Store(object_store::Error::NotFound { .. })) => Ok(()),
            Err(CompactError::Store(error)) => Err(ReaderError::Store(error)),
            Err(error) => {
                tracing::warn!(sequence, %error, "parquet twin has no readable event count, trusting the wal");
                Ok(())
            }
        }
    }

    /// A damaged WAL is replaced by its Parquet twin only when the twin
    /// verifies by its own stamps; otherwise the damage is the error.
    async fn fall_back_to_twin(
        &self,
        files: &SegmentFiles,
        source: WalError,
    ) -> Result<Vec<LogEvent>, ReaderError> {
        let sequence = files.sequence;
        let damaged = ReaderError::Wal { sequence, source };
        if !files.parquet {
            return Err(damaged);
        }
        match self.read_twin(sequence).await {
            Ok(events) => {
                tracing::warn!(sequence, error = %damaged, "wal is damaged, reading its parquet twin");
                Ok(events)
            }
            Err(twin_error) => {
                tracing::warn!(sequence, error = %twin_error, "parquet twin of a damaged wal does not verify");
                Err(damaged)
            }
        }
    }
}

/// The segments from `from` through `through`, after checking that none of
/// them is missing.
fn list_segments(
    dir: &Path,
    from: Option<LogOffset>,
    through: Option<LogOffset>,
) -> Result<Vec<SegmentFiles>, ReaderError> {
    // WAL files are listed first: a compaction between the two listings then
    // shows up as a Parquet beside a WAL file, never as a segment in neither.
    let wals = segment_paths(dir)?;
    let parquets = numbered_files(dir, PARQUET_EXTENSION)?;
    let open = open_segment_sequences(&wals);
    let mut segments = BTreeMap::new();
    for (sequence, wal) in wals {
        let files = SegmentFiles {
            sequence,
            wal: Some(wal),
            parquet: false,
            open: open.contains(&sequence),
        };
        segments.insert(sequence, files);
    }
    for (sequence, _) in parquets {
        segments
            .entry(sequence)
            .or_insert(SegmentFiles {
                sequence,
                wal: None,
                parquet: false,
                open: false,
            })
            .parquet = true;
    }

    let start = from.map_or(0, |from| from.segment);
    let end = through.map_or(u64::MAX, |through| through.segment);
    let mut expected = start;
    let mut wanted = Vec::new();
    for (sequence, files) in segments {
        if !(start..=end).contains(&sequence) {
            continue;
        }
        if sequence != expected {
            return Err(ReaderError::MissingSegment { sequence: expected });
        }
        expected = sequence.saturating_add(1);
        wanted.push(files);
    }
    Ok(wanted)
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::ops::Range;

    use futures_util::StreamExt;
    use tempfile::TempDir;

    use super::*;
    use crate::fixtures::{
        compact_segments, event, frame_len, parquet_path, put_parquet, rotating_config, snapshot,
        wal_path, write_log,
    };
    use crate::wal::{SystemClock, WalWriter, HEADER_BYTES};
    use crate::LogWriter;

    type Logged = Vec<(LogOffset, LogEvent)>;

    const PER_SEGMENT: usize = 2;
    /// Seven events at two per segment: segments 0 to 2 are closed and
    /// segment 3 is the open tail holding one event.
    const EVENTS: usize = 7;
    const CLOSED: [u64; 3] = [0, 1, 2];
    const GARBAGE: &[u8] = b"not a parquet file";

    fn at(segment: u64, index: u64) -> LogOffset {
        LogOffset { segment, index }
    }

    fn standard_log() -> (TempDir, Logged) {
        let dir = tempfile::tempdir().expect("temp dir");
        let logged = write_log(dir.path(), EVENTS, PER_SEGMENT);
        (dir, logged)
    }

    fn local_reader(dir: &Path) -> SequentialScanReader {
        SequentialScanReader::local(dir).expect("reader")
    }

    fn pair(record: LogRecord) -> (LogOffset, LogEvent) {
        (record.offset, record.event)
    }

    /// The records before the stream's first error, and that error. Asserts
    /// the stream ends right after an error instead of skipping past it.
    async fn drain(mut stream: LogStream) -> (Logged, Option<ReaderError>) {
        let mut records = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(record) => records.push(pair(record)),
                Err(error) => {
                    assert!(
                        stream.next().await.is_none(),
                        "a scan must stop at its error"
                    );
                    return (records, Some(error));
                }
            }
        }
        (records, None)
    }

    async fn scanned(dir: &Path, from: Option<LogOffset>) -> (Logged, Option<ReaderError>) {
        drain(local_reader(dir).scan(from)).await
    }

    /// Every record of a scan that must not fail.
    async fn records(dir: &Path, from: Option<LogOffset>) -> Logged {
        let (records, error) = scanned(dir, from).await;
        assert!(error.is_none(), "scan failed: {error:?}");
        records
    }

    fn after(logged: &Logged, from: Option<LogOffset>) -> Logged {
        logged
            .iter()
            .filter(|(offset, _)| from.is_none_or(|from| *offset > from))
            .cloned()
            .collect()
    }

    fn before_segment(logged: &Logged, sequence: u64) -> Logged {
        logged
            .iter()
            .filter(|(offset, _)| offset.segment < sequence)
            .cloned()
            .collect()
    }

    fn segment_events(logged: &Logged, sequence: u64) -> Vec<LogEvent> {
        logged
            .iter()
            .filter(|(offset, _)| offset.segment == sequence)
            .map(|(_, event)| event.clone())
            .collect()
    }

    /// The bytes of the Parquet object for `sequence` holding `events`.
    async fn parquet_bytes(sequence: u64, events: &[LogEvent]) -> Vec<u8> {
        let scratch = tempfile::tempdir().expect("temp dir");
        put_parquet(scratch.path(), sequence, events).await;
        fs::read(parquet_path(scratch.path(), sequence)).expect("read parquet")
    }

    /// Files that cannot stand in for the Parquet of segment 1.
    async fn broken_parquets(events: &[LogEvent]) -> Vec<(&'static str, Vec<u8>)> {
        let whole = parquet_bytes(1, events).await;
        vec![
            ("garbage bytes", GARBAGE.to_vec()),
            ("zero byte file", Vec::new()),
            ("truncated file", whole[..whole.len() / 2].to_vec()),
            (
                "stamped for another segment",
                parquet_bytes(11, events).await,
            ),
        ]
    }

    /// Compacts `sequences` but puts the WAL file back, the state a crash
    /// leaves between the verified Parquet and the WAL removal.
    async fn compact_keeping_wal(dir: &Path, sequences: &[u64]) {
        for sequence in sequences {
            let wal = fs::read(wal_path(dir, *sequence)).expect("read wal");
            compact_segments(dir, &[*sequence]).await;
            fs::write(wal_path(dir, *sequence), wal).expect("restore wal");
        }
    }

    fn append_bytes(path: &Path, bytes: &[u8]) {
        OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open segment")
            .write_all(bytes)
            .expect("append bytes");
    }

    /// The start of a real record cut short, as an interrupted append leaves it.
    fn torn_record() -> Vec<u8> {
        let scratch = tempfile::tempdir().expect("temp dir");
        let mut writer =
            WalWriter::open_with_system_clock(scratch.path(), rotating_config(PER_SEGMENT))
                .expect("open wal");
        writer.append(&event(900)).expect("append event");
        let mut bytes = fs::read(wal_path(scratch.path(), 0)).expect("read wal");
        bytes.truncate(bytes.len() - 3);
        bytes
    }

    /// Flips a payload byte of the first record, a checksum failure that is not
    /// at the tail when the segment holds more records.
    fn corrupt_first_record(path: &Path) {
        let mut bytes = fs::read(path).expect("read segment");
        bytes[HEADER_BYTES + 1] ^= 0xFF;
        fs::write(path, bytes).expect("write segment");
    }

    fn append_all(writer: &mut WalWriter<SystemClock>, logged: &mut Logged, range: Range<usize>) {
        for index in range {
            let offset = writer.append(&event(index)).expect("append");
            logged.push((offset, event(index)));
        }
    }

    #[tokio::test]
    async fn full_scan_yields_every_event_in_write_order_with_its_segment_and_index() {
        // Arrange
        let cases: [(&str, &[u64], &[u64]); 7] = [
            ("nothing compacted", &[], &[]),
            ("first segment compacted", &[0], &[]),
            ("middle segment compacted", &[1], &[]),
            ("all closed segments compacted", &CLOSED, &[]),
            ("first and last closed compacted", &[0, 2], &[]),
            ("one wal beside its parquet twin", &[], &[1]),
            ("every closed wal beside its twin", &[], &CLOSED),
        ];
        for (label, compacted, twinned) in cases {
            let (dir, logged) = standard_log();
            compact_segments(dir.path(), compacted).await;
            compact_keeping_wal(dir.path(), twinned).await;

            // Act
            let scanned = records(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, logged, "{label}");
        }
    }

    #[tokio::test]
    async fn the_wal_wins_over_a_twin_whose_stamped_count_it_cannot_contradict() {
        // Arrange
        let (_, logged) = standard_log();
        let events = segment_events(&logged, 1);
        let mut twins = broken_parquets(&events).await;
        twins.push((
            "same count, other content",
            parquet_bytes(1, &[event(50), event(51)]).await,
        ));
        for (label, twin) in twins {
            let (dir, logged) = standard_log();
            fs::write(parquet_path(dir.path(), 1), twin).expect("write twin");

            // Act
            let scanned = records(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, logged, "{label}");
        }
    }

    #[tokio::test]
    async fn a_twin_counting_other_events_than_the_wal_is_a_typed_error() {
        // Arrange
        let cases = [
            ("parquet holds fewer", 1, 2),
            ("parquet holds more", 3, 2),
            ("wal cut at a record boundary", 2, 1),
        ];
        for (label, parquet_events, wal_events) in cases {
            let (dir, logged) = standard_log();
            let mut events = segment_events(&logged, 1);
            events.resize_with(parquet_events, || event(99));
            fs::write(parquet_path(dir.path(), 1), parquet_bytes(1, &events).await)
                .expect("write twin");
            OpenOptions::new()
                .write(true)
                .open(wal_path(dir.path(), 1))
                .expect("open wal")
                .set_len(wal_events * frame_len(&event(0)))
                .expect("cut wal");

            // Act
            let (scanned, error) = scanned(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, before_segment(&logged, 1), "{label}");
            assert!(
                matches!(
                    error,
                    Some(ReaderError::TwinMismatch { sequence: 1, wal_events: w, parquet_events: p })
                        if w == wal_events && p == parquet_events as u64
                ),
                "{label}: {error:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_damaged_wal_falls_back_to_a_parquet_twin_that_verifies() {
        // Arrange
        type Damage = fn(&Path);
        let cases: [(&str, Damage); 2] = [
            ("torn tail", |path| append_bytes(path, &torn_record())),
            ("corrupt first record", corrupt_first_record),
        ];
        for (label, damage) in cases {
            let (dir, logged) = standard_log();
            compact_keeping_wal(dir.path(), &[1]).await;
            damage(&wal_path(dir.path(), 1));

            // Act
            let scanned = records(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, logged, "{label}");
        }
    }

    #[tokio::test]
    async fn a_damaged_wal_beside_a_twin_that_does_not_verify_is_a_typed_error() {
        // Arrange
        let (_, logged) = standard_log();
        for (label, twin) in broken_parquets(&segment_events(&logged, 1)).await {
            let (dir, logged) = standard_log();
            corrupt_first_record(&wal_path(dir.path(), 1));
            fs::write(parquet_path(dir.path(), 1), twin).expect("write twin");

            // Act
            let (scanned, error) = scanned(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, before_segment(&logged, 1), "{label}");
            assert!(
                matches!(error, Some(ReaderError::Wal { sequence: 1, .. })),
                "{label}"
            );
        }
    }

    #[tokio::test]
    async fn scan_from_an_offset_is_exclusive_and_the_same_before_and_after_compaction() {
        // Arrange
        let (dir, logged) = standard_log();
        let last = logged.last().expect("events").0;
        let mut offsets = vec![None];
        offsets.extend(logged.iter().map(|(offset, _)| Some(*offset)));
        offsets.extend(
            [
                at(last.segment, last.index + 50),
                at(last.segment + 1, 0),
                at(u64::MAX, u64::MAX),
            ]
            .map(Some),
        );
        let mut uncompacted = Vec::new();
        for from in &offsets {
            uncompacted.push(records(dir.path(), *from).await);
        }

        // Act
        compact_segments(dir.path(), &CLOSED).await;

        // Assert
        for (from, before) in offsets.iter().zip(&uncompacted) {
            assert_eq!(before, &after(&logged, *from), "before, from {from:?}");
            assert_eq!(
                &records(dir.path(), *from).await,
                before,
                "compacted, from {from:?}"
            );
        }
    }

    #[tokio::test]
    async fn scan_through_stops_after_the_record_at_through_and_reads_no_later_segment() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED).await;
        let reader = local_reader(dir.path());
        let reader: &dyn LogReader = &reader;
        let mut froms = vec![None];
        froms.extend(logged.iter().map(|(offset, _)| Some(*offset)));
        let mut throughs: Vec<LogOffset> = logged.iter().map(|(offset, _)| *offset).collect();
        throughs.push(at(99, 0));

        for from in froms {
            for through in &throughs {
                // Act
                let (scanned, error) = drain(reader.scan_through(from, *through)).await;

                // Assert
                let expected: Logged = after(&logged, from)
                    .into_iter()
                    .filter(|(offset, _)| offset <= through)
                    .collect();
                assert!(error.is_none(), "from {from:?} through {through:?}");
                assert_eq!(scanned, expected, "from {from:?} through {through:?}");
            }
        }
        fs::write(parquet_path(dir.path(), 2), GARBAGE).expect("corrupt a later segment");
        let (scanned, error) = drain(reader.scan_through(None, at(1, 1))).await;
        assert!(error.is_none());
        assert_eq!(scanned, before_segment(&logged, 2));
    }

    #[tokio::test]
    async fn nothing_to_read_yields_nothing_and_a_log_written_later_is_picked_up() {
        // Arrange
        type Setup = fn(&Path);
        let cases: [(&str, Setup); 3] = [
            ("empty directory", |_| {}),
            ("only a log id", |dir| {
                fs::write(dir.join("log.id"), uuid::Uuid::now_v7().to_string()).expect("write id");
            }),
            ("an open segment with no records", |dir| {
                WalWriter::open_with_system_clock(dir, rotating_config(PER_SEGMENT))
                    .expect("open wal");
            }),
        ];
        for (label, setup) in cases {
            let dir = tempfile::tempdir().expect("temp dir");
            setup(dir.path());

            // Act
            let empty = scanned(dir.path(), None).await;
            let logged = write_log(dir.path(), 1, PER_SEGMENT);

            // Assert
            assert_eq!(empty.0, Logged::new(), "{label}");
            assert!(empty.1.is_none(), "{label}");
            assert_eq!(records(dir.path(), None).await, logged, "{label}");
        }
    }

    #[tokio::test]
    async fn a_missing_segment_is_a_typed_error_before_any_record_is_yielded() {
        // Arrange
        let cases = [
            ("front gap", vec![0], None, Some(0)),
            ("middle gap", vec![1], None, Some(1)),
            ("gap after the cursor", vec![2], Some(at(0, 1)), Some(2)),
            (
                "the cursor's segment is missing",
                vec![1],
                Some(at(1, 1)),
                Some(1),
            ),
            ("gap before the cursor", vec![0, 1], Some(at(2, 0)), None),
        ];
        for (label, removed, from, missing) in cases {
            let (dir, logged) = standard_log();
            compact_segments(dir.path(), &CLOSED).await;
            for sequence in removed {
                fs::remove_file(parquet_path(dir.path(), sequence)).expect("remove segment");
            }

            // Act
            let (scanned, error) = scanned(dir.path(), from).await;

            // Assert
            match missing {
                Some(sequence) => {
                    assert!(scanned.is_empty(), "{label}");
                    assert!(
                        matches!(error, Some(ReaderError::MissingSegment { sequence: s }) if s == sequence),
                        "{label}: {error:?}"
                    );
                }
                None => {
                    assert!(error.is_none(), "{label}");
                    assert_eq!(scanned, after(&logged, from), "{label}");
                }
            }
        }
    }

    #[tokio::test]
    async fn a_corrupt_parquet_without_a_wal_twin_is_a_typed_error_naming_the_segment() {
        // Arrange
        let (_, logged) = standard_log();
        for (label, corrupt) in broken_parquets(&segment_events(&logged, 1)).await {
            let (dir, logged) = standard_log();
            compact_segments(dir.path(), &CLOSED).await;
            fs::write(parquet_path(dir.path(), 1), corrupt).expect("write");

            // Act
            let (scanned, error) = scanned(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, before_segment(&logged, 1), "{label}");
            assert!(
                matches!(error, Some(ReaderError::Parquet { sequence: 1, .. })),
                "{label}"
            );
            let message = error.expect("error").to_string();
            assert!(message.contains("segment 1"), "{label}: {message}");
        }
    }

    #[tokio::test]
    async fn a_valid_parquet_holding_no_events_yields_nothing_for_its_segment() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED).await;
        put_parquet(dir.path(), 1, &[]).await;

        // Act
        let scanned = records(dir.path(), None).await;

        // Assert
        let expected: Logged = logged
            .iter()
            .filter(|(offset, _)| offset.segment != 1)
            .cloned()
            .collect();
        assert_eq!(scanned, expected);
    }

    #[tokio::test]
    async fn a_torn_tail_on_an_open_wal_segment_is_dropped_without_an_error() {
        // Arrange
        type Setup = fn(&Path);
        let cases: [(&str, Setup); 2] = [
            ("the highest segment", |dir| {
                append_bytes(&wal_path(dir, 3), &torn_record());
            }),
            ("below an empty successor", |dir| {
                append_bytes(&wal_path(dir, 3), &torn_record());
                fs::write(wal_path(dir, 4), b"").expect("failed rotation's successor");
            }),
        ];
        for (label, setup) in cases {
            let (dir, logged) = standard_log();
            compact_segments(dir.path(), &CLOSED).await;
            setup(dir.path());

            // Act
            let scanned = records(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, logged, "{label}");
        }
    }

    #[tokio::test]
    async fn damage_in_a_wal_segment_without_a_twin_is_a_typed_error_that_stops_the_scan() {
        // Arrange
        type Damage = fn(&Path);
        let cases: [(&str, usize, usize, u64, Damage); 3] = [
            (
                "torn tail of a closed segment",
                EVENTS,
                PER_SEGMENT,
                1,
                |path| {
                    append_bytes(path, &torn_record());
                },
            ),
            (
                "corrupt first record",
                EVENTS,
                PER_SEGMENT,
                1,
                corrupt_first_record,
            ),
            (
                "corrupt record before the end of the open segment",
                8,
                3,
                2,
                corrupt_first_record,
            ),
        ];
        for (label, events, per_segment, damaged, damage) in cases {
            let dir = tempfile::tempdir().expect("temp dir");
            let logged = write_log(dir.path(), events, per_segment);
            damage(&wal_path(dir.path(), damaged));

            // Act
            let (scanned, error) = scanned(dir.path(), None).await;

            // Assert
            assert_eq!(scanned, before_segment(&logged, damaged), "{label}");
            assert!(
                matches!(error, Some(ReaderError::Wal { sequence, .. }) if sequence == damaged),
                "{label}"
            );
        }
    }

    #[tokio::test]
    async fn files_that_are_not_log_segments_are_ignored() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED).await;
        for name in [
            "notes.txt",
            "9.wal",
            "+3.wal",
            "00000000000000000009.wal.tmp",
            "00000000000000000009.parquet.tmp",
            "99999999999999999999.txt",
            "log.id.tmp",
        ] {
            fs::write(dir.path().join(name), b"stray").expect("write stray file");
        }
        fs::create_dir(wal_path(dir.path(), 9)).expect("stray dir");
        fs::create_dir(parquet_path(dir.path(), 8)).expect("stray dir");

        // Act
        let scanned = records(dir.path(), None).await;

        // Assert
        assert_eq!(scanned, logged);
    }

    #[tokio::test]
    async fn scanning_never_modifies_the_directory() {
        // Arrange
        let (mixed, logged) = standard_log();
        compact_segments(mixed.path(), &[0, 1]).await;
        fs::write(parquet_path(mixed.path(), 2), GARBAGE).expect("write");
        append_bytes(&wal_path(mixed.path(), 3), &torn_record());
        fs::write(mixed.path().join("notes.txt"), b"stray").expect("write stray file");
        let (corrupt, _) = standard_log();
        compact_segments(corrupt.path(), &CLOSED).await;
        fs::write(parquet_path(corrupt.path(), 1), GARBAGE).expect("write");
        let (torn, _) = standard_log();
        append_bytes(&wal_path(torn.path(), 1), &torn_record());
        let empty = tempfile::tempdir().expect("temp dir");
        assert_eq!(records(mixed.path(), None).await, logged);

        for dir in [mixed, corrupt, torn, empty] {
            let before = snapshot(dir.path());

            // Act
            scanned(dir.path(), None).await;
            scanned(dir.path(), Some(at(1, 0))).await;

            // Assert
            assert_eq!(snapshot(dir.path()), before);
        }
    }

    #[tokio::test]
    async fn a_scan_started_before_appends_returns_a_prefix_with_later_records_of_its_open_segment()
    {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let mut writer =
            WalWriter::open_with_system_clock(dir.path(), rotating_config(PER_SEGMENT))
                .expect("open wal");
        let mut logged = Logged::new();
        append_all(&mut writer, &mut logged, 0..3);
        let mut stream = local_reader(dir.path()).scan(None);
        let first = pair(
            stream
                .next()
                .await
                .expect("first item")
                .expect("first record"),
        );

        // Act
        append_all(&mut writer, &mut logged, 3..7);
        let (rest, error) = drain(stream).await;

        // Assert
        assert!(error.is_none());
        let seen = [vec![first], rest].concat();
        assert_eq!(
            seen,
            logged[..4],
            "events 0 to 3, the last one appended after the scan began"
        );
    }

    #[tokio::test]
    async fn scans_racing_a_rotating_writer_only_ever_see_a_prefix_of_the_log() {
        // Arrange
        const TOTAL: usize = 60;
        const ATTEMPTS: usize = 5;
        let mut saw_partial_prefix = false;
        for _ in 0..ATTEMPTS {
            let dir = tempfile::tempdir().expect("temp dir");
            let writer_dir = dir.path().to_path_buf();
            let writer = std::thread::spawn(move || {
                let mut writer = WalWriter::open(&writer_dir, rotating_config(4), SystemClock)
                    .expect("open wal");
                let mut logged = Logged::new();
                append_all(&mut writer, &mut logged, 0..TOTAL);
                logged
            });

            // Act
            let mut racing = Vec::new();
            while !writer.is_finished() {
                let (scan, error) = scanned(dir.path(), None).await;
                assert!(error.is_none(), "a racing scan must not fail: {error:?}");
                racing.push(scan);
            }
            let logged = writer.join().expect("writer thread");

            // Assert
            for (number, scan) in racing.iter().enumerate() {
                assert_eq!(scan, &logged[..scan.len()], "racing scan {number}");
            }
            assert_eq!(records(dir.path(), None).await, logged);
            saw_partial_prefix = racing
                .iter()
                .any(|scan| !scan.is_empty() && scan.len() < TOTAL);
            if saw_partial_prefix {
                break;
            }
        }
        assert!(
            saw_partial_prefix,
            "no scan overlapped the writer in {ATTEMPTS} attempts"
        );
    }

    #[tokio::test]
    async fn compacting_listed_segments_mid_scan_does_not_change_what_the_scan_yields() {
        // Arrange
        let (dir, mut logged) = standard_log();
        let mut stream = local_reader(dir.path()).scan(None);
        let first = pair(
            stream
                .next()
                .await
                .expect("first item")
                .expect("first record"),
        );

        // Act
        compact_segments(dir.path(), &CLOSED).await;
        let (rest, error) = drain(stream).await;

        // Assert
        assert!(error.is_none());
        assert_eq!([vec![first], rest].concat(), logged);

        // Arrange: the listed open segment rotates and is compacted mid-scan
        let (dir, _) = standard_log();
        let mut stream = local_reader(dir.path()).scan(None);
        let first = pair(
            stream
                .next()
                .await
                .expect("first item")
                .expect("first record"),
        );
        let mut writer =
            WalWriter::open_with_system_clock(dir.path(), rotating_config(PER_SEGMENT))
                .expect("open wal");
        append_all(&mut writer, &mut logged, EVENTS..EVENTS + 2);

        // Act
        compact_segments(dir.path(), &[3]).await;
        let (rest, error) = drain(stream).await;

        // Assert
        assert!(error.is_none());
        assert_eq!(
            [vec![first], rest].concat(),
            logged[..EVENTS + 1],
            "up to segment 3, now Parquet"
        );
    }

    #[tokio::test]
    async fn a_scan_reads_only_the_segments_it_reaches() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED).await;
        let mut stream = local_reader(dir.path()).scan(None);
        let first = pair(
            stream
                .next()
                .await
                .expect("first item")
                .expect("first record"),
        );

        // Act
        fs::write(parquet_path(dir.path(), 2), GARBAGE).expect("corrupt a later segment");
        let (rest, error) = drain(stream).await;
        let taken: Vec<_> = local_reader(dir.path()).scan(None).take(1).collect().await;

        // Assert
        assert_eq!([vec![first], rest].concat(), before_segment(&logged, 2));
        assert!(matches!(
            error,
            Some(ReaderError::Parquet { sequence: 2, .. })
        ));
        assert!(
            matches!(taken.as_slice(), [Ok(_)]),
            "take(1) never reaches segment 2"
        );

        // Arrange: a resume cursor skips the corrupt segment before it
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED).await;
        fs::write(parquet_path(dir.path(), 0), GARBAGE).expect("corrupt an earlier segment");

        // Act
        let resumed = records(dir.path(), Some(at(1, 0))).await;

        // Assert
        assert_eq!(resumed, after(&logged, Some(at(1, 0))));
    }
}
