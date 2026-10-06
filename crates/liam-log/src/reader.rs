// SPDX-License-Identifier: Apache-2.0

//! Read side of the log: a pluggable `LogReader` and a sequential scan over a
//! log directory.
//!
//! A scan is a pull iterator that loads one segment's events at a time, so a
//! very large log is never held in memory whole and a caller that stops early
//! never pays for the segments it did not reach. A DuckDB-backed reader can
//! implement `LogReader` later without changing the writer or the on-disk
//! format.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::vec;

use object_store::local::LocalFileSystem;
use tokio::runtime::{Builder, Runtime};

use crate::compactor::{read_closed_segment, read_parquet, CompactError};
use crate::event::LogEvent;
use crate::wal::{numbered_files, scan_segment, segment_paths, WalError, PARQUET_EXTENSION};
use crate::LogOffset;

/// One event with the position it was written at.
///
/// A struct so a later read outcome can carry more than the event, without
/// changing the `LogReader` trait.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRecord {
    pub offset: LogOffset,
    pub event: LogEvent,
}

/// Why a scan could not continue. The scan yields nothing after an error.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[error("log io failed: {0}")]
    Io(#[from] io::Error),
    #[error("segment {sequence} could not be read: {source}")]
    Segment { sequence: u64, source: CompactError },
}

/// Records in write order, pulled lazily.
pub type LogScan = Box<dyn Iterator<Item = Result<LogRecord, ReaderError>> + Send>;

/// Reads the log back in write order.
pub trait LogReader {
    /// Yields every record written after `from`, exclusive, or every record
    /// when `from` is `None`.
    fn scan(&self, from: Option<LogOffset>) -> LogScan;
}

/// Scans the Parquet segments and the open WAL tail of one log directory.
///
/// Pulling a scan blocks on its own runtime, so call it from a blocking
/// context, not from inside an async task.
#[derive(Debug, Clone)]
pub struct SequentialScanReader {
    dir: PathBuf,
}

impl SequentialScanReader {
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
        }
    }
}

impl LogReader for SequentialScanReader {
    fn scan(&self, from: Option<LogOffset>) -> LogScan {
        Box::new(SequentialScan {
            dir: self.dir.clone(),
            from,
            segments: None,
            records: Vec::new().into_iter(),
            parquet: None,
            finished: false,
        })
    }
}

/// One segment of the directory and where its events can be read from.
struct SegmentFiles {
    sequence: u64,
    /// The WAL file, if the segment has not been compacted away.
    wal: Option<PathBuf>,
    /// The highest WAL segment is the one the writer appends to, so a torn
    /// tail there is an interrupted append rather than damage.
    open: bool,
}

/// Reads Parquet objects, which are async, from the synchronous scan.
struct ParquetSource {
    runtime: Runtime,
    store: LocalFileSystem,
}

impl ParquetSource {
    fn open(dir: &Path) -> Result<Self, ReaderError> {
        Ok(Self {
            runtime: Builder::new_current_thread().build()?,
            store: LocalFileSystem::new_with_prefix(dir).map_err(io::Error::other)?,
        })
    }
}

/// The pull iterator behind `SequentialScanReader::scan`. The directory is
/// listed on the first pull and one segment's records are resident at a time.
struct SequentialScan {
    dir: PathBuf,
    from: Option<LogOffset>,
    segments: Option<vec::IntoIter<SegmentFiles>>,
    records: vec::IntoIter<LogRecord>,
    parquet: Option<ParquetSource>,
    finished: bool,
}

impl Iterator for SequentialScan {
    type Item = Result<LogRecord, ReaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let item = self.pull().transpose();
        self.finished = !matches!(item, Some(Ok(_)));
        item
    }
}

impl SequentialScan {
    fn pull(&mut self) -> Result<Option<LogRecord>, ReaderError> {
        loop {
            if let Some(record) = self.records.next() {
                return Ok(Some(record));
            }
            let Some(files) = self.segments()?.next() else {
                return Ok(None);
            };
            self.records = self.load(&files)?.into_iter();
        }
    }

    fn segments(&mut self) -> Result<&mut vec::IntoIter<SegmentFiles>, ReaderError> {
        let segments = match self.segments.take() {
            Some(segments) => segments,
            None => list_segments(&self.dir, self.from)?.into_iter(),
        };
        Ok(self.segments.insert(segments))
    }

    fn parquet(&mut self) -> Result<&ParquetSource, ReaderError> {
        let parquet = match self.parquet.take() {
            Some(parquet) => parquet,
            None => ParquetSource::open(&self.dir)?,
        };
        Ok(self.parquet.insert(parquet))
    }

    /// The records of one segment that come after `from`.
    fn load(&mut self, files: &SegmentFiles) -> Result<Vec<LogRecord>, ReaderError> {
        let segment = files.sequence;
        let from = self.from;
        let events = self.read_events(files)?;
        Ok(events
            .into_iter()
            .zip(0..)
            .map(|(event, index)| LogRecord {
                offset: LogOffset { segment, index },
                event,
            })
            .filter(|record| from.is_none_or(|from| record.offset > from))
            .collect())
    }

    /// A WAL file wins while it exists: a Parquet beside it was verified
    /// against it before the file could be removed, so both hold the same
    /// events and a damaged Parquet needs no special case. Without a WAL file
    /// the Parquet is the segment, held to its own stamps.
    fn read_events(&mut self, files: &SegmentFiles) -> Result<Vec<LogEvent>, ReaderError> {
        let sequence = files.sequence;
        let segment = |source| ReaderError::Segment { sequence, source };
        if let Some(wal) = &files.wal {
            let read = if files.open {
                read_open_segment(wal)
            } else {
                read_closed_segment(wal)
            };
            if let Some(events) = read.map_err(segment)? {
                return Ok(events);
            }
            // Compacted after the directory was listed, so the Parquet exists.
        }
        let parquet = self.parquet()?;
        parquet
            .runtime
            .block_on(read_parquet(&parquet.store, sequence))
            .map_err(segment)
    }
}

/// Segments in sequence order that can hold records after `from`.
fn list_segments(dir: &Path, from: Option<LogOffset>) -> io::Result<Vec<SegmentFiles>> {
    // WAL files are listed first: a compaction between the two listings then
    // shows up as a Parquet beside a WAL file, never as a segment in neither.
    let wals = regular_files(segment_paths(dir)?);
    let parquets = regular_files(numbered_files(dir, PARQUET_EXTENSION)?);
    let open_sequence = wals.last().map(|(sequence, _)| *sequence);
    let mut segments: BTreeMap<u64, Option<PathBuf>> = parquets
        .into_iter()
        .map(|(sequence, _)| (sequence, None))
        .collect();
    segments.extend(
        wals.into_iter()
            .map(|(sequence, wal)| (sequence, Some(wal))),
    );
    Ok(segments
        .into_iter()
        .filter(|(sequence, _)| from.is_none_or(|from| *sequence >= from.segment))
        .map(|(sequence, wal)| SegmentFiles {
            sequence,
            wal,
            open: Some(sequence) == open_sequence,
        })
        .collect())
}

fn regular_files(files: Vec<(u64, PathBuf)>) -> Vec<(u64, PathBuf)> {
    files
        .into_iter()
        .filter(|(_, path)| path.is_file())
        .collect()
}

/// Reads the segment the writer may still be appending to, dropping a torn
/// tail. `None` when the file is gone.
fn read_open_segment(segment: &Path) -> Result<Option<Vec<LogEvent>>, CompactError> {
    let bytes = match fs::read(segment) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(WalError::Io(error).into()),
    };
    Ok(Some(scan_segment(segment, &bytes)?.events))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::time::SystemTime;

    use tempfile::TempDir;

    use super::*;
    use crate::compactor::parquet_name;
    use crate::fixtures::{compact_segments, event, put_parquet, rotating_config, write_log};
    use crate::wal::{segment_name, SystemClock, WalWriter, HEADER_BYTES};
    use crate::LogWriter;

    type Logged = Vec<(LogOffset, LogEvent)>;

    const PER_SEGMENT: usize = 2;
    /// Seven events at two per segment: segments 0 to 2 are closed and
    /// segment 3 is the open tail holding one event.
    const EVENTS: usize = 7;
    const CLOSED: [u64; 3] = [0, 1, 2];

    fn at(segment: u64, index: u64) -> LogOffset {
        LogOffset { segment, index }
    }

    fn wal_path(dir: &Path, sequence: u64) -> PathBuf {
        dir.join(segment_name(sequence))
    }

    fn parquet_path(dir: &Path, sequence: u64) -> PathBuf {
        dir.join(parquet_name(sequence))
    }

    fn standard_log() -> (TempDir, Logged) {
        let dir = tempfile::tempdir().expect("temp dir");
        let logged = write_log(dir.path(), EVENTS, PER_SEGMENT);
        (dir, logged)
    }

    /// Every record of a scan that must not fail.
    fn records(dir: &Path, from: Option<LogOffset>) -> Logged {
        SequentialScanReader::new(dir)
            .scan(from)
            .map(|item| {
                let record = item.expect("scan item");
                (record.offset, record.event)
            })
            .collect()
    }

    /// The records before the scan's first error, and that error. Asserts the
    /// scan ends right after an error instead of skipping past it.
    fn records_then_error(dir: &Path, from: Option<LogOffset>) -> (Logged, Option<ReaderError>) {
        let mut scan = SequentialScanReader::new(dir).scan(from);
        let mut records = Vec::new();
        for item in scan.by_ref() {
            match item {
                Ok(record) => records.push((record.offset, record.event)),
                Err(error) => {
                    assert!(scan.next().is_none(), "a scan must stop at its first error");
                    return (records, Some(error));
                }
            }
        }
        (records, None)
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

    fn failed_segment(error: &Option<ReaderError>) -> Option<u64> {
        match error {
            Some(ReaderError::Segment { sequence, .. }) => Some(*sequence),
            _ => None,
        }
    }

    /// Compacts `sequences` but puts the WAL file back, the state a crash
    /// leaves between the verified Parquet and the WAL removal.
    fn compact_keeping_wal(dir: &Path, sequences: &[u64]) {
        for sequence in sequences {
            let wal = fs::read(wal_path(dir, *sequence)).expect("read wal");
            compact_segments(dir, &[*sequence]);
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

    fn snapshot(dir: &Path) -> BTreeMap<String, (Vec<u8>, SystemTime)> {
        fs::read_dir(dir)
            .expect("read dir")
            .map(|entry| {
                let entry = entry.expect("dir entry");
                let modified = entry
                    .metadata()
                    .expect("metadata")
                    .modified()
                    .expect("mtime");
                let bytes = fs::read(entry.path()).unwrap_or_default();
                (
                    entry.file_name().to_string_lossy().into_owned(),
                    (bytes, modified),
                )
            })
            .collect()
    }

    #[test]
    fn full_scan_yields_every_event_in_write_order_with_its_segment_and_index() {
        // Arrange
        let cases: [(&str, &[u64]); 5] = [
            ("nothing compacted", &[]),
            ("first segment compacted", &[0]),
            ("middle segment compacted", &[1]),
            ("all closed segments compacted", &CLOSED),
            ("first and last closed compacted", &[0, 2]),
        ];
        for (label, compacted) in cases {
            let (dir, logged) = standard_log();
            compact_segments(dir.path(), compacted);

            // Act
            let scanned = records(dir.path(), None);

            // Assert
            assert_eq!(scanned, logged, "{label}");
        }
    }

    #[test]
    fn offsets_are_segment_sequence_and_ordinal_within_the_segment() {
        // Arrange
        let (dir, _) = standard_log();
        compact_segments(dir.path(), &CLOSED);

        // Act
        let offsets: Vec<LogOffset> = records(dir.path(), None)
            .into_iter()
            .map(|(offset, _)| offset)
            .collect();

        // Assert
        let expected = [
            at(0, 0),
            at(0, 1),
            at(1, 0),
            at(1, 1),
            at(2, 0),
            at(2, 1),
            at(3, 0),
        ];
        assert_eq!(offsets, expected);
    }

    #[test]
    fn a_wal_segment_with_a_parquet_twin_yields_each_event_once() {
        // Arrange
        let cases: [(&str, &[u64]); 3] = [
            ("one twin", &[1]),
            ("every closed segment has a twin", &CLOSED),
            ("first and last closed", &[0, 2]),
        ];
        for (label, twins) in cases {
            let (dir, logged) = standard_log();
            compact_keeping_wal(dir.path(), twins);

            // Act
            let scanned = records(dir.path(), None);

            // Assert
            assert_eq!(scanned, logged, "{label}");
        }
    }

    #[test]
    fn a_parquet_twin_that_fails_verification_falls_back_to_the_wal() {
        // Arrange
        type Defect = fn(&Path, u64, &[LogEvent]);
        let cases: [(&str, Defect); 7] = [
            ("garbage bytes", |dir, sequence, _| {
                fs::write(parquet_path(dir, sequence), b"not a parquet file").expect("write");
            }),
            ("zero byte file", |dir, sequence, _| {
                fs::write(parquet_path(dir, sequence), []).expect("write");
            }),
            ("truncated file", |dir, sequence, events| {
                put_parquet(dir, sequence, events);
                let bytes = fs::read(parquet_path(dir, sequence)).expect("read");
                fs::write(parquet_path(dir, sequence), &bytes[..bytes.len() / 2]).expect("write");
            }),
            ("fewer events than the segment", |dir, sequence, events| {
                put_parquet(dir, sequence, &events[..1]);
            }),
            ("more events than the segment", |dir, sequence, events| {
                let mut longer = events.to_vec();
                longer.push(event(99));
                put_parquet(dir, sequence, &longer);
            }),
            ("same count with different content", |dir, sequence, _| {
                put_parquet(dir, sequence, &[event(50), event(51)]);
            }),
            ("stamped for another segment", |dir, sequence, events| {
                put_parquet(dir, sequence + 10, events);
                fs::rename(
                    parquet_path(dir, sequence + 10),
                    parquet_path(dir, sequence),
                )
                .expect("rename");
            }),
        ];
        for (label, defect) in cases {
            let (dir, logged) = standard_log();
            defect(dir.path(), 1, &segment_events(&logged, 1));

            // Act
            let scanned = records(dir.path(), None);

            // Assert
            assert_eq!(scanned, logged, "{label}");
        }
    }

    #[test]
    fn scan_from_an_offset_returns_only_later_events_exclusive_of_the_offset() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED);

        for (offset, _) in &logged {
            // Act
            let scanned = records(dir.path(), Some(*offset));

            // Assert
            assert_eq!(scanned, after(&logged, Some(*offset)), "from {offset:?}");
        }
    }

    #[test]
    fn the_same_offset_scans_the_same_before_and_after_compaction() {
        // Arrange
        let (dir, logged) = standard_log();
        let mut offsets: Vec<Option<LogOffset>> = vec![None];
        offsets.extend(logged.iter().map(|(offset, _)| Some(*offset)));
        offsets.push(Some(at(99, 0)));
        let uncompacted: Vec<Logged> = offsets
            .iter()
            .map(|from| records(dir.path(), *from))
            .collect();

        // Act
        compact_segments(dir.path(), &CLOSED);
        let compacted: Vec<Logged> = offsets
            .iter()
            .map(|from| records(dir.path(), *from))
            .collect();

        // Assert
        for ((from, before), after_rotation) in offsets.iter().zip(&uncompacted).zip(&compacted) {
            assert_eq!(before, &after(&logged, *from), "before, from {from:?}");
            assert_eq!(after_rotation, before, "after compaction, from {from:?}");
        }
    }

    #[test]
    fn an_empty_directory_yields_nothing_and_picks_up_a_log_written_later() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");

        // Act
        let empty = records(dir.path(), None);
        let logged = write_log(dir.path(), 2, PER_SEGMENT);
        let later = records(dir.path(), None);

        // Assert
        assert!(empty.is_empty());
        assert_eq!(later, logged);
    }

    #[test]
    fn a_directory_holding_only_a_log_id_yields_nothing() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        fs::write(dir.path().join("log.id"), uuid::Uuid::now_v7().to_string()).expect("write id");

        // Act
        let (scanned, error) = records_then_error(dir.path(), None);
        let logged = write_log(dir.path(), 1, PER_SEGMENT);

        // Assert
        assert!(scanned.is_empty());
        assert!(error.is_none());
        assert_eq!(records(dir.path(), None), logged);
    }

    #[test]
    fn an_open_segment_with_no_records_yields_nothing() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let writer = WalWriter::open_with_system_clock(dir.path(), rotating_config(PER_SEGMENT))
            .expect("open wal");

        // Act
        let (scanned, error) = records_then_error(dir.path(), None);
        drop(writer);
        let logged = write_log(dir.path(), 1, PER_SEGMENT);

        // Assert
        assert!(scanned.is_empty());
        assert!(error.is_none());
        assert_eq!(records(dir.path(), None), logged);
    }

    #[test]
    fn an_offset_at_or_past_the_end_yields_nothing_rather_than_an_error() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED);
        let last = logged.last().expect("events").0;
        let offsets = [
            ("the last event", last),
            ("past the last index", at(last.segment, last.index + 50)),
            ("the next segment", at(last.segment + 1, 0)),
            ("a far segment", at(u64::MAX, u64::MAX)),
        ];

        for (label, offset) in offsets {
            // Act
            let (scanned, error) = records_then_error(dir.path(), Some(offset));

            // Assert
            assert!(scanned.is_empty(), "{label}");
            assert!(error.is_none(), "{label}");
        }
        assert_eq!(records(dir.path(), None), logged);
    }

    #[test]
    fn an_offset_inside_the_wal_tail_returns_the_rest_of_the_tail() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let logged = write_log(dir.path(), 8, 3);
        compact_segments(dir.path(), &[0, 1]);
        let inside_tail = at(2, 0);

        // Act
        let scanned = records(dir.path(), Some(inside_tail));

        // Assert
        assert_eq!(scanned, vec![logged[7].clone()]);
    }

    #[test]
    fn an_offset_in_a_segment_that_no_longer_exists_returns_the_events_after_it() {
        // Arrange
        let offsets = [at(0, 1), at(1, 0), at(1, 1)];
        for offset in offsets {
            let (dir, logged) = standard_log();
            compact_segments(dir.path(), &CLOSED);
            fs::remove_file(parquet_path(dir.path(), 0)).expect("remove segment 0");
            fs::remove_file(parquet_path(dir.path(), 1)).expect("remove segment 1");

            // Act
            let scanned = records(dir.path(), Some(offset));

            // Assert
            assert_eq!(scanned, after(&logged, Some(at(1, 1))), "from {offset:?}");
        }
    }

    #[test]
    fn a_corrupt_parquet_without_a_wal_twin_is_a_typed_error_naming_the_segment() {
        // Arrange
        type Corrupt = fn(&Path, &Logged);
        let cases: [(&str, Corrupt); 3] = [
            ("garbage bytes", |dir, _| {
                fs::write(parquet_path(dir, 1), b"not a parquet file").expect("write");
            }),
            ("zero byte file", |dir, _| {
                fs::write(parquet_path(dir, 1), []).expect("write");
            }),
            ("stamped for another segment", |dir, logged| {
                put_parquet(dir, 11, &segment_events(logged, 1));
                fs::rename(parquet_path(dir, 11), parquet_path(dir, 1)).expect("rename");
            }),
        ];
        for (label, corrupt) in cases {
            let (dir, logged) = standard_log();
            compact_segments(dir.path(), &CLOSED);
            corrupt(dir.path(), &logged);

            // Act
            let (scanned, error) = records_then_error(dir.path(), None);

            // Assert
            assert_eq!(scanned, before_segment(&logged, 1), "{label}");
            assert_eq!(failed_segment(&error), Some(1), "{label}");
            let message = error.expect("error").to_string();
            assert!(message.contains("segment 1"), "{label}: {message}");
        }
    }

    #[test]
    fn a_valid_parquet_holding_no_events_yields_nothing_for_its_segment() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED);
        put_parquet(dir.path(), 1, &[]);

        // Act
        let (scanned, error) = records_then_error(dir.path(), None);

        // Assert
        assert!(error.is_none());
        let expected: Logged = logged
            .iter()
            .filter(|(offset, _)| offset.segment != 1)
            .cloned()
            .collect();
        assert_eq!(scanned, expected);
    }

    #[test]
    fn a_torn_tail_on_the_open_wal_segment_is_dropped_without_an_error() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED);
        append_bytes(&wal_path(dir.path(), 3), &torn_record());

        // Act
        let (scanned, error) = records_then_error(dir.path(), None);

        // Assert
        assert!(error.is_none());
        assert_eq!(scanned, logged);
    }

    #[test]
    fn damage_in_a_closed_wal_segment_is_a_typed_error_that_stops_the_scan() {
        // Arrange
        type Damage = fn(&Path);
        let cases: [(&str, Damage); 2] = [
            ("torn tail", |path| append_bytes(path, &torn_record())),
            ("corrupt first record", corrupt_first_record),
        ];
        for (label, damage) in cases {
            let (dir, logged) = standard_log();
            damage(&wal_path(dir.path(), 1));

            // Act
            let (scanned, error) = records_then_error(dir.path(), None);

            // Assert
            assert_eq!(scanned, before_segment(&logged, 1), "{label}");
            assert_eq!(failed_segment(&error), Some(1), "{label}");
        }
    }

    #[test]
    fn a_corrupt_record_before_the_end_of_the_open_segment_is_a_typed_error() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let logged = write_log(dir.path(), 8, 3);
        corrupt_first_record(&wal_path(dir.path(), 2));

        // Act
        let (scanned, error) = records_then_error(dir.path(), None);

        // Assert
        assert_eq!(scanned, before_segment(&logged, 2));
        assert_eq!(failed_segment(&error), Some(2));
    }

    #[test]
    fn files_that_are_not_log_segments_are_ignored() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &CLOSED);
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
        fs::create_dir(dir.path().join("00000000000000000009.wal")).expect("stray dir");

        // Act
        let (scanned, error) = records_then_error(dir.path(), None);

        // Assert
        assert!(error.is_none());
        assert_eq!(scanned, logged);
    }

    #[test]
    fn scanning_never_modifies_the_directory() {
        // Arrange
        let (dir, logged) = standard_log();
        compact_segments(dir.path(), &[0, 1]);
        fs::write(parquet_path(dir.path(), 2), b"not a parquet file").expect("write");
        append_bytes(&wal_path(dir.path(), 3), &torn_record());
        fs::write(dir.path().join("notes.txt"), b"stray").expect("write stray file");
        let before = snapshot(dir.path());

        // Act
        let first = records(dir.path(), None);
        let second = records(dir.path(), Some(at(1, 0)));

        // Assert
        assert_eq!(first, logged);
        assert_eq!(second, after(&logged, Some(at(1, 0))));
        assert_eq!(snapshot(dir.path()), before);
    }

    #[test]
    fn a_scan_started_before_an_append_returns_a_prefix_of_the_log() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let mut writer =
            WalWriter::open_with_system_clock(dir.path(), rotating_config(100)).expect("open wal");
        let mut logged = Logged::new();
        let mut append = |writer: &mut WalWriter<SystemClock>, index: usize| {
            let offset = writer.append(&event(index)).expect("append");
            logged.push((offset, event(index)));
        };
        (0..3).for_each(|index| append(&mut writer, index));
        let scan = SequentialScanReader::new(dir.path()).scan(None);

        // Act
        (3..5).for_each(|index| append(&mut writer, index));
        let scanned: Logged = scan
            .map(|item| item.expect("scan item"))
            .map(|record| (record.offset, record.event))
            .collect();

        // Assert
        assert!(scanned.len() >= 3, "the records written before the scan");
        assert_eq!(scanned, logged[..scanned.len()]);
    }

    #[test]
    fn scans_racing_a_rotating_writer_only_ever_see_a_prefix_of_the_log() {
        // Arrange
        const TOTAL: usize = 60;
        let dir = tempfile::tempdir().expect("temp dir");
        let writer_dir = dir.path().to_path_buf();
        let writer = std::thread::spawn(move || {
            let mut writer =
                WalWriter::open(&writer_dir, rotating_config(4), SystemClock).expect("open wal");
            (0..TOTAL)
                .map(|index| {
                    let offset = writer.append(&event(index)).expect("append");
                    (offset, event(index))
                })
                .collect::<Logged>()
        });
        let reader = SequentialScanReader::new(dir.path());

        // Act
        let mut racing_scans = Vec::new();
        while !writer.is_finished() {
            racing_scans.push(
                reader
                    .scan(None)
                    .map(|item| item.expect("a racing scan must not fail"))
                    .map(|record| (record.offset, record.event))
                    .collect::<Logged>(),
            );
        }
        let logged = writer.join().expect("writer thread");
        let settled = records(dir.path(), None);

        // Assert
        for (number, scan) in racing_scans.iter().enumerate() {
            assert_eq!(scan, &logged[..scan.len()], "racing scan {number}");
        }
        assert_eq!(settled, logged);
    }

    #[test]
    fn the_reader_is_usable_as_a_trait_object() {
        // Arrange
        let (dir, logged) = standard_log();
        let reader: Box<dyn LogReader> = Box::new(SequentialScanReader::new(dir.path()));

        // Act
        let scanned: Vec<LogRecord> = reader
            .scan(Some(at(0, 1)))
            .collect::<Result<_, _>>()
            .expect("scan");

        // Assert
        let expected: Vec<LogRecord> = after(&logged, Some(at(0, 1)))
            .into_iter()
            .map(|(offset, event)| LogRecord { offset, event })
            .collect();
        assert_eq!(scanned, expected);
    }
}
