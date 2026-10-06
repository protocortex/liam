// SPDX-License-Identifier: Apache-2.0

//! Write-ahead log segments: length-prefixed postcard records, fsynced per append.
//!
//! The writer owns rotation so a segment is only ever read by others once it
//! is closed; callers serialize appends themselves.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

use crate::event::{EventError, LogEvent};
use crate::{LogOffset, LogWriter};

const SEGMENT_EXTENSION: &str = "wal";
const MANIFEST_NAME: &str = "log.id";
const LENGTH_PREFIX_BYTES: usize = 4;
const SEQUENCE_DIGITS: usize = 20;

/// Why a WAL operation failed.
#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("wal io failed: {0}")]
    Io(#[from] io::Error),
    #[error("wal record could not be encoded: {0}")]
    Encode(#[from] EventError),
    #[error("wal record of {0} bytes exceeds the 4 byte length prefix")]
    RecordTooLarge(usize),
    #[error("wal segment {} has a corrupt record at offset {offset}", segment.display())]
    Corrupt { segment: PathBuf, offset: u64 },
    #[error("wal log id manifest {} does not hold a uuid", manifest.display())]
    InvalidLogId { manifest: PathBuf },
}

/// Source of the current time, injected so rotation tests never sleep.
pub trait RotationClock {
    fn now_secs(&self) -> u64;
}

/// Wall clock in seconds since the UNIX epoch; a clock set before it reads 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl RotationClock for SystemClock {
    fn now_secs(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs())
    }
}

/// Told about each segment once the writer has fully closed it, so a consumer
/// never reads a segment that is still being appended to.
pub trait SegmentSink: Send + Sync {
    fn segment_closed(&self, path: &Path);
}

/// Thresholds that close the open segment and start the next one.
#[derive(Debug, Clone, Copy)]
pub struct WalConfig {
    pub segment_max_bytes: u64,
    pub rotate_interval_secs: u64,
}

struct OpenSegment {
    file: File,
    sequence: u64,
    size: u64,
    records: u64,
    opened_at: u64,
}

/// Appends events to the open segment of a log directory.
pub struct WalWriter<C: RotationClock> {
    dir: PathBuf,
    config: WalConfig,
    clock: C,
    log_id: Uuid,
    segment: OpenSegment,
    sink: Option<Box<dyn SegmentSink>>,
}

impl WalWriter<SystemClock> {
    /// Opens the log directory timing rotation by the wall clock.
    pub fn open_with_system_clock(dir: &Path, config: WalConfig) -> Result<Self, WalError> {
        Self::open(dir, config, SystemClock)
    }
}

impl<C: RotationClock> WalWriter<C> {
    /// Opens the log directory, creating the first segment when none exists.
    pub fn open(dir: &Path, config: WalConfig, clock: C) -> Result<Self, WalError> {
        fs::create_dir_all(dir)?;
        let log_id = load_or_create_log_id(dir)?;
        let segment = match segment_paths(dir)?.last() {
            Some((sequence, path)) => reopen_segment(path, *sequence, clock.now_secs())?,
            None => create_segment(dir, 0, clock.now_secs())?,
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            clock,
            log_id,
            segment,
            sink: None,
        })
    }

    /// Delivers a notification to `sink` for every segment closed from now on.
    pub fn with_sink(mut self, sink: Box<dyn SegmentSink>) -> Self {
        self.sink = Some(sink);
        self
    }

    fn should_rotate(&self) -> bool {
        let elapsed = self.clock.now_secs().saturating_sub(self.segment.opened_at);
        self.segment.size >= self.config.segment_max_bytes
            || elapsed >= self.config.rotate_interval_secs
    }

    fn rotate(&mut self) -> Result<(), WalError> {
        let closed = self.dir.join(segment_name(self.segment.sequence));
        self.segment = create_segment(&self.dir, self.segment.sequence + 1, self.clock.now_secs())?;
        // Every append is already fsynced, so the closed segment is complete here.
        if let Some(sink) = &self.sink {
            sink.segment_closed(&closed);
        }
        Ok(())
    }
}

impl<C: RotationClock + Send> LogWriter for WalWriter<C> {
    fn append(&mut self, event: &LogEvent) -> Result<LogOffset, WalError> {
        let payload = event.encode()?;
        let length = record_length(payload.len())?;
        let mut record = Vec::with_capacity(LENGTH_PREFIX_BYTES + payload.len());
        record.extend_from_slice(&length.to_le_bytes());
        record.extend_from_slice(&payload);

        let offset = LogOffset {
            segment: self.segment.sequence,
            index: self.segment.records,
        };
        self.segment.file.write_all(&record)?;
        self.segment.file.sync_all()?;
        self.segment.size += record.len() as u64;
        self.segment.records += 1;

        if self.should_rotate() {
            self.rotate()?;
        }
        Ok(offset)
    }

    /// Identifies this log; stable across reopens of the same directory.
    fn log_id(&self) -> Uuid {
        self.log_id
    }
}

/// The length prefix value for a payload, rejecting one the prefix cannot hold.
fn record_length(len: usize) -> Result<u32, WalError> {
    u32::try_from(len).map_err(|_| WalError::RecordTooLarge(len))
}

/// Reads every event in the log directory, oldest segment first.
#[cfg(test)]
pub(crate) fn replay(dir: &Path) -> Result<Vec<LogEvent>, WalError> {
    let mut events = Vec::new();
    for (_, path) in segment_paths(dir)? {
        events.extend(scan_segment(&path, &fs::read(&path)?)?.events);
    }
    Ok(events)
}

struct Scan {
    events: Vec<LogEvent>,
    /// Length of the prefix made of complete, decodable records.
    valid_len: u64,
}

/// Decodes records until the bytes end or a torn final record is reached.
///
/// An undecodable record that is followed by more data cannot be a torn write,
/// so it is reported instead of being dropped.
fn scan_segment(path: &Path, bytes: &[u8]) -> Result<Scan, WalError> {
    let mut events = Vec::new();
    let mut offset = 0;
    while let Some((payload, remainder)) = split_record(&bytes[offset..]) {
        match LogEvent::decode(payload) {
            Ok(event) => events.push(event),
            Err(_) if remainder.is_empty() => break,
            Err(_) => {
                return Err(WalError::Corrupt {
                    segment: path.to_path_buf(),
                    offset: offset as u64,
                })
            }
        }
        offset = bytes.len() - remainder.len();
    }
    Ok(Scan {
        events,
        valid_len: offset as u64,
    })
}

/// Splits off the first record, or returns `None` when it is incomplete.
fn split_record(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let (prefix, body) = bytes.split_at_checked(LENGTH_PREFIX_BYTES)?;
    let length = u32::from_le_bytes(prefix.try_into().expect("prefix is 4 bytes")) as usize;
    body.split_at_checked(length)
}

/// Reads the log identity, creating it on first open.
///
/// The id is staged in a temp file and renamed so a crash never leaves a
/// half-written manifest that a later open would mistake for the identity.
fn load_or_create_log_id(dir: &Path) -> Result<Uuid, WalError> {
    let manifest = dir.join(MANIFEST_NAME);
    match fs::read_to_string(&manifest) {
        Ok(text) => {
            return Uuid::parse_str(text.trim()).map_err(|_| WalError::InvalidLogId { manifest })
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let log_id = Uuid::now_v7();
    let staged = dir.join(format!("{MANIFEST_NAME}.tmp"));
    let mut file = File::create(&staged)?;
    file.write_all(log_id.hyphenated().to_string().as_bytes())?;
    file.sync_all()?;
    fs::rename(&staged, &manifest)?;
    File::open(dir)?.sync_all()?;
    Ok(log_id)
}

fn segment_name(sequence: u64) -> String {
    format!("{sequence:020}.{SEGMENT_EXTENSION}")
}

/// Only names written by `segment_name` count, so a stray `9.wal` or `+3.wal` is ignored.
fn segment_sequence(path: &Path) -> Option<u64> {
    if path.extension()? != SEGMENT_EXTENSION {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    if stem.len() != SEQUENCE_DIGITS || !stem.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    stem.parse().ok()
}

/// Segment files with their sequence, in creation order.
fn segment_paths(dir: &Path) -> io::Result<Vec<(u64, PathBuf)>> {
    let mut segments = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if let Some(sequence) = segment_sequence(&path) {
            segments.push((sequence, path));
        }
    }
    segments.sort();
    Ok(segments)
}

fn create_segment(dir: &Path, sequence: u64, now: u64) -> io::Result<OpenSegment> {
    let file = OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(dir.join(segment_name(sequence)))?;
    // The new directory entry must be durable, not just the file contents.
    File::open(dir)?.sync_all()?;
    Ok(OpenSegment {
        file,
        sequence,
        size: 0,
        records: 0,
        opened_at: now,
    })
}

/// Reopens a segment, truncating a torn final record so the next append lands cleanly.
fn reopen_segment(path: &Path, sequence: u64, now: u64) -> Result<OpenSegment, WalError> {
    let file = OpenOptions::new().append(true).open(path)?;
    let on_disk = file.metadata()?.len();
    let scan = scan_segment(path, &fs::read(path)?)?;
    let valid = scan.valid_len;
    if valid < on_disk {
        tracing::warn!(
            segment = %path.display(),
            dropped_bytes = on_disk - valid,
            "truncating torn final wal record"
        );
        file.set_len(valid)?;
        file.sync_all()?;
    }
    Ok(OpenSegment {
        file,
        sequence,
        size: valid,
        records: scan.events.len() as u64,
        opened_at: now,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::fixtures::event;
    use crate::LogOffset;

    const INTERVAL_SECS: u64 = 60;
    const LARGE: u64 = 1 << 20;

    /// `Send`, which the `LogWriter` trait object requires.
    #[derive(Clone, Default)]
    struct FakeClock(Arc<AtomicU64>);

    impl FakeClock {
        fn advance(&self, secs: u64) {
            self.0.fetch_add(secs, Ordering::SeqCst);
        }
    }

    impl RotationClock for FakeClock {
        fn now_secs(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn config(segment_max_bytes: u64) -> WalConfig {
        WalConfig {
            segment_max_bytes,
            rotate_interval_secs: INTERVAL_SECS,
        }
    }

    fn frame(event: &LogEvent) -> Vec<u8> {
        let payload = event.encode().expect("encode event");
        let mut bytes = (payload.len() as u32).to_le_bytes().to_vec();
        bytes.extend(payload);
        bytes
    }

    fn frame_len(event: &LogEvent) -> u64 {
        frame(event).len() as u64
    }

    fn segment_files(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .expect("read log dir")
            .map(|entry| entry.expect("dir entry").path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "wal"))
            .collect();
        files.sort();
        files
    }

    fn open(dir: &Path, max_bytes: u64, clock: &FakeClock) -> WalWriter<FakeClock> {
        WalWriter::open(dir, config(max_bytes), clock.clone()).expect("open wal")
    }

    #[test]
    fn open_on_empty_directory_creates_one_segment_and_replays_nothing() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");

        // Act
        let _writer = open(dir.path(), LARGE, &FakeClock::default());
        let replayed = replay(dir.path()).expect("replay");

        // Assert
        assert_eq!(segment_files(dir.path()).len(), 1);
        assert!(replayed.is_empty());
    }

    #[test]
    fn single_event_is_replayed_unchanged_after_reopen() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let first = event(0);
        let mut writer = open(dir.path(), LARGE, &clock);
        writer.append(&first).expect("append");
        drop(writer);

        // Act
        let _reopened = open(dir.path(), LARGE, &clock);
        let replayed = replay(dir.path()).expect("replay");

        // Assert
        assert_eq!(replayed, vec![first]);
    }

    #[test]
    fn appended_events_are_replayed_in_order_after_reopen() {
        for count in [2, 10, 100] {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
            let events: Vec<LogEvent> = (0..count).map(event).collect();
            let mut writer = open(dir.path(), LARGE, &clock);
            for appended in &events {
                writer.append(appended).expect("append");
            }
            drop(writer);

            // Act
            let _reopened = open(dir.path(), LARGE, &clock);
            let replayed = replay(dir.path()).expect("replay");

            // Assert
            assert_eq!(replayed, events, "count {count}");
        }
    }

    #[test]
    fn append_returns_the_segment_and_record_index_it_was_written_at() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let mut writer = open(dir.path(), LARGE, &FakeClock::default());

        // Act
        let first_offset = writer.append(&event(0)).expect("append first");
        let second_offset = writer.append(&event(1)).expect("append second");

        // Assert
        assert_eq!(
            first_offset,
            LogOffset {
                segment: 0,
                index: 0
            }
        );
        assert_eq!(
            second_offset,
            LogOffset {
                segment: 0,
                index: 1
            }
        );
    }

    #[test]
    fn offsets_strictly_increase_across_rotations() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let max_bytes = frame_len(&event(0)) * 2;
        let mut writer = open(dir.path(), max_bytes, &FakeClock::default());

        // Act
        let offsets: Vec<LogOffset> = (0..5)
            .map(|index| writer.append(&event(index)).expect("append"))
            .collect();

        // Assert
        let expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
            .map(|(segment, index)| LogOffset { segment, index });
        assert_eq!(offsets, expected);
        assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn offset_after_reopen_continues_the_index_of_the_last_valid_record() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        write_events(dir.path(), &[event(0), event(1)], &clock);
        let segment = segment_files(dir.path()).remove(0);
        let torn = frame(&event(2));
        append_raw(&segment, &torn[..torn.len() / 2]);
        let mut writer = open(dir.path(), LARGE, &clock);

        // Act
        let offset = writer.append(&event(3)).expect("append after reopen");

        // Assert
        assert_eq!(
            offset,
            LogOffset {
                segment: 0,
                index: 2
            }
        );
    }

    #[test]
    fn record_is_a_little_endian_length_prefix_then_the_encoded_event() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let mut writer = open(dir.path(), LARGE, &FakeClock::default());
        let appended = event(0);
        let payload = appended.encode().expect("encode event");

        // Act
        writer.append(&appended).expect("append");
        let files = segment_files(dir.path());

        // Assert
        assert_eq!(files.len(), 1);
        let bytes = fs::read(&files[0]).expect("read segment");
        let (prefix, body) = bytes.split_at(LENGTH_PREFIX_BYTES);
        assert_eq!(prefix, (payload.len() as u32).to_le_bytes());
        assert_eq!(body, payload);
    }

    #[test]
    fn append_below_both_thresholds_does_not_rotate() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let appended = event(0);
        let mut writer = open(dir.path(), frame_len(&appended) + 1, &clock);
        clock.advance(INTERVAL_SECS - 1);

        // Act
        writer.append(&appended).expect("append");

        // Assert
        assert_eq!(segment_files(dir.path()).len(), 1);
        assert_eq!(replay(dir.path()).expect("replay"), vec![appended]);
    }

    #[test]
    fn append_reaching_the_size_limit_exactly_rotates() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let appended = event(0);
        let mut writer = open(dir.path(), frame_len(&appended), &FakeClock::default());

        // Act
        writer.append(&appended).expect("append");

        // Assert
        assert_eq!(segment_files(dir.path()).len(), 2);
    }

    #[test]
    fn append_crossing_the_size_limit_rotates_and_keeps_the_new_segment_appendable() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let (first, second) = (event(0), event(1));
        let mut writer = open(dir.path(), frame_len(&first) - 1, &FakeClock::default());

        // Act
        writer.append(&first).expect("append first");
        let files_after_rotation = segment_files(dir.path()).len();
        writer.append(&second).expect("append second");

        // Assert
        assert_eq!(files_after_rotation, 2);
        assert_eq!(replay(dir.path()).expect("replay"), vec![first, second]);
    }

    #[test]
    fn append_after_the_interval_elapses_rotates_on_time_alone() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (first, second) = (event(0), event(1));
        let mut writer = open(dir.path(), LARGE, &clock);
        writer.append(&first).expect("append first");
        let files_before = segment_files(dir.path()).len();
        clock.advance(INTERVAL_SECS + 1);

        // Act
        writer.append(&second).expect("append second");

        // Assert
        assert_eq!(files_before, 1);
        assert_eq!(segment_files(dir.path()).len(), 2);
        assert_eq!(replay(dir.path()).expect("replay"), vec![first, second]);
    }

    #[test]
    fn append_after_exactly_the_interval_rotates() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let mut writer = open(dir.path(), LARGE, &clock);
        clock.advance(INTERVAL_SECS);

        // Act
        writer.append(&event(0)).expect("append");

        // Assert
        assert_eq!(segment_files(dir.path()).len(), 2);
    }

    #[test]
    fn time_rotation_resets_the_timer_for_the_new_segment() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let mut writer = open(dir.path(), LARGE, &clock);
        clock.advance(INTERVAL_SECS);
        writer.append(&event(0)).expect("append rotates");

        // Act
        writer
            .append(&event(1))
            .expect("append with an unmoved clock");

        // Assert
        assert_eq!(segment_files(dir.path()).len(), 2);
    }

    #[test]
    fn reopen_counts_the_interval_from_the_reopen() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        write_events(dir.path(), &[event(0)], &clock);
        clock.advance(1_000);
        let mut writer = open(dir.path(), LARGE, &clock);

        // Act
        writer.append(&event(1)).expect("append at reopen time");
        let files_at_reopen = segment_files(dir.path()).len();
        clock.advance(INTERVAL_SECS);
        writer.append(&event(2)).expect("append after the interval");

        // Assert
        assert_eq!(files_at_reopen, 1);
        assert_eq!(segment_files(dir.path()).len(), 2);
    }

    #[test]
    fn events_are_replayed_in_order_across_rotated_segments() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let events: Vec<LogEvent> = (0..5).map(event).collect();
        let mut writer = open(dir.path(), frame_len(&events[0]), &FakeClock::default());

        // Act
        for appended in &events {
            writer.append(appended).expect("append");
        }
        drop(writer);

        // Assert
        assert_eq!(segment_files(dir.path()).len(), 6);
        assert_eq!(replay(dir.path()).expect("replay"), events);
    }

    /// Writes the events through a real writer, then drops it so the file is quiescent.
    fn write_events(dir: &Path, events: &[LogEvent], clock: &FakeClock) {
        let mut writer = open(dir, LARGE, clock);
        for appended in events {
            writer.append(appended).expect("append");
        }
    }

    fn append_raw(path: &Path, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open segment");
        file.write_all(bytes).expect("write raw bytes");
    }

    fn open_error(dir: &Path, clock: &FakeClock) -> WalError {
        WalWriter::open(dir, config(LARGE), clock.clone())
            .err()
            .expect("open should fail")
    }

    #[test]
    fn truncated_final_record_is_dropped_on_disk_and_the_next_append_follows_the_valid_prefix() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (first, second, torn, next) = (event(0), event(1), event(2), event(3));
        write_events(dir.path(), &[first.clone(), second.clone()], &clock);
        let segment = segment_files(dir.path()).remove(0);
        let torn_frame = frame(&torn);
        append_raw(&segment, &torn_frame[..torn_frame.len() / 2]);

        // Act
        let mut writer = open(dir.path(), LARGE, &clock);
        writer.append(&next).expect("append after recovery");
        drop(writer);
        let _reopened = open(dir.path(), LARGE, &clock);

        // Assert
        let expected: Vec<u8> = [&first, &second, &next]
            .into_iter()
            .flat_map(frame)
            .collect();
        assert_eq!(fs::read(&segment).expect("read segment"), expected);
        assert_eq!(
            replay(dir.path()).expect("replay"),
            vec![first, second, next]
        );
    }

    #[test]
    fn truncation_inside_the_length_prefix_or_before_the_payload_is_a_recoverable_tail() {
        // 1 to 3 bytes of the 4 byte prefix, then the full prefix with 0 payload bytes.
        for kept_bytes in [1, 2, 3, 4] {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
            let (valid, torn, next) = (event(0), event(1), event(2));
            write_events(dir.path(), std::slice::from_ref(&valid), &clock);
            let segment = segment_files(dir.path()).remove(0);
            append_raw(&segment, &frame(&torn)[..kept_bytes]);

            // Act
            let mut writer = open(dir.path(), LARGE, &clock);
            let size_after_open = fs::metadata(&segment).expect("segment metadata").len();
            writer.append(&next).expect("append after recovery");
            drop(writer);

            // Assert
            assert_eq!(
                size_after_open,
                frame_len(&valid),
                "kept {kept_bytes} bytes"
            );
            assert_eq!(
                replay(dir.path()).expect("replay"),
                vec![valid, next],
                "kept {kept_bytes} bytes"
            );
        }
    }

    #[test]
    fn length_prefix_claiming_more_bytes_than_remain_is_treated_as_a_truncated_tail() {
        const REMAINING: u32 = 10;
        // One byte more than remains, a plausible large record, and the largest prefix.
        for claimed in [REMAINING + 1, 1_000_000, u32::MAX] {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
            let (valid, next) = (event(0), event(1));
            write_events(dir.path(), std::slice::from_ref(&valid), &clock);
            let segment = segment_files(dir.path()).remove(0);
            let mut bogus = claimed.to_le_bytes().to_vec();
            bogus.extend([0xAB; REMAINING as usize]);
            append_raw(&segment, &bogus);

            // Act
            let mut writer = open(dir.path(), LARGE, &clock);
            let size_after_open = fs::metadata(&segment).expect("segment metadata").len();
            writer.append(&next).expect("append after recovery");
            drop(writer);

            // Assert
            assert_eq!(
                size_after_open,
                frame_len(&valid),
                "claimed {claimed} bytes"
            );
            assert_eq!(
                replay(dir.path()).expect("replay"),
                vec![valid, next],
                "claimed {claimed} bytes"
            );
        }
    }

    /// Writes three records, then overwrites the second record's payload.
    ///
    /// Returns the segment, its corrupted bytes, and the second record's offset.
    fn write_with_corrupt_second_record(
        dir: &Path,
        clock: &FakeClock,
    ) -> (PathBuf, Vec<u8>, usize) {
        let (first, second) = (event(0), event(1));
        write_events(dir, &[first.clone(), second.clone(), event(2)], clock);
        let segment = segment_files(dir).remove(0);
        let mut bytes = fs::read(&segment).expect("read segment");
        let corrupt_start = frame_len(&first) as usize;
        let payload_start = corrupt_start + LENGTH_PREFIX_BYTES;
        bytes[payload_start..corrupt_start + frame_len(&second) as usize].fill(0xFF);
        fs::write(&segment, &bytes).expect("write corrupted segment");
        (segment, bytes, corrupt_start)
    }

    #[test]
    fn corrupt_record_in_the_middle_fails_open_without_modifying_the_segment() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (segment, bytes, corrupt_start) = write_with_corrupt_second_record(dir.path(), &clock);

        // Act
        let error = open_error(dir.path(), &clock);

        // Assert
        assert!(
            matches!(&error, WalError::Corrupt { offset, .. } if *offset == corrupt_start as u64),
            "unexpected error: {error}"
        );
        assert_eq!(fs::read(&segment).expect("read segment"), bytes);
    }

    #[test]
    fn corrupt_record_in_the_middle_fails_replay_instead_of_being_skipped() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (_, _, corrupt_start) = write_with_corrupt_second_record(dir.path(), &clock);

        // Act
        let result = replay(dir.path());

        // Assert
        assert!(
            matches!(&result, Err(WalError::Corrupt { offset, .. }) if *offset == corrupt_start as u64),
            "unexpected result: {result:?}"
        );
    }

    /// What the filesystem looked like at the instant a notification arrived.
    #[derive(Debug, Clone, PartialEq)]
    struct Notification {
        path: PathBuf,
        size_when_notified: u64,
        successor_existed: bool,
    }

    #[derive(Clone, Default)]
    struct RecordingSink(Arc<Mutex<Vec<Notification>>>);

    impl RecordingSink {
        fn received(&self) -> Vec<Notification> {
            self.0.lock().expect("sink lock").clone()
        }

        fn paths(&self) -> Vec<PathBuf> {
            self.received().into_iter().map(|n| n.path).collect()
        }
    }

    impl SegmentSink for RecordingSink {
        fn segment_closed(&self, path: &Path) {
            let successor = path.with_file_name(segment_name(
                segment_sequence(path).expect("segment name") + 1,
            ));
            let notification = Notification {
                path: path.to_path_buf(),
                size_when_notified: fs::metadata(path).expect("closed segment metadata").len(),
                successor_existed: successor.exists(),
            };
            self.0.lock().expect("sink lock").push(notification);
        }
    }

    fn open_with_sink(
        dir: &Path,
        max_bytes: u64,
        clock: &FakeClock,
        sink: &RecordingSink,
    ) -> WalWriter<FakeClock> {
        open(dir, max_bytes, clock).with_sink(Box::new(sink.clone()))
    }

    fn manifest_id(dir: &Path) -> Uuid {
        let text = fs::read_to_string(dir.join(MANIFEST_NAME)).expect("read log.id");
        Uuid::parse_str(text.trim()).expect("log.id holds a uuid")
    }

    #[test]
    fn wal_writer_is_usable_as_a_log_writer_trait_object() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let first = event(0);
        let writer =
            WalWriter::open(dir.path(), config(LARGE), FakeClock::default()).expect("open wal");
        let mut writer: Box<dyn LogWriter> = Box::new(writer);

        // Act
        let offset = writer.append(&first).expect("append through the trait");
        let log_id = writer.log_id();

        // Assert
        assert_eq!(
            offset,
            LogOffset {
                segment: 0,
                index: 0
            }
        );
        assert_eq!(replay(dir.path()).expect("replay"), vec![first]);
        assert_eq!(log_id, manifest_id(dir.path()));
    }

    #[test]
    fn first_open_writes_a_non_nil_log_id_manifest() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");

        // Act
        let writer = open(dir.path(), LARGE, &FakeClock::default());

        // Assert
        assert!(dir.path().join(MANIFEST_NAME).is_file());
        assert!(!writer.log_id().is_nil());
        assert_eq!(writer.log_id(), manifest_id(dir.path()));
    }

    #[test]
    fn reopening_reads_the_same_log_id_and_leaves_the_manifest_untouched() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let first_id = open(dir.path(), LARGE, &clock).log_id();
        let manifest_before = fs::read(dir.path().join(MANIFEST_NAME)).expect("read log.id");

        // Act
        let second_id = open(dir.path(), LARGE, &clock).log_id();

        // Assert
        assert!(!first_id.is_nil());
        assert_eq!(second_id, first_id);
        assert_eq!(
            fs::read(dir.path().join(MANIFEST_NAME)).expect("read log.id"),
            manifest_before
        );
    }

    #[test]
    fn separate_log_directories_get_distinct_ids() {
        // Arrange
        let (first_dir, second_dir) = (
            tempfile::tempdir().expect("tempdir"),
            tempfile::tempdir().expect("tempdir"),
        );
        let clock = FakeClock::default();

        // Act
        let first_id = open(first_dir.path(), LARGE, &clock).log_id();
        let second_id = open(second_dir.path(), LARGE, &clock).log_id();

        // Assert
        assert_ne!(first_id, second_id);
    }

    #[test]
    fn size_rotation_notifies_the_sink_only_after_the_segment_is_closed() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let sink = RecordingSink::default();
        let appended = event(0);
        let mut writer = open_with_sink(
            dir.path(),
            frame_len(&appended),
            &FakeClock::default(),
            &sink,
        );
        let closed = segment_files(dir.path()).remove(0);

        // Act
        writer.append(&appended).expect("append");

        // Assert
        assert_eq!(
            sink.received(),
            vec![Notification {
                path: closed,
                size_when_notified: frame_len(&appended),
                successor_existed: true,
            }]
        );
    }

    #[test]
    fn time_rotation_notifies_the_sink_with_the_closed_segment() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let sink = RecordingSink::default();
        let mut writer = open_with_sink(dir.path(), LARGE, &clock, &sink);
        let closed = segment_files(dir.path()).remove(0);
        writer.append(&event(0)).expect("append first");
        clock.advance(INTERVAL_SECS + 1);

        // Act
        writer.append(&event(1)).expect("append second");

        // Assert
        assert_eq!(sink.paths(), vec![closed]);
    }

    #[test]
    fn below_threshold_append_sends_nothing_until_a_rotation_delivers_a_notification() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let sink = RecordingSink::default();
        let (first, second) = (event(0), event(1));
        let mut writer = open_with_sink(dir.path(), frame_len(&first) + 1, &clock, &sink);
        let first_segment = segment_files(dir.path()).remove(0);

        // Act
        writer.append(&first).expect("append below threshold");
        let notified_before_rotation = sink.received().len();
        writer.append(&second).expect("append crossing threshold");

        // Assert
        assert_eq!(notified_before_rotation, 0);
        assert_eq!(sink.paths(), vec![first_segment]);
    }

    #[test]
    fn the_open_segment_is_never_notified_and_stays_untouched_after_a_notification() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let sink = RecordingSink::default();
        let appended = event(0);
        let mut writer = open_with_sink(
            dir.path(),
            frame_len(&appended),
            &FakeClock::default(),
            &sink,
        );
        writer.append(&appended).expect("append triggers rotation");

        // Act
        let files = segment_files(dir.path());
        let open_segment = files.last().expect("open segment").clone();

        // Assert
        assert_eq!(sink.paths(), vec![files[0].clone()], "positive control");
        assert!(!sink.paths().contains(&open_segment));
        assert_eq!(fs::metadata(&open_segment).expect("open segment").len(), 0);
    }

    #[test]
    fn every_rotation_notifies_each_closed_segment_once_in_order() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let sink = RecordingSink::default();
        let events: Vec<LogEvent> = (0..4).map(event).collect();
        let mut writer = open_with_sink(
            dir.path(),
            frame_len(&events[0]),
            &FakeClock::default(),
            &sink,
        );

        // Act
        for appended in &events {
            writer.append(appended).expect("append");
        }

        // Assert
        let files = segment_files(dir.path());
        assert_eq!(files.len(), 5);
        assert_eq!(sink.paths(), files[..4].to_vec());
    }

    #[test]
    fn unparseable_log_id_manifest_fails_open_with_invalid_log_id() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join(MANIFEST_NAME);
        fs::write(&manifest, "not-a-uuid").expect("write bad manifest");

        // Act
        let error = open_error(dir.path(), &FakeClock::default());

        // Assert
        assert!(
            matches!(&error, WalError::InvalidLogId { manifest: path } if *path == manifest),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn twelve_rotated_segments_replay_in_order_and_reopen_appends_to_the_newest() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let events: Vec<LogEvent> = (0..12).map(event).collect();
        let mut writer = open(dir.path(), frame_len(&events[0]), &clock);
        for appended in &events {
            writer.append(appended).expect("append");
        }
        drop(writer);
        fs::write(dir.path().join("9.wal"), b"stray").expect("write stray");
        fs::write(dir.path().join("+3.wal"), b"stray").expect("write stray");
        let segments_before = segment_paths(dir.path()).expect("list segments");

        // Act
        let mut reopened = WalWriter::open(dir.path(), config(LARGE), clock).expect("reopen");
        let offset = reopened.append(&event(12)).expect("append after reopen");
        drop(reopened);

        // Assert
        let segments = segment_paths(dir.path()).expect("list segments");
        let (newest_sequence, newest) = segments.last().expect("newest segment");
        assert_eq!(segments.len(), 13);
        assert_eq!(segments, segments_before);
        assert_eq!(*newest_sequence, 12);
        assert_eq!(
            newest.file_name().expect("file name"),
            segment_name(12).as_str()
        );
        assert_eq!(
            offset,
            LogOffset {
                segment: 12,
                index: 0
            }
        );
        assert_eq!(fs::read(newest).expect("read newest"), frame(&event(12)));
        let mut expected = events;
        expected.push(event(12));
        assert_eq!(replay(dir.path()).expect("replay"), expected);
    }

    #[test]
    fn segment_sequence_accepts_only_twenty_digit_stems() {
        // Arrange
        let accepted = Path::new("00000000000000000012.wal");
        let rejected = [
            "9.wal",
            "+3.wal",
            "0000000000000000001a.wal",
            "00000000000000000012.tmp",
        ];

        // Act
        let parsed = segment_sequence(accepted);

        // Assert
        assert_eq!(parsed, Some(12));
        for name in rejected {
            assert_eq!(segment_sequence(Path::new(name)), None, "{name}");
        }
    }

    #[test]
    fn record_length_accepts_the_largest_prefix_value() {
        assert_eq!(record_length(u32::MAX as usize).expect("fits"), u32::MAX);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn record_length_rejects_one_byte_past_the_prefix_range() {
        // Arrange
        let too_large = u32::MAX as usize + 1;

        // Act
        let result = record_length(too_large);

        // Assert
        assert!(
            matches!(result, Err(WalError::RecordTooLarge(len)) if len == too_large),
            "unexpected result: {result:?}"
        );
    }

    #[test]
    fn system_clock_reports_seconds_since_the_unix_epoch() {
        // Arrange
        const SEPTEMBER_2020: u64 = 1_600_000_000;

        // Act
        let now = SystemClock.now_secs();

        // Assert
        assert!(now > SEPTEMBER_2020, "now_secs was {now}");
    }

    #[test]
    fn open_with_system_clock_creates_a_usable_writer() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");

        // Act
        let mut writer =
            WalWriter::open_with_system_clock(dir.path(), config(LARGE)).expect("open wal");
        writer.append(&event(0)).expect("append");

        // Assert
        assert_eq!(replay(dir.path()).expect("replay"), vec![event(0)]);
    }
}
