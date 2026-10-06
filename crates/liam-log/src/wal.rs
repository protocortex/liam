// SPDX-License-Identifier: Apache-2.0

//! Write-ahead log segments: checksummed postcard records, fsynced per append.
//!
//! A record is `[len u32 LE][header_crc u32 LE][payload_crc u32 LE][payload]`.
//! `header_crc` covers the 4 length bytes, so a damaged length is told apart
//! from a torn payload; `payload_crc` covers the payload. On open, damage at
//! the final position is a torn tail and is truncated away, unless a complete
//! valid frame follows it: then the damage is mid-file and fails the open with
//! the file untouched. A record that checks out but cannot be decoded also
//! fails the open and is never truncated.
//!
//! The writer owns rotation so a segment is only ever read by others once it
//! is closed; callers serialize appends themselves. Segments closed before a
//! crash are not announced again after a reopen: the compactor recovers them
//! by scanning the log directory when it starts.
//!
//! Known limitation: the time rotation timer is not persisted, so it restarts
//! on every reopen.

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
const CRC_BYTES: usize = 4;
const HEADER_BYTES: usize = LENGTH_PREFIX_BYTES + 2 * CRC_BYTES;
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
    #[error("wal segment {} has an undecodable record at offset {offset}: {source}", segment.display())]
    Decode {
        segment: PathBuf,
        offset: u64,
        source: EventError,
    },
    #[error("wal segment {} already exists and is not empty", segment.display())]
    SegmentNotEmpty { segment: PathBuf },
    #[error("wal writer is poisoned by an earlier failed append and must be reopened")]
    Poisoned,
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
///
/// A segment closed before a crash is not announced again after a reopen; a
/// consumer must scan the log directory on startup to find those.
pub trait SegmentSink: Send + Sync {
    fn segment_closed(&self, path: &Path);
}

/// Thresholds that close the open segment and start the next one.
#[derive(Debug, Clone, Copy)]
pub struct WalConfig {
    pub segment_max_bytes: u64,
    pub rotate_interval_secs: u64,
}

/// The disk operations a writer performs, so tests can inject failures.
pub(crate) trait FileOps: Send {
    fn write_all(&mut self, file: &mut File, bytes: &[u8]) -> io::Result<()>;
    fn sync_file(&mut self, file: &File) -> io::Result<()>;
    fn truncate(&mut self, file: &File, len: u64) -> io::Result<()>;
    fn sync_dir(&mut self, dir: &Path) -> io::Result<()>;
}

pub(crate) struct OsFileOps;

impl FileOps for OsFileOps {
    fn write_all(&mut self, file: &mut File, bytes: &[u8]) -> io::Result<()> {
        file.write_all(bytes)
    }

    fn sync_file(&mut self, file: &File) -> io::Result<()> {
        file.sync_all()
    }

    fn truncate(&mut self, file: &File, len: u64) -> io::Result<()> {
        file.set_len(len)
    }

    fn sync_dir(&mut self, dir: &Path) -> io::Result<()> {
        File::open(dir)?.sync_all()
    }
}

/// Writes `bytes` to a temp file beside `path` and renames it into place, so a
/// crash leaves the old file or the new one, never a partial write.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8], ops: &mut dyn FileOps) -> io::Result<()> {
    let mut staged = path.as_os_str().to_owned();
    staged.push(".tmp");
    let staged = PathBuf::from(staged);
    let mut file = File::create(&staged)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&staged, path)?;
    ops.sync_dir(sync_dir_for(path))
}

/// The directory whose entry for `path` must reach disk. A bare file name has
/// an empty parent, which means the current directory.
fn sync_dir_for(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
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
    ops: Box<dyn FileOps>,
    /// Set when the file may hold bytes the writer cannot account for.
    failed: bool,
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
        Self::open_with_file_ops(dir, config, clock, OsFileOps)
    }

    fn open_with_file_ops(
        dir: &Path,
        config: WalConfig,
        clock: C,
        ops: impl FileOps + 'static,
    ) -> Result<Self, WalError> {
        let mut ops: Box<dyn FileOps> = Box::new(ops);
        create_dir_all_durably(dir, ops.as_mut())?;
        let log_id = load_or_create_log_id(dir, ops.as_mut())?;
        let segment = match segment_paths(dir)?.last() {
            Some((sequence, path)) => {
                reopen_segment(path, *sequence, clock.now_secs(), ops.as_mut())?
            }
            None => create_segment(dir, 0, clock.now_secs(), ops.as_mut())?,
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            clock,
            log_id,
            segment,
            sink: None,
            ops,
            failed: false,
        })
    }

    #[cfg(test)]
    fn with_file_ops(mut self, ops: impl FileOps + 'static) -> Self {
        self.ops = Box::new(ops);
        self
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
        self.segment = create_segment(
            &self.dir,
            self.segment.sequence + 1,
            self.clock.now_secs(),
            self.ops.as_mut(),
        )?;
        // Every append is already fsynced, so the closed segment is complete here.
        if let Some(sink) = &self.sink {
            sink.segment_closed(&closed);
        }
        Ok(())
    }

    /// Writes and fsyncs one framed record, undoing a failed attempt.
    ///
    /// The writer is poisoned when the file may keep bytes it cannot account
    /// for: the rollback failed, or fsync failed and left the data's fate unknown.
    fn write_durably(&mut self, record: &[u8]) -> io::Result<()> {
        let (error, sync_failed) = match self.ops.write_all(&mut self.segment.file, record) {
            Err(error) => (error, false),
            Ok(()) => match self.ops.sync_file(&self.segment.file) {
                Ok(()) => return Ok(()),
                Err(error) => (error, true),
            },
        };
        tracing::warn!(segment = self.segment.sequence, %error, "wal append failed, rolling back");
        let rolled_back = self.rollback();
        if let Err(rollback_error) = &rolled_back {
            tracing::warn!(segment = self.segment.sequence, error = %rollback_error, "wal rollback failed");
        }
        self.failed = sync_failed || rolled_back.is_err();
        Err(error)
    }

    fn rollback(&mut self) -> io::Result<()> {
        self.ops.truncate(&self.segment.file, self.segment.size)?;
        self.ops.sync_file(&self.segment.file)
    }
}

impl<C: RotationClock + Send> LogWriter for WalWriter<C> {
    fn append(&mut self, event: &LogEvent) -> Result<LogOffset, WalError> {
        if self.failed {
            return Err(WalError::Poisoned);
        }
        let payload = event.encode()?;
        let length = record_length(payload.len())?;
        let record = encode_frame(length, &payload);

        let offset = LogOffset {
            segment: self.segment.sequence,
            index: self.segment.records,
        };
        self.write_durably(&record)?;
        self.segment.size += record.len() as u64;
        self.segment.records += 1;

        // The record is durable, so a rotation failure must not fail the append;
        // the next append retries because the thresholds still hold.
        if self.should_rotate() {
            if let Err(error) = self.rotate() {
                tracing::warn!(segment = self.segment.sequence, %error, "wal rotation failed, will retry on the next append");
            }
        }
        Ok(offset)
    }

    /// Identifies this log; stable across reopens of the same directory.
    fn log_id(&self) -> Uuid {
        self.log_id
    }
}

fn encode_frame(length: u32, payload: &[u8]) -> Vec<u8> {
    let length_bytes = length.to_le_bytes();
    let mut frame = Vec::with_capacity(HEADER_BYTES + payload.len());
    frame.extend_from_slice(&length_bytes);
    frame.extend_from_slice(&crc32fast::hash(&length_bytes).to_le_bytes());
    frame.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
    frame.extend_from_slice(payload);
    frame
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
    /// Length of the prefix made of complete, valid records.
    valid_len: u64,
}

/// Decodes records until the bytes end or a torn final record is reached.
///
/// Damage followed by more data cannot be a torn write, so it is reported
/// instead of dropped. A record whose checksums hold but whose payload does
/// not decode is also reported: it is valid data this build cannot read, such
/// as an event from a newer schema, and must never be truncated.
fn scan_segment(path: &Path, bytes: &[u8]) -> Result<Scan, WalError> {
    let mut events = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let rest = &bytes[offset..];
        match read_frame(rest) {
            Frame::Valid { payload, len } => {
                let event = LogEvent::decode(payload).map_err(|source| WalError::Decode {
                    segment: path.to_path_buf(),
                    offset: offset as u64,
                    source,
                })?;
                events.push(event);
                offset += len;
            }
            Frame::TornPayload => break,
            Frame::BadPayloadChecksum { len } if len == rest.len() => break,
            Frame::BadHeader if is_torn_tail(rest) => break,
            Frame::BadPayloadChecksum { .. } | Frame::BadHeader => {
                return Err(WalError::Corrupt {
                    segment: path.to_path_buf(),
                    offset: offset as u64,
                })
            }
        }
    }
    Ok(Scan {
        events,
        valid_len: offset as u64,
    })
}

/// Whether a damaged header at the start of `rest` is a torn tail: zero fill,
/// or no complete valid frame anywhere after it.
fn is_torn_tail(rest: &[u8]) -> bool {
    rest.iter().all(|byte| *byte == 0)
        || !(1..rest.len()).any(|start| matches!(read_frame(&rest[start..]), Frame::Valid { .. }))
}

enum Frame<'a> {
    Valid {
        payload: &'a [u8],
        len: usize,
    },
    /// The header is intact but claims more payload than the bytes hold.
    TornPayload,
    /// The header is incomplete or fails its checksum, so its length is untrusted.
    BadHeader,
    BadPayloadChecksum {
        len: usize,
    },
}

/// Classifies the record at the start of `bytes`.
fn read_frame(bytes: &[u8]) -> Frame<'_> {
    let Some((header, body)) = bytes.split_at_checked(HEADER_BYTES) else {
        return Frame::BadHeader;
    };
    let (prefix, checksums) = header.split_at(LENGTH_PREFIX_BYTES);
    let (header_crc, payload_crc) = checksums.split_at(CRC_BYTES);
    if crc32fast::hash(prefix).to_le_bytes() != header_crc {
        return Frame::BadHeader;
    }
    let length = u32::from_le_bytes(prefix.try_into().expect("prefix is 4 bytes")) as usize;
    let Some((payload, _)) = body.split_at_checked(length) else {
        return Frame::TornPayload;
    };
    let len = HEADER_BYTES + length;
    if crc32fast::hash(payload).to_le_bytes() == payload_crc {
        Frame::Valid { payload, len }
    } else {
        Frame::BadPayloadChecksum { len }
    }
}

/// Reads the log identity, creating it on first open.
///
/// The id is staged in a temp file and renamed so a crash never leaves a
/// half-written manifest that a later open would mistake for the identity.
fn load_or_create_log_id(dir: &Path, ops: &mut dyn FileOps) -> Result<Uuid, WalError> {
    let manifest = dir.join(MANIFEST_NAME);
    match fs::read_to_string(&manifest) {
        Ok(text) => {
            return Uuid::parse_str(text.trim()).map_err(|_| WalError::InvalidLogId { manifest })
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let log_id = Uuid::now_v7();
    write_atomically(&manifest, log_id.hyphenated().to_string().as_bytes(), ops)?;
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

/// Creates the directory and fsyncs every directory it newly created plus the
/// first pre-existing ancestor, so no new entry is lost to a crash.
fn create_dir_all_durably(dir: &Path, ops: &mut dyn FileOps) -> io::Result<()> {
    let created: Vec<&Path> = dir
        .ancestors()
        .take_while(|ancestor| !ancestor.as_os_str().is_empty() && !ancestor.exists())
        .collect();
    let Some(topmost) = created.last() else {
        return Ok(());
    };
    fs::create_dir_all(dir)?;
    for path in &created {
        ops.sync_dir(path)?;
    }
    match topmost.parent() {
        None => Ok(()),
        // A bare relative name has an empty parent, which means the current directory.
        Some(parent) if parent.as_os_str().is_empty() => ops.sync_dir(Path::new(".")),
        Some(parent) => ops.sync_dir(parent),
    }
}

/// Creates the segment, adopting an empty file left by an earlier failed attempt.
fn create_segment(
    dir: &Path,
    sequence: u64,
    now: u64,
    ops: &mut dyn FileOps,
) -> Result<OpenSegment, WalError> {
    let path = dir.join(segment_name(sequence));
    let file = match OpenOptions::new().create_new(true).append(true).open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let file = OpenOptions::new().append(true).open(&path)?;
            if file.metadata()?.len() != 0 {
                return Err(WalError::SegmentNotEmpty { segment: path });
            }
            file
        }
        Err(error) => return Err(error.into()),
    };
    // The new directory entry must be durable, not just the file contents.
    ops.sync_dir(dir)?;
    Ok(OpenSegment {
        file,
        sequence,
        size: 0,
        records: 0,
        opened_at: now,
    })
}

/// Reopens a segment, truncating a torn final record so the next append lands cleanly.
fn reopen_segment(
    path: &Path,
    sequence: u64,
    now: u64,
    ops: &mut dyn FileOps,
) -> Result<OpenSegment, WalError> {
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
        ops.truncate(&file, valid)?;
        ops.sync_file(&file)?;
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
    use crate::event::CURRENT_SCHEMA_VERSION;
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

    fn frame_payload(payload: &[u8]) -> Vec<u8> {
        let length = (payload.len() as u32).to_le_bytes();
        let mut bytes = length.to_vec();
        bytes.extend(crc32fast::hash(&length).to_le_bytes());
        bytes.extend(crc32fast::hash(payload).to_le_bytes());
        bytes.extend(payload);
        bytes
    }

    fn frame(event: &LogEvent) -> Vec<u8> {
        frame_payload(&event.encode().expect("encode event"))
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
    fn record_is_a_length_prefix_then_a_header_crc_then_a_payload_crc_then_the_encoded_event() {
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
        let (prefix, rest) = bytes.split_at(LENGTH_PREFIX_BYTES);
        let (header_crc, rest) = rest.split_at(CRC_BYTES);
        let (payload_crc, body) = rest.split_at(CRC_BYTES);
        assert_eq!(HEADER_BYTES, 12);
        assert_eq!(prefix, (payload.len() as u32).to_le_bytes());
        assert_eq!(header_crc, crc32fast::hash(prefix).to_le_bytes());
        assert_eq!(payload_crc, crc32fast::hash(&payload).to_le_bytes());
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
    fn every_cut_point_of_a_torn_final_record_is_a_recoverable_tail() {
        let (valid, torn, next) = (event(0), event(1), event(2));
        for kept_bytes in 1..frame(&torn).len() {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
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
                vec![valid.clone(), next.clone()],
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
            let prefix = claimed.to_le_bytes();
            let mut bogus = prefix.to_vec();
            bogus.extend(crc32fast::hash(&prefix).to_le_bytes());
            bogus.extend(0xDEAD_BEEF_u32.to_le_bytes());
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
        let payload_start = corrupt_start + HEADER_BYTES;
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

    #[test]
    fn bad_checksum_on_the_last_record_is_a_torn_tail_that_is_truncated() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (first, last, next) = (event(0), event(1), event(2));
        write_events(dir.path(), &[first.clone(), last], &clock);
        let segment = segment_files(dir.path()).remove(0);
        let mut bytes = fs::read(&segment).expect("read segment");
        let last_payload_byte = bytes.len() - 1;
        bytes[last_payload_byte] ^= 0xFF;
        fs::write(&segment, &bytes).expect("write corrupted segment");

        // Act
        let mut writer = open(dir.path(), LARGE, &clock);
        let size_after_open = fs::metadata(&segment).expect("segment metadata").len();
        writer.append(&next).expect("append after recovery");
        drop(writer);

        // Assert
        assert_eq!(size_after_open, frame_len(&first));
        assert_eq!(replay(dir.path()).expect("replay"), vec![first, next]);
    }

    #[test]
    fn zero_filled_tails_are_a_torn_tail_that_is_truncated() {
        // A header cut short, a whole zeroed header, then more zeros.
        for zero_bytes in [4, 8, HEADER_BYTES, 64] {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
            let (valid, next) = (event(0), event(1));
            write_events(dir.path(), std::slice::from_ref(&valid), &clock);
            let segment = segment_files(dir.path()).remove(0);
            append_raw(&segment, &vec![0; zero_bytes]);

            // Act
            let mut writer = open(dir.path(), LARGE, &clock);
            let size_after_open = fs::metadata(&segment).expect("segment metadata").len();
            writer.append(&next).expect("append after recovery");
            drop(writer);

            // Assert
            assert_eq!(
                size_after_open,
                frame_len(&valid),
                "{zero_bytes} zero bytes"
            );
            assert_eq!(
                replay(dir.path()).expect("replay"),
                vec![valid, next],
                "{zero_bytes} zero bytes"
            );
        }
    }

    #[test]
    fn zeroed_header_followed_by_data_with_no_valid_frame_is_a_torn_tail() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (valid, next) = (event(0), event(1));
        write_events(dir.path(), std::slice::from_ref(&valid), &clock);
        let segment = segment_files(dir.path()).remove(0);
        let mut tail = vec![0; HEADER_BYTES];
        tail.extend([1, 2, 3]);
        append_raw(&segment, &tail);

        // Act
        let mut writer = open(dir.path(), LARGE, &clock);
        let size_after_open = fs::metadata(&segment).expect("segment metadata").len();
        writer.append(&next).expect("append after recovery");
        drop(writer);

        // Assert
        assert_eq!(size_after_open, frame_len(&valid));
        assert_eq!(replay(dir.path()).expect("replay"), vec![valid, next]);
    }

    /// Writes `events` and returns the segment, its bytes, and each record's start offset.
    fn write_segment(
        dir: &Path,
        events: &[LogEvent],
        clock: &FakeClock,
    ) -> (PathBuf, Vec<u8>, Vec<usize>) {
        write_events(dir, events, clock);
        let segment = segment_files(dir).remove(0);
        let starts = events
            .iter()
            .scan(0, |start, appended| {
                let current = *start;
                *start += frame(appended).len();
                Some(current)
            })
            .collect();
        let bytes = fs::read(&segment).expect("read segment");
        (segment, bytes, starts)
    }

    #[test]
    fn damaged_length_prefix_of_a_middle_record_is_corrupt_and_leaves_the_segment_unchanged() {
        let events: Vec<LogEvent> = (0..3).map(event).collect();
        let length = (frame(&events[1]).len() - HEADER_BYTES) as u32;
        // One byte down and up, a large claim, and the largest prefix.
        for damaged in [length - 1, length + 1, length + 1_000, u32::MAX] {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
            let (segment, mut bytes, starts) = write_segment(dir.path(), &events, &clock);
            bytes[starts[1]..starts[1] + LENGTH_PREFIX_BYTES]
                .copy_from_slice(&damaged.to_le_bytes());
            fs::write(&segment, &bytes).expect("write damaged segment");

            // Act
            let error = open_error(dir.path(), &clock);

            // Assert
            assert!(
                matches!(&error, WalError::Corrupt { offset, .. } if *offset == starts[1] as u64),
                "damaged {damaged}: unexpected error: {error}"
            );
            assert_eq!(fs::read(&segment).expect("read segment"), bytes);
        }
    }

    #[test]
    fn flipped_crc_bytes_of_the_final_record_are_a_torn_tail_that_is_truncated() {
        // The four header_crc bytes, then the four payload_crc bytes.
        for flipped in LENGTH_PREFIX_BYTES..HEADER_BYTES {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
            let (first, last, next) = (event(0), event(1), event(2));
            let (segment, mut bytes, starts) =
                write_segment(dir.path(), &[first.clone(), last], &clock);
            bytes[starts[1] + flipped] ^= 0xFF;
            fs::write(&segment, &bytes).expect("write damaged segment");

            // Act
            let mut writer = open(dir.path(), LARGE, &clock);
            let size_after_open = fs::metadata(&segment).expect("segment metadata").len();
            writer.append(&next).expect("append after recovery");
            drop(writer);

            // Assert
            assert_eq!(size_after_open, frame_len(&first), "byte {flipped}");
            assert_eq!(
                replay(dir.path()).expect("replay"),
                vec![first, next],
                "byte {flipped}"
            );
        }
    }

    #[test]
    fn flipped_crc_bytes_of_a_middle_record_are_corrupt_and_leave_the_segment_unchanged() {
        for flipped in LENGTH_PREFIX_BYTES..HEADER_BYTES {
            // Arrange
            let dir = tempfile::tempdir().expect("tempdir");
            let clock = FakeClock::default();
            let events: Vec<LogEvent> = (0..3).map(event).collect();
            let (segment, mut bytes, starts) = write_segment(dir.path(), &events, &clock);
            bytes[starts[1] + flipped] ^= 0xFF;
            fs::write(&segment, &bytes).expect("write damaged segment");

            // Act
            let error = open_error(dir.path(), &clock);

            // Assert
            assert!(
                matches!(&error, WalError::Corrupt { offset, .. } if *offset == starts[1] as u64),
                "byte {flipped}: unexpected error: {error}"
            );
            assert_eq!(fs::read(&segment).expect("read segment"), bytes);
        }
    }

    #[test]
    fn valid_checksum_with_a_newer_schema_version_fails_open_and_keeps_the_frame() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        write_events(dir.path(), &[event(0)], &clock);
        let segment = segment_files(dir.path()).remove(0);
        let mut payload = event(1).encode().expect("encode event");
        payload[0] = u8::try_from(CURRENT_SCHEMA_VERSION + 1).expect("one byte version");
        append_raw(&segment, &frame_payload(&payload));
        let bytes = fs::read(&segment).expect("read segment");

        // Act
        let error = open_error(dir.path(), &clock);

        // Assert
        assert!(
            matches!(
                &error,
                WalError::Decode {
                    source: EventError::UnsupportedSchemaVersion { .. },
                    offset,
                    ..
                } if *offset == frame_len(&event(0))
            ),
            "unexpected error: {error}"
        );
        assert_eq!(fs::read(&segment).expect("read segment"), bytes);
    }

    #[test]
    fn valid_checksum_over_an_undecodable_final_payload_fails_open_and_keeps_the_frame() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        write_events(dir.path(), &[event(0)], &clock);
        let segment = segment_files(dir.path()).remove(0);
        append_raw(&segment, &frame_payload(&[0xFF; 5]));
        let bytes = fs::read(&segment).expect("read segment");

        // Act
        let error = open_error(dir.path(), &clock);

        // Assert
        assert!(
            matches!(&error, WalError::Decode { offset, .. } if *offset == frame_len(&event(0))),
            "unexpected error: {error}"
        );
        assert_eq!(fs::read(&segment).expect("read segment"), bytes);
    }

    /// Fails the next call of each armed operation exactly once, then behaves normally.
    #[derive(Default)]
    struct FaultyOps {
        /// Bytes that reach the file before the next write fails.
        partial_write: Option<usize>,
        fail_sync: bool,
        fail_truncate: bool,
        fail_dir_sync: bool,
    }

    fn injected(operation: &str) -> io::Error {
        io::Error::other(format!("injected {operation} failure"))
    }

    impl FileOps for FaultyOps {
        fn write_all(&mut self, file: &mut File, bytes: &[u8]) -> io::Result<()> {
            match self.partial_write.take() {
                Some(kept) => {
                    file.write_all(&bytes[..kept])?;
                    Err(injected("write"))
                }
                None => file.write_all(bytes),
            }
        }

        fn sync_file(&mut self, file: &File) -> io::Result<()> {
            if std::mem::take(&mut self.fail_sync) {
                return Err(injected("sync"));
            }
            file.sync_all()
        }

        fn truncate(&mut self, file: &File, len: u64) -> io::Result<()> {
            if std::mem::take(&mut self.fail_truncate) {
                return Err(injected("truncate"));
            }
            file.set_len(len)
        }

        fn sync_dir(&mut self, dir: &Path) -> io::Result<()> {
            if std::mem::take(&mut self.fail_dir_sync) {
                return Err(injected("directory sync"));
            }
            File::open(dir)?.sync_all()
        }
    }

    #[test]
    fn failed_write_is_rolled_back_so_the_next_append_lands_at_the_correct_offset() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (first, lost, next) = (event(0), event(1), event(2));
        let mut writer = open(dir.path(), LARGE, &clock);
        writer.append(&first).expect("append first");
        writer.ops = Box::new(FaultyOps {
            partial_write: Some(HEADER_BYTES + 3),
            ..FaultyOps::default()
        });

        // Act
        let failure = writer.append(&lost).expect_err("write should fail");
        let offset = writer.append(&next).expect("append after rollback");
        drop(writer);

        // Assert
        assert!(matches!(failure, WalError::Io(_)), "unexpected: {failure}");
        assert_eq!(
            offset,
            LogOffset {
                segment: 0,
                index: 1
            }
        );
        let segment = segment_files(dir.path()).remove(0);
        let expected: Vec<u8> = [&first, &next].into_iter().flat_map(frame).collect();
        assert_eq!(fs::read(&segment).expect("read segment"), expected);
    }

    #[test]
    fn failed_write_with_a_failed_rollback_poisons_the_writer_until_reopen_recovers_the_tail() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (first, lost, next) = (event(0), event(1), event(2));
        let mut writer = open(dir.path(), LARGE, &clock);
        writer.append(&first).expect("append first");
        writer.ops = Box::new(FaultyOps {
            partial_write: Some(HEADER_BYTES + 3),
            fail_truncate: true,
            ..FaultyOps::default()
        });

        // Act
        let failure = writer.append(&lost).expect_err("write should fail");
        let blocked = writer.append(&next).expect_err("poisoned append");
        drop(writer);
        let mut reopened = open(dir.path(), LARGE, &clock);
        let offset = reopened.append(&next).expect("append after reopen");

        // Assert
        assert!(matches!(failure, WalError::Io(_)), "unexpected: {failure}");
        assert!(
            matches!(blocked, WalError::Poisoned),
            "unexpected: {blocked}"
        );
        assert_eq!(
            offset,
            LogOffset {
                segment: 0,
                index: 1
            }
        );
        assert_eq!(replay(dir.path()).expect("replay"), vec![first, next]);
    }

    #[test]
    fn failed_sync_poisons_the_writer_even_when_the_rollback_succeeds() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let mut writer = open(dir.path(), LARGE, &FakeClock::default()).with_file_ops(FaultyOps {
            fail_sync: true,
            ..FaultyOps::default()
        });

        // Act
        let failure = writer.append(&event(0)).expect_err("sync should fail");
        let blocked = writer.append(&event(1)).expect_err("poisoned append");

        // Assert
        assert!(matches!(failure, WalError::Io(_)), "unexpected: {failure}");
        assert!(
            matches!(blocked, WalError::Poisoned),
            "unexpected: {blocked}"
        );
        assert!(replay(dir.path()).expect("replay").is_empty());
    }

    #[test]
    fn rotation_adopts_a_pre_created_empty_successor_segment() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let (first, second) = (event(0), event(1));
        let mut writer = open(dir.path(), frame_len(&first), &FakeClock::default());
        let successor = dir.path().join(segment_name(1));
        File::create(&successor).expect("pre-create successor");

        // Act
        let rotating = writer.append(&first).expect("append that rotates");
        let adopted = writer
            .append(&second)
            .expect("append to the adopted segment");

        // Assert
        assert_eq!(
            rotating,
            LogOffset {
                segment: 0,
                index: 0
            }
        );
        assert_eq!(
            adopted,
            LogOffset {
                segment: 1,
                index: 0
            }
        );
        assert_eq!(
            fs::read(&successor).expect("read successor"),
            frame(&second)
        );
    }

    #[test]
    fn directory_sync_failure_during_rotation_still_acknowledges_the_append_and_retries_next() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let sink = RecordingSink::default();
        let events: Vec<LogEvent> = (0..3).map(event).collect();
        let mut writer = open_with_sink(
            dir.path(),
            frame_len(&events[0]),
            &FakeClock::default(),
            &sink,
        )
        .with_file_ops(FaultyOps {
            fail_dir_sync: true,
            ..FaultyOps::default()
        });

        // Act
        let first = writer
            .append(&events[0])
            .expect("append survives the failed rotation");
        let notified_after_failure = sink.received().len();
        let second = writer
            .append(&events[1])
            .expect("append retries the rotation");
        let third = writer
            .append(&events[2])
            .expect("append to the rotated segment");

        // Assert
        assert_eq!(
            first,
            LogOffset {
                segment: 0,
                index: 0
            }
        );
        assert_eq!(notified_after_failure, 0);
        assert_eq!(
            second,
            LogOffset {
                segment: 0,
                index: 1
            }
        );
        assert_eq!(
            third,
            LogOffset {
                segment: 1,
                index: 0
            }
        );
        let files = segment_files(dir.path());
        assert_eq!(sink.paths(), files[..2].to_vec());
        assert_eq!(replay(dir.path()).expect("replay"), events);
    }

    #[test]
    fn creating_a_segment_over_a_non_empty_file_is_an_error() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let existing = dir.path().join(segment_name(1));
        fs::write(&existing, b"data").expect("pre-create non-empty successor");

        // Act
        let result = create_segment(dir.path(), 1, 0, &mut OsFileOps);

        // Assert
        assert!(
            matches!(&result, Err(WalError::SegmentNotEmpty { segment }) if *segment == existing),
            "unexpected result: {:?}",
            result.err()
        );
        assert_eq!(fs::read(&existing).expect("read existing"), b"data");
    }

    #[test]
    fn rotation_onto_a_non_empty_successor_keeps_the_append_and_does_not_notify() {
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
        let foreign = dir.path().join(segment_name(1));
        fs::write(&foreign, b"data").expect("pre-create successor");

        // Act
        let offset = writer
            .append(&appended)
            .expect("append survives failed rotation");
        let next_offset = writer
            .append(&event(1))
            .expect("append stays in the unrotated segment");

        // Assert
        assert_eq!(
            offset,
            LogOffset {
                segment: 0,
                index: 0
            }
        );
        assert_eq!(
            next_offset,
            LogOffset {
                segment: 0,
                index: 1
            }
        );
        assert_eq!(fs::read(&foreign).expect("read foreign file"), b"data");
        assert!(sink.received().is_empty());
    }

    /// A file operation as seen by `RecordingOps`.
    #[derive(Debug, Clone, PartialEq)]
    enum Op {
        /// `manifest_renamed` is whether `log.id` was in place, with no staged copy, in `dir`.
        SyncDir {
            dir: PathBuf,
            manifest_renamed: bool,
        },
        SyncFile,
        Truncate(u64),
    }

    /// Delegates to the real disk while recording the order of durability operations.
    #[derive(Clone, Default)]
    struct RecordingOps(Arc<Mutex<Vec<Op>>>);

    impl RecordingOps {
        fn recorded(&self) -> Vec<Op> {
            self.0.lock().expect("ops lock").clone()
        }

        fn push(&self, op: Op) {
            self.0.lock().expect("ops lock").push(op);
        }
    }

    impl FileOps for RecordingOps {
        fn write_all(&mut self, file: &mut File, bytes: &[u8]) -> io::Result<()> {
            file.write_all(bytes)
        }

        fn sync_file(&mut self, file: &File) -> io::Result<()> {
            self.push(Op::SyncFile);
            file.sync_all()
        }

        fn truncate(&mut self, file: &File, len: u64) -> io::Result<()> {
            self.push(Op::Truncate(len));
            file.set_len(len)
        }

        fn sync_dir(&mut self, dir: &Path) -> io::Result<()> {
            self.push(Op::SyncDir {
                dir: dir.to_path_buf(),
                manifest_renamed: dir.join(MANIFEST_NAME).exists()
                    && !dir.join(format!("{MANIFEST_NAME}.tmp")).exists(),
            });
            File::open(dir)?.sync_all()
        }
    }

    fn sync_dir_op(dir: &Path, manifest_renamed: bool) -> Op {
        Op::SyncDir {
            dir: dir.to_path_buf(),
            manifest_renamed,
        }
    }

    fn open_recording(dir: &Path) -> (WalWriter<FakeClock>, RecordingOps) {
        let ops = RecordingOps::default();
        let writer =
            WalWriter::open_with_file_ops(dir, config(LARGE), FakeClock::default(), ops.clone())
                .expect("open wal");
        (writer, ops)
    }

    #[test]
    fn open_syncs_the_log_directory_after_the_manifest_rename_and_after_creating_the_segment() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");

        // Act
        let (_writer, ops) = open_recording(dir.path());

        // Assert
        assert_eq!(
            ops.recorded(),
            vec![sync_dir_op(dir.path(), true), sync_dir_op(dir.path(), true)]
        );
    }

    #[test]
    fn open_on_a_nested_new_path_syncs_every_created_directory_and_the_first_existing_one() {
        // Arrange
        let root = tempfile::tempdir().expect("tempdir");
        let (a, b) = (root.path().join("a"), root.path().join("a").join("b"));
        let nested = b.join("log");

        // Act
        let (mut writer, ops) = open_recording(&nested);
        writer.append(&event(0)).expect("append");

        // Assert
        assert_eq!(
            ops.recorded()[..4],
            [
                sync_dir_op(&nested, false),
                sync_dir_op(&b, false),
                sync_dir_op(&a, false),
                sync_dir_op(root.path(), false),
            ]
        );
        assert_eq!(replay(&nested).expect("replay"), vec![event(0)]);
    }

    #[test]
    fn open_on_an_existing_directory_syncs_no_ancestor() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        write_events(dir.path(), &[event(0)], &clock);

        // Act
        let (_writer, ops) = open_recording(dir.path());

        // Assert
        assert!(ops.recorded().is_empty(), "{:?}", ops.recorded());
    }

    #[test]
    fn recovery_truncate_is_followed_by_a_file_sync() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (valid, torn) = (event(0), event(1));
        write_events(dir.path(), std::slice::from_ref(&valid), &clock);
        append_raw(
            &segment_files(dir.path()).remove(0),
            &frame(&torn)[..HEADER_BYTES + 2],
        );

        // Act
        let (_writer, ops) = open_recording(dir.path());

        // Assert
        assert_eq!(
            ops.recorded(),
            vec![Op::Truncate(frame_len(&valid)), Op::SyncFile]
        );
    }

    #[test]
    fn failed_recovery_truncate_fails_open_and_leaves_the_segment_unchanged() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        write_events(dir.path(), &[event(0)], &clock);
        let segment = segment_files(dir.path()).remove(0);
        append_raw(&segment, &frame(&event(1))[..HEADER_BYTES + 2]);
        let bytes = fs::read(&segment).expect("read segment");
        let faulty = FaultyOps {
            fail_truncate: true,
            ..FaultyOps::default()
        };

        // Act
        let result = WalWriter::open_with_file_ops(dir.path(), config(LARGE), clock, faulty);

        // Assert
        assert!(
            matches!(&result, Err(WalError::Io(error)) if error.to_string().contains("truncate")),
            "unexpected result: {:?}",
            result.err()
        );
        assert_eq!(fs::read(&segment).expect("read segment"), bytes);
    }

    #[test]
    fn append_with_failed_sync_and_failed_rollback_may_still_be_recovered_on_reopen() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let clock = FakeClock::default();
        let (first, unknown) = (event(0), event(1));
        let mut writer = open(dir.path(), LARGE, &clock);
        writer.append(&first).expect("append first");
        writer.ops = Box::new(FaultyOps {
            fail_sync: true,
            fail_truncate: true,
            ..FaultyOps::default()
        });

        // Act
        let failure = writer.append(&unknown).expect_err("sync should fail");
        let blocked = writer.append(&event(2)).expect_err("poisoned append");
        drop(writer);
        let _reopened = open(dir.path(), LARGE, &clock);

        // Assert
        assert!(matches!(failure, WalError::Io(_)), "unexpected: {failure}");
        assert!(
            matches!(blocked, WalError::Poisoned),
            "unexpected: {blocked}"
        );
        assert_eq!(
            replay(dir.path()).expect("replay"),
            vec![first, unknown],
            "an Err from a poisoned append does not mean the record is absent"
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
    fn a_bare_file_name_syncs_the_current_directory() {
        // Act
        let dir = sync_dir_for(Path::new("bloom.bin"));

        // Assert
        assert_eq!(dir, Path::new("."));
    }

    #[test]
    fn a_nested_path_syncs_its_parent_directory() {
        // Act
        let dir = sync_dir_for(Path::new("a/b.bin"));

        // Assert
        assert_eq!(dir, Path::new("a"));
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
