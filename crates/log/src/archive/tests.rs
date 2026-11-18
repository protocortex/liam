// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use futures_util::StreamExt;
use tempfile::TempDir;

use super::*;
use crate::crypto::{KeyRegistry, TestMaster};
use crate::envelope::{KeyHints, OpeningReader, Sealer};
use crate::event::LogEvent;
use crate::fixtures::{
    compact_segments, event, log_event, node_row, parquet_path, rotating_config,
};
use crate::reader::{LogReader, LogStream, ReaderError, SequentialScanReader};
use crate::wal::WalWriter;
use crate::LogWriter;

const AFTER_DAYS: u32 = 5;
const NOW: i64 = 100 * MILLIS_PER_DAY;
const POLICY: ArchivePolicy = ArchivePolicy {
    after_days: AFTER_DAYS,
    zstd_level: 3,
};
const PER_SEGMENT: usize = 2;

/// An event of the fixture log ingested `days_ago` days before `NOW`.
fn aged(index: usize, days_ago: i64) -> LogEvent {
    LogEvent {
        ingested_at: NOW - days_ago * MILLIS_PER_DAY,
        ..event(index)
    }
}

struct Env {
    dir: TempDir,
    archive: TempDir,
}

impl Env {
    fn new() -> Self {
        Self {
            dir: tempfile::tempdir().unwrap(),
            archive: tempfile::tempdir().unwrap(),
        }
    }

    fn log(&self) -> &Path {
        self.dir.path()
    }

    fn archive(&self) -> &Path {
        self.archive.path()
    }

    /// Appends `events`, two to a segment, and closes the writer. A partition
    /// of two events is compacted for every full segment.
    async fn write(&self, events: &[LogEvent]) -> Vec<(LogOffset, LogEvent)> {
        let mut writer =
            WalWriter::open_with_system_clock(self.log(), rotating_config(PER_SEGMENT)).unwrap();
        let logged: Vec<_> = events
            .iter()
            .map(|event| (writer.append(event).unwrap(), event.clone()))
            .collect();
        drop(writer);
        let closed: Vec<u64> = (0..logged.last().unwrap().0.segment).collect();
        compact_segments(self.log(), &closed).await;
        logged
    }

    fn run(&self) -> Result<ArchiveReport, ArchiveError> {
        self.run_with(&NoFault)
    }

    fn run_with(&self, fault: &dyn ArchiveFault) -> Result<ArchiveReport, ArchiveError> {
        archive_partitions(self.log(), self.archive(), POLICY, NOW, fault)
    }

    async fn scan(&self) -> (Vec<(LogOffset, LogEvent)>, Option<ReaderError>) {
        drain(SequentialScanReader::local(self.log()).unwrap().scan(None)).await
    }
}

async fn drain(mut stream: LogStream) -> (Vec<(LogOffset, LogEvent)>, Option<ReaderError>) {
    let mut records = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(record) => records.push((record.offset, record.event)),
            Err(error) => return (records, Some(error)),
        }
    }
    (records, None)
}

/// Six events: segments 0 and 1 are 10 and 5 days old, segment 2 is a day old,
/// and segment 3 is the open tail.
fn events_by_age() -> Vec<LogEvent> {
    [10, 10, 5, 5, 1, 1, 1]
        .iter()
        .enumerate()
        .map(|(index, days)| aged(index, *days))
        .collect()
}

fn at(segment: u64, index: u64) -> LogOffset {
    LogOffset { segment, index }
}

fn has_parquet(dir: &Path, sequence: u64) -> bool {
    parquet_path(dir, sequence).exists()
}

#[tokio::test]
async fn partitions_older_than_the_age_and_exactly_at_it_move_and_a_scan_still_returns_every_event()
{
    // Arrange
    let env = Env::new();
    let logged = env.write(&events_by_age()).await;

    // Act
    let report = env.run().unwrap();

    // Assert
    assert_eq!(report.archived, vec![0, 1]);
    assert!(!has_parquet(env.log(), 0) && !has_parquet(env.log(), 1));
    assert!(has_parquet(env.log(), 2), "a day old stays");
    assert!(has_parquet(env.archive(), 0) && has_parquet(env.archive(), 1));
    let manifest = read_manifest(env.log()).unwrap();
    let ranges: Vec<_> = manifest
        .iter()
        .map(|partition| (partition.first(), partition.last()))
        .collect();
    assert_eq!(
        ranges,
        vec![(at(0, 0), Some(at(0, 1))), (at(1, 0), Some(at(1, 1)))]
    );
    let (scanned, error) = env.scan().await;
    assert!(error.is_none(), "{error:?}");
    assert_eq!(scanned, logged);
}

#[tokio::test]
async fn a_second_run_moves_nothing_more() {
    // Arrange
    let env = Env::new();
    env.write(&events_by_age()).await;
    env.run().unwrap();

    // Act
    let report = env.run().unwrap();

    // Assert
    assert_eq!(report, ArchiveReport::default());
}

struct StopAt(ArchiveStep);

impl ArchiveFault for StopAt {
    fn reached(&self, step: ArchiveStep, _sequence: u64) -> io::Result<()> {
        if step == self.0 {
            return Err(io::Error::other("stopped"));
        }
        Ok(())
    }
}

#[tokio::test]
async fn a_run_stopped_after_placing_the_copy_is_finished_by_the_next() {
    for step in [
        ArchiveStep::Written,
        ArchiveStep::Placed,
        ArchiveStep::Verified,
        ArchiveStep::Recorded,
    ] {
        // Arrange
        let env = Env::new();
        let logged = env.write(&events_by_age()).await;
        env.run_with(&StopAt(step)).expect_err("the fault stops it");
        let (midway, error) = env.scan().await;
        assert!(error.is_none(), "{step:?}: {error:?}");
        assert_eq!(midway, logged, "{step:?}: no event is missing midway");

        // Act
        let report = env.run().unwrap();

        // Assert
        let moved = [report.archived.clone(), report.finished.clone()].concat();
        assert!(moved.contains(&0), "{step:?}: {report:?}");
        let (after, error) = env.scan().await;
        assert!(error.is_none(), "{step:?}: {error:?}");
        assert_eq!(after, logged, "{step:?}");
        assert!(!has_parquet(env.log(), 0), "{step:?}");
    }
}

#[tokio::test]
async fn a_corrupt_copy_keeps_the_original_and_is_a_typed_error() {
    // Arrange
    let env = Env::new();
    let logged = env.write(&events_by_age()).await;
    struct Corrupt(PathBuf);
    impl ArchiveFault for Corrupt {
        fn reached(&self, step: ArchiveStep, sequence: u64) -> io::Result<()> {
            if step == ArchiveStep::Placed {
                fs::write(self.0.join(parquet_name(sequence)), b"garbage")?;
            }
            Ok(())
        }
    }

    // Act
    let result = env.run_with(&Corrupt(env.archive().to_path_buf()));

    // Assert
    assert!(
        matches!(result, Err(ArchiveError::CopyMismatch { sequence: 0, .. })),
        "{result:?}"
    );
    assert!(has_parquet(env.log(), 0));
    assert!(read_manifest(env.log()).unwrap().is_empty());
    assert_eq!(env.scan().await.0, logged);
}

#[tokio::test]
async fn an_archived_partition_that_went_missing_fails_the_scan_for_exactly_its_range() {
    // Arrange
    let env = Env::new();
    env.write(&events_by_age()).await;
    env.run().unwrap();
    fs::remove_file(parquet_path(env.archive(), 1)).unwrap();

    // Act
    let (before, error) = env.scan().await;

    // Assert
    assert_eq!(before.len(), 2, "segment 0 was read before the failure");
    assert!(
        matches!(
            error,
            Some(ReaderError::PartitionUnreachable {
                first_segment: 1,
                last_segment: 1
            })
        ),
        "{error:?}"
    );
    let reader = SequentialScanReader::local(env.log()).unwrap();
    let (later, error) = drain(reader.scan(Some(at(2, 0)))).await;
    assert!(
        error.is_none(),
        "a scan past the range does not need it: {error:?}"
    );
    assert_eq!(later.len(), 2);
}

#[tokio::test]
async fn archived_rows_of_a_destroyed_key_still_read_as_shredded() {
    // Arrange
    let env = Env::new();
    let keys = tempfile::tempdir().unwrap();
    let registry = Arc::new(KeyRegistry::open(keys.path(), Arc::new(TestMaster::new(3))).unwrap());
    let plain = log_event("sealed-0", LogPayload::NodeWrite(node_row("n-0", "secret")));
    let sealed = Sealer::new(Arc::clone(&registry))
        .seal(
            &LogEvent {
                ingested_at: NOW - 10 * MILLIS_PER_DAY,
                ..plain
            },
            &KeyHints::default(),
        )
        .unwrap();
    let key = sealed
        .encryption_key_id
        .clone()
        .expect("a sealed event names its key");
    let filler = aged(1, 10);
    env.write(&[sealed, filler, aged(2, 1), aged(3, 1), aged(4, 1)])
        .await;
    env.run().unwrap();
    registry.destroy(&key).unwrap();

    // Act
    let reader = OpeningReader::new(SequentialScanReader::local(env.log()).unwrap(), registry);
    let (records, error) = {
        let mut stream = reader.scan(None);
        let mut shredded = Vec::new();
        let mut failure = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(record) => shredded.push(record.shredded),
                Err(error) => failure = Some(error),
            }
        }
        (shredded, failure)
    };

    // Assert
    assert!(error.is_none(), "{error:?}");
    assert_eq!(records[0], vec![key]);
}

#[tokio::test]
async fn a_partition_with_a_plaintext_row_stops_the_run_before_anything_moves() {
    // Arrange
    let env = Env::new();
    let plain = LogEvent {
        ingested_at: NOW - 10 * MILLIS_PER_DAY,
        ..log_event(
            "plain-0",
            LogPayload::NodeWrite(node_row("n-0", "readable")),
        )
    };
    let mut events = vec![aged(0, 10), aged(1, 10), plain, aged(3, 10)];
    events.extend([aged(4, 1), aged(5, 1)]);
    env.write(&events).await;

    // Act
    let result = env.run();

    // Assert
    assert!(
        matches!(
            result,
            Err(ArchiveError::PlaintextLogPresent { sequence: 1 })
        ),
        "{result:?}"
    );
    assert!(has_parquet(env.log(), 0), "the clean partition stays too");
    assert!(!env.log().join(MANIFEST_FILE).exists());
}

#[tokio::test]
async fn a_missing_archive_directory_is_a_typed_error_before_anything_moves() {
    // Arrange
    let env = Env::new();
    env.write(&events_by_age()).await;
    let missing = env.archive().join("not-there");

    // Act
    let result = archive_partitions(env.log(), &missing, POLICY, NOW, &NoFault);

    // Assert
    assert!(
        matches!(result, Err(ArchiveError::DirMissing { .. })),
        "{result:?}"
    );
    assert!(has_parquet(env.log(), 0));
}

#[tokio::test]
async fn a_dormant_log_reopens_after_its_highest_archived_sequence() {
    // Arrange: every partition is old, so all closed ones move.
    let env = Env::new();
    let old: Vec<LogEvent> = (0..6).map(|index| aged(index, 10)).collect();
    let logged = env.write(&old).await;
    env.run().unwrap();
    let highest_before = logged.last().unwrap().0.segment;

    // Act
    let mut writer =
        WalWriter::open_with_system_clock(env.log(), rotating_config(PER_SEGMENT)).unwrap();
    let appended = writer.append(&aged(6, 10)).unwrap();
    drop(writer);

    // Assert
    assert!(appended.segment >= highest_before, "{appended:?}");
    let (scanned, error) = env.scan().await;
    assert!(error.is_none(), "{error:?}");
    let ids: Vec<&str> = scanned.iter().map(|(_, e)| e.event_id.as_str()).collect();
    let expected: Vec<String> = (0..7).map(|index| format!("event-{index:03}")).collect();
    assert_eq!(ids, expected);
}

#[tokio::test]
async fn a_writer_never_reuses_the_sequence_of_a_partition_that_is_only_archived() {
    // Arrange: archive the tail's predecessor and remove the tail file, so the
    // highest sequence in the directory is below an archived one.
    let env = Env::new();
    env.write(&events_by_age()).await;
    env.run().unwrap();
    let manifest = read_manifest(env.log()).unwrap();
    let top = manifest.last().unwrap().sequence;
    for (sequence, path) in segment_paths(env.log()).unwrap() {
        if sequence > top {
            fs::remove_file(path).unwrap();
        }
    }
    for (sequence, path) in numbered_files(env.log(), PARQUET_EXTENSION).unwrap() {
        if sequence > top {
            fs::remove_file(path).unwrap();
        }
    }

    // Act
    let writer =
        WalWriter::open_with_system_clock(env.log(), rotating_config(PER_SEGMENT)).unwrap();

    // Assert
    assert_eq!(writer.open_sequence(), top + 1);
}
