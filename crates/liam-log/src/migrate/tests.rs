// SPDX-License-Identifier: Apache-2.0

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures_util::StreamExt;

use super::*;
use crate::crypto::TestMaster;
use crate::envelope::OpeningReader;
use crate::event::{LogPayload, NodeRow, CURRENT_SCHEMA_VERSION};
use crate::fixtures::{compact_segments, parquet_path, wal_path};
use crate::reader::{LogReader, SequentialScanReader};
use crate::wal::{WalConfig, WalWriter};
use crate::LogWriter;

const EVENTS: usize = 40;
const SUBJECTS: usize = 5;
const SECRET: &str = "secret content";

fn node_event(index: usize) -> LogEvent {
    LogEvent {
        event_id: format!("event-{index:03}"),
        content_hash: [index as u8; 32],
        source: "client".into(),
        trust_score: 0.9,
        observed_at: 1_000,
        ingested_at: 2_000,
        encryption_key_id: None,
        schema_version: CURRENT_SCHEMA_VERSION,
        payload: LogPayload::NodeWrite(NodeRow {
            id: format!("node-{index:03}"),
            kind: "fact".into(),
            label: format!("label {index}"),
            content: format!("{SECRET} {index}"),
            producer: "agent-a".into(),
            attributes: "{}".into(),
            scope: Some("proj/a".into()),
            subject: Some(format!("subject-{}", index % SUBJECTS)),
            confidence: 0.75,
            valid_from: 1_000,
            valid_from_supplied: true,
            valid_until: 9_000,
            tx_from: 2_000,
            tx_to: 9_000,
            derivation: None,
            basis: Vec::new(),
        }),
    }
}

struct Fixture {
    dir: tempfile::TempDir,
    _keys: tempfile::TempDir,
    registry: Arc<KeyRegistry>,
    plain: Vec<LogEvent>,
}

/// A plaintext log of `EVENTS` events over several segments, the first two
/// compacted to Parquet beside their WAL files' removal.
async fn plaintext_log() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let keys = tempfile::tempdir().unwrap();
    let registry = Arc::new(KeyRegistry::open(keys.path(), Arc::new(TestMaster::new(2))).unwrap());
    let config = WalConfig {
        segment_max_bytes: 1_500,
        rotate_interval_secs: u64::MAX,
    };
    let mut writer = WalWriter::open_with_system_clock(dir.path(), config).unwrap();
    let plain: Vec<LogEvent> = (0..EVENTS).map(node_event).collect();
    for event in &plain {
        writer.append(event).unwrap();
    }
    drop(writer);
    compact_segments(dir.path(), &[0, 1]).await;
    Fixture {
        dir,
        _keys: keys,
        registry,
        plain,
    }
}

fn run(fixture: &Fixture, fault: &dyn MigrateFault) -> Result<MigrateReport, MigrateError> {
    seal_log(
        fixture.dir.path(),
        &Sealer::new(Arc::clone(&fixture.registry)),
        &fixture.registry,
        &mut NodeKeyHints::default(),
        fault,
    )
}

async fn opened(fixture: &Fixture) -> Vec<(crate::LogOffset, LogEvent)> {
    let reader = OpeningReader::new(
        SequentialScanReader::local(fixture.dir.path()).unwrap(),
        Arc::clone(&fixture.registry),
    );
    reader
        .scan(None)
        .map(|record| {
            let record = record.expect("record");
            (record.offset, record.event)
        })
        .collect()
        .await
}

async fn raw(fixture: &Fixture) -> Vec<(crate::LogOffset, LogEvent)> {
    SequentialScanReader::local(fixture.dir.path())
        .unwrap()
        .scan(None)
        .map(|record| {
            let record = record.expect("record");
            (record.offset, record.event)
        })
        .collect()
        .await
}

fn file_bytes(dir: &Path) -> Vec<u8> {
    let mut all = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            all.extend(fs::read(path).unwrap());
        }
    }
    all
}

fn contains(haystack: &[u8], needle: &str) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle.as_bytes())
}

/// Fails once at `step`, as a crash there would.
struct CrashAt {
    step: MigrateStep,
    fired: AtomicBool,
}

impl CrashAt {
    fn new(step: MigrateStep) -> Self {
        Self {
            step,
            fired: AtomicBool::new(false),
        }
    }
}

impl MigrateFault for CrashAt {
    fn reached(&self, step: MigrateStep, _sequence: u64) -> io::Result<()> {
        if step == self.step && !self.fired.swap(true, Ordering::SeqCst) {
            return Err(io::Error::other("simulated crash"));
        }
        Ok(())
    }
}

#[tokio::test]
async fn every_event_reads_back_with_its_offset_and_id_and_no_plaintext_is_left() {
    // Arrange
    let fixture = plaintext_log().await;
    let before = raw(&fixture).await;
    assert!(contains(&file_bytes(fixture.dir.path()), SECRET));

    // Act
    let report = run(&fixture, &NoFault).unwrap();

    // Assert
    let after = opened(&fixture).await;
    assert_eq!(after.len(), EVENTS);
    for ((offset, original), (sealed_offset, event)) in before.iter().zip(&after) {
        assert_eq!(offset, sealed_offset);
        assert_eq!(original.event_id, event.event_id);
        assert_eq!(original.payload, event.payload);
    }
    assert!(report.partitions_rewritten >= 3);
    assert_eq!(report.events_sealed, EVENTS as u64);
    assert!(!contains(&file_bytes(fixture.dir.path()), SECRET));
    assert!(!fixture.dir.path().join(STATE_FILE).exists());
    assert!(parquet_path(fixture.dir.path(), 0).exists());
}

#[tokio::test]
async fn the_plaintext_file_is_zeroed_before_it_is_unlinked() {
    // Arrange
    let fixture = plaintext_log().await;
    let mut held = File::open(wal_path(fixture.dir.path(), 2)).unwrap();

    // Act
    run(&fixture, &NoFault).unwrap();

    // Assert
    let mut left = Vec::new();
    held.read_to_end(&mut left).unwrap();
    assert!(!left.is_empty());
    assert!(left.iter().all(|byte| *byte == 0));
}

#[tokio::test]
async fn a_crash_at_any_step_is_finished_by_the_next_run() {
    for step in [
        MigrateStep::Written,
        MigrateStep::Verified,
        MigrateStep::Zeroed,
    ] {
        // Arrange
        let fixture = plaintext_log().await;
        let crashed = run(&fixture, &CrashAt::new(step));

        // Act
        let resumed = run(&fixture, &NoFault);

        // Assert
        assert!(crashed.is_err(), "{step:?}");
        resumed.unwrap();
        let after = opened(&fixture).await;
        assert_eq!(after.len(), EVENTS, "{step:?}");
        for (index, (_, event)) in after.iter().enumerate() {
            assert_eq!(event.payload, fixture.plain[index].payload, "{step:?}");
        }
        assert!(
            !contains(&file_bytes(fixture.dir.path()), SECRET),
            "{step:?}"
        );
        assert!(!fixture.dir.path().join(STATE_FILE).exists(), "{step:?}");
    }
}

#[tokio::test]
async fn a_crash_before_the_copy_verified_leaves_the_plaintext_whole() {
    // Arrange
    let fixture = plaintext_log().await;

    // Act
    let crashed = run(&fixture, &CrashAt::new(MigrateStep::Written));

    // Assert
    assert!(crashed.is_err());
    let events = raw(&fixture).await;
    assert_eq!(events.len(), EVENTS);
    assert!(events.iter().zip(&fixture.plain).all(|((_, a), b)| a == b));
}

#[tokio::test]
async fn a_second_run_changes_nothing() {
    // Arrange
    let fixture = plaintext_log().await;
    run(&fixture, &NoFault).unwrap();
    let before = file_bytes(fixture.dir.path());

    // Act
    let report = run(&fixture, &NoFault).unwrap();

    // Assert
    assert_eq!(report, MigrateReport::default());
    assert_eq!(file_bytes(fixture.dir.path()), before);
}

#[tokio::test]
async fn a_duplicate_pointer_keeps_its_target_and_loses_its_header_hash() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let keys = tempfile::tempdir().unwrap();
    let registry = Arc::new(KeyRegistry::open(keys.path(), Arc::new(TestMaster::new(2))).unwrap());
    let mut duplicate = node_event(1);
    duplicate.payload = LogPayload::DuplicateOf {
        first_event_id: "event-000".into(),
    };
    let mut writer = WalWriter::open_with_system_clock(
        dir.path(),
        WalConfig {
            segment_max_bytes: 1 << 20,
            rotate_interval_secs: u64::MAX,
        },
    )
    .unwrap();
    writer.append(&node_event(0)).unwrap();
    writer.append(&duplicate).unwrap();
    drop(writer);
    let fixture = Fixture {
        dir,
        _keys: keys,
        registry,
        plain: Vec::new(),
    };

    // Act
    run(&fixture, &NoFault).unwrap();

    // Assert
    let events = raw(&fixture).await;
    assert_eq!(events[1].1.content_hash, [0; 32]);
    assert_eq!(events[1].1.payload, duplicate.payload);
    assert!(matches!(events[0].1.payload, LogPayload::Sealed(_)));
}
