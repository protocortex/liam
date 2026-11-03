// SPDX-License-Identifier: Apache-2.0
//! A log whose archived partition cannot be reached is read as a hard error by
//! every consumer, never as a log with a gap in it.

use liam_log::archive::{archive_partitions, ArchivePolicy, NoFault};
use liam_log::compactor::Compactor;
use liam_log::crypto::KeyRegistry;
use liam_log::envelope::{KeyHints, Sealer};
use liam_log::event::LogEvent;
use liam_log::event::TombstoneTable;
use liam_log::reader::ReaderError;
use tempfile::TempDir;

use super::support::{key_registry, node_write, tombstone, Env, SEGMENT_PER_EVENT};
use super::*;
use crate::types::OriginClaim;
use crate::DefaultBackend;

const EVERYTHING: ArchivePolicy = ArchivePolicy {
    after_days: 0,
    zstd_level: 3,
};

struct Lost {
    env: Env,
    registry: Arc<KeyRegistry>,
    // Keep the directories alive for the test.
    _keys: TempDir,
    _archive: TempDir,
}

/// A sealed log of three events in a segment each. The first two are compacted,
/// archived and then lost from the archive directory.
async fn log_with_a_lost_archive() -> Lost {
    let env = Env::new();
    let (keys, registry) = key_registry();
    let sealer = Sealer::new(Arc::clone(&registry));
    let sealed = |event: LogEvent| sealer.seal(&event, &KeyHints::default()).unwrap();
    env.append_with(
        SEGMENT_PER_EVENT,
        &[
            sealed(node_write("n1", "one")),
            sealed(node_write("n2", "two")),
            tombstone("gc-1", &[(TombstoneTable::Nodes, "n2")]),
        ],
    );
    let wal_dir = env.wal_dir();
    let mut wals: Vec<_> = std::fs::read_dir(&wal_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "wal"))
        .collect();
    wals.sort();
    wals.pop();
    let compactor = Compactor::local(&wal_dir).unwrap();
    for wal in wals {
        compactor.compact_segment(&wal).await.unwrap();
    }
    let archive = TempDir::new().unwrap();
    let report =
        archive_partitions(&wal_dir, archive.path(), EVERYTHING, i64::MAX, &NoFault).unwrap();
    assert!(!report.archived.is_empty());
    for entry in std::fs::read_dir(archive.path()).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    Lost {
        env,
        registry,
        _keys: keys,
        _archive: archive,
    }
}

fn is_unreachable(error: &Error) -> bool {
    matches!(
        error,
        Error::LogRead(ReaderError::PartitionUnreachable { .. })
    )
}

async fn open(lost: &Lost) -> (DefaultGraph, SharedLog) {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let log = lost.env.sealing_log(&lost.registry);
    lost.env
        .open_over::<DefaultBackend>(log, "archive.db", clock)
        .await
}

#[tokio::test]
async fn catch_up_stops_on_an_unreachable_partition() {
    // Arrange
    let lost = log_with_a_lost_archive().await;
    let (graph, _log) = open(&lost).await;

    // Act
    let result = graph.catch_up().await;

    // Assert
    let error = result.expect_err("a replay must not skip the lost range");
    assert!(is_unreachable(&error), "{error:?}");
}

#[tokio::test]
async fn a_rebuild_stops_on_an_unreachable_partition() {
    // Arrange
    let lost = log_with_a_lost_archive().await;
    let (graph, log) = open(&lost).await;

    // Act
    let result = graph.rebuild_from_log(log, RebuildMode::RequireEmpty).await;

    // Assert
    let error = result
        .err()
        .expect("a rebuild must not skip the lost range");
    assert!(is_unreachable(&error), "{error:?}");
}

#[tokio::test]
async fn the_origin_check_stops_on_an_unreachable_partition() {
    // Arrange
    let lost = log_with_a_lost_archive().await;
    let (graph, _log) = open(&lost).await;
    let claim = OriginClaim {
        node_id: NodeId::from_raw("n1"),
        origin_event_ids: vec!["event-n1".to_owned()],
        label: "label".to_owned(),
        content: "one".to_owned(),
    };

    // Act
    let result = graph.verify_origins(&[claim]).await;

    // Assert
    let error = result.expect_err("a check must not judge on a log it cannot read");
    assert!(is_unreachable(&error), "{error:?}");
}
