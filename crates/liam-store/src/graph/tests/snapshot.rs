// SPDX-License-Identifier: Apache-2.0
//! Snapshot and restore of a log-backed store. Tests run against real file
//! databases and a real WAL in a temp directory. A restore brings the database
//! back to the snapshot's point in the log and `catch_up` carries it forward, so
//! the log is never rolled back and every record is applied exactly once.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::log_write::within_deadline;
use super::support::{count, fact, snapshot as rows_of, Env};
use super::*;
use crate::error::Error;
use crate::graph::snapshot::restore_snapshot;
use crate::DefaultBackend;

const DIMS: usize = 8;
const SHA256_HEX_LEN: usize = 64;
const LOCK_PROBE: Duration = Duration::from_millis(150);

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

async fn open(env: &Env, db: &str) -> (DefaultGraph, SharedLog) {
    env.open::<DefaultBackend>(db, clock()).await
}

fn snapshot_dir(env_dir: &TempDir) -> PathBuf {
    env_dir.path().join("snapshots")
}

fn bytes(path: &Path) -> Vec<u8> {
    fs::read(path).expect("read file")
}

/// Rows of `db` after a store over the whole log has caught up, which is what
/// the log says the store holds.
async fn full_log_state(env: &Env, db: &str) -> (String, String) {
    let (graph, _log) = open(env, db).await;
    graph.catch_up().await.expect("catch up");
    rows_of(&graph).await
}

#[tokio::test]
async fn a_snapshot_records_the_log_id_the_cursor_and_a_checksum() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("first")).await.unwrap();
    graph.insert(fact("second")).await.unwrap();
    let position = graph.log_position().await.expect("a logged store");

    // Act
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();

    // Assert
    let recorded = taken.meta.log.expect("log fields recorded");
    assert_eq!(recorded.log_id, position.log_id);
    assert_eq!(
        recorded.cursor, position.head,
        "every write is accounted for"
    );
    assert_eq!(recorded.head, position.head);
    assert_eq!(taken.meta.sha256.len(), SHA256_HEX_LEN);
    assert_eq!(taken.meta.bytes, bytes(&taken.database_path).len() as u64);
    drop(log);
}

#[tokio::test]
async fn a_snapshot_of_a_store_without_a_log_records_no_log_fields() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("plain.db");
    let graph = DefaultGraph::open(path.to_str().unwrap(), GraphConfig::new(DIMS))
        .await
        .unwrap();
    graph.insert(fact("only")).await.unwrap();

    // Act
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();

    // Assert
    assert!(taken.meta.log.is_none());
    assert!(graph.log_position().await.is_none());
}

#[tokio::test]
async fn committed_data_not_yet_checkpointed_is_in_the_snapshot_file() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, _log) = open(&env, "live.db").await;
    for content in ["a", "b", "c"] {
        graph.insert(fact(content)).await.unwrap();
    }

    // Act
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();

    // Assert
    let standalone = DefaultGraph::open(
        taken.database_path.to_str().unwrap(),
        GraphConfig::new(DIMS),
    )
    .await
    .unwrap();
    assert_eq!(count(&standalone, "nodes").await, 3);
}

#[tokio::test]
async fn a_snapshot_waits_for_a_write_that_holds_the_log() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    let target = snapshot_dir(&dir);
    let held = log.lock().await;

    // Act
    let mut pending = Box::pin(graph.snapshot(&target));
    let early = tokio::time::timeout(LOCK_PROBE, &mut pending).await;

    // Assert
    assert!(early.is_err(), "the snapshot ran while the log was held");
    drop(held);
    within_deadline(pending).await.unwrap();
}

#[tokio::test]
async fn a_failed_snapshot_leaves_nothing_at_the_snapshot_location() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, _log) = open(&env, "live.db").await;
    graph.insert(fact("kept")).await.unwrap();
    let target = snapshot_dir(&dir);
    fs::create_dir_all(&target).unwrap();
    let mut permissions = fs::metadata(&target).unwrap().permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&target, permissions.clone()).unwrap();
    if fs::write(target.join("probe"), b"").is_ok() {
        // Running with rights that ignore directory permissions: the failure
        // cannot be injected this way.
        return;
    }

    // Act
    let result = graph.snapshot(&target).await;

    // Assert
    assert!(matches!(result, Err(Error::Snapshot(_))), "{result:?}");
    #[allow(clippy::permissions_set_readonly_false)]
    permissions.set_readonly(false);
    fs::set_permissions(&target, permissions).unwrap();
    let left: Vec<_> = fs::read_dir(&target).unwrap().collect();
    assert!(left.is_empty(), "left behind: {left:?}");
}

#[tokio::test]
async fn restoring_puts_the_snapshot_file_in_place_byte_for_byte() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    graph.insert(fact("after")).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));

    // Act
    let restored = restore_snapshot(&live, &taken.meta_path, position.as_ref()).unwrap();

    // Assert
    assert_eq!(restored, taken.meta);
    assert_eq!(bytes(&live), bytes(&taken.database_path));
}

#[tokio::test]
async fn a_restore_then_catch_up_ends_at_the_logs_full_state_applying_each_record_once() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    graph
        .upsert_by(fact("old").with_subject("topic"))
        .await
        .unwrap();
    let at_snapshot = rows_of(&graph).await;
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    // The writes of a run that is about to be rolled back.
    graph.insert(fact("after")).await.unwrap();
    graph
        .upsert_by(fact("new").with_subject("topic"))
        .await
        .unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));
    restore_snapshot(&live, &taken.meta_path, position.as_ref()).unwrap();

    // Act
    let (restored, log) = open(&env, "live.db").await;
    let after_restore = rows_of(&restored).await;
    let report = restored.catch_up().await.unwrap();
    let after_catch_up = rows_of(&restored).await;
    drop((restored, log));

    // Assert
    assert_eq!(after_restore, at_snapshot, "the restore reverts the writes");
    assert_eq!(report.already_applied, 0);
    assert!(report.applied >= 2, "{report:?}");
    assert_eq!(after_catch_up, full_log_state(&env, "full.db").await);
}

#[tokio::test]
async fn a_second_catch_up_after_a_restore_applies_nothing() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    graph.insert(fact("after")).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    restore_snapshot(
        Path::new(&env.db("live.db")),
        &taken.meta_path,
        position.as_ref(),
    )
    .unwrap();
    let (restored, _log) = open(&env, "live.db").await;
    restored.catch_up().await.unwrap();

    // Act
    let again = restored.catch_up().await.unwrap();

    // Assert
    assert_eq!((again.applied, again.already_applied), (0, 0));
    assert_eq!(count(&restored, "nodes").await, 2);
}

#[tokio::test]
async fn a_snapshot_from_another_log_is_refused_and_the_live_file_is_untouched() {
    // Arrange
    let other = Env::new();
    let dir = TempDir::new().unwrap();
    let (foreign, foreign_log) = open(&other, "foreign.db").await;
    foreign.insert(fact("elsewhere")).await.unwrap();
    let taken = foreign.snapshot(&snapshot_dir(&dir)).await.unwrap();
    drop((foreign, foreign_log));
    let env = Env::new();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("mine")).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));
    let before = bytes(&live);

    // Act
    let result = restore_snapshot(&live, &taken.meta_path, position.as_ref());

    // Assert
    assert!(
        matches!(result, Err(Error::LogIdMismatch { .. })),
        "{result:?}"
    );
    assert_eq!(bytes(&live), before);
}

#[tokio::test]
async fn a_snapshot_ahead_of_the_log_is_refused_and_the_live_file_is_untouched() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("one")).await.unwrap();
    graph.insert(fact("two")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    let mut shorter = graph.log_position().await.expect("a logged store");
    shorter.head = None;
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));
    let before = bytes(&live);

    // Act
    let result = restore_snapshot(&live, &taken.meta_path, Some(&shorter));

    // Assert
    assert!(
        matches!(result, Err(Error::CursorBeyondLog { .. })),
        "{result:?}"
    );
    assert_eq!(bytes(&live), before);
}

#[tokio::test]
async fn a_snapshot_taken_without_a_log_is_refused_for_a_logged_store() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("plain.db");
    let plain = DefaultGraph::open(plain_path.to_str().unwrap(), GraphConfig::new(DIMS))
        .await
        .unwrap();
    let taken = plain.snapshot(&snapshot_dir(&dir)).await.unwrap();
    let (graph, log) = open(&env, "live.db").await;
    let position = graph.log_position().await;
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));

    // Act
    let result = restore_snapshot(&live, &taken.meta_path, position.as_ref());

    // Assert
    assert!(matches!(result, Err(Error::Snapshot(_))), "{result:?}");
}

#[tokio::test]
async fn a_corrupted_snapshot_is_refused_and_the_live_file_is_untouched() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    graph.insert(fact("after")).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    let mut damaged = bytes(&taken.database_path);
    let middle = damaged.len() / 2;
    damaged[middle] ^= 0xff;
    fs::write(&taken.database_path, damaged).unwrap();
    let live = PathBuf::from(env.db("live.db"));
    let before = bytes(&live);

    // Act
    let result = restore_snapshot(&live, &taken.meta_path, position.as_ref());

    // Assert
    assert!(
        matches!(result, Err(Error::SnapshotCorrupt(_))),
        "{result:?}"
    );
    assert_eq!(bytes(&live), before);
    assert!(!live.with_extension("db.restore").exists());
}

#[tokio::test]
async fn a_truncated_snapshot_is_refused() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    let kept = bytes(&taken.database_path);
    fs::write(&taken.database_path, &kept[..kept.len() / 2]).unwrap();

    // Act
    let result = restore_snapshot(
        Path::new(&env.db("live.db")),
        &taken.meta_path,
        position.as_ref(),
    );

    // Assert
    assert!(
        matches!(result, Err(Error::SnapshotCorrupt(_))),
        "{result:?}"
    );
}

#[tokio::test]
async fn a_restore_whose_swap_fails_returns_a_distinct_error_and_leaves_no_partial_file() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    // A directory cannot be replaced by renaming a file over it.
    let blocked = dir.path().join("blocked.db");
    fs::create_dir(&blocked).unwrap();
    let snapshot_before = bytes(&taken.database_path);

    // Act
    let result = restore_snapshot(&blocked, &taken.meta_path, position.as_ref());

    // Assert
    assert!(matches!(result, Err(Error::RestoreFailed(_))), "{result:?}");
    assert!(blocked.is_dir());
    assert_eq!(bytes(&taken.database_path), snapshot_before);
    let siblings: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(
        siblings
            .iter()
            .all(|name| !name.to_string_lossy().ends_with(".restore")),
        "{siblings:?}"
    );
}

#[tokio::test]
async fn restore_clears_a_stale_wal_so_it_cannot_be_replayed_onto_the_snapshot() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));
    let wal = PathBuf::from(format!("{}-wal", live.display()));
    let shm = PathBuf::from(format!("{}-shm", live.display()));
    fs::write(&wal, b"stale frames").unwrap();
    fs::write(&shm, b"stale index").unwrap();

    // Act
    restore_snapshot(&live, &taken.meta_path, position.as_ref()).unwrap();

    // Assert
    assert!(!wal.exists() && !shm.exists());
}
