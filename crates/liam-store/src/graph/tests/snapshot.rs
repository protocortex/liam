// SPDX-License-Identifier: Apache-2.0
//! Snapshot and restore of a log-backed store. Tests run against real file
//! databases and a real WAL in a temp directory. A restore brings the database
//! back to the snapshot's point in the log and `catch_up` carries it forward, so
//! the log is never rolled back and every record is applied exactly once.
//!
//! The publish and the restore are also run over a disk that fails one step at
//! a time, and that stops working from one step on to stand in for a crash.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use liam_log::wal::FileOps;

use super::log_write::within_deadline;
use super::support::{count, fact, snapshot as rows_of, Env};
use super::*;
use crate::error::Error;
use crate::graph::snapshot::{
    recover_interrupted_restore, restore_files, LogPosition, OsOps, SnapshotOps,
};
use crate::DefaultBackend;

const DIMS: usize = 8;
const SHA256_HEX_LEN: usize = 64;
const LOCK_PROBE: Duration = Duration::from_millis(150);
const OLD_WAL: &[u8] = b"committed frames not yet checkpointed";
const OLD_SHM: &[u8] = b"index of the old wal";
const SNAPSHOT_ID_STEM: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";

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

/// Every file in `dir` with its contents.
fn files_in(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .map(|entry| entry.unwrap())
                .filter(|entry| entry.path().is_file())
                .map(|entry| {
                    (
                        entry.file_name().to_string_lossy().into_owned(),
                        bytes(&entry.path()),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The files of the database `live` and its sidecars, which a restore may touch.
fn files_beside(live: &Path) -> BTreeMap<String, Vec<u8>> {
    let stem = live.file_name().unwrap().to_string_lossy().into_owned();
    files_in(live.parent().unwrap())
        .into_iter()
        .filter(|(name, _)| name.starts_with(&stem))
        .collect()
}

/// Rows of `db` after a store over the whole log has caught up, which is what
/// the log says the store holds.
async fn full_log_state(env: &Env, db: &str) -> (String, String) {
    let (graph, _log) = open(env, db).await;
    graph.catch_up().await.expect("catch up");
    rows_of(&graph).await
}

/// A disk that fails the step with a given index, or that and every step after
/// it, which is what a crash leaves behind. Records each step it is asked for.
#[derive(Clone, Default)]
struct Faults(Arc<StdMutex<FaultState>>);

#[derive(Default)]
struct FaultState {
    fail_at: Option<usize>,
    from_then_on: bool,
    steps: Vec<String>,
}

impl Faults {
    fn never() -> Self {
        Self::default()
    }

    fn at(step: usize) -> Self {
        Self(Arc::new(StdMutex::new(FaultState {
            fail_at: Some(step),
            ..FaultState::default()
        })))
    }

    fn crash_at(step: usize) -> Self {
        Self(Arc::new(StdMutex::new(FaultState {
            fail_at: Some(step),
            from_then_on: true,
            steps: Vec::new(),
        })))
    }

    fn steps(&self) -> Vec<String> {
        self.0.lock().unwrap().steps.clone()
    }

    fn step(&self, what: String) -> io::Result<()> {
        let mut state = self.0.lock().unwrap();
        let index = state.steps.len();
        state.steps.push(what.clone());
        let failing = state
            .fail_at
            .is_some_and(|at| index == at || (state.from_then_on && index > at));
        if failing {
            Err(io::Error::other(format!(
                "injected failure at step {index}: {what}"
            )))
        } else {
            Ok(())
        }
    }
}

fn name_of(path: &Path) -> String {
    path.file_name().unwrap().to_string_lossy().into_owned()
}

impl FileOps for Faults {
    fn write_all(&mut self, file: &mut File, bytes: &[u8]) -> io::Result<()> {
        use std::io::Write;
        let stepped = self.step("write".into());
        if stepped.is_err() {
            // A torn write: some of the bytes reach the file.
            file.write_all(&bytes[..bytes.len() / 2])?;
            return stepped;
        }
        file.write_all(bytes)
    }

    fn sync_file(&mut self, file: &File) -> io::Result<()> {
        self.step("sync file".into())?;
        file.sync_all()
    }

    fn truncate(&mut self, file: &File, len: u64) -> io::Result<()> {
        self.step("truncate".into())?;
        file.set_len(len)
    }

    fn sync_dir(&mut self, dir: &Path) -> io::Result<()> {
        self.step("sync dir".into())?;
        File::open(dir)?.sync_all()
    }

    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        self.step(format!("rename {} to {}", name_of(from), name_of(to)))?;
        fs::rename(from, to)
    }
}

impl SnapshotOps for Faults {
    fn remove_file(&mut self, path: &Path) -> io::Result<()> {
        self.step(format!("remove {}", name_of(path)))?;
        fs::remove_file(path)
    }
}

/// A closed store with a WAL the database file does not hold yet, and a
/// snapshot of it taken earlier.
struct Restorable {
    _env: Env,
    _dir: TempDir,
    live: PathBuf,
    meta_path: PathBuf,
    position: Option<LogPosition>,
    snapshot_bytes: Vec<u8>,
    before: BTreeMap<String, Vec<u8>>,
}

async fn restorable() -> Restorable {
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("before")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    graph.insert(fact("after")).await.unwrap();
    let position = graph.log_position().await;
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));
    fs::write(format!("{}-wal", live.display()), OLD_WAL).unwrap();
    fs::write(format!("{}-shm", live.display()), OLD_SHM).unwrap();
    Restorable {
        before: files_beside(&live),
        snapshot_bytes: bytes(&taken.database_path),
        meta_path: taken.meta_path,
        _env: env,
        _dir: dir,
        live,
        position,
    }
}

impl Restorable {
    fn restore_over(
        &self,
        ops: &mut impl SnapshotOps,
    ) -> crate::error::Result<crate::SnapshotMeta> {
        restore_files(ops, &self.live, &self.meta_path, self.position.as_ref())
    }

    /// What the files beside the database are once the snapshot has replaced
    /// it: the snapshot's file and no WAL.
    fn swapped(&self) -> BTreeMap<String, Vec<u8>> {
        BTreeMap::from([(name_of(&self.live), self.snapshot_bytes.clone())])
    }
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
async fn the_log_stays_locked_from_reading_the_cursor_until_the_copy_is_taken() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, _log) = open(&env, "live.db").await;
    graph.insert(fact("one")).await.unwrap();
    graph.insert(fact("two")).await.unwrap();
    let at_snapshot = graph.log_position().await.expect("a logged store").head;
    let graph = Arc::new(graph);
    let racing = Arc::clone(&graph);
    // Between the cursor read and the copy, a write tries to land. While the
    // snapshot holds the log it cannot, so the pause ends without it.
    let pause = move || {
        let racing = Arc::clone(&racing);
        async move {
            let write = tokio::spawn(async move { racing.insert(fact("raced")).await });
            let _ = tokio::time::timeout(LOCK_PROBE, write).await;
        }
    };

    // Act
    let taken = graph
        .snapshot_with(&snapshot_dir(&dir), OsOps, pause)
        .await
        .unwrap();

    // Assert
    let copied = DefaultGraph::open(
        taken.database_path.to_str().unwrap(),
        GraphConfig::new(DIMS),
    )
    .await
    .unwrap();
    assert_eq!(taken.meta.log.expect("log fields").cursor, at_snapshot);
    assert_eq!(
        count(&copied, "nodes").await,
        2,
        "the copy holds a later write"
    );
}

#[tokio::test]
async fn a_snapshot_into_a_path_that_is_a_file_fails_and_leaves_that_file_alone() {
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, _log) = open(&env, "live.db").await;
    graph.insert(fact("kept")).await.unwrap();
    let target = dir.path().join("not-a-directory");
    fs::write(&target, b"data").unwrap();

    // Act
    let result = graph.snapshot(&target).await;

    // Assert
    assert!(matches!(result, Err(Error::Snapshot(_))), "{result:?}");
    assert_eq!(bytes(&target), b"data");
}

/// Takes one snapshot over `ops` of a store with two writes, into `dir`.
async fn snapshot_over(
    graph: &DefaultGraph,
    dir: &Path,
    ops: Faults,
) -> crate::error::Result<crate::Snapshot> {
    graph
        .snapshot_with(dir, ops, || std::future::ready(()))
        .await
}

/// Every `.db` has its `.json` and the other way round, and no unfinished
/// file is left.
fn assert_only_whole_snapshots(dir: &Path) {
    let names: Vec<String> = files_in(dir).into_keys().collect();
    for name in &names {
        let (stem, extension) = name.split_once('.').expect("a snapshot file name");
        let partner = match extension {
            "db" => format!("{stem}.json"),
            "json" => format!("{stem}.db"),
            other => panic!("unfinished file left behind: {stem}.{other}"),
        };
        assert!(
            names.contains(&partner),
            "{name} has no {partner}: {names:?}"
        );
    }
}

#[tokio::test]
async fn a_snapshot_publishes_the_copy_then_the_metadata_in_that_order() {
    // Arrange
    let env = Env::new();
    let (graph, _log) = open(&env, "live.db").await;
    graph.insert(fact("kept")).await.unwrap();
    let dir = TempDir::new().unwrap();
    let ops = Faults::never();

    // Act
    snapshot_over(&graph, dir.path(), ops.clone())
        .await
        .unwrap();

    // Assert
    let steps = ops.steps();
    let shape: Vec<&str> = steps
        .iter()
        .map(|step| match step.as_str() {
            s if s.starts_with("rename") && s.contains(".db.partial to") => "publish the copy",
            s if s.starts_with("rename") && s.contains(".json.tmp to") => "publish the metadata",
            s => s,
        })
        .collect();
    assert_eq!(
        shape,
        [
            "sync file",
            "publish the copy",
            "publish the metadata",
            "sync dir"
        ]
    );
}

#[tokio::test]
async fn a_snapshot_that_fails_at_any_step_leaves_nothing_behind() {
    // Arrange
    let env = Env::new();
    let (graph, _log) = open(&env, "live.db").await;
    graph.insert(fact("kept")).await.unwrap();
    let recorder = Faults::never();
    let clean = TempDir::new().unwrap();
    snapshot_over(&graph, clean.path(), recorder.clone())
        .await
        .unwrap();
    let step_count = recorder.steps().len();

    for step in 0..step_count {
        let dir = TempDir::new().unwrap();

        // Act
        let result = snapshot_over(&graph, dir.path(), Faults::at(step)).await;

        // Assert
        assert!(
            matches!(result, Err(Error::Snapshot(_))),
            "step {step}: {result:?}"
        );
        let left: Vec<_> = files_in(dir.path()).into_keys().collect();
        assert!(left.is_empty(), "step {step} left {left:?}");
    }
}

#[tokio::test]
async fn a_snapshot_cut_off_at_any_step_never_leaves_a_torn_snapshot_and_the_next_one_sweeps_up() {
    // Arrange
    let env = Env::new();
    let (graph, _log) = open(&env, "live.db").await;
    graph.insert(fact("kept")).await.unwrap();
    let recorder = Faults::never();
    let clean = TempDir::new().unwrap();
    snapshot_over(&graph, clean.path(), recorder.clone())
        .await
        .unwrap();

    for step in 0..recorder.steps().len() {
        let dir = TempDir::new().unwrap();
        let crashed = snapshot_over(&graph, dir.path(), Faults::crash_at(step)).await;
        assert!(crashed.is_err(), "step {step}");
        let published: Vec<String> = files_in(dir.path())
            .into_keys()
            .filter(|name| name.ends_with(".json"))
            .collect();
        for meta in &published {
            let meta_path = dir.path().join(meta);
            let database =
                fs::read(meta_path.with_extension("db")).expect("a published meta has its db");
            let stored: serde_json::Value = serde_json::from_slice(&bytes(&meta_path)).unwrap();
            assert_eq!(
                stored["bytes"],
                database.len(),
                "step {step}: torn database"
            );
        }

        // Act
        let next = snapshot_over(&graph, dir.path(), Faults::never())
            .await
            .unwrap();

        // Assert
        assert_only_whole_snapshots(dir.path());
        assert!(next.meta_path.exists());
        assert_eq!(
            files_in(dir.path()).len(),
            2 * (published.len() + 1),
            "step {step}: unfinished files survived the sweep"
        );
    }
}

#[tokio::test]
async fn a_snapshot_sweeps_unfinished_files_of_a_crashed_one_and_leaves_other_files_alone() {
    // Arrange
    let env = Env::new();
    let (graph, _log) = open(&env, "live.db").await;
    graph.insert(fact("kept")).await.unwrap();
    let dir = TempDir::new().unwrap();
    let whole = graph.snapshot(dir.path()).await.unwrap();
    let leftovers = [
        format!("{SNAPSHOT_ID_STEM}.db.partial"),
        format!("{SNAPSHOT_ID_STEM}.json.tmp"),
        format!("{SNAPSHOT_ID_STEM}.db"),
    ];
    for name in &leftovers {
        fs::write(dir.path().join(name), b"crashed").unwrap();
    }
    fs::write(dir.path().join("notes.db"), b"mine").unwrap();
    fs::write(dir.path().join("readme.txt"), b"mine").unwrap();

    // Act
    let next = graph.snapshot(dir.path()).await.unwrap();

    // Assert
    let after: Vec<String> = files_in(dir.path()).into_keys().collect();
    for name in &leftovers {
        assert!(!after.contains(name), "{name} survived: {after:?}");
    }
    for kept in ["notes.db", "readme.txt"] {
        assert!(after.contains(&kept.to_owned()), "{kept} was removed");
    }
    assert!(whole.meta_path.exists() && whole.database_path.exists());
    assert!(next.meta_path.exists() && next.database_path.exists());
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
    let live = PathBuf::from(env.db("live.db"));

    // Act
    let restored = graph
        .restore_snapshot(&live, &taken.meta_path)
        .await
        .unwrap();

    // Assert
    assert_eq!(restored, taken.meta);
    assert_eq!(bytes(&live), bytes(&taken.database_path));
    drop(log);
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
    let live = PathBuf::from(env.db("live.db"));
    graph
        .restore_snapshot(&live, &taken.meta_path)
        .await
        .unwrap();
    drop(log);

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
    graph
        .restore_snapshot(Path::new(&env.db("live.db")), &taken.meta_path)
        .await
        .unwrap();
    drop(log);
    let (restored, _log) = open(&env, "live.db").await;
    restored.catch_up().await.unwrap();

    // Act
    let again = restored.catch_up().await.unwrap();

    // Assert
    assert_eq!((again.applied, again.already_applied), (0, 0));
    assert_eq!(count(&restored, "nodes").await, 2);
}

#[tokio::test]
async fn a_snapshot_from_another_log_is_refused_and_the_live_files_are_untouched() {
    // Arrange
    let other = Env::new();
    let dir = TempDir::new().unwrap();
    let (foreign, foreign_log) = open(&other, "foreign.db").await;
    foreign.insert(fact("elsewhere")).await.unwrap();
    let taken = foreign.snapshot(&snapshot_dir(&dir)).await.unwrap();
    drop((foreign, foreign_log));
    let target = restorable().await;
    let foreign_id = taken.meta.log.unwrap().log_id;
    let own_id = target.position.unwrap().log_id;

    // Act
    let result = restore_files(
        &mut OsOps,
        &target.live,
        &taken.meta_path,
        target.position.as_ref(),
    );

    // Assert
    assert!(
        matches!(&result, Err(Error::SnapshotLogMismatch { snapshot, store })
            if *snapshot == foreign_id && *store == own_id),
        "{result:?}"
    );
    assert_eq!(files_beside(&target.live), target.before);
}

#[tokio::test]
async fn a_snapshot_ahead_of_an_empty_log_is_refused_and_the_live_files_are_untouched() {
    // Arrange
    let mut target = restorable().await;
    target.position.as_mut().unwrap().head = None;

    // Act
    let result = target.restore_over(&mut OsOps);

    // Assert
    assert!(
        matches!(result, Err(Error::SnapshotAheadOfLog { head: None, .. })),
        "{result:?}"
    );
    assert_eq!(files_beside(&target.live), target.before);
}

#[tokio::test]
async fn a_snapshot_ahead_of_a_log_with_a_shorter_head_is_refused_and_the_live_files_are_untouched()
{
    // Arrange
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, log) = open(&env, "live.db").await;
    graph.insert(fact("one")).await.unwrap();
    let shorter = graph.log_position().await.expect("a logged store");
    graph.insert(fact("two")).await.unwrap();
    graph.insert(fact("three")).await.unwrap();
    let taken = graph.snapshot(&snapshot_dir(&dir)).await.unwrap();
    drop((graph, log));
    let live = PathBuf::from(env.db("live.db"));
    let before = files_beside(&live);

    // Act
    let result = restore_files(&mut OsOps, &live, &taken.meta_path, Some(&shorter));

    // Assert
    assert!(
        matches!(&result, Err(Error::SnapshotAheadOfLog { cursor, head })
            if Some(*cursor) == taken.meta.log.unwrap().cursor && *head == shorter.head),
        "{result:?}"
    );
    assert_eq!(files_beside(&live), before);
}

#[tokio::test]
async fn a_snapshot_whose_cursor_equals_the_log_head_is_accepted() {
    // Arrange
    let target = restorable().await;

    // Act
    let result = target.restore_over(&mut OsOps);

    // Assert
    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
async fn a_snapshot_taken_without_a_log_is_refused_for_a_logged_store() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let plain_path = dir.path().join("plain.db");
    let plain = DefaultGraph::open(plain_path.to_str().unwrap(), GraphConfig::new(DIMS))
        .await
        .unwrap();
    let taken = plain.snapshot(&snapshot_dir(&dir)).await.unwrap();
    let target = restorable().await;

    // Act
    let result = restore_files(
        &mut OsOps,
        &target.live,
        &taken.meta_path,
        target.position.as_ref(),
    );

    // Assert
    assert!(
        matches!(&result, Err(Error::SnapshotWithoutLog { snapshot_id })
            if *snapshot_id == taken.meta.snapshot_id),
        "{result:?}"
    );
    assert_eq!(files_beside(&target.live), target.before);
}

#[tokio::test]
async fn a_snapshot_of_a_logged_store_is_refused_for_a_store_without_a_log() {
    // Arrange
    let target = restorable().await;

    // Act
    let result = restore_files(&mut OsOps, &target.live, &target.meta_path, None);

    // Assert
    assert!(matches!(result, Err(Error::NoEventLog)), "{result:?}");
    assert_eq!(files_beside(&target.live), target.before);
}

#[tokio::test]
async fn a_corrupted_snapshot_is_refused_and_the_live_files_are_untouched() {
    // Arrange
    let target = restorable().await;
    let database = target.meta_path.with_extension("db");
    let mut damaged = bytes(&database);
    let middle = damaged.len() / 2;
    damaged[middle] ^= 0xff;
    fs::write(&database, damaged).unwrap();

    // Act
    let result = target.restore_over(&mut OsOps);

    // Assert
    assert!(
        matches!(result, Err(Error::SnapshotChecksumMismatch { .. })),
        "{result:?}"
    );
    assert_eq!(files_beside(&target.live), target.before);
}

#[tokio::test]
async fn a_truncated_snapshot_is_refused_and_the_live_files_are_untouched() {
    // Arrange
    let target = restorable().await;
    let database = target.meta_path.with_extension("db");
    let kept = bytes(&database);
    fs::write(&database, &kept[..kept.len() / 2]).unwrap();

    // Act
    let result = target.restore_over(&mut OsOps);

    // Assert
    assert!(
        matches!(result, Err(Error::SnapshotChecksumMismatch { .. })),
        "{result:?}"
    );
    assert_eq!(files_beside(&target.live), target.before);
}

#[tokio::test]
async fn unreadable_snapshot_metadata_is_refused_as_corrupt() {
    // Arrange
    let target = restorable().await;
    fs::write(&target.meta_path, b"not json").unwrap();

    // Act
    let result = target.restore_over(&mut OsOps);

    // Assert
    assert!(
        matches!(result, Err(Error::SnapshotCorrupt(_))),
        "{result:?}"
    );
    assert_eq!(files_beside(&target.live), target.before);
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
    let result = restore_files(&mut OsOps, &blocked, &taken.meta_path, position.as_ref());

    // Assert
    assert!(matches!(result, Err(Error::RestoreFailed(_))), "{result:?}");
    assert!(blocked.is_dir());
    assert_eq!(bytes(&taken.database_path), snapshot_before);
    let siblings: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        siblings.iter().all(|name| !name.ends_with(".restore")),
        "{siblings:?}"
    );
}

#[tokio::test]
async fn a_restore_replaces_the_wal_and_shm_with_the_snapshot_and_leaves_no_side_files() {
    // Arrange
    let target = restorable().await;

    // Act
    target.restore_over(&mut OsOps).unwrap();

    // Assert
    assert_eq!(files_beside(&target.live), target.swapped());
}

#[tokio::test]
async fn a_restore_moves_the_wal_aside_and_deletes_it_only_after_the_swap() {
    // Arrange
    let target = restorable().await;
    let ops = Faults::never();

    // Act
    target.restore_over(&mut ops.clone()).unwrap();

    // Assert
    let mut steps = ops.steps();
    steps.dedup();
    assert_eq!(
        steps,
        [
            "write",
            "sync file",
            "rename live.db-wal to live.db-wal.pre-restore",
            "rename live.db-shm to live.db-shm.pre-restore",
            "rename live.db.restore to live.db",
            "sync dir",
            "remove live.db-wal.pre-restore",
            "remove live.db-shm.pre-restore",
        ]
    );
}

#[tokio::test]
async fn a_restore_that_fails_at_any_step_leaves_the_old_database_and_wal_or_the_whole_snapshot() {
    // Arrange
    let recorder = Faults::never();
    restorable()
        .await
        .restore_over(&mut recorder.clone())
        .unwrap();

    for step in 0..recorder.steps().len() {
        let target = restorable().await;

        // Act
        let result = target.restore_over(&mut Faults::at(step));

        // Assert
        let now = files_beside(&target.live);
        if now == target.before {
            assert!(
                result.is_err(),
                "step {step}: nothing changed yet it succeeded"
            );
            continue;
        }
        let swapped_with_asides = now.get("live.db") == Some(&target.snapshot_bytes);
        assert!(
            swapped_with_asides,
            "step {step}: neither the old files nor the snapshot: {:?}",
            now.keys().collect::<Vec<_>>()
        );
        recover_interrupted_restore(&target.live).unwrap();
        assert_eq!(files_beside(&target.live), target.swapped(), "step {step}");
    }
}

#[tokio::test]
async fn a_restore_cut_off_at_any_step_is_recovered_to_the_old_files_or_the_snapshot() {
    // Arrange
    let recorder = Faults::never();
    restorable()
        .await
        .restore_over(&mut recorder.clone())
        .unwrap();

    for step in 0..recorder.steps().len() {
        let target = restorable().await;
        // A cut-off while deleting the side files after the swap still succeeds.
        let _ = target.restore_over(&mut Faults::crash_at(step));

        // Act
        recover_interrupted_restore(&target.live).unwrap();

        // Assert
        let now = files_beside(&target.live);
        assert!(
            now == target.before || now == target.swapped(),
            "step {step}: {:?}",
            now.keys().collect::<Vec<_>>()
        );

        // A restore started again completes whatever the first one left.
        target.restore_over(&mut OsOps).unwrap();
        assert_eq!(files_beside(&target.live), target.swapped(), "step {step}");
    }
}

#[tokio::test]
async fn a_restore_started_over_an_interrupted_one_first_recovers_it() {
    // Arrange
    let target = restorable().await;
    let steps = Faults::never();
    restorable().await.restore_over(&mut steps.clone()).unwrap();
    // Stop right after both side files are moved and before the swap.
    let swap = steps
        .steps()
        .iter()
        .position(|step| step.starts_with("rename live.db.restore"))
        .unwrap();
    target
        .restore_over(&mut Faults::crash_at(swap))
        .unwrap_err();
    assert!(!files_beside(&target.live).contains_key("live.db-wal"));

    // Act
    let again = target.restore_over(&mut OsOps);

    // Assert
    assert!(again.is_ok(), "{again:?}");
    assert_eq!(files_beside(&target.live), target.swapped());
}

#[tokio::test]
async fn opening_a_store_after_a_cut_off_restore_puts_its_wal_back() {
    // Arrange
    let env = Env::new();
    let path = PathBuf::from(env.db("live.db"));
    DefaultGraph::open(path.to_str().unwrap(), GraphConfig::new(DIMS))
        .await
        .unwrap();
    let wal = PathBuf::from(format!("{}-wal", path.display()));
    let aside = PathBuf::from(format!("{}.pre-restore", wal.display()));
    let staged = PathBuf::from(format!("{}.restore", path.display()));
    fs::write(&aside, b"").unwrap();
    fs::write(&staged, b"half a restore").unwrap();

    // Act
    DefaultGraph::open(path.to_str().unwrap(), GraphConfig::new(DIMS))
        .await
        .unwrap();

    // Assert
    assert!(!aside.exists() && !staged.exists());
}

#[tokio::test]
async fn recovery_refuses_to_overwrite_a_wal_that_appeared_beside_a_pending_swap() {
    // Arrange
    let target = restorable().await;
    let wal = PathBuf::from(format!("{}-wal", target.live.display()));
    let aside = PathBuf::from(format!("{}.pre-restore", wal.display()));
    let staged = PathBuf::from(format!("{}.restore", target.live.display()));
    fs::write(&aside, b"moved aside").unwrap();
    fs::write(&staged, b"half a restore").unwrap();

    // Act
    let result = recover_interrupted_restore(&target.live);

    // Assert
    assert!(matches!(result, Err(Error::RestoreFailed(_))), "{result:?}");
    assert_eq!(bytes(&wal), OLD_WAL);
    assert_eq!(bytes(&aside), b"moved aside");
}
