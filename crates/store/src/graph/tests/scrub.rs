// SPDX-License-Identifier: Apache-2.0
//! Scrubbing deleted text out of the database file, its log and its free pages.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tempfile::TempDir;

use super::*;
use crate::backend::{CheckpointMode, CheckpointResult};
use crate::graph::lineage::LineageStore;
use crate::types::LineageTarget;
use crate::DefaultBackend;

const KEY: &str = "key-1";
/// One lowercase alphanumeric token that stemming leaves alone, so the full-text
/// index holds these bytes whole.
const CANARY: &str = "zqjxkvw4271";
const NO_WAIT: Duration = Duration::ZERO;

struct Rig {
    graph: Graph<DefaultBackend>,
    db: PathBuf,
    _dir: TempDir,
}

async fn rig(secure_delete: bool) -> Rig {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("scrub.db");
    let config = GraphConfig::new(8).with_secure_delete(secure_delete);
    let graph = Graph::<DefaultBackend>::open(db.to_str().unwrap(), config)
        .await
        .unwrap();
    // A checkpoint on commit would move the log into the file before the test looks.
    graph
        .backend
        .execute_batch("PRAGMA wal_autocheckpoint = 0")
        .await
        .unwrap();
    Rig {
        graph,
        db,
        _dir: dir,
    }
}

fn wal_of(db: &Path) -> PathBuf {
    let mut name = db.as_os_str().to_owned();
    name.push("-wal");
    PathBuf::from(name)
}

fn holds(path: &Path, needle: &str) -> bool {
    let bytes = std::fs::read(path).unwrap_or_default();
    bytes
        .windows(needle.len())
        .any(|window| window == needle.as_bytes())
}

/// Writes the canary into a node, names its key group and returns the target.
async fn write_canary(rig: &Rig) -> LineageTarget {
    let id = rig
        .graph
        .insert(NewNode::now("fact", "note", format!("about {CANARY} only")))
        .await
        .unwrap();
    rig.graph
        .backend
        .execute(
            "INSERT INTO node_keys (node_id, key_id, source) VALUES (?1, ?2, 'event')",
            &[id.as_str().into(), KEY.into()],
        )
        .await
        .unwrap();
    LineageTarget {
        job_id: "job-1".into(),
        key_ids: vec![KEY.into()],
        ..LineageTarget::default()
    }
}

async fn put_log_in_file(rig: &Rig) {
    rig.graph
        .backend
        .wal_checkpoint(CheckpointMode::Passive, NO_WAIT)
        .await
        .unwrap();
}

#[tokio::test]
async fn a_purged_canary_is_gone_from_the_file_and_its_log_after_a_scrub() {
    // Arrange
    let rig = rig(true).await;
    let target = write_canary(&rig).await;
    put_log_in_file(&rig).await;
    assert!(holds(&rig.db, CANARY) && holds(&wal_of(&rig.db), CANARY));

    // Act
    rig.graph.purge(&target).await.unwrap();
    let report = rig.graph.scrub().await.unwrap();

    // Assert
    assert!(report.checkpoint.is_complete());
    assert!(!holds(&rig.db, CANARY), "the database file");
    assert!(!holds(&wal_of(&rig.db), CANARY), "the write-ahead log");
}

#[tokio::test]
async fn without_secure_delete_and_a_scrub_the_canary_stays_in_each_file() {
    // Arrange
    let rig = rig(false).await;
    let target = write_canary(&rig).await;
    put_log_in_file(&rig).await;

    // Act
    rig.graph.purge(&target).await.unwrap();

    // Assert
    assert!(holds(&rig.db, CANARY), "the database file");
    assert!(holds(&wal_of(&rig.db), CANARY), "the write-ahead log");
}

#[tokio::test]
async fn secure_delete_reads_back_as_set_and_optimize_runs() {
    // Arrange
    let rig = rig(true).await;

    // Act
    let on = rig.graph.backend.set_secure_delete(true).await.unwrap();
    let off = rig.graph.backend.set_secure_delete(false).await.unwrap();
    let scrubbed = rig.graph.scrub().await;

    // Assert
    assert!(on && !off);
    assert!(scrubbed.is_ok(), "{scrubbed:?}");
}

/// A second connection that reads inside a transaction, so the frames written
/// after it began cannot be checkpointed away.
async fn hold_a_reader(db: &Path) -> libsql::Connection {
    let database = libsql::Builder::new_local(db.to_str().unwrap())
        .build()
        .await
        .unwrap();
    let reader = database.connect().unwrap();
    reader.execute("BEGIN", ()).await.unwrap();
    let mut rows = reader
        .query("SELECT COUNT(*) FROM nodes", ())
        .await
        .unwrap();
    while rows.next().await.unwrap().is_some() {}
    reader
}

#[tokio::test]
async fn a_busy_checkpoint_leaves_the_scrub_open_until_the_reader_is_gone() {
    // Arrange
    let rig = rig(true).await;
    write_canary(&rig).await;
    let reader = hold_a_reader(&rig.db).await;
    rig.graph
        .insert(NewNode::now(
            "fact",
            "later",
            "written after the reader began",
        ))
        .await
        .unwrap();

    // Act
    let busy: CheckpointResult = rig
        .graph
        .backend
        .wal_checkpoint(CheckpointMode::Truncate, NO_WAIT)
        .await
        .unwrap();
    let blocked = rig.graph.scrub().await;
    let log_while_held = std::fs::metadata(wal_of(&rig.db)).unwrap().len();
    drop(reader);
    let done = rig.graph.scrub().await.unwrap();

    // Assert
    assert!(
        busy.busy || busy.log_frames != busy.checkpointed,
        "{busy:?}"
    );
    assert!(
        matches!(blocked, Err(Error::ScrubIncomplete { .. })),
        "{blocked:?}"
    );
    assert!(log_while_held > 0);
    assert_eq!(done.checkpoint.log_frames, 0);
    assert_eq!(std::fs::metadata(wal_of(&rig.db)).unwrap().len(), 0);
}

#[tokio::test]
async fn a_failure_in_any_step_names_it_and_a_rerun_completes() {
    for (prefix, step) in [
        ("INSERT INTO nodes_fts", "optimize the full-text index"),
        ("PRAGMA wal_checkpoint", "empty the write-ahead log"),
        ("PRAGMA incremental_vacuum", "reclaim the free pages"),
    ] {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("fail.db");
        let graph = Graph::<FailingBackend>::open(path.to_str().unwrap(), GraphConfig::new(8))
            .await
            .unwrap();
        graph.backend.set_fail_on_sql(Some(prefix));

        // Act
        let failed = graph.scrub().await;
        let fired = graph.backend.sql_failures();
        graph.backend.set_fail_on_sql(None);
        let rerun = graph.scrub().await;

        // Assert
        assert!(
            matches!(&failed, Err(Error::ScrubStep { step: named, .. }) if *named == step),
            "{prefix}: {failed:?}"
        );
        assert_eq!(fired, 1, "{prefix}");
        assert!(rerun.is_ok(), "{prefix}: {rerun:?}");
    }
}
