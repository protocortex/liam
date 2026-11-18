// SPDX-License-Identifier: Apache-2.0
//! A snapshot's metadata file is refused when it was written by another
//! format or names a log that is not a UUID.

use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value};

use super::support::{fact, Env};
use super::*;
use crate::error::Error;
use crate::graph::snapshot::verify_snapshot;
use crate::DefaultBackend;

const UNKNOWN_FORMAT: u64 = 9999;
const NOT_A_UUID: &str = "not-a-uuid";

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

async fn taken_meta_path() -> (PathBuf, Env, TempDir) {
    let env = Env::new();
    let dir = TempDir::new().unwrap();
    let (graph, _log) = env.open::<DefaultBackend>("live.db", clock()).await;
    graph.insert(fact("kept")).await.unwrap();
    let taken = graph.snapshot(&dir.path().join("snapshots")).await.unwrap();
    (taken.meta_path, env, dir)
}

fn edit_meta(path: &PathBuf, change: impl FnOnce(&mut Value)) {
    let mut meta: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    change(&mut meta);
    fs::write(path, serde_json::to_vec(&meta).unwrap()).unwrap();
}

#[tokio::test]
async fn metadata_in_another_format_is_refused_as_corrupt() {
    // Arrange
    let (meta_path, _env, _dir) = taken_meta_path().await;
    edit_meta(&meta_path, |meta| meta["format"] = json!(UNKNOWN_FORMAT));

    // Act
    let result = verify_snapshot(&meta_path);

    // Assert
    match result {
        Err(Error::SnapshotCorrupt(text)) => {
            assert!(text.contains(&UNKNOWN_FORMAT.to_string()), "{text}")
        }
        other => panic!("expected a corrupt snapshot, got {other:?}"),
    }
}

#[tokio::test]
async fn metadata_naming_a_log_that_is_not_a_uuid_is_refused_as_corrupt() {
    // Arrange
    let (meta_path, _env, _dir) = taken_meta_path().await;
    edit_meta(&meta_path, |meta| meta["log"]["log_id"] = json!(NOT_A_UUID));

    // Act
    let result = verify_snapshot(&meta_path);

    // Assert
    match result {
        Err(Error::SnapshotCorrupt(text)) => assert!(text.contains(NOT_A_UUID), "{text}"),
        other => panic!("expected a corrupt snapshot, got {other:?}"),
    }
}
