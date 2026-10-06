// SPDX-License-Identifier: Apache-2.0
//! `liam rebuild` as a person runs it: the real binary, so the exit code, the
//! `~` expansion of its path flags and the config lookup are the ones shipped.

use std::path::Path;
use std::process::{Command, Output};
use std::sync::Arc;

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{WalConfig, WalWriter};
use liam_store::{DefaultGraph, EventLog, GraphConfig, NewNode};

const DIMS: usize = 8;

/// A directory holding a config file, a store and the log it was written
/// through, with the store and log named the way the flags default to.
async fn fixture(dir: &Path) {
    std::fs::write(dir.join("liam.toml"), format!("embedding_dims = {DIMS}\n")).unwrap();
    let log_dir = dir.join("liam.log");
    let writer = WalWriter::open_with_system_clock(
        &log_dir,
        WalConfig {
            segment_max_bytes: 1 << 20,
            rotate_interval_secs: 3600,
        },
    )
    .unwrap();
    let reader = SequentialScanReader::local(&log_dir).unwrap();
    let log = Arc::new(tokio::sync::Mutex::new(EventLog::new(
        Box::new(writer),
        Arc::new(reader),
        HashBloom::new(BloomConfig::default()),
    )));
    DefaultGraph::open(
        dir.join("liam.db").to_str().unwrap(),
        GraphConfig::new(DIMS),
    )
    .await
    .unwrap()
    .with_log(log)
    .await
    .unwrap()
    .insert(NewNode::now("fact", "label", "alpha"))
    .await
    .unwrap();
}

fn liam(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_liam"))
        .args(args)
        .current_dir(dir)
        .env("HOME", dir)
        .env_remove("LIAM_CONFIG")
        .output()
        .expect("run liam")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[tokio::test]
async fn a_rebuild_that_succeeds_exits_zero() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path()).await;
    let config = dir.path().join("liam.toml");

    // Act
    let output = liam(
        dir.path(),
        &[
            "--config",
            config.to_str().unwrap(),
            "rebuild",
            "--database",
            "liam.db",
        ],
    );

    // Assert
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(text(&output.stdout).contains("nodes: 1"));
}

#[tokio::test]
async fn a_refused_rebuild_exits_one() {
    // Arrange: a row only the database holds, which the log cannot account for
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path()).await;
    DefaultGraph::open(
        dir.path().join("liam.db").to_str().unwrap(),
        GraphConfig::new(DIMS),
    )
    .await
    .unwrap()
    .insert(NewNode::now("fact", "label", "unlogged"))
    .await
    .unwrap();
    let config = dir.path().join("liam.toml");

    // Act
    let output = liam(
        dir.path(),
        &[
            "--config",
            config.to_str().unwrap(),
            "rebuild",
            "--database",
            "liam.db",
        ],
    );

    // Assert
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).contains("mismatch"));
    assert!(output.stdout.is_empty());
}

#[tokio::test]
async fn a_rebuild_without_a_config_file_exits_one_and_says_how_to_name_one() {
    // Arrange: the default liam.toml is looked up in the working directory,
    // where there is none
    let dir = tempfile::tempdir().unwrap();

    // Act
    let output = liam(dir.path(), &["rebuild", "--database", "liam.db"]);

    // Assert
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).contains("LIAM_CONFIG"));
}

#[tokio::test]
async fn a_tilde_in_the_path_flags_expands_to_the_home_directory() {
    // Arrange: HOME is the directory holding the store and its log, which are
    // not reachable from the working directory by a literal `~`
    let home = tempfile::tempdir().unwrap();
    fixture(home.path()).await;
    let elsewhere = tempfile::tempdir().unwrap();
    let config = home.path().join("liam.toml");

    // Act
    let output = Command::new(env!("CARGO_BIN_EXE_liam"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "rebuild",
            "--database",
            "~/liam.db",
            "--log-dir",
            "~/liam.log",
        ])
        .current_dir(elsewhere.path())
        .env("HOME", home.path())
        .output()
        .expect("run liam");

    // Assert
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(text(&output.stdout).contains(&home.path().join("liam.db").display().to_string()));
}
