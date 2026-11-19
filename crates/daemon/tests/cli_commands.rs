// SPDX-License-Identifier: Apache-2.0
//! The `liam` commands that need no store and no model: the real binary, so
//! the exit code and the text a person reads are the shipped ones. Every case
//! stops before any weights would be fetched.

mod support;

use std::path::Path;
use std::process::{Command, Output};

const MOCK_CONFIG: &str = "embedding_dims = 8\n";
const FAILURE_EXIT: i32 = 1;
const USAGE_EXIT: i32 = 2;

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

fn with_config(dir: &Path) {
    std::fs::write(dir.join("liam.toml"), MOCK_CONFIG).unwrap();
}

#[test]
fn devices_prints_the_platform_and_says_mock_providers_load_no_model() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    // Act
    let output = liam(dir.path(), &["devices"]);

    // Assert
    let stdout = text(&output.stdout);
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(
        stdout.contains(&format!(
            "platform: {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        )),
        "{stdout}"
    );
    assert!(!stdout.contains("embedder.device"), "{stdout}");
    assert!(
        stdout.contains("llm.device auto resolves to not used (provider mock)"),
        "{stdout}"
    );
}

#[test]
fn devices_as_json_is_one_document_naming_the_platform_and_the_llm_resolution() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    // Act
    let output = liam(dir.path(), &["devices", "--json"]);

    // Assert
    assert!(output.status.success(), "{}", text(&output.stderr));
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("one JSON document");
    assert_eq!(report["os"], std::env::consts::OS);
    assert_eq!(report["arch"], std::env::consts::ARCH);
    assert!(report.get("embedder").is_none());
    assert_eq!(report["llm"]["configured"], "auto");
}

#[test]
fn fetch_models_with_only_mock_providers_exits_one_and_says_nothing_is_fetched() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    with_config(dir.path());

    // Act
    let output = liam(dir.path(), &["--config", "liam.toml", "fetch-models"]);

    // Assert
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(FAILURE_EXIT));
    assert!(stderr.contains("nothing to fetch"), "{stderr}");
    assert!(
        !stderr.contains("does not exist"),
        "the config file was there, so no missing-file note: {stderr}"
    );
}

#[test]
fn fetch_models_names_a_config_path_that_does_not_exist() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    // Act
    let output = liam(dir.path(), &["--config", "missing.toml", "fetch-models"]);

    // Assert
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(FAILURE_EXIT));
    assert!(
        stderr
            .contains("Note that missing.toml does not exist, so the built-in defaults were used"),
        "{stderr}"
    );
}

#[test]
fn consolidate_status_on_a_database_with_no_run_says_so_and_exits_zero() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    // Act
    let output = liam(dir.path(), &["consolidate", "status"]);

    // Assert
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(
        text(&output.stdout).trim(),
        "no consolidation run is recorded for this database"
    );
}

#[test]
fn consolidate_status_reads_the_database_named_by_the_flag() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let database = dir.path().join("elsewhere.db");

    // Act
    let output = liam(
        dir.path(),
        &[
            "consolidate",
            "--database",
            database.to_str().unwrap(),
            "status",
        ],
    );

    // Assert
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(text(&output.stdout).contains("no consolidation run is recorded"));
}

#[test]
fn a_retrieval_eval_refuses_the_vectors_flag_before_loading_any_model() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    // Act
    let output = liam(
        dir.path(),
        &["eval", "retrieval", "--vectors-out", "vectors.json"],
    );

    // Assert
    assert_eq!(output.status.code(), Some(FAILURE_EXIT));
    assert!(
        text(&output.stderr).contains("--vectors-out is only supported by `liam eval embedder`"),
        "{}",
        text(&output.stderr)
    );
    assert!(!dir.path().join("vectors.json").exists());
}

#[test]
fn an_eval_with_a_removed_model_or_device_flag_is_refused_before_loading_any_model() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    for command in ["retrieval", "embedder"] {
        for flag in [["--model", "acme/embed-9000"], ["--device", "cpu"]] {
            // Act
            let output = liam(dir.path(), &["eval", command, flag[0], flag[1]]);

            // Assert
            assert_eq!(output.status.code(), Some(USAGE_EXIT), "{command} {flag:?}");
            assert!(
                text(&output.stderr).contains(flag[0]),
                "{command}: {}",
                text(&output.stderr)
            );
        }
    }
}

const STEPS_DONE: &str =
    "steps done: snapshot, baseline, dedup, maintenance, extract:skipped, extract, \
     summarize:skipped, summarize, tier:skipped, tier, probes";

/// A store with its log, and the embedder record that `rebuild` writes.
async fn recorded_store(dir: &Path) {
    support::fixture(dir).await;
    let rebuilt = liam(
        dir,
        &["--config", "liam.toml", "rebuild", "--database", "liam.db"],
    );
    assert_eq!(rebuilt.status.code(), Some(0), "{}", text(&rebuilt.stderr));
}

#[tokio::test]
async fn a_consolidation_commits_and_a_later_status_reports_the_same_run() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    recorded_store(dir.path()).await;
    let consolidate = [
        "--config",
        "liam.toml",
        "consolidate",
        "--database",
        "liam.db",
    ];
    let status = [consolidate.as_slice(), &["status"]].concat();

    // Act
    let ran = liam(dir.path(), &consolidate);
    let reported = liam(dir.path(), &status);

    // Assert
    assert_eq!(ran.status.code(), Some(0), "{}", text(&ran.stderr));
    let ran_text = text(&ran.stdout);
    assert!(ran_text.contains(": committed\n"), "{ran_text}");
    assert!(ran_text.contains(STEPS_DONE), "{ran_text}");
    assert_eq!(text(&reported.stdout), ran_text);
}

#[test]
fn a_consolidation_over_a_database_with_no_log_exits_one_and_says_how_to_make_one() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    with_config(dir.path());

    // Act
    let output = liam(
        dir.path(),
        &[
            "--config",
            "liam.toml",
            "consolidate",
            "--database",
            "liam.db",
        ],
    );

    // Assert
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(FAILURE_EXIT));
    assert!(stderr.starts_with("liam consolidate: "), "{stderr}");
    assert!(stderr.contains("start liamd once to create it"), "{stderr}");
}
