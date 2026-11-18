// SPDX-License-Identifier: Apache-2.0
//! The `liam` commands that need no store and no model: the real binary, so
//! the exit code and the text a person reads are the shipped ones. Every case
//! stops before any weights would be fetched.

use std::path::Path;
use std::process::{Command, Output};

const MOCK_CONFIG: &str = "embedding_dims = 8\n";
const QWEN3: &str = "Qwen/Qwen3-Embedding-0.6B";
const FAILURE_EXIT: i32 = 1;

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
    assert!(
        stdout.contains("embedder.device auto resolves to not used (provider mock)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("llm.device auto resolves to not used (provider mock)"),
        "{stdout}"
    );
}

#[test]
fn devices_as_json_is_one_document_naming_the_platform_and_both_resolutions() {
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
    assert_eq!(report["embedder"]["configured"], "auto");
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
        &[
            "eval",
            "retrieval",
            "--model",
            QWEN3,
            "--vectors-out",
            "vectors.json",
        ],
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
fn an_eval_with_an_unknown_device_is_refused_before_loading_any_model() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();

    for command in ["retrieval", "embedder"] {
        // Act
        let output = liam(
            dir.path(),
            &["eval", command, "--model", QWEN3, "--device", "bogus"],
        );

        // Assert
        assert_eq!(output.status.code(), Some(FAILURE_EXIT), "{command}");
        assert!(
            text(&output.stderr)
                .contains(r#"embedder.device = "bogus" is not one of auto, metal, cuda, cpu"#),
            "{command}: {}",
            text(&output.stderr)
        );
    }
}
