// SPDX-License-Identifier: Apache-2.0
//! What an operator reads when the sidecar cannot start, and the production
//! wiring of the system dependencies.

use std::error::Error;

use super::*;

const FREE_MB: u64 = 100;
const BUDGET_MB: u64 = 2048;
const EXIT_CODE: i32 = 7;
const FIXED_PORT: u16 = 18080;
const LOOPBACK: &str = "127.0.0.1";
const SECRET: &str = "s3cret-value";

#[test]
fn each_start_failure_names_its_cause_in_plain_words() {
    // Arrange
    let cases: Vec<(SidecarError, &str)> = vec![
        (
            bad_path("model_gguf", "does not point to a file"),
            "extraction.model_gguf does not point to a file",
        ),
        (
            SidecarError::RamGate {
                free_mb: FREE_MB,
                budget_mb: BUDGET_MB,
            },
            "100 MB of RAM is free, below extraction.ram_budget_mb of 2048 MB",
        ),
        (
            SidecarError::Exited(Some(EXIT_CODE)),
            "the sidecar exited with code 7 at start",
        ),
        (
            SidecarError::Exited(None),
            "the sidecar was killed by a signal at start",
        ),
        (
            SidecarError::StartTimeout,
            "the sidecar was not ready in time",
        ),
    ];

    for (error, expected) in cases {
        // Act
        let text = error.to_string();

        // Assert
        assert_eq!(text, expected);
    }
}

#[test]
fn setup_and_spawn_failures_keep_their_source_and_the_others_have_none() {
    // Arrange
    let setup = setup_failure("no random bytes");
    let spawn = SidecarError::Spawn(io::Error::other("exec failed"));

    // Act
    let setup_source = setup.source().map(ToString::to_string);
    let spawn_source = spawn.source().map(ToString::to_string);

    // Assert
    assert_eq!(setup_source.as_deref(), Some("no random bytes"));
    assert_eq!(spawn_source.as_deref(), Some("exec failed"));
    assert!(setup.to_string().contains("no random bytes"));
    assert!(spawn.to_string().contains("exec failed"));
    assert!(SidecarError::StartTimeout.source().is_none());
    assert!(SidecarError::Exited(None).source().is_none());
}

#[test]
fn a_secret_never_shows_in_its_debug_form() {
    // Arrange
    let secret = Secret(SECRET.to_string());

    // Act
    let printed = format!("{secret:?}");

    // Assert
    assert_eq!(printed, "Secret(<redacted>)");
    assert!(!printed.contains(SECRET));
}

#[test]
fn the_system_ram_reading_is_in_megabytes_and_never_exceeds_the_installed_memory() {
    // Arrange
    let installed_mb = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    )
    .total_memory()
        / BYTES_PER_MB;

    // Act
    let free = SystemRam.free_mb();

    // Assert
    assert!(installed_mb > 0);
    assert!(free <= installed_mb, "{free} MB free of {installed_mb} MB");
}

#[test]
fn a_fixed_port_is_used_as_configured() {
    // Arrange
    let config = ExtractionConfig {
        host: LOOPBACK.to_string(),
        port: FIXED_PORT,
        ..ExtractionConfig::default()
    };
    let mut deps = Deps::system(&config, Path::new("pid"), Path::new("server"));

    // Act
    let port = (deps.next_port)().unwrap();

    // Assert
    assert_eq!(port, FIXED_PORT);
    assert_eq!(deps.pidfile, Path::new("pid"));
    assert_eq!(deps.server_exe, Path::new("server"));
}

#[test]
fn port_zero_asks_the_system_for_a_free_loopback_port() {
    // Arrange
    let config = ExtractionConfig {
        host: LOOPBACK.to_string(),
        port: 0,
        ..ExtractionConfig::default()
    };
    let mut deps = Deps::system(&config, Path::new("pid"), Path::new("server"));

    // Act
    let port = (deps.next_port)().unwrap();

    // Assert
    assert_ne!(port, 0);
}

#[test]
fn a_host_that_is_not_an_address_cannot_pick_a_port() {
    // Arrange
    let config = ExtractionConfig {
        host: "not-an-address".to_string(),
        port: 0,
        ..ExtractionConfig::default()
    };
    let mut deps = Deps::system(&config, Path::new("pid"), Path::new("server"));

    // Act
    let error = (deps.next_port)().unwrap_err();

    // Assert
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
}

#[tokio::test]
async fn starting_with_the_system_wiring_rejects_unset_paths_before_any_spawn() {
    // Arrange
    let config = ExtractionConfig::default();
    let dir = tempfile::tempdir().unwrap();

    // Act
    let result = SidecarGuard::start(&config, &dir.path().join("pid")).await;

    // Assert
    match result {
        Err(SidecarError::BadPath { key, why }) => {
            assert_eq!(key, "llama_server_path");
            assert_eq!(why, "must be set when llm_enabled is true");
        }
        other => panic!("expected a bad path error, got {:?}", other.map(|_| ())),
    }
}
