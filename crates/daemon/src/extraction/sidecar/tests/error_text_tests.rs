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
fn each_start_failure_carries_the_value_an_operator_needs() {
    // Arrange
    let cases: Vec<(SidecarError, Vec<String>)> = vec![
        (
            bad_path("model_gguf", "does not point to a file"),
            vec!["model_gguf".into(), "does not point to a file".into()],
        ),
        (
            SidecarError::RamGate {
                free_mb: FREE_MB,
                budget_mb: BUDGET_MB,
            },
            vec![
                FREE_MB.to_string(),
                BUDGET_MB.to_string(),
                "ram_budget_mb".into(),
            ],
        ),
        (
            SidecarError::Exited(Some(EXIT_CODE)),
            vec![EXIT_CODE.to_string()],
        ),
        (SidecarError::Exited(None), vec!["signal".into()]),
        (SidecarError::StartTimeout, vec!["sidecar".into()]),
    ];

    for (error, carried) in cases {
        // Act
        let text = error.to_string();

        // Assert
        for part in carried {
            assert!(text.contains(&part), "{text:?} should carry {part:?}");
        }
    }
}

#[test]
fn a_start_that_ended_by_signal_reads_differently_from_one_that_exited() {
    // Act
    let signalled = SidecarError::Exited(None).to_string();
    let exited = SidecarError::Exited(Some(EXIT_CODE)).to_string();

    // Assert
    assert_ne!(signalled, exited);
    assert!(!signalled.contains(&EXIT_CODE.to_string()));
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
    let setup_text = setup.to_string();
    let spawn_text = spawn.to_string();
    assert!(setup_text.contains("no random bytes"), "{setup_text}");
    assert!(spawn_text.contains("exec failed"), "{spawn_text}");
    // The operator must be able to tell a failed preparation from a failed spawn.
    assert!(!setup_text.contains("exec failed"), "{setup_text}");
    assert!(setup_text.contains("sidecar"), "{setup_text}");
    assert!(spawn_text.contains("process"), "{spawn_text}");
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
