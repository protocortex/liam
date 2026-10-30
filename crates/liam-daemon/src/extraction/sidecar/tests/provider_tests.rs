// SPDX-License-Identifier: Apache-2.0
//! The production provider against the fake server: a session reaches it
//! through the client, reports its death, and stops it when dropped.

use std::sync::Mutex;

use async_trait::async_trait;

use super::*;
use crate::extraction::extractor::ExtractorProvider;
use crate::extraction::identity::GgufFile;
use crate::extraction::model_double::FixedIdentity;
use crate::extraction::provider::{pidfile_path, SidecarProvider, SidecarStarter};

type CommandFactory = Box<dyn Fn(&LaunchParams) -> Command + Send + Sync>;

/// Starts the fake server once, with the given ports and memory.
struct FakeStarter {
    factory: CommandFactory,
    deps: Mutex<Option<Deps>>,
}

impl FakeStarter {
    fn with(factory: CommandFactory, deps: Deps) -> Arc<Self> {
        Arc::new(Self {
            factory,
            deps: Mutex::new(Some(deps)),
        })
    }
}

#[async_trait]
impl SidecarStarter for FakeStarter {
    async fn start(&self, config: &ExtractionConfig) -> Result<SidecarGuard, SidecarError> {
        let deps = self
            .deps
            .lock()
            .unwrap()
            .take()
            .expect("one start per starter");
        SidecarGuard::start_with(config, |launch| (self.factory)(launch), deps).await
    }
}

fn provider(starter: Arc<FakeStarter>) -> SidecarProvider {
    SidecarProvider::new(starter, Arc::new(FixedIdentity(identity())))
}

fn read_pid(path: &std::path::Path) -> u32 {
    std::fs::read_to_string(path).unwrap().parse().unwrap()
}

#[tokio::test]
async fn a_session_answers_through_the_client_and_stops_the_server_when_dropped() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let pid_file = dir.path().join("pid");
    let (factory, _) = fake("", Some(pid_file.clone()));
    let starter = FakeStarter::with(Box::new(factory), deps(vec![free_port()]));
    let provider = provider(starter);

    // Act
    let session = provider.open(&config(START_TIMEOUT_SECS)).await.unwrap();
    let answer = session.extractor.extract_json(&request()).await.unwrap();
    let alive_while_open = (session.alive)();
    drop(session);

    // Assert
    assert_eq!(answer, "{}");
    assert!(alive_while_open);
    assert!(wait_until_gone(read_pid(&pid_file)));
}

#[tokio::test]
async fn a_completion_slower_than_the_request_timeout_fails_at_the_timeout() {
    // Arrange
    let (factory, _) = fake(MODE_SLOW_COMPLETION, None);
    let starter = FakeStarter::with(Box::new(factory), deps(vec![free_port()]));
    let config = ExtractionConfig {
        request_timeout_secs: SHORT_TIMEOUT_SECS,
        ..config(START_TIMEOUT_SECS)
    };
    let session = provider(starter).open(&config).await.unwrap();
    let started = Instant::now();

    // Act
    let result = session.extractor.extract_json(&request()).await;

    // Assert
    assert!(result.is_err());
    assert!(started.elapsed() < FAKE_COMPLETION_PAUSE / 2);
}

#[tokio::test]
async fn the_alive_check_turns_false_when_the_server_process_dies() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let pid_file = dir.path().join("pid");
    let (factory, _) = fake("", Some(pid_file.clone()));
    let starter = FakeStarter::with(Box::new(factory), deps(vec![free_port()]));
    let session = provider(starter)
        .open(&config(START_TIMEOUT_SECS))
        .await
        .unwrap();

    // Act
    // SAFETY: the pid is the fake server this test started.
    unsafe { libc::kill(read_pid(&pid_file) as libc::pid_t, libc::SIGKILL) };
    let give_up = Instant::now() + GONE_WAIT;
    while (session.alive)() && Instant::now() < give_up {
        std::thread::sleep(GONE_POLL);
    }

    // Assert
    assert!(!(session.alive)());
}

#[tokio::test]
async fn a_server_that_cannot_start_is_an_error_naming_why() {
    // Arrange
    let (factory, spawns) = fake("", None);
    let low_memory = Deps {
        free_ram: Box::new(FixedRam(100)),
        ..deps(vec![free_port()])
    };
    let starter = FakeStarter::with(Box::new(factory), low_memory);
    let config = ExtractionConfig {
        ram_budget_mb: 4096,
        ..config(START_TIMEOUT_SECS)
    };

    // Act
    let error = provider(starter).open(&config).await.err().unwrap();

    // Assert
    let gate = SidecarError::RamGate {
        free_mb: 100,
        budget_mb: 4096,
    };
    assert_eq!(error.downcast_ref::<SidecarError>(), Some(&gate));
    assert_eq!(spawns.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_provider_without_a_model_identity_starts_no_server() {
    // Arrange
    let (factory, spawns) = fake("", None);
    let starter = FakeStarter::with(Box::new(factory), deps(vec![free_port()]));
    let provider = SidecarProvider::new(starter, Arc::new(GgufFile::new(None)));

    // Act
    let result = provider.open(&config(START_TIMEOUT_SECS)).await;

    // Assert
    assert!(result.is_err());
    assert_eq!(spawns.load(Ordering::SeqCst), 0);
}

#[test]
fn the_pidfile_sits_beside_the_database() {
    // Arrange
    let database = std::path::Path::new("/var/lib/liam/liam.db");

    // Act
    let pidfile = pidfile_path(database);

    // Assert
    assert_eq!(
        pidfile,
        std::path::Path::new("/var/lib/liam/liam.db.sidecar.pid")
    );
}
