// SPDX-License-Identifier: Apache-2.0
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};

use tempfile::TempDir;

use super::*;
use crate::extraction::client::{
    ExtractError, ExtractRequest, JsonExtractor, ModelIdentity, SidecarClient,
};
use crate::extraction::test_double::{ok, spawn_double};

/// The fake server is this test binary run with only the entry test selected;
/// libtest rejects unknown flags, so its settings travel in the environment.
const ENTRY_TEST: &str = "extraction::sidecar::tests::fake_server_entry";
const FAKE_PORT: &str = "FAKE_PORT";
const FAKE_MODE: &str = "FAKE_MODE";
const FAKE_PID_FILE: &str = "FAKE_PID_FILE";
const MODE_EXIT: &str = "exit";
const MODE_LOADING: &str = "loading";
const MODE_IGNORE_TERM: &str = "ignore_term";
const START_TIMEOUT_SECS: u64 = 10;
const SHORT_TIMEOUT_SECS: u64 = 1;
const TIMEOUT_SLACK_FACTOR: u64 = 5;
const ENOUGH_RAM_MB: u64 = 1 << 20;
const FAKE_ACCEPT_PAUSE: Duration = Duration::from_millis(20);
const FAKE_EXIT_CODE: i32 = 3;
const FAKE_BIND_FAIL_CODE: i32 = 4;
const GONE_POLL: Duration = Duration::from_millis(20);
const GONE_WAIT: Duration = Duration::from_secs(5);
const CHAT_BODY: &str = r#"{"choices":[{"message":{"content":"{}"}}]}"#;
const LOOPBACK: &str = "127.0.0.1";
const LEAK_PROBE: &str = "LIAM_TEST_SIDECAR_LEAK";

struct FixedRam(u64);

impl FreeRam for FixedRam {
    fn free_mb(&self) -> u64 {
        self.0
    }
}

fn config(timeout_secs: u64) -> ExtractionConfig {
    ExtractionConfig {
        start_timeout_secs: timeout_secs,
        ram_budget_mb: 1,
        ..ExtractionConfig::default()
    }
}

fn deps(ports: Vec<u16>) -> Deps {
    let mut ports = ports.into_iter();
    Deps {
        free_ram: Box::new(FixedRam(ENOUGH_RAM_MB)),
        next_port: Box::new(move || Ok(ports.next().expect("a port was scripted"))),
    }
}

fn free_port() -> u16 {
    free_loopback_port(LOOPBACK).unwrap()
}

/// A command factory for the fake server, plus a count of its spawns.
fn fake(
    mode: &'static str,
    pid_file: Option<PathBuf>,
) -> (impl Fn(&LaunchParams) -> Command, Arc<AtomicUsize>) {
    let spawns = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&spawns);
    let factory = move |launch: &LaunchParams| {
        counter.fetch_add(1, Ordering::SeqCst);
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", ENTRY_TEST, "--nocapture", "--test-threads=1"])
            .env(FAKE_PORT, launch.addr.port().to_string())
            .env(FAKE_MODE, mode)
            .env(API_KEY_ENV, launch.key)
            .env(ALIAS_ENV, launch.alias);
        if let Some(path) = &pid_file {
            command.env(FAKE_PID_FILE, path);
        }
        command
    };
    (factory, spawns)
}

async fn started(factory: impl Fn(&LaunchParams) -> Command) -> SidecarGuard {
    let deps = deps(vec![free_port()]);
    SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, deps)
        .await
        .unwrap()
}

fn process_exists(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the pid exists; a zombie still counts.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

fn wait_until_gone(pid: u32) -> bool {
    let give_up = Instant::now() + GONE_WAIT;
    while process_exists(pid) && Instant::now() < give_up {
        std::thread::sleep(GONE_POLL);
    }
    !process_exists(pid)
}

fn identity() -> ModelIdentity {
    ModelIdentity {
        model_id: "fake".into(),
        model_sha256: "0".into(),
    }
}

fn request() -> ExtractRequest {
    ExtractRequest {
        system: String::new(),
        prompt: String::new(),
        schema: serde_json::json!({}),
        source_event_ids: vec![],
        schema_hash: String::new(),
        temperature: 0.0,
        seed: 0,
        max_tokens: 1,
    }
}

/// Runs only as a child of these tests, where it plays `llama-server`.
#[test]
fn fake_server_entry() {
    let Ok(port) = std::env::var(FAKE_PORT) else {
        return;
    };
    let mode = std::env::var(FAKE_MODE).unwrap_or_default();
    if let Ok(path) = std::env::var(FAKE_PID_FILE) {
        std::fs::write(path, std::process::id().to_string()).unwrap();
    }
    if mode == MODE_EXIT {
        std::process::exit(FAKE_EXIT_CODE);
    }
    if mode == MODE_IGNORE_TERM {
        // SAFETY: installing SIG_IGN for SIGTERM is async-signal-safe.
        unsafe { libc::signal(libc::SIGTERM, libc::SIG_IGN) };
    }
    let key = std::env::var(API_KEY_ENV).unwrap();
    let alias = std::env::var(ALIAS_ENV).unwrap();
    let Ok(listener) = TcpListener::bind((LOOPBACK, port.parse().unwrap())) else {
        eprintln!("bind failed for key {key} and alias {alias}");
        std::process::exit(FAKE_BIND_FAIL_CODE);
    };
    listener.set_nonblocking(true).unwrap();
    // SAFETY: getppid has no preconditions.
    let parent = unsafe { libc::getppid() };
    loop {
        // An aborted test run must not leave this server behind.
        // SAFETY: as above.
        if unsafe { libc::getppid() } != parent {
            std::process::exit(0);
        }
        match listener.accept() {
            Ok((stream, _)) => serve(stream, &mode, &key, &alias),
            Err(_) => std::thread::sleep(FAKE_ACCEPT_PAUSE),
        }
    }
}

fn serve(stream: TcpStream, mode: &str, key: &str, alias: &str) {
    stream.set_nonblocking(false).unwrap();
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).unwrap();
    let mut authorized = false;
    let mut length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        authorized |= line
            .trim()
            .eq_ignore_ascii_case(&format!("authorization: bearer {key}"));
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap();
        }
        if line.trim().is_empty() {
            break;
        }
    }
    std::io::copy(&mut reader.by_ref().take(length), &mut std::io::sink()).unwrap();
    let path = request_line.split_whitespace().nth(1).unwrap_or_default();
    let (status, body) = match path {
        _ if mode == MODE_LOADING => ("503 Service Unavailable", String::new()),
        HEALTH_PATH => ("200 OK", r#"{"status":"ok"}"#.into()),
        MODELS_PATH => ("200 OK", format!(r#"{{"data":[{{"id":"{alias}"}}]}}"#)),
        _ if authorized => ("200 OK", CHAT_BODY.into()),
        _ => ("401 Unauthorized", String::new()),
    };
    let reply = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = reader.get_mut().write_all(reply.as_bytes());
}

#[test]
fn the_entry_constant_names_a_real_test() {
    // Arrange
    let exe = std::env::current_exe().unwrap();

    // Act
    let listing = Command::new(exe)
        .args(["--list", "--exact", ENTRY_TEST])
        .output()
        .unwrap();

    // Assert
    let text = String::from_utf8_lossy(&listing.stdout);
    assert!(text.contains(&format!("{ENTRY_TEST}: test")), "{text}");
}

#[tokio::test]
async fn a_healthy_server_answers_to_its_key_only_and_leaves_nothing_behind() {
    // Arrange
    let (factory, _) = fake("", None);
    let guard = started(factory).await;
    let timeout = Duration::from_secs(START_TIMEOUT_SECS);
    let client =
        SidecarClient::new(guard.addr(), guard.api_key().into(), timeout, identity()).unwrap();
    let wrong = SidecarClient::new(guard.addr(), "wrong".into(), timeout, identity()).unwrap();

    // Act
    let healthy = client.health().await;
    let answer = client.extract_json(&request()).await;
    let refused = wrong.extract_json(&request()).await;

    // Assert
    assert!(healthy);
    assert_eq!(answer.unwrap(), "{}");
    assert_eq!(refused, Err(ExtractError::Status(401)));
    let pid = guard.pid();
    drop(guard);
    assert!(
        !process_exists(pid),
        "the child was left running or unreaped"
    );
}

#[tokio::test]
async fn too_little_free_ram_refuses_before_spawning() {
    // Arrange
    let (factory, spawns) = fake("", None);
    let config = ExtractionConfig {
        ram_budget_mb: 4096,
        ..config(START_TIMEOUT_SECS)
    };
    let low = Deps {
        free_ram: Box::new(FixedRam(100)),
        ..deps(vec![])
    };

    // Act
    let result = SidecarGuard::start_with(&config, factory, low).await;

    // Assert
    let gate = SidecarError::RamGate {
        free_mb: 100,
        budget_mb: 4096,
    };
    assert_eq!(result.err(), Some(gate));
    assert_eq!(spawns.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_server_that_never_loads_times_out_and_is_killed() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let pid_file = dir.path().join("pid");
    let (factory, _) = fake(MODE_LOADING, Some(pid_file.clone()));
    let config = config(SHORT_TIMEOUT_SECS);
    let begun = Instant::now();

    // Act
    let result = SidecarGuard::start_with(&config, factory, deps(vec![free_port()])).await;

    // Assert
    assert_eq!(result.err(), Some(SidecarError::StartTimeout));
    assert!(begun.elapsed() < Duration::from_secs(SHORT_TIMEOUT_SECS * TIMEOUT_SLACK_FACTOR));
    let pid: u32 = std::fs::read_to_string(pid_file).unwrap().parse().unwrap();
    assert!(wait_until_gone(pid));
}

#[tokio::test]
async fn a_server_that_exits_at_once_fails_after_the_attempt_budget() {
    // Arrange
    let (factory, spawns) = fake(MODE_EXIT, None);
    let ports = (0..MAX_START_ATTEMPTS).map(|_| free_port()).collect();

    // Act
    let result = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, deps(ports)).await;

    // Assert
    let exited = SidecarError::Exited(Some(FAKE_EXIT_CODE));
    assert_eq!(result.err(), Some(exited));
    assert_eq!(spawns.load(Ordering::SeqCst), MAX_START_ATTEMPTS as usize);
}

#[tokio::test]
async fn a_taken_port_is_retried_on_a_new_one() {
    // Arrange
    let held = TcpListener::bind((LOOPBACK, 0)).unwrap();
    let taken = held.local_addr().unwrap().port();
    let (factory, spawns) = fake("", None);
    let ports = vec![taken, free_port()];

    // Act
    let guard = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, deps(ports)).await;

    // Assert
    assert!(guard.unwrap().is_alive());
    assert_eq!(spawns.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_hostile_listener_gets_no_key_and_no_body() {
    // Arrange
    let hostile = spawn_double(vec![ok(r#"{"data":[{"id":"wrong"}]}"#)]).await;
    let (factory, spawns) = fake("", None);
    let ports = vec![hostile.addr.port(), free_port()];

    // Act
    let guard = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, deps(ports)).await;

    // Assert
    assert!(guard.is_ok());
    assert_eq!(spawns.load(Ordering::SeqCst), 2);
    let seen = hostile.requests();
    assert!(!seen.is_empty());
    assert!(seen
        .iter()
        .all(|r| r.starts_with("GET ") && !r.to_ascii_lowercase().contains("authorization")));
}

#[tokio::test]
async fn a_killed_child_reads_as_dead_and_is_reaped_with_the_guard() {
    // Arrange
    let (factory, _) = fake("", None);
    let guard = started(factory).await;
    let handle = guard.alive_handle();
    let pid = guard.pid();

    // Act
    // SAFETY: kills our own child.
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    let give_up = Instant::now() + GONE_WAIT;
    while guard.is_alive() && Instant::now() < give_up {
        tokio::time::sleep(GONE_POLL).await;
    }

    // Assert
    assert!(!guard.is_alive());
    assert!(!handle.is_alive());
    drop(guard);
    assert!(!process_exists(pid));
}

#[tokio::test]
async fn a_panic_while_the_guard_is_held_still_reaps_the_child() {
    // Arrange
    let (factory, _) = fake("", None);
    let guard = started(factory).await;
    let pid = guard.pid();

    // Act
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _held = guard;
        panic!("boom");
    }));

    // Assert
    assert!(outcome.is_err());
    assert!(!process_exists(pid));
}

#[tokio::test]
async fn stop_ends_a_server_that_honours_sigterm_within_the_grace() {
    // Arrange
    let (factory, _) = fake("", None);
    let guard = started(factory).await;
    let (pid, handle) = (guard.pid(), guard.alive_handle());
    let begun = Instant::now();

    // Act
    guard.stop().await;

    // Assert
    assert!(!handle.is_alive());
    assert!(begun.elapsed() < STOP_GRACE);
    assert!(!process_exists(pid));
}

#[tokio::test]
async fn stop_kills_a_server_that_ignores_sigterm() {
    // Arrange
    let (factory, _) = fake(MODE_IGNORE_TERM, None);
    let guard = started(factory).await;
    let pid = guard.pid();

    // Act
    guard.stop().await;

    // Assert
    assert!(!process_exists(pid));
}

#[test]
fn the_production_command_keeps_the_secrets_off_argv() {
    // Arrange
    let config = ExtractionConfig {
        extra_args: vec!["--threads".into(), "2".into()],
        ..ExtractionConfig::default()
    };
    let paths = Paths {
        server: "/bin/llama-server".into(),
        model: "/models/m.gguf".into(),
    };
    let launch = LaunchParams {
        addr: "127.0.0.1:9".parse().unwrap(),
        alias: "the-alias",
        key: "the-key",
    };

    // Act
    let command = server_command(&config, &paths, &launch);

    // Assert
    let args: Vec<String> = command
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert!(args
        .iter()
        .all(|a| !a.contains("the-key") && !a.contains("the-alias")));
    assert!(args.windows(2).any(|w| w == ["--host", LOOPBACK]));
    assert!(args.windows(2).any(|w| w == ["--threads", "2"]));
    let envs: Vec<_> = command.get_envs().collect();
    assert!(envs.contains(&(API_KEY_ENV.as_ref(), Some("the-key".as_ref()))));
    assert!(envs.contains(&(ALIAS_ENV.as_ref(), Some("the-alias".as_ref()))));
}

#[test]
fn bad_paths_are_typed_errors_naming_the_key() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let plain = dir.path().join("plain");
    std::fs::write(&plain, "x").unwrap();
    let plain = plain.to_str().unwrap();
    let missing = "/nonexistent/file";
    let cases = [
        (None, None, "llama_server_path"),
        (Some(missing), Some(missing), "llama_server_path"),
        (Some(plain), Some(plain), "llama_server_path"),
        (Some("/bin/sh"), Some(missing), "model_gguf"),
        (Some("/bin/sh"), None, "model_gguf"),
    ];

    for (server, model, key) in cases {
        // Act
        let config = ExtractionConfig {
            llama_server_path: server.map(Into::into),
            model_gguf: model.map(Into::into),
            ..ExtractionConfig::default()
        };
        let result = Paths::resolve(&config);

        // Assert
        assert!(
            matches!(result, Err(SidecarError::BadPath { key: k, .. }) if k == key),
            "{server:?} {model:?}"
        );
    }
}

#[test]
fn stderr_keeps_only_the_tail_and_never_the_secrets() {
    // Arrange
    let mut tail = StderrTail::default();
    let filler = "x".repeat(STDERR_CAP_BYTES);

    // Act
    tail.push(format!("key=SECRETKEY\n{filler}\nlast alias=ALIAS1 line\n").as_bytes());
    let text = tail.redacted(&["SECRETKEY", "ALIAS1"]);

    // Assert
    assert!(text.len() <= STDERR_CAP_BYTES);
    assert!(text.contains("last alias=[redacted] line"));
    assert!(!text.contains("SECRETKEY") && !text.contains("ALIAS1"));
}

#[test]
fn a_secret_cut_by_the_cap_leaves_no_fragment() {
    // Arrange
    let mut tail = StderrTail::default();
    let filler = "y".repeat(STDERR_CAP_BYTES - 5);

    // Act
    tail.push(format!("ABCDEFGH{filler}").as_bytes());
    let text = tail.redacted(&["ABCDEFGH"]);

    // Assert
    assert!(!text.contains("DEFGH"), "a fragment of the secret survived");
}

#[test]
fn the_child_inherits_only_the_allowlist_and_its_own_secrets() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let script = dir.path().join("print-env");
    std::fs::write(&script, "#!/bin/sh\nenv\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let paths = Paths {
        server: script,
        model: "/models/m.gguf".into(),
    };
    let launch = LaunchParams {
        addr: "127.0.0.1:9".parse().unwrap(),
        alias: "the-alias",
        key: "the-key",
    };
    std::env::set_var(LEAK_PROBE, "leaked");

    // Act
    let output = server_command(&ExtractionConfig::default(), &paths, &launch)
        .output()
        .unwrap();

    // Assert
    let printed = String::from_utf8_lossy(&output.stdout);
    assert!(printed.contains("LLAMA_API_KEY=the-key"), "{printed}");
    assert!(printed.contains("PATH="), "{printed}");
    assert!(!printed.contains(LEAK_PROBE), "{printed}");
}

#[test]
fn the_random_values_differ_and_are_hex() {
    // Arrange / Act
    let (a, b) = (
        random_hex(KEY_BYTES).unwrap(),
        random_hex(KEY_BYTES).unwrap(),
    );

    // Assert
    assert_ne!(a, b);
    assert_eq!(a.len(), KEY_BYTES * 2);
    assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
}
