// SPDX-License-Identifier: Apache-2.0
//! `liamd` as a supervisor or a person runs it: the real binary on mock models
//! in a scratch directory, so the exit code and the text are the shipped ones.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

const FAILURE_EXIT: i32 = 1;
const READY_LINE: &str = "socket listener started";
const SHUTDOWN_LINE: &str = "shutting down";
const SOCKET_NAME: &str = "s.sock";

/// A short directory under `/tmp`: a socket path inside the system temp
/// directory can pass the address length limit on macOS.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        Self::with_config("")
    }

    /// `extra` is appended to the config, for the tables it opens.
    fn with_config(extra: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(format!("/tmp/liam-it-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("liam.toml"),
            format!(
                "embedding_dims = 8\ndatabase_path = \"{0}/liam.db\"\n\
                 socket_path = \"{0}/{SOCKET_NAME}\"\nlog_filter = \"info\"\n{extra}",
                dir.display()
            ),
        )
        .unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn liamd(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_liamd"));
        command
            .args(["--config", "liam.toml"])
            .args(args)
            .current_dir(&self.0)
            .env("HOME", &self.0)
            .env_remove("LIAM_CONFIG")
            .stdin(Stdio::null());
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        self.liamd(args).output().expect("run liamd")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Kills the daemon if a test fails before it was stopped.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn delete_status_with_no_jobs_says_so_and_exits_zero() {
    // Arrange
    let scratch = Scratch::new();

    // Act
    let output = scratch.run(&["delete", "--status"]);

    // Assert
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert_eq!(text(&output.stdout).trim(), "no delete jobs");
}

#[test]
fn encrypt_log_without_a_master_key_is_refused() {
    // Arrange
    let scratch = Scratch::new();

    // Act
    let output = scratch.run(&["encrypt-log"]);

    // Assert
    assert_eq!(output.status.code(), Some(FAILURE_EXIT));
    assert!(
        text(&output.stderr)
            .contains("encrypt-log needs [encryption] master_key in the configuration"),
        "{}",
        text(&output.stderr)
    );
}

#[test]
fn the_proxy_without_a_daemon_says_where_it_looked_and_how_to_start_one() {
    // Arrange
    let scratch = Scratch::new();

    // Act
    let output = scratch.run(&["proxy"]);

    // Assert
    let stderr = text(&output.stderr);
    assert_eq!(output.status.code(), Some(FAILURE_EXIT));
    assert!(
        stderr.contains(&format!(
            "no liam daemon is listening at {}",
            scratch.path().join(SOCKET_NAME).display()
        )),
        "{stderr}"
    );
    assert!(stderr.contains("Start one with `liamd serve`"), "{stderr}");
}

#[test]
fn a_stdio_session_closed_before_it_starts_is_an_error() {
    // Arrange
    let scratch = Scratch::new();

    // Act
    let output = scratch.run(&[]);

    // Assert
    assert_eq!(output.status.code(), Some(FAILURE_EXIT));
    assert!(
        text(&output.stderr).contains("connection closed: initialize request"),
        "{}",
        text(&output.stderr)
    );
}

/// Starts `liamd serve` in `scratch`, stops it with SIGTERM once it listens,
/// and checks the orderly exit.
fn serve_then_stop_with_sigterm(scratch: &Scratch) {
    // Arrange
    let mut child = scratch
        .liamd(&["serve"])
        .stderr(Stdio::piped())
        .spawn()
        .expect("start liamd");
    let stderr = child.stderr.take().expect("piped stderr");
    let pid = child.id();
    let mut daemon = Running(child);
    let mut lines = BufReader::new(stderr);
    let mut seen = String::new();
    loop {
        let mut line = String::new();
        let read = lines.read_line(&mut line).expect("read the daemon log");
        assert!(read > 0, "the daemon ended before it listened:\n{seen}");
        seen.push_str(&line);
        if line.contains(READY_LINE) {
            break;
        }
    }
    assert!(scratch.path().join(SOCKET_NAME).exists());

    // Act
    // SAFETY: the pid is that of the child this test started and still holds.
    let sent = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    let mut rest = String::new();
    lines
        .read_to_string(&mut rest)
        .expect("read the rest of the log");
    let status = daemon.0.wait().expect("wait for liamd");

    // Assert
    assert_eq!(sent, 0);
    assert!(status.success(), "{status:?}\n{seen}{rest}");
    assert!(rest.contains(SHUTDOWN_LINE), "{rest}");
    assert!(
        !scratch.path().join(SOCKET_NAME).exists(),
        "the socket is removed"
    );
}

#[test]
fn a_serving_daemon_stops_cleanly_on_sigterm_and_removes_its_socket() {
    serve_then_stop_with_sigterm(&Scratch::new());
}

#[test]
fn a_daemon_with_tiering_and_a_tuned_generation_limit_also_stops_cleanly() {
    // A generation limit of zero asks the daemon to measure its own capacity.
    let scratch = Scratch::with_config(
        "[llm]\nmax_concurrent_generations = 0\n\n[tiering]\nenabled = true\n",
    );

    serve_then_stop_with_sigterm(&scratch);
}
