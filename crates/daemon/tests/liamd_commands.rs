// SPDX-License-Identifier: Apache-2.0
//! `liamd` as a supervisor or a person runs it: the real binary on mock models
//! in a scratch directory, so the exit code and the text are the shipped ones.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{
    Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Output, Stdio,
};
use std::sync::atomic::{AtomicU64, Ordering};

const FAILURE_EXIT: i32 = 1;
const READY_LINE: &str = "socket listener started";
const SHUTDOWN_LINE: &str = "shutting down";
const SOCKET_NAME: &str = "s.sock";
const INITIALIZE: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"liamd-test","version":"0"}}}"#;
const INITIALIZED_NOTIFICATION: &str = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
const LIST_TOOLS: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;

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
        self.liamd_with_config("liam.toml", args)
    }

    fn liamd_with_config(&self, config: &str, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_liamd"));
        command
            .args(["--config", config])
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

/// The client side of an MCP session over a child's piped stdin and stdout.
struct Session {
    to_server: ChildStdin,
    replies: BufReader<ChildStdout>,
}

impl Session {
    fn over(child: &mut Child) -> Self {
        Self {
            to_server: child.stdin.take().expect("piped stdin"),
            replies: BufReader::new(child.stdout.take().expect("piped stdout")),
        }
    }

    /// Sends a request and returns the one line that answers it.
    fn request(&mut self, message: &str) -> String {
        self.notify(message);
        let mut reply = String::new();
        self.replies.read_line(&mut reply).expect("read a reply");
        reply
    }

    fn notify(&mut self, message: &str) {
        writeln!(self.to_server, "{message}").expect("write a message");
    }

    /// Opens the session and returns what `tools/list` says.
    fn open_and_list_tools(&mut self) -> (String, String) {
        let initialized = self.request(INITIALIZE);
        self.notify(INITIALIZED_NOTIFICATION);
        (initialized, self.request(LIST_TOOLS))
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

/// A `liamd serve` that has logged that it listens.
struct Daemon {
    process: Running,
    log: BufReader<ChildStderr>,
    seen: String,
}

impl Daemon {
    fn start(scratch: &Scratch) -> Self {
        let mut child = scratch
            .liamd(&["serve"])
            .stderr(Stdio::piped())
            .spawn()
            .expect("start liamd");
        let stderr = child.stderr.take().expect("piped stderr");
        let mut daemon = Self {
            process: Running(child),
            log: BufReader::new(stderr),
            seen: String::new(),
        };
        loop {
            let mut line = String::new();
            let read = daemon
                .log
                .read_line(&mut line)
                .expect("read the daemon log");
            assert!(
                read > 0,
                "the daemon ended before it listened:\n{}",
                daemon.seen
            );
            daemon.seen.push_str(&line);
            if line.contains(READY_LINE) {
                break;
            }
        }
        assert!(scratch.path().join(SOCKET_NAME).exists());
        daemon
    }

    /// Sends SIGTERM and returns the exit status with the rest of the log.
    fn stop_with_sigterm(&mut self) -> (ExitStatus, String) {
        // SAFETY: the pid is that of the child this test started and still holds.
        let sent = unsafe { libc::kill(self.process.0.id() as libc::pid_t, libc::SIGTERM) };
        assert_eq!(sent, 0);
        let mut rest = String::new();
        self.log
            .read_to_string(&mut rest)
            .expect("read the rest of the log");
        let status = self.process.0.wait().expect("wait for liamd");
        (status, rest)
    }
}

/// Starts `liamd serve` in `scratch`, stops it with SIGTERM once it listens,
/// and checks the orderly exit.
fn serve_then_stop_with_sigterm(scratch: &Scratch) {
    // Arrange
    let mut daemon = Daemon::start(scratch);

    // Act
    let (status, rest) = daemon.stop_with_sigterm();

    // Assert
    assert!(status.success(), "{status:?}\n{}{rest}", daemon.seen);
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

#[test]
fn a_client_behind_the_proxy_can_list_the_daemons_tools() {
    // Arrange
    let scratch = Scratch::new();
    let mut daemon = Daemon::start(&scratch);
    let mut proxy = Running(
        scratch
            .liamd(&["proxy"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the proxy"),
    );
    let mut session = Session::over(&mut proxy.0);

    // Act
    let (initialized, tools) = session.open_and_list_tools();

    // Assert
    assert!(initialized.contains("\"serverInfo\""), "{initialized}");
    assert!(tools.contains("\"name\":\"remember\""), "{tools}");
    assert!(tools.contains("\"name\":\"recall\""), "{tools}");
    drop(session);
    let proxy_status = proxy.0.wait().expect("wait for the proxy");
    let (status, _) = daemon.stop_with_sigterm();
    assert!(proxy_status.success(), "{proxy_status:?}");
    assert!(status.success(), "{status:?}");
}

#[test]
fn a_config_file_that_does_not_exist_is_said_to_mean_mock_models() {
    // Arrange
    let scratch = Scratch::new();

    // Act
    let output = scratch
        .liamd_with_config("absent.toml", &[])
        .output()
        .expect("run liamd");

    // Assert
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains("no config file at")
            && stderr.contains("so the built-in defaults are used with mock models"),
        "{stderr}"
    );
}

#[test]
fn a_stdio_session_lists_the_tools_and_ends_when_the_client_closes_its_input() {
    // Arrange
    let scratch = Scratch::new();
    let mut daemon = Running(
        scratch
            .liamd(&[])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start liamd"),
    );
    let mut session = Session::over(&mut daemon.0);

    // Act
    let (initialized, tools) = session.open_and_list_tools();
    drop(session);
    let status = daemon.0.wait().expect("wait for liamd");

    // Assert
    assert!(initialized.contains("\"serverInfo\""), "{initialized}");
    assert!(tools.contains("\"name\":\"remember\""), "{tools}");
    assert!(status.success(), "{status:?}");
}
