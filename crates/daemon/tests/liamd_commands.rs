// SPDX-License-Identifier: Apache-2.0
//! `liamd` as a supervisor or a person runs it: the real binary on mock models
//! in a scratch directory, so the exit code and the text are the shipped ones.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Output, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const FAILURE_EXIT: i32 = 1;
const READY_LINE: &str = "socket listener started";
const SHUTDOWN_LINE: &str = "shutting down";
const SOCKET_NAME: &str = "s.sock";
/// How long a spawned `liamd` may live before a wait on it kills it, and how
/// long a reply may take, so no wait can hang the test run.
const CHILD_DEADLINE: Duration = Duration::from_secs(60);
/// How often a wait for the child to end looks at it again.
const EXIT_POLL: Duration = Duration::from_millis(20);
/// How long a daemon may take to log that it listens.
const READY_DEADLINE: Duration = Duration::from_secs(30);
/// Prefix of the scratch directories, kept short for the socket path limit.
const SCRATCH_PREFIX: &str = "liam-it-";
/// Short base for the scratch directories: a socket path inside the system
/// temp directory can pass the address length limit on macOS.
const SCRATCH_BASE: &str = "/tmp";
const INITIALIZE: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"liamd-test","version":"0"}}}"#;
const INITIALIZED_NOTIFICATION: &str = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
const LIST_TOOLS: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;

/// A scratch directory with a config file, removed when dropped.
struct Scratch(tempfile::TempDir);

impl Scratch {
    fn new() -> Self {
        Self::with_config("")
    }

    /// `extra` is appended to the config, for the tables it opens.
    fn with_config(extra: &str) -> Self {
        let dir = tempfile::Builder::new()
            .prefix(SCRATCH_PREFIX)
            .tempdir_in(SCRATCH_BASE)
            .expect("create the scratch directory");
        std::fs::write(
            dir.path().join("liam.toml"),
            format!(
                "embedding_dims = 8\ndatabase_path = \"{0}/liam.db\"\n\
                 socket_path = \"{0}/{SOCKET_NAME}\"\nlog_filter = \"info\"\n{extra}",
                dir.path().display()
            ),
        )
        .unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        self.0.path()
    }

    fn liamd(&self, args: &[&str]) -> Command {
        self.liamd_with_config("liam.toml", args)
    }

    fn liamd_with_config(&self, config: &str, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_liamd"));
        command
            .args(["--config", config])
            .args(args)
            .current_dir(self.path())
            .env("HOME", self.path())
            .env_remove("LIAM_CONFIG")
            .stdin(Stdio::null());
        command
    }

    /// Runs `liamd` to its end, bounded by [`CHILD_DEADLINE`].
    fn run(&self, args: &[&str]) -> Output {
        self.run_command(self.liamd(args))
    }

    fn run_command(&self, mut command: Command) -> Output {
        let mut child = Running::spawn(command.stdout(Stdio::piped()));
        let mut stdout = child.child.stdout.take().expect("piped stdout");
        let collected = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        });
        let status = child.wait();
        Output {
            status,
            stdout: collected.join().expect("stdout reader thread"),
            stderr: child.log.text().into_bytes(),
        }
    }
}

/// What a child's stderr said so far, and whether it has ended.
#[derive(Default)]
struct LogState {
    text: String,
    ended: bool,
}

/// The child's stderr, read on its own thread so a full pipe never blocks it
/// and a test can wait for a line with a deadline.
#[derive(Clone, Default)]
struct Log(Arc<(Mutex<LogState>, Condvar)>);

impl Log {
    fn text(&self) -> String {
        self.0 .0.lock().unwrap().text.clone()
    }

    /// Waits until the log holds `needle`; the error is the log so far.
    fn wait_for(&self, needle: &str, deadline: Duration) -> Result<(), String> {
        let (lock, changed) = &*self.0;
        let (state, _) = changed
            .wait_timeout_while(lock.lock().unwrap(), deadline, |state| {
                !state.text.contains(needle) && !state.ended
            })
            .unwrap();
        if state.text.contains(needle) {
            Ok(())
        } else {
            Err(state.text.clone())
        }
    }
}

/// A spawned `liamd` that is killed and reaped on every exit path, and by
/// [`Running::wait`] once [`CHILD_DEADLINE`] passes.
struct Running {
    child: Child,
    log: Log,
    reader: Option<JoinHandle<()>>,
}

impl Running {
    fn spawn(command: &mut Command) -> Self {
        let mut child = command.stderr(Stdio::piped()).spawn().expect("start liamd");
        let log = Log::default();
        let stderr = child.stderr.take().expect("piped stderr");
        let reader_log = log.clone();
        let reader = std::thread::spawn(move || {
            let (lock, changed) = &*reader_log.0;
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut state = lock.lock().unwrap();
                state.text.push_str(&line);
                state.text.push('\n');
                changed.notify_all();
            }
            lock.lock().unwrap().ended = true;
            changed.notify_all();
        });
        Self {
            child,
            log,
            reader: Some(reader),
        }
    }

    /// Waits for the child to end. A child still running at the deadline is
    /// killed and reaped, and that fails the test with its log.
    fn wait(&mut self) -> ExitStatus {
        let deadline = Instant::now() + CHILD_DEADLINE;
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("poll liamd") {
                break status;
            }
            if Instant::now() >= deadline {
                self.kill_and_reap();
                panic!(
                    "liamd was still running after {CHILD_DEADLINE:?} and was killed:\n{}",
                    self.log.text()
                );
            }
            std::thread::park_timeout(EXIT_POLL);
        };
        if let Some(reader) = self.reader.take() {
            reader.join().expect("log reader thread");
        }
        status
    }

    fn kill_and_reap(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }

    fn pid(&self) -> libc::pid_t {
        self.child.id() as libc::pid_t
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.kill_and_reap();
    }
}

/// The client side of an MCP session over a child's piped stdin and stdout.
struct Session {
    to_server: ChildStdin,
    replies: mpsc::Receiver<String>,
    log: Log,
}

impl Session {
    fn over(running: &mut Running) -> Self {
        let stdout = running.child.stdout.take().expect("piped stdout");
        let (send, replies) = mpsc::channel();
        // Ends when the child closes stdout, which `Running` ensures on drop.
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if send.send(line).is_err() {
                    break;
                }
            }
        });
        Self {
            to_server: running.child.stdin.take().expect("piped stdin"),
            replies,
            log: running.log.clone(),
        }
    }

    /// Sends a request and returns the one line that answers it, or fails
    /// with the daemon log when none comes within [`CHILD_DEADLINE`].
    fn request(&mut self, message: &str) -> String {
        self.notify(message);
        match self.replies.recv_timeout(CHILD_DEADLINE) {
            Ok(line) => line,
            Err(error) => panic!(
                "no reply to {message} ({error}); the daemon log:\n{}",
                self.log.text()
            ),
        }
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
struct Daemon(Running);

impl Daemon {
    fn start(scratch: &Scratch) -> Self {
        let process = Running::spawn(&mut scratch.liamd(&["serve"]));
        if let Err(seen) = process.log.wait_for(READY_LINE, READY_DEADLINE) {
            panic!("the daemon did not listen within {READY_DEADLINE:?}:\n{seen}");
        }
        assert!(scratch.path().join(SOCKET_NAME).exists());
        Self(process)
    }

    /// Sends SIGTERM and returns the exit status with the whole log.
    fn stop_with_sigterm(&mut self) -> (ExitStatus, String) {
        // SAFETY: the pid is that of a child this test has not yet reaped.
        let sent = unsafe { libc::kill(self.0.pid(), libc::SIGTERM) };
        assert_eq!(sent, 0);
        let status = self.0.wait();
        (status, self.0.log.text())
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
    assert!(status.success(), "{status:?}\n{rest}");
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
    let mut proxy = Running::spawn(
        scratch
            .liamd(&["proxy"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped()),
    );
    let mut session = Session::over(&mut proxy);

    // Act
    let (initialized, tools) = session.open_and_list_tools();

    // Assert
    assert!(initialized.contains("\"serverInfo\""), "{initialized}");
    assert!(tools.contains("\"name\":\"remember\""), "{tools}");
    assert!(tools.contains("\"name\":\"recall\""), "{tools}");
    drop(session);
    let proxy_status = proxy.wait();
    let (status, _) = daemon.stop_with_sigterm();
    assert!(proxy_status.success(), "{proxy_status:?}");
    assert!(status.success(), "{status:?}");
}

#[test]
fn a_config_file_that_does_not_exist_is_said_to_mean_mock_models() {
    // Arrange
    let scratch = Scratch::new();

    // Act
    let output = scratch.run_command(scratch.liamd_with_config("absent.toml", &[]));

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
    let mut daemon = Running::spawn(
        scratch
            .liamd(&[])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped()),
    );
    let mut session = Session::over(&mut daemon);

    // Act
    let (initialized, tools) = session.open_and_list_tools();
    drop(session);
    let status = daemon.wait();

    // Assert
    assert!(initialized.contains("\"serverInfo\""), "{initialized}");
    assert!(tools.contains("\"name\":\"remember\""), "{tools}");
    assert!(status.success(), "{status:?}");
}
