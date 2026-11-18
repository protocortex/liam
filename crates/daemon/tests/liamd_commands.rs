// SPDX-License-Identifier: Apache-2.0
//! `liamd` as a supervisor or a person runs it: the real binary on mock models
//! in a scratch directory, so the exit code and the text are the shipped ones.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

const FAILURE_EXIT: i32 = 1;
const SOCKET_NAME: &str = "s.sock";

/// A short directory under `/tmp`: a socket path inside the system temp
/// directory can pass the address length limit on macOS.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(format!("/tmp/liam-it-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("liam.toml"),
            format!(
                "embedding_dims = 8\ndatabase_path = \"{0}/liam.db\"\n\
                 socket_path = \"{0}/{SOCKET_NAME}\"\nlog_filter = \"info\"\n",
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
