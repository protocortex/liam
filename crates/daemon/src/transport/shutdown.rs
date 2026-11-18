// SPDX-License-Identifier: Apache-2.0
//! Ordered shutdown: what stops the daemon, and in which order.
//!
//! The order is the whole point, because with concurrent clients the naive
//! version loses work:
//!
//! 1. Stop accepting. No new session can start once we are on the way out.
//! 2. Drain: let live sessions finish, bounded by [`DEFAULT_DRAIN_DEADLINE`].
//! 3. Abort whatever is still running when that deadline passes, so a stuck
//!    session cannot hold the process open forever.
//! 4. Unlink the socket, and ONLY if this process bound it.
//!
//! Step 4 is why [`crate::transport::ListenerSource`] is an enum rather than
//! a bare listener: unlinking a path launchd owns leaves the supervisor
//! holding a descriptor for a name that no longer exists, so on-demand start
//! quietly stops working until the job is reloaded.

use std::time::Duration;

use tokio::task::JoinSet;

use crate::transport::ListenerSource;

/// How long live sessions get to finish after the daemon stops accepting.
///
/// MUST stay below the supervisor's grace period, currently `ExitTimeOut`
/// 20 in `packaging/dev.protocortex.liamd.plist`. launchd sends SIGTERM,
/// waits out that grace, then SIGKILLs, so a drain allowed to run longer
/// than the grace does not finish gracefully at all: it gets killed
/// mid-write, which is the exact outcome draining exists to avoid.
/// `tests::the_drain_deadline_stays_under_the_supervisor_grace_period`
/// pins that relationship against the shipped plist.
///
pub const DEFAULT_DRAIN_DEADLINE: Duration = Duration::from_secs(10);

/// Which signal asked the daemon to stop. Carried only so the log line can
/// name it; both take the identical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// SIGTERM, what launchd and `kill` send.
    Term,
    /// SIGINT, what Ctrl-C in a terminal sends.
    Int,
}

impl Trigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Trigger::Term => "SIGTERM",
            Trigger::Int => "SIGINT",
        }
    }
}

/// SIGTERM and SIGINT handlers, installed. Once this exists a signal is held
/// for [`SignalWatch::wait`] instead of ending the process, so it is made
/// before the socket appears: a stop that arrives right after a client can
/// connect must still run the orderly shutdown.
///
/// Both signals map to the same [`Trigger`] handling on purpose: an operator
/// pressing Ctrl-C and launchd stopping the job want the same thing, and
/// having two shutdown paths is how one of them rots untested.
pub struct SignalWatch {
    term: tokio::signal::unix::Signal,
    interrupt: tokio::signal::unix::Signal,
}

impl SignalWatch {
    /// Installing the handlers can fail (the process is out of signal handler
    /// slots, or the platform refuses), and that is surfaced rather than
    /// swallowed: a daemon that silently cannot hear SIGTERM would be SIGKILLed
    /// on every stop and lose in-flight work every time.
    pub fn install() -> std::io::Result<Self> {
        use tokio::signal::unix::{signal, SignalKind};

        Ok(Self {
            term: signal(SignalKind::terminate())?,
            interrupt: signal(SignalKind::interrupt())?,
        })
    }

    /// Resolves when SIGTERM or SIGINT arrives, whichever lands first.
    ///
    /// Not exercised by raising a signal inside the unit test process: signal
    /// disposition is process-global and `cargo test` runs every test in one
    /// process, so it would race the others. `tests/liamd_commands.rs` stops
    /// a real daemon process with SIGTERM instead, and the orderly shutdown
    /// is tested directly through `drain` and `unlink_owned_socket`.
    pub async fn wait(mut self) -> Trigger {
        tokio::select! {
            _ = self.term.recv() => Trigger::Term,
            _ = self.interrupt.recv() => Trigger::Int,
        }
    }
}

/// Installs the signal handlers, and only then runs `resolve`, the step that
/// makes the socket appear. The order is what lets a stop that lands as soon
/// as a client can connect run the orderly shutdown instead of ending the
/// process. The install result is returned as is, for the caller to report.
pub async fn watch_signals_before<T, E, F, Fut>(
    resolve: F,
) -> Result<(std::io::Result<SignalWatch>, T), E>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
{
    let signals = SignalWatch::install();
    let resolved = resolve().await?;
    Ok((signals, resolved))
}

/// Waits for every live session to finish, then aborts whatever is left
/// once `deadline` passes. Returns `true` if the drain completed on its own.
///
/// Sessions are NOT told to stop first. rmcp's own session cancellation
/// drops the running service, which kills an in-flight tool call partway
/// through, so cancelling up front would defeat the drain this function
/// exists to perform. Instead a session ends the way it normally does, when
/// its client disconnects or its work finishes, and the deadline is what
/// bounds the wait.
pub async fn drain(sessions: &mut JoinSet<()>, deadline: Duration) -> bool {
    if sessions.is_empty() {
        return true;
    }

    let live = sessions.len();
    tracing::info!(
        sessions = live,
        deadline_secs = deadline.as_secs(),
        "draining live sessions before exit"
    );

    let drained = tokio::time::timeout(deadline, async {
        while sessions.join_next().await.is_some() {}
    })
    .await;

    match drained {
        Ok(()) => {
            tracing::info!(
                sessions = live,
                "all sessions finished before the drain deadline"
            );
            true
        }
        Err(_) => {
            // `shutdown` aborts every remaining task and waits for the
            // aborts to land, so this returns having actually stopped them
            // rather than leaving them running into process exit.
            let stuck = sessions.len();
            sessions.shutdown().await;
            tracing::warn!(
                aborted = stuck,
                deadline_secs = deadline.as_secs(),
                "drain deadline passed; aborted the sessions still running"
            );
            false
        }
    }
}

/// Removes the socket file, but only when this process is the one that
/// created it.
///
/// Idempotent, and deliberately tolerant of the file already being gone: a
/// second SIGTERM arriving during shutdown, or an operator deleting the
/// socket by hand, must not turn into a failed exit.
pub fn unlink_owned_socket(source: &ListenerSource) {
    let Some(path) = source.owned_path() else {
        tracing::debug!("socket was activated by a supervisor; leaving the path in place");
        return;
    };

    match std::fs::remove_file(path) {
        Ok(()) => tracing::info!(path = %path.display(), "removed the socket this process bound"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(path = %path.display(), "socket already gone at shutdown");
        }
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                error = %error,
                "failed to remove the socket; a later start will treat it as stale and replace it"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    use tokio::net::UnixListener;

    use super::*;

    fn temp_socket_path() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        PathBuf::from(format!(
            "/tmp/liam-shutdown-{}-{unique}",
            std::process::id()
        ))
    }

    const CHILD_MARKER: &str = "LIAM_SIGNAL_ORDER_CHILD";
    const CHILD_TEST: &str = "transport::shutdown::tests::a_stop_during_listener_setup_is_held";
    const HELD_LINE: &str = "held SIGTERM";
    /// Longest the child may take to see the signal it sent itself.
    const HELD_DEADLINE: Duration = Duration::from_secs(10);
    /// Longest the parent waits for the child process to end.
    const CHILD_DEADLINE: Duration = Duration::from_secs(60);

    /// A SIGTERM raised while the listener is still being set up must be held
    /// for the shutdown. Run in a child of the test binary: with the handlers
    /// installed after setup, the signal would end the process, and a signal
    /// disposition is process-wide, so it cannot be tried inside the shared
    /// test process.
    #[tokio::test]
    async fn a_stop_during_listener_setup_is_held() {
        if std::env::var_os(CHILD_MARKER).is_none() {
            return;
        }

        let (signals, ()) = watch_signals_before(|| async {
            // SAFETY: raising a signal at this process has no preconditions.
            let raised = unsafe { libc::raise(libc::SIGTERM) };
            assert_eq!(raised, 0);
            Ok::<(), std::io::Error>(())
        })
        .await
        .expect("setup");
        let trigger = tokio::time::timeout(HELD_DEADLINE, signals.expect("install").wait()).await;

        println!("{HELD_LINE}: {trigger:?}");
    }

    #[test]
    fn a_stop_during_listener_setup_runs_the_ordered_shutdown_path() {
        // Arrange
        let child = Command::new(std::env::current_exe().expect("test binary"))
            .args([CHILD_TEST, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD_MARKER, "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("start the child test");
        let pid = child.id();
        let (done, finished) = std::sync::mpsc::channel::<()>();
        let watchdog = std::thread::spawn(move || {
            if finished.recv_timeout(CHILD_DEADLINE)
                == Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            {
                // SAFETY: the child is not reaped before `done` is dropped.
                unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
            }
        });

        // Act
        let output = child.wait_with_output().expect("wait for the child");
        drop(done);
        watchdog.join().expect("watchdog");

        // Assert
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{:?}\n{stdout}{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains(&format!("{HELD_LINE}: Ok(Term)")),
            "{stdout}"
        );
    }

    #[test]
    fn each_trigger_is_named_by_its_signal() {
        // Act / Assert
        assert_eq!(Trigger::Term.as_str(), "SIGTERM");
        assert_eq!(Trigger::Int.as_str(), "SIGINT");
    }

    #[test]
    fn the_drain_deadline_stays_under_the_supervisor_grace_period() {
        // Given the shipped launchd job's ExitTimeOut
        const PLIST: &str = include_str!("../../../../packaging/dev.protocortex.liamd.plist");
        let after = PLIST
            .split_once("<key>ExitTimeOut</key>")
            .expect("the plist must set ExitTimeOut")
            .1;
        let open = after.find("<integer>").expect("integer value") + "<integer>".len();
        let close = after.find("</integer>").expect("closed integer");
        let grace: u64 = after[open..close]
            .trim()
            .parse()
            .expect("numeric ExitTimeOut");

        // Then the drain finishes inside it. Draining longer than the grace
        // means launchd SIGKILLs mid-drain, which loses the work the drain
        // exists to protect.
        assert!(
            DEFAULT_DRAIN_DEADLINE.as_secs() < grace,
            "drain deadline {}s must stay under the launchd grace {grace}s",
            DEFAULT_DRAIN_DEADLINE.as_secs()
        );
    }

    #[tokio::test]
    async fn draining_with_no_sessions_returns_immediately() {
        // Given no live sessions
        let mut sessions: JoinSet<()> = JoinSet::new();

        // When drained
        let completed = drain(&mut sessions, Duration::from_secs(30)).await;

        // Then it reports a clean drain without waiting on the deadline
        assert!(completed);
    }

    #[tokio::test]
    async fn an_in_flight_session_is_allowed_to_finish() {
        // Given a session that finishes well within the deadline
        let mut sessions = JoinSet::new();
        let (tx, rx) = tokio::sync::oneshot::channel();
        sessions.spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let _ = tx.send("finished");
        });

        // When drained with room to spare
        let completed = drain(&mut sessions, Duration::from_secs(30)).await;

        // Then it ran to completion rather than being cut off
        assert!(completed, "a short session must finish inside the deadline");
        assert_eq!(
            rx.await.expect("the session must have completed its work"),
            "finished"
        );
    }

    #[tokio::test]
    async fn a_session_outliving_the_deadline_is_aborted_rather_than_hanging() {
        // Given a session that would never end on its own
        let mut sessions = JoinSet::new();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        sessions.spawn(async move {
            // Far longer than any deadline this test would use.
            tokio::time::sleep(Duration::from_secs(3600)).await;
            let _ = tx.send(());
        });

        // When the deadline passes
        let completed = drain(&mut sessions, Duration::from_millis(50)).await;

        // Then drain reports the deadline was missed, returns instead of
        // hanging, and the task is genuinely gone rather than still running.
        assert!(!completed, "an endless session must miss the deadline");
        assert!(sessions.is_empty(), "aborted sessions must be reaped");
        assert!(
            rx.await.is_err(),
            "the aborted session must never have completed its work"
        );
    }

    #[tokio::test]
    async fn a_bound_socket_is_unlinked_on_shutdown() {
        // Given a socket this process bound itself
        let path = temp_socket_path();
        let listener = UnixListener::bind(&path).expect("bind");
        let source = ListenerSource::Bound {
            listener,
            path: path.clone(),
        };
        assert!(path.exists());

        // When shutting down
        unlink_owned_socket(&source);

        // Then the path is gone, since we own it
        assert!(!path.exists(), "a Bound socket must be unlinked");
    }

    #[tokio::test]
    async fn an_activated_socket_is_left_in_place_on_shutdown() {
        // Given a listener standing in for one a supervisor handed over
        let path = temp_socket_path();
        let listener = UnixListener::bind(&path).expect("bind");
        let source = ListenerSource::Activated(listener);
        assert!(path.exists());

        // When shutting down
        unlink_owned_socket(&source);

        // Then the path survives: launchd owns it, and removing it would
        // leave the supervisor holding a descriptor for a name that is gone.
        assert!(
            path.exists(),
            "an Activated socket must never be unlinked by this process"
        );

        drop(source);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn unlinking_twice_is_safe() {
        // Given a Bound socket already unlinked once, which is what a second
        // SIGTERM arriving during shutdown looks like
        let path = temp_socket_path();
        let listener = UnixListener::bind(&path).expect("bind");
        let source = ListenerSource::Bound {
            listener,
            path: path.clone(),
        };
        unlink_owned_socket(&source);

        // When it runs again
        unlink_owned_socket(&source);

        // Then it neither panics nor reports failure
        assert!(!path.exists());
    }
}
