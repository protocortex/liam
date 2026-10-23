// SPDX-License-Identifier: Apache-2.0
//! The pidfile and the sweep that clears a sidecar left behind by a daemon
//! that was killed before `Drop` could run.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
use tracing::{info, warn};

const PIDFILE_MODE: u32 = 0o600;
/// A pidfile is three short numbers; anything longer is not ours.
const PIDFILE_MAX_BYTES: u64 = 256;
const PORT_FLAG: &str = "--port";
const EXIT_WAIT: Duration = Duration::from_secs(2);
const EXIT_POLL: Duration = Duration::from_millis(20);

/// What the OS says about one process.
#[derive(Clone)]
pub struct ProcessInfo {
    pub exe: Option<PathBuf>,
    /// Seconds since the epoch; a reused pid has a different one.
    pub start_time: u64,
    /// Read from argv, so `None` when argv is hidden or truncated.
    pub port: Option<u16>,
    pub uid: Option<u32>,
}

/// The process table, behind a trait so a test can script it.
pub trait ProcessTable: Send {
    fn inspect(&self, pid: u32) -> Option<ProcessInfo>;
}

pub struct SystemProcesses;

impl ProcessTable for SystemProcesses {
    fn inspect(&self, pid: u32) -> Option<ProcessInfo> {
        let pid = Pid::from_u32(pid);
        let mut system = System::new();
        let what = ProcessRefreshKind::nothing()
            .with_exe(UpdateKind::Always)
            .with_cmd(UpdateKind::Always)
            .with_user(UpdateKind::Always);
        system.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, what);
        let process = system.process(pid)?;
        Some(ProcessInfo {
            exe: process.exe().map(Path::to_path_buf),
            start_time: process.start_time(),
            port: port_from_args(process.cmd()),
            uid: process.user_id().map(|uid| **uid),
        })
    }
}

fn port_from_args(args: &[OsString]) -> Option<u16> {
    let at = args.iter().position(|arg| arg == PORT_FLAG)?;
    args.get(at + 1)?.to_str()?.parse().ok()
}

/// Sidecars started by guards alive in this process: one is never stale, even
/// when a second start shares its pidfile.
static CLAIMED: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());

fn claimed() -> std::sync::MutexGuard<'static, BTreeSet<u32>> {
    CLAIMED.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn claim(pid: u32) {
    claimed().insert(pid);
}

pub fn release(pid: u32) {
    claimed().remove(&pid);
}

pub fn effective_uid() -> u32 {
    // SAFETY: `geteuid` has no preconditions and cannot fail.
    unsafe { libc::geteuid() }
}

/// The three facts that identify the sidecar a daemon started.
#[derive(Debug, PartialEq, Eq)]
pub struct Pidfile {
    pub pid: u32,
    pub start_time: u64,
    pub port: u16,
}

enum PidfileRead {
    Missing,
    /// A symlink, a non-file, or a file owned by someone else.
    Untrusted,
    Text(String),
}

impl Pidfile {
    /// Fails if the path exists: the sweep removed any old file first, so a
    /// survivor was planted or belongs to a live guard. A symlink is never
    /// followed.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_NOFOLLOW)
            .mode(PIDFILE_MODE)
            .open(path)?;
        writeln!(file, "{} {} {}", self.pid, self.start_time, self.port)
    }

    /// The pid recorded in `path`, if it holds a readable record of ours.
    pub fn recorded_pid(path: &Path) -> Option<u32> {
        match read_trusted(path, effective_uid()) {
            PidfileRead::Text(text) => Self::parse(&text).map(|record| record.pid),
            PidfileRead::Missing | PidfileRead::Untrusted => None,
        }
    }

    /// A pid of 0 or past `i32::MAX` is rejected: `kill` would read it as a
    /// process group or as -1, which signals every process we may signal.
    fn parse(text: &str) -> Option<Self> {
        let mut fields = text.split_whitespace();
        let pid: u32 = fields.next()?.parse().ok()?;
        let start_time = fields.next()?.parse().ok()?;
        let port = fields.next()?.parse().ok()?;
        let pid_is_safe = (1..=i32::MAX as u32).contains(&pid);
        (pid_is_safe && fields.next().is_none()).then_some(Self {
            pid,
            start_time,
            port,
        })
    }
}

fn read_trusted(path: &Path, uid: u32) -> PidfileRead {
    let opened = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path);
    let file = match opened {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return PidfileRead::Missing,
        Err(_) => return PidfileRead::Untrusted,
    };
    let trusted = file
        .metadata()
        .is_ok_and(|meta| meta.is_file() && meta.uid() == uid);
    let mut text = String::new();
    if trusted
        && file
            .take(PIDFILE_MAX_BYTES)
            .read_to_string(&mut text)
            .is_ok()
    {
        PidfileRead::Text(text)
    } else {
        PidfileRead::Untrusted
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SweepOutcome {
    NoPidfile,
    Killed(u32),
    /// The file was stale, unreadable, untrusted, or named a process that is
    /// not ours.
    Cleared,
    /// The file names a sidecar a guard in this process still owns.
    LiveGuard,
}

/// `sweep_as` for the current user and this process's own guards.
pub fn sweep(pidfile: &Path, server_exe: &Path, table: &dyn ProcessTable) -> SweepOutcome {
    sweep_as(pidfile, server_exe, table, effective_uid(), &|pid| {
        claimed().contains(&pid)
    })
}

/// Kills the recorded process only when it is owned by `uid` and its
/// executable, start time, and (when readable) port all match, then removes
/// the pidfile and waits for the process to exit. A pid that cannot be
/// verified is treated as unrelated, and this process and its parent are never
/// killed.
pub fn sweep_as(
    pidfile: &Path,
    server_exe: &Path,
    table: &dyn ProcessTable,
    uid: u32,
    is_claimed: &dyn Fn(u32) -> bool,
) -> SweepOutcome {
    let record = match read_trusted(pidfile, uid) {
        PidfileRead::Missing => return SweepOutcome::NoPidfile,
        PidfileRead::Untrusted => None,
        PidfileRead::Text(text) => Pidfile::parse(&text),
    };
    if record.as_ref().is_some_and(|record| is_claimed(record.pid)) {
        return SweepOutcome::LiveGuard;
    }
    let ours = record.filter(|record| is_killable(record, server_exe, table, uid));
    // Unlinking a symlink removes the link, never its target.
    if let Err(error) = std::fs::remove_file(pidfile) {
        warn!(%error, "could not remove the stale sidecar pidfile");
    }
    let Some(record) = ours else {
        return SweepOutcome::Cleared;
    };
    info!(
        pid = record.pid,
        "killing a sidecar left by an earlier daemon"
    );
    // SAFETY: `kill` only sends a signal, and `parse` kept the pid in the
    // range of a single process. The identity check just above makes a
    // recycled pid unlikely to match.
    unsafe { libc::kill(record.pid as libc::pid_t, libc::SIGKILL) };
    wait_for_exit(record.pid);
    SweepOutcome::Killed(record.pid)
}

/// The memory check that follows a sweep must see the memory it freed.
fn wait_for_exit(pid: u32) {
    let give_up = Instant::now() + EXIT_WAIT;
    // SAFETY: signal 0 only checks that the pid exists.
    while unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 && Instant::now() < give_up {
        std::thread::sleep(EXIT_POLL);
    }
}

fn is_killable(record: &Pidfile, server_exe: &Path, table: &dyn ProcessTable, uid: u32) -> bool {
    // SAFETY: `getppid` has no preconditions and cannot fail.
    let parent = unsafe { libc::getppid() } as u32;
    if record.pid == std::process::id() || record.pid == parent {
        return false;
    }
    let Some(info) = table.inspect(record.pid) else {
        return false;
    };
    let same_exe = info
        .exe
        .as_deref()
        .is_some_and(|exe| resolved(exe) == resolved(server_exe));
    let same_port = info.port.is_none_or(|port| port == record.port);
    same_exe && info.start_time == record.start_time && same_port && info.uid == Some(uid)
}

fn resolved(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pidfile_round_trips_and_odd_pids_are_refused() {
        // Arrange
        let record = Pidfile {
            pid: 42,
            start_time: 1000,
            port: 8080,
        };

        // Act
        let parsed = Pidfile::parse("42 1000 8080\n");

        // Assert
        assert_eq!(parsed, Some(record));
        for bad in ["0 1 2", "4294967295 1 2", "-1 1 2", "1 2", "1 2 3 4", "x"] {
            assert_eq!(Pidfile::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_system_table_describes_this_process() {
        // Arrange
        let own_exe = std::env::current_exe().unwrap();

        // Act
        let info = SystemProcesses.inspect(std::process::id()).unwrap();

        // Assert
        assert_eq!(info.exe.map(|exe| resolved(&exe)), Some(resolved(&own_exe)));
        assert!(info.start_time > 0);
        assert_eq!(info.uid, Some(effective_uid()));
    }

    #[test]
    fn the_port_is_read_from_argv_only_when_present() {
        // Arrange
        let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();

        // Act / Assert
        assert_eq!(
            port_from_args(&args(&["srv", "--port", "9090"])),
            Some(9090)
        );
        assert_eq!(port_from_args(&args(&["srv", "--port"])), None);
        assert_eq!(port_from_args(&args(&["srv"])), None);
    }
}
