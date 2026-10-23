// SPDX-License-Identifier: Apache-2.0
//! The pidfile and the sweep that clears a sidecar left behind by a daemon
//! that was killed before `Drop` could run.

use std::ffi::OsString;
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
use tracing::{info, warn};

const PIDFILE_MODE: u32 = 0o600;
const PORT_FLAG: &str = "--port";

/// What the OS says about one process.
#[derive(Clone)]
pub struct ProcessInfo {
    pub exe: Option<PathBuf>,
    /// Seconds since the epoch; a reused pid has a different one.
    pub start_time: u64,
    /// Read from argv, so `None` when argv is hidden or truncated.
    pub port: Option<u16>,
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
            .with_cmd(UpdateKind::Always);
        system.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), true, what);
        let process = system.process(pid)?;
        Some(ProcessInfo {
            exe: process.exe().map(Path::to_path_buf),
            start_time: process.start_time(),
            port: port_from_args(process.cmd()),
        })
    }
}

fn port_from_args(args: &[OsString]) -> Option<u16> {
    let at = args.iter().position(|arg| arg == PORT_FLAG)?;
    args.get(at + 1)?.to_str()?.parse().ok()
}

/// The three facts that identify the sidecar a daemon started.
#[derive(Debug, PartialEq, Eq)]
pub struct Pidfile {
    pub pid: u32,
    pub start_time: u64,
    pub port: u16,
}

impl Pidfile {
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(PIDFILE_MODE)
            .open(path)?;
        writeln!(file, "{} {} {}", self.pid, self.start_time, self.port)
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

#[derive(Debug, PartialEq, Eq)]
pub enum SweepOutcome {
    NoPidfile,
    Killed(u32),
    /// The file was stale, unreadable, or named a process that is not ours.
    Cleared,
}

/// Kills the recorded process only when its executable, start time, and (when
/// readable) port all match, then removes the pidfile. A pid that cannot be
/// verified is treated as unrelated.
pub fn sweep(pidfile: &Path, server_exe: &Path, table: &dyn ProcessTable) -> SweepOutcome {
    let Ok(text) = std::fs::read_to_string(pidfile) else {
        return SweepOutcome::NoPidfile;
    };
    let ours = Pidfile::parse(&text).filter(|record| is_our_sidecar(record, server_exe, table));
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
    SweepOutcome::Killed(record.pid)
}

fn is_our_sidecar(record: &Pidfile, server_exe: &Path, table: &dyn ProcessTable) -> bool {
    let Some(info) = table.inspect(record.pid) else {
        return false;
    };
    let same_exe = info
        .exe
        .as_deref()
        .is_some_and(|exe| resolved(exe) == resolved(server_exe));
    let same_port = info.port.is_none_or(|port| port == record.port);
    same_exe && info.start_time == record.start_time && same_port
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
