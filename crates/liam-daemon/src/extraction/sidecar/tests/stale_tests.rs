use std::os::unix::process::ExitStatusExt;

use super::*;
use crate::extraction::sidecar::stale::{self, ProcessInfo, ProcessTable, SweepOutcome};

const MATCHING_START: u64 = 777;
const RECORDED_PORT: u16 = 4242;

struct Scripted(Option<ProcessInfo>);

impl ProcessTable for Scripted {
    fn inspect(&self, _pid: u32) -> Option<ProcessInfo> {
        self.0.clone()
    }
}

/// A fake server started outside any guard, killed on drop if still running.
struct Orphan(Child);

impl Orphan {
    fn start() -> Self {
        let (factory, _) = fake("", None);
        let launch = LaunchParams {
            addr: SocketAddr::new(LOOPBACK.parse().unwrap(), free_port()),
            alias: "orphan",
            key: "orphan",
        };
        Self(factory(&launch).spawn().unwrap())
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }

    fn running(&mut self) -> bool {
        matches!(self.0.try_wait(), Ok(None))
    }
}

impl Drop for Orphan {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn info(exe: &str, start_time: u64, port: Option<u16>) -> Option<ProcessInfo> {
    Some(ProcessInfo {
        exe: Some(exe.into()),
        start_time,
        port,
    })
}

fn our_exe() -> PathBuf {
    std::env::current_exe().unwrap()
}

fn our_exe_text() -> String {
    our_exe().to_string_lossy().into_owned()
}

fn write_record(dir: &TempDir, pid: u32) -> PathBuf {
    let path = dir.path().join("sidecar.pid");
    let record = Pidfile {
        pid,
        start_time: MATCHING_START,
        port: RECORDED_PORT,
    };
    record.write(&path).unwrap();
    path
}

#[test]
fn a_stale_sidecar_whose_exe_start_time_and_port_match_is_killed() {
    // Arrange
    let (dir, mut orphan) = (TempDir::new().unwrap(), Orphan::start());
    let pidfile = write_record(&dir, orphan.pid());
    let table = Scripted(info(&our_exe_text(), MATCHING_START, Some(RECORDED_PORT)));

    // Act
    let outcome = stale::sweep(&pidfile, &our_exe(), &table);

    // Assert
    assert_eq!(outcome, SweepOutcome::Killed(orphan.pid()));
    assert!(!pidfile.exists());
    assert_eq!(orphan.0.wait().unwrap().signal(), Some(libc::SIGKILL));
}

#[test]
fn a_process_that_cannot_be_verified_as_ours_is_never_killed() {
    // Arrange
    let exe = our_exe_text();
    let cases = [
        ("dead pid", None),
        (
            "reused pid",
            info(&exe, MATCHING_START + 1, Some(RECORDED_PORT)),
        ),
        (
            "unrelated program",
            info("/bin/sh", MATCHING_START, Some(RECORDED_PORT)),
        ),
        (
            "different port",
            info(&exe, MATCHING_START, Some(RECORDED_PORT + 1)),
        ),
        (
            "unreadable exe",
            Some(ProcessInfo {
                exe: None,
                start_time: MATCHING_START,
                port: None,
            }),
        ),
    ];

    for (name, process) in cases {
        let (dir, mut orphan) = (TempDir::new().unwrap(), Orphan::start());
        let pidfile = write_record(&dir, orphan.pid());

        // Act
        let outcome = stale::sweep(&pidfile, &our_exe(), &Scripted(process));

        // Assert
        assert_eq!(outcome, SweepOutcome::Cleared, "{name}");
        assert!(orphan.running(), "{name}: an unrelated process was killed");
        assert!(!pidfile.exists(), "{name}: the stale file was kept");
    }
}

#[test]
fn an_unreadable_port_is_compared_by_exe_and_start_time_only() {
    // Arrange
    let (dir, mut orphan) = (TempDir::new().unwrap(), Orphan::start());
    let pidfile = write_record(&dir, orphan.pid());
    let table = Scripted(info(&our_exe_text(), MATCHING_START, None));

    // Act
    let outcome = stale::sweep(&pidfile, &our_exe(), &table);

    // Assert
    assert_eq!(outcome, SweepOutcome::Killed(orphan.pid()));
    orphan.0.wait().unwrap();
}

#[test]
fn a_missing_or_garbled_pidfile_kills_nothing() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("sidecar.pid");
    let table = Scripted(info(&our_exe_text(), MATCHING_START, None));

    // Act
    let missing = stale::sweep(&pidfile, &our_exe(), &table);
    std::fs::write(&pidfile, "0 777 4242").unwrap();
    let garbled = stale::sweep(&pidfile, &our_exe(), &table);

    // Assert
    assert_eq!(missing, SweepOutcome::NoPidfile);
    assert_eq!(garbled, SweepOutcome::Cleared);
    assert!(!pidfile.exists());
}

#[tokio::test]
async fn start_kills_a_matching_stale_sidecar_before_it_launches() {
    // Arrange
    let (dir, mut orphan) = (TempDir::new().unwrap(), Orphan::start());
    let pidfile = write_record(&dir, orphan.pid());
    let (factory, _) = fake("", None);
    let scripted = Deps {
        processes: Box::new(Scripted(info(
            &our_exe_text(),
            MATCHING_START,
            Some(RECORDED_PORT),
        ))),
        pidfile,
        ..deps(vec![free_port()])
    };

    // Act
    let guard = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, scripted).await;

    // Assert
    assert!(guard.unwrap().is_alive());
    assert!(!orphan.running());
}

#[tokio::test]
async fn the_pidfile_is_private_names_the_child_and_goes_with_the_guard() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("sidecar.pid");
    let (factory, _) = fake("", None);
    let with_file = Deps {
        pidfile: pidfile.clone(),
        ..deps(vec![free_port()])
    };

    // Act
    let guard = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, with_file)
        .await
        .unwrap();
    let mode = std::fs::metadata(&pidfile).unwrap().permissions().mode() & 0o777;
    let text = std::fs::read_to_string(&pidfile).unwrap();
    let pid = guard.pid();
    drop(guard);

    // Assert
    assert_eq!(mode, 0o600);
    assert_eq!(
        text.split_whitespace().next(),
        Some(pid.to_string().as_str())
    );
    assert!(!pidfile.exists());
}

#[tokio::test]
async fn dropping_a_guard_whose_child_and_pidfile_are_already_gone_does_not_panic() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("sidecar.pid");
    let (factory, _) = fake("", None);
    let with_file = Deps {
        pidfile: pidfile.clone(),
        ..deps(vec![free_port()])
    };
    let guard = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, with_file)
        .await
        .unwrap();
    // SAFETY: kills our own child.
    unsafe { libc::kill(guard.pid() as libc::pid_t, libc::SIGKILL) };
    std::fs::remove_file(&pidfile).unwrap();

    // Act / Assert
    drop(guard);
}
