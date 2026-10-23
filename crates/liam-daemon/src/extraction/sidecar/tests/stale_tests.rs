use std::os::unix::fs::symlink;
use std::os::unix::process::ExitStatusExt;
use std::sync::atomic::AtomicBool;

use super::*;
use crate::extraction::sidecar::stale::{
    self, effective_uid, ProcessInfo, ProcessTable, SweepOutcome,
};

const MATCHING_START: u64 = 777;
const RECORDED_PORT: u16 = 4242;

struct Scripted(Option<ProcessInfo>);

impl ProcessTable for Scripted {
    fn inspect(&self, _pid: u32) -> Option<ProcessInfo> {
        self.0.clone()
    }
}

/// A fake server started outside any guard. A thread reaps it, so it is truly
/// gone once it dies; it is killed on drop if still running.
struct Orphan {
    pid: u32,
    status: Arc<Mutex<Option<ExitStatus>>>,
}

impl Orphan {
    fn start() -> Self {
        let (factory, _) = fake("", None);
        let launch = LaunchParams {
            addr: SocketAddr::new(LOOPBACK.parse().unwrap(), free_port()),
            alias: "orphan",
            probe_key: "orphan-probe",
            key: "orphan",
        };
        let mut child = factory(&launch).spawn().unwrap();
        let (pid, status) = (child.id(), Arc::new(Mutex::new(None)));
        let slot = Arc::clone(&status);
        std::thread::spawn(move || *slot.lock().unwrap() = child.wait().ok());
        Self { pid, status }
    }

    fn pid(&self) -> u32 {
        self.pid
    }

    fn running(&self) -> bool {
        self.status.lock().unwrap().is_none()
    }

    fn exit_status(&self) -> ExitStatus {
        let give_up = Instant::now() + GONE_WAIT;
        while self.running() && Instant::now() < give_up {
            std::thread::sleep(GONE_POLL);
        }
        self.status.lock().unwrap().expect("the orphan exited")
    }
}

impl Drop for Orphan {
    fn drop(&mut self) {
        if self.running() {
            // SAFETY: kills our own child, which the reaper thread collects.
            unsafe { libc::kill(self.pid as libc::pid_t, libc::SIGKILL) };
        }
    }
}

fn info(exe: &str, start_time: u64, port: Option<u16>) -> Option<ProcessInfo> {
    Some(ProcessInfo {
        exe: Some(exe.into()),
        start_time,
        port,
        uid: Some(effective_uid()),
    })
}

fn our_exe() -> PathBuf {
    std::env::current_exe().unwrap()
}

fn our_exe_text() -> String {
    our_exe().to_string_lossy().into_owned()
}

fn matching_table() -> Scripted {
    Scripted(info(&our_exe_text(), MATCHING_START, Some(RECORDED_PORT)))
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
    let (dir, orphan) = (TempDir::new().unwrap(), Orphan::start());
    let pidfile = write_record(&dir, orphan.pid());

    // Act
    let outcome = stale::sweep(&pidfile, &our_exe(), &matching_table());

    // Assert
    assert_eq!(outcome, SweepOutcome::Killed(orphan.pid()));
    assert!(!pidfile.exists());
    assert_eq!(orphan.exit_status().signal(), Some(libc::SIGKILL));
}

#[test]
fn a_process_that_cannot_be_verified_as_ours_is_never_killed() {
    // Arrange
    let exe = our_exe_text();
    let other_user = ProcessInfo {
        uid: Some(effective_uid() + 1),
        ..info(&exe, MATCHING_START, Some(RECORDED_PORT)).unwrap()
    };
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
        ("another user's process", Some(other_user)),
        (
            "unreadable exe",
            Some(ProcessInfo {
                exe: None,
                start_time: MATCHING_START,
                port: None,
                uid: Some(effective_uid()),
            }),
        ),
    ];

    for (name, process) in cases {
        let (dir, orphan) = (TempDir::new().unwrap(), Orphan::start());
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
    let (dir, orphan) = (TempDir::new().unwrap(), Orphan::start());
    let pidfile = write_record(&dir, orphan.pid());
    let table = Scripted(info(&our_exe_text(), MATCHING_START, None));

    // Act
    let outcome = stale::sweep(&pidfile, &our_exe(), &table);

    // Assert
    assert_eq!(outcome, SweepOutcome::Killed(orphan.pid()));
}

#[test]
fn this_process_and_its_parent_are_never_killed_even_when_the_record_matches() {
    // Arrange
    // SAFETY: `getppid` has no preconditions.
    let parent = unsafe { libc::getppid() } as u32;
    let dir = TempDir::new().unwrap();

    for pid in [std::process::id(), parent] {
        let pidfile = write_record(&dir, pid);

        // Act
        let outcome = stale::sweep(&pidfile, &our_exe(), &matching_table());

        // Assert
        assert_eq!(outcome, SweepOutcome::Cleared, "pid {pid}");
    }
}

#[test]
fn a_pidfile_owned_by_someone_else_is_not_trusted() {
    // Arrange
    let (dir, orphan) = (TempDir::new().unwrap(), Orphan::start());
    let pidfile = write_record(&dir, orphan.pid());
    let not_the_owner = effective_uid() + 1;

    // Act
    let outcome = stale::sweep_as(
        &pidfile,
        &our_exe(),
        &matching_table(),
        not_the_owner,
        &|_| false,
    );

    // Assert
    assert_eq!(outcome, SweepOutcome::Cleared);
    assert!(orphan.running());
}

#[test]
fn a_symlinked_pidfile_is_unlinked_not_followed() {
    // Arrange
    let (dir, orphan) = (TempDir::new().unwrap(), Orphan::start());
    let target = dir.path().join("real.pid");
    Pidfile {
        pid: orphan.pid(),
        start_time: MATCHING_START,
        port: RECORDED_PORT,
    }
    .write(&target)
    .unwrap();
    let link = dir.path().join("sidecar.pid");
    symlink(&target, &link).unwrap();

    // Act
    let outcome = stale::sweep(&link, &our_exe(), &matching_table());

    // Assert
    assert_eq!(outcome, SweepOutcome::Cleared);
    assert!(orphan.running());
    assert!(link.symlink_metadata().is_err(), "the link was kept");
    assert!(target.exists(), "the link's target was removed");
}

#[test]
fn writing_a_pidfile_never_replaces_or_follows_an_existing_path() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let record = Pidfile {
        pid: 1,
        start_time: 1,
        port: 1,
    };
    let existing = dir.path().join("existing.pid");
    std::fs::write(&existing, "planted").unwrap();
    let target = dir.path().join("victim");
    std::fs::write(&target, "precious").unwrap();
    let link = dir.path().join("link.pid");
    symlink(&target, &link).unwrap();

    // Act
    let over_file = record.write(&existing);
    let over_link = record.write(&link);

    // Assert
    assert!(over_file.is_err() && over_link.is_err());
    assert_eq!(std::fs::read_to_string(&existing).unwrap(), "planted");
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "precious");
}

#[test]
fn a_missing_or_garbled_pidfile_kills_nothing() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let pidfile = dir.path().join("sidecar.pid");

    // Act
    let missing = stale::sweep(&pidfile, &our_exe(), &matching_table());
    std::fs::write(&pidfile, "0 777 4242").unwrap();
    let garbled = stale::sweep(&pidfile, &our_exe(), &matching_table());

    // Assert
    assert_eq!(missing, SweepOutcome::NoPidfile);
    assert_eq!(garbled, SweepOutcome::Cleared);
    assert!(!pidfile.exists());
}

/// Reports whether the stale process was already gone when RAM was read.
struct RamAfterSweep {
    stale_pid: u32,
    was_gone: Arc<AtomicBool>,
}

impl FreeRam for RamAfterSweep {
    fn free_mb(&self) -> u64 {
        // SAFETY: signal 0 only checks that the pid exists.
        let alive = unsafe { libc::kill(self.stale_pid as libc::pid_t, 0) } == 0;
        self.was_gone.store(!alive, Ordering::SeqCst);
        ENOUGH_RAM_MB
    }
}

#[tokio::test]
async fn start_clears_a_matching_stale_sidecar_before_it_reads_free_ram() {
    // Arrange
    let (dir, orphan) = (TempDir::new().unwrap(), Orphan::start());
    let pidfile = write_record(&dir, orphan.pid());
    let (factory, _) = fake("", None);
    let was_gone = Arc::new(AtomicBool::new(false));
    let scripted = Deps {
        processes: Box::new(matching_table()),
        pidfile,
        free_ram: Box::new(RamAfterSweep {
            stale_pid: orphan.pid(),
            was_gone: Arc::clone(&was_gone),
        }),
        ..deps(vec![free_port()])
    };

    // Act
    let guard = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, scripted).await;

    // Assert
    assert!(guard.unwrap().is_alive());
    assert!(!orphan.running());
    assert!(
        was_gone.load(Ordering::SeqCst),
        "RAM was read before the stale exit"
    );
}

#[tokio::test]
async fn the_written_pidfile_is_private_and_really_identifies_the_child() {
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

    // Act
    let mode = std::fs::metadata(&pidfile).unwrap().permissions().mode() & 0o777;
    let outcome = stale::sweep_as(
        &pidfile,
        &our_exe(),
        &stale::SystemProcesses,
        effective_uid(),
        &|_| false,
    );

    // Assert
    assert_eq!(mode, 0o600);
    assert_eq!(outcome, SweepOutcome::Killed(guard.pid()));
    drop(guard);
    assert!(!pidfile.exists());
}

#[tokio::test]
async fn a_child_the_table_cannot_describe_fails_the_start_and_is_killed() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let (pid_file, pidfile) = (dir.path().join("child-pid"), dir.path().join("sidecar.pid"));
    let (factory, spawns) = fake("", Some(pid_file.clone()));
    let blind = Deps {
        processes: Box::new(Scripted(None)),
        pidfile: pidfile.clone(),
        ..deps(vec![free_port()])
    };

    // Act
    let result = SidecarGuard::start_with(&config(START_TIMEOUT_SECS), factory, blind).await;

    // Assert
    assert!(matches!(result, Err(SidecarError::Setup(_))));
    assert_eq!(spawns.load(Ordering::SeqCst), 1);
    assert!(!pidfile.exists());
    // Killed at once, the child may never have written its pid.
    let written = std::fs::read_to_string(&pid_file).ok();
    if let Some(pid) = written.and_then(|text| text.parse::<u32>().ok()) {
        assert!(wait_until_gone(pid));
    }
}

#[tokio::test]
async fn a_second_start_sharing_the_pidfile_leaves_the_first_guard_alone() {
    // Arrange
    let (factory_a, _) = fake("", None);
    let first = started(factory_a).await;
    let pidfile = deps(vec![]).pidfile;
    let (factory_b, _) = fake("", None);

    // Act
    let second = SidecarGuard::start_with(
        &config(START_TIMEOUT_SECS),
        factory_b,
        deps(vec![free_port()]),
    )
    .await;

    // Assert
    assert!(matches!(second, Err(SidecarError::Setup(_))));
    assert!(
        first.is_alive(),
        "the live guard's child was swept as stale"
    );
    assert_eq!(stale::Pidfile::recorded_pid(&pidfile), Some(first.pid()));
    drop(first);
    assert!(!pidfile.exists());
}

#[tokio::test]
async fn dropping_a_guard_whose_child_and_pidfile_are_already_gone_does_not_panic() {
    // Arrange
    let (factory, _) = fake("", None);
    let guard = started(factory).await;
    // SAFETY: kills our own child.
    unsafe { libc::kill(guard.pid() as libc::pid_t, libc::SIGKILL) };
    std::fs::remove_file(deps(vec![]).pidfile).unwrap();

    // Act / Assert
    drop(guard);
}
