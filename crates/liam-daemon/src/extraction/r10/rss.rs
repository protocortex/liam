// SPDX-License-Identifier: Apache-2.0
//! Peak resident memory of a process and everything it started, for the
//! real-model tier. The sampler reads at start, on a fixed interval and at
//! stop, so a short spike between two reads is missed: the figure is a lower
//! bound on the true peak.

use std::collections::BTreeSet;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

/// Resident bytes of `root` plus every process descended from it.
fn tree_rss_bytes(system: &mut System, root: Pid) -> u64 {
    system.refresh_processes_specifics(
        ProcessesToUpdate::All,
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let mut members = BTreeSet::from([root]);
    loop {
        let before = members.len();
        // On Linux sysinfo lists each thread as a process that shares its
        // owner's memory, which would count that memory once per thread.
        for (pid, process) in system
            .processes()
            .iter()
            .filter(|(_, process)| process.thread_kind().is_none())
        {
            if process
                .parent()
                .is_some_and(|parent| members.contains(&parent))
            {
                members.insert(*pid);
            }
        }
        if members.len() == before {
            break;
        }
    }
    members
        .iter()
        .filter_map(|pid| system.process(*pid))
        .map(sysinfo::Process::memory)
        .sum()
}

/// Samples in a background thread until stopped.
pub(super) struct PeakRss {
    stop: Sender<()>,
    sampler: JoinHandle<u64>,
}

impl PeakRss {
    /// Takes the first sample before returning.
    pub(super) fn start(root: Pid) -> Self {
        let (stop, stopped) = mpsc::channel();
        let mut system = System::new();
        let mut peak = tree_rss_bytes(&mut system, root);
        let sampler = std::thread::spawn(move || {
            while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(SAMPLE_INTERVAL) {
                peak = peak.max(tree_rss_bytes(&mut system, root));
            }
            peak.max(tree_rss_bytes(&mut system, root))
        });
        Self { stop, sampler }
    }

    /// Takes a last sample and returns the largest of all of them, in bytes.
    pub(super) fn stop(self) -> u64 {
        // The sampler ends on a closed channel as well, so a failed send is
        // the same as a delivered one.
        let _ = self.stop.send(());
        self.sampler
            .join()
            .expect("the RSS sampler thread panicked")
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};

    use rand_chacha::rand_core::{RngCore, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    const BYTES_PER_MB: u64 = 1 << 20;
    const ALLOC_MB: u64 = 64;
    /// How far above the allocation the measured growth may sit: the child's
    /// own buffers and allocator overhead.
    const SLACK_MB: u64 = 32;
    const HOLD: Duration = Duration::from_millis(300);
    const ALLOC_ENV: &str = "R10_RSS_ALLOC_MB";
    const READY_LINE: &str = "alloc child ready";
    const ENTRY_TEST: &str = "extraction::r10::rss::tests::alloc_child_entry";
    const CHILD_SEED: u64 = 7;

    /// Runs only as a child of the test below: announces itself, waits for one
    /// byte, then holds `R10_RSS_ALLOC_MB` of memory for a moment.
    #[test]
    fn alloc_child_entry() {
        let Ok(megabytes) = std::env::var(ALLOC_ENV) else {
            return;
        };
        println!("{READY_LINE}");
        std::io::stdout().flush().unwrap();
        let mut release = [0u8; 1];
        std::io::Read::read_exact(&mut std::io::stdin(), &mut release).unwrap();
        // Random bytes: a zero-filled or repeating block can be compressed by
        // the OS and would not count as resident.
        let mut block = vec![0u8; (megabytes.parse::<u64>().unwrap() * BYTES_PER_MB) as usize];
        ChaCha8Rng::seed_from_u64(CHILD_SEED).fill_bytes(&mut block);
        std::thread::sleep(HOLD);
        std::hint::black_box(&block);
    }

    #[test]
    fn the_peak_covers_the_memory_a_child_holds() {
        // Arrange
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", ENTRY_TEST, "--nocapture", "--test-threads=1"])
            .env(ALLOC_ENV, ALLOC_MB.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        // The libtest harness prints its own text on the same line first.
        while !line.contains(READY_LINE) {
            line.clear();
            assert_ne!(output.read_line(&mut line).unwrap(), 0, "the child exited");
        }
        let root = Pid::from_u32(child.id());
        let baseline = tree_rss_bytes(&mut System::new(), root);
        let sampler = PeakRss::start(root);

        // Act
        child.stdin.take().unwrap().write_all(&[1]).unwrap();
        child.wait().unwrap();
        let peak = sampler.stop();

        // Assert
        let grown = peak.saturating_sub(baseline);
        assert!(grown >= ALLOC_MB * BYTES_PER_MB, "grew by {grown} bytes");
        assert!(
            grown < (ALLOC_MB + SLACK_MB) * BYTES_PER_MB,
            "grew by {grown} bytes"
        );
    }

    #[test]
    fn a_run_shorter_than_one_interval_still_reports_memory() {
        // Arrange
        let sampler = PeakRss::start(Pid::from_u32(std::process::id()));

        // Act
        let peak = sampler.stop();

        // Assert
        assert!(peak > 0);
    }

    #[test]
    fn a_process_that_does_not_exist_holds_nothing() {
        // Arrange
        let mut system = System::new();

        // Act
        let bytes = tree_rss_bytes(&mut system, Pid::from_u32(u32::MAX));

        // Assert
        assert_eq!(bytes, 0);
    }
}
