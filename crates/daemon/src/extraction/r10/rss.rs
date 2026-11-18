// SPDX-License-Identifier: Apache-2.0
//! Peak resident memory of a process and everything it started, for the
//! real-model tier. The sampler reads at start, on a fixed interval and at
//! stop, so a short spike between two reads is missed: the figure is a lower
//! bound on the true peak.

use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use sysinfo::{Pid, System};

use crate::metrics::memory::tree_rss_bytes;

const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

/// Samples in a background thread until stopped.
pub(super) struct PeakRss {
    stop: Sender<()>,
    sampler: JoinHandle<u64>,
}

impl PeakRss {
    /// Takes the first sample before returning.
    pub(super) fn start(root: Pid) -> Self {
        let mut system = System::new();
        Self::start_with(move || tree_rss_bytes(&mut system, root))
    }

    fn start_with(mut sample: impl FnMut() -> u64 + Send + 'static) -> Self {
        let (stop, stopped) = mpsc::channel();
        let mut peak = sample();
        let sampler = std::thread::spawn(move || {
            while let Err(RecvTimeoutError::Timeout) = stopped.recv_timeout(SAMPLE_INTERVAL) {
                peak = peak.max(sample());
            }
            peak.max(sample())
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

    use crate::metrics::memory::{ProcessMemory, SystemMemory};

    use super::*;

    const READY_LINE: &str = "sleeper child ready";
    const CHILD_ENV: &str = "R10_RSS_SLEEPER";
    const ENTRY_TEST: &str = "extraction::r10::rss::tests::sleeper_child_entry";
    const LOW: u64 = 10;
    const HIGH: u64 = 90;
    const AFTER_THE_SPIKE: u64 = 20;
    /// The first, spike and post-spike readings the scripted sampler takes.
    const SCRIPTED_READINGS: usize = 3;
    /// A byte the parent sends to let the sleeper child end.
    const RELEASE_BYTE: u8 = 1;
    /// A pid no process can have, so the tree is empty.
    /// The test process and its sleeper child.
    const PARENT_AND_CHILD: usize = 2;
    const NO_SUCH_PID: u32 = u32::MAX;

    /// Runs only as a child of the test below: announces itself, then waits
    /// for one byte so the parent controls how long it lives.
    #[test]
    fn sleeper_child_entry() {
        if std::env::var(CHILD_ENV).is_err() {
            return;
        }
        println!("{READY_LINE}");
        std::io::stdout().flush().unwrap();
        std::io::Read::read_exact(&mut std::io::stdin(), &mut [0u8; 1]).unwrap();
    }

    // What the OS reports as resident depends on memory pressure, so the
    // magnitude is not asserted against a real process. The sampler's own
    // guarantee is tested with scripted readings, and the process tree with
    // its membership.
    #[test]
    fn the_peak_is_the_largest_reading_not_the_last() {
        // Arrange
        let mut readings = [LOW, HIGH]
            .into_iter()
            .chain(std::iter::repeat(AFTER_THE_SPIKE));
        let (taken, reading_taken) = mpsc::channel();

        // Act
        let sampler = PeakRss::start_with(move || {
            let _ = taken.send(());
            readings.next().unwrap()
        });
        // The first reading, the spike and one reading after it.
        for _ in 0..SCRIPTED_READINGS {
            reading_taken.recv().unwrap();
        }
        let peak = sampler.stop();

        // Assert
        assert_eq!(peak, HIGH);
    }

    #[test]
    fn the_tree_includes_a_child_process() {
        // Arrange
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", ENTRY_TEST, "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
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

        // Act
        let members = SystemMemory::new().resident_bytes().len();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&[RELEASE_BYTE])
            .unwrap();
        child.wait().unwrap();

        // Assert
        assert!(
            members >= PARENT_AND_CHILD,
            "the tree held {members} process(es)"
        );
    }

    #[test]
    fn the_tree_reading_is_at_least_the_parents_own_with_a_child_running() {
        // Arrange
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", ENTRY_TEST, "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        while !line.contains(READY_LINE) {
            line.clear();
            assert_ne!(output.read_line(&mut line).unwrap(), 0, "the child exited");
        }
        let parent = Pid::from_u32(std::process::id());
        let mut system = System::new();

        // Act
        let tree = tree_rss_bytes(&mut system, parent);
        let own = system
            .process(parent)
            .expect("the parent is listed")
            .memory();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&[RELEASE_BYTE])
            .unwrap();
        child.wait().unwrap();

        // Assert
        assert!(tree >= own, "tree {tree} is below the parent's own {own}");
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
        let bytes = tree_rss_bytes(&mut system, Pid::from_u32(NO_SUCH_PID));

        // Assert
        assert_eq!(bytes, 0);
    }
}
