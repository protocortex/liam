// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exclusive advisory lock so only one process opens the store at a time.
//!
//! The plan deliberately keeps plain `liamd` a store-opening stdio server so
//! existing MCP configs keep working, which means a user who also runs
//! `liamd serve` ends up with two processes writing the same libSQL file.
//! `liam-store`'s write mutex only serializes writers WITHIN one process, so
//! without this lock a second store-opening process would still write
//! concurrently at the OS level. This lock makes that impossible instead of
//! merely unlikely.
//!
//! # Why an advisory `flock`, not a PID file
//!
//! A PID file records who holds a lock but not whether they are still
//! alive, so a crash leaves a stale file that a fresh process has to detect
//! and clean up by hand. [`std::fs::File::try_lock`] is `flock`-based on
//! Unix: the OS releases it the moment the holding process exits, for any
//! reason including a crash, so there is never a stale lock left to clean
//! up.
//!
//! # Why per process, not per store open
//!
//! `spawn_gc` in `main.rs` opens a SECOND connection to the same database on
//! purpose, so GC never contends with request handling. This lock guards
//! against a second PROCESS, not a second connection, so it is acquired
//! exactly once, in `run`, before the first `DefaultGraph::open`, and must
//! never be retaken around `spawn_gc`'s connection: doing so would have the
//! process deadlock against itself.
//!
//! # Contract for the stdio proxy (WU-9)
//!
//! The proxy mode opens no store: it only shuttles bytes to the socket a
//! `serve` process already owns, so it must acquire no lock here. A proxy
//! that took this lock would fail to start whenever a `serve` process is
//! already running, which is the one case it exists to support.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Holds the exclusive lock on `<database_path>.lock` for as long as it is
/// alive. Dropping it releases the lock immediately, since the OS unlocks on
/// file close, so callers must bind it to a named variable that lives as
/// long as the store stays open: `let _lock = StoreLock::acquire(path)?;`,
/// never `let _ = ...`, which drops on the spot and releases the lock right
/// away.
#[derive(Debug)]
pub struct StoreLock(
    // Never read: the field exists so the lock is released by `Drop` when
    // the guard goes out of scope, not for its contents.
    #[allow(dead_code)] File,
);

impl StoreLock {
    /// Try to acquire the lock, failing immediately, never blocking, if
    /// another process already holds it.
    pub fn acquire(database_path: &Path) -> anyhow::Result<Self> {
        let lock_path = lock_path_for(database_path);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|source| {
                anyhow::anyhow!(
                    "failed to open store lock file {}: {source}",
                    lock_path.display()
                )
            })?;

        file.try_lock().map_err(|source| {
            anyhow::anyhow!(
                "could not acquire the store lock at {} ({source}): another \
                 liamd process is likely already running against this \
                 database; point your MCP client at `liamd proxy` instead of \
                 starting a second store-opening process",
                lock_path.display()
            )
        })?;

        Ok(Self(file))
    }
}

fn lock_path_for(database_path: &Path) -> PathBuf {
    let mut lock_path = database_path.as_os_str().to_owned();
    lock_path.push(".lock");
    PathBuf::from(lock_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquiring_with_no_lock_held_succeeds() {
        // Arrange: a database path with no lock file yet.
        let dir = tempfile::tempdir().expect("temp dir");
        let database_path = dir.path().join("liam.db");

        // Act
        let result = StoreLock::acquire(&database_path);

        // Assert: the lock is granted.
        assert!(result.is_ok(), "expected the lock to be free: {result:?}");
    }

    #[test]
    fn a_held_lock_fails_fast_and_names_the_file_and_the_proxy() {
        // Arrange: one process (this test) already holds the lock.
        let dir = tempfile::tempdir().expect("temp dir");
        let database_path = dir.path().join("liam.db");
        let _first = StoreLock::acquire(&database_path).expect("first acquisition must succeed");

        // Act: a second acquisition on the same path, in the same process.
        // This is genuinely exclusive across handles in one process (`flock`
        // is per open file description, not per process), so no subprocess
        // is needed to pin this behaviour.
        let result = StoreLock::acquire(&database_path);

        // Assert: it fails immediately and the error names the lock file and
        // points at the proxy as the fix.
        let message = result
            .expect_err("a second acquisition must fail")
            .to_string();
        let lock_path = lock_path_for(&database_path);
        assert!(
            message.contains(&lock_path.display().to_string()),
            "message should name the lock file: {message}"
        );
        assert!(
            message.contains("liamd proxy"),
            "message should point at the proxy as the fix: {message}"
        );
    }

    #[test]
    fn dropping_the_guard_releases_the_lock_for_a_fresh_acquisition() {
        // Arrange: acquire and then drop the lock.
        let dir = tempfile::tempdir().expect("temp dir");
        let database_path = dir.path().join("liam.db");
        let first = StoreLock::acquire(&database_path).expect("first acquisition must succeed");
        drop(first);

        // Act: a fresh acquisition on the same path.
        let result = StoreLock::acquire(&database_path);

        // Assert: it succeeds. This is the in-process stand-in for "a killed
        // holder leaves no stale lock"; the OS-level release on process
        // death is what `flock` gives us and is documented above, not
        // unit-testable here.
        assert!(
            result.is_ok(),
            "expected the lock to be free again: {result:?}"
        );
    }
}
