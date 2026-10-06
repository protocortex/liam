// SPDX-License-Identifier: Apache-2.0
//! Exclusive advisory lock so only one process opens the store at a time.
//!
//! `liam-store`'s write mutex only serializes writers within one process, so
//! without this lock a second process opening the same libSQL file would still
//! write concurrently at the OS level.
//!
//! It is an advisory `flock`, not a PID file: the OS releases it when the
//! holder exits for any reason, so a crash leaves no stale lock to clean up.
//!
//! A process takes it once, before its first store open. Taking it again in the
//! same process fails against itself.

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
    /// Creates the database's directory, then tries to acquire the lock,
    /// failing immediately, never blocking, if another process already holds
    /// it.
    pub fn acquire(database_path: &Path) -> anyhow::Result<Self> {
        // The lock file lives beside the database, and libSQL will not create
        // the directory for either.
        if let Some(parent) = database_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|source| {
                anyhow::anyhow!(
                    "failed to create the database directory {}: {source}",
                    parent.display()
                )
            })?;
        }
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

        file.try_lock()
            .map_err(|error| anyhow::anyhow!("{}", lock_failure_message(&lock_path, &error)))?;

        Ok(Self(file))
    }
}

fn lock_path_for(database_path: &Path) -> PathBuf {
    let mut lock_path = database_path.as_os_str().to_owned();
    lock_path.push(".lock");
    PathBuf::from(lock_path)
}

/// Explains why `try_lock` failed, in words that fit the actual cause.
///
/// [`std::fs::File::try_lock`] fails two different ways and each calls for
/// a different diagnosis. [`std::fs::TryLockError::WouldBlock`] means the
/// lock is genuinely held: another process got there first. But
/// [`std::fs::TryLockError::Error`] means the OS could not take the lock at
/// all, most often because the filesystem does not support advisory
/// locking, which network mounts (NFS and similar) frequently do not.
/// Telling an operator on a network mount to "stop the other liamd
/// process" sends them looking for a process that does not exist; this
/// function gives that case its own message instead.
///
/// Both branches still refuse to start the store. If we cannot prove no
/// other process holds it, single-writer safety cannot be guaranteed, so
/// failing closed is correct either way. Do not change that outcome when
/// editing this function, only the wording.
fn lock_failure_message(lock_path: &Path, error: &std::fs::TryLockError) -> String {
    use std::fs::TryLockError;

    match error {
        TryLockError::WouldBlock => format!(
            "could not acquire the store lock at {} ({error}): another liam \
             process, the daemon or `liam rebuild`, already holds it",
            lock_path.display()
        ),
        TryLockError::Error(source) => format!(
            "could not acquire the store lock at {} because locking it \
             failed ({source}): the filesystem may not support advisory \
             locking, which network mounts frequently do not; the store \
             is designed to live on a local filesystem, so move it there \
             or use one that supports flock",
            lock_path.display()
        ),
    }
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
    fn a_held_lock_fails_fast_and_names_the_file_and_any_holder() {
        // Arrange: one process (this test) already holds the lock.
        let dir = tempfile::tempdir().expect("temp dir");
        let database_path = dir.path().join("liam.db");
        let _first = StoreLock::acquire(&database_path).expect("first acquisition must succeed");

        // Act: a second acquisition on the same path, in the same process.
        // This is genuinely exclusive across handles in one process (`flock`
        // is per open file description, not per process), so no subprocess
        // is needed to pin this behaviour.
        let result = StoreLock::acquire(&database_path);

        // Assert: it fails immediately, names the lock file, and does not
        // assume which kind of process holds it.
        let message = result
            .expect_err("a second acquisition must fail")
            .to_string();
        let lock_path = lock_path_for(&database_path);
        assert!(
            message.contains(&lock_path.display().to_string()),
            "message should name the lock file: {message}"
        );
        assert!(
            message.contains("another liam process"),
            "message should say another process holds the lock: {message}"
        );
        assert!(
            !message.contains("liamd proxy"),
            "the holder may be `liam rebuild`, so the proxy is not the library's advice: {message}"
        );
    }

    #[test]
    fn acquiring_creates_the_missing_database_directory() {
        // Arrange: a database path whose directory does not exist yet.
        let dir = tempfile::tempdir().expect("temp dir");
        let database_path = dir.path().join("fresh").join("liam.db");

        // Act
        let result = StoreLock::acquire(&database_path);

        // Assert
        assert!(result.is_ok(), "{result:?}");
        assert!(database_path.parent().unwrap().is_dir());
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

    #[test]
    fn would_block_message_names_the_file_and_blames_another_process() {
        // Arrange: the contention branch, the one a normal filesystem can
        // actually produce (see `a_held_lock_fails_fast_and_names_the_file_and_the_fix`
        // for the end-to-end version of this case).
        let lock_path = PathBuf::from("/tmp/example/liam.db.lock");

        // Act
        let message = lock_failure_message(&lock_path, &std::fs::TryLockError::WouldBlock);

        // Assert: names the lock file and points at the other process.
        assert!(
            message.contains(&lock_path.display().to_string()),
            "message should name the lock file: {message}"
        );
        assert!(
            message.contains("another liam process"),
            "message should say another process holds the lock: {message}"
        );
    }

    #[test]
    fn locking_unsupported_message_names_the_file_and_blames_the_filesystem_not_a_process() {
        // Arrange: the "locking itself failed" branch. A normal filesystem
        // never produces this, so it can only be exercised by constructing
        // the error directly and driving the pure message function.
        let lock_path = PathBuf::from("/tmp/example/liam.db.lock");
        let error = std::fs::TryLockError::Error(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "operation not supported on this filesystem",
        ));

        // Act
        let message = lock_failure_message(&lock_path, &error);

        // Assert: names the lock file, blames the filesystem rather than a
        // process, and gives a diagnosis distinct from the contention case.
        assert!(
            message.contains(&lock_path.display().to_string()),
            "message should name the lock file: {message}"
        );
        assert!(
            message.contains("filesystem may not support advisory locking"),
            "message should name the likely cause: {message}"
        );
        assert!(
            !message.contains("another liamd process"),
            "message must not claim another process holds the lock: {message}"
        );
    }
}
