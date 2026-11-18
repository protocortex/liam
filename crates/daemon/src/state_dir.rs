// SPDX-License-Identifier: Apache-2.0
//! Keeps the directories and files LIAM owns readable by the owner only.
//!
//! New directories and files are created private (see `liam_log::private_fs`).
//! This module covers what already exists: a state directory an older start
//! left at the umask's mode, and the database files libsql creates itself.

use std::path::{Path, PathBuf};

use liam_log::private_fs::narrow_to_owner;

/// The folder under the home directory that holds LIAM's default state.
const STATE_DIR_NAME: &str = ".liam";
/// Suffixes of the files SQLite keeps beside the database.
const DATABASE_SIDE_FILES: [&str; 2] = ["-wal", "-shm"];

/// What `secure_dir` did about one existing directory.
#[derive(Debug, PartialEq, Eq)]
pub enum DirOutcome {
    /// Already owner-only, or narrower than the default, so it was not touched.
    Untouched,
    /// Was wider, and is now owner-only. Holds the old mode.
    Tightened(u32),
    /// Wider, but not LIAM's own directory, so it was left alone. Holds the mode.
    WideAndNotOwned(u32),
}

/// The directory LIAM keeps its state in by default.
pub fn default_state_dir(home: &Path) -> PathBuf {
    home.join(STATE_DIR_NAME)
}

/// Narrows `dir` to owner-only when it lies inside `state_root`, the only
/// place LIAM can be sure holds nothing but its own files. A directory the
/// user pointed elsewhere may hold other files that others need, so it is
/// reported and left as it is. A directory that does not exist yet is left for
/// its creator, which makes it private.
pub fn secure_dir(dir: &Path, state_root: &Path) -> std::io::Result<DirOutcome> {
    if !dir.is_dir() {
        return Ok(DirOutcome::Untouched);
    }
    if dir.starts_with(state_root) {
        return Ok(narrow_to_owner(dir)?.map_or(DirOutcome::Untouched, DirOutcome::Tightened));
    }
    Ok(wide_mode(dir)?.map_or(DirOutcome::Untouched, DirOutcome::WideAndNotOwned))
}

#[cfg(unix)]
fn wide_mode(dir: &Path) -> std::io::Result<Option<u32>> {
    use std::os::unix::fs::PermissionsExt;
    const GROUP_AND_OTHER: u32 = 0o077;

    let mode = std::fs::metadata(dir)?.permissions().mode() & 0o7777;
    Ok((mode & GROUP_AND_OTHER != 0).then_some(mode))
}

#[cfg(not(unix))]
fn wide_mode(_dir: &Path) -> std::io::Result<Option<u32>> {
    Ok(None)
}

/// Secures each directory and logs what changed. A failure is a warning: the
/// daemon runs with the modes it found rather than refusing to start.
pub fn secure_dirs(dirs: &[&Path], state_root: &Path) {
    for dir in dirs {
        match secure_dir(dir, state_root) {
            Ok(DirOutcome::Untouched) => {}
            Ok(DirOutcome::Tightened(from)) => {
                tracing::info!(dir = %dir.display(), from = format!("{from:o}"), "restricted the state directory to its owner");
            }
            Ok(DirOutcome::WideAndNotOwned(mode)) => {
                tracing::warn!(dir = %dir.display(), mode = format!("{mode:o}"), "this directory is readable by other users and is not LIAM's own state directory, so its mode was left as it is");
            }
            Err(error) => {
                tracing::warn!(dir = %dir.display(), %error, "could not check the mode of this directory");
            }
        }
    }
}

/// Restricts the database and its `-wal` and `-shm` files to the owner. The
/// store creates them with the umask's mode, so this runs once it has opened.
/// A failure is a warning, not a failed start.
pub fn secure_database_files(database: &Path) {
    let mut files = vec![database.to_path_buf()];
    for suffix in DATABASE_SIDE_FILES {
        let mut name = database.as_os_str().to_owned();
        name.push(suffix);
        files.push(PathBuf::from(name));
    }
    for file in files.iter().filter(|file| file.exists()) {
        if let Err(error) = narrow_to_owner(file) {
            tracing::warn!(file = %file.display(), %error, "could not restrict this database file to its owner");
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn set_mode(path: &Path, mode: u32) {
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
    }

    fn mode_of(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn an_existing_wide_directory_inside_the_state_root_is_tightened() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let root = default_state_dir(home.path());
        fs::create_dir(&root).unwrap();
        set_mode(&root, 0o755);

        // Act
        let outcome = secure_dir(&root, &root).unwrap();

        // Assert
        assert_eq!(outcome, DirOutcome::Tightened(0o755));
        assert_eq!(mode_of(&root), 0o700);
    }

    #[test]
    fn a_narrower_directory_inside_the_state_root_is_left_alone() {
        for mode in [0o700, 0o500] {
            // Arrange
            let home = tempfile::tempdir().unwrap();
            let root = default_state_dir(home.path());
            fs::create_dir(&root).unwrap();
            set_mode(&root, mode);

            // Act
            let outcome = secure_dir(&root, &root).unwrap();

            // Assert
            assert_eq!(outcome, DirOutcome::Untouched, "mode {mode:o}");
            assert_eq!(mode_of(&root), mode);
            set_mode(&root, 0o700);
        }
    }

    #[test]
    fn a_wide_directory_outside_the_state_root_is_reported_and_not_changed() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        set_mode(elsewhere.path(), 0o755);

        // Act
        let outcome = secure_dir(elsewhere.path(), &default_state_dir(home.path())).unwrap();

        // Assert
        assert_eq!(outcome, DirOutcome::WideAndNotOwned(0o755));
        assert_eq!(mode_of(elsewhere.path()), 0o755);
    }

    #[test]
    fn a_directory_that_does_not_exist_is_left_for_its_creator() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let root = default_state_dir(home.path());

        // Act
        let outcome = secure_dir(&root, &root).unwrap();

        // Assert
        assert_eq!(outcome, DirOutcome::Untouched);
        assert!(!root.exists());
    }

    #[test]
    fn the_database_and_its_side_files_are_restricted_and_a_missing_one_is_skipped() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let wal = dir.path().join("liam.db-wal");
        fs::write(&database, b"db").unwrap();
        fs::write(&wal, b"wal").unwrap();
        set_mode(&database, 0o644);
        set_mode(&wal, 0o640);

        // Act
        secure_database_files(&database);

        // Assert
        assert_eq!(mode_of(&database), 0o600);
        assert_eq!(mode_of(&wal), 0o600);
        assert!(!dir.path().join("liam.db-shm").exists());
    }

    #[test]
    fn a_database_file_the_user_narrowed_further_is_left_alone() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        fs::write(&database, b"db").unwrap();
        set_mode(&database, 0o400);

        // Act
        secure_database_files(&database);

        // Assert
        assert_eq!(mode_of(&database), 0o400);
    }
}
