// SPDX-License-Identifier: Apache-2.0
//! Keeps the directories and files LIAM owns readable by the owner only.
//!
//! New directories and files are created private (see `liam_log::private_fs`).
//! This module covers what already exists: a state directory an older start
//! left at the umask's mode, and the database files libsql creates itself.

use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use liam_log::private_fs::{narrow_dir_to_owner, narrow_to_owner};

/// The folder under the home directory that holds LIAM's default state.
const STATE_DIR_NAME: &str = ".liam";
/// Suffixes of the files SQLite keeps beside the database.
const DATABASE_SIDE_FILES: [&str; 2] = ["-wal", "-shm"];

/// Error kinds that mean the directory is not there, or is not a directory.
const NOT_THERE: [std::io::ErrorKind; 2] = [
    std::io::ErrorKind::NotFound,
    std::io::ErrorKind::NotADirectory,
];

/// What `secure_dir` did about one existing directory.
#[derive(Debug, PartialEq, Eq)]
pub enum DirOutcome {
    /// Already owner-only, narrower than the default, or not there yet, so it
    /// was not touched.
    Untouched,
    /// Was wider, and is now owner-only. Holds the old mode.
    Tightened(u32),
    /// Wider, but not LIAM's own directory, so it was left alone. Holds the mode.
    WideAndNotOwned(u32),
    /// Could not be shown to be LIAM's own directory, so it was left alone.
    Skipped(Skip),
}

/// Why a directory was not checked or changed.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Skip {
    /// The directory or the state root is not an absolute path.
    NotAbsolute,
    /// The path climbs above the filesystem root.
    EscapesRoot,
    /// A component from the state root down is a symbolic link.
    Symlink,
}

impl Skip {
    fn reason(self) -> &'static str {
        match self {
            Self::NotAbsolute => "the path is not absolute",
            Self::EscapesRoot => "the path climbs above the filesystem root",
            Self::Symlink => "a part of the path is a symbolic link",
        }
    }
}

/// The directory LIAM keeps its state in by default.
pub fn default_state_dir(home: &Path) -> PathBuf {
    home.join(STATE_DIR_NAME)
}

/// The state root for a `HOME` value, or `None` when `HOME` is unset or not an
/// absolute path, in which case no directory can be shown to be LIAM's own.
pub fn state_root_from_home(home: Option<OsString>) -> Option<PathBuf> {
    let home = PathBuf::from(home?);
    home.is_absolute().then(|| default_state_dir(&home))
}

/// `path` with `.` and `..` resolved by name, without touching the filesystem.
fn normalise(path: &Path) -> Result<PathBuf, Skip> {
    if !path.is_absolute() {
        return Err(Skip::NotAbsolute);
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                out.push(component);
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return Err(Skip::EscapesRoot);
                }
            }
        }
    }
    Ok(out)
}

/// Whether `root` or any directory between it and `dir` is a symbolic link.
/// `Ok(false)` also covers a part that does not exist. `dir` must be inside
/// `root`.
fn has_symlink_from(root: &Path, dir: &Path) -> std::io::Result<bool> {
    let below = dir.strip_prefix(root).unwrap_or(Path::new(""));
    let mut current = root.to_path_buf();
    let steps = std::iter::once(None).chain(below.components().map(Some));
    for step in steps {
        if let Some(step) = step {
            current.push(step);
        }
        match std::fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => return Ok(true),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        }
    }
    Ok(false)
}

/// Narrows `dir` to owner-only when it lies inside `state_root`, the only
/// place LIAM can be sure holds nothing but its own files. Both paths are
/// resolved by name first, so `..` cannot lead out of the root, and a symbolic
/// link anywhere from the root down is refused rather than followed. A
/// directory the user pointed elsewhere may hold other files that others need,
/// so it is reported and left as it is. A directory that does not exist yet is
/// left for its creator, which makes it private.
pub fn secure_dir(dir: &Path, state_root: &Path) -> std::io::Result<DirOutcome> {
    let (dir, root) = match (normalise(dir), normalise(state_root)) {
        (Ok(dir), Ok(root)) => (dir, root),
        (Err(skip), _) | (_, Err(skip)) => return Ok(DirOutcome::Skipped(skip)),
    };
    if !dir.starts_with(&root) {
        return outside_root(&dir);
    }
    if has_symlink_from(&root, &dir)
        .map_err(|error| failed("check for symbolic links in", error))?
    {
        return Ok(DirOutcome::Skipped(Skip::Symlink));
    }
    match narrow_dir_to_owner(&dir) {
        Ok(changed) => Ok(changed.map_or(DirOutcome::Untouched, DirOutcome::Tightened)),
        Err(error) if NOT_THERE.contains(&error.kind()) => Ok(DirOutcome::Untouched),
        Err(error) => Err(error),
    }
}

/// Keeps the error's kind and says which step failed.
fn failed(step: &str, error: std::io::Error) -> std::io::Error {
    std::io::Error::new(error.kind(), format!("could not {step} the path: {error}"))
}

/// A directory outside the state root is never changed, only reported when wide.
fn outside_root(dir: &Path) -> std::io::Result<DirOutcome> {
    if !dir.is_dir() {
        return Ok(DirOutcome::Untouched);
    }
    let wide = wide_mode(dir).map_err(|error| failed("read the mode of", error))?;
    Ok(wide.map_or(DirOutcome::Untouched, DirOutcome::WideAndNotOwned))
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

/// Directories already warned about, so a directory two settings share, such
/// as the database directory and the socket's parent, is reported once.
fn first_report(dir: &Path) -> bool {
    static REPORTED: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);
    REPORTED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashSet::new)
        .insert(dir.to_path_buf())
}

/// Secures each directory and logs what changed. A failure is a warning: the
/// daemon runs with the modes it found rather than refusing to start.
/// `risk` says what a wide directory exposes, and is added to the warning.
pub fn secure_dirs(dirs: &[&Path], state_root: &Path, risk: &str) {
    for dir in dirs {
        match secure_dir(dir, state_root) {
            Ok(DirOutcome::Untouched) => {}
            Ok(DirOutcome::Tightened(from)) => {
                tracing::info!(dir = %dir.display(), from = format!("{from:o}"), "restricted the state directory to its owner");
            }
            Ok(DirOutcome::WideAndNotOwned(mode)) if first_report(dir) => {
                tracing::warn!(dir = %dir.display(), mode = format!("{mode:o}"), risk, "this directory is readable by other users and is not LIAM's own state directory, so its mode was left as it is");
            }
            Ok(DirOutcome::Skipped(skip)) if first_report(dir) => {
                tracing::warn!(dir = %dir.display(), reason = skip.reason(), risk, "this directory could not be shown to be LIAM's own state directory, so its mode was left as it is");
            }
            Ok(DirOutcome::WideAndNotOwned(_) | DirOutcome::Skipped(_)) => {}
            Err(error) if first_report(dir) => {
                tracing::warn!(dir = %dir.display(), %error, "could not secure this directory");
            }
            Err(_) => {}
        }
    }
}

/// Restricts the database and its `-wal` and `-shm` files to the owner. The
/// store creates them with the process umask's mode, so this is a second
/// layer, run once the store has opened. A symbolic link is refused, not
/// followed. A failure is a warning, not a failed start.
pub fn secure_database_files(database: &Path) {
    let mut files = vec![database.to_path_buf()];
    for suffix in DATABASE_SIDE_FILES {
        let mut name = database.as_os_str().to_owned();
        name.push(suffix);
        files.push(PathBuf::from(name));
    }
    for file in files {
        match narrow_to_owner(&file) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                tracing::warn!(file = %file.display(), %error, "could not restrict this database file to its owner");
            }
            _ => {}
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::os::unix::fs::{symlink, PermissionsExt};

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

    #[test]
    fn a_path_that_climbs_out_of_the_state_root_is_not_changed() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let root = default_state_dir(home.path());
        let shared = home.path().join("shared");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&shared).unwrap();
        set_mode(&shared, 0o755);
        let escaping = root.join("..").join("shared");

        // Act
        let outcome = secure_dir(&escaping, &root).unwrap();

        // Assert
        assert_eq!(outcome, DirOutcome::WideAndNotOwned(0o755));
        assert_eq!(mode_of(&shared), 0o755);
    }

    #[test]
    fn dot_parts_that_stay_inside_the_state_root_are_resolved_and_tightened() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let root = default_state_dir(home.path());
        let log = root.join("log");
        fs::create_dir_all(&log).unwrap();
        set_mode(&log, 0o755);
        let roundabout = root.join(".").join("other").join("..").join("log");

        // Act
        let outcome = secure_dir(&roundabout, &root).unwrap();

        // Assert
        assert_eq!(outcome, DirOutcome::Tightened(0o755));
        assert_eq!(mode_of(&log), 0o700);
    }

    #[test]
    fn a_symlinked_directory_inside_the_state_root_is_not_changed() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let root = default_state_dir(home.path());
        let target = home.path().join("target");
        fs::create_dir(&root).unwrap();
        fs::create_dir_all(target.join("below")).unwrap();
        set_mode(&target, 0o755);
        set_mode(&target.join("below"), 0o755);
        symlink(&target, root.join("link")).unwrap();

        // Act
        let link = secure_dir(&root.join("link"), &root).unwrap();
        let below = secure_dir(&root.join("link").join("below"), &root).unwrap();

        // Assert
        assert_eq!(link, DirOutcome::Skipped(Skip::Symlink));
        assert_eq!(below, DirOutcome::Skipped(Skip::Symlink));
        assert_eq!(mode_of(&target), 0o755);
        assert_eq!(mode_of(&target.join("below")), 0o755);
    }

    #[test]
    fn a_state_root_that_is_a_symlink_is_not_changed() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let root = default_state_dir(home.path());
        let target = home.path().join("target");
        fs::create_dir(&target).unwrap();
        set_mode(&target, 0o755);
        symlink(&target, &root).unwrap();

        // Act
        let outcome = secure_dir(&root, &root).unwrap();

        // Assert
        assert_eq!(outcome, DirOutcome::Skipped(Skip::Symlink));
        assert_eq!(mode_of(&target), 0o755);
    }

    #[test]
    fn a_relative_path_or_state_root_changes_nothing() {
        // Arrange
        let home = tempfile::tempdir().unwrap();
        let root = default_state_dir(home.path());
        fs::create_dir(&root).unwrap();
        set_mode(&root, 0o755);

        // Act
        let relative_dir = secure_dir(Path::new("log"), &root).unwrap();
        let relative_root = secure_dir(&root, Path::new(".liam")).unwrap();

        // Assert
        assert_eq!(relative_dir, DirOutcome::Skipped(Skip::NotAbsolute));
        assert_eq!(relative_root, DirOutcome::Skipped(Skip::NotAbsolute));
        assert_eq!(mode_of(&root), 0o755);
    }

    #[test]
    fn a_home_that_is_unset_or_relative_gives_no_state_root() {
        // Arrange
        let home = std::ffi::OsString::from("/home/alice");

        // Act and Assert
        assert_eq!(state_root_from_home(None), None);
        assert_eq!(state_root_from_home(Some("relative/home".into())), None);
        assert_eq!(
            state_root_from_home(Some(home)),
            Some(PathBuf::from("/home/alice/.liam"))
        );
    }

    #[test]
    fn a_database_file_that_is_a_symlink_is_not_changed() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.db");
        let database = dir.path().join("liam.db");
        fs::write(&real, b"db").unwrap();
        set_mode(&real, 0o644);
        symlink(&real, &database).unwrap();

        // Act
        secure_database_files(&database);

        // Assert
        assert_eq!(mode_of(&real), 0o644);
    }
}
