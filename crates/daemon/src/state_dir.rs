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

use liam_log::crypto::REGISTRY_FILE;
use liam_log::private_fs::{narrow_dir_to_owner, narrow_to_owner};
use liam_log::wal::{LOCK_NAME, MANIFEST_NAME, PARQUET_EXTENSION, SEGMENT_EXTENSION};

use crate::config::Config;
use crate::models;

/// The folder under the home directory that holds LIAM's default state.
const STATE_DIR_NAME: &str = ".liam";
/// Suffixes of the files SQLite keeps beside the database.
const DATABASE_SIDE_FILES: [&str; 2] = ["-wal", "-shm"];
/// Extensions of the files LIAM writes beside the database, after the database
/// file name: its sidecars, lock and staging files.
const DATABASE_SIDECAR_EXTENSIONS: [&str; 5] = ["json", "jsonl", "lock", "pid", "tmp"];
/// Extensions of the snapshot files: the copied database and its metadata.
const SNAPSHOT_FILE_EXTENSIONS: [&str; 2] = ["db", "json"];
/// The group and other permission bits.
#[cfg(unix)]
const GROUP_AND_OTHER_BITS: u32 = 0o077;
/// Where a bare file name sits.
const CURRENT_DIR: &str = ".";
/// How the empty parent of a bare file name is shown.
const CURRENT_DIR_LABEL: &str = ".";

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
            Self::Symlink => "a part of the path is a symbolic link, so if you want the real directory owner-only, run `chmod 700` on it yourself",
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

    let mode = std::fs::metadata(dir)?.permissions().mode() & 0o7777;
    Ok((mode & GROUP_AND_OTHER_BITS != 0).then_some(mode))
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

/// `dir` as written, with the empty parent of a bare file name shown as the
/// current directory.
fn dir_label(dir: &Path) -> String {
    if dir.as_os_str().is_empty() {
        CURRENT_DIR_LABEL.to_string()
    } else {
        dir.display().to_string()
    }
}

/// The directory a relative `dir` resolves against, or `None` for an absolute
/// one.
fn resolved_against(dir: &Path, cwd: Option<&Path>) -> Option<String> {
    if dir.is_absolute() {
        return None;
    }
    cwd.map(|cwd| cwd.display().to_string())
}

/// Secures each directory and logs what changed. A failure is a warning: the
/// daemon runs with the modes it found rather than refusing to start.
/// `risk` says what a wide directory exposes, and is added to the warning.
pub fn secure_dirs(dirs: &[&Path], state_root: &Path, risk: &str) {
    for dir in dirs {
        match secure_dir(dir, state_root) {
            Ok(DirOutcome::Untouched) => {}
            Ok(DirOutcome::Tightened(from)) => {
                tracing::info!(dir = %dir_label(dir), from = format!("{from:o}"), "restricted the state directory to its owner");
            }
            Ok(DirOutcome::WideAndNotOwned(mode)) if first_report(dir) => {
                tracing::warn!(dir = %dir_label(dir), mode = format!("{mode:o}"), risk, "this directory is readable by other users and is not LIAM's own state directory, so its mode was left as it is");
            }
            Ok(DirOutcome::Skipped(skip)) if first_report(dir) => {
                let cwd = std::env::current_dir().ok();
                tracing::warn!(dir = %dir_label(dir), resolves_against = resolved_against(dir, cwd.as_deref()), reason = skip.reason(), risk, "this directory could not be shown to be LIAM's own state directory, so its mode was left as it is");
            }
            Ok(DirOutcome::WideAndNotOwned(_) | DirOutcome::Skipped(_)) => {}
            Err(error) if first_report(dir) => {
                tracing::warn!(dir = %dir_label(dir), %error, "could not secure this directory");
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

/// Whether `name` is a sidecar the daemon writes next to the database `db_name`,
/// such as `liam.db.embedder.json`. `liam.toml` and other files do not match.
fn is_database_sidecar(db_name: &str, name: &str) -> bool {
    name.strip_prefix(db_name)
        .and_then(|rest| rest.strip_prefix('.'))
        .is_some_and(|rest| has_extension(rest, &DATABASE_SIDECAR_EXTENSIONS))
}

/// Whether `name` is a file the event log keeps in its directory.
fn is_log_file(name: &str) -> bool {
    [MANIFEST_NAME, LOCK_NAME, REGISTRY_FILE].contains(&name)
        || has_extension(name, &[SEGMENT_EXTENSION, PARQUET_EXTENSION])
}

fn is_snapshot_file(name: &str) -> bool {
    has_extension(name, &SNAPSHOT_FILE_EXTENSIONS)
}

fn has_extension(name: &str, extensions: &[&str]) -> bool {
    Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extensions.contains(&extension))
}

/// What one sweep over a directory did.
#[derive(Debug, Default, PartialEq, Eq)]
struct Sweep {
    /// Files that were wider and are now owner-only.
    tightened: usize,
    /// Files opened to narrow them. A file already owner-only is never opened.
    opened: usize,
    /// Files that could not be narrowed.
    failed: usize,
    /// The first file that could not be narrowed, with its error.
    first_failure: Option<(PathBuf, String)>,
}

impl Sweep {
    fn record_failure(&mut self, path: PathBuf, error: &std::io::Error) {
        self.failed += 1;
        self.first_failure
            .get_or_insert_with(|| (path, error.to_string()));
    }
}

/// Whether the mode has a group or other bit set.
#[cfg(unix)]
fn is_wide(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & GROUP_AND_OTHER_BITS != 0
}

#[cfg(not(unix))]
fn is_wide(_meta: &std::fs::Metadata) -> bool {
    false
}

/// Narrows the regular files directly in `dir` that `is_own` accepts. Nothing
/// happens for a directory outside `state_root` or one reached through a
/// symbolic link, because only the state root is known to hold nothing but
/// LIAM's files. The directory entry's own metadata (it does not follow links)
/// tells a regular file from a link and shows the mode, so a symbolic link and
/// a file already owner-only are passed over without being opened.
fn secure_files_in(
    dir: &Path,
    state_root: &Path,
    is_own: impl Fn(&str) -> bool,
) -> std::io::Result<Sweep> {
    let mut sweep = Sweep::default();
    let (Ok(dir), Ok(root)) = (normalise(dir), normalise(state_root)) else {
        return Ok(sweep);
    };
    if !dir.starts_with(&root)
        || has_symlink_from(&root, &dir)
            .map_err(|error| failed("check for symbolic links in", error))?
    {
        return Ok(sweep);
    }
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if NOT_THERE.contains(&error.kind()) => return Ok(sweep),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        if !entry.file_name().to_str().is_some_and(&is_own) {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() || !is_wide(&meta) {
            continue;
        }
        sweep.opened += 1;
        match narrow_to_owner(&entry.path()) {
            Ok(Some(_)) => sweep.tightened += 1,
            Ok(None) => {}
            Err(error) => sweep.record_failure(entry.path(), &error),
        }
    }
    Ok(sweep)
}

/// Restricts the files the daemon keeps to the owner: the database and its
/// sidecars, the event log's files and the snapshot files. Run after the store
/// has opened, and at the end of a command that opened it. Files outside
/// `state_root` and files that are not LIAM's by name are never touched, and a
/// failure is a warning, not a failed start. A file that is a hard link is
/// narrowed like any other file the same user owns, which narrows every name
/// it has.
pub fn secure_state_files(database: &Path, log_dir: &Path, snapshot_dir: &Path, state_root: &Path) {
    secure_database_files(database);
    secure_sidecar_files(database, log_dir, snapshot_dir, state_root);
}

/// The part of `secure_state_files` that is limited to the state root.
fn secure_sidecar_files(database: &Path, log_dir: &Path, snapshot_dir: &Path, state_root: &Path) {
    let db_name = database
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string();
    let database_dir = database
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or(Path::new(CURRENT_DIR));
    secure_group(database_dir, state_root, |name| {
        is_database_sidecar(&db_name, name)
    });
    secure_group(log_dir, state_root, is_log_file);
    secure_group(snapshot_dir, state_root, is_snapshot_file);
}

/// `secure_files_in`, with what it did logged: one warning per directory, not
/// one per file, so a large directory cannot flood the log.
fn secure_group(dir: &Path, state_root: &Path, is_own: impl Fn(&str) -> bool) {
    match secure_files_in(dir, state_root, is_own) {
        Ok(sweep) => log_sweep(dir, &sweep),
        Err(error) => {
            tracing::warn!(dir = %dir.display(), %error, "could not secure the files in this directory");
        }
    }
}

fn log_sweep(dir: &Path, sweep: &Sweep) {
    let dir = dir.display();
    if sweep.tightened > 0 {
        tracing::info!(dir = %dir, count = sweep.tightened, "restricted the state files to their owner");
    }
    if let Some((first, error)) = &sweep.first_failure {
        tracing::warn!(dir = %dir, count = sweep.failed, first = %first.display(), %error, "could not restrict some files to their owner");
    }
}

/// `secure_state_files` for the directories `config` names around `database`,
/// with `log_dir` taking the place of the configured log directory when the
/// command was given one. The database files are narrowed whatever `HOME` is,
/// because the user pointed the tool at them. The rest needs a state root from
/// `HOME`, and is skipped without one.
pub fn secure_state_files_for(config: &Config, database: &Path, log_dir: Option<&Path>) {
    secure_state_files_with_home(config, database, log_dir, std::env::var_os("HOME"));
}

fn secure_state_files_with_home(
    config: &Config,
    database: &Path,
    log_dir: Option<&Path>,
    home: Option<OsString>,
) {
    secure_database_files(database);
    let Some(state_root) = state_root_from_home(home) else {
        return;
    };
    let dirs = match log_dir {
        Some(dir) => Ok(dir.to_path_buf()),
        None => models::resolve_log_dir(config, database),
    }
    .and_then(|log| Ok((log, models::resolve_snapshot_dir(config, database)?)));
    match dirs {
        Ok((log_dir, snapshot_dir)) => {
            secure_sidecar_files(database, &log_dir, &snapshot_dir, &state_root);
        }
        Err(error) => {
            tracing::warn!(%error, "could not work out the state directories, so their files were not checked");
        }
    }
}

#[cfg(all(test, unix))]
mod failure_tests;

#[cfg(all(test, unix))]
mod secure_dirs_tests;

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

    /// A state root with the database, sidecars, log and snapshot files, every
    /// file 0644.
    struct Layout {
        _home: tempfile::TempDir,
        root: PathBuf,
        database: PathBuf,
        log_dir: PathBuf,
        snapshot_dir: PathBuf,
    }

    impl Layout {
        fn new() -> Self {
            let home = tempfile::tempdir().unwrap();
            let root = default_state_dir(home.path());
            let log_dir = root.join("liam.log");
            let snapshot_dir = root.join("liam.snapshots");
            for dir in [&root, &log_dir, &snapshot_dir] {
                fs::create_dir_all(dir).unwrap();
            }
            let layout = Self {
                database: root.join("liam.db"),
                _home: home,
                root,
                log_dir,
                snapshot_dir,
            };
            for file in layout.state_files() {
                layout.write_wide(&file);
            }
            layout
        }

        fn state_files(&self) -> Vec<PathBuf> {
            vec![
                self.database.clone(),
                self.root.join("liam.db.embedder.json"),
                self.root.join("liam.db.consolidation.dedup.json"),
                self.log_dir.join("00000000000000000000.wal"),
                self.log_dir.join("00000000000000000001.parquet"),
                self.log_dir.join(MANIFEST_NAME),
                self.snapshot_dir.join("01ABC.db"),
                self.snapshot_dir.join("01ABC.json"),
            ]
        }

        fn write_wide(&self, file: &Path) {
            fs::write(file, b"data").unwrap();
            set_mode(file, 0o644);
        }

        fn secure(&self) {
            secure_state_files(
                &self.database,
                &self.log_dir,
                &self.snapshot_dir,
                &self.root,
            );
        }
    }

    #[test]
    fn a_sidecar_is_created_owner_only_even_with_no_umask() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let fingerprint = crate::embedder_fingerprint::EmbedderFingerprint {
            provider: "mock".to_string(),
            model: "m".to_string(),
            dims: 8,
            prompt_scheme: 1,
        };

        // Act: a mask that hides nothing, so only the explicit mode can narrow
        liam_log::private_fs::with_umask(0, || {
            crate::embedder_fingerprint::record(&database, &fingerprint).unwrap();
            crate::consolidation::tier_history::save(
                &crate::consolidation::tier_history::history_path_for(&database),
                &Default::default(),
            )
            .unwrap();
        });

        // Assert
        for path in [
            crate::embedder_fingerprint::fingerprint_path_for(&database),
            crate::consolidation::tier_history::history_path_for(&database),
        ] {
            assert_eq!(mode_of(&path), 0o600, "{}", path.display());
        }
    }

    #[test]
    fn wide_state_files_become_owner_only() {
        // Arrange
        let layout = Layout::new();

        // Act
        layout.secure();

        // Assert
        for file in layout.state_files() {
            assert_eq!(mode_of(&file), 0o600, "{}", file.display());
        }
    }

    #[test]
    fn a_file_already_narrower_than_owner_only_is_left_as_it_is() {
        // Arrange
        let layout = Layout::new();
        let sidecar = layout.root.join("liam.db.embedder.json");
        set_mode(&sidecar, 0o400);

        // Act
        layout.secure();

        // Assert
        assert_eq!(mode_of(&sidecar), 0o400);
    }

    #[test]
    fn a_symbolic_link_is_skipped_and_its_target_is_not_changed() {
        // Arrange
        let layout = Layout::new();
        let target = tempfile::NamedTempFile::new().unwrap();
        set_mode(target.path(), 0o644);
        let link = layout.root.join("liam.db.metrics-state.json");
        symlink(target.path(), &link).unwrap();

        // Act
        layout.secure();

        // Assert
        assert_eq!(mode_of(target.path()), 0o644);
    }

    #[test]
    fn files_in_a_directory_outside_the_state_root_are_not_touched() {
        // Arrange
        let layout = Layout::new();
        let outside = tempfile::tempdir().unwrap();
        let database = outside.path().join("liam.db");
        let sidecar = outside.path().join("liam.db.embedder.json");
        let wal = outside.path().join("00000000000000000000.wal");
        for file in [&sidecar, &wal] {
            layout.write_wide(file);
        }

        // Act
        secure_state_files(&database, outside.path(), outside.path(), &layout.root);

        // Assert
        assert_eq!(mode_of(&sidecar), 0o644);
        assert_eq!(mode_of(&wal), 0o644);
    }

    #[test]
    fn a_file_that_is_not_one_of_the_daemons_own_is_not_touched() {
        // Arrange
        let layout = Layout::new();
        let others = [
            layout.root.join("liam.toml"),
            layout.root.join("notes.json"),
            layout.root.join("liam.db.backup"),
            layout.log_dir.join("notes.txt"),
            layout.snapshot_dir.join("readme.md"),
        ];
        for file in &others {
            layout.write_wide(file);
        }

        // Act
        layout.secure();

        // Assert
        for file in &others {
            assert_eq!(mode_of(file), 0o644, "{}", file.display());
        }
    }

    #[test]
    fn a_state_directory_reached_through_a_symbolic_link_is_not_followed() {
        // Arrange
        let layout = Layout::new();
        let real = tempfile::tempdir().unwrap();
        let wal = real.path().join("00000000000000000000.wal");
        layout.write_wide(&wal);
        let link = layout.root.join("linked.log");
        symlink(real.path(), &link).unwrap();

        // Act
        secure_state_files(&layout.database, &link, &layout.snapshot_dir, &layout.root);

        // Assert
        assert_eq!(mode_of(&wal), 0o644);
    }

    #[test]
    fn a_bare_file_names_parent_is_shown_as_the_current_directory() {
        // Arrange
        let parent = Path::new("s.sock").parent().unwrap();

        // Act / Assert
        assert_eq!(dir_label(parent), CURRENT_DIR_LABEL);
        assert_eq!(dir_label(Path::new("run/dir")), "run/dir");
    }

    #[test]
    fn a_relative_directory_names_what_it_resolves_against_and_an_absolute_one_does_not() {
        // Arrange
        let cwd = Path::new("/work");

        // Act / Assert
        assert_eq!(
            resolved_against(Path::new("run"), Some(cwd)).as_deref(),
            Some("/work")
        );
        assert_eq!(resolved_against(Path::new("/run"), Some(cwd)), None);
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

    #[test]
    fn the_database_files_are_narrowed_when_home_gives_no_state_root() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let database = dir.path().join("liam.db");
        let files = [
            database.clone(),
            dir.path().join("liam.db-wal"),
            dir.path().join("liam.db-shm"),
        ];
        for file in &files {
            fs::write(file, b"data").unwrap();
            set_mode(file, 0o644);
        }

        // Act
        for home in [None, Some(OsString::from("relative/home"))] {
            secure_state_files_with_home(&Config::default(), &database, None, home);
        }

        // Assert
        for file in &files {
            assert_eq!(mode_of(file), 0o600, "{}", file.display());
        }
    }

    #[test]
    fn a_log_directory_given_by_the_command_is_the_one_secured() {
        // Arrange
        let layout = Layout::new();
        let other_log = layout.root.join("other.log");
        fs::create_dir(&other_log).unwrap();
        let wal = other_log.join("00000000000000000000.wal");
        layout.write_wide(&wal);
        let home = layout.root.parent().unwrap().as_os_str().to_owned();

        // Act
        secure_state_files_with_home(
            &Config::default(),
            &layout.database,
            Some(&other_log),
            Some(home),
        );

        // Assert
        assert_eq!(mode_of(&wal), 0o600);
    }

    #[test]
    fn an_already_narrow_file_is_not_opened() {
        // Arrange
        let layout = Layout::new();
        let wal = layout.log_dir.join("00000000000000000000.wal");
        let segment = layout.log_dir.join("00000000000000000002.wal");
        layout.write_wide(&segment);
        set_mode(&wal, 0o600);
        fs::remove_file(layout.log_dir.join("00000000000000000001.parquet")).unwrap();
        fs::remove_file(layout.log_dir.join(MANIFEST_NAME)).unwrap();

        // Act
        let sweep = secure_files_in(&layout.log_dir, &layout.root, is_log_file).unwrap();

        // Assert
        assert_eq!(sweep.opened, 1);
        assert_eq!(sweep.tightened, 1);
        assert_eq!(sweep.failed, 0);
    }

    #[test]
    fn failing_files_in_one_directory_make_one_summary_with_a_count() {
        // Arrange: a mode with no owner read bit makes the open fail
        if liam_log::private_fs::skipped_as_root(
            "failing_files_in_one_directory_make_one_summary_with_a_count",
        ) {
            return;
        }
        let layout = Layout::new();
        let names = [
            "00000000000000000010.wal",
            "00000000000000000011.wal",
            "00000000000000000012.wal",
        ];
        for name in names {
            let file = layout.log_dir.join(name);
            layout.write_wide(&file);
            set_mode(&file, 0o070);
        }

        // Act
        let sweep = secure_files_in(&layout.log_dir, &layout.root, is_log_file).unwrap();

        // Assert
        assert_eq!(sweep.failed, 3);
        let (first, error) = sweep.first_failure.unwrap();
        assert!(names.iter().any(|name| first.ends_with(name)), "{first:?}");
        assert!(!error.is_empty());
    }
}
