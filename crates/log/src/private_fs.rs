// SPDX-License-Identifier: Apache-2.0
//! Owner-only directories and files for everything LIAM keeps on disk.
//!
//! Memory is stored in plaintext unless `[encryption]` is on, so a state
//! directory that the umask leaves readable by other users would expose it.

use std::fs::{DirBuilder, OpenOptions};
use std::io;
use std::path::Path;

/// Mode of a directory LIAM creates: the owner only.
pub const PRIVATE_DIR_MODE: u32 = 0o700;
/// Mode of a file LIAM creates: the owner only.
pub const PRIVATE_FILE_MODE: u32 = 0o600;
/// The group and other permission bits, which a private path must not have.
#[cfg(unix)]
const GROUP_AND_OTHER: u32 = 0o077;
/// The process mask that removes the group and other bits from every file a
/// library or a child process creates without an explicit mode.
#[cfg(unix)]
const PRIVATE_UMASK: libc::mode_t = GROUP_AND_OTHER as libc::mode_t;

/// Makes every file this process creates from now on owner-only, unless the
/// creator asks for a mode of its own. It closes the window in which a
/// library that creates files with the default mode, such as the database
/// engine, leaves a new file readable by others until it is narrowed.
///
/// The mask is process-wide, so call this once at the top of `main`, before
/// any thread starts.
pub fn restrict_umask() {
    #[cfg(unix)]
    set_umask(PRIVATE_UMASK);
}

/// Sets the process mask and returns the one it replaces.
#[cfg(unix)]
fn set_umask(mask: libc::mode_t) -> libc::mode_t {
    // SAFETY: `umask` takes and returns a plain integer, reads no memory and
    // cannot fail.
    unsafe { libc::umask(mask) }
}

/// Runs `run` with the process mask set to `mask`, then puts the old mask
/// back. Calls are serialised, so a test that creates files and checks their
/// modes does not see another test's mask.
#[cfg(all(unix, any(test, feature = "test-support")))]
pub fn with_umask<R>(mask: libc::mode_t, run: impl FnOnce() -> R) -> R {
    use std::sync::{Mutex, PoisonError};

    struct Restore(libc::mode_t);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_umask(self.0);
        }
    }

    static SERIAL: Mutex<()> = Mutex::new(());
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let _restore = Restore(set_umask(mask));
    run()
}

/// Creates `dir` and every missing ancestor, each with mode 0700. A directory
/// that already exists keeps its mode. The mode applies at creation, so there
/// is no moment when a new directory is wider.
pub fn create_private_dir_all(dir: &Path) -> io::Result<()> {
    let mut builder = DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, PRIVATE_DIR_MODE);
    builder.create(dir)
}

/// Open options that create a file with mode 0600.
#[cfg(unix)]
pub fn private_file_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, PRIVATE_FILE_MODE);
    options
}

/// Open options for a new file. There are no mode bits to set on this platform.
#[cfg(not(unix))]
pub fn private_file_options() -> OpenOptions {
    OpenOptions::new()
}

/// Removes the group and other permission bits from the file at `path`, never
/// adding any. Returns the old mode when it changed one, and `None` when the
/// file was already owner-only. A symlink is refused, not followed.
#[cfg(unix)]
pub fn narrow_to_owner(path: &Path) -> io::Result<Option<u32>> {
    narrow_handle(&open_without_following(path, 0)?)
}

/// Like `narrow_to_owner`, for a directory: one the user narrowed further, such
/// as to 0500, is left as it is.
#[cfg(unix)]
pub fn narrow_dir_to_owner(path: &Path) -> io::Result<Option<u32>> {
    narrow_handle(&open_without_following(path, libc::O_DIRECTORY)?)
}

/// The mode is read from, and applied to, the open handle, so a path swapped
/// for a symlink after the open cannot redirect the change.
#[cfg(unix)]
fn open_without_following(path: &Path, extra_flags: libc::c_int) -> io::Result<std::fs::File> {
    std::os::unix::fs::OpenOptionsExt::custom_flags(
        OpenOptions::new().read(true),
        libc::O_NOFOLLOW | extra_flags,
    )
    .open(path)
    .map_err(|error| {
        let error = step_failed("open without following links", error);
        // A symlink fails the open with ELOOP; say what the user can do about it.
        if error.raw_os_error() == Some(libc::ELOOP) {
            return io::Error::new(
                error.kind(),
                format!("{error}; if the real path should be owner-only, run `chmod 700` on it yourself"),
            );
        }
        error
    })
}

/// Keeps the error's kind and says which step failed.
#[cfg(unix)]
fn step_failed(step: &str, error: io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("could not {step} the path: {error}"))
}

#[cfg(unix)]
fn narrow_handle(handle: &std::fs::File) -> io::Result<Option<u32>> {
    use std::os::unix::fs::PermissionsExt;

    let mode = handle
        .metadata()
        .map_err(|error| step_failed("read the mode of", error))?
        .permissions()
        .mode()
        & 0o7777;
    if mode & GROUP_AND_OTHER == 0 {
        return Ok(None);
    }
    handle
        .set_permissions(std::fs::Permissions::from_mode(mode & !GROUP_AND_OTHER))
        .map_err(|error| step_failed("change the mode of", error))?;
    Ok(Some(mode))
}

/// There are no mode bits to narrow on this platform.
#[cfg(not(unix))]
pub fn narrow_to_owner(_path: &Path) -> io::Result<Option<u32>> {
    Ok(None)
}

/// There are no mode bits to narrow on this platform.
#[cfg(not(unix))]
pub fn narrow_dir_to_owner(_path: &Path) -> io::Result<Option<u32>> {
    Ok(None)
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    const NO_MASK: libc::mode_t = 0;

    fn mode_of(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn every_directory_created_is_owner_only() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("state").join("log");

        // Act: a mask that hides nothing, so only the explicit mode can narrow it
        with_umask(NO_MASK, || create_private_dir_all(&nested).unwrap());

        // Assert
        assert_eq!(mode_of(&nested), PRIVATE_DIR_MODE);
        assert_eq!(mode_of(nested.parent().unwrap()), PRIVATE_DIR_MODE);
    }

    #[test]
    fn an_existing_directory_keeps_its_mode_when_created_again() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o755)).unwrap();

        // Act
        create_private_dir_all(root.path()).unwrap();

        // Assert
        assert_eq!(mode_of(root.path()), 0o755);
    }

    #[test]
    fn a_file_created_with_the_private_options_is_owner_only() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("file");

        // Act: a mask that hides nothing, so only the explicit mode can narrow it
        with_umask(NO_MASK, || {
            private_file_options()
                .create_new(true)
                .write(true)
                .open(&path)
                .unwrap();
        });

        // Assert
        assert_eq!(mode_of(&path), PRIVATE_FILE_MODE);
    }

    #[test]
    fn a_wide_mode_loses_its_group_and_other_bits_and_keeps_the_owner_bits() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("file");
        fs::write(&path, b"x").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        // Act
        let changed = narrow_to_owner(&path).unwrap();

        // Assert
        assert_eq!(changed, Some(0o644));
        assert_eq!(mode_of(&path), 0o600);
    }

    #[test]
    fn a_mode_that_is_already_narrower_is_left_alone() {
        for mode in [0o700, 0o500, 0o600, 0o400] {
            // Arrange
            let root = tempfile::tempdir().unwrap();
            fs::set_permissions(root.path(), fs::Permissions::from_mode(mode)).unwrap();

            // Act
            let changed = narrow_to_owner(root.path()).unwrap();

            // Assert
            assert_eq!(changed, None, "mode {mode:o}");
            assert_eq!(mode_of(root.path()), mode);
            fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
    }

    #[test]
    fn a_directory_is_narrowed_through_its_own_handle() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("wide");
        fs::create_dir(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // Act
        let changed = narrow_dir_to_owner(&dir).unwrap();

        // Assert
        assert_eq!(changed, Some(0o755));
        assert_eq!(mode_of(&dir), 0o700);
    }

    #[test]
    fn a_symlink_is_refused_and_its_target_keeps_its_mode() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target");
        fs::create_dir(&target).unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755)).unwrap();
        let file_target = root.path().join("file");
        fs::write(&file_target, b"x").unwrap();
        fs::set_permissions(&file_target, fs::Permissions::from_mode(0o644)).unwrap();
        let dir_link = root.path().join("dir-link");
        let file_link = root.path().join("file-link");
        std::os::unix::fs::symlink(&target, &dir_link).unwrap();
        std::os::unix::fs::symlink(&file_target, &file_link).unwrap();

        // Act
        let dir_result = narrow_dir_to_owner(&dir_link);
        let file_result = narrow_to_owner(&file_link);

        // Assert
        assert!(dir_result.is_err());
        assert!(file_result.is_err());
        assert_eq!(mode_of(&target), 0o755);
        assert_eq!(mode_of(&file_target), 0o644);
    }

    #[test]
    fn a_file_is_not_opened_as_a_directory() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("file");
        fs::write(&file, b"x").unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).unwrap();

        // Act
        let result = narrow_dir_to_owner(&file);

        // Assert
        assert!(result.is_err());
        assert_eq!(mode_of(&file), 0o644);
    }

    #[test]
    fn the_umask_is_set_and_restored() {
        // Arrange
        const OUTER: libc::mode_t = 0o022;

        // Act
        let (inner, restored) = with_umask(OUTER, || {
            let inner = set_umask(PRIVATE_UMASK);
            (inner, set_umask(inner))
        });
        let after = with_umask(OUTER, || set_umask(OUTER));

        // Assert
        assert_eq!(inner, OUTER);
        assert_eq!(restored, PRIVATE_UMASK);
        assert_eq!(after, OUTER);
    }
}
