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
const GROUP_AND_OTHER: u32 = 0o077;

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
pub fn private_file_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, PRIVATE_FILE_MODE);
    options
}

/// Removes the group and other permission bits from `path`, never adding any.
/// Returns the old mode when it changed one, and `None` when the path was
/// already owner-only, including a path the user narrowed further, such as
/// a directory at 0500.
#[cfg(unix)]
pub fn narrow_to_owner(path: &Path) -> io::Result<Option<u32>> {
    use std::os::unix::fs::PermissionsExt;

    let mode = std::fs::metadata(path)?.permissions().mode() & 0o7777;
    if mode & GROUP_AND_OTHER == 0 {
        return Ok(None);
    }
    std::fs::set_permissions(
        path,
        std::fs::Permissions::from_mode(mode & !GROUP_AND_OTHER),
    )?;
    Ok(Some(mode))
}

/// There are no mode bits to narrow on this platform.
#[cfg(not(unix))]
pub fn narrow_to_owner(_path: &Path) -> io::Result<Option<u32>> {
    Ok(None)
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn mode_of(path: &Path) -> u32 {
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn every_directory_created_is_owner_only() {
        // Arrange
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("state").join("log");

        // Act
        create_private_dir_all(&nested).unwrap();

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

        // Act
        private_file_options()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();

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
}
