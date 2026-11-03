// SPDX-License-Identifier: Apache-2.0
//! Reading a provider key from a `0600` file or the Keychain, and holding it
//! so that printing it never shows it.

use std::fmt;
use std::path::PathBuf;

use liam_log::crypto::KeychainItems;

/// The account every hosted key is stored under in the Keychain; the service
/// names the provider.
const KEYCHAIN_ACCOUNT: &str = "api-key";
/// Any permission for the group or others.
#[cfg(unix)]
const GROUP_AND_OTHER_BITS: u32 = 0o077;

/// A key that prints as a placeholder.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

const REDACTED: &str = "<redacted>";

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The key itself, for the one place that puts it on the wire.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretString({REDACTED})")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Why a key could not be read.
#[derive(Debug)]
pub enum SecretError {
    LooseMode { path: PathBuf, mode: u32 },
    Unreadable { path: PathBuf, reason: String },
    EmptyFile { path: PathBuf },
    NotInKeychain { service: String },
    Keychain { service: String, reason: String },
}

impl fmt::Display for SecretError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LooseMode { path, mode } => write!(
                f,
                "key file {} is readable by other users (mode {mode:o}); run chmod 600 on it",
                path.display()
            ),
            Self::Unreadable { path, reason } => {
                write!(f, "key file {} could not be read: {reason}", path.display())
            }
            Self::EmptyFile { path } => write!(f, "the key file {} is empty", path.display()),
            Self::NotInKeychain { service } => {
                write!(f, "the Keychain has no key for service {service}")
            }
            Self::Keychain { service, reason } => write!(
                f,
                "the Keychain could not be read for service {service}: {reason}"
            ),
        }
    }
}

impl std::error::Error for SecretError {}

/// Where a provider key is kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretSource {
    File(PathBuf),
    Keychain { service: String },
}

impl SecretSource {
    pub fn read(&self, keychain: &dyn KeychainItems) -> Result<SecretString, SecretError> {
        match self {
            Self::File(path) => read_file(path),
            Self::Keychain { service } => read_keychain(keychain, service),
        }
    }
}

fn read_file(path: &PathBuf) -> Result<SecretString, SecretError> {
    let unreadable = |error: std::io::Error| SecretError::Unreadable {
        path: path.clone(),
        reason: error.kind().to_string(),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)
            .map_err(unreadable)?
            .permissions()
            .mode();
        if mode & GROUP_AND_OTHER_BITS != 0 {
            return Err(SecretError::LooseMode {
                path: path.clone(),
                mode: mode & 0o777,
            });
        }
    }
    let text = std::fs::read_to_string(path).map_err(unreadable)?;
    let key = text.trim();
    if key.is_empty() {
        return Err(SecretError::EmptyFile { path: path.clone() });
    }
    Ok(SecretString::new(key))
}

fn read_keychain(keychain: &dyn KeychainItems, service: &str) -> Result<SecretString, SecretError> {
    let item = keychain
        .read(service, KEYCHAIN_ACCOUNT)
        .map_err(|error| SecretError::Keychain {
            service: service.to_owned(),
            reason: error.to_string(),
        })?
        .ok_or_else(|| SecretError::NotInKeychain {
            service: service.to_owned(),
        })?;
    let key = String::from_utf8_lossy(&item);
    Ok(SecretString::new(key.trim()))
}

#[cfg(test)]
mod tests {
    use liam_log::crypto::MemoryKeychain;

    use super::*;

    const KEY: &str = "sk-test-0123456789";

    fn key_file(dir: &std::path::Path, mode: u32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("provider.key");
        std::fs::write(&path, format!("{KEY}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        path
    }

    #[test]
    fn a_secret_prints_as_a_placeholder_in_debug_and_display() {
        // Arrange
        let secret = SecretString::new(KEY);

        // Act
        let shown = format!("{secret} {secret:?} {:?}", Some(&secret));

        // Assert
        assert!(!shown.contains(KEY), "{shown}");
        assert_eq!(secret.expose(), KEY);
    }

    #[test]
    fn a_key_file_with_owner_only_access_is_read_and_trimmed() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let source = SecretSource::File(key_file(dir.path(), 0o600));

        // Act
        let key = source.read(&MemoryKeychain::default()).unwrap();

        // Assert
        assert_eq!(key.expose(), KEY);
    }

    #[test]
    fn a_key_file_open_to_the_group_or_others_is_refused_without_its_content() {
        for mode in [0o644, 0o640, 0o604] {
            // Arrange
            let dir = tempfile::tempdir().unwrap();
            let source = SecretSource::File(key_file(dir.path(), mode));

            // Act
            let error = source.read(&MemoryKeychain::default()).unwrap_err();

            // Assert
            assert!(matches!(error, SecretError::LooseMode { .. }), "{mode:o}");
            assert!(!error.to_string().contains(KEY));
        }
    }

    #[test]
    fn a_missing_or_empty_key_file_is_a_typed_error() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.key");
        std::fs::write(&empty, "  \n").unwrap();
        std::fs::set_permissions(&empty, std::os::unix::fs::PermissionsExt::from_mode(0o600))
            .unwrap();

        // Act
        let missing = SecretSource::File(dir.path().join("none")).read(&MemoryKeychain::default());
        let blank = SecretSource::File(empty).read(&MemoryKeychain::default());

        // Assert
        assert!(matches!(missing, Err(SecretError::Unreadable { .. })));
        assert!(matches!(blank, Err(SecretError::EmptyFile { .. })));
    }

    #[test]
    fn a_keychain_item_is_read_and_a_missing_one_is_named_by_its_service() {
        // Arrange
        let keychain = MemoryKeychain::default();
        keychain
            .write(
                "liam-anthropic",
                KEYCHAIN_ACCOUNT,
                format!("{KEY}\n").as_bytes(),
            )
            .unwrap();

        // Act
        let found = SecretSource::Keychain {
            service: "liam-anthropic".into(),
        }
        .read(&keychain);
        let missing = SecretSource::Keychain {
            service: "liam-openai".into(),
        }
        .read(&keychain);

        // Assert
        assert_eq!(found.unwrap().expose(), KEY);
        let message = missing.unwrap_err().to_string();
        assert!(message.contains("liam-openai"), "{message}");
    }
}
