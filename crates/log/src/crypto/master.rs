// SPDX-License-Identifier: Apache-2.0

//! Master key providers: the secret that wraps every data key in the registry.
//! The master key never sits beside the log, so a copy of the log directory
//! alone opens nothing.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use zeroize::Zeroizing;

use super::{
    aead_open, aead_seal, hmac_sha256, random_array, CryptoError, DataKey, KEY_LEN, NONCE_LEN,
};

/// What a master key wraps: a nonce followed by the sealed body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wrapped(Vec<u8>);

impl Wrapped {
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Wraps and unwraps small secrets under a master key. `aad` binds a wrapped
/// secret to what it is for, so one cannot be moved to another record.
pub trait MasterKeyProvider: Send + Sync {
    /// A fingerprint of the master key (not the key), recorded beside each
    /// wrapped secret so a registry opened with the wrong master says so.
    fn master_key_id(&self) -> &str;
    fn wrap(&self, aad: &[u8], secret: &[u8]) -> Result<Wrapped, CryptoError>;
    fn unwrap(&self, aad: &[u8], wrapped: &Wrapped) -> Result<Zeroizing<Vec<u8>>, CryptoError>;
}

/// Domain separator for the master key fingerprint.
const FINGERPRINT_LABEL: &[u8] = b"liam master key id v1";
/// Hex characters of the fingerprint kept as the id.
const FINGERPRINT_CHARS: usize = 16;

/// The wrapping logic every provider shares: AES-256-GCM under a 32-byte master.
struct Master {
    id: String,
    key: DataKey,
}

impl Master {
    fn new(key: DataKey) -> Self {
        let fingerprint = hmac_sha256(key.as_bytes(), FINGERPRINT_LABEL);
        let id = fingerprint
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()[..FINGERPRINT_CHARS]
            .to_owned();
        Self { id, key }
    }

    fn wrap(&self, aad: &[u8], secret: &[u8]) -> Result<Wrapped, CryptoError> {
        let sealed = aead_seal(&self.key, aad, secret)?;
        let mut bytes = Vec::with_capacity(NONCE_LEN + sealed.body.len());
        bytes.extend_from_slice(&sealed.nonce);
        bytes.extend_from_slice(&sealed.body);
        Ok(Wrapped(bytes))
    }

    fn unwrap(&self, aad: &[u8], wrapped: &Wrapped) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
        let (nonce, body) = wrapped
            .0
            .split_at_checked(NONCE_LEN)
            .ok_or(CryptoError::Tampered)?;
        let nonce: [u8; NONCE_LEN] = nonce.try_into().map_err(|_| CryptoError::Tampered)?;
        aead_open(&self.key, aad, &nonce, body)
    }
}

macro_rules! master_provider {
    ($name:ident) => {
        impl MasterKeyProvider for $name {
            fn master_key_id(&self) -> &str {
                &self.0.id
            }

            fn wrap(&self, aad: &[u8], secret: &[u8]) -> Result<Wrapped, CryptoError> {
                self.0.wrap(aad, secret)
            }

            fn unwrap(
                &self,
                aad: &[u8],
                wrapped: &Wrapped,
            ) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
                self.0.unwrap(aad, wrapped)
            }
        }
    };
}

/// A master key read from a file that only its owner may access.
pub struct FileMaster(Master);

master_provider!(FileMaster);

impl FileMaster {
    /// Reads the key file. A missing file is an error, never a reason to make
    /// a key: generating one silently would orphan every key wrapped before.
    pub fn open(path: &Path) -> Result<Self, CryptoError> {
        let metadata = fs::metadata(path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => CryptoError::KeyFileMissing {
                path: path.to_path_buf(),
            },
            _ => CryptoError::Io(error),
        })?;
        refuse_loose_mode(path, &metadata)?;
        let bytes = Zeroizing::new(fs::read(path)?);
        let key: [u8; KEY_LEN] =
            bytes
                .as_slice()
                .try_into()
                .map_err(|_| CryptoError::KeyFileLength {
                    path: path.to_path_buf(),
                    found: bytes.len(),
                })?;
        Ok(Self(Master::new(DataKey::from_bytes(key))))
    }

    /// Writes a new random key file, readable by its owner only. Fails if the
    /// file exists, so an existing key is never replaced.
    pub fn generate(path: &Path) -> Result<(), CryptoError> {
        let key = Zeroizing::new(random_array::<KEY_LEN>()?);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(path)?;
        file.write_all(key.as_slice())?;
        file.sync_all()?;
        Ok(())
    }
}

#[cfg(unix)]
fn refuse_loose_mode(path: &Path, metadata: &fs::Metadata) -> Result<(), CryptoError> {
    use std::os::unix::fs::PermissionsExt;

    /// Any bit for group or others.
    const OTHERS_AND_GROUP: u32 = 0o077;
    let mode = metadata.permissions().mode() & 0o777;
    if mode & OTHERS_AND_GROUP != 0 {
        return Err(CryptoError::KeyFileMode {
            path: path.to_path_buf(),
            mode,
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn refuse_loose_mode(_path: &Path, _metadata: &fs::Metadata) -> Result<(), CryptoError> {
    Ok(())
}

/// Where a keychain keeps a secret, so tests need no real keychain.
pub trait KeychainItems: Send + Sync {
    fn read(&self, service: &str, account: &str)
        -> Result<Option<Zeroizing<Vec<u8>>>, CryptoError>;
    fn write(&self, service: &str, account: &str, secret: &[u8]) -> Result<(), CryptoError>;
    fn remove(&self, service: &str, account: &str) -> Result<(), CryptoError>;
}

/// An in-memory keychain for tests.
#[derive(Default)]
pub struct MemoryKeychain {
    items: Mutex<HashMap<(String, String), Vec<u8>>>,
}

impl KeychainItems for MemoryKeychain {
    fn read(
        &self,
        service: &str,
        account: &str,
    ) -> Result<Option<Zeroizing<Vec<u8>>>, CryptoError> {
        let items = self.items.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(items
            .get(&(service.to_owned(), account.to_owned()))
            .cloned()
            .map(Zeroizing::new))
    }

    fn write(&self, service: &str, account: &str, secret: &[u8]) -> Result<(), CryptoError> {
        let mut items = self.items.lock().unwrap_or_else(PoisonError::into_inner);
        items.insert((service.to_owned(), account.to_owned()), secret.to_vec());
        Ok(())
    }

    fn remove(&self, service: &str, account: &str) -> Result<(), CryptoError> {
        let mut items = self.items.lock().unwrap_or_else(PoisonError::into_inner);
        items.remove(&(service.to_owned(), account.to_owned()));
        Ok(())
    }
}

/// The macOS login keychain.
#[cfg(target_os = "macos")]
pub struct SystemKeychain;

#[cfg(target_os = "macos")]
impl SystemKeychain {
    /// `errSecItemNotFound`: the item does not exist, which is not a failure.
    const ITEM_NOT_FOUND: i32 = -25300;
}

#[cfg(target_os = "macos")]
impl KeychainItems for SystemKeychain {
    fn read(
        &self,
        service: &str,
        account: &str,
    ) -> Result<Option<Zeroizing<Vec<u8>>>, CryptoError> {
        match security_framework::passwords::get_generic_password(service, account) {
            Ok(secret) => Ok(Some(Zeroizing::new(secret))),
            Err(error) if error.code() == Self::ITEM_NOT_FOUND => Ok(None),
            Err(error) => Err(CryptoError::Keychain(error.to_string())),
        }
    }

    fn write(&self, service: &str, account: &str, secret: &[u8]) -> Result<(), CryptoError> {
        security_framework::passwords::set_generic_password(service, account, secret)
            .map_err(|error| CryptoError::Keychain(error.to_string()))
    }

    fn remove(&self, service: &str, account: &str) -> Result<(), CryptoError> {
        match security_framework::passwords::delete_generic_password(service, account) {
            Ok(()) => Ok(()),
            Err(error) if error.code() == Self::ITEM_NOT_FOUND => Ok(()),
            Err(error) => Err(CryptoError::Keychain(error.to_string())),
        }
    }
}

/// A master key kept in a keychain item, made on first use.
pub struct KeychainMaster(Master);

master_provider!(KeychainMaster);

impl KeychainMaster {
    /// Reads the item, or makes and stores a random key when there is none.
    /// Choosing the keychain in the configuration is the consent to make one.
    pub fn open_or_create(
        items: &Arc<dyn KeychainItems>,
        service: &str,
        account: &str,
    ) -> Result<Self, CryptoError> {
        let key = match items.read(service, account)? {
            Some(stored) => {
                let bytes: [u8; KEY_LEN] = stored.as_slice().try_into().map_err(|_| {
                    CryptoError::Keychain(format!(
                        "the keychain item holds {} bytes, expected {KEY_LEN}",
                        stored.len()
                    ))
                })?;
                DataKey::from_bytes(bytes)
            }
            None => {
                let key = DataKey::random()?;
                items.write(service, account, key.as_bytes())?;
                key
            }
        };
        Ok(Self(Master::new(key)))
    }
}

/// A master key made from a fixed byte, for tests in this and other crates.
#[cfg(any(test, feature = "test-support"))]
pub struct TestMaster(Master);

#[cfg(any(test, feature = "test-support"))]
master_provider!(TestMaster);

#[cfg(any(test, feature = "test-support"))]
impl TestMaster {
    pub fn new(seed: u8) -> Self {
        Self(Master::new(DataKey::from_bytes([seed; KEY_LEN])))
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn key_file(dir: &Path, mode: u32) -> std::path::PathBuf {
        let path = dir.join("master.key");
        fs::write(&path, [9u8; KEY_LEN]).expect("write key file");
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).expect("set mode");
        path
    }

    fn round_trips(provider: &dyn MasterKeyProvider) {
        // Arrange
        let secret = [5u8; KEY_LEN];

        // Act
        let wrapped = provider.wrap(b"slot", &secret).expect("wrap");
        let unwrapped = provider.unwrap(b"slot", &wrapped).expect("unwrap");

        // Assert
        assert_eq!(unwrapped.as_slice(), secret);
        assert_ne!(wrapped.as_bytes(), secret);
    }

    fn rejects_a_flipped_bit(provider: &dyn MasterKeyProvider) {
        // Arrange
        let wrapped = provider.wrap(b"slot", &[5u8; KEY_LEN]).expect("wrap");
        let mut bytes = wrapped.as_bytes().to_vec();
        bytes[NONCE_LEN + 3] ^= 1;

        // Act
        let result = provider.unwrap(b"slot", &Wrapped::from_bytes(bytes));

        // Assert
        assert!(matches!(result, Err(CryptoError::Tampered)));
    }

    #[test]
    fn each_provider_wraps_and_unwraps_and_rejects_a_flipped_bit() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let items: Arc<dyn KeychainItems> = Arc::new(MemoryKeychain::default());
        let file = FileMaster::open(&key_file(dir.path(), 0o600)).expect("file master");
        let keychain = KeychainMaster::open_or_create(&items, "svc", "acct").expect("keychain");
        let test = TestMaster::new(1);

        // Act and Assert
        for provider in [&file as &dyn MasterKeyProvider, &keychain, &test] {
            round_trips(provider);
            rejects_a_flipped_bit(provider);
        }
    }

    #[test]
    fn a_wrapped_secret_is_bound_to_its_aad() {
        // Arrange
        let master = TestMaster::new(2);
        let wrapped = master.wrap(b"key-a", &[1u8; KEY_LEN]).expect("wrap");

        // Act
        let result = master.unwrap(b"key-b", &wrapped);

        // Assert
        assert!(matches!(result, Err(CryptoError::Tampered)));
    }

    #[test]
    fn a_truncated_wrapped_blob_is_tampered_not_a_panic() {
        // Arrange
        let master = TestMaster::new(2);

        // Act
        let result = master.unwrap(b"", &Wrapped::from_bytes(vec![1, 2, 3]));

        // Assert
        assert!(matches!(result, Err(CryptoError::Tampered)));
    }

    #[test]
    fn master_key_ids_differ_between_keys_and_match_for_one_key() {
        // Arrange and Act
        let first = TestMaster::new(1);
        let again = TestMaster::new(1);
        let other = TestMaster::new(2);

        // Assert
        assert_eq!(first.master_key_id(), again.master_key_id());
        assert_ne!(first.master_key_id(), other.master_key_id());
        assert_eq!(first.master_key_id().len(), FINGERPRINT_CHARS);
    }

    #[test]
    fn file_master_refuses_a_key_file_open_to_others() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let path = key_file(dir.path(), 0o644);

        // Act
        let result = FileMaster::open(&path);

        // Assert
        assert!(matches!(
            result,
            Err(CryptoError::KeyFileMode { mode: 0o644, .. })
        ));
    }

    #[test]
    fn file_master_does_not_make_a_missing_key_file() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("missing.key");

        // Act
        let result = FileMaster::open(&path);

        // Assert
        assert!(matches!(result, Err(CryptoError::KeyFileMissing { .. })));
        assert!(!path.exists());
    }

    #[test]
    fn file_master_refuses_a_key_file_of_the_wrong_length() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("short.key");
        fs::write(&path, [1u8; 5]).expect("write");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).expect("mode");

        // Act
        let result = FileMaster::open(&path);

        // Assert
        assert!(matches!(
            result,
            Err(CryptoError::KeyFileLength { found: 5, .. })
        ));
    }

    #[test]
    fn generate_writes_an_owner_only_key_and_never_replaces_one() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("new.key");

        // Act
        FileMaster::generate(&path).expect("generate");
        let first = fs::read(&path).expect("read");
        let again = FileMaster::generate(&path);
        let opened = FileMaster::open(&path);

        // Assert
        assert_eq!(first.len(), KEY_LEN);
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(again.is_err(), "an existing key must not be replaced");
        assert_eq!(fs::read(&path).unwrap(), first);
        assert!(opened.is_ok());
    }

    #[test]
    fn keychain_master_makes_a_key_once_and_reuses_it() {
        // Arrange
        let items: Arc<dyn KeychainItems> = Arc::new(MemoryKeychain::default());

        // Act
        let first = KeychainMaster::open_or_create(&items, "svc", "acct").expect("first");
        let second = KeychainMaster::open_or_create(&items, "svc", "acct").expect("second");

        // Assert
        assert_eq!(first.master_key_id(), second.master_key_id());
    }

    #[test]
    fn keychain_master_refuses_an_item_of_the_wrong_length() {
        // Arrange
        let memory = MemoryKeychain::default();
        memory.write("svc", "acct", &[1, 2, 3]).expect("write");
        let items: Arc<dyn KeychainItems> = Arc::new(memory);

        // Act
        let result = KeychainMaster::open_or_create(&items, "svc", "acct");

        // Assert
        assert!(matches!(result, Err(CryptoError::Keychain(_))));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_system_keychain_stores_and_removes_an_item() {
        // Arrange
        let items = SystemKeychain;
        let service = format!("liam-test-{}", uuid::Uuid::now_v7());
        let cleanup = || items.remove(&service, "master").expect("remove");

        // Act
        let missing = items.read(&service, "master").expect("read missing");
        items
            .write(&service, "master", &[4u8; KEY_LEN])
            .expect("write");
        let stored = items.read(&service, "master").expect("read stored");
        cleanup();
        let gone = items.read(&service, "master").expect("read gone");

        // Assert
        assert!(missing.is_none());
        assert_eq!(stored.expect("stored").as_slice(), [4u8; KEY_LEN]);
        assert!(gone.is_none());
    }
}
