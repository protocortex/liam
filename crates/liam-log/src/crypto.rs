// SPDX-License-Identifier: Apache-2.0

//! Keys and the primitives built on them: AES-256-GCM sealing, HMAC-SHA256,
//! master key providers that wrap data keys, and the append-only registry that
//! stores the wrapped keys.
//!
//! Deleting content means destroying the key that seals it, so every secret
//! here zeroizes on drop and prints nothing when formatted.

use std::fmt;
use std::io;
use std::path::PathBuf;

use aes_gcm::aead::{Aead, Payload};
use aes_gcm::{Aes256Gcm, KeyInit};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use zeroize::Zeroizing;

mod master;
mod registry;

#[cfg(target_os = "macos")]
pub use master::SystemKeychain;
#[cfg(any(test, feature = "test-support"))]
pub use master::TestMaster;
pub use master::{
    FileMaster, KeychainItems, KeychainMaster, MasterKeyProvider, MemoryKeychain, Wrapped,
};
pub use registry::{KeyLookup, KeyMaterial, KeyRegistry, KeyState};

/// Bytes in a data key, a dedup key, and a master key.
pub const KEY_LEN: usize = 32;
/// Bytes in an AES-GCM nonce.
pub const NONCE_LEN: usize = 12;
/// Bytes in an AES-GCM authentication tag.
pub const TAG_LEN: usize = 16;

/// Why a key operation failed. No variant carries a subject, a scope, or any
/// content, only ids and paths.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("authentication failed: the data was altered or sealed under another key")]
    Tampered,
    #[error("key file {} has mode {mode:o}, it must not be accessible to group or others (0600)", path.display())]
    KeyFileMode { path: PathBuf, mode: u32 },
    #[error("key file {} does not exist", path.display())]
    KeyFileMissing { path: PathBuf },
    #[error("key file {} must hold exactly {KEY_LEN} bytes, it holds {found}", path.display())]
    KeyFileLength { path: PathBuf, found: usize },
    #[error("key registry {} is corrupt: {reason}", path.display())]
    RegistryCorrupt { path: PathBuf, reason: String },
    #[error("key group {key_id} is being deleted or is deleted, nothing may be written under it")]
    KeyGroupLocked { key_id: String },
    #[error("key {key_id} is not in the registry")]
    KeyMissing { key_id: String },
    #[error("key {key_id} cannot return to active: its key bytes are already destroyed")]
    KeyDestroyed { key_id: String },
    #[error("the scope token {token:?} is reserved for entries without a scope")]
    ReservedScope { token: &'static str },
    #[error("key was wrapped by master key {found}, the configured master key is {expected}")]
    WrongMaster { expected: String, found: String },
    #[error("keychain failed: {0}")]
    Keychain(String),
    #[error("random bytes unavailable: {0}")]
    Random(String),
    #[error("registry record could not be decoded: {0}")]
    Decode(postcard::Error),
    #[error("key storage io failed: {0}")]
    Io(#[from] io::Error),
}

/// A 256-bit AES key. Zeroized on drop; `Debug` prints no bytes.
#[derive(Clone)]
pub struct DataKey(Zeroizing<[u8; KEY_LEN]>);

impl DataKey {
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self(Zeroizing::new(bytes))
    }

    pub fn random() -> Result<Self, CryptoError> {
        Ok(Self::from_bytes(random_array()?))
    }

    pub(crate) fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }
}

impl fmt::Debug for DataKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DataKey(<redacted>)")
    }
}

/// A 256-bit HMAC key that scopes content hashes to one key group.
#[derive(Clone)]
pub struct DedupKey(DataKey);

impl DedupKey {
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self(DataKey::from_bytes(bytes))
    }

    pub fn random() -> Result<Self, CryptoError> {
        Ok(Self(DataKey::random()?))
    }

    pub(crate) fn as_bytes(&self) -> &[u8; KEY_LEN] {
        self.0.as_bytes()
    }
}

impl fmt::Debug for DedupKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DedupKey(<redacted>)")
    }
}

/// A ciphertext with the nonce it was sealed under. The tag is the last
/// [`TAG_LEN`] bytes of `body`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedBytes {
    pub nonce: [u8; NONCE_LEN],
    pub body: Vec<u8>,
}

/// Fills a fixed-size array from the operating system's random source.
pub(crate) fn random_array<const N: usize>() -> Result<[u8; N], CryptoError> {
    let mut bytes = [0u8; N];
    getrandom::fill(&mut bytes).map_err(|error| CryptoError::Random(error.to_string()))?;
    Ok(bytes)
}

/// Seals `plaintext` under `key` with a fresh random nonce. `aad` is
/// authenticated but not stored, so opening needs the same bytes.
pub fn aead_seal(key: &DataKey, aad: &[u8], plaintext: &[u8]) -> Result<SealedBytes, CryptoError> {
    let nonce = random_array::<NONCE_LEN>()?;
    let body = cipher(key)
        .encrypt(
            &nonce.into(),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| CryptoError::Tampered)?;
    Ok(SealedBytes { nonce, body })
}

/// Opens what [`aead_seal`] produced. Any change to the nonce, the body, the
/// key, or `aad` fails with [`CryptoError::Tampered`].
pub fn aead_open(
    key: &DataKey,
    aad: &[u8],
    nonce: &[u8; NONCE_LEN],
    body: &[u8],
) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
    cipher(key)
        .decrypt(&(*nonce).into(), Payload { msg: body, aad })
        .map(Zeroizing::new)
        .map_err(|_| CryptoError::Tampered)
}

fn cipher(key: &DataKey) -> Aes256Gcm {
    Aes256Gcm::new(&(*key.as_bytes()).into())
}

/// HMAC-SHA256 of `data` under `key`.
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    // HMAC takes a key of any length, so construction cannot fail.
    let mut mac =
        <Hmac<Sha256> as KeyInit>::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> DataKey {
        DataKey::from_bytes([byte; KEY_LEN])
    }

    #[test]
    fn seal_then_open_returns_the_plaintext() {
        // Arrange
        let key = key(7);

        // Act
        let sealed = aead_seal(&key, b"aad", b"secret text").expect("seal");
        let opened = aead_open(&key, b"aad", &sealed.nonce, &sealed.body).expect("open");

        // Assert
        assert_eq!(opened.as_slice(), b"secret text");
        assert_eq!(sealed.body.len(), b"secret text".len() + TAG_LEN);
    }

    #[test]
    fn open_fails_when_the_aad_the_key_or_a_bit_changes() {
        // Arrange
        let sealed = aead_seal(&key(1), b"aad", b"secret").expect("seal");
        let mut flipped = sealed.body.clone();
        flipped[0] ^= 1;

        // Act
        let wrong_aad = aead_open(&key(1), b"other", &sealed.nonce, &sealed.body);
        let wrong_key = aead_open(&key(2), b"aad", &sealed.nonce, &sealed.body);
        let flipped_bit = aead_open(&key(1), b"aad", &sealed.nonce, &flipped);

        // Assert
        assert!(matches!(wrong_aad, Err(CryptoError::Tampered)));
        assert!(matches!(wrong_key, Err(CryptoError::Tampered)));
        assert!(matches!(flipped_bit, Err(CryptoError::Tampered)));
    }

    #[test]
    fn two_seals_of_one_payload_differ() {
        // Arrange
        let key = key(3);

        // Act
        let first = aead_seal(&key, b"", b"same").expect("seal");
        let second = aead_seal(&key, b"", b"same").expect("seal");

        // Assert
        assert_ne!(first.nonce, second.nonce);
        assert_ne!(first.body, second.body);
    }

    #[test]
    fn keys_print_no_bytes() {
        // Arrange
        let data = DataKey::from_bytes([0xAB; KEY_LEN]);
        let dedup = DedupKey::from_bytes([0xCD; KEY_LEN]);

        // Act
        let printed = format!("{data:?} {dedup:?} {data:#?}");

        // Assert
        assert!(!printed.contains("171"), "{printed}");
        assert!(!printed.to_lowercase().contains("ab"), "{printed}");
        assert!(printed.contains("redacted"));
    }

    #[test]
    fn hmac_matches_the_published_rfc_4231_vector() {
        // Arrange: RFC 4231 test case 2.
        let expected = [
            0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e, 0x6a, 0x04, 0x24, 0x26, 0x08, 0x95,
            0x75, 0xc7, 0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27, 0x39, 0x83, 0x9d, 0xec, 0x58, 0xb9,
            0x64, 0xec, 0x38, 0x43,
        ];

        // Act
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");

        // Assert
        assert_eq!(mac, expected);
    }
}
