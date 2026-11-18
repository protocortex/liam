// SPDX-License-Identifier: Apache-2.0
//! Registry files that are damaged or hand-built wrong, and the typed error
//! each one gets.

use super::*;

const UNKNOWN_TAG: u8 = 9;
const OVERSIZED_HEADER: u32 = (MAX_HEADER_LEN + 1) as u32;
const UNSUPPORTED_VERSION: u32 = FORMAT_VERSION + 1;
const GHOST_KEY: &str = "ghost-key";

fn file_header(version: u32) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes
}

fn write_registry(dir: &Path, bytes: &[u8]) {
    fs::write(dir.join(REGISTRY_FILE), bytes).unwrap();
}

fn corrupt_reason(result: Result<KeyRegistry, CryptoError>) -> String {
    match result {
        Err(CryptoError::RegistryCorrupt { reason, .. }) => reason,
        other => panic!("expected RegistryCorrupt, got {:?}", other.map(|_| ())),
    }
}

fn state_record(key_id: &str, state: KeyState) -> Vec<u8> {
    let header = StateHeader {
        key_id: key_id.to_owned(),
        state,
    };
    encode_record(TAG_STATE, &header, None).unwrap()
}

#[test]
fn a_file_that_does_not_start_with_the_magic_is_corrupt() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = file_header(FORMAT_VERSION);
    bytes[0] ^= 0xFF;
    write_registry(dir.path(), &bytes);

    // Act
    let reason = corrupt_reason(KeyRegistry::open(dir.path(), master()));

    // Assert
    assert_eq!(reason, "the file does not start with the registry magic");
}

#[test]
fn a_file_written_by_another_format_version_is_corrupt() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    write_registry(dir.path(), &file_header(UNSUPPORTED_VERSION));

    // Act
    let reason = corrupt_reason(KeyRegistry::open(dir.path(), master()));

    // Assert
    assert_eq!(
        reason,
        format!("unsupported format version {UNSUPPORTED_VERSION}")
    );
}

#[test]
fn a_record_with_an_unknown_tag_is_corrupt_and_names_its_offset() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = file_header(FORMAT_VERSION);
    bytes.push(UNKNOWN_TAG);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    write_registry(dir.path(), &bytes);

    // Act
    let reason = corrupt_reason(KeyRegistry::open(dir.path(), master()));

    // Assert
    assert_eq!(
        reason,
        format!("unknown record tag {UNKNOWN_TAG} at offset {FILE_HEADER_LEN}")
    );
}

#[test]
fn a_record_claiming_an_oversized_header_is_corrupt() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let mut bytes = file_header(FORMAT_VERSION);
    bytes.push(TAG_SECRET);
    bytes.extend_from_slice(&OVERSIZED_HEADER.to_le_bytes());
    write_registry(dir.path(), &bytes);

    // Act
    let reason = corrupt_reason(KeyRegistry::open(dir.path(), master()));

    // Assert
    assert_eq!(
        reason,
        format!("record at offset {FILE_HEADER_LEN} claims a {OVERSIZED_HEADER} byte header")
    );
}

#[test]
fn keys_without_the_secret_record_are_corrupt() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    open(dir.path()).new_event_key().unwrap();
    let bytes = registry_bytes(dir.path());
    let secret_len = secret_record_len(&bytes);
    let mut without_secret = bytes[..FILE_HEADER_LEN].to_vec();
    without_secret.extend_from_slice(&bytes[FILE_HEADER_LEN + secret_len..]);
    write_registry(dir.path(), &without_secret);

    // Act
    let reason = corrupt_reason(KeyRegistry::open(dir.path(), master()));

    // Assert
    assert_eq!(
        reason,
        "keys exist but the registry secret record is missing"
    );
}

#[test]
fn a_header_only_file_gets_a_fresh_secret_and_can_make_keys() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    write_registry(dir.path(), &file_header(FORMAT_VERSION));

    // Act
    let registry = open(dir.path());
    let key_id = registry.key_for_subject(None, "alice").unwrap();

    // Assert
    assert!(matches!(
        registry.open_key(&key_id).unwrap(),
        KeyLookup::Active(_)
    ));
    assert!(registry_bytes(dir.path()).len() > FILE_HEADER_LEN);
}

#[test]
fn a_state_record_for_a_key_the_registry_does_not_hold_is_corrupt() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    open(dir.path());
    let mut bytes = registry_bytes(dir.path());
    bytes.extend_from_slice(&state_record(GHOST_KEY, KeyState::Destroying));
    write_registry(dir.path(), &bytes);

    // Act
    let reason = corrupt_reason(KeyRegistry::open(dir.path(), master()));

    // Assert
    assert_eq!(
        reason,
        "a state record names a key the registry does not hold"
    );
}

#[test]
fn a_destroyed_key_that_a_later_record_makes_live_again_is_corrupt() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let registry = open(dir.path());
    let key_id = registry.new_event_key().unwrap();
    registry.destroy(&key_id).unwrap();
    drop(registry);
    let mut bytes = registry_bytes(dir.path());
    bytes.extend_from_slice(&state_record(&key_id, KeyState::Active));
    write_registry(dir.path(), &bytes);

    // Act
    let reason = corrupt_reason(KeyRegistry::open(dir.path(), master()));

    // Assert
    assert_eq!(reason, "a destroyed key returns to a live state");
}

#[test]
fn destroying_a_key_the_registry_never_held_names_the_key() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let registry = open(dir.path());

    // Act
    let begin = registry.begin_destroy(GHOST_KEY);
    let finish = registry.finish_destroy(GHOST_KEY);
    let cancel = registry.cancel_destroy(GHOST_KEY);

    // Assert
    for result in [begin, finish, cancel] {
        assert!(
            matches!(&result, Err(CryptoError::KeyMissing { key_id }) if key_id == GHOST_KEY),
            "{result:?}"
        );
    }
}

#[test]
fn an_unreadable_destroyed_index_fails_the_open_instead_of_being_ignored() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    open(dir.path());
    fs::create_dir(dir.path().join(DESTROYED_INDEX_FILE)).unwrap();

    // Act
    let result = KeyRegistry::open(dir.path(), master());

    // Assert
    assert!(
        matches!(result, Err(CryptoError::Io(_))),
        "{:?}",
        result.map(|_| ())
    );
}

#[test]
fn cancelling_a_marked_key_whose_index_cannot_be_read_is_an_io_error() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let registry = open(dir.path());
    let key_id = registry.new_event_key().unwrap();
    registry.begin_destroy(&key_id).unwrap();
    fs::create_dir(dir.path().join(DESTROYED_INDEX_FILE)).unwrap();

    // Act
    let result = registry.cancel_destroy(&key_id);

    // Assert
    assert!(matches!(result, Err(CryptoError::Io(_))), "{result:?}");
    assert_eq!(registry.state(&key_id), Some(KeyState::Destroying));
}
