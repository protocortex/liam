// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};

use super::*;
use crate::crypto::TestMaster;

const MASTER_SEED: u8 = 4;

fn master() -> Arc<dyn MasterKeyProvider> {
    Arc::new(TestMaster::new(MASTER_SEED))
}

fn open(dir: &Path) -> KeyRegistry {
    KeyRegistry::open(dir, master()).expect("open registry")
}

/// A hook that fails the first time the registry reaches `point`, standing in
/// for a crash there.
fn crash_at(point: Point) -> FaultHook {
    let fired = AtomicBool::new(false);
    Arc::new(move |reached| {
        if reached == point && !fired.swap(true, Ordering::SeqCst) {
            return Err(io::Error::other("simulated crash"));
        }
        Ok(())
    })
}

fn registry_bytes(dir: &Path) -> Vec<u8> {
    fs::read(dir.join(REGISTRY_FILE)).expect("read registry")
}

fn active(registry: &KeyRegistry, key_id: &str) -> KeyMaterial {
    match registry.open_key(key_id).expect("lookup") {
        KeyLookup::Active(material) => material,
        other => panic!("expected an active key, got {other:?}"),
    }
}

#[test]
fn a_new_key_survives_a_reopen_with_the_same_bytes() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry
        .key_for_subject(Some("work"), "alice")
        .expect("key");
    let before = active(&registry, &key_id);
    drop(registry);

    // Act
    let reopened = open(dir.path());
    let after = active(&reopened, &key_id);

    // Assert
    assert_eq!(before.data.as_bytes(), after.data.as_bytes());
    assert_eq!(before.dedup.as_bytes(), after.dedup.as_bytes());
    assert_eq!(
        reopened.key_for_subject(Some("work"), "alice").unwrap(),
        key_id
    );
}

#[test]
fn subjects_share_a_group_by_normalised_text_and_scope() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());

    // Act
    let alice = registry.key_for_subject(Some("work"), "Alice").unwrap();
    let alice_again = registry.key_for_subject(Some("work"), "  alice ").unwrap();
    let bob = registry.key_for_subject(Some("work"), "bob").unwrap();
    let alice_elsewhere = registry.key_for_subject(Some("home"), "alice").unwrap();
    let alice_unscoped = registry.key_for_subject(None, "alice").unwrap();

    // Assert
    assert_eq!(alice, alice_again);
    assert_ne!(alice, bob);
    assert_ne!(alice, alice_elsewhere);
    assert_ne!(alice, alice_unscoped);
    assert_ne!(alice_elsewhere, alice_unscoped);
}

#[test]
fn the_no_scope_token_is_refused_as_a_literal_scope() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());

    // Act
    let result = registry.key_for_subject(Some(NO_SCOPE_TOKEN), "alice");

    // Assert
    assert!(matches!(result, Err(CryptoError::ReservedScope { .. })));
}

#[test]
fn event_keys_are_distinct_and_never_found_by_subject() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());

    // Act
    let first = registry.new_event_key().unwrap();
    let second = registry.new_event_key().unwrap();

    // Assert
    assert_ne!(first, second);
    assert!(registry.by_subject_is_empty());
}

#[test]
fn a_destroyed_key_reads_destroyed_and_its_bytes_and_subject_hash_are_gone() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.key_for_subject(Some("work"), "alice").unwrap();
    let before = registry_bytes(dir.path());
    let slot = before[before.len() - SLOT_LEN..].to_vec();

    // Act
    registry.destroy(&key_id).expect("destroy");
    let after = registry_bytes(dir.path());
    let replacement = registry.key_for_subject(Some("work"), "alice").unwrap();

    // Assert
    assert!(matches!(
        registry.open_key(&key_id).unwrap(),
        KeyLookup::Destroyed
    ));
    assert!(after[before.len() - SLOT_LEN..before.len()]
        .iter()
        .all(|byte| *byte == 0));
    assert!(!after
        .windows(WRAPPED_LEN)
        .any(|window| window == &slot[..WRAPPED_LEN]));
    assert_ne!(replacement, key_id, "a destroyed subject gets a new group");
}

#[test]
fn a_destroyed_key_stays_destroyed_after_a_reopen() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.key_for_subject(None, "alice").unwrap();
    registry.destroy(&key_id).unwrap();
    drop(registry);

    // Act
    let reopened = open(dir.path());

    // Assert
    assert!(matches!(
        reopened.open_key(&key_id).unwrap(),
        KeyLookup::Destroyed
    ));
    assert_eq!(reopened.state(&key_id), Some(KeyState::Destroyed));
}

#[test]
fn destroy_is_idempotent() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.new_event_key().unwrap();
    registry.destroy(&key_id).unwrap();
    let size = registry_bytes(dir.path()).len();

    // Act
    let again = registry.destroy(&key_id);

    // Assert
    assert!(again.is_ok());
    assert_eq!(
        registry_bytes(dir.path()).len(),
        size,
        "nothing is appended twice"
    );
}

#[test]
fn an_unknown_key_is_missing_not_destroyed() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());

    // Act
    let lookup = registry.open_key("never-added").unwrap();

    // Assert
    assert!(matches!(lookup, KeyLookup::Missing));
}

#[test]
fn a_registry_cut_at_a_record_boundary_or_swapped_for_an_older_copy_reads_missing() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let older = registry_bytes(dir.path());
    let key_id = registry.key_for_subject(None, "alice").unwrap();
    drop(registry);

    // Act
    fs::write(dir.path().join(REGISTRY_FILE), &older).unwrap();
    let reopened = open(dir.path());

    // Assert
    assert!(matches!(
        reopened.open_key(&key_id).unwrap(),
        KeyLookup::Missing
    ));
}

#[test]
fn a_torn_final_record_is_discarded_and_earlier_keys_read_intact() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let kept = registry.key_for_subject(None, "alice").unwrap();
    let kept_len = registry_bytes(dir.path()).len();
    let lost = registry.key_for_subject(None, "bob").unwrap();
    drop(registry);
    let bytes = registry_bytes(dir.path());
    fs::write(dir.path().join(REGISTRY_FILE), &bytes[..bytes.len() - 10]).unwrap();

    // Act
    let reopened = open(dir.path());

    // Assert
    assert!(matches!(
        reopened.open_key(&kept).unwrap(),
        KeyLookup::Active(_)
    ));
    assert!(matches!(
        reopened.open_key(&lost).unwrap(),
        KeyLookup::Missing
    ));
    assert_eq!(
        registry_bytes(dir.path()).len(),
        kept_len,
        "the torn bytes are cut off"
    );
}

#[test]
fn a_complete_final_record_with_a_bad_checksum_is_corruption_not_a_torn_write() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    registry.key_for_subject(None, "alice").unwrap();
    drop(registry);
    let mut bytes = registry_bytes(dir.path());
    let key_record_start = FILE_HEADER_LEN + secret_record_len(&bytes);
    bytes[key_record_start + RECORD_PREFIX_LEN] ^= 0xFF;
    fs::write(dir.path().join(REGISTRY_FILE), &bytes).unwrap();

    // Act
    let result = KeyRegistry::open(dir.path(), master());

    // Assert
    assert!(matches!(result, Err(CryptoError::RegistryCorrupt { .. })));
}

#[test]
fn a_checksum_failure_in_the_middle_of_the_file_is_corruption() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    registry.key_for_subject(None, "alice").unwrap();
    registry.key_for_subject(None, "bob").unwrap();
    drop(registry);
    let mut bytes = registry_bytes(dir.path());
    let first_key_record = FILE_HEADER_LEN + secret_record_len(&bytes);
    bytes[first_key_record + RECORD_PREFIX_LEN] ^= 0xFF;
    fs::write(dir.path().join(REGISTRY_FILE), &bytes).unwrap();

    // Act
    let result = KeyRegistry::open(dir.path(), master());

    // Assert
    assert!(matches!(result, Err(CryptoError::RegistryCorrupt { .. })));
}

#[test]
fn a_flipped_bit_in_a_slot_does_not_wedge_startup() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.key_for_subject(None, "alice").unwrap();
    drop(registry);
    let mut bytes = registry_bytes(dir.path());
    let last = bytes.len() - 1;
    bytes[last - HASH_LEN - 3] ^= 1;
    fs::write(dir.path().join(REGISTRY_FILE), &bytes).unwrap();

    // Act
    let reopened = open(dir.path());
    let lookup = reopened.open_key(&key_id);

    // Assert
    assert!(
        matches!(lookup, Err(CryptoError::Tampered)),
        "the wrap, not the CRC, catches it"
    );
}

#[test]
fn a_registry_opened_with_another_master_key_says_so() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    open(dir.path()).new_event_key().unwrap();

    // Act
    let result = KeyRegistry::open(dir.path(), Arc::new(TestMaster::new(MASTER_SEED + 1)));

    // Assert
    assert!(matches!(result, Err(CryptoError::WrongMaster { .. })));
}

#[test]
fn a_stray_temp_file_beside_the_registry_is_ignored() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let key_id = open(dir.path()).new_event_key().unwrap();
    fs::write(dir.path().join("keys.registry.tmp"), b"garbage").unwrap();

    // Act
    let reopened = open(dir.path());

    // Assert
    assert!(matches!(
        reopened.open_key(&key_id).unwrap(),
        KeyLookup::Active(_)
    ));
}

#[test]
fn a_crash_before_the_directory_sync_hands_back_no_registry() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");

    // Act
    let result = KeyRegistry::open_with(dir.path(), master(), Some(crash_at(Point::DirectorySync)));

    // Assert
    assert!(
        result.is_err(),
        "no key may be returned before the directory is durable"
    );
    let retried = open(dir.path());
    assert!(
        retried.new_event_key().is_ok(),
        "a retry recovers the half made file"
    );
}

#[test]
fn a_group_marked_destroying_is_locked_for_writers_and_reads_destroying() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.key_for_subject(None, "alice").unwrap();

    // Act
    registry.begin_destroy(&key_id).unwrap();
    let write = registry.key_for_subject(None, "alice");

    // Assert
    assert!(matches!(write, Err(CryptoError::KeyGroupLocked { .. })));
    assert!(matches!(
        registry.open_key(&key_id).unwrap(),
        KeyLookup::Destroying
    ));
}

#[test]
fn a_cancelled_destroy_returns_the_group_to_active() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.key_for_subject(None, "alice").unwrap();
    registry.begin_destroy(&key_id).unwrap();

    // Act
    registry.cancel_destroy(&key_id).unwrap();
    drop(registry);
    let reopened = open(dir.path());

    // Assert
    assert!(matches!(
        reopened.open_key(&key_id).unwrap(),
        KeyLookup::Active(_)
    ));
    assert_eq!(reopened.key_for_subject(None, "alice").unwrap(), key_id);
}

#[test]
fn a_destroy_that_reached_the_index_cannot_be_cancelled() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry =
        KeyRegistry::open_with(dir.path(), master(), Some(crash_at(Point::AfterIndexLine)))
            .unwrap();
    let key_id = registry.new_event_key().unwrap();
    assert!(registry.destroy(&key_id).is_err());

    // Act
    let cancel = registry.cancel_destroy(&key_id);

    // Assert
    assert!(matches!(cancel, Err(CryptoError::KeyDestroyed { .. })));
}

#[test]
fn cancelling_a_destroyed_key_is_refused() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.new_event_key().unwrap();
    registry.destroy(&key_id).unwrap();

    // Act
    let cancel = registry.cancel_destroy(&key_id);

    // Assert
    assert!(matches!(cancel, Err(CryptoError::KeyDestroyed { .. })));
}

#[test]
fn a_crash_at_every_destroy_step_is_finished_by_the_next_open() {
    for point in [
        Point::AfterDestroyingRecord,
        Point::AfterIndexLine,
        Point::AfterSlotOverwrite,
    ] {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        let registry = KeyRegistry::open_with(dir.path(), master(), Some(crash_at(point))).unwrap();
        let key_id = registry.key_for_subject(None, "alice").unwrap();
        let before = registry_bytes(dir.path());
        let slot = before[before.len() - SLOT_LEN..][..WRAPPED_LEN].to_vec();

        // Act
        let crashed = registry.destroy(&key_id);
        let mid_state = registry.state(&key_id);
        drop(registry);
        let reopened = open(dir.path());
        reopened.finish_destroy(&key_id).unwrap();

        // Assert
        assert!(crashed.is_err(), "{point:?} must interrupt the destroy");
        assert_ne!(mid_state, Some(KeyState::Active), "{point:?}");
        assert!(
            matches!(reopened.open_key(&key_id).unwrap(), KeyLookup::Destroyed),
            "{point:?}"
        );
        let after = registry_bytes(dir.path());
        assert!(
            !after.windows(WRAPPED_LEN).any(|window| window == slot),
            "{point:?}"
        );
    }
}

#[test]
fn between_the_slot_overwrite_and_the_destroyed_record_the_key_reads_destroying_never_missing() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = KeyRegistry::open_with(
        dir.path(),
        master(),
        Some(crash_at(Point::AfterSlotOverwrite)),
    )
    .unwrap();
    let key_id = registry.new_event_key().unwrap();

    // Act
    let crashed = registry.destroy(&key_id);
    let lookup = registry.open_key(&key_id).unwrap();

    // Assert
    assert!(crashed.is_err());
    assert!(matches!(lookup, KeyLookup::Destroying));
}

#[test]
fn an_older_registry_copy_is_destroyed_again_from_the_sidecar_index() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let key_id = registry.key_for_subject(None, "alice").unwrap();
    let older = registry_bytes(dir.path());
    registry.destroy(&key_id).unwrap();
    drop(registry);
    fs::write(dir.path().join(REGISTRY_FILE), &older).unwrap();

    // Act
    let reopened = open(dir.path());

    // Assert
    assert!(matches!(
        reopened.open_key(&key_id).unwrap(),
        KeyLookup::Destroyed
    ));
    let after = registry_bytes(dir.path());
    assert!(after[older.len() - SLOT_LEN..older.len()]
        .iter()
        .all(|byte| *byte == 0));
}

#[test]
fn a_thousand_keys_grow_the_file_by_appended_records_only() {
    // Arrange
    let dir = tempfile::tempdir().expect("temp dir");
    let registry = open(dir.path());
    let first = registry.new_event_key().unwrap();
    let after_one = registry_bytes(dir.path());
    let record_len = after_one.len() - FILE_HEADER_LEN - secret_record_len(&after_one);

    // Act
    for _ in 1..1000 {
        registry.new_event_key().unwrap();
    }
    let after_all = registry_bytes(dir.path());

    // Assert
    assert_eq!(after_all.len(), after_one.len() + 999 * record_len);
    assert_eq!(
        &after_all[..after_one.len()],
        &after_one[..],
        "earlier bytes never change"
    );
    assert!(matches!(
        registry.open_key(&first).unwrap(),
        KeyLookup::Active(_)
    ));
}

/// Length of the secret record that follows the file header.
fn secret_record_len(bytes: &[u8]) -> usize {
    let header_len = u32::from_le_bytes(
        bytes[FILE_HEADER_LEN + 1..FILE_HEADER_LEN + RECORD_PREFIX_LEN]
            .try_into()
            .unwrap(),
    );
    RECORD_PREFIX_LEN + header_len as usize + CRC_BYTES
}

impl KeyRegistry {
    fn by_subject_is_empty(&self) -> bool {
        self.lock().by_subject.is_empty()
    }
}
