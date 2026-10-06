// SPDX-License-Identifier: Apache-2.0

//! On-disk persistence for the dedup bloom filter.
//!
//! A saved filter carries the hash scheme, sizing, log identity and log
//! position it was built under, so a mismatch is detected and the caller
//! rebuilds from the log rather than trusting a filter that no longer fits.

use std::fs::{self, File};
use std::io::{self, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::dedup::{BloomConfig, HashBloom};
use crate::LogOffset;

/// Version of the canonical encoding and digest behind every content hash.
/// It is stored with a persisted filter, so changing either must bump it.
pub const CONTENT_HASH_SCHEME: u32 = 1;

/// What lands on disk: everything the filter was built under travels alongside
/// it, so a changed hash scheme, config, log or log position is detected
/// instead of silently misread.
#[derive(Serialize, Deserialize)]
struct PersistedFilter {
    scheme: u32,
    expected_items: u64,
    false_positive_rate: f64,
    log_id: [u8; 16],
    /// `(segment, index)` of the last event folded into the filter.
    folded_through: Option<(u64, u64)>,
    num_hashes: u32,
    bits: Vec<u64>,
}

/// A rate of 1e-9, the lowest `BloomConfig` allows, needs 30 hashes, so a stored
/// count above this is damage and would make every lookup needlessly slow.
const MAX_NUM_HASHES: u32 = 64;

const FILE_MAGIC: [u8; 4] = *b"LBF1";
const HEADER_LEN: usize = FILE_MAGIC.len() + size_of::<u32>();

impl HashBloom {
    /// Records `config`, the sizing this filter was built with, alongside it.
    /// Writes magic, checksum, then the encoded filter to a temp file beside
    /// `path` and renames it into place, so a crash never leaves a partial file.
    /// `folded_through` is the offset of the last event the filter has seen, or
    /// `None` for an empty log.
    pub fn save(
        &self,
        path: &Path,
        config: &BloomConfig,
        log_id: Uuid,
        folded_through: Option<LogOffset>,
    ) -> io::Result<()> {
        let persisted = PersistedFilter {
            scheme: CONTENT_HASH_SCHEME,
            expected_items: config.expected_items() as u64,
            false_positive_rate: config.false_positive_rate(),
            log_id: log_id.into_bytes(),
            folded_through: folded_through.map(|offset| (offset.segment, offset.index)),
            num_hashes: self.filter.num_hashes(),
            bits: self.filter.iter().collect(),
        };
        let bytes = frame(&persisted)?;

        let mut staged = path.as_os_str().to_owned();
        staged.push(".tmp");
        let staged = PathBuf::from(staged);
        let mut file = File::create(&staged)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&staged, path)?;
        let dir = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        File::open(dir)?.sync_all()
    }

    /// Never panics: any unreadable, damaged, differently configured, foreign
    /// or out of date file comes back as `Stale` so the caller rebuilds from
    /// the log. `head` is the offset of the log's last event, or `None` when
    /// the log is empty. A filter saved ahead of `head` is kept: it can only
    /// add false positives, which the exact index overrules.
    pub fn load(
        path: &Path,
        config: &BloomConfig,
        log_id: Uuid,
        head: Option<LogOffset>,
    ) -> FilterLoad {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return FilterLoad::Stale(StaleReason::Missing)
            }
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "bloom filter file could not be read, rebuilding from the log"
                );
                return FilterLoad::Stale(StaleReason::Unreadable);
            }
        };
        let Some(persisted) = decode(&bytes) else {
            return FilterLoad::Stale(StaleReason::Corrupt);
        };
        if persisted.scheme != CONTENT_HASH_SCHEME
            || persisted.expected_items != config.expected_items() as u64
            || persisted.false_positive_rate != config.false_positive_rate()
        {
            return FilterLoad::Stale(StaleReason::ConfigMismatch);
        }
        if persisted.log_id != log_id.into_bytes() {
            return FilterLoad::Stale(StaleReason::LogMismatch);
        }
        let folded_through = persisted
            .folded_through
            .map(|(segment, index)| LogOffset { segment, index });
        if folded_through < head {
            return FilterLoad::Stale(StaleReason::BehindLog);
        }
        FilterLoad::Fresh(Self::from_parts(persisted.bits, persisted.num_hashes))
    }
}

fn frame(persisted: &PersistedFilter) -> io::Result<Vec<u8>> {
    let payload = postcard::to_stdvec(persisted).map_err(io::Error::other)?;
    Ok(frame_payload(&payload))
}

fn frame_payload(payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend(FILE_MAGIC);
    bytes.extend(crc32fast::hash(payload).to_le_bytes());
    bytes.extend(payload);
    bytes
}

fn decode(bytes: &[u8]) -> Option<PersistedFilter> {
    let (magic, rest) = bytes.split_at_checked(FILE_MAGIC.len())?;
    let (checksum, payload) = rest.split_at_checked(size_of::<u32>())?;
    if magic != FILE_MAGIC || checksum != crc32fast::hash(payload).to_le_bytes() {
        return None;
    }
    // The checksum cannot vouch for a payload that was written wrong, and a
    // filter with no bits or a wild hash count would panic or crawl on first use.
    // Postcard sizes the bit vector from the bytes actually present, so a
    // corrupt length prefix fails to decode instead of allocating.
    let persisted: PersistedFilter = postcard::from_bytes(payload).ok()?;
    let usable = !persisted.bits.is_empty() && (1..=MAX_NUM_HASHES).contains(&persisted.num_hashes);
    usable.then_some(persisted)
}

/// Why a persisted filter cannot be used as is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleReason {
    Missing,
    /// The file exists but could not be read for a reason other than absence.
    Unreadable,
    Corrupt,
    /// Saved under a different hash scheme or bloom sizing.
    ConfigMismatch,
    /// Saved for a different log.
    LogMismatch,
    /// The log holds events the saved filter has not seen.
    BehindLog,
}

/// Result of loading a persisted filter: usable, or stale so the caller
/// rebuilds from the log before serving writes.
#[derive(Debug)]
pub enum FilterLoad {
    Fresh(HashBloom),
    Stale(StaleReason),
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::sync::{Arc, Mutex};

    use sha2::{Digest, Sha256};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber};

    use super::*;
    use crate::dedup::PreCheck;
    use crate::fixtures::{filter_with, sample_hashes, small_config};

    fn log_id() -> Uuid {
        Uuid::from_u128(0x1234)
    }

    fn offset(segment: u64, index: u64) -> LogOffset {
        LogOffset { segment, index }
    }

    fn assert_stale(loaded: FilterLoad, expected: StaleReason) {
        assert!(
            matches!(&loaded, FilterLoad::Stale(reason) if *reason == expected),
            "expected stale {expected:?}, got {loaded:?}"
        );
    }

    fn assert_fresh(loaded: FilterLoad) -> HashBloom {
        match loaded {
            FilterLoad::Fresh(filter) => filter,
            FilterLoad::Stale(reason) => panic!("expected a usable filter, got stale: {reason:?}"),
        }
    }

    /// Saves a real 500-hash filter to `bloom.bin` and returns its path and bytes.
    fn save_real_filter(dir: &Path) -> (PathBuf, Vec<u8>) {
        let path = dir.join("bloom.bin");
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path, &small_config(), log_id(), None)
            .expect("save");
        let bytes = fs::read(&path).expect("read back");
        (path, bytes)
    }

    fn persisted(bits: Vec<u64>, num_hashes: u32) -> PersistedFilter {
        let config = small_config();
        PersistedFilter {
            scheme: CONTENT_HASH_SCHEME,
            expected_items: config.expected_items() as u64,
            false_positive_rate: config.false_positive_rate(),
            log_id: log_id().into_bytes(),
            folded_through: None,
            num_hashes,
            bits,
        }
    }

    fn load_small(path: &Path) -> FilterLoad {
        HashBloom::load(path, &small_config(), log_id(), None)
    }

    /// Collects the messages of warn events emitted while it is the subscriber.
    #[derive(Clone, Default)]
    struct WarnCapture(Arc<Mutex<Vec<String>>>);

    struct Message(String);

    impl Visit for Message {
        fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }

    impl Subscriber for WarnCapture {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, event: &Event<'_>) {
            if *event.metadata().level() == Level::WARN {
                let mut message = Message(String::new());
                event.record(&mut message);
                self.0.lock().expect("capture lock").push(message.0);
            }
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }

    #[test]
    fn a_saved_filter_loads_back_equal() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let hashes = sample_hashes(0..1_000);
        let original = filter_with(small_config(), &hashes);
        let head = Some(offset(2, 40));

        // Act
        original
            .save(&path, &small_config(), log_id(), head)
            .expect("save");
        let loaded = HashBloom::load(&path, &small_config(), log_id(), head);

        // Assert
        let filter = assert_fresh(loaded);
        assert_eq!(filter, original);
        assert!(hashes.iter().all(|hash| filter.might_contain(hash)));
    }

    #[test]
    fn a_filter_saved_for_an_empty_log_loads_fresh_against_an_empty_log() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        HashBloom::new(small_config())
            .save(&path, &small_config(), log_id(), None)
            .expect("save");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_fresh(loaded);
    }

    #[test]
    fn a_missing_filter_file_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("absent.bin");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_stale(loaded, StaleReason::Missing);
    }

    #[test]
    fn a_filter_path_that_cannot_be_read_is_unreadable_and_warns() {
        // Arrange: a directory exists at the path but cannot be read as a file.
        let dir = tempfile::tempdir().expect("tempdir");
        let capture = WarnCapture::default();

        // Act
        let loaded = tracing::subscriber::with_default(capture.clone(), || load_small(dir.path()));

        // Assert
        assert_stale(loaded, StaleReason::Unreadable);
        let warnings = capture.0.lock().expect("capture lock");
        assert_eq!(warnings.len(), 1, "got {warnings:?}");
        assert!(
            warnings[0].contains("could not be read"),
            "got {warnings:?}"
        );
    }

    #[test]
    fn a_missing_filter_file_does_not_warn() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let capture = WarnCapture::default();

        // Act
        tracing::subscriber::with_default(capture.clone(), || {
            load_small(&dir.path().join("absent.bin"))
        });

        // Assert
        assert!(capture.0.lock().expect("capture lock").is_empty());
    }

    #[test]
    fn a_file_that_is_not_a_filter_is_corrupt() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        fs::write(&path, b"this is not a bloom filter").expect("write garbage");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_stale(loaded, StaleReason::Corrupt);
    }

    #[test]
    fn a_flipped_bit_vector_byte_is_rejected_by_the_checksum_alone() {
        // Arrange: the bit vector is the last field, so the tail is inside it.
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, saved) = save_real_filter(dir.path());
        let mut damaged = saved.clone();
        let flipped = damaged.len() - 100;
        damaged[flipped] ^= 0x01;
        fs::write(&path, &damaged).expect("write damaged file");

        // Act
        let rejected = load_small(&path);
        fs::write(&path, frame_payload(&damaged[HEADER_LEN..])).expect("write repaired file");
        let accepted = load_small(&path);

        // Assert: with the checksum made to match, the same flip is accepted.
        assert_stale(rejected, StaleReason::Corrupt);
        let filter = assert_fresh(accepted);
        let original = assert_fresh({
            fs::write(&path, &saved).expect("restore original");
            load_small(&path)
        });
        assert_ne!(filter, original);
    }

    #[test]
    fn a_wrong_magic_with_a_matching_checksum_is_corrupt() {
        // Arrange: the checksum covers the payload only, so it still matches.
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, mut bytes) = save_real_filter(dir.path());
        bytes[..FILE_MAGIC.len()].copy_from_slice(b"XXXX");
        fs::write(&path, &bytes).expect("write");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_stale(loaded, StaleReason::Corrupt);
    }

    #[test]
    fn a_file_cut_short_at_any_boundary_is_corrupt() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, full) = save_real_filter(dir.path());
        let lengths = [
            0,
            1,
            3,
            4,
            5,
            HEADER_LEN - 1,
            HEADER_LEN,
            HEADER_LEN + 1,
            full.len() / 2,
            full.len() - 1,
        ];

        for kept in lengths {
            fs::write(&path, &full[..kept]).expect("truncate");

            // Act
            let loaded = load_small(&path);

            // Assert
            assert!(
                matches!(&loaded, FilterLoad::Stale(StaleReason::Corrupt)),
                "{kept} of {} bytes: got {loaded:?}",
                full.len()
            );
        }
    }

    #[test]
    fn a_header_only_file_with_a_valid_checksum_is_corrupt() {
        // Arrange: magic plus the checksum of an empty payload, nothing after.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let header_only = frame_payload(&[]);
        assert_eq!(header_only.len(), HEADER_LEN);
        fs::write(&path, header_only).expect("write");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_stale(loaded, StaleReason::Corrupt);
    }

    #[test]
    fn the_on_disk_format_and_hash_seed_are_pinned() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let filter = filter_with(small_config(), &sample_hashes(0..10));

        // Act
        filter
            .save(&path, &small_config(), log_id(), Some(offset(2, 40)))
            .expect("save");
        let digest = Sha256::digest(fs::read(&path).expect("read back"));
        let answers: String = sample_hashes(0..64)
            .iter()
            .map(|hash| if filter.might_contain(hash) { '1' } else { '0' })
            .collect();

        // Assert: a change here means saved filters no longer load as they were
        // built, so the format version must be bumped with it.
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        assert_eq!(
            hex,
            "5bfeb3bc399079b6c6521368cdd5385a40171b534e56e82ba4e8f2280cf80abe"
        );
        assert_eq!(
            answers,
            "1111111111000000000000000000000000000000000000000000000000000000"
        );
    }

    #[test]
    fn a_filter_saved_with_a_different_expected_items_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, _) = save_real_filter(dir.path());
        let other = BloomConfig::new(small_config().expected_items() + 1, 0.01).expect("config");

        // Act
        let loaded = HashBloom::load(&path, &other, log_id(), None);

        // Assert
        assert_stale(loaded, StaleReason::ConfigMismatch);
    }

    #[test]
    fn a_filter_saved_with_a_different_false_positive_rate_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let (path, _) = save_real_filter(dir.path());
        let other = BloomConfig::new(small_config().expected_items(), 0.02).expect("config");

        // Act
        let loaded = HashBloom::load(&path, &other, log_id(), None);

        // Assert
        assert_stale(loaded, StaleReason::ConfigMismatch);
    }

    #[test]
    fn a_filter_saved_under_another_content_hash_scheme_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let mut other_scheme = persisted(vec![0; 8], 3);
        other_scheme.scheme = CONTENT_HASH_SCHEME + 1;
        fs::write(&path, frame(&other_scheme).expect("frame")).expect("write");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_stale(loaded, StaleReason::ConfigMismatch);
    }

    #[test]
    fn saving_over_an_existing_file_replaces_it_and_leaves_no_staged_file() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let first = filter_with(small_config(), &sample_hashes(0..100));
        let second = filter_with(small_config(), &sample_hashes(100..200));
        let head = Some(offset(0, 2));
        first
            .save(&path, &small_config(), log_id(), Some(offset(0, 1)))
            .expect("save first");

        // Act
        second
            .save(&path, &small_config(), log_id(), head)
            .expect("save second");
        let loaded = HashBloom::load(&path, &small_config(), log_id(), head);

        // Assert
        assert_eq!(assert_fresh(loaded), second);
        let names: Vec<_> = fs::read_dir(dir.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert_eq!(names, ["bloom.bin"]);
    }

    #[test]
    fn a_checksummed_filter_with_no_bits_is_corrupt() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        fs::write(&path, frame(&persisted(Vec::new(), 7)).expect("frame")).expect("write");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_stale(loaded, StaleReason::Corrupt);
    }

    #[test]
    fn a_checksummed_filter_with_an_out_of_range_hash_count_is_corrupt() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");

        for num_hashes in [0, MAX_NUM_HASHES + 1, u32::MAX] {
            fs::write(
                &path,
                frame(&persisted(vec![0; 8], num_hashes)).expect("frame"),
            )
            .expect("write");

            // Act
            let loaded = load_small(&path);

            // Assert
            assert!(
                matches!(&loaded, FilterLoad::Stale(StaleReason::Corrupt)),
                "{num_hashes} hashes: got {loaded:?}"
            );
        }
    }

    #[test]
    fn a_checksummed_filter_claiming_an_enormous_bit_vector_is_corrupt() {
        // Arrange: bits is the last field, so swap its empty length prefix for a
        // claim of 2^40 words with no data behind it.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let mut payload = postcard::to_stdvec(&persisted(Vec::new(), 7)).expect("encode");
        payload.pop();
        payload.extend(postcard::to_stdvec(&(1_u64 << 40)).expect("encode length"));
        fs::write(&path, frame_payload(&payload)).expect("write");

        // Act
        let loaded = load_small(&path);

        // Assert
        assert_stale(loaded, StaleReason::Corrupt);
    }

    #[test]
    fn a_filter_saved_for_another_log_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let head = Some(offset(0, 5));
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path, &small_config(), log_id(), head)
            .expect("save");

        // Act
        let loaded = HashBloom::load(&path, &small_config(), Uuid::from_u128(0x9999), head);

        // Assert
        assert_stale(loaded, StaleReason::LogMismatch);
    }

    #[test]
    fn a_filter_behind_the_log_head_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path, &small_config(), log_id(), Some(offset(1, 9)))
            .expect("save");

        for head in [offset(1, 10), offset(2, 0)] {
            // Act
            let loaded = HashBloom::load(&path, &small_config(), log_id(), Some(head));

            // Assert
            assert_stale(loaded, StaleReason::BehindLog);
        }
    }

    #[test]
    fn a_filter_saved_for_an_empty_log_is_behind_a_log_that_has_since_grown() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        HashBloom::new(small_config())
            .save(&path, &small_config(), log_id(), None)
            .expect("save");

        // Act
        let loaded = HashBloom::load(&path, &small_config(), log_id(), Some(offset(0, 0)));

        // Assert
        assert_stale(loaded, StaleReason::BehindLog);
    }

    #[test]
    fn a_filter_at_the_log_head_is_fresh() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let head = Some(offset(3, 7));
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path, &small_config(), log_id(), head)
            .expect("save");

        // Act
        let loaded = HashBloom::load(&path, &small_config(), log_id(), head);

        // Assert
        assert_fresh(loaded);
    }

    #[test]
    fn a_filter_ahead_of_the_log_head_is_kept() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path, &small_config(), log_id(), Some(offset(3, 7)))
            .expect("save");

        // Act
        let loaded = HashBloom::load(&path, &small_config(), log_id(), Some(offset(3, 6)));

        // Assert
        assert_fresh(loaded);
    }
}
