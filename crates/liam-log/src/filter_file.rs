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

use fastbloom::BloomFilter;
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
    filter: BloomFilter,
}

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
            filter: self.filter.clone(),
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
        FilterLoad::Fresh(Self {
            filter: persisted.filter,
        })
    }
}

fn frame(persisted: &PersistedFilter) -> io::Result<Vec<u8>> {
    let payload = postcard::to_stdvec(persisted).map_err(io::Error::other)?;
    let mut bytes = Vec::with_capacity(HEADER_LEN + payload.len());
    bytes.extend(FILE_MAGIC);
    bytes.extend(crc32fast::hash(&payload).to_le_bytes());
    bytes.extend(payload);
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Option<PersistedFilter> {
    let (magic, rest) = bytes.split_at_checked(FILE_MAGIC.len())?;
    let (checksum, payload) = rest.split_at_checked(size_of::<u32>())?;
    if magic != FILE_MAGIC || checksum != crc32fast::hash(payload).to_le_bytes() {
        return None;
    }
    postcard::from_bytes(payload).ok()
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

    #[test]
    fn a_saved_filter_loads_back_equal() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let hashes = sample_hashes(0..1_000);
        let original = filter_with(small_config(), &hashes);
        let head = Some(offset(2, 40));

        // Act
        original.save(&path, &small_config(), log_id(), head).expect("save");
        let loaded = HashBloom::load(&path, &small_config(), log_id(), head);

        // Assert
        match loaded {
            FilterLoad::Fresh(filter) => {
                assert_eq!(filter, original);
                assert!(hashes.iter().all(|hash| filter.might_contain(hash)));
            }
            FilterLoad::Stale(reason) => panic!("expected a usable filter, got stale: {reason:?}"),
        }
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
        let loaded = HashBloom::load(&path, &small_config(), log_id(), None);

        // Assert
        assert!(matches!(loaded, FilterLoad::Fresh(_)), "got {loaded:?}");
    }

    #[test]
    fn a_missing_filter_file_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("absent.bin");

        // Act
        let loaded = HashBloom::load(&path, &small_config(), log_id(), None);

        // Assert
        assert_stale(loaded, StaleReason::Missing);
    }

    #[test]
    fn a_filter_path_that_cannot_be_read_is_unreadable_not_missing() {
        // Arrange: a directory exists at the path but cannot be read as a file.
        let dir = tempfile::tempdir().expect("tempdir");

        // Act
        let loaded = HashBloom::load(dir.path(), &small_config(), log_id(), None);

        // Assert
        assert_stale(loaded, StaleReason::Unreadable);
    }

    #[test]
    fn a_corrupt_filter_file_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        std::fs::write(&path, b"this is not a bloom filter").expect("write garbage");

        // Act
        let loaded = HashBloom::load(&path, &small_config(), log_id(), None);

        // Assert
        assert_stale(loaded, StaleReason::Corrupt);
    }

    #[test]
    fn a_truncated_filter_file_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path, &small_config(), log_id(), None)
            .expect("save");
        let full = std::fs::read(&path).expect("read back");

        for kept in [0, 1, full.len() / 2, full.len() - 1] {
            std::fs::write(&path, &full[..kept]).expect("truncate");

            // Act
            let loaded = HashBloom::load(&path, &small_config(), log_id(), None);

            // Assert
            assert_stale(loaded, StaleReason::Corrupt);
        }
    }

    #[test]
    fn a_filter_saved_under_a_different_config_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        filter_with(small_config(), &sample_hashes(0..500))
            .save(&path, &small_config(), log_id(), None)
            .expect("save");
        let other_config = BloomConfig::new(50_000, 0.001).expect("valid config");

        // Act
        let loaded = HashBloom::load(&path, &other_config, log_id(), None);

        // Assert
        assert_stale(loaded, StaleReason::ConfigMismatch);
    }

    #[test]
    fn a_filter_saved_under_another_content_hash_scheme_is_stale() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bloom.bin");
        let config = small_config();
        let persisted = PersistedFilter {
            scheme: CONTENT_HASH_SCHEME + 1,
            expected_items: config.expected_items() as u64,
            false_positive_rate: config.false_positive_rate(),
            log_id: log_id().into_bytes(),
            folded_through: None,
            filter: HashBloom::new(config.clone()).filter,
        };
        std::fs::write(&path, frame(&persisted).expect("frame")).expect("write");

        // Act
        let loaded = HashBloom::load(&path, &config, log_id(), None);

        // Assert
        assert_stale(loaded, StaleReason::ConfigMismatch);
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
        assert!(matches!(loaded, FilterLoad::Fresh(_)), "got {loaded:?}");
    }
}
