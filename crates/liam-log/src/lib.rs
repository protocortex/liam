// SPDX-License-Identifier: Apache-2.0

//! Append-only event log for LIAM.
//!
//! The store keeps derived state (graph, facts, summaries); this crate keeps
//! the immutable record those are rebuilt from, so a write-ahead log and its
//! compacted Parquet segments outlive any change to the derived schema.

pub mod compactor;
pub mod dedup;
pub mod event;
pub mod filter_file;
pub mod hash;
pub mod reader;
pub mod wal;

use uuid::Uuid;

use crate::event::LogEvent;
use crate::wal::WalError;

/// Where an appended record sits: the sequence number of its segment and its
/// ordinal within that segment, so the position does not depend on record size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogOffset {
    pub segment: u64,
    pub index: u64,
}

/// The append side of the log, so the store can swap the real WAL for a double.
///
/// `append` blocks on fsync, so async callers must offload it with
/// `spawn_blocking` or `block_in_place`. Callers serialize appends behind one
/// lock; the trait takes `&mut self` to say so.
pub trait LogWriter: Send {
    /// Appends one event durably and returns the position it was written at.
    ///
    /// After an `Err` from a writer that then reports `WalError::Poisoned`, the
    /// outcome is unknown: the record may still exist after a reopen. Callers
    /// must dedupe by `event_id` when replaying.
    fn append(&mut self, event: &LogEvent) -> Result<LogOffset, WalError>;

    /// Identifies this log across restarts.
    fn log_id(&self) -> Uuid;

    /// The offset of the last record this writer appended or found when it
    /// opened, or `None` for an empty log.
    fn head(&self) -> Option<LogOffset>;
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    //! A `LogWriter` that fails on demand, to exercise the store's
    //! projection-write failure paths without a real disk fault.

    use std::io;

    use super::*;

    enum Failure {
        OnNth(u64),
        AfterFirst(u64),
    }

    /// In-memory writer that rejects appends according to its failure plan.
    pub struct FailingLogWriter {
        log_id: Uuid,
        failure: Failure,
        attempts: u64,
        appended: Vec<LogEvent>,
    }

    impl FailingLogWriter {
        /// Fails only the `nth` append (1 based); every other append succeeds.
        pub fn fail_on_nth(nth: u64) -> Self {
            assert!(nth >= 1, "appends are 1 based, so nth must be at least 1");
            Self::new(Failure::OnNth(nth))
        }

        /// Succeeds for the first `successes` appends, then fails every later one.
        pub fn fail_after(successes: u64) -> Self {
            Self::new(Failure::AfterFirst(successes))
        }

        /// Reports `log_id` instead of a generated one, to stand in for the
        /// same log after a restart.
        pub fn with_log_id(mut self, log_id: Uuid) -> Self {
            self.log_id = log_id;
            self
        }

        /// Events accepted so far, in append order.
        pub fn appended(&self) -> &[LogEvent] {
            &self.appended
        }

        fn new(failure: Failure) -> Self {
            Self {
                log_id: Uuid::now_v7(),
                failure,
                attempts: 0,
                appended: Vec::new(),
            }
        }

        fn injected_error() -> WalError {
            WalError::Io(io::Error::other("injected append failure"))
        }
    }

    impl LogWriter for FailingLogWriter {
        fn append(&mut self, event: &LogEvent) -> Result<LogOffset, WalError> {
            self.attempts += 1;
            let fails = match self.failure {
                Failure::OnNth(nth) => self.attempts == nth,
                Failure::AfterFirst(successes) => self.attempts > successes,
            };
            if fails {
                return Err(Self::injected_error());
            }
            let index = self.appended.len() as u64;
            self.appended.push(event.clone());
            Ok(LogOffset { segment: 0, index })
        }

        fn log_id(&self) -> Uuid {
            self.log_id
        }

        fn head(&self) -> Option<LogOffset> {
            let index = self.appended.len().checked_sub(1)?;
            Some(LogOffset {
                segment: 0,
                index: index as u64,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::fixtures::event;

        fn outcomes(writer: &mut dyn LogWriter, count: usize) -> Vec<bool> {
            (0..count)
                .map(|index| writer.append(&event(index)).is_ok())
                .collect()
        }

        #[test]
        fn fail_on_nth_rejects_only_that_append() {
            // Arrange
            let mut writer = FailingLogWriter::fail_on_nth(3);

            // Act
            let results = outcomes(&mut writer, 5);

            // Assert
            assert_eq!(results, vec![true, true, false, true, true]);
        }

        #[test]
        fn fail_after_rejects_every_append_past_the_point() {
            // Arrange
            let mut writer = FailingLogWriter::fail_after(2);

            // Act
            let results = outcomes(&mut writer, 5);

            // Assert
            assert_eq!(results, vec![true, true, false, false, false]);
        }

        #[test]
        fn fail_on_nth_one_rejects_only_the_first_append() {
            // Arrange
            let mut writer = FailingLogWriter::fail_on_nth(1);

            // Act
            let results = outcomes(&mut writer, 3);

            // Assert
            assert_eq!(results, vec![false, true, true]);
        }

        #[test]
        fn fail_after_zero_rejects_every_append() {
            // Arrange
            let mut writer = FailingLogWriter::fail_after(0);

            // Act
            let results = outcomes(&mut writer, 3);

            // Assert
            assert_eq!(results, vec![false, false, false]);
        }

        #[test]
        #[should_panic(expected = "1 based")]
        fn fail_on_nth_zero_is_rejected_because_appends_count_from_one() {
            FailingLogWriter::fail_on_nth(0);
        }

        #[test]
        fn accepted_appends_return_a_running_index_in_segment_zero() {
            // Arrange
            let mut writer = FailingLogWriter::fail_on_nth(2);

            // Act
            let first = writer.append(&event(0)).expect("first append");
            let rejected = writer.append(&event(1)).is_err();
            let third = writer.append(&event(2)).expect("third append");

            // Assert
            assert_eq!(
                first,
                LogOffset {
                    segment: 0,
                    index: 0
                }
            );
            assert!(rejected);
            assert_eq!(
                third,
                LogOffset {
                    segment: 0,
                    index: 1
                }
            );
        }

        #[test]
        fn rejected_appends_are_not_recorded() {
            // Arrange
            let mut writer = FailingLogWriter::fail_on_nth(2);

            // Act
            outcomes(&mut writer, 3);

            // Assert
            assert_eq!(writer.appended(), [event(0), event(2)]);
        }

        #[test]
        fn injected_failure_is_an_io_error() {
            // Arrange
            let mut writer = FailingLogWriter::fail_after(0);

            // Act
            let error = writer.append(&event(0)).expect_err("append should fail");

            // Assert
            assert!(
                matches!(error, WalError::Io(_)),
                "unexpected error: {error}"
            );
        }

        #[test]
        fn head_is_the_last_accepted_append_and_ignores_a_rejected_one() {
            // Arrange
            let mut writer = FailingLogWriter::fail_on_nth(2);
            let empty = writer.head();

            // Act
            let first = writer.append(&event(0)).expect("first append");
            let after_first = writer.head();
            writer.append(&event(1)).expect_err("rejected append");
            let after_rejected = writer.head();
            let third = writer.append(&event(2)).expect("third append");

            // Assert
            assert_eq!(empty, None);
            assert_eq!(after_first, Some(first));
            assert_eq!(after_rejected, Some(first));
            assert_eq!(writer.head(), Some(third));
        }

        #[test]
        fn double_reports_a_stable_non_nil_log_id() {
            // Arrange
            let writer = FailingLogWriter::fail_after(0);

            // Act
            let (first, second) = (writer.log_id(), writer.log_id());

            // Assert
            assert_eq!(first, second);
            assert!(!first.is_nil());
        }
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::SystemTime;

    use object_store::local::LocalFileSystem;
    use sha2::{Digest, Sha256};

    use crate::compactor::{parquet_name, write_parquet, Compactor};
    use crate::dedup::{BloomConfig, HashBloom};
    use crate::event::{
        EdgeRow, LogEvent, LogPayload, NodeRow, TombstoneTable, TombstoneTarget,
        CURRENT_SCHEMA_VERSION,
    };
    use crate::wal::{segment_name, WalConfig, WalWriter};
    use crate::{LogOffset, LogWriter};

    /// Open-ended end of a validity or transaction interval, in epoch millis.
    pub(crate) const FOREVER: i64 = 4_102_444_800_000;

    /// A distinct event per index; every index encodes to the same byte length.
    pub(crate) fn event(index: usize) -> LogEvent {
        LogEvent {
            event_id: format!("event-{index:03}"),
            content_hash: [index as u8; 32],
            source: "agent-a".into(),
            trust_score: 0.9,
            observed_at: 1_000,
            ingested_at: 2_000,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload: LogPayload::Tombstone(vec![TombstoneTarget {
                table: TombstoneTable::Nodes,
                id: format!("node-{index:03}"),
            }]),
        }
    }

    /// A resolved node row whose hashed content is set by `content`, so rows
    /// with equal `content` hash alike whatever their ids.
    pub(crate) fn node_row(id: &str, content: &str) -> NodeRow {
        NodeRow {
            id: id.into(),
            kind: "fact".into(),
            label: "label".into(),
            content: content.into(),
            producer: "agent-a".into(),
            attributes: "{}".into(),
            scope: Some("proj/a".into()),
            subject: None,
            confidence: 0.75,
            valid_from: 1_000,
            valid_from_supplied: true,
            valid_until: FOREVER,
            tx_from: 2_000,
            tx_to: FOREVER,
        }
    }

    /// A resolved edge row whose hashed content is set by `dst`.
    pub(crate) fn edge_row(id: &str, dst: &str) -> EdgeRow {
        EdgeRow {
            id: id.into(),
            src: "node-1".into(),
            dst: dst.into(),
            edge_type: "relates_to".into(),
            attributes: "{}".into(),
            tx_from: 2_000,
            tx_to: FOREVER,
        }
    }

    /// An event around `payload`. Its source differs from the rows' producer
    /// and its content hash is a sentinel, so a rebuild that reads either
    /// instead of the row content shows up as a wrong hash.
    pub(crate) fn log_event(event_id: &str, payload: LogPayload) -> LogEvent {
        LogEvent {
            event_id: event_id.into(),
            content_hash: EVENT_HASH_SENTINEL,
            source: "log-writer".into(),
            trust_score: 0.9,
            observed_at: 1_000,
            ingested_at: 2_000,
            encryption_key_id: None,
            schema_version: CURRENT_SCHEMA_VERSION,
            payload,
        }
    }

    pub(crate) const EVENT_HASH_SENTINEL: [u8; 32] = [0xEE; 32];

    /// Bytes one record for `event` occupies in a segment.
    pub(crate) fn frame_len(event: &LogEvent) -> u64 {
        (crate::wal::HEADER_BYTES + event.encode().expect("encode event").len()) as u64
    }

    pub(crate) fn wal_path(dir: &Path, sequence: u64) -> PathBuf {
        dir.join(segment_name(sequence))
    }

    pub(crate) fn parquet_path(dir: &Path, sequence: u64) -> PathBuf {
        dir.join(parquet_name(sequence))
    }

    /// Every entry of `dir` with its bytes and modification time, so a test can
    /// tell a rewrite that restores the same bytes from no write at all.
    pub(crate) fn snapshot(dir: &Path) -> BTreeMap<String, (Vec<u8>, SystemTime)> {
        fs::read_dir(dir)
            .expect("list dir")
            .map(|entry| {
                let entry = entry.expect("dir entry");
                let modified = entry.metadata().and_then(|meta| meta.modified());
                let bytes = fs::read(entry.path()).unwrap_or_default();
                let name = entry.file_name().to_string_lossy().into_owned();
                (name, (bytes, modified.expect("mtime")))
            })
            .collect()
    }

    /// Rotates after `events_per_segment` fixture events and never on time.
    pub(crate) fn rotating_config(events_per_segment: usize) -> WalConfig {
        WalConfig {
            segment_max_bytes: events_per_segment as u64 * frame_len(&event(0)),
            rotate_interval_secs: u64::MAX,
        }
    }

    /// Appends `event(0)..event(count)` to a WAL in `dir` and returns each event
    /// with the offset the writer reported, then closes the writer.
    pub(crate) fn write_log(
        dir: &Path,
        count: usize,
        events_per_segment: usize,
    ) -> Vec<(LogOffset, LogEvent)> {
        let mut writer =
            WalWriter::open_with_system_clock(dir, rotating_config(events_per_segment))
                .expect("open wal");
        (0..count)
            .map(|index| {
                let event = event(index);
                let offset = writer.append(&event).expect("append event");
                (offset, event)
            })
            .collect()
    }

    /// Compacts each named closed segment into Parquet and removes its WAL file.
    pub(crate) async fn compact_segments(dir: &Path, sequences: &[u64]) {
        let compactor = Compactor::local(dir).expect("compactor");
        for sequence in sequences {
            let compacted = compactor.compact_segment(&wal_path(dir, *sequence)).await;
            compacted.expect("compact segment");
        }
    }

    /// Writes the Parquet object for `sequence` holding exactly `events`.
    pub(crate) async fn put_parquet(dir: &Path, sequence: u64, events: &[LogEvent]) {
        let store = LocalFileSystem::new_with_prefix(dir).expect("local store");
        write_parquet(&store, sequence, events)
            .await
            .expect("write parquet");
    }

    pub(crate) fn sample_hash(index: u32) -> [u8; 32] {
        Sha256::digest(index.to_le_bytes()).into()
    }

    pub(crate) fn sample_hashes(range: std::ops::Range<u32>) -> Vec<[u8; 32]> {
        range.map(sample_hash).collect()
    }

    pub(crate) fn small_config() -> BloomConfig {
        BloomConfig::new(5_000, 0.01).expect("valid config")
    }

    pub(crate) fn filter_with(config: BloomConfig, hashes: &[[u8; 32]]) -> HashBloom {
        let mut filter = HashBloom::new(config);
        for hash in hashes {
            filter.insert(hash);
        }
        filter
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        assert_eq!(env!("CARGO_PKG_NAME"), "liam-log");
    }
}
