// SPDX-License-Identifier: Apache-2.0

//! Append-only event log for LIAM.
//!
//! The store keeps derived state (graph, facts, summaries); this crate keeps
//! the immutable record those are rebuilt from, so a write-ahead log and its
//! compacted Parquet segments outlive any change to the derived schema.

pub mod event;
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
    fn append(&mut self, event: &LogEvent) -> Result<LogOffset, WalError>;

    /// Identifies this log across restarts.
    fn log_id(&self) -> Uuid;
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
    use crate::event::{
        LogEvent, LogPayload, TombstoneTable, TombstoneTarget, CURRENT_SCHEMA_VERSION,
    };

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
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        assert_eq!(env!("CARGO_PKG_NAME"), "liam-log");
    }
}
