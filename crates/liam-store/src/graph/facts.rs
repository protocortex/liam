// SPDX-License-Identifier: Apache-2.0
//! The state extraction keeps between runs: one log cursor per extractor and a
//! failure counter, with a quarantine flag, per batch. All are plain rows, so a
//! reopened store resumes where the last run stopped.

use std::collections::BTreeSet;

use liam_log::LogOffset;
use sha2::{Digest, Sha256};

use super::log_cursor::{offset_from_columns, offset_to_columns};
use super::Graph;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::types::ExtractorName;
use crate::value::{Row, Value};

const READ_CURSOR_SQL: &str =
    "SELECT last_segment, last_index FROM extraction_cursor WHERE name = ?1";

// The WHERE makes the compare-and-set one statement, so two writers cannot
// each pass a read check and then move the cursor back past the other. It
// returns a row only when the cursor moved.
const ADVANCE_CURSOR_SQL: &str = "INSERT INTO extraction_cursor
       (name, last_segment, last_index, updated_at) VALUES (?1, ?2, ?3, ?4)
     ON CONFLICT(name) DO UPDATE SET
       last_segment = excluded.last_segment,
       last_index = excluded.last_index,
       updated_at = excluded.updated_at
     WHERE (excluded.last_segment, excluded.last_index)
         > (extraction_cursor.last_segment, extraction_cursor.last_index)
     RETURNING last_segment";

// Returns the new count, or no row when the count is already at the limit.
const RECORD_FAILURE_SQL: &str = "INSERT INTO extraction_attempts
       (batch_key, event_ids, failures, last_error, updated_at) VALUES (?1, ?2, 1, ?3, ?4)
     ON CONFLICT(batch_key) DO UPDATE SET
       failures = failures + 1,
       event_ids = excluded.event_ids,
       last_error = excluded.last_error,
       updated_at = excluded.updated_at
     WHERE failures < ?5
     RETURNING failures";

const READ_FAILURES_SQL: &str = "SELECT failures FROM extraction_attempts WHERE batch_key = ?1";

const CLEAR_FAILURES_SQL: &str = "DELETE FROM extraction_attempts WHERE batch_key = ?1";

const QUARANTINE_SQL: &str =
    "UPDATE extraction_attempts SET quarantined = 1, updated_at = ?2 WHERE batch_key = ?1";

const IS_QUARANTINED_SQL: &str = "SELECT quarantined FROM extraction_attempts WHERE batch_key = ?1";

const QUARANTINED_EVENTS_SQL: &str =
    "SELECT event_ids FROM extraction_attempts WHERE quarantined = 1";

/// The key a batch of events is tracked under: SHA-256 over the prompt
/// version, the model hash and the event ids, as lowercase hex. Event order
/// does not matter, and every part is length-prefixed so `("ab", "c")` and
/// `("a", "bc")` differ. A new prompt or model gives every batch a fresh key,
/// which re-drives events quarantined under the old one.
pub fn batch_key(prompt_version: &str, model_sha256: &str, event_ids: &[String]) -> String {
    let mut sorted: Vec<&str> = event_ids.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let mut hasher = Sha256::new();
    let mut part = |bytes: &[u8]| {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    part(prompt_version.as_bytes());
    part(model_sha256.as_bytes());
    for id in &sorted {
        part(id.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn corrupt(detail: String) -> Error {
    Error::CorruptExtractionState(detail)
}

fn offset_from_row(row: &Row) -> Result<LogOffset> {
    let (segment, index) = (row.get_i64(0)?, row.get_i64(1)?);
    offset_from_columns(segment, index).ok_or_else(|| {
        corrupt(format!(
            "extraction_cursor holds the offset {segment}, {index}"
        ))
    })
}

fn failures_from_row(row: &Row, batch_key: &str) -> Result<u32> {
    let stored = row.get_i64(0)?;
    u32::try_from(stored).map_err(|_| {
        corrupt(format!(
            "extraction_attempts holds {stored} failures for batch {batch_key}"
        ))
    })
}

impl<B: Backend> Graph<B> {
    /// Where `name` has read the log up to, or `None` before its first advance.
    /// Offset 0 is a real position and is not the same as `None`.
    pub async fn extraction_cursor(&self, name: ExtractorName) -> Result<Option<LogOffset>> {
        let rows = self
            .backend
            .query(READ_CURSOR_SQL, &[name.as_str().into()])
            .await?;
        rows.first().map(offset_from_row).transpose()
    }

    /// Moves the `name` cursor to `offset` and returns where it stands. An
    /// offset equal to the stored one changes nothing; a lower one is refused.
    /// Segment and index compare together, so a later segment with a lower
    /// index is forward.
    pub async fn advance_extraction_cursor(
        &self,
        name: ExtractorName,
        offset: LogOffset,
    ) -> Result<LogOffset> {
        let (segment, index) =
            offset_to_columns(offset).ok_or(Error::ExtractionOffsetOutOfRange(offset))?;
        // One transaction, so a concurrent advance cannot land between the
        // refused update and the read that tells equal from behind.
        let mut tx = self.backend.begin().await?;
        let moved = tx
            .query(
                ADVANCE_CURSOR_SQL,
                &[
                    name.as_str().into(),
                    segment.into(),
                    index.into(),
                    self.clock.now().into(),
                ],
            )
            .await?;
        if !moved.is_empty() {
            tx.commit().await?;
            return Ok(offset);
        }
        let rows = tx.query(READ_CURSOR_SQL, &[name.as_str().into()]).await?;
        tx.rollback().await?;
        let stored = rows
            .first()
            .map(offset_from_row)
            .transpose()?
            .ok_or_else(|| {
                Error::Backend("extraction_cursor has no row right after a refused advance".into())
            })?;
        if stored == offset {
            Ok(offset)
        } else {
            Err(Error::ExtractionCursorRegressed {
                extractor: name.as_str(),
                stored,
                attempted: offset,
            })
        }
    }

    /// Counts one more failure for the batch and returns the new total. The
    /// batch's event ids and last error are replaced by this failure's.
    pub async fn record_batch_failure(
        &self,
        batch_key: &str,
        event_ids: &[String],
        error: &str,
    ) -> Result<u32> {
        let mut tx = self.backend.begin().await?;
        let rows = tx
            .query(
                RECORD_FAILURE_SQL,
                &[
                    batch_key.into(),
                    serde_json::to_string(event_ids)?.into(),
                    error.into(),
                    self.clock.now().into(),
                    Value::Int(i64::from(u32::MAX)),
                ],
            )
            .await?;
        let Some(row) = rows.first() else {
            tx.rollback().await?;
            return Err(Error::ExtractionCounterOverflow {
                batch_key: batch_key.to_owned(),
            });
        };
        let count = failures_from_row(row, batch_key)?;
        tx.commit().await?;
        Ok(count)
    }

    /// How many times the batch has failed since its last reset.
    pub async fn batch_failures(&self, batch_key: &str) -> Result<u32> {
        let rows = self
            .backend
            .query(READ_FAILURES_SQL, &[batch_key.into()])
            .await?;
        rows.first()
            .map_or(Ok(0), |row| failures_from_row(row, batch_key))
    }

    /// Forgets the batch's failures, as after it succeeded.
    pub async fn clear_batch_failures(&self, batch_key: &str) -> Result<()> {
        self.backend
            .execute(CLEAR_FAILURES_SQL, &[batch_key.into()])
            .await?;
        Ok(())
    }

    /// Marks a batch that has failed for good, so its events are skipped. The
    /// batch must have a recorded failure.
    pub async fn quarantine(&self, batch_key: &str) -> Result<()> {
        let changed = self
            .backend
            .execute(QUARANTINE_SQL, &[batch_key.into(), self.clock.now().into()])
            .await?;
        if changed == 0 {
            return Err(Error::ExtractionBatchUnknown {
                batch_key: batch_key.to_owned(),
            });
        }
        Ok(())
    }

    /// Whether this exact batch key is quarantined. A batch built from a new
    /// prompt version has a new key, so it is not.
    pub async fn is_quarantined(&self, batch_key: &str) -> Result<bool> {
        let rows = self
            .backend
            .query(IS_QUARANTINED_SQL, &[batch_key.into()])
            .await?;
        rows.first()
            .map_or(Ok(false), |row| Ok(row.get_i64(0)? != 0))
    }

    /// Every event id in any quarantined batch, whatever prompt version
    /// quarantined it. Use `is_quarantined` to ask about one batch key.
    pub async fn quarantined_event_ids(&self) -> Result<BTreeSet<String>> {
        let rows = self.backend.query(QUARANTINED_EVENTS_SQL, &[]).await?;
        let mut ids = BTreeSet::new();
        for row in &rows {
            let batch: Vec<String> = serde_json::from_str(&row.get_string(0)?)?;
            ids.extend(batch);
        }
        Ok(ids)
    }
}

#[cfg(all(test, feature = "backend-libsql"))]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;
    use crate::clock::FixedClock;
    use crate::ids::Millis;
    use crate::types::GraphConfig;
    use crate::DefaultGraph;

    const DIMS: usize = 8;
    const KEY: &str = "batch-a";
    const I64_MAX: u64 = i64::MAX as u64;
    const CONCURRENT_OFFSETS: u64 = 50;

    fn at(segment: u64, index: u64) -> LogOffset {
        LogOffset { segment, index }
    }

    fn db_path(dir: &TempDir) -> String {
        dir.path().join("graph.db").to_str().unwrap().to_owned()
    }

    async fn open(path: &str) -> DefaultGraph {
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock)
            .await
            .unwrap()
    }

    async fn seed_failures(g: &DefaultGraph, failures: i64) {
        g.backend
            .execute(
                "INSERT INTO extraction_attempts
                   (batch_key, event_ids, failures, last_error, updated_at)
                 VALUES (?1, '[]', ?2, 'seeded', 0)",
                &[KEY.into(), Value::Int(failures)],
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_cursor_never_advanced_reads_as_none() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;

        // Act
        let cursor = g.extraction_cursor(ExtractorName::Rules).await.unwrap();

        // Assert
        assert_eq!(cursor, None);
    }

    #[tokio::test]
    async fn the_first_position_zero_zero_is_distinct_from_none() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;

        // Act
        let returned = g
            .advance_extraction_cursor(ExtractorName::Rules, at(0, 0))
            .await
            .unwrap();

        // Assert
        assert_eq!(returned, at(0, 0));
        assert_eq!(
            g.extraction_cursor(ExtractorName::Rules).await.unwrap(),
            Some(at(0, 0))
        );
    }

    #[tokio::test]
    async fn a_lower_offset_is_refused_and_the_cursor_stays() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        g.advance_extraction_cursor(ExtractorName::Llm, at(2, 9))
            .await
            .unwrap();

        // Act
        let result = g
            .advance_extraction_cursor(ExtractorName::Llm, at(2, 7))
            .await;

        // Assert
        assert!(matches!(
            result,
            Err(Error::ExtractionCursorRegressed { stored, attempted, .. })
                if stored == at(2, 9) && attempted == at(2, 7)
        ));
        assert_eq!(
            g.extraction_cursor(ExtractorName::Llm).await.unwrap(),
            Some(at(2, 9))
        );
    }

    #[tokio::test]
    async fn the_same_offset_again_succeeds_unchanged() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        g.advance_extraction_cursor(ExtractorName::Llm, at(2, 9))
            .await
            .unwrap();

        // Act
        let returned = g
            .advance_extraction_cursor(ExtractorName::Llm, at(2, 9))
            .await
            .unwrap();

        // Assert
        assert_eq!(returned, at(2, 9));
    }

    #[tokio::test]
    async fn a_later_segment_with_a_lower_index_is_forward() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        g.advance_extraction_cursor(ExtractorName::Rules, at(1, 50))
            .await
            .unwrap();

        // Act
        let returned = g
            .advance_extraction_cursor(ExtractorName::Rules, at(2, 0))
            .await
            .unwrap();

        // Assert
        assert_eq!(returned, at(2, 0));
        assert_eq!(
            g.extraction_cursor(ExtractorName::Rules).await.unwrap(),
            Some(at(2, 0))
        );
    }

    #[tokio::test]
    async fn an_earlier_segment_with_a_higher_index_is_backwards() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        g.advance_extraction_cursor(ExtractorName::Rules, at(2, 0))
            .await
            .unwrap();

        // Act
        let result = g
            .advance_extraction_cursor(ExtractorName::Rules, at(1, 50))
            .await;

        // Assert
        assert!(matches!(
            result,
            Err(Error::ExtractionCursorRegressed { .. })
        ));
    }

    #[tokio::test]
    async fn the_two_cursors_move_independently() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;

        // Act
        g.advance_extraction_cursor(ExtractorName::Rules, at(0, 9))
            .await
            .unwrap();
        let llm_after_rules = g.extraction_cursor(ExtractorName::Llm).await.unwrap();
        g.advance_extraction_cursor(ExtractorName::Llm, at(0, 4))
            .await
            .unwrap();

        // Assert
        assert_eq!(llm_after_rules, None);
        assert_eq!(
            g.extraction_cursor(ExtractorName::Rules).await.unwrap(),
            Some(at(0, 9))
        );
        assert_eq!(
            g.extraction_cursor(ExtractorName::Llm).await.unwrap(),
            Some(at(0, 4))
        );
    }

    #[tokio::test]
    async fn the_largest_storable_offset_is_accepted() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        let max = at(I64_MAX, I64_MAX);

        // Act
        g.advance_extraction_cursor(ExtractorName::Rules, max)
            .await
            .unwrap();

        // Assert
        assert_eq!(
            g.extraction_cursor(ExtractorName::Rules).await.unwrap(),
            Some(max)
        );
    }

    #[tokio::test]
    async fn an_offset_part_above_i64_max_is_refused_and_nothing_is_stored() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;

        for offset in [at(I64_MAX + 1, 0), at(0, I64_MAX + 1)] {
            // Act
            let result = g
                .advance_extraction_cursor(ExtractorName::Rules, offset)
                .await;

            // Assert
            assert!(matches!(
                result,
                Err(Error::ExtractionOffsetOutOfRange(refused)) if refused == offset
            ));
        }
        assert_eq!(
            g.extraction_cursor(ExtractorName::Rules).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn a_stored_negative_offset_is_reported_as_corrupt() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        g.backend
            .execute(
                "INSERT INTO extraction_cursor (name, last_segment, last_index, updated_at)
                 VALUES ('rules', -1, 0, 0)",
                &[],
            )
            .await
            .unwrap();

        // Act
        let result = g.extraction_cursor(ExtractorName::Rules).await;

        // Assert
        assert!(matches!(result, Err(Error::CorruptExtractionState(_))));
    }

    #[tokio::test]
    async fn cursors_and_counters_survive_a_reopen() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = db_path(&dir);
        {
            let g = open(&path).await;
            g.advance_extraction_cursor(ExtractorName::Rules, at(3, 8))
                .await
                .unwrap();
            g.record_batch_failure(KEY, &["e1".into()], "timeout")
                .await
                .unwrap();
            g.record_batch_failure(KEY, &["e1".into()], "timeout")
                .await
                .unwrap();
        }

        // Act
        let g = open(&path).await;

        // Assert
        assert_eq!(
            g.extraction_cursor(ExtractorName::Rules).await.unwrap(),
            Some(at(3, 8))
        );
        assert_eq!(g.extraction_cursor(ExtractorName::Llm).await.unwrap(), None);
        assert_eq!(g.batch_failures(KEY).await.unwrap(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_tasks_advancing_at_once_leave_the_cursor_at_the_highest() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = Arc::new(open(&db_path(&dir)).await);
        let advance = |g: Arc<DefaultGraph>, offsets: Vec<LogOffset>| {
            tokio::spawn(async move {
                for offset in offsets {
                    // Losing the race is refused; anything else is a real failure.
                    match g
                        .advance_extraction_cursor(ExtractorName::Llm, offset)
                        .await
                    {
                        Ok(_) | Err(Error::ExtractionCursorRegressed { .. }) => {}
                        Err(other) => panic!("unexpected advance error: {other}"),
                    }
                }
            })
        };
        let ascending = (0..CONCURRENT_OFFSETS).map(|i| at(0, i)).collect();
        let descending = (0..CONCURRENT_OFFSETS).rev().map(|i| at(0, i)).collect();

        // Act
        let first = advance(g.clone(), ascending);
        let second = advance(g.clone(), descending);
        first.await.unwrap();
        second.await.unwrap();

        // Assert
        assert_eq!(
            g.extraction_cursor(ExtractorName::Llm).await.unwrap(),
            Some(at(0, CONCURRENT_OFFSETS - 1))
        );
    }

    #[tokio::test]
    async fn an_unseen_batch_has_no_failures() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;

        // Act
        let failures = g.batch_failures(KEY).await.unwrap();

        // Assert
        assert_eq!(failures, 0);
    }

    #[tokio::test]
    async fn each_failure_adds_one_and_clearing_resets_to_zero() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        let events = ["e1".to_owned(), "e2".to_owned()];

        // Act
        let counts = [
            g.record_batch_failure(KEY, &events, "first").await.unwrap(),
            g.record_batch_failure(KEY, &events, "second")
                .await
                .unwrap(),
            g.record_batch_failure(KEY, &events, "third").await.unwrap(),
        ];
        let before_clear = g.batch_failures(KEY).await.unwrap();
        g.clear_batch_failures(KEY).await.unwrap();
        let after_clear = g.batch_failures(KEY).await.unwrap();

        // Assert
        assert_eq!(counts, [1, 2, 3]);
        assert_eq!(before_clear, 3);
        assert_eq!(after_clear, 0);
    }

    #[tokio::test]
    async fn clearing_a_batch_never_seen_is_fine() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;

        // Act
        let result = g.clear_batch_failures(KEY).await;

        // Assert
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn batches_count_separately() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        g.record_batch_failure("batch-a", &[], "x").await.unwrap();

        // Act
        let other = g.batch_failures("batch-b").await.unwrap();

        // Assert
        assert_eq!(other, 0);
    }

    #[tokio::test]
    async fn the_counter_reaches_u32_max_and_then_refuses_to_wrap() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        seed_failures(&g, i64::from(u32::MAX) - 1).await;

        // Act
        let at_max = g.record_batch_failure(KEY, &[], "x").await.unwrap();
        let past_max = g.record_batch_failure(KEY, &[], "x").await;

        // Assert
        assert_eq!(at_max, u32::MAX);
        assert!(matches!(
            past_max,
            Err(Error::ExtractionCounterOverflow { batch_key }) if batch_key == KEY
        ));
        assert_eq!(g.batch_failures(KEY).await.unwrap(), u32::MAX);
    }

    #[tokio::test]
    async fn a_stored_count_outside_u32_is_reported_as_corrupt() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        seed_failures(&g, i64::from(u32::MAX) + 1).await;

        // Act
        let result = g.batch_failures(KEY).await;

        // Assert
        assert!(matches!(result, Err(Error::CorruptExtractionState(_))));
    }

    #[tokio::test]
    async fn five_then_nine_within_one_segment_moves_forward() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        g.advance_extraction_cursor(ExtractorName::Rules, at(0, 5))
            .await
            .unwrap();

        // Act
        g.advance_extraction_cursor(ExtractorName::Rules, at(0, 9))
            .await
            .unwrap();

        // Assert
        assert_eq!(
            g.extraction_cursor(ExtractorName::Rules).await.unwrap(),
            Some(at(0, 9))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_equal_advance_is_not_called_a_regression_by_a_later_forward_one() {
        // Arrange: hold the write lock so the equal advance queues first, then
        // a forward advance queues behind it.
        let dir = TempDir::new().unwrap();
        let g = Arc::new(open(&db_path(&dir)).await);
        g.advance_extraction_cursor(ExtractorName::Llm, at(0, 5))
            .await
            .unwrap();
        let lock = g.backend.begin().await.unwrap();
        let spawn_advance = |offset: LogOffset| {
            let g = g.clone();
            tokio::spawn(async move {
                g.advance_extraction_cursor(ExtractorName::Llm, offset)
                    .await
            })
        };
        let mut equal = spawn_advance(at(0, 5));
        let queued = tokio::time::timeout(std::time::Duration::from_millis(100), &mut equal).await;
        assert!(queued.is_err(), "the advance must wait for the write lock");
        let forward = spawn_advance(at(0, 9));

        // Act
        lock.rollback().await.unwrap();
        let equal = equal.await.unwrap();
        let forward = forward.await.unwrap();

        // Assert
        assert_eq!(equal.unwrap(), at(0, 5));
        assert_eq!(forward.unwrap(), at(0, 9));
    }

    #[tokio::test]
    async fn a_second_failure_replaces_the_events_and_the_error() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        let first = g
            .record_batch_failure(KEY, &["e1".to_owned()], "old error")
            .await
            .unwrap();

        // Act
        let second = g
            .record_batch_failure(KEY, &["e1".to_owned(), "e2".to_owned()], "new error")
            .await
            .unwrap();

        // Assert
        let rows = g
            .backend
            .query(
                "SELECT event_ids, last_error, failures FROM extraction_attempts WHERE batch_key = ?1",
                &[KEY.into()],
            )
            .await
            .unwrap();
        assert_eq!((first, second), (1, 2));
        assert_eq!(rows[0].get_string(0).unwrap(), r#"["e1","e2"]"#);
        assert_eq!(rows[0].get_string(1).unwrap(), "new error");
        assert_eq!(rows[0].get_i64(2).unwrap(), 2);
    }

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    #[test]
    fn a_batch_key_is_stable_and_ignores_event_order() {
        // Arrange
        let forward = ids(&["e1", "e2", "e3"]);
        let shuffled = ids(&["e3", "e1", "e2"]);

        // Act
        let first = batch_key("p1", "m1", &forward);
        let again = batch_key("p1", "m1", &forward);
        let reordered = batch_key("p1", "m1", &shuffled);

        // Assert
        assert_eq!(first, again);
        assert_eq!(first, reordered);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn a_batch_key_changes_with_each_input_and_resists_shifted_boundaries() {
        // Arrange
        let events = ids(&["e1", "e2"]);
        let base = batch_key("p1", "m1", &events);

        // Act
        let other_prompt = batch_key("p2", "m1", &events);
        let other_model = batch_key("p1", "m2", &events);
        let other_events = batch_key("p1", "m1", &ids(&["e1", "e3"]));
        let shifted_a = batch_key("ab", "c", &events);
        let shifted_b = batch_key("a", "bc", &events);
        let joined = batch_key("p1", "m1", &ids(&["ab", "c"]));
        let split = batch_key("p1", "m1", &ids(&["a", "bc"]));

        // Assert
        assert_ne!(base, other_prompt);
        assert_ne!(base, other_model);
        assert_ne!(base, other_events);
        assert_ne!(shifted_a, shifted_b);
        assert_ne!(joined, split);
    }

    #[tokio::test]
    async fn a_new_prompt_version_key_is_not_quarantined_for_the_same_events() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        let events = ids(&["e1", "e2"]);
        let old = batch_key("p1", "m1", &events);
        let new = batch_key("p2", "m1", &events);
        g.record_batch_failure(&old, &events, "bad").await.unwrap();
        g.quarantine(&old).await.unwrap();

        // Act
        let old_quarantined = g.is_quarantined(&old).await.unwrap();
        let new_quarantined = g.is_quarantined(&new).await.unwrap();

        // Assert
        assert!(old_quarantined);
        assert!(!new_quarantined);
    }

    #[tokio::test]
    async fn quarantined_event_ids_merge_every_quarantined_batch_sorted_and_deduplicated() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;
        for (key, events) in [
            ("k1", ids(&["e3", "e1"])),
            ("k2", ids(&["e1", "e2"])),
            ("k3", ids(&["e9"])),
        ] {
            g.record_batch_failure(key, &events, "bad").await.unwrap();
        }
        g.quarantine("k1").await.unwrap();
        g.quarantine("k2").await.unwrap();

        // Act
        let quarantined = g.quarantined_event_ids().await.unwrap();

        // Assert
        let listed: Vec<&str> = quarantined.iter().map(String::as_str).collect();
        assert_eq!(listed, ["e1", "e2", "e3"]);
    }

    #[tokio::test]
    async fn quarantining_an_unknown_key_is_a_typed_error() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let g = open(&db_path(&dir)).await;

        // Act
        let result = g.quarantine("nope").await;

        // Assert
        assert!(matches!(
            result,
            Err(Error::ExtractionBatchUnknown { batch_key }) if batch_key == "nope"
        ));
    }

    #[tokio::test]
    async fn quarantine_persists_across_a_reopen() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = db_path(&dir);
        {
            let g = open(&path).await;
            g.record_batch_failure(KEY, &ids(&["e1"]), "bad")
                .await
                .unwrap();
            g.quarantine(KEY).await.unwrap();
        }

        // Act
        let g = open(&path).await;

        // Assert
        assert!(g.is_quarantined(KEY).await.unwrap());
        assert_eq!(g.quarantined_event_ids().await.unwrap().len(), 1);
    }
}
