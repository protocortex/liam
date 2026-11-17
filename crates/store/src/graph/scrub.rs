// SPDX-License-Identifier: Apache-2.0
//! Pushing deleted text out of the database file after a purge. A purge deletes
//! rows; the bytes live on in the full-text index, in the write-ahead log and in
//! free pages until these steps overwrite or drop them.

use std::time::Duration;

use super::Graph;
use crate::backend::{Backend, CheckpointMode, CheckpointResult};
use crate::error::{Error, Result};

/// Times a busy checkpoint is tried before the scrub reports it incomplete.
const CHECKPOINT_ATTEMPTS: usize = 5;
/// How long a checkpoint waits for a reader before it reports busy.
const CHECKPOINT_WAIT: Duration = Duration::from_millis(200);

const FTS_OPTIMIZE: &str = "INSERT INTO nodes_fts(nodes_fts) VALUES('optimize')";
const INCREMENTAL_VACUUM: &str = "PRAGMA incremental_vacuum";
const PAGE_COUNT: &str = "PRAGMA page_count";
const PAGE_SIZE: &str = "PRAGMA page_size";

/// What a scrub did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrubReport {
    /// Size of the database file, in bytes, before and after.
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub checkpoint: CheckpointResult,
}

impl<B: Backend> Graph<B> {
    /// Merges the full-text index so the entries of deleted rows go, empties the
    /// write-ahead log into the database file, and returns the pages the delete
    /// freed. Safe to run again. It is complete only when the log is empty; while
    /// a reader holds frames it fails with `Error::ScrubIncomplete` and the caller
    /// runs it again later.
    pub async fn scrub(&self) -> Result<ScrubReport> {
        let bytes_before = self.file_bytes().await?;
        self.backend
            .execute(FTS_OPTIMIZE, &[])
            .await
            .map_err(|error| step("optimize the full-text index", error))?;
        let checkpoint = self.empty_the_log().await?;
        self.backend
            .execute(INCREMENTAL_VACUUM, &[])
            .await
            .map_err(|error| step("reclaim the free pages", error))?;
        let report = ScrubReport {
            bytes_before,
            bytes_after: self.file_bytes().await?,
            checkpoint,
        };
        tracing::info!(?report, "database scrubbed");
        Ok(report)
    }

    async fn empty_the_log(&self) -> Result<CheckpointResult> {
        let mut last = None;
        for attempt in 1..=CHECKPOINT_ATTEMPTS {
            let result = self
                .backend
                .wal_checkpoint(CheckpointMode::Truncate, CHECKPOINT_WAIT)
                .await
                .map_err(|error| step("empty the write-ahead log", error))?;
            tracing::debug!(attempt, ?result, "checkpoint attempted");
            if result.is_complete() {
                return Ok(result);
            }
            last = Some(result);
        }
        let left = last.expect("at least one attempt is made");
        Err(Error::ScrubIncomplete {
            busy: left.busy,
            log_frames: left.log_frames,
            checkpointed: left.checkpointed,
        })
    }

    async fn file_bytes(&self) -> Result<u64> {
        let pages = self.backend.query(PAGE_COUNT, &[]).await?[0].get_i64(0)?;
        let size = self.backend.query(PAGE_SIZE, &[]).await?[0].get_i64(0)?;
        Ok((pages * size).max(0) as u64)
    }
}

fn step(step: &'static str, error: Error) -> Error {
    Error::ScrubStep {
        step,
        reason: error.to_string(),
    }
}
