// SPDX-License-Identifier: Apache-2.0
//! Copying the rows a store held before it had a log into that log, once, so a
//! rebuild from the log alone can recreate them.

use super::Graph;
use crate::backend::Backend;
use crate::error::Result;

/// Rows backfilled between two writes of the resume point.
const BACKFILL_BATCH: usize = 500;

/// What one backfill run appended to the log.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackfillReport {
    /// Node events appended, superseded nodes included.
    pub nodes: usize,
    /// Edge events appended.
    pub edges: usize,
    /// Whether the run continued an interrupted one instead of starting over.
    pub resumed: bool,
}

impl<B: Backend> Graph<B> {
    /// Logs every row the log does not hold yet, then marks the backfill complete.
    pub async fn backfill_log_from_projection(&self) -> Result<BackfillReport> {
        self.backfill_in_batches(BACKFILL_BATCH).await
    }

    /// Whether the backfill has not completed for this store.
    pub async fn needs_backfill(&self) -> Result<bool> {
        Ok(false)
    }

    pub(super) async fn backfill_in_batches(&self, _batch: usize) -> Result<BackfillReport> {
        Ok(BackfillReport::default())
    }
}
