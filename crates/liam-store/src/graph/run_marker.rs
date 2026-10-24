// SPDX-License-Identifier: Apache-2.0
//! The log record that cancels one consolidation run's writes.

use liam_log::event::LogPayload;

use super::logged_plan::{event, STORE_SOURCE, STORE_TRUST};
use super::logged_write::HeldLog;
use super::Graph;
use crate::backend::Backend;
use crate::error::{Error, Result};

/// The hash of a record that covers no content.
const NO_CONTENT_HASH: [u8; 32] = [0; 32];

impl<B: Backend> Graph<B> {
    /// Appends a `RunRolledBack` record for `run_id`. A replay or rebuild then
    /// skips every write whose source is `consolidation:<run_id>`, so a
    /// rollback that restores a snapshot and replays forward does not bring the
    /// run's own writes back. The cursor is left for `catch_up` to move over
    /// the record, since the marker carries no rows.
    ///
    /// # Errors
    ///
    /// Fails without an event log, on a poisoned log, and when the append fails.
    pub async fn mark_run_rolled_back(&self, run_id: &str) -> Result<()> {
        let log = self.log.as_ref().ok_or(Error::NoEventLog)?;
        let mut held = HeldLog::acquire_for_write(log).await?;
        let now = self.clock.now();
        let marker = event(
            NO_CONTENT_HASH,
            (STORE_SOURCE, STORE_TRUST),
            now.0,
            now,
            LogPayload::RunRolledBack {
                run_id: run_id.to_owned(),
            },
        );
        held.append(marker).await?;
        held.reconcile(&[]);
        tracing::info!(run_id, "consolidation run marked rolled back in the log");
        Ok(())
    }
}
