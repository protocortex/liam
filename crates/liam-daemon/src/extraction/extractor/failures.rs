// SPDX-License-Identifier: Apache-2.0
//! What a run remembers about events the model could not handle. A single
//! event that fails is counted in the store and, after `max_batch_failures`
//! runs, quarantined so the cursor can pass it. The count lives in the store,
//! so it survives a crash and is reverted together with the facts on a restore.

use anyhow::Result;
use liam_store::batch_key;
use tracing::{info, warn};

use super::{Run, Status, PROMPT_VERSION};

impl Run<'_> {
    /// The key one event is tracked under. It changes with the prompt version
    /// and the model, so a fix re-drives events quarantined before it.
    fn event_key(&self, position: usize) -> String {
        let id = self.inputs[position].event_id.clone();
        batch_key(
            PROMPT_VERSION,
            &self.deps.identity.model_sha256,
            std::slice::from_ref(&id),
        )
    }

    /// Marks events quarantined by an earlier run, so their batches are
    /// skipped and the cursor passes them, and remembers which others have
    /// failures to clear.
    pub(super) async fn skip_quarantined(&mut self) -> Result<()> {
        for position in 0..self.inputs.len() {
            let key = self.event_key(position);
            if self.deps.store.batch_failures(&key).await? == 0 {
                continue;
            }
            if self.deps.store.is_quarantined(&key).await? {
                self.status[position] = Status::Quarantined;
                self.report.batches_quarantined += 1;
            } else {
                self.failing.insert(position);
            }
        }
        Ok(())
    }

    /// An event that succeeded is no longer failing.
    pub(super) async fn clear_failures(&mut self, piece: &[usize]) -> Result<()> {
        for &position in piece {
            if self.failing.remove(&position) {
                self.deps
                    .store
                    .clear_batch_failures(&self.event_key(position))
                    .await?;
            }
        }
        Ok(())
    }

    /// Counts one failure for an event the model cannot handle on its own.
    /// True while it is still failing, false once it is quarantined.
    pub(super) async fn fail_single(&mut self, position: usize, reason: &str) -> Result<bool> {
        let key = self.event_key(position);
        let id = self.inputs[position].event_id.clone();
        let failures = self
            .deps
            .store
            .record_batch_failure(&key, std::slice::from_ref(&id), reason)
            .await?;
        self.failing.insert(position);
        if failures >= self.deps.config.max_batch_failures {
            self.quarantine(position, &key).await?;
            return Ok(false);
        }
        self.status[position] = Status::Failed;
        warn!(event_id = %id, failures, "an event the model cannot handle");
        Ok(true)
    }

    async fn quarantine(&mut self, position: usize, key: &str) -> Result<()> {
        self.deps.store.quarantine(key).await?;
        self.status[position] = Status::Quarantined;
        self.report.batches_quarantined += 1;
        warn!(event_id = %self.inputs[position].event_id, "event quarantined");
        Ok(())
    }

    /// Counts the run's timed out events, now that it is known whether the
    /// model answered anything. A run where it answered nothing proves
    /// nothing about the events, so no timeout is counted. At most half of the
    /// events the run asked about are quarantined for a timeout, so a limit set
    /// too low cannot quarantine a backlog.
    pub(super) async fn settle_timeouts(&mut self) -> Result<()> {
        if self.timeouts.is_empty() {
            return Ok(());
        }
        if !self.model_answered {
            info!(
                timed_out = self.timeouts.len(),
                "no call was answered, so the timeouts count against no event"
            );
            return Ok(());
        }
        let mut room = self.attempted.len() / 2;
        for position in std::mem::take(&mut self.timeouts) {
            let key = self.event_key(position);
            let id = self.inputs[position].event_id.clone();
            let failures = self
                .deps
                .store
                .record_batch_failure(&key, std::slice::from_ref(&id), "the model call timed out")
                .await?;
            self.failing.insert(position);
            if failures >= self.deps.config.max_batch_failures && room > 0 {
                room -= 1;
                self.quarantine(position, &key).await?;
            }
        }
        Ok(())
    }
}
