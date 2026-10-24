// SPDX-License-Identifier: Apache-2.0
//! What a run remembers about events the model could not handle. A single
//! event that fails is counted in the store and, after `max_batch_failures`
//! runs, quarantined so the cursor can pass it. The count lives in the store,
//! so it survives a crash and is reverted together with the facts on a restore.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use liam_log::LogOffset;
use liam_store::{batch_key, Millis};
use tracing::{info, warn};

use super::{is_parseable, ExtractionInput, Run, Status, PROMPT_VERSION};

const SCOPE_SEPARATOR: char = '\u{1f}';
const CANARY_EVENT_ID: &str = "canary";
/// A fixed, trivial statement: a working model answers it quickly.
pub(super) const CANARY_TEXT: &str = "Alice works at Acme.";

/// Joins a scope and an event id into one id no other pair can produce: scopes
/// hold no control characters.
pub(crate) fn tracking_id(scope: Option<&str>, event_id: &str) -> String {
    format!("{}{SCOPE_SEPARATOR}{event_id}", scope.unwrap_or_default())
}

impl Run<'_> {
    /// The key one event is tracked under in one scope. It changes with the
    /// prompt version and the model, so a fix re-drives events quarantined
    /// before it, and it includes the scope, so an event that fails in one
    /// scope is not blamed in, or cleared by, another.
    fn event_key(&self, position: usize) -> String {
        let input = &self.inputs[position];
        batch_key(
            PROMPT_VERSION,
            &self.deps.identity.model_sha256,
            &[tracking_id(input.scope.as_deref(), &input.event_id)],
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

    /// Counts the run's timed out and refused events, now that it is known
    /// whether the model works. That is proven by a live answer in this run,
    /// or else by a canary question that the model answers after the fact. A
    /// cached answer proves nothing. Without proof no event is blamed. At most
    /// half of the events the run asked about are quarantined this way, so a
    /// limit set too low cannot quarantine a backlog.
    pub(super) async fn settle_suspects(&mut self) -> Result<()> {
        if self.suspects.is_empty() || self.report.sidecar_lost {
            return Ok(());
        }
        if !self.model_answered && !self.canary_answered().await {
            info!(
                suspects = self.suspects.len(),
                "the model gave no proof it works, so no failure counts against an event"
            );
            return Ok(());
        }
        let mut room = self.attempted.len() / 2;
        for (position, reason) in std::mem::take(&mut self.suspects) {
            let key = self.event_key(position);
            let id = self.inputs[position].event_id.clone();
            let failures = self
                .deps
                .store
                .record_batch_failure(&key, std::slice::from_ref(&id), &reason)
                .await?;
            self.failing.insert(position);
            if failures >= self.deps.config.max_batch_failures && room > 0 {
                room -= 1;
                self.quarantine(position, &key).await?;
            }
        }
        Ok(())
    }

    /// Asks the model a tiny fixed question straight through the extractor,
    /// not the cache, within the time the run has left. True when it gives a
    /// readable answer.
    async fn canary_answered(&mut self) -> bool {
        let Some(opened) = self.session.as_ref() else {
            return false;
        };
        let (extractor, alive) = (Arc::clone(&opened.extractor), Arc::clone(&opened.alive));
        if self.expired() || !alive() {
            return false;
        }
        let canary = ExtractionInput {
            event_id: CANARY_EVENT_ID.into(),
            offset: LogOffset {
                segment: 0,
                index: 0,
            },
            scope: None,
            source: String::new(),
            observed_at: Millis(0),
            ingested_at: Millis(0),
            text: CANARY_TEXT.into(),
        };
        let request = self.request(&[&canary]);
        let Ok(Ok(_permit)) =
            tokio::time::timeout(self.remaining(), self.deps.permits.acquire()).await
        else {
            return false;
        };
        let limit = self
            .remaining()
            .min(Duration::from_secs(self.deps.config.request_timeout_secs));
        let answered = matches!(
            tokio::time::timeout(limit, extractor.extract_json(&request)).await,
            Ok(Ok(text)) if is_parseable(&text)
        );
        info!(
            answered,
            "canary question asked after a timeout or a refusal"
        );
        answered
    }
}
