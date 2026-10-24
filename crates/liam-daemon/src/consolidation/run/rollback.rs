// SPDX-License-Identifier: Apache-2.0
//! The rollback of a run: check the snapshot, commit to the rollback, cancel the
//! run's writes in the log, put the snapshot back and replay the log forward, so
//! concurrent client writes survive.
//!
//! Three manifest steps order it. The snapshot is checked before anything is
//! recorded, because once the marker is in the log a replay skips the run's
//! writes, and a snapshot that then cannot be restored would leave the log and
//! the live store disagreeing. `rollback:pending` is the commitment: a run that
//! has it never resumes forward. `rollback:marked` follows the marker.

use std::sync::Arc;

use anyhow::Context;
use liam_store::verify_snapshot;

use super::{Consolidator, RunManifest, RunOutcome, SharedStore};

/// Recorded before anything else of the rollback, so a run that dies from here
/// on finishes the rollback instead of resuming forward.
pub(super) const ROLLBACK_PENDING: &str = "rollback:pending";
/// Recorded once the marker that cancels the run's writes is in the log.
pub(super) const ROLLBACK_MARKED: &str = "rollback:marked";
/// A crash point between the marker's append and the step that records it.
pub(super) const MARKER_APPENDED: &str = "rollback:appended";
/// A crash point after the restore and replay, before the outcome is saved.
pub(super) const ROLLBACK_RESTORED: &str = "rollback:restored";

impl Consolidator {
    pub(super) async fn roll_back(
        &self,
        shared: &SharedStore,
        mut manifest: RunManifest,
        reason: &str,
    ) -> anyhow::Result<RunManifest> {
        tracing::error!(run_id = %manifest.run_id, reason, "consolidation: rolling the run back");
        let Some(snapshot) = manifest.snapshot_meta_path.clone() else {
            tracing::info!(run_id = %manifest.run_id, "consolidation: the run never reached its snapshot, so it changed nothing");
            return self.end(manifest, RunOutcome::RolledBack);
        };
        if !manifest.is_complete(ROLLBACK_MARKED) {
            if let Err(error) = verify_snapshot(&snapshot) {
                tracing::error!(run_id = %manifest.run_id, %error, "consolidation: the snapshot is unusable, so the run cannot be rolled back; the marker was not appended and the store is unchanged, needing manual inspection");
                return self.end(manifest, RunOutcome::RestoreFailed);
            }
            if !manifest.is_complete(ROLLBACK_PENDING) {
                // Once pending is recorded the run never goes forward, so a
                // restore that could not happen must be found out first.
                shared.ensure_restorable().await?;
                self.finish_step(&mut manifest, ROLLBACK_PENDING)?;
            }
            shared
                .get()
                .await?
                .mark_run_rolled_back(&manifest.run_id)
                .await
                .context("could not append the rollback marker to the log")?;
            self.crash_point(MARKER_APPENDED)?;
            self.finish_step(&mut manifest, ROLLBACK_MARKED)?;
        }
        let outcome = self.restore(shared, &manifest, &snapshot).await?;
        self.crash_point(ROLLBACK_RESTORED)?;
        self.end(manifest, outcome)
    }

    /// Swaps the restored store into the shared slot, waiting for the requests
    /// in flight and holding new ones back until the replay is done.
    async fn restore(
        &self,
        shared: &SharedStore,
        manifest: &RunManifest,
        snapshot: &std::path::Path,
    ) -> anyhow::Result<RunOutcome> {
        let mut slot = shared.exclusive().await;
        let held = slot.take().context("the store is not in its slot")?;
        let store = match Arc::try_unwrap(held) {
            Ok(store) => store,
            Err(held) => {
                *slot = Some(held);
                anyhow::bail!(
                    "the store is still held outside its slot, so it cannot be closed for the restore"
                );
            }
        };
        let restored = store.restore_snapshot(&self.database_path, snapshot).await;
        let reopened = self.parts.reopener.reopen().await;
        let store = match reopened {
            Ok(store) => store,
            Err(error) => {
                let mut failed = manifest.clone();
                failed.outcome = Some(RunOutcome::RestoreFailed);
                failed.save(&self.manifest_path())?;
                return Err(error.context("could not reopen the database after the rollback"));
            }
        };
        let outcome = match restored {
            Err(error) => {
                tracing::error!(run_id = %manifest.run_id, %error, "consolidation: the snapshot could not be restored, so the database may be in a mixed state and needs manual inspection");
                RunOutcome::RestoreFailed
            }
            Ok(_) => match store.catch_up().await {
                Ok(report) => {
                    tracing::info!(run_id = %manifest.run_id, ?report, "consolidation: rolled back and caught up");
                    RunOutcome::RolledBack
                }
                Err(error) => {
                    tracing::error!(run_id = %manifest.run_id, %error, "consolidation: the snapshot was restored but the replay failed, so the database is behind the log and needs manual inspection");
                    RunOutcome::RestoreFailed
                }
            },
        };
        *slot = Some(Arc::new(store));
        Ok(outcome)
    }

    /// Records how the rollback ended. A snapshot that was restored or never
    /// needed goes; one that could not be restored stays for inspection.
    fn end(&self, mut manifest: RunManifest, outcome: RunOutcome) -> anyhow::Result<RunManifest> {
        manifest.outcome = Some(outcome);
        manifest.save(&self.manifest_path())?;
        if outcome == RunOutcome::RolledBack {
            self.discard_leftovers(&manifest);
        }
        Ok(manifest)
    }
}
