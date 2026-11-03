// SPDX-License-Identifier: Apache-2.0
//! The `consolidate` tool: one consolidation run now, instead of waiting for
//! the schedule, reported as the run's manifest.

use std::fmt::Write;

use liam_daemon::consolidation::manifest::{RunManifest, RunOutcome};

use super::MemoryServer;
use crate::consolidation_job::RunNowError;

const NOT_AVAILABLE: &str = "this server has no consolidation job";
const BUSY: &str = "a consolidation run is already in progress";

pub(super) async fn run(server: &MemoryServer) -> String {
    if let Err(refusal) = server.require_admin("consolidate") {
        return refusal.to_string();
    }
    let Some(job) = &server.consolidator else {
        return format!("consolidate failed: {NOT_AVAILABLE}");
    };
    match job.run_now().await {
        Ok(manifest) => render(&manifest),
        Err(RunNowError::Busy) => format!("consolidate refused: {BUSY}"),
        Err(RunNowError::Failed(error)) => format!("consolidate failed: {error:#}"),
    }
}

fn render(manifest: &RunManifest) -> String {
    let outcome = match manifest.outcome {
        Some(RunOutcome::Committed) => "committed",
        Some(RunOutcome::RolledBack) => "rolled back; the store is as it was",
        Some(RunOutcome::RestoreFailed) => "could not restore the store; it needs inspection",
        None => "ended with no outcome",
    };
    let mut text = format!("consolidation run {}: {outcome}", manifest.run_id);
    let _ = write!(
        text,
        "\nsteps completed: {}",
        manifest.completed_steps.join(", ")
    );
    text
}
