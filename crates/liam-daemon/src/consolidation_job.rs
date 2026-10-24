// SPDX-License-Identifier: Apache-2.0
//! The daemon's side of the consolidation job: the maintenance steps it runs
//! inside a snapshot, the reopen after a rollback, and the scheduled tick.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use liam_daemon::config::Config;
use liam_daemon::consolidation::hooks::ConsolidationHookSet;
use liam_daemon::consolidation::manifest::RunOutcome;
use liam_daemon::consolidation::run::{
    Consolidator, Maintenance, Parts, ProbeRunner, StoreProbes, StoreReopener,
};
use liam_daemon::consolidation::settings::{identity, run_settings};
use liam_daemon::consolidation::shared_store::SharedStore;
use liam_daemon::models::StoreEmbedder;
use liam_model::Embedder;
use liam_store::{DefaultGraph, RetentionPolicy, SharedLog};
use tokio_util::sync::CancellationToken;

use crate::log_startup::{open_database, open_store_on_log};
use crate::mcp::MemoryServer;
use crate::{maintenance_steps, Resynthesis};

/// Runs the tick's steps over the store the run guards, stamping every log
/// write with the run's source so a rollback cancels them.
struct DaemonMaintenance {
    server: MemoryServer,
    policy: RetentionPolicy,
    resynthesis: Resynthesis,
    /// An autotuning server is still calibrating its generation permits at the
    /// first run, which then skips resynthesis so it does not compete for them.
    skip_resynthesis_once: AtomicBool,
}

#[async_trait]
impl Maintenance for DaemonMaintenance {
    async fn run(&self, store: &DefaultGraph, source: &str) {
        let skip = self.skip_resynthesis_once.swap(false, Ordering::SeqCst);
        let resynthesis = (!skip).then_some(self.resynthesis);
        maintenance_steps(&self.server, store, &self.policy, resynthesis, Some(source)).await;
    }
}

/// Opens the database again on the log the daemon already writes through.
struct DaemonReopener {
    config: Config,
    database_path: PathBuf,
    log_dir: PathBuf,
    log: SharedLog,
    embedder: Arc<dyn Embedder>,
}

#[async_trait]
impl StoreReopener for DaemonReopener {
    async fn reopen(&self) -> anyhow::Result<DefaultGraph> {
        let store = open_database(&self.config, &self.database_path).await?;
        open_store_on_log(
            store,
            &self.database_path,
            &self.log_dir,
            self.log.clone(),
            self.embedder.clone(),
        )
        .await
    }
}

/// What the daemon needs to run the job: where the database and its log are,
/// and the embedder both use.
pub struct JobSetup<'a> {
    pub config: &'a Config,
    pub database_path: &'a Path,
    pub log_dir: &'a Path,
    pub log: SharedLog,
    pub embedder: Arc<dyn Embedder>,
    pub server: &'a MemoryServer,
}

/// The consolidation job as the daemon runs it, on a schedule and at start.
pub struct ConsolidationJob {
    consolidator: Consolidator,
    store: Arc<SharedStore>,
    stopping: CancellationToken,
}

impl ConsolidationJob {
    /// # Errors
    ///
    /// Fails when the snapshot directory cannot be resolved.
    pub fn new(setup: JobSetup<'_>) -> anyhow::Result<Self> {
        Self::with_probes(setup, Arc::new(StoreProbes))
    }

    fn with_probes(setup: JobSetup<'_>, probes: Arc<dyn ProbeRunner>) -> anyhow::Result<Self> {
        let JobSetup {
            config,
            database_path,
            log_dir,
            log,
            embedder,
            server,
        } = setup;
        let maintenance = DaemonMaintenance {
            server: server.clone(),
            policy: config.gc_policy(),
            resynthesis: Resynthesis::from_config(config),
            skip_resynthesis_once: AtomicBool::new(
                config.gc.run_on_start && config.llm.max_concurrent_generations == 0,
            ),
        };
        let parts = Parts {
            embedder: Arc::new(StoreEmbedder(embedder.clone())),
            hooks: ConsolidationHookSet::default(),
            maintenance: Arc::new(maintenance),
            reopener: Arc::new(DaemonReopener {
                config: config.clone(),
                database_path: database_path.to_path_buf(),
                log_dir: log_dir.to_path_buf(),
                log,
                embedder,
            }),
            probes,
        };
        let consolidator = Consolidator::new(
            database_path.to_path_buf(),
            run_settings(config, database_path)?,
            identity(config),
            parts,
        );
        Ok(Self {
            consolidator,
            store: server.shared_store(),
            stopping: server.shutdown_token(),
        })
    }

    pub fn stopping(&self) -> CancellationToken {
        self.stopping.clone()
    }

    /// Finishes a rollback a crash cut short, before anything serves the
    /// store. A run that crashed going forward is resumed by the first tick.
    ///
    /// # Errors
    ///
    /// Fails when the rollback cannot be finished, which leaves the manifest as
    /// the next start's resume point.
    pub async fn finish_pending_rollback(&self) -> anyhow::Result<()> {
        if let Some(run) = self
            .consolidator
            .finish_pending_rollback(&self.store)
            .await?
        {
            report(&run.run_id, run.outcome);
        }
        Ok(())
    }

    /// True when a run is waiting to be resumed, so it should not wait a full
    /// interval for the next tick.
    pub fn has_unfinished_run(&self) -> bool {
        match self.consolidator.unfinished_run() {
            Ok(run) => run.is_some(),
            Err(error) => {
                tracing::warn!(error = %format!("{error:#}"), "could not read the consolidation manifest");
                false
            }
        }
    }

    /// One scheduled run. A failure is logged and left for the next tick, which
    /// resumes the run from the step it stopped at.
    pub async fn tick(&self) -> Option<RunOutcome> {
        if self.stopping.is_cancelled() {
            return None;
        }
        match self.consolidator.consolidate(&self.store).await {
            Ok(run) => {
                report(&run.run_id, run.outcome);
                run.outcome
            }
            Err(error) => {
                tracing::error!(error = %format!("{error:#}"), "the consolidation run failed; the next tick resumes it");
                None
            }
        }
    }
}

fn report(run_id: &str, outcome: Option<RunOutcome>) {
    match outcome {
        Some(RunOutcome::Committed) => tracing::info!(run_id, "consolidation committed"),
        Some(RunOutcome::RolledBack) => {
            tracing::error!(run_id, "consolidation rolled back; the store is as it was")
        }
        Some(RunOutcome::RestoreFailed) => {
            tracing::error!(
                run_id,
                "consolidation could not restore the store; it needs manual inspection"
            )
        }
        None => tracing::warn!(run_id, "consolidation ended with no outcome"),
    }
}

#[cfg(test)]
mod tests;
