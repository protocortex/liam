// SPDX-License-Identifier: Apache-2.0
//! The daemon's side of the consolidation job: the maintenance steps it runs
//! inside a snapshot, the reopen after a rollback, and the scheduled tick.

use std::path::Path;
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
use liam_daemon::store_open::LogReopener;
use liam_model::Embedder;
use liam_store::{DefaultGraph, RetentionPolicy, SharedLog};
use tokio_util::sync::CancellationToken;

use crate::extraction::setup::{self as extraction_setup, sweep_stale_sidecar, DaemonFiles};
use crate::mcp::MemoryServer;
use crate::shutdown::Shutdown;
use crate::{maintenance_steps, Resynthesis};

/// Runs the tick's steps over the store the run guards, stamping every log
/// write with the run's source so a rollback cancels them.
struct DaemonMaintenance {
    server: MemoryServer,
    policy: RetentionPolicy,
    resynthesis: Resynthesis,
    /// Set for the run made at start of an autotuning server, which is still
    /// calibrating its generation permits: that one run skips resynthesis so it
    /// does not compete for them.
    skip_resynthesis_once: Arc<AtomicBool>,
}

#[async_trait]
impl Maintenance for DaemonMaintenance {
    async fn run(&self, store: &DefaultGraph, source: &str) -> anyhow::Result<()> {
        let skip = self.skip_resynthesis_once.swap(false, Ordering::SeqCst);
        let resynthesis = (!skip).then_some(self.resynthesis);
        let finished =
            maintenance_steps(&self.server, store, &self.policy, resynthesis, Some(source)).await;
        // A step cut short must not be recorded as done: the run resumes at it.
        anyhow::ensure!(
            finished,
            "the daemon is stopping, so the maintenance step did not finish"
        );
        Ok(())
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
    pub shutdown: &'a Shutdown,
}

/// The hook parts a run calls: extraction when the config turns it on. A model
/// server an earlier daemon left behind is cleared first, on or off.
async fn hooks_for(setup: &JobSetup<'_>) -> anyhow::Result<ConsolidationHookSet> {
    let config = &setup.config.extraction;
    sweep_stale_sidecar(config, setup.database_path);
    if !config.enabled {
        return Ok(ConsolidationHookSet::default());
    }
    let files = DaemonFiles {
        database_path: setup.database_path,
        log_dir: setup.log_dir,
        permits: setup.server.generation_permits_handle(),
        stopping: setup.shutdown.token(),
    };
    let extraction = extraction_setup::build(config, files).await?;
    Ok(ConsolidationHookSet {
        extraction: Some(Arc::new(extraction)),
        ..Default::default()
    })
}

/// The consolidation job as the daemon runs it, on a schedule and at start.
pub struct ConsolidationJob {
    consolidator: Consolidator,
    store: Arc<SharedStore>,
    shutdown: Shutdown,
    skip_resynthesis_once: Arc<AtomicBool>,
    autotuning: bool,
}

impl ConsolidationJob {
    /// # Errors
    ///
    /// Fails when the snapshot directory cannot be resolved, or when extraction
    /// is on and its response cache or the event log cannot be opened.
    pub async fn new(setup: JobSetup<'_>) -> anyhow::Result<Self> {
        let hooks = hooks_for(&setup).await?;
        Self::with_parts(setup, Arc::new(StoreProbes), None, hooks)
    }

    /// The job over the given probe runner, hooks and, when given, reopener,
    /// which tests use to make a run regress or a reopen fail.
    fn with_parts(
        setup: JobSetup<'_>,
        probes: Arc<dyn ProbeRunner>,
        reopener: Option<Arc<dyn StoreReopener>>,
        hooks: ConsolidationHookSet,
    ) -> anyhow::Result<Self> {
        let JobSetup {
            config,
            database_path,
            log_dir,
            log,
            embedder,
            server,
            shutdown,
        } = setup;
        let skip_resynthesis_once = Arc::new(AtomicBool::new(false));
        let maintenance = DaemonMaintenance {
            server: server.clone(),
            policy: config.gc_policy(),
            resynthesis: Resynthesis::from_config(config),
            skip_resynthesis_once: skip_resynthesis_once.clone(),
        };
        let reopener = reopener.unwrap_or_else(|| {
            Arc::new(LogReopener {
                config: config.clone(),
                database_path: database_path.to_path_buf(),
                log_dir: log_dir.to_path_buf(),
                log,
                embedder: embedder.clone(),
            })
        });
        let parts = Parts {
            embedder: Arc::new(StoreEmbedder(embedder.clone())),
            hooks,
            maintenance: Arc::new(maintenance),
            reopener,
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
            shutdown: shutdown.clone(),
            skip_resynthesis_once,
            autotuning: config.llm.max_concurrent_generations == 0,
        })
    }

    pub fn stopping(&self) -> CancellationToken {
        self.shutdown.token()
    }

    /// What the daemon does before it serves: finishes a rollback a crash cut
    /// short, then says whether a run should start now, which it should when
    /// the config asks for one at start or a crashed run is waiting to resume.
    /// That first run leaves resynthesis out when the server is still
    /// calibrating its generation permits.
    ///
    /// # Errors
    ///
    /// Fails when the rollback cannot be finished, which leaves the manifest as
    /// the next start's resume point.
    pub async fn start(&self, run_on_start: bool) -> anyhow::Result<bool> {
        self.finish_pending_rollback().await?;
        let run_now = run_on_start || self.has_unfinished_run();
        if run_now && self.autotuning {
            self.skip_resynthesis_once.store(true, Ordering::SeqCst);
        }
        Ok(run_now)
    }

    async fn finish_pending_rollback(&self) -> anyhow::Result<()> {
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
    fn has_unfinished_run(&self) -> bool {
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
        if self.shutdown.token().is_cancelled() {
            return None;
        }
        match self.consolidator.consolidate(&self.store).await {
            Ok(run) => {
                report(&run.run_id, run.outcome);
                run.outcome
            }
            Err(error) => {
                tracing::error!(error = %format!("{error:#}"), "the consolidation run did not finish");
                // A rollback that could not reopen the store leaves nothing to serve.
                if self.store.get().await.is_err() {
                    self.shutdown.store_lost();
                }
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
