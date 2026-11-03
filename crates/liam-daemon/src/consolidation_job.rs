// SPDX-License-Identifier: Apache-2.0
//! The daemon's side of the consolidation job: the maintenance steps it runs
//! inside a snapshot, the reopen after a rollback, and the scheduled tick.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use liam_daemon::config::Config;
use liam_daemon::consolidation::hooks::{
    ConsolidationHookSet, ExtractionHook, SummaryHook, TierHook,
};
use liam_daemon::consolidation::manifest::{RunManifest, RunOutcome};
use liam_daemon::consolidation::run::{
    Consolidator, Maintenance, Parts, ProbeRunner, StoreProbes, StoreReopener,
};
use liam_daemon::consolidation::settings::{identity, run_settings};
use liam_daemon::consolidation::shared_store::SharedStore;
use liam_daemon::models::StoreEmbedder;
use liam_daemon::store_open::LogReopener;
use liam_model::Embedder;
use liam_store::{DefaultGraph, RetentionPolicy, SharedLog, SystemClock};
use tokio_util::sync::CancellationToken;

use crate::extraction::client::ModelIdentity;
use liam_log::crypto::KeyRegistry;

use crate::extraction::setup::{self as extraction_setup, sweep_stale_sidecar, DaemonFiles};
use crate::extraction::sidecar::SystemRam;
use crate::mcp::MemoryServer;
use crate::shutdown::Shutdown;
use crate::summary::file_cache::{summary_cache_path, FileSummaryCache};
use crate::summary::hooks::SummaryHooks;
use crate::tiering::hooks::TierHooks;
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
    /// The key registry of a sealed log, so the extraction pass reads it open.
    pub keys: Option<Arc<KeyRegistry>>,
    pub embedder: Arc<dyn Embedder>,
    pub server: &'a MemoryServer,
    pub shutdown: &'a Shutdown,
}

/// The hook parts a run calls: extraction, summaries and tiering, each when the config
/// turns it on. A model server an earlier daemon left behind is cleared first,
/// whether extraction is on or off, when the server's path is configured.
fn hooks_for(setup: &JobSetup<'_>) -> anyhow::Result<ConsolidationHookSet> {
    sweep_stale_sidecar(&setup.config.extraction, setup.database_path);
    Ok(ConsolidationHookSet {
        extraction: extraction_hook(setup)?,
        summary: summary_hook(setup)?,
        tiering: tiering_hook(setup),
    })
}

fn tiering_hook(setup: &JobSetup<'_>) -> Option<Arc<dyn TierHook>> {
    let config = &setup.config.tiering;
    config.enabled.then(|| {
        Arc::new(TierHooks {
            config: config.clone(),
            extraction: setup.config.extraction.clone(),
            embedder: setup.embedder.clone(),
            permits: setup.server.generation_permits_handle(),
            free_ram: Arc::new(SystemRam),
            clock: Arc::new(SystemClock),
        }) as Arc<dyn TierHook>
    })
}

fn extraction_hook(setup: &JobSetup<'_>) -> anyhow::Result<Option<Arc<dyn ExtractionHook>>> {
    let config = &setup.config.extraction;
    if !config.enabled {
        return Ok(None);
    }
    let files = DaemonFiles {
        database_path: setup.database_path,
        log_dir: setup.log_dir,
        keys: setup.keys.clone(),
        permits: setup.server.generation_permits_handle(),
        stopping: setup.shutdown.token(),
    };
    Ok(Some(Arc::new(extraction_setup::build(config, files)?)))
}

fn summary_hook(setup: &JobSetup<'_>) -> anyhow::Result<Option<Arc<dyn SummaryHook>>> {
    let config = &setup.config.summary;
    if !config.enabled {
        return Ok(None);
    }
    let llm = &setup.config.llm;
    // The in-process model is named by its repository and file, and not hashed.
    let identity = ModelIdentity {
        model_id: format!("{}:{}/{}", llm.provider, llm.model, llm.gguf_file),
        model_sha256: String::new(),
    };
    let cache = FileSummaryCache::open(summary_cache_path(setup.database_path), identity.clone())
        .context("could not open the summary cache")?;
    Ok(Some(Arc::new(SummaryHooks {
        config: config.clone(),
        llm: setup.server.llm_handle(),
        permits: setup.server.generation_permits_handle(),
        cache: Arc::new(cache),
        embedder: Arc::new(StoreEmbedder(setup.embedder.clone())),
        identity,
        context_tokens: llm.context_tokens,
    })))
}

/// Why a run asked for by hand did not produce a manifest.
#[derive(Debug)]
pub enum RunNowError {
    /// Another run holds the in-process lock.
    Busy,
    Failed(anyhow::Error),
}

/// The consolidation job as the daemon runs it, on a schedule and at start.
pub struct ConsolidationJob {
    /// Held for the length of a run, so a scheduled tick and a run asked for by
    /// hand never overlap.
    running: tokio::sync::Mutex<()>,
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
    pub fn new(setup: JobSetup<'_>) -> anyhow::Result<Self> {
        let hooks = hooks_for(&setup)?;
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
            keys: _,
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
            running: tokio::sync::Mutex::new(()),
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
    /// resumes the run from the step it stopped at. A tick that meets a run in
    /// progress is skipped.
    pub async fn tick(&self) -> Option<RunOutcome> {
        if self.shutdown.token().is_cancelled() {
            return None;
        }
        let Ok(_running) = self.running.try_lock() else {
            tracing::info!("a consolidation run is in progress, so this tick is skipped");
            return None;
        };
        self.consolidate().await.ok().and_then(|run| run.outcome)
    }

    /// One run now, for the `consolidate` tool.
    ///
    /// # Errors
    ///
    /// `Busy` when a run is in progress, and `Failed` when the run did not
    /// finish; the next call resumes it.
    pub async fn run_now(&self) -> Result<RunManifest, RunNowError> {
        let _running = self.running.try_lock().map_err(|_| RunNowError::Busy)?;
        self.consolidate().await.map_err(RunNowError::Failed)
    }

    async fn consolidate(&self) -> anyhow::Result<RunManifest> {
        let run = self.consolidator.consolidate(&self.store).await;
        match &run {
            Ok(run) => report(&run.run_id, run.outcome),
            Err(error) => {
                tracing::error!(error = %format!("{error:#}"), "the consolidation run did not finish");
                // A rollback that could not reopen the store leaves nothing to serve.
                if self.store.get().await.is_err() {
                    self.shutdown.store_lost();
                }
            }
        }
        run
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
