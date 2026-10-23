// SPDX-License-Identifier: Apache-2.0
//! One consolidation run: snapshot, score the probes, merge near-duplicates,
//! maintain, call the hooks, score the probes again, then commit or roll back.
//!
//! Every step is recorded in the manifest as it finishes, so a run that dies
//! resumes at its first unfinished step. A run that regressed the probes, or
//! failed a step it cannot do without, rolls back: a marker in the log cancels
//! the run's own writes, the snapshot is restored, and `catch_up` replays
//! everything else that was logged since, so concurrent client writes survive.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use liam_store::{ContentEmbedder, DefaultGraph, Millis, FOREVER};

use super::dedup::merge_near_duplicates;
use super::determinism::rng_for;
use super::hooks::{ConsolidationHookSet, ExtractCtx, SummaryCtx, TierCtx};
use super::manifest::{manifest_path_for, RunManifest, RunOutcome};
use super::probe_set::{pick_probes, probe_set_path_for, ProbeSet};
use super::probes::{compare, run_probes, Probe, ProbeError, ProbeReport};
use crate::embedder_fingerprint::EmbedderFingerprint;

const SNAPSHOT: &str = "snapshot";
const BASELINE: &str = "baseline";
const DEDUP: &str = "dedup";
const MAINTENANCE: &str = "maintenance";
const EXTRACT: &str = "extract";
const SUMMARIZE: &str = "summarize";
const TIER: &str = "tier";
const PROBES: &str = "probes";
/// Suffix of the step name recorded when a hook fails; the step still counts
/// as finished, since a hook failure never fails the run.
const FAILED_SUFFIX: &str = ":failed";

/// The steps of a forward run, in order.
const STEPS: [&str; 8] = [
    SNAPSHOT,
    BASELINE,
    DEDUP,
    MAINTENANCE,
    EXTRACT,
    SUMMARIZE,
    TIER,
    PROBES,
];

/// Just before the end of time, so every live row is visible.
const ALL_LIVE: Millis = Millis(FOREVER.0 - 1);

/// What a run is tuned by.
#[derive(Debug, Clone)]
pub struct RunSettings {
    /// A probe whose recall drops by at least this much fails the run.
    pub probe_recall_margin: f64,
    /// Cosine similarity at or above which two nodes count as near-duplicates.
    pub dedup_similarity_threshold: f32,
    pub snapshot_dir: PathBuf,
    /// Most probes drawn from the store.
    pub probe_count: usize,
}

/// The models a run records, so a resume under other models is refused.
#[derive(Debug, Clone)]
pub struct Identity {
    pub embedder: EmbedderFingerprint,
    pub llm_model: String,
}

/// The existing maintenance steps (sweep, provenance repair, cluster refresh,
/// resynthesis), run as one step of the job.
#[async_trait]
pub trait Maintenance: Send + Sync {
    async fn run(&self, store: &DefaultGraph);
}

/// Opens the database again, on the same log and embedder, after a restore
/// replaced its file. The caller replays the log.
#[async_trait]
pub trait StoreReopener: Send + Sync {
    async fn reopen(&self) -> anyhow::Result<DefaultGraph>;
}

/// Scores probes against a store.
#[async_trait]
pub trait ProbeRunner: Send + Sync {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError>;
}

/// Scores probes with the store's own retrieval.
pub struct StoreProbes;

#[async_trait]
impl ProbeRunner for StoreProbes {
    async fn run(&self, store: &DefaultGraph, probes: &[Probe]) -> Result<ProbeReport, ProbeError> {
        run_probes(store, probes).await
    }
}

/// What a run calls.
pub struct Parts {
    pub embedder: Arc<dyn ContentEmbedder>,
    pub hooks: ConsolidationHookSet,
    pub maintenance: Arc<dyn Maintenance>,
    pub reopener: Arc<dyn StoreReopener>,
    pub probes: Arc<dyn ProbeRunner>,
}

/// A finished run: the manifest says how it ended, and the store is the one to
/// serve from now on (reopened when the run rolled back).
pub struct Finished {
    pub store: DefaultGraph,
    pub manifest: RunManifest,
}

/// Why a run stopped short of a verdict on the forward path.
enum StepOutcome {
    Done,
    /// The run cannot be committed; the reason is logged with the rollback.
    Failed(String),
}

enum Verdict {
    Commit,
    Rollback(String),
}

pub struct Consolidator {
    database_path: PathBuf,
    settings: RunSettings,
    identity: Identity,
    parts: Parts,
    seed: Option<u64>,
    #[cfg(test)]
    crash_after: Option<&'static str>,
}

impl Consolidator {
    pub fn new(
        database_path: PathBuf,
        settings: RunSettings,
        identity: Identity,
        parts: Parts,
    ) -> Self {
        Self {
            database_path,
            settings,
            identity,
            parts,
            seed: None,
            #[cfg(test)]
            crash_after: None,
        }
    }

    /// Fixes the seed of a new run, so two runs choose alike.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Runs the job over `store`, resuming an unfinished run that the manifest
    /// beside the database records. The store is consumed because a rollback
    /// must close it before the snapshot replaces its file; `Finished` hands
    /// back the store to use afterwards. An `Err` leaves the manifest as the
    /// next call's resume point and drops the store.
    ///
    /// # Errors
    ///
    /// Fails when an unfinished run was made under other models, when the
    /// manifest or snapshot cannot be written, and when a rollback cannot
    /// reopen the store. A rollback whose restore fails is not an `Err`: it
    /// ends as `RestoreFailed`.
    pub async fn consolidate(&self, store: DefaultGraph) -> anyhow::Result<Finished> {
        let mut manifest = self.begin()?;
        match self.forward(&store, &mut manifest).await? {
            Verdict::Commit => self.commit(store, manifest),
            Verdict::Rollback(reason) => {
                anyhow::bail!("the run cannot be committed and cannot yet be rolled back: {reason}")
            }
        }
    }

    fn manifest_path(&self) -> PathBuf {
        manifest_path_for(&self.database_path)
    }

    fn probe_set_path(&self) -> PathBuf {
        probe_set_path_for(&self.database_path)
    }

    /// The unfinished run on disk, or a new one.
    fn begin(&self) -> anyhow::Result<RunManifest> {
        if let Some(unfinished) =
            RunManifest::load(&self.manifest_path())?.filter(|run| run.outcome.is_none())
        {
            unfinished.check_resumable(&self.identity.embedder, &self.identity.llm_model)?;
            tracing::info!(run_id = %unfinished.run_id, done = ?unfinished.completed_steps, "consolidation: resuming an unfinished run");
            return Ok(unfinished);
        }
        let id = ulid::Ulid::gen();
        let manifest = RunManifest::new(
            id.to_string(),
            self.seed.unwrap_or(id.random() as u64),
            self.identity.embedder.clone(),
            self.identity.llm_model.clone(),
            Millis::now(),
        );
        manifest.save(&self.manifest_path())?;
        tracing::info!(run_id = %manifest.run_id, seed = manifest.seed, "consolidation: run started");
        Ok(manifest)
    }

    async fn forward(
        &self,
        store: &DefaultGraph,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<Verdict> {
        for step in STEPS.iter().skip(manifest.resume_point(&STEPS)) {
            tracing::info!(run_id = %manifest.run_id, step, "consolidation: step starting");
            if let StepOutcome::Failed(reason) = self.run_step(step, store, manifest).await? {
                return Ok(Verdict::Rollback(reason));
            }
            self.finish_step(manifest, step)?;
        }
        Ok(Verdict::Commit)
    }

    async fn run_step(
        &self,
        step: &str,
        store: &DefaultGraph,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<StepOutcome> {
        match step {
            SNAPSHOT => self.take_snapshot(store, manifest).await?,
            BASELINE => self.record_baseline(store, manifest).await?,
            DEDUP => return Ok(self.merge_duplicates(store, manifest).await),
            MAINTENANCE => self.parts.maintenance.run(store).await,
            EXTRACT => {
                let result = self.parts.hooks.extract(&ExtractCtx).await;
                self.note_hook(step, result.map(drop), manifest)?;
            }
            SUMMARIZE => {
                let result = self.parts.hooks.summarize(&SummaryCtx).await;
                self.note_hook(step, result.map(drop), manifest)?;
            }
            TIER => {
                let result = self.parts.hooks.tier(&TierCtx).await;
                self.note_hook(step, result.map(drop), manifest)?;
            }
            PROBES => return self.check_probes(store).await,
            other => anyhow::bail!("the consolidation run has no step named {other}"),
        }
        Ok(StepOutcome::Done)
    }

    async fn take_snapshot(
        &self,
        store: &DefaultGraph,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<()> {
        let snapshot = store
            .snapshot(&self.settings.snapshot_dir)
            .await
            .context("could not snapshot the database before the run")?;
        manifest.snapshot_meta_path = Some(snapshot.meta_path);
        Ok(())
    }

    async fn record_baseline(
        &self,
        store: &DefaultGraph,
        manifest: &RunManifest,
    ) -> anyhow::Result<()> {
        let mut rng = rng_for(manifest);
        let probes = pick_probes(store, &mut rng, ALL_LIVE, self.settings.probe_count).await?;
        let baseline = self.parts.probes.run(store, &probes).await?;
        ProbeSet { probes, baseline }.save(&self.probe_set_path())
    }

    async fn merge_duplicates(&self, store: &DefaultGraph, manifest: &RunManifest) -> StepOutcome {
        let merged = merge_near_duplicates(
            store,
            self.parts.embedder.as_ref(),
            None,
            ALL_LIVE,
            self.settings.dedup_similarity_threshold,
            &manifest.write_source(),
        )
        .await;
        match merged {
            Ok(report) => {
                tracing::info!(?report, "consolidation: near-duplicates merged");
                StepOutcome::Done
            }
            Err(error) => StepOutcome::Failed(format!("the near-duplicate merge failed: {error}")),
        }
    }

    fn note_hook(
        &self,
        step: &str,
        result: anyhow::Result<()>,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<()> {
        if let Err(error) = result {
            tracing::warn!(step, %error, "consolidation: a hook failed, the run goes on");
            manifest
                .mark_step_complete(&format!("{step}{FAILED_SUFFIX}"), &self.manifest_path())?;
        }
        Ok(())
    }

    async fn check_probes(&self, store: &DefaultGraph) -> anyhow::Result<StepOutcome> {
        let set = ProbeSet::load(&self.probe_set_path())?;
        if set.probes.is_empty() {
            tracing::info!("consolidation: the store had nothing to probe, the gate is skipped");
            return Ok(StepOutcome::Done);
        }
        let current = match self.parts.probes.run(store, &set.probes).await {
            Ok(report) => report,
            Err(error) => return Ok(StepOutcome::Failed(format!("the probes failed: {error}"))),
        };
        if compare(&set.baseline, &current, self.settings.probe_recall_margin) {
            return Ok(StepOutcome::Done);
        }
        Ok(StepOutcome::Failed(format!(
            "a probe's recall dropped by at least {}",
            self.settings.probe_recall_margin
        )))
    }

    fn finish_step(&self, manifest: &mut RunManifest, step: &str) -> anyhow::Result<()> {
        manifest.mark_step_complete(step, &self.manifest_path())?;
        self.crash_point(step)
    }

    fn commit(&self, store: DefaultGraph, mut manifest: RunManifest) -> anyhow::Result<Finished> {
        manifest.outcome = Some(RunOutcome::Committed);
        manifest.save(&self.manifest_path())?;
        self.discard_leftovers(&manifest);
        tracing::info!(run_id = %manifest.run_id, "consolidation: run committed");
        Ok(Finished { store, manifest })
    }

    /// Removes the snapshot and probe set once nothing can need them. A
    /// failure only leaves files behind, so it is logged and not raised.
    fn discard_leftovers(&self, manifest: &RunManifest) {
        let mut files = vec![self.probe_set_path()];
        if let Some(meta) = &manifest.snapshot_meta_path {
            files.push(meta.with_extension("db"));
            files.push(meta.clone());
        }
        for file in files {
            remove_if_present(&file);
        }
    }

    #[cfg(test)]
    fn crash_point(&self, point: &str) -> anyhow::Result<()> {
        if self.crash_after == Some(point) {
            anyhow::bail!("simulated crash after {point}");
        }
        Ok(())
    }

    #[cfg(not(test))]
    fn crash_point(&self, _point: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

fn remove_if_present(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "consolidation: could not remove a leftover file")
        }
    }
}

#[cfg(test)]
mod tests;
