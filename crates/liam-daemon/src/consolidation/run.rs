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
use super::dedup_cursor;
use super::determinism::rng_for;
use super::hooks::{ConsolidationHookSet, ExtractCtx, SummaryCtx, TierCtx};
use super::manifest::{manifest_path_for, ResumeRefused, RunManifest, RunOutcome};
use super::probe_set::{pick_probes, probe_set_path_for, ProbeSet};
use super::probes::{compare, run_probes, Probe, ProbeError, ProbeReport};
use super::shared_store::SharedStore;
use crate::embedder_fingerprint::EmbedderFingerprint;

const SNAPSHOT: &str = "snapshot";
const BASELINE: &str = "baseline";
const DEDUP: &str = "dedup";
const MAINTENANCE: &str = "maintenance";
const EXTRACT: &str = "extract";
const SUMMARIZE: &str = "summarize";
const TIER: &str = "tier";
const PROBES: &str = "probes";
/// Suffix of the step name recorded when a hook part fails, or ran only in
/// part. The step still counts as finished, since a hook never fails the run.
const PARTIAL_SUFFIX: &str = ":partial";
/// Suffix of the step name recorded when no hook part is configured for it.
const SKIPPED_SUFFIX: &str = ":skipped";

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
/// resynthesis), run as one step of the job. Every log write they make must be
/// stamped with `source`, so a rollback of the run cancels them.
///
/// An `Err` means the step did not finish, as when the daemon is stopping. The
/// run stops there without a verdict and resumes at this step.
#[async_trait]
pub trait Maintenance: Send + Sync {
    async fn run(&self, store: &DefaultGraph, source: &str) -> anyhow::Result<()>;
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

    /// Runs the job over the shared store, resuming an unfinished run that the
    /// manifest beside the database records. The run takes a read guard per
    /// step, so clients keep working; a rollback takes the write guard to swap
    /// the restored store in. The returned manifest says how the run ended.
    ///
    /// An unfinished run made under other models cannot be resumed, so it is
    /// rolled back and returned; the next call starts a fresh run.
    ///
    /// # Errors
    ///
    /// Fails when `probe_count` is zero, when the manifest or snapshot cannot
    /// be written, and when a rollback cannot reopen the store. A rollback whose
    /// restore fails is not an `Err`: it ends as `RestoreFailed`. An `Err`
    /// leaves the manifest as the next call's resume point.
    pub async fn consolidate(&self, shared: &SharedStore) -> anyhow::Result<RunManifest> {
        anyhow::ensure!(
            self.settings.probe_count > 0,
            "probe_count is zero, so the run would have no probes to gate it"
        );
        let (mut manifest, refused) = self.begin().await?;
        let verdict = if let Some(refused) = refused {
            Verdict::Rollback(refused.to_string())
        } else if manifest.is_complete(rollback::ROLLBACK_PENDING) {
            Verdict::Rollback("finishing an interrupted rollback".to_owned())
        } else {
            self.forward(shared, &mut manifest).await?
        };
        match verdict {
            Verdict::Commit => self.commit(manifest),
            Verdict::Rollback(reason) => self.roll_back(shared, manifest, &reason).await,
        }
    }

    /// Finishes a rollback that a crash interrupted, so the store is right
    /// before anything serves it. A run that crashed on its forward path is
    /// left for the next call, which resumes it.
    ///
    /// # Errors
    ///
    /// Fails like `consolidate` does.
    pub async fn finish_pending_rollback(
        &self,
        shared: &SharedStore,
    ) -> anyhow::Result<Option<RunManifest>> {
        let pending = self
            .unfinished_run()?
            .is_some_and(|run| run.is_complete(rollback::ROLLBACK_PENDING));
        if !pending {
            return Ok(None);
        }
        self.consolidate(shared).await.map(Some)
    }

    /// The run the manifest beside the database records as not yet ended.
    ///
    /// # Errors
    ///
    /// Fails when the manifest exists and cannot be read.
    pub fn unfinished_run(&self) -> anyhow::Result<Option<RunManifest>> {
        Ok(RunManifest::load(&self.manifest_path())?.filter(|run| run.outcome.is_none()))
    }

    fn manifest_path(&self) -> PathBuf {
        manifest_path_for(&self.database_path)
    }

    fn probe_set_path(&self) -> PathBuf {
        probe_set_path_for(&self.database_path)
    }

    /// A run that ended in `RestoreFailed` keeps its snapshot for inspection and
    /// leaves the store possibly mixed. A new run would sweep that snapshot away
    /// and build on the store, so it waits for an operator to clear the manifest.
    fn refuse_after_a_failed_restore(&self) -> anyhow::Result<()> {
        let Some(last) = RunManifest::load(&self.manifest_path())? else {
            return Ok(());
        };
        anyhow::ensure!(
            last.outcome != Some(RunOutcome::RestoreFailed),
            "consolidation run {} ended with a failed restore, so the store needs manual \
             inspection and its snapshot is kept in {}; once it is checked, remove {} to allow \
             new runs",
            last.run_id,
            self.snapshot_dir_for(&last).display(),
            self.manifest_path().display()
        );
        Ok(())
    }

    /// The unfinished run on disk, or a new one. A run made under other models
    /// comes back with the reason it cannot be resumed.
    async fn begin(&self) -> anyhow::Result<(RunManifest, Option<ResumeRefused>)> {
        self.refuse_after_a_failed_restore()?;
        if let Some(unfinished) = self.unfinished_run()? {
            let extraction_model = self.parts.hooks.extraction_model().await;
            let refused = unfinished
                .check_resumable(
                    &self.identity.embedder,
                    &self.identity.llm_model,
                    extraction_model.as_ref(),
                )
                .err();
            tracing::info!(run_id = %unfinished.run_id, done = ?unfinished.completed_steps, refused = refused.is_some(), "consolidation: an unfinished run found");
            return Ok((unfinished, refused));
        }
        self.sweep_dead_snapshots();
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
        Ok((manifest, None))
    }

    async fn forward(
        &self,
        shared: &SharedStore,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<Verdict> {
        for step in STEPS.iter().skip(manifest.resume_point(&STEPS)) {
            tracing::info!(run_id = %manifest.run_id, step, "consolidation: step starting");
            let store = shared.get().await?;
            let outcome = self.run_step(step, &store, manifest).await?;
            drop(store);
            if let StepOutcome::Failed(reason) = outcome {
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
        let source = manifest.write_source();
        match step {
            SNAPSHOT => self.take_snapshot(store, manifest).await?,
            BASELINE => self.record_baseline(store, manifest).await?,
            DEDUP => return self.merge_duplicates(store, manifest).await,
            MAINTENANCE => self.parts.maintenance.run(store, &source).await?,
            EXTRACT => {
                let configured = self.parts.hooks.extraction.is_some();
                // Recorded before the pass writes anything, so a crash during
                // it still leaves the run knowing which model it began with.
                if manifest.extraction_model.is_none() {
                    manifest.extraction_model = self.parts.hooks.extraction_model().await;
                    manifest.save(&self.manifest_path())?;
                }
                let ctx = ExtractCtx { store, source };
                let result = self.parts.hooks.extract(&ctx).await.map(|report| {
                    manifest.extraction_model = report.model.or(manifest.extraction_model.take());
                    report.partial
                });
                self.note_hook(step, configured, result, manifest)?;
            }
            SUMMARIZE => {
                let configured = self.parts.hooks.summary.is_some();
                let result = self.parts.hooks.summarize(&SummaryCtx { source }).await;
                self.note_hook(step, configured, result.map(|_| false), manifest)?;
            }
            TIER => {
                let configured = self.parts.hooks.tiering.is_some();
                let result = self.parts.hooks.tier(&TierCtx { source }).await;
                self.note_hook(step, configured, result.map(|_| false), manifest)?;
            }
            PROBES => return self.check_probes(store, manifest).await,
            other => anyhow::bail!("the consolidation run has no step named {other}"),
        }
        Ok(StepOutcome::Done)
    }

    /// Each run snapshots into a directory of its own, named by its id, so an
    /// attempt that died before the manifest recorded it can be found and
    /// removed by the retry, whatever it left behind.
    fn snapshot_dir_for(&self, manifest: &RunManifest) -> PathBuf {
        self.settings.snapshot_dir.join(&manifest.run_id)
    }

    async fn take_snapshot(
        &self,
        store: &DefaultGraph,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<()> {
        let dir = self.snapshot_dir_for(manifest);
        remove_dir_if_present(&dir);
        let snapshot = store
            .snapshot(&dir)
            .await
            .context("could not snapshot the database before the run")?;
        self.crash_point("snapshot:published")?;
        manifest.snapshot_meta_path = Some(snapshot.meta_path);
        manifest.snapshot_database_path = Some(snapshot.database_path);
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

    async fn merge_duplicates(
        &self,
        store: &DefaultGraph,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<StepOutcome> {
        let after = dedup_cursor::load(&dedup_cursor::cursor_path_for(&self.database_path))?;
        let merged = merge_near_duplicates(
            store,
            self.parts.embedder.as_ref(),
            after,
            ALL_LIVE,
            self.settings.dedup_similarity_threshold,
            &manifest.write_source(),
        )
        .await;
        match merged {
            Ok(report) => {
                tracing::info!(?report, "consolidation: near-duplicates merged");
                manifest.dedup_resume_from = report.resume_from;
                Ok(StepOutcome::Done)
            }
            Err(error) => Ok(StepOutcome::Failed(format!(
                "the near-duplicate merge failed: {error}"
            ))),
        }
    }

    /// Records how a hook step went. A part that is not configured is skipped;
    /// one that fails, or reports it did only part of its work (`Ok(true)`), is
    /// partial. Neither fails the run.
    fn note_hook(
        &self,
        step: &str,
        configured: bool,
        result: anyhow::Result<bool>,
        manifest: &mut RunManifest,
    ) -> anyhow::Result<()> {
        let suffix = match (configured, result) {
            (false, _) => SKIPPED_SUFFIX,
            (true, Ok(false)) => return Ok(()),
            (true, Ok(true)) => PARTIAL_SUFFIX,
            (true, Err(error)) => {
                tracing::warn!(step, %error, "consolidation: a hook failed, the run goes on");
                PARTIAL_SUFFIX
            }
        };
        manifest.mark_step_complete(&format!("{step}{suffix}"), &self.manifest_path())
    }

    async fn check_probes(
        &self,
        store: &DefaultGraph,
        manifest: &RunManifest,
    ) -> anyhow::Result<StepOutcome> {
        let loaded = ProbeSet::load(&self.probe_set_path())?;
        if loaded.probes.is_empty() {
            tracing::info!(
                "consolidation: the store had no live node to probe, the gate is skipped"
            );
            return Ok(StepOutcome::Done);
        }
        let swept = store.nodes_swept_by(&manifest.write_source()).await?;
        let surviving = loaded.without_swept(store, &swept, ALL_LIVE).await?;
        if !surviving.lost.is_empty() {
            return Ok(StepOutcome::Failed(format!(
                "probed nodes vanished that the run's sweep did not remove: {}",
                surviving.lost.join(", ")
            )));
        }
        let set = surviving.set;
        if set.probes.is_empty() {
            return Ok(StepOutcome::Failed(
                "the run's sweep removed every probed node, so nothing is left to check it against"
                    .to_owned(),
            ));
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

    fn commit(&self, mut manifest: RunManifest) -> anyhow::Result<RunManifest> {
        if let Some(position) = manifest.dedup_resume_from {
            dedup_cursor::save(
                &dedup_cursor::cursor_path_for(&self.database_path),
                position,
            )?;
        }
        manifest.outcome = Some(RunOutcome::Committed);
        manifest.save(&self.manifest_path())?;
        self.discard_leftovers(&manifest);
        tracing::info!(run_id = %manifest.run_id, "consolidation: run committed");
        Ok(manifest)
    }

    /// Removes the snapshot and probe set once nothing can need them. A
    /// failure only leaves files behind, so it is logged and not raised.
    fn discard_leftovers(&self, manifest: &RunManifest) {
        remove_if_present(&self.probe_set_path());
        remove_dir_if_present(&self.snapshot_dir_for(manifest));
    }

    /// Removes the snapshot directories of runs that no manifest can resume.
    /// Called when a fresh run starts, so no unfinished run owns any of them.
    fn sweep_dead_snapshots(&self) {
        let Ok(entries) = std::fs::read_dir(&self.settings.snapshot_dir) else {
            return;
        };
        for entry in entries.flatten() {
            remove_dir_if_present(&entry.path());
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

fn remove_dir_if_present(path: &Path) {
    match std::fs::remove_dir_all(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "consolidation: could not remove a leftover directory")
        }
    }
}

mod rollback;
#[cfg(test)]
mod rollback_tests;
#[cfg(test)]
mod tests;
