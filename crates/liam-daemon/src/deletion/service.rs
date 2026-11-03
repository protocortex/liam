// SPDX-License-Identifier: Apache-2.0
//! Running a delete job. The steps go in a fixed order and each one is safe to
//! run again, so a job that stopped anywhere, by a crash or by a failure, is
//! resumed from the state it saved:
//!
//! 1. scan the log and save what it found (`Requested` to `Destroying`)
//! 2. mark the key groups, record the deletion, destroy the keys one by one
//! 3. purge every store, then scrub the database and forget probe nodes
//!
//! The caller holds the store lock for as long as the service lives, which is
//! what keeps every other writer out.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use liam_log::crypto::KeyRegistry;
use liam_log::reader::LogReader;
use liam_store::{Clock, DefaultGraph, DeletionRecord, LineageHits, LineageTarget};

use super::job::{job_file_path_for, DeadlineAlert, DeleteJob, Failure, JobState, JobStore};
use super::memory::NodeState;
use super::registry::LineageRegistry;
use super::scan::{scan, Found, ScanError};
use super::scrub::{scrub_database, ScrubOutcome};
use super::target::DeleteTarget;
use crate::config::Config;
use crate::storelock::StoreLock;
use liam_daemon::consolidation::probe_set::{forget_expected_ids, probe_set_path_for};
use liam_daemon::models::resolve_snapshot_dir;

/// Where a job stops in a test, standing in for a crash. The step before it has
/// been saved and the step after it has not started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashPoint {
    AfterRequest,
    AfterScan,
    AfterMark,
    AfterDeletionRecorded,
    /// The bytes of key `n` are gone and the log does not say so yet.
    AfterKeyDestroyed(usize),
    AfterKeyRecorded(usize),
    AfterStorePurged(usize),
    AfterPurge,
    AfterScrub,
    BeforeDone,
}

/// Why a call on the service did not do what it was asked.
#[derive(Debug)]
pub enum DeleteError {
    /// The target cannot be deleted as asked. Nothing was destroyed and the job
    /// is cancelled.
    Refused(ScanError),
    UnknownJob(String),
    /// The job has destroyed a key, or is past the point of destroying, so it
    /// can only be finished.
    CancelRefused {
        job_id: String,
        state: JobState,
    },
    /// A step failed. The job is `Failed` and is retried by an explicit resume.
    JobFailed {
        job_id: String,
        reason: String,
    },
    /// The job file could not be read or written.
    JobFile(anyhow::Error),
    #[cfg(test)]
    Crashed(CrashPoint),
    /// Used inside a step; the runner turns it into `JobFailed`.
    Step(anyhow::Error),
}

impl fmt::Display for DeleteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(error) => write!(f, "the delete was refused: {error}"),
            Self::UnknownJob(id) => write!(f, "there is no delete job {id}"),
            Self::CancelRefused { job_id, state } => write!(
                f,
                "delete job {job_id} is {state:?} and past the point where it can be cancelled"
            ),
            Self::JobFailed { job_id, reason } => {
                write!(f, "delete job {job_id} failed: {reason}")
            }
            Self::JobFile(error) => write!(f, "{error:#}"),
            #[cfg(test)]
            Self::Crashed(point) => write!(f, "stopped at {point:?}"),
            Self::Step(error) => write!(f, "{error:#}"),
        }
    }
}

impl std::error::Error for DeleteError {}

impl From<anyhow::Error> for DeleteError {
    fn from(error: anyhow::Error) -> Self {
        Self::Step(error)
    }
}

/// Jobs that did not finish when the daemon tried to resume them.
#[derive(Debug, PartialEq, Eq)]
pub struct DeleteJobPending {
    pub job_ids: Vec<String>,
}

impl fmt::Display for DeleteJobPending {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "delete jobs {} did not finish, and serving would expose a half-deleted subject; \
             resolve them with `liamd delete --resume` before starting",
            self.job_ids.join(", ")
        )
    }
}

impl std::error::Error for DeleteJobPending {}

/// How a run that did not fail ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutcome {
    Done,
    /// A reader held the database log, so the scrub is open. Resume once it is
    /// released.
    ScrubPending,
}

/// What `resume_all` found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ResumeReport {
    pub finished: Vec<String>,
    /// Jobs still open: failed, or waiting on a scrub.
    pub pending: Vec<String>,
}

/// What the service works on.
pub struct DeleteDeps {
    pub jobs: JobStore,
    pub store: Arc<DefaultGraph>,
    pub keys: Arc<KeyRegistry>,
    pub registry: LineageRegistry,
    /// A reader that opens sealed records.
    pub reader: Arc<dyn LogReader>,
    pub probe_set: PathBuf,
    pub clock: Arc<dyn Clock>,
    pub deadline_days: u32,
}

/// The open store and the log around it.
pub struct StoreParts {
    pub store: Arc<DefaultGraph>,
    pub keys: Arc<KeyRegistry>,
    /// A reader that opens sealed records.
    pub reader: Arc<dyn LogReader>,
    pub clock: Arc<dyn Clock>,
}

pub struct DeleteService {
    deps: DeleteDeps,
    #[cfg(test)]
    fault: Option<CrashPoint>,
}

/// What a step leaves for the runner.
enum Next {
    Continue,
    Stop(RunOutcome),
}

impl DeleteService {
    /// The lock is only asked for, as proof that no other process has the store.
    pub fn new(deps: DeleteDeps, _lock: &StoreLock) -> Self {
        Self {
            deps,
            #[cfg(test)]
            fault: None,
        }
    }

    /// The service for the database at `database_path`, with the stores a
    /// delete reaches and the job file beside it. `states` are the in-memory
    /// holders of node ids that are alive in this process.
    pub fn open(
        config: &Config,
        database_path: &Path,
        parts: StoreParts,
        states: Vec<Arc<dyn NodeState>>,
        lock: &StoreLock,
    ) -> anyhow::Result<Self> {
        let snapshot_dir = resolve_snapshot_dir(config, database_path)?;
        let registry = LineageRegistry::for_database(
            database_path,
            &snapshot_dir,
            Arc::clone(&parts.store),
            states,
        )?;
        Ok(Self::new(
            DeleteDeps {
                jobs: JobStore::new(job_file_path_for(database_path)),
                store: parts.store,
                keys: parts.keys,
                registry,
                reader: parts.reader,
                probe_set: probe_set_path_for(database_path),
                clock: parts.clock,
                deadline_days: config.deletion.deadline_days,
            },
            lock,
        ))
    }

    #[cfg(test)]
    pub fn stopping_at(mut self, point: CrashPoint) -> Self {
        self.fault = Some(point);
        self
    }

    fn crash_at(&self, point: CrashPoint) -> Result<(), DeleteError> {
        #[cfg(test)]
        if self.fault == Some(point) {
            return Err(DeleteError::Crashed(point));
        }
        let _ = point;
        Ok(())
    }

    /// Saves a job for `target` and returns its id. Nothing runs yet.
    pub fn request(&self, target: DeleteTarget) -> Result<String, DeleteError> {
        let id = ulid::Ulid::gen().to_string();
        let job = DeleteJob::new(
            id.clone(),
            target,
            self.deps.clock.now().0,
            self.deps.deadline_days,
        );
        self.save(&job)?;
        tracing::info!(job = %id, "delete job requested");
        self.crash_at(CrashPoint::AfterRequest)?;
        Ok(id)
    }

    pub fn jobs(&self) -> Result<Vec<DeleteJob>, DeleteError> {
        self.deps.jobs.load().map_err(DeleteError::JobFile)
    }

    /// The deadline state of every job still open.
    pub fn alerts(&self) -> Result<Vec<(String, DeadlineAlert)>, DeleteError> {
        let now = self.deps.clock.now().0;
        Ok(self
            .jobs()?
            .into_iter()
            .map(|job| (job.alert(now), job.id))
            .filter(|(alert, _)| *alert != DeadlineAlert::None)
            .map(|(alert, id)| (id, alert))
            .collect())
    }

    /// Runs job `id` from the state it saved. A failed job is retried from the
    /// step it failed at.
    pub async fn run(&self, id: &str) -> Result<RunOutcome, DeleteError> {
        let mut job = self.load(id)?;
        if job.state == JobState::Failed {
            let failure = job.failure.take();
            job.state = failure.map_or(JobState::Requested, |failure| failure.at);
            self.save(&job)?;
        }
        self.run_job(job).await
    }

    /// Runs every job that is not final, for the daemon's start. A failed job
    /// is left alone unless `retry_failed`, because it asked for an explicit
    /// resume.
    pub async fn resume_all(&self, retry_failed: bool) -> Result<ResumeReport, DeleteError> {
        let mut report = ResumeReport::default();
        for job in self.jobs()? {
            if job.state.is_final() {
                continue;
            }
            if job.state == JobState::Failed && !retry_failed {
                report.pending.push(job.id);
                continue;
            }
            match self.run(&job.id).await {
                Ok(RunOutcome::Done) => report.finished.push(job.id),
                Ok(RunOutcome::ScrubPending) => report.pending.push(job.id),
                Err(DeleteError::JobFailed { job_id, reason }) => {
                    tracing::error!(job = %job_id, %reason, "delete job failed");
                    report.pending.push(job_id);
                }
                Err(DeleteError::Refused(error)) => {
                    tracing::warn!(job = %job.id, %error, "delete job refused");
                }
                Err(other) => return Err(other),
            }
        }
        Ok(report)
    }

    /// Returns a job that has not destroyed a key to where it began and ends it
    /// as `Cancelled`. A job past that point can only be finished.
    pub async fn cancel(&self, id: &str) -> Result<(), DeleteError> {
        let mut job = self.load(id)?;
        let before_destroy = matches!(
            job.state,
            JobState::Requested | JobState::Destroying | JobState::Failed
        ) && !job.destroy_started;
        if !before_destroy {
            return Err(DeleteError::CancelRefused {
                job_id: job.id,
                state: job.state,
            });
        }
        for key in &job.key_ids {
            self.deps
                .keys
                .cancel_destroy(key)
                .map_err(|error| DeleteError::Step(error.into()))?;
        }
        if self.may_have_recorded(&job) {
            self.record(DeletionRecord::Cancelled {
                job_id: job.id.clone(),
            })
            .await?;
        }
        job.state = JobState::Cancelled;
        self.save(&job)?;
        tracing::info!(job = %job.id, "delete job cancelled");
        Ok(())
    }

    /// Whether the deletion could be in the log: the job got to marking.
    fn may_have_recorded(&self, job: &DeleteJob) -> bool {
        let reached = |state| job.state == state;
        reached(JobState::Destroying)
            || job
                .failure
                .as_ref()
                .is_some_and(|failure| failure.at == JobState::Destroying)
    }

    fn load(&self, id: &str) -> Result<DeleteJob, DeleteError> {
        self.deps
            .jobs
            .get(id)
            .map_err(DeleteError::JobFile)?
            .ok_or_else(|| DeleteError::UnknownJob(id.to_owned()))
    }

    fn save(&self, job: &DeleteJob) -> Result<(), DeleteError> {
        self.deps.jobs.save(job).map_err(DeleteError::JobFile)
    }

    async fn record(&self, record: DeletionRecord) -> Result<(), DeleteError> {
        let appended = self
            .deps
            .store
            .record_deletion(&record)
            .await
            .map_err(|error| DeleteError::Step(error.into()))?;
        tracing::debug!(?record, appended, "deletion record written");
        Ok(())
    }

    async fn run_job(&self, mut job: DeleteJob) -> Result<RunOutcome, DeleteError> {
        loop {
            let step = match job.state {
                JobState::Requested => self.start(&mut job).await,
                JobState::Destroying => self.destroy(&mut job).await,
                JobState::KeyDestroyed => self.move_to(&mut job, JobState::Purging),
                JobState::Purging => self.purge(&mut job).await,
                JobState::Scrubbing => self.scrub(&mut job).await,
                JobState::Done | JobState::Cancelled | JobState::Failed => {
                    return Ok(RunOutcome::Done)
                }
            };
            match step {
                Ok(Next::Continue) => {}
                Ok(Next::Stop(outcome)) => return Ok(outcome),
                Err(DeleteError::Step(error)) => return self.fail(job, &error),
                Err(DeleteError::Refused(error)) => {
                    job.state = JobState::Cancelled;
                    job.failure = Some(Failure {
                        at: JobState::Requested,
                        reason: error.to_string(),
                    });
                    self.save(&job)?;
                    return Err(DeleteError::Refused(error));
                }
                Err(other) => return Err(other),
            }
        }
    }

    fn fail(&self, mut job: DeleteJob, error: &anyhow::Error) -> Result<RunOutcome, DeleteError> {
        let reason = format!("{error:#}");
        tracing::error!(job = %job.id, state = ?job.state, %reason, "delete job failed");
        job.failure = Some(Failure {
            at: job.state,
            reason: reason.clone(),
        });
        job.state = JobState::Failed;
        self.save(&job)?;
        Err(DeleteError::JobFailed {
            job_id: job.id,
            reason,
        })
    }

    fn move_to(&self, job: &mut DeleteJob, state: JobState) -> Result<Next, DeleteError> {
        job.state = state;
        self.save(job)?;
        tracing::info!(job = %job.id, ?state, "delete job moved");
        Ok(Next::Continue)
    }

    /// What deleting `target` would destroy, from the log and the stores as they
    /// stand. Nothing is changed.
    pub async fn dry_run(&self, target: &DeleteTarget) -> Result<Preview, DeleteError> {
        preview(
            &self.deps.store,
            self.deps.reader.as_ref(),
            &self.deps.keys,
            &self.deps.registry,
            target,
        )
        .await
    }

    /// Scans the log, checks the registry knows every key, and fixes the
    /// nodes the stores outside the database will delete by.
    async fn start(&self, job: &mut DeleteJob) -> Result<Next, DeleteError> {
        let found = find(&self.deps.store, self.deps.reader.as_ref(), &job.target).await?;
        self.ensure_known(&found.key_ids)?;
        job.event_ids = found.event_ids;
        job.key_ids = found.key_ids;
        job.node_ids = self.resolved(job).await?.node_ids;
        job.state = JobState::Destroying;
        self.save(job)?;
        tracing::info!(
            job = %job.id,
            events = job.event_ids.len(),
            keys = job.key_ids.len(),
            "log scanned"
        );
        self.crash_at(CrashPoint::AfterScan)?;
        Ok(Next::Continue)
    }

    async fn destroy(&self, job: &mut DeleteJob) -> Result<Next, DeleteError> {
        self.ensure_known(&job.key_ids)?;
        for key in &job.key_ids {
            self.deps
                .keys
                .begin_destroy(key)
                .map_err(|error| DeleteError::Step(error.into()))?;
        }
        self.crash_at(CrashPoint::AfterMark)?;
        self.record(DeletionRecord::Recorded {
            job_id: job.id.clone(),
            event_ids: job.event_ids.clone(),
            key_ids: job.key_ids.clone(),
        })
        .await?;
        self.crash_at(CrashPoint::AfterDeletionRecorded)?;
        job.destroy_started = true;
        self.save(job)?;
        for (index, key) in job.key_ids.iter().enumerate() {
            self.deps
                .keys
                .finish_destroy(key)
                .map_err(|error| DeleteError::Step(error.into()))?;
            self.crash_at(CrashPoint::AfterKeyDestroyed(index))?;
            self.record(DeletionRecord::KeyDestroyed {
                key_id: key.clone(),
            })
            .await?;
            self.crash_at(CrashPoint::AfterKeyRecorded(index))?;
        }
        self.move_to(job, JobState::KeyDestroyed)
    }

    fn ensure_known(&self, key_ids: &[String]) -> Result<(), DeleteError> {
        ensure_known(&self.deps.keys, key_ids)
    }

    /// The target every store is given: resolved once, from the outbox that
    /// the scan filled, so a rerun deletes by the same sets.
    async fn resolved(&self, job: &DeleteJob) -> Result<LineageTarget, DeleteError> {
        self.deps
            .store
            .resolve_target(&LineageTarget {
                job_id: job.id.clone(),
                event_ids: job.event_ids.clone(),
                key_ids: job.key_ids.clone(),
                ..LineageTarget::default()
            })
            .await
            .map_err(|error| DeleteError::Step(error.into()))
    }

    async fn purge(&self, job: &mut DeleteJob) -> Result<Next, DeleteError> {
        let target = self.resolved(job).await?;
        for index in 0..self.deps.registry.store_count() {
            self.deps.registry.purge_at(index, &target).await?;
            self.crash_at(CrashPoint::AfterStorePurged(index))?;
        }
        self.crash_at(CrashPoint::AfterPurge)?;
        self.move_to(job, JobState::Scrubbing)
    }

    async fn scrub(&self, job: &mut DeleteJob) -> Result<Next, DeleteError> {
        if scrub_database(&self.deps.store).await? == ScrubOutcome::Incomplete {
            return Ok(Next::Stop(RunOutcome::ScrubPending));
        }
        self.crash_at(CrashPoint::AfterScrub)?;
        let nodes: HashSet<String> = job.node_ids.iter().cloned().collect();
        forget_expected_ids(&self.deps.probe_set, &nodes)?;
        let target = self.resolved(job).await?;
        let residuals = self.deps.registry.verify_all(&target).await?;
        if !residuals.is_empty() {
            let stores: Vec<String> = residuals
                .iter()
                .map(|residual| format!("{}:{}", residual.store, residual.kind))
                .collect();
            return Err(DeleteError::Step(anyhow::anyhow!(
                "{} rows still refer to the deleted data: {}",
                residuals.len(),
                stores.join(", ")
            )));
        }
        self.crash_at(CrashPoint::BeforeDone)?;
        self.move_to(job, JobState::Done)?;
        Ok(Next::Stop(RunOutcome::Done))
    }
}

/// What a delete of a target would do.
#[derive(Debug)]
pub struct Preview {
    pub found: Found,
    /// The rows each store would remove, in the order the stores are purged.
    pub hits: Vec<(String, LineageHits)>,
}

impl fmt::Display for Preview {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let found = &self.found;
        write!(
            f,
            "{} events, {} nodes and {} key groups\nkey groups: {}",
            found.event_ids.len(),
            found.node_ids.len(),
            found.key_ids.len(),
            found.key_ids.join(", ")
        )?;
        for (store, hits) in &self.hits {
            write!(f, "\nstore {store}: {} rows", hits.total())?;
            for (table, rows) in &hits.rows {
                write!(f, ", {table} {rows}")?;
            }
        }
        Ok(())
    }
}

/// The job id a preview asks the stores about. A real job id is a ULID, so no
/// outbox row carries this one.
const PREVIEW_JOB: &str = "preview";

/// Reads the log and the stores for `target` and changes nothing.
pub async fn preview(
    store: &DefaultGraph,
    reader: &dyn LogReader,
    keys: &KeyRegistry,
    registry: &LineageRegistry,
    target: &DeleteTarget,
) -> Result<Preview, DeleteError> {
    let found = find(store, reader, target).await?;
    ensure_known(keys, &found.key_ids)?;
    let asked = LineageTarget {
        job_id: PREVIEW_JOB.to_owned(),
        event_ids: found.event_ids.clone(),
        key_ids: found.key_ids.clone(),
        ..LineageTarget::default()
    };
    let resolved = store
        .preview_target(&asked)
        .await
        .map_err(|error| DeleteError::Step(error.into()))?;
    let hits = registry.find_all(&resolved).await?;
    Ok(Preview { found, hits })
}

async fn find(
    store: &DefaultGraph,
    reader: &dyn LogReader,
    target: &DeleteTarget,
) -> Result<Found, DeleteError> {
    let groups = match target {
        DeleteTarget::Subject {
            subject,
            scope,
            include_children,
        } => {
            // No group is not an early answer: plain rows of the subject have
            // none, and the scan is what finds them.
            store
                .key_groups_of_subject(subject, scope.as_deref(), *include_children)
                .await
                .map_err(|error| DeleteError::Step(error.into()))?
        }
        DeleteTarget::Event { .. } => Vec::new(),
    };
    scan(reader, target, &groups).await.map_err(scan_failure)
}

/// A key the registry does not know is damage, not a delete, so the job stops
/// rather than call it destroyed. A destroyed key is known, which is what lets
/// a resume pass the destroy.
fn ensure_known(keys: &KeyRegistry, key_ids: &[String]) -> Result<(), DeleteError> {
    match key_ids.iter().find(|key| keys.state(key).is_none()) {
        Some(key) => Err(DeleteError::Step(anyhow::anyhow!(
            "key {key} is not in the key registry, so the job cannot destroy it"
        ))),
        None => Ok(()),
    }
}

/// An unreachable archive or a failed read fails the job; the other errors say
/// the target cannot be deleted as asked.
fn scan_failure(error: ScanError) -> DeleteError {
    match error {
        ScanError::PartitionUnreachable { .. } | ScanError::Reader(_) => {
            DeleteError::Step(anyhow::anyhow!("{error}"))
        }
        refusal => DeleteError::Refused(refusal),
    }
}

#[cfg(test)]
mod tests;
