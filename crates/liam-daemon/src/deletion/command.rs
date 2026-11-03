// SPDX-License-Identifier: Apache-2.0
//! `liamd delete`: counts what a delete would destroy, starts the job when
//! told to, and finishes, cancels or lists the jobs a stopped daemon left.
//! Everything but `--status` takes the store lock, so it refuses to run beside
//! a daemon.

use std::fmt;
use std::io::Write;
use std::sync::Arc;

use anyhow::Context;
use liam_daemon::event_log;
use liam_store::{validate_scope, Clock, SystemClock};

use super::job::{job_file_path_for, DeadlineAlert, DeleteJob, JobStore};
use super::service::{DeleteError, DeleteService, RunOutcome, StoreParts};
use super::target::DeleteTarget;
use crate::config::Config;
use crate::log_startup::open_store_for_delete;
use crate::storelock::StoreLock;
use crate::{resolve_config_path, resolve_log_dir};

const MILLIS_PER_DAY: i64 = 86_400_000;

/// What `liamd delete` was asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeleteAction {
    /// Count what `target` would destroy, and start the job only with `yes`.
    Request {
        target: DeleteTarget,
        yes: bool,
    },
    /// Finish one job, or every open job, failed ones included.
    Resume(Option<String>),
    Cancel(String),
    /// List the jobs by id and state.
    Status,
}

/// Another process, most likely the daemon, holds the store.
#[derive(Debug)]
pub struct StoreHeld;

impl fmt::Display for StoreHeld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "liamd delete needs the store to itself: stop the daemon, then run it again"
        )
    }
}

impl std::error::Error for StoreHeld {}

/// Runs `action`, writing what an operator needs to read to `out`.
pub async fn run(
    config: &Config,
    action: DeleteAction,
    out: &mut impl Write,
) -> anyhow::Result<()> {
    let database_path = resolve_config_path("database_path", &config.database_path)?;
    if action == DeleteAction::Status {
        let jobs = JobStore::new(job_file_path_for(&database_path)).load()?;
        return write_status(&jobs, SystemClock.now().0, out);
    }
    let log_dir = resolve_log_dir(config, &database_path)?;
    let lock = StoreLock::acquire(&database_path)
        .map_err(|source| anyhow::Error::new(StoreHeld).context(format!("{source:#}")))?;
    let (store, keys) = open_store_for_delete(config, &database_path, &log_dir).await?;
    let keys = keys.context(
        "the log has no key registry, so nothing in it can be deleted by key; \
         configure [encryption] first",
    )?;
    let parts = StoreParts {
        reader: event_log::log_reader(&log_dir, Some(&keys))?,
        store: Arc::new(store),
        keys,
        clock: Arc::new(SystemClock),
    };
    let service = DeleteService::open(config, &database_path, parts, Vec::new(), &lock)?;
    match action {
        DeleteAction::Request { target, yes } => request(&service, target, yes, out).await,
        DeleteAction::Resume(None) => resume_all(&service, out).await,
        DeleteAction::Resume(Some(id)) => finish(&service, &id, out).await,
        DeleteAction::Cancel(id) => {
            service.cancel(&id).await?;
            writeln!(out, "delete job {id} cancelled")?;
            Ok(())
        }
        DeleteAction::Status => unreachable!("status returned before the store was opened"),
    }
}

async fn request(
    service: &DeleteService,
    target: DeleteTarget,
    yes: bool,
    out: &mut impl Write,
) -> anyhow::Result<()> {
    let target = checked(target)?;
    let preview = service.dry_run(&target).await?;
    writeln!(out, "this would destroy {preview}")?;
    if !yes {
        writeln!(out, "nothing was changed; add --yes to delete")?;
        return Ok(());
    }
    let id = service.request(target)?;
    writeln!(out, "delete job {id} started")?;
    finish(service, &id, out).await
}

/// The scope as the store keeps it, so a blank one is refused here.
fn checked(target: DeleteTarget) -> anyhow::Result<DeleteTarget> {
    match target {
        DeleteTarget::Subject {
            subject,
            scope,
            include_children,
        } => Ok(DeleteTarget::Subject {
            subject,
            scope: validate_scope(&scope)?,
            include_children,
        }),
        event => Ok(event),
    }
}

async fn finish(service: &DeleteService, id: &str, out: &mut impl Write) -> anyhow::Result<()> {
    match service.run(id).await {
        Ok(RunOutcome::Done) => {
            writeln!(out, "delete job {id} is done")?;
            Ok(())
        }
        Ok(RunOutcome::ScrubPending) => {
            writeln!(
                out,
                "delete job {id} is waiting for a reader to release the database; \
                 run `liamd delete --resume` once it has"
            )?;
            Ok(())
        }
        Err(error @ DeleteError::JobFailed { .. }) => Err(anyhow::Error::new(error)
            .context("the job is saved; fix the cause and run `liamd delete --resume`")),
        Err(error) => Err(error.into()),
    }
}

async fn resume_all(service: &DeleteService, out: &mut impl Write) -> anyhow::Result<()> {
    let report = service.resume_all(true).await?;
    writeln!(out, "{} delete jobs finished", report.finished.len())?;
    anyhow::ensure!(
        report.pending.is_empty(),
        "delete jobs {} did not finish; see the log for why, then run `liamd delete --resume` again",
        report.pending.join(", ")
    );
    Ok(())
}

/// One line per job: its id, where it is and how close to its deadline. The
/// subject is left out, because the output is read where the subject should
/// not be.
fn write_status(jobs: &[DeleteJob], now: i64, out: &mut impl Write) -> anyhow::Result<()> {
    if jobs.is_empty() {
        writeln!(out, "no delete jobs")?;
    }
    for job in jobs {
        let late = match job.alert(now) {
            DeadlineAlert::None => "",
            DeadlineAlert::Approaching => ", close to its deadline",
            DeadlineAlert::Overdue => ", past its deadline",
        };
        let days_left = (job.deadline_at - now).div_euclid(MILLIS_PER_DAY);
        writeln!(
            out,
            "job {}: {:?}, {days_left} days to its deadline{late}",
            job.id, job.state
        )?;
        if let Some(failure) = &job.failure {
            writeln!(out, "  failed at {:?}: {}", failure.at, failure.reason)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::deletion::job::{Failure, JobState};
    use crate::deletion::test_support::sealed_subjects;
    use crate::log_startup::test_support::database_in;

    const SUBJECT: &str = "alice";

    fn alice(yes: bool) -> DeleteAction {
        DeleteAction::Request {
            target: DeleteTarget::Subject {
                subject: SUBJECT.into(),
                scope: None,
                include_children: false,
            },
            yes,
        }
    }

    /// A sealed store with alice and bob, closed, and the config that reaches it.
    async fn sealed() -> (TempDir, Config) {
        let dir = TempDir::new().unwrap();
        let config = sealed_subjects(dir.path()).await.config.clone();
        (dir, config)
    }

    fn jobs(dir: &TempDir) -> Vec<DeleteJob> {
        JobStore::new(job_file_path_for(&database_in(dir.path())))
            .load()
            .unwrap()
    }

    async fn output(config: &Config, action: DeleteAction) -> (anyhow::Result<()>, String) {
        let mut out = Vec::new();
        let result = run(config, action, &mut out).await;
        (result, String::from_utf8(out).unwrap())
    }

    #[tokio::test]
    async fn without_yes_the_counts_are_printed_and_nothing_starts() {
        // Arrange
        let (dir, config) = sealed().await;

        // Act
        let (result, text) = output(&config, alice(false)).await;

        // Assert
        result.expect("a count succeeds");
        assert!(text.contains("this would destroy"), "{text}");
        assert!(text.contains("add --yes"), "{text}");
        assert!(jobs(&dir).is_empty(), "no job was saved");
    }

    #[tokio::test]
    async fn with_yes_the_job_runs_to_done() {
        // Arrange
        let (dir, config) = sealed().await;

        // Act
        let (result, text) = output(&config, alice(true)).await;

        // Assert
        result.expect("the job runs");
        let jobs = jobs(&dir);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].state, JobState::Done);
        assert!(
            text.contains(&format!("delete job {} is done", jobs[0].id)),
            "{text}"
        );
    }

    #[tokio::test]
    async fn a_subject_with_nothing_held_is_refused_without_a_job() {
        // Arrange
        let (dir, config) = sealed().await;
        let action = DeleteAction::Request {
            target: DeleteTarget::Subject {
                subject: "nobody".into(),
                scope: None,
                include_children: false,
            },
            yes: true,
        };

        // Act
        let (result, _) = output(&config, action).await;

        // Assert
        let error = format!("{:#}", result.expect_err("nothing to delete"));
        assert!(error.contains("nothing"), "{error}");
        assert!(jobs(&dir).is_empty());
    }

    #[tokio::test]
    async fn a_running_daemon_holding_the_store_is_a_typed_refusal() {
        // Arrange
        let (dir, config) = sealed().await;
        let _daemon = StoreLock::acquire(&database_in(dir.path())).unwrap();

        // Act
        let (result, _) = output(&config, alice(true)).await;

        // Assert
        let error = result.expect_err("the store is held");
        assert!(error.downcast_ref::<StoreHeld>().is_some(), "{error:#}");
        assert!(jobs(&dir).is_empty());
    }

    #[tokio::test]
    async fn status_lists_ids_and_states_and_never_the_subject() {
        // Arrange
        let (dir, config) = sealed().await;
        let mut job = DeleteJob::new("job-failed".into(), alice_target(), 0, 30);
        job.state = JobState::Failed;
        job.failure = Some(Failure {
            at: JobState::Purging,
            reason: "purging graph failed".into(),
        });
        JobStore::new(job_file_path_for(&database_in(dir.path())))
            .save(&job)
            .unwrap();
        let _daemon = StoreLock::acquire(&database_in(dir.path())).unwrap();

        // Act
        let (result, text) = output(&config, DeleteAction::Status).await;

        // Assert
        result.expect("status needs no lock");
        assert!(text.contains("job job-failed: Failed"), "{text}");
        assert!(text.contains("purging graph failed"), "{text}");
        assert!(!text.contains(SUBJECT), "{text}");
    }

    #[tokio::test]
    async fn resume_finishes_a_failed_job_and_cancel_ends_one_that_has_not_started() {
        // Arrange
        let (dir, config) = sealed().await;
        let store = JobStore::new(job_file_path_for(&database_in(dir.path())));
        store
            .save(&DeleteJob::new("job-cancel".into(), alice_target(), 0, 30))
            .unwrap();

        // Act
        let (cancelled, text) = output(&config, DeleteAction::Cancel("job-cancel".into())).await;
        let (resumed, _) = output(&config, DeleteAction::Resume(None)).await;

        // Assert
        cancelled.expect("cancel");
        assert!(text.contains("job-cancel cancelled"), "{text}");
        resumed.expect("nothing left to resume");
        assert_eq!(jobs(&dir)[0].state, JobState::Cancelled);
    }

    fn alice_target() -> DeleteTarget {
        DeleteTarget::Subject {
            subject: SUBJECT.into(),
            scope: None,
            include_children: false,
        }
    }
}
