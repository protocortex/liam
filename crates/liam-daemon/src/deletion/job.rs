// SPDX-License-Identifier: Apache-2.0
//! A delete job and the file that keeps it. The job records how far it got, so
//! a crash at any point resumes where it stopped.

use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use serde::{Deserialize, Serialize};

use super::target::DeleteTarget;

/// Appended to the database path to name the job file.
const JOB_FILE_SUFFIX: &str = ".delete-jobs.json";
/// Layout version of the file. Any other is refused.
const FORMAT_VERSION: u32 = 1;
pub(super) const MILLIS_PER_DAY: i64 = 86_400_000;
/// The share of the deadline window after which a job is reported as late.
const APPROACHING_PERCENT: i64 = 80;
const PERCENT: i64 = 100;

/// Where the job file of the database at `database_path` lives.
pub fn job_file_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(JOB_FILE_SUFFIX);
    PathBuf::from(path)
}

/// The step a job is at. A job runs these in order; `Failed` and `Cancelled`
/// leave the line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobState {
    /// Saved, nothing found or touched yet.
    Requested,
    /// The log was scanned and the ids saved. The key groups are marked, the
    /// deletion recorded, and the keys destroyed one by one.
    Destroying,
    KeyDestroyed,
    Purging,
    Scrubbing,
    Done,
    Cancelled,
    Failed,
}

impl JobState {
    /// A job in a final state has nothing left to run.
    pub fn is_final(self) -> bool {
        matches!(self, Self::Done | Self::Cancelled)
    }
}

/// Why a job stopped, and the step it retries from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    pub at: JobState,
    pub reason: String,
}

/// How late a job is against its deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlineAlert {
    None,
    /// 80 percent of the window has passed.
    Approaching,
    /// The deadline has been reached.
    Overdue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeleteJob {
    pub id: String,
    pub target: DeleteTarget,
    pub state: JobState,
    /// Epoch milliseconds.
    pub requested_at: i64,
    /// The inclusive end of the window: reached at `requested_at` plus the
    /// configured days.
    pub deadline_at: i64,
    pub event_ids: Vec<String>,
    pub node_ids: Vec<String>,
    /// The key groups of the target and the row keys of every write citing it.
    pub key_ids: Vec<String>,
    /// Set before the first key is destroyed. From then on the job cannot be
    /// cancelled.
    pub destroy_started: bool,
    pub failure: Option<Failure>,
    /// What the job could not remove, for the operator to act on. A job file
    /// written before notices existed reads as having none.
    #[serde(default)]
    pub notices: Vec<String>,
}

impl DeleteJob {
    pub fn new(id: String, target: DeleteTarget, requested_at: i64, deadline_days: u32) -> Self {
        Self {
            id,
            target,
            state: JobState::Requested,
            requested_at,
            deadline_at: requested_at + i64::from(deadline_days) * MILLIS_PER_DAY,
            event_ids: Vec::new(),
            node_ids: Vec::new(),
            key_ids: Vec::new(),
            destroy_started: false,
            failure: None,
            notices: Vec::new(),
        }
    }

    /// A job counts against its deadline until it is done or cancelled, a
    /// failed one included.
    pub fn alert(&self, now: i64) -> DeadlineAlert {
        if self.state.is_final() {
            return DeadlineAlert::None;
        }
        if now >= self.deadline_at {
            return DeadlineAlert::Overdue;
        }
        let window = self.deadline_at - self.requested_at;
        if now - self.requested_at >= window * APPROACHING_PERCENT / PERCENT {
            return DeadlineAlert::Approaching;
        }
        DeadlineAlert::None
    }
}

#[derive(Serialize, Deserialize)]
struct JobFile {
    version: u32,
    jobs: Vec<DeleteJob>,
}

/// The jobs of one database, kept in a file that is replaced whole after every
/// change.
pub struct JobStore {
    path: PathBuf,
}

impl JobStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Every job in the file, oldest first. A file that does not exist holds
    /// none.
    pub fn load(&self) -> anyhow::Result<Vec<DeleteJob>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&self.path).map_err(|source| {
            anyhow::anyhow!(
                "the delete job file {} could not be read: {source}",
                self.path.display()
            )
        })?;
        let file: JobFile = serde_json::from_str(&text).map_err(|source| {
            anyhow::anyhow!(
                "the delete job file {} could not be parsed: {source}",
                self.path.display()
            )
        })?;
        anyhow::ensure!(
            file.version == FORMAT_VERSION,
            "the delete job file {} has version {}, but this build reads version {FORMAT_VERSION}",
            self.path.display(),
            file.version
        );
        Ok(file.jobs)
    }

    pub fn get(&self, id: &str) -> anyhow::Result<Option<DeleteJob>> {
        Ok(self.load()?.into_iter().find(|job| job.id == id))
    }

    /// Saves `job`, replacing the job with its id or adding it last.
    pub fn save(&self, job: &DeleteJob) -> anyhow::Result<()> {
        let mut jobs = self.load()?;
        match jobs.iter_mut().find(|saved| saved.id == job.id) {
            Some(saved) => *saved = job.clone(),
            None => jobs.push(job.clone()),
        }
        let bytes = serde_json::to_vec_pretty(&JobFile {
            version: FORMAT_VERSION,
            jobs,
        })?;
        write_atomically(&self.path, &bytes, &mut OsFileOps).map_err(|source| {
            anyhow::anyhow!(
                "could not save the delete job file {}: {source}",
                self.path.display()
            )
        })
    }

    /// Adds the `notices` job `id` does not have yet, so a step that runs
    /// again adds each once.
    pub fn add_notices(&self, id: &str, notices: &[String]) -> anyhow::Result<()> {
        let mut job = self
            .get(id)?
            .ok_or_else(|| anyhow::anyhow!("there is no delete job {id} to add a notice to"))?;
        let before = job.notices.len();
        for notice in notices {
            if !job.notices.contains(notice) {
                job.notices.push(notice.clone());
            }
        }
        if job.notices.len() == before {
            return Ok(());
        }
        self.save(&job)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAYS: u32 = 30;
    const START: i64 = 1_000;
    const WINDOW: i64 = DAYS as i64 * MILLIS_PER_DAY;
    /// 80 percent of the window.
    const APPROACHING_AT: i64 = START + 24 * MILLIS_PER_DAY;

    fn job() -> DeleteJob {
        DeleteJob::new(
            "job-1".into(),
            DeleteTarget::Event {
                event_id: "e".into(),
            },
            START,
            DAYS,
        )
    }

    #[test]
    fn a_notice_is_added_once_and_a_job_saved_without_notices_reads_as_having_none() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = JobStore::new(dir.path().join("jobs.json"));
        store.save(&job()).unwrap();
        let notice = "openai keeps a copy".to_owned();

        // Act
        store
            .add_notices("job-1", std::slice::from_ref(&notice))
            .unwrap();
        store
            .add_notices("job-1", std::slice::from_ref(&notice))
            .unwrap();
        let missing = store
            .add_notices("job-9", std::slice::from_ref(&notice))
            .unwrap_err();
        let mut old: serde_json::Value = serde_json::to_value(job()).unwrap();
        old.as_object_mut().unwrap().remove("notices");
        let read: DeleteJob = serde_json::from_value(old).unwrap();

        // Assert
        assert_eq!(store.get("job-1").unwrap().unwrap().notices, [notice]);
        assert!(missing.to_string().contains("job-9"), "{missing}");
        assert!(read.notices.is_empty());
    }

    #[test]
    fn the_deadline_is_the_window_after_the_request() {
        // Arrange and Act
        let job = job();

        // Assert
        assert_eq!(job.deadline_at, START + WINDOW);
    }

    #[test]
    fn the_alert_changes_at_the_stated_instants() {
        // Arrange
        let job = job();
        let at = |now: i64| job.alert(now);

        // Act and Assert
        assert_eq!(at(START), DeadlineAlert::None);
        assert_eq!(at(APPROACHING_AT - 1), DeadlineAlert::None);
        assert_eq!(at(APPROACHING_AT), DeadlineAlert::Approaching);
        assert_eq!(at(job.deadline_at - 1), DeadlineAlert::Approaching);
        assert_eq!(at(job.deadline_at), DeadlineAlert::Overdue);
        assert_eq!(at(job.deadline_at + 1), DeadlineAlert::Overdue);
    }

    #[test]
    fn a_failed_job_keeps_its_alert_and_a_finished_one_has_none() {
        // Arrange
        let mut failed = job();
        failed.state = JobState::Failed;
        let mut done = job();
        done.state = JobState::Done;
        let mut cancelled = job();
        cancelled.state = JobState::Cancelled;
        let late = job().deadline_at;

        // Act and Assert
        assert_eq!(failed.alert(late), DeadlineAlert::Overdue);
        assert_eq!(done.alert(late), DeadlineAlert::None);
        assert_eq!(cancelled.alert(late), DeadlineAlert::None);
    }

    #[test]
    fn a_saved_job_loads_back_and_a_second_save_replaces_it() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = JobStore::new(job_file_path_for(&dir.path().join("liam.db")));
        let mut first = job();

        // Act
        store.save(&first).unwrap();
        first.state = JobState::Destroying;
        first.key_ids = vec!["k".into()];
        store.save(&first).unwrap();
        let mut second = job();
        second.id = "job-2".into();
        store.save(&second).unwrap();

        // Assert
        let jobs = store.load().unwrap();
        assert_eq!(jobs, [first.clone(), second]);
        assert_eq!(store.get("job-1").unwrap(), Some(first));
        assert_eq!(store.get("missing").unwrap(), None);
    }

    #[test]
    fn a_missing_file_holds_no_jobs() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let store = JobStore::new(job_file_path_for(&dir.path().join("liam.db")));

        // Act
        let jobs = store.load().unwrap();

        // Assert
        assert!(jobs.is_empty());
    }

    #[test]
    fn an_unreadable_or_foreign_file_is_an_error_that_names_it() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = job_file_path_for(&dir.path().join("liam.db"));
        let store = JobStore::new(path.clone());

        // Act
        std::fs::write(&path, "not json").unwrap();
        let broken = store.load().unwrap_err().to_string();
        std::fs::write(&path, r#"{"version":99,"jobs":[]}"#).unwrap();
        let foreign = store.load().unwrap_err().to_string();

        // Assert
        assert!(broken.contains("could not be parsed"), "{broken}");
        assert!(foreign.contains("version 99"), "{foreign}");
    }

    #[test]
    fn a_directory_in_place_of_the_file_is_an_error() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = job_file_path_for(&dir.path().join("liam.db"));
        std::fs::create_dir(&path).unwrap();

        // Act
        let error = JobStore::new(path).load().unwrap_err().to_string();

        // Assert
        assert!(error.contains("could not be read"), "{error}");
    }
}
