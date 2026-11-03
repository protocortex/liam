// SPDX-License-Identifier: Apache-2.0
//! The delete job end to end over a sealed store: the order of its steps, a
//! stop at every durable effect, and the refusals.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures_util::{stream, StreamExt};
use liam_daemon::consolidation::probe_set::ProbeSet;
use liam_daemon::consolidation::probes::{Probe, ProbeReport};
use liam_daemon::event_log;
use liam_log::crypto::KeyState;
use liam_log::event::LogPayload;
use liam_log::reader::{LogStream, ReaderError};
use liam_log::LogOffset;
use liam_store::{
    Error, EventRole, FixedClock, LineageHits, LineageStore, Millis, NewNode, NodeId, PurgeReport,
    Residual, BEGINNING,
};
use tempfile::TempDir;

use super::*;
use crate::deletion::job::MILLIS_PER_DAY;
use crate::deletion::test_support::{sealed_subjects, Subjects};
use crate::log_startup::test_support::{database_in, log_dir_in};

const START: i64 = 1_000;
/// The stores a job purges when the database has no extraction cache: the
/// database, the dedup filter, the snapshots, the state files, and memory.
const STORES: usize = 5;
const DEADLINE_DAYS: u32 = 30;
const APPROACHING_DAYS: i64 = 24;

struct Env {
    dir: TempDir,
    subjects: Subjects,
    clock: Arc<FixedClock>,
    lock: StoreLock,
}

fn alice() -> DeleteTarget {
    DeleteTarget::Subject {
        subject: "alice".into(),
        scope: None,
        include_children: false,
    }
}

impl Env {
    async fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let subjects = sealed_subjects(dir.path()).await;
        let lock = StoreLock::acquire(&database_in(dir.path())).unwrap();
        Self {
            dir,
            subjects,
            clock: Arc::new(FixedClock::new(Millis(START))),
            lock,
        }
    }

    fn database(&self) -> PathBuf {
        database_in(self.dir.path())
    }

    fn reader(&self) -> Arc<dyn LogReader> {
        event_log::log_reader(&log_dir_in(self.dir.path()), Some(&self.subjects.keys)).unwrap()
    }

    fn parts(&self) -> StoreParts {
        StoreParts {
            store: Arc::clone(&self.subjects.store),
            keys: Arc::clone(&self.subjects.keys),
            reader: self.reader(),
            clock: Arc::clone(&self.clock) as Arc<dyn Clock>,
        }
    }

    fn service(&self) -> DeleteService {
        DeleteService::open(
            &self.subjects.config,
            &self.database(),
            self.parts(),
            Vec::new(),
            &self.lock,
        )
        .unwrap()
    }

    fn deps(&self, registry: LineageRegistry) -> DeleteDeps {
        let parts = self.parts();
        DeleteDeps {
            jobs: self.jobs(),
            store: parts.store,
            keys: parts.keys,
            registry,
            reader: parts.reader,
            probe_set: probe_set_path_for(&self.database()),
            clock: parts.clock,
            deadline_days: DEADLINE_DAYS,
        }
    }

    fn jobs(&self) -> JobStore {
        JobStore::new(job_file_path_for(&self.database()))
    }

    fn only_job(&self) -> DeleteJob {
        let mut jobs = self.jobs().load().unwrap();
        assert_eq!(jobs.len(), 1, "{jobs:?}");
        jobs.remove(0)
    }

    async fn payloads(&self) -> Vec<LogPayload> {
        let mut records = self.reader().scan(None);
        let mut payloads = Vec::new();
        while let Some(record) = records.next().await {
            payloads.push(record.unwrap().event.payload);
        }
        payloads
    }

    async fn count(&self, wanted: fn(&LogPayload) -> bool) -> usize {
        self.payloads().await.iter().filter(|p| wanted(p)).count()
    }

    async fn live_nodes(&self) -> usize {
        let changes = self.subjects.store.changes_since(BEGINNING).await.unwrap();
        changes.iter().filter(|change| !change.closed).count()
    }

    fn key_states(&self, job: &DeleteJob) -> Vec<Option<KeyState>> {
        job.key_ids
            .iter()
            .map(|key| self.subjects.keys.state(key))
            .collect()
    }

    async fn end_state(&self) -> EndState {
        let job = self.only_job();
        EndState {
            state: job.state,
            recorded: self
                .count(|p| matches!(p, LogPayload::DeletionRecorded { .. }))
                .await,
            destroyed: self
                .count(|p| matches!(p, LogPayload::KeyDestroyed { .. }))
                .await,
            removals: self
                .count(|p| matches!(p, LogPayload::LineageRemoval { .. }))
                .await,
            keys: self.key_states(&job),
            nodes: self.live_nodes().await,
        }
    }

    async fn alice_can_write(&self) -> bool {
        let note = NewNode::now("fact", "again", "written later").with_subject("alice");
        self.subjects.store.insert(note).await.is_ok()
    }
}

#[derive(Debug, PartialEq)]
struct EndState {
    state: JobState,
    recorded: usize,
    destroyed: usize,
    removals: usize,
    keys: Vec<Option<KeyState>>,
    nodes: usize,
}

/// Requests and runs a delete of alice, returning how it ended.
async fn delete_alice(service: &DeleteService) -> Result<RunOutcome, DeleteError> {
    let id = service.request(alice())?;
    service.run(&id).await
}

fn crashed_at(result: &Result<RunOutcome, DeleteError>, point: CrashPoint) -> bool {
    matches!(result, Err(DeleteError::Crashed(at)) if *at == point)
}

#[tokio::test]
async fn a_subject_is_deleted_and_nothing_refers_to_it_afterwards() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();

    // Act
    let outcome = delete_alice(&service).await.unwrap();

    // Assert
    let job = env.only_job();
    assert_eq!(outcome, RunOutcome::Done);
    assert_eq!(job.state, JobState::Done);
    assert_eq!(env.key_states(&job), [Some(KeyState::Destroyed)]);
    assert_eq!(env.live_nodes().await, 1, "only bob's note stays");
    let target = env
        .subjects
        .store
        .resolve_target(&LineageTarget {
            job_id: job.id.clone(),
            event_ids: job.event_ids.clone(),
            key_ids: job.key_ids.clone(),
            ..LineageTarget::default()
        })
        .await
        .unwrap();
    assert!(service
        .deps
        .registry
        .verify_all(&target)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn a_job_stopped_at_any_step_resumes_to_the_end_state_of_an_uninterrupted_run() {
    // Arrange
    let baseline = {
        let env = Env::new().await;
        delete_alice(&env.service()).await.unwrap();
        env.end_state().await
    };
    let mut points = vec![
        CrashPoint::AfterRequest,
        CrashPoint::AfterScan,
        CrashPoint::AfterMark,
        CrashPoint::AfterDeletionRecorded,
        CrashPoint::AfterKeyDestroyed(0),
        CrashPoint::AfterKeyRecorded(0),
        CrashPoint::AfterPurge,
        CrashPoint::AfterScrub,
        CrashPoint::BeforeDone,
    ];
    points.extend((0..STORES).map(CrashPoint::AfterStorePurged));

    for point in points {
        let env = Env::new().await;

        // Act
        let stopped = delete_alice(&env.service().stopping_at(point)).await;
        let resumed = env.service().resume_all(false).await.unwrap();

        // Assert
        assert!(crashed_at(&stopped, point), "{point:?}: {stopped:?}");
        assert_eq!(resumed.finished.len(), 1, "{point:?}");
        assert_eq!(env.end_state().await, baseline, "{point:?}");
        assert_eq!(baseline.recorded, 1);
        assert_eq!(baseline.destroyed, 1);
        assert_eq!(baseline.removals, 1);
    }
}

#[tokio::test]
async fn a_group_being_destroyed_refuses_writes_and_no_event_lands() {
    // Arrange
    let env = Env::new().await;
    delete_alice(&env.service().stopping_at(CrashPoint::AfterMark))
        .await
        .unwrap_err();
    let before = env.payloads().await.len();

    // Act
    let note = NewNode::now("fact", "late", "late write").with_subject("alice");
    let result = env.subjects.store.insert(note).await;

    // Assert
    assert!(
        matches!(result, Err(Error::KeyGroupLocked(_))),
        "{result:?}"
    );
    assert_eq!(env.payloads().await.len(), before);
}

#[tokio::test]
async fn a_second_process_cannot_take_the_store_while_the_service_holds_it() {
    // Arrange
    let env = Env::new().await;

    // Act
    let second = StoreLock::acquire(&env.database());

    // Assert
    assert!(second.is_err());
}

#[tokio::test]
async fn a_crash_between_two_keys_finishes_each_key_once() {
    // Arrange
    let env = Env::new().await;
    for scope in ["work", "home"] {
        let note = NewNode::now("fact", scope, "scoped secret")
            .with_subject("alice")
            .with_scope(scope);
        env.subjects.store.insert(note).await.unwrap();
    }
    let everywhere = DeleteTarget::Subject {
        subject: "alice".into(),
        scope: None,
        include_children: true,
    };
    let first = env.service().stopping_at(CrashPoint::AfterKeyRecorded(0));
    let id = first.request(everywhere).unwrap();

    // Act
    let stopped = first.run(&id).await;
    env.service().run(&id).await.unwrap();

    // Assert
    assert!(crashed_at(&stopped, CrashPoint::AfterKeyRecorded(0)));
    let job = env.only_job();
    assert_eq!(job.key_ids.len(), 3);
    assert!(env
        .key_states(&job)
        .iter()
        .all(|state| *state == Some(KeyState::Destroyed)));
    assert_eq!(
        env.count(|p| matches!(p, LogPayload::KeyDestroyed { .. }))
            .await,
        3
    );
}

#[tokio::test]
async fn a_key_the_registry_does_not_know_fails_the_job() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();
    let mut job = DeleteJob::new("job-ghost".into(), alice(), START, DEADLINE_DAYS);
    job.state = JobState::Destroying;
    job.key_ids = vec!["ghost-key".into()];
    env.jobs().save(&job).unwrap();

    // Act
    let result = service.run("job-ghost").await;

    // Assert
    let saved = env.jobs().get("job-ghost").unwrap().unwrap();
    assert!(
        matches!(result, Err(DeleteError::JobFailed { .. })),
        "{result:?}"
    );
    assert_eq!(saved.state, JobState::Failed);
    assert!(saved.failure.unwrap().reason.contains("ghost-key"));
}

#[tokio::test]
async fn cancel_ends_a_job_that_has_not_destroyed_a_key_and_refuses_the_rest() {
    // Arrange
    let cancellable = [
        CrashPoint::AfterRequest,
        CrashPoint::AfterScan,
        CrashPoint::AfterMark,
        CrashPoint::AfterDeletionRecorded,
    ];
    let past_destroy = [
        CrashPoint::AfterKeyDestroyed(0),
        CrashPoint::AfterKeyRecorded(0),
        CrashPoint::AfterPurge,
        CrashPoint::BeforeDone,
    ];

    for point in cancellable {
        let env = Env::new().await;
        delete_alice(&env.service().stopping_at(point))
            .await
            .unwrap_err();
        let id = env.only_job().id;

        // Act
        env.service().cancel(&id).await.unwrap();
        let resumed = env.service().resume_all(false).await.unwrap();

        // Assert
        let job = env.only_job();
        assert_eq!(job.state, JobState::Cancelled, "{point:?}");
        assert_eq!(resumed, ResumeReport::default(), "{point:?}");
        assert!(env
            .key_states(&job)
            .iter()
            .all(|state| *state != Some(KeyState::Destroyed)));
        assert!(env.alice_can_write().await, "{point:?}");
    }

    for point in past_destroy {
        let env = Env::new().await;
        delete_alice(&env.service().stopping_at(point))
            .await
            .unwrap_err();
        let job = env.only_job();
        let before = env.key_states(&job);

        // Act
        let result = env.service().cancel(&job.id).await;

        // Assert
        assert!(
            matches!(result, Err(DeleteError::CancelRefused { .. })),
            "{point:?}: {result:?}"
        );
        assert_eq!(env.key_states(&job), before, "{point:?}");
    }
}

#[tokio::test]
async fn a_final_job_cannot_be_cancelled() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();
    delete_alice(&service).await.unwrap();
    let id = env.only_job().id;

    // Act
    let result = service.cancel(&id).await;

    // Assert
    assert!(matches!(
        result,
        Err(DeleteError::CancelRefused {
            state: JobState::Done,
            ..
        })
    ));
    assert!(matches!(
        service.cancel("missing").await,
        Err(DeleteError::UnknownJob(_))
    ));
}

#[tokio::test]
async fn a_job_that_failed_after_destroying_a_key_cannot_be_cancelled() {
    // Arrange
    let env = Env::new().await;
    let mut job = DeleteJob::new("job-late".into(), alice(), START, DEADLINE_DAYS);
    job.state = JobState::Failed;
    job.destroy_started = true;
    env.jobs().save(&job).unwrap();

    // Act
    let result = env.service().cancel("job-late").await;

    // Assert
    assert!(matches!(
        result,
        Err(DeleteError::CancelRefused {
            state: JobState::Failed,
            ..
        })
    ));
}

#[tokio::test]
async fn a_cancel_after_the_deletion_was_recorded_appends_the_cancellation() {
    // Arrange
    let env = Env::new().await;
    delete_alice(&env.service().stopping_at(CrashPoint::AfterDeletionRecorded))
        .await
        .unwrap_err();
    let id = env.only_job().id;

    // Act
    env.service().cancel(&id).await.unwrap();

    // Assert
    let cancelled = env
        .count(|p| matches!(p, LogPayload::DeletionCancelled { .. }))
        .await;
    assert_eq!(cancelled, 1);
}

/// Serves the failure of an archive that cannot be reached.
struct Unreachable;

impl LogReader for Unreachable {
    fn scan(&self, _from: Option<LogOffset>) -> LogStream {
        Box::pin(stream::iter([Err(ReaderError::PartitionUnreachable {
            first_segment: 1,
            last_segment: 2,
        })]))
    }

    fn scan_through(&self, from: Option<LogOffset>, _through: LogOffset) -> LogStream {
        self.scan(from)
    }
}

#[tokio::test]
async fn an_unreachable_archive_fails_the_job_before_any_key_is_touched() {
    // Arrange
    let env = Env::new().await;
    let registry = LineageRegistry::new().with(Arc::clone(&env.subjects.store));
    let mut deps = env.deps(registry);
    deps.reader = Arc::new(Unreachable);
    let service = DeleteService::new(deps, &env.lock);

    // Act
    let result = delete_alice(&service).await;
    let pending = service.resume_all(false).await.unwrap();
    let id = env.only_job().id;
    service.cancel(&id).await.unwrap();

    // Assert
    assert!(
        matches!(result, Err(DeleteError::JobFailed { .. })),
        "{result:?}"
    );
    assert_eq!(pending.pending, std::slice::from_ref(&id));
    assert_eq!(env.only_job().state, JobState::Cancelled);
    assert!(env.alice_can_write().await);
}

#[tokio::test]
async fn a_failed_job_counts_against_its_deadline_and_runs_again_only_when_asked() {
    // Arrange
    let env = Env::new().await;
    let healthy = Arc::new(AtomicBool::new(false));
    let registry = LineageRegistry::new()
        .with(Arc::clone(&env.subjects.store))
        .with(Flaky(Arc::clone(&healthy)));
    let service = DeleteService::new(env.deps(registry), &env.lock);

    // Act
    let failed = delete_alice(&service).await;
    let job = env.only_job();
    env.clock.set(Millis(job.deadline_at));
    let alerts = service.alerts().unwrap();
    let left_alone = service.resume_all(false).await.unwrap();
    healthy.store(true, Ordering::SeqCst);
    let retried = service.resume_all(true).await.unwrap();

    // Assert
    assert!(
        matches!(failed, Err(DeleteError::JobFailed { .. })),
        "{failed:?}"
    );
    assert_eq!(job.state, JobState::Failed);
    assert_eq!(job.failure.unwrap().at, JobState::Purging);
    assert_eq!(alerts, [(job.id.clone(), DeadlineAlert::Overdue)]);
    assert_eq!(left_alone.pending, std::slice::from_ref(&job.id));
    assert_eq!(retried.finished, [job.id]);
}

/// A store that fails its purge until told it is healthy.
struct Flaky(Arc<AtomicBool>);

#[async_trait]
impl LineageStore for Flaky {
    fn name(&self) -> &str {
        "flaky"
    }

    async fn find(&self, _target: &LineageTarget) -> liam_store::Result<LineageHits> {
        Ok(LineageHits::default())
    }

    async fn purge(&self, _target: &LineageTarget) -> liam_store::Result<PurgeReport> {
        if self.0.load(Ordering::SeqCst) {
            return Ok(PurgeReport::default());
        }
        Err(Error::LineageStore {
            store: "flaky".into(),
            reason: "the store is down".into(),
        })
    }

    async fn verify(&self, _target: &LineageTarget) -> liam_store::Result<Vec<Residual>> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn the_deadline_alert_follows_the_injected_clock() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();
    delete_alice(&service.stopping_at(CrashPoint::AfterScan))
        .await
        .unwrap_err();
    let service = env.service();
    let job = env.only_job();
    let at = |offset: i64| {
        env.clock.set(Millis(job.requested_at + offset));
        service.alerts().unwrap()
    };
    let approaching = APPROACHING_DAYS * MILLIS_PER_DAY;
    let deadline = i64::from(DEADLINE_DAYS) * MILLIS_PER_DAY;

    // Act and Assert
    assert!(at(approaching - 1).is_empty());
    assert_eq!(
        at(approaching),
        [(job.id.clone(), DeadlineAlert::Approaching)]
    );
    assert_eq!(
        at(deadline - 1),
        [(job.id.clone(), DeadlineAlert::Approaching)]
    );
    assert_eq!(at(deadline), [(job.id.clone(), DeadlineAlert::Overdue)]);
}

#[tokio::test]
async fn a_reader_holding_the_database_keeps_the_job_in_scrubbing_until_released() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();
    let database = libsql::Builder::new_local(env.database().to_str().unwrap())
        .build()
        .await
        .unwrap();
    let reader = database.connect().unwrap();
    reader.execute("BEGIN", ()).await.unwrap();
    let mut rows = reader
        .query("SELECT COUNT(*) FROM nodes", ())
        .await
        .unwrap();
    while rows.next().await.unwrap().is_some() {}
    env.subjects
        .store
        .insert(NewNode::now("fact", "later", "after the reader"))
        .await
        .unwrap();

    // Act
    let held = delete_alice(&service).await.unwrap();
    let state_while_held = env.only_job().state;
    env.clock.set(Millis(env.only_job().deadline_at));
    let alert_while_held = service.alerts().unwrap().len();
    drop(rows);
    drop(reader);
    let resumed = service.resume_all(false).await.unwrap();

    // Assert
    assert_eq!(held, RunOutcome::ScrubPending);
    assert_eq!(state_while_held, JobState::Scrubbing);
    assert_eq!(alert_while_held, 1);
    assert_eq!(resumed.finished.len(), 1);
    assert_eq!(env.only_job().state, JobState::Done);
}

#[tokio::test]
async fn a_subject_that_does_not_exist_is_refused_and_makes_no_key() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();
    let before = env.payloads().await.len();
    let target = DeleteTarget::Subject {
        subject: "nobody".into(),
        scope: None,
        include_children: false,
    };

    // Act
    let id = service.request(target).unwrap();
    let result = service.run(&id).await;

    // Assert
    assert!(matches!(
        result,
        Err(DeleteError::Refused(ScanError::NothingToDelete))
    ));
    assert_eq!(env.only_job().state, JobState::Cancelled);
    assert_eq!(env.payloads().await.len(), before);
    assert_eq!(
        service.resume_all(false).await.unwrap(),
        ResumeReport::default()
    );
}

#[tokio::test]
async fn an_event_without_a_subject_is_deleted_alone_and_one_under_a_subject_is_refused() {
    // Arrange
    let env = Env::new().await;
    let store = &env.subjects.store;
    let free = store
        .insert(NewNode::now("fact", "free", "free text"))
        .await
        .unwrap();
    let event_of = |id: NodeId| async move {
        store
            .node_event_ids(&id, EventRole::Origin)
            .await
            .unwrap()
            .remove(0)
            .event_id
    };
    let free_event = event_of(free).await;
    let alice_event = event_of(env.subjects.alice.clone()).await;
    let service = env.service();

    // Act
    let refused_id = service
        .request(DeleteTarget::Event {
            event_id: alice_event,
        })
        .unwrap();
    let refused = service.run(&refused_id).await;
    let id = service
        .request(DeleteTarget::Event {
            event_id: free_event,
        })
        .unwrap();
    let outcome = service.run(&id).await.unwrap();

    // Assert
    assert!(matches!(
        refused,
        Err(DeleteError::Refused(ScanError::EventInSubjectGroup { .. }))
    ));
    assert_eq!(outcome, RunOutcome::Done);
    assert_eq!(env.live_nodes().await, 2, "alice and bob stay");
}

#[tokio::test]
async fn the_nodes_a_probe_expected_leave_the_saved_probe_set() {
    // Arrange
    let env = Env::new().await;
    let alice_id = env.subjects.alice.as_str().to_owned();
    let probe = |name: &str, id: &str| Probe {
        name: name.into(),
        query: name.into(),
        relevant: vec![NodeId::from_raw(id)],
        embedding: None,
    };
    let set = ProbeSet {
        probes: vec![probe("alice", &alice_id), probe("other", "someone-else")],
        baseline: ProbeReport::default(),
    };
    let path = probe_set_path_for(&env.database());
    set.save(&path).unwrap();

    // Act
    delete_alice(&env.service()).await.unwrap();

    // Assert
    let saved = ProbeSet::load(&path).unwrap();
    let names: HashSet<&str> = saved.probes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, HashSet::from(["other"]));
}

#[tokio::test]
async fn a_dry_run_names_what_a_delete_would_destroy_and_changes_nothing() {
    // Arrange
    let env = Env::new().await;
    for scope in ["work", "home"] {
        let note = NewNode::now("fact", scope, "scoped secret")
            .with_subject("alice")
            .with_scope(scope);
        env.subjects.store.insert(note).await.unwrap();
    }
    let service = env.service();
    let tree = |include_children| DeleteTarget::Subject {
        subject: "alice".into(),
        scope: None,
        include_children,
    };
    let before = env.payloads().await.len();

    // Act
    let exact = service.dry_run(&tree(false)).await.unwrap();
    let all = service.dry_run(&tree(true)).await.unwrap();

    // Assert
    assert_eq!(exact.key_ids.len(), 1);
    assert_eq!(all.key_ids.len(), 3);
    assert_eq!(env.payloads().await.len(), before);
    assert!(env.jobs().load().unwrap().is_empty());
}

#[tokio::test]
async fn the_job_deletes_only_the_groups_it_was_asked_about() {
    // Arrange
    let env = Env::new().await;
    let note = NewNode::now("fact", "work", "scoped secret")
        .with_subject("alice")
        .with_scope("work");
    env.subjects.store.insert(note).await.unwrap();

    // Act
    delete_alice(&env.service()).await.unwrap();

    // Assert
    let job = env.only_job();
    assert_eq!(job.key_ids.len(), 1);
    assert_eq!(env.live_nodes().await, 2, "bob and the scoped note stay");
}

#[tokio::test]
async fn a_plain_row_of_the_subject_refuses_the_job_and_destroys_nothing() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let config = crate::log_startup::test_support::config_in(dir.path());
    let started = crate::log_startup::open_store_with_log(
        &config,
        &database_in(dir.path()),
        &log_dir_in(dir.path()),
        crate::log_startup::test_support::embedder(),
        &crate::shutdown::Shutdown::default(),
    )
    .await
    .unwrap();
    let note = NewNode::now("fact", "plain", "plain secret").with_subject("alice");
    started.store.insert(note).await.unwrap();
    let encryption = crate::config::EncryptionConfig {
        master_key: Some(crate::config::MasterKeySource::File),
        key_file: Some(dir.path().join("master.key").to_str().unwrap().to_owned()),
    };
    let keys = event_log::open_key_registry(&log_dir_in(dir.path()), &encryption)
        .unwrap()
        .unwrap();
    let store = Arc::new(started.store);
    let lock = StoreLock::acquire(&database_in(dir.path())).unwrap();
    let parts = StoreParts {
        store,
        keys,
        reader: event_log::log_reader(&log_dir_in(dir.path()), None).unwrap(),
        clock: Arc::new(FixedClock::new(Millis(START))),
    };
    let mut config = config;
    config.encryption = encryption;
    let service =
        DeleteService::open(&config, &database_in(dir.path()), parts, Vec::new(), &lock).unwrap();

    // Act
    let result = delete_alice(&service).await;

    // Assert
    assert!(matches!(
        result,
        Err(DeleteError::Refused(ScanError::PlaintextLogPresent { .. }))
    ));
}

#[tokio::test]
async fn every_error_says_what_failed() {
    // Arrange
    let errors = [
        DeleteError::UnknownJob("job-1".into()),
        DeleteError::CancelRefused {
            job_id: "job-1".into(),
            state: JobState::Done,
        },
        DeleteError::JobFailed {
            job_id: "job-1".into(),
            reason: "the store is down".into(),
        },
        DeleteError::Refused(ScanError::NothingToDelete),
        DeleteError::Step(anyhow::anyhow!("a step broke")),
        DeleteError::JobFile(anyhow::anyhow!("the file broke")),
    ];

    // Act
    let texts: Vec<String> = errors.iter().map(ToString::to_string).collect();

    // Assert
    assert!(texts[0].contains("job-1"));
    assert!(texts[1].contains("can be cancelled"));
    assert!(texts[2].contains("the store is down"));
    assert!(texts[3].contains("refused"));
    assert!(texts[4].contains("a step broke"));
    assert!(texts[5].contains("the file broke"));
}

impl Env {
    async fn resume_pending(&self, keys: bool) -> anyhow::Result<()> {
        super::super::resume_pending(
            &self.subjects.config,
            &self.database(),
            &log_dir_in(self.dir.path()),
            Arc::clone(&self.subjects.store),
            keys.then(|| Arc::clone(&self.subjects.keys)),
            Vec::new(),
            &self.lock,
        )
        .await
    }

    /// A job saved as failed before it did anything, as a store failure leaves it.
    fn save_failed_job(&self, id: &str) {
        let mut job = DeleteJob::new(id.into(), alice(), START, DEADLINE_DAYS);
        job.state = JobState::Failed;
        job.failure = Some(Failure {
            at: JobState::Requested,
            reason: "the store was down".into(),
        });
        self.jobs().save(&job).unwrap();
    }
}

#[tokio::test]
async fn a_start_with_no_open_job_needs_no_key_registry() {
    // Arrange
    let env = Env::new().await;
    delete_alice(&env.service()).await.unwrap();

    // Act
    let result = env.resume_pending(false).await;

    // Assert
    assert!(result.is_ok(), "{result:?}");
}

#[tokio::test]
async fn a_start_finishes_the_job_a_crash_left_and_then_serves() {
    // Arrange
    let env = Env::new().await;
    delete_alice(&env.service().stopping_at(CrashPoint::AfterKeyDestroyed(0)))
        .await
        .unwrap_err();
    let handles = Arc::strong_count(&env.subjects.store);

    // Act
    let result = env.resume_pending(true).await;

    // Assert
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(env.only_job().state, JobState::Done);
    assert_eq!(Arc::strong_count(&env.subjects.store), handles);
}

#[tokio::test]
async fn a_failed_job_refuses_the_start_until_it_is_resumed() {
    // Arrange
    let env = Env::new().await;
    env.save_failed_job("job-failed");

    // Act
    let refused = env.resume_pending(true).await.unwrap_err();
    env.service().run("job-failed").await.unwrap();
    let started = env.resume_pending(true).await;

    // Assert
    let pending = refused.downcast_ref::<DeleteJobPending>().expect("typed");
    assert_eq!(pending.job_ids, ["job-failed"]);
    assert!(refused.to_string().contains("job-failed"));
    assert!(started.is_ok(), "{started:?}");
}

#[tokio::test]
async fn a_scrub_a_reader_holds_refuses_the_start_until_it_is_released() {
    // Arrange
    let env = Env::new().await;
    let database = libsql::Builder::new_local(env.database().to_str().unwrap())
        .build()
        .await
        .unwrap();
    let reader = database.connect().unwrap();
    reader.execute("BEGIN", ()).await.unwrap();
    let mut rows = reader
        .query("SELECT COUNT(*) FROM nodes", ())
        .await
        .unwrap();
    while rows.next().await.unwrap().is_some() {}
    env.subjects
        .store
        .insert(NewNode::now("fact", "later", "after the reader"))
        .await
        .unwrap();
    assert_eq!(
        delete_alice(&env.service()).await.unwrap(),
        RunOutcome::ScrubPending
    );

    // Act
    let refused = env.resume_pending(true).await;
    drop(rows);
    drop(reader);
    let started = env.resume_pending(true).await;

    // Assert
    assert!(refused
        .unwrap_err()
        .downcast_ref::<DeleteJobPending>()
        .is_some());
    assert!(started.is_ok(), "{started:?}");
    assert_eq!(env.only_job().state, JobState::Done);
}

#[tokio::test]
async fn an_open_job_with_no_key_registry_cannot_be_resumed() {
    // Arrange
    let env = Env::new().await;
    env.save_failed_job("job-failed");

    // Act
    let error = env.resume_pending(false).await.unwrap_err();

    // Assert
    assert!(format!("{error:#}").contains("key registry"), "{error:#}");
}
