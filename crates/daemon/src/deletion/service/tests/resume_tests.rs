// SPDX-License-Identifier: Apache-2.0
//! Resuming every open job, and running a job that is already over.

use super::*;

fn ghost_job(id: &str) -> DeleteJob {
    let mut job = DeleteJob::new(id.into(), alice(), START, DEADLINE_DAYS);
    job.state = JobState::Destroying;
    job.key_ids = vec!["ghost-key".into()];
    job
}

#[tokio::test]
async fn a_job_that_fails_while_resuming_stays_pending_and_the_others_still_run() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();
    let mut finished = DeleteJob::new("job-done".into(), alice(), START, DEADLINE_DAYS);
    finished.state = JobState::Done;
    env.jobs().save(&finished).unwrap();
    env.jobs().save(&ghost_job("job-ghost")).unwrap();

    // Act
    let report = service.resume_all(true).await.unwrap();

    // Assert
    assert_eq!(report.pending, vec!["job-ghost".to_string()]);
    assert!(report.finished.is_empty(), "{report:?}");
    assert_eq!(
        env.jobs().get("job-ghost").unwrap().unwrap().state,
        JobState::Failed
    );
}

#[tokio::test]
async fn running_a_job_that_is_already_over_changes_nothing() {
    // Arrange
    let env = Env::new().await;
    let service = env.service();
    for (id, state) in [
        ("job-done", JobState::Done),
        ("job-cancelled", JobState::Cancelled),
    ] {
        let mut job = DeleteJob::new(id.into(), alice(), START, DEADLINE_DAYS);
        job.state = state;
        env.jobs().save(&job).unwrap();

        // Act
        let outcome = service.run(id).await;

        // Assert
        assert!(matches!(outcome, Ok(RunOutcome::Done)), "{id}: {outcome:?}");
        assert_eq!(env.jobs().get(id).unwrap().unwrap().state, state);
    }
}

#[test]
fn a_simulated_stop_says_where_it_stopped() {
    // Act
    let text = DeleteError::Crashed(CrashPoint::BeforeDone).to_string();

    // Assert
    assert_eq!(text, "stopped at BeforeDone");
}

const RESIDUAL_STORE: &str = "mirror";
const RESIDUAL_KIND: &str = "row";

/// A store that still holds a row of the target after its purge.
struct KeepsARow;

#[async_trait]
impl LineageStore for KeepsARow {
    fn name(&self) -> &str {
        RESIDUAL_STORE
    }

    async fn find(&self, _target: &LineageTarget) -> liam_store::Result<LineageHits> {
        Ok(LineageHits::default())
    }

    async fn purge(&self, _target: &LineageTarget) -> liam_store::Result<PurgeReport> {
        Ok(PurgeReport::default())
    }

    async fn verify(&self, _target: &LineageTarget) -> liam_store::Result<Vec<Residual>> {
        Ok(vec![Residual {
            store: RESIDUAL_STORE.into(),
            kind: RESIDUAL_KIND.into(),
            id: "row-1".into(),
        }])
    }
}

#[tokio::test]
async fn a_row_left_after_the_scrub_fails_the_job_and_names_the_store() {
    // Arrange
    let env = Env::new().await;
    let registry = LineageRegistry::new()
        .with(Arc::clone(&env.subjects.store))
        .with(KeepsARow);
    let service = DeleteService::new(env.deps(registry), &env.lock);

    // Act
    let outcome = delete_alice(&service).await;

    // Assert
    match outcome {
        Err(DeleteError::JobFailed { reason, .. }) => {
            assert!(reason.contains(RESIDUAL_STORE), "{reason}");
            assert!(reason.contains(RESIDUAL_KIND), "{reason}");
        }
        other => panic!("expected a failed job, got {other:?}"),
    }
    assert_eq!(env.only_job().state, JobState::Failed);
}
