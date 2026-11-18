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
