// SPDX-License-Identifier: Apache-2.0
//! What `liamd delete` prints for the job list and for a job named on its own.

use tempfile::TempDir;

use super::*;
use crate::deletion::test_support::sealed_subjects;
use crate::log_startup::test_support::database_in;

const SUBJECT: &str = "alice";
const DEADLINE_DAYS: u32 = 30;
const DAYS_LEFT_WHEN_CLOSE: i64 = 1;
const NOTICE: &str = "a cached summary still names the subject";

fn target() -> DeleteTarget {
    DeleteTarget::Subject {
        subject: SUBJECT.into(),
        scope: None,
        include_children: false,
    }
}

fn status_text(jobs: &[DeleteJob], now: i64) -> String {
    let mut out = Vec::new();
    write_status(jobs, now, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

#[test]
fn an_empty_job_list_says_there_are_no_jobs() {
    // Act
    let text = status_text(&[], 0);

    // Assert
    assert_eq!(text, "no delete jobs\n");
}

#[test]
fn a_job_far_from_its_deadline_carries_no_warning() {
    // Arrange
    let job = DeleteJob::new("job-new".into(), target(), 0, DEADLINE_DAYS);

    // Act
    let text = status_text(&[job], 0);

    // Assert
    assert_eq!(
        text,
        format!("job job-new: Requested, {DEADLINE_DAYS} days to its deadline\n")
    );
}

#[test]
fn a_job_in_the_last_stretch_of_its_window_is_close_to_its_deadline() {
    // Arrange
    let job = DeleteJob::new("job-late".into(), target(), 0, DEADLINE_DAYS);
    let now = job.deadline_at - DAYS_LEFT_WHEN_CLOSE * MILLIS_PER_DAY;

    // Act
    let text = status_text(&[job], now);

    // Assert
    assert_eq!(
        text,
        format!("job job-late: Requested, {DAYS_LEFT_WHEN_CLOSE} days to its deadline, close to its deadline\n")
    );
}

#[test]
fn a_job_past_its_deadline_says_so() {
    // Arrange
    let job = DeleteJob::new("job-over".into(), target(), 0, DEADLINE_DAYS);
    let now = job.deadline_at + MILLIS_PER_DAY;

    // Act
    let text = status_text(&[job], now);

    // Assert
    assert!(text.ends_with(", past its deadline\n"), "{text}");
    assert!(text.contains("-1 days to its deadline"), "{text}");
}

#[test]
fn a_notice_is_listed_under_its_job_without_the_subject() {
    // Arrange
    let mut job = DeleteJob::new("job-notice".into(), target(), 0, DEADLINE_DAYS);
    job.notices.push(NOTICE.into());

    // Act
    let text = status_text(&[job], 0);

    // Assert
    assert!(text.contains(&format!("  notice: {NOTICE}\n")), "{text}");
    assert!(!text.contains(&format!("job-notice: {SUBJECT}")), "{text}");
}

#[test]
fn the_refusal_to_share_a_store_tells_the_operator_what_to_do() {
    // Act
    let text = StoreHeld.to_string();

    // Assert
    assert_eq!(
        text,
        "liamd delete needs the store to itself: stop the daemon, then run it again"
    );
}

async fn sealed() -> (TempDir, Config) {
    let dir = TempDir::new().unwrap();
    let config = sealed_subjects(dir.path()).await.config.clone();
    (dir, config)
}

fn save(dir: &TempDir, job: &DeleteJob) {
    JobStore::new(job_file_path_for(&database_in(dir.path())))
        .save(job)
        .unwrap();
}

#[tokio::test]
async fn resuming_one_job_by_id_finishes_it_and_says_so() {
    // Arrange
    let (dir, config) = sealed().await;
    save(
        &dir,
        &DeleteJob::new("job-one".into(), target(), 0, DEADLINE_DAYS),
    );
    let mut out = Vec::new();

    // Act
    let result = run(
        &config,
        DeleteAction::Resume(Some("job-one".into())),
        &mut out,
    )
    .await;

    // Assert
    result.expect("the job runs");
    assert!(String::from_utf8(out)
        .unwrap()
        .contains("delete job job-one is done"),);
}

#[tokio::test]
async fn resuming_a_job_that_does_not_exist_is_an_error_naming_it() {
    // Arrange
    let (_dir, config) = sealed().await;
    let mut out = Vec::new();

    // Act
    let result = run(
        &config,
        DeleteAction::Resume(Some("job-ghost".into())),
        &mut out,
    )
    .await;

    // Assert
    let error = format!("{:#}", result.expect_err("no such job"));
    assert!(error.contains("job-ghost"), "{error}");
}

#[tokio::test]
async fn a_delete_without_a_key_registry_asks_for_encryption_first() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let config = Config {
        database_path: database_in(dir.path()).to_string_lossy().into_owned(),
        embedding_dims: 8,
        ..Config::default()
    };
    let mut out = Vec::new();

    // Act
    let result = run(&config, DeleteAction::Resume(None), &mut out).await;

    // Assert
    let error = format!("{:#}", result.expect_err("no keys"));
    assert!(error.contains("configure [encryption] first"), "{error}");
}
