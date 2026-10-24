// SPDX-License-Identifier: Apache-2.0
//! The offline run over a real database and log in a temp directory.

use std::path::PathBuf;

use liam_model::MockEmbedder;
use liam_store::{Millis, NewNode};

use super::*;

const DIMS: usize = 8;

struct Setup {
    dir: tempfile::TempDir,
    config: Config,
}

impl Setup {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut config = Config {
            database_path: dir.path().join("liam.db").to_str().unwrap().into(),
            embedding_dims: DIMS,
            ..Config::default()
        };
        config.log.dir = Some(dir.path().join("events").to_str().unwrap().into());
        std::fs::write(dir.path().join("liam.toml"), "").unwrap();
        Self { dir, config }
    }

    fn database(&self) -> PathBuf {
        self.dir.path().join("liam.db")
    }

    fn config_path(&self) -> PathBuf {
        self.dir.path().join("liam.toml")
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        Arc::new(MockEmbedder::new(DIMS))
    }

    /// A store with a few facts, written through its log as the daemon would.
    async fn seeded(self) -> Self {
        let log_dir = models::resolve_log_dir(&self.config, &self.database()).unwrap();
        let writer = event_log::open_writer(&log_dir, self.config.log.wal()).unwrap();
        let log = event_log::shared_log(writer, &log_dir, self.config.log.bloom().unwrap(), None)
            .unwrap();
        let pending = embedder_fingerprint::verify(
            &self.database(),
            &EmbedderFingerprint::from_config(&self.config),
        )
        .unwrap();
        let store = open_database(&self.config, &self.database()).await.unwrap();
        let store = open_store_on_log(store, &self.database(), &log_dir, log, self.embedder())
            .await
            .unwrap();
        pending.unwrap().commit().unwrap();
        for i in 0..3 {
            let content = format!("fact number {i}");
            store
                .insert(NewNode::now("fact", &content, &content))
                .await
                .unwrap();
        }
        self
    }

    async fn run(&self) -> anyhow::Result<(RunOutcome, String)> {
        let mut out = Vec::new();
        let outcome = run_offline(
            &self.config,
            &self.config_path(),
            &self.database(),
            self.embedder(),
            &mut out,
        )
        .await?;
        Ok((outcome, String::from_utf8(out).unwrap()))
    }
}

#[tokio::test]
async fn a_run_over_a_stopped_store_commits_and_says_so() {
    // Arrange
    let setup = Setup::new().seeded().await;

    // Act
    let (outcome, printed) = setup.run().await.unwrap();

    // Assert
    assert_eq!(outcome, RunOutcome::Committed);
    assert!(printed.contains("committed"), "{printed}");
    let recorded = RunManifest::load(&manifest_path_for(&setup.database())).unwrap();
    assert_eq!(recorded.unwrap().outcome, Some(RunOutcome::Committed));
}

#[tokio::test]
async fn a_store_a_daemon_holds_is_refused_before_anything_is_written() {
    // Arrange
    let setup = Setup::new().seeded().await;
    let _daemon = StoreLock::acquire(&setup.database()).unwrap();

    // Act
    let error = setup.run().await.unwrap_err();

    // Assert
    assert!(
        error.to_string().contains("needs the store to itself"),
        "{error:#}"
    );
    assert!(!manifest_path_for(&setup.database()).exists());
}

#[tokio::test]
async fn a_missing_config_file_is_refused_so_the_mock_embedder_is_not_used_by_accident() {
    // Arrange
    let setup = Setup::new().seeded().await;
    std::fs::remove_file(setup.config_path()).unwrap();

    // Act
    let error = setup.run().await.unwrap_err();

    // Assert
    assert!(error.to_string().contains("does not exist"), "{error:#}");
}

#[tokio::test]
async fn a_database_with_no_log_is_refused() {
    // Arrange
    let setup = Setup::new();

    // Act
    let error = setup.run().await.unwrap_err();

    // Assert
    assert!(
        error.to_string().contains("is not an event log"),
        "{error:#}"
    );
}

#[tokio::test]
async fn an_unfinished_run_is_resumed_not_restarted() {
    // Arrange
    let setup = Setup::new().seeded().await;
    let config = &setup.config;
    let unfinished = RunManifest::new(
        "01HZZZZZZZZZZZZZZZZZZZZZZZ".to_owned(),
        3,
        identity(config).embedder,
        identity(config).llm_model,
        Millis(1),
    );
    unfinished
        .save(&manifest_path_for(&setup.database()))
        .unwrap();

    // Act
    let (outcome, printed) = setup.run().await.unwrap();

    // Assert
    assert_eq!(outcome, RunOutcome::Committed);
    assert!(printed.contains(&unfinished.run_id), "{printed}");
}

#[tokio::test]
async fn status_reports_no_run_a_finished_run_and_an_unfinished_one() {
    // Arrange
    let setup = Setup::new().seeded().await;
    let mut none = Vec::new();
    print_status(&setup.database(), &mut none).unwrap();
    setup.run().await.unwrap();
    let mut finished = Vec::new();
    print_status(&setup.database(), &mut finished).unwrap();
    let path = manifest_path_for(&setup.database());
    let mut run = RunManifest::load(&path).unwrap().unwrap();
    run.outcome = None;
    run.save(&path).unwrap();

    // Act
    let mut unfinished = Vec::new();
    print_status(&setup.database(), &mut unfinished).unwrap();

    // Assert
    let text = |bytes: Vec<u8>| String::from_utf8(bytes).unwrap();
    assert!(text(none).contains("no consolidation run"));
    assert!(text(finished).contains("committed"));
    let unfinished = text(unfinished);
    assert!(unfinished.contains("unfinished"), "{unfinished}");
    assert!(unfinished.contains("liam consolidate"), "{unfinished}");
}

#[tokio::test]
async fn a_run_is_refused_while_the_last_one_ended_restore_failed() {
    // Arrange
    let setup = Setup::new().seeded().await;
    let mut failed = RunManifest::new(
        "01HZZZZZZZZZZZZZZZZZZZZZZZ".to_owned(),
        3,
        identity(&setup.config).embedder,
        identity(&setup.config).llm_model,
        Millis(1),
    );
    failed.outcome = Some(RunOutcome::RestoreFailed);
    failed.save(&manifest_path_for(&setup.database())).unwrap();

    // Act
    let error = setup.run().await.unwrap_err();

    // Assert
    let message = format!("{error:#}");
    assert!(message.contains("failed restore"), "{message}");
    assert!(message.contains("remove"), "{message}");
    let after = RunManifest::load(&manifest_path_for(&setup.database())).unwrap();
    assert_eq!(after, Some(failed), "no new run replaced the record");
}
