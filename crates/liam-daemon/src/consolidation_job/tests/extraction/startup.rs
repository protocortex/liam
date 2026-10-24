// SPDX-License-Identifier: Apache-2.0
//! The job as the daemon builds it: what start does and does not do, and a
//! model file that changes between runs.

use super::rules::{mention_facts, with_ada};
use super::*;
use crate::extraction::identity::{hash_file_for, model_identity, GgufFile};
use crate::extraction::provider::pidfile_path;

const ABSENT_SERVER: &str = "/nonexistent/llama-server";

fn started(harness: &Harness, server: &MemoryServer, config: &Config) -> ConsolidationJob {
    ConsolidationJob::new(JobSetup {
        config,
        database_path: &database_in(&harness.root),
        log_dir: &log_dir_in(&harness.root),
        log: harness.log.clone(),
        embedder: embedder(),
        server,
        shutdown: &harness.shutdown,
    })
    .unwrap()
}

fn rules_only(harness: &Harness, enabled: bool) -> Config {
    let mut config = harness.config.clone();
    config.extraction = ExtractionConfig {
        enabled,
        llm_enabled: false,
        min_event_chars: SHORT_EVENT_CHARS,
        ..ExtractionConfig::default()
    };
    config
}

#[tokio::test]
async fn start_clears_a_leftover_pidfile_whether_extraction_is_on_or_off() {
    for enabled in [false, true] {
        // Arrange
        let harness = Harness::new();
        let server = seeded(&harness, 0).await;
        let mut config = rules_only(&harness, enabled);
        config.extraction.llama_server_path = Some(ABSENT_SERVER.into());
        let pidfile = pidfile_path(&database_in(&harness.root));
        std::fs::write(&pidfile, "not a record").unwrap();

        // Act
        started(&harness, &server, &config);

        // Assert
        assert!(!pidfile.exists(), "enabled = {enabled}");
    }
}

#[tokio::test]
async fn start_with_extraction_on_does_not_read_the_model_file() {
    // Arrange
    let harness = Harness::new();
    let server = seeded(&harness, 0).await;
    let model = harness.root.join("model.gguf");
    std::fs::write(&model, b"weights").unwrap();
    let mut config = harness.config.clone();
    config.extraction = ExtractionConfig {
        enabled: true,
        model_gguf: Some(model.to_string_lossy().into_owned()),
        ..ExtractionConfig::default()
    };

    // Act
    started(&harness, &server, &config);

    // Assert
    assert!(!hash_file_for(&model).exists());
}

#[tokio::test]
async fn extraction_runs_only_when_the_config_turns_it_on() {
    for (enabled, facts) in [(true, 1), (false, 0)] {
        // Arrange
        let harness = Harness::new();
        let server = with_ada(&harness, 1).await;
        let job = started(&harness, &server, &rules_only(&harness, enabled));

        // Act
        let outcome = job.tick().await;

        // Assert
        assert_eq!(outcome, Some(RunOutcome::Committed), "enabled = {enabled}");
        let store = server.store().await.unwrap();
        assert_eq!(
            mention_facts(&store).await.len(),
            facts,
            "enabled = {enabled}"
        );
    }
}

#[tokio::test]
async fn a_model_file_replaced_between_runs_is_not_trusted_as_the_model_that_answered() {
    // Arrange: the first run's model double answers as the file's first contents
    let harness = Harness::new();
    let server = seeded(&harness, 1).await;
    let model = harness.root.join("model.gguf");
    std::fs::write(&model, b"first weights").unwrap();
    let first = model_identity(&model).unwrap();
    let provider = ProviderDouble::answering_as(first.clone());
    let mut part = extraction_part(&harness, &provider, config());
    part.identity = Arc::new(GgufFile::new(Some(model.clone())));
    let job = harness.job_with_hooks(&server, honest(), None, hooks_of(part));
    assert_eq!(job.tick().await, Some(RunOutcome::Committed));
    let store = server.store().await.unwrap();
    let facts_before = facts_of(&store).await.len();
    let recorded = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap()
        .extraction_model
        .expect("the model was recorded");
    assert_eq!(recorded.model_sha256, first.model_sha256);
    std::fs::write(&model, b"other weights, other size").unwrap();
    store.insert(employment_note("bob")).await.unwrap();
    drop(store);

    // Act
    let outcome = job.tick().await;

    // Assert: the run names the new file, which the old server does not match
    assert_eq!(outcome, Some(RunOutcome::Committed));
    let manifest = RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap();
    let second = model_identity(&model).unwrap();
    assert_eq!(
        manifest
            .extraction_model
            .clone()
            .map(|model| model.model_sha256),
        Some(second.model_sha256)
    );
    assert!(done(&manifest, "extract:partial"));
    let store = server.store().await.unwrap();
    assert_eq!(facts_of(&store).await.len(), facts_before);
}
