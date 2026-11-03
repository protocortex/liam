// SPDX-License-Identifier: Apache-2.0
//! The summary part inside whole runs: a scripted model on real database and
//! log files, driven through `tick()`.

use liam_daemon::config::SummaryConfig;
use liam_daemon::consolidation::hooks::ConsolidationHookSet;
use liam_store::{ContentEmbedder, EmbedError};
use tokio::sync::Semaphore;

use super::*;
use crate::extraction::client::ModelIdentity;
use crate::summary::generate::SummaryCache;
use crate::summary::hooks::SummaryHooks;
use crate::summary::summariser::LocalSummariser;
use crate::summary::test_support::{MemoryCache, ScriptedLlm};

const NOTES: usize = 4;
const REPLY: &str = "These ledger notes cover quarterly forecasts for the team.";

struct FixedEmbedder;

#[async_trait]
impl ContentEmbedder for FixedEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut vector = vec![0.0; DIMS];
        vector[0] = 1.0;
        Ok(vector)
    }
}

fn note(n: usize) -> NewNode {
    let mut vector = vec![0.0; DIMS];
    vector[0] = 1.0;
    NewNode::now(
        "fact",
        format!("Note {n}"),
        format!("ledger notes number {n} cover quarterly forecasts for the team"),
    )
    .with_embedding(vector)
}

async fn filled(harness: &Harness) -> MemoryServer {
    let (store, _clock) = harness.store_at(Millis(1000)).await;
    for n in 0..NOTES {
        store.insert(note(n)).await.unwrap();
    }
    harness.server(store, Arc::new(liam_model::MockLlm))
}

fn hooks_over(
    llm: &Arc<ScriptedLlm>,
    cache: &Arc<MemoryCache>,
    config: SummaryConfig,
) -> ConsolidationHookSet {
    ConsolidationHookSet {
        summary: Some(Arc::new(SummaryHooks {
            config,
            summariser: Arc::new(LocalSummariser::new(
                llm.clone(),
                ModelIdentity {
                    model_id: "model".to_string(),
                    model_sha256: String::new(),
                },
            )),
            permits: Arc::new(Semaphore::new(1)),
            cache: cache.clone() as Arc<dyn SummaryCache>,
            embedder: Arc::new(FixedEmbedder),
            context_tokens: 8192,
        })),
        ..Default::default()
    }
}

fn enabled() -> SummaryConfig {
    SummaryConfig {
        enabled: true,
        ..SummaryConfig::default()
    }
}

fn manifest_of(harness: &Harness) -> RunManifest {
    RunManifest::load(&harness.manifest_path())
        .unwrap()
        .unwrap()
}

fn done(manifest: &RunManifest, step: &str) -> bool {
    manifest.completed_steps.iter().any(|done| done == step)
}

async fn live_summaries(server: &MemoryServer) -> usize {
    server
        .store()
        .await
        .unwrap()
        .live_summaries(1, "")
        .await
        .unwrap()
        .len()
}

#[tokio::test]
async fn a_committed_run_keeps_the_summaries_it_wrote() {
    // Arrange
    let harness = Harness::new();
    let server = filled(&harness).await;
    let llm = Arc::new(ScriptedLlm::replying(&[REPLY]));
    let cache = Arc::new(MemoryCache::default());
    let job = harness.job_with_hooks(&server, honest(), None, hooks_over(&llm, &cache, enabled()));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(live_summaries(&server).await, 1);
    let manifest = manifest_of(&harness);
    assert!(done(&manifest, "summarize"));
    assert!(!done(&manifest, "summarize:partial"));
}

#[tokio::test]
async fn a_rolled_back_run_leaves_no_summary_and_the_next_run_reuses_the_cached_text() {
    // Arrange
    let harness = Harness::new();
    let server = filled(&harness).await;
    let llm = Arc::new(ScriptedLlm::replying(&[REPLY]));
    let cache = Arc::new(MemoryCache::default());
    let rolled_back = harness.job_with_hooks(
        &server,
        regresses(),
        None,
        hooks_over(&llm, &cache, enabled()),
    );

    // Act
    let first = rolled_back.tick().await;

    // Assert
    assert_eq!(first, Some(RunOutcome::RolledBack));
    assert_eq!(live_summaries(&server).await, 0);
    assert_eq!(cache.len(), 1, "the response outlives the rollback");

    // Act
    let retry =
        harness.job_with_hooks(&server, honest(), None, hooks_over(&llm, &cache, enabled()));
    let second = retry.tick().await;

    // Assert
    assert_eq!(second, Some(RunOutcome::Committed));
    assert_eq!(live_summaries(&server).await, 1);
    assert_eq!(llm.calls(), 1, "the retry asked the model nothing");
}

#[tokio::test]
async fn a_model_failure_makes_the_step_partial_and_the_run_still_commits() {
    // Arrange
    let harness = Harness::new();
    let server = filled(&harness).await;
    let llm = Arc::new(ScriptedLlm::failing("the model fell over"));
    let cache = Arc::new(MemoryCache::default());
    let job = harness.job_with_hooks(&server, honest(), None, hooks_over(&llm, &cache, enabled()));

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert_eq!(live_summaries(&server).await, 0);
    assert!(done(&manifest_of(&harness), "summarize:partial"));
}

#[tokio::test]
async fn a_run_without_a_summary_part_records_the_step_as_skipped() {
    // Arrange
    let harness = Harness::new();
    let server = filled(&harness).await;
    let job = harness.job(&server, honest());

    // Act
    let outcome = job.tick().await;

    // Assert
    assert_eq!(outcome, Some(RunOutcome::Committed));
    assert!(done(&manifest_of(&harness), "summarize:skipped"));
}

fn hooks_for_config(harness: &Harness, server: &MemoryServer, enabled: bool) -> bool {
    let mut config = harness.config.clone();
    config.summary.enabled = enabled;
    let setup = JobSetup {
        keys: None,
        config: &config,
        database_path: &database_in(&harness.root),
        log_dir: &log_dir_in(&harness.root),
        log: harness.log.clone(),
        embedder: embedder(),
        server,
        shutdown: &harness.shutdown,
        metrics: None,
    };
    hooks_for(&setup).unwrap().summary.is_some()
}

#[tokio::test]
async fn the_daemon_builds_the_summary_part_only_when_the_config_turns_it_on() {
    // Arrange
    let harness = Harness::new();
    let server = filled(&harness).await;

    // Act
    let on = hooks_for_config(&harness, &server, true);
    let off = hooks_for_config(&harness, &server, false);

    // Assert
    assert!(on);
    assert!(!off);
}
