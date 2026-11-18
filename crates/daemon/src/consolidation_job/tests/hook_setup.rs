// SPDX-License-Identifier: Apache-2.0
//! Which hooks a job builds from the config, and what stops it building them.

use liam_daemon::config::Provider;
use liam_store::FixedClock;

use super::*;
use crate::metrics::sink::MemorySink;

const MIN_PAIRS: usize = 2;
const HOSTED: Provider = Provider::Anthropic;

fn metrics() -> Arc<Metrics> {
    let clock = Arc::new(FixedClock::new(Millis(1_000)));
    let sink = Arc::new(MemorySink::new(clock.clone()));
    Arc::new(Metrics::new(sink, clock, MIN_PAIRS))
}

async fn server_for(harness: &Harness) -> MemoryServer {
    let (store, _clock) = harness.store_at(Millis(1_000)).await;
    harness.server(store, Arc::new(liam_model::MockLlm))
}

struct Paths {
    database: PathBuf,
    log_dir: PathBuf,
}

impl Paths {
    fn of(harness: &Harness) -> Self {
        Self {
            database: database_in(&harness.root),
            log_dir: log_dir_in(&harness.root),
        }
    }
}

fn setup<'a>(
    harness: &'a Harness,
    paths: &'a Paths,
    config: &'a Config,
    server: &'a MemoryServer,
    metrics: Option<Arc<Metrics>>,
) -> JobSetup<'a> {
    JobSetup {
        config,
        database_path: &paths.database,
        log_dir: &paths.log_dir,
        log: harness.log.clone(),
        keys: None,
        embedder: embedder(),
        server,
        shutdown: &harness.shutdown,
        metrics,
    }
}

fn refusal(built: anyhow::Result<ConsolidationHookSet>) -> String {
    match built {
        Err(error) => format!("{error:#}"),
        Ok(_) => panic!("the hooks should not have been built"),
    }
}

#[tokio::test]
async fn hosted_extraction_without_a_metrics_recorder_is_refused() {
    // Arrange
    let harness = Harness::new();
    let server = server_for(&harness).await;
    let paths = Paths::of(&harness);
    let mut config = harness.config.clone();
    config.extraction.enabled = true;
    config.extraction.provider = HOSTED;

    // Act
    let message = refusal(hooks_for(&setup(&harness, &paths, &config, &server, None)));

    // Assert
    assert_eq!(message, "hosted extraction needs the metrics recorder");
}

#[tokio::test]
async fn hosted_summaries_without_a_metrics_recorder_are_refused() {
    // Arrange
    let harness = Harness::new();
    let server = server_for(&harness).await;
    let paths = Paths::of(&harness);
    let mut config = harness.config.clone();
    config.summary.enabled = true;
    config.summary.provider = HOSTED;

    // Act
    let message = refusal(hooks_for(&setup(&harness, &paths, &config, &server, None)));

    // Assert
    assert_eq!(message, "hosted summaries need the metrics recorder");
}

#[tokio::test]
async fn the_tiering_hook_exists_only_when_tiering_is_enabled() {
    // Arrange
    let harness = Harness::new();
    let server = server_for(&harness).await;
    let paths = Paths::of(&harness);
    let mut on = harness.config.clone();
    on.tiering.enabled = true;
    let mut off = harness.config.clone();
    off.tiering.enabled = false;

    // Act
    let with_tiering = hooks_for(&setup(&harness, &paths, &on, &server, None)).unwrap();
    let without_tiering = hooks_for(&setup(&harness, &paths, &off, &server, None)).unwrap();

    // Assert
    assert!(with_tiering.tiering.is_some());
    assert!(without_tiering.tiering.is_none());
}

#[tokio::test]
async fn local_summaries_are_built_with_or_without_a_metrics_recorder() {
    // Arrange
    let harness = Harness::new();
    let server = server_for(&harness).await;
    let paths = Paths::of(&harness);
    let mut config = harness.config.clone();
    config.summary.enabled = true;
    config.summary.provider = Provider::Local;

    // Act
    let measured = hooks_for(&setup(&harness, &paths, &config, &server, Some(metrics()))).unwrap();
    let unmeasured = hooks_for(&setup(&harness, &paths, &config, &server, None)).unwrap();

    // Assert
    assert!(measured.summary.is_some());
    assert!(unmeasured.summary.is_some());
}
