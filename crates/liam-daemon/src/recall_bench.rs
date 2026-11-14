// SPDX-License-Identifier: Apache-2.0
//! Recall benchmark: drives the real `MemoryServer` over an isolated store and
//! reports per-stage p50 and p95 latency (read from the metrics sink) and
//! nDCG@10 for a labelled pair file, once on an idle server and once while an
//! `ask` is in flight.
//!
//! The pair file is tab separated, one pair per line: `query<TAB>relevant fact`.
//! Every distinct fact is remembered; the pair's fact is the one relevant
//! result for its query.
//!
//! # Two tiers
//!
//! - **Mock tier, always-on:** `MockEmbedder`, `IdentityReranker` and a gated
//!   fake model, no network and never `~/.liam`. Runs under
//!   `cargo test -p liam-daemon --bin liamd recall_bench`.
//! - **Real tier, `#[ignore]`d:** the local embedder and reranker, downloading
//!   weights on first run. Reads the pair file named by `LIAM_RECALL_BENCH_PAIRS`
//!   (default: a small built-in set) and `LIAM_RECALL_BENCH_RUNS` (default 50),
//!   and the embedder from `LIAM_TOOL_EVAL_MODEL` like `tool_eval`. The store
//!   is isolated in a temporary directory, but the model weights are read from
//!   and may be downloaded into the configured `embedder.cache_dir`;
//!   `LIAM_RECALL_BENCH_CACHE_DIR` overrides that directory:
//!
//!   ```text
//!   cargo test --release -p liam-daemon --bin liamd -- --ignored --nocapture recall_bench
//!   ```
//!
//! The metrics sink keeps the latest 50 samples per stage, so percentiles
//! describe the last 50 recalls of a phase however many runs are asked for.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use liam_daemon::retrieval_eval::ndcg_at_10;
use liam_model::{Embedder, Llm, Reranker};
use liam_store::{DefaultGraph, GraphConfig, SystemClock};
use rmcp::handler::server::wrapper::Parameters;
use tokio::sync::{Notify, Semaphore};

use crate::config::Config;
use crate::mcp::{AskArgs, MemoryServer, RecallArgs, RememberArgs};
use crate::metrics::monotonic::Monotonic;
use crate::metrics::recorder::Metrics;
use crate::metrics::sink::{MemorySink, MetricsSink, Stage, Tool};
use crate::tiering::usage::UsageRecorder;
use crate::tool_eval::labels_in_order;

/// The environment variable that sets the number of runs per phase.
const RUNS_ENV: &str = "LIAM_RECALL_BENCH_RUNS";
/// The environment variable that overrides the model cache directory.
const CACHE_DIR_ENV: &str = "LIAM_RECALL_BENCH_CACHE_DIR";
/// Recalls per phase in the real tier unless `RUNS_ENV` says otherwise.
const DEFAULT_REAL_RUNS: usize = 50;
/// The name of the row for the whole `recall` call, after the stages.
const TOTAL_STAGE: &str = "total";

/// The kind every seeded fact carries.
const FACT_KIND: &str = "fact";
/// The database file inside the store directory.
const DB_FILE: &str = "recall-bench.db";
/// Results requested per recall, the depth nDCG@10 scores.
const RECALL_K: usize = 10;
/// Deadline `MemoryServer::new` requires for `ask`.
const ASK_TIMEOUT_SECS: u64 = 300;
/// How long the in-flight `ask` may take to reach the model before the bench
/// gives up on it.
const ASK_ENTRY_TIMEOUT: Duration = Duration::from_secs(30);
/// Alert minimum sample count; the bench raises no alerts.
const MIN_SAMPLES: usize = 1;

/// What one run of the benchmark needs.
#[derive(Debug, Clone)]
pub struct RecallBenchConfig {
    /// The labelled pair file.
    pub pairs: PathBuf,
    /// How many recalls to run per phase. Zero is an error.
    pub runs: usize,
    /// Where the isolated store is created.
    pub store_dir: PathBuf,
    /// A directory the store must never live under: the real `~/.liam`.
    pub forbidden_root: PathBuf,
}

/// Latency percentiles of one stage, in milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub struct StageLatency {
    pub stage: &'static str,
    pub p50_ms: f64,
    pub p95_ms: f64,
    /// Samples the sink recorded for the stage.
    pub samples: u64,
}

/// The result of one phase of recalls.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseReport {
    pub stages: Vec<StageLatency>,
    pub ndcg_at_10: f64,
}

/// The benchmark result.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallBenchReport {
    pub runs: usize,
    pub store_dir: PathBuf,
    /// Recalls on an otherwise idle server.
    pub idle: PhaseReport,
    /// Recalls while one `ask` is held in flight.
    pub during_ask: PhaseReport,
}

impl RecallBenchReport {
    /// The report as plain text, one line per stage.
    pub fn render(&self) -> String {
        let mut out = format!(
            "recall bench: {} runs per phase, store {}\n",
            self.runs,
            self.store_dir.display()
        );
        for (name, phase) in [("idle", &self.idle), ("during ask", &self.during_ask)] {
            let _ = writeln!(out, "[{name}] nDCG@10 {:.4}", phase.ndcg_at_10);
            for stage in &phase.stages {
                let _ = writeln!(
                    out,
                    "  {:<14} p50 {:>9.3} ms  p95 {:>9.3} ms  ({} samples)",
                    stage.stage, stage.p50_ms, stage.p95_ms, stage.samples
                );
            }
        }
        out
    }
}

struct Pair {
    query: String,
    fact: String,
}

/// A model that parks inside `complete` until released, so an `ask` stays in
/// flight for as long as the bench needs. Releasing closes the gate for good,
/// because `ask` may call the model more than once.
struct GatedLlm {
    entered: Notify,
    gate: Semaphore,
}

#[async_trait::async_trait]
impl Llm for GatedLlm {
    async fn complete(&self, _system: &str, _prompt: &str) -> liam_model::Result<String> {
        self.entered.notify_one();
        let _ = self.gate.acquire().await;
        Ok("released".to_string())
    }
}

/// Runs the benchmark with the given models and timer. The timer measures the
/// stages, so a real run passes `InstantSource` and a test passes a ticking one.
///
/// # Errors
///
/// Fails when `runs` is zero, the store path is under the forbidden root, the
/// pair file cannot be read or parsed, a seed fails, or the `ask` never
/// reaches the model.
pub async fn run_recall_bench(
    config: &RecallBenchConfig,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    timer: Arc<dyn Monotonic>,
) -> anyhow::Result<RecallBenchReport> {
    tracing::debug!(runs = config.runs, "recall bench: validating");
    anyhow::ensure!(
        config.runs >= 1,
        "{RUNS_ENV} must be at least 1, got {}",
        config.runs
    );
    anyhow::ensure!(
        !config.store_dir.starts_with(&config.forbidden_root),
        "refusing to benchmark in {}: it is under {}",
        config.store_dir.display(),
        config.forbidden_root.display()
    );
    let pairs = read_pairs(&config.pairs)?;

    tracing::debug!(pairs = pairs.len(), "recall bench: seeding the store");
    let llm = Arc::new(GatedLlm {
        entered: Notify::new(),
        gate: Semaphore::new(0),
    });
    let server = open_server(&config.store_dir, embedder, reranker, llm.clone()).await?;
    seed(&server, &pairs).await?;

    tracing::debug!("recall bench: idle phase");
    let idle = run_phase(&server, &pairs, config.runs, &timer).await?;

    tracing::debug!("recall bench: phase with an ask in flight");
    let during_ask = run_phase_during_ask(&server, &llm, &pairs, config.runs, &timer).await?;

    Ok(RecallBenchReport {
        runs: config.runs,
        store_dir: config.store_dir.clone(),
        idle,
        during_ask,
    })
}

fn read_pairs(path: &Path) -> anyhow::Result<Vec<Pair>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the pair file {}", path.display()))?;
    let mut pairs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (query, fact) = line.split_once('\t').with_context(|| {
            format!(
                "pair file line {} needs `query<TAB>relevant fact`",
                index + 1
            )
        })?;
        pairs.push(Pair {
            query: query.trim().to_string(),
            fact: fact.trim().to_string(),
        });
    }
    anyhow::ensure!(
        !pairs.is_empty(),
        "the pair file {} has no pairs",
        path.display()
    );
    Ok(pairs)
}

async fn open_server(
    dir: &Path,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
) -> anyhow::Result<MemoryServer> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating the store directory {}", dir.display()))?;
    let db = dir.join(DB_FILE);
    let store = DefaultGraph::open(&db.to_string_lossy(), GraphConfig::new(embedder.dims()))
        .await
        .context("opening the benchmark store")?;
    let defaults = Config::default();
    // Without these the record_usage and issue_handles stages would never run.
    let server = MemoryServer::new(
        Arc::new(store),
        embedder,
        reranker,
        llm,
        ASK_TIMEOUT_SECS,
        defaults.ask_sufficiency_check,
        defaults.llm.context_tokens,
        defaults.llm.max_concurrent_generations,
    );
    Ok(server
        .with_usage_recorder(Arc::new(UsageRecorder::new(
            defaults.tiering.max_tracked_items,
        )))
        .with_feedback(
            defaults.tiering.feedback_per_producer_per_hour,
            defaults.tiering.max_tracked_items,
        ))
}

/// Remembers each distinct fact once, labelled with its own text.
async fn seed(server: &MemoryServer, pairs: &[Pair]) -> anyhow::Result<()> {
    let mut seen = HashMap::new();
    for pair in pairs {
        if seen.insert(pair.fact.as_str(), ()).is_some() {
            continue;
        }
        let reply = server
            .remember(Parameters(RememberArgs {
                kind: FACT_KIND.to_string(),
                label: pair.fact.clone(),
                content: pair.fact.clone(),
                scope: None,
                subject: None,
                attributes: None,
                valid_from: None,
                confidence: None,
                episode: None,
            }))
            .await;
        anyhow::ensure!(reply.starts_with("remembered "), "seeding failed: {reply}");
    }
    Ok(())
}

/// `runs` recalls on a copy of `server` that reports to a fresh sink; scores
/// their mean nDCG@10 and reads each stage's percentiles from that sink.
async fn run_phase(
    server: &MemoryServer,
    pairs: &[Pair],
    runs: usize,
    timer: &Arc<dyn Monotonic>,
) -> anyhow::Result<PhaseReport> {
    let sink = Arc::new(MemorySink::new(Arc::new(SystemClock)));
    let metrics = Arc::new(
        Metrics::new(sink.clone(), Arc::new(SystemClock), MIN_SAMPLES)
            .with_monotonic(timer.clone()),
    );
    let metered = server.clone().with_metrics(metrics);

    let mut total = 0.0;
    for run in 0..runs {
        let pair = &pairs[run % pairs.len()];
        let reply = metered
            .recall(Parameters(RecallArgs {
                query: pair.query.clone(),
                kind: None,
                scope: None,
                k: Some(RECALL_K),
                as_of: None,
                mode: None,
            }))
            .await;
        let ranked = labels_in_order(&reply);
        total += ndcg_at_10(&ranked, &[pair.fact.as_str()].into());
    }
    Ok(PhaseReport {
        stages: read_stages(sink.as_ref()),
        ndcg_at_10: total / runs as f64,
    })
}

/// As `run_phase`, with one `ask` parked inside the model for its duration.
async fn run_phase_during_ask(
    server: &MemoryServer,
    llm: &Arc<GatedLlm>,
    pairs: &[Pair],
    runs: usize,
    timer: &Arc<dyn Monotonic>,
) -> anyhow::Result<PhaseReport> {
    let asker = server.clone();
    let question = pairs[0].query.clone();
    let ask = tokio::spawn(async move {
        asker
            .ask(Parameters(AskArgs {
                question,
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await
    });
    let entered = tokio::time::timeout(ASK_ENTRY_TIMEOUT, llm.entered.notified()).await;

    // The ask is released whatever happened above, so it never outlives the bench.
    let phase = match entered {
        Ok(()) => run_phase(server, pairs, runs, timer).await,
        Err(_) => Err(anyhow::anyhow!(
            "the in-flight ask never reached the model within {ASK_ENTRY_TIMEOUT:?}"
        )),
    };
    llm.gate.close();
    ask.await.context("the in-flight ask panicked")?;
    phase
}

fn read_stages(sink: &MemorySink) -> Vec<StageLatency> {
    let readings = sink.snapshot().readings;
    let find = |metric| readings.iter().find(|reading| reading.metric == metric);
    let stages = Stage::ALL.map(|stage| (stage.name(), stage.metrics()));
    let total = (TOTAL_STAGE, Tool::Recall.metrics());
    stages
        .into_iter()
        .chain([total])
        .map(|(stage, [p50, p95])| StageLatency {
            stage,
            p50_ms: find(p50).map_or(0.0, |reading| reading.value),
            p95_ms: find(p95).map_or(0.0, |reading| reading.value),
            samples: find(p50).map_or(0, |reading| reading.samples),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use crate::metrics::monotonic::{InstantSource, TickingMonotonic};

    const RUNS: usize = 5;
    const PAIRS: &str = "how do we retry safely\tclients send an idempotency key on every retry\n\
        why did login loop\tthe sso callback dropped the session cookie\n\
        which db pool size\tledger-core caps the postgres pool at 40 connections\n";
    const MOCK_DIMS: usize = 8;

    /// A tick under a millisecond, so every stage lasts a nonzero time.
    const TICK: Duration = Duration::from_micros(250);

    fn config(dir: &Path, runs: usize) -> RecallBenchConfig {
        let pairs = dir.join("pairs.tsv");
        std::fs::write(&pairs, PAIRS).unwrap();
        RecallBenchConfig {
            pairs,
            runs,
            store_dir: dir.join("store"),
            forbidden_root: dir.join("fake-home").join(".liam"),
        }
    }

    async fn run(config: &RecallBenchConfig) -> anyhow::Result<RecallBenchReport> {
        run_recall_bench(
            config,
            Arc::new(liam_model::MockEmbedder::new(MOCK_DIMS)),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(TickingMonotonic::new(TICK)),
        )
        .await
    }

    fn stage<'a>(phase: &'a PhaseReport, name: &str) -> &'a StageLatency {
        phase.stages.iter().find(|s| s.stage == name).unwrap()
    }

    #[tokio::test]
    async fn the_report_has_p50_and_p95_for_every_stage_and_an_ndcg() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = config(dir.path(), RUNS);

        // Act
        let report = run(&config).await.unwrap();

        // Assert
        let expected: Vec<&str> = Stage::ALL
            .iter()
            .map(|stage| stage.name())
            .chain([TOTAL_STAGE])
            .collect();
        for phase in [&report.idle, &report.during_ask] {
            let names: Vec<&str> = phase.stages.iter().map(|s| s.stage).collect();
            assert_eq!(names, expected);
            for s in &phase.stages {
                assert!(s.p50_ms > 0.0, "{} p50", s.stage);
                assert!(s.p95_ms >= s.p50_ms, "{} p95 below p50", s.stage);
            }
            assert!(phase.ndcg_at_10 > 0.0 && phase.ndcg_at_10 <= 1.0);
        }
        let text = report.render();
        assert!(text.contains("p50") && text.contains("p95") && text.contains("nDCG@10"));
    }

    #[tokio::test]
    async fn zero_runs_is_a_clear_error() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = config(dir.path(), 0);

        // Act
        let error = run(&config).await.unwrap_err();

        // Assert
        assert!(
            error
                .to_string()
                .contains("LIAM_RECALL_BENCH_RUNS must be at least 1"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn the_store_lives_under_the_injected_dir_and_a_forbidden_root_is_refused() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let allowed = config(dir.path(), RUNS);
        let mut forbidden = config(dir.path(), RUNS);
        forbidden.store_dir = forbidden.forbidden_root.join("store");

        // Act
        let report = run(&allowed).await.unwrap();
        let error = run(&forbidden).await.unwrap_err();

        // Assert
        assert!(report.store_dir.starts_with(dir.path()));
        assert!(report.store_dir.join(DB_FILE).exists());
        assert!(error.to_string().contains("refusing"), "{error:#}");
        assert!(!forbidden.store_dir.exists());
    }

    #[tokio::test]
    async fn record_usage_and_issue_handles_are_sampled() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = config(dir.path(), RUNS);

        // Act
        let report = run(&config).await.unwrap();

        // Assert
        for name in ["record_usage", "issue_handles"] {
            assert!(stage(&report.idle, name).samples > 0, "{name}");
        }
        assert_eq!(stage(&report.idle, "total").samples, RUNS as u64);
    }

    #[tokio::test]
    async fn recalls_still_complete_while_an_ask_is_in_flight() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let config = config(dir.path(), RUNS);

        // Act
        let report = run(&config).await.unwrap();

        // Assert
        assert_eq!(stage(&report.during_ask, "total").samples, RUNS as u64);
        assert!(stage(&report.during_ask, "record_usage").samples > 0);
    }

    /// Real tier: prints the report for the local embedder and reranker. See
    /// the module doc for the run command.
    #[tokio::test]
    #[ignore = "downloads model weights; see module doc for the run command"]
    async fn real_models_report_recall_latency_and_ndcg() {
        // Arrange
        let defaults = Config::default();
        let model_id = std::env::var("LIAM_TOOL_EVAL_MODEL").unwrap_or(defaults.embedder.model);
        let cache_dir = match std::env::var(CACHE_DIR_ENV) {
            Ok(dir) => dir,
            Err(_) => liam_daemon::models::resolve_path_with_home(
                "embedder.cache_dir",
                &defaults.embedder.cache_dir,
                &std::env::var("HOME").unwrap_or_default(),
            )
            .expect("resolve embedder.cache_dir"),
        };
        let embedder: Arc<dyn Embedder> = Arc::new(
            liam_model::FastEmbedEmbedder::load(
                &model_id,
                defaults.embedding_dims,
                &cache_dir,
                liam_model::llm::DevicePreference::Cpu,
            )
            .expect("load real embedder; needs network access for the first download"),
        );
        let reranker: Arc<dyn Reranker> = Arc::new(
            liam_model::FastEmbedReranker::load()
                .expect("load real reranker; needs network access for the first download"),
        );
        let dir = tempfile::tempdir().unwrap();
        let mut config = config(dir.path(), DEFAULT_REAL_RUNS);
        if let Ok(pairs) = std::env::var("LIAM_RECALL_BENCH_PAIRS") {
            config.pairs = PathBuf::from(pairs);
        }
        if let Ok(runs) = std::env::var(RUNS_ENV) {
            config.runs = runs.parse().expect("LIAM_RECALL_BENCH_RUNS is a number");
        }
        config.forbidden_root = Path::new(&std::env::var("HOME").unwrap_or_default()).join(".liam");

        // Act
        let report = run_recall_bench(&config, embedder, reranker, Arc::new(InstantSource::new()))
            .await
            .unwrap();

        // Assert
        println!("{}", report.render());
        assert!(report.idle.ndcg_at_10 > 0.0);
    }
}
