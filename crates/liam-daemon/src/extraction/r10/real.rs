// SPDX-License-Identifier: Apache-2.0
//! The R10 tier that runs the real model. It starts the packaged
//! `llama-server` on the pinned GGUF, extracts the whole corpus and holds the
//! result to the committed baseline. It is `#[ignore]`d because it needs the
//! files `liam fetch-models` installs and takes minutes:
//!
//! `cargo test -p liam-daemon real_model -- --ignored --nocapture`
//!
//! A run prints the numbers to put in `baseline.json` when the prompt or the
//! model changes.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use sysinfo::Pid;

use super::baseline::{Baseline, SANITY_FLOOR_F1};
use super::rss::PeakRss;
use super::tiers::{corpus, model_labels, run_pipeline};
use super::{score, score_by_predicate};
use crate::config::ExtractionConfig;
use crate::extraction::identity::{model_identity, GgufFile};
use crate::extraction::prompt::PROMPT_VERSION;
use crate::extraction::provider::{pidfile_path, SidecarProvider, SystemStarter};
use crate::extraction::sidecar::SystemRam;
use liam_daemon::fetch::{default_model_path, default_server_dir, SERVER_BINARY};

const BYTES_PER_MB: u64 = 1 << 20;
const PIDFILE_NAME: &str = "r10.db";

fn installed_files() -> (PathBuf, PathBuf) {
    let home = std::env::var("HOME").expect("HOME is set");
    (
        default_server_dir(&home).join(SERVER_BINARY),
        default_model_path(&home),
    )
}

#[tokio::test]
#[ignore = "needs the model and server from `liam fetch-models`"]
async fn the_real_model_meets_the_committed_gate() {
    // Arrange
    // The run's reasons for stopping early are only in the logs.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init();
    let (server, model) = installed_files();
    let identity = model_identity(&model).expect("the model file is installed");
    let baseline = Baseline::committed();
    if let Err(message) = baseline.check_bound(PROMPT_VERSION, &identity.model_sha256) {
        panic!("{message}");
    }
    let config = ExtractionConfig {
        enabled: true,
        llama_server_path: Some(server.to_string_lossy().into_owned()),
        model_gguf: Some(model.to_string_lossy().into_owned()),
        ..ExtractionConfig::default()
    };
    let scratch = tempfile::TempDir::new().expect("tempdir");
    let provider = SidecarProvider::new(
        Arc::new(SystemStarter {
            pidfile: pidfile_path(&scratch.path().join(PIDFILE_NAME)),
        }),
        Arc::new(GgufFile::new(Some(model))),
    );
    let events = corpus();
    let sampler = PeakRss::start(Pid::from_u32(std::process::id()));

    // Act
    let ram_budget_mb = config.ram_budget_mb;
    let run = run_pipeline(&events, &provider, &identity, config, &SystemRam).await;
    let peak_rss_mb = sampler.stop() / BYTES_PER_MB;

    // Assert
    let extracted = run.extracted(&events).await;
    let overall = score(&extracted, &model_labels(&events));
    let by_predicate = score_by_predicate(&extracted, &model_labels(&events));
    let per_predicate_f1: std::collections::BTreeMap<_, _> = by_predicate
        .iter()
        .map(|(predicate, scored)| (predicate.as_str(), scored.f1))
        .collect();
    let facts_dropped_ungrounded = run.report.facts_dropped_ungrounded;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "prompt_version": PROMPT_VERSION,
            "model_sha256": identity.model_sha256,
            "seed": baseline.seed,
            "measured": {
                "f1": overall.f1,
                "per_predicate_f1": per_predicate_f1,
                "peak_rss_mb": peak_rss_mb,
                "facts_dropped_ungrounded": facts_dropped_ungrounded,
            },
        }))
        .unwrap()
    );
    println!("{:?}", run.report);
    assert!(!run.report.sidecar_lost, "the server died mid-run");
    assert!(!run.report.deadline_hit, "the run hit its deadline");
    assert!(
        overall.f1 >= SANITY_FLOOR_F1,
        "F1 {} is below the sanity floor",
        overall.f1
    );
    assert!(
        overall.f1 >= baseline.gate.min_f1,
        "F1 {} is below the gate {}",
        overall.f1,
        baseline.gate.min_f1
    );
    assert!(
        facts_dropped_ungrounded <= baseline.gate.max_facts_dropped_ungrounded,
        "{facts_dropped_ungrounded} facts were dropped as ungrounded, the ceiling is {}",
        baseline.gate.max_facts_dropped_ungrounded
    );
    // The sampler can miss a short peak, so this is a lower bound on the real one.
    assert!(
        peak_rss_mb < ram_budget_mb,
        "peak RSS {peak_rss_mb} MB is not under the {ram_budget_mb} MB budget"
    );
}
