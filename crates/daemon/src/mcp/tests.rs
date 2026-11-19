// SPDX-License-Identifier: Apache-2.0
//! Tests of the MCP tools. The doubles and servers they share live here; the
//! tests are split by topic into the modules below.

use super::episode::test_support::{episode_edge, episode_entity, episode_fact};
use super::episode::{EpisodeEntityArgs, EpisodeFactArgs};
use super::test_doubles::CountingEmbedder;
use super::*;
use liam_store::{FixedClock, GraphConfig, Millis, HANDLE_LEN};
use serde_json::json;

mod argument_errors;
mod as_of_and_round_trips;
mod clusters_and_timeline;
mod entity_page_build;
mod entity_pages_and_modes;
mod entity_pages_in_timeline;
mod error_replies;
mod generation_limit;
mod query_limits_and_embed_kinds;
mod recall_filters_and_format;
mod relate;
mod remember;
mod remember_episode;
mod remember_episode_entities;
mod removed_ask;
mod store_swap;
mod timing_samples;

/// Fresh in-memory server wired with the given llm, a 30s synthesis
/// timeout, and the shipped default 8192-token context budget. Dims 8 to
/// match `MockEmbedder::new(8)`.
async fn server_with(llm: Arc<dyn Llm>) -> MemoryServer {
    server_with_generation_limit(llm, 30, 8192, 1).await
}

/// As `server_with`, plus an explicit synthesis timeout, context budget and
/// `max_concurrent_generations`, for the tests that exercise the generation
/// semaphore itself; every other test goes through `server_with`'s fixed 1.
async fn server_with_generation_limit(
    llm: Arc<dyn Llm>,
    synthesis_timeout_secs: u64,
    context_tokens: usize,
    max_concurrent_generations: usize,
) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
        .await
        .expect("open in-memory store");
    MemoryServer::new(
        Arc::new(store),
        Arc::new(liam_model::MockEmbedder::new(8)),
        llm,
        synthesis_timeout_secs,
        context_tokens,
        max_concurrent_generations,
    )
}

/// Returns the written node's id, so a test that needs to name the node
/// (`relate`, or pinning recall's handle) does not have to go around the
/// tool surface to find it.
async fn seed(server: &MemoryServer, kind: &str, label: &str, content: &str) -> String {
    seed_scoped(server, kind, label, content, None).await
}

async fn seed_scoped(
    server: &MemoryServer,
    kind: &str,
    label: &str,
    content: &str,
    scope: Option<&str>,
) -> String {
    let out = server
        .remember(Parameters(RememberArgs {
            kind: kind.to_string(),
            label: label.to_string(),
            content: content.to_string(),
            scope: scope.map(str::to_string),
            subject: None,
            attributes: None,
            valid_from: None,
            confidence: None,
            episode: None,
        }))
        .await;
    assert!(out.starts_with("remembered "), "seed failed: {out}");
    out.trim_start_matches("remembered ").to_string()
}

/// Server with the neutral doubles used by the filter tests: the store's
/// ordering and the echo llm.
async fn plain_server() -> MemoryServer {
    server_with(Arc::new(liam_model::MockLlm)).await
}

/// Server backed by a `FixedClock`-driven store instead of the real
/// clock `plain_server` uses. WHY: the `as_of`
/// tests need to pin exact millisecond instants across two writes, and a
/// real clock on an in-memory store is flaky at that resolution; the
/// clock is handed back so a test can `.set()` it between writes. Same
/// neutral doubles as `plain_server` otherwise.
async fn server_with_clock(clock: Arc<FixedClock>) -> MemoryServer {
    let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock)
        .await
        .expect("open in-memory store");
    MemoryServer::new(
        Arc::new(store),
        Arc::new(liam_model::MockEmbedder::new(8)),
        Arc::new(liam_model::MockLlm),
        30,
        8192,
        1,
    )
}

/// Writes one version of the fixed subject `"price"` for the `as_of`
/// tests: a second call with the clock advanced supersedes the first
/// instead of inserting a competing live node, mirroring
/// `Graph::upsert_by_supersedes_same_subject` at the store layer.
async fn remember_version(server: &MemoryServer, label: &str, content: &str) -> String {
    let out = server
        .remember(Parameters(RememberArgs {
            kind: "fact".to_string(),
            label: label.to_string(),
            content: content.to_string(),
            scope: None,
            subject: Some("price".to_string()),
            attributes: None,
            valid_from: None,
            confidence: None,
            episode: None,
        }))
        .await;
    assert!(out.starts_with("remembered "), "seed failed: {out}");
    out.trim_start_matches("remembered ").to_string()
}

/// Self-cleaning temp database path, unique per call so parallel test
/// binaries and leftovers from a crashed run never collide. Needed only
/// by the producer-stamping test below: every other test in this module
/// uses `:memory:`, but that test asserts through a SECOND connection
/// (`producer` is deliberately absent from `Hit`, so `recall`/`query`
/// cannot see it), and each `:memory:` connection is its own private
/// database, so only a file can be read back from outside the server
/// that wrote it.
fn temp_db_path() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "liam-daemon-mcp-producer-test-{}-{unique}.db",
        std::process::id()
    ))
}

/// Bare-bones args for tests that only exercise one of `attributes`,
/// `valid_from`, or `confidence`; every other field is a fixed, unique
/// content string so a test can find its own node with `Query::text`.
fn remember_args(content: &str) -> RememberArgs {
    RememberArgs {
        kind: "fact".to_string(),
        label: "label".to_string(),
        content: content.to_string(),
        scope: None,
        subject: None,
        attributes: None,
        valid_from: None,
        confidence: None,
        episode: None,
    }
}

/// In-memory server over a recording embedder, plus that embedder.
async fn server_with_recording_embedder() -> (MemoryServer, Arc<CountingEmbedder>) {
    let embedder = Arc::new(CountingEmbedder::new(8));
    (server_with_embedder(embedder.clone()).await, embedder)
}

/// In-memory server over `embedder`, with the neutral llm.
async fn server_with_embedder(embedder: Arc<dyn Embedder>) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
        .await
        .expect("open in-memory store");
    MemoryServer::new(
        Arc::new(store),
        embedder,
        Arc::new(liam_model::MockLlm),
        30,
        8192,
        1,
    )
}

fn recall_args_for(query: &str) -> RecallArgs {
    RecallArgs {
        query: query.to_string(),
        kind: None,
        scope: None,
        k: None,
        as_of: None,
        mode: None,
    }
}

fn recall_args() -> RecallArgs {
    RecallArgs {
        query: "anything".to_string(),
        kind: None,
        scope: None,
        k: None,
        as_of: None,
        mode: None,
    }
}
