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
mod entity_pages_and_modes;
mod entity_synthesis;
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

/// Always-errors `Llm`, so a synthesis must report the failure.
struct FailingLlm;
#[async_trait::async_trait]
impl liam_model::Llm for FailingLlm {
    async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
        Err(liam_model::ModelError::Llm("boom".into()))
    }
}

/// Records what reached the model and replies with a canned answer. WHY not
/// `MockLlm`: its echo repeats the system rules, which the grounding check of
/// entity synthesis rejects as not derived from the mentions. Recording the
/// prompt asserts prompt delivery directly instead of inferring it from an echo.
struct RecordingLlm {
    reply: &'static str,
    seen: std::sync::Mutex<Vec<String>>,
}

impl RecordingLlm {
    fn new(reply: &'static str) -> Self {
        Self {
            reply,
            seen: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn last_prompt(&self) -> String {
        self.seen
            .lock()
            .expect("prompt log")
            .last()
            .cloned()
            .expect("the llm was never called")
    }
}

#[async_trait::async_trait]
impl liam_model::Llm for RecordingLlm {
    async fn complete(&self, system: &str, prompt: &str) -> liam_model::Result<String> {
        self.seen
            .lock()
            .expect("prompt log")
            .push(format!("{system}\n{prompt}"));
        Ok(self.reply.to_string())
    }
}

/// Blocks inside `complete()` on a `Notify` the test controls, so the test
/// decides exactly when a call may finish. Tracks the PEAK number of
/// concurrent `complete()` calls it has ever seen (fetch_add, compare
/// against a running max, fetch_sub on exit), because a final count of 0
/// proves nothing: every call reaches 0 eventually whether or not two of
/// them were ever in flight together.
struct GatedLlm {
    release: Arc<tokio::sync::Notify>,
    in_flight: std::sync::atomic::AtomicUsize,
    peak: std::sync::atomic::AtomicUsize,
}

impl GatedLlm {
    fn new(release: Arc<tokio::sync::Notify>) -> Self {
        Self {
            release,
            in_flight: std::sync::atomic::AtomicUsize::new(0),
            peak: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn in_flight(&self) -> usize {
        self.in_flight.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn peak(&self) -> usize {
        self.peak.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl liam_model::Llm for GatedLlm {
    async fn complete(&self, _s: &str, prompt: &str) -> liam_model::Result<String> {
        let now = self
            .in_flight
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        self.peak
            .fetch_max(now, std::sync::atomic::Ordering::SeqCst);
        self.release.notified().await;
        self.in_flight
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        Ok(grounded_entity_reply(prompt))
    }
}

/// Cooperatively yields until `condition()` is true, bounded so a real bug
/// panics the test instead of hanging the suite. No sleeps: progress here
/// depends only on the single-threaded test runtime getting a chance to
/// poll the other spawned task, which `yield_now` grants deterministically
/// (no wall-clock wait, real or paused).
async fn wait_until(mut condition: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if condition() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("condition never became true; the two tasks likely deadlocked");
}

/// Extracts "(kind) label" from a synthesis prompt's "Entity: (kind) label" line and
/// echoes it back verbatim, which `is_grounded` always accepts (the reply is a subset
/// of its own vocabulary seed) regardless of which entity or how short its label is.
fn grounded_entity_reply(prompt: &str) -> String {
    let line = prompt
        .lines()
        .find(|l| l.starts_with("Entity: ("))
        .expect("synthesis prompt must have an Entity line");
    let rest = line.trim_start_matches("Entity: (");
    let close = rest
        .find(')')
        .expect("Entity line must have a closing paren");
    format!("{} {}", &rest[..close], rest[close + 1..].trim())
}

/// Counts calls and always succeeds with a reply grounded in the
/// entity's own kind/label, for entity synthesis tests that assert on
/// call count.
struct CountingLlm {
    calls: std::sync::atomic::AtomicUsize,
}

impl CountingLlm {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl liam_model::Llm for CountingLlm {
    async fn complete(&self, _s: &str, prompt: &str) -> liam_model::Result<String> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(grounded_entity_reply(prompt))
    }
}

/// Errors on its first call only, succeeds after with a reply grounded
/// in the entity's own kind/label: pins that one entity's failure must
/// not drop another entity's success.
struct FirstCallErrorsLlm {
    calls: std::sync::atomic::AtomicUsize,
}

impl FirstCallErrorsLlm {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl liam_model::Llm for FirstCallErrorsLlm {
    async fn complete(&self, _s: &str, prompt: &str) -> liam_model::Result<String> {
        let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if n == 0 {
            Err(liam_model::ModelError::Llm("boom".into()))
        } else {
            Ok(grounded_entity_reply(prompt))
        }
    }
}

/// Fresh in-memory server wired with the given llm, a 30s synthesis
/// timeout, and the shipped default 8192-token context budget. Dims 8 to
/// match `MockEmbedder::new(8)`.
async fn server_with(llm: Arc<dyn Llm>) -> MemoryServer {
    server_with_timeout(llm, 30, 8192).await
}

/// Fresh in-memory server wired with the given llm/synthesis
/// timeout/context budget, generation capped to 1 concurrent call (the
/// shipped default). Dims 8 to match `MockEmbedder::new(8)`.
async fn server_with_timeout(
    llm: Arc<dyn Llm>,
    synthesis_timeout_secs: u64,
    context_tokens: usize,
) -> MemoryServer {
    server_with_generation_limit(llm, synthesis_timeout_secs, context_tokens, 1).await
}

/// As `server_with_timeout`, plus an explicit `max_concurrent_generations`
/// for the tests in this module that exercise the generation semaphore
/// itself; every other test goes through `server_with_timeout`'s fixed 1.
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
/// clock `plain_server`/`server_with_timeout` use. WHY: the `as_of`
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
    server_with_recording_embedder_and_llm(Arc::new(liam_model::MockLlm)).await
}

async fn server_with_recording_embedder_and_llm(
    llm: Arc<dyn Llm>,
) -> (MemoryServer, Arc<CountingEmbedder>) {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
        .await
        .expect("open in-memory store");
    let embedder = Arc::new(CountingEmbedder::new(8));
    let server = MemoryServer::new(Arc::new(store), embedder.clone(), llm, 30, 8192, 1);
    (server, embedder)
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
