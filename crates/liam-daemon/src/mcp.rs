// SPDX-License-Identifier: Apache-2.0
//! MCP tool surface: `remember`, `recall`, `relate`, `ask`, `clusters`,
//! `timeline`, and `facts`, wiring the store and the model.
//!
//! VERSION CHECK: the rmcp macro surface (`#[tool_router]`, `#[tool]`,
//! `#[tool_handler]`, `Parameters`) moves across releases. Confirmed unchanged
//! from 2.x through rmcp 3.1.2, the version pinned in `Cargo.toml`; re-confirm
//! against the rmcp version you pin before bumping again.

mod admin;
#[cfg(test)]
mod admin_tests;
mod consolidate;
#[cfg(test)]
mod deadline_tests;
mod delete;
pub mod facts;
pub mod feedback;
mod metrics;
#[cfg(test)]
mod metrics_exit_criteria;
#[cfg(test)]
mod metrics_tests;
pub mod producer;
mod reply;
pub mod summaries;
#[cfg(test)]
mod test_doubles;
#[cfg(test)]
mod usage_tests;

use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use liam_daemon::consolidation::shared_store::SharedStore;
use liam_model::{Embedder, Llm, Reranker, TextKind};
use liam_store::types::{EpisodeEdge, EpisodeRef};
use liam_store::{
    relation, DefaultGraph, EventRef, EventRole, Millis, NewNode, NodeId, Query, Vocabulary,
};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerConfig};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use self::delete::DeleteArgs;
use self::episode::{resolved_id, validate_episode, EpisodeArgs};
use self::facts::FactsArgs;
use self::feedback::{Feedback, FeedbackArgs};
use self::metrics::MetricsArgs;
use self::reply::ToolReply;
use self::summaries::SummariesArgs;
use crate::ask::{
    self, build_ask_prompt, clamp_ask_k, estimate_tokens, fallback_answer, fit_evidence_to_budget,
    format_answer,
};
use crate::clusters::{narrow_groups, render_clusters};
use crate::config::{
    resolve_vocabulary, AdminConfig, ExtractionConfig, DEFAULT_MAX_BASIS_IDS_RENDERED,
    DEFAULT_MIN_SUMMARY_SIMILARITY, DEFAULT_RECALL_TIMEOUT_SECS,
};
use crate::consolidation_job::ConsolidationJob;
use crate::degraded::Degraded;
use crate::deletion::DeletePreview;
use crate::metrics::recorder::{Metrics, ToolTimer};
use crate::metrics::sink::{Stage, Tool};
use crate::render;
use crate::synthesis;
use crate::tiering::usage::UsageRecorder;
use crate::tuning::{record_and_maybe_evaluate, AimdHandles, RollingWindow};

/// Runs one store call under a read guard that lasts only for that call, and
/// yields the call's `Result` (or the error of the store being unavailable).
macro_rules! on_store {
    ($server:expr, |$store:ident| $call:expr) => {
        match $server.store().await {
            Ok($store) => $call.await,
            Err(error) => Err(error),
        }
    };
}

// Declared after `on_store!` so the module can use the macro.
pub mod episode;

/// Token budget for the sufficiency pre-pass. Only "YES"/"NO" is wanted, with a
/// little slack for models that add punctuation or a stray word; anything longer
/// is not a verdict and `ask::parse_sufficiency` rejects it anyway.
const SUFFICIENCY_MAX_TOKENS: usize = 8;

/// Producer id a `MemoryServer` reads until something calls
/// `set_producer`. That is every stdio connection
/// (`main.rs` never calls it) and a socket connection during the brief
/// window between `serve` returning and the accept loop's post-handshake
/// `set_producer` call landing (see `producer`'s field doc). Matches both
/// `config::ProducersConfig::default().unknown_id` and
/// `liam_store::NewNode`'s own default, so a server nobody has stamped a
/// producer onto records exactly what it recorded before this field existed.
const DEFAULT_PRODUCER: &str = "unknown";

/// Cap on serialized `attributes` size for `remember`. Deliberately a
/// distinct constant from `ask::MAX_EVIDENCE_CHARS`, not a reuse of it:
/// this bounds write-time input, that bounds a read-time prompt budget.
/// Same value by consistency, not by coupling.
const MAX_ATTRIBUTES_CHARS: usize = 2000;

/// Cap on `content`'s length for `remember` (top-level and every
/// `episode.facts[i]`). A starting point, not measured from production
/// data, matching how `MAX_SCOPE_CHARS` is framed in `liam-store`. The
/// value is kept comfortably under `liam-model`'s `EMBED_MAX_INPUT_TOKENS`
/// (8192 tokens): that cap is the tokenizer's own `max_length`, which
/// silently truncates input rather than erroring, so content passing this
/// check does not silently lose embedding fidelity to an unnoticed
/// truncation downstream.
const MAX_CONTENT_CHARS: usize = 16_000;

/// Cap on `recall`'s `query` and `ask`'s `question`, in characters.
const MAX_QUERY_CHARS: usize = 4000;

/// Cap on distinct words handed to the lexical arm of a query.
const MAX_QUERY_TERMS: usize = 64;

/// Cap on `1 + episode.facts.len() + episode.entities.len() +
/// episode.edges.len()` for `remember`'s `episode` field (the leading `1`
/// is the always-present top-level fact). Bounds how long
/// `Graph::ingest_episode`'s transaction holds the write lock; see the
/// plan's Architecture section, "Cost of the transaction staying open for
/// the whole call."
const MAX_EPISODE_ITEMS: usize = 100;

/// Safety cap on `Graph::mentions` reads per triggered entity: generous
/// relative to `MAX_EPISODE_ITEMS`, not a tuning knob.
const MENTIONS_FETCH_LIMIT: usize = 200;

/// The origin events of `mentions`, each listed once, for the page written from
/// them.
async fn mention_basis(
    store: &DefaultGraph,
    mentions: &[liam_store::Candidate],
) -> liam_store::Result<Vec<EventRef>> {
    let mut basis = std::collections::BTreeSet::new();
    for mention in mentions {
        basis.extend(store.node_event_ids(&mention.id, EventRole::Origin).await?);
    }
    Ok(basis.into_iter().collect())
}

/// Cap on entity-page synthesis output: a short compiled profile, not a
/// full answer. Measured real output was ~55 tokens / ~270 chars.
pub(crate) const ENTITY_SYNTHESIS_MAX_NEW_TOKENS: usize = 256;

/// Default and upper bound on the number of mentions `timeline` shows.
/// WHY: caller-supplied `limit` is otherwise unbounded, and each mention's
/// full content lands in the response, so a large `limit` on a well-mentioned
/// entity would blow the response size. Mirrors `ask`'s `clamp_ask_k`
/// (see `ask.rs`): default keeps `timeline`'s existing behavior when `limit`
/// is omitted, and the cap bounds a single response regardless of how many
/// mentions the entity has.
const DEFAULT_TIMELINE_LIMIT: usize = 50;
const MAX_TIMELINE_LIMIT: usize = 200;

/// Resolve the caller's `limit` into the usable range: absent means
/// `DEFAULT_TIMELINE_LIMIT`, and anything else is clamped to
/// `1..=MAX_TIMELINE_LIMIT`. WHY clamp the bottom too: mirrors `clamp_ask_k`'s
/// reasoning, a `limit = 0` would render no mentions, which reads as a claim
/// that the entity has none rather than a claim about the argument.
fn clamp_timeline_limit(limit: Option<usize>) -> usize {
    limit
        .unwrap_or(DEFAULT_TIMELINE_LIMIT)
        .clamp(1, MAX_TIMELINE_LIMIT)
}

/// Shared embed→build-`Query` sequence for `recall` and `ask`. WHY: both
/// handlers apply the same k/embedding/kind/scope/as_of shape to a `Query`;
/// keeping it in one place means a future filter only needs to change here.
fn build_query(
    text: &str,
    k: Option<usize>,
    kind: Option<String>,
    scope: Option<String>,
    embedding: Option<Vec<f32>>,
    as_of: Option<i64>,
    include_inferred: bool,
) -> Query {
    let mut q = Query::text(text.to_string())
        .with_k(k.unwrap_or(8))
        .with_inferred(include_inferred);
    if let Some(e) = embedding {
        q = q.with_embedding(e);
    }
    if let Some(kind) = kind {
        q = q.with_kind(kind);
    }
    if let Some(scope) = scope {
        q = q.with_scope(scope);
    }
    if let Some(as_of) = as_of {
        q = q.with_as_of(liam_store::Millis(as_of));
    }
    q
}

/// Confidence range check shared by `remember`'s top-level field and every
/// `episode.facts[i]`'s own field: `None` when absent or in `0.0..=1.0`,
/// otherwise the problem, without the `"remember failed: "`/item prefix a
/// caller adds.
fn confidence_problem(confidence: Option<f64>) -> Option<String> {
    let c = confidence?;
    if (0.0..=1.0).contains(&c) {
        None
    } else {
        Some("confidence must be between 0.0 and 1.0".to_string())
    }
}

/// Attributes shape/size check shared by `remember`'s top-level field and
/// every `episode.facts[i]`'s own field: `None` when absent or a JSON object
/// within `MAX_ATTRIBUTES_CHARS`, otherwise the problem.
fn attributes_problem(attributes: &Option<serde_json::Value>) -> Option<String> {
    let v = attributes.as_ref()?;
    if !v.is_object() {
        return Some("attributes must be a JSON object".to_string());
    }
    if v.to_string().chars().count() > MAX_ATTRIBUTES_CHARS {
        return Some(format!(
            "attributes exceeds {MAX_ATTRIBUTES_CHARS} characters"
        ));
    }
    None
}

/// Content size check for `remember`'s top-level `content` field: `None`
/// within `MAX_CONTENT_CHARS`, otherwise the problem. Unlike
/// `attributes_problem`, `content` is a plain `&str`, always present
/// rather than `Option`, since it is a required top-level field. Counts
/// `chars()`, not bytes, so a multi-byte-but-single-scalar character
/// (e.g. `'é'`) is not double-counted against the cap.
fn content_problem(content: &str) -> Option<String> {
    if content.chars().count() > MAX_CONTENT_CHARS {
        Some(format!("content exceeds {MAX_CONTENT_CHARS} characters"))
    } else {
        None
    }
}

/// Validates a `recall` query or `ask` question: `None` when usable.
fn query_problem(query: &str) -> Option<String> {
    if query.trim().is_empty() {
        Some("query is empty".to_string())
    } else if query.contains('\0') {
        Some("query contains a NUL character".to_string())
    } else if query.chars().count() > MAX_QUERY_CHARS {
        Some(format!("query is longer than {MAX_QUERY_CHARS} characters"))
    } else {
        None
    }
}

/// The query text handed to the lexical arm, capped to `MAX_QUERY_TERMS`.
/// Words are distinct by lowercase; the first spelling is kept.
fn lexical_text(query: &str) -> String {
    let mut seen = std::collections::HashSet::new();
    query
        .split_whitespace()
        .filter(|word| seen.insert(word.to_lowercase()))
        .take(MAX_QUERY_TERMS)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Applies `remember`'s five optional per-node fields to an
/// already-constructed `NewNode`, shared by every site that builds a node
/// from `RememberArgs`/`EpisodeFactArgs`: the non-episode path, the
/// episode's top-level fact, and each `episode.facts` entry. Each field is
/// applied only when present, in the same order those call sites already
/// applied it, so the resulting node shape is unchanged for every
/// combination of present/absent fields.
fn apply_optional_fields(
    mut node: NewNode,
    scope: Option<String>,
    attributes: Option<serde_json::Value>,
    valid_from: Option<i64>,
    confidence: Option<f64>,
    subject: Option<String>,
) -> NewNode {
    if let Some(scope) = scope {
        node = node.with_scope(scope);
    }
    if let Some(attributes) = attributes {
        node = node.with_attributes(attributes);
    }
    if let Some(valid_from) = valid_from {
        node = node.with_valid_from(liam_store::Millis(valid_from));
    }
    if let Some(confidence) = confidence {
        node = node.with_confidence(confidence);
    }
    if let Some(subject) = subject {
        node = node.with_subject(subject);
    }
    node
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RememberArgs {
    /// decision | fact | symbol | episode (opaque to the store).
    pub kind: String,
    pub label: String,
    pub content: String,
    /// Optional partition (project, agent). Trimmed; rejected if empty
    /// after trimming, over 200 characters, contains a character outside
    /// ASCII letters, digits, `-`, `_`, `/`, or has a leading/trailing `/`
    /// or an empty segment (`//`).
    pub scope: Option<String>,
    /// Optional identity; a new value with the same subject supersedes the old.
    pub subject: Option<String>,
    /// Optional JSON object of extra fields the store returns verbatim.
    /// Rejected if not an object, or if it serializes past
    /// `MAX_ATTRIBUTES_CHARS`.
    pub attributes: Option<serde_json::Value>,
    /// Optional backdated "true as of" instant, epoch milliseconds. Omitted
    /// means "now"; unlike `confidence` this takes no range check, since any
    /// past or future instant is a meaningful valid time.
    pub valid_from: Option<i64>,
    /// Optional confidence override in `0.0..=1.0`. Omitted defaults to 1.0.
    pub confidence: Option<f64>,
    /// Optional additional facts and the edges between them, written
    /// atomically with the top-level fact above through
    /// `Graph::ingest_episode`. Omitted, `remember` behaves exactly as
    /// before this field existed.
    pub episode: Option<EpisodeArgs>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallArgs {
    pub query: String,
    pub kind: Option<String>,
    /// Optional partition filter; matches memories written with this scope
    /// or any descendant scope (for example `work` also matches
    /// `work/liam`, but not `work-archive`). Trimmed; rejected if empty
    /// after trimming, over 200 characters, contains a character outside
    /// ASCII letters, digits, `-`, `_`, `/`, or has a leading/trailing `/`
    /// or an empty segment (`//`).
    pub scope: Option<String>,
    pub k: Option<usize>,
    /// Optional point-in-time recall, epoch milliseconds. Omitted means
    /// "now"; a past instant surfaces the version of each subject that was
    /// live then.
    pub as_of: Option<i64>,
    /// Only `strict` is supported: recall returns recorded memory.
    pub mode: Option<Mode>,
}

/// What a read may return. Strict returns only what a client recorded; infer
/// may also return what a model composed from it, each item labelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Strict,
    Infer,
}

/// A tool that only reads recorded memory refuses `mode = infer`.
fn require_strict(tool: &str, mode: Option<Mode>) -> Result<(), String> {
    match mode {
        Some(Mode::Infer) => Err(format!(
            "{tool} failed: mode infer is not supported, {tool} returns recorded memory only"
        )),
        _ => Ok(()),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RelateArgs {
    /// Handle of the source memory, as shown by `recall`. A full id works too.
    pub from: String,
    /// Handle of the target memory, as shown by `recall`. A full id works too.
    pub to: String,
    /// Relation type, for example `mentions`. `supersedes` is reserved.
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AskArgs {
    pub question: String,
    pub kind: Option<String>,
    /// Optional partition filter; matches memories written with this scope
    /// or any descendant scope (for example `work` also matches
    /// `work/liam`, but not `work-archive`). Trimmed; rejected if empty
    /// after trimming, over 200 characters, contains a character outside
    /// ASCII letters, digits, `-`, `_`, `/`, or has a leading/trailing `/`
    /// or an empty segment (`//`).
    pub scope: Option<String>,
    pub k: Option<usize>,
    /// Optional point-in-time recall, epoch milliseconds. Omitted means
    /// "now"; a past instant retrieves the version of each subject that was
    /// live then.
    pub as_of: Option<i64>,
    /// `infer` (the default) may compose the answer from entity pages and
    /// summaries as well as recorded memory and labels what it used. `strict`
    /// returns recorded evidence only and makes no model call.
    pub mode: Option<Mode>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ClustersArgs {
    /// Number of clusters to show, largest first. Narrows within the token
    /// budget; cannot widen past it.
    pub k: Option<usize>,
    /// Number of memories to show per cluster. Narrows within the token
    /// budget; cannot widen past it.
    pub members: Option<usize>,
    /// Only `strict` is supported: clusters list recorded memory.
    pub mode: Option<Mode>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TimelineArgs {
    /// Handle of the entity, as shown by `recall`. A full id works too.
    pub entity: String,
    /// How many mentions to show, most recent first. Absent defaults to
    /// `DEFAULT_TIMELINE_LIMIT` (50); anything above `MAX_TIMELINE_LIMIT`
    /// (200) is capped, not rejected.
    pub limit: Option<usize>,
    /// Only `strict` is supported: the entity page is shown labelled inferred,
    /// and the mentions are recorded memory.
    pub mode: Option<Mode>,
}

#[derive(Clone)]
pub struct MemoryServer {
    /// Shared with the consolidation job, whose rollback swaps the store in the
    /// slot: reach it through `store()`, never keeping the guard across calls.
    store: Arc<SharedStore>,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
    /// Wall-clock deadline for the `ask` stages after retrieval (permit acquire,
    /// sufficiency pre-pass, synthesis); `synthesize_entity` shares it too.
    /// Retrieval has its own `recall_timeout_secs` budget.
    ask_timeout_secs: u64,
    /// Seconds `recall` and `ask` may spend retrieving; see `retrieval_deadline`.
    recall_timeout_secs: u64,
    /// Whether `ask` runs the yes/no sufficiency pre-pass before synthesizing;
    /// see `config::Config::ask_sufficiency_check`.
    ask_sufficiency_check: bool,
    /// Token budget `ask` trims retrieved evidence to before either prompt is
    /// built; see `config::LlmConfig::context_tokens`, which this must match so
    /// a full-size prompt never overflows the model's own context window.
    /// `clusters` reads this too, as one tenth of it (ADR-0002 S4): same
    /// underlying config value, not an `ask`-only budget despite the name.
    ask_context_tokens: usize,
    /// Event ids a provenance line lists before it counts the rest; see
    /// `config::SummaryConfig::max_basis_ids_rendered`.
    max_basis_ids_rendered: usize,
    /// Least similarity of a question to a summary for `ask` to use it; see
    /// `config::SummaryConfig::min_summary_similarity`.
    min_summary_similarity: f32,
    /// Bounds how many `ask` calls may be inside a model call (sufficiency
    /// pre-pass or synthesis) at once; see
    /// `config::LlmConfig::max_concurrent_generations`. A semaphore, not a
    /// lock, so an operator with memory headroom can raise the limit above 1
    /// instead of every request serializing permanently.
    generation_permits: Arc<Semaphore>,
    /// AIMD latency samples from every `ask` and entity-synthesis call,
    /// shared process-wide: `Arc`-wrapped since `MemoryServer` clones per connection.
    rolling_window: Arc<Mutex<RollingWindow>>,
    /// Authoritative total permit count AIMD grows/shrinks against, distinct
    /// from `generation_permits.available_permits()`.
    granted_capacity: Arc<AtomicUsize>,
    /// The single shrink slot: `Some` while one permit is held back.
    held_permit: Arc<Mutex<Option<OwnedSemaphorePermit>>>,
    /// Single-flight guard so only one AIMD evaluation runs at a time.
    evaluating: Arc<AtomicBool>,
    /// The prior evaluation's window average, `None` until a second
    /// evaluation has one to compare against.
    previous_window_average: Arc<Mutex<Option<Duration>>>,
    /// Producer id stamped on every node this connection's `remember` calls
    /// write. Per-CONNECTION, unlike every field above it, which is set once
    /// for the process's whole lifetime: that mismatch is why this is not a
    /// ninth constructor argument. Instead the accept loop clones an
    /// already-constructed `MemoryServer` per connection (see
    /// `#[derive(Clone)]` above) and calls `set_producer` on the clone.
    ///
    /// An `OnceLock`, not a plain `String`, because of WHEN the accept loop
    /// (WU-8) learns which producer to stamp: only after the MCP
    /// `initialize` handshake completes, which happens inside `rmcp`'s
    /// `serve`. By then this `MemoryServer` is already moved into the `Arc`
    /// backing the running session and reachable only as `&MemoryServer`
    /// (`RunningService::service`), never owned again, so the mutation has
    /// to go through `&self`; `OnceLock::set` is a `&self` method for
    /// exactly that reason. Each connection's clone gets its OWN,
    /// independently empty cell, which is the same per-connection isolation
    /// every other `Clone` field here already gets, just via a different
    /// mechanism.
    ///
    /// INVARIANT, and it is load-bearing: the template `MemoryServer` handed
    /// to `accept_loop` must never be stamped. `OnceLock`'s `Clone` yields a
    /// fresh unset cell only when the source is unset; cloning an
    /// already-set lock COPIES the value, and the clone's own `set` then
    /// fails. Stamp the template and every connection silently inherits that
    /// one producer, so every write is misattributed. `set_producer` logs on
    /// a failed set so that mistake surfaces instead of passing silently.
    ///
    /// Reads (`remember`, through the `producer` method below) fall back to
    /// `DEFAULT_PRODUCER` while this is unset: every stdio connection
    /// forever, and a socket connection during the brief window between
    /// `serve` returning and the accept loop's `set_producer` call landing.
    producer: OnceLock<String>,
    /// Cancelled when the daemon is stopping, so the maintenance tick stops
    /// at its next step instead of starting work no one will wait for.
    shutdown: CancellationToken,
    /// Tells `facts` which predicates hold one value, so it can mark competing
    /// claims. Shared across connections like the other process-wide state.
    vocabulary: Arc<Vocabulary>,
    /// Counts the results `recall` and `ask` return, when tiering is on.
    usage: Option<Arc<UsageRecorder>>,
    /// The verdict budget and this connection's issued handles, when tiering
    /// is on. A clone starts with no handles issued.
    feedback: Option<Feedback>,
    /// Which producers may call `delete` and `consolidate`. Empty refuses all.
    admin: Arc<AdminConfig>,
    /// What `delete` asks, when the log is sealed.
    delete_preview: Option<Arc<DeletePreview>>,
    /// What `consolidate` runs.
    consolidator: Option<Arc<ConsolidationJob>>,
    /// Where `recall` and `ask` report their timings, when metrics are on.
    metrics: Option<Arc<Metrics>>,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}

/// RAII guard: records `ask`'s permit-held duration on drop, covering every
/// one of its early-return branches without a record call at each site.
struct PermitTimer {
    window: Arc<Mutex<RollingWindow>>,
    aimd: AimdHandles,
    queue_wait: Duration,
    start: tokio::time::Instant,
    context_tokens: usize,
}

impl PermitTimer {
    fn new(
        window: Arc<Mutex<RollingWindow>>,
        aimd: AimdHandles,
        queue_wait: Duration,
        context_tokens: usize,
    ) -> Self {
        Self {
            window,
            aimd,
            queue_wait,
            start: tokio::time::Instant::now(),
            context_tokens,
        }
    }
}

impl Drop for PermitTimer {
    fn drop(&mut self) {
        record_and_maybe_evaluate(
            &self.window,
            self.aimd.clone(),
            self.queue_wait,
            self.start.elapsed(),
            self.context_tokens,
        );
    }
}

/// Least time worth giving the reranker; below it retrieval keeps store order.
const MIN_RERANK_BUDGET: Duration = Duration::from_millis(200);

/// Why a retrieval stage stopped before it finished.
enum Interrupted {
    Deadline,
    Cancelled,
}

fn cancelled_reply(tool: &str) -> String {
    format!("{tool} cancelled")
}

/// Races `work` against the deadline and the caller's cancel token.
async fn until_interrupted<T>(
    deadline: tokio::time::Instant,
    cancel: &CancellationToken,
    work: impl std::future::Future<Output = T>,
) -> Result<T, Interrupted> {
    tokio::select! {
        out = work => Ok(out),
        _ = tokio::time::sleep_until(deadline) => Err(Interrupted::Deadline),
        _ = cancel.cancelled() => Err(Interrupted::Cancelled),
    }
}

impl MemoryServer {
    /// The instant retrieval must finish by.
    fn retrieval_deadline(&self) -> tokio::time::Instant {
        tokio::time::Instant::now() + Duration::from_secs(self.recall_timeout_secs)
    }

    /// Counts each reason of a degraded reply. A failure or a cancel carries
    /// none, so it counts nothing.
    fn count_degraded(&self, tool: Tool, reply: &ToolReply) {
        let Some(metrics) = &self.metrics else { return };
        for reason in reply.degraded() {
            metrics.record_degraded(tool, *reason);
        }
    }

    fn interruption_reply(&self, tool: &str, why: Interrupted) -> ToolReply {
        match why {
            Interrupted::Deadline => ToolReply::failed(format!(
                "{tool} failed: retrieval took longer than {} s",
                self.recall_timeout_secs
            )),
            Interrupted::Cancelled => ToolReply::ok(cancelled_reply(tool)),
        }
    }

    /// Reranks `docs`. When the reranker fails, is skipped for lack of time,
    /// or outlasts the deadline, returns store order and why. `Err` is the cancel reply.
    async fn rerank_within(
        &self,
        tool: &str,
        query: &str,
        docs: &[String],
        deadline: tokio::time::Instant,
        cancel: &CancellationToken,
    ) -> Result<(Vec<usize>, Option<Degraded>), ToolReply> {
        let store_order = || (0..docs.len()).collect();
        if deadline.saturating_duration_since(tokio::time::Instant::now()) < MIN_RERANK_BUDGET {
            tracing::warn!(
                tool,
                "too little retrieval time left, skipping the reranker"
            );
            return Ok((store_order(), Some(Degraded::RerankSkipped)));
        }
        let started = self.stage_start();
        match until_interrupted(deadline, cancel, self.reranker.order(query, docs)).await {
            Ok(Ok(order)) => {
                self.stage_end(Stage::Rerank, started);
                Ok((order, None))
            }
            Ok(Err(e)) => {
                tracing::warn!(tool, error = %e, "answered from unreranked evidence");
                Ok((store_order(), Some(Degraded::RerankFailed)))
            }
            Err(Interrupted::Deadline) => {
                tracing::warn!(tool, "reranker outlasted the retrieval deadline");
                Ok((store_order(), Some(Degraded::RerankTimeout)))
            }
            Err(Interrupted::Cancelled) => Err(ToolReply::ok(cancelled_reply(tool))),
        }
    }
}

/// Adds each present reason to `reply`, in order.
fn with_reasons(
    reply: ToolReply,
    reasons: impl IntoIterator<Item = Option<Degraded>>,
) -> ToolReply {
    reasons
        .into_iter()
        .flatten()
        .fold(reply, ToolReply::with_degraded)
}

#[tool_router]
impl MemoryServer {
    // Each argument is a distinct config field threaded straight through, the
    // established pattern in this constructor (see `ask_timeout_secs`,
    // `ask_sufficiency_check`, `ask_context_tokens` above); a config struct
    // parameter would hide which fields `MemoryServer` actually depends on.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<DefaultGraph>,
        embedder: Arc<dyn Embedder>,
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn Llm>,
        ask_timeout_secs: u64,
        ask_sufficiency_check: bool,
        ask_context_tokens: usize,
        max_concurrent_generations: usize,
    ) -> Self {
        // `Semaphore::new(0)` would deadlock every `ask` call forever. `0` means
        // auto-tune (see `main.rs::build_autotuned_server`); this floor only
        // matters for a direct explicit-`0` construction that bypasses it.
        let max_concurrent_generations = max_concurrent_generations.max(1);
        Self {
            store: Arc::new(SharedStore::from_arc(store)),
            embedder,
            reranker,
            llm,
            ask_timeout_secs,
            recall_timeout_secs: DEFAULT_RECALL_TIMEOUT_SECS,
            ask_sufficiency_check,
            ask_context_tokens,
            max_basis_ids_rendered: DEFAULT_MAX_BASIS_IDS_RENDERED,
            min_summary_similarity: DEFAULT_MIN_SUMMARY_SIMILARITY,
            generation_permits: Arc::new(Semaphore::new(max_concurrent_generations)),
            rolling_window: Arc::new(Mutex::new(RollingWindow::new())),
            granted_capacity: Arc::new(AtomicUsize::new(max_concurrent_generations)),
            held_permit: Arc::new(Mutex::new(None)),
            evaluating: Arc::new(AtomicBool::new(false)),
            previous_window_average: Arc::new(Mutex::new(None)),
            producer: OnceLock::new(),
            shutdown: CancellationToken::new(),
            vocabulary: Arc::new(resolve_vocabulary(&ExtractionConfig::default())),
            usage: None,
            feedback: None,
            admin: Arc::new(AdminConfig::default()),
            delete_preview: None,
            consolidator: None,
            metrics: None,
            tool_router: Self::tool_router(),
        }
    }

    /// Ties this server to the daemon's shutdown token.
    pub(crate) fn with_shutdown(mut self, shutdown: CancellationToken) -> Self {
        self.shutdown = shutdown;
        self
    }

    /// Sets the seconds `recall` and `ask` may spend retrieving. Stored as at
    /// least one, so a configured `0` cannot time every request out at once.
    pub(crate) fn with_recall_timeout(mut self, secs: u64) -> Self {
        self.recall_timeout_secs = secs.max(1);
        self
    }

    /// Lists at most `max` event ids on a provenance line.
    pub(crate) fn with_max_basis_ids_rendered(mut self, max: usize) -> Self {
        self.max_basis_ids_rendered = max;
        self
    }

    /// Uses summaries at least this close to a question in `ask`.
    pub(crate) fn with_min_summary_similarity(mut self, min: f32) -> Self {
        self.min_summary_similarity = min;
        self
    }

    /// Uses the operator's resolved predicate vocabulary for `facts`.
    pub(crate) fn with_vocabulary(mut self, vocabulary: Vocabulary) -> Self {
        self.vocabulary = Arc::new(vocabulary);
        self
    }

    /// Counts the results of `recall` and `ask` in `recorder`.
    pub(crate) fn with_usage_recorder(mut self, recorder: Arc<UsageRecorder>) -> Self {
        self.usage = Some(recorder);
        self
    }

    /// Accepts verdicts through `feedback`, `per_hour` from each producer, on
    /// up to `max_issued` results shown to a connection.
    pub(crate) fn with_feedback(mut self, per_hour: u32, max_issued: usize) -> Self {
        self.feedback = Some(Feedback::new(per_hour, max_issued));
        self
    }

    /// Lets the producers `admin` lists call `delete` and `consolidate`.
    pub(crate) fn with_admin(mut self, admin: AdminConfig) -> Self {
        self.admin = Arc::new(admin);
        self
    }

    /// Answers `delete` from `preview`.
    pub(crate) fn with_delete_preview(mut self, preview: Arc<DeletePreview>) -> Self {
        self.delete_preview = Some(preview);
        self
    }

    /// Runs `consolidate` on `job`.
    pub(crate) fn with_consolidator(mut self, job: Arc<ConsolidationJob>) -> Self {
        self.consolidator = Some(job);
        self
    }

    /// Reports the timings of `recall` and `ask` to `metrics`.
    pub(crate) fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    fn time(&self, tool: Tool) -> Option<ToolTimer> {
        self.metrics.as_ref().map(|metrics| metrics.time(tool))
    }

    fn stage_start(&self) -> Option<Millis> {
        self.metrics.as_ref().map(|metrics| metrics.now())
    }

    fn stage_end(&self, stage: Stage, started: Option<Millis>) {
        if let (Some(metrics), Some(started)) = (&self.metrics, started) {
            metrics.record_stage(stage, started);
        }
    }

    /// Runs `q` and reports how long each retrieval arm took.
    async fn explained(
        &self,
        q: &Query,
    ) -> Result<Vec<liam_store::ExplainedHit>, liam_store::Error> {
        let started = self.stage_start();
        let searched = on_store!(self, |store| store.query_timed(q));
        self.stage_end(Stage::StoreSearch, started);
        let (hits, timings) = searched?;
        if let Some(metrics) = &self.metrics {
            metrics.record_arms(timings);
        }
        Ok(hits)
    }

    /// Makes the results a client is shown ones it can give a verdict on.
    fn issue_handles<'a>(&self, nodes: impl IntoIterator<Item = &'a NodeId>) {
        let started = self.stage_start();
        if let Some(feedback) = &self.feedback {
            feedback.issue(nodes);
        }
        self.stage_end(Stage::IssueHandles, started);
    }

    /// Counts one hit on each node a client was given as a ranked result.
    /// Neighbours pulled in beside a result are not credited.
    fn record_usage<'a>(&self, nodes: impl IntoIterator<Item = &'a NodeId>) {
        let started = self.stage_start();
        if let Some(recorder) = &self.usage {
            recorder.record(&self.producer(), nodes, Millis::now());
        }
        self.stage_end(Stage::RecordUsage, started);
    }

    fn record_evidence_usage(&self, evidence: &[ask::Evidence]) {
        self.record_usage(
            evidence
                .iter()
                .filter(|e| !e.expanded)
                .filter_map(|e| e.node_id.as_ref()),
        );
    }

    pub(crate) fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Stamps `id` as this connection's producer through `&self`. The
    /// `&self` receiver, not `self`, is the reason `producer` is an
    /// `OnceLock` rather than a plain `String` at all: the accept loop
    /// (WU-8) only learns a connecting client's declared name once `rmcp`'s
    /// `serve` returns a `RunningService`, and by then the `MemoryServer`
    /// driving that connection is reachable only as `&MemoryServer`
    /// (`RunningService::service`). `OnceLock::set` is a `&self` method for
    /// exactly this reason.
    ///
    /// Warns rather than panicking if `producer` is already set. Every
    /// caller stamps at most once per instance, so a failed `set` means the
    /// template `MemoryServer` was stamped before `accept_loop` cloned it
    /// (see the field doc's invariant), and every connection is now writing
    /// under one inherited producer. That is a caller bug worth seeing in
    /// the log: it runs once per connection, never on a request path, so
    /// the check costs nothing measurable, and staying silent would make
    /// wholesale misattribution look exactly like correct operation.
    pub(crate) fn set_producer(&self, id: impl Into<String>) {
        let id = id.into();
        if let Err(rejected) = self.producer.set(id) {
            tracing::warn!(
                already = %self.producer.get().map(String::as_str).unwrap_or(DEFAULT_PRODUCER),
                rejected = %rejected,
                "producer was already set on this MemoryServer; the template must never be \
                 stamped before the accept loop clones it, or every connection inherits one \
                 producer and its writes are misattributed"
            );
        }
    }

    /// Shares this server's language model, so a job that generates text takes
    /// turns on the same permits as the server does.
    pub(crate) fn llm_handle(&self) -> Arc<dyn Llm> {
        self.llm.clone()
    }

    /// Shares this server's generation semaphore so startup can grow it
    /// once the auto-tune benchmark learns the real ceiling.
    pub(crate) fn generation_permits_handle(&self) -> Arc<Semaphore> {
        self.generation_permits.clone()
    }

    /// Shares this server's authoritative granted-capacity counter, so a
    /// cold-start benchmark that grows the semaphore can keep it in step.
    pub(crate) fn granted_capacity_handle(&self) -> Arc<AtomicUsize> {
        self.granted_capacity.clone()
    }

    /// Shares this server's store slot with the consolidation job, whose
    /// rollback swaps the store in it.
    pub(crate) fn shared_store(&self) -> Arc<SharedStore> {
        self.store.clone()
    }

    /// The store for one call. The guard holds back a rollback's restore, so
    /// it is never kept across another `store()`: a restore queued in between
    /// would block the second read behind it, and neither would proceed.
    ///
    /// # Errors
    ///
    /// Fails when a rollback could not reopen the store, so the slot is empty.
    pub(crate) async fn store(
        &self,
    ) -> liam_store::Result<tokio::sync::RwLockReadGuard<'_, DefaultGraph>> {
        self.store
            .get()
            .await
            .map_err(|error| liam_store::Error::Backend(error.to_string()))
    }

    /// Bundles this connection's AIMD state for one evaluation to use.
    fn aimd_handles(&self) -> AimdHandles {
        AimdHandles {
            generation_permits: self.generation_permits.clone(),
            granted_capacity: self.granted_capacity.clone(),
            held_permit: self.held_permit.clone(),
            evaluating: self.evaluating.clone(),
            previous_window_average: self.previous_window_average.clone(),
        }
    }

    /// Test-only: exposes the shared window so a test can confirm `ask` and
    /// entity synthesis land in the same one, not two independent copies.
    #[cfg(test)]
    pub(crate) fn rolling_window_handle(&self) -> Arc<Mutex<RollingWindow>> {
        self.rolling_window.clone()
    }

    /// The producer id to stamp on this connection's writes right now:
    /// whatever `set_producer` set, or `DEFAULT_PRODUCER` if it has not run
    /// yet. See `producer`'s field doc for the two windows where that
    /// fallback applies.
    fn producer(&self) -> String {
        self.producer
            .get()
            .cloned()
            .unwrap_or_else(|| DEFAULT_PRODUCER.to_string())
    }

    // `pub(crate)` so the tool-eval grounding harness (see `tool_eval.rs`) drives
    // the same code path an MCP client hits, rather than a re-implementation of it.
    #[tool(description = "Record a durable decision or fact into long-term memory.")]
    pub(crate) async fn remember(&self, Parameters(args): Parameters<RememberArgs>) -> String {
        // Cheap checks first, before the embed call pays for a rejected
        // request: the `relate` handler establishes this convention.
        if let Some(problem) = confidence_problem(args.confidence) {
            return format!("remember failed: {problem}");
        }
        if let Some(problem) = attributes_problem(&args.attributes) {
            return format!("remember failed: {problem}");
        }
        if let Some(problem) = content_problem(&args.content) {
            return format!("remember failed: {problem}");
        }

        let Some(episode) = args.episode else {
            let embedding = match self.embedder.embed(&args.content, TextKind::Document).await {
                Ok(v) => v,
                Err(e) => return format!("embed failed: {e}"),
            };
            let node = NewNode::now(args.kind, args.label, args.content)
                .with_embedding(embedding)
                .with_producer(self.producer());
            let has_subject = args.subject.is_some();
            let node = apply_optional_fields(
                node,
                args.scope,
                args.attributes,
                args.valid_from,
                args.confidence,
                args.subject,
            );
            let write = if has_subject {
                on_store!(self, |store| store.upsert_by(node))
            } else {
                on_store!(self, |store| store.insert(node))
            };
            return match write {
                Ok(id) => format!("remembered {}", id.as_str()),
                Err(e) => format!("remember failed: {e}"),
            };
        };

        // `args.episode` was `Some`: an episode of additional facts and
        // edges, written atomically with the top-level fact through
        // `Graph::ingest_episode`. Bounded first, before any per-item check
        // or DB call, so a caller-sized episode can't hold
        // `ingest_episode`'s transaction open indefinitely.
        let total = 1 + episode.facts.len() + episode.entities.len() + episode.edges.len();
        if total > MAX_EPISODE_ITEMS {
            return format!(
                "remember failed: episode has {total} items, over the {MAX_EPISODE_ITEMS} max"
            );
        }

        // Every fact's confidence/attributes, and every edge's kind and
        // from/to syntax, validated up front by `validate_episode`: no DB
        // call happens until that whole pass is clean.
        let fact_count = episode.facts.len();
        let entity_count = episode.entities.len();
        let problems = validate_episode(&episode, fact_count, entity_count);
        if !problems.is_empty() {
            return format!("remember failed: {}", problems.join("; "));
        }

        // Every from/to is now known syntactically valid: either "fact:N" in
        // bounds, or handle-shaped. Resolve the handle-shaped ones now.
        let refs = match self
            .resolve_episode_refs(&episode, fact_count, entity_count)
            .await
        {
            Ok(refs) => refs,
            Err(resolve_problems) => {
                return format!("remember failed: {}", resolve_problems.join("; "));
            }
        };

        // Embed the top-level fact plus every episode.facts entry, then
        // build the combined node list in order: index 0 is the top-level
        // fact, 1..=fact_count is episode.facts, and
        // 1+fact_count..1+fact_count+entity_count is episode.entities.
        let embedding = match self.embedder.embed(&args.content, TextKind::Document).await {
            Ok(v) => v,
            Err(e) => return format!("embed failed: {e}"),
        };
        let mut nodes = Vec::with_capacity(1 + fact_count + entity_count);
        // Cloned once and reused below: an episode's facts and entities
        // share the top-level fact's scope (there is no per-item
        // override), so every node built from here on applies the same
        // `scope`. Cheap to clone at this size: episodes are capped at
        // `MAX_EPISODE_ITEMS`.
        let scope = args.scope;
        let top = NewNode::now(args.kind, args.label, args.content)
            .with_embedding(embedding)
            .with_producer(self.producer());
        let top = apply_optional_fields(
            top,
            scope.clone(),
            args.attributes,
            args.valid_from,
            args.confidence,
            args.subject,
        );
        nodes.push(top);

        for fact in episode.facts {
            let embedding = match self.embedder.embed(&fact.content, TextKind::Document).await {
                Ok(v) => v,
                Err(e) => return format!("embed failed: {e}"),
            };
            let node = NewNode::now(fact.kind, fact.label, fact.content)
                .with_embedding(embedding)
                .with_producer(self.producer());
            let node = apply_optional_fields(
                node,
                scope.clone(),
                fact.attributes,
                fact.valid_from,
                fact.confidence,
                fact.subject,
            );
            nodes.push(node);
        }

        // Entities land after every fact, in episode.entities order,
        // landing entity j at combined index 1 + fact_count + j. No embed
        // call: an entity's content is always empty by construction
        // (`NewNode::entity`), so there is nothing to embed.
        for entity in episode.entities {
            let mut node =
                NewNode::entity(entity.entity_type, entity.name).with_producer(self.producer());
            if let Some(scope) = scope.clone() {
                node = node.with_scope(scope);
            }
            nodes.push(node);
        }

        let edges: Vec<EpisodeEdge> = episode
            .edges
            .iter()
            .zip(&refs)
            .map(|(edge, (from, to))| EpisodeEdge {
                from: from.clone(),
                to: to.clone(),
                kind: edge.kind.trim().to_lowercase(),
                attributes: serde_json::json!({}),
            })
            .collect();

        match on_store!(self, |store| store.ingest_episode(nodes, edges)) {
            Ok(result) => {
                // Only a fresh `entity:N` from THIS episode triggers
                // resynthesis; a handle-referenced existing entity does not.
                let entity_start = 1 + fact_count;
                let entity_end = entity_start + entity_count;
                let is_fresh_entity = |r: &EpisodeRef| matches!(r, EpisodeRef::New(i) if (entity_start..entity_end).contains(i));
                let resolved_node_id = |r: &EpisodeRef| -> liam_store::NodeId {
                    match r {
                        EpisodeRef::New(i) => result.node_ids[*i].clone(),
                        EpisodeRef::Existing(id) => id.clone(),
                    }
                };
                let mut triggered: Vec<liam_store::NodeId> = Vec::new();
                for ((from, to), edge) in refs.iter().zip(&episode.edges) {
                    if edge.kind.trim().to_lowercase() != relation::MENTIONS {
                        continue;
                    }
                    for r in [from, to] {
                        if is_fresh_entity(r) {
                            let id = resolved_node_id(r);
                            if !triggered.contains(&id) {
                                triggered.push(id);
                            }
                        }
                    }
                }

                let mut lines: Vec<String> = result
                    .node_ids
                    .iter()
                    .map(|id| format!("remembered {}", id.as_str()))
                    .collect();
                for ((from, to), edge) in refs.iter().zip(&episode.edges) {
                    lines.push(format!(
                        "related {} -{}-> {}",
                        resolved_id(from, &result.node_ids),
                        edge.kind.trim().to_lowercase(),
                        resolved_id(to, &result.node_ids)
                    ));
                }

                if !triggered.is_empty() {
                    // One shared deadline for the whole triggered batch,
                    // mirroring `ask`'s "one deadline for the whole request" pattern.
                    let entity_synthesis_deadline = tokio::time::Instant::now()
                        + Duration::from_secs(self.ask_timeout_secs.max(1));
                    // Spawned to run concurrently; collection stays
                    // sequential so no outcome is ever dropped.
                    let mut handles = Vec::with_capacity(triggered.len());
                    for entity_id in triggered {
                        let server = self.clone();
                        let deadline = entity_synthesis_deadline;
                        handles.push(tokio::spawn(async move {
                            server
                                .resynthesize_entity(
                                    entity_id,
                                    deadline,
                                    ENTITY_SYNTHESIS_MAX_NEW_TOKENS,
                                )
                                .await
                        }));
                    }
                    for handle in handles {
                        match handle.await {
                            Ok(Ok(())) => {}
                            Ok(Err(failure)) => {
                                lines.push(format!("synthesis failed for {failure}"))
                            }
                            Err(join_err) => {
                                lines.push(format!("synthesis failed for a task: {join_err}"))
                            }
                        }
                    }
                }

                lines.join("\n")
            }
            Err(e) => format!("remember failed: {e}"),
        }
    }

    /// Recompiles one entity's page from its mentions using `remember`'s
    /// shared batch deadline; a failure is reported, never panicked.
    pub(crate) async fn resynthesize_entity(
        &self,
        entity_id: liam_store::NodeId,
        deadline: tokio::time::Instant,
        max_new_tokens: usize,
    ) -> Result<(), String> {
        // Held through the model call: a rollback waits for it, as for an `ask`.
        let store = self
            .store()
            .await
            .map_err(|e| format!("{}: {e}", entity_id.as_str()))?;
        self.resynthesize_entity_as(&store, entity_id, deadline, max_new_tokens, None)
            .await
    }

    /// `resynthesize_entity` with the new page written under `write_source`
    /// instead of the server's producer, so a consolidation run's rollback
    /// cancels the write and the supersession that comes with it. Works on the
    /// caller's `store`, never locking the slot again: a second read behind a
    /// queued restore would deadlock.
    pub(crate) async fn resynthesize_entity_as(
        &self,
        store: &DefaultGraph,
        entity_id: liam_store::NodeId,
        deadline: tokio::time::Instant,
        max_new_tokens: usize,
        write_source: Option<&str>,
    ) -> Result<(), String> {
        let now = liam_store::Millis::now();
        let candidate = match store.get(&entity_id, now).await {
            Ok(Some(c)) => c,
            Ok(None) => {
                return Err(format!(
                    "{}: entity vanished before synthesis could run",
                    entity_id.as_str()
                ))
            }
            Err(e) => return Err(format!("{}: {e}", entity_id.as_str())),
        };
        let label = candidate.label.clone();
        let rows = match store.mentions(&entity_id, now, MENTIONS_FETCH_LIMIT).await {
            Ok(rows) => rows,
            Err(e) => return Err(format!("{label}: {e}")),
        };
        let mentions: Vec<ask::Evidence> = rows.iter().map(ask::Evidence::from_candidate).collect();
        let synthesized = match synthesis::synthesize_entity(
            &*self.llm,
            &self.aimd_handles(),
            &self.rolling_window,
            deadline,
            &candidate.kind,
            &candidate.label,
            &mentions,
            self.ask_context_tokens,
            max_new_tokens,
        )
        .await
        {
            Ok(synthesized) => synthesized,
            Err(e) => return Err(format!("{label}: {e}")),
        };
        let new_content = synthesized.profile;
        let basis = match mention_basis(store, &rows[..synthesized.mentions_kept]).await {
            Ok(basis) => basis,
            Err(e) => return Err(format!("{label}: {e}")),
        };
        let embedding = match self.embedder.embed(&new_content, TextKind::Document).await {
            Ok(v) => v,
            Err(e) => return Err(format!("{label}: {e}")),
        };
        let mut node = NewNode::entity(candidate.kind, candidate.label)
            .with_attributes(candidate.attributes)
            .with_embedding(embedding)
            .with_basis(basis)
            .with_producer(write_source.map_or_else(|| self.producer(), str::to_owned));
        if let Some(scope) = candidate.scope.clone() {
            node = node.with_scope(scope);
        }
        node.content = new_content;
        match store.upsert_by(node).await {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("{label}: {e}")),
        }
    }

    // `pub(crate)` so the tool-eval grounding harness (see `tool_eval.rs`) drives
    // the same code path an MCP client hits, rather than a re-implementation of it.
    #[tool(
        name = "recall",
        description = "Retrieve relevant long-term memory, reranked for precision.",
        output_schema = reply::output_schema()
    )]
    async fn recall_tool(
        &self,
        Parameters(args): Parameters<RecallArgs>,
        cancel: CancellationToken,
    ) -> rmcp::model::CallToolResult {
        let reply = self.recall_with(args, cancel).await;
        self.count_degraded(Tool::Recall, &reply);
        reply.into_call_result()
    }

    // Test-only: the server routes through the token-taking wrapper above.
    #[cfg(test)]
    pub(crate) async fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        self.recall_with(args, CancellationToken::new())
            .await
            .text()
            .to_string()
    }

    /// Embeds `text` as a query. An embedder error is logged and reported as
    /// `EmbedFailed`, and the search then runs on keywords alone.
    async fn embed_or_degrade(
        &self,
        tool: &str,
        text: &str,
    ) -> (Option<Vec<f32>>, Option<Degraded>) {
        let started = self.stage_start();
        let embedded = self.embedder.embed(text, TextKind::Query).await;
        self.stage_end(Stage::Embed, started);
        match embedded {
            Ok(vector) => (Some(vector), None),
            Err(e) => {
                tracing::warn!(tool, error = %e, "embed failed, searching without a vector");
                (None, Some(Degraded::EmbedFailed))
            }
        }
    }

    pub(crate) async fn recall_with(
        &self,
        args: RecallArgs,
        cancel: CancellationToken,
    ) -> ToolReply {
        if let Err(refusal) = require_strict("recall", args.mode) {
            return ToolReply::failed(refusal);
        }
        if let Some(problem) = query_problem(&args.query) {
            return ToolReply::failed(format!("recall failed: {problem}"));
        }
        let _timer = self.time(Tool::Recall);
        if cancel.is_cancelled() {
            return ToolReply::ok(cancelled_reply("recall"));
        }
        let deadline = self.retrieval_deadline();
        let k = crate::ask::clamp_recall_k(args.k);
        let retrieve = async {
            let (embedding, embed_degraded) = self.embed_or_degrade("recall", &args.query).await;
            let q = build_query(
                &lexical_text(&args.query),
                Some(crate::ask::rerank_pool(k)),
                args.kind,
                args.scope,
                embedding,
                args.as_of,
                false,
            );
            (self.explained(&q).await, embed_degraded)
        };
        let (hits, embed_degraded) = match until_interrupted(deadline, &cancel, retrieve).await {
            Ok((Ok(h), degraded)) => (h, degraded),
            Ok((Err(e), _)) => return ToolReply::failed(format!("recall failed: {e}")),
            Err(why) => return self.interruption_reply("recall", why),
        };
        if hits.is_empty() {
            return with_reasons(ToolReply::ok("no relevant memory"), [embed_degraded]);
        }
        let docs: Vec<String> = hits
            .iter()
            .map(|h| h.hit.retrieved.as_text().to_string())
            .collect();
        let (order, rerank_degraded) = match self
            .rerank_within("recall", &args.query, &docs, deadline, &cancel)
            .await
        {
            Ok(ranked) => ranked,
            Err(reply) => return reply,
        };
        let order = &order[..order.len().min(k)];
        // Usage and handles cover only the hits returned, not the wider pool.
        self.record_usage(
            order
                .iter()
                .map(|&i| &hits[i])
                .filter(|h| !h.expanded)
                .map(|h| &h.hit.id),
        );
        self.issue_handles(order.iter().map(|&i| &hits[i].hit.id));
        // The handle rides inside the kind bracket so a hit still opens with
        // one bracketed field. It is here because an agent cannot link what it
        // recalled without a name for it, which is the gap ADR-0001 exists to
        // close. 13 characters, not the full 26: see `liam_store::HANDLE_LEN`.
        //
        // Confidence and attributes render as trailing lines instead, never
        // inside the bracket. ADR-0004 records putting confidence in the
        // bracket as a rejected alternative: it broke `resolve_handle`'s
        // alphanumeric-only gate, so the bracket stays `[{kind} {handle}]`
        // and nothing else, under every combination of these fields.
        let text = order
            .iter()
            .map(|&i| {
                let hit = &hits[i];
                let mut trailing = Vec::new();
                if hit.confidence != 1.0 {
                    trailing.push(format!("confidence: {:.2}", hit.confidence));
                }
                if let serde_json::Value::Object(map) = &hit.hit.attributes {
                    if !map.is_empty() {
                        trailing.push(format!("attributes: {}", hit.hit.attributes));
                    }
                }
                let head =
                    render::bracket_head(&hit.hit.kind, hit.hit.id.handle(), &hit.hit.label, None);
                render::render_item(
                    &head,
                    &hit.hit.retrieved,
                    &trailing,
                    self.max_basis_ids_rendered,
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        with_reasons(ToolReply::ok(text), [embed_degraded, rerank_degraded])
    }

    #[tool(
        description = "Record a relationship between two memories, using the handles recall returns."
    )]
    async fn relate(&self, Parameters(args): Parameters<RelateArgs>) -> String {
        // Lower-cased here, at the client boundary, so `Mentions` and
        // `mentions` are one relation instead of two. The clustering dedup keys
        // on the type string, so leaving both would let a pair reach weight 2.0
        // through nothing worse than inconsistent capitalisation, which is the
        // bias ADR-0002 Amendment 2 exists to remove. Inventing genuinely
        // different types still inflates a pair, and that risk stays open in
        // ADR-0001; this only closes the accidental half of it.
        let kind = args.kind.trim().to_lowercase();
        if kind.is_empty() {
            return "relate failed: type must not be empty".to_string();
        }
        // `supersedes` is written only inside `Graph::supersede`'s transaction
        // and carries version history. A client able to assert it could rewrite
        // what superseded what, so the refusal is at the door (ADR-0001).
        // Normalising first is what makes this exact comparison enough: the
        // store keeps types verbatim while the clustering filter matches
        // `type != 'supersedes'` exactly, so an accepted `SUPERSEDES` would
        // read as version history in the table and still count as a semantic
        // edge in every recompute.
        if kind == relation::SUPERSEDES {
            return format!(
                "relate failed: '{}' is reserved for version history",
                relation::SUPERSEDES
            );
        }
        let (from, to) = match (
            on_store!(self, |store| store.resolve_handle(&args.from)),
            on_store!(self, |store| store.resolve_handle(&args.to)),
        ) {
            (Err(e), _) => return format!("relate failed: from: {e}"),
            (_, Err(e)) => return format!("relate failed: to: {e}"),
            (Ok(from), Ok(to)) => (from, to),
        };
        // Compared after resolution, not before: two different handles can name
        // the same node, and a self-loop carries no meaning for clustering
        // while looking exactly like a client bug worth reporting back.
        if from == to {
            return format!(
                "relate failed: from and to are the same node ({})",
                from.handle()
            );
        }
        // The edge id is dropped rather than echoed. Nothing takes one as
        // input, and a ULID costs 19 tokens in the reply for no reachable use.
        match on_store!(self, |store| store.relate(&from, &to, &kind)) {
            Ok(_) => format!("related {} -{kind}-> {}", from.handle(), to.handle()),
            Err(e) => format!("relate failed: {e}"),
        }
    }

    #[tool(description = "List memory clusters found by community detection, largest first.")]
    pub(crate) async fn clusters(&self, Parameters(args): Parameters<ClustersArgs>) -> String {
        if let Err(refusal) = require_strict("clusters", args.mode) {
            return refusal;
        }
        let groups = match on_store!(self, |store| store.community_groups(false)) {
            Ok(g) => g,
            Err(e) => return format!("clusters failed: {e}"),
        };
        let narrowed = narrow_groups(&groups, args.k, args.members);
        // One tenth of the configured context (ADR-0002 S4): the operator's
        // declared "how much text is reasonable on this machine", not a
        // promise about the MCP client's own context.
        let budget = self.ask_context_tokens / 10;
        render_clusters(&narrowed, budget, |s| {
            self.llm
                .count_tokens(s)
                .unwrap_or_else(|| estimate_tokens(s))
        })
    }

    #[tool(
        description = "Show an entity's compiled page and the mentions behind it, most recent first."
    )]
    pub(crate) async fn timeline(&self, Parameters(args): Parameters<TimelineArgs>) -> String {
        if let Err(refusal) = require_strict("timeline", args.mode) {
            return refusal;
        }
        let id = match on_store!(self, |store| store.resolve_handle(&args.entity)) {
            Ok(id) => id,
            Err(e) => return format!("timeline failed: {e}"),
        };
        let now = liam_store::Millis::now();
        let entity = match on_store!(self, |store| store.get(&id, now)) {
            Ok(Some(c)) => c,
            Ok(None) => return format!("timeline failed: {} is not currently live", id.handle()),
            Err(e) => return format!("timeline failed: {e}"),
        };
        let limit = clamp_timeline_limit(args.limit);
        // Over-fetch by one: a row past `limit` proves "there is at least
        // one more" without a specific, unprovable total count. It says
        // nothing about `Graph::mentions`'s own internal over-fetch pool
        // (`limit.max(1) * 3`), which exists to survive hydration drops and
        // is unrelated to this trailer.
        let mut mentions = match on_store!(self, |store| store.mentions(&id, now, limit + 1)) {
            Ok(m) => m,
            Err(e) => return format!("timeline failed: {e}"),
        };
        // The page header is the one inferred item this tool shows; a mention
        // that is itself inferred is left out.
        mentions.retain(|m| !m.retrieved.is_inferred());
        let truncated = mentions.len() > limit;
        if truncated {
            mentions.truncate(limit);
        }
        let header = render::render_item(
            &render::bracket_head(&entity.kind, id.handle(), &entity.label, None),
            &entity.retrieved,
            &[],
            self.max_basis_ids_rendered,
        );
        let mentions_section = if mentions.is_empty() {
            "no mentions yet".to_string()
        } else {
            mentions
                .iter()
                .map(|m| {
                    let head = render::bracket_head(
                        &m.kind,
                        m.id.handle(),
                        &m.label,
                        Some(m.valid_from.0),
                    );
                    render::render_item(&head, &m.retrieved, &[], self.max_basis_ids_rendered)
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        let trailer = if truncated {
            "\n\n(more mentions exist)"
        } else {
            ""
        };
        format!("{header}\n\n{mentions_section}{trailer}")
    }

    #[tool(
        description = "List the facts held about a subject. Default: what is true now. as_of \
                       (epoch ms) asks what was true then; believed_at (epoch ms) asks what the \
                       store believed then."
    )]
    pub(crate) async fn facts(&self, Parameters(args): Parameters<FactsArgs>) -> String {
        facts::read(self, args).await
    }

    #[tool(
        description = "Say whether a result recall or ask returned was correct, using its handle. \
                       Verdicts decide which memories stay fast to search."
    )]
    pub(crate) async fn feedback(&self, Parameters(args): Parameters<FeedbackArgs>) -> String {
        feedback::give(self, args).await
    }

    #[tool(
        description = "List the summaries LIAM has written over memory, or show one with what it \
         was built from. Summaries are inferred: model-written from recorded memory, never \
         recorded themselves, and each lists the events it rests on. Filter by level, scope or a \
         query; pass id to see one summary's children."
    )]
    pub(crate) async fn summaries(&self, Parameters(args): Parameters<SummariesArgs>) -> String {
        summaries::read(self, args).await
    }

    #[tool(
        description = "Count what deleting a subject or one event would destroy, per store, \
                       without changing anything. Deleting happens with `liamd delete`, with the \
                       daemon stopped. Only producers listed under [admin] may call it."
    )]
    pub(crate) async fn delete(&self, Parameters(args): Parameters<DeleteArgs>) -> String {
        delete::read(self, args).await
    }

    #[tool(
        description = "Run one consolidation now and report how it ended. Only producers listed \
                       under [admin] may call it."
    )]
    pub(crate) async fn consolidate(&self) -> String {
        consolidate::run(self).await
    }

    #[tool(
        description = "Show the latest reading of each metric LIAM keeps about itself, and the \
                       alerts that are active. Pass metric to see one, limit to cap the readings."
    )]
    pub(crate) async fn metrics(&self, Parameters(args): Parameters<MetricsArgs>) -> String {
        metrics::read(self, args).await
    }

    /// Acquires one slot from `generation_permits`, racing the shared request
    /// `deadline` rather than a fresh timeout of its own: see the `deadline`
    /// comment in `ask` for why every stage has to share one instant. Returns
    /// the exact reason text `ask` hands to `fallback_answer` on either
    /// failure path.
    async fn acquire_generation_permit(
        &self,
        deadline: tokio::time::Instant,
    ) -> Result<OwnedSemaphorePermit, &'static str> {
        match tokio::time::timeout_at(deadline, self.generation_permits.clone().acquire_owned())
            .await
        {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err("no generation slot is available"),
            // Without this timeout, the k-th queued caller would wait up to
            // k * ask_timeout_secs before its own generation budget even
            // started, so a queue would turn one slow request into a pile of
            // requests that each look like a hang. The shared deadline is
            // what keeps that wait from silently eating into budget the
            // later stages never get back.
            Err(_) => Err("timed out waiting for a generation slot"),
        }
    }

    /// Runs the sufficiency pre-pass against the shared `deadline`: does the
    /// evidence actually contain the answer? `true` means proceed to
    /// synthesis; only an explicit parsed "no" returns `false`, so a
    /// timeout, an LLM error, or an unparseable reply fails open rather than
    /// denying memory the store really holds.
    async fn passes_sufficiency_check(
        &self,
        question: &str,
        evidence: &[ask::Evidence],
        deadline: tokio::time::Instant,
    ) -> bool {
        let (system, user) = ask::build_sufficiency_prompt(question, evidence);
        let verdict = tokio::time::timeout_at(
            deadline,
            // Capped hard: the verdict is one word, and an uncapped pre-pass
            // let a rambling model spend 50s per question (see eval.rs).
            self.llm
                .complete_capped(&system, &user, SUFFICIENCY_MAX_TOKENS),
        )
        .await;
        if let Ok(Ok(reply)) = verdict {
            ask::parse_sufficiency(&reply) != Some(false)
        } else {
            true
        }
    }

    /// Builds the answer prompt, synthesizes against the shared `deadline`,
    /// and dispatches to the grounded answer or one of the fallback
    /// variants: the natural end of `ask`'s orchestration.
    async fn synthesize_answer(
        &self,
        question: &str,
        evidence: &[ask::Evidence],
        deadline: tokio::time::Instant,
    ) -> (String, Option<Degraded>) {
        let (system, user) = build_ask_prompt(question, evidence);
        let synth = tokio::time::timeout_at(deadline, self.llm.complete(&system, &user)).await;
        let unavailable = |reason: &str| {
            (
                fallback_answer(reason, evidence, self.max_basis_ids_rendered),
                Some(Degraded::SynthesisUnavailable),
            )
        };
        match synth {
            Ok(Ok(a)) if !a.trim().is_empty() => {
                let answer = a.trim();
                // Last line of defence against prompt injection and free-running
                // fabrication: an answer that shares almost no vocabulary with the
                // evidence is not a synthesis of it, whatever the model intended.
                // Unlike the prompt rules, this does not depend on the model
                // cooperating. See `ask::is_grounded`.
                if ask::is_grounded(answer, question, evidence) {
                    (
                        format_answer(answer, evidence, self.max_basis_ids_rendered),
                        None,
                    )
                } else {
                    unavailable("the answer was not grounded in the evidence")
                }
            }
            Ok(Ok(_)) => unavailable("the model returned an empty answer"),
            Ok(Err(_)) => unavailable("the model failed"),
            Err(_) => unavailable("synthesis timed out"),
        }
    }

    // `pub(crate)` so the grounding eval (see `eval.rs`) drives the same code
    // path an MCP client hits, rather than a re-implementation of it.
    #[tool(
        name = "ask",
        description = "Answer a question from long-term memory, synthesized and cited.",
        output_schema = reply::output_schema()
    )]
    async fn ask_tool(
        &self,
        Parameters(args): Parameters<AskArgs>,
        cancel: CancellationToken,
    ) -> rmcp::model::CallToolResult {
        let reply = self.ask_with(args, cancel).await;
        self.count_degraded(Tool::Ask, &reply);
        reply.into_call_result()
    }

    // Test-only: the server routes through the token-taking wrapper above.
    #[cfg(test)]
    pub(crate) async fn ask(&self, Parameters(args): Parameters<AskArgs>) -> String {
        self.ask_with(args, CancellationToken::new())
            .await
            .text()
            .to_string()
    }

    pub(crate) async fn ask_with(&self, args: AskArgs, cancel: CancellationToken) -> ToolReply {
        if let Some(problem) = query_problem(&args.question) {
            return ToolReply::failed(format!("ask failed: {problem}"));
        }
        let _timer = self.time(Tool::Ask);
        if cancel.is_cancelled() {
            return ToolReply::ok(cancelled_reply("ask"));
        }
        let retrieval_deadline = self.retrieval_deadline();
        let mode = args.mode.unwrap_or(Mode::Infer);
        let k = clamp_ask_k(args.k);
        let retrieve = async {
            let (embedding, embed_degraded) = self.embed_or_degrade("ask", &args.question).await;
            let q = build_query(
                &lexical_text(&args.question),
                Some(ask::rerank_pool(k)),
                args.kind,
                args.scope.clone(),
                embedding.clone(),
                args.as_of,
                mode == Mode::Infer,
            );
            let hits = match self.explained(&q).await {
                Ok(hits) => hits,
                Err(e) => return Err(ToolReply::failed(format!("ask failed: {e}"))),
            };
            // Summaries are model-written, so only infer mode looks for them.
            let summaries = match (&embedding, mode) {
                (Some(question), Mode::Infer) => {
                    summaries::matching(self, question, &args.scope, args.as_of).await
                }
                _ => Vec::new(),
            };
            Ok((hits, summaries, embed_degraded))
        };
        let (hits, summaries, embed_degraded) =
            match until_interrupted(retrieval_deadline, &cancel, retrieve).await {
                Ok(Ok(found)) => found,
                Ok(Err(reply)) => return reply,
                Err(why) => return self.interruption_reply("ask", why),
            };
        if hits.is_empty() && summaries.is_empty() {
            return with_reasons(ToolReply::ok("no relevant memory"), [embed_degraded]);
        }
        let candidates: Vec<ask::Evidence> = hits
            .iter()
            .map(ask::Evidence::from_hit)
            .chain(summaries)
            .collect();
        let docs: Vec<String> = candidates
            .iter()
            .map(|e| e.retrieved.as_text().to_string())
            .collect();
        let (order, rerank_degraded) = match self
            .rerank_within("ask", &args.question, &docs, retrieval_deadline, &cancel)
            .await
        {
            Ok(ranked) => ranked,
            Err(reply) => return reply,
        };
        let retrieval_reasons = [embed_degraded, rerank_degraded];
        // Candidates list the store hits first, so an index below `hit_count`
        // is a store hit. Keep k of them plus every summary.
        let hit_count = hits.len();
        let mut kept_hits = 0;
        let evidence: Vec<ask::Evidence> = order
            .into_iter()
            .filter(|&i| {
                if i >= hit_count {
                    return true;
                }
                kept_hits += 1;
                kept_hits <= k
            })
            .map(|i| candidates[i].clone())
            .collect();
        if mode == Mode::Strict {
            self.issue_handles(evidence.iter().filter_map(|e| e.node_id.as_ref()));
            self.record_evidence_usage(&evidence);
            return with_reasons(
                ToolReply::ok(ask::strict_answer(&evidence, self.max_basis_ids_rendered)),
                retrieval_reasons,
            );
        }
        // Trim ONCE, before the sufficiency pre-pass, and reuse this same
        // slice for both prompts below. Trimming twice, or only for the
        // answer prompt, would let the pre-pass vouch for evidence the answer
        // never sees. The closure prefers the model's real count and only
        // falls back to the estimate when the provider cannot count.
        let evidence = fit_evidence_to_budget(
            |slice| build_ask_prompt(&args.question, slice),
            &evidence,
            self.ask_context_tokens,
            |s| {
                self.llm
                    .count_tokens(s)
                    .unwrap_or_else(|| estimate_tokens(s))
            },
        );

        // Every reply from here on shows these results, with their handles.
        self.issue_handles(evidence.iter().filter_map(|e| e.node_id.as_ref()));

        // One deadline for the stages after retrieval (retrieval has its own
        // `recall_timeout_secs` budget): the permit acquire, the
        // sufficiency pre-pass, and synthesis below all race against this
        // same instant instead of each getting its own fresh
        // `ask_timeout_secs`, so a slow stage eats into the budget the later
        // stages get rather than tripling the wall-clock cap. `max(1)` guards
        // against an operator typo of `ask_timeout_secs = 0`, which would
        // otherwise make the whole request time out immediately.
        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.ask_timeout_secs.max(1));

        // Bound concurrent generation before either model call below: both
        // the sufficiency pre-pass and synthesis hit the model, so a permit
        // held only across synthesis would let N sufficiency calls pile onto
        // the GPU while only synthesis was bounded. `acquire_owned` on a
        // clone of the `Arc` keeps the permit's lifetime independent of any
        // borrow of `self` across the awaits below; it drops, releasing the
        // slot, when `ask` returns.
        let acquire_start = tokio::time::Instant::now();
        let _generation_permit = match self.acquire_generation_permit(deadline).await {
            Ok(permit) => permit,
            Err(reason) => {
                return with_reasons(
                    ToolReply::ok(fallback_answer(
                        reason,
                        evidence,
                        self.max_basis_ids_rendered,
                    )),
                    retrieval_reasons
                        .into_iter()
                        .chain([Some(Degraded::SynthesisUnavailable)]),
                )
            }
        };
        // Covers every return path below, not just one model call.
        let _permit_timer = PermitTimer::new(
            self.rolling_window.clone(),
            self.aimd_handles(),
            acquire_start.elapsed(),
            self.ask_context_tokens,
        );

        // Sufficiency pre-pass: ask whether the evidence answers the question at
        // all, and refuse outright if it does not. See
        // `ask::build_sufficiency_prompt` for why this is a separate call.
        if self.ask_sufficiency_check
            && !self
                .passes_sufficiency_check(&args.question, evidence, deadline)
                .await
        {
            return with_reasons(
                ToolReply::ok(ask::insufficient_answer(
                    evidence,
                    self.max_basis_ids_rendered,
                )),
                retrieval_reasons,
            );
        }
        // From here the model sees the evidence, so it counts even if the
        // answer then fails.
        self.record_evidence_usage(evidence);

        let (answer, synthesis_degraded) = self
            .synthesize_answer(&args.question, evidence, deadline)
            .await;
        with_reasons(
            ToolReply::ok(answer),
            retrieval_reasons.into_iter().chain([synthesis_degraded]),
        )
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

#[cfg(test)]
mod tests {
    use super::episode::test_support::{episode_edge, episode_entity, episode_fact};
    use super::episode::{EpisodeEntityArgs, EpisodeFactArgs};
    use super::test_doubles::{CountingEmbedder, FailingReranker};
    use super::*;
    use liam_store::{FixedClock, GraphConfig, Millis, HANDLE_LEN};
    use serde_json::json;

    /// Always-errors `Llm`, so `ask` must fall back to the retrieved evidence.
    struct FailingLlm;
    #[async_trait::async_trait]
    impl liam_model::Llm for FailingLlm {
        async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
            Err(liam_model::ModelError::Llm("boom".into()))
        }
    }

    /// Sleeps past any short test timeout, so `ask` must fall back instead of
    /// waiting on synthesis. Used to exercise the `tokio::time::timeout` arm
    /// without a real 30s test.
    struct SlowLlm;
    #[async_trait::async_trait]
    impl liam_model::Llm for SlowLlm {
        async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok("too late".to_string())
        }
    }

    /// Records what reached the model and replies with a canned answer. WHY not
    /// `MockLlm` for the synthesized-path tests: its echo repeats the system
    /// rules, which `ask::is_grounded` correctly rejects as not derived from the
    /// evidence. Recording the prompt asserts prompt delivery directly instead of
    /// inferring it from an echo.
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

    /// Answers with fluent text that shares nothing with the evidence: what a
    /// model does when it answers from its own priors instead of the retrieved
    /// facts.
    struct UngroundedLlm;
    #[async_trait::async_trait]
    impl liam_model::Llm for UngroundedLlm {
        async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
            Ok(
                "Kubernetes clusters orchestrate containerized microservice deployments."
                    .to_string(),
            )
        }
    }

    /// Answers the sufficiency pre-pass and the synthesis call differently, so a
    /// test can drive one without the other. Told apart by the pre-pass prompt's
    /// own wording rather than call order, which would break the moment the
    /// handler reorders its calls.
    struct SufficiencyLlm {
        verdict: &'static str,
        answer: &'static str,
    }

    #[async_trait::async_trait]
    impl liam_model::Llm for SufficiencyLlm {
        async fn complete(&self, _s: &str, prompt: &str) -> liam_model::Result<String> {
            if prompt.contains("Reply YES or NO") {
                Ok(self.verdict.to_string())
            } else {
                Ok(self.answer.to_string())
            }
        }
    }

    /// Succeeds but with a blank answer, so `ask` must fall back to evidence
    /// instead of returning whitespace as if it were a real synthesis.
    struct EmptyLlm;
    #[async_trait::async_trait]
    impl liam_model::Llm for EmptyLlm {
        async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
            Ok("   ".to_string())
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
        grounded: bool,
    }

    impl GatedLlm {
        fn new(release: Arc<tokio::sync::Notify>) -> Self {
            Self {
                release,
                in_flight: std::sync::atomic::AtomicUsize::new(0),
                peak: std::sync::atomic::AtomicUsize::new(0),
                grounded: false,
            }
        }

        /// As `new`, but replies with `grounded_entity_reply(prompt)` instead
        /// of the fixed literal, for the one call site that gates entity
        /// synthesis rather than `ask`.
        fn new_grounded(release: Arc<tokio::sync::Notify>) -> Self {
            Self {
                release,
                in_flight: std::sync::atomic::AtomicUsize::new(0),
                peak: std::sync::atomic::AtomicUsize::new(0),
                grounded: true,
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
            if self.grounded {
                Ok(grounded_entity_reply(prompt))
            } else {
                Ok("the wibbleflux service runs nightly [1].".to_string())
            }
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

    /// Fresh in-memory server wired with the given reranker/llm, a 30s ask
    /// timeout, and the shipped default 8192-token context budget. Dims 8 to
    /// match `MockEmbedder::new(8)`.
    async fn server_with(reranker: Arc<dyn Reranker>, llm: Arc<dyn Llm>) -> MemoryServer {
        server_with_timeout(reranker, llm, 30, false, 8192).await
    }

    /// Fresh in-memory server wired with the given reranker/llm/ask
    /// timeout/context budget, generation capped to 1 concurrent call (the
    /// shipped default). Dims 8 to match `MockEmbedder::new(8)`.
    async fn server_with_timeout(
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn Llm>,
        ask_timeout_secs: u64,
        ask_sufficiency_check: bool,
        ask_context_tokens: usize,
    ) -> MemoryServer {
        server_with_generation_limit(
            reranker,
            llm,
            ask_timeout_secs,
            ask_sufficiency_check,
            ask_context_tokens,
            1,
        )
        .await
    }

    /// As `server_with_timeout`, plus an explicit `max_concurrent_generations`
    /// for the tests in this module that exercise the generation semaphore
    /// itself; every other test goes through `server_with_timeout`'s fixed 1.
    async fn server_with_generation_limit(
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn Llm>,
        ask_timeout_secs: u64,
        ask_sufficiency_check: bool,
        ask_context_tokens: usize,
        max_concurrent_generations: usize,
    ) -> MemoryServer {
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        MemoryServer::new(
            Arc::new(store),
            Arc::new(liam_model::MockEmbedder::new(8)),
            reranker,
            llm,
            ask_timeout_secs,
            ask_sufficiency_check,
            ask_context_tokens,
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

    /// Server with the neutral doubles used by the filter tests: identity rerank
    /// (so ordering is the store's) and the echo llm (so `ask`'s answer contains
    /// whatever evidence reached the prompt, which is what they assert on).
    async fn plain_server() -> MemoryServer {
        server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
        )
        .await
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
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
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

    #[tokio::test]
    async fn remember_stamps_the_servers_producer_on_the_written_node() {
        use liam_store::{Backend, DefaultBackend};

        // Given a MemoryServer carrying a producer
        let db_path = temp_db_path();
        let db_path_str = db_path.to_str().expect("temp path is valid utf-8");
        let store = DefaultGraph::open(db_path_str, GraphConfig::new(8))
            .await
            .expect("open file-backed store");
        let server = MemoryServer::new(
            Arc::new(store),
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        );
        server.set_producer("agent-a");

        // When remember writes a node
        let out = server
            .remember(Parameters(RememberArgs {
                kind: "fact".to_string(),
                label: "label".to_string(),
                content: "content".to_string(),
                scope: None,
                subject: None,
                attributes: None,
                valid_from: None,
                confidence: None,
                episode: None,
            }))
            .await;
        assert!(out.starts_with("remembered "), "remember failed: {out}");

        // Then the stored row records that producer. Read it back through a
        // fresh connection to the same file, since `producer` is
        // deliberately absent from `Hit` and cannot be asserted through
        // `recall`.
        let raw = DefaultBackend::open(db_path_str, 1)
            .await
            .expect("open a second connection to the same file");
        let rows = raw
            .query("SELECT producer FROM nodes", &[])
            .await
            .expect("query nodes");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get_string(0).unwrap(), "agent-a");

        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(format!("{db_path_str}-wal"));
        let _ = std::fs::remove_file(format!("{db_path_str}-shm"));
    }

    #[tokio::test]
    async fn remember_collapses_entity_pages_across_producers_via_episode() {
        use liam_store::{Backend, DefaultBackend};

        // Given two MemoryServer clones over one store, each stamped with a
        // different producer before either calls remember
        let db_path = temp_db_path();
        let db_path_str = db_path.to_str().expect("temp path is valid utf-8");
        let store = DefaultGraph::open(db_path_str, GraphConfig::new(8))
            .await
            .expect("open file-backed store");
        let template = MemoryServer::new(
            Arc::new(store),
            Arc::new(liam_model::MockEmbedder::new(8)),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        );
        let server_a = template.clone();
        server_a.set_producer("agent-a");
        let server_b = template.clone();
        server_b.set_producer("agent-b");

        // When each calls remember with an episode whose entities list
        // names "Alice" (same type, same scope)
        let out_a = server_a
            .remember(Parameters(RememberArgs {
                scope: Some("s".to_string()),
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Alice")],
                    edges: vec![],
                }),
                ..remember_args("two producer episode entity content a")
            }))
            .await;
        assert!(!out_a.contains("failed"), "{out_a}");

        let out_b = server_b
            .remember(Parameters(RememberArgs {
                scope: Some("s".to_string()),
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Alice")],
                    edges: vec![],
                }),
                ..remember_args("two producer episode entity content b")
            }))
            .await;
        assert!(!out_b.contains("failed"), "{out_b}");

        // Then exactly one live "Alice" entity page exists for that subject
        // and scope, read back through a fresh connection to the same file
        let raw = DefaultBackend::open(db_path_str, 1)
            .await
            .expect("open a second connection to the same file");
        let rows = raw
            .query(
                "SELECT COUNT(*) FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3",
                &["alice".into(), "s".into(), liam_store::FOREVER.into()],
            )
            .await
            .expect("query nodes");
        assert_eq!(
            rows[0].get_i64(0).unwrap(),
            1,
            "exactly one live Alice page must remain, ignoring producer"
        );

        // And both producers' writes landed as separate rows, with agent-b's
        // the live one, proving the entity loop stamped each write with its
        // own producer rather than collapsing both under one
        let producer_rows = raw
            .query(
                "SELECT producer, tx_to FROM nodes WHERE subject = ?1 AND scope = ?2",
                &["alice".into(), "s".into()],
            )
            .await
            .expect("query nodes by producer");
        assert_eq!(
            producer_rows.len(),
            2,
            "both agent-a's and agent-b's Alice pages must exist as separate rows"
        );
        let live_row = producer_rows
            .iter()
            .find(|r| r.get_i64(1).unwrap() == liam_store::FOREVER.0)
            .expect("one row must be live");
        assert_eq!(
            live_row.get_string(0).unwrap(),
            "agent-b",
            "the live Alice page must belong to agent-b"
        );

        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(format!("{db_path_str}-wal"));
        let _ = std::fs::remove_file(format!("{db_path_str}-shm"));
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

    /// A one-key JSON object whose `.to_string()` is exactly `total`
    /// characters, for pinning `MAX_ATTRIBUTES_CHARS`'s boundary exactly
    /// rather than guessing at serde_json's compact formatting.
    fn attributes_of_length(total: usize) -> serde_json::Value {
        let overhead = json!({"k": ""}).to_string().chars().count();
        let padding = "a".repeat(total - overhead);
        json!({ "k": padding })
    }

    #[tokio::test]
    async fn remember_with_attributes_round_trips_through_query_explained() {
        // Given a server and an attributes object to remember
        let server = plain_server().await;

        // When remember writes the node
        let out = server
            .remember(Parameters(RememberArgs {
                attributes: Some(json!({"k": "v"})),
                ..remember_args("attributes round trip content")
            }))
            .await;
        assert!(out.starts_with("remembered "), "{out}");

        // Then the store holds exactly the attributes sent
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("attributes round trip content"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hit.attributes, json!({"k": "v"}));
    }

    #[tokio::test]
    async fn remember_with_valid_from_sets_the_stored_instant_not_now() {
        // Given a server and an explicit backdated valid_from
        let server = plain_server().await;

        // When remember writes the node
        let out = server
            .remember(Parameters(RememberArgs {
                valid_from: Some(1000),
                ..remember_args("valid from backdated content")
            }))
            .await;
        assert!(out.starts_with("remembered "), "{out}");

        // Then the store holds that instant, not the store's clock
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("valid from backdated content"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].valid_from, Millis(1000));
    }

    #[tokio::test]
    async fn remember_with_confidence_round_trips_through_query_explained() {
        // Given a server and an explicit confidence
        let server = plain_server().await;

        // When remember writes the node
        let out = server
            .remember(Parameters(RememberArgs {
                confidence: Some(0.75),
                ..remember_args("confidence round trip content")
            }))
            .await;
        assert!(out.starts_with("remembered "), "{out}");

        // Then the store holds exactly that confidence
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("confidence round trip content"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].confidence, 0.75);
    }

    #[tokio::test]
    async fn remember_rejects_out_of_range_confidence_without_writing() {
        // Given a server, and several confidence values outside 0.0..=1.0
        let server = plain_server().await;
        let marker = "confidence boundary marker content";

        for confidence in [-0.1, 1.1, 5.0] {
            let before = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker))
                .await
                .unwrap()
                .len();

            // When remember is called with that confidence
            let out = server
                .remember(Parameters(RememberArgs {
                    confidence: Some(confidence),
                    ..remember_args(marker)
                }))
                .await;

            // Then it is refused with the exact error text, and no node lands
            assert_eq!(
                out, "remember failed: confidence must be between 0.0 and 1.0",
                "confidence {confidence}: {out}"
            );
            let after = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker))
                .await
                .unwrap()
                .len();
            assert_eq!(
                after, before,
                "confidence {confidence} wrote a node despite rejection"
            );
        }
    }

    #[tokio::test]
    async fn remember_accepts_confidence_at_its_inclusive_bounds() {
        // Given a server, and confidence at each inclusive bound
        let server = plain_server().await;

        // Distinct marker words per bound, not a shared phrase: `fts5_query`
        // ORs quoted terms together, so two contents sharing any word would
        // each match the other's query too.
        for (confidence, marker) in [(0.0, "confidencefloor"), (1.0, "confidenceceiling")] {
            // When remember is called with that confidence
            let out = server
                .remember(Parameters(RememberArgs {
                    confidence: Some(confidence),
                    ..remember_args(marker)
                }))
                .await;

            // Then it is accepted and round-trips exactly
            assert!(
                out.starts_with("remembered "),
                "confidence {confidence}: {out}"
            );
            let hits = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker))
                .await
                .unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].confidence, confidence);
        }
    }

    #[tokio::test]
    async fn remember_rejects_non_object_attributes() {
        // Given a server, and several non-object attributes shapes
        let server = plain_server().await;

        for attributes in [json!(["a"]), json!("x"), json!(1), json!(true)] {
            // When remember is called with that shape
            let out = server
                .remember(Parameters(RememberArgs {
                    attributes: Some(attributes.clone()),
                    ..remember_args("non object attributes content")
                }))
                .await;

            // Then it is refused with the exact error text
            assert_eq!(
                out, "remember failed: attributes must be a JSON object",
                "attributes {attributes:?}: {out}"
            );
        }
    }

    #[tokio::test]
    async fn remember_attributes_boundary_at_the_max_char_cap() {
        // Given attributes serialized to exactly MAX_ATTRIBUTES_CHARS
        let server = plain_server().await;
        let at_cap = attributes_of_length(MAX_ATTRIBUTES_CHARS);
        assert_eq!(at_cap.to_string().chars().count(), MAX_ATTRIBUTES_CHARS);

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                attributes: Some(at_cap),
                ..remember_args("attributes at cap content")
            }))
            .await;

        // Then it is accepted
        assert!(out.starts_with("remembered "), "{out}");

        // Given attributes one character past the cap
        let over_cap = attributes_of_length(MAX_ATTRIBUTES_CHARS + 1);
        assert_eq!(
            over_cap.to_string().chars().count(),
            MAX_ATTRIBUTES_CHARS + 1
        );
        let marker = "attributes over cap content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                attributes: Some(over_cap),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused with the exact error text, and no node lands
        assert_eq!(
            out,
            format!("remember failed: attributes exceeds {MAX_ATTRIBUTES_CHARS} characters")
        );
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(
            after, before,
            "over-cap attributes wrote a node despite rejection"
        );
    }

    #[tokio::test]
    async fn remember_content_boundary_at_the_max_char_cap() {
        // Given content built to exactly MAX_CONTENT_CHARS
        let server = plain_server().await;
        let at_cap = "x".repeat(MAX_CONTENT_CHARS);
        assert_eq!(at_cap.chars().count(), MAX_CONTENT_CHARS);

        // When remember is called with it
        let out = server.remember(Parameters(remember_args(&at_cap))).await;

        // Then it is accepted
        assert!(out.starts_with("remembered "), "{out}");

        // Given content one character past the cap, built around a unique
        // marker so the before/after query below can detect a written node
        let marker = "content over cap marker";
        let over_cap = format!(
            "{marker} {}",
            "x".repeat(MAX_CONTENT_CHARS - marker.chars().count())
        );
        assert_eq!(over_cap.chars().count(), MAX_CONTENT_CHARS + 1);
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server.remember(Parameters(remember_args(&over_cap))).await;

        // Then it is refused with the exact error text, and no node lands
        assert_eq!(
            out,
            format!("remember failed: content exceeds {MAX_CONTENT_CHARS} characters")
        );
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(
            after, before,
            "over-cap content wrote a node despite rejection"
        );
    }

    #[tokio::test]
    async fn remember_content_boundary_counts_unicode_scalars_not_bytes() {
        // Given content built from the precomposed 'é' scalar (U+00E9, 2
        // UTF-8 bytes) repeated to exactly MAX_CONTENT_CHARS characters: a
        // regression that swapped `.chars().count()` for `.len()` (byte
        // count) in `content_problem` would double-count this and reject it
        // at the cap, something ASCII-only fixtures can't catch since ASCII
        // chars and bytes are 1:1.
        let server = plain_server().await;
        let at_cap: String = "é".repeat(MAX_CONTENT_CHARS);
        assert_eq!(at_cap.chars().count(), MAX_CONTENT_CHARS);

        // When remember is called with it
        let out = server.remember(Parameters(remember_args(&at_cap))).await;

        // Then it is accepted
        assert!(out.starts_with("remembered "), "{out}");

        // Given content one character past the cap, built the same way
        let over_cap: String = "é".repeat(MAX_CONTENT_CHARS + 1);
        assert_eq!(over_cap.chars().count(), MAX_CONTENT_CHARS + 1);

        // When remember is called with it
        let out = server.remember(Parameters(remember_args(&over_cap))).await;

        // Then it is refused with the exact error text
        assert_eq!(
            out,
            format!("remember failed: content exceeds {MAX_CONTENT_CHARS} characters")
        );
    }

    #[tokio::test]
    async fn remember_without_the_new_args_keeps_prior_defaults() {
        // Given the pre-WU1 call shape: no attributes/valid_from/confidence
        let server = plain_server().await;
        let before = Millis::now();

        // When remember writes the node
        let out = server
            .remember(Parameters(remember_args("regression pin content")))
            .await;
        assert!(out.starts_with("remembered "), "{out}");
        let after = Millis::now();

        // Then attributes/confidence keep their old defaults, and valid_from
        // resolves to "now" within the call's own wall-clock window
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("regression pin content"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hit.attributes, json!({}));
        assert_eq!(hits[0].confidence, 1.0);
        assert!(
            hits[0].valid_from.0 >= before.0 && hits[0].valid_from.0 <= after.0,
            "valid_from {:?} not within [{before:?}, {after:?}]",
            hits[0].valid_from
        );
    }

    #[tokio::test]
    async fn remember_without_episode_is_unchanged() {
        // Given a server, and a remember call with `episode` explicitly
        // absent, the shape every pre-WU4 caller sends
        let server = plain_server().await;

        // When remember is called
        let out = server
            .remember(Parameters(remember_args(
                "episode absent regression content",
            )))
            .await;

        // Then the response is the same "remembered {id}" shape as before,
        // and exactly one node lands: no episode-shaped side effect at all
        assert!(out.starts_with("remembered "), "{out}");
        assert!(!out.contains('\n'), "unexpected extra line: {out}");
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("episode absent regression content"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn remember_with_episode_facts_and_an_edge_between_them_writes_atomically() {
        // Given a top-level fact, one episode.facts entry, and an edge
        // between them by combined index
        let server = plain_server().await;

        // When remember is called with episode set
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("episodenestedfactmarker")],
                    entities: vec![],
                    edges: vec![episode_edge("fact:0", "fact:1", "relates_to")],
                }),
                ..remember_args("episodetoplevelfactmarker")
            }))
            .await;

        // Then it succeeds, both facts exist, and the edge exists
        assert!(!out.contains("failed"), "{out}");
        let mut lines = out.lines();
        let top_id = lines
            .next()
            .expect("top-level fact line")
            .trim_start_matches("remembered ")
            .to_string();
        let nested_id = lines
            .next()
            .expect("nested fact line")
            .trim_start_matches("remembered ")
            .to_string();

        // Counted by content match, not by result length: the edge just
        // created between them makes each reachable as a graph-expanded
        // neighbour of the other's text hit too (`ExplainedHit::expanded`),
        // so a bare length assertion would be pinning that expansion
        // behaviour instead of "both facts exist".
        let top = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("episodetoplevelfactmarker"))
            .await
            .unwrap();
        assert!(
            top.iter()
                .any(|h| h.hit.retrieved.as_text() == "episodetoplevelfactmarker"),
            "{out}"
        );
        let nested = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("episodenestedfactmarker"))
            .await
            .unwrap();
        assert!(
            nested
                .iter()
                .any(|h| h.hit.retrieved.as_text() == "episodenestedfactmarker"),
            "{out}"
        );

        // The edge exists: relating the same ordered triple again reports
        // "already relates", the existing idempotency signal `relate` uses.
        let relate_again = server
            .relate(Parameters(RelateArgs {
                from: top_id,
                to: nested_id,
                kind: "relates_to".to_string(),
            }))
            .await;
        assert!(
            relate_again.contains("already relates"),
            "edge missing: {relate_again}"
        );
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_supersedes_edge_kind() {
        // Given an episode whose one edge asserts the reserved `supersedes`
        // type
        let server = plain_server().await;
        let marker = "episode supersedes rejection content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("episode supersedes nested content")],
                    entities: vec![],
                    edges: vec![episode_edge("fact:0", "fact:1", "supersedes")],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused, and nothing lands, not even the top-level fact
        assert!(out.contains("reserved"), "{out}");
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_a_mentions_edge_with_a_fresh_entity_as_to() {
        // Given an episode whose one edge is a `mentions` edge with a fresh
        // entity as `to` (backwards: `to` should be a fact) and a
        // handle-shaped `from`, so only the `to`-is-entity check fires,
        // independent of the `from`-is-fact check
        let server = plain_server().await;
        let existing_handle = seed(
            &server,
            "fact",
            "Existing handle target",
            "backwards mentions to-entity handle content",
        )
        .await;
        let marker = "backwards mentions to entity content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Backwards To Entity")],
                    edges: vec![episode_edge(&existing_handle, "entity:0", "mentions")],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused, naming the edge index and the backwards
        // direction, and nothing lands, not even the top-level fact
        assert!(out.contains("edge 0"), "{out}");
        assert!(out.contains("backwards"), "{out}");
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_a_mentions_edge_with_a_fresh_fact_as_from() {
        // Given an episode whose one edge is a `mentions` edge with a fresh
        // fact as `from` (backwards: `from` should be an entity) and a
        // handle-shaped `to`, so only the `from`-is-fact check fires,
        // independent of the `to`-is-entity check
        let server = plain_server().await;
        let existing_handle = seed(
            &server,
            "fact",
            "Existing handle target",
            "backwards mentions from-fact handle content",
        )
        .await;
        let marker = "backwards mentions from fact content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("backwards mentions nested fact content")],
                    entities: vec![],
                    edges: vec![episode_edge("fact:1", &existing_handle, "mentions")],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused, naming the edge index and the backwards
        // direction, and nothing lands, not even the top-level fact
        assert!(out.contains("edge 0"), "{out}");
        assert!(out.contains("backwards"), "{out}");
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_writes_a_mentions_edge_whose_from_is_a_handle_reference_to_an_existing_entity(
    ) {
        // Given a live entity from a prior call, referenced afterward by its
        // handle rather than an episode-local `entity:N` index
        let server = plain_server().await;
        let prior_out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Handle Mentions Entity")],
                    edges: vec![],
                }),
                ..remember_args("handle mentions entity top content")
            }))
            .await;
        assert!(!prior_out.contains("failed"), "{prior_out}");
        let entity_handle = prior_out
            .lines()
            .nth(1)
            .expect("entity line")
            .trim_start_matches("remembered ")
            .to_string();

        // When a mentions edge uses that handle as `from` and a fresh fact
        // as `to`: the correct direction, just with `from` not "entity:N"
        // syntax
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("handle mentions fact content")],
                    entities: vec![],
                    edges: vec![episode_edge(&entity_handle, "fact:1", "mentions")],
                }),
                ..remember_args("handle mentions second top content")
            }))
            .await;

        // Then it succeeds, and the edge is positively confirmed written: a
        // "related" line names the exact from/to pair, not just the absence
        // of a rejection string
        assert!(!out.contains("failed"), "{out}");
        let fact_id = out
            .lines()
            .nth(1)
            .expect("fact line")
            .trim_start_matches("remembered ")
            .to_string();
        assert!(
            out.contains(&format!("related {entity_handle} -mentions-> {fact_id}")),
            "expected the mentions edge confirmed in the response: {out}"
        );
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_out_of_range_confidence_on_a_nested_fact() {
        // Given an episode whose nested fact carries an out-of-range
        // confidence
        let server = plain_server().await;
        let marker = "episode nested confidence rejection content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![EpisodeFactArgs {
                        confidence: Some(5.0),
                        ..episode_fact("episode nested confidence nested content")
                    }],
                    entities: vec![],
                    edges: vec![],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused, and nothing lands: per-item validation runs,
        // not just the top-level field's
        assert!(out.contains("confidence"), "{out}");
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_oversized_content_on_a_nested_fact() {
        // Given an episode whose nested fact's content is one character over
        // MAX_CONTENT_CHARS, among otherwise-valid facts
        let server = plain_server().await;
        let marker = "episode nested content ceiling rejection content";
        let over_cap = "x".repeat(MAX_CONTENT_CHARS + 1);
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact(&over_cap)],
                    entities: vec![],
                    edges: vec![],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused with the fact-prefixed content problem, and
        // nothing lands: per-item content validation runs, not just the
        // top-level field's
        assert!(out.contains("fact:1: content exceeds"), "{out}");
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_with_episode_reports_both_content_and_confidence_problems_on_the_same_fact() {
        // Given a single episode fact with BOTH oversized content AND an
        // out-of-range confidence
        let server = plain_server().await;
        let marker = "episode nested content and confidence content";
        let over_cap = "x".repeat(MAX_CONTENT_CHARS + 1);

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![EpisodeFactArgs {
                        confidence: Some(5.0),
                        ..episode_fact(&over_cap)
                    }],
                    entities: vec![],
                    edges: vec![],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then the response reports BOTH problems under the same "fact:1"
        // prefix, proving accumulation on a single item rather than
        // short-circuiting at the first problem found
        assert!(out.contains("fact:1: content exceeds"), "{out}");
        assert!(out.contains("fact:1: confidence must be"), "{out}");
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_an_out_of_bounds_fact_index() {
        // Given an episode with two valid combined fact indices (0 = the
        // top-level fact, 1 = the one episode.facts entry) and an edge
        // naming "fact:5", well past the valid 0..=1 range
        let server = plain_server().await;
        let marker = "episode out of bounds index rejection content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("episode out of bounds nested content")],
                    entities: vec![],
                    edges: vec![episode_edge("fact:0", "fact:5", "mentions")],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused specifically for that reference, and nothing
        // lands
        assert!(
            out.contains("not a recognized reference") && out.contains("fact:5"),
            "{out}"
        );
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");

        // And the true boundary, one past the last valid index, is rejected
        // too: "fact:2" is not a valid combined index in this same
        // 2-valid-index episode, so a one-off bug in parse_fact_ref's bound
        // check (e.g. "<=" instead of "<") would wrongly accept it while
        // still correctly rejecting the far-out-of-range "fact:5" above
        let boundary_marker = "episode boundary index rejection content";
        let boundary_out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("episode boundary nested content")],
                    entities: vec![],
                    edges: vec![episode_edge("fact:0", "fact:2", "mentions")],
                }),
                ..remember_args(boundary_marker)
            }))
            .await;
        assert!(
            boundary_out.contains("not a recognized reference") && boundary_out.contains("fact:2"),
            "{boundary_out}"
        );
        let boundary_after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(boundary_marker))
            .await
            .unwrap()
            .len();
        assert_eq!(
            boundary_after, 0,
            "wrote a node despite boundary rejection: {boundary_out}"
        );
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_an_unrecognized_reference_form() {
        // Given an episode whose edge references "bogus:0", deliberately not
        // "entity:0" (which becomes valid from WU-5 on)
        let server = plain_server().await;
        let marker = "episode unrecognized reference rejection content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![],
                    edges: vec![episode_edge("fact:0", "bogus:0", "mentions")],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused specifically for that reference, and nothing
        // lands
        assert!(
            out.contains("not a recognized reference") && out.contains("bogus:0"),
            "{out}"
        );
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_with_episode_accepts_a_whitespace_padded_handle_reference() {
        // Given a pre-existing handle target, and an episode edge that names
        // it with incidental leading/trailing whitespace: is_handle_shaped
        // must trim before its alphanumeric check, the same way
        // Graph::resolve_handle trims before its own, or this reference is
        // wrongly rejected before resolve_handle ever runs
        let server = plain_server().await;
        let existing_handle = seed(
            &server,
            "fact",
            "Existing handle target",
            "padded handle existing content",
        )
        .await;
        let marker = "padded handle top content";

        // When remember is called with the edge's "to" padded with whitespace
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![],
                    edges: vec![episode_edge(
                        "fact:0",
                        &format!("  {existing_handle}  "),
                        "relates_to",
                    )],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is accepted, not refused as "not a recognized reference",
        // and the edge resolves to the existing handle's real id
        assert!(!out.contains("failed"), "{out}");
        let top_id = out
            .lines()
            .next()
            .expect("remembered line")
            .trim_start_matches("remembered ")
            .to_string();
        assert!(
            out.contains(&format!("related {top_id} -relates_to-> {existing_handle}")),
            "{out}"
        );
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_a_request_over_max_episode_items() {
        // Given an episode one item over MAX_EPISODE_ITEMS once the
        // always-present top-level fact is counted
        let server = plain_server().await;
        let marker = "episode over max rejection content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        let facts: Vec<EpisodeFactArgs> = (0..MAX_EPISODE_ITEMS)
            .map(|i| episode_fact(&format!("episode over max nested content {i}")))
            .collect();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts,
                    entities: vec![],
                    edges: vec![],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused before any DB call, and nothing lands
        assert!(out.contains("failed"), "{out}");
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");

        // Given the same shape, but exactly at the limit (one fewer fact)
        let at_cap_marker = "episode at max cap content";
        let facts: Vec<EpisodeFactArgs> = (0..MAX_EPISODE_ITEMS - 1)
            .map(|i| episode_fact(&format!("episode at max cap nested content {i}")))
            .collect();

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts,
                    entities: vec![],
                    edges: vec![],
                }),
                ..remember_args(at_cap_marker)
            }))
            .await;

        // Then the limit itself is accepted, not an off-by-one
        assert!(!out.contains("failed"), "{out}");
    }

    #[tokio::test]
    async fn remember_with_episode_reports_an_edge_to_a_since_superseded_fact_as_a_failure() {
        // Given two episode facts sharing a subject, so the second
        // supersedes the first per `Graph::ingest_episode`'s own dedup
        // rules, plus an edge referencing the first by its original combined
        // index
        let server = plain_server().await;
        let top_marker = "episode superseded edge top content";
        let first_marker = "episode superseded edge first content";
        let second_marker = "episode superseded edge second content";

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![
                        EpisodeFactArgs {
                            subject: Some("episode-supersede-subject".to_string()),
                            ..episode_fact(first_marker)
                        },
                        EpisodeFactArgs {
                            subject: Some("episode-supersede-subject".to_string()),
                            ..episode_fact(second_marker)
                        },
                    ],
                    entities: vec![],
                    edges: vec![episode_edge("fact:1", "fact:2", "relates_to")],
                }),
                ..remember_args(top_marker)
            }))
            .await;

        // Then the whole call fails, naming the failing reference (the dead
        // node's id, embedded in `Graph::ingest_episode`'s own "is not live"
        // message), not a generic "remember failed" with no detail
        assert!(out.starts_with("remember failed:"), "{out}");
        assert!(out.contains("is not live"), "{out}");

        // And nothing landed: not the top-level fact, not either episode
        // fact, proving the whole episode rolled back together
        for marker in [top_marker, first_marker, second_marker] {
            let hits = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker))
                .await
                .unwrap();
            assert_eq!(
                hits.len(),
                0,
                "{marker} wrote a node despite rejection: {out}"
            );
        }
    }

    #[tokio::test]
    async fn remember_with_episode_entities_upserts_by_name_across_calls() {
        // Given two SEPARATE remember calls, each naming entity "Alice" in
        // its own episode
        let server = plain_server().await;

        // When the first call remembers "Alice"
        let first_out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Alice")],
                    edges: vec![],
                }),
                ..remember_args("episode entity first call marker")
            }))
            .await;
        assert!(!first_out.contains("failed"), "{first_out}");
        let first_entity_id = first_out
            .lines()
            .nth(1)
            .expect("entity line")
            .trim_start_matches("remembered ")
            .to_string();

        // And the second call remembers "Alice" again, in a wholly separate
        // episode
        let second_out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Alice")],
                    edges: vec![],
                }),
                ..remember_args("episode entity second call marker")
            }))
            .await;
        assert!(!second_out.contains("failed"), "{second_out}");
        let second_entity_id = second_out
            .lines()
            .nth(1)
            .expect("entity line")
            .trim_start_matches("remembered ")
            .to_string();

        // Then the second supersedes the first rather than duplicating: the
        // first id is no longer live (resolve_handle only finds live rows),
        // the second is
        assert_ne!(
            first_entity_id, second_entity_id,
            "expected two distinct ids"
        );
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(&first_entity_id)
                .await
                .is_err(),
            "first entity's id is still live; expected the second call to supersede it"
        );
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(&second_entity_id)
                .await
                .is_ok(),
            "second entity's id should be live"
        );
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
        let server = MemoryServer::new(
            Arc::new(store),
            embedder.clone(),
            Arc::new(liam_model::IdentityReranker),
            llm,
            30,
            false,
            8192,
            1,
        );
        (server, embedder)
    }

    #[tokio::test]
    async fn recall_embeds_the_query_as_a_query() {
        // Arrange
        let (server, embedder) = server_with_recording_embedder().await;

        // Act
        server
            .recall(Parameters(RecallArgs {
                query: "where do I live".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert
        assert_eq!(
            embedder.calls(),
            vec![("where do I live".to_string(), TextKind::Query)]
        );
    }

    #[tokio::test]
    async fn ask_embeds_the_question_as_a_query() {
        // Arrange
        let (server, embedder) = server_with_recording_embedder().await;

        // Act
        server
            .ask(Parameters(AskArgs {
                question: "where do I live".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert
        assert_eq!(
            embedder.calls(),
            vec![("where do I live".to_string(), TextKind::Query)]
        );
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

    fn ask_args_for(question: &str) -> AskArgs {
        AskArgs {
            question: question.to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }
    }

    fn hundred_words() -> String {
        (0..100)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[tokio::test]
    async fn recall_and_ask_reject_invalid_queries_with_exact_replies() {
        // Arrange
        let server = server_with_recording_embedder().await.0;
        let cases: Vec<(String, &str)> = vec![
            (String::new(), "query is empty"),
            ("   \n\t".to_string(), "query is empty"),
            ("alpha\0port".to_string(), "query contains a NUL character"),
            ("x".repeat(4001), "query is longer than 4000 characters"),
            (
                "\u{e9}".repeat(4001),
                "query is longer than 4000 characters",
            ),
        ];

        for (text, problem) in cases {
            // Act
            let recall = server.recall(Parameters(recall_args_for(&text))).await;
            let ask = server.ask(Parameters(ask_args_for(&text))).await;

            // Assert
            assert_eq!(recall, format!("recall failed: {problem}"));
            assert_eq!(ask, format!("ask failed: {problem}"));
        }
    }

    #[tokio::test]
    async fn recall_and_ask_accept_queries_at_the_character_limit() {
        // Arrange
        let server = server_with_recording_embedder().await.0;
        let cases = ["x".repeat(4000), "\u{e9}".repeat(4000)];

        for text in cases {
            assert_eq!(text.chars().count(), 4000);

            // Act
            let recall = server.recall(Parameters(recall_args_for(&text))).await;
            let ask = server.ask(Parameters(ask_args_for(&text))).await;

            // Assert
            assert!(!recall.starts_with("recall failed"), "recall: {recall}");
            assert!(!ask.starts_with("ask failed"), "ask: {ask}");
        }
    }

    #[test]
    fn lexical_text_keeps_the_first_64_distinct_words_in_order() {
        // Arrange
        let question = hundred_words();

        // Act
        let out = lexical_text(&question);

        // Assert
        let expected: Vec<String> = (0..64).map(|i| format!("w{i}")).collect();
        let got: Vec<&str> = out.split_whitespace().collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn lexical_text_treats_case_variants_as_one_term() {
        // Arrange
        let question = "Word word WORD";

        // Act
        let out = lexical_text(question);

        // Assert
        assert_eq!(out.split_whitespace().count(), 1);
    }

    #[tokio::test]
    async fn recall_and_ask_embed_the_full_question_while_lexical_text_is_capped() {
        // Arrange
        let question = hundred_words();
        let (server, embedder) = server_with_recording_embedder().await;

        // Act
        server.recall(Parameters(recall_args_for(&question))).await;
        server.ask(Parameters(ask_args_for(&question))).await;

        // Assert
        let calls = embedder.calls();
        assert_eq!(calls.len(), 2);
        for (text, kind) in &calls {
            assert_eq!(text, &question);
            assert_eq!(*kind, TextKind::Query);
        }
        assert_eq!(lexical_text(&question).split_whitespace().count(), 64);
    }

    #[tokio::test]
    async fn remember_embeds_the_content_as_a_document() {
        // Arrange
        let (server, embedder) = server_with_recording_embedder().await;

        // Act
        let out = server
            .remember(Parameters(remember_args("I live in Lisbon")))
            .await;

        // Assert
        assert!(!out.contains("failed"), "{out}");
        assert_eq!(
            embedder.calls(),
            vec![("I live in Lisbon".to_string(), TextKind::Document)]
        );
    }

    #[tokio::test]
    async fn remember_episode_embeds_the_top_fact_and_every_nested_fact_as_documents() {
        // Arrange
        let (server, embedder) = server_with_recording_embedder().await;

        // Act
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("nested fact content")],
                    entities: vec![],
                    edges: vec![],
                }),
                ..remember_args("top fact content")
            }))
            .await;

        // Assert
        assert!(!out.contains("failed"), "{out}");
        assert_eq!(
            embedder.calls(),
            vec![
                ("top fact content".to_string(), TextKind::Document),
                ("nested fact content".to_string(), TextKind::Document),
            ]
        );
    }

    #[tokio::test]
    async fn resynthesize_entity_embeds_the_new_content_as_a_document() {
        // Arrange
        let (server, embedder) =
            server_with_recording_embedder_and_llm(Arc::new(CountingLlm::new())).await;
        let entity_id = server
            .store()
            .await
            .unwrap()
            .upsert_by(NewNode::entity("person", "Kim"))
            .await
            .expect("seed entity");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        server
            .resynthesize_entity(entity_id, deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
            .await
            .expect("resynthesis should succeed");

        // Assert
        let calls = embedder.calls();
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].1, TextKind::Document);
    }

    /// A page that mentions three facts: two written in one episode, so one
    /// event, and a third written on its own.
    async fn page_with_three_mentions(store: &DefaultGraph) -> liam_store::NodeId {
        let page = store
            .upsert_by(NewNode::entity("person", "Kim"))
            .await
            .unwrap();
        let episode = store
            .ingest_episode(
                vec![
                    NewNode::now("fact", "Role", "Kim works as an engineer."),
                    NewNode::now("fact", "Home", "Kim lives in Lisbon."),
                ],
                vec![],
            )
            .await
            .unwrap();
        let alone = store
            .insert(NewNode::now("fact", "Pet", "Kim has a cat."))
            .await
            .unwrap();
        for id in episode.node_ids.iter().chain([&alone]) {
            store.relate(&page, id, relation::MENTIONS).await.unwrap();
        }
        page
    }

    /// The basis of the live page labelled Kim.
    async fn basis_of_kim(store: &DefaultGraph) -> Vec<EventRef> {
        let hits = store
            .query_explained(&Query::text("person Kim").with_inferred(true))
            .await
            .unwrap();
        let written = hits
            .iter()
            .find(|h| h.hit.label == "Kim")
            .expect("resynthesized page missing");
        store
            .node_event_ids(&written.hit.id, EventRole::Basis)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn resynthesize_entity_lists_each_mentions_origin_event_once_as_the_pages_basis() {
        // Arrange
        let llm = Arc::new(CountingLlm::new());
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm).await;
        let store = server.store().await.unwrap();
        let page = page_with_three_mentions(&store).await;
        let mut expected = std::collections::BTreeSet::new();
        for mention in store
            .mentions(&page, liam_store::Millis::now(), 10)
            .await
            .unwrap()
        {
            expected.extend(
                store
                    .node_event_ids(&mention.id, EventRole::Origin)
                    .await
                    .unwrap(),
            );
        }
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        server
            .resynthesize_entity(page, deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
            .await
            .expect("resynthesis should succeed");

        // Assert
        assert_eq!(expected.len(), 2);
        assert_eq!(
            basis_of_kim(&store).await,
            expected.into_iter().collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn resynthesize_entity_leaves_a_mention_the_budget_dropped_out_of_the_basis() {
        // Arrange: a context budget that keeps only the first mention
        let llm = Arc::new(CountingLlm::new());
        let server =
            server_with_timeout(Arc::new(liam_model::IdentityReranker), llm, 30, false, 1).await;
        let store = server.store().await.unwrap();
        let page = page_with_three_mentions(&store).await;
        let mentions = store
            .mentions(&page, liam_store::Millis::now(), 10)
            .await
            .unwrap();
        let kept = store
            .node_event_ids(&mentions[0].id, EventRole::Origin)
            .await
            .unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        server
            .resynthesize_entity(page, deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
            .await
            .expect("resynthesis should succeed");

        // Assert
        assert_eq!(basis_of_kim(&store).await, kept);
    }

    #[tokio::test]
    async fn resynthesize_entity_as_writes_the_page_under_the_given_source() {
        // Arrange
        let (server, _embedder) =
            server_with_recording_embedder_and_llm(Arc::new(CountingLlm::new())).await;
        let entity_id = server
            .store()
            .await
            .unwrap()
            .upsert_by(NewNode::entity("person", "Kim"))
            .await
            .expect("seed entity");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        server
            .resynthesize_entity_as(
                &server.store().await.unwrap(),
                entity_id.clone(),
                deadline,
                ENTITY_SYNTHESIS_MAX_NEW_TOKENS,
                Some("consolidation:9"),
            )
            .await
            .expect("resynthesis should succeed");

        // Assert
        let pages = server
            .store()
            .await
            .unwrap()
            .changes_since(liam_store::BEGINNING)
            .await
            .expect("changes");
        let newest = pages.iter().rfind(|change| !change.closed).expect("a page");
        let page = server
            .store()
            .await
            .unwrap()
            .get(&newest.id, liam_store::Millis(liam_store::FOREVER.0 - 1))
            .await
            .expect("get")
            .expect("live page");
        assert_eq!(page.producer, "consolidation:9");
    }

    #[tokio::test]
    async fn remember_with_episode_entities_get_no_embedding() {
        // Given a server wired with a call-counting embedder, and an
        // episode with one nested fact and two entities
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let embedder = Arc::new(CountingEmbedder::new(8));
        let server = MemoryServer::new(
            Arc::new(store),
            embedder.clone(),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        );

        // When remember is called with 1 nested fact and 2 entities
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("episode entity embedding nested content")],
                    entities: vec![
                        episode_entity("person", "Bob"),
                        episode_entity("person", "Carol"),
                    ],
                    edges: vec![],
                }),
                ..remember_args("episode entity embedding top content")
            }))
            .await;

        // Then it succeeds, and the embedder was called exactly once per
        // fact (the top-level fact plus the one nested fact), never for
        // either entity
        assert!(!out.contains("failed"), "{out}");
        assert_eq!(
            embedder.call_count(),
            2,
            "expected 1 (top-level) + 1 (nested fact) = 2 embed calls, entities must not embed"
        );
    }

    #[tokio::test]
    async fn remember_with_episode_edge_references_an_entity_by_index() {
        // Given one entity and an edge "fact:0" -> "entity:0". Kind
        // "relates_to", not "mentions", so the entity stays live below.
        let server = plain_server().await;

        // When remember is called
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Dave")],
                    edges: vec![episode_edge("fact:0", "entity:0", "relates_to")],
                }),
                ..remember_args("episode entity edge top content")
            }))
            .await;

        // Then it succeeds, the entity exists, and the edge's reported type
        // is exactly what was sent
        assert!(!out.contains("failed"), "{out}");
        let mut lines = out.lines();
        let top_id = lines
            .next()
            .expect("top-level fact line")
            .trim_start_matches("remembered ")
            .to_string();
        let entity_id = lines
            .next()
            .expect("entity line")
            .trim_start_matches("remembered ")
            .to_string();
        assert!(
            out.contains(&format!("related {top_id} -relates_to-> {entity_id}")),
            "{out}"
        );

        // And the edge exists: relating it again reports "already relates".
        let relate_again = server
            .relate(Parameters(RelateArgs {
                from: top_id,
                to: entity_id,
                kind: "relates_to".to_string(),
            }))
            .await;
        assert!(
            relate_again.contains("already relates"),
            "edge missing: {relate_again}"
        );
    }

    #[tokio::test]
    async fn remember_with_episode_edge_references_the_correct_entity_among_several() {
        // Given one nested fact and two entities, with a correctly-directed
        // mentions edge from "entity:1" to "fact:1". Combined indices: 0 =
        // top-level fact, 1 = the nested fact, 2 = entity 0, 3 = entity 1, so
        // this exercises the nontrivial `1 + fact_count + j` arithmetic on
        // the entity side. Uses `CountingLlm` instead of `plain_server`'s
        // `MockLlm` because this episode triggers entity synthesis, which
        // `MockLlm`'s echo cannot pass `is_grounded`.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(CountingLlm::new()),
        )
        .await;

        // When remember is called
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("episode entity among several nested content")],
                    entities: vec![
                        episode_entity("person", "Eve"),
                        episode_entity("person", "Frank"),
                    ],
                    edges: vec![episode_edge("entity:1", "fact:1", "mentions")],
                }),
                ..remember_args("episode entity among several top content")
            }))
            .await;

        // Then it succeeds, and the edge's source is SPECIFICALLY the
        // second entity's node, not the first's
        assert!(!out.contains("failed"), "{out}");
        let mut lines = out.lines();
        let top_id = lines
            .next()
            .expect("top-level fact line")
            .trim_start_matches("remembered ")
            .to_string();
        let fact_id = lines
            .next()
            .expect("nested fact line")
            .trim_start_matches("remembered ")
            .to_string();
        let entity0_id = lines
            .next()
            .expect("first entity line")
            .trim_start_matches("remembered ")
            .to_string();
        let entity1_id = lines
            .next()
            .expect("second entity line")
            .trim_start_matches("remembered ")
            .to_string();
        assert!(
            out.contains(&format!("related {entity1_id} -mentions-> {fact_id}")),
            "{out}"
        );
        for wrong in [&top_id, &entity0_id] {
            assert!(
                !out.contains(&format!("related {wrong} -mentions-> {fact_id}")),
                "edge sourced from the wrong node {wrong}: {out}"
            );
        }
    }

    #[tokio::test]
    async fn remember_with_episode_rejects_a_request_over_max_episode_items_counting_entities() {
        // Given an episode where facts.len() + edges.len() alone is under
        // MAX_EPISODE_ITEMS, but adding entities.len() pushes the total
        // over
        let server = plain_server().await;
        let marker = "episode entities over max rejection content";
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        let facts: Vec<EpisodeFactArgs> = (0..MAX_EPISODE_ITEMS / 2)
            .map(|i| episode_fact(&format!("episode entities over max nested content {i}")))
            .collect();
        let entities: Vec<EpisodeEntityArgs> = (0..(MAX_EPISODE_ITEMS - facts.len()))
            .map(|i| episode_entity("person", &format!("Entity {i}")))
            .collect();
        assert!(
            facts.len() < MAX_EPISODE_ITEMS,
            "facts + edges alone must stay under the cap for this test to prove anything"
        );

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts,
                    entities,
                    edges: vec![],
                }),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused, and nothing lands
        assert!(out.contains("failed"), "{out}");
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(after, before, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_with_episode_reports_an_edge_to_a_since_superseded_entity_as_a_failure() {
        // Given two entities naming the same person, differing only in case
        // and trailing whitespace (so `NewNode::entity`'s
        // `name.trim().to_lowercase()` normalization, not an accidental
        // exact-string match, is what makes them collide), plus an edge
        // referencing the first by its original combined index
        let server = plain_server().await;
        let top_marker = "episode superseded entity edge top content";

        // When remember is called with it
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![
                        episode_entity("person", "Alice"),
                        episode_entity("person", "alice "),
                    ],
                    edges: vec![episode_edge("entity:0", "entity:1", "relates_to")],
                }),
                ..remember_args(top_marker)
            }))
            .await;

        // Then the whole call fails, naming the failing reference (the dead
        // node's id, embedded in `Graph::ingest_episode`'s own "is not live"
        // message)
        assert!(out.starts_with("remember failed:"), "{out}");
        assert!(out.contains("is not live"), "{out}");

        // And nothing landed, not even the top-level fact, proving the
        // whole episode rolled back together
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(top_marker))
            .await
            .unwrap();
        assert_eq!(after.len(), 0, "wrote a node despite rejection: {out}");
    }

    #[tokio::test]
    async fn remember_synthesizes_a_mentioned_entity_exactly_once_despite_two_mentions_edges() {
        // Given an episode where 2 separate facts each carry a mentions
        // edge to the SAME fresh entity
        let llm = Arc::new(CountingLlm::new());
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;

        // When remember runs
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![
                        episode_fact("dedup trigger fact one content"),
                        episode_fact("dedup trigger fact two content"),
                    ],
                    entities: vec![episode_entity("person", "Dedup Entity")],
                    edges: vec![
                        episode_edge("entity:0", "fact:1", "mentions"),
                        episode_edge("entity:0", "fact:2", "mentions"),
                    ],
                }),
                ..remember_args("dedup trigger top content")
            }))
            .await;

        // Then synthesis ran exactly once for that entity, not twice
        assert!(!out.contains("synthesis failed"), "{out}");
        assert_eq!(
            llm.call_count(),
            1,
            "expected exactly one synthesis call: {out}"
        );
    }

    #[tokio::test]
    async fn remember_synthesizes_every_freshly_mentioned_entity_before_returning() {
        // Given an episode mentioning 2 distinct fresh entities
        let llm = Arc::new(CountingLlm::new());
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;

        // When remember runs
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![
                        episode_entity("person", "Both One"),
                        episode_entity("person", "Both Two"),
                    ],
                    edges: vec![
                        episode_edge("entity:0", "fact:0", "mentions"),
                        episode_edge("entity:1", "fact:0", "mentions"),
                    ],
                }),
                ..remember_args("both entities top content")
            }))
            .await;

        // Then both synthesized, and both completed before returning: the
        // ORIGINAL entity ids are already superseded by their own writes
        assert!(!out.contains("synthesis failed"), "{out}");
        assert_eq!(llm.call_count(), 2, "{out}");
        let entity0_id = out
            .lines()
            .nth(1)
            .expect("first entity line")
            .trim_start_matches("remembered ")
            .to_string();
        let entity1_id = out
            .lines()
            .nth(2)
            .expect("second entity line")
            .trim_start_matches("remembered ")
            .to_string();
        for id in [&entity0_id, &entity1_id] {
            assert!(
                server
                    .store()
                    .await
                    .unwrap()
                    .resolve_handle(id)
                    .await
                    .is_err(),
                "original entity id should have been superseded by its own resynthesis: {out}"
            );
        }
        for label in ["Both One", "Both Two"] {
            let expected_content = format!("person {label}");
            let hits = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(expected_content.as_str()).with_inferred(true))
                .await
                .unwrap();
            assert!(
                hits.iter()
                    .any(|h| h.hit.label == label && h.hit.retrieved.as_text() == expected_content),
                "{label}'s resynthesized content should already be queryable: {out}"
            );
        }
    }

    #[tokio::test]
    async fn resynthesize_entity_preserves_existing_attributes_on_the_new_node() {
        // Given a live entity that already carries attributes
        let llm = Arc::new(CountingLlm::new());
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        let entity_id = server
            .store()
            .await
            .unwrap()
            .upsert_by(
                NewNode::entity("person", "Kim").with_attributes(json!({"role": "engineer"})),
            )
            .await
            .expect("seed entity with attributes");

        // When its resynthesis runs, the same per-entity flow `remember`
        // triggers for a freshly mentioned entity
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        server
            .resynthesize_entity(entity_id.clone(), deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
            .await
            .expect("resynthesis should succeed");

        // Then the new node's attributes match the old one's, not wiped
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(entity_id.as_str())
                .await
                .is_err(),
            "the old version should have been superseded"
        );
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("person Kim").with_inferred(true))
            .await
            .unwrap();
        let new_entity = hits
            .iter()
            .find(|h| h.hit.label == "Kim")
            .expect("resynthesized entity missing");
        assert_eq!(new_entity.hit.retrieved.as_text(), "person Kim");
        assert_eq!(new_entity.hit.attributes, json!({"role": "engineer"}));
    }

    #[tokio::test]
    async fn resynthesize_entity_preserves_the_original_entitys_scope() {
        // Given a live entity scoped to "proj-x"
        let llm = Arc::new(CountingLlm::new());
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        let entity_id = server
            .store()
            .await
            .unwrap()
            .upsert_by(NewNode::entity("person", "Priya").with_scope("proj-x"))
            .await
            .expect("seed scoped entity");

        // When its resynthesis runs, the same per-entity flow `remember`
        // triggers for a freshly mentioned entity
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        server
            .resynthesize_entity(entity_id.clone(), deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
            .await
            .expect("resynthesis should succeed");

        // Then the original id is genuinely superseded, not left live
        // alongside a second, unscoped duplicate: a scope-dropping
        // rebuild fails to match the live "proj-x" competitor by subject
        // AND scope, so `upsert_by` falls through to `insert` and the
        // old id survives
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(entity_id.as_str())
                .await
                .is_err(),
            "the old scoped version should have been superseded, not forked"
        );

        // And the resynthesized node kept the same scope: it is visible
        // to a proj-x-scoped query...
        let scoped = server
            .store()
            .await
            .unwrap()
            .query_explained(
                &Query::text("person Priya")
                    .with_scope("proj-x")
                    .with_inferred(true),
            )
            .await
            .unwrap();
        assert!(
            scoped.iter().any(|h| h.hit.label == "Priya"),
            "resynthesized entity missing from its own proj-x scope"
        );

        // ...but not to a different scope's query, the isolation
        // guarantee a scope-dropping rebuild would silently break
        let other_scope = server
            .store()
            .await
            .unwrap()
            .query_explained(
                &Query::text("person Priya")
                    .with_scope("proj-y")
                    .with_inferred(true),
            )
            .await
            .unwrap();
        assert!(
            !other_scope.iter().any(|h| h.hit.label == "Priya"),
            "resynthesized entity leaked into the proj-y scope"
        );
    }

    #[tokio::test]
    async fn resynthesize_entity_honors_the_deadline_it_is_given_not_a_fresh_one() {
        // Given an already-elapsed deadline, a sole permit held all test long,
        // and an hour-long `ask_timeout_secs` a fresh deadline would never miss.
        let llm = Arc::new(CountingLlm::new());
        let server = server_with_generation_limit(
            Arc::new(liam_model::IdentityReranker),
            llm,
            3600,
            false,
            8192,
            1,
        )
        .await;
        let _held = server
            .generation_permits
            .clone()
            .acquire_owned()
            .await
            .expect("hold the sole permit");
        let entity_id = server
            .store()
            .await
            .unwrap()
            .upsert_by(NewNode::entity("person", "Deadline Carrier"))
            .await
            .expect("seed entity");
        let already_elapsed = tokio::time::Instant::now() - Duration::from_millis(1);

        // When resynthesis runs with that deadline, guarded by a short real
        // timeout so a regression that recomputes its own fails, not hangs
        let outcome = tokio::time::timeout(
            Duration::from_millis(200),
            server.resynthesize_entity(entity_id, already_elapsed, ENTITY_SYNTHESIS_MAX_NEW_TOKENS),
        )
        .await
        .expect("must fail fast on the passed deadline, not hang on a freshly computed one");

        // Then it fails immediately for having missed THAT deadline, proving
        // the passed-in value governed the wait rather than a fresh one
        let err = outcome.expect_err("an already-elapsed deadline must not grant a permit");
        assert!(
            err.contains("timed out waiting for a generation slot"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn remember_still_commits_the_episode_when_entity_synthesis_fails() {
        // Given a mock llm that always errors
        let llm = Arc::new(FailingLlm);
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm).await;

        // When remember runs an episode mentioning a fresh entity
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Failing Entity")],
                    edges: vec![episode_edge("entity:0", "fact:0", "mentions")],
                }),
                ..remember_args("synthesis failure top content")
            }))
            .await;

        // Then the episode's facts/entities/edges are still fully
        // committed
        assert!(!out.starts_with("remember failed:"), "{out}");
        let top = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("synthesis failure top content"))
            .await
            .unwrap();
        assert!(
            top.iter()
                .any(|h| h.hit.retrieved.as_text() == "synthesis failure top content"),
            "{out}"
        );
        let entity_id = out
            .lines()
            .nth(1)
            .expect("entity line")
            .trim_start_matches("remembered ")
            .to_string();
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(&entity_id)
                .await
                .is_ok(),
            "entity should still be live: {out}"
        );

        // And the response names the entity synthesis failed for
        assert!(out.contains("synthesis failed for Failing Entity"), "{out}");
    }

    #[tokio::test]
    async fn remember_runs_entity_synthesis_concurrently_not_sequentially() {
        // 2 permits, not the shipped default of 1: only then can peak
        // reach 2, proving true concurrency, not just bounded queuing.
        let release = Arc::new(tokio::sync::Notify::new());
        let llm = Arc::new(GatedLlm::new_grounded(release.clone()));
        let server = Arc::new(
            server_with_generation_limit(
                Arc::new(liam_model::IdentityReranker),
                llm.clone(),
                30,
                false,
                8192,
                2,
            )
            .await,
        );

        // When remember runs an episode mentioning 2 distinct fresh
        // entities, spawned since the gated llm blocks until released
        let remember_server = server.clone();
        let handle = tokio::spawn(async move {
            remember_server
                .remember(Parameters(RememberArgs {
                    episode: Some(EpisodeArgs {
                        facts: vec![],
                        entities: vec![
                            episode_entity("person", "Concurrent One"),
                            episode_entity("person", "Concurrent Two"),
                        ],
                        edges: vec![
                            episode_edge("entity:0", "fact:0", "mentions"),
                            episode_edge("entity:1", "fact:0", "mentions"),
                        ],
                    }),
                    ..remember_args("concurrent entity synthesis top content")
                }))
                .await
        });
        wait_until(|| llm.in_flight() == 2).await;
        release.notify_one();
        release.notify_one();
        let out = handle.await.expect("remember task panicked");

        // Then the PEAK proves both syntheses were in flight together,
        // not one after the other
        assert!(!out.contains("failed"), "{out}");
        assert_eq!(
            llm.peak(),
            2,
            "expected both entity syntheses concurrently in flight: {out}"
        );
    }

    #[tokio::test]
    async fn remember_records_both_outcomes_when_one_of_two_entity_syntheses_fails() {
        // Given a llm that errors on its first call and succeeds after,
        // and an episode mentioning 2 distinct fresh entities
        let llm = Arc::new(FirstCallErrorsLlm::new());
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm).await;

        // When remember runs
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![
                        episode_entity("person", "Mixed One"),
                        episode_entity("person", "Mixed Two"),
                    ],
                    edges: vec![
                        episode_edge("entity:0", "fact:0", "mentions"),
                        episode_edge("entity:1", "fact:0", "mentions"),
                    ],
                }),
                ..remember_args("mixed outcome top content")
            }))
            .await;

        // Then both outcomes are recorded: one entity's content was
        // compiled, the other is named in the failure list. Filtered to an
        // exact content match, not just a lexical hit: the FTS search for
        // "person Mixed One" also surfaces "person Mixed Two" (they share
        // both words), so raw hit count alone would not tell the two apart.
        let mut succeeded: Vec<_> = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("person Mixed One").with_inferred(true))
            .await
            .unwrap()
            .into_iter()
            .filter(|h| h.hit.retrieved.as_text() == "person Mixed One")
            .collect();
        if succeeded.is_empty() {
            succeeded = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text("person Mixed Two").with_inferred(true))
                .await
                .unwrap()
                .into_iter()
                .filter(|h| h.hit.retrieved.as_text() == "person Mixed Two")
                .collect();
        }
        assert_eq!(
            succeeded.len(),
            1,
            "expected exactly one entity to succeed: {out}"
        );
        let survivor_label = succeeded[0].hit.label.clone();
        let failed_label = if survivor_label == "Mixed One" {
            "Mixed Two"
        } else {
            "Mixed One"
        };
        assert!(
            out.contains(&format!("synthesis failed for {failed_label}")),
            "expected the OTHER entity named as failed: {out}"
        );
        assert!(
            !out.contains(&format!("synthesis failed for {survivor_label}")),
            "the successful entity must not also be reported as failed: {out}"
        );
    }

    #[tokio::test]
    async fn remember_does_not_resynthesize_an_existing_entity_referenced_only_by_handle() {
        // Given a live entity from a prior call
        let llm = Arc::new(CountingLlm::new());
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        let prior_out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Handle Only Entity")],
                    edges: vec![],
                }),
                ..remember_args("handle only entity top content")
            }))
            .await;
        assert!(!prior_out.contains("failed"), "{prior_out}");
        let entity_handle = prior_out
            .lines()
            .nth(1)
            .expect("entity line")
            .trim_start_matches("remembered ")
            .to_string();

        // When a second episode mentions the SAME entity only by handle,
        // never as an entity:N reference
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("handle only mention fact content")],
                    entities: vec![],
                    edges: vec![episode_edge(&entity_handle, "fact:1", "mentions")],
                }),
                ..remember_args("handle only second top content")
            }))
            .await;

        // Then synthesis does not run for it
        assert!(!out.contains("failed"), "{out}");
        assert_eq!(
            llm.call_count(),
            0,
            "existing entities referenced only by handle must not trigger resynthesis: {out}"
        );
    }

    #[tokio::test]
    async fn remember_synthesizes_a_mentioned_entity_from_its_real_mention_content() {
        // Given a fresh entity linked to a fact via a correctly-directed
        // mentions edge (`from`: entity, `to`: fact)
        let llm = Arc::new(RecordingLlm::new("person Xyzzy Entity"));
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;

        // When remember runs
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact("the xyzzy-marker-42 distinctive fact content")],
                    entities: vec![episode_entity("person", "Xyzzy Entity")],
                    edges: vec![episode_edge("entity:0", "fact:1", "mentions")],
                }),
                ..remember_args("xyzzy marker top content")
            }))
            .await;

        // Then the fact's real content reached the synthesis prompt, not an
        // empty mentions list
        assert!(!out.contains("synthesis failed"), "{out}");
        assert!(
            llm.last_prompt()
                .contains("the xyzzy-marker-42 distinctive fact content"),
            "expected the mentioned fact's real content in the synthesis prompt: {}",
            llm.last_prompt()
        );
    }

    #[tokio::test]
    async fn remember_with_a_full_mixed_episode_round_trips_through_mcp() {
        // Given a prior, separate `remember` call establishing entity
        // "Grace" as live, and a separate fact remembered before this
        // episode, so this episode's third edge can reference it by its
        // handle string rather than an episode-local index. Uses
        // `CountingLlm` instead of `plain_server`'s `MockLlm` because this
        // episode triggers entity synthesis, which `MockLlm`'s echo cannot
        // pass `is_grounded`.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(CountingLlm::new()),
        )
        .await;

        let prior_entity_out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![episode_entity("person", "Grace")],
                    edges: vec![],
                }),
                ..remember_args("mixed episode prior entity top content")
            }))
            .await;
        assert!(!prior_entity_out.contains("failed"), "{prior_entity_out}");
        let prior_entity_id = prior_entity_out
            .lines()
            .nth(1)
            .expect("prior entity line")
            .trim_start_matches("remembered ")
            .to_string();

        let existing_handle = seed(
            &server,
            "fact",
            "Existing handle target",
            "mixed episode existing handle content",
        )
        .await;

        // When remember is called with a full mixed episode: 2 nested facts
        // (plus the top-level fact makes 3 fact-shaped nodes total), 2
        // entities (one brand new, one superseding "Grace" above), and 3
        // edges: fact-to-fact by index, fact-to-entity by index, and
        // fact-to-existing-handle by handle string
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![
                        episode_fact("mixed episode fact one content"),
                        episode_fact("mixed episode fact two content"),
                    ],
                    entities: vec![
                        episode_entity("person", "Henry"),
                        episode_entity("person", "Grace"),
                    ],
                    edges: vec![
                        episode_edge("fact:1", "fact:2", "supports"),
                        episode_edge("entity:0", "fact:1", "mentions"),
                        episode_edge("fact:2", &existing_handle, "relates_to"),
                    ],
                }),
                ..remember_args("mixed episode top content")
            }))
            .await;

        // Then the response has one "remembered {id}" line per written node
        // (top fact, 2 nested facts, 2 entities, in order) followed by one
        // "related ..." line per edge, in order; ids are read out of the
        // response text itself, since they are runtime-generated ULIDs
        assert!(!out.contains("failed"), "{out}");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 8, "expected 5 remembered + 3 related: {out}");
        for line in &lines[..5] {
            assert!(line.starts_with("remembered "), "{out}");
        }
        for line in &lines[5..] {
            assert!(line.starts_with("related "), "{out}");
        }
        let top_id = lines[0].trim_start_matches("remembered ").to_string();
        let fact1_id = lines[1].trim_start_matches("remembered ").to_string();
        let fact2_id = lines[2].trim_start_matches("remembered ").to_string();
        let henry_id = lines[3].trim_start_matches("remembered ").to_string();
        let grace_id = lines[4].trim_start_matches("remembered ").to_string();

        assert_eq!(
            lines[5],
            format!("related {fact1_id} -supports-> {fact2_id}")
        );
        assert_eq!(
            lines[6],
            format!("related {henry_id} -mentions-> {fact1_id}")
        );
        assert_eq!(
            lines[7],
            format!("related {fact2_id} -relates_to-> {existing_handle}")
        );

        // Every written item is independently recall-able afterward: the
        // top fact and both nested facts by content, the pre-existing
        // handle target too
        for marker in [
            "mixed episode top content",
            "mixed episode fact one content",
            "mixed episode fact two content",
            "mixed episode existing handle content",
        ] {
            let hits = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker))
                .await
                .unwrap();
            assert!(
                hits.iter().any(|h| h.hit.retrieved.as_text() == marker),
                "{marker} missing: {out}"
            );
        }

        // Henry's own id was superseded by its own resynthesis (mentioned
        // via a "mentions" edge); a live, resynthesized Henry still exists
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(&henry_id)
                .await
                .is_err(),
            "Henry's original id should have been superseded by its own resynthesis"
        );
        let resynthesized_henry = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("Henry").with_inferred(true))
            .await
            .unwrap();
        assert!(
            resynthesized_henry.iter().any(|h| h.hit.label == "Henry"),
            "expected a resynthesized, still-live Henry entity: {out}"
        );
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(&grace_id)
                .await
                .is_ok(),
            "superseding entity should be live"
        );
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(&prior_entity_id)
                .await
                .is_err(),
            "prior entity should have been superseded, not left live"
        );
        assert_ne!(prior_entity_id, grace_id, "expected two distinct ids");

        // The other 2 edges still exist, each reporting "already relates";
        // Henry's own edge is checked above instead, since its target is superseded.
        for (from, to, kind) in [
            (fact1_id.clone(), fact2_id.clone(), "supports"),
            (fact2_id.clone(), existing_handle.clone(), "relates_to"),
        ] {
            let relate_again = server
                .relate(Parameters(RelateArgs {
                    from,
                    to,
                    kind: kind.to_string(),
                }))
                .await;
            assert!(
                relate_again.contains("already relates"),
                "edge missing for {kind}: {relate_again}"
            );
        }
        // top_id is written but not referenced by any edge in this episode;
        // confirming it round-trips through recall proves it was still
        // written even though nothing points at it.
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(&top_id)
                .await
                .is_ok(),
            "top-level fact should be live"
        );
    }

    #[tokio::test]
    async fn remember_with_episode_reports_every_validation_problem_before_writing_anything() {
        // Given an episode with two independent problems in the same call:
        // an out-of-range confidence on episode.facts[1], and a reserved
        // edge kind on episode.edges[0]
        let server = plain_server().await;
        let top_marker = "episode multi problem top content";
        let fact0_marker = "episode multi problem fact zero content";
        let fact1_marker = "episode multi problem fact one content";

        // When remember is called with both problems present at once
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![
                        episode_fact(fact0_marker),
                        EpisodeFactArgs {
                            confidence: Some(5.0),
                            ..episode_fact(fact1_marker)
                        },
                    ],
                    entities: vec![],
                    edges: vec![episode_edge("fact:0", "fact:1", "supersedes")],
                }),
                ..remember_args(top_marker)
            }))
            .await;

        // Then the response reports BOTH problems, not just the first one
        // encountered, proving `problems.push(...)` accumulates across the
        // whole validation pass rather than returning at the first failure
        assert!(out.starts_with("remember failed:"), "{out}");
        assert!(
            out.contains("confidence"),
            "confidence problem missing: {out}"
        );
        assert!(out.contains("reserved"), "edge kind problem missing: {out}");

        // And nothing landed at all: not the top-level fact, not either
        // nested fact
        for marker in [top_marker, fact0_marker, fact1_marker] {
            let hits = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker))
                .await
                .unwrap();
            assert_eq!(
                hits.len(),
                0,
                "{marker} wrote a node despite rejection: {out}"
            );
        }
    }

    #[tokio::test]
    async fn remember_with_episode_facts_and_entities_inherit_the_top_level_scope() {
        // Given a top-level fact scoped to "proj-x", plus one nested fact
        // and one entity in the same episode
        let server = plain_server().await;
        let top_marker = "episode scope top content";
        let fact_marker = "episode scope nested content";
        let entity_marker = "Zorbnaxia";

        // When remember is called with `scope: Some("proj-x")` on the
        // top-level args
        let out = server
            .remember(Parameters(RememberArgs {
                scope: Some("proj-x".to_string()),
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact(fact_marker)],
                    entities: vec![episode_entity("person", entity_marker)],
                    edges: vec![],
                }),
                ..remember_args(top_marker)
            }))
            .await;

        // Then it succeeds, and a query scoped to "proj-x" finds every one
        // of the three nodes: the doc comment on `EpisodeFactArgs` promises
        // nested facts "share the top-level fact's scope"; this exercises
        // entities too, which share the same contract. An entity has no
        // content (`NewNode::entity` always leaves it empty), so its
        // marker is matched on `label` instead of `content`.
        assert!(!out.contains("failed"), "{out}");
        for marker in [top_marker, fact_marker, entity_marker] {
            let scoped = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker).with_scope("proj-x").with_inferred(true))
                .await
                .unwrap();
            assert!(
                scoped
                    .iter()
                    .any(|h| h.hit.retrieved.as_text() == marker || h.hit.label == marker),
                "{marker} missing from the proj-x scope: {out}"
            );

            // And the same marker is invisible to a DIFFERENT scope's
            // query, the isolation guarantee `scope` exists for. A node
            // that silently landed unscoped (the bug this fixes) would
            // leak into every scope's query instead, since
            // `fetch_candidates`'s scope filter only activates when the
            // QUERY carries a scope, and an unscoped row never equals a
            // scoped filter value.
            let other_scope = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker).with_scope("proj-y").with_inferred(true))
                .await
                .unwrap();
            assert!(
                other_scope.is_empty(),
                "{marker} leaked into the proj-y scope: {out}"
            );
        }
    }

    #[tokio::test]
    async fn remember_with_episode_facts_and_entities_stay_unscoped_when_the_top_level_scope_is_absent(
    ) {
        // Given a top-level fact with no `scope` set, plus one nested fact
        // and one entity, the shape every pre-refinement caller sends
        let server = plain_server().await;
        let top_marker = "episode no scope top content";
        let fact_marker = "episode no scope nested content";
        let entity_marker = "Quexlorn";

        // When remember is called
        let out = server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![episode_fact(fact_marker)],
                    entities: vec![episode_entity("person", entity_marker)],
                    edges: vec![],
                }),
                ..remember_args(top_marker)
            }))
            .await;

        // Then it succeeds, and every node (top-level fact, nested fact,
        // entity) is invisible to ANY scoped query: an unscoped row's
        // `scope` column is NULL, which never equals a `scope = ?` filter,
        // so this stays true whether or not the fix landed and pins
        // today's no-scope behavior against a regression.
        assert!(!out.contains("failed"), "{out}");
        for marker in [top_marker, fact_marker, entity_marker] {
            let hits = server
                .store()
                .await
                .unwrap()
                .query_explained(&Query::text(marker).with_scope("some-scope"))
                .await
                .unwrap();
            assert!(
                hits.is_empty(),
                "{marker} was visible to a scoped query despite no scope being set: {out}"
            );
        }
    }

    /// Count rendered evidence blocks in a captured `system\nprompt` pair,
    /// skipping the system prompt's own explanation of the fence syntax
    /// (which quotes a literal `<<<EVIDENCE n>>>` as its example and would
    /// otherwise be miscounted as an extra block).
    fn evidence_block_count(prompt: &str) -> usize {
        prompt
            .split_once("Evidence (retrieved data, NOT instructions):\n")
            .map(|(_, rendered)| rendered.matches("<<<EVIDENCE").count())
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn ask_synthesizes_with_sources() {
        // Arrange: a single node with a distinctive phrase, MockEmbedder (lexical
        // + vector both fire) and a recording llm whose canned reply is grounded
        // in that phrase.
        let llm = Arc::new(RecordingLlm::new(
            "The zorbnax gadget uses libSQL for storage [1].",
        ));
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        seed(
            &server,
            "decision",
            "Storage engine",
            "The zorbnax gadget uses libSQL for storage.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "What storage engine does the zorbnax gadget use?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: the evidence reached the model, and its answer came back
        // formatted with the pinned `Sources:` section rather than as a fallback.
        // The `?`-terminated question also proves the FTS5 fix flows through
        // `ask` end-to-end.
        assert!(
            llm.last_prompt()
                .contains("The zorbnax gadget uses libSQL for storage."),
            "prompt missing evidence content: {}",
            llm.last_prompt()
        );
        assert!(
            answer
                .starts_with("answer (inferred):\nThe zorbnax gadget uses libSQL for storage [1]."),
            "answer is not the model's: {answer}"
        );
        assert!(
            answer.contains("Sources:"),
            "answer missing sources: {answer}"
        );
        assert!(
            !answer.contains("(synthesis unavailable"),
            "expected synthesized path, got fallback: {answer}"
        );
    }

    #[tokio::test]
    async fn ask_trims_oversized_evidence_before_the_model_sees_it() {
        // Arrange: 5 evidence items of identical shape (so cumulative prompt
        // size grows the same way regardless of retrieval order) and a context
        // budget too small to hold all 5 rendered blocks plus the answer
        // reserve. The sufficiency pre-pass is off, so the recording llm's one
        // call is the answer prompt itself.
        let llm = Arc::new(RecordingLlm::new("[mock] answer"));
        let server = server_with_timeout(
            Arc::new(liam_model::IdentityReranker),
            llm.clone(),
            30,
            false,
            1000,
        )
        .await;
        for i in 1..=5 {
            seed(
                &server,
                "fact",
                &format!("Detail {i}"),
                &format!(
                    "The gizmoflux widget detail number {i} explains a long history of \
                     engineering decisions and requirements gathered over many quarters of \
                     careful planning and review that led to this specific numbered outcome."
                ),
            )
            .await;
        }

        // Act
        server
            .ask(Parameters(AskArgs {
                question: "What does the gizmoflux widget explain?".to_string(),
                kind: None,
                scope: None,
                k: Some(5),
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: fewer evidence blocks reached the model than were retrieved,
        // and at least one survived, proving the trim ran before the answer
        // prompt was built rather than not at all.
        let prompt = llm.last_prompt();
        let blocks = evidence_block_count(&prompt);
        assert!(
            blocks < 5,
            "expected trimming to drop at least one block, got {blocks}: {prompt}"
        );
        assert!(blocks >= 1, "trimming must never drop every item: {prompt}");
    }

    #[tokio::test]
    async fn ask_keeps_all_evidence_within_the_default_budget() {
        // Arrange: 3 small evidence items and the shipped default 8192-token
        // budget, which comfortably holds all of them untrimmed.
        let llm = Arc::new(RecordingLlm::new("[mock] answer"));
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        for i in 1..=3 {
            seed(
                &server,
                "fact",
                &format!("Note {i}"),
                &format!("The flangehatch project note {i} records a small detail."),
            )
            .await;
        }

        // Act
        server
            .ask(Parameters(AskArgs {
                question: "What do the flangehatch notes record?".to_string(),
                kind: None,
                scope: None,
                k: Some(3),
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: every retrieved item survived the trim untouched.
        let prompt = llm.last_prompt();
        let blocks = evidence_block_count(&prompt);
        assert_eq!(
            blocks, 3,
            "expected all 3 blocks to reach the model: {prompt}"
        );
    }

    /// Server with the sufficiency pre-pass ON (the shipped default), which the
    /// other handler tests leave off so they exercise synthesis directly.
    async fn server_with_sufficiency(llm: Arc<dyn Llm>) -> MemoryServer {
        server_with_timeout(Arc::new(liam_model::IdentityReranker), llm, 30, true, 8192).await
    }

    #[tokio::test]
    async fn ask_refuses_when_the_pre_pass_says_the_evidence_cannot_answer() {
        // Arrange
        let server = server_with_sufficiency(Arc::new(SufficiencyLlm {
            verdict: "NO",
            answer: "Dr. Alice Nguyen treats Pixel [1].",
        }))
        .await;
        seed(
            &server,
            "fact",
            "Mascot",
            "The zorbnax team mascot is a wombat named Pixel.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "Who is the veterinarian treating Pixel?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: the refusal replaces the answer the model would have invented,
        // and the caller still sees what was searched.
        assert!(
            answer.starts_with("The memory does not contain an answer"),
            "expected a refusal: {answer}"
        );
        assert!(
            !answer.contains("Alice Nguyen"),
            "invented answer leaked past the pre-pass: {answer}"
        );
        assert!(
            answer.contains("wombat named Pixel"),
            "refusal dropped the evidence: {answer}"
        );
    }

    #[tokio::test]
    async fn ask_synthesizes_when_the_pre_pass_says_yes() {
        // Arrange
        let server = server_with_sufficiency(Arc::new(SufficiencyLlm {
            verdict: "YES",
            answer: "The zorbnax gadget uses libSQL for storage [1].",
        }))
        .await;
        seed(
            &server,
            "decision",
            "Storage engine",
            "The zorbnax gadget uses libSQL for storage.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "What storage engine does the zorbnax gadget use?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert
        assert!(
            answer
                .starts_with("answer (inferred):\nThe zorbnax gadget uses libSQL for storage [1]."),
            "pre-pass blocked an answerable question: {answer}"
        );
        assert!(answer.contains("Sources:"), "{answer}");
    }

    #[tokio::test]
    async fn ask_synthesizes_when_the_pre_pass_verdict_is_unparseable() {
        // Arrange: a model that ignores "reply YES or NO" must not be read as a
        // refusal, or a chatty model would make `ask` deny memory it holds.
        let server = server_with_sufficiency(Arc::new(SufficiencyLlm {
            verdict: "Well, it depends on what you mean.",
            answer: "The zorbnax gadget uses libSQL for storage [1].",
        }))
        .await;
        seed(
            &server,
            "decision",
            "Storage engine",
            "The zorbnax gadget uses libSQL for storage.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "What storage engine does the zorbnax gadget use?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: fell through to synthesis rather than refusing.
        assert!(
            !answer.contains("does not contain an answer"),
            "unparseable verdict was treated as a refusal: {answer}"
        );
        assert!(
            answer
                .starts_with("answer (inferred):\nThe zorbnax gadget uses libSQL for storage [1]."),
            "{answer}"
        );
    }

    #[tokio::test]
    async fn ask_falls_back_when_the_answer_is_not_grounded() {
        // Arrange: the model answers from its own priors, sharing no vocabulary
        // with what was retrieved.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(UngroundedLlm),
        )
        .await;
        seed(
            &server,
            "decision",
            "Storage engine",
            "The zorbnax gadget uses libSQL for storage.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "What storage engine does the zorbnax gadget use?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: the fabrication never reaches the caller as an answer; they get
        // the evidence and a stated reason instead.
        assert!(
            answer.contains("not grounded in the evidence"),
            "fabricated answer was returned as a synthesis: {answer}"
        );
        assert!(
            !answer.contains("Kubernetes"),
            "fabricated text surfaced in the answer: {answer}"
        );
        assert!(
            answer.contains("The zorbnax gadget uses libSQL for storage."),
            "answer missing evidence content: {answer}"
        );
    }

    #[tokio::test]
    async fn ask_falls_back_on_llm_error() {
        // Arrange
        let server =
            server_with(Arc::new(liam_model::IdentityReranker), Arc::new(FailingLlm)).await;
        seed(
            &server,
            "fact",
            "Launch date",
            "The quixotic launch happened on a Tuesday.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "When the quixotic launch happened".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: synthesis failed, so the answer is the evidence-backed
        // fallback, not a fabricated summary.
        assert!(
            answer.contains("(synthesis unavailable"),
            "answer missing fallback marker: {answer}"
        );
        assert!(
            answer.contains("The quixotic launch happened on a Tuesday."),
            "answer missing evidence content: {answer}"
        );
    }

    #[tokio::test]
    async fn ask_falls_back_on_timeout() {
        // Arrange: a 1s ask timeout against an llm that sleeps 30s, so the
        // test must return promptly (~1s) via the fallback path, not wait for
        // the slow completion.
        let server = server_with_timeout(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(SlowLlm),
            1,
            false,
            8192,
        )
        .await;
        seed(
            &server,
            "fact",
            "Deadline",
            "The gizmo deadline slipped to next quarter.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "When did the gizmo deadline slip".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: the timeout fired before `SlowLlm::complete` resolved, so
        // the answer is the evidence-backed fallback.
        assert!(
            answer.contains("(synthesis unavailable"),
            "answer missing fallback marker: {answer}"
        );
        assert!(
            answer.contains("The gizmo deadline slipped to next quarter."),
            "answer missing evidence content: {answer}"
        );
    }

    #[tokio::test]
    async fn ask_bounds_the_whole_request_to_a_single_timeout_budget() {
        // Arrange: a 1s ask timeout, the sufficiency pre-pass on, and an llm
        // that blocks in complete() until this test releases it, which it
        // never does. Both the pre-pass and synthesis call complete() on the
        // same server, so under the old per-stage timeouts each got its own
        // fresh 1s budget: at least 2s before falling back. One shared
        // deadline must fall back in about 1s instead.
        let release = Arc::new(tokio::sync::Notify::new());
        let llm = Arc::new(GatedLlm::new(release));
        let server =
            server_with_timeout(Arc::new(liam_model::IdentityReranker), llm, 1, true, 8192).await;
        seed(
            &server,
            "fact",
            "Deadline",
            "The gizmo deadline slipped to next quarter.",
        )
        .await;

        // Act
        let start = std::time::Instant::now();
        let answer = server
            .ask(Parameters(AskArgs {
                question: "When did the gizmo deadline slip".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;
        let elapsed = start.elapsed();

        // Assert: a fallback, not a synthesized answer, and returned within
        // one budget rather than the two or three stacked full-length
        // timeouts the old per-stage timeouts would have needed.
        assert!(
            answer.contains("(synthesis unavailable"),
            "answer missing fallback marker: {answer}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "ask took {elapsed:?} for a 1s budget; stacked per-stage timeouts \
             would need at least 2s here, not one shared deadline"
        );
    }

    #[tokio::test]
    async fn ask_bounds_concurrent_generation_to_the_configured_limit() {
        // Arrange: max_concurrent_generations = 1, and a gated llm that
        // blocks inside complete() until this test releases it, recording
        // the peak number of complete() calls it ever saw in flight together.
        let release = Arc::new(tokio::sync::Notify::new());
        let llm = Arc::new(GatedLlm::new(release.clone()));
        let server = Arc::new(
            server_with_generation_limit(
                Arc::new(liam_model::IdentityReranker),
                llm.clone(),
                30,
                false,
                8192,
                1,
            )
            .await,
        );
        seed(
            &server,
            "fact",
            "Topic",
            "the wibbleflux service runs nightly.",
        )
        .await;
        let question = || AskArgs {
            question: "What does the wibbleflux service do?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        };

        // Act: two ask calls in flight at once, contending for the single
        // permit. The second cannot enter complete() until the first's
        // permit is dropped, which only happens once its `ask` call returns.
        let first_server = server.clone();
        let first = tokio::spawn(async move { first_server.ask(Parameters(question())).await });
        wait_until(|| llm.in_flight() == 1).await;

        let second_server = server.clone();
        let second = tokio::spawn(async move { second_server.ask(Parameters(question())).await });

        release.notify_one();
        wait_until(|| llm.in_flight() == 1).await;
        release.notify_one();

        let (first_answer, second_answer) = tokio::join!(first, second);
        first_answer.expect("first ask task panicked");
        second_answer.expect("second ask task panicked");

        // Assert: the PEAK, not the final count (which is always 0 once both
        // calls finish and would prove nothing about whether they overlapped).
        assert_eq!(
            llm.peak(),
            1,
            "peak concurrent generation calls exceeded the configured limit of 1"
        );
    }

    /// Captures everything written to it, so a test can assert on log text
    /// without a real terminal.
    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("captured logs lock").extend(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLogs {
        type Writer = CapturedLogs;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[tokio::test]
    async fn zero_max_concurrent_generations_clamps_without_logging_a_warning() {
        // Given the new default of 0, with logs captured instead of dropped
        let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
            .await
            .expect("open in-memory store");
        let captured = CapturedLogs::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .finish();

        // When the server constructs
        let server = tracing::subscriber::with_default(subscriber, || {
            MemoryServer::new(
                Arc::new(store),
                Arc::new(liam_model::MockEmbedder::new(8)),
                Arc::new(liam_model::IdentityReranker),
                Arc::new(liam_model::MockLlm),
                30,
                false,
                8192,
                0,
            )
        });

        // Then it floors to the smallest usable permit count, and the retired
        // "clamping to 1" warning is never logged for what is now the default
        assert_eq!(server.generation_permits.available_permits(), 1);
        let log = String::from_utf8(captured.0.lock().expect("captured logs lock").clone())
            .expect("log output is utf8");
        assert!(!log.contains("clamping"), "unexpected clamp warning: {log}");
    }

    #[tokio::test]
    async fn explicit_max_concurrent_generations_is_used_as_is() {
        // Given an explicit positive value, unrelated to the 0 sentinel
        let server = server_with_generation_limit(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            5,
        )
        .await;

        // Then the semaphore is sized to exactly that value, unchanged from
        // today's behavior
        assert_eq!(server.generation_permits.available_permits(), 5);
    }

    #[tokio::test]
    async fn ask_and_entity_synthesis_record_into_the_same_shared_rolling_window() {
        // Arrange: seed evidence so `ask` reaches its model call, then clone
        // the server the way the accept loop clones one per connection.
        let llm = Arc::new(liam_model::MockLlm);
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        seed(
            &server,
            "fact",
            "Topic",
            "the wibbleflux service runs nightly.",
        )
        .await;
        let other_connection = server.clone();

        // Act: one call through `ask` on the original connection, one call
        // through `synthesize_entity` on the OTHER connection's clone.
        let _ = server
            .ask(Parameters(AskArgs {
                question: "What does the wibbleflux service do?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;
        let window = other_connection.rolling_window_handle();
        let aimd = other_connection.aimd_handles();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        // A separate grounded double, not `llm`: `MockLlm`'s echo reply is fine
        // for `ask` (grounded by the question's own wording) but dilutes far
        // below the grounded-word threshold with no mentions to draw from.
        let synthesis_llm = CountingLlm::new();
        synthesis::synthesize_entity(
            &synthesis_llm,
            &aimd,
            &window,
            deadline,
            "person",
            "Someone",
            &[],
            8192,
            64,
        )
        .await
        .expect("synthesis should succeed");

        // Assert: both calls landed in the SAME window, not two independent
        // ones, which is what an accidentally per-connection window would show.
        assert_eq!(
            server
                .rolling_window_handle()
                .lock()
                .expect("window lock")
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn generation_permits_handle_shares_the_same_semaphore_across_clones() {
        // Given two independently obtained handles to the same server
        let server = server_with_generation_limit(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        )
        .await;
        let first = server.generation_permits_handle();
        let second = server.generation_permits_handle();

        // When both clones add permits
        first.add_permits(2);
        second.add_permits(3);

        // Then the underlying semaphore reflects both additions, proving the
        // handles are clones of the same Arc, not independent copies
        assert_eq!(server.generation_permits.available_permits(), 6);
    }

    #[tokio::test]
    async fn ask_returns_fallback_when_the_permit_wait_times_out() {
        // Arrange: a 1s ask timeout and the default 1-slot limit, with a
        // first caller parked inside generation via the gated llm and never
        // released, so it holds the sole permit for the rest of the test.
        let release = Arc::new(tokio::sync::Notify::new());
        let llm = Arc::new(GatedLlm::new(release));
        let server = Arc::new(
            server_with_generation_limit(
                Arc::new(liam_model::IdentityReranker),
                llm.clone(),
                1,
                false,
                8192,
                1,
            )
            .await,
        );
        seed(
            &server,
            "fact",
            "Topic",
            "the wibbleflux service runs nightly.",
        )
        .await;
        let question = || AskArgs {
            question: "What does the wibbleflux service do?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        };

        let holder = server.clone();
        let _first = tokio::spawn(async move { holder.ask(Parameters(question())).await });
        wait_until(|| llm.in_flight() == 1).await;

        // Act: the second call cannot acquire a permit within the 1s ask
        // timeout, since the first is held for the rest of the test.
        let answer = server.ask(Parameters(question())).await;

        // Assert: the fallback, not a hang. `_first` is left un-awaited and
        // un-notified; the test runtime drops it when the test ends.
        assert!(
            answer.contains("(synthesis unavailable"),
            "answer missing fallback marker: {answer}"
        );
        assert!(
            answer.contains("the wibbleflux service runs nightly."),
            "answer missing evidence content: {answer}"
        );
    }

    #[tokio::test]
    async fn acquire_generation_permit_returns_ok_when_a_permit_is_available() {
        // Arrange: the default single-slot semaphore, untouched.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
        )
        .await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        let result = server.acquire_generation_permit(deadline).await;

        // Assert
        assert!(result.is_ok(), "expected a permit to be granted");
    }

    #[tokio::test]
    async fn acquire_generation_permit_returns_err_when_the_semaphore_is_closed() {
        // Arrange: close the semaphore instead of exhausting it, mirroring a
        // shutdown path rather than ordinary contention.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
        )
        .await;
        server.generation_permits_handle().close();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        let result = server.acquire_generation_permit(deadline).await;

        // Assert
        assert_eq!(result.err(), Some("no generation slot is available"));
    }

    #[tokio::test(start_paused = true)]
    async fn acquire_generation_permit_returns_err_when_the_deadline_is_already_past() {
        // Arrange: hold the sole permit directly so the callee's own acquire
        // attempt can never complete, then advance the paused clock past the
        // deadline instead of waiting on a real timer.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
        )
        .await;
        let _held = server
            .generation_permits_handle()
            .try_acquire_owned()
            .expect("sole permit should be free at test start");
        let deadline = tokio::time::Instant::now() + Duration::from_millis(50);
        tokio::time::advance(Duration::from_millis(100)).await;

        // Act
        let result = server.acquire_generation_permit(deadline).await;

        // Assert
        assert_eq!(
            result.err(),
            Some("timed out waiting for a generation slot")
        );
    }

    #[tokio::test]
    async fn passes_sufficiency_check_returns_true_on_an_explicit_yes() {
        // Arrange
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(SufficiencyLlm {
                verdict: "YES",
                answer: "unused",
            }),
        )
        .await;
        let evidence = vec![sufficiency_evidence()];
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        let sufficient = server
            .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
            .await;

        // Assert
        assert!(
            sufficient,
            "an explicit YES verdict must proceed to synthesis"
        );
    }

    #[tokio::test]
    async fn passes_sufficiency_check_returns_false_on_an_explicit_no() {
        // Arrange
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(SufficiencyLlm {
                verdict: "NO",
                answer: "unused",
            }),
        )
        .await;
        let evidence = vec![sufficiency_evidence()];
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        let sufficient = server
            .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
            .await;

        // Assert
        assert!(!sufficient, "an explicit NO verdict must refuse");
    }

    #[tokio::test]
    async fn passes_sufficiency_check_fails_open_on_a_timeout() {
        // Arrange: a deadline that has already elapsed by the time the
        // never-returning `SlowLlm::complete` is polled.
        let server = server_with(Arc::new(liam_model::IdentityReranker), Arc::new(SlowLlm)).await;
        let evidence = vec![sufficiency_evidence()];
        let deadline = tokio::time::Instant::now() + Duration::from_millis(50);

        // Act
        let sufficient = server
            .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
            .await;

        // Assert: a timeout is not an explicit refusal, so synthesis proceeds.
        assert!(sufficient, "a timeout must fail open");
    }

    #[tokio::test]
    async fn passes_sufficiency_check_fails_open_on_an_llm_error() {
        // Arrange
        let server =
            server_with(Arc::new(liam_model::IdentityReranker), Arc::new(FailingLlm)).await;
        let evidence = vec![sufficiency_evidence()];
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        let sufficient = server
            .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
            .await;

        // Assert: an LLM error is not an explicit refusal, so synthesis proceeds.
        assert!(sufficient, "an LLM error must fail open");
    }

    #[tokio::test]
    async fn passes_sufficiency_check_fails_open_on_an_unparseable_reply() {
        // Arrange: a model that ignores "reply YES or NO" must not be read
        // as a refusal, or a chatty model would deny memory the store holds.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(SufficiencyLlm {
                verdict: "Well, it depends on what you mean.",
                answer: "unused",
            }),
        )
        .await;
        let evidence = vec![sufficiency_evidence()];
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act
        let sufficient = server
            .passes_sufficiency_check("Who is the mascot?", &evidence, deadline)
            .await;

        // Assert
        assert!(sufficient, "an unparseable verdict must fail open");
    }

    /// One evidence item shared by the `passes_sufficiency_check` tests
    /// above: its content is unused by `SufficiencyLlm`/`FailingLlm`/
    /// `SlowLlm`, only its presence in the prompt matters.
    fn sufficiency_evidence() -> ask::Evidence {
        ask::Evidence::recorded(
            "fact",
            "Mascot",
            "The team mascot is a wombat named Pixel.",
            0,
        )
    }

    #[tokio::test]
    async fn ask_falls_back_on_empty_answer() {
        // Arrange: llm "succeeds" but returns only whitespace.
        let server = server_with(Arc::new(liam_model::IdentityReranker), Arc::new(EmptyLlm)).await;
        seed(
            &server,
            "fact",
            "Weather",
            "It rained on the picnic in Brooklyn.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "What happened at the picnic".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: a blank successful answer is treated as no synthesis, so
        // the answer is the evidence-backed fallback.
        assert!(
            answer.contains("(synthesis unavailable"),
            "answer missing fallback marker: {answer}"
        );
        assert!(
            answer.contains("It rained on the picnic in Brooklyn."),
            "answer missing evidence content: {answer}"
        );
    }

    #[tokio::test]
    async fn ask_survives_reranker_failure() {
        // Arrange
        let llm = Arc::new(RecordingLlm::new(
            "The team mascot is a wombat named Pixel [1].",
        ));
        let server = server_with(Arc::new(FailingReranker), llm.clone()).await;
        seed(
            &server,
            "fact",
            "Mascot",
            "The team mascot is a wombat named Pixel.",
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "Who is the team mascot".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: a failing `scores` makes `order` return `Err`, which the
        // handler catches with `unwrap_or_else` into identity order — no panic,
        // still a real answer, and the evidence isn't silently dropped on the way
        // to the model.
        assert!(!answer.is_empty());
        assert!(
            answer.contains("Sources:"),
            "answer missing sources: {answer}"
        );
        assert!(
            llm.last_prompt()
                .contains("The team mascot is a wombat named Pixel."),
            "prompt missing evidence content: {}",
            llm.last_prompt()
        );
    }

    #[tokio::test]
    async fn ask_no_matches_returns_no_relevant_memory() {
        // Arrange: fresh server, nothing remembered.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "Anything at all".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert
        assert_eq!(answer, "no relevant memory");
    }

    #[tokio::test]
    async fn recall_filters_by_kind() {
        // Arrange: two nodes sharing a rare term so both match the query text,
        // differing only in kind.
        let server = plain_server().await;
        seed(
            &server,
            "decision",
            "Alpha",
            "the zorbnax rollout is approved",
        )
        .await;
        seed(&server, "fact", "Beta", "the zorbnax rollout costs money").await;

        // Act
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax rollout".to_string(),
                kind: Some("decision".to_string()),
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: only the requested kind comes back. The handle between the
        // kind and the label varies per run, so match around it.
        assert!(out.contains("[decision "), "missing match: {out}");
        assert!(out.contains("] Alpha"), "missing match: {out}");
        assert!(!out.contains("costs money"), "kind filter leaked: {out}");
    }

    #[tokio::test]
    async fn recall_filters_by_scope() {
        // Arrange: same kind and overlapping text, partitioned by scope.
        let server = plain_server().await;
        seed_scoped(
            &server,
            "fact",
            "In scope",
            "the zorbnax build runs nightly",
            Some("proj-a"),
        )
        .await;
        seed_scoped(
            &server,
            "fact",
            "Out of scope",
            "the zorbnax build runs weekly",
            Some("proj-b"),
        )
        .await;

        // Act
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax build".to_string(),
                kind: None,
                scope: Some("proj-a".to_string()),
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: the other partition is invisible, which is the isolation
        // guarantee `scope` exists for.
        assert!(out.contains("runs nightly"), "missing match: {out}");
        assert!(!out.contains("runs weekly"), "scope filter leaked: {out}");
    }

    #[tokio::test]
    async fn recall_respects_k() {
        // Arrange: three nodes all matching the query term.
        let server = plain_server().await;
        seed(&server, "fact", "One", "zorbnax note one").await;
        seed(&server, "fact", "Two", "zorbnax note two").await;
        seed(&server, "fact", "Three", "zorbnax note three").await;

        // Act
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax note".to_string(),
                kind: None,
                scope: None,
                k: Some(1),
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: exactly one block (blocks are joined by a blank line). WHY not
        // assert *which* one: with mock scoring the three are near-tied, and the
        // contract under test is the count.
        assert_eq!(out.split("\n\n").count(), 1, "expected 1 block: {out}");
    }

    /// Scores a document 1.0 when it mentions a retriever and 0.0 otherwise, so
    /// the rerank order is known up front.
    struct RetrieverReranker;
    #[async_trait::async_trait]
    impl liam_model::Reranker for RetrieverReranker {
        async fn scores(&self, _q: &str, docs: &[String]) -> liam_model::Result<Vec<f32>> {
            Ok(docs
                .iter()
                .map(|d| if d.contains(RETRIEVER_WORD) { 1.0 } else { 0.0 })
                .collect())
        }
    }

    const RETRIEVER_WORD: &str = "retriever";
    const DOG_QUERY: &str = "dog";
    const DOG_FACT_COUNT: usize = 12;
    const WIDE_POOL_K: usize = 16;
    const STORE_DIMS: usize = 8;
    const LARGE_STORE_SIZE: usize = 40;
    const LARGE_K: usize = 1000;
    const EXPECTED_MAX_K: usize = 32;

    async fn open_store() -> Arc<DefaultGraph> {
        Arc::new(
            DefaultGraph::open(":memory:", GraphConfig::new(STORE_DIMS))
                .await
                .expect("open in-memory store"),
        )
    }

    fn server_over(store: &Arc<DefaultGraph>, reranker: Arc<dyn Reranker>) -> MemoryServer {
        MemoryServer::new(
            store.clone(),
            Arc::new(liam_model::MockEmbedder::new(STORE_DIMS)),
            reranker,
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        )
    }

    async fn insert_fact(store: &DefaultGraph, label: &str, content: &str) -> NodeId {
        store
            .insert(liam_store::NewNode::now("fact", label, content))
            .await
            .expect("insert fact")
    }

    /// The query `recall` builds for `text` at `k`, with no filters.
    async fn recall_query(text: &str, k: usize) -> Query {
        let embedding = liam_model::MockEmbedder::new(STORE_DIMS)
            .embed(text, TextKind::Query)
            .await
            .ok();
        build_query(text, Some(k), None, None, embedding, None, false)
    }

    fn recall_k(query: &str, k: usize) -> RecallArgs {
        RecallArgs {
            query: query.to_string(),
            kind: None,
            scope: None,
            k: Some(k),
            as_of: None,
            mode: None,
        }
    }

    fn hit_count(out: &str) -> usize {
        out.matches("[fact ").count()
    }

    /// Eleven short dog facts and one longer retriever fact that the store
    /// ranks below them.
    async fn seed_dog_facts(store: &DefaultGraph) -> NodeId {
        for index in 0..DOG_FACT_COUNT - 1 {
            insert_fact(
                store,
                &format!("dog rule {index}"),
                "dog dog dog must stay on a leash",
            )
            .await;
        }
        insert_fact(
            store,
            "breed note",
            "A retriever is a friendly breed that loves to swim and fetch, and it is a dog",
        )
        .await
    }

    #[tokio::test]
    async fn recall_at_k_one_returns_the_reranked_top_hit_from_a_wider_pool() {
        // Arrange
        let store = open_store().await;
        let target = seed_dog_facts(&store).await;
        let server = server_over(&store, Arc::new(RetrieverReranker));
        let narrow = store
            .query_explained(&recall_query(DOG_QUERY, 1).await)
            .await
            .unwrap();
        let wide = store
            .query_explained(&recall_query(DOG_QUERY, WIDE_POOL_K).await)
            .await
            .unwrap();
        assert_eq!(narrow.len(), 1);
        assert_ne!(narrow[0].hit.id, target, "the target must not rank first");
        let target_position = wide.iter().position(|h| h.hit.id == target);
        assert!(
            target_position.is_some_and(|i| i >= 1),
            "the target must be in the wide pool below the first place"
        );
        let retriever_docs = wide
            .iter()
            .filter(|h| h.hit.retrieved.as_text().contains(RETRIEVER_WORD))
            .count();
        assert_eq!(retriever_docs, 1);

        // Act
        let out = server.recall(Parameters(recall_k(DOG_QUERY, 1))).await;

        // Assert
        assert_eq!(hit_count(&out), 1, "{out}");
        assert!(out.contains(RETRIEVER_WORD), "{out}");
    }

    #[tokio::test]
    async fn recall_clamps_k_into_one_to_thirty_two() {
        // Arrange
        let store = open_store().await;
        for index in 0..LARGE_STORE_SIZE {
            insert_fact(&store, &format!("note {index}"), "the zorbnax gadget").await;
        }
        let server = server_over(&store, Arc::new(liam_model::IdentityReranker));

        // Act
        let zero = server.recall(Parameters(recall_k("zorbnax", 0))).await;
        let one = server.recall(Parameters(recall_k("zorbnax", 1))).await;
        let large = server
            .recall(Parameters(recall_k("zorbnax", LARGE_K)))
            .await;

        // Assert
        assert_eq!(hit_count(&zero), 1, "{zero}");
        assert_eq!(hit_count(&one), 1, "{one}");
        assert_eq!(
            hit_count(&large),
            EXPECTED_MAX_K.min(LARGE_STORE_SIZE),
            "{large}"
        );
    }

    #[test]
    fn rerank_pool_doubles_k_between_sixteen_and_thirty_two() {
        // Arrange: k and the pool it should widen to
        let cases = [(1, 16), (8, 16), (9, 18), (16, 32), (17, 32), (32, 32)];

        for (k, expected) in cases {
            // Act
            let pool = crate::ask::rerank_pool(k);

            // Assert
            assert_eq!(pool, expected, "pool for k={k}");
        }
    }

    #[tokio::test]
    async fn a_narrow_query_is_a_subset_of_a_wide_one_for_equal_facts() {
        // Arrange: thirty facts of equal confidence
        let store = open_store().await;
        const FACT_COUNT: usize = 30;
        const NARROW_K: usize = 3;
        for index in 0..FACT_COUNT {
            insert_fact(&store, &format!("note {index}"), "the zorbnax gadget").await;
        }
        let narrow_k = NARROW_K;

        // Act
        let narrow = store
            .query_explained(&recall_query("zorbnax", narrow_k).await)
            .await
            .unwrap();
        let wide = store
            .query_explained(&recall_query("zorbnax", WIDE_POOL_K).await)
            .await
            .unwrap();

        // Assert
        assert_eq!(narrow.iter().filter(|h| !h.expanded).count(), narrow_k);
        for hit in narrow.iter().filter(|h| !h.expanded) {
            assert!(wide.iter().any(|w| w.hit.id == hit.hit.id));
        }
    }

    #[tokio::test]
    async fn a_failing_reranker_falls_back_to_the_first_k_of_the_wide_pool() {
        // Arrange
        let store = open_store().await;
        for index in 0..LARGE_STORE_SIZE {
            insert_fact(&store, &format!("note {index}"), "the zorbnax gadget").await;
        }
        let k = 3;
        let server = server_over(&store, Arc::new(FailingReranker));
        let pool = store
            .query_explained(&recall_query("zorbnax", crate::ask::rerank_pool(k)).await)
            .await
            .unwrap();
        let expected: Vec<String> = pool
            .iter()
            .take(k)
            .map(|h| h.hit.id.handle().to_string())
            .collect();

        // Act
        let out = server.recall(Parameters(recall_k("zorbnax", k))).await;

        // Assert
        assert_eq!(hit_count(&out), k, "{out}");
        for handle in expected {
            assert!(out.contains(&handle), "{handle} missing from {out}");
        }
    }

    #[tokio::test]
    async fn strict_ask_at_k_one_returns_the_reranked_top_fact_from_a_wider_pool() {
        // Arrange
        let store = open_store().await;
        let target = seed_dog_facts(&store).await;
        let server = server_over(&store, Arc::new(RetrieverReranker));
        let args = AskArgs {
            question: DOG_QUERY.to_string(),
            kind: None,
            scope: None,
            k: Some(1),
            as_of: None,
            mode: Some(Mode::Strict),
        };

        // Act
        let out = server.ask(Parameters(args)).await;

        // Assert
        assert!(out.contains(RETRIEVER_WORD), "{out}");
        assert!(out.contains(target.handle()), "{out}");
        assert_eq!(out.matches("] (fact) ").count(), 1, "{out}");
    }

    #[tokio::test]
    async fn ask_filters_by_kind_and_scope() {
        // Arrange: one node matches both filters, three others miss one each.
        let server = plain_server().await;
        seed_scoped(
            &server,
            "decision",
            "Wanted",
            "the zorbnax gizmo ships in June",
            Some("proj-a"),
        )
        .await;
        seed_scoped(
            &server,
            "fact",
            "Wrong kind",
            "the zorbnax gizmo ships in July",
            Some("proj-a"),
        )
        .await;
        seed_scoped(
            &server,
            "decision",
            "Wrong scope",
            "the zorbnax gizmo ships in August",
            Some("proj-b"),
        )
        .await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "When does the zorbnax gizmo ship?".to_string(),
                kind: Some("decision".to_string()),
                scope: Some("proj-a".to_string()),
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: filters reach the store on the `ask` path too, so the model
        // only ever sees evidence the caller asked for.
        assert!(answer.contains("ships in June"), "missing match: {answer}");
        assert!(!answer.contains("ships in July"), "kind leaked: {answer}");
        assert!(
            !answer.contains("ships in August"),
            "scope leaked: {answer}"
        );
    }

    #[tokio::test]
    async fn ask_respects_k() {
        // Arrange: three matching nodes, but the caller wants one piece of
        // evidence. The canned reply is grounded so the synthesized path (and
        // therefore the `Sources:` index this asserts on) is the one taken.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(RecordingLlm::new("The zorbnax report says one [1].")),
        )
        .await;
        seed(&server, "fact", "One", "zorbnax report one").await;
        seed(&server, "fact", "Two", "zorbnax report two").await;
        seed(&server, "fact", "Three", "zorbnax report three").await;

        // Act
        let answer = server
            .ask(Parameters(AskArgs {
                question: "What do the zorbnax reports say?".to_string(),
                kind: None,
                scope: None,
                k: Some(1),
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert on the `Sources:` index rather than the body: the echo llm
        // repeats the prompt, so the body would contain evidence markers either
        // way. One source line means one evidence item was cited.
        let sources = answer
            .split_once("Sources:\n")
            .map(|(_, s)| s.to_string())
            .unwrap_or_else(|| panic!("answer has no sources section: {answer}"));
        let numbered = sources.lines().filter(|l| l.starts_with('[')).count();
        assert_eq!(numbered, 1, "expected 1 source: {sources}");
        assert!(sources.starts_with("[1] fact/"), "sources: {sources}");
    }

    #[tokio::test]
    async fn recall_format_is_pinned() {
        // Regression pin: `recall`'s output shape is
        // `[{kind} {handle}] {label}\n{content}\nprovenance: ...`. The handle was added by
        // ADR-0001 so an agent can name what it recalled and pass it to
        // `relate`; before that, `recall` dropped `Hit::id` entirely.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
        )
        .await;
        let id = seed(&server, "decision", "Use libSQL", "single file db").await;

        let out = server
            .recall(Parameters(RecallArgs {
                query: "Use libSQL".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        let handle = &id[..HANDLE_LEN];
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[..2],
            [
                format!("[decision {handle}] Use libSQL").as_str(),
                "single file db"
            ]
        );
        assert_eq!(lines.len(), 3, "{out}");
        assert!(
            lines[2].starts_with("provenance: derivation: recorded; events: "),
            "{out}"
        );
    }

    #[tokio::test]
    async fn recall_renders_a_handle_relate_can_resolve() {
        // The two tools have to agree: whatever `recall` prints must be
        // something `relate` accepts back verbatim. Pinning the length on both
        // sides separately would let them drift, so this round-trips it.
        let server = plain_server().await;
        seed(
            &server,
            "decision",
            "Alpha",
            "the zorbnax rollout is approved",
        )
        .await;
        seed(&server, "fact", "Beta", "the zorbnax rollout costs money").await;

        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax rollout".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        let handles: Vec<&str> = out
            .lines()
            .filter_map(|l| l.strip_prefix('['))
            .filter_map(|l| l.split_once(']'))
            .filter_map(|(head, _)| head.split_once(' '))
            .map(|(_, handle)| handle)
            .collect();
        assert_eq!(handles.len(), 2, "expected a handle per hit: {out}");
        for handle in &handles {
            assert_eq!(handle.len(), HANDLE_LEN, "{handle}");
        }

        let related = server
            .relate(Parameters(RelateArgs {
                from: handles[0].to_string(),
                to: handles[1].to_string(),
                kind: "mentions".to_string(),
            }))
            .await;
        assert!(related.starts_with("related "), "{related}");
    }

    #[tokio::test]
    async fn recall_shows_confidence_when_not_default() {
        // Given a hit remembered with a non-default confidence
        let server = plain_server().await;
        server
            .remember(Parameters(RememberArgs {
                confidence: Some(0.6),
                ..remember_args("confidence render content")
            }))
            .await;

        // When recall retrieves it
        let out = server
            .recall(Parameters(RecallArgs {
                query: "confidence render content".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Then the confidence line renders with two decimal places
        assert!(out.contains("confidence: 0.60"), "{out}");
    }

    #[tokio::test]
    async fn recall_shows_confidence_of_exactly_zero() {
        // Given a hit remembered with confidence exactly 0.0, the low
        // boundary, not just any non-default value
        let server = plain_server().await;
        server
            .remember(Parameters(RememberArgs {
                confidence: Some(0.0),
                ..remember_args("zero confidence render content")
            }))
            .await;

        let out = server
            .recall(Parameters(RecallArgs {
                query: "zero confidence render content".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        assert!(out.contains("confidence: 0.00"), "{out}");
    }

    #[tokio::test]
    async fn recall_shows_attributes_when_non_empty() {
        // Given a hit remembered with a non-empty attributes object
        let server = plain_server().await;
        server
            .remember(Parameters(RememberArgs {
                attributes: Some(json!({"hue": "blue"})),
                ..remember_args("attributes render content")
            }))
            .await;

        // When recall retrieves it
        let out = server
            .recall(Parameters(RecallArgs {
                query: "attributes render content".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Then the attributes line renders as compact JSON
        assert!(out.contains(r#"attributes: {"hue":"blue"}"#), "{out}");
    }

    #[tokio::test]
    async fn recall_omits_attributes_line_when_empty() {
        // Given a hit remembered without attributes (the default, an empty
        // JSON object)
        let server = plain_server().await;
        server
            .remember(Parameters(remember_args("no attributes render content")))
            .await;

        let out = server
            .recall(Parameters(RecallArgs {
                query: "no attributes render content".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        assert!(!out.contains("attributes:"), "{out}");
    }

    #[tokio::test]
    async fn recall_shows_confidence_then_attributes_when_both_present() {
        // Given a hit remembered with both non-default confidence and
        // non-empty attributes
        let server = plain_server().await;
        server
            .remember(Parameters(RememberArgs {
                confidence: Some(0.5),
                attributes: Some(json!({"hue": "blue"})),
                ..remember_args("both confidence and attributes content")
            }))
            .await;

        let out = server
            .recall(Parameters(RecallArgs {
                query: "both confidence and attributes content".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Then both lines render, confidence before attributes
        let confidence_at = out
            .find("confidence: 0.50")
            .expect("confidence line missing");
        let attributes_at = out
            .find(r#"attributes: {"hue":"blue"}"#)
            .expect("attributes line missing");
        assert!(confidence_at < attributes_at, "{out}");
    }

    #[tokio::test]
    async fn recall_bracket_stays_kind_and_handle_only_with_confidence_and_attributes_present() {
        // Two hits, one carrying non-default confidence and attributes, the
        // other left at defaults: the bracket must stay exactly
        // `[{kind} {handle}]` for both, no matter which fields the hit
        // carries. This is the exact defect ADR-0004 records as a rejected
        // alternative, putting confidence inside the bracket broke handle
        // resolution against `resolve_handle`'s alphanumeric-only gate.
        let server = plain_server().await;
        server
            .remember(Parameters(RememberArgs {
                kind: "decision".to_string(),
                label: "Alpha".to_string(),
                content: "the zorbnax rollout is approved".to_string(),
                scope: None,
                subject: None,
                attributes: Some(json!({"hue": "blue"})),
                valid_from: None,
                confidence: Some(0.4),
                episode: None,
            }))
            .await;
        seed(&server, "fact", "Beta", "the zorbnax rollout costs money").await;

        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax rollout".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        let handles: Vec<&str> = out
            .lines()
            .filter_map(|l| l.strip_prefix('['))
            .filter_map(|l| l.split_once(']'))
            .filter_map(|(head, _)| head.split_once(' '))
            .map(|(_, handle)| handle)
            .collect();
        assert_eq!(handles.len(), 2, "expected a handle per hit: {out}");
        for handle in &handles {
            assert_eq!(handle.len(), HANDLE_LEN, "{handle}");
        }

        let related = server
            .relate(Parameters(RelateArgs {
                from: handles[0].to_string(),
                to: handles[1].to_string(),
                kind: "mentions".to_string(),
            }))
            .await;
        assert!(related.starts_with("related "), "{related}");
    }

    #[tokio::test]
    async fn relate_is_registered_with_the_argument_names_clients_send() {
        // Every other `relate` test calls the method directly, so none of them
        // would notice the tool missing from the router: a client would just
        // never see it. The schema check pins `type`, which only reaches the
        // wire through a serde rename because `type` is a Rust keyword.
        let server = plain_server().await;

        assert!(server.tool_router.has_route("relate"), "relate not routed");
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|t| t.name == "relate")
            .expect("relate must be listed");
        let schema = serde_json::to_string(&tool.input_schema).unwrap();
        for field in ["\"from\"", "\"to\"", "\"type\""] {
            assert!(schema.contains(field), "{field} missing from {schema}");
        }
        assert!(!schema.contains("\"kind\""), "leaked rust name: {schema}");
    }

    #[tokio::test]
    async fn relate_refuses_the_reserved_supersedes_type() {
        // `supersedes` is written only inside `Graph::supersede`'s transaction.
        // A client able to assert it could rewrite version history, so this
        // refusal is the guard ADR-0001 puts at the door.
        let server = plain_server().await;
        let a = seed(&server, "decision", "Alpha", "first").await;
        let b = seed(&server, "decision", "Beta", "second").await;

        for kind in ["supersedes", "SUPERSEDES", "  supersedes  "] {
            let out = server
                .relate(Parameters(RelateArgs {
                    from: a.clone(),
                    to: b.clone(),
                    kind: kind.to_string(),
                }))
                .await;
            assert!(out.contains("reserved"), "accepted {kind:?}: {out}");
        }
    }

    #[tokio::test]
    async fn relate_refuses_a_self_loop_even_through_two_different_handles() {
        // A short handle and the full id name the same node, so the check has
        // to run AFTER resolution. Comparing the raw arguments would let this
        // pair through and write an edge that carries no meaning.
        let server = plain_server().await;
        let a = seed(&server, "decision", "Alpha", "first").await;

        let out = server
            .relate(Parameters(RelateArgs {
                from: a[..HANDLE_LEN].to_string(),
                to: a.clone(),
                kind: "mentions".to_string(),
            }))
            .await;

        assert!(out.contains("same node"), "{out}");
    }

    #[tokio::test]
    async fn relate_reports_an_unknown_handle_without_writing() {
        let server = plain_server().await;
        let a = seed(&server, "decision", "Alpha", "first").await;

        let out = server
            .relate(Parameters(RelateArgs {
                from: a,
                to: "0000000000000".to_string(),
                kind: "mentions".to_string(),
            }))
            .await;

        assert!(out.starts_with("relate failed: to:"), "{out}");
        assert!(out.contains("no live node"), "{out}");
        // The "without writing" half of the name, asserted rather than assumed.
        // Community detection builds its node set purely from live edges, so a
        // zero count means nothing landed in the table.
        assert_eq!(
            server
                .store()
                .await
                .unwrap()
                .recompute_communities()
                .await
                .unwrap(),
            0,
            "a refused relate still wrote an edge"
        );
    }

    #[tokio::test]
    async fn relation_types_are_case_normalised_into_one_relation() {
        // `Mentions` and `mentions` are distinct keys in the clustering dedup,
        // so leaving both would double the pair's weight on nothing worse than
        // inconsistent capitalisation. Normalising at this boundary is also
        // what lets the reserved-word check be an exact comparison.
        let server = plain_server().await;
        let a = seed(&server, "decision", "Alpha", "first").await;
        let b = seed(&server, "decision", "Beta", "second").await;
        let relate = |kind: &str| {
            server.relate(Parameters(RelateArgs {
                from: a.clone(),
                to: b.clone(),
                kind: kind.to_string(),
            }))
        };

        let first = relate("mentions").await;
        let second = relate("Mentions").await;
        let third = relate("  MENTIONS  ").await;

        assert!(first.starts_with("related "), "{first}");
        assert!(second.contains("already relates"), "{second}");
        assert!(third.contains("already relates"), "{third}");
        assert!(
            first.contains("-mentions->"),
            "type not normalised: {first}"
        );
    }

    #[tokio::test]
    async fn relate_refuses_a_glob_wildcard_instead_of_matching_a_node() {
        // `*` is a GLOB wildcard, and resolution is the only place a
        // client-supplied string reaches a pattern. ONE seeded node on purpose:
        // with two, an unguarded `*` matches both and comes back as an ambiguous
        // handle, which still starts with "relate failed: from:" and lets the
        // test pass with the guard deleted. With one node it would resolve
        // successfully instead, to a node the caller never named.
        let server = plain_server().await;
        let only = seed(&server, "decision", "Alpha", "first").await;

        let out = server
            .relate(Parameters(RelateArgs {
                from: "*".to_string(),
                to: only,
                kind: "mentions".to_string(),
            }))
            .await;

        assert!(out.contains("no live node"), "wildcard resolved: {out}");
    }

    #[tokio::test]
    async fn relate_is_idempotent_on_the_same_ordered_triple() {
        let server = plain_server().await;
        let a = seed(&server, "decision", "Alpha", "first").await;
        let b = seed(&server, "decision", "Beta", "second").await;
        let args = || RelateArgs {
            from: a.clone(),
            to: b.clone(),
            kind: "mentions".to_string(),
        };

        let first = server.relate(Parameters(args())).await;
        let second = server.relate(Parameters(args())).await;

        assert!(first.starts_with("related "), "{first}");
        assert!(second.contains("already relates"), "{second}");
    }

    #[tokio::test]
    async fn clusters_on_an_empty_store_says_so_and_does_not_panic() {
        let server = plain_server().await;

        let out = server
            .clusters(Parameters(ClustersArgs {
                k: None,
                members: None,
                mode: None,
            }))
            .await;

        assert_eq!(out, "no clusters yet");
    }

    #[tokio::test]
    async fn clusters_budget_is_a_fraction_of_the_configured_context_not_all_of_it() {
        // Enough groups, each with a label long enough to matter, that the
        // real (one tenth) budget forces truncation while the full 8192
        // context would not: catches the handler passing the whole context
        // through un-scaled.
        let server = plain_server().await;
        let pad = "x".repeat(300);
        for i in 0..15 {
            let a = seed(&server, "fact", &format!("group{i}a {pad}"), "x").await;
            let b = seed(&server, "fact", &format!("group{i}b {pad}"), "x").await;
            server
                .relate(Parameters(RelateArgs {
                    from: a,
                    to: b,
                    kind: "mentions".to_string(),
                }))
                .await;
        }

        let out = server
            .clusters(Parameters(ClustersArgs {
                k: None,
                members: None,
                mode: None,
            }))
            .await;

        assert!(out.contains("withheld"), "{out}");
    }

    #[tokio::test]
    async fn clusters_reports_a_group_for_two_related_memories() {
        let server = plain_server().await;
        let a = seed(&server, "fact", "Rollout plan", "x").await;
        let b = seed(&server, "fact", "Rollout owner", "y").await;
        server
            .relate(Parameters(RelateArgs {
                from: a.clone(),
                to: b.clone(),
                kind: "mentions".to_string(),
            }))
            .await;

        let out = server
            .clusters(Parameters(ClustersArgs {
                k: None,
                members: None,
                mode: None,
            }))
            .await;

        assert!(out.contains("Rollout plan"), "{out}");
        assert!(out.contains("Rollout owner"), "{out}");
    }

    #[tokio::test]
    async fn a_handle_rendered_by_clusters_resolves_through_relate() {
        // `clusters` and `recall` must resolve through the same handle
        // scheme, so a client acting on what `clusters` shows does not need
        // a second round trip through `recall` first.
        let server = plain_server().await;
        let a = seed(&server, "fact", "Rollout plan", "first").await;
        let b = seed(&server, "fact", "Rollout owner", "second").await;
        let c = seed(&server, "fact", "Unrelated", "third").await;
        let related = server
            .relate(Parameters(RelateArgs {
                from: a.clone(),
                to: b.clone(),
                kind: "mentions".to_string(),
            }))
            .await;
        assert!(related.starts_with("related "), "{related}");

        let rendered = server
            .clusters(Parameters(ClustersArgs {
                k: None,
                members: None,
                mode: None,
            }))
            .await;
        let handle = rendered
            .lines()
            .find_map(|l| l.strip_prefix("[fact "))
            .and_then(|rest| rest.split(']').next())
            .unwrap_or_else(|| panic!("no member line with a handle: {rendered}"))
            .to_string();

        let linked = server
            .relate(Parameters(RelateArgs {
                from: handle,
                to: c.clone(),
                kind: "mentions".to_string(),
            }))
            .await;
        assert!(
            linked.starts_with("related "),
            "handle did not resolve: {linked}"
        );
    }

    #[tokio::test]
    async fn clusters_is_registered_with_the_argument_names_clients_send() {
        let server = plain_server().await;

        assert!(
            server.tool_router.has_route("clusters"),
            "clusters not routed"
        );
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|t| t.name == "clusters")
            .expect("clusters must be listed");
        let schema = serde_json::to_string(&tool.input_schema).unwrap();
        for field in ["\"k\"", "\"members\""] {
            assert!(schema.contains(field), "{field} missing from {schema}");
        }
    }

    #[tokio::test]
    async fn timeline_returns_compiled_content_and_mentions_most_recent_first() {
        // Given an entity with a compiled page and 2 live mentions, linked
        // at two different instants
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        let mut entity_node = NewNode::entity("person", "Ada Lovelace");
        entity_node.content = "compiled bio of ada".to_string();
        let entity_id = server
            .store()
            .await
            .unwrap()
            .insert(entity_node)
            .await
            .expect("seed entity");
        let fact1 = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::now("fact", "Fact One", "first mention content"))
            .await
            .expect("seed fact one");
        let fact2 = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::now("fact", "Fact Two", "second mention content"))
            .await
            .expect("seed fact two");
        clock.set(Millis(2000));
        server
            .store()
            .await
            .unwrap()
            .relate(&entity_id, &fact1, relation::MENTIONS)
            .await
            .expect("link fact one");
        clock.set(Millis(3000));
        server
            .store()
            .await
            .unwrap()
            .relate(&entity_id, &fact2, relation::MENTIONS)
            .await
            .expect("link fact two");

        // When timeline runs with no caller-supplied limit
        let out = server
            .timeline(Parameters(TimelineArgs {
                entity: entity_id.as_str().to_string(),
                limit: None,
                mode: None,
            }))
            .await;

        // Then the compiled content and both mentions appear, most
        // recently linked mention first, with no "more" trailer
        assert!(out.contains("compiled bio of ada"), "{out}");
        let fact1_pos = out.find("first mention content").expect("fact one missing");
        let fact2_pos = out
            .find("second mention content")
            .expect("fact two missing");
        assert!(
            fact2_pos < fact1_pos,
            "most recently linked mention should come first: {out}"
        );
        assert!(!out.contains("more mentions exist"), "{out}");
    }

    #[tokio::test]
    async fn timeline_reports_an_unresolvable_handle_as_not_found() {
        // Given a handle that resolves to no node
        let server = plain_server().await;

        // When timeline runs
        let out = server
            .timeline(Parameters(TimelineArgs {
                entity: "0000000000000".to_string(),
                limit: None,
                mode: None,
            }))
            .await;

        // Then it fails clearly instead of panicking or succeeding empty
        assert!(out.starts_with("timeline failed:"), "{out}");
        assert!(out.contains("no live node"), "{out}");
    }

    #[tokio::test]
    async fn timeline_reports_not_live_for_a_node_with_a_future_valid_from() {
        // Given a resolvable handle whose node isn't valid-time-live yet
        let server = plain_server().await;
        let one_day_ms = 24 * 60 * 60 * 1000;
        let future_entity = NewNode::entity("person", "Future Person")
            .with_valid_from(Millis(Millis::now().0 + one_day_ms));
        let entity_id = server
            .store()
            .await
            .unwrap()
            .insert(future_entity)
            .await
            .expect("seed future entity");

        // When timeline runs
        let out = server
            .timeline(Parameters(TimelineArgs {
                entity: entity_id.as_str().to_string(),
                limit: None,
                mode: None,
            }))
            .await;

        // Then it reports the node as not currently live, not a crash
        // or an unresolvable-handle error
        assert_eq!(
            out,
            format!(
                "timeline failed: {} is not currently live",
                entity_id.handle()
            )
        );
    }

    #[tokio::test]
    async fn timeline_on_an_entity_with_no_mentions_says_so() {
        // Given an entity created directly, with no mentions edges at all
        let server = plain_server().await;
        let entity_id = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::entity("person", "Lonely Entity"))
            .await
            .expect("seed entity");

        // When timeline runs
        let out = server
            .timeline(Parameters(TimelineArgs {
                entity: entity_id.as_str().to_string(),
                limit: None,
                mode: None,
            }))
            .await;

        // Then it returns the entity's own data with an explicit
        // "no mentions yet" section, not an error
        assert!(out.contains("Lonely Entity"), "{out}");
        assert!(out.contains("no mentions yet"), "{out}");
        assert!(!out.starts_with("timeline failed:"), "{out}");
    }

    #[tokio::test]
    async fn timeline_on_a_non_entity_node_returns_its_own_data_without_crashing() {
        // Given a handle that resolves to a live node that was never an
        // entity (an ordinary remembered fact)
        let server = plain_server().await;
        let handle = seed(&server, "fact", "Ordinary Fact", "ordinary content").await;

        // When timeline runs
        let out = server
            .timeline(Parameters(TimelineArgs {
                entity: handle,
                limit: None,
                mode: None,
            }))
            .await;

        // Then it still returns that node's own data and an empty
        // mentions list, without crashing
        assert!(out.contains("Ordinary Fact"), "{out}");
        assert!(out.contains("ordinary content"), "{out}");
        assert!(out.contains("no mentions yet"), "{out}");
    }

    #[test]
    fn clamp_timeline_limit_defaults_and_bounds_the_mention_count() {
        // Arrange / Act / Assert
        assert_eq!(
            clamp_timeline_limit(None),
            DEFAULT_TIMELINE_LIMIT,
            "absent limit defaults"
        );
        assert_eq!(
            clamp_timeline_limit(Some(10)),
            10,
            "in-range limit passes through"
        );
        assert_eq!(
            clamp_timeline_limit(Some(MAX_TIMELINE_LIMIT)),
            MAX_TIMELINE_LIMIT,
            "the cap itself is allowed"
        );
        assert_eq!(
            clamp_timeline_limit(Some(10_000)),
            MAX_TIMELINE_LIMIT,
            "oversized limit is capped, not passed to the store"
        );
        assert_eq!(
            clamp_timeline_limit(Some(0)),
            1,
            "limit=0 shows one mention, not none"
        );
    }

    #[tokio::test]
    async fn timeline_truncates_to_the_caller_supplied_limit_and_appends_a_trailer() {
        // Given an entity with 3 live mentions, linked at three different
        // instants, and a caller-supplied limit of 2
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        let entity_id = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::entity("person", "Ada Lovelace"))
            .await
            .expect("seed entity");
        let oldest = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::now(
                "fact",
                "Fact Oldest",
                "oldest mention content",
            ))
            .await
            .expect("seed oldest fact");
        let middle = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::now(
                "fact",
                "Fact Middle",
                "middle mention content",
            ))
            .await
            .expect("seed middle fact");
        let newest = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::now(
                "fact",
                "Fact Newest",
                "newest mention content",
            ))
            .await
            .expect("seed newest fact");
        clock.set(Millis(2000));
        server
            .store()
            .await
            .unwrap()
            .relate(&entity_id, &oldest, relation::MENTIONS)
            .await
            .expect("link oldest fact");
        clock.set(Millis(3000));
        server
            .store()
            .await
            .unwrap()
            .relate(&entity_id, &middle, relation::MENTIONS)
            .await
            .expect("link middle fact");
        clock.set(Millis(4000));
        server
            .store()
            .await
            .unwrap()
            .relate(&entity_id, &newest, relation::MENTIONS)
            .await
            .expect("link newest fact");

        // When timeline runs with limit: Some(2)
        let out = server
            .timeline(Parameters(TimelineArgs {
                entity: entity_id.as_str().to_string(),
                limit: Some(2),
                mode: None,
            }))
            .await;

        // Then exactly the 2 most recently linked mentions show, followed
        // by the trailer, and the excluded oldest mention is absent
        assert!(out.contains("newest mention content"), "{out}");
        assert!(out.contains("middle mention content"), "{out}");
        assert!(!out.contains("oldest mention content"), "{out}");
        assert!(out.contains("more mentions exist"), "{out}");
    }

    #[tokio::test]
    async fn timeline_shows_all_mentions_with_no_trailer_when_count_equals_the_limit() {
        // Given an entity with exactly 2 live mentions, linked at two
        // different instants, and a caller-supplied limit of 2
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        let entity_id = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::entity("person", "Ada Lovelace"))
            .await
            .expect("seed entity");
        let fact1 = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::now("fact", "Fact One", "first mention content"))
            .await
            .expect("seed fact one");
        let fact2 = server
            .store()
            .await
            .unwrap()
            .insert(NewNode::now("fact", "Fact Two", "second mention content"))
            .await
            .expect("seed fact two");
        clock.set(Millis(2000));
        server
            .store()
            .await
            .unwrap()
            .relate(&entity_id, &fact1, relation::MENTIONS)
            .await
            .expect("link fact one");
        clock.set(Millis(3000));
        server
            .store()
            .await
            .unwrap()
            .relate(&entity_id, &fact2, relation::MENTIONS)
            .await
            .expect("link fact two");

        // When timeline runs with limit: Some(2)
        let out = server
            .timeline(Parameters(TimelineArgs {
                entity: entity_id.as_str().to_string(),
                limit: Some(2),
                mode: None,
            }))
            .await;

        // Then both mentions show and no trailer appears (the "+1"
        // over-fetch comparison is strictly greater-than, not >=)
        assert!(out.contains("first mention content"), "{out}");
        assert!(out.contains("second mention content"), "{out}");
        assert!(!out.contains("more mentions exist"), "{out}");
    }

    #[tokio::test]
    async fn timeline_is_registered_with_the_argument_names_clients_send() {
        let server = plain_server().await;

        assert!(
            server.tool_router.has_route("timeline"),
            "timeline not routed"
        );
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|t| t.name == "timeline")
            .expect("timeline must be listed");
        let schema = serde_json::to_string(&tool.input_schema).unwrap();
        assert!(
            schema.contains("\"entity\""),
            "entity missing from {schema}"
        );
    }

    #[tokio::test]
    async fn recall_as_of_returns_only_the_version_live_at_that_instant() {
        // Arrange: two versions of one subject, written at two clock instants.
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        remember_version(&server, "v1", "zorbnax price is 10").await;
        clock.set(Millis(2000));
        remember_version(&server, "v2", "zorbnax price is 20").await;

        // Act: as_of pinned to the first write's instant.
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax price".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: Some(1000),
                mode: None,
            }))
            .await;

        // Assert: only the first version was live at t0.
        assert!(out.contains("] v1"), "missing first version: {out}");
        assert!(!out.contains("] v2"), "second version leaked: {out}");
    }

    #[tokio::test]
    async fn ask_as_of_reflects_only_the_version_live_at_that_instant() {
        // Arrange: same two-version setup, checked through ask's evidence
        // rather than recall's rendered text.
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        remember_version(&server, "v1", "zorbnax price is 10").await;
        clock.set(Millis(2000));
        remember_version(&server, "v2", "zorbnax price is 20").await;

        // Act: as_of pinned to the first write's instant.
        let answer = server
            .ask(Parameters(AskArgs {
                question: "What is the zorbnax price?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: Some(1000),
                mode: None,
            }))
            .await;

        // Assert: only the first version's content reached the model as
        // evidence (the echo llm repeats whatever it was handed).
        assert!(
            answer.contains("price is 10"),
            "missing first version: {answer}"
        );
        assert!(
            !answer.contains("price is 20"),
            "second version leaked: {answer}"
        );
    }

    #[tokio::test]
    async fn recall_as_of_before_the_first_write_returns_no_hits() {
        // Arrange: one node written at t0.
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock).await;
        remember_version(&server, "v1", "zorbnax price is 10").await;

        // Act: as_of set before the subject existed at all.
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax price".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: Some(500),
                mode: None,
            }))
            .await;

        // Assert
        assert_eq!(out, "no relevant memory", "expected zero hits: {out}");
    }

    #[tokio::test]
    async fn recall_as_of_after_all_writes_returns_only_the_current_version() {
        // Arrange: two versions of one subject.
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        remember_version(&server, "v1", "zorbnax price is 10").await;
        clock.set(Millis(2000));
        remember_version(&server, "v2", "zorbnax price is 20").await;

        // Act: as_of well after both writes.
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax price".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: Some(3000),
                mode: None,
            }))
            .await;

        // Assert: exactly one hit, and it is the current version, not the
        // superseded one this pins exclusion of.
        assert_eq!(out.split("\n\n").count(), 1, "expected 1 block: {out}");
        assert!(out.contains("] v2"), "missing current version: {out}");
        assert!(!out.contains("] v1"), "superseded version leaked: {out}");
    }

    #[tokio::test]
    async fn recall_as_of_exactly_at_the_second_writes_valid_from_is_live() {
        // `live_at` uses `valid_from <= t` and `tx_from <= t`, both
        // inclusive, so as_of pinned to the exact instant of a write already
        // sees that write; see `live_at` in `liam-store`'s `graph.rs`.
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        remember_version(&server, "v1", "zorbnax price is 10").await;
        clock.set(Millis(2000));
        remember_version(&server, "v2", "zorbnax price is 20").await;

        // Act: as_of pinned exactly to the second write's instant.
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax price".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: Some(2000),
                mode: None,
            }))
            .await;

        // Assert: the boundary instant already counts as live for the
        // second version, and the first is already superseded by then.
        assert_eq!(out.split("\n\n").count(), 1, "expected 1 block: {out}");
        assert!(
            out.contains("] v2"),
            "missing current version at boundary: {out}"
        );
        assert!(
            !out.contains("] v1"),
            "superseded version should not be live: {out}"
        );
    }

    #[tokio::test]
    async fn recall_without_as_of_behaves_like_no_time_filter() {
        // Arrange: a single node, as_of omitted.
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock).await;
        seed(&server, "fact", "Now", "zorbnax status is stable").await;

        // Act
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax status".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert: omitting as_of still finds the current version, the same
        // behavior every pre-WU-2 recall test already pins.
        assert!(out.contains("] Now"), "missing match: {out}");
    }

    #[tokio::test]
    async fn remember_then_recall_round_trips_confidence_and_attributes_through_the_tool_call() {
        // Given a fixture written with attributes, confidence, and valid_from
        // all set, through the real `remember` tool call rather than a direct
        // store write like WU-1's per-field tests used.
        let server = plain_server().await;
        let out = server
            .remember(Parameters(RememberArgs {
                attributes: Some(json!({"hue": "blue"})),
                valid_from: Some(1_700_000_000_000),
                confidence: Some(0.42),
                ..remember_args("full round trip via recall content")
            }))
            .await;
        assert!(out.starts_with("remembered "), "{out}");

        // When recall retrieves it through the real `recall` tool call
        let recalled = server
            .recall(Parameters(RecallArgs {
                query: "full round trip via recall content".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Then both fields carry the values written, proven across the
        // actual tool-call boundary end to end.
        assert!(recalled.contains("confidence: 0.42"), "{recalled}");
        assert!(
            recalled.contains(r#"attributes: {"hue":"blue"}"#),
            "{recalled}"
        );
    }

    #[tokio::test]
    async fn remember_then_ask_round_trips_confidence_and_attributes_through_the_tool_call() {
        // Same fixture as the recall round trip above, checked through `ask`
        // instead: `plain_server`'s echo llm repeats whatever evidence
        // reached the prompt, so this proves the same fields survive that
        // tool-call boundary too, not just recall's.
        let server = plain_server().await;
        let out = server
            .remember(Parameters(RememberArgs {
                attributes: Some(json!({"hue": "blue"})),
                valid_from: Some(1_700_000_000_000),
                confidence: Some(0.42),
                ..remember_args("full round trip via ask content")
            }))
            .await;
        assert!(out.starts_with("remembered "), "{out}");

        let answer = server
            .ask(Parameters(AskArgs {
                question: "full round trip via ask content".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        assert!(answer.contains("confidence: 0.42"), "{answer}");
        assert!(answer.contains(r#"attributes: {"hue":"blue"}"#), "{answer}");
    }

    #[tokio::test]
    async fn recall_as_of_round_trip_through_remember_and_recall_tool_calls() {
        // Given a subject remembered twice through the real `remember` tool
        // call, the second superseding the first exactly as `upsert_by`
        // semantics work. The clock is advanced explicitly between the two
        // writes rather than relying on the real clock's resolution, the
        // exact flakiness `server_with_clock` (WU-2) exists to avoid.
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let server = server_with_clock(clock.clone()).await;
        remember_version(&server, "before update", "zorbnax widget status is draft").await;
        clock.set(Millis(2000));
        remember_version(&server, "after update", "zorbnax widget status is shipped").await;

        // When recall, through the real `recall` tool call, is pinned to an
        // instant before the second write
        let out = server
            .recall(Parameters(RecallArgs {
                query: "zorbnax widget status".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: Some(1000),
                mode: None,
            }))
            .await;

        // Then only the pre-update version's label appears
        assert!(
            out.contains("] before update"),
            "missing pre-update version: {out}"
        );
        assert!(
            !out.contains("] after update"),
            "post-update version leaked: {out}"
        );
    }

    #[tokio::test]
    async fn remember_recall_ask_are_registered_with_the_argument_names_clients_send() {
        // As `relate_is_registered_with_the_argument_names_clients_send`:
        // every other test in this module calls these tools directly, so
        // none would notice a tool falling out of the router, or one of
        // WU-1/WU-2's new fields silently dropped from its schema.
        let server = plain_server().await;
        let tools = server.tool_router.list_all();

        for (name, fields) in [
            (
                "remember",
                vec!["\"attributes\"", "\"valid_from\"", "\"confidence\""],
            ),
            ("recall", vec!["\"as_of\""]),
            ("ask", vec!["\"as_of\""]),
        ] {
            assert!(server.tool_router.has_route(name), "{name} not routed");
            let tool = tools
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{name} must be listed"));
            let schema = serde_json::to_string(&tool.input_schema).unwrap();
            for field in fields {
                assert!(
                    schema.contains(field),
                    "{field} missing from {name}'s schema: {schema}"
                );
            }
        }
    }

    /// Yields until the shared slot is held or awaited by a writer.
    async fn until_contended(shared: &SharedStore) {
        for _ in 0..YIELDS {
            if shared.is_contended() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("no writer ever queued");
    }

    /// Yields enough times for a spawned task to run to its first wait.
    async fn settle() {
        for _ in 0..YIELDS {
            tokio::task::yield_now().await;
        }
    }

    const YIELDS: usize = 200;
    const BOUND: Duration = Duration::from_secs(10);

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

    #[tokio::test]
    async fn resynthesis_on_the_callers_store_finishes_while_a_restore_is_queued() {
        // Arrange: the caller holds a read guard and a writer queues behind it
        let (server, _embedder) =
            server_with_recording_embedder_and_llm(Arc::new(CountingLlm::new())).await;
        let shared = server.shared_store();
        let guard = server.store().await.unwrap();
        let entity_id = guard
            .upsert_by(NewNode::entity("person", "Kim"))
            .await
            .unwrap();
        let queued = {
            let shared = shared.clone();
            tokio::spawn(async move { drop(shared.exclusive().await) })
        };
        until_contended(&shared).await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

        // Act: a second read behind the queued writer would never return
        let resynthesized = tokio::time::timeout(
            BOUND,
            server.resynthesize_entity_as(&guard, entity_id, deadline, 64, None),
        )
        .await;
        drop(guard);

        // Assert
        assert!(matches!(resynthesized, Ok(Ok(()))), "{resynthesized:?}");
        tokio::time::timeout(BOUND, queued).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn a_request_during_a_swap_waits_and_is_then_served() {
        // Arrange: a restore holds the slot
        let (server, _embedder) = server_with_recording_embedder().await;
        let shared = server.shared_store();
        let swap = shared.exclusive().await;
        let request = {
            let server = server.clone();
            tokio::spawn(async move { server.recall(Parameters(recall_args())).await })
        };

        // Act
        settle().await;
        let waiting = !request.is_finished();
        drop(swap);
        let answer = tokio::time::timeout(BOUND, request).await.unwrap().unwrap();

        // Assert
        assert!(waiting, "the request must wait for the swap");
        assert!(!answer.contains("failed"), "{answer}");
    }

    #[tokio::test]
    async fn a_request_after_a_swap_that_lost_the_store_gets_a_clean_error() {
        // Arrange: the restore could not reopen the store
        let (server, _embedder) = server_with_recording_embedder().await;
        let shared = server.shared_store();
        let mut swap = shared.exclusive().await;
        *swap = None;
        let request = {
            let server = server.clone();
            tokio::spawn(async move { server.recall(Parameters(recall_args())).await })
        };
        settle().await;

        // Act
        drop(swap);
        let answer = tokio::time::timeout(BOUND, request).await.unwrap().unwrap();

        // Assert
        assert!(answer.starts_with("recall failed"), "{answer}");
        assert!(answer.contains("store is unavailable"), "{answer}");
    }

    // ---- strict and infer modes ----

    const PAGE_LABEL: &str = "Zorbnax";

    /// A recorded fact and an entity page that both match `zorbnax`, the page
    /// mentioning the fact. Returns the page's id.
    async fn fact_and_page(server: &MemoryServer) -> liam_store::NodeId {
        let fact = seed(server, "fact", "Preference", "zorbnax likes green tea").await;
        let store = server.store().await.unwrap();
        let mut page = NewNode::entity("person", PAGE_LABEL);
        page.content = "zorbnax profile: likes green tea".to_string();
        let page = store.upsert_by(page).await.unwrap();
        let fact = store.resolve_handle(&fact).await.unwrap();
        store
            .relate(&page, &fact, relation::MENTIONS)
            .await
            .unwrap();
        page
    }

    fn ask_args(mode: Option<Mode>) -> AskArgs {
        AskArgs {
            question: "What does zorbnax like?".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode,
        }
    }

    #[tokio::test]
    async fn recall_leaves_out_an_entity_page_and_refuses_infer_mode() {
        // Arrange
        let server = plain_server().await;
        fact_and_page(&server).await;
        let args = |mode| RecallArgs {
            query: "zorbnax".to_string(),
            mode,
            ..recall_args()
        };

        // Act
        let default = server.recall(Parameters(args(None))).await;
        let infer = server.recall(Parameters(args(Some(Mode::Infer)))).await;

        // Assert
        assert!(default.contains("green tea"), "{default}");
        assert!(!default.contains("profile"), "{default}");
        assert!(
            infer.starts_with("recall failed") && infer.contains("mode infer"),
            "{infer}"
        );
    }

    #[tokio::test]
    async fn ask_defaults_to_infer_with_labelled_evidence_and_an_inferred_answer() {
        // Arrange
        let llm = Arc::new(RecordingLlm::new("Zorbnax likes green tea [1]."));
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        fact_and_page(&server).await;

        // Act
        let answer = server.ask(Parameters(ask_args(None))).await;

        // Assert: the model's block is marked, the page is among the sources
        // labelled inferred, and the prompt carries derivations but no ids
        assert!(answer.starts_with("answer (inferred):\n"), "{answer}");
        assert!(
            answer.contains("provenance: derivation: inferred; basis: unknown"),
            "{answer}"
        );
        let prompt = llm.last_prompt();
        assert!(prompt.contains("derivation: inferred"), "{prompt}");
        assert!(prompt.contains("derivation: recorded"), "{prompt}");
        assert!(!prompt.contains("provenance:"), "{prompt}");
    }

    #[tokio::test]
    async fn ask_in_strict_mode_makes_no_model_call_and_shows_no_inferred_row() {
        // Arrange
        let llm = Arc::new(RecordingLlm::new("Zorbnax likes green tea [1]."));
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm.clone()).await;
        fact_and_page(&server).await;

        // Act
        let answer = server.ask(Parameters(ask_args(Some(Mode::Strict)))).await;

        // Assert
        assert!(llm.seen.lock().unwrap().is_empty(), "the model was called");
        assert!(answer.starts_with("(strict mode:"), "{answer}");
        assert!(answer.contains("zorbnax likes green tea"), "{answer}");
        assert!(!answer.contains("answer (inferred)"), "{answer}");
        assert!(!answer.contains("profile"), "{answer}");
        assert!(!answer.contains("derivation: inferred"), "{answer}");
    }

    #[tokio::test]
    async fn the_ask_fallback_labels_every_evidence_block() {
        // Arrange
        let server =
            server_with(Arc::new(liam_model::IdentityReranker), Arc::new(FailingLlm)).await;
        fact_and_page(&server).await;

        // Act
        let answer = server.ask(Parameters(ask_args(None))).await;

        // Assert
        let blocks = answer.matches("<<<EVIDENCE ").count();
        let labelled = answer.matches("provenance: derivation: ").count();
        assert_eq!(blocks, 2, "{answer}");
        assert_eq!(labelled, blocks, "{answer}");
        assert!(answer.contains("derivation: inferred"), "{answer}");
    }

    #[tokio::test]
    async fn clusters_leave_out_a_page_that_mentions_recorded_memory() {
        // Arrange
        let server = plain_server().await;
        let page = fact_and_page(&server).await;
        let other = seed(&server, "fact", "Rollout owner", "y").await;
        let store = server.store().await.unwrap();
        let other = store.resolve_handle(&other).await.unwrap();
        let fact = store
            .mentions(&page, liam_store::Millis::now(), 10)
            .await
            .unwrap()
            .remove(0)
            .id;
        store
            .relate(&fact, &other, relation::MENTIONS)
            .await
            .unwrap();
        drop(store);

        // Act
        let out = server
            .clusters(Parameters(ClustersArgs {
                k: None,
                members: None,
                mode: None,
            }))
            .await;
        let infer = server
            .clusters(Parameters(ClustersArgs {
                k: None,
                members: None,
                mode: Some(Mode::Infer),
            }))
            .await;

        // Assert
        assert!(out.contains("Preference"), "{out}");
        assert!(out.contains("Rollout owner"), "{out}");
        assert!(!out.contains(PAGE_LABEL), "{out}");
        assert!(infer.contains("mode infer"), "{infer}");
    }

    #[tokio::test]
    async fn timeline_shows_the_page_labelled_inferred_and_refuses_infer_mode() {
        // Arrange
        let server = plain_server().await;
        let page = fact_and_page(&server).await;
        let args = |mode| TimelineArgs {
            entity: page.as_str().to_string(),
            limit: None,
            mode,
        };

        // Act
        let out = server.timeline(Parameters(args(None))).await;
        let infer = server.timeline(Parameters(args(Some(Mode::Infer)))).await;

        // Assert
        let header: Vec<&str> = out.lines().take(3).collect();
        assert!(
            header[0].starts_with(&format!("[person {}", page.handle())),
            "{out}"
        );
        assert!(
            header[2].starts_with("provenance: derivation: inferred"),
            "{out}"
        );
        assert!(out.contains("zorbnax likes green tea"), "{out}");
        assert!(infer.contains("mode infer"), "{infer}");
    }

    fn metered(server: MemoryServer) -> (MemoryServer, Arc<crate::metrics::sink::MemorySink>) {
        let clock = Arc::new(FixedClock::new(Millis(1_000)));
        let sink = Arc::new(crate::metrics::sink::MemorySink::new(clock.clone()));
        let metrics = Arc::new(Metrics::new(sink.clone(), clock, 2));
        (server.with_metrics(metrics), sink)
    }

    fn reported(
        sink: &crate::metrics::sink::MemorySink,
        metric: crate::metrics::sink::MetricId,
    ) -> bool {
        use crate::metrics::sink::MetricsSink;
        sink.snapshot()
            .readings
            .iter()
            .any(|reading| reading.metric == metric)
    }

    #[tokio::test]
    async fn recall_times_itself_and_the_arms_it_ran_and_not_ask() {
        use crate::metrics::sink::MetricId;
        // Arrange
        let (server, sink) = metered(plain_server().await);
        seed(&server, "fact", "One", "zorbnax note one").await;

        // Act
        server
            .recall(Parameters(RecallArgs {
                query: "zorbnax note".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert
        assert!(reported(&sink, MetricId::RecallLatencyP95Ms));
        assert!(reported(&sink, MetricId::ArmLatencyP95Ms));
        assert!(!reported(&sink, MetricId::AskLatencyP95Ms));
    }

    /// A clock that moves one millisecond on every read, so each stage lasts
    /// a fixed, nonzero time and the stages nest inside the total.
    struct TickingClock(std::sync::atomic::AtomicI64);

    impl liam_store::Clock for TickingClock {
        fn now(&self) -> Millis {
            Millis(self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
        }
    }

    fn samples_of(
        sink: &crate::metrics::sink::MemorySink,
        metric: crate::metrics::sink::MetricId,
    ) -> u64 {
        use crate::metrics::sink::MetricsSink;
        sink.snapshot()
            .readings
            .iter()
            .filter(|reading| reading.metric == metric)
            .map(|reading| reading.samples)
            .sum()
    }

    fn latest(
        sink: &crate::metrics::sink::MemorySink,
        metric: crate::metrics::sink::MetricId,
    ) -> f64 {
        use crate::metrics::sink::MetricsSink;
        sink.snapshot()
            .readings
            .iter()
            .find(|reading| reading.metric == metric)
            .map_or(f64::NAN, |reading| reading.value)
    }

    #[tokio::test]
    async fn recall_records_one_sample_per_stage_and_the_stages_fit_in_the_total() {
        use crate::metrics::sink::{MemorySink, MetricId};
        // Arrange
        let clock = Arc::new(TickingClock(std::sync::atomic::AtomicI64::new(1_000)));
        let sink = Arc::new(MemorySink::new(clock.clone()));
        let metrics = Arc::new(Metrics::new(sink.clone(), clock, 2));
        let server = plain_server().await.with_metrics(metrics);
        seed(&server, "fact", "One", "zorbnax note one").await;
        let stages = [
            MetricId::StageEmbedP50Ms,
            MetricId::StageStoreSearchP50Ms,
            MetricId::StageRerankP50Ms,
            MetricId::StageRecordUsageP50Ms,
            MetricId::StageIssueHandlesP50Ms,
        ];

        // Act
        server
            .recall(Parameters(RecallArgs {
                query: "zorbnax note".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert
        for stage in stages {
            assert_eq!(samples_of(&sink, stage), 1, "{stage}");
        }
        assert_eq!(samples_of(&sink, MetricId::RecallLatencyP50Ms), 1);
        let staged: f64 = stages.into_iter().map(|stage| latest(&sink, stage)).sum();
        assert!(staged <= latest(&sink, MetricId::RecallLatencyP50Ms));
    }

    /// Embeds like `MockEmbedder` after waiting `delay`.
    struct DelayedEmbedder {
        delay: Duration,
        inner: liam_model::MockEmbedder,
    }

    #[async_trait::async_trait]
    impl liam_model::Embedder for DelayedEmbedder {
        fn dims(&self) -> usize {
            self.inner.dims()
        }
        async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
            tokio::time::sleep(self.delay).await;
            self.inner.embed(text, kind).await
        }
    }

    /// Scores after waiting `delay`.
    struct DelayedReranker(Duration);

    #[async_trait::async_trait]
    impl Reranker for DelayedReranker {
        async fn scores(&self, _q: &str, docs: &[String]) -> liam_model::Result<Vec<f32>> {
            tokio::time::sleep(self.0).await;
            Ok(vec![0.0; docs.len()])
        }
    }

    #[tokio::test]
    async fn ask_records_one_sample_per_stage() {
        use crate::metrics::sink::{MemorySink, MetricId};
        // Arrange
        let clock = Arc::new(TickingClock(std::sync::atomic::AtomicI64::new(1_000)));
        let sink = Arc::new(MemorySink::new(clock.clone()));
        let metrics = Arc::new(Metrics::new(sink.clone(), clock, 2));
        let llm = Arc::new(RecordingLlm::new(
            "The zorbnax gadget uses libSQL for storage [1].",
        ));
        let server = server_with(Arc::new(liam_model::IdentityReranker), llm)
            .await
            .with_metrics(metrics);
        seed(
            &server,
            "decision",
            "Storage engine",
            "The zorbnax gadget uses libSQL for storage.",
        )
        .await;
        let stages = [
            MetricId::StageEmbedP50Ms,
            MetricId::StageStoreSearchP50Ms,
            MetricId::StageRerankP50Ms,
            MetricId::StageRecordUsageP50Ms,
            MetricId::StageIssueHandlesP50Ms,
        ];

        // Act
        server
            .ask(Parameters(ask_args_for(
                "What storage engine does the zorbnax gadget use?",
            )))
            .await;

        // Assert
        for stage in stages {
            assert_eq!(samples_of(&sink, stage), 1, "{stage}");
        }
        assert_eq!(samples_of(&sink, MetricId::AskLatencyP50Ms), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_degraded_rerank_records_no_rerank_sample() {
        use crate::metrics::sink::{MemorySink, MetricId};
        // Arrange
        const SLOW: Duration = Duration::from_secs(30);
        const NEARLY_ALL_THE_BUDGET: Duration = Duration::from_millis(900);
        let cases: [(&str, Duration, Arc<dyn Reranker>); 3] = [
            (
                "skipped",
                NEARLY_ALL_THE_BUDGET,
                Arc::new(DelayedReranker(Duration::ZERO)),
            ),
            ("failed", Duration::ZERO, Arc::new(FailingReranker)),
            ("timeout", Duration::ZERO, Arc::new(DelayedReranker(SLOW))),
        ];
        for (case, embed_delay, reranker) in cases {
            let clock = Arc::new(TickingClock(std::sync::atomic::AtomicI64::new(1_000)));
            let sink = Arc::new(MemorySink::new(clock.clone()));
            let metrics = Arc::new(Metrics::new(sink.clone(), clock, 2));
            let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
                .await
                .expect("open in-memory store");
            let server = MemoryServer::new(
                Arc::new(store),
                Arc::new(DelayedEmbedder {
                    delay: embed_delay,
                    inner: liam_model::MockEmbedder::new(8),
                }),
                reranker,
                Arc::new(liam_model::MockLlm),
                30,
                false,
                8192,
                1,
            )
            .with_recall_timeout(1)
            .with_metrics(metrics);
            seed(&server, "fact", "One", "zorbnax note one").await;

            // Act
            server
                .recall(Parameters(RecallArgs {
                    query: "zorbnax note".to_string(),
                    kind: None,
                    scope: None,
                    k: None,
                    as_of: None,
                    mode: None,
                }))
                .await;

            // Assert
            assert_eq!(samples_of(&sink, MetricId::RecallLatencyP50Ms), 1, "{case}");
            assert_eq!(samples_of(&sink, MetricId::StageEmbedP50Ms), 1, "{case}");
            assert_eq!(samples_of(&sink, MetricId::StageRerankP50Ms), 0, "{case}");
        }
    }

    #[tokio::test]
    async fn ask_times_itself() {
        use crate::metrics::sink::MetricId;
        // Arrange
        let (server, sink) = metered(plain_server().await);

        // Act
        server
            .ask(Parameters(AskArgs {
                question: "anything?".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
                mode: None,
            }))
            .await;

        // Assert
        assert!(reported(&sink, MetricId::AskLatencyP50Ms));
        assert!(!reported(&sink, MetricId::RecallLatencyP50Ms));
    }

    /// Rewrites the recall and ask schema fixture from the live tool list.
    /// Run it by hand with `--ignored` when an input schema changes on purpose.
    #[tokio::test]
    #[ignore = "writes the recall and ask schema fixture"]
    async fn capture_recall_ask_schemas() {
        // Arrange
        let server = plain_server().await;

        // Act
        let schemas = recall_ask_schemas(&server);
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/mcp/fixtures/recall_ask_schemas.json"
        );
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
        std::fs::write(path, serde_json::to_string_pretty(&schemas).unwrap() + "\n").unwrap();

        // Assert
        assert!(schemas["recall"].is_object() && schemas["ask"].is_object());
    }

    #[tokio::test]
    async fn recall_and_ask_list_the_schemas_clients_already_rely_on() {
        // Arrange
        let server = plain_server().await;
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("mcp/fixtures/recall_ask_schemas.json"))
                .expect("fixture is JSON");

        // Act
        let listed = recall_ask_schemas(&server);

        // Assert
        assert_eq!(listed, fixture);
    }

    /// The `recall` and `ask` input schemas as listed by the server, by tool name.
    fn recall_ask_schemas(server: &MemoryServer) -> serde_json::Value {
        let tools = server.tool_router.list_all();
        let mut out = serde_json::Map::new();
        for name in ["recall", "ask"] {
            let tool = tools
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{name} must be listed"));
            out.insert(
                name.to_string(),
                serde_json::to_value(&tool.input_schema).unwrap(),
            );
        }
        serde_json::Value::Object(out)
    }
}
