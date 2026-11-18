// SPDX-License-Identifier: Apache-2.0
//! MCP tool surface: `remember`, `recall`, `relate`, `ask`, `clusters`,
//! `timeline`, `facts`, `feedback`, `summaries`, `delete`, `consolidate` and
//! `metrics`, wiring the store and the model.
//!
//! VERSION CHECK: the rmcp macro surface (`#[tool_router]`, `#[tool]`,
//! `#[tool_handler]`, `Parameters`) moves across releases. Confirmed unchanged
//! from 2.x through rmcp 3.9.0, the version `Cargo.lock` resolves; re-confirm
//! against the rmcp version you resolve before bumping again.
//!
//! The daemon sets no cap on the size of an incoming message line. It relies on
//! rmcp's default of 16 MiB (`DEFAULT_MAX_LINE_LENGTH`, added in 3.6.0), so a
//! rmcp release that lowers or removes it changes what a client may send.

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
use liam_model::{Embedder, Llm, TextKind};
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

    fn stage_start(&self) -> Option<Duration> {
        self.metrics.as_ref().map(|metrics| metrics.now())
    }

    fn stage_end(&self, stage: Stage, started: Option<Duration>) {
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
        if let Some(feedback) = &self.feedback {
            let started = self.stage_start();
            feedback.issue(nodes);
            self.stage_end(Stage::IssueHandles, started);
        }
    }

    /// Counts one hit on each node a client was given as a ranked result.
    /// Neighbours pulled in beside a result are not credited.
    fn record_usage<'a>(&self, nodes: impl IntoIterator<Item = &'a NodeId>) {
        if let Some(recorder) = &self.usage {
            let started = self.stage_start();
            recorder.record(&self.producer(), nodes, Millis::now());
            self.stage_end(Stage::RecordUsage, started);
        }
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
        description = "Retrieve relevant long-term memory.",
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
                Some(crate::ask::candidate_pool(k)),
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
        let shown = &hits[..hits.len().min(k)];
        // Usage and handles cover only the hits returned, not the wider pool.
        self.record_usage(shown.iter().filter(|h| !h.expanded).map(|h| &h.hit.id));
        self.issue_handles(shown.iter().map(|h| &h.hit.id));
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
        let text = shown
            .iter()
            .map(|hit| {
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
        with_reasons(ToolReply::ok(text), [embed_degraded])
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

    /// Loads the model within the shared `deadline`, so the token count and the
    /// generation timer that follow start on a ready model. The error is the
    /// reason text `ask` hands to `fallback_answer`.
    async fn ensure_llm_loaded(&self, deadline: tokio::time::Instant) -> Result<(), &'static str> {
        match tokio::time::timeout_at(deadline, self.llm.ensure_loaded()).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err("the model could not be loaded"),
            Err(_) => Err("timed out loading the model"),
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
                Some(ask::candidate_pool(k)),
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
        let retrieval_reasons = [embed_degraded];
        // Candidates list the store hits first, so an index below `hit_count`
        // is a store hit. Keep k of them plus every summary.
        let hit_count = hits.len();
        let mut kept_hits = 0;
        let evidence: Vec<ask::Evidence> = candidates
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| {
                if i >= hit_count {
                    return true;
                }
                kept_hits += 1;
                kept_hits <= k
            })
            .map(|(_, candidate)| candidate)
            .collect();
        if mode == Mode::Strict {
            self.issue_handles(evidence.iter().filter_map(|e| e.node_id.as_ref()));
            self.record_evidence_usage(&evidence);
            return with_reasons(
                ToolReply::ok(ask::strict_answer(&evidence, self.max_basis_ids_rendered)),
                retrieval_reasons,
            );
        }
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

        // Trim ONCE, before the sufficiency pre-pass, and reuse this same
        // slice for both prompts below. Trimming twice, or only for the
        // answer prompt, would let the pre-pass vouch for evidence the answer
        // never sees. The model loads first, inside the request deadline, so
        // the count below is the real one. The closure prefers the model's
        // real count and only falls back to the estimate when the provider
        // cannot count.
        let loaded = self.ensure_llm_loaded(deadline).await;
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
        if let Err(reason) = loaded {
            return with_reasons(
                ToolReply::ok(fallback_answer(
                    reason,
                    evidence,
                    self.max_basis_ids_rendered,
                )),
                retrieval_reasons
                    .into_iter()
                    .chain([Some(Degraded::SynthesisUnavailable)]),
            );
        }

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
mod tests;
