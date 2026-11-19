// SPDX-License-Identifier: Apache-2.0
//! MCP tool surface: `remember`, `recall`, `relate`, `clusters`,
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
mod argument_errors;
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

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, OnceLock};
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
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use self::delete::DeleteArgs;
use self::episode::{resolved_id, validate_episode, EpisodeArgs};
use self::facts::FactsArgs;
use self::feedback::{Feedback, FeedbackArgs};
use self::metrics::MetricsArgs;
use self::reply::{RecalledPage, ToolReply, PAGE_KIND};
use self::summaries::SummariesArgs;
use crate::clusters::{narrow_groups, render_clusters};
use crate::config::{
    resolve_vocabulary, AdminConfig, ExtractionConfig, DEFAULT_MAX_BASIS_IDS_RENDERED,
    DEFAULT_MIN_SUMMARY_SIMILARITY, DEFAULT_RECALL_TIMEOUT_SECS,
};
use crate::consolidation_job::ConsolidationJob;
use crate::degraded::Degraded;
use crate::deletion::DeletePreview;
use crate::entity_page;
use crate::metrics::recorder::{Metrics, ToolTimer};
use crate::metrics::sink::{Stage, Tool};
use crate::prompt::estimate_tokens;
use crate::recall::{self, candidate_pool, clamp_recall_k};
use crate::render;
use crate::tiering::usage::UsageRecorder;

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
/// distinct constant from `prompt::MAX_EVIDENCE_CHARS`, not a reuse of it:
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

/// Cap on `recall`'s `query`, in characters.
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

/// Default and upper bound on the number of mentions `timeline` shows.
/// WHY: caller-supplied `limit` is otherwise unbounded, and each mention's
/// full content lands in the response, so a large `limit` on a well-mentioned
/// entity would blow the response size. The default keeps `timeline`'s
/// existing behavior when `limit` is omitted, and the cap bounds a single
/// response regardless of how many mentions the entity has.
const DEFAULT_TIMELINE_LIMIT: usize = 50;
const MAX_TIMELINE_LIMIT: usize = 200;

/// Resolve the caller's `limit` into the usable range: absent means
/// `DEFAULT_TIMELINE_LIMIT`, and anything else is clamped to
/// `1..=MAX_TIMELINE_LIMIT`. WHY clamp the bottom too: a `limit = 0` would render no mentions, which reads as a claim
/// that the entity has none rather than a claim about the argument.
fn clamp_timeline_limit(limit: Option<usize>) -> usize {
    limit
        .unwrap_or(DEFAULT_TIMELINE_LIMIT)
        .clamp(1, MAX_TIMELINE_LIMIT)
}

/// Builds the `Query` a `recall` runs from its k, embedding, kind, scope and
/// as_of.
fn build_query(
    text: &str,
    k: Option<usize>,
    kind: Option<String>,
    scope: Option<String>,
    embedding: Option<Vec<f32>>,
    as_of: Option<i64>,
) -> Query {
    let mut q = Query::text(text.to_string()).with_k(k.unwrap_or(8));
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

/// Validates a `recall` query: `None` when usable.
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
    /// Wall-clock deadline for one batch of entity-page builds (embedding the
    /// mentions and the page); see `config::Config::ask_timeout_secs`, whose
    /// name predates the removal of `ask`.
    synthesis_timeout_secs: u64,
    /// Seconds `recall` may spend retrieving; see `retrieval_deadline`.
    recall_timeout_secs: u64,
    /// The model's context window in tokens, see `config::LlmConfig::context_tokens`.
    /// `clusters` takes one tenth of it as its output budget (ADR-0002 S4).
    context_tokens: usize,
    /// Event ids a provenance line lists before it counts the rest; see
    /// `config::SummaryConfig::max_basis_ids_rendered`.
    max_basis_ids_rendered: usize,
    /// Least similarity of a query to a summary for `summaries` to list it; see
    /// `config::SummaryConfig::min_summary_similarity`.
    min_summary_similarity: f32,
    /// Bounds how many generations may be inside a model call at once; see
    /// `config::LlmConfig::max_concurrent_generations`. A semaphore, not a
    /// lock, so an operator with memory headroom can raise the limit above 1
    /// instead of every request serializing permanently.
    generation_permits: Arc<Semaphore>,
    /// Authoritative total permit count the cold-start benchmark sets, distinct
    /// from `generation_permits.available_permits()`.
    granted_capacity: Arc<AtomicUsize>,
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
    /// Counts the results `recall` returns, when tiering is on.
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
    /// Where `recall` reports its timings, when metrics are on.
    metrics: Option<Arc<Metrics>>,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
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
    // established pattern in this constructor (see `synthesis_timeout_secs`
    // and `context_tokens` above); a config struct
    // parameter would hide which fields `MemoryServer` actually depends on.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<DefaultGraph>,
        embedder: Arc<dyn Embedder>,
        llm: Arc<dyn Llm>,
        synthesis_timeout_secs: u64,
        context_tokens: usize,
        max_concurrent_generations: usize,
    ) -> Self {
        // `Semaphore::new(0)` would deadlock every entity synthesis forever. `0` means
        // auto-tune (see `main.rs::build_autotuned_server`); this floor only
        // matters for a direct explicit-`0` construction that bypasses it.
        let max_concurrent_generations = max_concurrent_generations.max(1);
        Self {
            store: Arc::new(SharedStore::from_arc(store)),
            embedder,
            llm,
            synthesis_timeout_secs,
            recall_timeout_secs: DEFAULT_RECALL_TIMEOUT_SECS,
            context_tokens,
            max_basis_ids_rendered: DEFAULT_MAX_BASIS_IDS_RENDERED,
            min_summary_similarity: DEFAULT_MIN_SUMMARY_SIMILARITY,
            generation_permits: Arc::new(Semaphore::new(max_concurrent_generations)),
            granted_capacity: Arc::new(AtomicUsize::new(max_concurrent_generations)),
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

    /// Sets the seconds `recall` may spend retrieving. Stored as at
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

    /// Lists summaries at least this close to a query in `summaries`.
    pub(crate) fn with_min_summary_similarity(mut self, min: f32) -> Self {
        self.min_summary_similarity = min;
        self
    }

    /// Uses the operator's resolved predicate vocabulary for `facts`.
    pub(crate) fn with_vocabulary(mut self, vocabulary: Vocabulary) -> Self {
        self.vocabulary = Arc::new(vocabulary);
        self
    }

    /// Counts the results of `recall` in `recorder`.
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

    /// Reports the timings of `recall` to `metrics`.
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

    /// The ranking of recorded and inferred rows together for `q`, which says
    /// where a page stands against the memories.
    async fn explained_with_pages(
        &self,
        q: &Query,
    ) -> Result<Vec<liam_store::ExplainedHit>, liam_store::Error> {
        let joint = q.clone().with_inferred(true);
        on_store!(self, |store| store.query_explained(&joint))
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
                    // so one slow entity cannot stretch the whole request.
                    let entity_synthesis_deadline = tokio::time::Instant::now()
                        + Duration::from_secs(self.synthesis_timeout_secs.max(1));
                    // Spawned to run concurrently; collection stays
                    // sequential so no outcome is ever dropped.
                    let mut handles = Vec::with_capacity(triggered.len());
                    for entity_id in triggered {
                        let server = self.clone();
                        let deadline = entity_synthesis_deadline;
                        handles.push(tokio::spawn(async move {
                            server.resynthesize_entity(entity_id, deadline).await
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

    /// Rebuilds one entity's page from its mentions within `remember`'s shared
    /// batch deadline; a failure is reported, never panicked.
    pub(crate) async fn resynthesize_entity(
        &self,
        entity_id: liam_store::NodeId,
        deadline: tokio::time::Instant,
    ) -> Result<(), String> {
        // Held through the embedder calls: a rollback waits for them.
        let store = self
            .store()
            .await
            .map_err(|e| format!("{}: {e}", entity_id.as_str()))?;
        self.resynthesize_entity_as(&store, entity_id, deadline, None)
            .await
    }

    /// `resynthesize_entity` with the new page written under `write_source`
    /// instead of the server's producer, so a consolidation run's rollback
    /// cancels the write and the supersession that comes with it. Works on the
    /// caller's `store`, never locking the slot again: a second read behind a
    /// queued restore would deadlock.
    ///
    /// The steps: read the entity and its recorded mentions, give each a
    /// vector, build the page, embed it, write it. An entity with no mention
    /// gets no page.
    pub(crate) async fn resynthesize_entity_as(
        &self,
        store: &DefaultGraph,
        entity_id: liam_store::NodeId,
        deadline: tokio::time::Instant,
        write_source: Option<&str>,
    ) -> Result<(), String> {
        let now = liam_store::Millis::now();
        let candidate = match store.get(&entity_id, now).await {
            Ok(Some(c)) => c,
            Ok(None) => {
                return Err(format!(
                    "{}: entity vanished before its page could be built",
                    entity_id.as_str()
                ))
            }
            Err(e) => return Err(format!("{}: {e}", entity_id.as_str())),
        };
        let label = candidate.label.clone();
        self.write_page(store, candidate, deadline, write_source)
            .await
            .map_err(|e| format!("{label}: {e}"))
    }

    /// The steps after the entity is read; an error carries no entity label,
    /// the caller adds it.
    async fn write_page(
        &self,
        store: &DefaultGraph,
        candidate: liam_store::Candidate,
        deadline: tokio::time::Instant,
        write_source: Option<&str>,
    ) -> Result<(), String> {
        let label = candidate.label.clone();
        let mut rows = store
            .mentions(
                &candidate.id,
                liam_store::Millis::now(),
                MENTIONS_FETCH_LIMIT,
            )
            .await
            .map_err(|e| e.to_string())?;
        // A summary or an earlier page that mentions the entity is derived
        // text, so a page built from it would state a statement twice removed.
        rows.retain(|row| !row.retrieved.is_inferred());
        // A scoped entity is read inside its scope, so its page may quote only
        // memories a reader of that scope can already see.
        if let Some(scope) = &candidate.scope {
            rows.retain(|row| {
                row.scope
                    .as_deref()
                    .is_some_and(|s| liam_store::scope_is_within(s, scope))
            });
        }
        tracing::debug!(entity = %label, mentions = rows.len(), "building the entity page");
        let mentions = tokio::time::timeout_at(deadline, self.page_mentions(store, &rows))
            .await
            .map_err(|_| "timed out embedding the mentions".to_string())??;
        let page = entity_page::build_page(&candidate.kind, &candidate.label, &mentions);
        if page.source_ids.is_empty() {
            tracing::debug!(entity = %label, "the entity has no mention, so it gets no page");
            return Ok(());
        }
        let sources: Vec<liam_store::Candidate> = rows
            .into_iter()
            .filter(|row| page.source_ids.contains(&row.id))
            .collect();
        let basis = mention_basis(store, &sources)
            .await
            .map_err(|e| e.to_string())?;
        let embedding = tokio::time::timeout_at(
            deadline,
            self.embedder.embed(&page.text, TextKind::Document),
        )
        .await
        .map_err(|_| "timed out embedding the page".to_string())?
        .map_err(|e| e.to_string())?;
        let mut node = NewNode::entity(candidate.kind, candidate.label)
            .with_attributes(entity_page::stamped(candidate.attributes))
            .with_embedding(embedding)
            .with_basis(basis)
            .with_producer(write_source.map_or_else(|| self.producer(), str::to_owned));
        if let Some(scope) = candidate.scope.clone() {
            node = node.with_scope(scope);
        }
        node.content = page.text;
        store
            .upsert_by(node)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    /// Each row as a page mention: with the vector the store holds for it, or
    /// one from the embedder when the store holds none (a demoted memory).
    async fn page_mentions(
        &self,
        store: &DefaultGraph,
        rows: &[liam_store::Candidate],
    ) -> Result<Vec<entity_page::Mention>, String> {
        let mut mentions = Vec::with_capacity(rows.len());
        for row in rows {
            let text = row.retrieved.as_text().to_string();
            let stored = store.embedding(&row.id).await.map_err(|e| e.to_string())?;
            let embedding = match stored {
                Some(vector) => vector,
                None => self
                    .embedder
                    .embed(&text, TextKind::Document)
                    .await
                    .map_err(|e| e.to_string())?,
            };
            mentions.push(entity_page::Mention {
                id: row.id.clone(),
                valid_from_ms: row.valid_from.0,
                text,
                embedding,
            });
        }
        Ok(mentions)
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
        let k = clamp_recall_k(args.k);
        let retrieve = async {
            let (embedding, embed_degraded) = self.embed_or_degrade("recall", &args.query).await;
            let q = build_query(
                &lexical_text(&args.query),
                Some(candidate_pool(k)),
                args.kind,
                args.scope,
                embedding,
                args.as_of,
            );
            // The memories come from a query that never ranks a page, so a
            // page cannot change which memories are returned or in what order.
            let recorded = self.explained(&q).await;
            let found = match recorded {
                Ok(recorded) => self
                    .explained_with_pages(&q)
                    .await
                    .map(|pages| (recorded, pages)),
                Err(e) => Err(e),
            };
            (found, embed_degraded)
        };
        let (shown, embed_degraded) = match until_interrupted(deadline, &cancel, retrieve).await {
            Ok((Ok((recorded, pages)), degraded)) => {
                (recall::with_pages(recorded, pages, k), degraded)
            }
            Ok((Err(e), _)) => return ToolReply::failed(format!("recall failed: {e}")),
            Err(why) => return self.interruption_reply("recall", why),
        };
        if shown.is_empty() {
            return with_reasons(ToolReply::ok("no relevant memory"), [embed_degraded]);
        }
        let shown = &shown[..];
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
                let attributes = recall::shown_attributes(hit);
                if let serde_json::Value::Object(map) = &attributes {
                    if !map.is_empty() {
                        trailing.push(format!("attributes: {attributes}"));
                    }
                }
                if hit.hit.retrieved.is_inferred() {
                    trailing.push(format!("kind: {PAGE_KIND}"));
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
        let pages = shown
            .iter()
            .filter_map(|hit| RecalledPage::of(hit.hit.id.handle(), &hit.hit.retrieved))
            .collect();
        with_reasons(ToolReply::ok(text).with_pages(pages), [embed_degraded])
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
        let budget = self.context_tokens / 10;
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
        description = "Say whether a result recall returned was correct, using its handle. \
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
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, rmcp::ErrorData> {
        let tool = request.name.to_string();
        let call = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        let response = self.tool_router.call(call).await?;
        Ok(argument_errors::clarify(&tool, response))
    }

    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

#[cfg(test)]
mod tests;
