//! MCP tool surface: `remember`, `recall`, and `ask`, wiring the store and the
//! model.
//!
//! VERSION CHECK: the rmcp macro surface (`#[tool_router]`, `#[tool]`,
//! `#[tool_handler]`, `Parameters`) moves across releases. Confirm against the
//! rmcp version you pin.

use std::sync::Arc;
use std::time::Duration;

use liam_model::{Embedder, Llm, Reranker};
use liam_store::{DefaultGraph, NewNode, Query};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::ask::{self, build_ask_prompt, clamp_ask_k, fallback_answer, format_answer};

/// Token budget for the sufficiency pre-pass. Only "YES"/"NO" is wanted, with a
/// little slack for models that add punctuation or a stray word; anything longer
/// is not a verdict and `ask::parse_sufficiency` rejects it anyway.
const SUFFICIENCY_MAX_TOKENS: usize = 8;

/// Shared embed→build-`Query` sequence for `recall` and `ask`. WHY: both
/// handlers apply the same k/embedding/kind/scope shape to a `Query`; keeping
/// it in one place means a future filter only needs to change here.
fn build_query(
    text: &str,
    k: Option<usize>,
    kind: Option<String>,
    scope: Option<String>,
    embedding: Option<Vec<f32>>,
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
    q
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RememberArgs {
    /// decision | fact | symbol | episode (opaque to the store).
    pub kind: String,
    pub label: String,
    pub content: String,
    /// Optional partition (project, agent).
    pub scope: Option<String>,
    /// Optional identity; a new value with the same subject supersedes the old.
    pub subject: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallArgs {
    pub query: String,
    pub kind: Option<String>,
    pub scope: Option<String>,
    pub k: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AskArgs {
    pub question: String,
    pub kind: Option<String>,
    pub scope: Option<String>,
    pub k: Option<usize>,
}

#[derive(Clone)]
pub struct MemoryServer {
    store: Arc<DefaultGraph>,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    llm: Arc<dyn Llm>,
    /// Wall-clock cap on `ask` synthesis before falling back to ranked
    /// evidence; see `config::Config::ask_timeout_secs`.
    ask_timeout_secs: u64,
    /// Whether `ask` runs the yes/no sufficiency pre-pass before synthesizing;
    /// see `config::Config::ask_sufficiency_check`.
    ask_sufficiency_check: bool,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}

#[tool_router]
impl MemoryServer {
    pub fn new(
        store: Arc<DefaultGraph>,
        embedder: Arc<dyn Embedder>,
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn Llm>,
        ask_timeout_secs: u64,
        ask_sufficiency_check: bool,
    ) -> Self {
        Self {
            store,
            embedder,
            reranker,
            llm,
            ask_timeout_secs,
            ask_sufficiency_check,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Record a durable decision or fact into long-term memory.")]
    async fn remember(&self, Parameters(args): Parameters<RememberArgs>) -> String {
        let embedding = match self.embedder.embed(&args.content).await {
            Ok(v) => v,
            Err(e) => return format!("embed failed: {e}"),
        };
        let mut node = NewNode::now(args.kind, args.label, args.content).with_embedding(embedding);
        if let Some(scope) = args.scope {
            node = node.with_scope(scope);
        }
        let write = match args.subject {
            Some(subject) => self.store.upsert_by(node.with_subject(subject)).await,
            None => self.store.insert(node).await,
        };
        match write {
            Ok(id) => format!("remembered {}", id.as_str()),
            Err(e) => format!("remember failed: {e}"),
        }
    }

    #[tool(description = "Retrieve relevant long-term memory, reranked for precision.")]
    async fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> String {
        let embedding = self.embedder.embed(&args.query).await.ok();
        let q = build_query(&args.query, args.k, args.kind, args.scope, embedding);
        let hits = match self.store.query(&q).await {
            Ok(h) => h,
            Err(e) => return format!("recall failed: {e}"),
        };
        if hits.is_empty() {
            return "no relevant memory".to_string();
        }
        let docs: Vec<String> = hits.iter().map(|h| h.content.clone()).collect();
        let order = self
            .reranker
            .order(&args.query, &docs)
            .await
            .unwrap_or_else(|_| (0..hits.len()).collect());
        order
            .iter()
            .map(|&i| format!("[{}] {}\n{}", hits[i].kind, hits[i].label, hits[i].content))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    // `pub(crate)` so the grounding eval (see `eval.rs`) drives the same code
    // path an MCP client hits, rather than a re-implementation of it.
    #[tool(description = "Answer a question from long-term memory, synthesized and cited.")]
    pub(crate) async fn ask(&self, Parameters(args): Parameters<AskArgs>) -> String {
        let embedding = self.embedder.embed(&args.question).await.ok();
        let k = clamp_ask_k(args.k);
        let q = build_query(&args.question, Some(k), args.kind, args.scope, embedding);
        let hits = match self.store.query_explained(&q).await {
            Ok(h) => h,
            Err(e) => return format!("ask failed: {e}"),
        };
        if hits.is_empty() {
            return "no relevant memory".to_string();
        }
        let docs: Vec<String> = hits.iter().map(|h| h.hit.content.clone()).collect();
        let order = self
            .reranker
            .order(&args.question, &docs)
            .await
            .unwrap_or_else(|_| (0..hits.len()).collect());
        let evidence: Vec<ask::Evidence> = order
            .iter()
            .map(|&i| ask::Evidence::from_hit(&hits[i]))
            .collect();
        // Sufficiency pre-pass: ask whether the evidence answers the question at
        // all, and refuse outright if it does not. See
        // `ask::build_sufficiency_prompt` for why this is a separate call.
        if self.ask_sufficiency_check {
            let (system, user) = ask::build_sufficiency_prompt(&args.question, &evidence);
            let verdict = tokio::time::timeout(
                Duration::from_secs(self.ask_timeout_secs.max(1)),
                // Capped hard: the verdict is one word, and an uncapped pre-pass
                // let a rambling model spend 50s per question (see eval.rs).
                self.llm
                    .complete_capped(&system, &user, SUFFICIENCY_MAX_TOKENS),
            )
            .await;
            // Only an explicit NO refuses. A timeout, an error, or an
            // unparseable reply falls through to synthesis: failing closed here
            // would turn any model hiccup into "I don't know" about memory the
            // store really holds.
            if let Ok(Ok(reply)) = verdict {
                if ask::parse_sufficiency(&reply) == Some(false) {
                    return ask::insufficient_answer(&evidence);
                }
            }
        }

        let (system, user) = build_ask_prompt(&args.question, &evidence);
        let synth = tokio::time::timeout(
            // max(1) guards against an operator typo of `ask_timeout_secs = 0`,
            // which would otherwise make every call time out immediately.
            Duration::from_secs(self.ask_timeout_secs.max(1)),
            self.llm.complete(&system, &user),
        )
        .await;
        match synth {
            Ok(Ok(a)) if !a.trim().is_empty() => {
                let answer = a.trim();
                // Last line of defence against prompt injection and free-running
                // fabrication: an answer that shares almost no vocabulary with the
                // evidence is not a synthesis of it, whatever the model intended.
                // Unlike the prompt rules, this does not depend on the model
                // cooperating. See `ask::is_grounded`.
                if ask::is_grounded(answer, &args.question, &evidence) {
                    format_answer(answer, &evidence)
                } else {
                    fallback_answer("the answer was not grounded in the evidence", &evidence)
                }
            }
            Ok(Ok(_)) => fallback_answer("the model returned an empty answer", &evidence),
            Ok(Err(_)) => fallback_answer("the model failed", &evidence),
            Err(_) => fallback_answer("synthesis timed out", &evidence),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_store::GraphConfig;

    /// Always-errors `Llm`, so `ask` must fall back to the retrieved evidence.
    struct FailingLlm;
    #[async_trait::async_trait]
    impl liam_model::Llm for FailingLlm {
        async fn complete(&self, _s: &str, _p: &str) -> liam_model::Result<String> {
            Err(liam_model::ModelError::Llm("boom".into()))
        }
    }

    /// Always-errors `Reranker`, so `ask` must fall back to identity order
    /// (via `Reranker::order`'s default `unwrap_or_else`) instead of panicking.
    struct FailingReranker;
    #[async_trait::async_trait]
    impl liam_model::Reranker for FailingReranker {
        async fn scores(&self, _q: &str, _d: &[String]) -> liam_model::Result<Vec<f32>> {
            Err(liam_model::ModelError::Rerank("boom".into()))
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

    /// Fresh in-memory server wired with the given reranker/llm and a 30s ask
    /// timeout. Dims 8 to match `MockEmbedder::new(8)`.
    async fn server_with(reranker: Arc<dyn Reranker>, llm: Arc<dyn Llm>) -> MemoryServer {
        server_with_timeout(reranker, llm, 30, false).await
    }

    /// Fresh in-memory server wired with the given reranker/llm/ask timeout.
    /// Dims 8 to match `MockEmbedder::new(8)`.
    async fn server_with_timeout(
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn Llm>,
        ask_timeout_secs: u64,
        ask_sufficiency_check: bool,
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
        )
    }

    async fn seed(server: &MemoryServer, kind: &str, label: &str, content: &str) {
        seed_scoped(server, kind, label, content, None).await;
    }

    async fn seed_scoped(
        server: &MemoryServer,
        kind: &str,
        label: &str,
        content: &str,
        scope: Option<&str>,
    ) {
        let out = server
            .remember(Parameters(RememberArgs {
                kind: kind.to_string(),
                label: label.to_string(),
                content: content.to_string(),
                scope: scope.map(str::to_string),
                subject: None,
            }))
            .await;
        assert!(out.starts_with("remembered "), "seed failed: {out}");
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
            answer.starts_with("The zorbnax gadget uses libSQL for storage [1]."),
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

    /// Server with the sufficiency pre-pass ON (the shipped default), which the
    /// other handler tests leave off so they exercise synthesis directly.
    async fn server_with_sufficiency(llm: Arc<dyn Llm>) -> MemoryServer {
        server_with_timeout(Arc::new(liam_model::IdentityReranker), llm, 30, true).await
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
            }))
            .await;

        // Assert
        assert!(
            answer.starts_with("The zorbnax gadget uses libSQL for storage [1]."),
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
            }))
            .await;

        // Assert: fell through to synthesis rather than refusing.
        assert!(
            !answer.contains("does not contain an answer"),
            "unparseable verdict was treated as a refusal: {answer}"
        );
        assert!(
            answer.starts_with("The zorbnax gadget uses libSQL for storage [1]."),
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
            }))
            .await;

        // Assert: only the requested kind comes back.
        assert!(out.contains("[decision] Alpha"), "missing match: {out}");
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
            }))
            .await;

        // Assert: exactly one block (blocks are joined by a blank line). WHY not
        // assert *which* one: with mock scoring the three are near-tied, and the
        // contract under test is the count.
        assert_eq!(out.split("\n\n").count(), 1, "expected 1 block: {out}");
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
            }))
            .await;

        // Assert on the `Sources:` index rather than the body: the echo llm
        // repeats the prompt, so the body would contain evidence markers either
        // way. One source line means one evidence item was cited.
        let sources = answer
            .split_once("Sources:\n")
            .map(|(_, s)| s.to_string())
            .unwrap_or_else(|| panic!("answer has no sources section: {answer}"));
        assert_eq!(sources.lines().count(), 1, "expected 1 source: {sources}");
        assert!(sources.starts_with("[1] fact/"), "sources: {sources}");
    }

    #[tokio::test]
    async fn recall_format_is_pinned() {
        // Regression pin: `recall`'s output shape must stay
        // `[{kind}] {label}\n{content}`, unchanged by the `ask` addition.
        let server = server_with(
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
        )
        .await;
        seed(&server, "decision", "Use libSQL", "single file db").await;

        let out = server
            .recall(Parameters(RecallArgs {
                query: "Use libSQL".to_string(),
                kind: None,
                scope: None,
                k: None,
            }))
            .await;

        assert_eq!(out, "[decision] Use libSQL\nsingle file db");
    }
}
