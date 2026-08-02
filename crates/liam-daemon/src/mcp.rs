//! MCP tool surface: `remember` and `recall`, wiring the store and the model.
//!
//! VERSION CHECK: the rmcp macro surface (`#[tool_router]`, `#[tool]`,
//! `#[tool_handler]`, `Parameters`) moves across releases. Confirm against the
//! rmcp version you pin.

use std::sync::Arc;

use liam_model::{Embedder, Llm, Reranker};
use liam_store::{DefaultGraph, NewNode, Query};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;

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

#[derive(Clone)]
pub struct MemoryServer {
    store: Arc<DefaultGraph>,
    embedder: Arc<dyn Embedder>,
    reranker: Arc<dyn Reranker>,
    // Wired in M1; first consumed by synthesis/extraction in M2. The derived
    // `Clone` impl makes rustc's dead-code pass ignore field reads, so an
    // otherwise-unused field warns until M2 reads it.
    #[allow(dead_code)]
    llm: Arc<dyn Llm>,
    tool_router: rmcp::handler::server::tool::ToolRouter<Self>,
}

#[tool_router]
impl MemoryServer {
    pub fn new(
        store: Arc<DefaultGraph>,
        embedder: Arc<dyn Embedder>,
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn Llm>,
    ) -> Self {
        Self {
            store,
            embedder,
            reranker,
            llm,
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
        let mut q = Query::text(args.query.clone()).with_k(args.k.unwrap_or(8));
        if let Some(e) = embedding {
            q = q.with_embedding(e);
        }
        if let Some(kind) = args.kind {
            q = q.with_kind(kind);
        }
        if let Some(scope) = args.scope {
            q = q.with_scope(scope);
        }
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
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}
