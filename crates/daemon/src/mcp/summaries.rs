// SPDX-License-Identifier: Apache-2.0
//! Summaries as the read side sees them: the ones `ask` adds to its evidence in
//! infer mode, and the `summaries` tool that lists them and walks one down to
//! what it was built from. Every summary is model-written, so every one is
//! rendered with its inferred label and the events it rests on.

use liam_model::TextKind;
use liam_store::{validate_scope, ChildKind, DefaultGraph, Millis, NodeId, Summary, SummaryEntry};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::warn;

use super::MemoryServer;
use crate::ask::Evidence;
use crate::render;
use crate::summary::as_retrieved;
use crate::summary::tree::cosine;

/// Rows returned when the caller gives no `limit`.
pub(super) const DEFAULT_SUMMARIES_LIMIT: usize = 20;

/// Most rows one call returns, whatever `limit` asks for.
pub(super) const MAX_SUMMARIES_LIMIT: usize = 100;

/// Most summaries `ask` adds to the evidence it composes an answer from.
const MAX_ASK_SUMMARIES: usize = 3;

const NO_SUMMARIES: &str = "no matching summaries";

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SummariesArgs {
    /// Ranks the summaries by closeness to this text and leaves out the ones
    /// that are not close. Absent lists them by level, then id.
    pub query: Option<String>,
    /// Only summaries of this level (1 is the lowest).
    pub level: Option<u8>,
    /// Narrows to this scope and its descendants. Absent matches every scope.
    pub scope: Option<String>,
    /// Shows this one summary, open or closed, with the memories or summaries
    /// it was built from and the events behind it. The other arguments are
    /// ignored.
    pub id: Option<String>,
    /// Rows to return. Absent defaults to 20; anything above 100 is capped.
    pub limit: Option<usize>,
}

/// The summaries whose vector is at least `min_similarity` close to `question`,
/// closest first and by id among equals. One without a vector, or with one of
/// another width than the question's, is left out.
fn rank(entries: Vec<SummaryEntry>, question: &[f32], min_similarity: f32) -> Vec<Summary> {
    let mut scored: Vec<(f32, Summary)> = entries
        .into_iter()
        .filter_map(|entry| {
            let vector = entry
                .vector
                .filter(|vector| vector.len() == question.len())?;
            let score = cosine(question, &vector);
            (score >= min_similarity).then_some((score, entry.summary))
        })
        .collect();
    scored.sort_by(|(a, left), (b, right)| b.total_cmp(a).then_with(|| left.id.cmp(&right.id)));
    scored.into_iter().map(|(_, summary)| summary).collect()
}

/// The evidence `ask` adds in infer mode: the live summaries of the question's
/// scope that are close enough to it, and, with `as_of`, known by then. A read
/// that fails leaves them out, so the answer is composed from what was recorded.
pub(super) async fn matching(
    server: &MemoryServer,
    question: &[f32],
    scope: &Option<String>,
    as_of: Option<i64>,
) -> Vec<Evidence> {
    let Ok(scope) = validate_scope(scope) else {
        return Vec::new();
    };
    let found = async {
        let store = server.store().await?;
        let entries = store
            .summaries_within(scope.as_deref(), None, as_of.map(Millis))
            .await?;
        let mut evidence = Vec::new();
        for summary in rank(entries, question, server.min_summary_similarity)
            .into_iter()
            .take(MAX_ASK_SUMMARIES)
        {
            let sources = store.summary_sources(&summary.id).await?;
            evidence.push(Evidence::from_summary(&summary, sources));
        }
        Ok::<_, liam_store::Error>(evidence)
    }
    .await;
    found.unwrap_or_else(|error| {
        warn!(%error, "ask: summaries could not be read and are left out");
        Vec::new()
    })
}

pub(super) async fn read(server: &MemoryServer, args: SummariesArgs) -> String {
    match run(server, args).await {
        Ok(text) => text,
        Err(reason) => format!("summaries failed: {reason}"),
    }
}

async fn run(server: &MemoryServer, args: SummariesArgs) -> Result<String, String> {
    let scope = validate_scope(&args.scope).map_err(|error| error.to_string())?;
    let cap = server.max_basis_ids_rendered;
    if let Some(id) = &args.id {
        let store = server.store().await.map_err(|error| error.to_string())?;
        return walk_down(&store, id, cap).await;
    }
    let limit = args
        .limit
        .unwrap_or(DEFAULT_SUMMARIES_LIMIT)
        .min(MAX_SUMMARIES_LIMIT);
    if limit == 0 {
        return Ok(NO_SUMMARIES.to_string());
    }
    let query = args
        .query
        .as_deref()
        .map(str::trim)
        .filter(|q| !q.is_empty());
    let question = match query {
        Some(text) => Some(
            server
                .embedder
                .embed(text, TextKind::Query)
                .await
                .map_err(|error| format!("the query could not be embedded: {error}"))?,
        ),
        None => None,
    };
    let store = server.store().await.map_err(|error| error.to_string())?;
    let entries = store
        .summaries_within(scope.as_deref(), args.level, None)
        .await
        .map_err(|error| error.to_string())?;
    let mut summaries = match question {
        Some(question) => rank(entries, &question, server.min_summary_similarity),
        None => entries.into_iter().map(|entry| entry.summary).collect(),
    };
    summaries.truncate(limit);
    if summaries.is_empty() {
        return Ok(NO_SUMMARIES.to_string());
    }
    let mut rendered = Vec::new();
    for summary in &summaries {
        rendered.push(render_summary(&store, summary, cap).await?);
    }
    Ok(rendered.join("\n\n"))
}

async fn render_summary(
    store: &DefaultGraph,
    summary: &Summary,
    cap: usize,
) -> Result<String, String> {
    let sources = store
        .summary_sources(&summary.id)
        .await
        .map_err(|error| error.to_string())?;
    let head = render::bracket_head(
        "summary",
        &summary.id,
        &format!("level {}", summary.level),
        Some(summary.tx_from.0),
    );
    let scope = (!summary.scope.is_empty()).then(|| format!("scope: {}", summary.scope));
    let trailing: Vec<String> = scope.into_iter().collect();
    Ok(render::render_item(
        &head,
        &as_retrieved(summary, sources),
        &trailing,
        cap,
    ))
}

/// One summary, then each child as it stood when the summary was written. A
/// child that no longer exists is named, not skipped.
async fn walk_down(store: &DefaultGraph, id: &str, cap: usize) -> Result<String, String> {
    let summary = store
        .summary(id)
        .await
        .map_err(|error| error.to_string())?
        .ok_or_else(|| format!("no summary {id}"))?;
    let mut parts = vec![render_summary(store, &summary, cap).await?];
    let children = store
        .summary_children(id)
        .await
        .map_err(|error| error.to_string())?;
    parts.push(format!("children ({}):", children.len()));
    for child in children {
        parts.push(render_child(store, &summary, &child.id, child.kind, cap).await?);
    }
    Ok(parts.join("\n\n"))
}

async fn render_child(
    store: &DefaultGraph,
    parent: &Summary,
    id: &str,
    kind: ChildKind,
    cap: usize,
) -> Result<String, String> {
    let missing = || format!("[missing child {id}]");
    match kind {
        ChildKind::Summary => match store.summary(id).await.map_err(|e| e.to_string())? {
            Some(child) => render_summary(store, &child, cap).await,
            None => Ok(missing()),
        },
        ChildKind::Node => {
            let node = store
                .get(&NodeId::from_raw(id), parent.tx_from)
                .await
                .map_err(|error| error.to_string())?;
            Ok(match node {
                Some(node) => {
                    let head = render::bracket_head(
                        &node.kind,
                        node.id.handle(),
                        &node.label,
                        Some(node.valid_from.0),
                    );
                    render::render_item(&head, &node.retrieved, &[], cap)
                }
                None => missing(),
            })
        }
    }
}

#[cfg(test)]
mod tests;
