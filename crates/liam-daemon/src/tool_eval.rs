// SPDX-License-Identifier: Apache-2.0
//! Grounding eval for `remember`/`recall`/`relate`: whether what goes IN comes
//! back OUT intact, findable under the label it was written with.
//!
//! Where `eval.rs` scores whether the LLM's synthesized ANSWER stays grounded
//! in its evidence, and `retrieval_eval.rs` scores whether `Graph::query`
//! surfaces the right memories by category, this module scores the
//! write/read tools themselves: does a fact survive `remember`, and does
//! `recall` return it under a query that shares vocabulary with its content.
//!
//! # Two tiers
//!
//! - **Mock tier, always-on:** `MockEmbedder`/`IdentityReranker`/`MockLlm`, no
//!   model download. A plain `#[tokio::test]`, runs under
//!   `cargo test -p liam-daemon --bin liamd tool_eval`.
//! - **Real-embedder, gated + `#[ignore]`d (added in a later WU):** the actual
//!   local embedder behind `#[cfg(feature = "local")]`, downloads weights on
//!   first run:
//!
//!   ```text
//!   cargo test --release -p liam-daemon --features local -- --ignored --nocapture tool_eval
//!   ```

use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;

use crate::mcp::{self, RememberArgs};

/// One seeded memory: kind, label, content.
type Fact = (&'static str, &'static str, &'static str);

/// Wall-clock deadline `MemoryServer::new` requires; this module never calls
/// `ask`, but the constructor still needs a value. Matches `eval.rs`'s own
/// constant.
const ASK_TIMEOUT_SECS: u64 = 300;

/// Opens a fresh in-memory store and wires it into a `MemoryServer` with the
/// given embedder/reranker and a `MockLlm` (this module never exercises
/// `ask`).
///
/// WHY this doesn't reuse `eval.rs`'s server construction: `eval.rs` builds
/// its server once, inline, inside a single `#[ignore]`d test behind a fixed
/// real LLM and reranker, because it only ever measures answer synthesis.
/// This module's tests each need a fresh store per test (`seed` mutates it)
/// and vary the embedder/reranker across tiers (mock now, a real embedder in
/// a later WU), so the construction has to be a reusable function rather than
/// one inline block; there being no third caller yet, that function stays
/// here instead of becoming a shared helper across the eval modules.
async fn build_server(
    embedder: Arc<dyn liam_model::Embedder>,
    reranker: Arc<dyn liam_model::Reranker>,
    dims: usize,
) -> mcp::MemoryServer {
    let store = Arc::new(
        liam_store::DefaultGraph::open(":memory:", liam_store::GraphConfig::new(dims))
            .await
            .expect("open in-memory store"),
    );
    mcp::MemoryServer::new(
        store,
        embedder,
        reranker,
        Arc::new(liam_model::MockLlm),
        ASK_TIMEOUT_SECS,
        crate::config::Config::default().ask_sufficiency_check,
        crate::config::Config::default().llm.context_tokens,
        crate::config::Config::default()
            .llm
            .max_concurrent_generations,
    )
}

/// Remembers every fact in `facts` into `server`.
///
/// WHY the return value is discarded: `remember`'s own return string is
/// already covered by `mcp.rs`'s handler tests; this module's tests assert on
/// what `recall` reports back afterward, not on what `remember` echoes at
/// write time, so asserting here too would just duplicate that coverage.
async fn seed(server: &mcp::MemoryServer, facts: &[Fact]) {
    for (kind, label, content) in facts {
        server
            .remember(Parameters(RememberArgs {
                kind: kind.to_string(),
                label: label.to_string(),
                content: content.to_string(),
                scope: None,
                subject: None,
                attributes: None,
                valid_from: None,
                confidence: None,
                episode: None,
            }))
            .await;
    }
}

/// Splits `recall_text` on `"\n\n"` blocks and extracts each block's label
/// from its first line, `"[{kind} {handle}] {label}"`.
///
/// WHY split on the literal `"] "` instead of a general bracket parser:
/// `recall`'s real output always places exactly one `"] "` between the
/// handle and the label on that first line, and the label itself may
/// legitimately contain further `[`/`]` characters (arbitrary user text), so
/// a parser that matched brackets in general would have to guess which pair
/// is the delimiter. `split_once` takes the first occurrence, which is
/// always the delimiter in this format, and needs none of that guessing.
fn labels_in_order(recall_text: &str) -> Vec<String> {
    recall_text
        .split("\n\n")
        .filter(|block| !block.is_empty())
        .filter_map(|block| {
            let first_line = block.lines().next()?;
            let (_, label) = first_line.split_once("] ")?;
            Some(label.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rmcp::handler::server::wrapper::Parameters;

    use super::*;
    use crate::mcp::RecallArgs;

    #[tokio::test]
    async fn seeded_fact_is_recalled_by_label() {
        // Arrange
        const DIMS: usize = 8;
        let embedder = Arc::new(liam_model::MockEmbedder::new(DIMS));
        let reranker = Arc::new(liam_model::IdentityReranker);
        let server = build_server(embedder, reranker, DIMS).await;
        let fact: Fact = (
            "fact",
            "Storage engine",
            "LIAM stores all memory in libSQL, a single-file SQLite fork.",
        );
        seed(&server, &[fact]).await;

        // Act
        let recall_text = server
            .recall(Parameters(RecallArgs {
                query: "libSQL SQLite storage".to_string(),
                kind: None,
                scope: None,
                k: None,
                as_of: None,
            }))
            .await;

        // Assert
        assert!(labels_in_order(&recall_text).contains(&"Storage engine".to_string()));
    }

    #[test]
    fn labels_in_order_empty_input_returns_empty_vec() {
        // Arrange
        let recall_text = "";

        // Act
        let labels = labels_in_order(recall_text);

        // Assert
        assert!(labels.is_empty());
    }

    #[test]
    fn labels_in_order_single_block_returns_one_label() {
        // Arrange
        let recall_text = "[fact abcd1234efgh] Storage engine\nLIAM stores memory in libSQL.";

        // Act
        let labels = labels_in_order(recall_text);

        // Assert
        assert_eq!(labels, vec!["Storage engine".to_string()]);
    }

    #[test]
    fn labels_in_order_duplicate_labels_both_appear_in_order() {
        // Arrange
        // Two blocks intentionally share the same label. `labels_in_order`
        // must not deduplicate: `reciprocal_rank`'s `HashSet` matching only
        // cares about first-occurrence position, so a genuine duplicate
        // label in a fixture would be a fixture-authoring bug, not a parser
        // bug.
        let recall_text = "[fact abcd1234efgh] Same Label\nFirst content.\n\n\
             [fact ijkl5678mnop] Same Label\nSecond content.";

        // Act
        let labels = labels_in_order(recall_text);

        // Assert
        assert_eq!(
            labels,
            vec!["Same Label".to_string(), "Same Label".to_string()]
        );
    }
}
