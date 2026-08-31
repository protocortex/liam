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
//! - **Real-tier, gated + `#[ignore]`d:** the actual local embedder
//!   (`FastEmbedEmbedder`, Qwen3) AND the actual local reranker
//!   (`FastEmbedReranker`, a BGE cross-encoder) behind `#[cfg(feature =
//!   "local")]`, downloading weights on first run:
//!
//!   ```text
//!   cargo test --release -p liam-daemon --bin liamd --features local -- --ignored --nocapture reranking
//!   ```
//!
//!   Env override: `LIAM_TOOL_EVAL_MODEL`, mirroring `eval.rs`'s
//!   `LIAM_EVAL_MODEL` and `retrieval_eval.rs`'s `LIAM_RETRIEVAL_EVAL_MODEL`
//!   for a local A/B run against a different embedding model.
//!
//! # Real-tier baseline (2026-08-31, Apple M1 Pro, macOS 15.7.5,
//! # Qwen/Qwen3-Embedding-0.6B, 768 dims, default BGE reranker via fastembed,
//! # `cargo test --release -p liam-daemon --bin liamd --features local --
//! # --ignored --nocapture reranking`)
//!
//! `reranking_promotes_the_correct_target_over_a_surface_decoy`: under
//! `IdentityReranker` (real embedding + lexical RRF, no rerank), recall
//! ordered `["Gizmo ship date", "Nightjar ship date"]` for the query "When
//! does the zorbnax gizmo's successor ship?" (decoy reciprocal rank 1.0,
//! target reciprocal rank 0.5). Under `FastEmbedReranker`, the same query
//! against the same fixture reordered to `["Nightjar ship date", "Gizmo ship
//! date"]` (target reciprocal rank 1.0). See `mod real_tier` below for why
//! this fixture reproduces the flaw.
//!
//! `semantic_paraphrase_recovers_the_remembered_fact` (2026-08-31, same
//! machine/model as above,
//! `cargo test --release -p liam-daemon --bin liamd --features local --
//! --ignored --nocapture semantic_paraphrase`): recalling under
//! `IdentityReranker` with the paraphrase query "Will they fix it for free
//! if liquid gets inside accidentally?" against the sole seeded fact
//! "Nightjar's warranty only covers manufacturing defects, never water
//! damage." (zero shared non-stopword tokens between query and fact
//! content) returned that fact, label "Nightjar warranty terms", as the
//! only and top hit (reciprocal rank 1.0), confirming the real embedder
//! alone, with no lexical overlap to lean on, recovers the fact.

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

/// Real-tier: gated behind `feature = "local"` and `#[ignore]`d, since it
/// downloads model weights on first run and takes real time (see the module
/// doc's "Two tiers" section for the exact run command). A sibling of `mod
/// tests`, not nested inside it, specifically so it can `use super::*;` and
/// reuse `build_server`/`seed`/`labels_in_order` the same way `mod tests`
/// does, mirroring `retrieval_eval.rs`'s own `mod real_embedder_run`.
#[cfg(feature = "local")]
mod real_tier {
    use std::collections::HashSet;

    use super::*;
    use crate::mcp::RecallArgs;

    /// Read an env override, falling back to `default` when unset. Same
    /// small shape as `eval.rs`'s and `retrieval_eval.rs`'s own `env_or`;
    /// redefined locally rather than shared across modules, per those
    /// modules' own doc comments, since it is a two-line function with no
    /// state to keep in sync.
    fn env_or(key: &str, default: &str) -> String {
        std::env::var(key).unwrap_or_else(|_| default.to_string())
    }

    /// Label of the fixture's correct answer to `QUERY`.
    const TARGET_LABEL: &str = "Nightjar ship date";
    /// Label of the fixture's surface-similar wrong answer to `QUERY`.
    const DECOY_LABEL: &str = "Gizmo ship date";

    /// Query asking specifically about the zorbnax gizmo's SUCCESSOR, not
    /// the gizmo itself.
    const QUERY: &str = "When does the zorbnax gizmo's successor ship?";

    /// The two-fact fixture that reproduces the ranking flaw. WHY the decoy
    /// beats the target under `IdentityReranker` (real embedding + lexical
    /// RRF, no rerank) despite answering the wrong question: `DECOY`'s
    /// wording ("The zorbnax gizmo ships in June 2026") repeats `QUERY`'s
    /// most literal phrase, "zorbnax gizmo ... ship", verbatim, but never
    /// mentions a successor at all: it's the ORIGINAL product's ship date,
    /// not the one `QUERY` actually asks for. `TARGET` answers the actual
    /// question (the successor's ship date, under its own name "Nightjar")
    /// and shares the word "successor" with `QUERY`, but paraphrases the
    /// verb ("launches" instead of "ship") and never repeats `QUERY`'s
    /// "zorbnax gizmo ships" phrase, so it has less raw surface/n-gram
    /// overlap with `QUERY` than `DECOY` does, despite being the fact
    /// `QUERY` is actually about. A cross-encoder reranker scores `QUERY`
    /// and each document JOINTLY (not via separately-computed vectors), so
    /// it can tell "this document is about the ORIGINAL gizmo, not its
    /// successor" in a way that surface-overlap-driven ranking cannot; see
    /// this module's top-of-file doc comment for the real ranks this fixture
    /// produced.
    const FIXTURE: [Fact; 2] = [
        ("fact", DECOY_LABEL, "The zorbnax gizmo ships in June 2026."),
        (
            "fact",
            TARGET_LABEL,
            "Nightjar, the zorbnax gizmo's successor, launches in November 2026.",
        ),
    ];

    /// Given a target fact and a surface-similar decoy fact seeded
    /// identically into two real-embedding stores, when the same query is
    /// recalled under `IdentityReranker`, then the decoy ranks first and the
    /// target is outranked; when the same query is recalled under
    /// `FastEmbedReranker`, then the target is promoted to rank first.
    #[tokio::test]
    #[ignore = "downloads reranker weights; see module doc for the run command"]
    async fn reranking_promotes_the_correct_target_over_a_surface_decoy() {
        // Arrange: one real embedder, loaded once, shared by both servers,
        // since both need to embed the SAME facts/query identically for the
        // comparison to isolate the reranker as the only variable that
        // differs between them.
        let model_id = env_or(
            "LIAM_TOOL_EVAL_MODEL",
            &crate::config::Config::default().embedder.model,
        );
        let dims = crate::config::Config::default().embedding_dims;
        let embedder = Arc::new(liam_model::FastEmbedEmbedder::load(&model_id, dims).expect(
            "load real embedder (Qwen3); requires network access for first-time model \
                 download",
        ));

        let baseline_server = build_server(
            embedder.clone(),
            Arc::new(liam_model::IdentityReranker),
            dims,
        )
        .await;
        seed(&baseline_server, &FIXTURE).await;

        let real_reranker = liam_model::FastEmbedReranker::load().expect(
            "load reranker (BGE cross-encoder); requires network access for first-time model \
             download",
        );
        let real_server = build_server(embedder, Arc::new(real_reranker), dims).await;
        seed(&real_server, &FIXTURE).await;

        // Act
        let baseline_recall_text = baseline_server
            .recall(Parameters(RecallArgs {
                query: QUERY.to_string(),
                kind: None,
                scope: None,
                k: Some(FIXTURE.len()),
                as_of: None,
            }))
            .await;
        let real_recall_text = real_server
            .recall(Parameters(RecallArgs {
                query: QUERY.to_string(),
                kind: None,
                scope: None,
                k: Some(FIXTURE.len()),
                as_of: None,
            }))
            .await;
        println!("baseline (IdentityReranker) order: {baseline_recall_text}");
        println!("real (FastEmbedReranker) order: {real_recall_text}");

        let baseline_labels = labels_in_order(&baseline_recall_text);
        let real_labels = labels_in_order(&real_recall_text);
        println!("baseline labels: {baseline_labels:?}");
        println!("real labels: {real_labels:?}");

        // Assert
        let decoy = HashSet::from([DECOY_LABEL]);
        let target = HashSet::from([TARGET_LABEL]);
        let baseline_decoy_rr = crate::retrieval_eval::reciprocal_rank(&baseline_labels, &decoy);
        let baseline_target_rr = crate::retrieval_eval::reciprocal_rank(&baseline_labels, &target);
        let real_target_rr = crate::retrieval_eval::reciprocal_rank(&real_labels, &target);
        println!(
            "reciprocal rank: baseline decoy={baseline_decoy_rr}, baseline \
             target={baseline_target_rr}, real target={real_target_rr}"
        );

        assert_eq!(
            baseline_decoy_rr, 1.0,
            "expected the decoy to rank first under IdentityReranker, proving it is a real, \
             retrieved competitor rather than the target simply being missing: {baseline_labels:?}"
        );
        assert!(
            baseline_target_rr < 1.0,
            "expected the target to be outranked (not missing) under IdentityReranker: \
             {baseline_labels:?}"
        );
        assert_eq!(
            real_target_rr, 1.0,
            "expected the target to be promoted to rank first under FastEmbedReranker: \
             {real_labels:?}"
        );
    }

    /// Label of the sole fact seeded for
    /// `semantic_paraphrase_recovers_the_remembered_fact`.
    const PARAPHRASE_FACT_LABEL: &str = "Nightjar warranty terms";

    /// The fact, phrased around coverage/defects/water damage.
    const PARAPHRASE_FACT_CONTENT: &str =
        "Nightjar's warranty only covers manufacturing defects, never water damage.";

    /// A paraphrase of the question `PARAPHRASE_FACT_CONTENT` answers,
    /// sharing MINIMAL lexical overlap with it. Verified by eye before
    /// writing this test: `PARAPHRASE_FACT_CONTENT`'s non-stopword/content
    /// words are {nightjar, warranty, covers, manufacturing, defects,
    /// water, damage}; `PARAPHRASE_QUERY`'s are {fix, free, liquid, gets,
    /// inside, accidentally}. Zero words in common, so plain lexical/FTS
    /// matching has nothing to latch onto here and only a real semantic
    /// embedding (mapping "water damage" near "liquid ... inside" and
    /// "covers" near "fix ... for free") can recover the fact for this
    /// query.
    const PARAPHRASE_QUERY: &str = "Will they fix it for free if liquid gets inside accidentally?";

    /// The single-fact fixture for
    /// `semantic_paraphrase_recovers_the_remembered_fact`.
    const PARAPHRASE_FACT: Fact = ("fact", PARAPHRASE_FACT_LABEL, PARAPHRASE_FACT_CONTENT);

    /// Given a fact remembered with wording A, when recalled with a
    /// paraphrase B sharing minimal lexical overlap with A, then the
    /// fact's reciprocal rank in the recall output is exactly 1.0. Uses
    /// `IdentityReranker`, not `FastEmbedReranker`: the sibling test above
    /// already covers reranking quality, so this test isolates the
    /// embedder as the only variable being measured, per this module's
    /// doc comment. Loads its OWN `Arc<FastEmbedEmbedder>` rather than
    /// reusing the sibling test's instance: each `#[tokio::test]` function
    /// gets a fresh instance, no cross-test-function shared state.
    #[tokio::test]
    #[ignore = "downloads embedder weights; see module doc for the run command"]
    async fn semantic_paraphrase_recovers_the_remembered_fact() {
        // Arrange
        let model_id = env_or(
            "LIAM_TOOL_EVAL_MODEL",
            &crate::config::Config::default().embedder.model,
        );
        let dims = crate::config::Config::default().embedding_dims;
        let embedder = Arc::new(liam_model::FastEmbedEmbedder::load(&model_id, dims).expect(
            "load real embedder (Qwen3); requires network access for first-time model \
             download",
        ));
        let server = build_server(embedder, Arc::new(liam_model::IdentityReranker), dims).await;
        seed(&server, &[PARAPHRASE_FACT]).await;

        // Act
        let recall_text = server
            .recall(Parameters(RecallArgs {
                query: PARAPHRASE_QUERY.to_string(),
                kind: None,
                scope: None,
                k: Some(1),
                as_of: None,
            }))
            .await;
        println!("recall order: {recall_text}");
        let labels = labels_in_order(&recall_text);
        println!("labels: {labels:?}");

        // Assert
        let rr = crate::retrieval_eval::reciprocal_rank(
            &labels,
            &HashSet::from([PARAPHRASE_FACT_LABEL]),
        );
        println!("reciprocal rank: {rr}");
        assert_eq!(
            rr, 1.0,
            "expected the real embedder to recover the fact for a lexically-disjoint paraphrase \
             query: {labels:?}"
        );
    }
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
