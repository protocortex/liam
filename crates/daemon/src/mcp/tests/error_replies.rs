// SPDX-License-Identifier: Apache-2.0
//! The one-line replies tools give when an input or a dependency fails.

use std::sync::atomic::{AtomicUsize, Ordering};

use liam_model::{Embedder, MockEmbedder, TextKind};

use super::super::feedback::FeedbackArgs;
use super::super::test_doubles::ErroringEmbedder;
use super::*;

const DIMS: usize = 8;

/// Embeds the first `working` calls, then fails every one after.
struct FailsAfter {
    working: usize,
    calls: AtomicUsize,
    inner: MockEmbedder,
}

#[async_trait::async_trait]
impl Embedder for FailsAfter {
    fn dims(&self) -> usize {
        DIMS
    }

    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        if self.calls.fetch_add(1, Ordering::SeqCst) < self.working {
            self.inner.embed(text, kind).await
        } else {
            Err(liam_model::ModelError::Embed("model unavailable".into()))
        }
    }
}

async fn server_over(embedder: Arc<dyn Embedder>) -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
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

async fn stored_matches(server: &MemoryServer, text: &str) -> usize {
    server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(text))
        .await
        .unwrap()
        .len()
}

#[tokio::test]
async fn remember_reports_an_embedding_failure_and_stores_nothing() {
    // Arrange
    let embedder = Arc::new(ErroringEmbedder::new(DIMS));
    let server = server_over(embedder.clone()).await;

    // Act
    let out = server
        .remember(Parameters(RememberArgs {
            episode: None,
            ..remember_args("unembeddedmarker")
        }))
        .await;

    // Assert
    assert!(out.starts_with("embed failed: "), "{out}");
    assert!(out.contains("boom"), "{out}");
    assert_eq!(embedder.call_count(), 1);
    assert_eq!(stored_matches(&server, "unembeddedmarker").await, 0);
}

#[tokio::test]
async fn an_episode_reports_a_failed_embedding_of_its_top_level_fact() {
    // Arrange
    let server = server_over(Arc::new(ErroringEmbedder::new(DIMS))).await;

    // Act
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("nestedmarker")],
                entities: vec![],
                edges: vec![],
            }),
            ..remember_args("topmarker")
        }))
        .await;

    // Assert
    assert!(out.starts_with("embed failed: "), "{out}");
    assert_eq!(stored_matches(&server, "topmarker").await, 0);
}

#[tokio::test]
async fn an_episode_reports_a_failed_embedding_of_a_nested_fact_and_writes_none_of_it() {
    // Arrange
    let server = server_over(Arc::new(FailsAfter {
        working: 1,
        calls: AtomicUsize::new(0),
        inner: MockEmbedder::new(DIMS),
    }))
    .await;

    // Act
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("nestedmarker")],
                entities: vec![],
                edges: vec![],
            }),
            ..remember_args("topmarker")
        }))
        .await;

    // Assert
    assert_eq!(out, "embed failed: embedder: model unavailable");
    assert_eq!(stored_matches(&server, "topmarker").await, 0);
    assert_eq!(stored_matches(&server, "nestedmarker").await, 0);
}

#[tokio::test]
async fn relate_refuses_a_blank_type_and_writes_no_edge() {
    // Arrange
    let server = plain_server().await;
    let a = seed(&server, "decision", "Alpha", "first").await;
    let b = seed(&server, "decision", "Beta", "second").await;

    for blank in ["", "   "] {
        // Act
        let out = server
            .relate(Parameters(RelateArgs {
                from: a.clone(),
                to: b.clone(),
                kind: blank.to_string(),
            }))
            .await;

        // Assert
        assert_eq!(out, "relate failed: type must not be empty", "{blank:?}");
    }
}

#[tokio::test]
async fn the_feedback_tool_says_when_tiering_is_off() {
    // Arrange
    let server = plain_server().await;
    let id = seed(&server, "fact", "Alpha", "first").await;

    // Act
    let out = server
        .feedback(Parameters(FeedbackArgs {
            result_id: id,
            correct: true,
        }))
        .await;

    // Assert
    assert_eq!(
        out,
        "feedback failed: tiering is not enabled, so verdicts are not kept"
    );
}
