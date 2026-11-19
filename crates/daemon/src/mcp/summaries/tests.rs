// SPDX-License-Identifier: Apache-2.0
//! The `summaries` tool: which summaries are read, how they are labelled, and
//! how one is walked down.

use std::sync::Arc;

use liam_model::{Embedder, MockEmbedder};
use liam_store::{
    EventRef, GraphConfig, Millis, NewNode, NewSummary, PutOutcome, SummaryChild, SummaryEntry,
};
use rmcp::handler::server::wrapper::Parameters;

use super::*;
use crate::summary::generate::SUMMARY_PROMPT_VERSION;

const DIMS: usize = 8;
const QUESTION: &str = "quarterly forecasts for the finance team";
const SUMMARY_TEXT: &str = "quarterly forecasts for the finance team";
const EVENT: &str = "event-one";
/// Close enough that only a text equal to the question passes.
const STRICT_SIMILARITY: f32 = 0.999;

async fn server() -> MemoryServer {
    let store = DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
        .await
        .unwrap();
    MemoryServer::new(Arc::new(store), Arc::new(MockEmbedder::new(DIMS)), 30)
        .with_min_summary_similarity(STRICT_SIMILARITY)
}

/// A summary of `children` with `text`, embedded the way a pass would embed it.
struct Seed<'a> {
    scope: &'a str,
    level: u8,
    text: &'a str,
    children: Vec<(String, ChildKind)>,
    valid_max: Millis,
}

impl Default for Seed<'_> {
    fn default() -> Self {
        Self {
            scope: "",
            level: 1,
            text: SUMMARY_TEXT,
            children: vec![("child".to_string(), ChildKind::Node)],
            valid_max: Millis(1),
        }
    }
}

async fn seed(server: &MemoryServer, seed: Seed<'_>) -> String {
    let vector = MockEmbedder::new(DIMS)
        .embed(seed.text, liam_model::TextKind::Document)
        .await
        .unwrap();
    let children: Vec<SummaryChild> = seed
        .children
        .into_iter()
        .map(|(id, kind)| SummaryChild { id, kind })
        .collect();
    let summary = NewSummary {
        level: seed.level,
        scope: seed.scope.to_string(),
        text: seed.text.to_string(),
        events_fed: 1,
        valid_min: Millis(0),
        valid_max: seed.valid_max,
        model_id: "model".to_string(),
        prompt_version: SUMMARY_PROMPT_VERSION.to_string(),
    };
    let source = EventRef {
        event_id: EVENT.to_string(),
        synthetic: false,
    };
    let store = server.store().await.unwrap();
    match store
        .put_summary(summary, &children, &[source], &vector, &[], "test")
        .await
        .unwrap()
    {
        PutOutcome::Inserted(id) | PutOutcome::Unchanged(id) => id,
    }
}

fn tool_args() -> SummariesArgs {
    SummariesArgs {
        query: None,
        level: None,
        scope: None,
        id: None,
        limit: None,
    }
}

#[test]
fn ranking_skips_a_vector_of_another_width_and_breaks_ties_by_id() {
    // Arrange
    let entry = |id: &str, vector: Option<Vec<f32>>| SummaryEntry {
        summary: liam_store::Summary {
            id: id.to_string(),
            level: 1,
            scope: String::new(),
            child_hash: id.to_string(),
            text: liam_store::InferredText::for_test(id),
            derivation: liam_store::Derivation::Inferred,
            events_fed: 1,
            valid_min: Millis(0),
            valid_max: Millis(0),
            model_id: String::new(),
            prompt_version: String::new(),
            tx_from: Millis(0),
            tx_to: liam_store::FOREVER,
        },
        vector,
    };
    let entries = vec![
        entry("b", Some(vec![1.0, 0.0])),
        entry("a", Some(vec![1.0, 0.0])),
        entry("narrow", Some(vec![1.0])),
        entry("none", None),
        entry("far", Some(vec![0.0, 1.0])),
    ];

    // Act
    let ranked = rank(entries, &[1.0, 0.0], 0.5);

    // Assert
    let ids: Vec<&str> = ranked.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["a", "b"]);
}

#[tokio::test]
async fn the_tool_filters_by_level_and_says_none_for_an_unknown_one() {
    // Arrange
    let server = server().await;
    let low = seed(&server, Seed::default()).await;
    let high = seed(
        &server,
        Seed {
            level: 2,
            children: vec![(low.clone(), ChildKind::Summary)],
            ..Seed::default()
        },
    )
    .await;

    // Act
    let level_two = server
        .summaries(Parameters(SummariesArgs {
            level: Some(2),
            ..tool_args()
        }))
        .await;
    let unknown = server
        .summaries(Parameters(SummariesArgs {
            level: Some(9),
            ..tool_args()
        }))
        .await;

    // Assert
    assert!(level_two.contains(&high), "{level_two}");
    assert!(!level_two.contains(&low), "{level_two}");
    assert_eq!(unknown, NO_SUMMARIES);
}

#[tokio::test]
async fn the_tool_returns_nothing_for_a_limit_of_zero_and_caps_a_large_limit() {
    // Arrange
    let server = server().await;
    for n in 0..=MAX_SUMMARIES_LIMIT {
        seed(
            &server,
            Seed {
                children: vec![(format!("child-{n}"), ChildKind::Node)],
                ..Seed::default()
            },
        )
        .await;
    }

    // Act
    let none = server
        .summaries(Parameters(SummariesArgs {
            limit: Some(0),
            ..tool_args()
        }))
        .await;
    let capped = server
        .summaries(Parameters(SummariesArgs {
            limit: Some(MAX_SUMMARIES_LIMIT * 10),
            ..tool_args()
        }))
        .await;

    // Assert
    assert_eq!(none, NO_SUMMARIES);
    assert_eq!(capped.matches("[summary ").count(), MAX_SUMMARIES_LIMIT);
}

#[tokio::test]
async fn the_tool_lists_twenty_rows_by_level_then_id_when_no_limit_is_given() {
    // Arrange
    let server = server().await;
    let mut expected = Vec::new();
    for n in 0..=DEFAULT_SUMMARIES_LIMIT {
        let id = seed(
            &server,
            Seed {
                children: vec![(format!("child-{n}"), ChildKind::Node)],
                ..Seed::default()
            },
        )
        .await;
        expected.push(id);
    }
    expected.sort();
    expected.truncate(DEFAULT_SUMMARIES_LIMIT);

    // Act
    let listed = server.summaries(Parameters(tool_args())).await;

    // Assert
    let shown: Vec<&str> = listed
        .lines()
        .filter_map(|line| line.strip_prefix("[summary "))
        .filter_map(|rest| rest.split(']').next())
        .collect();
    assert_eq!(shown, expected);
}

#[tokio::test]
async fn the_tool_ranks_by_a_query_and_drops_what_is_not_close() {
    // Arrange
    let server = server().await;
    let close = seed(&server, Seed::default()).await;
    let far = seed(
        &server,
        Seed {
            text: "an unrelated note about holiday rotas",
            children: vec![("other".to_string(), ChildKind::Node)],
            ..Seed::default()
        },
    )
    .await;

    // Act
    let found = server
        .summaries(Parameters(SummariesArgs {
            query: Some(QUESTION.to_string()),
            ..tool_args()
        }))
        .await;

    // Assert
    assert!(found.contains(&close), "{found}");
    assert!(!found.contains(&far), "{found}");
}

#[tokio::test]
async fn one_summary_is_walked_down_to_its_children_derivations_and_closure() {
    // Arrange
    let server = server().await;
    let node = {
        let store = server.store().await.unwrap();
        store
            .insert(NewNode::now("fact", "Forecast", "forecasts are quarterly"))
            .await
            .unwrap()
    };
    let id = seed(
        &server,
        Seed {
            children: vec![
                (node.as_str().to_string(), ChildKind::Node),
                ("gone-node".to_string(), ChildKind::Node),
            ],
            ..Seed::default()
        },
    )
    .await;

    // Act
    let shown = server
        .summaries(Parameters(SummariesArgs {
            id: Some(id.clone()),
            ..tool_args()
        }))
        .await;

    // Assert
    assert!(
        shown.contains(&format!(
            "provenance: derivation: inferred; basis: {EVENT}; summary: 1"
        )),
        "{shown}"
    );
    assert!(shown.contains("children (2):"), "{shown}");
    assert!(shown.contains("forecasts are quarterly"), "{shown}");
    assert!(shown.contains("derivation: recorded"), "{shown}");
    assert!(shown.contains("[missing child gone-node]"), "{shown}");
}

#[tokio::test]
async fn an_unknown_summary_id_is_an_error_naming_it() {
    // Arrange
    let server = server().await;

    // Act
    let shown = server
        .summaries(Parameters(SummariesArgs {
            id: Some("nope".to_string()),
            ..tool_args()
        }))
        .await;

    // Assert
    assert_eq!(shown, "summaries failed: no summary nope");
}

#[tokio::test]
async fn the_tool_is_routed_and_says_it_is_inferred() {
    // Arrange
    let server = server().await;

    // Act
    let tools = server.tool_router.list_all();

    // Assert
    assert!(server.tool_router.has_route("summaries"));
    let tool = tools.iter().find(|t| t.name == "summaries").unwrap();
    let description = tool.description.as_deref().unwrap_or_default();
    assert!(description.contains("inferred"), "{description}");
}
