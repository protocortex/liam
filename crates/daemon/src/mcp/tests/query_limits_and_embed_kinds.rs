// SPDX-License-Identifier: Apache-2.0
//! The limits on a query, and which text is embedded as a query or a document.

use super::*;

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

fn hundred_words() -> String {
    (0..100)
        .map(|i| format!("w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::test]
async fn recall_rejects_invalid_queries_with_exact_replies() {
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

        // Assert
        assert_eq!(recall, format!("recall failed: {problem}"));
    }
}

#[tokio::test]
async fn recall_accepts_queries_at_the_character_limit() {
    // Arrange
    let server = server_with_recording_embedder().await.0;
    let cases = ["x".repeat(4000), "\u{e9}".repeat(4000)];

    for text in cases {
        assert_eq!(text.chars().count(), 4000);

        // Act
        let recall = server.recall(Parameters(recall_args_for(&text))).await;

        // Assert
        assert!(!recall.starts_with("recall failed"), "recall: {recall}");
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
async fn recall_embeds_the_full_query_while_lexical_text_is_capped() {
    // Arrange
    let question = hundred_words();
    let (server, embedder) = server_with_recording_embedder().await;

    // Act
    server.recall(Parameters(recall_args_for(&question))).await;

    // Assert
    let calls = embedder.calls();
    assert_eq!(calls.len(), 1);
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
