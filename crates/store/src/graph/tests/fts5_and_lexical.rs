// SPDX-License-Identifier: Apache-2.0
//! Query text turned into an FTS5 match, and lexical search over it.

use super::*;

#[test]
fn fts5_query_collapses_a_repeated_term() {
    // Arrange
    let text = "word ".repeat(2000);

    // Act
    let query = fts5_query(&text);

    // Assert
    assert_eq!(query, "\"word\"");
}

#[test]
fn fts5_query_dedupes_case_insensitively() {
    // Arrange
    let text = "Word word WORD";

    // Act
    let query = fts5_query(text);

    // Assert
    assert_eq!(query, "\"word\"");
}

#[test]
fn fts5_query_still_doubles_quotes_after_dedupe() {
    // Arrange
    let text = "a \"b\" a";

    // Act
    let query = fts5_query(text);

    // Assert
    assert_eq!(query, "\"a\" OR \"\"\"b\"\"\"");
}

#[tokio::test]
async fn lexical_query_tolerates_question_punctuation() {
    // FTS5 parses a raw MATCH value as its own query language; a natural
    // question's trailing `?` or an embedded apostrophe previously raised
    // "fts5: syntax error" and aborted the whole hybrid query. The terms
    // must still match. Regression pin for a natural-language
    // question as the query. Quoting is what protects this, so it must hold regardless of
    // how the quoted terms are joined.
    // Arrange
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now(
        "fact",
        "Storage",
        "the gadget uses libsql for storage",
    ))
    .await
    .unwrap();

    // Act: a question-shaped query whose terms appear in the content.
    let hits = g.query(&Query::text("storage gadget?")).await.unwrap();

    // Assert
    assert!(hits.iter().any(|h| h.label == "Storage"));

    // An apostrophe also opens FTS5 syntax; a question containing one must
    // not raise a syntax error either.
    assert!(g
        .query(&Query::text("what's the gadget's storage?"))
        .await
        .is_ok());

    // A punctuation-only query hits the "no searchable term" guard: it must
    // return Ok (empty), never an FTS5 syntax error.
    assert!(g.query(&Query::text("???")).await.unwrap().is_empty());
}

#[tokio::test]
async fn lexical_question_finds_the_statement_that_answers_it() {
    // Regression pin for the whole bug: a question and the statement it
    // answers share only content words. The question adds interrogatives
    // ("when", "does") that a stored statement never contains, so joining
    // the quoted terms with implicit AND required every word to appear and
    // always returned zero hits. This fails on the pre-fix separator.
    // Arrange
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now(
        "fact",
        "Gizmo ship date",
        "The zorbnax gizmo ships in June 2026.",
    ))
    .await
    .unwrap();

    // Act
    let hits = g
        .query(&Query::text("When does the zorbnax gizmo ship?"))
        .await
        .unwrap();

    // Assert
    assert!(hits.iter().any(|h| h.label == "Gizmo ship date"));
}

#[tokio::test]
async fn lexical_rare_term_still_ranks_first_under_or() {
    // ORing the quoted terms recovers recall, but recall alone would also
    // pass if precision at the top were destroyed. This proves BM25's
    // inverse-document-frequency weighting still does its job: "zorbnax"
    // appears in only one of three otherwise-similar nodes, so it must
    // dominate the common words ("the", "ships", "June", "2026") every node
    // shares.
    // Arrange
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now(
        "fact",
        "Rare",
        "the zorbnax gizmo ships in June 2026",
    ))
    .await
    .unwrap();
    g.insert(NewNode::now(
        "fact",
        "Common A",
        "the gadget ships in June 2026",
    ))
    .await
    .unwrap();
    g.insert(NewNode::now(
        "fact",
        "Common B",
        "the widget ships in June 2026",
    ))
    .await
    .unwrap();

    // Act
    let hits = g
        .query(&Query::text("when does the zorbnax gizmo ship"))
        .await
        .unwrap();

    // Assert
    assert_eq!(hits[0].label, "Rare");
}

#[tokio::test]
async fn lexical_scope_filter_includes_descendant_scopes() {
    // Given nodes scoped "work" and "work/liam" that share the same
    // searchable terms, so a lexical hit is the only thing that can
    // distinguish an ancestor-scope query from a scope miss.
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("decision", "Root", "zorbnax rollout approved").with_scope("work"))
        .await
        .unwrap();
    g.insert(NewNode::now("decision", "Child", "zorbnax rollout approved").with_scope("work/liam"))
        .await
        .unwrap();

    // When querying with scope "work"
    let hits = g
        .query(&Query::text("zorbnax rollout").with_scope("work"))
        .await
        .unwrap();

    // Then both the "work"-scoped and "work/liam"-scoped nodes match.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        labels.contains(&"Root"),
        "expected Root in hits: {labels:?}"
    );
    assert!(
        labels.contains(&"Child"),
        "expected Child in hits: {labels:?}"
    );
}

#[tokio::test]
async fn lexical_scope_filter_excludes_ancestor_scope() {
    // Given the same "work" / "work/liam" nodes as the descendant-
    // inclusion case, but querying the more specific child scope.
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("decision", "Root", "zorbnax rollout approved").with_scope("work"))
        .await
        .unwrap();
    g.insert(NewNode::now("decision", "Child", "zorbnax rollout approved").with_scope("work/liam"))
        .await
        .unwrap();

    // When querying with scope "work/liam"
    let hits = g
        .query(&Query::text("zorbnax rollout").with_scope("work/liam"))
        .await
        .unwrap();

    // Then only the "work/liam"-scoped node matches; the ancestor does
    // not.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert_eq!(labels, vec!["Child"]);
}

#[tokio::test]
async fn lexical_scope_filter_matches_true_descendants_not_string_prefixes() {
    // Given nodes at "work" (exact match), "work-archive" (shares the
    // string prefix but has no "/" boundary), "work/a_b" (a true child
    // whose segment contains a literal underscore), and "wo" (shorter
    // than the query scope, so no "/" boundary is even possible), all
    // sharing the same searchable content.
    let g = graph_at(Millis(1000)).await;
    let exact_id = g
        .insert(NewNode::now("decision", "Exact", "zorbnax rollout approved").with_scope("work"))
        .await
        .unwrap();
    g.insert(
        NewNode::now("decision", "FalsePrefix", "zorbnax rollout approved")
            .with_scope("work-archive"),
    )
    .await
    .unwrap();
    let underscore_id = g
        .insert(
            NewNode::now("decision", "Underscore", "zorbnax rollout approved")
                .with_scope("work/a_b"),
        )
        .await
        .unwrap();
    g.insert(NewNode::now("decision", "Short", "zorbnax rollout approved").with_scope("wo"))
        .await
        .unwrap();

    // When calling `lexical` directly, bypassing `fetch_candidates`'s
    // own scope re-check, so a broken `lexical` predicate cannot hide
    // behind it.
    let lexical_ids = g
        .lexical(
            "zorbnax rollout",
            Filter {
                kind: None,
                scope: Some("work"),
                include_inferred: true,
            },
            Millis(1000),
            10,
        )
        .await
        .unwrap();
    let lexical_id_set: HashSet<NodeId> = lexical_ids.into_iter().collect();
    let expected_id_set: HashSet<NodeId> = [exact_id, underscore_id].into_iter().collect();
    assert_eq!(lexical_id_set, expected_id_set);

    // When querying with scope "work" through the full pipeline
    let hits = g
        .query(&Query::text("zorbnax rollout").with_scope("work"))
        .await
        .unwrap();

    // Then only the exact match and the true descendant match.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        labels.contains(&"Exact"),
        "expected Exact in hits: {labels:?}"
    );
    assert!(
        labels.contains(&"Underscore"),
        "expected Underscore in hits: {labels:?}"
    );
    assert!(
        !labels.contains(&"FalsePrefix"),
        "work-archive must not match scope work: {labels:?}"
    );
    assert!(
        !labels.contains(&"Short"),
        "wo must not match scope work: {labels:?}"
    );
}

#[tokio::test]
async fn lexical_scope_filter_underscore_in_query_scope_is_literal() {
    // Given nodes at "a_b/c" (a true descendant of query scope "a_b")
    // and "axb/c" (a different scope a LIKE pattern would conflate,
    // since '_' is a wildcard only in the LIKE pattern, not the value;
    // the earlier underscore test only put the underscore in the
    // stored scope, which a naive LIKE implementation gets right by
    // accident).
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("decision", "Real", "zorbnax rollout approved").with_scope("a_b/c"))
        .await
        .unwrap();
    g.insert(NewNode::now("decision", "Fake", "zorbnax rollout approved").with_scope("axb/c"))
        .await
        .unwrap();

    // When querying with scope "a_b"
    let hits = g
        .query(&Query::text("zorbnax rollout").with_scope("a_b"))
        .await
        .unwrap();

    // Then only the true descendant matches.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(
        labels.contains(&"Real"),
        "expected Real in hits: {labels:?}"
    );
    assert!(
        !labels.contains(&"Fake"),
        "axb/c must not match query scope a_b: {labels:?}"
    );
}

#[tokio::test]
async fn query_kind_and_scope_filters_combine() {
    // Given a decision and a fact, both scoped "work/liam", sharing the
    // same searchable content, exercising the placeholder-index path
    // where kind and scope filters are both active together.
    let g = graph_at(Millis(1000)).await;
    g.insert(
        NewNode::now("decision", "Decided", "zorbnax rollout approved").with_scope("work/liam"),
    )
    .await
    .unwrap();
    g.insert(NewNode::now("fact", "Stated", "zorbnax rollout approved").with_scope("work/liam"))
        .await
        .unwrap();

    // When querying with kind "decision" and scope "work"
    let hits = g
        .query(
            &Query::text("zorbnax rollout")
                .with_kind("decision")
                .with_scope("work"),
        )
        .await
        .unwrap();

    // Then only the decision comes back.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert_eq!(labels, vec!["Decided"]);
}

#[tokio::test]
async fn lexical_no_scope_filter_matches_every_scope() {
    // Given nodes across unrelated scopes
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("decision", "A", "zorbnax rollout approved").with_scope("work"))
        .await
        .unwrap();
    g.insert(NewNode::now("decision", "B", "zorbnax rollout approved").with_scope("personal"))
        .await
        .unwrap();

    // When querying with no scope filter
    let hits = g.query(&Query::text("zorbnax rollout")).await.unwrap();

    // Then both nodes match regardless of scope, unchanged from
    // today's behaviour.
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(labels.contains(&"A"), "expected A in hits: {labels:?}");
    assert!(labels.contains(&"B"), "expected B in hits: {labels:?}");
}
