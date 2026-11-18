// SPDX-License-Identifier: Apache-2.0
//! `recall` and `ask` filters, the pool they read, and how a hit is rendered.

use super::*;

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
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: only the requested kind comes back. The handle between the
    // kind and the label varies per run, so match around it.
    assert!(out.contains("[decision "), "missing match: {out}");
    assert!(out.contains("] Alpha"), "missing match: {out}");
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
            as_of: None,
            mode: None,
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
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: exactly one block (blocks are joined by a blank line). WHY not
    // assert *which* one: with mock scoring the three are near-tied, and the
    // contract under test is the count.
    assert_eq!(out.split("\n\n").count(), 1, "expected 1 block: {out}");
}

const WIDE_POOL_K: usize = 16;
const STORE_DIMS: usize = 8;
const LARGE_STORE_SIZE: usize = 40;
const LARGE_K: usize = 1000;
const EXPECTED_MAX_K: usize = 32;

async fn open_store() -> Arc<DefaultGraph> {
    Arc::new(
        DefaultGraph::open(":memory:", GraphConfig::new(STORE_DIMS))
            .await
            .expect("open in-memory store"),
    )
}

fn server_over(store: &Arc<DefaultGraph>) -> MemoryServer {
    MemoryServer::new(
        store.clone(),
        Arc::new(liam_model::MockEmbedder::new(STORE_DIMS)),
        Arc::new(liam_model::MockLlm),
        30,
        false,
        8192,
        1,
    )
}

async fn insert_fact(store: &DefaultGraph, label: &str, content: &str) -> NodeId {
    store
        .insert(liam_store::NewNode::now("fact", label, content))
        .await
        .expect("insert fact")
}

/// The query `recall` builds for `text` at `k`, with no filters.
async fn recall_query(text: &str, k: usize) -> Query {
    let embedding = liam_model::MockEmbedder::new(STORE_DIMS)
        .embed(text, TextKind::Query)
        .await
        .ok();
    build_query(text, Some(k), None, None, embedding, None, false)
}

fn recall_k(query: &str, k: usize) -> RecallArgs {
    RecallArgs {
        query: query.to_string(),
        kind: None,
        scope: None,
        k: Some(k),
        as_of: None,
        mode: None,
    }
}

fn hit_count(out: &str) -> usize {
    out.matches("[fact ").count()
}

#[tokio::test]
async fn recall_clamps_k_into_one_to_thirty_two() {
    // Arrange
    let store = open_store().await;
    for index in 0..LARGE_STORE_SIZE {
        insert_fact(&store, &format!("note {index}"), "the zorbnax gadget").await;
    }
    let server = server_over(&store);

    // Act
    let zero = server.recall(Parameters(recall_k("zorbnax", 0))).await;
    let one = server.recall(Parameters(recall_k("zorbnax", 1))).await;
    let large = server
        .recall(Parameters(recall_k("zorbnax", LARGE_K)))
        .await;

    // Assert
    assert_eq!(hit_count(&zero), 1, "{zero}");
    assert_eq!(hit_count(&one), 1, "{one}");
    assert_eq!(
        hit_count(&large),
        EXPECTED_MAX_K.min(LARGE_STORE_SIZE),
        "{large}"
    );
}

#[tokio::test]
async fn a_narrow_query_is_a_subset_of_a_wide_one_for_equal_facts() {
    // Arrange: thirty facts of equal confidence
    let store = open_store().await;
    const FACT_COUNT: usize = 30;
    const NARROW_K: usize = 3;
    for index in 0..FACT_COUNT {
        insert_fact(&store, &format!("note {index}"), "the zorbnax gadget").await;
    }
    let narrow_k = NARROW_K;

    // Act
    let narrow = store
        .query_explained(&recall_query("zorbnax", narrow_k).await)
        .await
        .unwrap();
    let wide = store
        .query_explained(&recall_query("zorbnax", WIDE_POOL_K).await)
        .await
        .unwrap();

    // Assert
    assert_eq!(narrow.iter().filter(|h| !h.expanded).count(), narrow_k);
    for hit in narrow.iter().filter(|h| !h.expanded) {
        assert!(wide.iter().any(|w| w.hit.id == hit.hit.id));
    }
}

#[tokio::test]
async fn recall_returns_the_first_k_of_the_wide_pool_in_store_order() {
    // Arrange
    let store = open_store().await;
    for index in 0..LARGE_STORE_SIZE {
        insert_fact(&store, &format!("note {index}"), "the zorbnax gadget").await;
    }
    let k = 3;
    let server = server_over(&store);
    let pool = store
        .query_explained(&recall_query("zorbnax", crate::ask::candidate_pool(k)).await)
        .await
        .unwrap();
    let expected: Vec<String> = pool
        .iter()
        .take(k)
        .map(|h| h.hit.id.handle().to_string())
        .collect();

    // Act
    let out = server.recall(Parameters(recall_k("zorbnax", k))).await;

    // Assert
    assert_eq!(hit_count(&out), k, "{out}");
    for handle in expected {
        assert!(out.contains(&handle), "{handle} missing from {out}");
    }
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
            as_of: None,
            mode: None,
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
    let server = server_with(Arc::new(RecordingLlm::new(
        "The zorbnax report says one [1].",
    )))
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
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert on the `Sources:` index rather than the body: the echo llm
    // repeats the prompt, so the body would contain evidence markers either
    // way. One source line means one evidence item was cited.
    let sources = answer
        .split_once("Sources:\n")
        .map(|(_, s)| s.to_string())
        .unwrap_or_else(|| panic!("answer has no sources section: {answer}"));
    let numbered = sources.lines().filter(|l| l.starts_with('[')).count();
    assert_eq!(numbered, 1, "expected 1 source: {sources}");
    assert!(sources.starts_with("[1] fact/"), "sources: {sources}");
}

#[tokio::test]
async fn recall_format_is_pinned() {
    // Regression pin: `recall`'s output shape is
    // `[{kind} {handle}] {label}\n{content}\nprovenance: ...`. The handle was added by
    // ADR-0001 so an agent can name what it recalled and pass it to
    // `relate`; before that, `recall` dropped `Hit::id` entirely.
    let server = server_with(Arc::new(liam_model::MockLlm)).await;
    let id = seed(&server, "decision", "Use libSQL", "single file db").await;

    let out = server
        .recall(Parameters(RecallArgs {
            query: "Use libSQL".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    let handle = &id[..HANDLE_LEN];
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines[..2],
        [
            format!("[decision {handle}] Use libSQL").as_str(),
            "single file db"
        ]
    );
    assert_eq!(lines.len(), 3, "{out}");
    assert!(
        lines[2].starts_with("provenance: derivation: recorded; events: "),
        "{out}"
    );
}

#[tokio::test]
async fn recall_renders_a_handle_relate_can_resolve() {
    // The two tools have to agree: whatever `recall` prints must be
    // something `relate` accepts back verbatim. Pinning the length on both
    // sides separately would let them drift, so this round-trips it.
    let server = plain_server().await;
    seed(
        &server,
        "decision",
        "Alpha",
        "the zorbnax rollout is approved",
    )
    .await;
    seed(&server, "fact", "Beta", "the zorbnax rollout costs money").await;

    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax rollout".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    let handles: Vec<&str> = out
        .lines()
        .filter_map(|l| l.strip_prefix('['))
        .filter_map(|l| l.split_once(']'))
        .filter_map(|(head, _)| head.split_once(' '))
        .map(|(_, handle)| handle)
        .collect();
    assert_eq!(handles.len(), 2, "expected a handle per hit: {out}");
    for handle in &handles {
        assert_eq!(handle.len(), HANDLE_LEN, "{handle}");
    }

    let related = server
        .relate(Parameters(RelateArgs {
            from: handles[0].to_string(),
            to: handles[1].to_string(),
            kind: "mentions".to_string(),
        }))
        .await;
    assert!(related.starts_with("related "), "{related}");
}

#[tokio::test]
async fn recall_shows_confidence_when_not_default() {
    // Given a hit remembered with a non-default confidence
    let server = plain_server().await;
    server
        .remember(Parameters(RememberArgs {
            confidence: Some(0.6),
            ..remember_args("confidence render content")
        }))
        .await;

    // When recall retrieves it
    let out = server
        .recall(Parameters(RecallArgs {
            query: "confidence render content".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Then the confidence line renders with two decimal places
    assert!(out.contains("confidence: 0.60"), "{out}");
}

#[tokio::test]
async fn recall_shows_confidence_of_exactly_zero() {
    // Given a hit remembered with confidence exactly 0.0, the low
    // boundary, not just any non-default value
    let server = plain_server().await;
    server
        .remember(Parameters(RememberArgs {
            confidence: Some(0.0),
            ..remember_args("zero confidence render content")
        }))
        .await;

    let out = server
        .recall(Parameters(RecallArgs {
            query: "zero confidence render content".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    assert!(out.contains("confidence: 0.00"), "{out}");
}

#[tokio::test]
async fn recall_shows_attributes_when_non_empty() {
    // Given a hit remembered with a non-empty attributes object
    let server = plain_server().await;
    server
        .remember(Parameters(RememberArgs {
            attributes: Some(json!({"hue": "blue"})),
            ..remember_args("attributes render content")
        }))
        .await;

    // When recall retrieves it
    let out = server
        .recall(Parameters(RecallArgs {
            query: "attributes render content".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Then the attributes line renders as compact JSON
    assert!(out.contains(r#"attributes: {"hue":"blue"}"#), "{out}");
}

#[tokio::test]
async fn recall_omits_attributes_line_when_empty() {
    // Given a hit remembered without attributes (the default, an empty
    // JSON object)
    let server = plain_server().await;
    server
        .remember(Parameters(remember_args("no attributes render content")))
        .await;

    let out = server
        .recall(Parameters(RecallArgs {
            query: "no attributes render content".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    assert!(!out.contains("attributes:"), "{out}");
}

#[tokio::test]
async fn recall_shows_confidence_then_attributes_when_both_present() {
    // Given a hit remembered with both non-default confidence and
    // non-empty attributes
    let server = plain_server().await;
    server
        .remember(Parameters(RememberArgs {
            confidence: Some(0.5),
            attributes: Some(json!({"hue": "blue"})),
            ..remember_args("both confidence and attributes content")
        }))
        .await;

    let out = server
        .recall(Parameters(RecallArgs {
            query: "both confidence and attributes content".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Then both lines render, confidence before attributes
    let confidence_at = out
        .find("confidence: 0.50")
        .expect("confidence line missing");
    let attributes_at = out
        .find(r#"attributes: {"hue":"blue"}"#)
        .expect("attributes line missing");
    assert!(confidence_at < attributes_at, "{out}");
}

#[tokio::test]
async fn recall_bracket_stays_kind_and_handle_only_with_confidence_and_attributes_present() {
    // Two hits, one carrying non-default confidence and attributes, the
    // other left at defaults: the bracket must stay exactly
    // `[{kind} {handle}]` for both, no matter which fields the hit
    // carries. This is the exact defect ADR-0004 records as a rejected
    // alternative, putting confidence inside the bracket broke handle
    // resolution against `resolve_handle`'s alphanumeric-only gate.
    let server = plain_server().await;
    server
        .remember(Parameters(RememberArgs {
            kind: "decision".to_string(),
            label: "Alpha".to_string(),
            content: "the zorbnax rollout is approved".to_string(),
            scope: None,
            subject: None,
            attributes: Some(json!({"hue": "blue"})),
            valid_from: None,
            confidence: Some(0.4),
            episode: None,
        }))
        .await;
    seed(&server, "fact", "Beta", "the zorbnax rollout costs money").await;

    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax rollout".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    let handles: Vec<&str> = out
        .lines()
        .filter_map(|l| l.strip_prefix('['))
        .filter_map(|l| l.split_once(']'))
        .filter_map(|(head, _)| head.split_once(' '))
        .map(|(_, handle)| handle)
        .collect();
    assert_eq!(handles.len(), 2, "expected a handle per hit: {out}");
    for handle in &handles {
        assert_eq!(handle.len(), HANDLE_LEN, "{handle}");
    }

    let related = server
        .relate(Parameters(RelateArgs {
            from: handles[0].to_string(),
            to: handles[1].to_string(),
            kind: "mentions".to_string(),
        }))
        .await;
    assert!(related.starts_with("related "), "{related}");
}
