// SPDX-License-Identifier: Apache-2.0
//! `remember`: the producer it stamps, and the limits on what it accepts.

use super::*;

#[tokio::test]
async fn remember_stamps_the_servers_producer_on_the_written_node() {
    use liam_store::{Backend, DefaultBackend};

    // Given a MemoryServer carrying a producer
    let db_path = temp_db_path();
    let db_path_str = db_path.to_str().expect("temp path is valid utf-8");
    let store = DefaultGraph::open(db_path_str, GraphConfig::new(8))
        .await
        .expect("open file-backed store");
    let server = MemoryServer::new(
        Arc::new(store),
        Arc::new(liam_model::MockEmbedder::new(8)),
        30,
    );
    server.set_producer("agent-a");

    // When remember writes a node
    let out = server
        .remember(Parameters(RememberArgs {
            kind: "fact".to_string(),
            label: "label".to_string(),
            content: "content".to_string(),
            scope: None,
            subject: None,
            attributes: None,
            valid_from: None,
            confidence: None,
            episode: None,
        }))
        .await;
    assert!(out.starts_with("remembered "), "remember failed: {out}");

    // Then the stored row records that producer. Read it back through a
    // fresh connection to the same file, since `producer` is
    // deliberately absent from `Hit` and cannot be asserted through
    // `recall`.
    let raw = DefaultBackend::open(db_path_str, 1)
        .await
        .expect("open a second connection to the same file");
    let rows = raw
        .query("SELECT producer FROM nodes", &[])
        .await
        .expect("query nodes");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_string(0).unwrap(), "agent-a");

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(format!("{db_path_str}-wal"));
    let _ = std::fs::remove_file(format!("{db_path_str}-shm"));
}

#[tokio::test]
async fn remember_collapses_entity_pages_across_producers_via_episode() {
    use liam_store::{Backend, DefaultBackend};

    // Given two MemoryServer clones over one store, each stamped with a
    // different producer before either calls remember
    let db_path = temp_db_path();
    let db_path_str = db_path.to_str().expect("temp path is valid utf-8");
    let store = DefaultGraph::open(db_path_str, GraphConfig::new(8))
        .await
        .expect("open file-backed store");
    let template = MemoryServer::new(
        Arc::new(store),
        Arc::new(liam_model::MockEmbedder::new(8)),
        30,
    );
    let server_a = template.clone();
    server_a.set_producer("agent-a");
    let server_b = template.clone();
    server_b.set_producer("agent-b");

    // When each calls remember with an episode whose entities list
    // names "Alice" (same type, same scope)
    let out_a = server_a
        .remember(Parameters(RememberArgs {
            scope: Some("s".to_string()),
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Alice")],
                edges: vec![],
            }),
            ..remember_args("two producer episode entity content a")
        }))
        .await;
    assert!(!out_a.contains("failed"), "{out_a}");

    let out_b = server_b
        .remember(Parameters(RememberArgs {
            scope: Some("s".to_string()),
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Alice")],
                edges: vec![],
            }),
            ..remember_args("two producer episode entity content b")
        }))
        .await;
    assert!(!out_b.contains("failed"), "{out_b}");

    // Then exactly one live "Alice" entity page exists for that subject
    // and scope, read back through a fresh connection to the same file
    let raw = DefaultBackend::open(db_path_str, 1)
        .await
        .expect("open a second connection to the same file");
    let rows = raw
        .query(
            "SELECT COUNT(*) FROM nodes WHERE subject = ?1 AND scope = ?2 AND tx_to = ?3",
            &["alice".into(), "s".into(), liam_store::FOREVER.into()],
        )
        .await
        .expect("query nodes");
    assert_eq!(
        rows[0].get_i64(0).unwrap(),
        1,
        "exactly one live Alice page must remain, ignoring producer"
    );

    // And both producers' writes landed as separate rows, with agent-b's
    // the live one, proving the entity loop stamped each write with its
    // own producer rather than collapsing both under one
    let producer_rows = raw
        .query(
            "SELECT producer, tx_to FROM nodes WHERE subject = ?1 AND scope = ?2",
            &["alice".into(), "s".into()],
        )
        .await
        .expect("query nodes by producer");
    assert_eq!(
        producer_rows.len(),
        2,
        "both agent-a's and agent-b's Alice pages must exist as separate rows"
    );
    let live_row = producer_rows
        .iter()
        .find(|r| r.get_i64(1).unwrap() == liam_store::FOREVER.0)
        .expect("one row must be live");
    assert_eq!(
        live_row.get_string(0).unwrap(),
        "agent-b",
        "the live Alice page must belong to agent-b"
    );

    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(format!("{db_path_str}-wal"));
    let _ = std::fs::remove_file(format!("{db_path_str}-shm"));
}

/// A one-key JSON object whose `.to_string()` is exactly `total`
/// characters, for pinning `MAX_ATTRIBUTES_CHARS`'s boundary exactly
/// rather than guessing at serde_json's compact formatting.
fn attributes_of_length(total: usize) -> serde_json::Value {
    let overhead = json!({"k": ""}).to_string().chars().count();
    let padding = "a".repeat(total - overhead);
    json!({ "k": padding })
}

#[tokio::test]
async fn remember_with_attributes_round_trips_through_query_explained() {
    // Given a server and an attributes object to remember
    let server = plain_server().await;

    // When remember writes the node
    let out = server
        .remember(Parameters(RememberArgs {
            attributes: Some(json!({"k": "v"})),
            ..remember_args("attributes round trip content")
        }))
        .await;
    assert!(out.starts_with("remembered "), "{out}");

    // Then the store holds exactly the attributes sent
    let hits = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("attributes round trip content"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].hit.attributes, json!({"k": "v"}));
}

#[tokio::test]
async fn remember_with_valid_from_sets_the_stored_instant_not_now() {
    // Given a server and an explicit backdated valid_from
    let server = plain_server().await;

    // When remember writes the node
    let out = server
        .remember(Parameters(RememberArgs {
            valid_from: Some(1000),
            ..remember_args("valid from backdated content")
        }))
        .await;
    assert!(out.starts_with("remembered "), "{out}");

    // Then the store holds that instant, not the store's clock
    let hits = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("valid from backdated content"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].valid_from, Millis(1000));
}

#[tokio::test]
async fn remember_with_confidence_round_trips_through_query_explained() {
    // Given a server and an explicit confidence
    let server = plain_server().await;

    // When remember writes the node
    let out = server
        .remember(Parameters(RememberArgs {
            confidence: Some(0.75),
            ..remember_args("confidence round trip content")
        }))
        .await;
    assert!(out.starts_with("remembered "), "{out}");

    // Then the store holds exactly that confidence
    let hits = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("confidence round trip content"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].confidence, 0.75);
}

#[tokio::test]
async fn remember_rejects_out_of_range_confidence_without_writing() {
    // Given a server, and several confidence values outside 0.0..=1.0
    let server = plain_server().await;
    let marker = "confidence boundary marker content";

    for confidence in [-0.1, 1.1, 5.0] {
        let before = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();

        // When remember is called with that confidence
        let out = server
            .remember(Parameters(RememberArgs {
                confidence: Some(confidence),
                ..remember_args(marker)
            }))
            .await;

        // Then it is refused with the exact error text, and no node lands
        assert_eq!(
            out, "remember failed: confidence must be between 0.0 and 1.0",
            "confidence {confidence}: {out}"
        );
        let after = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap()
            .len();
        assert_eq!(
            after, before,
            "confidence {confidence} wrote a node despite rejection"
        );
    }
}

#[tokio::test]
async fn remember_accepts_confidence_at_its_inclusive_bounds() {
    // Given a server, and confidence at each inclusive bound
    let server = plain_server().await;

    // Distinct marker words per bound, not a shared phrase: `fts5_query`
    // ORs quoted terms together, so two contents sharing any word would
    // each match the other's query too.
    for (confidence, marker) in [(0.0, "confidencefloor"), (1.0, "confidenceceiling")] {
        // When remember is called with that confidence
        let out = server
            .remember(Parameters(RememberArgs {
                confidence: Some(confidence),
                ..remember_args(marker)
            }))
            .await;

        // Then it is accepted and round-trips exactly
        assert!(
            out.starts_with("remembered "),
            "confidence {confidence}: {out}"
        );
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].confidence, confidence);
    }
}

#[tokio::test]
async fn remember_rejects_non_object_attributes() {
    // Given a server, and several non-object attributes shapes
    let server = plain_server().await;

    for attributes in [json!(["a"]), json!("x"), json!(1), json!(true)] {
        // When remember is called with that shape
        let out = server
            .remember(Parameters(RememberArgs {
                attributes: Some(attributes.clone()),
                ..remember_args("non object attributes content")
            }))
            .await;

        // Then it is refused with the exact error text
        assert_eq!(
            out, "remember failed: attributes must be a JSON object",
            "attributes {attributes:?}: {out}"
        );
    }
}

#[tokio::test]
async fn remember_attributes_boundary_at_the_max_char_cap() {
    // Given attributes serialized to exactly MAX_ATTRIBUTES_CHARS
    let server = plain_server().await;
    let at_cap = attributes_of_length(MAX_ATTRIBUTES_CHARS);
    assert_eq!(at_cap.to_string().chars().count(), MAX_ATTRIBUTES_CHARS);

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            attributes: Some(at_cap),
            ..remember_args("attributes at cap content")
        }))
        .await;

    // Then it is accepted
    assert!(out.starts_with("remembered "), "{out}");

    // Given attributes one character past the cap
    let over_cap = attributes_of_length(MAX_ATTRIBUTES_CHARS + 1);
    assert_eq!(
        over_cap.to_string().chars().count(),
        MAX_ATTRIBUTES_CHARS + 1
    );
    let marker = "attributes over cap content";
    let before = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            attributes: Some(over_cap),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused with the exact error text, and no node lands
    assert_eq!(
        out,
        format!("remember failed: attributes exceeds {MAX_ATTRIBUTES_CHARS} characters")
    );
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(
        after, before,
        "over-cap attributes wrote a node despite rejection"
    );
}

#[tokio::test]
async fn remember_content_boundary_at_the_max_char_cap() {
    // Given content built to exactly MAX_CONTENT_CHARS
    let server = plain_server().await;
    let at_cap = "x".repeat(MAX_CONTENT_CHARS);
    assert_eq!(at_cap.chars().count(), MAX_CONTENT_CHARS);

    // When remember is called with it
    let out = server.remember(Parameters(remember_args(&at_cap))).await;

    // Then it is accepted
    assert!(out.starts_with("remembered "), "{out}");

    // Given content one character past the cap, built around a unique
    // marker so the before/after query below can detect a written node
    let marker = "content over cap marker";
    let over_cap = format!(
        "{marker} {}",
        "x".repeat(MAX_CONTENT_CHARS - marker.chars().count())
    );
    assert_eq!(over_cap.chars().count(), MAX_CONTENT_CHARS + 1);
    let before = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();

    // When remember is called with it
    let out = server.remember(Parameters(remember_args(&over_cap))).await;

    // Then it is refused with the exact error text, and no node lands
    assert_eq!(
        out,
        format!("remember failed: content exceeds {MAX_CONTENT_CHARS} characters")
    );
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(
        after, before,
        "over-cap content wrote a node despite rejection"
    );
}

#[tokio::test]
async fn remember_content_boundary_counts_unicode_scalars_not_bytes() {
    // Given content built from the precomposed 'é' scalar (U+00E9, 2
    // UTF-8 bytes) repeated to exactly MAX_CONTENT_CHARS characters: a
    // regression that swapped `.chars().count()` for `.len()` (byte
    // count) in `content_problem` would double-count this and reject it
    // at the cap, something ASCII-only fixtures can't catch since ASCII
    // chars and bytes are 1:1.
    let server = plain_server().await;
    let at_cap: String = "é".repeat(MAX_CONTENT_CHARS);
    assert_eq!(at_cap.chars().count(), MAX_CONTENT_CHARS);

    // When remember is called with it
    let out = server.remember(Parameters(remember_args(&at_cap))).await;

    // Then it is accepted
    assert!(out.starts_with("remembered "), "{out}");

    // Given content one character past the cap, built the same way
    let over_cap: String = "é".repeat(MAX_CONTENT_CHARS + 1);
    assert_eq!(over_cap.chars().count(), MAX_CONTENT_CHARS + 1);

    // When remember is called with it
    let out = server.remember(Parameters(remember_args(&over_cap))).await;

    // Then it is refused with the exact error text
    assert_eq!(
        out,
        format!("remember failed: content exceeds {MAX_CONTENT_CHARS} characters")
    );
}

#[tokio::test]
async fn remember_without_the_new_args_keeps_prior_defaults() {
    // Given the pre-WU1 call shape: no attributes/valid_from/confidence
    let server = plain_server().await;
    let before = Millis::now();

    // When remember writes the node
    let out = server
        .remember(Parameters(remember_args("regression pin content")))
        .await;
    assert!(out.starts_with("remembered "), "{out}");
    let after = Millis::now();

    // Then attributes/confidence keep their old defaults, and valid_from
    // resolves to "now" within the call's own wall-clock window
    let hits = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("regression pin content"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].hit.attributes, json!({}));
    assert_eq!(hits[0].confidence, 1.0);
    assert!(
        hits[0].valid_from.0 >= before.0 && hits[0].valid_from.0 <= after.0,
        "valid_from {:?} not within [{before:?}, {after:?}]",
        hits[0].valid_from
    );
}

#[tokio::test]
async fn remember_without_episode_is_unchanged() {
    // Given a server, and a remember call with `episode` explicitly
    // absent, the shape every pre-WU4 caller sends
    let server = plain_server().await;

    // When remember is called
    let out = server
        .remember(Parameters(remember_args(
            "episode absent regression content",
        )))
        .await;

    // Then the response is the same "remembered {id}" shape as before,
    // and exactly one node lands: no episode-shaped side effect at all
    assert!(out.starts_with("remembered "), "{out}");
    assert!(!out.contains('\n'), "unexpected extra line: {out}");
    let hits = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("episode absent regression content"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[tokio::test]
async fn a_second_producer_stamp_is_refused_and_the_first_stays() {
    // Arrange
    let server = plain_server().await;
    assert_eq!(server.producer(), DEFAULT_PRODUCER);
    server.set_producer("agent-a");

    // Act
    server.set_producer("agent-b");

    // Assert
    assert_eq!(server.producer(), "agent-a");
}
