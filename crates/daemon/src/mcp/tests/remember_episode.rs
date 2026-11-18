// SPDX-License-Identifier: Apache-2.0
//! `remember` with an episode of facts and edges.

use super::*;

#[tokio::test]
async fn remember_with_episode_facts_and_an_edge_between_them_writes_atomically() {
    // Given a top-level fact, one episode.facts entry, and an edge
    // between them by combined index
    let server = plain_server().await;

    // When remember is called with episode set
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("episodenestedfactmarker")],
                entities: vec![],
                edges: vec![episode_edge("fact:0", "fact:1", "relates_to")],
            }),
            ..remember_args("episodetoplevelfactmarker")
        }))
        .await;

    // Then it succeeds, both facts exist, and the edge exists
    assert!(!out.contains("failed"), "{out}");
    let mut lines = out.lines();
    let top_id = lines
        .next()
        .expect("top-level fact line")
        .trim_start_matches("remembered ")
        .to_string();
    let nested_id = lines
        .next()
        .expect("nested fact line")
        .trim_start_matches("remembered ")
        .to_string();

    // Counted by content match, not by result length: the edge just
    // created between them makes each reachable as a graph-expanded
    // neighbour of the other's text hit too (`ExplainedHit::expanded`),
    // so a bare length assertion would be pinning that expansion
    // behaviour instead of "both facts exist".
    let top = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("episodetoplevelfactmarker"))
        .await
        .unwrap();
    assert!(
        top.iter()
            .any(|h| h.hit.retrieved.as_text() == "episodetoplevelfactmarker"),
        "{out}"
    );
    let nested = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("episodenestedfactmarker"))
        .await
        .unwrap();
    assert!(
        nested
            .iter()
            .any(|h| h.hit.retrieved.as_text() == "episodenestedfactmarker"),
        "{out}"
    );

    // The edge exists: relating the same ordered triple again reports
    // "already relates", the existing idempotency signal `relate` uses.
    let relate_again = server
        .relate(Parameters(RelateArgs {
            from: top_id,
            to: nested_id,
            kind: "relates_to".to_string(),
        }))
        .await;
    assert!(
        relate_again.contains("already relates"),
        "edge missing: {relate_again}"
    );
}

#[tokio::test]
async fn remember_with_episode_rejects_supersedes_edge_kind() {
    // Given an episode whose one edge asserts the reserved `supersedes`
    // type
    let server = plain_server().await;
    let marker = "episode supersedes rejection content";
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
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("episode supersedes nested content")],
                entities: vec![],
                edges: vec![episode_edge("fact:0", "fact:1", "supersedes")],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused, and nothing lands, not even the top-level fact
    assert!(out.contains("reserved"), "{out}");
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");
}

#[tokio::test]
async fn remember_with_episode_rejects_a_mentions_edge_with_a_fresh_entity_as_to() {
    // Given an episode whose one edge is a `mentions` edge with a fresh
    // entity as `to` (backwards: `to` should be a fact) and a
    // handle-shaped `from`, so only the `to`-is-entity check fires,
    // independent of the `from`-is-fact check
    let server = plain_server().await;
    let existing_handle = seed(
        &server,
        "fact",
        "Existing handle target",
        "backwards mentions to-entity handle content",
    )
    .await;
    let marker = "backwards mentions to entity content";
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
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Backwards To Entity")],
                edges: vec![episode_edge(&existing_handle, "entity:0", "mentions")],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused, naming the edge index and the backwards
    // direction, and nothing lands, not even the top-level fact
    assert!(out.contains("edge 0"), "{out}");
    assert!(out.contains("backwards"), "{out}");
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");
}

#[tokio::test]
async fn remember_with_episode_rejects_a_mentions_edge_with_a_fresh_fact_as_from() {
    // Given an episode whose one edge is a `mentions` edge with a fresh
    // fact as `from` (backwards: `from` should be an entity) and a
    // handle-shaped `to`, so only the `from`-is-fact check fires,
    // independent of the `to`-is-entity check
    let server = plain_server().await;
    let existing_handle = seed(
        &server,
        "fact",
        "Existing handle target",
        "backwards mentions from-fact handle content",
    )
    .await;
    let marker = "backwards mentions from fact content";
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
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("backwards mentions nested fact content")],
                entities: vec![],
                edges: vec![episode_edge("fact:1", &existing_handle, "mentions")],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused, naming the edge index and the backwards
    // direction, and nothing lands, not even the top-level fact
    assert!(out.contains("edge 0"), "{out}");
    assert!(out.contains("backwards"), "{out}");
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");
}

#[tokio::test]
async fn remember_writes_a_mentions_edge_whose_from_is_a_handle_reference_to_an_existing_entity() {
    // Given a live entity from a prior call, referenced afterward by its
    // handle rather than an episode-local `entity:N` index
    let server = plain_server().await;
    let prior_out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Handle Mentions Entity")],
                edges: vec![],
            }),
            ..remember_args("handle mentions entity top content")
        }))
        .await;
    assert!(!prior_out.contains("failed"), "{prior_out}");
    let entity_handle = prior_out
        .lines()
        .nth(1)
        .expect("entity line")
        .trim_start_matches("remembered ")
        .to_string();

    // When a mentions edge uses that handle as `from` and a fresh fact
    // as `to`: the correct direction, just with `from` not "entity:N"
    // syntax
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("handle mentions fact content")],
                entities: vec![],
                edges: vec![episode_edge(&entity_handle, "fact:1", "mentions")],
            }),
            ..remember_args("handle mentions second top content")
        }))
        .await;

    // Then it succeeds, and the edge is positively confirmed written: a
    // "related" line names the exact from/to pair, not just the absence
    // of a rejection string
    assert!(!out.contains("failed"), "{out}");
    let fact_id = out
        .lines()
        .nth(1)
        .expect("fact line")
        .trim_start_matches("remembered ")
        .to_string();
    assert!(
        out.contains(&format!("related {entity_handle} -mentions-> {fact_id}")),
        "expected the mentions edge confirmed in the response: {out}"
    );
}

#[tokio::test]
async fn remember_with_episode_rejects_out_of_range_confidence_on_a_nested_fact() {
    // Given an episode whose nested fact carries an out-of-range
    // confidence
    let server = plain_server().await;
    let marker = "episode nested confidence rejection content";
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
            episode: Some(EpisodeArgs {
                facts: vec![EpisodeFactArgs {
                    confidence: Some(5.0),
                    ..episode_fact("episode nested confidence nested content")
                }],
                entities: vec![],
                edges: vec![],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused, and nothing lands: per-item validation runs,
    // not just the top-level field's
    assert!(out.contains("confidence"), "{out}");
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");
}

#[tokio::test]
async fn remember_with_episode_rejects_oversized_content_on_a_nested_fact() {
    // Given an episode whose nested fact's content is one character over
    // MAX_CONTENT_CHARS, among otherwise-valid facts
    let server = plain_server().await;
    let marker = "episode nested content ceiling rejection content";
    let over_cap = "x".repeat(MAX_CONTENT_CHARS + 1);
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
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact(&over_cap)],
                entities: vec![],
                edges: vec![],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused with the fact-prefixed content problem, and
    // nothing lands: per-item content validation runs, not just the
    // top-level field's
    assert!(out.contains("fact:1: content exceeds"), "{out}");
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");
}

#[tokio::test]
async fn remember_with_episode_reports_both_content_and_confidence_problems_on_the_same_fact() {
    // Given a single episode fact with BOTH oversized content AND an
    // out-of-range confidence
    let server = plain_server().await;
    let marker = "episode nested content and confidence content";
    let over_cap = "x".repeat(MAX_CONTENT_CHARS + 1);

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![EpisodeFactArgs {
                    confidence: Some(5.0),
                    ..episode_fact(&over_cap)
                }],
                entities: vec![],
                edges: vec![],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then the response reports BOTH problems under the same "fact:1"
    // prefix, proving accumulation on a single item rather than
    // short-circuiting at the first problem found
    assert!(out.contains("fact:1: content exceeds"), "{out}");
    assert!(out.contains("fact:1: confidence must be"), "{out}");
}

#[tokio::test]
async fn remember_with_episode_rejects_an_out_of_bounds_fact_index() {
    // Given an episode with two valid combined fact indices (0 = the
    // top-level fact, 1 = the one episode.facts entry) and an edge
    // naming "fact:5", well past the valid 0..=1 range
    let server = plain_server().await;
    let marker = "episode out of bounds index rejection content";
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
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("episode out of bounds nested content")],
                entities: vec![],
                edges: vec![episode_edge("fact:0", "fact:5", "mentions")],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused specifically for that reference, and nothing
    // lands
    assert!(
        out.contains("not a recognized reference") && out.contains("fact:5"),
        "{out}"
    );
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");

    // And the true boundary, one past the last valid index, is rejected
    // too: "fact:2" is not a valid combined index in this same
    // 2-valid-index episode, so a one-off bug in parse_fact_ref's bound
    // check (e.g. "<=" instead of "<") would wrongly accept it while
    // still correctly rejecting the far-out-of-range "fact:5" above
    let boundary_marker = "episode boundary index rejection content";
    let boundary_out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("episode boundary nested content")],
                entities: vec![],
                edges: vec![episode_edge("fact:0", "fact:2", "mentions")],
            }),
            ..remember_args(boundary_marker)
        }))
        .await;
    assert!(
        boundary_out.contains("not a recognized reference") && boundary_out.contains("fact:2"),
        "{boundary_out}"
    );
    let boundary_after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(boundary_marker))
        .await
        .unwrap()
        .len();
    assert_eq!(
        boundary_after, 0,
        "wrote a node despite boundary rejection: {boundary_out}"
    );
}

#[tokio::test]
async fn remember_with_episode_rejects_an_unrecognized_reference_form() {
    // Given an episode whose edge references "bogus:0", deliberately not
    // "entity:0" (which becomes valid from WU-5 on)
    let server = plain_server().await;
    let marker = "episode unrecognized reference rejection content";
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
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![],
                edges: vec![episode_edge("fact:0", "bogus:0", "mentions")],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused specifically for that reference, and nothing
    // lands
    assert!(
        out.contains("not a recognized reference") && out.contains("bogus:0"),
        "{out}"
    );
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");
}

#[tokio::test]
async fn remember_with_episode_accepts_a_whitespace_padded_handle_reference() {
    // Given a pre-existing handle target, and an episode edge that names
    // it with incidental leading/trailing whitespace: is_handle_shaped
    // must trim before its alphanumeric check, the same way
    // Graph::resolve_handle trims before its own, or this reference is
    // wrongly rejected before resolve_handle ever runs
    let server = plain_server().await;
    let existing_handle = seed(
        &server,
        "fact",
        "Existing handle target",
        "padded handle existing content",
    )
    .await;
    let marker = "padded handle top content";

    // When remember is called with the edge's "to" padded with whitespace
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![],
                edges: vec![episode_edge(
                    "fact:0",
                    &format!("  {existing_handle}  "),
                    "relates_to",
                )],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is accepted, not refused as "not a recognized reference",
    // and the edge resolves to the existing handle's real id
    assert!(!out.contains("failed"), "{out}");
    let top_id = out
        .lines()
        .next()
        .expect("remembered line")
        .trim_start_matches("remembered ")
        .to_string();
    assert!(
        out.contains(&format!("related {top_id} -relates_to-> {existing_handle}")),
        "{out}"
    );
}

#[tokio::test]
async fn remember_with_episode_rejects_a_request_over_max_episode_items() {
    // Given an episode one item over MAX_EPISODE_ITEMS once the
    // always-present top-level fact is counted
    let server = plain_server().await;
    let marker = "episode over max rejection content";
    let before = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    let facts: Vec<EpisodeFactArgs> = (0..MAX_EPISODE_ITEMS)
        .map(|i| episode_fact(&format!("episode over max nested content {i}")))
        .collect();

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts,
                entities: vec![],
                edges: vec![],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused before any DB call, and nothing lands
    assert!(out.contains("failed"), "{out}");
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    assert_eq!(after, before, "wrote a node despite rejection: {out}");

    // Given the same shape, but exactly at the limit (one fewer fact)
    let at_cap_marker = "episode at max cap content";
    let facts: Vec<EpisodeFactArgs> = (0..MAX_EPISODE_ITEMS - 1)
        .map(|i| episode_fact(&format!("episode at max cap nested content {i}")))
        .collect();

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts,
                entities: vec![],
                edges: vec![],
            }),
            ..remember_args(at_cap_marker)
        }))
        .await;

    // Then the limit itself is accepted, not an off-by-one
    assert!(!out.contains("failed"), "{out}");
}

#[tokio::test]
async fn remember_with_episode_reports_an_edge_to_a_since_superseded_fact_as_a_failure() {
    // Given two episode facts sharing a subject, so the second
    // supersedes the first per `Graph::ingest_episode`'s own dedup
    // rules, plus an edge referencing the first by its original combined
    // index
    let server = plain_server().await;
    let top_marker = "episode superseded edge top content";
    let first_marker = "episode superseded edge first content";
    let second_marker = "episode superseded edge second content";

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![
                    EpisodeFactArgs {
                        subject: Some("episode-supersede-subject".to_string()),
                        ..episode_fact(first_marker)
                    },
                    EpisodeFactArgs {
                        subject: Some("episode-supersede-subject".to_string()),
                        ..episode_fact(second_marker)
                    },
                ],
                entities: vec![],
                edges: vec![episode_edge("fact:1", "fact:2", "relates_to")],
            }),
            ..remember_args(top_marker)
        }))
        .await;

    // Then the whole call fails, naming the failing reference (the dead
    // node's id, embedded in `Graph::ingest_episode`'s own "is not live"
    // message), not a generic "remember failed" with no detail
    assert!(out.starts_with("remember failed:"), "{out}");
    assert!(out.contains("is not live"), "{out}");

    // And nothing landed: not the top-level fact, not either episode
    // fact, proving the whole episode rolled back together
    for marker in [top_marker, first_marker, second_marker] {
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap();
        assert_eq!(
            hits.len(),
            0,
            "{marker} wrote a node despite rejection: {out}"
        );
    }
}

#[tokio::test]
async fn remember_with_episode_entities_upserts_by_name_across_calls() {
    // Given two SEPARATE remember calls, each naming entity "Alice" in
    // its own episode
    let server = plain_server().await;

    // When the first call remembers "Alice"
    let first_out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Alice")],
                edges: vec![],
            }),
            ..remember_args("episode entity first call marker")
        }))
        .await;
    assert!(!first_out.contains("failed"), "{first_out}");
    let first_entity_id = first_out
        .lines()
        .nth(1)
        .expect("entity line")
        .trim_start_matches("remembered ")
        .to_string();

    // And the second call remembers "Alice" again, in a wholly separate
    // episode
    let second_out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Alice")],
                edges: vec![],
            }),
            ..remember_args("episode entity second call marker")
        }))
        .await;
    assert!(!second_out.contains("failed"), "{second_out}");
    let second_entity_id = second_out
        .lines()
        .nth(1)
        .expect("entity line")
        .trim_start_matches("remembered ")
        .to_string();

    // Then the second supersedes the first rather than duplicating: the
    // first id is no longer live (resolve_handle only finds live rows),
    // the second is
    assert_ne!(
        first_entity_id, second_entity_id,
        "expected two distinct ids"
    );
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(&first_entity_id)
            .await
            .is_err(),
        "first entity's id is still live; expected the second call to supersede it"
    );
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(&second_entity_id)
            .await
            .is_ok(),
        "second entity's id should be live"
    );
}
