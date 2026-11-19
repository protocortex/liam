// SPDX-License-Identifier: Apache-2.0
//! `remember` with an episode that names entities, and the full mixed episode.

use super::*;

#[tokio::test]
async fn remember_with_episode_entities_get_no_embedding() {
    // Given a server wired with a call-counting embedder, and an
    // episode with one nested fact and two entities
    let store = DefaultGraph::open(":memory:", GraphConfig::new(8))
        .await
        .expect("open in-memory store");
    let embedder = Arc::new(CountingEmbedder::new(8));
    let server = MemoryServer::new(Arc::new(store), embedder.clone(), 30);

    // When remember is called with 1 nested fact and 2 entities
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("episode entity embedding nested content")],
                entities: vec![
                    episode_entity("person", "Bob"),
                    episode_entity("person", "Carol"),
                ],
                edges: vec![],
            }),
            ..remember_args("episode entity embedding top content")
        }))
        .await;

    // Then it succeeds, and the embedder was called exactly once per
    // fact (the top-level fact plus the one nested fact), never for
    // either entity
    assert!(!out.contains("failed"), "{out}");
    assert_eq!(
        embedder.call_count(),
        2,
        "expected 1 (top-level) + 1 (nested fact) = 2 embed calls, entities must not embed"
    );
}

#[tokio::test]
async fn remember_with_episode_edge_references_an_entity_by_index() {
    // Given one entity and an edge "fact:0" -> "entity:0". Kind
    // "relates_to", not "mentions", so the entity stays live below.
    let server = plain_server().await;

    // When remember is called
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Dave")],
                edges: vec![episode_edge("fact:0", "entity:0", "relates_to")],
            }),
            ..remember_args("episode entity edge top content")
        }))
        .await;

    // Then it succeeds, the entity exists, and the edge's reported type
    // is exactly what was sent
    assert!(!out.contains("failed"), "{out}");
    let mut lines = out.lines();
    let top_id = lines
        .next()
        .expect("top-level fact line")
        .trim_start_matches("remembered ")
        .to_string();
    let entity_id = lines
        .next()
        .expect("entity line")
        .trim_start_matches("remembered ")
        .to_string();
    assert!(
        out.contains(&format!("related {top_id} -relates_to-> {entity_id}")),
        "{out}"
    );

    // And the edge exists: relating it again reports "already relates".
    let relate_again = server
        .relate(Parameters(RelateArgs {
            from: top_id,
            to: entity_id,
            kind: "relates_to".to_string(),
        }))
        .await;
    assert!(
        relate_again.contains("already relates"),
        "edge missing: {relate_again}"
    );
}

#[tokio::test]
async fn remember_with_episode_edge_references_the_correct_entity_among_several() {
    // Given one nested fact and two entities, with a correctly-directed
    // mentions edge from "entity:1" to "fact:1". Combined indices: 0 =
    // top-level fact, 1 = the nested fact, 2 = entity 0, 3 = entity 1, so
    // this exercises the nontrivial `1 + fact_count + j` arithmetic on
    // the entity side.
    let server = plain_server().await;

    // When remember is called
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("episode entity among several nested content")],
                entities: vec![
                    episode_entity("person", "Eve"),
                    episode_entity("person", "Frank"),
                ],
                edges: vec![episode_edge("entity:1", "fact:1", "mentions")],
            }),
            ..remember_args("episode entity among several top content")
        }))
        .await;

    // Then it succeeds, and the edge's source is SPECIFICALLY the
    // second entity's node, not the first's
    assert!(!out.contains("failed"), "{out}");
    let mut lines = out.lines();
    let top_id = lines
        .next()
        .expect("top-level fact line")
        .trim_start_matches("remembered ")
        .to_string();
    let fact_id = lines
        .next()
        .expect("nested fact line")
        .trim_start_matches("remembered ")
        .to_string();
    let entity0_id = lines
        .next()
        .expect("first entity line")
        .trim_start_matches("remembered ")
        .to_string();
    let entity1_id = lines
        .next()
        .expect("second entity line")
        .trim_start_matches("remembered ")
        .to_string();
    assert!(
        out.contains(&format!("related {entity1_id} -mentions-> {fact_id}")),
        "{out}"
    );
    for wrong in [&top_id, &entity0_id] {
        assert!(
            !out.contains(&format!("related {wrong} -mentions-> {fact_id}")),
            "edge sourced from the wrong node {wrong}: {out}"
        );
    }
}

#[tokio::test]
async fn remember_with_episode_rejects_a_request_over_max_episode_items_counting_entities() {
    // Given an episode where facts.len() + edges.len() alone is under
    // MAX_EPISODE_ITEMS, but adding entities.len() pushes the total
    // over
    let server = plain_server().await;
    let marker = "episode entities over max rejection content";
    let before = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(marker))
        .await
        .unwrap()
        .len();
    let facts: Vec<EpisodeFactArgs> = (0..MAX_EPISODE_ITEMS / 2)
        .map(|i| episode_fact(&format!("episode entities over max nested content {i}")))
        .collect();
    let entities: Vec<EpisodeEntityArgs> = (0..(MAX_EPISODE_ITEMS - facts.len()))
        .map(|i| episode_entity("person", &format!("Entity {i}")))
        .collect();
    assert!(
        facts.len() < MAX_EPISODE_ITEMS,
        "facts + edges alone must stay under the cap for this test to prove anything"
    );

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts,
                entities,
                edges: vec![],
            }),
            ..remember_args(marker)
        }))
        .await;

    // Then it is refused, and nothing lands
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
}

#[tokio::test]
async fn remember_with_episode_reports_an_edge_to_a_since_superseded_entity_as_a_failure() {
    // Given two entities naming the same person, differing only in case
    // and trailing whitespace (so `NewNode::entity`'s
    // `name.trim().to_lowercase()` normalization, not an accidental
    // exact-string match, is what makes them collide), plus an edge
    // referencing the first by its original combined index
    let server = plain_server().await;
    let top_marker = "episode superseded entity edge top content";

    // When remember is called with it
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![
                    episode_entity("person", "Alice"),
                    episode_entity("person", "alice "),
                ],
                edges: vec![episode_edge("entity:0", "entity:1", "relates_to")],
            }),
            ..remember_args(top_marker)
        }))
        .await;

    // Then the whole call fails, naming the failing reference (the dead
    // node's id, embedded in `Graph::ingest_episode`'s own "is not live"
    // message)
    assert!(out.starts_with("remember failed:"), "{out}");
    assert!(out.contains("is not live"), "{out}");

    // And nothing landed, not even the top-level fact, proving the
    // whole episode rolled back together
    let after = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text(top_marker))
        .await
        .unwrap();
    assert_eq!(after.len(), 0, "wrote a node despite rejection: {out}");
}

#[tokio::test]
async fn remember_with_a_full_mixed_episode_round_trips_through_mcp() {
    // Given a prior, separate `remember` call establishing entity
    // "Grace" as live, and a separate fact remembered before this
    // episode, so this episode's third edge can reference it by its
    // handle string rather than an episode-local index.
    let server = plain_server().await;

    let prior_entity_out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Grace")],
                edges: vec![],
            }),
            ..remember_args("mixed episode prior entity top content")
        }))
        .await;
    assert!(!prior_entity_out.contains("failed"), "{prior_entity_out}");
    let prior_entity_id = prior_entity_out
        .lines()
        .nth(1)
        .expect("prior entity line")
        .trim_start_matches("remembered ")
        .to_string();

    let existing_handle = seed(
        &server,
        "fact",
        "Existing handle target",
        "mixed episode existing handle content",
    )
    .await;

    // When remember is called with a full mixed episode: 2 nested facts
    // (plus the top-level fact makes 3 fact-shaped nodes total), 2
    // entities (one brand new, one superseding "Grace" above), and 3
    // edges: fact-to-fact by index, fact-to-entity by index, and
    // fact-to-existing-handle by handle string
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![
                    episode_fact("mixed episode fact one content"),
                    episode_fact("mixed episode fact two content"),
                ],
                entities: vec![
                    episode_entity("person", "Henry"),
                    episode_entity("person", "Grace"),
                ],
                edges: vec![
                    episode_edge("fact:1", "fact:2", "supports"),
                    episode_edge("entity:0", "fact:1", "mentions"),
                    episode_edge("fact:2", &existing_handle, "relates_to"),
                ],
            }),
            ..remember_args("mixed episode top content")
        }))
        .await;

    // Then the response has one "remembered {id}" line per written node
    // (top fact, 2 nested facts, 2 entities, in order) followed by one
    // "related ..." line per edge, in order; ids are read out of the
    // response text itself, since they are runtime-generated ULIDs
    assert!(!out.contains("failed"), "{out}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 8, "expected 5 remembered + 3 related: {out}");
    for line in &lines[..5] {
        assert!(line.starts_with("remembered "), "{out}");
    }
    for line in &lines[5..] {
        assert!(line.starts_with("related "), "{out}");
    }
    let top_id = lines[0].trim_start_matches("remembered ").to_string();
    let fact1_id = lines[1].trim_start_matches("remembered ").to_string();
    let fact2_id = lines[2].trim_start_matches("remembered ").to_string();
    let henry_id = lines[3].trim_start_matches("remembered ").to_string();
    let grace_id = lines[4].trim_start_matches("remembered ").to_string();

    assert_eq!(
        lines[5],
        format!("related {fact1_id} -supports-> {fact2_id}")
    );
    assert_eq!(
        lines[6],
        format!("related {henry_id} -mentions-> {fact1_id}")
    );
    assert_eq!(
        lines[7],
        format!("related {fact2_id} -relates_to-> {existing_handle}")
    );

    // Every written item is independently recall-able afterward: the
    // top fact and both nested facts by content, the pre-existing
    // handle target too
    for marker in [
        "mixed episode top content",
        "mixed episode fact one content",
        "mixed episode fact two content",
        "mixed episode existing handle content",
    ] {
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker))
            .await
            .unwrap();
        assert!(
            hits.iter().any(|h| h.hit.retrieved.as_text() == marker),
            "{marker} missing: {out}"
        );
    }

    // Henry's own id was superseded by its own resynthesis (mentioned
    // via a "mentions" edge); a live, resynthesized Henry still exists
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(&henry_id)
            .await
            .is_err(),
        "Henry's original id should have been superseded by its own resynthesis"
    );
    let resynthesized_henry = live_row(&server.store().await.unwrap(), "Henry", None).await;
    assert!(
        resynthesized_henry.is_some_and(|h| h.hit.label == "Henry"),
        "expected a resynthesized, still-live Henry entity: {out}"
    );
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(&grace_id)
            .await
            .is_ok(),
        "superseding entity should be live"
    );
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(&prior_entity_id)
            .await
            .is_err(),
        "prior entity should have been superseded, not left live"
    );
    assert_ne!(prior_entity_id, grace_id, "expected two distinct ids");

    // The other 2 edges still exist, each reporting "already relates";
    // Henry's own edge is checked above instead, since its target is superseded.
    for (from, to, kind) in [
        (fact1_id.clone(), fact2_id.clone(), "supports"),
        (fact2_id.clone(), existing_handle.clone(), "relates_to"),
    ] {
        let relate_again = server
            .relate(Parameters(RelateArgs {
                from,
                to,
                kind: kind.to_string(),
            }))
            .await;
        assert!(
            relate_again.contains("already relates"),
            "edge missing for {kind}: {relate_again}"
        );
    }
    // top_id is written but not referenced by any edge in this episode;
    // confirming it round-trips through recall proves it was still
    // written even though nothing points at it.
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(&top_id)
            .await
            .is_ok(),
        "top-level fact should be live"
    );
}

#[tokio::test]
async fn remember_with_episode_reports_every_validation_problem_before_writing_anything() {
    // Given an episode with two independent problems in the same call:
    // an out-of-range confidence on episode.facts[1], and a reserved
    // edge kind on episode.edges[0]
    let server = plain_server().await;
    let top_marker = "episode multi problem top content";
    let fact0_marker = "episode multi problem fact zero content";
    let fact1_marker = "episode multi problem fact one content";

    // When remember is called with both problems present at once
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![
                    episode_fact(fact0_marker),
                    EpisodeFactArgs {
                        confidence: Some(5.0),
                        ..episode_fact(fact1_marker)
                    },
                ],
                entities: vec![],
                edges: vec![episode_edge("fact:0", "fact:1", "supersedes")],
            }),
            ..remember_args(top_marker)
        }))
        .await;

    // Then the response reports BOTH problems, not just the first one
    // encountered, proving `problems.push(...)` accumulates across the
    // whole validation pass rather than returning at the first failure
    assert!(out.starts_with("remember failed:"), "{out}");
    assert!(
        out.contains("confidence"),
        "confidence problem missing: {out}"
    );
    assert!(out.contains("reserved"), "edge kind problem missing: {out}");

    // And nothing landed at all: not the top-level fact, not either
    // nested fact
    for marker in [top_marker, fact0_marker, fact1_marker] {
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
async fn remember_with_episode_facts_and_entities_inherit_the_top_level_scope() {
    // Given a top-level fact scoped to "proj-x", plus one nested fact
    // and one entity in the same episode
    let server = plain_server().await;
    let top_marker = "episode scope top content";
    let fact_marker = "episode scope nested content";
    let entity_marker = "Zorbnaxia";

    // When remember is called with `scope: Some("proj-x")` on the
    // top-level args
    let out = server
        .remember(Parameters(RememberArgs {
            scope: Some("proj-x".to_string()),
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact(fact_marker)],
                entities: vec![episode_entity("person", entity_marker)],
                edges: vec![],
            }),
            ..remember_args(top_marker)
        }))
        .await;

    // Then it succeeds, and a query scoped to "proj-x" finds every one
    // of the three nodes: the doc comment on `EpisodeFactArgs` promises
    // nested facts "share the top-level fact's scope"; this exercises
    // entities too, which share the same contract. An entity has no
    // content (`NewNode::entity` always leaves it empty), so its
    // marker is matched on `label` instead of `content`.
    assert!(!out.contains("failed"), "{out}");
    for marker in [top_marker, fact_marker, entity_marker] {
        let scoped = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker).with_scope("proj-x").with_inferred(true))
            .await
            .unwrap();
        assert!(
            scoped
                .iter()
                .any(|h| h.hit.retrieved.as_text() == marker || h.hit.label == marker),
            "{marker} missing from the proj-x scope: {out}"
        );

        // And the same marker is invisible to a DIFFERENT scope's
        // query, the isolation guarantee `scope` exists for. A node
        // that silently landed unscoped (the bug this fixes) would
        // leak into every scope's query instead, since
        // `fetch_candidates`'s scope filter only activates when the
        // QUERY carries a scope, and an unscoped row never equals a
        // scoped filter value.
        let other_scope = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker).with_scope("proj-y").with_inferred(true))
            .await
            .unwrap();
        assert!(
            other_scope.is_empty(),
            "{marker} leaked into the proj-y scope: {out}"
        );
    }
}

#[tokio::test]
async fn remember_with_episode_facts_and_entities_stay_unscoped_when_the_top_level_scope_is_absent()
{
    // Given a top-level fact with no `scope` set, plus one nested fact
    // and one entity, the shape every pre-refinement caller sends
    let server = plain_server().await;
    let top_marker = "episode no scope top content";
    let fact_marker = "episode no scope nested content";
    let entity_marker = "Quexlorn";

    // When remember is called
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact(fact_marker)],
                entities: vec![episode_entity("person", entity_marker)],
                edges: vec![],
            }),
            ..remember_args(top_marker)
        }))
        .await;

    // Then it succeeds, and every node (top-level fact, nested fact,
    // entity) is invisible to ANY scoped query: an unscoped row's
    // `scope` column is NULL, which never equals a `scope = ?` filter,
    // so this stays true whether or not the fix landed and pins
    // today's no-scope behavior against a regression.
    assert!(!out.contains("failed"), "{out}");
    for marker in [top_marker, fact_marker, entity_marker] {
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(marker).with_scope("some-scope"))
            .await
            .unwrap();
        assert!(
            hits.is_empty(),
            "{marker} was visible to a scoped query despite no scope being set: {out}"
        );
    }
}
