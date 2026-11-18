// SPDX-License-Identifier: Apache-2.0
//! Synthesis of entity pages: what it writes, keeps and records when it fails.

use super::*;

#[tokio::test]
async fn resynthesize_entity_embeds_the_new_content_as_a_document() {
    // Arrange
    let (server, embedder) =
        server_with_recording_embedder_and_llm(Arc::new(CountingLlm::new())).await;
    let entity_id = server
        .store()
        .await
        .unwrap()
        .upsert_by(NewNode::entity("person", "Kim"))
        .await
        .expect("seed entity");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    server
        .resynthesize_entity(entity_id, deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
        .await
        .expect("resynthesis should succeed");

    // Assert
    let calls = embedder.calls();
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].1, TextKind::Document);
}

/// A page that mentions three facts: two written in one episode, so one
/// event, and a third written on its own.
async fn page_with_three_mentions(store: &DefaultGraph) -> liam_store::NodeId {
    let page = store
        .upsert_by(NewNode::entity("person", "Kim"))
        .await
        .unwrap();
    let episode = store
        .ingest_episode(
            vec![
                NewNode::now("fact", "Role", "Kim works as an engineer."),
                NewNode::now("fact", "Home", "Kim lives in Lisbon."),
            ],
            vec![],
        )
        .await
        .unwrap();
    let alone = store
        .insert(NewNode::now("fact", "Pet", "Kim has a cat."))
        .await
        .unwrap();
    for id in episode.node_ids.iter().chain([&alone]) {
        store.relate(&page, id, relation::MENTIONS).await.unwrap();
    }
    page
}

/// The basis of the live page labelled Kim.
async fn basis_of_kim(store: &DefaultGraph) -> Vec<EventRef> {
    let hits = store
        .query_explained(&Query::text("person Kim").with_inferred(true))
        .await
        .unwrap();
    let written = hits
        .iter()
        .find(|h| h.hit.label == "Kim")
        .expect("resynthesized page missing");
    store
        .node_event_ids(&written.hit.id, EventRole::Basis)
        .await
        .unwrap()
}

#[tokio::test]
async fn resynthesize_entity_lists_each_mentions_origin_event_once_as_the_pages_basis() {
    // Arrange
    let llm = Arc::new(CountingLlm::new());
    let server = server_with(llm).await;
    let store = server.store().await.unwrap();
    let page = page_with_three_mentions(&store).await;
    let mut expected = std::collections::BTreeSet::new();
    for mention in store
        .mentions(&page, liam_store::Millis::now(), 10)
        .await
        .unwrap()
    {
        expected.extend(
            store
                .node_event_ids(&mention.id, EventRole::Origin)
                .await
                .unwrap(),
        );
    }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    server
        .resynthesize_entity(page, deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
        .await
        .expect("resynthesis should succeed");

    // Assert
    assert_eq!(expected.len(), 2);
    assert_eq!(
        basis_of_kim(&store).await,
        expected.into_iter().collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn resynthesize_entity_leaves_a_mention_the_budget_dropped_out_of_the_basis() {
    // Arrange: a context budget that keeps only the first mention
    let llm = Arc::new(CountingLlm::new());
    let server = server_with_timeout(llm, 30, false, 1).await;
    let store = server.store().await.unwrap();
    let page = page_with_three_mentions(&store).await;
    let mentions = store
        .mentions(&page, liam_store::Millis::now(), 10)
        .await
        .unwrap();
    let kept = store
        .node_event_ids(&mentions[0].id, EventRole::Origin)
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    server
        .resynthesize_entity(page, deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
        .await
        .expect("resynthesis should succeed");

    // Assert
    assert_eq!(basis_of_kim(&store).await, kept);
}

#[tokio::test]
async fn resynthesize_entity_as_writes_the_page_under_the_given_source() {
    // Arrange
    let (server, _embedder) =
        server_with_recording_embedder_and_llm(Arc::new(CountingLlm::new())).await;
    let entity_id = server
        .store()
        .await
        .unwrap()
        .upsert_by(NewNode::entity("person", "Kim"))
        .await
        .expect("seed entity");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act
    server
        .resynthesize_entity_as(
            &server.store().await.unwrap(),
            entity_id.clone(),
            deadline,
            ENTITY_SYNTHESIS_MAX_NEW_TOKENS,
            Some("consolidation:9"),
        )
        .await
        .expect("resynthesis should succeed");

    // Assert
    let pages = server
        .store()
        .await
        .unwrap()
        .changes_since(liam_store::BEGINNING)
        .await
        .expect("changes");
    let newest = pages.iter().rfind(|change| !change.closed).expect("a page");
    let page = server
        .store()
        .await
        .unwrap()
        .get(&newest.id, liam_store::Millis(liam_store::FOREVER.0 - 1))
        .await
        .expect("get")
        .expect("live page");
    assert_eq!(page.producer, "consolidation:9");
}

#[tokio::test]
async fn remember_synthesizes_a_mentioned_entity_exactly_once_despite_two_mentions_edges() {
    // Given an episode where 2 separate facts each carry a mentions
    // edge to the SAME fresh entity
    let llm = Arc::new(CountingLlm::new());
    let server = server_with(llm.clone()).await;

    // When remember runs
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![
                    episode_fact("dedup trigger fact one content"),
                    episode_fact("dedup trigger fact two content"),
                ],
                entities: vec![episode_entity("person", "Dedup Entity")],
                edges: vec![
                    episode_edge("entity:0", "fact:1", "mentions"),
                    episode_edge("entity:0", "fact:2", "mentions"),
                ],
            }),
            ..remember_args("dedup trigger top content")
        }))
        .await;

    // Then synthesis ran exactly once for that entity, not twice
    assert!(!out.contains("synthesis failed"), "{out}");
    assert_eq!(
        llm.call_count(),
        1,
        "expected exactly one synthesis call: {out}"
    );
}

#[tokio::test]
async fn remember_synthesizes_every_freshly_mentioned_entity_before_returning() {
    // Given an episode mentioning 2 distinct fresh entities
    let llm = Arc::new(CountingLlm::new());
    let server = server_with(llm.clone()).await;

    // When remember runs
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![
                    episode_entity("person", "Both One"),
                    episode_entity("person", "Both Two"),
                ],
                edges: vec![
                    episode_edge("entity:0", "fact:0", "mentions"),
                    episode_edge("entity:1", "fact:0", "mentions"),
                ],
            }),
            ..remember_args("both entities top content")
        }))
        .await;

    // Then both synthesized, and both completed before returning: the
    // ORIGINAL entity ids are already superseded by their own writes
    assert!(!out.contains("synthesis failed"), "{out}");
    assert_eq!(llm.call_count(), 2, "{out}");
    let entity0_id = out
        .lines()
        .nth(1)
        .expect("first entity line")
        .trim_start_matches("remembered ")
        .to_string();
    let entity1_id = out
        .lines()
        .nth(2)
        .expect("second entity line")
        .trim_start_matches("remembered ")
        .to_string();
    for id in [&entity0_id, &entity1_id] {
        assert!(
            server
                .store()
                .await
                .unwrap()
                .resolve_handle(id)
                .await
                .is_err(),
            "original entity id should have been superseded by its own resynthesis: {out}"
        );
    }
    for label in ["Both One", "Both Two"] {
        let expected_content = format!("person {label}");
        let hits = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text(expected_content.as_str()).with_inferred(true))
            .await
            .unwrap();
        assert!(
            hits.iter()
                .any(|h| h.hit.label == label && h.hit.retrieved.as_text() == expected_content),
            "{label}'s resynthesized content should already be queryable: {out}"
        );
    }
}

#[tokio::test]
async fn resynthesize_entity_preserves_existing_attributes_on_the_new_node() {
    // Given a live entity that already carries attributes
    let llm = Arc::new(CountingLlm::new());
    let server = server_with(llm.clone()).await;
    let entity_id = server
        .store()
        .await
        .unwrap()
        .upsert_by(NewNode::entity("person", "Kim").with_attributes(json!({"role": "engineer"})))
        .await
        .expect("seed entity with attributes");

    // When its resynthesis runs, the same per-entity flow `remember`
    // triggers for a freshly mentioned entity
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    server
        .resynthesize_entity(entity_id.clone(), deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
        .await
        .expect("resynthesis should succeed");

    // Then the new node's attributes match the old one's, not wiped
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(entity_id.as_str())
            .await
            .is_err(),
        "the old version should have been superseded"
    );
    let hits = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("person Kim").with_inferred(true))
        .await
        .unwrap();
    let new_entity = hits
        .iter()
        .find(|h| h.hit.label == "Kim")
        .expect("resynthesized entity missing");
    assert_eq!(new_entity.hit.retrieved.as_text(), "person Kim");
    assert_eq!(new_entity.hit.attributes, json!({"role": "engineer"}));
}

#[tokio::test]
async fn resynthesize_entity_preserves_the_original_entitys_scope() {
    // Given a live entity scoped to "proj-x"
    let llm = Arc::new(CountingLlm::new());
    let server = server_with(llm.clone()).await;
    let entity_id = server
        .store()
        .await
        .unwrap()
        .upsert_by(NewNode::entity("person", "Priya").with_scope("proj-x"))
        .await
        .expect("seed scoped entity");

    // When its resynthesis runs, the same per-entity flow `remember`
    // triggers for a freshly mentioned entity
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    server
        .resynthesize_entity(entity_id.clone(), deadline, ENTITY_SYNTHESIS_MAX_NEW_TOKENS)
        .await
        .expect("resynthesis should succeed");

    // Then the original id is genuinely superseded, not left live
    // alongside a second, unscoped duplicate: a scope-dropping
    // rebuild fails to match the live "proj-x" competitor by subject
    // AND scope, so `upsert_by` falls through to `insert` and the
    // old id survives
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(entity_id.as_str())
            .await
            .is_err(),
        "the old scoped version should have been superseded, not forked"
    );

    // And the resynthesized node kept the same scope: it is visible
    // to a proj-x-scoped query...
    let scoped = server
        .store()
        .await
        .unwrap()
        .query_explained(
            &Query::text("person Priya")
                .with_scope("proj-x")
                .with_inferred(true),
        )
        .await
        .unwrap();
    assert!(
        scoped.iter().any(|h| h.hit.label == "Priya"),
        "resynthesized entity missing from its own proj-x scope"
    );

    // ...but not to a different scope's query, the isolation
    // guarantee a scope-dropping rebuild would silently break
    let other_scope = server
        .store()
        .await
        .unwrap()
        .query_explained(
            &Query::text("person Priya")
                .with_scope("proj-y")
                .with_inferred(true),
        )
        .await
        .unwrap();
    assert!(
        !other_scope.iter().any(|h| h.hit.label == "Priya"),
        "resynthesized entity leaked into the proj-y scope"
    );
}

#[tokio::test]
async fn resynthesize_entity_honors_the_deadline_it_is_given_not_a_fresh_one() {
    // Given an already-elapsed deadline, a sole permit held all test long,
    // and an hour-long `ask_timeout_secs` a fresh deadline would never miss.
    let llm = Arc::new(CountingLlm::new());
    let server = server_with_generation_limit(llm, 3600, false, 8192, 1).await;
    let _held = server
        .generation_permits
        .clone()
        .acquire_owned()
        .await
        .expect("hold the sole permit");
    let entity_id = server
        .store()
        .await
        .unwrap()
        .upsert_by(NewNode::entity("person", "Deadline Carrier"))
        .await
        .expect("seed entity");
    let already_elapsed = tokio::time::Instant::now() - Duration::from_millis(1);

    // When resynthesis runs with that deadline, guarded by a short real
    // timeout so a regression that recomputes its own fails, not hangs
    let outcome = tokio::time::timeout(
        Duration::from_millis(200),
        server.resynthesize_entity(entity_id, already_elapsed, ENTITY_SYNTHESIS_MAX_NEW_TOKENS),
    )
    .await
    .expect("must fail fast on the passed deadline, not hang on a freshly computed one");

    // Then it fails immediately for having missed THAT deadline, proving
    // the passed-in value governed the wait rather than a fresh one
    let err = outcome.expect_err("an already-elapsed deadline must not grant a permit");
    assert!(
        err.contains("timed out waiting for a generation slot"),
        "{err}"
    );
}

#[tokio::test]
async fn remember_still_commits_the_episode_when_entity_synthesis_fails() {
    // Given a mock llm that always errors
    let llm = Arc::new(FailingLlm);
    let server = server_with(llm).await;

    // When remember runs an episode mentioning a fresh entity
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Failing Entity")],
                edges: vec![episode_edge("entity:0", "fact:0", "mentions")],
            }),
            ..remember_args("synthesis failure top content")
        }))
        .await;

    // Then the episode's facts/entities/edges are still fully
    // committed
    assert!(!out.starts_with("remember failed:"), "{out}");
    let top = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("synthesis failure top content"))
        .await
        .unwrap();
    assert!(
        top.iter()
            .any(|h| h.hit.retrieved.as_text() == "synthesis failure top content"),
        "{out}"
    );
    let entity_id = out
        .lines()
        .nth(1)
        .expect("entity line")
        .trim_start_matches("remembered ")
        .to_string();
    assert!(
        server
            .store()
            .await
            .unwrap()
            .resolve_handle(&entity_id)
            .await
            .is_ok(),
        "entity should still be live: {out}"
    );

    // And the response names the entity synthesis failed for
    assert!(out.contains("synthesis failed for Failing Entity"), "{out}");
}

#[tokio::test]
async fn remember_runs_entity_synthesis_concurrently_not_sequentially() {
    // 2 permits, not the shipped default of 1: only then can peak
    // reach 2, proving true concurrency, not just bounded queuing.
    let release = Arc::new(tokio::sync::Notify::new());
    let llm = Arc::new(GatedLlm::new_grounded(release.clone()));
    let server = Arc::new(server_with_generation_limit(llm.clone(), 30, false, 8192, 2).await);

    // When remember runs an episode mentioning 2 distinct fresh
    // entities, spawned since the gated llm blocks until released
    let remember_server = server.clone();
    let handle = tokio::spawn(async move {
        remember_server
            .remember(Parameters(RememberArgs {
                episode: Some(EpisodeArgs {
                    facts: vec![],
                    entities: vec![
                        episode_entity("person", "Concurrent One"),
                        episode_entity("person", "Concurrent Two"),
                    ],
                    edges: vec![
                        episode_edge("entity:0", "fact:0", "mentions"),
                        episode_edge("entity:1", "fact:0", "mentions"),
                    ],
                }),
                ..remember_args("concurrent entity synthesis top content")
            }))
            .await
    });
    wait_until(|| llm.in_flight() == 2).await;
    release.notify_one();
    release.notify_one();
    let out = handle.await.expect("remember task panicked");

    // Then the PEAK proves both syntheses were in flight together,
    // not one after the other
    assert!(!out.contains("failed"), "{out}");
    assert_eq!(
        llm.peak(),
        2,
        "expected both entity syntheses concurrently in flight: {out}"
    );
}

#[tokio::test]
async fn remember_records_both_outcomes_when_one_of_two_entity_syntheses_fails() {
    // Given a llm that errors on its first call and succeeds after,
    // and an episode mentioning 2 distinct fresh entities
    let llm = Arc::new(FirstCallErrorsLlm::new());
    let server = server_with(llm).await;

    // When remember runs
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![
                    episode_entity("person", "Mixed One"),
                    episode_entity("person", "Mixed Two"),
                ],
                edges: vec![
                    episode_edge("entity:0", "fact:0", "mentions"),
                    episode_edge("entity:1", "fact:0", "mentions"),
                ],
            }),
            ..remember_args("mixed outcome top content")
        }))
        .await;

    // Then both outcomes are recorded: one entity's content was
    // compiled, the other is named in the failure list. Filtered to an
    // exact content match, not just a lexical hit: the FTS search for
    // "person Mixed One" also surfaces "person Mixed Two" (they share
    // both words), so raw hit count alone would not tell the two apart.
    let mut succeeded: Vec<_> = server
        .store()
        .await
        .unwrap()
        .query_explained(&Query::text("person Mixed One").with_inferred(true))
        .await
        .unwrap()
        .into_iter()
        .filter(|h| h.hit.retrieved.as_text() == "person Mixed One")
        .collect();
    if succeeded.is_empty() {
        succeeded = server
            .store()
            .await
            .unwrap()
            .query_explained(&Query::text("person Mixed Two").with_inferred(true))
            .await
            .unwrap()
            .into_iter()
            .filter(|h| h.hit.retrieved.as_text() == "person Mixed Two")
            .collect();
    }
    assert_eq!(
        succeeded.len(),
        1,
        "expected exactly one entity to succeed: {out}"
    );
    let survivor_label = succeeded[0].hit.label.clone();
    let failed_label = if survivor_label == "Mixed One" {
        "Mixed Two"
    } else {
        "Mixed One"
    };
    assert!(
        out.contains(&format!("synthesis failed for {failed_label}")),
        "expected the OTHER entity named as failed: {out}"
    );
    assert!(
        !out.contains(&format!("synthesis failed for {survivor_label}")),
        "the successful entity must not also be reported as failed: {out}"
    );
}

#[tokio::test]
async fn remember_does_not_resynthesize_an_existing_entity_referenced_only_by_handle() {
    // Given a live entity from a prior call
    let llm = Arc::new(CountingLlm::new());
    let server = server_with(llm.clone()).await;
    let prior_out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Handle Only Entity")],
                edges: vec![],
            }),
            ..remember_args("handle only entity top content")
        }))
        .await;
    assert!(!prior_out.contains("failed"), "{prior_out}");
    let entity_handle = prior_out
        .lines()
        .nth(1)
        .expect("entity line")
        .trim_start_matches("remembered ")
        .to_string();

    // When a second episode mentions the SAME entity only by handle,
    // never as an entity:N reference
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("handle only mention fact content")],
                entities: vec![],
                edges: vec![episode_edge(&entity_handle, "fact:1", "mentions")],
            }),
            ..remember_args("handle only second top content")
        }))
        .await;

    // Then synthesis does not run for it
    assert!(!out.contains("failed"), "{out}");
    assert_eq!(
        llm.call_count(),
        0,
        "existing entities referenced only by handle must not trigger resynthesis: {out}"
    );
}

#[tokio::test]
async fn remember_synthesizes_a_mentioned_entity_from_its_real_mention_content() {
    // Given a fresh entity linked to a fact via a correctly-directed
    // mentions edge (`from`: entity, `to`: fact)
    let llm = Arc::new(RecordingLlm::new("person Xyzzy Entity"));
    let server = server_with(llm.clone()).await;

    // When remember runs
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("the xyzzy-marker-42 distinctive fact content")],
                entities: vec![episode_entity("person", "Xyzzy Entity")],
                edges: vec![episode_edge("entity:0", "fact:1", "mentions")],
            }),
            ..remember_args("xyzzy marker top content")
        }))
        .await;

    // Then the fact's real content reached the synthesis prompt, not an
    // empty mentions list
    assert!(!out.contains("synthesis failed"), "{out}");
    assert!(
        llm.last_prompt()
            .contains("the xyzzy-marker-42 distinctive fact content"),
        "expected the mentioned fact's real content in the synthesis prompt: {}",
        llm.last_prompt()
    );
}
