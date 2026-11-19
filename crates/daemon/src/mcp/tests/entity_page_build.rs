// SPDX-License-Identifier: Apache-2.0
//! Building entity pages: what is written, what it rests on, and what
//! happens when the embedder fails.

use liam_model::{Embedder, MockEmbedder};
use liam_store::{Millis, NodeId};

use super::test_doubles::GatedEmbedder;
use super::*;

const DAY_MS: i64 = 86_400_000;
const DIMS: usize = 8;

/// The text of a memory that has no stored vector and that the embedder refuses,
/// which is how a page build fails: only such a mention is embedded.
const REFUSED_MENTION: &str = "a mention the embedder refuses";

/// An embedder that fails on `REFUSED_MENTION` and works for everything else.
struct MentionFailingEmbedder {
    inner: MockEmbedder,
}

#[async_trait::async_trait]
impl Embedder for MentionFailingEmbedder {
    fn dims(&self) -> usize {
        self.inner.dims()
    }
    async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
        if text == REFUSED_MENTION {
            return Err(liam_model::ModelError::embed_failed(
                "mention refused",
                "refused",
            ));
        }
        self.inner.embed(text, kind).await
    }
}

/// A server whose embedder refuses `REFUSED_MENTION`, and the id of a memory
/// with that text and no vector.
async fn server_with_a_refused_mention() -> (MemoryServer, NodeId) {
    let server = server_with_embedder(Arc::new(MentionFailingEmbedder {
        inner: MockEmbedder::new(DIMS),
    }))
    .await;
    let refused = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now("fact", "note", REFUSED_MENTION))
        .await
        .unwrap();
    (server, refused)
}

/// A unit vector along `axis`, so two mentions on different axes are never
/// near duplicates.
fn axis(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMS];
    vector[axis] = 1.0;
    vector
}

/// Writes a fact valid from `day`, stored under `embedding`, and links the
/// entity `page` to it.
async fn mention(
    store: &DefaultGraph,
    page: &NodeId,
    text: &str,
    day: i64,
    embedding: Vec<f32>,
) -> NodeId {
    let id = store
        .insert(
            NewNode::now("fact", "note", text)
                .with_valid_from(Millis(day * DAY_MS))
                .with_embedding(embedding),
        )
        .await
        .unwrap();
    store.relate(page, &id, relation::MENTIONS).await.unwrap();
    id
}

async fn entity(store: &DefaultGraph, name: &str) -> NodeId {
    store
        .upsert_by(NewNode::entity("person", name))
        .await
        .unwrap()
}

fn soon() -> tokio::time::Instant {
    tokio::time::Instant::now() + Duration::from_secs(5)
}

/// The live page labelled `name`.
async fn page_named(store: &DefaultGraph, name: &str) -> Found {
    live_row(store, name, None)
        .await
        .unwrap_or_else(|| panic!("no page labelled {name}"))
}

/// A page with three mentions: two written in one episode, so one event, and a
/// third written on its own. Each is on its own axis.
async fn page_with_three_mentions(store: &DefaultGraph) -> NodeId {
    let page = entity(store, "Kim").await;
    let episode = store
        .ingest_episode(
            vec![
                NewNode::now("fact", "Role", "Kim works as an engineer.").with_embedding(axis(0)),
                NewNode::now("fact", "Home", "Kim lives in Lisbon.").with_embedding(axis(1)),
            ],
            vec![],
        )
        .await
        .unwrap();
    for id in &episode.node_ids {
        store.relate(&page, id, relation::MENTIONS).await.unwrap();
    }
    mention(store, &page, "Kim has a cat.", 3, axis(2)).await;
    page
}

#[tokio::test]
async fn the_page_is_not_embedded_and_stored_mention_vectors_are_reused() {
    // Arrange: the one mention already has a vector
    let (server, embedder) = server_with_recording_embedder().await;
    let store = server.store().await.unwrap();
    let page = entity(&store, "Kim").await;
    mention(&store, &page, "Kim likes tea.", 3, axis(0)).await;

    // Act
    server
        .resynthesize_entity_as(&store, page, soon(), None)
        .await
        .expect("the page should be built");

    // Assert: no embed call at all, and the page row holds the text and no vector
    assert!(embedder.calls().is_empty(), "{:?}", embedder.calls());
    let written = page_named(&store, "Kim").await;
    assert_eq!(
        written.hit.retrieved.as_text(),
        "Kim (person)\n1970-01-04: Kim likes tea."
    );
    assert!(store.embedding(&written.hit.id).await.unwrap().is_none());
}

#[tokio::test]
async fn a_mention_without_a_stored_vector_is_embedded_as_a_document() {
    // Arrange: a mention written with no vector
    let (server, embedder) = server_with_recording_embedder().await;
    let store = server.store().await.unwrap();
    let page = entity(&store, "Kim").await;
    let bare = store
        .insert(NewNode::now("fact", "note", "Kim likes tea."))
        .await
        .unwrap();
    store
        .relate(&page, &bare, relation::MENTIONS)
        .await
        .unwrap();

    // Act
    server
        .resynthesize_entity_as(&store, page, soon(), None)
        .await
        .expect("the page should be built");

    // Assert: the mention is embedded, and the page is not
    let calls = embedder.calls();
    assert_eq!(calls, [("Kim likes tea.".to_string(), TextKind::Document)]);
}

#[tokio::test]
async fn the_page_lists_each_mentions_origin_event_once_as_its_basis() {
    // Arrange
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let page = page_with_three_mentions(&store).await;
    let mut expected = std::collections::BTreeSet::new();
    for mention in store.mentions(&page, Millis::now(), 10).await.unwrap() {
        expected.extend(
            store
                .node_event_ids(&mention.id, EventRole::Origin)
                .await
                .unwrap(),
        );
    }

    // Act
    server
        .resynthesize_entity_as(&store, page, soon(), None)
        .await
        .expect("the page should be built");

    // Assert
    let written = page_named(&store, "Kim").await;
    let basis = store
        .node_event_ids(&written.hit.id, EventRole::Basis)
        .await
        .unwrap();
    assert_eq!(expected.len(), 2);
    assert_eq!(basis, expected.into_iter().collect::<Vec<_>>());
}

#[tokio::test]
async fn a_mention_the_page_dropped_as_a_duplicate_is_left_out_of_the_basis() {
    // Arrange: two mentions with the same vector, the older one written first
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let page = entity(&store, "Kim").await;
    let older = mention(&store, &page, "Kim lives in Lisbon.", 1, axis(0)).await;
    let newer = mention(&store, &page, "Kim lives in Porto.", 2, axis(0)).await;
    let origin = |id: NodeId| {
        let store = &store;
        async move { store.node_event_ids(&id, EventRole::Origin).await.unwrap() }
    };
    let (older_events, newer_events) = (origin(older).await, origin(newer).await);

    // Act
    server
        .resynthesize_entity_as(&store, page, soon(), None)
        .await
        .expect("the page should be built");

    // Assert
    let written = page_named(&store, "Kim").await;
    assert!(!written.hit.retrieved.as_text().contains("Lisbon"));
    let basis = store
        .node_event_ids(&written.hit.id, EventRole::Basis)
        .await
        .unwrap();
    assert_eq!(basis, newer_events);
    assert!(older_events.iter().all(|event| !basis.contains(event)));
}

#[tokio::test]
async fn an_entity_with_no_mention_keeps_its_row_and_gets_no_page() {
    // Arrange
    let (server, embedder) = server_with_recording_embedder().await;
    let store = server.store().await.unwrap();
    let id = entity(&store, "Kim").await;

    // Act
    let built = server
        .resynthesize_entity_as(&store, id.clone(), soon(), None)
        .await;

    // Assert
    assert_eq!(built, Ok(()));
    assert_eq!(embedder.call_count(), 0);
    assert!(store.resolve_handle(id.as_str()).await.is_ok());
    assert_eq!(page_named(&store, "Kim").await.hit.retrieved.as_text(), "");
}

#[tokio::test]
async fn an_entity_that_mentions_only_derived_text_gets_no_page() {
    // Arrange: the only thing the entity links to is another page
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let id = entity(&store, "Kim").await;
    let other = entity(&store, "Other").await;
    store.relate(&id, &other, relation::MENTIONS).await.unwrap();

    // Act
    let built = server
        .resynthesize_entity_as(&store, id, soon(), None)
        .await;

    // Assert
    assert_eq!(built, Ok(()));
    assert_eq!(page_named(&store, "Kim").await.hit.retrieved.as_text(), "");
}

#[tokio::test]
async fn the_page_is_written_under_the_given_source() {
    // Arrange
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let page = entity(&store, "Kim").await;
    mention(&store, &page, "Kim likes tea.", 3, axis(0)).await;

    // Act
    server
        .resynthesize_entity_as(&store, page, soon(), Some("consolidation:9"))
        .await
        .expect("the page should be built");

    // Assert
    let changes = store.changes_since(liam_store::BEGINNING).await.unwrap();
    let newest = changes.iter().rfind(|change| !change.closed).unwrap();
    let written = store
        .get(&newest.id, Millis(liam_store::FOREVER.0 - 1))
        .await
        .unwrap()
        .expect("live page");
    assert_eq!(written.producer, "consolidation:9");
}

#[tokio::test]
async fn the_page_keeps_the_attributes_and_the_scope_of_the_entity() {
    // Arrange
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let page = store
        .upsert_by(
            NewNode::entity("person", "Kim")
                .with_attributes(json!({"role": "engineer"}))
                .with_scope("proj-x"),
        )
        .await
        .unwrap();
    store
        .relate(
            &page,
            &store
                .insert(
                    NewNode::now("fact", "note", "Kim likes tea.")
                        .with_embedding(axis(0))
                        .with_scope("proj-x"),
                )
                .await
                .unwrap(),
            relation::MENTIONS,
        )
        .await
        .unwrap();

    // Act
    server
        .resynthesize_entity_as(&store, page.clone(), soon(), None)
        .await
        .expect("the page should be built");

    // Assert: the old row is superseded, and the new one is still in the scope
    assert!(store.resolve_handle(page.as_str()).await.is_err());
    let written = live_row(&store, "Kim", Some("proj-x"))
        .await
        .expect("page in scope");
    assert_eq!(
        written.hit.attributes,
        crate::entity_page::stamped(json!({"role": "engineer"}))
    );
    assert!(live_row(&store, "Kim", Some("proj-y")).await.is_none());
}

#[tokio::test]
async fn building_honors_the_deadline_it_is_given_not_a_fresh_one() {
    // Arrange: an embedder that never answers, and a deadline already past
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    let server = server_with_embedder(Arc::new(GatedEmbedder::new(DIMS, gate.clone()))).await;
    let store = server.store().await.unwrap();
    let page = entity(&store, "Kim").await;
    let bare = store
        .insert(NewNode::now("fact", "note", "Kim likes tea."))
        .await
        .unwrap();
    store
        .relate(&page, &bare, relation::MENTIONS)
        .await
        .unwrap();
    let _held = gate.lock().await;
    let already_elapsed = tokio::time::Instant::now() - Duration::from_millis(1);

    // Act
    let outcome = tokio::time::timeout(
        Duration::from_millis(200),
        server.resynthesize_entity_as(&store, page, already_elapsed, None),
    )
    .await
    .expect("must fail fast on the passed deadline, not hang");

    // Assert
    let error = outcome.expect_err("an elapsed deadline cannot embed the mentions");
    assert!(
        error.contains("timed out embedding the mentions"),
        "{error}"
    );
}

#[tokio::test]
async fn an_entity_that_vanished_is_reported_by_its_id() {
    // Arrange
    let server = plain_server().await;
    let store = server.store().await.unwrap();

    // Act
    let outcome = server
        .resynthesize_entity_as(&store, NodeId::from_raw("missing"), soon(), None)
        .await;

    // Assert
    let error = outcome.unwrap_err();
    assert!(error.starts_with("missing: entity vanished"), "{error}");
}

#[tokio::test]
async fn remember_builds_one_page_holding_both_facts_of_a_twice_mentioned_entity() {
    // Arrange / Act: two facts each carry a mentions edge to the same entity
    let server = plain_server().await;
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

    // Assert
    assert!(!out.contains("synthesis failed"), "{out}");
    let store = server.store().await.unwrap();
    let pages: Vec<_> = live_row(&store, "Dedup Entity", None)
        .await
        .into_iter()
        .filter(|h| !h.hit.retrieved.as_text().is_empty())
        .collect();
    assert_eq!(pages.len(), 1, "{out}");
    assert!(pages[0]
        .hit
        .retrieved
        .as_text()
        .starts_with("Dedup Entity (person)\n"));
}

#[tokio::test]
async fn remember_builds_every_freshly_mentioned_entity_before_returning() {
    // Arrange / Act: an episode mentioning two fresh entities
    let server = plain_server().await;
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

    // Assert: both pages exist already, and the original rows are superseded
    assert!(!out.contains("synthesis failed"), "{out}");
    let store = server.store().await.unwrap();
    for (line, label) in [(1, "Both One"), (2, "Both Two")] {
        let original = out
            .lines()
            .nth(line)
            .expect("entity line")
            .trim_start_matches("remembered ");
        assert!(store.resolve_handle(original).await.is_err(), "{out}");
        let page = page_named(&store, label).await;
        assert!(
            page.hit
                .retrieved
                .as_text()
                .contains("both entities top content"),
            "{out}"
        );
    }
}

#[tokio::test]
async fn remember_still_commits_the_episode_when_a_page_cannot_be_built() {
    // Arrange: "Failing Entity" mentions a memory the embedder refuses
    let (server, refused) = server_with_a_refused_mention().await;

    // Act
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Failing Entity")],
                edges: vec![episode_edge("entity:0", refused.as_str(), "mentions")],
            }),
            ..remember_args("synthesis failure top content")
        }))
        .await;

    // Assert: the facts are committed, the entity is live, and the reply names it
    assert!(!out.starts_with("remember failed:"), "{out}");
    let store = server.store().await.unwrap();
    let top = store
        .query_explained(&Query::text("synthesis failure top content"))
        .await
        .unwrap();
    assert!(top
        .iter()
        .any(|h| h.hit.retrieved.as_text() == "synthesis failure top content"));
    let entity_id = out
        .lines()
        .nth(1)
        .expect("entity line")
        .trim_start_matches("remembered ");
    assert!(store.resolve_handle(entity_id).await.is_ok(), "{out}");
    assert!(out.contains("synthesis failed for Failing Entity"), "{out}");
}

#[tokio::test]
async fn remember_records_both_outcomes_when_one_of_two_pages_fails() {
    // Arrange: only "Mixed One" mentions the memory the embedder refuses
    let (server, refused) = server_with_a_refused_mention().await;

    // Act
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![
                    episode_entity("person", "Mixed One"),
                    episode_entity("person", "Mixed Two"),
                ],
                edges: vec![
                    episode_edge("entity:0", refused.as_str(), "mentions"),
                    episode_edge("entity:1", "fact:0", "mentions"),
                ],
            }),
            ..remember_args("mixed outcome top content")
        }))
        .await;

    // Assert
    let store = server.store().await.unwrap();
    assert!(out.contains("synthesis failed for Mixed One"), "{out}");
    assert!(!out.contains("synthesis failed for Mixed Two"), "{out}");
    assert!(page_named(&store, "Mixed Two")
        .await
        .hit
        .retrieved
        .as_text()
        .starts_with("Mixed Two (person)\n"));
}

#[tokio::test]
async fn remember_does_not_rebuild_an_existing_entity_referenced_only_by_handle() {
    // Arrange: an entity from a prior call
    let (server, embedder) = server_with_recording_embedder().await;
    let prior = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![],
                entities: vec![episode_entity("person", "Handle Only Entity")],
                edges: vec![],
            }),
            ..remember_args("handle only entity top content")
        }))
        .await;
    assert!(!prior.contains("failed"), "{prior}");
    let handle = prior
        .lines()
        .nth(1)
        .expect("entity line")
        .trim_start_matches("remembered ")
        .to_string();

    // Act: a second episode mentions it only by handle
    let out = server
        .remember(Parameters(RememberArgs {
            episode: Some(EpisodeArgs {
                facts: vec![episode_fact("handle only mention fact content")],
                entities: vec![],
                edges: vec![episode_edge(&handle, "fact:1", "mentions")],
            }),
            ..remember_args("handle only second top content")
        }))
        .await;

    // Assert: no page text was embedded
    assert!(!out.contains("failed"), "{out}");
    let page_embeds = embedder
        .calls()
        .into_iter()
        .filter(|(text, _)| text.starts_with("Handle Only Entity (person)"))
        .count();
    assert_eq!(page_embeds, 0, "{out}");
}

#[tokio::test]
async fn the_page_of_a_remembered_entity_carries_the_real_content_of_its_mention() {
    // Arrange / Act
    let server = plain_server().await;
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

    // Assert
    assert!(!out.contains("synthesis failed"), "{out}");
    let store = server.store().await.unwrap();
    let page = page_named(&store, "Xyzzy Entity").await;
    assert!(
        page.hit
            .retrieved
            .as_text()
            .contains("the xyzzy-marker-42 distinctive fact content"),
        "{}",
        page.hit.retrieved.as_text()
    );
}

/// Writes a fact in `scope`, linked from the entity `page`.
async fn scoped_mention(
    store: &DefaultGraph,
    page: &NodeId,
    text: &str,
    scope: Option<&str>,
    embedding: Vec<f32>,
) {
    let mut fact = NewNode::now("fact", "note", text).with_embedding(embedding);
    if let Some(scope) = scope {
        fact = fact.with_scope(scope);
    }
    let id = store.insert(fact).await.unwrap();
    store.relate(page, &id, relation::MENTIONS).await.unwrap();
}

#[tokio::test]
async fn a_scoped_entity_page_holds_only_mentions_inside_its_scope() {
    // Arrange: mentions in the scope, in a child scope, in another scope, and in none
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let page = store
        .upsert_by(NewNode::entity("person", "Kim").with_scope("proj-x"))
        .await
        .unwrap();
    for (n, (text, scope)) in [
        ("Kim likes tea.", Some("proj-x")),
        ("Kim runs the build.", Some("proj-x/ci")),
        ("Kim plans a secret move.", Some("proj-y")),
        ("Kim has an unscoped note.", None),
    ]
    .into_iter()
    .enumerate()
    {
        scoped_mention(&store, &page, text, scope, axis(n)).await;
    }

    // Act
    server
        .resynthesize_entity_as(&store, page, soon(), None)
        .await
        .unwrap();

    // Assert
    let written = live_row(&store, "Kim", Some("proj-x"))
        .await
        .filter(|found| found.hit.retrieved.is_inferred())
        .expect("a page in scope");
    let text = written.hit.retrieved.as_text();
    assert!(text.contains("Kim likes tea."), "{text}");
    assert!(text.contains("Kim runs the build."), "{text}");
    assert!(!text.contains("secret move"), "{text}");
    assert!(!text.contains("unscoped note"), "{text}");
}

#[tokio::test]
async fn a_scoped_entity_whose_mentions_are_all_outside_its_scope_gets_no_page() {
    // Arrange
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let page = store
        .upsert_by(NewNode::entity("person", "Kim").with_scope("proj-x"))
        .await
        .unwrap();
    scoped_mention(
        &store,
        &page,
        "Kim plans a secret move.",
        Some("proj-y"),
        axis(0),
    )
    .await;

    // Act
    server
        .resynthesize_entity_as(&store, page.clone(), soon(), None)
        .await
        .unwrap();

    // Assert: the entity row is untouched
    assert!(store.resolve_handle(page.as_str()).await.is_ok());
}

#[tokio::test]
async fn an_unscoped_entity_page_may_hold_mentions_of_every_scope() {
    // Arrange: an unscoped page is only read by a reader who sees every scope
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let page = entity(&store, "Kim").await;
    scoped_mention(&store, &page, "Kim likes tea.", Some("proj-x"), axis(0)).await;
    scoped_mention(&store, &page, "Kim runs the build.", None, axis(1)).await;

    // Act
    server
        .resynthesize_entity_as(&store, page, soon(), None)
        .await
        .unwrap();

    // Assert
    let text = page_named(&store, "Kim")
        .await
        .hit
        .retrieved
        .as_text()
        .to_string();
    assert!(text.contains("Kim likes tea."), "{text}");
    assert!(text.contains("Kim runs the build."), "{text}");
}
