// SPDX-License-Identifier: Apache-2.0
//! The entity page `timeline` shows: built by the maintenance tick from the
//! sentences recorded about the entity, never from text a model wrote.

use super::*;

const LEGACY_TEXT: &str = "Old is a person who invented a claim about tea.";
const TICK_CAP: usize = 5;
const TICK_TIMEOUT_SECS: u64 = 30;
const MENTION: &str = "Old likes tea in the morning";

fn axis(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 8];
    vector[axis] = 1.0;
    vector
}

async fn tick(server: &MemoryServer) {
    let policy = liam_store::RetentionPolicy::keep("nonexistent-kind", Millis(1));
    crate::maintenance_tick(server, &policy, TICK_CAP, TICK_TIMEOUT_SECS, false).await;
}

async fn timeline_of(server: &MemoryServer, entity: &liam_store::NodeId) -> String {
    server
        .timeline(Parameters(TimelineArgs {
            entity: entity.handle().to_string(),
            limit: None,
            mode: None,
        }))
        .await
}

/// A page written the way the deleted model-based code wrote one: an entity row
/// holding text and the source events, with no builder marker among its
/// attributes. Each page is linked to the mention `fact`.
async fn legacy_page(store: &DefaultGraph, fact: &liam_store::NodeId) -> liam_store::NodeId {
    let mut page = NewNode::entity("person", "Old").with_basis(
        store
            .node_event_ids(fact, liam_store::EventRole::Origin)
            .await
            .unwrap(),
    );
    page.content = LEGACY_TEXT.to_string();
    let page = store.insert(page).await.unwrap();
    store.relate(&page, fact, relation::MENTIONS).await.unwrap();
    page
}

async fn live_old_pages(store: &DefaultGraph) -> Vec<liam_store::ExplainedHit> {
    store
        .query_explained(&Query::text("Old").with_inferred(true).with_k(10))
        .await
        .unwrap()
        .into_iter()
        .filter(|hit| hit.hit.label == "Old")
        .collect()
}

#[tokio::test]
async fn timeline_shows_the_page_the_tick_builds_from_the_recorded_mentions() {
    // Arrange: an entity mentioned by several memories, as `remember` leaves it
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let entity = store
        .upsert_by(NewNode::entity("person", "Kim"))
        .await
        .unwrap();
    for (day, text) in [
        (1, "Kim lives in Lisbon."),
        (2, "Kim moved to Porto in May."),
    ] {
        let fact = store
            .insert(
                NewNode::now("fact", "Home", text)
                    .with_valid_from(Millis(day * 86_400_000))
                    .with_embedding(axis(usize::try_from(day).unwrap())),
            )
            .await
            .unwrap();
        store
            .relate(&entity, &fact, relation::MENTIONS)
            .await
            .unwrap();
    }

    // Act
    tick(&server).await;

    // Assert: the page, found by label, heads the timeline with dated lines
    let page = live_old_pages_named(&store, "Kim").await;
    let out = timeline_of(&server, &page).await;
    assert!(out.contains("Kim (person)\n"), "{out}");
    assert!(out.contains("1970-01-02: Kim lives in Lisbon."), "{out}");
    assert!(
        out.contains("1970-01-03: Kim moved to Porto in May."),
        "{out}"
    );
    assert!(out.contains("derivation: inferred"), "{out}");
}

async fn live_old_pages_named(store: &DefaultGraph, name: &str) -> liam_store::NodeId {
    store
        .query_explained(&Query::text(name).with_inferred(true))
        .await
        .unwrap()
        .into_iter()
        .find(|hit| hit.hit.label == name)
        .expect("a live row for the entity")
        .hit
        .id
}

#[tokio::test]
async fn a_page_an_earlier_version_wrote_is_not_shown_then_rebuilt_and_its_row_closed() {
    // Arrange: a legacy page, and the fact it mentions
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let fact = store
        .insert(NewNode::now("fact", "note", MENTION))
        .await
        .unwrap();
    let legacy = legacy_page(&store, &fact).await;

    // Act
    let before = timeline_of(&server, &legacy).await;
    tick(&server).await;

    // Assert: hidden before, then a new page replaces it and the old row is closed
    assert!(!before.contains(LEGACY_TEXT), "{before}");
    assert!(before.contains(MENTION), "{before}");
    assert!(store.resolve_handle(legacy.as_str()).await.is_err());
    let live = live_old_pages(&store).await;
    assert_eq!(live.len(), 1);
    assert!(live[0]
        .hit
        .retrieved
        .as_text()
        .starts_with("Old (person)\n"));
    let after = timeline_of(&server, &live[0].hit.id).await;
    assert!(after.contains("Old (person)\n"), "{after}");
    assert!(!after.contains(LEGACY_TEXT), "{after}");
}

#[tokio::test]
async fn three_live_rows_of_one_entity_become_one_page_on_the_next_tick() {
    // Arrange: three legacy pages of one entity, as the old collision filter left
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let fact = store
        .insert(NewNode::now("fact", "note", MENTION))
        .await
        .unwrap();
    let mut legacy = Vec::new();
    for _ in 0..3 {
        legacy.push(legacy_page(&store, &fact).await);
    }
    assert_eq!(live_old_pages(&store).await.len(), 3);

    // Act
    tick(&server).await;

    // Assert
    let live = live_old_pages(&store).await;
    assert_eq!(live.len(), 1, "{live:?}");
    assert!(crate::entity_page::is_stamped(&live[0].hit.attributes));
    for old in legacy {
        assert!(store.resolve_handle(old.as_str()).await.is_err());
    }
}

#[tokio::test]
async fn a_page_the_builder_wrote_is_not_rebuilt_by_the_next_tick() {
    // Arrange: a built page
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let fact = store
        .insert(NewNode::now("fact", "note", MENTION))
        .await
        .unwrap();
    let legacy = legacy_page(&store, &fact).await;
    tick(&server).await;
    let built = live_old_pages(&store).await.remove(0).hit.id;

    // Act
    tick(&server).await;

    // Assert
    assert!(store.resolve_handle(legacy.as_str()).await.is_err());
    assert_eq!(live_old_pages(&store).await[0].hit.id, built);
}
