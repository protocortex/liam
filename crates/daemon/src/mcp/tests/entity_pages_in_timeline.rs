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

async fn live_old_pages(store: &DefaultGraph) -> Vec<Found> {
    live_row(store, "Old", None).await.into_iter().collect()
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
    let page = live_row(&store, "Kim", None).await.unwrap().hit.id;
    let out = timeline_of(&server, &page).await;
    assert!(out.contains("Kim (person)\n"), "{out}");
    assert!(out.contains("1970-01-02: Kim lives in Lisbon."), "{out}");
    assert!(
        out.contains("1970-01-03: Kim moved to Porto in May."),
        "{out}"
    );
    assert!(out.contains("derivation: inferred"), "{out}");
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
    for old in &legacy {
        assert!(store.resolve_handle(old.as_str()).await.is_ok());
    }

    // Act
    tick(&server).await;

    // Assert
    let live = live_old_pages(&store).await;
    assert_eq!(live.len(), 1);
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

/// A page the old code wrote for an entity that nothing mentions.
async fn orphan_legacy_page(store: &DefaultGraph, name: &str) -> liam_store::NodeId {
    let mut page = NewNode::entity("person", name);
    page.content = LEGACY_TEXT.to_string();
    store.insert(page).await.unwrap()
}

#[tokio::test]
async fn legacy_pages_with_no_mention_do_not_starve_a_legacy_page_that_has_one() {
    // Arrange: more orphan legacy pages than one tick rebuilds, written first,
    // and one legacy page with a mention, written last
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let mut orphans = Vec::new();
    for n in 0..=TICK_CAP {
        orphans.push(orphan_legacy_page(&store, &format!("Orphan {n}")).await);
    }
    let fact = store
        .insert(NewNode::now("fact", "note", MENTION))
        .await
        .unwrap();
    legacy_page(&store, &fact).await;

    // Act: two ticks are enough, because an orphan is replaced and not listed again
    tick(&server).await;
    tick(&server).await;

    // Assert: the page with a mention was rebuilt, and no orphan keeps model text
    let rebuilt = live_row(&store, "Old", None).await.unwrap();
    assert!(rebuilt
        .hit
        .retrieved
        .as_text()
        .starts_with("Old (person)\n"));
    for orphan in orphans {
        assert!(store.resolve_handle(orphan.as_str()).await.is_err());
    }
    let listed = store
        .pages_without_marker(crate::entity_page::PAGE_MARKER_KEY, TICK_CAP, Millis::now())
        .await
        .unwrap();
    assert!(listed.is_empty(), "{listed:?}");
}

const RECALL_MEMORIES: [(&str, &str); 4] = [
    (
        "Maria Costa",
        "A Maria Costa mudou-se para o Porto em maio.",
    ),
    (
        "Maria Costa",
        "A Maria Costa trabalha na equipa de plataforma.",
    ),
    ("Tomas Becker", "Tomas Becker se mudo a Sevilla en mayo."),
    (
        "Tomas Becker",
        "Tomas Becker trabaja en el equipo de plataforma.",
    ),
];

/// A server holding the memories above, each linked from its entity, and the
/// pages of the entities when `with_pages`.
async fn server_with_memories(with_pages: bool) -> MemoryServer {
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    for name in ["Maria Costa", "Tomas Becker"] {
        let entity = store
            .upsert_by(NewNode::entity("person", name))
            .await
            .unwrap();
        for (owner, text) in RECALL_MEMORIES {
            if owner == name {
                let fact = store
                    .insert(NewNode::now("fact", "note", text))
                    .await
                    .unwrap();
                store
                    .relate(&entity, &fact, relation::MENTIONS)
                    .await
                    .unwrap();
            }
        }
        if with_pages {
            server
                .resynthesize_entity_as(
                    &store,
                    entity,
                    tokio::time::Instant::now() + Duration::from_secs(5),
                    None,
                )
                .await
                .unwrap();
        }
    }
    drop(store);
    server
}

/// The memory text lines of a recall reply, in order.
async fn recalled_lines(server: &MemoryServer, query: &str) -> Vec<String> {
    let text = server
        .recall(Parameters(RecallArgs {
            query: query.to_string(),
            ..recall_args()
        }))
        .await;
    text.split("\n\n")
        .map(|block| block.lines().nth(1).unwrap_or_default().to_string())
        .collect()
}

#[tokio::test]
async fn recall_returns_the_same_memories_in_the_same_order_with_pages_built_and_without() {
    // Arrange: one Portuguese and one Spanish entity, with and without pages
    let without = server_with_memories(false).await;
    let with = server_with_memories(true).await;

    for query in [
        "Maria Costa plataforma maio",
        "Tomas Becker equipo plataforma Sevilla",
    ] {
        // Act
        let plain = recalled_lines(&without, query).await;
        let paged = recalled_lines(&with, query).await;

        // Assert
        assert!(plain.len() > 1, "{plain:?}");
        assert_eq!(paged, plain, "{query}");
    }
}
