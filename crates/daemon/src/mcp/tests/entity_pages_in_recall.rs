// SPDX-License-Identifier: Apache-2.0
//! The pages a `recall` adds to the recorded memories it returns, and the
//! verdict a client gives on one.

use super::*;
use crate::recall::MAX_PAGES_IN_RECALL;

const K: usize = 4;
const QUERY: &str = "likes tea in the morning";
const PAGE_MARK: &str = "kind: entity_page";
const ENTITIES: [&str; 3] = ["Kim", "Zed", "Ann"];
const FACTS_PER_ENTITY: usize = 3;
const VERDICTS_PER_HOUR: u32 = 10;
const TRACKED: usize = 100;

fn soon() -> tokio::time::Instant {
    tokio::time::Instant::now() + Duration::from_secs(5)
}

/// Three entities with three matching facts each, and their pages when asked.
/// The facts are written in the same order whether or not pages are built.
async fn populated(build_pages: bool) -> MemoryServer {
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    for name in ENTITIES {
        let entity = store
            .upsert_by(NewNode::entity("person", name))
            .await
            .unwrap();
        for n in 0..FACTS_PER_ENTITY {
            let text = format!("{name} likes tea in the morning number {n}");
            let fact = store
                .insert(NewNode::now("fact", "note", text))
                .await
                .unwrap();
            store
                .relate(&entity, &fact, relation::MENTIONS)
                .await
                .unwrap();
        }
        if build_pages {
            server
                .resynthesize_entity_as(&store, entity, soon(), None)
                .await
                .unwrap();
        }
    }
    drop(store);
    server
}

async fn recalled(server: &MemoryServer, k: usize) -> Vec<String> {
    let reply = server
        .recall_with(
            RecallArgs {
                query: QUERY.to_string(),
                k: Some(k),
                ..recall_args()
            },
            CancellationToken::new(),
        )
        .await;
    reply.into_call_result().content[0]
        .as_text()
        .unwrap()
        .text
        .split("\n\n")
        .map(str::to_string)
        .collect()
}

fn is_page(block: &str) -> bool {
    block.contains(PAGE_MARK)
}

/// The text line of each recorded block: the handle and event ids differ per
/// store.
fn recorded_texts(blocks: &[String]) -> Vec<String> {
    blocks
        .iter()
        .filter(|block| !is_page(block))
        .map(|block| block.lines().nth(1).unwrap_or_default().to_string())
        .collect()
}

#[tokio::test]
async fn the_recorded_memories_are_the_same_in_order_with_pages_and_without() {
    // Arrange: two stores holding the same facts, one with pages built
    let without = populated(false).await;
    let with = populated(true).await;

    // Act
    let plain = recalled(&without, K).await;
    let paged = recalled(&with, K).await;

    // Assert
    assert_eq!(recorded_texts(&paged), recorded_texts(&plain));
    assert_eq!(recorded_texts(&plain).len(), K);
    assert!(paged.iter().any(|block| is_page(block)), "{paged:?}");
}

#[tokio::test]
async fn a_reply_holds_k_memories_plus_at_most_the_page_cap() {
    // Arrange: three pages match, one more than the cap
    let server = populated(true).await;

    // Act
    let blocks = recalled(&server, K).await;

    // Assert
    let pages = blocks.iter().filter(|block| is_page(block)).count();
    assert_eq!(ENTITIES.len(), MAX_PAGES_IN_RECALL + 1);
    assert_eq!(pages, MAX_PAGES_IN_RECALL);
    assert_eq!(blocks.len(), K + MAX_PAGES_IN_RECALL);
}

#[tokio::test]
async fn a_page_that_outranks_a_memory_still_leaves_all_k_memories() {
    // Arrange
    let server = populated(true).await;

    // Act
    let blocks = recalled(&server, K).await;

    // Assert: a page comes before the last memory, and every memory slot is kept
    let first_page = blocks.iter().position(|block| is_page(block)).unwrap();
    let last_memory = blocks.iter().rposition(|block| !is_page(block)).unwrap();
    assert!(first_page < last_memory, "{blocks:?}");
    assert_eq!(recorded_texts(&blocks).len(), K);
}

#[tokio::test]
async fn feedback_with_a_page_handle_is_recorded_like_one_for_a_memory() {
    // Arrange
    let server = populated(true)
        .await
        .with_feedback(VERDICTS_PER_HOUR, TRACKED);
    let blocks = recalled(&server, K).await;
    let page = blocks.iter().find(|block| is_page(block)).unwrap();
    let handle = page
        .split([' ', ']'])
        .nth(1)
        .expect("a handle in the bracket")
        .to_string();

    // Act
    let out = server
        .feedback(Parameters(super::super::feedback::FeedbackArgs {
            result_id: handle,
            correct: true,
        }))
        .await;

    // Assert
    assert_eq!(out, "feedback recorded");
}

const LEGACY_TEXT: &str = "Old is a person who invented a claim about tea.";
const TICK_CAP: usize = 5;
const TICK_TIMEOUT_SECS: u64 = 30;

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

async fn tick(server: &MemoryServer) {
    let policy = liam_store::RetentionPolicy::keep("nonexistent-kind", Millis(1));
    crate::maintenance_tick(server, &policy, TICK_CAP, TICK_TIMEOUT_SECS, false).await;
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
async fn a_page_an_earlier_version_wrote_is_not_recalled_then_rebuilt_and_its_row_closed() {
    // Arrange: a legacy page that matches the query, and the fact it mentions
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let fact = store
        .insert(NewNode::now("fact", "note", "Old likes tea in the morning"))
        .await
        .unwrap();
    let legacy = legacy_page(&store, &fact).await;
    let query = || RecallArgs {
        query: "Old likes tea invented claim".to_string(),
        ..recall_args()
    };

    // Act
    let before = server.recall(Parameters(query())).await;
    tick(&server).await;
    let after = server.recall(Parameters(query())).await;

    // Assert: hidden before, the memory still found, then a new page replaces it
    assert!(!before.contains(LEGACY_TEXT), "{before}");
    assert!(before.contains("Old likes tea in the morning"), "{before}");
    assert!(store.resolve_handle(legacy.as_str()).await.is_err());
    let live = live_old_pages(&store).await;
    assert_eq!(live.len(), 1);
    assert!(live[0]
        .hit
        .retrieved
        .as_text()
        .starts_with("Old (person)\n"));
    assert!(!after.contains(LEGACY_TEXT), "{after}");
    assert!(after.contains(PAGE_MARK), "{after}");
}

#[tokio::test]
async fn three_live_rows_of_one_entity_become_one_page_on_the_next_tick() {
    // Arrange: three legacy pages of one entity, as the old collision filter left
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let fact = store
        .insert(NewNode::now("fact", "note", "Old likes tea in the morning"))
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
        .insert(NewNode::now("fact", "note", "Old likes tea in the morning"))
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
