// SPDX-License-Identifier: Apache-2.0
//! Retrieval returns typed text, and a strict query keeps inferred rows out of
//! the ranking pool, graph expansion, and cluster groups.

use super::support::fact;
use super::*;
use crate::types::{Basis, EventRef, EventRole, Retrieved};

const DIMS: usize = 8;
const MENTIONS: &str = relation::MENTIONS;

async fn open() -> DefaultGraph {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock)
        .await
        .unwrap()
}

/// An entity page whose text carries `words`, so a text query finds it.
fn page(name: &str, words: &str) -> NewNode {
    let mut page = NewNode::entity("person", name);
    page.content = words.to_string();
    page
}

fn texts(hits: &[ExplainedHit]) -> Vec<&str> {
    let mut texts: Vec<_> = hits.iter().map(|h| h.hit.retrieved.as_text()).collect();
    texts.sort_unstable();
    texts
}

#[tokio::test]
async fn a_strict_query_returns_the_fact_and_an_inferring_one_returns_the_page_as_inferred() {
    // Arrange
    let g = open().await;
    g.insert(fact("zorbnax met the team")).await.unwrap();
    g.insert(page("Zorbnax", "zorbnax profile")).await.unwrap();
    let query = Query::text("zorbnax");

    // Act
    let strict = g.query_explained(&query).await.unwrap();
    let inferring = g.query_explained(&query.with_inferred(true)).await.unwrap();

    // Assert
    assert_eq!(texts(&strict), ["zorbnax met the team"]);
    assert!(matches!(strict[0].hit.retrieved, Retrieved::Recorded(_)));
    assert_eq!(
        texts(&inferring),
        ["zorbnax met the team", "zorbnax profile"]
    );
    let inferred = inferring
        .iter()
        .filter(|h| h.hit.retrieved.is_inferred())
        .count();
    assert_eq!(inferred, 1);
}

#[tokio::test]
async fn a_strict_vector_query_leaves_out_the_page_that_is_nearest() {
    // Arrange
    let g = open().await;
    let near = vec![1.0; DIMS];
    g.insert(page("Near", "profile").with_embedding(near.clone()))
        .await
        .unwrap();
    g.insert(fact("far away").with_embedding(vec![-1.0; DIMS]))
        .await
        .unwrap();

    // Act
    let strict = g.query_explained(&Query::vector(near)).await.unwrap();

    // Assert
    assert_eq!(texts(&strict), ["far away"]);
}

#[tokio::test]
async fn a_page_that_matches_the_query_does_not_pull_in_its_mentions_when_strict() {
    // Arrange: only the page matches, and it mentions a fact that does not
    let g = open().await;
    let page = g.insert(page("Zorbnax", "zorbnax profile")).await.unwrap();
    let unrelated = g.insert(fact("quiet unrelated note")).await.unwrap();
    g.relate(&page, &unrelated, MENTIONS).await.unwrap();
    let query = Query::text("zorbnax");

    // Act
    let strict = g.query_explained(&query).await.unwrap();
    let inferring = g.query_explained(&query.with_inferred(true)).await.unwrap();

    // Assert
    assert!(strict.is_empty(), "{:?}", texts(&strict));
    assert_eq!(
        texts(&inferring),
        ["quiet unrelated note", "zorbnax profile"]
    );
}

#[tokio::test]
async fn page_neighbours_do_not_take_ranked_slots_from_recorded_rows_when_strict() {
    // Arrange: one match whose neighbours are five pages, written first so they
    // win a tie on score, and three recorded facts
    let g = open().await;
    let seed_text = fact("zorbnax seed");
    let mut pages = Vec::new();
    for i in 0..5 {
        pages.push(
            g.insert(page(&format!("Page {i}"), "profile"))
                .await
                .unwrap(),
        );
    }
    let seed = g.insert(seed_text).await.unwrap();
    for page in &pages {
        g.relate(page, &seed, MENTIONS).await.unwrap();
    }
    for text in ["linked one", "linked two", "linked three"] {
        let linked = g.insert(fact(text)).await.unwrap();
        g.relate(&seed, &linked, "cites").await.unwrap();
    }
    let query = Query::text("zorbnax").with_k(3);

    // Act
    let strict = g.query_explained(&query).await.unwrap();
    let inferring = g.query_explained(&query.with_inferred(true)).await.unwrap();

    // Assert
    assert_eq!(strict.len(), 3);
    assert!(strict.iter().all(|h| !h.hit.retrieved.is_inferred()));
    assert!(inferring.iter().any(|h| h.hit.retrieved.is_inferred()));
}

#[tokio::test]
async fn an_inferred_hit_reads_its_basis_or_unknown_when_none_is_listed() {
    // Arrange
    let g = open().await;
    let mention = g.insert(fact("zorbnax note")).await.unwrap();
    let basis = g.node_event_ids(&mention, EventRole::Origin).await.unwrap();
    g.insert(page("Zorbnax", "zorbnax with basis").with_basis(basis.clone()))
        .await
        .unwrap();
    g.insert(page("Legacy", "zorbnax without basis"))
        .await
        .unwrap();

    // Act
    let hits = g
        .query_explained(&Query::text("zorbnax").with_inferred(true))
        .await
        .unwrap();

    // Assert
    let basis_of = |text: &str| {
        hits.iter()
            .find_map(|h| match &h.hit.retrieved {
                Retrieved::Inferred(page) if page.text.as_str() == text => Some(page.basis.clone()),
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(basis_of("zorbnax with basis"), Basis::Known(basis));
    assert_eq!(basis_of("zorbnax without basis"), Basis::Unknown);
}

#[tokio::test]
async fn a_recorded_hit_carries_the_events_that_wrote_it() {
    // Arrange
    let g = open().await;
    let id = g.insert(fact("zorbnax note")).await.unwrap();
    let origin: Vec<EventRef> = g.node_event_ids(&id, EventRole::Origin).await.unwrap();

    // Act
    let hits = g.query_explained(&Query::text("zorbnax")).await.unwrap();

    // Assert
    match &hits[0].hit.retrieved {
        Retrieved::Recorded(recorded) => assert_eq!(recorded.events, origin),
        other => panic!("expected recorded text, got {other:?}"),
    }
}

#[tokio::test]
async fn cluster_groups_leave_out_a_page_when_strict_and_keep_the_stored_assignment() {
    // Arrange: a page mentioning two linked facts, so all three share a community
    let g = open().await;
    let page = g.insert(page("Zorbnax", "profile")).await.unwrap();
    let one = g.insert(fact("one")).await.unwrap();
    let two = g.insert(fact("two")).await.unwrap();
    g.relate(&page, &one, MENTIONS).await.unwrap();
    g.relate(&page, &two, MENTIONS).await.unwrap();
    g.relate(&one, &two, "cites").await.unwrap();
    let before = g.communities().await.unwrap();

    // Act
    let all = g.community_groups(true).await.unwrap();
    let strict = g.community_groups(false).await.unwrap();

    // Assert
    let members = |groups: &[Vec<ClusterMember>]| -> usize { groups.iter().map(Vec::len).sum() };
    assert_eq!(members(&all), 3);
    assert_eq!(members(&strict), 2);
    assert!(strict.iter().flatten().all(|m| m.id != page));
    assert_eq!(g.communities().await.unwrap(), before);
}
