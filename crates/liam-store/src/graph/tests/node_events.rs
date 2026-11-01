// SPDX-License-Identifier: Apache-2.0
//! The log events listed beside each node: the origin every write adds in its
//! own transaction, the basis a composed node carries, and what a rebuild, a
//! backfill, and a sweep do to them.

use liam_log::event::{LogPayload, RowEffect};

use super::support::{count, fact, Env};
use super::*;
use crate::types::{EpisodeEdge, EpisodeRef, EventRef, EventRole};
use crate::DefaultBackend;

const DIMS: usize = 8;
const MENTIONS: &str = relation::MENTIONS;

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

async fn origins<B: Backend>(g: &Graph<B>, id: &NodeId) -> Vec<EventRef> {
    g.node_event_ids(id, EventRole::Origin).await.unwrap()
}

fn real(event_id: &str) -> EventRef {
    EventRef {
        event_id: event_id.to_string(),
        synthetic: false,
    }
}

/// The ids of the log's events that wrote nodes, in log order.
async fn node_event_ids_in_log(env: &Env) -> Vec<String> {
    env.records()
        .await
        .into_iter()
        .filter(|record| match &record.event.payload {
            LogPayload::NodeWrite(_) => true,
            LogPayload::EpisodeBatch(effects) => effects
                .iter()
                .any(|effect| matches!(effect, RowEffect::Node(_))),
            _ => false,
        })
        .map(|record| record.event.event_id)
        .collect()
}

#[tokio::test]
async fn a_logged_write_lists_the_log_event_that_wrote_the_node() {
    // Arrange
    let env = Env::new();
    let (g, _log) = env.open::<DefaultBackend>("origin.db", clock()).await;

    // Act
    let id = g.insert(fact("alpha")).await.unwrap();

    // Assert
    let logged = node_event_ids_in_log(&env).await;
    assert_eq!(origins(&g, &id).await, [real(&logged[0])]);
}

#[tokio::test]
async fn a_write_with_no_log_lists_a_synthetic_origin() {
    // Arrange
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
        .await
        .unwrap();

    // Act
    let id = g.insert(fact("alpha")).await.unwrap();

    // Assert
    let listed = origins(&g, &id).await;
    assert_eq!(listed.len(), 1);
    assert!(listed[0].synthetic, "{listed:?}");
}

#[tokio::test]
async fn every_row_of_an_episode_lists_the_batch_event() {
    // Arrange
    let env = Env::new();
    let (g, _log) = env.open::<DefaultBackend>("episode.db", clock()).await;
    let edge = EpisodeEdge {
        from: EpisodeRef::New(0),
        to: EpisodeRef::New(1),
        kind: MENTIONS.into(),
        attributes: serde_json::json!({}),
    };

    // Act
    let episode = g
        .ingest_episode(vec![fact("one"), fact("two")], vec![edge])
        .await
        .unwrap();

    // Assert
    let batch = node_event_ids_in_log(&env).await.remove(0);
    for id in &episode.node_ids {
        assert_eq!(origins(&g, id).await, [real(&batch)]);
    }
}

#[tokio::test]
async fn a_version_that_replaces_a_node_lists_its_own_event_and_leaves_the_old_one() {
    // Arrange
    let env = Env::new();
    let (g, _log) = env.open::<DefaultBackend>("versions.db", clock()).await;
    let first = g
        .upsert_by(fact("v1").with_subject("subject"))
        .await
        .unwrap();

    // Act
    let second = g
        .upsert_by(fact("v2").with_subject("subject"))
        .await
        .unwrap();

    // Assert
    let logged = node_event_ids_in_log(&env).await;
    assert_eq!(logged.len(), 2);
    assert_eq!(origins(&g, &first).await, [real(&logged[0])]);
    assert_eq!(origins(&g, &second).await, [real(&logged[1])]);
}

#[tokio::test]
async fn a_node_written_with_a_basis_lists_it_apart_from_its_origin() {
    // Arrange
    let env = Env::new();
    let (g, _log) = env.open::<DefaultBackend>("basis.db", clock()).await;
    let mention = g.insert(fact("mention")).await.unwrap();
    let mut basis = origins(&g, &mention).await;
    basis.push(EventRef {
        event_id: "placeholder".to_string(),
        synthetic: true,
    });
    let mut page = NewNode::entity("person", "Ada").with_basis(basis.clone());
    page.content = "profile".into();

    // Act
    let page = g.upsert_by(page).await.unwrap();

    // Assert
    let listed = g.node_event_ids(&page, EventRole::Basis).await.unwrap();
    assert_eq!(listed, [real(&basis[0].event_id), basis[1].clone()]);
    let page_origin = origins(&g, &page).await;
    assert_eq!(page_origin.len(), 1);
    assert!(!basis.contains(&page_origin[0]));
}

#[tokio::test]
async fn a_basis_event_keeps_its_flag_when_the_write_has_no_log() {
    // Arrange
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
        .await
        .unwrap();
    let basis = vec![real("backed-by-a-log")];

    // Act
    let page = g
        .insert(NewNode::entity("person", "Ada").with_basis(basis.clone()))
        .await
        .unwrap();

    // Assert
    let listed = g.node_event_ids(&page, EventRole::Basis).await.unwrap();
    assert_eq!(listed, basis);
}

#[tokio::test]
async fn a_legacy_node_with_no_listed_events_reads_as_empty() {
    // Arrange
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
        .await
        .unwrap();
    let id = g.insert(fact("alpha")).await.unwrap();
    g.backend
        .execute("DELETE FROM node_events", &[])
        .await
        .unwrap();

    // Act
    let listed = g.node_event_ids(&id, EventRole::Origin).await.unwrap();

    // Assert
    assert!(listed.is_empty());
}

#[tokio::test]
async fn sweeping_a_node_removes_the_events_listed_for_it() {
    // Arrange
    let env = Env::new();
    let now = clock();
    let (g, _log) = env.open::<DefaultBackend>("gc.db", Arc::clone(&now)).await;
    let doomed = g.insert(fact("old")).await.unwrap();
    g.upsert_by(NewNode::entity("person", "Ada").with_basis(origins(&g, &doomed).await))
        .await
        .unwrap();
    assert_eq!(count(&g, "node_events").await, 3);
    now.set(Millis(50_000));

    // Act
    g.gc(&RetentionPolicy::keep("fact", Millis(1)).without_reclaim())
        .await
        .unwrap();

    // Assert: the page and its listed events stay, the swept fact's go
    assert_eq!(count(&g, "node_events").await, 2);
    assert!(origins(&g, &doomed).await.is_empty());
}

#[tokio::test]
async fn a_rebuild_lists_the_same_origins_again() {
    // Arrange
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let one = g.insert(fact("one")).await.unwrap();
    let two = g
        .ingest_episode(vec![fact("two")], vec![])
        .await
        .unwrap()
        .node_ids
        .remove(0);
    let before = (origins(&g, &one).await, origins(&g, &two).await);

    // Act
    let (rebuilt, _) = g
        .rebuild_from_log(log, RebuildMode::ResetChecked)
        .await
        .unwrap();

    // Assert
    assert_eq!(
        (origins(&rebuilt, &one).await, origins(&rebuilt, &two).await),
        before
    );
}

#[tokio::test]
async fn a_row_the_backfill_logs_lists_a_synthetic_origin() {
    // Arrange: a node written before the store had a log
    let env = Env::new();
    let before_log =
        DefaultGraph::open_with_clock(&env.db("old.db"), GraphConfig::new(DIMS), clock())
            .await
            .unwrap();
    let id = before_log.insert(fact("predates")).await.unwrap();
    drop(before_log);
    let (g, _log) = env.open::<DefaultBackend>("old.db", clock()).await;

    // Act
    g.backfill_log_from_projection().await.unwrap();

    // Assert: beside the placeholder the unlogged write left
    let logged = node_event_ids_in_log(&env).await;
    let listed = origins(&g, &id).await;
    assert!(listed.contains(&EventRef {
        event_id: logged[0].clone(),
        synthetic: true,
    }));
}

/// A write path and what it changes, for the failure sweep below.
#[derive(Clone, Copy, Debug)]
enum Site {
    Insert,
    UpsertInsert,
    UpsertSupersede,
    Supersede,
    Episode,
}

const SITES: [Site; 5] = [
    Site::Insert,
    Site::UpsertInsert,
    Site::UpsertSupersede,
    Site::Supersede,
    Site::Episode,
];

async fn write(g: &Graph<FailingBackend>, site: Site, old: &NodeId) -> Result<()> {
    match site {
        Site::Insert => g.insert(fact("new")).await.map(drop),
        Site::UpsertInsert => g
            .upsert_by(fact("fresh").with_subject("other"))
            .await
            .map(drop),
        Site::UpsertSupersede => g
            .upsert_by(fact("newer").with_subject("subject"))
            .await
            .map(drop),
        Site::Supersede => g.supersede(old, fact("replacement")).await.map(drop),
        Site::Episode => g
            .ingest_episode(vec![fact("one"), fact("two")], vec![])
            .await
            .map(drop),
    }
}

/// Fails the write at each statement in turn until it goes through. A failed
/// attempt must leave neither a node nor a listed event behind.
async fn assert_no_half_write(g: &Graph<FailingBackend>, site: Site) {
    let old = g.insert(fact("old").with_subject("subject")).await.unwrap();
    let nodes = count(g, "nodes").await;
    let listed = count(g, "node_events").await;
    for statement in 0..40 {
        g.backend.set_fail_on_execute(statement);
        if write(g, site, &old).await.is_ok() {
            assert_eq!(
                count(g, "node_events").await - listed,
                count(g, "nodes").await - nodes,
                "{site:?}: one origin per node written"
            );
            assert!(statement > 1, "{site:?}: failed only {statement} times");
            return;
        }
        assert_eq!(count(g, "nodes").await, nodes, "{site:?}, {statement}");
        assert_eq!(
            count(g, "node_events").await,
            listed,
            "{site:?}, {statement}"
        );
    }
    panic!("{site:?} still fails after 40 statements");
}

#[tokio::test]
async fn a_write_that_fails_part_way_lists_no_event_for_a_node_it_did_not_keep() {
    for site in SITES {
        // Arrange: a store with no log, and one with a log
        let unlogged: Graph<FailingBackend> =
            Graph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
                .await
                .unwrap();
        let env = Env::new();
        let (logged, _log) = env.open::<FailingBackend>("faults.db", clock()).await;

        // Act and Assert
        assert_no_half_write(&unlogged, site).await;
        assert_no_half_write(&logged, site).await;
    }
}
