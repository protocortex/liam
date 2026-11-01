// SPDX-License-Identifier: Apache-2.0
//! The `derivation` column: which write paths set it, and the backfill that
//! marks the entity pages written before the column existed.

use super::support::{count, Env};
use super::*;
use crate::types::{Derivation, EpisodeEdge, EpisodeRef};
use crate::DefaultBackend;

const DIMS: usize = 8;
const MENTIONS: &str = relation::MENTIONS;

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

/// A store with no log, in a file of its own.
async fn open_unlogged(dir: &TempDir) -> DefaultGraph {
    let path = dir.path().join("derivation.db");
    DefaultGraph::open_with_clock(path.to_str().unwrap(), GraphConfig::new(DIMS), clock())
        .await
        .unwrap()
}

async fn derivation_of<B: Backend>(g: &Graph<B>, id: &NodeId) -> Derivation {
    let rows = g
        .backend
        .query(
            "SELECT derivation FROM nodes WHERE id = ?1",
            &[id.as_str().into()],
        )
        .await
        .unwrap();
    Derivation::parse(&rows[0].get_string(0).unwrap()).unwrap()
}

/// Puts the store back as a database from before the column was filled in:
/// every row recorded and the pass not yet run.
async fn forget_derivations<B: Backend>(g: &Graph<B>) {
    for sql in [
        "UPDATE nodes SET derivation = 'recorded'",
        "DELETE FROM derivation_backfill_state",
    ] {
        g.backend.execute(sql, &[]).await.unwrap();
    }
}

struct Seeded {
    page: NodeId,
    accented_page: NodeId,
    fact: NodeId,
    plain: NodeId,
    empty_page: NodeId,
    user_node: NodeId,
}

/// One node of each kind the backfill must tell apart. A page is an entity
/// node with a `mentions` edge to a fact and a subject equal to its label,
/// lowercased.
async fn seed<B: Backend>(g: &Graph<B>) -> Seeded {
    let fact = g
        .insert(NewNode::now("fact", "note", "met Ada"))
        .await
        .unwrap();
    let plain = g
        .insert(NewNode::now("fact", "plain", "nothing"))
        .await
        .unwrap();
    let mut page = NewNode::entity("person", " Ada ");
    page.content = "synthesized".into();
    let page = g.insert(page).await.unwrap();
    let accented_page = g.insert(NewNode::entity("person", "Émile")).await.unwrap();
    let empty_page = g.insert(NewNode::entity("person", "Nobody")).await.unwrap();
    let user_node = g
        .insert(NewNode::now("fact", "Shout", "text").with_subject("quiet"))
        .await
        .unwrap();
    for src in [&page, &accented_page, &user_node] {
        g.relate(src, &fact, MENTIONS).await.unwrap();
    }
    Seeded {
        page,
        accented_page,
        fact,
        plain,
        empty_page,
        user_node,
    }
}

async fn assert_only_pages_are_inferred<B: Backend>(g: &Graph<B>, s: &Seeded) {
    for page in [&s.page, &s.accented_page] {
        assert_eq!(derivation_of(g, page).await, Derivation::Inferred);
    }
    for kept in [&s.fact, &s.plain, &s.empty_page, &s.user_node] {
        assert_eq!(derivation_of(g, kept).await, Derivation::Recorded);
    }
}

#[tokio::test]
async fn opening_a_database_without_the_column_adds_it_and_marks_only_the_pages() {
    // Arrange: a database as an older build left it, with no derivation column
    let dir = TempDir::new().unwrap();
    let seeded = {
        let g = open_unlogged(&dir).await;
        let seeded = seed(&g).await;
        for sql in [
            "ALTER TABLE nodes DROP COLUMN derivation",
            "DROP TABLE derivation_backfill_state",
        ] {
            g.backend.execute(sql, &[]).await.unwrap();
        }
        seeded
    };

    // Act
    let g = open_unlogged(&dir).await;

    // Assert
    assert_only_pages_are_inferred(&g, &seeded).await;
    assert_eq!(count(&g, "derivation_backfill_state").await, 1);
}

#[tokio::test]
async fn the_backfill_runs_once_and_a_second_open_changes_nothing() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let g = open_unlogged(&dir).await;
    let seeded = seed(&g).await;
    forget_derivations(&g).await;

    // Act
    let first = crate::migrate::backfill_derivation(&g.backend, Millis(2000)).await;
    let second = crate::migrate::backfill_derivation(&g.backend, Millis(3000)).await;
    drop(g);
    let reopened = open_unlogged(&dir).await;

    // Assert
    assert_eq!(first.unwrap(), 2);
    assert_eq!(second.unwrap(), 0);
    assert_only_pages_are_inferred(&reopened, &seeded).await;
    let marks = reopened
        .backend
        .query(
            "SELECT completed_at, flagged FROM derivation_backfill_state",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(marks.len(), 1);
    assert_eq!(
        (marks[0].get_i64(0).unwrap(), marks[0].get_i64(1).unwrap()),
        (2000, 2)
    );
}

#[tokio::test]
async fn a_page_whose_mentions_are_all_closed_is_still_marked() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let g = open_unlogged(&dir).await;
    let seeded = seed(&g).await;
    g.backend
        .execute(
            "UPDATE edges SET tx_to = 5 WHERE type = ?1",
            &[MENTIONS.into()],
        )
        .await
        .unwrap();
    forget_derivations(&g).await;

    // Act
    crate::migrate::backfill_derivation(&g.backend, Millis(2000))
        .await
        .unwrap();

    // Assert
    assert_only_pages_are_inferred(&g, &seeded).await;
}

#[tokio::test]
async fn every_write_path_stores_the_derivation_of_its_constructor() {
    // Arrange: the same paths on a store with no log and one with a log
    let dir = TempDir::new().unwrap();
    let env = Env::new();
    let unlogged = open_unlogged(&dir).await;
    let (logged, _log) = env.open::<DefaultBackend>("logged.db", clock()).await;

    for g in [&unlogged, &logged] {
        // Act
        let inserted = g.insert(NewNode::entity("person", "Insert")).await.unwrap();
        let upserted = g
            .upsert_by(NewNode::entity("person", "Upsert"))
            .await
            .unwrap();
        let mut newer = NewNode::entity("person", "Upsert");
        newer.content = "second version".into();
        let upserted_again = g.upsert_by(newer).await.unwrap();
        let recorded = g
            .insert(NewNode::now("fact", "plain", "text"))
            .await
            .unwrap();
        let superseding = g
            .supersede(&recorded, NewNode::entity("person", "Supersede"))
            .await
            .unwrap();
        let episode = g
            .ingest_episode(
                vec![
                    NewNode::now("fact", "episode fact", "text"),
                    NewNode::entity("person", "Episode"),
                ],
                vec![EpisodeEdge {
                    from: EpisodeRef::New(1),
                    to: EpisodeRef::New(0),
                    kind: MENTIONS.into(),
                    attributes: serde_json::json!({}),
                }],
            )
            .await
            .unwrap();

        // Assert
        for page in [
            &inserted,
            &upserted,
            &upserted_again,
            &superseding,
            &episode.node_ids[1],
        ] {
            assert_eq!(derivation_of(g, page).await, Derivation::Inferred);
        }
        for note in [&recorded, &episode.node_ids[0]] {
            assert_eq!(derivation_of(g, note).await, Derivation::Recorded);
        }
        assert_ne!(upserted, upserted_again, "the second upsert supersedes");
    }
}

#[tokio::test]
async fn a_rebuild_marks_the_replayed_pages_inferred_again() {
    // Arrange: a logged store, whose log does not carry the derivation
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let seeded = seed(&g).await;

    // Act
    let (rebuilt, _) = g
        .rebuild_from_log(Arc::clone(&log), RebuildMode::ResetChecked)
        .await
        .unwrap();

    // Assert
    assert_only_pages_are_inferred(&rebuilt, &seeded).await;
    assert_eq!(count(&rebuilt, "derivation_backfill_state").await, 1);
}
