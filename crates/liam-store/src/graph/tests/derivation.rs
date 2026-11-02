// SPDX-License-Identifier: Apache-2.0
//! The `derivation` column: which write paths set it, and the backfill that
//! marks the entity pages written before the column existed.

use liam_log::event::{EdgeRow, LogEvent, LogPayload};
use liam_log::hash::{edge_row_hash, node_row_hash};

use super::support::{count, edge, event, node, Env, ONE_SEGMENT};
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

/// What the live store held right after `seed`: the pages and the stub
/// synthesized from `NewNode::entity` are inferred, everything else recorded.
async fn assert_derivations_as_written<B: Backend>(g: &Graph<B>, s: &Seeded) {
    for inferred in [&s.page, &s.accented_page, &s.empty_page] {
        assert_eq!(derivation_of(g, inferred).await, Derivation::Inferred);
    }
    for recorded in [&s.fact, &s.plain, &s.user_node] {
        assert_eq!(derivation_of(g, recorded).await, Derivation::Recorded);
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
async fn a_rebuild_restores_every_derivation_from_the_log() {
    // Arrange: a logged store, with a stub that has no mentions edge to mark it by
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let seeded = seed(&g).await;
    assert_derivations_as_written(&g, &seeded).await;

    // Act
    let (rebuilt, report) = g
        .rebuild_from_log(Arc::clone(&log), RebuildMode::ResetChecked)
        .await
        .unwrap();

    // Assert
    assert_derivations_as_written(&rebuilt, &seeded).await;
    assert_eq!(report.replayed.pages_marked, 0);
    assert_eq!(count(&rebuilt, "derivation_backfill_state").await, 1);
}

#[tokio::test]
async fn a_rebuild_keeps_a_recorded_node_shaped_like_a_page_recorded() {
    // Arrange: a user's own node with a page's subject and a mentions edge
    let env = Env::new();
    let (g, log) = env.open::<DefaultBackend>("live.db", clock()).await;
    let fact = g
        .insert(NewNode::now("fact", "note", "met Ada"))
        .await
        .unwrap();
    let lookalike = g
        .insert(NewNode::now("person", "Ada", "my own note").with_subject("ada"))
        .await
        .unwrap();
    g.relate(&lookalike, &fact, MENTIONS).await.unwrap();

    // Act
    let (rebuilt, _) = g
        .rebuild_from_log(Arc::clone(&log), RebuildMode::ResetChecked)
        .await
        .unwrap();

    // Assert
    assert_eq!(
        derivation_of(&rebuilt, &lookalike).await,
        Derivation::Recorded
    );
}

/// A store attached to the log in `env`, opened over a new handle on its file.
async fn reopen_logged(env: &Env, db: &str) -> DefaultGraph {
    let (g, _log) = env
        .open_over::<DefaultBackend>(env.log(ONE_SEGMENT), db, clock())
        .await;
    g
}

#[tokio::test]
async fn pages_replayed_into_an_empty_store_by_catch_up_are_inferred() {
    // Arrange: a log holding pages, and a new database that has never seen it
    let env = Env::new();
    let seeded = {
        let (writer, _log) = env.open::<DefaultBackend>("writer.db", clock()).await;
        seed(&writer).await
    };
    let fresh = reopen_logged(&env, "fresh.db").await;

    // Act
    let report = fresh.catch_up().await.unwrap();

    // Assert
    assert!(report.applied > 0);
    assert_derivations_as_written(&fresh, &seeded).await;
    assert_eq!(report.pages_marked, 0);
}

#[tokio::test]
async fn pages_replayed_into_a_store_whose_backfill_is_done_are_inferred() {
    // Arrange: a store that finished its backfill while the log was empty, as a
    // restored snapshot does, then pages that other clients wrote to the log
    let env = Env::new();
    drop(reopen_logged(&env, "snapshot.db").await);
    let seeded = {
        let (writer, _log) = env.open::<DefaultBackend>("writer.db", clock()).await;
        seed(&writer).await
    };
    let restored = reopen_logged(&env, "snapshot.db").await;
    assert_eq!(count(&restored, "derivation_backfill_state").await, 1);

    // Act
    restored.catch_up().await.unwrap();

    // Assert
    assert_derivations_as_written(&restored, &seeded).await;
}

#[tokio::test]
async fn a_finished_backfill_leaves_a_page_reset_to_recorded_alone() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let g = open_unlogged(&dir).await;
    let seeded = seed(&g).await;
    forget_derivations(&g).await;
    crate::migrate::backfill_derivation(&g.backend, Millis(2000))
        .await
        .unwrap();
    g.backend
        .execute(
            "UPDATE nodes SET derivation = 'recorded' WHERE id = ?1",
            &[seeded.page.as_str().into()],
        )
        .await
        .unwrap();

    // Act
    let second = crate::migrate::backfill_derivation(&g.backend, Millis(3000)).await;

    // Assert
    assert_eq!(second.unwrap(), 0);
    assert_eq!(derivation_of(&g, &seeded.page).await, Derivation::Recorded);
}

#[tokio::test]
async fn the_backfill_never_overwrites_a_derivation_other_than_recorded() {
    // Arrange: a page-shaped node an extractor already owns
    let dir = TempDir::new().unwrap();
    let g = open_unlogged(&dir).await;
    let seeded = seed(&g).await;
    forget_derivations(&g).await;
    g.backend
        .execute(
            "UPDATE nodes SET derivation = ?1 WHERE id = ?2",
            &[
                Derivation::Extracted.as_str().into(),
                seeded.page.as_str().into(),
            ],
        )
        .await
        .unwrap();

    // Act
    let flagged = crate::migrate::backfill_derivation(&g.backend, Millis(2000))
        .await
        .unwrap();

    // Assert
    assert_eq!(flagged, 1);
    assert_eq!(derivation_of(&g, &seeded.page).await, Derivation::Extracted);
    assert_eq!(
        derivation_of(&g, &seeded.accented_page).await,
        Derivation::Inferred
    );
}

#[tokio::test]
async fn the_backfill_logs_how_many_page_shaped_nodes_it_left_recorded() {
    // Arrange: one stub with a page's subject and no mentions edge
    let dir = TempDir::new().unwrap();
    let g = open_unlogged(&dir).await;
    seed(&g).await;
    forget_derivations(&g).await;
    super::support::capture_logs();

    // Act
    crate::migrate::backfill_derivation(&g.backend, Millis(2000))
        .await
        .unwrap();

    // Assert
    let logs = super::support::captured_logs();
    assert!(logs.contains("flagged=2"), "{logs}");
    assert!(logs.contains("stubs=1"), "{logs}");
}

fn legacy_node(id: &str, label: &str, subject: Option<&str>) -> LogEvent {
    let mut row = node(id, "text");
    row.label = label.into();
    row.subject = subject.map(String::from);
    event(
        &format!("event-{id}"),
        node_row_hash(&row),
        LogPayload::NodeWrite(row),
    )
}

fn legacy_mentions(id: &str, src: &str, dst: &str) -> LogEvent {
    let mut row: EdgeRow = edge(id, src, dst);
    row.edge_type = MENTIONS.into();
    event(
        &format!("event-{id}"),
        edge_row_hash(&row),
        LogPayload::EdgeWrite(row),
    )
}

#[tokio::test]
async fn nodes_from_events_without_a_derivation_are_classified_once_the_replay_is_in() {
    // Arrange: events as an older build wrote them, with the mentions edges after the nodes
    let env = Env::new();
    env.append(&[
        legacy_node("fact", "note", None),
        legacy_node("page", " Ada ", Some("ada")),
        legacy_node("user", "Shout", Some("quiet")),
        legacy_mentions("m-page", "page", "fact"),
        legacy_mentions("m-user", "user", "fact"),
    ]);
    let fresh = reopen_logged(&env, "fresh.db").await;

    // Act
    let report = fresh.catch_up().await.unwrap();

    // Assert
    let derivation = |id: &str| {
        let id = NodeId::from_raw(id);
        let fresh = &fresh;
        async move { derivation_of(fresh, &id).await }
    };
    assert_eq!(derivation("page").await, Derivation::Inferred);
    assert_eq!(derivation("user").await, Derivation::Recorded);
    assert_eq!(derivation("fact").await, Derivation::Recorded);
    assert_eq!(report.pages_marked, 1);
}

#[tokio::test]
async fn a_second_replay_of_the_same_log_marks_nothing_again() {
    // Arrange
    let env = Env::new();
    env.append(&[
        legacy_node("fact", "note", None),
        legacy_node("page", "Ada", Some("ada")),
        legacy_mentions("m-page", "page", "fact"),
    ]);
    let fresh = reopen_logged(&env, "fresh.db").await;
    fresh.catch_up().await.unwrap();

    // Act
    let again = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(again.pages_marked, 0);
    assert_eq!(
        derivation_of(&fresh, &NodeId::from_raw("page")).await,
        Derivation::Inferred
    );
}

#[tokio::test]
async fn a_logged_derivation_that_names_no_derivation_is_quarantined() {
    // Arrange
    let env = Env::new();
    let mut row = node("odd", "text");
    row.derivation = Some("guessed".into());
    env.append(&[event(
        "event-odd",
        node_row_hash(&row),
        LogPayload::NodeWrite(row),
    )]);
    let fresh = reopen_logged(&env, "fresh.db").await;

    // Act
    let report = fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!(report.quarantined, 1);
    assert!(!super::support::has_node(&fresh, "odd").await);
}
