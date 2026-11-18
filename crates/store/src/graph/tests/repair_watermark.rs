// SPDX-License-Identifier: Apache-2.0
//! The watermark of the mention repair, and the repair of superseded mentions.

use super::*;

#[test]
fn next_watermark_is_none_for_an_empty_list() {
    // Arrange
    let processed: Vec<(Millis, EdgeOutcome)> = Vec::new();

    // Act
    let watermark = next_watermark(&processed);

    // Assert
    assert_eq!(watermark, None);
}

#[test]
fn next_watermark_advances_to_the_newest_tx_from_when_everything_succeeds() {
    let processed = vec![
        (Millis(100), EdgeOutcome::Inserted),
        (Millis(300), EdgeOutcome::AlreadyPresent),
        (Millis(200), EdgeOutcome::NotLive),
    ];

    let watermark = next_watermark(&processed);

    assert_eq!(watermark, Some(Millis(300)));
}

#[test]
fn next_watermark_never_advances_past_a_real_failure_despite_later_successes() {
    // The entry at 300 succeeds but comes after the failure at 200, so it
    // must not pull the watermark forward: a later scan has to revisit
    // the failed edge, which it can only do if the watermark stayed
    // behind it.
    let processed = vec![
        (Millis(100), EdgeOutcome::Inserted),
        (Millis(200), EdgeOutcome::RealFailure),
        (Millis(300), EdgeOutcome::Inserted),
    ];

    let watermark = next_watermark(&processed);

    assert_eq!(watermark, Some(Millis(100)));
}

#[test]
fn next_watermark_is_none_when_the_first_entry_is_a_real_failure() {
    let processed = vec![
        (Millis(100), EdgeOutcome::RealFailure),
        (Millis(200), EdgeOutcome::Inserted),
    ];

    let watermark = next_watermark(&processed);

    assert_eq!(watermark, None);
}

#[test]
fn next_watermark_excludes_a_tied_timestamp_when_the_success_is_processed_first() {
    // Both entries share tx_from=100: the success is processed before
    // the failure, but nothing is safely before the failure's own
    // timestamp, so the watermark must not advance to it.
    let processed = vec![
        (Millis(100), EdgeOutcome::Inserted),
        (Millis(100), EdgeOutcome::RealFailure),
    ];

    let watermark = next_watermark(&processed);

    assert_eq!(watermark, None);
}

#[test]
fn next_watermark_excludes_a_tied_timestamp_regardless_of_processing_order() {
    // Same tie as above with the failure processed first, confirming the
    // bound comes from the timestamp, not from list position.
    let processed = vec![
        (Millis(100), EdgeOutcome::RealFailure),
        (Millis(100), EdgeOutcome::Inserted),
    ];

    let watermark = next_watermark(&processed);

    assert_eq!(watermark, None);
}

// ---- repair_superseded_mentions ----

#[tokio::test]
async fn repair_inserts_a_mentions_edge_to_the_node_that_superseded_the_original() {
    // Arrange: an entity mentions a fact, which is then superseded once.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    g.relate(&entity, &fact, relation::MENTIONS).await.unwrap();
    clock.set(Millis(2000));
    let fact2 = g
        .supersede(&fact, NewNode::now("fact", "second", "x"))
        .await
        .unwrap();

    // Act
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert
    assert_eq!(repaired, 1);
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                entity.as_str().into(),
                fact2.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        rows[0].get_i64(0).unwrap(),
        1,
        "a live mentions edge to the new node must exist"
    );
}

#[tokio::test]
async fn repair_follows_a_two_hop_supersession_chain_to_the_terminal_node() {
    // Arrange: fact A is superseded by B, and B is superseded by C, both
    // landing inside one repair window, so the entity's mention must jump
    // straight to C rather than stopping at the intermediate B.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact_a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    g.relate(&entity, &fact_a, relation::MENTIONS)
        .await
        .unwrap();
    clock.set(Millis(2000));
    let fact_b = g
        .supersede(&fact_a, NewNode::now("fact", "b", "x"))
        .await
        .unwrap();
    clock.set(Millis(3000));
    let fact_c = g
        .supersede(&fact_b, NewNode::now("fact", "c", "x"))
        .await
        .unwrap();

    // Act
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert
    assert_eq!(repaired, 1);
    let to_terminal = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                entity.as_str().into(),
                fact_c.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        to_terminal[0].get_i64(0).unwrap(),
        1,
        "must land on the terminal node C"
    );
    let to_intermediate = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                entity.as_str().into(),
                fact_b.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        to_intermediate[0].get_i64(0).unwrap(),
        0,
        "must not stop at the intermediate node B"
    );
}

#[tokio::test]
async fn repair_excludes_an_already_processed_supersession_on_a_second_run() {
    // Arrange: repair once, so the watermark moves past this supersession.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    g.relate(&entity, &fact, relation::MENTIONS).await.unwrap();
    clock.set(Millis(2000));
    let fact2 = g
        .supersede(&fact, NewNode::now("fact", "second", "x"))
        .await
        .unwrap();
    let first_pass = g.repair_superseded_mentions().await.unwrap();
    assert_eq!(first_pass, 1, "sanity: the first pass must repair it");

    // Act: run again with nothing new since the watermark advanced past it.
    let second_pass = g.repair_superseded_mentions().await.unwrap();

    // Assert: the watermark excludes the edge from the scan entirely,
    // rather than `relate` refusing a second attempt that never fires.
    assert_eq!(second_pass, 0);
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                entity.as_str().into(),
                fact2.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        rows[0].get_i64(0).unwrap(),
        1,
        "exactly one mentions edge to the terminal, not a second insert attempt"
    );
}

#[tokio::test]
async fn repair_lets_relates_own_duplicate_guard_absorb_a_second_attempt_within_one_pass() {
    // Arrange: two live mentions edges from the same entity to the same
    // original node, inserted as rows since `link` and `relate` both refuse
    // a duplicate. Both fall inside one repair pass.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    insert_edge_row(&g, &entity, &fact, relation::MENTIONS).await;
    insert_edge_row(&g, &entity, &fact, relation::MENTIONS).await;
    clock.set(Millis(2000));
    let fact2 = g
        .supersede(&fact, NewNode::now("fact", "second", "x"))
        .await
        .unwrap();

    // Act
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert: `relate`'s own duplicate-triple guard absorbs the second
    // attempt within this single pass, distinct from the watermark
    // exclusion across separate passes.
    assert_eq!(repaired, 1);
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                entity.as_str().into(),
                fact2.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_i64(0).unwrap(), 1);
}

#[tokio::test]
async fn repair_ignores_a_supersession_edge_older_than_the_watermark() {
    // Arrange: a supersession, with the watermark manually advanced past
    // it (standing in for an earlier repair pass), then a fresh mention
    // added to the now-dead original node afterward.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    clock.set(Millis(2000));
    let fact2 = g
        .supersede(&fact, NewNode::now("fact", "second", "x"))
        .await
        .unwrap();
    g.backend
        .execute(
            "INSERT INTO provenance_repair_state (id, last_repaired_at) VALUES (1, ?1)",
            &[Millis(2000).into()],
        )
        .await
        .unwrap();
    clock.set(Millis(3000));
    insert_edge_row(&g, &entity, &fact, relation::MENTIONS).await;

    // Act
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert: bounded scan, not full-graph, so the old supersession never
    // resurfaces regardless of when a mention lands on its dead node.
    assert_eq!(repaired, 0);
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                entity.as_str().into(),
                fact2.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_i64(0).unwrap(), 0);
}

#[tokio::test]
async fn repair_picks_up_a_pre_restart_supersession_after_reopening_the_graph() {
    // Arrange: a supersession happens, then the `Graph` handle is dropped
    // and a fresh one reopened over the same backing file, simulating a
    // daemon restart with the watermark never advanced manually.
    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("repair.db");
    let path_str = path.to_str().expect("temp path is valid utf-8");

    let (entity, fact2) = {
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let g = DefaultGraph::open_with_clock(path_str, GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
        let fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
        g.relate(&entity, &fact, relation::MENTIONS).await.unwrap();
        clock.set(Millis(2000));
        let fact2 = g
            .supersede(&fact, NewNode::now("fact", "second", "x"))
            .await
            .unwrap();
        (entity, fact2)
    }; // g drops here, releasing the file before reopening.

    // Act
    let clock = Arc::new(FixedClock::new(Millis(3000)));
    let g = DefaultGraph::open_with_clock(path_str, GraphConfig::new(8), clock)
        .await
        .expect("reopen graph over the same backing file");
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert
    assert_eq!(
        repaired, 1,
        "repair after a restart must still find the pre-restart supersession"
    );
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                entity.as_str().into(),
                fact2.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_i64(0).unwrap(), 1);
}

#[tokio::test]
async fn repair_does_nothing_for_a_mention_of_a_fact_that_was_never_superseded() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    g.relate(&entity, &fact, relation::MENTIONS).await.unwrap();

    // Act
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert
    assert_eq!(repaired, 0);
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE type = ?1",
            &[relation::MENTIONS.into()],
        )
        .await
        .unwrap();
    assert_eq!(
        rows[0].get_i64(0).unwrap(),
        1,
        "no new mentions edge should have been added"
    );
}

#[tokio::test]
async fn repair_moves_a_superseded_entitys_own_mention_to_the_terminal_entity() {
    // Arrange: an entity mentions a fact, then the entity itself is
    // superseded directly, standing in for what resynthesis does. Unlike
    // `supersede`, this leaves the entity's own `mentions` edge attached
    // to the now-dead original id.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let entity = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "first", "x")).await.unwrap();
    g.relate(&entity, &fact, relation::MENTIONS).await.unwrap();
    clock.set(Millis(2000));
    let terminal_entity = g
        .supersede(&entity, NewNode::entity("person", "Ada"))
        .await
        .unwrap();

    // Act
    let repaired = g.repair_superseded_mentions().await.unwrap();

    // Assert: the mention now originates from the terminal entity id,
    // not the dead original.
    assert_eq!(repaired, 1);
    let fingerprint = g
        .entity_mentions_fingerprint("ada", None, Millis(3000))
        .await
        .unwrap();
    assert_eq!(
        fingerprint.edge_count, 1,
        "the live entity node must have exactly one mentions edge"
    );
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4",
            &[
                terminal_entity.as_str().into(),
                fact.as_str().into(),
                relation::MENTIONS.into(),
                FOREVER.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        rows[0].get_i64(0).unwrap(),
        1,
        "a live mentions edge from the terminal entity to the fact must exist"
    );
}
