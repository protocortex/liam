// SPDX-License-Identifier: Apache-2.0
//! Tiering from end to end, over a real store and event log: a cold item comes
//! back when it is used, a demoted item keeps everything but its vectors, and
//! the moves settle instead of flipping.

use std::path::Path;
use std::sync::Arc;

use liam_daemon::config::TieringConfig;
use liam_daemon::consolidation::hooks::TierHook;
use liam_daemon::consolidation::tier_history::TierHistory;
use liam_daemon::models::StoreEmbedder;
use liam_daemon::store_open::open_store_on_log;
use liam_model::{IdentityReranker, MockEmbedder, MockLlm};
use liam_store::{
    ChildKind, Clock, DefaultGraph, EventRole, FixedClock, GraphConfig, ItemView, LineageKey,
    Millis, NewEdge, NewNode, NewSummary, NodeId, OriginClaim, OriginVerdict, Query, RebuildMode,
    SharedLog, SummaryChild, Tier, UsageEntry,
};
use rmcp::handler::server::wrapper::Parameters;

use super::hooks::TierHooks;
use super::test_support::*;
use super::usage::UsageRecorder;
use super::utility::{utility, TierParams};
use crate::mcp::{MemoryServer, RecallArgs};

const TRACKED: usize = 100;
const WRITER: &str = "consolidation:exit-criteria";
/// The query a person would type for the item, sharing no words with it.
const PARAPHRASE: &str = "something else entirely";
/// A signature far from every other fixture signature.
const TARGET_PATTERN: u8 = 0b0000_0111;
const OTHER_PATTERN: u8 = 0b1111_1000;
/// Hits at which the hits term is 1, and a count that leaves it inside the
/// hysteresis band.
const BAND_HITS_CAP: u32 = 5;

fn recall(query: &str) -> RecallArgs {
    RecallArgs {
        query: query.to_string(),
        kind: None,
        scope: None,
        k: Some(5),
        as_of: None,
        mode: None,
    }
}

async fn found_with_ranks(
    store: &DefaultGraph,
    query: &Query,
    id: &NodeId,
) -> Option<(Option<usize>, Option<usize>, Option<usize>)> {
    let hits = store.query_explained(query).await.unwrap();
    hits.iter()
        .find(|hit| hit.hit.id == *id)
        .map(|hit| (hit.lexical_rank, hit.vector_rank, hit.cold_rank))
}

fn embedding_of(label: &str, pattern: u8) -> ScriptedEmbedder {
    ScriptedEmbedder::new(
        DIMS,
        &[
            (&text_of(label), signed(pattern, DIMS)),
            (PARAPHRASE, signed(pattern, DIMS)),
        ],
    )
}

// ---- promotion on access ----

#[tokio::test]
async fn a_cold_item_found_by_a_paraphrase_is_counted_and_promoted_with_its_vectors() {
    // Arrange
    let fixture = fixture().await;
    let (target_id, target) = add(&fixture, "target", TARGET_PATTERN).await;
    let (_, _other) = add(&fixture, "other", OTHER_PATTERN).await;
    demote(&fixture, &target).await;
    fixture
        .store
        .put_usage_logged(START.day_number(), &[])
        .await
        .unwrap();
    fixture.clock.set(Millis(START.0 + 3 * DAY_MS));
    let recorder = Arc::new(UsageRecorder::new(TRACKED));
    let server = MemoryServer::new(
        fixture.store.clone(),
        Arc::new(embedding_of("target", TARGET_PATTERN)),
        Arc::new(IdentityReranker),
        Arc::new(MockLlm),
        1,
        false,
        8192,
        1,
    )
    .with_usage_recorder(recorder.clone());

    // Act
    let reply = server.recall(Parameters(recall(PARAPHRASE))).await;
    recorder
        .flush(&*fixture.store, fixture.clock.now())
        .await
        .unwrap();
    let hook = hooks(
        &fixture,
        TieringConfig {
            hits_cap: 20,
            ..config()
        },
        Arc::new(embedding_of("target", TARGET_PATTERN)),
    );
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert!(reply.contains("target"), "the cold arm missed it: {reply}");
    assert_eq!(report.moved_lineages, vec![target.as_str().to_string()]);
    assert_eq!(tier_of(&fixture, &target).await, Tier::Hot);
    let nodes = fixture.store.lineage_nodes(&target).await.unwrap();
    assert!(nodes
        .iter()
        .all(|node| node.has_vector && node.has_signature));
    let ranks = found_with_ranks(
        &fixture.store,
        &Query::vector(signed(TARGET_PATTERN, DIMS)).with_k(5),
        &target_id,
    )
    .await
    .unwrap();
    assert_eq!((ranks.1, ranks.2), (Some(0), None), "ranked by the f32 arm");
}

// ---- demotion keeps data ----

#[tokio::test]
async fn a_demoted_item_keeps_everything_but_its_vectors_and_a_rebuild_leaves_it_cold() {
    // Arrange
    let fixture = fixture().await;
    let (old_id, old) = add(&fixture, "old", TARGET_PATTERN).await;
    let (mate_id, mate) = add(&fixture, "mate", OTHER_PATTERN).await;
    fixture
        .store
        .link(NewEdge::new(&old_id, &mate_id, "mentions"))
        .await
        .unwrap();
    observed(&fixture, &[&mate]).await;
    let as_of = fixture.clock.now();
    let hook = hooks(&fixture, config(), Arc::new(scripted_owned(&[])));

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert: the vectors are gone and the rest is there
    assert_eq!(report.moved_lineages, vec![old.as_str().to_string()]);
    assert_eq!(tier_of(&fixture, &old).await, Tier::Cold);
    let nodes = fixture.store.lineage_nodes(&old).await.unwrap();
    assert!(nodes
        .iter()
        .all(|node| !node.has_vector && node.has_signature));
    assert!(fixture.store.get(&old_id, as_of).await.unwrap().is_some());
    let lexical = found_with_ranks(&fixture.store, &Query::text(text_of("old")), &old_id).await;
    assert!(lexical.unwrap().0.is_some(), "the text index still has it");
    let paraphrase = found_with_ranks(
        &fixture.store,
        &Query::vector(signed(TARGET_PATTERN, DIMS)).with_k(5),
        &old_id,
    )
    .await;
    assert!(
        paraphrase.unwrap().2.is_some(),
        "the cold arm still finds it"
    );
    let neighbours = fixture.store.neighbors(&mate_id, as_of).await.unwrap();
    assert!(neighbours.contains(&old_id), "the edge is kept");
    let origins = fixture
        .store
        .node_event_ids(&old_id, EventRole::Origin)
        .await
        .unwrap();
    assert!(!origins.is_empty(), "the node still points at its events");
    let claim = OriginClaim {
        node_id: old_id.clone(),
        origin_event_ids: origins.into_iter().map(|event| event.event_id).collect(),
        label: "old".to_string(),
        content: text_of("old"),
    };
    let verdicts = fixture.store.verify_origins(&[claim]).await.unwrap();
    assert_eq!(
        verdicts[&old_id],
        OriginVerdict::Verified,
        "the log holds it"
    );

    // Act: a rebuild from the log into an empty database
    let dir = tempfile::tempdir().unwrap();
    let rebuilt = DefaultGraph::open(
        dir.path().join("rebuilt.db").to_str().unwrap(),
        GraphConfig::new(DIMS),
    )
    .await
    .unwrap()
    .with_embedder(Arc::new(StoreEmbedder(Arc::new(MockEmbedder::new(DIMS)))));
    let (rebuilt, _report) = rebuilt
        .rebuild_from_log(fixture.log.clone(), RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert: still cold, still without a vector, signature intact
    assert_eq!(rebuilt.tier_of(&old).await.unwrap(), Tier::Cold);
    let nodes = rebuilt.lineage_nodes(&old).await.unwrap();
    assert!(nodes
        .iter()
        .all(|node| !node.has_vector && node.has_signature));
    assert_eq!(rebuilt.tier_of(&mate).await.unwrap(), Tier::Hot);
}

// ---- no oscillation ----

async fn run_at(fixture: &Fixture, hook: &TierHooks, day: i64) {
    fixture.clock.set(Millis(day * DAY_MS));
    hook.tier(&ctx(fixture, &[], TierHistory::default()))
        .await
        .unwrap();
}

async fn tiers_of(fixture: &Fixture, lineages: &[&LineageKey]) -> Vec<Tier> {
    let mut tiers = Vec::new();
    for lineage in lineages {
        tiers.push(tier_of(fixture, lineage).await);
    }
    tiers
}

async fn hit(fixture: &Fixture, lineage: &LineageKey, hits: u32, at: Millis) {
    let entry = UsageEntry {
        lineage_key: lineage.clone(),
        hits,
        last_access_at: at,
    };
    fixture
        .store
        .put_usage_logged(fixture.clock.now().day_number(), &[entry])
        .await
        .unwrap();
}

#[tokio::test]
async fn items_in_the_band_stay_where_they_are_and_an_idle_item_is_demoted_once_and_stays_cold() {
    // Arrange: one item used up, one held in the band by fresh hits each run,
    // and one cold item kept in the band the same way
    let fixture = fixture().await;
    let (_, busy) = add(&fixture, "busy", 0b001).await;
    let (_, band) = add(&fixture, "band", 0b010).await;
    let (_, bandcold) = add(&fixture, "bandcold", 0b100).await;
    demote(&fixture, &bandcold).await;
    fixture
        .store
        .put_usage_logged(START.day_number(), &[])
        .await
        .unwrap();
    let first = START.0 / DAY_MS + 45;
    fixture.clock.set(Millis(first * DAY_MS));
    hit(&fixture, &busy, BAND_HITS_CAP, fixture.clock.now()).await;
    let hook = hooks(
        &fixture,
        TieringConfig {
            hits_cap: BAND_HITS_CAP,
            ..config()
        },
        Arc::new(scripted_owned(&[])),
    );
    let lineages = [&busy, &band, &bandcold];
    let mut sequence = Vec::new();

    // Act: runs 45, 80, 81 and 120 days in; the busy item's hits fall out of
    // the 30 day window between the first two
    for day in [first, first + 35, first + 36, first + 75] {
        fixture.clock.set(Millis(day * DAY_MS));
        hit(&fixture, &band, 1, fixture.clock.now()).await;
        hit(&fixture, &bandcold, 1, START).await;
        run_at(&fixture, &hook, day).await;
        sequence.push(tiers_of(&fixture, &lineages).await);
    }

    // Assert
    let (hot, cold) = (Tier::Hot, Tier::Cold);
    assert_eq!(
        sequence,
        vec![
            vec![hot, hot, cold],
            vec![cold, hot, cold],
            vec![cold, hot, cold],
            vec![cold, hot, cold],
        ]
    );
}

// ---- supersede keeps the lineage ----

#[tokio::test]
async fn a_new_version_of_a_cold_item_shares_its_lineage_keeps_its_history_and_is_hot() {
    // Arrange
    let fixture = fixture().await;
    let (old_id, lineage) = add(&fixture, "fact", TARGET_PATTERN).await;
    demote(&fixture, &lineage).await;
    let entry = UsageEntry {
        lineage_key: lineage.clone(),
        hits: 3,
        last_access_at: START,
    };
    fixture
        .store
        .put_usage_logged(START.day_number(), &[entry])
        .await
        .unwrap();
    fixture
        .store
        .put_feedback_logged(&lineage, "producer-0", true)
        .await
        .unwrap();
    let before = item_of(&fixture, &lineage).await;
    fixture.clock.set(Millis(START.0 + DAY_MS));

    // Act
    let new_id = fixture
        .store
        .supersede(
            &old_id,
            NewNode::now("fact", "fact", "a newer text")
                .with_subject("fact")
                .with_producer("agent-a")
                .with_embedding(signed(OTHER_PATTERN, DIMS)),
        )
        .await
        .unwrap();

    // Assert
    assert_eq!(
        fixture.store.lineage_of(&new_id).await.unwrap(),
        Some(lineage.clone())
    );
    let nodes = fixture.store.lineage_nodes(&lineage).await.unwrap();
    assert_eq!(nodes.len(), 1);
    assert!(nodes[0].has_vector && nodes[0].has_signature);
    assert_eq!(tier_of(&fixture, &lineage).await, Tier::Hot);
    let after = item_of(&fixture, &lineage).await;
    assert_eq!(after.hits_by_day, before.hits_by_day);
    assert_eq!(after.feedback, before.feedback);
}

async fn item_of(fixture: &Fixture, lineage: &LineageKey) -> ItemView {
    let snapshot = fixture
        .store
        .usage_snapshot(fixture.clock.now(), i64::from(WINDOW_DAYS))
        .await
        .unwrap();
    snapshot
        .items
        .into_iter()
        .find(|item| &item.lineage_key == lineage)
        .unwrap()
}

// ---- exemptions and summary stability ----

#[tokio::test]
async fn entity_pages_and_summary_children_are_not_demoted_and_a_summary_keeps_its_children() {
    // Arrange
    let fixture = fixture().await;
    let page_id = fixture
        .store
        .upsert_by(NewNode::entity("person", "Ada"))
        .await
        .unwrap();
    let page = fixture.store.lineage_of(&page_id).await.unwrap().unwrap();
    let (child_id, child) = add(&fixture, "child", 0b001).await;
    let (_, plain) = add(&fixture, "plain", 0b010).await;
    let origins = fixture
        .store
        .node_event_ids(&child_id, EventRole::Origin)
        .await
        .unwrap();
    let put = fixture
        .store
        .put_summary(
            NewSummary {
                level: 1,
                scope: String::new(),
                text: "a summary of the child".to_string(),
                events_fed: 1,
                valid_min: START,
                valid_max: START,
                model_id: "model".to_string(),
                prompt_version: "1".to_string(),
            },
            &[SummaryChild {
                id: child_id.as_str().to_string(),
                kind: ChildKind::Node,
            }],
            &origins,
            &[0.5; DIMS],
            &[],
            WRITER,
        )
        .await
        .unwrap();
    let summary_id = summary_id_of(&put);
    let children_before = fixture.store.summary_children(&summary_id).await.unwrap();
    observed(&fixture, &[]).await;
    let hook = hooks(&fixture, config(), Arc::new(scripted_owned(&[])));

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.moved_lineages, vec![plain.as_str().to_string()]);
    assert_eq!(tier_of(&fixture, &page).await, Tier::Hot);
    assert_eq!(tier_of(&fixture, &child).await, Tier::Hot);
    let children_after = fixture.store.summary_children(&summary_id).await.unwrap();
    assert_eq!(children_after, children_before);
    assert_eq!(fixture.store.live_summaries(1, "").await.unwrap().len(), 1);
}

fn summary_id_of(outcome: &liam_store::PutOutcome) -> String {
    match outcome {
        liam_store::PutOutcome::Inserted(id) | liam_store::PutOutcome::Unchanged(id) => id.clone(),
    }
}

// ---- the feedback cap ----

async fn feedback_utility(fixture: &Fixture, lineage: &LineageKey, params: &TierParams) -> f64 {
    let snapshot = fixture
        .store
        .usage_snapshot(fixture.clock.now(), i64::from(WINDOW_DAYS))
        .await
        .unwrap();
    let item = snapshot
        .items
        .iter()
        .find(|item| &item.lineage_key == lineage)
        .unwrap();
    utility(item, params, snapshot.now).value
}

#[tokio::test]
async fn a_burst_of_verdicts_moves_the_score_no_further_than_the_cap_allows() {
    // Arrange
    let fixture = fixture().await;
    let (_, repeated) = add(&fixture, "repeated", 0b001).await;
    let (_, single) = add(&fixture, "single", 0b010).await;
    let (_, capped) = add(&fixture, "capped", 0b011).await;
    let (_, flooded) = add(&fixture, "flooded", 0b100).await;
    let cap = 20;
    for _ in 0..50 {
        fixture
            .store
            .put_feedback_logged(&repeated, "producer-0", true)
            .await
            .unwrap();
    }
    fixture
        .store
        .put_feedback_logged(&single, "producer-0", true)
        .await
        .unwrap();
    for producer in 0..cap {
        fixture
            .store
            .put_feedback_logged(&capped, &format!("producer-{producer}"), true)
            .await
            .unwrap();
    }
    for producer in 0..(cap * 3) {
        fixture
            .store
            .put_feedback_logged(&flooded, &format!("producer-{producer}"), true)
            .await
            .unwrap();
    }
    let config = TieringConfig {
        w1: 0.0,
        w3: 1.0,
        feedback_item_cap: cap,
        ..config()
    };
    let params = TierParams::new(&config, &Default::default());

    // Act
    let repeated_score = feedback_utility(&fixture, &repeated, &params).await;
    let single_score = feedback_utility(&fixture, &single, &params).await;
    let capped_score = feedback_utility(&fixture, &capped, &params).await;
    let flooded_score = feedback_utility(&fixture, &flooded, &params).await;

    // Assert
    assert_eq!(
        repeated_score, single_score,
        "one producer counts once a day"
    );
    assert_eq!(
        flooded_score, capped_score,
        "only the latest {cap} verdicts count"
    );
    assert!(capped_score > single_score);
    assert!(flooded_score < 1.0);
}

// ---- a store from before the signatures ----

/// The store as the daemon opens it: attached to the log and caught up.
async fn file_store(database: &Path, clock: Arc<FixedClock>, log: SharedLog) -> DefaultGraph {
    let store =
        DefaultGraph::open_with_clock(database.to_str().unwrap(), GraphConfig::new(DIMS), clock)
            .await
            .unwrap();
    let log_dir = database.parent().unwrap().join("events");
    open_store_on_log(
        store,
        database,
        &log_dir,
        log,
        Arc::new(MockEmbedder::new(DIMS)),
    )
    .await
    .unwrap()
}

/// Deletes every signature and the note that they were once filled in, as a
/// database written before they existed would have it.
async fn strip_signatures(database: &Path) {
    let db = libsql::Builder::new_local(database).build().await.unwrap();
    let connection = db.connect().unwrap();
    for sql in ["DELETE FROM node_quant", "DELETE FROM quant_backfill_state"] {
        connection.execute(sql, ()).await.unwrap();
    }
}

#[tokio::test]
async fn a_store_from_before_the_signatures_gets_them_and_then_demotes_an_unused_item() {
    // Arrange: a database file whose vectors have no signatures
    let dir = tempfile::tempdir().unwrap();
    let database = dir.path().join("upgraded.db");
    let log = shared_test_log(dir.path());
    let clock = Arc::new(FixedClock::new(START));
    let (old_id, old) = {
        let store = file_store(&database, clock.clone(), log.clone()).await;
        let old_id = store
            .insert(
                NewNode::now("fact", "old", text_of("old"))
                    .with_subject("old")
                    .with_producer("agent-a")
                    .with_embedding(signed(TARGET_PATTERN, DIMS)),
            )
            .await
            .unwrap();
        let lineage = store.lineage_of(&old_id).await.unwrap().unwrap();
        (old_id, lineage)
    };
    strip_signatures(&database).await;
    let store = Arc::new(file_store(&database, clock.clone(), log.clone()).await);
    store
        .put_usage_logged(START.day_number(), &[])
        .await
        .unwrap();
    clock.set(Millis(START.0 + 45 * DAY_MS));
    let fixture = Fixture::over(store, log, clock, dir);
    let hook = hooks(&fixture, config(), Arc::new(scripted_owned(&[])));
    let before = fixture.store.lineage_nodes(&old).await.unwrap();
    assert!(before
        .iter()
        .all(|node| node.has_vector && !node.has_signature));

    // Act
    let report = hook
        .tier(&ctx(&fixture, &[], TierHistory::default()))
        .await
        .unwrap();

    // Assert
    assert_eq!(report.moved_lineages, vec![old.as_str().to_string()]);
    assert_eq!(tier_of(&fixture, &old).await, Tier::Cold);
    let found = found_with_ranks(
        &fixture.store,
        &Query::vector(signed(TARGET_PATTERN, DIMS)).with_k(5),
        &old_id,
    )
    .await;
    assert!(found.unwrap().2.is_some(), "the cold arm finds it");
}
