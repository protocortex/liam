// SPDX-License-Identifier: Apache-2.0
//! Purging what a destroyed key group left in the derived tables: every table,
//! the rows that stay, and a crash at each point of the purge.

use std::collections::BTreeMap;

use liam_log::event::LogPayload;
use tokio::sync::Barrier;

use super::quant::{signed, DIMS};
use super::support::{count, key_registry, Env, StubEmbedder};
use super::*;
use crate::graph::lineage::{exempt_tables, target_tables, LineageStore, PurgeInput};
use crate::types::{
    ChildKind, EventRef, LineageTarget, NewSummary, PutOutcome, Residual, SummaryChild,
};
use crate::value::Row;
use crate::DefaultBackend;

const ALICE_KEY: &str = "key-alice";
const BOB_KEY: &str = "key-bob";
const ALICE_EVENT: &str = "event-alice";
const BOB_EVENT: &str = "event-bob";
const JOB: &str = "job-1";
const WRITER: &str = "consolidation:test";
/// More statements than any purge runs, so a loop over them ends.
const MOST_STATEMENTS: usize = 200;
const LOOPS: usize = 5;

fn clock() -> Arc<FixedClock> {
    Arc::new(FixedClock::new(Millis(1000)))
}

async fn failing() -> Graph<FailingBackend> {
    Graph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
        .await
        .unwrap()
}

async fn open() -> DefaultGraph {
    DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock())
        .await
        .unwrap()
}

fn target_of(event: &str, key: &str) -> LineageTarget {
    LineageTarget {
        job_id: JOB.to_owned(),
        event_ids: vec![event.to_owned()],
        key_ids: vec![key.to_owned()],
        ..LineageTarget::default()
    }
}

fn alice() -> LineageTarget {
    target_of(ALICE_EVENT, ALICE_KEY)
}

async fn exec<B: Backend>(g: &Graph<B>, sql: &str, params: &[Value]) {
    g.backend.execute(sql, params).await.unwrap();
}

async fn rows<B: Backend>(g: &Graph<B>, sql: &str, params: &[Value]) -> Vec<Row> {
    g.backend.query(sql, params).await.unwrap()
}

fn event_ref(id: &str) -> EventRef {
    EventRef {
        event_id: id.to_owned(),
        synthetic: false,
    }
}

fn summary(level: u8) -> NewSummary {
    NewSummary {
        level,
        scope: String::new(),
        text: "a summary".to_owned(),
        events_fed: 1,
        valid_min: Millis(100),
        valid_max: Millis(900),
        model_id: "model-a".to_owned(),
        prompt_version: "prompt-1".to_owned(),
    }
}

fn children(kind: ChildKind, ids: &[&str]) -> Vec<SummaryChild> {
    ids.iter()
        .map(|id| SummaryChild {
            id: (*id).to_owned(),
            kind,
        })
        .collect()
}

async fn put_summary<B: Backend>(
    g: &Graph<B>,
    level: u8,
    kids: &[SummaryChild],
    closes: &[String],
) -> String {
    let outcome = g
        .put_summary(
            summary(level),
            kids,
            &[event_ref(ALICE_EVENT)],
            &[0.5; DIMS],
            closes,
            WRITER,
        )
        .await
        .unwrap();
    match outcome {
        PutOutcome::Inserted(id) | PutOutcome::Unchanged(id) => id,
    }
}

async fn insert_fact<B: Backend>(
    g: &Graph<B>,
    (id, lineage): (&str, &str),
    subject: &str,
    (tx_to, closed_by): (i64, &str),
) {
    exec(
        g,
        "INSERT INTO facts (id, lineage_id, subject, predicate, object, object_key, evidence_at,
                            valid_from, tx_from, tx_to, closed_by, trust_score, extractor_id,
                            extractor_version)
         VALUES (?1, ?2, ?3, 'p', 'o', 'o', 1, 1, 1, ?4, ?5, 0.5, 'rules', '1')",
        &[
            id.into(),
            lineage.into(),
            subject.into(),
            tx_to.into(),
            closed_by.into(),
        ],
    )
    .await;
    exec(
        g,
        "INSERT OR IGNORE INTO fact_sources (lineage_id, event_id) VALUES (?1, ?2)",
        &[lineage.into(), ALICE_EVENT.into()],
    )
    .await;
}

async fn name_key<B: Backend>(g: &Graph<B>, node_id: &str, key: &str) {
    exec(
        g,
        "INSERT INTO node_keys (node_id, key_id, source) VALUES (?1, ?2, ?3)",
        &[node_id.into(), key.into(), "event".into()],
    )
    .await;
}

/// The ids a world holds, alice's first.
struct World {
    a1: NodeId,
    b1: NodeId,
    shared: String,
    ancestor: String,
    closed: String,
    bob_summary: String,
}

/// Alice's and bob's rows in every table a purge reaches. A summary is shared
/// between them, one is closed, and an ancestor sits above the shared one.
async fn world<B: Backend>(g: &Graph<B>) -> World {
    let a1 = g.insert(node_of("a1", 0b001, "alice")).await.unwrap();
    let a2 = g.insert(node_of("a2", 0b010, "alice")).await.unwrap();
    let b1 = g.insert(node_of("b1", 0b100, "bob")).await.unwrap();
    g.link(NewEdge::new(&a1, &a2, "relates_to")).await.unwrap();
    g.link(NewEdge::new(&a1, &b1, "relates_to")).await.unwrap();
    for (id, key) in [(&a1, ALICE_KEY), (&a2, ALICE_KEY), (&b1, BOB_KEY)] {
        name_key(g, id.as_str(), key).await;
    }
    let closed = put_summary(
        g,
        1,
        &children(ChildKind::Node, &[a1.as_str(), a2.as_str()]),
        &[],
    )
    .await;
    let shared = put_summary(
        g,
        1,
        &children(ChildKind::Node, &[a1.as_str(), b1.as_str()]),
        std::slice::from_ref(&closed),
    )
    .await;
    let ancestor = put_summary(g, 2, &children(ChildKind::Summary, &[shared.as_str()]), &[]).await;
    let bob_summary = put_summary(g, 1, &children(ChildKind::Node, &[b1.as_str()]), &[]).await;
    exec(
        g,
        "INSERT INTO summary_failures (child_hash, failures, last_error, prompt_version, model_id)
         SELECT child_hash, 1, 'timeout', 'p', 'm' FROM summaries WHERE id IN (?1, ?2)",
        &[shared.as_str().into(), bob_summary.as_str().into()],
    )
    .await;
    insert_fact(g, ("f-old", "line-1"), "alice", (500, "f-new")).await;
    insert_fact(g, ("f-new", "line-1"), "alice", (FOREVER.0, "")).await;
    insert_fact(g, ("f-alice", "line-a"), "alice", (FOREVER.0, "")).await;
    insert_fact(g, ("f-bob", "line-b"), "bob", (FOREVER.0, "")).await;
    for (kind, id, key) in [
        ("fact", "f-new", ALICE_KEY),
        ("fact", "f-alice", ALICE_KEY),
        ("fact", "f-bob", BOB_KEY),
    ] {
        exec(
            g,
            "INSERT INTO derived_keys (row_kind, row_id, key_id) VALUES (?1, ?2, ?3)",
            &[kind.into(), id.into(), key.into()],
        )
        .await;
    }
    for (node, event) in [(&a1, ALICE_EVENT), (&b1, BOB_EVENT)] {
        let lineage = g.lineage_of(node).await.unwrap().expect("a lineage");
        for sql in [
            "INSERT INTO item_tier (lineage_key, tier, changed_at) VALUES (?1, 'cold', 1)",
            "INSERT INTO item_usage_events (event_id, lineage_key, day, hits, last_access_at)
             VALUES (?2, ?1, 1, 1, 1)",
            "INSERT INTO item_feedback (event_id, lineage_key, producer, at, day, correct, counted)
             VALUES (?2, ?1, 'p', 1, 1, 1, 1)",
        ] {
            exec(g, sql, &[lineage.as_str().into(), event.into()]).await;
        }
    }
    for (subject, event) in [("alice", ALICE_EVENT), ("bob", BOB_EVENT)] {
        for sql in [
            "INSERT INTO entity_mention_state (subject, mentions_count, mentions_max_tx_from,
                                              last_synthesized_at) VALUES (?1, 1, 1, 1)",
            "INSERT INTO fact_suppressions (subject, predicate, object_key, trust_score, retracted_at)
             VALUES (?1, 'p', 'o', 0.5, 1)",
        ] {
            exec(g, sql, &[subject.into()]).await;
        }
        for sql in [
            "INSERT INTO extraction_attempts (batch_key, event_ids, failures, last_error, updated_at)
             VALUES ('batch-' || ?2, '[\"' || ?2 || '\"]', 1, 'timeout', 1)",
            "INSERT INTO log_hash_index (content_hash, first_event_id, row_ids)
             VALUES (CAST(?2 AS BLOB), ?2, '[]')",
            "INSERT INTO log_quarantine (event_id, segment, seg_index, reason, at)
             VALUES (?2, 0, 0, 'refused', 1)",
        ] {
            exec(g, sql, &[subject.into(), event.into()]).await;
        }
    }
    World {
        a1,
        b1,
        shared,
        ancestor,
        closed,
        bob_summary,
    }
}

fn node_of(label: &str, pattern: u8, subject: &str) -> NewNode {
    NewNode::now("fact", label, format!("content of {label}"))
        .with_embedding(signed(pattern))
        .with_subject(subject)
}

/// Rows per table a purge reaches, as a purge that removed nothing leaves them.
async fn table_counts<B: Backend>(g: &Graph<B>) -> BTreeMap<&'static str, i64> {
    let mut counts = BTreeMap::new();
    for table in target_tables() {
        counts.insert(table, count(g, table).await);
    }
    counts
}

async fn ids<B: Backend>(g: &Graph<B>, table: &str, column: &str) -> Vec<String> {
    let sql = format!("SELECT {column} FROM {table} ORDER BY {column}");
    rows(g, &sql, &[])
        .await
        .iter()
        .map(|row| row.get_string(0).unwrap())
        .collect()
}

async fn assert_verified<B: Backend>(g: &Graph<B>, target: &LineageTarget) {
    let left: Vec<Residual> = g.verify(target).await.unwrap();
    assert!(left.is_empty(), "{left:?}");
}

#[tokio::test]
async fn a_purge_removes_every_row_of_the_group_and_none_of_the_others() {
    // Arrange
    let g = open().await;
    let w = world(&g).await;

    // Act
    let report = g.purge(&alice()).await.unwrap();

    // Assert
    assert!(report.removed > 0);
    assert_verified(&g, &alice()).await;
    assert_eq!(ids(&g, "nodes", "id").await, vec![w.b1.as_str().to_owned()]);
    assert_eq!(count(&g, "edges").await, 0);
    assert_eq!(
        ids(&g, "node_vectors", "node_id").await,
        vec![w.b1.as_str().to_owned()]
    );
    for table in [
        "node_quant",
        "node_lineage",
        "item_tier",
        "item_usage_events",
        "item_feedback",
    ] {
        assert_eq!(count(&g, table).await, 1, "{table}");
    }
    assert_eq!(ids(&g, "facts", "id").await, vec!["f-bob", "f-old"]);
    assert_eq!(ids(&g, "derived_keys", "row_id").await, vec!["f-bob"]);
    assert_eq!(
        ids(&g, "entity_mention_state", "subject").await,
        vec!["bob"]
    );
    assert_eq!(ids(&g, "fact_suppressions", "subject").await, vec!["bob"]);
    assert_eq!(
        ids(&g, "extraction_attempts", "batch_key").await,
        vec!["batch-event-bob"]
    );
    assert_eq!(
        ids(&g, "log_hash_index", "first_event_id").await,
        vec![BOB_EVENT]
    );
    assert_eq!(ids(&g, "log_quarantine", "event_id").await, vec![BOB_EVENT]);
    assert_eq!(count(&g, "node_keys").await, 3, "the groups still resolve");
}

#[tokio::test]
async fn a_summary_built_from_a_deleted_and_a_surviving_node_goes_with_its_ancestors() {
    // Arrange
    let g = open().await;
    let w = world(&g).await;

    // Act
    g.purge(&alice()).await.unwrap();

    // Assert
    assert_eq!(
        ids(&g, "summaries", "id").await,
        vec![w.bob_summary.clone()]
    );
    for id in [&w.shared, &w.ancestor, &w.closed] {
        let left = rows(
            &g,
            "SELECT 1 FROM summary_children WHERE parent_id = ?1",
            &[id.as_str().into()],
        )
        .await;
        assert!(left.is_empty(), "{id}");
    }
    assert_eq!(count(&g, "summary_embeddings").await, 1);
    assert_eq!(count(&g, "summary_sources").await, 1);
    assert_eq!(count(&g, "summary_failures").await, 1);
    assert_eq!(count(&g, "summary_children").await, 1);
}

#[tokio::test]
async fn find_counts_what_a_purge_removes_and_a_purged_store_has_nothing_to_find() {
    // Arrange
    let g = open().await;
    world(&g).await;

    // Act
    let before = g.find(&alice()).await.unwrap();
    let report = g.purge(&alice()).await.unwrap();
    let after = g.find(&alice()).await.unwrap();

    // Assert
    assert_eq!(before.rows.get("nodes"), Some(&2));
    assert_eq!(before.rows.get("node_vectors"), Some(&2));
    assert_eq!(before.total(), report.removed);
    assert_eq!(after.total(), 0, "{after:?}");
}

#[tokio::test]
async fn a_second_purge_removes_nothing() {
    // Arrange
    let g = open().await;
    world(&g).await;
    g.purge(&alice()).await.unwrap();
    let after_first = table_counts(&g).await;

    // Act
    let again = g.purge(&alice()).await.unwrap();

    // Assert
    assert_eq!(again.removed, 0);
    assert_eq!(table_counts(&g).await, after_first);
}

#[tokio::test]
async fn a_target_that_names_nothing_is_a_no_op() {
    // Arrange
    let g = open().await;
    world(&g).await;
    let before = table_counts(&g).await;
    let nothing = LineageTarget {
        job_id: "job-empty".to_owned(),
        ..LineageTarget::default()
    };

    // Act
    let report = g.purge(&nothing).await.unwrap();

    // Assert
    assert_eq!(report.removed, 0);
    assert_eq!(table_counts(&g).await, before);
}

#[tokio::test]
async fn a_purge_over_more_ids_than_one_statement_binds_removes_them_all() {
    // Arrange
    let g = open().await;
    let total = 1200;
    for i in 0..total {
        let id = format!("bulk-{i}");
        exec(
            &g,
            "INSERT INTO nodes (id, kind, label, content, valid_from, tx_from)
             VALUES (?1, 'fact', 'l', 'c', 1, 1)",
            &[id.as_str().into()],
        )
        .await;
        name_key(&g, &id, ALICE_KEY).await;
    }

    // Act
    let report = g.purge(&alice()).await.unwrap();

    // Assert
    assert_eq!(report.removed, total);
    assert_eq!(count(&g, "nodes").await, 0);
}

#[tokio::test]
async fn a_node_swept_before_the_delete_still_takes_its_derived_rows_with_it() {
    // Arrange
    let g = open().await;
    let w = world(&g).await;
    exec(
        &g,
        "DELETE FROM edges WHERE src = ?1 OR dst = ?1",
        &[w.a1.as_str().into()],
    )
    .await;
    g.backend.vector_delete(w.a1.as_str()).await.unwrap();
    exec(
        &g,
        "DELETE FROM nodes WHERE id = ?1",
        &[w.a1.as_str().into()],
    )
    .await;

    // Act
    g.purge(&alice()).await.unwrap();

    // Assert
    assert_verified(&g, &alice()).await;
    assert_eq!(ids(&g, "summaries", "id").await, vec![w.bob_summary]);
}

#[tokio::test]
async fn a_fact_version_the_purged_one_replaced_is_current_again() {
    // Arrange
    let g = open().await;
    world(&g).await;

    // Act
    g.purge(&alice()).await.unwrap();

    // Assert
    let old = rows(
        &g,
        "SELECT tx_to, closed_by FROM facts WHERE id = 'f-old'",
        &[],
    )
    .await;
    assert_eq!(old[0].get_i64(0).unwrap(), FOREVER.0);
    assert_eq!(old[0].get_string(1).unwrap(), "");
    assert_eq!(
        count(&g, "fact_sources").await,
        2,
        "the lineage keeps its sources"
    );
}

#[tokio::test]
async fn an_entity_page_of_a_deleted_subject_takes_its_mention_state_and_basis_with_it() {
    // Arrange
    let g = open().await;
    let page = g
        .insert(NewNode::entity("person", "alice").with_embedding(signed(1)))
        .await
        .unwrap();
    name_key(&g, page.as_str(), ALICE_KEY).await;
    exec(
        &g,
        "INSERT INTO node_events (node_id, event_id, role) VALUES (?1, 'mention', 'basis')",
        &[page.as_str().into()],
    )
    .await;
    exec(
        &g,
        "INSERT INTO entity_mention_state (subject, mentions_count, mentions_max_tx_from,
                                          last_synthesized_at) VALUES ('alice', 1, 1, 1)",
        &[],
    )
    .await;

    // Act
    g.purge(&alice()).await.unwrap();

    // Assert
    assert_eq!(count(&g, "entity_mention_state").await, 0);
    assert_eq!(
        count(&g, "node_events").await,
        0,
        "the basis rows go with the page"
    );
}

#[tokio::test]
async fn a_subject_written_again_before_the_purge_commits_keeps_its_state() {
    // Arrange
    let g = failing().await;
    let w = world(&g).await;
    g.backend.set_fail_on_vector_delete(2);
    assert!(g.purge(&alice()).await.is_err());
    let again = g.insert(node_of("a3", 0b011, "alice")).await.unwrap();

    // Act
    g.purge(&alice()).await.unwrap();

    // Assert
    let survivors = ids(&g, "nodes", "id").await;
    assert!(survivors.contains(&again.as_str().to_owned()));
    assert!(survivors.contains(&w.b1.as_str().to_owned()));
    assert!(!survivors.contains(&w.a1.as_str().to_owned()));
    assert_eq!(
        count(&g, "item_tier").await,
        2,
        "alice's lineage has a node again"
    );
    assert_eq!(count(&g, "entity_mention_state").await, 2);
}

#[tokio::test]
async fn a_failure_after_some_vector_deletes_is_finished_by_a_rerun() {
    // Arrange
    let g = failing().await;
    world(&g).await;
    g.backend.set_fail_on_vector_delete(2);

    // Act
    let first = g.purge(&alice()).await;
    g.backend.set_fail_on_vector_delete(usize::MAX);
    let second = g.purge(&alice()).await;

    // Assert
    assert!(first.is_err());
    second.unwrap();
    assert_verified(&g, &alice()).await;
}

#[tokio::test]
async fn a_failed_commit_after_the_vector_deletes_is_finished_by_a_rerun() {
    // Arrange
    let g = failing().await;
    world(&g).await;
    let before = table_counts(&g).await;
    g.backend.set_fail_commit(true);

    // Act
    let first = g.purge(&alice()).await;
    let left_by_failure = table_counts(&g).await;
    g.backend.set_fail_commit(false);
    let second = g.purge(&alice()).await;

    // Assert
    assert!(first.is_err());
    assert_eq!(left_by_failure, before);
    second.unwrap();
    assert_verified(&g, &alice()).await;
}

#[tokio::test]
async fn a_failure_on_any_statement_of_the_transaction_removes_nothing() {
    // Arrange
    let g = failing().await;
    world(&g).await;
    let before = table_counts(&g).await;
    let mut failures = 0;

    // Act / Assert
    for statement in 0..MOST_STATEMENTS {
        g.backend.set_fail_on_execute(statement);
        if g.purge(&alice()).await.is_ok() {
            break;
        }
        failures += 1;
        assert_eq!(table_counts(&g).await, before, "statement {statement}");
    }
    assert!(
        failures >= target_tables().len(),
        "{failures} statements failed"
    );
    assert_verified(&g, &alice()).await;
}

#[tokio::test]
async fn a_purge_racing_a_write_of_another_subject_keeps_that_write() {
    for _ in 0..LOOPS {
        // Arrange
        let g = Arc::new(open().await);
        world(&g).await;
        let barrier = Arc::new(Barrier::new(2));
        let purge = {
            let (g, barrier) = (Arc::clone(&g), Arc::clone(&barrier));
            tokio::spawn(async move {
                barrier.wait().await;
                g.purge(&alice()).await.unwrap();
            })
        };
        let write = {
            let (g, barrier) = (Arc::clone(&g), Arc::clone(&barrier));
            tokio::spawn(async move {
                barrier.wait().await;
                g.insert(node_of("b2", 0b101, "bob")).await.unwrap()
            })
        };

        // Act
        purge.await.unwrap();
        let written = write.await.unwrap();

        // Assert
        let survivors = ids(&g, "nodes", "id").await;
        assert!(survivors.contains(&written.as_str().to_owned()));
        assert_eq!(survivors.len(), 2);
    }
}

#[tokio::test]
async fn the_inventory_covers_every_table_of_the_schema() {
    // Arrange
    let g = open().await;

    // Act
    let tables = ids(
        &g,
        "sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'nodes_fts%'",
        "name",
    )
    .await;

    // Assert
    for table in tables {
        let covered =
            target_tables().contains(&table.as_str()) || exempt_tables().contains(&table.as_str());
        assert!(
            covered,
            "{table} has neither a purge nor a reason to be left"
        );
    }
}

#[tokio::test]
async fn a_purge_is_logged_and_a_store_replaying_the_log_removes_the_same_rows() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let (live, _log) = env
        .open_over::<DefaultBackend>(env.sealing_log(&registry), "live.db", clock())
        .await;
    let live = live.with_embedder(StubEmbedder::new(DIMS));
    let ingested = live
        .ingest_episode(
            vec![
                NewNode::now("fact", "a", "alice secret").with_subject("alice"),
                NewNode::now("fact", "b", "bob note").with_subject("bob"),
            ],
            Vec::new(),
        )
        .await
        .unwrap();
    let alice_key = rows(
        &live,
        "SELECT key_id FROM node_keys WHERE node_id = ?1",
        &[ingested.node_ids[0].as_str().into()],
    )
    .await[0]
        .get_string(0)
        .unwrap();
    let target = target_of(ALICE_EVENT, &alice_key);

    // Act
    live.purge(&target).await.unwrap();
    let logged = env
        .records()
        .await
        .into_iter()
        .filter(|record| matches!(record.event.payload, LogPayload::LineageRemoval { .. }))
        .count();
    live.purge(&target).await.unwrap();
    let logged_again = env
        .records()
        .await
        .into_iter()
        .filter(|record| matches!(record.event.payload, LogPayload::LineageRemoval { .. }))
        .count();
    let live_nodes = count(&live, "nodes").await;
    drop((live, _log));
    let (fresh, _log) = env
        .open_over::<DefaultBackend>(env.sealing_log(&registry), "fresh.db", clock())
        .await;
    fresh.catch_up().await.unwrap();

    // Assert
    assert_eq!((logged, logged_again), (1, 1));
    assert_eq!(live_nodes, 1);
    assert_eq!(count(&fresh, "nodes").await, 1);
    assert_verified(&fresh, &target).await;
    assert_eq!(count(&fresh, "node_keys").await, 2);
}

#[tokio::test]
async fn replaying_a_removal_twice_changes_nothing() {
    // Arrange
    let g = open().await;
    world(&g).await;
    let input = PurgeInput {
        node_ids: ids(&g, "node_keys WHERE key_id = 'key-alice'", "node_id").await,
        event_ids: vec![ALICE_EVENT.to_owned()],
        key_ids: vec![ALICE_KEY.to_owned()],
    };
    let mut tx = g.backend.begin().await.unwrap();
    crate::graph::lineage::purge_in_tx(&mut *tx, &input, g.backend.vector_delete_sql())
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let once = table_counts(&g).await;

    // Act
    let mut tx = g.backend.begin().await.unwrap();
    let again = crate::graph::lineage::purge_in_tx(&mut *tx, &input, g.backend.vector_delete_sql())
        .await
        .unwrap();
    tx.commit().await.unwrap();

    // Assert
    assert_eq!(again, 0);
    assert_eq!(table_counts(&g).await, once);
}

#[tokio::test]
async fn node_keys_are_filled_from_the_headers_of_a_sealed_log() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let (g, _log) = env
        .open_over::<DefaultBackend>(env.sealing_log(&registry), "live.db", clock())
        .await;
    g.ingest_episode(
        vec![NewNode::now("fact", "a", "alice secret").with_subject("alice")],
        Vec::new(),
    )
    .await
    .unwrap();
    let written = ids(&g, "node_keys", "node_id").await;
    exec(&g, "DELETE FROM node_keys", &[]).await;

    // Act
    let sealed = g.backfill_node_keys().await.unwrap();
    let again = g.backfill_node_keys().await.unwrap();

    // Assert
    assert_eq!(sealed, 1);
    assert_eq!(again, 1);
    assert_eq!(ids(&g, "node_keys", "node_id").await, written);
}

#[tokio::test]
async fn a_store_without_a_log_has_no_headers_to_backfill_from() {
    // Arrange
    let g = open().await;

    // Act
    let sealed = g.backfill_node_keys().await.unwrap();

    // Assert
    assert_eq!(sealed, 0);
}

#[tokio::test]
async fn the_backfill_runs_only_for_a_store_with_no_key_group_recorded() {
    // Arrange
    let env = Env::new();
    let (_keys, registry) = key_registry();
    let (g, _log) = env
        .open_over::<DefaultBackend>(env.sealing_log(&registry), "live.db", clock())
        .await;
    let ingested = g
        .ingest_episode(
            vec![NewNode::now("fact", "a", "alice secret").with_subject("alice")],
            Vec::new(),
        )
        .await
        .unwrap();
    let key = g.key_of_node(&ingested.node_ids[0]).await.unwrap();
    exec(&g, "DELETE FROM node_keys", &[]).await;

    // Act
    let filled = g.backfill_node_keys_if_empty().await.unwrap();
    let skipped = g.backfill_node_keys_if_empty().await.unwrap();

    // Assert
    assert_eq!((filled, skipped), (1, 0));
    assert_eq!(g.key_of_node(&ingested.node_ids[0]).await.unwrap(), key);
    assert!(key.is_some());
}

#[tokio::test]
async fn resolving_a_target_fixes_the_nodes_and_lineage_keys_its_groups_reach() {
    // Arrange
    let g = open().await;
    let w = world(&g).await;
    let lineage = g.lineage_of(&w.a1).await.unwrap().expect("a lineage");

    // Act
    let resolved = g.resolve_target(&alice()).await.unwrap();
    let after_new_node = {
        name_key(&g, "late", ALICE_KEY).await;
        g.resolve_target(&alice()).await.unwrap()
    };

    // Assert
    assert_eq!(resolved.node_ids.len(), 2);
    assert_eq!(resolved.lineage_keys, vec![lineage.as_str().to_owned()]);
    assert_eq!(
        after_new_node, resolved,
        "the outbox keeps the first answer"
    );
}

#[tokio::test]
async fn previewing_a_target_answers_as_resolving_does_and_writes_nothing() {
    // Arrange
    let g = open().await;
    world(&g).await;
    let outboxes = "SELECT COUNT(*) FROM lineage_removals";
    let before = rows(&g, outboxes, &[]).await[0].get_i64(0).unwrap();

    // Act
    let previewed = g.preview_target(&alice()).await.unwrap();
    let after = rows(&g, outboxes, &[]).await[0].get_i64(0).unwrap();
    let resolved = g.resolve_target(&alice()).await.unwrap();

    // Assert
    assert_eq!(previewed, resolved);
    assert_eq!(after, before, "a preview leaves no outbox row");
}

#[tokio::test]
async fn the_dedup_filter_needs_a_log_to_rebuild_or_compare() {
    // Arrange
    let g = open().await;

    // Act
    let rebuilt = g.rebuild_dedup_filter(None).await;
    let compared = g.dedup_filter_matches_index().await;

    // Assert
    assert!(matches!(rebuilt, Err(Error::NoEventLog)), "{rebuilt:?}");
    assert!(matches!(compared, Err(Error::NoEventLog)), "{compared:?}");
}

#[tokio::test]
async fn a_purge_through_a_shared_handle_is_the_same_purge() {
    // Arrange
    let g = Arc::new(open().await);
    world(&g).await;

    // Act
    let store: Arc<dyn LineageStore> = g.clone();
    let report = store.purge(&alice()).await.unwrap();

    // Assert
    assert_eq!(store.name(), crate::GRAPH_STORE);
    assert!(report.removed > 0);
    assert!(store.verify(&alice()).await.unwrap().is_empty());
    assert_eq!(store.find(&alice()).await.unwrap().total(), 0);
}
