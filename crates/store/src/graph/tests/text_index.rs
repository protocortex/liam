// SPDX-License-Identifier: Apache-2.0
//! A written entity page stays out of the text index, so it cannot move the
//! ranking of recorded memory, and a store that indexed pages is cleaned once.

use super::*;
use crate::types::Derivation;

const DIMS: usize = 8;
const FIRST_TICK: Millis = Millis(1000);
const SECOND_TICK: Millis = Millis(2000);

/// What `schema` created before pages were kept out: every row indexed.
const OLD_TRIGGERS: [&str; 3] = [
    "CREATE TRIGGER nodes_ai AFTER INSERT ON nodes BEGIN
       INSERT INTO nodes_fts(rowid, label, content) VALUES (new.rowid, new.label, new.content);
     END",
    "CREATE TRIGGER nodes_ad AFTER DELETE ON nodes BEGIN
       INSERT INTO nodes_fts(nodes_fts, rowid, label, content)
         VALUES ('delete', old.rowid, old.label, old.content);
     END",
    "CREATE TRIGGER nodes_au AFTER UPDATE ON nodes BEGIN
       INSERT INTO nodes_fts(nodes_fts, rowid, label, content)
         VALUES ('delete', old.rowid, old.label, old.content);
       INSERT INTO nodes_fts(rowid, label, content) VALUES (new.rowid, new.label, new.content);
     END",
];

/// Puts the triggers of a store from before pages were kept out of the index.
pub(super) async fn use_old_triggers<B: Backend>(g: &Graph<B>) {
    for name in ["nodes_ai", "nodes_ad", "nodes_au"] {
        g.backend
            .execute(&format!("DROP TRIGGER {name}"), &[])
            .await
            .unwrap();
    }
    for sql in OLD_TRIGGERS {
        g.backend.execute(sql, &[]).await.unwrap();
    }
}

fn page(name: &str, text: &str) -> NewNode {
    let mut page = NewNode::entity("person", name);
    page.content = text.to_string();
    page
}

/// Ids of the nodes the text index finds for `word`.
async fn indexed_for<B: Backend>(g: &Graph<B>, word: &str) -> Vec<NodeId> {
    let rows = g
        .backend
        .query(
            "SELECT n.id FROM nodes_fts JOIN nodes n ON n.rowid = nodes_fts.rowid
             WHERE nodes_fts MATCH ?1 ORDER BY n.rowid",
            &[word.into()],
        )
        .await
        .unwrap();
    rows.iter()
        .map(|row| NodeId::from_raw(row.get_string(0).unwrap()))
        .collect()
}

async fn trigger_text<B: Backend>(g: &Graph<B>) -> String {
    let rows = g
        .backend
        .query(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'nodes_ai'",
            &[],
        )
        .await
        .unwrap();
    rows[0].get_string(0).unwrap()
}

#[tokio::test]
async fn a_written_page_is_not_indexed_for_text_and_has_no_vector() {
    // Arrange
    let g = graph_at(FIRST_TICK).await;
    let fact = g
        .insert(NewNode::now("fact", "note", "zorbnax likes tea"))
        .await
        .unwrap();
    let empty_entity = g
        .insert(NewNode::entity("person", "Zorbnax"))
        .await
        .unwrap();

    // Act
    let page = g
        .upsert_by(page(
            "Zorbnax",
            "Zorbnax (person)\n1970-01-01: zorbnax likes tea.",
        ))
        .await
        .unwrap();

    // Assert: the fact and the entity row with no text are indexed, the page is not
    assert_eq!(
        indexed_for(&g, "zorbnax").await,
        [fact.clone(), empty_entity]
    );
    assert!(g.embedding(&page).await.unwrap().is_none());
    let missing = g.backend.nodes_missing_vectors().await.unwrap();
    assert!(
        missing.contains(&fact) && !missing.contains(&page),
        "{missing:?}"
    );
}

#[tokio::test]
async fn updating_and_deleting_a_page_leave_the_text_index_as_it_was() {
    // Arrange: a fact and a page, the page then superseded by another
    let g = graph_at(FIRST_TICK).await;
    let fact = g
        .insert(NewNode::now("fact", "note", "zorbnax likes tea"))
        .await
        .unwrap();
    g.upsert_by(page("Zorbnax", "zorbnax first page"))
        .await
        .unwrap();

    // Act: the supersede updates the first page's `tx_to`
    g.upsert_by(page("Zorbnax", "zorbnax second page"))
        .await
        .unwrap();
    g.backend
        .execute(
            "DELETE FROM nodes WHERE derivation = ?1",
            &[Derivation::Inferred.as_str().into()],
        )
        .await
        .unwrap();

    // Assert: only the fact stays indexed, and a query still ranks it
    assert_eq!(indexed_for(&g, "zorbnax").await, [fact]);
    let hits = g.query(&Query::text("zorbnax")).await.unwrap();
    assert_eq!(hits.len(), 1);
}

#[tokio::test]
async fn a_store_that_indexed_pages_has_them_removed_once_and_gets_the_new_triggers() {
    // Arrange: a file store as the old build left it, a page indexed beside a fact
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("index.db");
    let path = path.to_str().unwrap();
    let clock = Arc::new(FixedClock::new(FIRST_TICK));
    let (fact, old_page) = {
        let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock.clone())
            .await
            .unwrap();
        use_old_triggers(&g).await;
        let fact = g
            .insert(NewNode::now("fact", "note", "zorbnax likes tea"))
            .await
            .unwrap();
        let old_page = g
            .insert(page("Zorbnax", "zorbnax model written profile"))
            .await
            .unwrap();
        assert_eq!(
            indexed_for(&g, "zorbnax").await,
            [fact.clone(), old_page.clone()]
        );
        (fact, old_page)
    };

    // Act
    let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let after_first_open = indexed_for(&g, "zorbnax").await;
    drop(g);
    let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock)
        .await
        .unwrap();

    // Assert: the page entry is gone, the fact stays, the new triggers are in place
    assert_eq!(after_first_open, std::slice::from_ref(&fact));
    assert_eq!(indexed_for(&g, "zorbnax").await, [fact]);
    assert!(trigger_text(&g).await.contains("derivation"));
    g.backend
        .execute(
            "DELETE FROM nodes WHERE id = ?1",
            &[old_page.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(indexed_for(&g, "zorbnax").await.len(), 1);
}

/// The ordered texts and scores a strict query returns.
async fn ranked(g: &DefaultGraph, query: &str) -> Vec<(String, f64)> {
    g.query_explained(&Query::text(query).with_k(10))
        .await
        .unwrap()
        .into_iter()
        .map(|hit| (hit.hit.retrieved.as_text().to_string(), hit.hit.score))
        .collect()
}

#[tokio::test]
async fn a_strict_query_ranks_the_same_with_a_page_written_and_without_for_pt_and_es() {
    // Arrange: two stores with the same memories and the same entity rows
    let memories = [
        "A Maria Costa mudou-se para o Porto em maio.",
        "A Maria Costa trabalha na equipa de plataforma.",
        "Tomas Becker se mudo a Sevilla en mayo.",
        "Tomas Becker trabaja en el equipo de plataforma.",
        "Notas de plataforma sobre a migracao de maio.",
    ];
    let queries = [
        "Maria Costa plataforma maio",
        "Tomas Becker equipo plataforma Sevilla",
    ];
    let mut stores = Vec::new();
    for with_pages in [false, true] {
        let clock = Arc::new(FixedClock::new(FIRST_TICK));
        let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(DIMS), clock.clone())
            .await
            .unwrap();
        for text in memories {
            g.insert(NewNode::now("fact", "note", text)).await.unwrap();
        }
        for name in ["Maria Costa", "Tomas Becker"] {
            g.upsert_by(NewNode::entity("person", name)).await.unwrap();
        }
        if with_pages {
            clock.set(SECOND_TICK);
            for name in ["Maria Costa", "Tomas Becker"] {
                let text =
                    format!("{name} (person)\n1970-01-01: {name} plataforma maio equipo Sevilla.");
                g.upsert_by(page(name, &text)).await.unwrap();
            }
        }
        stores.push(g);
    }

    // Act / Assert
    for query in queries {
        let without = ranked(&stores[0], query).await;
        let with = ranked(&stores[1], query).await;
        assert!(!without.is_empty(), "{query}");
        assert_eq!(with, without, "{query}");
    }
}
