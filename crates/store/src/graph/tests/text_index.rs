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
async fn a_strict_query_ranks_the_same_with_a_page_written_and_without_for_en_pt_and_es() {
    // Arrange: two stores with the same memories and the same entity rows
    let memories = [
        "Anna Weber moved to Leeds in May.",
        "Anna Weber works on the platform team.",
        "Platform notes about the May migration.",
        "A Maria Costa mudou-se para o Porto em maio.",
        "A Maria Costa trabalha na equipa de plataforma.",
        "Tomas Becker se mudo a Sevilla en mayo.",
        "Tomas Becker trabaja en el equipo de plataforma.",
        "Notas de plataforma sobre a migracao de maio.",
    ];
    let queries = [
        "Anna Weber platform May",
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
        for name in ["Anna Weber", "Maria Costa", "Tomas Becker"] {
            g.upsert_by(NewNode::entity("person", name)).await.unwrap();
        }
        if with_pages {
            clock.set(SECOND_TICK);
            for name in ["Anna Weber", "Maria Costa", "Tomas Becker"] {
                let text =
                    format!("{name} (person)\n1970-01-01: {name} platform plataforma May maio equipo Sevilla.");
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

async fn integrity_check<B: Backend>(g: &Graph<B>) -> Result<u64> {
    g.backend
        .execute(
            "INSERT INTO nodes_fts(nodes_fts) VALUES('integrity-check')",
            &[],
        )
        .await
}

#[tokio::test]
async fn a_page_written_by_the_new_triggers_passes_the_integrity_check() {
    // Arrange
    let g = graph_at(FIRST_TICK).await;
    g.insert(NewNode::now("fact", "note", "zorbnax likes tea"))
        .await
        .unwrap();

    // Act
    g.upsert_by(page("Zorbnax", "zorbnax page")).await.unwrap();

    // Assert
    integrity_check(&g).await.unwrap();
}

const SEEDED_FACTS: usize = 1500;
const SEEDED_ENTITIES: usize = 100;
const SEEDED_STUBS: usize = 100;
const PAGE_VERSIONS: usize = 3;
/// How often a seeded fact is also looked up by vector.
const VECTOR_PROBE_EVERY: usize = 100;

/// A unit vector whose angle grows with `n`, so no two seeded facts share one.
fn spread_vector(n: usize) -> Vec<f32> {
    let angle = n as f32 * 0.0037;
    let mut vector = vec![0.0; DIMS];
    vector[0] = angle.cos();
    vector[1] = angle.sin();
    vector
}

/// A file store left as the old build would have: every row indexed. It holds
/// recorded facts with vectors, entities each rewritten into closed and open
/// pages, and empty entity stubs. Summaries live in their own tables, outside
/// `nodes` and the text index, so they are not seeded here.
async fn seed_an_old_store(path: &str, clock: &Arc<FixedClock>) -> Vec<NodeId> {
    let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    use_old_triggers(&g).await;
    let mut facts = Vec::new();
    for n in 0..SEEDED_FACTS {
        let fact = NewNode::now("fact", "note", format!("seedfact{n} text"))
            .with_embedding(spread_vector(n));
        facts.push(g.insert(fact).await.unwrap());
    }
    for entity in 0..SEEDED_ENTITIES {
        for version in 0..PAGE_VERSIONS {
            clock.set(Millis(FIRST_TICK.0 + (version as i64 + 1) * 1000));
            let text = format!("seedentity{entity} seedpage version{version}");
            g.upsert_by(page(&format!("Seed Entity {entity}"), &text))
                .await
                .unwrap();
        }
    }
    for stub in 0..SEEDED_STUBS {
        g.insert(NewNode::entity("person", format!("Seed Stub {stub}")))
            .await
            .unwrap();
    }
    clock.set(FIRST_TICK);
    facts
}

#[tokio::test]
async fn migrating_a_seeded_old_store_keeps_every_fact_findable_and_drops_every_page_entry() {
    // Arrange
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("seeded.db");
    let path = path.to_str().unwrap();
    let clock = Arc::new(FixedClock::new(FIRST_TICK));
    let facts = seed_an_old_store(path, &clock).await;

    // Act
    let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock.clone())
        .await
        .unwrap();
    let after_first = trigger_text(&g).await;
    drop(g);
    let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock)
        .await
        .unwrap();

    // Assert
    integrity_check(&g).await.unwrap();
    assert_eq!(trigger_text(&g).await, after_first);
    assert!(indexed_for(&g, "seedpage").await.is_empty());
    for (n, fact) in facts.iter().enumerate() {
        assert_eq!(
            indexed_for(&g, &format!("seedfact{n}")).await,
            std::slice::from_ref(fact)
        );
        assert!(g.embedding(fact).await.unwrap().is_some());
    }
    for n in (0..SEEDED_FACTS).step_by(VECTOR_PROBE_EVERY) {
        let hits = g
            .query(&Query::vector(spread_vector(n)).with_k(3))
            .await
            .unwrap();
        assert_eq!(
            hits[0].retrieved.as_text(),
            format!("seedfact{n} text"),
            "fact {n}"
        );
    }
    let by_label = indexed_for(&g, "stub").await;
    assert_eq!(by_label.len(), SEEDED_STUBS);
}

#[tokio::test]
async fn two_openers_that_both_pass_the_old_trigger_check_remove_the_page_entries_once() {
    // Arrange: an old store, and a second opener that read the old triggers
    // before the first one committed
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("raced.db");
    let path = path.to_str().unwrap();
    let clock = Arc::new(FixedClock::new(FIRST_TICK));
    {
        let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock.clone())
            .await
            .unwrap();
        use_old_triggers(&g).await;
        g.insert(NewNode::now("fact", "note", "zorbnax likes tea"))
            .await
            .unwrap();
        g.insert(page("Zorbnax", "zorbnax model written profile"))
            .await
            .unwrap();
    }
    let first = crate::backends::DefaultBackend::open(path, 1)
        .await
        .unwrap();
    let second = crate::backends::DefaultBackend::open(path, 1)
        .await
        .unwrap();

    // Act: the first opener commits, then the second runs its transaction anyway
    crate::migrate::remove_pages_and_replace_triggers(&first)
        .await
        .unwrap();
    crate::migrate::remove_pages_and_replace_triggers(&second)
        .await
        .unwrap();

    // Assert
    let g = DefaultGraph::open_with_clock(path, GraphConfig::new(DIMS), clock)
        .await
        .unwrap();
    integrity_check(&g).await.unwrap();
    assert_eq!(indexed_for(&g, "zorbnax").await.len(), 1);
}
