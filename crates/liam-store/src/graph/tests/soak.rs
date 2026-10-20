// SPDX-License-Identifier: Apache-2.0
//! Zero-loss soak: rows written through the log-first path, then rebuilt
//! into a fresh store and compared row by row. The smoke test runs on every
//! `cargo test`; the full run is `#[ignore]`d and gated by the scheduled soak
//! workflow and the release workflow.

use super::rebuild::{all_rows_multiset, oracle_multiset};
use super::support::{count, fact_at, Env, ONE_SEGMENT};
use super::*;
use crate::types::{EpisodeEdge, EpisodeRef, GcReport};
use crate::DefaultBackend;

/// Rows the smoke test ingests, small enough for a debug `cargo test`.
const SMOKE_ROWS: usize = 10_000;
/// Rows the scheduled and release soak ingests.
const FULL_SOAK_ROWS: usize = 1_000_000;

/// Nodes in each episode batch, which is one durable log append however many
/// rows it carries. Batching is what lets a million rows land in minutes: every
/// append is fsynced, so one append per row would make the full run take hours.
const EPISODE_NODES: usize = 100;
/// Edges that chain one episode's nodes together.
const EPISODE_EDGES: usize = EPISODE_NODES - 1;
/// Distinct `upsert_by` subjects. Fewer than the rounds of the smoke run, so
/// every run replaces upserted rows rather than only inserting them.
const UPSERT_SUBJECTS: usize = 10;
/// A GC sweep runs after this many rounds, tombstoning the ephemeral nodes.
const GC_EVERY_ROUNDS: usize = 10;

/// Kind GC sweeps, and the retention that dooms every such node.
const EPHEMERAL_KIND: &str = "ephemeral";
const EPHEMERAL_MAX_AGE: Millis = Millis(100);
/// Where the clock starts; it then advances one millisecond per round, which
/// is past `fact_at`'s valid time by more than the retention.
const CLOCK_START: i64 = 1000;

/// What a round wrote: the rows it created and the log appends that took.
#[derive(Default)]
struct Written {
    rows: usize,
    appends: usize,
}

impl Written {
    /// Counts one log append that created `rows` rows.
    fn append(&mut self, rows: usize) {
        self.rows += rows;
        self.appends += 1;
    }
}

/// One round of writes through every logged path: an episode batch of nodes and
/// edges, then a single insert, upsert, supersede, relate, and ephemeral node
/// with an edge to the round's peer, which a sweep must tombstone with it.
async fn write_round(source: &DefaultGraph, round: usize) -> Written {
    let mut written = Written::default();
    let nodes = (0..EPISODE_NODES)
        .map(|n| fact_at(&format!("episode-{round}-{n}")))
        .collect();
    let edges = (0..EPISODE_EDGES)
        .map(|n| EpisodeEdge {
            from: EpisodeRef::New(n),
            to: EpisodeRef::New(n + 1),
            kind: "mentions".to_owned(),
            attributes: serde_json::json!({}),
        })
        .collect();
    source.ingest_episode(nodes, edges).await.unwrap();
    written.append(EPISODE_NODES + EPISODE_EDGES);

    let first = source
        .insert(fact_at(&format!("fact-{round}")))
        .await
        .unwrap();
    written.append(1);
    let subject = format!("subject-{}", round % UPSERT_SUBJECTS);
    let upserted = fact_at(&format!("upserted-{round}")).with_subject(subject);
    source.upsert_by(upserted).await.unwrap();
    // A replacing upsert also writes the `supersedes` edge that closes the
    // row it replaced, and every subject is taken after the first cycle.
    let replaces = round >= UPSERT_SUBJECTS;
    written.append(1 + usize::from(replaces));
    let second = source
        .supersede(&first, fact_at(&format!("next-{round}")))
        .await
        .unwrap();
    written.append(2);
    let peer = source
        .insert(fact_at(&format!("peer-{round}")))
        .await
        .unwrap();
    written.append(1);
    source.relate(&second, &peer, "mentions").await.unwrap();
    written.append(1);
    let ephemeral = NewNode {
        kind: EPHEMERAL_KIND.to_owned(),
        ..fact_at(&format!("ephemeral-{round}"))
    };
    let ephemeral = source.insert(ephemeral).await.unwrap();
    written.append(1);
    source.relate(&ephemeral, &peer, "mentions").await.unwrap();
    written.append(1);
    written
}

/// Drops a store without closing its connections, which still releases its
/// hold on the log. libsql intermittently crashes in `sqlite3Close` when a
/// release build closes connections to a database this size, and that would
/// fail the soak for a reason unrelated to the log. The process ends with the
/// test, so nothing outlives it. See #181.
fn leak(graph: DefaultGraph) {
    std::mem::forget(graph.backend);
}

/// Ingests at least `rows` rows through the real log-first path, rebuilds the
/// log into a fresh store, and asserts the sorted multisets of content hashes
/// are identical for the live rows and for all rows, closed ones included, so
/// neither a lost record nor a changed superseded row can hide.
async fn ingest_and_rebuild(rows: usize) {
    // Arrange
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(Millis(CLOCK_START)));
    let (source, log) = env
        .open::<DefaultBackend>("source.db", Arc::clone(&clock))
        .await;
    let retention = RetentionPolicy::keep(EPHEMERAL_KIND, EPHEMERAL_MAX_AGE).without_reclaim();

    // Act: ingest, then rebuild into a store that has never seen the rows
    let (mut written, mut round) = (Written::default(), 0);
    let mut removed = GcReport::default();
    while written.rows < rows {
        clock.set(Millis(CLOCK_START + round as i64));
        let batch = write_round(&source, round).await;
        written.rows += batch.rows;
        written.appends += batch.appends;
        round += 1;
        if round % GC_EVERY_ROUNDS == 0 {
            sweep(&source, &retention, &mut removed).await;
        }
    }
    sweep(&source, &retention, &mut removed).await;
    let live_before = oracle_multiset(&source).await;
    let all_before = all_rows_multiset(&source).await;
    let stored = count(&source, "nodes").await + count(&source, "edges").await;
    leak(source);
    drop(log);

    let (fresh, fresh_log) = env
        .open_over::<DefaultBackend>(env.log(ONE_SEGMENT), "fresh.db", Arc::clone(&clock))
        .await;
    let (rebuilt, report) = fresh
        .rebuild_from_log(fresh_log, RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert: nothing lost, nothing extra, and rows really were ingested
    eprintln!(
        "soak: {} rows in {} log appends over {round} rounds, gc removed {} nodes and {} edges",
        written.rows, written.appends, removed.nodes_removed, removed.edges_removed
    );
    assert!(!live_before.is_empty(), "the soak ingested no live rows");
    assert!(removed.nodes_removed > 0, "gc removed no nodes");
    assert!(
        removed.edges_removed >= removed.nodes_removed,
        "gc left the edges of {} swept nodes in place",
        removed.nodes_removed
    );
    assert_eq!(
        stored as u64,
        written.rows as u64 - removed.nodes_removed - removed.edges_removed,
        "rows written, swept and stored disagree"
    );
    let live_after = rebuilt.projection_hash_multiset().await.unwrap();
    assert!(
        live_after == live_before,
        "{} live rows rebuilt, {} expected",
        live_after.len(),
        live_before.len()
    );
    assert!(
        all_rows_multiset(&rebuilt).await == all_before,
        "the rebuilt rows differ from the {} stored, closed rows included",
        all_before.len()
    );
    assert_eq!(report.live_nodes + report.live_edges, live_before.len());
    leak(rebuilt);
}

async fn sweep(source: &DefaultGraph, retention: &RetentionPolicy, total: &mut GcReport) {
    let report = source.gc(retention).await.unwrap();
    total.nodes_removed += report.nodes_removed;
    total.edges_removed += report.edges_removed;
}

#[tokio::test]
async fn soak_smoke_rebuild_loses_no_record() {
    ingest_and_rebuild(SMOKE_ROWS).await;
}

#[tokio::test]
#[ignore = "full-scale soak: run by the scheduled soak workflow and the release gate"]
async fn soak_full_rebuild_loses_no_record() {
    ingest_and_rebuild(FULL_SOAK_ROWS).await;
}
