// SPDX-License-Identifier: Apache-2.0
//! Zero-loss soak: events written through the log-first path, then rebuilt
//! into a fresh store and compared row by row. The smoke test runs on every
//! `cargo test`; the full run is `#[ignore]`d and gated by the scheduled soak
//! workflow and the release workflow.

use liam_log::wal::WalConfig;

use super::rebuild::oracle_multiset;
use super::support::{fact_at, Env};
use super::*;
use crate::types::{EpisodeEdge, EpisodeRef};
use crate::DefaultBackend;

/// Events the smoke test ingests, small enough for a debug `cargo test`.
const SMOKE_EVENTS: usize = 10_000;
/// Events the scheduled and release soak ingests.
const FULL_SOAK_EVENTS: usize = 1_000_000;

/// Nodes in each episode batch, which is one durable log append however many
/// rows it carries. Batching is what lets a million rows land in minutes: every
/// append is fsynced, so one append per row would make the full run take hours.
const EPISODE_NODES: usize = 100;
/// Edges that chain one episode's nodes together.
const EPISODE_EDGES: usize = EPISODE_NODES - 1;
/// Distinct `upsert_by` subjects, so most upserts replace a live row.
const UPSERT_SUBJECTS: usize = 50;
/// A GC sweep runs after this many rounds, tombstoning the ephemeral nodes.
const GC_EVERY_ROUNDS: usize = 10;

/// Kind GC sweeps, and the retention that dooms every such node.
const EPHEMERAL_KIND: &str = "ephemeral";
const EPHEMERAL_MAX_AGE: Millis = Millis(100);
/// Where the clock starts; it then advances one millisecond per round, which
/// is past `fact_at`'s valid time by more than the retention.
const CLOCK_START: i64 = 1000;
/// Rotates segments as production does, so the soak crosses many of them.
const SOAK_WAL: WalConfig = WalConfig {
    segment_max_bytes: 1 << 20,
    rotate_interval_secs: 3600,
};

/// One round of writes through every logged path: an episode batch of nodes and
/// edges, then a single insert, upsert, supersede, relate, and ephemeral insert.
/// Returns how many rows it created.
async fn write_round(source: &DefaultGraph, round: usize) -> usize {
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

    let first = source
        .insert(fact_at(&format!("fact-{round}")))
        .await
        .unwrap();
    let subject = format!("subject-{}", round % UPSERT_SUBJECTS);
    let upserted = fact_at(&format!("upserted-{round}")).with_subject(subject);
    source.upsert_by(upserted).await.unwrap();
    let second = source
        .supersede(&first, fact_at(&format!("next-{round}")))
        .await
        .unwrap();
    let other = source
        .insert(fact_at(&format!("peer-{round}")))
        .await
        .unwrap();
    source.relate(&second, &other, "mentions").await.unwrap();
    let ephemeral = NewNode {
        kind: EPHEMERAL_KIND.to_owned(),
        ..fact_at(&format!("ephemeral-{round}"))
    };
    source.insert(ephemeral).await.unwrap();
    EPISODE_NODES + EPISODE_EDGES + ROUND_SINGLE_WRITES
}

/// Rows the single writes in `write_round` create: the first insert, the
/// upsert, the superseding node, the peer, the edge, and the ephemeral node.
const ROUND_SINGLE_WRITES: usize = 6;

/// Drops a store without closing its connections, which still releases its
/// hold on the log. libsql intermittently crashes in `sqlite3Close` when a
/// release build closes connections to a database this size, and that would
/// fail the soak for a reason unrelated to the log. The process ends with the
/// test, so nothing outlives it.
fn leak(graph: DefaultGraph) {
    std::mem::forget(graph.backend);
}

/// Ingests at least `events` rows through the real log-first path, rebuilds the
/// log into a fresh store, and asserts the full sorted multiset of live row
/// content hashes is identical, so a lost record cannot hide behind a
/// duplicate that keeps the row count unchanged.
async fn ingest_and_rebuild(events: usize) {
    // Arrange
    let env = Env::new();
    let clock = Arc::new(FixedClock::new(Millis(CLOCK_START)));
    let (source, log) = env
        .open_over::<DefaultBackend>(env.log(SOAK_WAL), "source.db", Arc::clone(&clock))
        .await;
    let retention = RetentionPolicy::keep(EPHEMERAL_KIND, EPHEMERAL_MAX_AGE).without_reclaim();

    // Act: ingest, then rebuild into a store that has never seen the rows
    let (mut ingested, mut round) = (0, 0);
    while ingested < events {
        clock.set(Millis(CLOCK_START + round as i64));
        ingested += write_round(&source, round).await;
        round += 1;
        if round % GC_EVERY_ROUNDS == 0 {
            source.gc(&retention).await.unwrap();
        }
    }
    source.gc(&retention).await.unwrap();
    let before = oracle_multiset(&source).await;
    leak(source);
    drop(log);

    let (fresh, fresh_log) = env
        .open_over::<DefaultBackend>(env.log(SOAK_WAL), "fresh.db", Arc::clone(&clock))
        .await;
    let (rebuilt, report) = fresh
        .rebuild_from_log(fresh_log, RebuildMode::RequireEmpty)
        .await
        .unwrap();

    // Assert: nothing lost, nothing extra, and rows really were ingested
    assert!(!before.is_empty(), "the soak ingested no live rows");
    let after = rebuilt.projection_hash_multiset().await.unwrap();
    assert!(
        after == before,
        "{} rows rebuilt, {} expected",
        after.len(),
        before.len()
    );
    assert_eq!(report.live_nodes + report.live_edges, before.len());
    leak(rebuilt);
}

#[tokio::test]
async fn soak_smoke_rebuild_loses_no_record() {
    ingest_and_rebuild(SMOKE_EVENTS).await;
}

#[tokio::test]
#[ignore = "full-scale soak: run by the scheduled soak workflow and the release gate"]
async fn soak_full_rebuild_loses_no_record() {
    ingest_and_rebuild(FULL_SOAK_EVENTS).await;
}
