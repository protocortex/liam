// SPDX-License-Identifier: Apache-2.0
//! When a community recompute runs, and the refresh that guards it.

use super::*;

// ---- WU-9: recompute against the fingerprint ----

#[tokio::test]
async fn the_edge_read_carries_the_fingerprint_of_the_rows_it_returned() {
    // The fingerprint and the graph come from ONE statement, so the value
    // stored always describes exactly the edge set that was clustered.
    //
    // What this pins is the invariant, that the count matches the rows
    // returned. The concurrency property itself, that no `relate` can land
    // between reading the edges and reading their fingerprint, holds by
    // construction because there is only one read, and no single-threaded
    // test can demonstrate it. Splitting this back into two queries is the
    // regression to watch for, and ADR-0002 Amendment 5 explains why.
    let (g, _clock, _a, _b) = seeded_pair(Millis(1000)).await;

    let (rows, fingerprint) = g.edges_with_fingerprint().await.unwrap();

    assert_eq!(rows.len() as i64, fingerprint.edge_count);
    assert_eq!(
        fingerprint,
        g.edge_fingerprint().await.unwrap(),
        "the cheap check and the recompute must agree on an idle store"
    );
}

#[tokio::test]
async fn a_second_call_with_nothing_changed_does_not_recompute() {
    let (g, clock, _a, _b) = seeded_pair(Millis(1000)).await;
    g.communities().await.unwrap();
    let first = g.read_cluster_state().await.unwrap().unwrap();

    clock.set(Millis(1000 + 3_600_000));
    g.communities().await.unwrap();

    let second = g.read_cluster_state().await.unwrap().unwrap();
    assert_eq!(
        first.computed_at, second.computed_at,
        "an unchanged store must not run Leiden again"
    );
}

#[tokio::test]
async fn an_edge_asserted_after_the_last_run_forces_a_recompute() {
    let (g, clock, a, _b) = seeded_pair(Millis(1000)).await;
    g.communities().await.unwrap();
    let before = g.read_cluster_state().await.unwrap().unwrap();

    clock.set(Millis(1000 + 3_600_000));
    let c = g.insert(NewNode::now("fact", "c", "x")).await.unwrap();
    g.relate(&a, &c, "mentions").await.unwrap();
    g.communities().await.unwrap();

    let after = g.read_cluster_state().await.unwrap().unwrap();
    assert_ne!(before.computed_at, after.computed_at, "a new edge is stale");
    assert_eq!(after.fingerprint.edge_count, 2);
}

#[tokio::test]
async fn a_swept_edge_forces_a_recompute_even_though_time_only_moves_forward() {
    // The COUNT half of the fingerprint. A deletion lowers the count while
    // MAX(tx_from) cannot fall, so a timestamp-only fingerprint would call
    // this unchanged.
    let (g, clock, _a, _b) = seeded_pair(Millis(1_000_000)).await;
    g.communities().await.unwrap();
    let before = g.read_cluster_state().await.unwrap().unwrap();

    clock.set(Millis(1_000_000 + 10_000));
    g.gc(&RetentionPolicy::keep("fact", Millis(1)))
        .await
        .unwrap();
    g.communities().await.unwrap();

    let after = g.read_cluster_state().await.unwrap().unwrap();
    assert_eq!(after.fingerprint.edge_count, 0);
    assert_ne!(before.computed_at, after.computed_at);
}

#[tokio::test]
async fn a_cold_run_advances_both_timestamps_and_a_warm_run_advances_only_one() {
    // ADR-0002 Amendment 1's whole point. `last_cold_start_at` must stay put
    // across a warm run, or the 24-hour escape hatch can never fire again.
    let t0 = Millis(1_000_000);
    let (g, clock, a, _b) = seeded_pair(t0).await;

    g.communities().await.unwrap();
    let cold = g.read_cluster_state().await.unwrap().unwrap();
    assert_eq!(cold.computed_at, t0);
    assert_eq!(cold.last_cold_start_at, t0, "first run is cold");

    let warm_at = Millis(t0.0 + 3_600_000);
    clock.set(warm_at);
    let c = g.insert(NewNode::now("fact", "c", "x")).await.unwrap();
    g.relate(&a, &c, "mentions").await.unwrap();
    g.communities().await.unwrap();

    let warm = g.read_cluster_state().await.unwrap().unwrap();
    assert_eq!(warm.computed_at, warm_at, "computed_at tracks every run");
    assert_eq!(
        warm.last_cold_start_at, t0,
        "a warm run must carry the cold-start time forward"
    );
}

#[tokio::test]
async fn a_warm_run_in_between_does_not_postpone_the_daily_cold_run() {
    // Reproduces the defect Amendment 1 fixed. Reading `computed_at` for the
    // 24-hour rule would see the warm run at +12h and never fire; reading
    // `last_cold_start_at` correctly sees the cold run at t0.
    let t0 = Millis(1_000_000);
    let (g, clock, a, _b) = seeded_pair(t0).await;
    g.communities().await.unwrap();

    clock.set(Millis(t0.0 + 12 * 3_600_000));
    let c = g.insert(NewNode::now("fact", "c", "x")).await.unwrap();
    g.relate(&a, &c, "mentions").await.unwrap();
    g.communities().await.unwrap();
    let warm = g.read_cluster_state().await.unwrap().unwrap();
    assert_eq!(warm.last_cold_start_at, t0);

    // 25h after the cold run, but only 13h after the most recent run.
    clock.set(Millis(t0.0 + 25 * 3_600_000));
    let d = g.insert(NewNode::now("fact", "d", "x")).await.unwrap();
    g.relate(&a, &d, "mentions").await.unwrap();
    g.communities().await.unwrap();

    let after = g.read_cluster_state().await.unwrap().unwrap();
    assert_eq!(
        after.last_cold_start_at,
        Millis(t0.0 + 25 * 3_600_000),
        "the escape hatch must fire on cold-start age, not on run age"
    );
}

#[tokio::test]
async fn a_prior_assignment_with_no_cluster_state_is_recomputed_from_cold() {
    // A database written before `cluster_state` existed: it carries an
    // assignment and no run state. On an EDGELESS store its live fingerprint
    // is (0, 0), so defaulting the missing state to zero would match and
    // serve that stale assignment forever.
    let g = graph_at(Millis(1000)).await;
    // A REAL node: `node_community.node_id` references `nodes(id)` and
    // libSQL enforces it, so a made-up id cannot be inserted at all.
    let orphan = g.insert(NewNode::now("fact", "lonely", "x")).await.unwrap();
    g.backend
        .execute(
            "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, 7, 1)",
            &[orphan.as_str().into()],
        )
        .await
        .unwrap();
    assert!(g.read_cluster_state().await.unwrap().is_none());

    let out = g.communities().await.unwrap();

    assert!(
        out.is_empty(),
        "the stale assignment must be cleared: {out:?}"
    );
    let state = g.read_cluster_state().await.unwrap().unwrap();
    assert_eq!(state.fingerprint.edge_count, 0);
}

#[tokio::test]
async fn repeated_recomputes_leave_exactly_one_cluster_state_row() {
    // `cluster_state` declares no PRIMARY KEY or UNIQUE, so nothing in the
    // schema stops rows accumulating. The DELETE in the write pair is the
    // only guard.
    let (g, clock, a, _b) = seeded_pair(Millis(1000)).await;
    for i in 1..4 {
        clock.set(Millis(1000 + i * 60_000));
        let n = g
            .insert(NewNode::now("fact", format!("n{i}"), "x"))
            .await
            .unwrap();
        g.relate(&a, &n, "mentions").await.unwrap();
        g.communities().await.unwrap();
    }

    let rows = g
        .backend
        .query("SELECT COUNT(*) FROM cluster_state", &[])
        .await
        .unwrap();
    assert_eq!(rows[0].get_i64(0).unwrap(), 1);
}

#[tokio::test]
async fn an_edge_landing_during_a_recompute_leaves_the_fingerprint_behind_not_ahead() {
    // THE guard for ADR-0002 Amendment 5, and the only one that needs a
    // concurrency seam. The recompute must persist the fingerprint captured
    // WITH the edge set it clustered. If it re-queried at commit instead, an
    // edge landing mid-run would be recorded as included when the assignment
    // never saw it, the next check would MATCH, and a stale assignment would
    // be served over a live edge.
    //
    // Behind is safe, ahead is corruption: the assertion is that a second
    // call still finds work to do.
    let _guard = TEST_LOCK.lock().await;
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g: Graph<Interposing> =
        Graph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();
    let c = g.insert(NewNode::now("fact", "c", "x")).await.unwrap();
    g.relate(&a, &b, "mentions").await.unwrap();

    // Fire right after the recompute reads the edge set, so the new edge is
    // in the table but not in the graph that was clustered.
    *INTERPOSE.lock().unwrap() = Some((
        "COUNT(*) OVER ()".to_string(),
        a.as_str().to_string(),
        c.as_str().to_string(),
    ));
    g.communities().await.unwrap();
    assert!(
        INTERPOSE.lock().unwrap().is_none(),
        "the injected write never fired, so this test proves nothing"
    );

    let stored = g.read_cluster_state().await.unwrap().unwrap();
    let live = g.edge_fingerprint().await.unwrap();
    assert_eq!(stored.fingerprint.edge_count, 1, "captured with the graph");
    assert_eq!(live.edge_count, 2, "the injected edge is really there");
    assert_ne!(
        stored.fingerprint, live,
        "the stored fingerprint must lag the live one, never match it"
    );

    // And the consequence that actually matters: the next call recomputes.
    let before = stored.computed_at;
    clock.set(Millis(2000));
    g.communities().await.unwrap();
    let after = g.read_cluster_state().await.unwrap().unwrap();
    assert_ne!(
        before, after.computed_at,
        "a fingerprint written ahead of the graph would skip this recompute"
    );
    assert_eq!(after.fingerprint.edge_count, 2);
}

// ---- WU-11a: the shared refresh seam (ADR-0002 Amendment 6) ----

#[tokio::test]
async fn the_staleness_check_reads_state_and_live_in_one_statement() {
    // Amendment 6's guard: exactly one query, so there's no "between two
    // reads" for a write to land in. `MAX(tx_from),` (trailing comma) is
    // unique to this statement among the store's other fingerprint reads.
    let _guard = TEST_LOCK.lock().await;
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g: Graph<Interposing> =
        Graph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();
    g.relate(&a, &b, "mentions").await.unwrap();

    *QUERY_MATCH_COUNT.lock().unwrap() = ("MAX(tx_from),".to_string(), 0);
    g.refresh_communities().await.unwrap();

    // Read the count into a local before asserting so the MutexGuard
    // drops before a failing assertion could panic while still holding
    // it, which would poison the mutex for every later test.
    let round_trips = QUERY_MATCH_COUNT.lock().unwrap().1;
    assert_eq!(
        round_trips, 1,
        "the staleness check must be one statement, not two"
    );
}

#[tokio::test]
async fn a_missing_cluster_state_row_reads_as_no_prior_run_on_the_combined_read() {
    // Defect 2, reproduced on the new read path: `None` must never be
    // defaulted into a zero fingerprint. `cluster_state`'s four columns
    // are declared NOT NULL, so on an empty table the combined statement's
    // scalar subqueries all come back NULL on the one aggregate row, and
    // that NULL group must map to `None`, not to `(0, 0)`.
    let g = graph_at(Millis(1000)).await;

    let (state, live) = g.staleness_snapshot().await.unwrap();

    assert!(state.is_none());
    assert_eq!(
        live.edge_count, 0,
        "an edgeless store's live side is (0, 0)"
    );
}

#[tokio::test]
async fn an_edgeless_store_with_a_prior_assignment_and_no_state_still_refreshes() {
    // The same fixture shape as the pre-Amendment-6 migration test,
    // exercised against the new combined read: a database with a
    // populated `node_community` and no `cluster_state` row. If `None`
    // were defaulted to `(0, 0)`, it would MATCH an edgeless store's live
    // fingerprint and serve that stale assignment forever.
    let g = graph_at(Millis(1000)).await;
    let orphan = g.insert(NewNode::now("fact", "lonely", "x")).await.unwrap();
    g.backend
        .execute(
            "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, 7, 1)",
            &[orphan.as_str().into()],
        )
        .await
        .unwrap();

    let did_work = g.refresh_communities().await.unwrap();

    assert!(did_work, "a missing cluster_state row must force a refresh");
    assert!(g.stored_communities().await.unwrap().is_empty());
}

#[tokio::test]
async fn refresh_reports_work_only_when_there_is_work() {
    let (g, clock, a, _b) = seeded_pair(Millis(1000)).await;

    assert!(
        g.refresh_communities().await.unwrap(),
        "first call on a store that has never clustered must do work"
    );
    assert!(
        !g.refresh_communities().await.unwrap(),
        "nothing changed since the first call"
    );

    clock.set(Millis(1000 + 3_600_000));
    let c = g.insert(NewNode::now("fact", "c", "x")).await.unwrap();
    g.relate(&a, &c, "mentions").await.unwrap();

    assert!(
        g.refresh_communities().await.unwrap(),
        "a new edge must be reported as work"
    );
}

// ---- WU-12a: community_groups ----
