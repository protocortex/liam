// SPDX-License-Identifier: Apache-2.0
//! Community detection, its warm start, and the groups it returns.

use super::*;

#[tokio::test]
async fn opening_an_old_schema_database_adds_producer_with_no_data_loss() {
    // Given a database created with the OLD schema (no `producer` column)
    // holding a row. Built by executing an explicit old `CREATE TABLE
    // nodes (...)` statement against a temp-file backend and inserting
    // directly through it, bypassing `Graph` entirely so the row
    // genuinely predates the column rather than merely having had it
    // dropped afterward.
    // This test exercises schema migration, not read pooling, so the
    // pool size is arbitrary; 1 keeps it minimal rather than implying
    // some other value matters here.
    const ARBITRARY_POOL_SIZE: usize = 1;
    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("old.db");
    let path_str = path.to_str().expect("temp path is valid utf-8");
    {
        let old_backend = crate::DefaultBackend::open(path_str, ARBITRARY_POOL_SIZE)
            .await
            .expect("open old-schema backend");
        old_backend
            .execute_batch(
                "CREATE TABLE nodes (
                   rowid       INTEGER PRIMARY KEY,
                   id          TEXT    NOT NULL UNIQUE,
                   kind        TEXT    NOT NULL,
                   label       TEXT    NOT NULL,
                   content     TEXT    NOT NULL,
                   attributes  TEXT    NOT NULL DEFAULT '{}',
                   scope       TEXT,
                   subject     TEXT,
                   confidence  REAL    NOT NULL DEFAULT 1.0,
                   valid_from  INTEGER NOT NULL,
                   valid_until INTEGER NOT NULL DEFAULT 4102444800000,
                   tx_from     INTEGER NOT NULL,
                   tx_to       INTEGER NOT NULL DEFAULT 4102444800000
                 )",
            )
            .await
            .expect("create old-schema nodes table");
        old_backend
            .execute(
                "INSERT INTO nodes
                 (id, kind, label, content, attributes, scope, subject, confidence,
                  valid_from, valid_until, tx_from, tx_to)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                &[
                    "pre-existing".into(),
                    "fact".into(),
                    "old label".into(),
                    "old content".into(),
                    "{}".into(),
                    "proj-a".into(),
                    "old-subject".into(),
                    Value::Real(0.75),
                    Millis(1000).into(),
                    FOREVER.into(),
                    Millis(1000).into(),
                    FOREVER.into(),
                ],
            )
            .await
            .expect("insert pre-existing row");
    } // old_backend drops here, releasing its connection before reopening.

    // When it is opened with the new code (a fresh `Graph` over the same
    // path, running the new schema plus the guarded migration).
    let clock = Arc::new(FixedClock::new(Millis(2000)));
    let g = DefaultGraph::open_with_clock(path_str, GraphConfig::new(8), clock)
        .await
        .expect("open graph over old-schema database");

    // Then the column exists (this SELECT would itself error if it did
    // not), the pre-existing row reads as "unknown", and every other
    // field on that row is intact.
    let rows = g
        .backend
        .query(
            "SELECT producer, kind, label, content, scope, subject, confidence,
                    valid_from, valid_until, tx_from, tx_to
             FROM nodes WHERE id = ?1",
            &["pre-existing".into()],
        )
        .await
        .expect("query the migrated nodes table");
    assert_eq!(rows.len(), 1, "the pre-existing row was not lost");
    let row = &rows[0];
    assert_eq!(
        row.get_string(0).unwrap(),
        "unknown",
        "producer defaults for rows written before the column existed"
    );
    assert_eq!(row.get_string(1).unwrap(), "fact");
    assert_eq!(row.get_string(2).unwrap(), "old label");
    assert_eq!(row.get_string(3).unwrap(), "old content");
    assert_eq!(row.get_string(4).unwrap(), "proj-a");
    assert_eq!(row.get_string(5).unwrap(), "old-subject");
    assert!((row_f64(row, 6) - 0.75).abs() < f64::EPSILON);
    assert_eq!(row.get_i64(7).unwrap(), 1000);
    assert_eq!(row.get_i64(8).unwrap(), FOREVER.0);
    assert_eq!(row.get_i64(9).unwrap(), 1000);
    assert_eq!(row.get_i64(10).unwrap(), FOREVER.0);
}

/// `cluster_state` is a new TABLE, not a new column, so it needs no
/// `migrate::` call: `schema()` is all `CREATE TABLE IF NOT EXISTS` and
/// `open_with_clock` re-runs the whole batch on every open. This pins that,
/// because the reasoning is easy to doubt later and the failure would be a
/// panic on the first `clusters` call against any pre-existing database.
#[tokio::test]
async fn opening_a_database_that_predates_cluster_state_creates_it_with_no_data_loss() {
    // This test exercises schema evolution, not read pooling.
    const ARBITRARY_POOL_SIZE: usize = 1;
    let dir = TempDir::new().expect("create temp dir");
    let path = dir.path().join("no-cluster-state.db");
    let path_str = path.to_str().expect("temp path is valid utf-8");

    // Given a database whose schema has node_community but NO
    // cluster_state, which is exactly what a build predating ADR-0002
    // left behind. Written through a bare backend so the table genuinely
    // never existed rather than having been dropped.
    {
        let old = crate::DefaultBackend::open(path_str, ARBITRARY_POOL_SIZE)
            .await
            .expect("open old-schema backend");
        old.execute_batch(
            "CREATE TABLE node_community (
               node_id     TEXT    NOT NULL PRIMARY KEY,
               community   INTEGER NOT NULL,
               computed_at INTEGER NOT NULL
             );",
        )
        .await
        .expect("create the old cluster tables");
        old.execute(
            "INSERT INTO node_community (node_id, community, computed_at)
             VALUES (?1, ?2, ?3)",
            &["pre-existing".into(), Value::Int(7), Millis(1000).into()],
        )
        .await
        .expect("insert a pre-existing assignment");
    } // dropped, releasing the connection before reopening

    // When it is opened by the new code
    let clock = Arc::new(FixedClock::new(Millis(2000)));
    let g = DefaultGraph::open_with_clock(path_str, GraphConfig::new(8), clock)
        .await
        .expect("open graph over a database with no cluster_state");

    // Then cluster_state exists. This SELECT errors if it does not, which
    // is the assertion.
    let state = g
        .backend
        .query("SELECT COUNT(*) FROM cluster_state", &[])
        .await
        .expect("cluster_state must exist after opening");
    assert_eq!(
        state[0].get_i64(0).unwrap(),
        0,
        "a table created by this open starts empty, so the fingerprint \
         check reads 'no prior run' and forces a cold recompute"
    );

    // And the assignment that was already there is untouched. Creating a
    // table must never disturb its neighbour.
    let kept = g
        .backend
        .query(
            "SELECT community, computed_at FROM node_community WHERE node_id = ?1",
            &["pre-existing".into()],
        )
        .await
        .expect("query the pre-existing assignment");
    assert_eq!(kept.len(), 1, "the pre-existing assignment was lost");
    assert_eq!(kept[0].get_i64(0).unwrap(), 7);
    assert_eq!(kept[0].get_i64(1).unwrap(), 1000);
}

/// The empty state is the one every store is in on the day ADR-0002 ships,
/// so it is worth naming rather than leaving implied.
#[tokio::test]
async fn a_fresh_database_has_an_empty_cluster_state() {
    let g = graph_at(Millis(1000)).await;

    let rows = g
        .backend
        .query("SELECT COUNT(*) FROM cluster_state", &[])
        .await
        .expect("cluster_state must exist on a fresh database");

    assert_eq!(rows[0].get_i64(0).unwrap(), 0);
}

/// Community detection over the edge graph, which had no test at all
/// before this. "Nothing calls it and nothing covers it" was used once as
/// an argument for dropping the feature; the approved design
/// (`2026-07-23-always-available-clusters-design.md`, folded into M5) says
/// the opposite, so this pins the behaviour the daemon will lean on when
/// `recompute_communities` moves onto the maintenance tick.
#[tokio::test]
async fn communities_separate_two_disconnected_groups_of_nodes() {
    // Given two clusters of linked nodes with NO edge between them: three
    // notes about auth that reference each other, and two about billing.
    let g = graph_at(Millis(1000)).await;

    let mut auth = Vec::new();
    for label in ["auth: token refresh", "auth: session expiry", "auth: login"] {
        auth.push(
            g.insert(NewNode::now("fact", label, "content"))
                .await
                .unwrap(),
        );
    }

    let mut billing = Vec::new();
    for label in ["billing: invoices", "billing: dunning"] {
        billing.push(
            g.insert(NewNode::now("fact", label, "content"))
                .await
                .unwrap(),
        );
    }

    for pair in [(0, 1), (1, 2), (0, 2)] {
        g.link(NewEdge::new(&auth[pair.0], &auth[pair.1], "mentions"))
            .await
            .unwrap();
    }
    g.link(NewEdge::new(&billing[0], &billing[1], "mentions"))
        .await
        .unwrap();

    // When communities are recomputed
    let count = g.recompute_communities().await.unwrap();

    // Then every node is assigned, and the two groups land in different
    // communities: that separation is the whole point, and it is what
    // makes clusters usable for finding sections nobody declared.
    // `recompute_communities` returns how many DISTINCT communities it
    // found, not how many nodes it touched.
    assert_eq!(count, 2, "the two disconnected groups are two communities");
    let assignments = g.communities().await.unwrap();
    assert_eq!(
        assignments.len(),
        5,
        "every node that appears on a live edge should be assigned"
    );

    let community_of = |id: &NodeId| {
        assignments
            .iter()
            .find(|(n, _)| n == id)
            .map(|(_, c)| *c)
            .expect("node must have a community")
    };
    let auth_community = community_of(&auth[0]);
    let billing_community = community_of(&billing[0]);

    assert!(
        auth.iter().all(|n| community_of(n) == auth_community),
        "the three linked auth nodes belong together: {assignments:?}"
    );
    assert!(
        billing.iter().all(|n| community_of(n) == billing_community),
        "the two linked billing nodes belong together: {assignments:?}"
    );
    assert_ne!(
        auth_community, billing_community,
        "disconnected groups must not be merged into one community: {assignments:?}"
    );
}

#[tokio::test]
async fn version_chains_do_not_form_communities() {
    // Before the `supersedes` filter, a store holding only version history
    // still produced communities, and they read as topics while describing
    // nothing but edit order. `supersede` is the only writer of that edge
    // type, so a store built purely by `upsert_by` has to cluster to
    // nothing at all (ADR-0001, Scope).
    let g = graph_at(Millis(1000)).await;
    for content in ["first", "second", "third"] {
        g.upsert_by(NewNode::now("fact", "Rollout status", content).with_subject("rollout"))
            .await
            .unwrap();
    }

    let count = g.recompute_communities().await.unwrap();

    assert_eq!(count, 0, "version history is not a topic");
    assert!(
        g.communities().await.unwrap().is_empty(),
        "no node should be assigned from supersedes edges alone"
    );
}

/// `(src, dst, type)` rows in the shape `build_cluster_input` reads, so the
/// dedup can be checked without a database or a Leiden run.
fn edge_rows(triples: &[(&str, &str, &str)]) -> Vec<Row> {
    triples
        .iter()
        .map(|(s, d, t)| {
            Row(vec![
                Value::Text((*s).to_string()),
                Value::Text((*d).to_string()),
                Value::Text((*t).to_string()),
            ])
        })
        .collect()
}

#[tokio::test]
async fn asserting_a_pair_in_both_directions_yields_one_edge() {
    // `relate` is idempotent on the ORDERED triple, so both directions
    // insert, which is right for a directed edge table. Clustering reads
    // the set as undirected and `leiden-rs` merges only consecutive exact
    // duplicates, so without this dedup the pair reaches weight 2.0
    // (ADR-0002 Amendment 2).
    //
    // Asserted on the edge list, NOT on the community count: a count does
    // not discriminate between the deduped graph and the doubled one, so
    // the earlier version of this test passed with the dedup deleted.
    let rows = edge_rows(&[("a", "b", "mentions"), ("b", "a", "mentions")]);

    let (labels, edges) = build_cluster_input(&rows).unwrap();

    assert_eq!(labels.len(), 2, "both endpoints stay in the node set");
    assert_eq!(
        edges.len(),
        1,
        "the reversed pair must collapse to one edge"
    );
}

#[tokio::test]
async fn two_relation_types_on_one_pair_stay_two_edges() {
    // The dedup key carries `type` on purpose, so this pins the OTHER
    // direction of the same guard. Dropping `type` from the key would weigh
    // a pair carrying two asserted relations exactly like a pair carrying
    // one, which is the error ADR-0002 Amendment 2 calls out as second bug.
    let rows = edge_rows(&[("a", "b", "mentions"), ("a", "b", "causes")]);

    let (_, edges) = build_cluster_input(&rows).unwrap();

    assert_eq!(edges.len(), 2, "distinct types are independent evidence");
}

#[tokio::test]
async fn a_repeated_triple_and_a_self_loop_each_collapse_to_one_edge() {
    // `relate` and `link` refuse an exact duplicate through their NOT EXISTS
    // guard, so only a row written outside both, an older store's or a raw
    // insert, can repeat a triple, and a `gc` can leave the table in shapes
    // nothing else produces.
    let repeated = edge_rows(&[("a", "b", "mentions"), ("a", "b", "mentions")]);
    assert_eq!(build_cluster_input(&repeated).unwrap().1.len(), 1);

    // A self-loop normalises to min == max and survives once.
    let loops = edge_rows(&[("a", "a", "mentions"), ("a", "a", "mentions")]);
    let (labels, edges) = build_cluster_input(&loops).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(labels, vec!["a".to_string()]);
}

#[tokio::test]
async fn a_deduped_row_never_drops_a_node_from_the_partition() {
    // Both endpoints are interned BEFORE the dedup check. A row that
    // dedupes shares its key with an earlier row, so its endpoints were
    // already interned and no node can be lost this way. Pinned because the
    // obvious "optimisation" of skipping the intern on a duplicate would
    // silently drop nodes out of clustering entirely.
    let rows = edge_rows(&[
        ("a", "b", "mentions"),
        ("b", "a", "mentions"),
        ("c", "d", "mentions"),
    ]);

    let (labels, edges) = build_cluster_input(&rows).unwrap();

    assert_eq!(labels.len(), 4, "every endpoint keeps a dense index");
    assert_eq!(edges.len(), 2);
}

#[tokio::test]
async fn the_dense_numbering_follows_row_order() {
    // `recompute_communities` orders by edge id so this numbering is a
    // function of the edge set alone. If it were not, Leiden's seeded
    // partition could differ run to run on identical data.
    let rows = edge_rows(&[("z", "y", "mentions"), ("y", "x", "mentions")]);

    let (labels, edges) = build_cluster_input(&rows).unwrap();

    assert_eq!(labels, vec!["z", "y", "x"], "interned in row order");
    assert_eq!((edges[0].0, edges[0].1), (0, 1));
    assert_eq!((edges[1].0, edges[1].1), (1, 2));
}

#[tokio::test]
async fn a_cold_start_is_due_only_strictly_after_twenty_four_hours() {
    let t0 = Millis(1_000_000);
    let day = Millis::days(1).0;

    assert!(!cold_start_due(t0, Millis(t0.0 + day - 1)));
    assert!(
        !cold_start_due(t0, Millis(t0.0 + day)),
        "strictly greater, so exactly 24h is not yet due"
    );
    assert!(cold_start_due(t0, Millis(t0.0 + day + 1)));
}

#[tokio::test]
async fn a_new_node_takes_a_community_id_above_every_id_in_the_seed() {
    let stored = vec![
        (NodeId::from_raw("a"), 0),
        (NodeId::from_raw("b"), 0),
        (NodeId::from_raw("c"), 1),
    ];
    let labels = ["a", "b", "c", "d", "e"].map(String::from).to_vec();

    let seed = build_seed(&labels, &stored);

    assert_eq!(seed, vec![0, 0, 1, 2, 3], "fresh ids continue above max");
}

#[tokio::test]
async fn no_fresh_id_ever_collides_with_a_stored_community() {
    // The stored set has a HOLE, which is the case a "fill the gaps" scheme
    // would get wrong while still passing the contiguous test above.
    // Reusing id 1 here would seed the new node into b's community.
    // A contiguous run AND a hole, so a scheme starting anywhere below the
    // maximum collides with a real community rather than landing in the gap.
    let stored = vec![
        (NodeId::from_raw("a"), 0),
        (NodeId::from_raw("b"), 1),
        (NodeId::from_raw("c"), 5),
    ];
    let labels = ["a", "b", "c", "new1", "new2"].map(String::from).to_vec();

    let seed = build_seed(&labels, &stored);

    let stored_ids: HashSet<usize> = stored
        .iter()
        .map(|(_, c)| usize::try_from(*c).unwrap())
        .collect();
    let fresh: Vec<usize> = seed[3..].to_vec();
    assert!(
        fresh.iter().all(|f| !stored_ids.contains(f)),
        "fresh {fresh:?} collides with stored {stored_ids:?}"
    );
    // Disjointness alone is satisfied by filling the gaps too, so pin the
    // rule ADR-0002 actually states: strictly above every stored id. It is
    // the simpler contract and it keeps the ids dense, which is what bounds
    // `Partition::renumber`'s allocation.
    let stored_max = stored_ids.iter().copied().max().unwrap();
    assert!(
        fresh.iter().all(|&f| f > stored_max),
        "fresh {fresh:?} must all exceed the stored maximum {stored_max}"
    );
    assert_eq!(&seed[..3], &[0, 1, 5], "known nodes keep their communities");
}

#[tokio::test]
async fn a_stored_node_that_no_longer_appears_on_an_edge_drops_out_of_the_seed() {
    // Walking `stored` instead of `labels` would produce a seed of the
    // wrong length, which surfaces as an InvalidPartition at runtime rather
    // than here.
    let stored = vec![
        (NodeId::from_raw("a"), 0),
        (NodeId::from_raw("b"), 1),
        (NodeId::from_raw("gone1"), 2),
        (NodeId::from_raw("gone2"), 3),
    ];
    let labels = ["a", "b"].map(String::from).to_vec();

    let seed = build_seed(&labels, &stored);

    assert_eq!(seed, vec![0, 1]);
}

#[tokio::test]
async fn the_seed_is_exactly_as_long_as_the_label_list() {
    let stored = vec![(NodeId::from_raw("known"), 7)];
    let labels = ["new1", "known", "new2"].map(String::from).to_vec();

    let seed = build_seed(&labels, &stored);

    assert_eq!(seed.len(), labels.len(), "one id per label, no skips");
    assert_eq!(seed[1], 7, "the known node keeps its community");
}

#[tokio::test]
async fn an_empty_stored_assignment_seeds_singletons() {
    let labels = ["a", "b", "c"].map(String::from).to_vec();

    assert_eq!(build_seed(&labels, &[]), vec![0, 1, 2]);
}

#[tokio::test]
async fn warm_starting_from_singletons_matches_a_cold_run() {
    // Pins that both entry points use the same LeidenConfig. A different
    // seed or resolution on the warm path is otherwise invisible.
    use crate::cluster::{detect, Edge};
    let edges = vec![Edge(0, 1), Edge(1, 2), Edge(0, 2)];

    let cold = detect(3, &edges, None);
    let warm = detect(3, &edges, Some(vec![0, 1, 2]));

    assert_eq!(cold, warm);
}

#[tokio::test]
async fn a_wrong_length_seed_falls_back_to_a_cold_run_not_to_singletons() {
    // A triangle, so the cold answer is ONE community and singletons would
    // be three. If the warm path degraded to singletons the way the cold
    // path does, the recompute would persist that alongside a matching
    // fingerprint and serve it until the next edge write.
    use crate::cluster::{detect, Edge};
    let edges = vec![Edge(0, 1), Edge(1, 2), Edge(0, 2)];

    let out = detect(3, &edges, Some(vec![0, 0]));

    assert_eq!(
        out.iter().collect::<HashSet<_>>().len(),
        1,
        "a connected triangle is one community: {out:?}"
    );
}

#[tokio::test]
async fn a_superseded_node_does_not_appear_in_any_group() {
    // Clustering filters edge liveness, not node liveness (ADR-0001), so
    // node_community can legitimately name a superseded node. Rendering
    // must not surface it.
    let g = graph_at(Millis(1000)).await;
    let old = g
        .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
        .await
        .unwrap();
    let other = g.insert(NewNode::now("fact", "other", "x")).await.unwrap();
    g.relate(&old, &other, "mentions").await.unwrap();
    g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
        .await
        .unwrap();

    let groups = g.community_groups(true).await.unwrap();

    let ids: Vec<&str> = groups.iter().flatten().map(|m| m.id.as_str()).collect();
    assert!(!ids.contains(&old.as_str()), "{ids:?}");
    assert!(
        ids.contains(&other.as_str()),
        "the live neighbour must still appear: {ids:?}"
    );
}

#[tokio::test]
async fn a_group_whose_members_are_all_superseded_is_not_returned() {
    let g = graph_at(Millis(1000)).await;
    let a = g
        .upsert_by(NewNode::now("fact", "A", "first").with_subject("a"))
        .await
        .unwrap();
    let b = g
        .upsert_by(NewNode::now("fact", "B", "first").with_subject("b"))
        .await
        .unwrap();
    g.relate(&a, &b, "mentions").await.unwrap();
    g.upsert_by(NewNode::now("fact", "A", "second").with_subject("a"))
        .await
        .unwrap();
    g.upsert_by(NewNode::now("fact", "B", "second").with_subject("b"))
        .await
        .unwrap();

    let groups = g.community_groups(true).await.unwrap();

    assert!(
        groups.is_empty(),
        "an all-superseded group must not appear: {groups:?}"
    );
}

#[tokio::test]
async fn groups_come_back_largest_first_with_a_deterministic_tie_break() {
    let g = graph_at(Millis(1000)).await;
    // A triangle (size 3) plus TWO same-sized pairs, so the sort has both
    // a real size difference and a real tie to resolve.
    let mut three = Vec::new();
    for label in ["t1", "t2", "t3"] {
        three.push(g.insert(NewNode::now("fact", label, "x")).await.unwrap());
    }
    for (u, v) in [(0, 1), (1, 2), (0, 2)] {
        g.relate(&three[u], &three[v], "mentions").await.unwrap();
    }
    let mut pair_a = Vec::new();
    for label in ["a1", "a2"] {
        pair_a.push(g.insert(NewNode::now("fact", label, "x")).await.unwrap());
    }
    g.relate(&pair_a[0], &pair_a[1], "mentions").await.unwrap();
    let mut pair_b = Vec::new();
    for label in ["b1", "b2"] {
        pair_b.push(g.insert(NewNode::now("fact", label, "x")).await.unwrap());
    }
    g.relate(&pair_b[0], &pair_b[1], "mentions").await.unwrap();

    let first = g.community_groups(true).await.unwrap();
    let second = g.community_groups(true).await.unwrap();

    assert_eq!(first[0].len(), 3, "the triangle sorts first: {first:?}");
    assert_eq!(first[1].len(), 2);
    assert_eq!(first[2].len(), 2);
    assert!(
        first[1][0].id.as_str() < first[2][0].id.as_str(),
        "tied groups must break on the first member's id: {first:?}"
    );
    assert_eq!(first, second, "ordering must be stable across calls");
}

#[test]
fn tied_groups_break_on_the_first_members_id_not_on_community_number() {
    // Community 1 holds the lexicographically LARGER id, community 9 the
    // smaller one, so a mutant that trusts community-number order instead
    // of breaking the tie on id gets this backwards on every run: no
    // dependence on Leiden's actual numbering or on hash order.
    let member = |id: &str| ClusterMember {
        id: NodeId::from_raw(id.to_string()),
        kind: "fact".to_string(),
        label: "x".to_string(),
    };
    let groups = group_and_sort_communities(vec![(1, member("z-member")), (9, member("a-member"))]);

    assert_eq!(groups[0][0].id.as_str(), "a-member", "{groups:?}");
    assert_eq!(groups[1][0].id.as_str(), "z-member", "{groups:?}");
}

#[tokio::test]
async fn every_member_carries_the_kind_and_label_recall_would_show() {
    let g = graph_at(Millis(1000)).await;
    let a = g
        .insert(NewNode::now("decision", "Adopt libSQL", "x"))
        .await
        .unwrap();
    let b = g
        .insert(NewNode::now("fact", "Metal enabled", "x"))
        .await
        .unwrap();
    g.relate(&a, &b, "mentions").await.unwrap();

    let groups = g.community_groups(true).await.unwrap();

    let members = &groups[0];
    assert!(members
        .iter()
        .any(|m| m.id == a && m.kind == "decision" && m.label == "Adopt libSQL"));
    assert!(members
        .iter()
        .any(|m| m.id == b && m.kind == "fact" && m.label == "Metal enabled"));
}

#[tokio::test]
async fn community_groups_refreshes_before_reading() {
    let (g, _clock, a, b) = seeded_pair(Millis(1000)).await;
    let _ = (&a, &b);

    let groups = g.community_groups(true).await.unwrap();

    assert!(
        !groups.is_empty(),
        "a store that has never clustered must still refresh first"
    );
}
