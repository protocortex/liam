// SPDX-License-Identifier: Apache-2.0
//! Writing edges with `relate` and `link`, and resolving node handles.

use super::*;

#[tokio::test]
async fn relate_writes_a_separate_row_per_relation_type() {
    // The write-side half: `relate`'s NOT EXISTS keys on the ordered triple
    // INCLUDING type, so a second type on the same pair is not a duplicate.
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();

    g.relate(&a, &b, "mentions").await.unwrap();
    g.relate(&a, &b, "causes").await.unwrap();

    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2",
            &[a.as_str().into(), b.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_i64(0).unwrap(), 2, "both types must be stored");
}

#[tokio::test]
async fn relate_stores_an_empty_json_object_as_attributes() {
    // Pins the claim behind sharing `EDGE_INSERT_SQL` with `ingest_episode`:
    // binding `relate`'s attributes as a parameter instead of hardcoding
    // the SQL literal `'{}'` must still store the same value.
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();

    let edge_id = g.relate(&a, &b, "mentions").await.unwrap();

    let rows = g
        .backend
        .query(
            "SELECT attributes FROM edges WHERE id = ?1",
            &[edge_id.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_string(0).unwrap(), "{}");
}

#[tokio::test]
async fn relate_refuses_an_endpoint_that_is_no_longer_live() {
    // The guarantee the unenforced foreign keys were supposed to give. The
    // check rides on the INSERT's own WHERE, so a `supersede` landing
    // between a caller's decision and this write still wins.
    let g = graph_at(Millis(1000)).await;
    let old = g
        .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
        .await
        .unwrap();
    let other = g.insert(NewNode::now("fact", "Other", "x")).await.unwrap();
    g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
        .await
        .unwrap();

    let err = g.relate(&old, &other, "mentions").await.unwrap_err();

    assert!(
        matches!(&err, Error::RelateRefused(m) if m.contains("source node")),
        "{err}"
    );
}

#[tokio::test]
async fn relate_refuses_a_destination_that_is_no_longer_live() {
    // The mirror of the test above, and not redundant: the two endpoints
    // are separate EXISTS subqueries binding separate parameters. Swapping
    // `?3` for `?2` in the second one is an easy slip in a statement that
    // mentions both four times, and with only the source case covered the
    // whole suite would stay green while `relate` wrote edges pointing at
    // superseded nodes.
    let g = graph_at(Millis(1000)).await;
    let other = g.insert(NewNode::now("fact", "Other", "x")).await.unwrap();
    let old = g
        .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
        .await
        .unwrap();
    g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
        .await
        .unwrap();

    let err = g.relate(&other, &old, "mentions").await.unwrap_err();

    assert!(
        matches!(&err, Error::RelateRefused(m) if m.contains("target node")),
        "{err}"
    );
    let rows = g
        .backend
        .query(
            "SELECT COUNT(*) FROM edges WHERE dst = ?1 AND type = ?2",
            &[old.as_str().into(), "mentions".into()],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_i64(0).unwrap(), 0, "no edge may have landed");
}

#[tokio::test]
async fn link_without_a_log_refuses_a_superseded_endpoint_and_a_repeated_triple() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let src = g.insert(NewNode::now("fact", "src", "x")).await.unwrap();
    let live = g.insert(NewNode::now("fact", "live", "x")).await.unwrap();
    let old = g
        .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
        .await
        .unwrap();
    g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
        .await
        .unwrap();
    let version_edges = support::count(&g, "edges").await;

    // Act
    let dead = g.link(NewEdge::new(&src, &old, "mentions")).await;
    let first = g.link(NewEdge::new(&src, &live, "mentions")).await;
    let repeated = g.link(NewEdge::new(&src, &live, "mentions")).await;

    // Assert
    assert!(
        matches!(&dead, Err(Error::RelateRefused(m)) if m.contains("not live")),
        "{dead:?}"
    );
    assert!(first.is_ok(), "{first:?}");
    assert!(
        matches!(&repeated, Err(Error::RelateRefused(m)) if m.contains("already relates")),
        "{repeated:?}"
    );
    assert_eq!(support::count(&g, "edges").await, version_edges + 1);
}

#[tokio::test]
async fn link_without_a_log_refuses_a_twin_with_other_attributes_and_names_only_the_keys() {
    // Arrange
    let g = graph_at(Millis(1000)).await;
    let src = g.insert(NewNode::now("fact", "src", "x")).await.unwrap();
    let dst = g.insert(NewNode::now("fact", "dst", "x")).await.unwrap();
    let weighted = |weight| {
        NewEdge::new(&src, &dst, "mentions").with_attributes(serde_json::json!({
            "weight": weight,
            "note": "same",
        }))
    };
    g.link(weighted(2)).await.unwrap();

    // Act
    let changed = g.link(weighted(731)).await;

    // Assert
    let Err(Error::RelateRefused(message)) = &changed else {
        panic!("expected a refusal: {changed:?}");
    };
    assert!(message.contains("already relates to"), "{message}");
    assert!(message.contains("weight"), "{message}");
    assert!(
        !message.contains("note"),
        "an unchanged key is not named: {message}"
    );
    assert!(
        !message.contains("731"),
        "a value is never named: {message}"
    );
    assert_eq!(support::count(&g, "edges").await, 1);
}

#[tokio::test]
async fn relate_is_idempotent_on_the_ordered_triple() {
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();

    g.relate(&a, &b, "mentions").await.unwrap();
    let err = g.relate(&a, &b, "mentions").await.unwrap_err();

    assert!(
        matches!(&err, Error::RelateRefused(m) if m.contains("already relates")),
        "{err}"
    );
    // The reverse direction is a different edge and must still land.
    g.relate(&b, &a, "mentions").await.unwrap();
}

#[tokio::test]
async fn a_handle_resolves_and_an_ambiguous_one_reports_its_candidates() {
    // The whole reason prefixes are acceptable where labels were not: a
    // collision is visible here, so the store refuses instead of guessing.
    // The refusal has to carry full ids, because a caller shown 13
    // characters cannot lengthen the handle on its own.
    let g = graph_at(Millis(1000)).await;
    let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
    let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();

    assert_eq!(g.resolve_handle(a.as_str()).await.unwrap(), a);
    assert_eq!(g.resolve_handle(a.handle()).await.unwrap(), a);
    // Crockford base32 is case-insensitive, GLOB is not.
    assert_eq!(
        g.resolve_handle(&a.handle().to_lowercase()).await.unwrap(),
        a
    );

    // Both ids start with `0`, so a one-character handle names neither.
    let err = g.resolve_handle("0").await.unwrap_err();
    let Error::AmbiguousHandle { candidates, .. } = &err else {
        panic!("expected an ambiguous handle, got {err}");
    };
    assert!(candidates.contains(&a.as_str().to_string()), "{err}");
    assert!(candidates.contains(&b.as_str().to_string()), "{err}");

    assert!(matches!(
        g.resolve_handle("ZZZZZZZZZZZZZ").await.unwrap_err(),
        Error::HandleNotFound(_)
    ));
}

#[tokio::test]
async fn a_wildcard_handle_never_resolves_even_with_one_live_node() {
    // With a single live node, an unfiltered `*` would GLOB-match it and
    // resolve successfully, handing back an arbitrary node the caller never
    // named. The alphabet check is what stops that.
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("fact", "only", "x")).await.unwrap();

    for handle in ["*", "?", "[0-9]*", "0*"] {
        assert!(
            matches!(
                g.resolve_handle(handle).await,
                Err(Error::HandleNotFound(_))
            ),
            "wildcard {handle:?} resolved"
        );
    }
}

#[tokio::test]
async fn a_superseded_node_is_not_reachable_by_handle() {
    let g = graph_at(Millis(1000)).await;
    let old = g
        .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
        .await
        .unwrap();
    g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
        .await
        .unwrap();

    assert!(matches!(
        g.resolve_handle(old.as_str()).await.unwrap_err(),
        Error::HandleNotFound(_)
    ));
}

#[test]
fn an_edge_refusal_with_every_guard_passing_is_a_retry_not_a_verdict_on_the_edge() {
    // Arrange
    let (src, dst) = (NodeId::new(), NodeId::new());

    // Act
    let error = edge_refusal(true, true, false, &src, &dst, "mentions");

    // Assert
    assert!(matches!(error, Error::ConcurrentWrite), "{error:?}");
}

#[test]
fn classify_relate_outcome_maps_an_insert_to_inserted() {
    // Arrange
    let result: Result<EdgeId> = Ok(EdgeId::new());

    // Act
    let outcome = classify_relate_outcome(&result);

    // Assert
    assert_eq!(outcome, EdgeOutcome::Inserted);
}

#[test]
fn classify_relate_outcome_maps_source_not_live_to_not_live() {
    let result: Result<EdgeId> = Err(Error::RelateRefused(
        "source node 01ARZ3NDEKTSV4RRFFQ69G5FAV is not live".to_string(),
    ));

    let outcome = classify_relate_outcome(&result);

    assert_eq!(outcome, EdgeOutcome::NotLive);
}

#[test]
fn classify_relate_outcome_maps_target_not_live_to_not_live() {
    let result: Result<EdgeId> = Err(Error::RelateRefused(
        "target node 01ARZ3NDEKTSV4RRFFQ69G5FAV is not live".to_string(),
    ));

    let outcome = classify_relate_outcome(&result);

    assert_eq!(outcome, EdgeOutcome::NotLive);
}

#[test]
fn classify_relate_outcome_maps_a_duplicate_triple_to_already_present() {
    let result: Result<EdgeId> = Err(Error::RelateRefused(
        "01ARZ3NDEKTSV4RRFFQ69G5FAV already relates to 01BX5ZZKBKACTAV9WEVGEMMVRZ as 'mentions'"
            .to_string(),
    ));

    let outcome = classify_relate_outcome(&result);

    assert_eq!(outcome, EdgeOutcome::AlreadyPresent);
}

#[test]
fn classify_relate_outcome_maps_a_concurrent_write_retry_to_real_failure() {
    // Every guard passed on the diagnostic re-read, so something else
    // flipped between the insert and the diagnostic: not a safe no-op.
    let result: Result<EdgeId> = Err(Error::RelateRefused(
        "a concurrent write took the row, retry".to_string(),
    ));

    let outcome = classify_relate_outcome(&result);

    assert_eq!(outcome, EdgeOutcome::RealFailure);
}

#[test]
fn classify_relate_outcome_maps_a_non_relate_refused_error_to_real_failure() {
    let result: Result<EdgeId> = Err(Error::Backend("connection reset".to_string()));

    let outcome = classify_relate_outcome(&result);

    assert_eq!(outcome, EdgeOutcome::RealFailure);
}
