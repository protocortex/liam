// SPDX-License-Identifier: Apache-2.0
//! `clusters` and `timeline`.

use super::*;

#[tokio::test]
async fn clusters_on_an_empty_store_says_so_and_does_not_panic() {
    let server = plain_server().await;

    let out = server
        .clusters(Parameters(ClustersArgs {
            k: None,
            members: None,
            mode: None,
        }))
        .await;

    assert_eq!(out, "no clusters yet");
}

#[tokio::test]
async fn clusters_budget_is_a_fraction_of_the_configured_context_not_all_of_it() {
    // Enough groups, each with a label long enough to matter, that the
    // real (one tenth) budget forces truncation while the full 8192
    // context would not: catches the handler passing the whole context
    // through un-scaled.
    let server = plain_server().await;
    let pad = "x".repeat(300);
    for i in 0..15 {
        let a = seed(&server, "fact", &format!("group{i}a {pad}"), "x").await;
        let b = seed(&server, "fact", &format!("group{i}b {pad}"), "x").await;
        server
            .relate(Parameters(RelateArgs {
                from: a,
                to: b,
                kind: "mentions".to_string(),
            }))
            .await;
    }

    let out = server
        .clusters(Parameters(ClustersArgs {
            k: None,
            members: None,
            mode: None,
        }))
        .await;

    assert!(out.contains("withheld"), "{out}");
}

#[tokio::test]
async fn clusters_reports_a_group_for_two_related_memories() {
    let server = plain_server().await;
    let a = seed(&server, "fact", "Rollout plan", "x").await;
    let b = seed(&server, "fact", "Rollout owner", "y").await;
    server
        .relate(Parameters(RelateArgs {
            from: a.clone(),
            to: b.clone(),
            kind: "mentions".to_string(),
        }))
        .await;

    let out = server
        .clusters(Parameters(ClustersArgs {
            k: None,
            members: None,
            mode: None,
        }))
        .await;

    assert!(out.contains("Rollout plan"), "{out}");
    assert!(out.contains("Rollout owner"), "{out}");
}

#[tokio::test]
async fn a_handle_rendered_by_clusters_resolves_through_relate() {
    // `clusters` and `recall` must resolve through the same handle
    // scheme, so a client acting on what `clusters` shows does not need
    // a second round trip through `recall` first.
    let server = plain_server().await;
    let a = seed(&server, "fact", "Rollout plan", "first").await;
    let b = seed(&server, "fact", "Rollout owner", "second").await;
    let c = seed(&server, "fact", "Unrelated", "third").await;
    let related = server
        .relate(Parameters(RelateArgs {
            from: a.clone(),
            to: b.clone(),
            kind: "mentions".to_string(),
        }))
        .await;
    assert!(related.starts_with("related "), "{related}");

    let rendered = server
        .clusters(Parameters(ClustersArgs {
            k: None,
            members: None,
            mode: None,
        }))
        .await;
    let handle = rendered
        .lines()
        .find_map(|l| l.strip_prefix("[fact "))
        .and_then(|rest| rest.split(']').next())
        .unwrap_or_else(|| panic!("no member line with a handle: {rendered}"))
        .to_string();

    let linked = server
        .relate(Parameters(RelateArgs {
            from: handle,
            to: c.clone(),
            kind: "mentions".to_string(),
        }))
        .await;
    assert!(
        linked.starts_with("related "),
        "handle did not resolve: {linked}"
    );
}

#[tokio::test]
async fn clusters_is_registered_with_the_argument_names_clients_send() {
    let server = plain_server().await;

    assert!(
        server.tool_router.has_route("clusters"),
        "clusters not routed"
    );
    let tool = server
        .tool_router
        .list_all()
        .into_iter()
        .find(|t| t.name == "clusters")
        .expect("clusters must be listed");
    let schema = serde_json::to_string(&tool.input_schema).unwrap();
    for field in ["\"k\"", "\"members\""] {
        assert!(schema.contains(field), "{field} missing from {schema}");
    }
}

#[tokio::test]
async fn timeline_returns_compiled_content_and_mentions_most_recent_first() {
    // Given an entity with a compiled page and 2 live mentions, linked
    // at two different instants
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock.clone()).await;
    let mut entity_node = NewNode::entity("person", "Ada Lovelace")
        .with_attributes(crate::entity_page::stamped(json!({})));
    entity_node.content = "compiled bio of ada".to_string();
    let entity_id = server
        .store()
        .await
        .unwrap()
        .insert(entity_node)
        .await
        .expect("seed entity");
    let fact1 = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now("fact", "Fact One", "first mention content"))
        .await
        .expect("seed fact one");
    let fact2 = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now("fact", "Fact Two", "second mention content"))
        .await
        .expect("seed fact two");
    clock.set(Millis(2000));
    server
        .store()
        .await
        .unwrap()
        .relate(&entity_id, &fact1, relation::MENTIONS)
        .await
        .expect("link fact one");
    clock.set(Millis(3000));
    server
        .store()
        .await
        .unwrap()
        .relate(&entity_id, &fact2, relation::MENTIONS)
        .await
        .expect("link fact two");

    // When timeline runs with no caller-supplied limit
    let out = server
        .timeline(Parameters(TimelineArgs {
            entity: entity_id.as_str().to_string(),
            limit: None,
            mode: None,
        }))
        .await;

    // Then the compiled content and both mentions appear, most
    // recently linked mention first, with no "more" trailer
    assert!(out.contains("compiled bio of ada"), "{out}");
    let fact1_pos = out.find("first mention content").expect("fact one missing");
    let fact2_pos = out
        .find("second mention content")
        .expect("fact two missing");
    assert!(
        fact2_pos < fact1_pos,
        "most recently linked mention should come first: {out}"
    );
    assert!(!out.contains("more mentions exist"), "{out}");
}

#[tokio::test]
async fn timeline_reports_an_unresolvable_handle_as_not_found() {
    // Given a handle that resolves to no node
    let server = plain_server().await;

    // When timeline runs
    let out = server
        .timeline(Parameters(TimelineArgs {
            entity: "0000000000000".to_string(),
            limit: None,
            mode: None,
        }))
        .await;

    // Then it fails clearly instead of panicking or succeeding empty
    assert!(out.starts_with("timeline failed:"), "{out}");
    assert!(out.contains("no live node"), "{out}");
}

#[tokio::test]
async fn timeline_reports_not_live_for_a_node_with_a_future_valid_from() {
    // Given a resolvable handle whose node isn't valid-time-live yet
    let server = plain_server().await;
    let one_day_ms = 24 * 60 * 60 * 1000;
    let future_entity = NewNode::entity("person", "Future Person")
        .with_valid_from(Millis(Millis::now().0 + one_day_ms));
    let entity_id = server
        .store()
        .await
        .unwrap()
        .insert(future_entity)
        .await
        .expect("seed future entity");

    // When timeline runs
    let out = server
        .timeline(Parameters(TimelineArgs {
            entity: entity_id.as_str().to_string(),
            limit: None,
            mode: None,
        }))
        .await;

    // Then it reports the node as not currently live, not a crash
    // or an unresolvable-handle error
    assert_eq!(
        out,
        format!(
            "timeline failed: {} is not currently live",
            entity_id.handle()
        )
    );
}

#[tokio::test]
async fn timeline_on_an_entity_with_no_mentions_says_so() {
    // Given an entity created directly, with no mentions edges at all
    let server = plain_server().await;
    let entity_id = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::entity("person", "Lonely Entity"))
        .await
        .expect("seed entity");

    // When timeline runs
    let out = server
        .timeline(Parameters(TimelineArgs {
            entity: entity_id.as_str().to_string(),
            limit: None,
            mode: None,
        }))
        .await;

    // Then it returns the entity's own data with an explicit
    // "no mentions yet" section, not an error
    assert!(out.contains("Lonely Entity"), "{out}");
    assert!(out.contains("no mentions yet"), "{out}");
    assert!(!out.starts_with("timeline failed:"), "{out}");
}

#[tokio::test]
async fn timeline_on_a_non_entity_node_returns_its_own_data_without_crashing() {
    // Given a handle that resolves to a live node that was never an
    // entity (an ordinary remembered fact)
    let server = plain_server().await;
    let handle = seed(&server, "fact", "Ordinary Fact", "ordinary content").await;

    // When timeline runs
    let out = server
        .timeline(Parameters(TimelineArgs {
            entity: handle,
            limit: None,
            mode: None,
        }))
        .await;

    // Then it still returns that node's own data and an empty
    // mentions list, without crashing
    assert!(out.contains("Ordinary Fact"), "{out}");
    assert!(out.contains("ordinary content"), "{out}");
    assert!(out.contains("no mentions yet"), "{out}");
}

#[test]
fn clamp_timeline_limit_defaults_and_bounds_the_mention_count() {
    // Arrange / Act / Assert
    assert_eq!(
        clamp_timeline_limit(None),
        DEFAULT_TIMELINE_LIMIT,
        "absent limit defaults"
    );
    assert_eq!(
        clamp_timeline_limit(Some(10)),
        10,
        "in-range limit passes through"
    );
    assert_eq!(
        clamp_timeline_limit(Some(MAX_TIMELINE_LIMIT)),
        MAX_TIMELINE_LIMIT,
        "the cap itself is allowed"
    );
    assert_eq!(
        clamp_timeline_limit(Some(10_000)),
        MAX_TIMELINE_LIMIT,
        "oversized limit is capped, not passed to the store"
    );
    assert_eq!(
        clamp_timeline_limit(Some(0)),
        1,
        "limit=0 shows one mention, not none"
    );
}

#[tokio::test]
async fn timeline_truncates_to_the_caller_supplied_limit_and_appends_a_trailer() {
    // Given an entity with 3 live mentions, linked at three different
    // instants, and a caller-supplied limit of 2
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock.clone()).await;
    let entity_id = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::entity("person", "Ada Lovelace"))
        .await
        .expect("seed entity");
    let oldest = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now(
            "fact",
            "Fact Oldest",
            "oldest mention content",
        ))
        .await
        .expect("seed oldest fact");
    let middle = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now(
            "fact",
            "Fact Middle",
            "middle mention content",
        ))
        .await
        .expect("seed middle fact");
    let newest = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now(
            "fact",
            "Fact Newest",
            "newest mention content",
        ))
        .await
        .expect("seed newest fact");
    clock.set(Millis(2000));
    server
        .store()
        .await
        .unwrap()
        .relate(&entity_id, &oldest, relation::MENTIONS)
        .await
        .expect("link oldest fact");
    clock.set(Millis(3000));
    server
        .store()
        .await
        .unwrap()
        .relate(&entity_id, &middle, relation::MENTIONS)
        .await
        .expect("link middle fact");
    clock.set(Millis(4000));
    server
        .store()
        .await
        .unwrap()
        .relate(&entity_id, &newest, relation::MENTIONS)
        .await
        .expect("link newest fact");

    // When timeline runs with limit: Some(2)
    let out = server
        .timeline(Parameters(TimelineArgs {
            entity: entity_id.as_str().to_string(),
            limit: Some(2),
            mode: None,
        }))
        .await;

    // Then exactly the 2 most recently linked mentions show, followed
    // by the trailer, and the excluded oldest mention is absent
    assert!(out.contains("newest mention content"), "{out}");
    assert!(out.contains("middle mention content"), "{out}");
    assert!(!out.contains("oldest mention content"), "{out}");
    assert!(out.contains("more mentions exist"), "{out}");
}

#[tokio::test]
async fn timeline_shows_all_mentions_with_no_trailer_when_count_equals_the_limit() {
    // Given an entity with exactly 2 live mentions, linked at two
    // different instants, and a caller-supplied limit of 2
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock.clone()).await;
    let entity_id = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::entity("person", "Ada Lovelace"))
        .await
        .expect("seed entity");
    let fact1 = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now("fact", "Fact One", "first mention content"))
        .await
        .expect("seed fact one");
    let fact2 = server
        .store()
        .await
        .unwrap()
        .insert(NewNode::now("fact", "Fact Two", "second mention content"))
        .await
        .expect("seed fact two");
    clock.set(Millis(2000));
    server
        .store()
        .await
        .unwrap()
        .relate(&entity_id, &fact1, relation::MENTIONS)
        .await
        .expect("link fact one");
    clock.set(Millis(3000));
    server
        .store()
        .await
        .unwrap()
        .relate(&entity_id, &fact2, relation::MENTIONS)
        .await
        .expect("link fact two");

    // When timeline runs with limit: Some(2)
    let out = server
        .timeline(Parameters(TimelineArgs {
            entity: entity_id.as_str().to_string(),
            limit: Some(2),
            mode: None,
        }))
        .await;

    // Then both mentions show and no trailer appears (the "+1"
    // over-fetch comparison is strictly greater-than, not >=)
    assert!(out.contains("first mention content"), "{out}");
    assert!(out.contains("second mention content"), "{out}");
    assert!(!out.contains("more mentions exist"), "{out}");
}

#[tokio::test]
async fn timeline_is_registered_with_the_argument_names_clients_send() {
    let server = plain_server().await;

    assert!(
        server.tool_router.has_route("timeline"),
        "timeline not routed"
    );
    let tool = server
        .tool_router
        .list_all()
        .into_iter()
        .find(|t| t.name == "timeline")
        .expect("timeline must be listed");
    let schema = serde_json::to_string(&tool.input_schema).unwrap();
    assert!(
        schema.contains("\"entity\""),
        "entity missing from {schema}"
    );
}
