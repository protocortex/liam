// SPDX-License-Identifier: Apache-2.0
//! Upsert by subject, scope rules, and the producer of a write.

use super::*;

#[tokio::test]
async fn upsert_by_supersedes_same_subject() {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    g.upsert_by(NewNode::now("fact", "v1", "price is 10").with_subject("price"))
        .await
        .unwrap();
    clock.set(Millis(2000));
    g.upsert_by(NewNode::now("fact", "v2", "price is 20").with_subject("price"))
        .await
        .unwrap();

    let hits = g.query(&Query::text("price")).await.unwrap();
    assert_eq!(hits.len(), 1, "only the current version is live");
    assert_eq!(hits[0].label, "v2");
}

#[tokio::test]
async fn upsert_by_supersedes_newest_competitor_deterministically() {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    // Two live nodes share a subject (via `insert`, which does not dedup).
    g.insert(
        NewNode::now("fact", "old", "price 10")
            .with_subject("price")
            .with_valid_from(Millis(1000)),
    )
    .await
    .unwrap();
    clock.set(Millis(2000));
    g.insert(
        NewNode::now("fact", "newer", "price 15")
            .with_subject("price")
            .with_valid_from(Millis(2000)),
    )
    .await
    .unwrap();
    // upsert_by supersedes the newest competitor, deterministically.
    clock.set(Millis(3000));
    g.upsert_by(
        NewNode::now("fact", "newest", "price 20")
            .with_subject("price")
            .with_valid_from(Millis(3000)),
    )
    .await
    .unwrap();

    let hits = g.query(&Query::text("price").with_k(10)).await.unwrap();
    let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
    assert!(labels.contains(&"newest"), "new version is live");
    assert!(
        !labels.contains(&"newer"),
        "newest competitor was superseded"
    );
    assert!(labels.contains(&"old"), "the older competitor is untouched");
}

#[tokio::test]
async fn insert_trims_scope_before_storing() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    g.insert(NewNode::now("decision", "Rollout", "zenith rollout notes").with_scope(" proj-a "))
        .await
        .unwrap();

    // Assert: found under the trimmed scope, not the raw untrimmed value
    let hits = g
        .query(&Query::text("zenith rollout").with_scope("proj-a"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].label, "Rollout");
}

#[tokio::test]
async fn insert_rejects_whitespace_only_scope() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let result = g
        .insert(NewNode::now("decision", "Bad", "whitespace scope body").with_scope("   "))
        .await;

    // Assert: rejected, and no row written
    assert!(matches!(result, Err(Error::InvalidScope(_))));
    let hits = g
        .query(&Query::text("whitespace scope body"))
        .await
        .unwrap();
    assert!(hits.is_empty(), "invalid scope must not reach storage");
}

#[tokio::test]
async fn insert_rejects_scope_with_embedded_space() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let result = g
        .insert(NewNode::now("decision", "Bad", "body").with_scope("proj a"))
        .await;

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[tokio::test]
async fn insert_rejects_malformed_leading_slash_scope() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let result = g
        .insert(NewNode::now("decision", "Bad", "body").with_scope("/bad"))
        .await;

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[tokio::test]
async fn upsert_by_normalizes_scope_before_finding_live_competitor() {
    // Arrange: the design doc's motivating bug, a live node under a
    // trimmed scope, then an update that names the same scope with
    // trailing whitespace.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    g.upsert_by(
        NewNode::now("fact", "v1", "widget price is 10")
            .with_subject("s1")
            .with_scope("proj-a"),
    )
    .await
    .unwrap();

    // Act
    clock.set(Millis(2000));
    g.upsert_by(
        NewNode::now("fact", "v2", "widget price is 20")
            .with_subject("s1")
            .with_scope("proj-a "),
    )
    .await
    .unwrap();

    // Assert: exactly one live node with subject s1, the old one superseded
    let hits = g
        .query(&Query::text("widget price").with_k(10))
        .await
        .unwrap();
    assert_eq!(
        hits.len(),
        1,
        "trailing-space scope must match the trimmed live node, not duplicate it"
    );
    assert_eq!(hits[0].label, "v2");

    // Assert: the stored scope is trimmed
    let scoped = g
        .query(&Query::text("widget price").with_scope("proj-a"))
        .await
        .unwrap();
    assert_eq!(scoped.len(), 1);
    assert_eq!(scoped[0].label, "v2");
}

#[tokio::test]
async fn upsert_by_supersedes_scoped_entity_rebuilt_from_candidate_scope() {
    // Arrange: a scoped entity, matching what `resynthesize_entity`
    // (crates/daemon/src/mcp.rs, a separate later fix) will read via
    // `Graph::get` and rebuild.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let original = g
        .upsert_by(
            NewNode::now("person", "kim", "profile v1")
                .with_subject("kim")
                .with_scope("project-a"),
        )
        .await
        .unwrap();

    // Act: read it back and rebuild a `NewNode` from the `Candidate`,
    // carrying its scope forward, then upsert again.
    clock.set(Millis(2000));
    let candidate = g
        .get(&original, Millis(2000))
        .await
        .unwrap()
        .expect("live node");
    let mut rebuilt = NewNode::now(candidate.kind, "kim", "profile v2").with_subject("kim");
    if let Some(scope) = candidate.scope.clone() {
        rebuilt = rebuilt.with_scope(scope);
    }
    g.upsert_by(rebuilt).await.unwrap();

    // Assert: the rebuild superseded the original scoped entity instead of
    // forking an unscoped duplicate.
    let live = g
        .query(&Query::text("kim").with_scope("project-a").with_k(10))
        .await
        .unwrap();
    assert_eq!(
        live.len(),
        1,
        "resynthesis must supersede the scoped entity, not fork a duplicate"
    );
    assert_eq!(live[0].retrieved.as_text(), "profile v2");
}

#[tokio::test]
async fn upsert_by_scope_collision_stays_exact_match_after_hierarchy_change() {
    // Arrange: a live node at the ancestor scope "work".
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let root = g
        .upsert_by(
            NewNode::now("fact", "v1", "widget price is 10")
                .with_subject("s1")
                .with_scope("work"),
        )
        .await
        .unwrap();

    // Act: a write with the same subject to the descendant scope
    // "work/liam".
    clock.set(Millis(2000));
    let child = g
        .upsert_by(
            NewNode::now("fact", "v2", "widget price is 20")
                .with_subject("s1")
                .with_scope("work/liam"),
        )
        .await
        .unwrap();

    // Assert: the descendant-scope write did not supersede the
    // ancestor-scope node. Write-side collision detection stays
    // exact-match, unaffected by the read-side scope hierarchy.
    assert!(
        g.exists_as_of(&root, Millis(2000)).await.unwrap(),
        "the ancestor-scoped node must still be live"
    );

    // Act: a write with the same subject to the ancestor scope "work"
    // must equally not supersede the descendant-scoped node.
    clock.set(Millis(3000));
    let _grandchild = g
        .upsert_by(
            NewNode::now("fact", "v3", "widget price is 30")
                .with_subject("s1")
                .with_scope("work"),
        )
        .await
        .unwrap();

    // Assert: the descendant-scoped node is still live. A hierarchical
    // write predicate would wrongly let this ancestor-scoped write
    // supersede it.
    assert!(
        g.exists_as_of(&child, Millis(3000)).await.unwrap(),
        "the descendant-scoped node must still be live after an ancestor-scope write"
    );
}

#[tokio::test]
async fn supersede_rejects_invalid_scope_and_normalizes_valid_scope() {
    // Arrange: a live node to supersede directly, matching the
    // direct-call usage shape at `as_of_recovers_superseded_history`.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old = g
        .insert(NewNode::now("decision", "Deno", "runtime"))
        .await
        .unwrap();

    // Act: an invalid scope on the direct supersede call
    clock.set(Millis(2000));
    let invalid = g
        .supersede(
            &old,
            NewNode::now("decision", "Rust", "runtime").with_scope("bad@scope"),
        )
        .await;

    // Assert: rejected, pinning that `validate_scope` runs inside
    // `supersede` itself, independent of `upsert_by`'s own normalization
    assert!(matches!(invalid, Err(Error::InvalidScope(_))));

    // Act: a valid but untrimmed scope on the direct supersede call
    let new_id = g
        .supersede(
            &old,
            NewNode::now("decision", "Rust", "runtime").with_scope(" proj-a "),
        )
        .await
        .unwrap();

    // Assert: the stored scope is trimmed
    let hits = g
        .query(&Query::text("runtime").with_scope("proj-a"))
        .await
        .unwrap();
    assert!(hits.iter().any(|h| h.id == new_id));
}

#[test]
fn new_node_entity_sets_kind_label_subject() {
    let n = NewNode::entity("person", "  Ada Lovelace ");
    assert_eq!(n.kind, "person");
    assert_eq!(n.label, "  Ada Lovelace ");
    assert_eq!(n.subject.as_deref(), Some("ada lovelace"));
}

#[test]
fn validate_scope_passes_none_through_unchanged() {
    // Arrange
    let scope: Option<String> = None;

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert_eq!(result.unwrap(), None);
}

#[test]
fn validate_scope_rejects_whitespace_only_scope() {
    // Arrange
    let scope = Some("  ".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[test]
fn validate_scope_trims_surrounding_whitespace() {
    // Arrange
    let scope = Some("  proj-a  ".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert_eq!(result.unwrap(), Some("proj-a".to_string()));
}

#[test]
fn validate_scope_accepts_exactly_max_chars() {
    // Arrange
    let scope = Some("a".repeat(MAX_SCOPE_CHARS));

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(result.is_ok());
}

#[test]
fn validate_scope_rejects_scope_over_max_chars() {
    // Arrange
    let scope = Some("a".repeat(MAX_SCOPE_CHARS + 1));

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[test]
fn validate_scope_rejects_embedded_space() {
    // Arrange
    let scope = Some("proj a".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[test]
fn validate_scope_rejects_non_ascii_alphanumeric_character() {
    // Arrange
    let scope = Some("proj@a".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[test]
fn validate_scope_rejects_leading_slash() {
    // Arrange
    let scope = Some("/proj-a".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[test]
fn validate_scope_rejects_trailing_slash() {
    // Arrange
    let scope = Some("proj-a/".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[test]
fn validate_scope_rejects_empty_segment() {
    // Arrange
    let scope = Some("proj//a".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[test]
fn validate_scope_accepts_valid_multi_segment_scope() {
    // Arrange
    let scope = Some("project-x/backend".to_string());

    // Act
    let result = validate_scope(&scope);

    // Assert
    assert_eq!(result.unwrap(), Some("project-x/backend".to_string()));
}

#[tokio::test]
async fn entity_mentions_edge_round_trips() {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let person = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g
        .insert(NewNode::now(
            "fact",
            "note",
            "Ada wrote the first algorithm",
        ))
        .await
        .unwrap();
    g.link(NewEdge::new(
        &person,
        &fact,
        crate::types::relation::MENTIONS,
    ))
    .await
    .unwrap();

    // `Graph::neighbors` traverses edges in either direction without
    // filtering by type, so it can't isolate a single relation. Assert
    // directly against the backend that the MENTIONS edge row exists.
    let rows = g
        .backend
        .query(
            "SELECT dst FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3",
            &[
                person.as_str().into(),
                fact.as_str().into(),
                crate::types::relation::MENTIONS.into(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(ids_from(&rows).unwrap(), vec![fact]);
}

#[tokio::test]
async fn query_rejects_invalid_scope() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let result = g
        .query(&Query::text("zenith rollout").with_scope("bad scope!"))
        .await;

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[tokio::test]
async fn query_explained_rejects_invalid_scope() {
    // Arrange
    let g = graph_at(Millis(1000)).await;

    // Act
    let result = g
        .query_explained(&Query::text("zenith rollout").with_scope("bad scope!"))
        .await;

    // Assert
    assert!(matches!(result, Err(Error::InvalidScope(_))));
}

#[tokio::test]
async fn query_trims_scope_before_filtering() {
    // Arrange: a node stored under an already-normalized scope, matching
    // the write side's trimming (`insert` runs `validate_scope` too).
    let g = graph_at(Millis(1000)).await;
    g.insert(NewNode::now("decision", "Rollout", "zenith rollout notes").with_scope("proj-a"))
        .await
        .unwrap();

    // Act: query with an untrimmed scope.
    let hits = g
        .query(&Query::text("zenith rollout").with_scope(" proj-a "))
        .await
        .unwrap();

    // Assert: the read side trims before filtering, so it still matches.
    assert!(hits.iter().any(|h| h.label == "Rollout"));
}

#[tokio::test]
async fn query_explained_carries_valid_from() {
    // The daemon's `ask` tool renders `valid_from` as a date, so
    // `ExplainedHit` must carry the raw instant, not just the decay
    // computed from it.
    // Arrange
    let g = graph_at(Millis(5000)).await;
    g.insert(NewNode::now("fact", "label", "body").with_valid_from(Millis(1234)))
        .await
        .unwrap();

    // Act
    let hits = g.query_explained(&Query::text("body")).await.unwrap();

    // Assert
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].valid_from, Millis(1234));
}

#[tokio::test]
async fn producer_round_trips_through_insert() {
    // Given a node inserted with a producer
    let g = graph_at(Millis(1000)).await;
    let id = g
        .insert(NewNode::now("fact", "label", "content").with_producer("agent-a"))
        .await
        .unwrap();

    // When read back, then the producer round-trips. `producer` is
    // deliberately absent from `Hit`, so query the `nodes` table
    // directly through the backend rather than through `query`.
    let rows = g
        .backend
        .query(
            "SELECT producer FROM nodes WHERE id = ?1",
            &[id.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_string(0).unwrap(), "agent-a");
}

#[tokio::test]
async fn producer_defaults_to_unknown_when_not_specified() {
    // Given a node inserted without specifying a producer
    let g = graph_at(Millis(1000)).await;
    let id = g
        .insert(NewNode::now("fact", "label", "content"))
        .await
        .unwrap();

    // When read back, then it is "unknown" rather than empty or null.
    let rows = g
        .backend
        .query(
            "SELECT producer FROM nodes WHERE id = ?1",
            &[id.as_str().into()],
        )
        .await
        .unwrap();
    assert_eq!(rows[0].get_string(0).unwrap(), "unknown");
}

#[tokio::test]
async fn upsert_by_carries_producer_through_supersede() {
    // Given a live node from producer A
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old_id = g
        .upsert_by(
            NewNode::now("fact", "v1", "price is 10")
                .with_subject("price")
                .with_producer("agent-a"),
        )
        .await
        .unwrap();

    // When producer B writes the same subject and scope via upsert_by
    clock.set(Millis(2000));
    let new_id = g
        .upsert_by(
            NewNode::now("fact", "v2", "price is 20")
                .with_subject("price")
                .with_producer("agent-b"),
        )
        .await
        .unwrap();

    // Then both versions stay live, each still recording its own writer:
    // a fact collides on (subject, scope, producer), so different
    // producers never compete for the same subject.
    let rows = g
        .backend
        .query(
            "SELECT id, producer, tx_to FROM nodes WHERE id IN (?1, ?2)",
            &[old_id.as_str().into(), new_id.as_str().into()],
        )
        .await
        .unwrap();
    let row_of = |id: &str| -> (String, i64) {
        let r = rows
            .iter()
            .find(|r| r.get_string(0).unwrap() == id)
            .unwrap();
        (r.get_string(1).unwrap(), r.get_i64(2).unwrap())
    };
    let (old_producer, old_tx_to) = row_of(old_id.as_str());
    let (new_producer, new_tx_to) = row_of(new_id.as_str());
    assert_eq!(old_producer, "agent-a");
    assert_eq!(new_producer, "agent-b");
    assert_eq!(old_tx_to, FOREVER.0, "producer agent-a's node stays live");
    assert_eq!(new_tx_to, FOREVER.0, "producer agent-b's node stays live");
}

#[tokio::test]
async fn upsert_by_keeps_both_producers_live_for_same_subject_and_scope() {
    // Arrange: a live node with subject "price", producer "agent-a"
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let a_id = g
        .upsert_by(
            NewNode::now("fact", "v1", "price is 10")
                .with_subject("price")
                .with_producer("agent-a"),
        )
        .await
        .unwrap();

    // Act: producer "agent-b" writes subject "price" in the same scope
    clock.set(Millis(2000));
    let b_id = g
        .upsert_by(
            NewNode::now("fact", "v2", "price is 20")
                .with_subject("price")
                .with_producer("agent-b"),
        )
        .await
        .unwrap();

    // Assert: both nodes remain live (neither superseded)
    let rows = g
        .backend
        .query(
            "SELECT id, tx_to FROM nodes WHERE id IN (?1, ?2)",
            &[a_id.as_str().into(), b_id.as_str().into()],
        )
        .await
        .unwrap();
    let tx_to_of = |id: &str| -> i64 {
        rows.iter()
            .find(|r| r.get_string(0).unwrap() == id)
            .unwrap()
            .get_i64(1)
            .unwrap()
    };
    assert_eq!(
        tx_to_of(a_id.as_str()),
        FOREVER.0,
        "producer agent-a's node stays live"
    );
    assert_eq!(
        tx_to_of(b_id.as_str()),
        FOREVER.0,
        "producer agent-b's node stays live"
    );
}

#[tokio::test]
async fn upsert_by_supersedes_entity_pages_across_producers() {
    // Arrange: a live entity page for "Alice" from producer "agent-a"
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let a_id = g
        .upsert_by(NewNode::entity("person", "Alice").with_producer("agent-a"))
        .await
        .unwrap();

    // Act: producer "agent-b" writes the same entity page in the same scope
    clock.set(Millis(2000));
    let b_id = g
        .upsert_by(NewNode::entity("person", "Alice").with_producer("agent-b"))
        .await
        .unwrap();

    // Assert: the second write supersedes the first, regardless of producer
    let rows = g
        .backend
        .query(
            "SELECT id, tx_to FROM nodes WHERE id IN (?1, ?2)",
            &[a_id.as_str().into(), b_id.as_str().into()],
        )
        .await
        .unwrap();
    let tx_to_of = |id: &str| -> i64 {
        rows.iter()
            .find(|r| r.get_string(0).unwrap() == id)
            .unwrap()
            .get_i64(1)
            .unwrap()
    };
    assert_ne!(
        tx_to_of(a_id.as_str()),
        FOREVER.0,
        "producer agent-a's page is superseded"
    );
    assert_eq!(
        tx_to_of(b_id.as_str()),
        FOREVER.0,
        "producer agent-b's page stays live"
    );
}

#[tokio::test]
async fn upsert_by_supersedes_when_same_producer_writes_twice() {
    // Arrange: a live node with subject "price", producer "agent-a"
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let old_id = g
        .upsert_by(
            NewNode::now("fact", "v1", "price is 10")
                .with_subject("price")
                .with_producer("agent-a"),
        )
        .await
        .unwrap();

    // Act: the same producer "agent-a" writes subject "price" again
    clock.set(Millis(2000));
    let new_id = g
        .upsert_by(
            NewNode::now("fact", "v2", "price is 20")
                .with_subject("price")
                .with_producer("agent-a"),
        )
        .await
        .unwrap();

    // Assert: the second write supersedes the first
    let rows = g
        .backend
        .query(
            "SELECT id, tx_to FROM nodes WHERE id IN (?1, ?2)",
            &[old_id.as_str().into(), new_id.as_str().into()],
        )
        .await
        .unwrap();
    let tx_to_of = |id: &str| -> i64 {
        rows.iter()
            .find(|r| r.get_string(0).unwrap() == id)
            .unwrap()
            .get_i64(1)
            .unwrap()
    };
    assert_ne!(
        tx_to_of(old_id.as_str()),
        FOREVER.0,
        "the first agent-a write is superseded"
    );
    assert_eq!(
        tx_to_of(new_id.as_str()),
        FOREVER.0,
        "the second agent-a write stays live"
    );
}
