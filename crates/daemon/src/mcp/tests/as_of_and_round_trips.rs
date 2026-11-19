// SPDX-License-Identifier: Apache-2.0
//! Reads as of a past instant, and what `remember` writes read back through the tools.

use super::*;

#[tokio::test]
async fn recall_as_of_returns_only_the_version_live_at_that_instant() {
    // Arrange: two versions of one subject, written at two clock instants.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock.clone()).await;
    remember_version(&server, "v1", "zorbnax price is 10").await;
    clock.set(Millis(2000));
    remember_version(&server, "v2", "zorbnax price is 20").await;

    // Act: as_of pinned to the first write's instant.
    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax price".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: Some(1000),
            mode: None,
        }))
        .await;

    // Assert: only the first version was live at t0.
    assert!(out.contains("] v1"), "missing first version: {out}");
    assert!(!out.contains("] v2"), "second version leaked: {out}");
}

#[tokio::test]
async fn recall_as_of_before_the_first_write_returns_no_hits() {
    // Arrange: one node written at t0.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock).await;
    remember_version(&server, "v1", "zorbnax price is 10").await;

    // Act: as_of set before the subject existed at all.
    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax price".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: Some(500),
            mode: None,
        }))
        .await;

    // Assert
    assert_eq!(out, "no relevant memory", "expected zero hits: {out}");
}

#[tokio::test]
async fn recall_as_of_after_all_writes_returns_only_the_current_version() {
    // Arrange: two versions of one subject.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock.clone()).await;
    remember_version(&server, "v1", "zorbnax price is 10").await;
    clock.set(Millis(2000));
    remember_version(&server, "v2", "zorbnax price is 20").await;

    // Act: as_of well after both writes.
    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax price".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: Some(3000),
            mode: None,
        }))
        .await;

    // Assert: exactly one hit, and it is the current version, not the
    // superseded one this pins exclusion of.
    assert_eq!(out.split("\n\n").count(), 1, "expected 1 block: {out}");
    assert!(out.contains("] v2"), "missing current version: {out}");
    assert!(!out.contains("] v1"), "superseded version leaked: {out}");
}

#[tokio::test]
async fn recall_as_of_exactly_at_the_second_writes_valid_from_is_live() {
    // `live_at` uses `valid_from <= t` and `tx_from <= t`, both
    // inclusive, so as_of pinned to the exact instant of a write already
    // sees that write; see `live_at` in `liam-store`'s `graph.rs`.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock.clone()).await;
    remember_version(&server, "v1", "zorbnax price is 10").await;
    clock.set(Millis(2000));
    remember_version(&server, "v2", "zorbnax price is 20").await;

    // Act: as_of pinned exactly to the second write's instant.
    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax price".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: Some(2000),
            mode: None,
        }))
        .await;

    // Assert: the boundary instant already counts as live for the
    // second version, and the first is already superseded by then.
    assert_eq!(out.split("\n\n").count(), 1, "expected 1 block: {out}");
    assert!(
        out.contains("] v2"),
        "missing current version at boundary: {out}"
    );
    assert!(
        !out.contains("] v1"),
        "superseded version should not be live: {out}"
    );
}

#[tokio::test]
async fn recall_without_as_of_behaves_like_no_time_filter() {
    // Arrange: a single node, as_of omitted.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock).await;
    seed(&server, "fact", "Now", "zorbnax status is stable").await;

    // Act
    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax status".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Assert: omitting as_of still finds the current version, the same
    // behavior every pre-WU-2 recall test already pins.
    assert!(out.contains("] Now"), "missing match: {out}");
}

#[tokio::test]
async fn remember_then_recall_round_trips_confidence_and_attributes_through_the_tool_call() {
    // Given a fixture written with attributes, confidence, and valid_from
    // all set, through the real `remember` tool call rather than a direct
    // store write like WU-1's per-field tests used.
    let server = plain_server().await;
    let out = server
        .remember(Parameters(RememberArgs {
            attributes: Some(json!({"hue": "blue"})),
            valid_from: Some(1_700_000_000_000),
            confidence: Some(0.42),
            ..remember_args("full round trip via recall content")
        }))
        .await;
    assert!(out.starts_with("remembered "), "{out}");

    // When recall retrieves it through the real `recall` tool call
    let recalled = server
        .recall(Parameters(RecallArgs {
            query: "full round trip via recall content".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: None,
            mode: None,
        }))
        .await;

    // Then both fields carry the values written, proven across the
    // actual tool-call boundary end to end.
    assert!(recalled.contains("confidence: 0.42"), "{recalled}");
    assert!(
        recalled.contains(r#"attributes: {"hue":"blue"}"#),
        "{recalled}"
    );
}

#[tokio::test]
async fn recall_as_of_round_trip_through_remember_and_recall_tool_calls() {
    // Given a subject remembered twice through the real `remember` tool
    // call, the second superseding the first exactly as `upsert_by`
    // semantics work. The clock is advanced explicitly between the two
    // writes rather than relying on the real clock's resolution, the
    // exact flakiness `server_with_clock` (WU-2) exists to avoid.
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let server = server_with_clock(clock.clone()).await;
    remember_version(&server, "before update", "zorbnax widget status is draft").await;
    clock.set(Millis(2000));
    remember_version(&server, "after update", "zorbnax widget status is shipped").await;

    // When recall, through the real `recall` tool call, is pinned to an
    // instant before the second write
    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax widget status".to_string(),
            kind: None,
            scope: None,
            k: None,
            as_of: Some(1000),
            mode: None,
        }))
        .await;

    // Then only the pre-update version's label appears
    assert!(
        out.contains("] before update"),
        "missing pre-update version: {out}"
    );
    assert!(
        !out.contains("] after update"),
        "post-update version leaked: {out}"
    );
}

#[tokio::test]
async fn remember_and_recall_are_registered_with_the_argument_names_clients_send() {
    // As `relate_is_registered_with_the_argument_names_clients_send`:
    // every other test in this module calls these tools directly, so
    // none would notice a tool falling out of the router, or one of
    // WU-1/WU-2's new fields silently dropped from its schema.
    let server = plain_server().await;
    let tools = server.tool_router.list_all();

    for (name, fields) in [
        (
            "remember",
            vec!["\"attributes\"", "\"valid_from\"", "\"confidence\""],
        ),
        ("recall", vec!["\"as_of\""]),
    ] {
        assert!(server.tool_router.has_route(name), "{name} not routed");
        let tool = tools
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} must be listed"));
        let schema = serde_json::to_string(&tool.input_schema).unwrap();
        for field in fields {
            assert!(
                schema.contains(field),
                "{field} missing from {name}'s schema: {schema}"
            );
        }
    }
}
