// SPDX-License-Identifier: Apache-2.0
//! `relate`: the arguments clients send and the edges it refuses.

use super::*;

#[tokio::test]
async fn relate_is_registered_with_the_argument_names_clients_send() {
    // Every other `relate` test calls the method directly, so none of them
    // would notice the tool missing from the router: a client would just
    // never see it. The schema check pins `type`, which only reaches the
    // wire through a serde rename because `type` is a Rust keyword.
    let server = plain_server().await;

    assert!(server.tool_router.has_route("relate"), "relate not routed");
    let tool = server
        .tool_router
        .list_all()
        .into_iter()
        .find(|t| t.name == "relate")
        .expect("relate must be listed");
    let schema = serde_json::to_string(&tool.input_schema).unwrap();
    for field in ["\"from\"", "\"to\"", "\"type\""] {
        assert!(schema.contains(field), "{field} missing from {schema}");
    }
    assert!(!schema.contains("\"kind\""), "leaked rust name: {schema}");
}

#[tokio::test]
async fn relate_refuses_the_reserved_supersedes_type() {
    // `supersedes` is written only inside `Graph::supersede`'s transaction.
    // A client able to assert it could rewrite version history, so this
    // refusal is the guard ADR-0001 puts at the door.
    let server = plain_server().await;
    let a = seed(&server, "decision", "Alpha", "first").await;
    let b = seed(&server, "decision", "Beta", "second").await;

    for kind in ["supersedes", "SUPERSEDES", "  supersedes  "] {
        let out = server
            .relate(Parameters(RelateArgs {
                from: a.clone(),
                to: b.clone(),
                kind: kind.to_string(),
            }))
            .await;
        assert!(out.contains("reserved"), "accepted {kind:?}: {out}");
    }
}

#[tokio::test]
async fn relate_refuses_a_self_loop_even_through_two_different_handles() {
    // A short handle and the full id name the same node, so the check has
    // to run AFTER resolution. Comparing the raw arguments would let this
    // pair through and write an edge that carries no meaning.
    let server = plain_server().await;
    let a = seed(&server, "decision", "Alpha", "first").await;

    let out = server
        .relate(Parameters(RelateArgs {
            from: a[..HANDLE_LEN].to_string(),
            to: a.clone(),
            kind: "mentions".to_string(),
        }))
        .await;

    assert!(out.contains("same node"), "{out}");
}

#[tokio::test]
async fn relate_reports_an_unknown_handle_without_writing() {
    let server = plain_server().await;
    let a = seed(&server, "decision", "Alpha", "first").await;

    let out = server
        .relate(Parameters(RelateArgs {
            from: a,
            to: "0000000000000".to_string(),
            kind: "mentions".to_string(),
        }))
        .await;

    assert!(out.starts_with("relate failed: to:"), "{out}");
    assert!(out.contains("no live node"), "{out}");
    // The "without writing" half of the name, asserted rather than assumed.
    // Community detection builds its node set purely from live edges, so a
    // zero count means nothing landed in the table.
    assert_eq!(
        server
            .store()
            .await
            .unwrap()
            .recompute_communities()
            .await
            .unwrap(),
        0,
        "a refused relate still wrote an edge"
    );
}

#[tokio::test]
async fn relation_types_are_case_normalised_into_one_relation() {
    // `Mentions` and `mentions` are distinct keys in the clustering dedup,
    // so leaving both would double the pair's weight on nothing worse than
    // inconsistent capitalisation. Normalising at this boundary is also
    // what lets the reserved-word check be an exact comparison.
    let server = plain_server().await;
    let a = seed(&server, "decision", "Alpha", "first").await;
    let b = seed(&server, "decision", "Beta", "second").await;
    let relate = |kind: &str| {
        server.relate(Parameters(RelateArgs {
            from: a.clone(),
            to: b.clone(),
            kind: kind.to_string(),
        }))
    };

    let first = relate("mentions").await;
    let second = relate("Mentions").await;
    let third = relate("  MENTIONS  ").await;

    assert!(first.starts_with("related "), "{first}");
    assert!(second.contains("already relates"), "{second}");
    assert!(third.contains("already relates"), "{third}");
    assert!(
        first.contains("-mentions->"),
        "type not normalised: {first}"
    );
}

#[tokio::test]
async fn relate_refuses_a_glob_wildcard_instead_of_matching_a_node() {
    // `*` is a GLOB wildcard, and resolution is the only place a
    // client-supplied string reaches a pattern. ONE seeded node on purpose:
    // with two, an unguarded `*` matches both and comes back as an ambiguous
    // handle, which still starts with "relate failed: from:" and lets the
    // test pass with the guard deleted. With one node it would resolve
    // successfully instead, to a node the caller never named.
    let server = plain_server().await;
    let only = seed(&server, "decision", "Alpha", "first").await;

    let out = server
        .relate(Parameters(RelateArgs {
            from: "*".to_string(),
            to: only,
            kind: "mentions".to_string(),
        }))
        .await;

    assert!(out.contains("no live node"), "wildcard resolved: {out}");
}

#[tokio::test]
async fn relate_is_idempotent_on_the_same_ordered_triple() {
    let server = plain_server().await;
    let a = seed(&server, "decision", "Alpha", "first").await;
    let b = seed(&server, "decision", "Beta", "second").await;
    let args = || RelateArgs {
        from: a.clone(),
        to: b.clone(),
        kind: "mentions".to_string(),
    };

    let first = server.relate(Parameters(args())).await;
    let second = server.relate(Parameters(args())).await;

    assert!(first.starts_with("related "), "{first}");
    assert!(second.contains("already relates"), "{second}");
}
