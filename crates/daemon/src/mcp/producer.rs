// SPDX-License-Identifier: Apache-2.0
//! Producer resolution: turn an MCP client's declared name into the
//! canonical producer id recorded on every node it writes.
//!
//! Kept PURE and free of any `rmcp` type on purpose. The MCP `initialize`
//! handshake (wired up in WU-8) is what actually reads a client's name off
//! the wire; by the time that name reaches this module it is already a plain
//! `Option<&str>`, so resolution itself needs no transport, no I/O, and no
//! async, and is testable with nothing more than a config value.

use std::collections::HashMap;

use crate::config::ProducersConfig;

/// Resolves an MCP client's declared name (`clientInfo.name` from the
/// `initialize` handshake) to a canonical producer id via
/// `config.clients`, falling back to `config.unknown_id` when the name is
/// absent or does not appear in the table.
///
/// **Matching is case-insensitive.** MCP clients do not agree on how they
/// capitalize their own names, so an operator who writes `claude-code` under
/// `[producers.clients]` in `liam.toml` should still match a client that
/// identifies itself as `Claude-Code`; both the configured key and the
/// connecting client's name are lowercased before comparison.
///
/// Resolution runs once per connection: `transport::socket::handle_connection`
/// calls this immediately after the MCP `initialize` handshake, so logging a
/// warning here on a fallback is naturally once-per-connection already and no
/// separate throttle is added.
///
/// Deliberately carries no `#[allow(dead_code)]`, unlike
/// `transport::socket::accept_loop`, which still waits on WU-9's mode
/// dispatch: this is genuinely reachable from the bin target now, so leaving
/// the lint armed means CI catches it if a later refactor ever unwires
/// producer resolution from the accept loop.
pub fn resolve(client_name: Option<&str>, config: &ProducersConfig) -> String {
    let Some(name) = client_name else {
        tracing::warn!(
            fallback = %config.unknown_id,
            "MCP client declared no name at initialize; recording its writes under the fallback producer"
        );
        return config.unknown_id.clone();
    };

    match lookup_case_insensitive(&config.clients, name) {
        Some(id) => id,
        None => {
            tracing::warn!(
                client_name = %name,
                fallback = %config.unknown_id,
                "MCP client name is not in [producers.clients]; recording its writes under the fallback producer"
            );
            config.unknown_id.clone()
        }
    }
}

/// Case-insensitive lookup over the configured client table. A linear scan
/// rather than pre-lowercasing into a second map: `config.clients` is
/// operator-sized (a handful of entries), so building and caching a second
/// map on every connection would cost more than it ever saves.
///
/// Takes the lexicographically smallest matching key rather than the first
/// one iteration happens to yield. TOML keys are case-sensitive, so an
/// operator may define both `claude-code` and `Claude-Code`, and both match
/// here. `HashMap` iteration order is not stable between runs, so a plain
/// `find` would resolve such a pair to a different producer id from one
/// daemon restart to the next. Picking a total order makes the answer
/// depend only on the config.
fn lookup_case_insensitive(clients: &HashMap<String, String>, name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    clients
        .iter()
        .filter(|(key, _)| key.to_lowercase() == lower)
        .min_by(|(a, _), (b, _)| a.cmp(b))
        .map(|(_, id)| id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(unknown_id: &str, pairs: &[(&str, &str)]) -> ProducersConfig {
        ProducersConfig {
            unknown_id: unknown_id.to_string(),
            clients: pairs
                .iter()
                .map(|(key, id)| (key.to_string(), id.to_string()))
                .collect(),
        }
    }

    #[test]
    fn a_name_present_in_the_table_resolves_to_its_canonical_id() {
        // Given a client name present in the config table
        let cfg = config("unknown", &[("claude-code", "claude")]);

        // When resolved
        let id = resolve(Some("claude-code"), &cfg);

        // Then the canonical id comes back
        assert_eq!(id, "claude");
    }

    #[test]
    fn a_name_absent_from_the_table_resolves_to_the_fallback() {
        // Given a name absent from the table
        let cfg = config("unknown", &[("claude-code", "claude")]);

        // When resolved
        let id = resolve(Some("some-other-client"), &cfg);

        // Then the configured fallback id comes back
        assert_eq!(id, "unknown");
    }

    #[test]
    fn no_client_name_resolves_to_the_fallback() {
        // Given no client name at all
        let cfg = config("guest", &[("claude-code", "claude")]);

        // When resolved
        let id = resolve(None, &cfg);

        // Then the fallback id comes back
        assert_eq!(id, "guest");
    }

    #[test]
    fn a_name_differing_only_by_case_still_resolves_to_the_canonical_id() {
        // Given a name differing from the configured key only by case
        let cfg = config("unknown", &[("claude-code", "claude")]);

        // When resolved
        let id = resolve(Some("Claude-Code"), &cfg);

        // Then it resolves the same as the exact-case name: matching is
        // documented above as case-insensitive.
        assert_eq!(id, "claude");
    }

    #[test]
    fn two_keys_differing_only_by_case_always_resolve_to_the_same_id() {
        // Given a table where two case-variant keys both match, which TOML
        // allows because its keys are case-sensitive
        let cfg = config(
            "unknown",
            &[("claude-code", "lowercase"), ("Claude-Code", "titlecase")],
        );

        // When the same name resolves repeatedly, across freshly built
        // tables so each one gets its own HashMap iteration order
        let ids: Vec<String> = (0..64)
            .map(|_| {
                let cfg = config(
                    "unknown",
                    &[("claude-code", "lowercase"), ("Claude-Code", "titlecase")],
                );
                resolve(Some("CLAUDE-CODE"), &cfg)
            })
            .collect();

        // Then every answer is the same one, and it is the
        // lexicographically smallest key's id rather than whichever entry
        // iteration happened to reach first.
        assert!(
            ids.iter().all(|id| id == "titlecase"),
            "resolution must not depend on HashMap iteration order, got: {ids:?}"
        );
        assert_eq!(resolve(Some("claude-code"), &cfg), "titlecase");
    }

    #[test]
    fn an_empty_client_table_still_falls_back_for_a_named_client() {
        // Given a config with no entries under [producers.clients]
        let cfg = config("unknown", &[]);

        // When a named client resolves
        let id = resolve(Some("anything"), &cfg);

        // Then it falls back, rather than panicking on an empty table
        assert_eq!(id, "unknown");
    }
}
