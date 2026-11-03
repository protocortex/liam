// SPDX-License-Identifier: Apache-2.0
//! The `episode` argument of `remember`: its argument types, the syntactic
//! validation pass, and the resolution of edge references into node indices
//! or existing nodes.

use liam_store::relation;
use liam_store::types::EpisodeRef;
use schemars::JsonSchema;
use serde::Deserialize;

use super::{attributes_problem, confidence_problem, content_problem, MemoryServer};

/// One nested fact inside `RememberArgs.episode.facts`. Same shape as
/// `RememberArgs`'s own fact fields, minus `scope` (an episode's facts share
/// the top-level fact's scope; there is no per-item override).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EpisodeFactArgs {
    pub kind: String,
    pub label: String,
    pub content: String,
    pub attributes: Option<serde_json::Value>,
    pub valid_from: Option<i64>,
    pub confidence: Option<f64>,
    pub subject: Option<String>,
}

/// One edge inside `RememberArgs.episode.edges`, linking two items of the
/// same episode, or an item to an already-existing node, by reference
/// instead of a real id neither may have yet.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EpisodeEdgeArgs {
    /// `"fact:N"` (0 is the top-level fact, 1..=N is `episode.facts`),
    /// `"entity:N"` (0-based into `episode.entities`), or a handle
    /// `recall`/`relate` would accept.
    pub from: String,
    pub to: String,
    /// Relation type, for example `mentions`: `from` must be the entity,
    /// `to` the fact. `supersedes` is reserved, same as `relate`.
    pub kind: String,
}

/// One entity inside `RememberArgs.episode.entities`, an entity page node
/// referenced by episode edges via `"entity:N"`. Written through
/// `liam_store::NewNode::entity(entity_type, name)`: `entity_type` becomes
/// its `kind` (for example "person", "company"), `name` becomes its label
/// and, normalized, its subject, so the same entity named across two
/// separate `remember` calls supersedes instead of duplicating. Gets no
/// embedding, since its content is always empty by construction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EpisodeEntityArgs {
    pub entity_type: String,
    pub name: String,
}

/// `RememberArgs.episode`: additional facts, entities, and the edges
/// between them and the top-level fact, written atomically.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EpisodeArgs {
    pub facts: Vec<EpisodeFactArgs>,
    pub entities: Vec<EpisodeEntityArgs>,
    pub edges: Vec<EpisodeEdgeArgs>,
}

/// Parses a `"fact:N"` episode edge reference into its combined node index
/// (0 is the always-present top-level fact, 1..=`fact_count` is
/// `episode.facts`), if `s` has that shape and `N` is in bounds. `None` for
/// anything else, including a handle, an `"entity:N"` reference (see
/// `parse_entity_ref`), or an unrecognized form; the caller distinguishes
/// those separately.
fn parse_fact_ref(s: &str, fact_count: usize) -> Option<usize> {
    let n: usize = s.strip_prefix("fact:")?.parse().ok()?;
    (n < 1 + fact_count).then_some(n)
}

/// Parses an `"entity:N"` episode edge reference into its entity-LOCAL index
/// (0-based into `episode.entities`), if `s` has that shape and `N` is in
/// bounds. Unlike `parse_fact_ref` there is no "top-level" slot to offset
/// by, so the bounds check is just `n < entity_count`. `None` for anything
/// else. The caller translates a `Some` result to the combined node index
/// (`1 + fact_count + n`) once `fact_count` is in scope; this function does
/// not need it.
fn parse_entity_ref(s: &str, entity_count: usize) -> Option<usize> {
    let n: usize = s.strip_prefix("entity:")?.parse().ok()?;
    (n < entity_count).then_some(n)
}

/// The same cheap syntactic pre-check `Graph::resolve_handle` runs before its
/// real DB read (`graph.rs:332`): non-empty, all-ASCII-alphanumeric. Used
/// here to recognize a handle-shaped episode edge reference up front,
/// without paying for the DB read until the reference is actually resolved.
fn is_handle_shaped(s: &str) -> bool {
    let s = s.trim();
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric())
}

/// The real node id an `EpisodeRef` names, once `Graph::ingest_episode` has
/// returned: the freshly assigned id at its combined index, for `New`, or
/// the id it already carried, for `Existing`.
pub(super) fn resolved_id(r: &EpisodeRef, node_ids: &[liam_store::NodeId]) -> String {
    match r {
        EpisodeRef::New(n) => node_ids[*n].as_str().to_string(),
        EpisodeRef::Existing(id) => id.as_str().to_string(),
    }
}

/// `remember`'s syntactic validation pass for an `episode`: every fact's
/// content/confidence/attributes, then every edge's kind and from/to syntax,
/// accumulating every problem instead of stopping at the first. Pure and
/// `&self`-free, since none of these checks touches the store; a handle
/// reference's existence is checked later, once this pass is clean. Problems
/// come back facts-then-edges, each group in index order, and within one
/// edge in a fixed order (empty-kind, then supersedes, then mentions
/// direction, then the from/to reference check), matching the order
/// `remember` has always reported them in.
pub(super) fn validate_episode(
    episode: &EpisodeArgs,
    fact_count: usize,
    entity_count: usize,
) -> Vec<String> {
    let mut problems: Vec<String> = Vec::new();
    for (i, fact) in episode.facts.iter().enumerate() {
        if let Some(problem) = content_problem(&fact.content) {
            problems.push(format!("fact:{}: {problem}", i + 1));
        }
        if let Some(problem) = confidence_problem(fact.confidence) {
            problems.push(format!("fact:{}: {problem}", i + 1));
        }
        if let Some(problem) = attributes_problem(&fact.attributes) {
            problems.push(format!("fact:{}: {problem}", i + 1));
        }
    }
    for (j, edge) in episode.edges.iter().enumerate() {
        let kind = edge.kind.trim().to_lowercase();
        if kind.is_empty() {
            problems.push(format!("edge {j}: type must not be empty"));
        } else if kind == relation::SUPERSEDES {
            problems.push(format!(
                "edge {j}: '{}' is reserved for version history",
                relation::SUPERSEDES
            ));
        } else if kind == relation::MENTIONS {
            if parse_entity_ref(&edge.to, entity_count).is_some() {
                problems.push(format!(
                    "edge {j}: mentions edge must have an entity as 'from' and a fact as \
                     'to' (to is backwards)"
                ));
            }
            if parse_fact_ref(&edge.from, fact_count).is_some() {
                problems.push(format!(
                    "edge {j}: mentions edge must have an entity as 'from' and a fact as \
                     'to' (from is backwards)"
                ));
            }
        }
        for (role, reference) in [("from", edge.from.as_str()), ("to", edge.to.as_str())] {
            if parse_fact_ref(reference, fact_count).is_none()
                && parse_entity_ref(reference, entity_count).is_none()
                && !is_handle_shaped(reference)
            {
                problems.push(format!(
                    "edge {j}: {role} '{reference}' is not a recognized reference"
                ));
            }
        }
    }
    problems
}

impl MemoryServer {
    /// Resolves one episode edge reference, already known syntactically
    /// valid by the caller's validation pass: `"fact:N"` or `"entity:N"`
    /// (parsing cannot fail here) to `EpisodeRef::New` at the combined
    /// index (an entity-local index `n` becomes `1 + fact_count + n`), or a
    /// handle to `EpisodeRef::Existing` via the one real `resolve_handle` DB
    /// read this reference gets. One method, not a fact/entity pair,
    /// because both branches produce the same `EpisodeRef::New` shape; the
    /// only difference is which parser and which index offset apply.
    async fn episode_ref(
        &self,
        s: &str,
        fact_count: usize,
        entity_count: usize,
    ) -> liam_store::Result<EpisodeRef> {
        if let Some(n) = parse_fact_ref(s, fact_count) {
            return Ok(EpisodeRef::New(n));
        }
        if let Some(n) = parse_entity_ref(s, entity_count) {
            return Ok(EpisodeRef::New(1 + fact_count + n));
        }
        on_store!(self, |store| store.resolve_handle(s)).map(EpisodeRef::Existing)
    }

    /// Resolves every episode edge's `from`/`to` into a real `EpisodeRef`,
    /// once `validate_episode` has confirmed both sides are syntactically
    /// valid: the one real DB read per handle-form reference. A resolution
    /// failure is a validation failure too, accumulated the same way
    /// `validate_episode` accumulates its own problems, even though the DB
    /// check itself couldn't happen until now. `Err` carries every problem
    /// found rather than the first, so a caller reports them all at once.
    pub(super) async fn resolve_episode_refs(
        &self,
        episode: &EpisodeArgs,
        fact_count: usize,
        entity_count: usize,
    ) -> Result<Vec<(EpisodeRef, EpisodeRef)>, Vec<String>> {
        let mut refs: Vec<(EpisodeRef, EpisodeRef)> = Vec::with_capacity(episode.edges.len());
        let mut resolve_problems: Vec<String> = Vec::new();
        for (j, edge) in episode.edges.iter().enumerate() {
            match (
                self.episode_ref(&edge.from, fact_count, entity_count).await,
                self.episode_ref(&edge.to, fact_count, entity_count).await,
            ) {
                (Ok(from), Ok(to)) => refs.push((from, to)),
                (from, to) => {
                    if let Err(e) = from {
                        resolve_problems.push(format!("edge {j}: from: {e}"));
                    }
                    if let Err(e) = to {
                        resolve_problems.push(format!("edge {j}: to: {e}"));
                    }
                }
            }
        }
        if resolve_problems.is_empty() {
            Ok(refs)
        } else {
            Err(resolve_problems)
        }
    }
}

#[cfg(test)]
pub(super) mod test_support {
    use super::{EpisodeEdgeArgs, EpisodeEntityArgs, EpisodeFactArgs};

    /// Bare-bones episode fact args, mirroring `remember_args`: every field
    /// but `content` is a fixed default, distinct content per call so a test
    /// can find its own node with `Query::text`.
    pub(crate) fn episode_fact(content: &str) -> EpisodeFactArgs {
        EpisodeFactArgs {
            kind: "fact".to_string(),
            label: "label".to_string(),
            content: content.to_string(),
            attributes: None,
            valid_from: None,
            confidence: None,
            subject: None,
        }
    }

    pub(crate) fn episode_edge(from: &str, to: &str, kind: &str) -> EpisodeEdgeArgs {
        EpisodeEdgeArgs {
            from: from.to_string(),
            to: to.to_string(),
            kind: kind.to_string(),
        }
    }

    pub(crate) fn episode_entity(entity_type: &str, name: &str) -> EpisodeEntityArgs {
        EpisodeEntityArgs {
            entity_type: entity_type.to_string(),
            name: name.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{episode_edge, episode_entity, episode_fact};
    use super::*;
    use crate::mcp::MAX_CONTENT_CHARS;

    #[test]
    fn validate_episode_returns_empty_for_a_clean_episode() {
        // Given an episode with one valid fact and no edges
        let episode = EpisodeArgs {
            facts: vec![episode_fact("clean content")],
            entities: vec![],
            edges: vec![],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then no problems are reported
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn validate_episode_reports_a_facts_content_problem() {
        // Given one fact whose content exceeds MAX_CONTENT_CHARS
        let over_cap = "x".repeat(MAX_CONTENT_CHARS + 1);
        let episode = EpisodeArgs {
            facts: vec![episode_fact(&over_cap)],
            entities: vec![],
            edges: vec![],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then the content problem is reported against fact:1
        assert_eq!(
            problems,
            vec![format!(
                "fact:1: content exceeds {MAX_CONTENT_CHARS} characters"
            )]
        );
    }

    #[test]
    fn validate_episode_reports_a_facts_confidence_problem() {
        // Given one fact with an out-of-range confidence
        let episode = EpisodeArgs {
            facts: vec![EpisodeFactArgs {
                confidence: Some(5.0),
                ..episode_fact("fine content")
            }],
            entities: vec![],
            edges: vec![],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then the confidence problem is reported against fact:1
        assert_eq!(
            problems,
            vec!["fact:1: confidence must be between 0.0 and 1.0".to_string()]
        );
    }

    #[test]
    fn validate_episode_reports_an_edge_with_an_empty_kind() {
        // Given an edge whose kind is empty
        let episode = EpisodeArgs {
            facts: vec![],
            entities: vec![],
            edges: vec![episode_edge("fact:0", "fact:0", "")],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then the empty-kind problem is reported against edge 0
        assert_eq!(problems, vec!["edge 0: type must not be empty".to_string()]);
    }

    #[test]
    fn validate_episode_reports_an_edge_asserting_supersedes() {
        // Given an edge asserting the reserved `supersedes` kind
        let episode = EpisodeArgs {
            facts: vec![],
            entities: vec![],
            edges: vec![episode_edge("fact:0", "fact:0", "supersedes")],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then the reserved-kind problem is reported against edge 0
        assert_eq!(
            problems,
            vec!["edge 0: 'supersedes' is reserved for version history".to_string()]
        );
    }

    #[test]
    fn validate_episode_reports_a_backwards_mentions_edge_with_a_fresh_entity_as_to() {
        // Given a mentions edge whose `to` is a fresh entity reference
        // (backwards: `to` should be a fact) and whose `from` is
        // handle-shaped, so only the to-is-entity check fires, independent
        // of the from-is-fact check
        let episode = EpisodeArgs {
            facts: vec![],
            entities: vec![episode_entity("person", "Someone")],
            edges: vec![episode_edge("abc123", "entity:0", "mentions")],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then only the to-is-backwards problem is reported
        assert_eq!(
            problems,
            vec![
                "edge 0: mentions edge must have an entity as 'from' and a fact as 'to' \
                 (to is backwards)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn validate_episode_reports_a_backwards_mentions_edge_with_a_fresh_fact_as_from() {
        // Given a mentions edge whose `from` is a fresh fact reference
        // (backwards: `from` should be an entity) and whose `to` is
        // handle-shaped, so only the from-is-fact check fires, independent
        // of the to-is-entity check
        let episode = EpisodeArgs {
            facts: vec![episode_fact("nested content")],
            entities: vec![],
            edges: vec![episode_edge("fact:1", "abc123", "mentions")],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then only the from-is-backwards problem is reported
        assert_eq!(
            problems,
            vec![
                "edge 0: mentions edge must have an entity as 'from' and a fact as 'to' \
                 (from is backwards)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn validate_episode_reports_an_edge_with_an_unrecognized_reference() {
        // Given an edge whose `to` is neither a fact/entity reference nor
        // handle-shaped
        let episode = EpisodeArgs {
            facts: vec![],
            entities: vec![],
            edges: vec![episode_edge("fact:0", "bogus:0", "relates_to")],
        };

        // When validated
        let problems = validate_episode(&episode, episode.facts.len(), episode.entities.len());

        // Then the unrecognized-reference problem is reported against edge 0's to
        assert_eq!(
            problems,
            vec!["edge 0: to 'bogus:0' is not a recognized reference".to_string()]
        );
    }
}
