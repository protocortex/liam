// SPDX-License-Identifier: Apache-2.0
//! The `facts` tool: read-only bitemporal queries over extracted facts.

use std::collections::HashMap;

use liam_store::{
    Cardinality, DefaultGraph, Fact, FactId, FactPage, Millis, Vocabulary, BEGINNING, FOREVER,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::MemoryServer;
use crate::ask::fmt_millis;

/// Rows returned when the caller gives no `limit`.
pub(super) const DEFAULT_FACTS_LIMIT: usize = 20;

/// Most rows one call returns, whatever `limit` asks for.
pub(super) const MAX_FACTS_LIMIT: usize = 100;

/// Longest accepted subject, in characters.
const MAX_SUBJECT_CHARS: usize = 256;

/// Characters in a lineage id, which is a ULID.
const LINEAGE_ID_CHARS: usize = 26;

/// Source ids listed per fact; the count always covers them all.
const MAX_SOURCES_SHOWN: usize = 10;

const MILLIS_PER_SECOND: i64 = 1000;

const NO_FACTS: &str = "no matching facts";
const MORE_FACTS_NOTE: &str = "more facts exist";
const MORE_SOURCES_NOTE: &str = "some facts have more sources than listed";

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FactsArgs {
    /// The entity the facts are about, matched after trimming and lower-casing.
    pub subject: String,
    /// Narrows to this scope and its descendants. Absent matches every scope.
    pub scope: Option<String>,
    /// Valid time, epoch milliseconds: the facts true at that instant. Absent
    /// means now.
    pub as_of: Option<i64>,
    /// Transaction time, epoch milliseconds: what the store believed then.
    /// Alone, it asks about the same instant; with `as_of`, about that instant.
    pub believed_at: Option<i64>,
    /// A lineage id shown on a fact: lists every stored version of it, oldest
    /// first. Cannot be combined with `scope`, `as_of` or `believed_at`.
    pub lineage: Option<String>,
    /// Rows to return. Absent defaults to 20; anything above 100 is capped.
    pub limit: Option<usize>,
}

/// What the call asks the store for, after validation.
enum View {
    /// Believed now, true at the instant (default now).
    Current { valid_at: Option<Millis> },
    /// Believed at one instant, true at another (default the same one).
    Past {
        believed_at: Millis,
        valid_at: Option<Millis>,
    },
    /// Every stored version of one fact.
    Lineage(FactId),
}

struct Request {
    subject: String,
    scope: Option<String>,
    view: View,
    limit: usize,
}

pub(super) async fn read(server: &MemoryServer, args: FactsArgs) -> String {
    let request = match Request::parse(args) {
        Ok(request) => request,
        Err(reason) => return format!("facts failed: {reason}"),
    };
    if request.limit == 0 {
        return NO_FACTS.to_string();
    }
    let Request {
        subject,
        scope,
        view,
        limit,
    } = request;
    let scope = scope.as_deref();
    let in_lineage = matches!(view, View::Lineage(_));
    // The guard lasts for this one read only, as in every other tool.
    let page = match server.store().await {
        Ok(store) => fetch(&store, &subject, scope, &view, limit).await,
        Err(error) => Err(error),
    };
    let mut page = match page {
        Ok(page) => page,
        Err(error) => return format!("facts failed: {error}"),
    };
    // The store keys a lineage by its id alone; the caller's subject must agree.
    page.facts.retain(|fact| fact.subject == subject);
    tracing::debug!(rows = page.facts.len(), in_lineage, "facts read");
    if page.facts.is_empty() {
        return NO_FACTS.to_string();
    }
    // Versions of one lineage are successive beliefs, not rival claims.
    let competing = if in_lineage {
        vec![false; page.facts.len()]
    } else {
        competing_rows(&page.facts, &server.vocabulary)
    };
    render(&page, &competing)
}

async fn fetch(
    store: &DefaultGraph,
    subject: &str,
    scope: Option<&str>,
    view: &View,
    limit: usize,
) -> liam_store::Result<FactPage> {
    match view {
        View::Current { valid_at } => {
            let valid_at = valid_at.unwrap_or_else(Millis::now);
            store.facts_valid_at(subject, scope, valid_at, limit).await
        }
        View::Past {
            believed_at,
            valid_at,
        } => {
            store
                .facts_believed_at(subject, scope, *believed_at, *valid_at, limit)
                .await
        }
        View::Lineage(lineage) => store.fact_lineage_history(lineage, limit).await,
    }
}

impl Request {
    fn parse(args: FactsArgs) -> Result<Self, String> {
        let subject = args.subject.trim().to_lowercase();
        if subject.is_empty() {
            return Err("subject must not be empty".into());
        }
        if subject.chars().count() > MAX_SUBJECT_CHARS {
            return Err(format!("subject exceeds {MAX_SUBJECT_CHARS} characters"));
        }
        let as_of = args.as_of.map(|ms| instant("as_of", ms)).transpose()?;
        let believed_at = args
            .believed_at
            .map(|ms| instant("believed_at", ms))
            .transpose()?;
        let view = match (args.lineage, believed_at) {
            (Some(lineage), _) => {
                if args.scope.is_some() || as_of.is_some() || believed_at.is_some() {
                    return Err(
                        "lineage cannot be combined with scope, as_of or believed_at".into(),
                    );
                }
                View::Lineage(lineage_id(&lineage)?)
            }
            (None, Some(believed_at)) => View::Past {
                believed_at,
                valid_at: as_of,
            },
            (None, None) => View::Current { valid_at: as_of },
        };
        Ok(Self {
            subject,
            scope: args.scope,
            view,
            limit: args
                .limit
                .unwrap_or(DEFAULT_FACTS_LIMIT)
                .min(MAX_FACTS_LIMIT),
        })
    }
}

/// An instant a calendar can hold and the open-ended sentinel has not reached.
fn instant(name: &str, ms: i64) -> Result<Millis, String> {
    let representable = ms < FOREVER.0
        && time::OffsetDateTime::from_unix_timestamp(ms.div_euclid(MILLIS_PER_SECOND)).is_ok();
    if representable {
        Ok(Millis(ms))
    } else {
        Err(format!(
            "{name} must be epoch milliseconds before the year 2100"
        ))
    }
}

fn lineage_id(raw: &str) -> Result<FactId, String> {
    let raw = raw.trim();
    if raw.len() == LINEAGE_ID_CHARS && raw.chars().all(|c| c.is_ascii_alphanumeric()) {
        Ok(FactId::from_raw(raw))
    } else {
        Err(format!(
            "lineage must be a {LINEAGE_ID_CHARS} character id as shown on a fact"
        ))
    }
}

/// Marks every row of a `(predicate, scope)` group that holds more than one
/// live fact for a predicate that allows a single value.
fn competing_rows(facts: &[Fact], vocabulary: &Vocabulary) -> Vec<bool> {
    let mut group_sizes: HashMap<(&str, Option<&str>), usize> = HashMap::new();
    for fact in facts {
        *group_sizes
            .entry((&fact.predicate, fact.scope.as_deref()))
            .or_default() += 1;
    }
    facts
        .iter()
        .map(|fact| {
            vocabulary.spec(&fact.predicate).cardinality == Cardinality::One
                && group_sizes[&(fact.predicate.as_str(), fact.scope.as_deref())] > 1
        })
        .collect()
}

fn render(page: &FactPage, competing: &[bool]) -> String {
    let mut blocks: Vec<String> = page
        .facts
        .iter()
        .zip(competing)
        .map(|(fact, competing)| render_fact(fact, *competing))
        .collect();
    if page.truncated {
        blocks.push(format!(
            "({MORE_FACTS_NOTE}: raise limit or narrow the query)"
        ));
    }
    if page.sources_truncated {
        blocks.push(format!("({MORE_SOURCES_NOTE})"));
    }
    blocks.join("\n\n")
}

/// The claim on one line, then one `name: value` line per field, as the other
/// tools render optional fields.
fn render_fact(fact: &Fact, competing: bool) -> String {
    let mut lines = vec![
        format!(
            "{} {} {}",
            one_line(&fact.subject),
            one_line(&fact.predicate),
            one_line(&fact.object)
        ),
        format!("trust: {:.2}", fact.trust_score),
        format!("confidence: {:.2}", fact.confidence),
        format!(
            "extractor: {} {}",
            fact.extractor_id.as_str(),
            fact.extractor_version
        ),
        format!(
            "valid: {} to {}",
            valid_bound(fact.valid_from),
            valid_bound(fact.valid_until)
        ),
        format!(
            "recorded: {} to {}",
            recorded_bound(fact.tx_from),
            recorded_bound(fact.tx_to)
        ),
    ];
    if let Some(scope) = &fact.scope {
        lines.push(format!("scope: {scope}"));
    }
    lines.push(format!("lineage: {}", fact.lineage_id.as_str()));
    if let Some(closer) = &fact.closed_by {
        lines.push(format!("closed_by: {}", closer.as_str()));
    }
    if competing {
        lines.push("competing: yes".to_string());
    }
    lines.push(sources_line(&fact.source_event_ids));
    lines.join("\n")
}

fn sources_line(ids: &[String]) -> String {
    let shown: Vec<&str> = ids
        .iter()
        .take(MAX_SOURCES_SHOWN)
        .map(String::as_str)
        .collect();
    let hidden = ids.len() - shown.len();
    let rest = if hidden > 0 {
        format!(", and {hidden} more")
    } else {
        String::new()
    };
    format!("sources: {} ({}{rest})", ids.len(), shown.join(", "))
}

/// Valid times are dates; the ends of time read as words.
fn valid_bound(at: Millis) -> String {
    if at == FOREVER {
        "open".to_string()
    } else if at == BEGINNING {
        "unbounded".to_string()
    } else {
        fmt_millis(at.0)
    }
}

/// Transaction times stay in the epoch milliseconds `believed_at` takes.
fn recorded_bound(at: Millis) -> String {
    if at == FOREVER {
        "open".to_string()
    } else {
        at.0.to_string()
    }
}

/// Stored text can hold line breaks that would pass for the next field.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests;
