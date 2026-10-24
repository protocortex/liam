// SPDX-License-Identifier: Apache-2.0
//! The `facts` tool: read-only bitemporal queries over extracted facts.

use std::collections::HashMap;

use liam_store::{
    validate_scope, Cardinality, DefaultGraph, Fact, FactPage, Millis, Vocabulary, BEGINNING,
    FOREVER, MAX_SOURCES_PER_FACT,
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

/// Source ids listed per fact. The count beside them is the store's, which
/// stops at `MAX_SOURCES_PER_FACT`.
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
    /// means now, or the `believed_at` instant when that is given.
    pub as_of: Option<i64>,
    /// Transaction time, epoch milliseconds: what the store believed then.
    /// Alone, it asks about the same instant; with `as_of`, about that instant.
    pub believed_at: Option<i64>,
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
    // The guard lasts for this one read only, as in every other tool.
    let page = match server.store().await {
        Ok(store) => fetch(&store, &subject, scope, &view, limit).await,
        Err(error) => Err(error),
    };
    let page = match page {
        Ok(page) => page,
        Err(error) => return format!("facts failed: {error}"),
    };
    tracing::debug!(rows = page.facts.len(), "facts read");
    if page.facts.is_empty() {
        return NO_FACTS.to_string();
    }
    let competing = competing_rows(&page.facts, &server.vocabulary);
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
        let scope = validate_scope(&args.scope).map_err(|error| error.to_string())?;
        let view = match believed_at {
            Some(believed_at) => View::Past {
                believed_at,
                valid_at: as_of,
            },
            None => View::Current { valid_at: as_of },
        };
        Ok(Self {
            subject,
            scope,
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
        .map(|(fact, competing)| render_fact(fact, *competing, page.sources_truncated))
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
fn render_fact(fact: &Fact, competing: bool, sources_cut: bool) -> String {
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
    if competing {
        lines.push("competing: yes".to_string());
    }
    lines.push(sources_line(&fact.source_event_ids, sources_cut));
    lines.join("\n")
}

/// A list that reaches the store's cap in a cut page is a lower bound.
fn sources_line(ids: &[String], page_cut: bool) -> String {
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
    let at_cap = page_cut && ids.len() >= MAX_SOURCES_PER_FACT;
    let plus = if at_cap { "+" } else { "" };
    format!("sources: {}{plus} ({}{rest})", ids.len(), shown.join(", "))
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
