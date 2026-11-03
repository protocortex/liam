// SPDX-License-Identifier: Apache-2.0
//! The two guarantees of the summary tree, as checks that run on a snapshot of
//! the store and on the text a tool printed:
//!
//! 1. every summary traces to real events: its sources are exactly the events of
//!    its children, at least one is a real log event, and every one is an event
//!    the log holds;
//! 2. no inferred content reaches a recorded field of rendered output: strict
//!    outputs leave inferred items out, and every item that is shown carries
//!    the derivation the store holds for it.
//!
//! Both prove the links, not that a summary is faithful to what it summarizes.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use liam_store::{ChildKind, Derivation, EventRef, SummaryChild};

use crate::mcp::Mode;

/// What the log holds: the event ids that were written to it.
pub trait EventLookup {
    fn knows(&self, event_id: &str) -> bool;
}

impl EventLookup for BTreeSet<String> {
    fn knows(&self, event_id: &str) -> bool {
        self.contains(event_id)
    }
}

/// One live summary with what it was built from.
pub struct SummaryRow {
    pub id: String,
    pub children: Vec<SummaryChild>,
    pub sources: Vec<EventRef>,
}

/// The live summaries and the origin events of the nodes under them.
pub struct StoreSnapshot {
    pub summaries: Vec<SummaryRow>,
    pub node_events: HashMap<String, Vec<EventRef>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum TraceError {
    /// Every event in the store is synthetic, so a trace proves nothing.
    SyntheticOnly,
    NoSources {
        summary: String,
    },
    /// A child names a summary or node the snapshot does not hold.
    UnknownChild {
        summary: String,
        child: String,
    },
    /// An event of a child that the summary's own events lack.
    MissingFromParent {
        summary: String,
        event: String,
    },
    /// An event of the summary that none of its children carries.
    NotFromChildren {
        summary: String,
        event: String,
    },
    /// Only synthetic events, though the store has real ones.
    SyntheticClosure {
        summary: String,
    },
    UnknownEvent {
        summary: String,
        event: String,
    },
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SyntheticOnly => write!(f, "synthetic only: no real event in the store"),
            other => write!(f, "{other:?}"),
        }
    }
}

fn ids(events: &[EventRef]) -> BTreeSet<&str> {
    events.iter().map(|e| e.event_id.as_str()).collect()
}

/// The events behind the children of `row`, found in the snapshot.
fn child_events<'a>(
    snapshot: &'a StoreSnapshot,
    row: &SummaryRow,
) -> Result<BTreeSet<&'a str>, TraceError> {
    let mut union = BTreeSet::new();
    for child in &row.children {
        let events = match child.kind {
            ChildKind::Node => snapshot.node_events.get(&child.id),
            ChildKind::Summary => snapshot
                .summaries
                .iter()
                .find(|s| s.id == child.id)
                .map(|s| &s.sources),
        };
        let events = events.ok_or_else(|| TraceError::UnknownChild {
            summary: row.id.clone(),
            child: child.id.clone(),
        })?;
        union.extend(ids(events));
    }
    Ok(union)
}

/// Checks the first guarantee over every live summary at every level.
pub fn check_trace(snapshot: &StoreSnapshot, log: &dyn EventLookup) -> Result<(), TraceError> {
    let all = snapshot
        .summaries
        .iter()
        .flat_map(|s| &s.sources)
        .chain(snapshot.node_events.values().flatten());
    if all.clone().next().is_some() && all.clone().all(|event| event.synthetic) {
        return Err(TraceError::SyntheticOnly);
    }
    for row in &snapshot.summaries {
        if row.sources.is_empty() {
            return Err(TraceError::NoSources {
                summary: row.id.clone(),
            });
        }
        let own = ids(&row.sources);
        let below = child_events(snapshot, row)?;
        if let Some(event) = below.difference(&own).next() {
            return Err(TraceError::MissingFromParent {
                summary: row.id.clone(),
                event: event.to_string(),
            });
        }
        if let Some(event) = own.difference(&below).next() {
            return Err(TraceError::NotFromChildren {
                summary: row.id.clone(),
                event: event.to_string(),
            });
        }
        if row.sources.iter().all(|event| event.synthetic) {
            return Err(TraceError::SyntheticClosure {
                summary: row.id.clone(),
            });
        }
        if let Some(event) = own.iter().find(|event| !log.knows(event)) {
            return Err(TraceError::UnknownEvent {
                summary: row.id.clone(),
                event: event.to_string(),
            });
        }
    }
    Ok(())
}

/// What the store holds for one item, as strings that appear in its rendering.
pub struct Entry {
    /// Text that shows the item is present: a sentinel in its body or label.
    pub tokens: Vec<String>,
    /// Handles or ids that print where the item is listed.
    pub ids: Vec<String>,
    pub derivation: Derivation,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SeparationError {
    /// An inferred item's text or id is in a strict output.
    InferredLeaked { found: String },
    /// An item is shown without the derivation the store holds for it.
    WrongLabel {
        token: String,
        expected: &'static str,
        found: Option<String>,
    },
    /// An exempted inferred item is listed without the inferred label.
    ExemptUnlabelled { id: String },
}

const PROVENANCE_PREFIX: &str = "provenance:";
const DERIVATION_KEY: &str = "derivation: ";
const ITEM_HEAD_START: char = '[';
const FENCE_START: &str = "<<<";

/// The derivation printed with the item whose text is on `lines[at]`: the first
/// provenance line from there on, unless another item starts first.
fn label_at(lines: &[&str], at: usize) -> Option<String> {
    for (offset, line) in lines[at..].iter().enumerate() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(PROVENANCE_PREFIX) {
            let rest = rest.trim_start().strip_prefix(DERIVATION_KEY)?;
            return rest.split(';').next().map(|label| label.trim().to_string());
        }
        let starts_item = trimmed.starts_with(ITEM_HEAD_START) || trimmed.starts_with(FENCE_START);
        if trimmed.is_empty() || (offset > 0 && starts_item) {
            return None;
        }
    }
    None
}

/// Checks the second guarantee on one tool's `rendered` text. In strict mode an
/// entry that is neither `exempt` nor recorded may not appear at all. In both,
/// every appearance of an entry's token carries the entry's derivation. An
/// `exempt` entry is the one inferred item a strict output may show, and it
/// must show it labelled inferred.
pub fn check_separation(
    rendered: &str,
    mode: Mode,
    truth: &[Entry],
    exempt: &[String],
) -> Result<(), SeparationError> {
    let lines: Vec<&str> = rendered.lines().collect();
    for entry in truth {
        let excused = entry.ids.iter().any(|id| exempt.contains(id));
        let inferred = entry.derivation == Derivation::Inferred;
        if mode == Mode::Strict && inferred && !excused {
            let leaked = entry.tokens.iter().chain(&entry.ids);
            if let Some(found) = leaked.into_iter().find(|text| rendered.contains(*text)) {
                return Err(SeparationError::InferredLeaked {
                    found: found.clone(),
                });
            }
        }
        if excused {
            for id in entry.ids.iter().filter(|id| exempt.contains(*id)) {
                let at = lines.iter().position(|line| line.contains(id.as_str()));
                if at.is_some_and(|at| label_at(&lines, at).as_deref() != Some("inferred")) {
                    return Err(SeparationError::ExemptUnlabelled { id: id.clone() });
                }
            }
        }
        for token in &entry.tokens {
            for at in (0..lines.len()).filter(|&at| lines[at].contains(token.as_str())) {
                let found = label_at(&lines, at);
                let matches = match &found {
                    Some(label) => label == entry.derivation.as_str(),
                    None => !inferred,
                };
                if !matches {
                    return Err(SeparationError::WrongLabel {
                        token: token.clone(),
                        expected: entry.derivation.as_str(),
                        found,
                    });
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
