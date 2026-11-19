// SPDX-License-Identifier: Apache-2.0
//! The entity page: a short, dated list of the sentences recorded about one
//! entity. It is cut from those sentences and never rewritten, so it cannot
//! state anything a memory did not.

use liam_store::NodeId;
use serde_json::{Map, Value};

use crate::render::fmt_millis;
use crate::summary::tree::cosine;

/// Two mentions at least this alike count as the same fact, and only the newer
/// one is kept.
const NEAR_DUPLICATE_COSINE: f32 = 0.92;

/// Newest mentions always kept, so the page shows the latest state.
const KEEP_NEWEST: usize = 2;

/// Further mentions kept for being the most like the rest, so the page carries
/// what the mentions share most.
const KEEP_CENTRAL: usize = 2;

/// Characters a page may hold, header included. A mention that would pass it
/// is left out, except the first shown, which is cut to fit.
const MAX_PAGE_CHARS: usize = 600;

/// Characters the header may take, so a long entity name cannot leave no room
/// for the newest mention.
const MAX_HEADER_CHARS: usize = MAX_PAGE_CHARS / HEADER_SHARE;

/// The header may take at most one part in this many of a page.
const HEADER_SHARE: usize = 3;

/// Characters the newline after a line takes.
const NEWLINE_CHARS: usize = 1;

/// Marks where a header or the newest mention was cut to fit the page.
const CUT_MARKER: &str = "...";

/// The attribute that tells a page this builder wrote from one an earlier
/// version wrote, which a model composed and which is never shown or trusted.
pub const PAGE_MARKER_KEY: &str = "page_builder";

/// One mention of an entity with the vector its text is stored under.
pub struct Mention {
    pub id: NodeId,
    pub valid_from_ms: i64,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// A page and the mentions it was built from, oldest first.
#[derive(Debug, PartialEq, Eq)]
pub struct EntityPage {
    pub text: String,
    pub source_ids: Vec<NodeId>,
}

/// Mentions newest first, without any that is a near duplicate of a newer one.
fn without_near_duplicates(mentions: &[Mention]) -> Vec<&Mention> {
    let mut newest_first: Vec<&Mention> = mentions.iter().collect();
    newest_first.sort_by(|a, b| {
        b.valid_from_ms
            .cmp(&a.valid_from_ms)
            .then_with(|| a.id.as_str().cmp(b.id.as_str()))
    });
    let mut kept: Vec<&Mention> = Vec::new();
    for mention in newest_first {
        let duplicate = kept
            .iter()
            .any(|k| cosine(&k.embedding, &mention.embedding) >= NEAR_DUPLICATE_COSINE);
        if !duplicate {
            kept.push(mention);
        }
    }
    kept
}

/// Mean cosine of `target` to every other mention in `all`.
fn centrality(target: &Mention, all: &[&Mention]) -> f32 {
    let others: Vec<f32> = all
        .iter()
        .filter(|m| m.id != target.id)
        .map(|m| cosine(&target.embedding, &m.embedding))
        .collect();
    // `max(1)` only guards the division: the target is always compared with
    // at least one other mention when a page is built.
    others.iter().sum::<f32>() / others.len().max(1) as f32
}

/// `attributes` of an entity with the page marker added.
pub fn stamped(attributes: Value) -> Value {
    let mut map = match attributes {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    map.insert(PAGE_MARKER_KEY.to_string(), Value::Bool(true));
    Value::Object(map)
}

/// True when the page with these `attributes` was written by this builder.
pub fn is_stamped(attributes: &Value) -> bool {
    attributes.get(PAGE_MARKER_KEY).is_some()
}

/// True for a row that holds page text the builder did not write: an earlier
/// version had a model compose it.
pub fn is_legacy_page(text: &str, attributes: &Value) -> bool {
    !text.is_empty() && !is_stamped(attributes)
}

/// `text` cut to at most `limit` characters, ending in the cut marker when
/// something was removed.
fn cut_to(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let room = limit.saturating_sub(CUT_MARKER.len());
    text.chars().take(room).collect::<String>() + CUT_MARKER
}

/// The dated line a mention takes on the page. Whitespace runs, newlines
/// included, become one space so a mention stays on its own line and cannot
/// pose as another dated line.
fn line(mention: &Mention) -> String {
    let text = mention
        .text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    format!("{}: {text}", fmt_millis(mention.valid_from_ms))
}

/// The mentions to show, newest first, then the most central of the rest.
fn selection<'a>(deduplicated: &[&'a Mention]) -> Vec<&'a Mention> {
    let (newest, rest) = deduplicated.split_at(KEEP_NEWEST.min(deduplicated.len()));
    let mut central: Vec<(f32, &Mention)> = rest
        .iter()
        .map(|m| (centrality(m, deduplicated), *m))
        .collect();
    // The sort is stable and `rest` is newest first, so equal scores keep the
    // newer mention.
    central.sort_by(|a, b| b.0.total_cmp(&a.0));
    newest
        .iter()
        .copied()
        .chain(central.into_iter().take(KEEP_CENTRAL).map(|(_, m)| m))
        .collect()
}

/// Builds the page of the entity `name` of `kind` from its `mentions`. The
/// same mentions always give the same page. An entity with no mention has an
/// empty page.
pub fn build_page(kind: &str, name: &str, mentions: &[Mention]) -> EntityPage {
    if mentions.is_empty() {
        return EntityPage {
            text: String::new(),
            source_ids: Vec::new(),
        };
    }
    let header = cut_to(&format!("{name} ({kind})"), MAX_HEADER_CHARS);
    let mut used = header.chars().count();
    let mut shown: Vec<&Mention> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    for mention in selection(&without_near_duplicates(mentions)) {
        let mut text = line(mention);
        // The separating newline counts as one character.
        let cost = text.chars().count() + NEWLINE_CHARS;
        if used + cost > MAX_PAGE_CHARS {
            if !shown.is_empty() {
                continue;
            }
            text = cut_to(&text, MAX_PAGE_CHARS - used - NEWLINE_CHARS);
        }
        used += text.chars().count() + NEWLINE_CHARS;
        shown.push(mention);
        lines.push(text);
    }
    let mut chosen: Vec<(&Mention, String)> = shown.into_iter().zip(lines).collect();
    chosen.sort_by(|a, b| {
        a.0.valid_from_ms
            .cmp(&b.0.valid_from_ms)
            .then_with(|| a.0.id.as_str().cmp(b.0.id.as_str()))
    });
    let source_ids = chosen.iter().map(|(m, _)| m.id.clone()).collect();
    let body: Vec<String> = chosen.into_iter().map(|(_, text)| text).collect();
    EntityPage {
        text: format!("{header}\n{}", body.join("\n")),
        source_ids,
    }
}

#[cfg(test)]
mod tests;
