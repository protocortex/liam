// SPDX-License-Identifier: Apache-2.0
//! The entity page: a short, dated list of the sentences recorded about one
//! entity. It is cut from those sentences and never rewritten, so it cannot
//! state anything a memory did not.

use liam_store::NodeId;

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
/// is left out, except the newest, which is cut to fit.
const MAX_PAGE_CHARS: usize = 600;

/// Marks where the newest mention was cut to fit the page.
const CUT_MARKER: &str = "...";

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

/// The dated line a mention takes on the page.
fn line(mention: &Mention) -> String {
    format!("{}: {}", fmt_millis(mention.valid_from_ms), mention.text)
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
    let header = format!("{name} ({kind})");
    let mut used = header.chars().count();
    let mut shown: Vec<&Mention> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    for mention in selection(&without_near_duplicates(mentions)) {
        let mut text = line(mention);
        // The separating newline counts as one character.
        let cost = text.chars().count() + 1;
        if used + cost > MAX_PAGE_CHARS {
            if !shown.is_empty() {
                continue;
            }
            let room = MAX_PAGE_CHARS.saturating_sub(used + 1 + CUT_MARKER.len());
            text = text.chars().take(room).collect::<String>() + CUT_MARKER;
        }
        used += text.chars().count() + 1;
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
mod evaluation;
#[cfg(test)]
mod tests;
