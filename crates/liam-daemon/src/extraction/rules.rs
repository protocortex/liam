// SPDX-License-Identifier: Apache-2.0
//! The cheap deterministic extractor: dates, URLs, ticket-like ids and known
//! entities found in one event's text. Pure: no model, no store access.
//!
//! Every pattern runs on the `regex` crate's linear-time engine or on plain
//! substring search, so no input can cause catastrophic backtracking, and the
//! text is cut to a character bound first so matching work is bounded too.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use liam_store::ExtractorName;
use regex::Regex;

/// Stamped on every fact this extractor produces, with the version below.
pub const RULES_EXTRACTOR: ExtractorName = ExtractorName::Rules;

/// Bumped whenever a pattern or a predicate changes what this extractor emits.
pub const RULES_EXTRACTOR_VERSION: &str = "1";

const PREDICATE_DATE: &str = "references_date";
const PREDICATE_URL: &str = "references_url";
const PREDICATE_ID: &str = "references_id";
const PREDICATE_MENTIONED_IN: &str = "mentioned_in";
const EVENT_SUBJECT_PREFIX: &str = "event:";

/// Dates outside this range are far more likely to be version strings or
/// noise than a calendar date.
const MIN_YEAR: u32 = 1000;
const MAX_YEAR: u32 = 9999;
const MONTHS_PER_YEAR: u32 = 12;
const FEBRUARY: u32 = 2;
const LEAP_FEBRUARY_DAYS: u32 = 29;
const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Trailing characters that end a sentence or a quotation, not a URL.
const URL_TRAILING_PUNCTUATION: [char; 7] = ['.', ',', ';', ':', '!', '?', '\''];
/// Closing brackets are trimmed only when the URL has no opener for them, so
/// a Wikipedia style `/Rust_(language)` keeps its parenthesis.
const URL_BRACKET_PAIRS: [(char, char); 3] = [('(', ')'), ('[', ']'), ('{', '}')];

static DATE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").expect("date pattern is valid"));
static URL_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)https?://[^\s\p{Cc}<>"`]+"#).expect("url pattern is valid"));
/// Two to ten capital letters or digits starting with a letter, a hyphen,
/// then one to nine digits: `ABC-123`.
static ID_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][A-Z0-9]{1,9}-[0-9]{1,9}").expect("id pattern is valid"));

/// A fact the rules found, before the hook turns it into a `NewFact`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RuleFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// The event the fact came from, its only source.
    pub source_event_id: String,
}

/// Finds dates, URLs, ids and the `known_entities` in `text`, looking at no
/// more than the first `max_chars` characters. Output is sorted by
/// `(subject, predicate, object)` and holds no duplicates.
pub fn extract_rules(
    event_id: &str,
    text: &str,
    known_entities: &[String],
    max_chars: usize,
) -> Vec<RuleFact> {
    let text = truncate_at_char_boundary(text, max_chars);
    let event_subject = format!("{EVENT_SUBJECT_PREFIX}{event_id}");
    let fact = |subject: &str, predicate: &str, object: &str| RuleFact {
        subject: subject.to_string(),
        predicate: predicate.to_string(),
        object: object.to_string(),
        source_event_id: event_id.to_string(),
    };

    let mut facts: BTreeSet<RuleFact> = BTreeSet::new();
    for (predicate, objects) in [
        (PREDICATE_DATE, find_dates(text)),
        (PREDICATE_URL, find_urls(text)),
        (PREDICATE_ID, find_ids(text)),
    ] {
        facts.extend(
            objects
                .iter()
                .map(|object| fact(&event_subject, predicate, object)),
        );
    }
    facts.extend(
        find_entities(text, known_entities)
            .iter()
            .map(|key| fact(key, PREDICATE_MENTIONED_IN, &event_subject)),
    );
    facts.into_iter().collect()
}

fn truncate_at_char_boundary(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte_index, _)) => &text[..byte_index],
        None => text,
    }
}

fn find_dates(text: &str) -> Vec<String> {
    DATE_PATTERN
        .find_iter(text)
        .filter(|m| is_whole_date_token(text, m.start(), m.end()))
        .map(|m| m.as_str())
        .filter(|candidate| is_calendar_date(candidate))
        .map(str::to_string)
        .collect()
}

/// A date is not part of a longer digit or hyphen chain such as
/// `2026-01-02-03` or `1-2026-01-02`.
fn is_whole_date_token(text: &str, start: usize, end: usize) -> bool {
    let mut before = text[..start].chars().rev();
    let mut after = text[end..].chars();
    let (prev, next) = (before.next(), after.next());
    let glued_before = prev.is_some_and(char::is_alphanumeric)
        || (prev == Some('-') && before.next().is_some_and(|c| c.is_ascii_digit()));
    let glued_after = next.is_some_and(char::is_alphanumeric)
        || (next == Some('-') && after.next().is_some_and(|c| c.is_ascii_digit()));
    !glued_before && !glued_after
}

/// `candidate` is `YYYY-MM-DD` by construction of the pattern.
fn is_calendar_date(candidate: &str) -> bool {
    let field = |range: std::ops::Range<usize>| candidate[range].parse::<u32>().ok();
    let (Some(year), Some(month), Some(day)) = (field(0..4), field(5..7), field(8..10)) else {
        return false;
    };
    if !(MIN_YEAR..=MAX_YEAR).contains(&year) || !(1..=MONTHS_PER_YEAR).contains(&month) {
        return false;
    }
    let mut days = DAYS_IN_MONTH[(month - 1) as usize];
    if month == FEBRUARY && is_leap_year(year) {
        days = LEAP_FEBRUARY_DAYS;
    }
    (1..=days).contains(&day)
}

fn is_leap_year(year: u32) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

fn find_urls(text: &str) -> Vec<String> {
    URL_PATTERN
        .find_iter(text)
        .filter_map(|m| trim_url(m.as_str()))
        .map(str::to_string)
        .collect()
}

/// Drops trailing punctuation and unmatched closing brackets, then the URL
/// when nothing follows the scheme. Bracket balance is counted once and
/// updated as characters are dropped, so a long run of closers stays linear.
fn trim_url(raw: &str) -> Option<&str> {
    let mut balance = URL_BRACKET_PAIRS.map(|(open, close)| {
        (raw.matches(close).count() as isize) - (raw.matches(open).count() as isize)
    });
    let mut url = raw;
    while let Some(last) = url.chars().next_back() {
        let unmatched_closer = URL_BRACKET_PAIRS
            .iter()
            .position(|&(_, close)| close == last)
            .filter(|&pair| balance[pair] > 0);
        if let Some(pair) = unmatched_closer {
            balance[pair] -= 1;
        } else if !URL_TRAILING_PUNCTUATION.contains(&last) {
            break;
        }
        url = &url[..url.len() - last.len_utf8()];
    }
    let (_, rest) = url.split_once("://")?;
    (!rest.is_empty()).then_some(url)
}

fn find_ids(text: &str) -> Vec<String> {
    ID_PATTERN
        .find_iter(text)
        .filter(|m| is_whole_word(text, m.start(), m.end()))
        .map(|m| m.as_str().to_string())
        .collect()
}

/// The keys of the known entities that appear as whole words, matched case
/// insensitively. A key is the entity name trimmed and lower-cased, the same
/// normalization the store applies to entity pages.
fn find_entities(text: &str, known_entities: &[String]) -> BTreeSet<String> {
    if known_entities.is_empty() {
        return BTreeSet::new();
    }
    let haystack = text.to_lowercase();
    known_entities
        .iter()
        .map(|name| name.trim().to_lowercase())
        .filter(|key| !key.is_empty())
        .filter(|key| {
            haystack
                .match_indices(key.as_str())
                .any(|(start, found)| is_whole_word(&haystack, start, start + found.len()))
        })
        .collect()
}

/// The characters on both sides of `start..end`, if any, are not word
/// characters.
fn is_whole_word(text: &str, start: usize, end: usize) -> bool {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    !text[..start].chars().next_back().is_some_and(is_word)
        && !text[end..].chars().next().is_some_and(is_word)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVENT: &str = "e1";
    const CAP: usize = 8192;
    const BACKSTOP_SECS: u64 = 5;
    const ONE_MB: usize = 1024 * 1024;
    const FUZZ_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
    const FUZZ_ROUNDS: usize = 200;
    const FUZZ_LEN: usize = 512;
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
    const LCG_OUTPUT_SHIFT: u32 = 56;

    fn run(text: &str) -> Vec<(String, String, String)> {
        run_with(text, &[])
    }

    fn run_with(text: &str, entities: &[&str]) -> Vec<(String, String, String)> {
        let entities: Vec<String> = entities.iter().map(|e| e.to_string()).collect();
        extract_rules(EVENT, text, &entities, CAP)
            .into_iter()
            .map(|f| (f.subject, f.predicate, f.object))
            .collect()
    }

    fn about_event(predicate: &str, object: &str) -> (String, String, String) {
        ("event:e1".into(), predicate.into(), object.into())
    }

    fn objects(facts: &[(String, String, String)], predicate: &str) -> Vec<String> {
        facts
            .iter()
            .filter(|(_, p, _)| p == predicate)
            .map(|(_, _, o)| o.clone())
            .collect()
    }

    /// Deterministic pseudo-random bytes, so a failure reproduces.
    fn pseudo_random_bytes(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(LCG_MULTIPLIER)
                    .wrapping_add(LCG_INCREMENT);
                (state >> LCG_OUTPUT_SHIFT) as u8
            })
            .collect()
    }

    #[test]
    fn finds_dates_urls_ids_and_known_entities() {
        // Arrange
        let text = "Alice met Bob on 2026-03-04, see https://x.y/a and ABC-123.";

        // Act
        let facts = run_with(text, &["alice"]);

        // Assert
        assert_eq!(
            facts,
            vec![
                ("alice".into(), "mentioned_in".into(), "event:e1".into()),
                about_event("references_date", "2026-03-04"),
                about_event("references_id", "ABC-123"),
                about_event("references_url", "https://x.y/a"),
            ]
        );
    }

    #[test]
    fn facts_carry_their_source_event() {
        // Arrange
        let text = "on 2026-03-04";

        // Act
        let facts = extract_rules(EVENT, text, &[], CAP);

        // Assert
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].source_event_id, EVENT);
    }

    #[test]
    fn the_extractor_identity_is_rules_at_a_fixed_version() {
        // Assert
        assert_eq!(RULES_EXTRACTOR.as_str(), "rules");
        assert!(!RULES_EXTRACTOR_VERSION.is_empty());
    }

    #[test]
    fn text_without_matches_yields_nothing() {
        for text in ["", "   \n\t ", "nothing to see here", "v1.2.3 and 12:30"] {
            // Act
            let facts = run_with(text, &["alice"]);

            // Assert
            assert!(facts.is_empty(), "{text:?} gave {facts:?}");
        }
    }

    #[test]
    fn only_real_calendar_dates_are_emitted() {
        for (text, expected) in [
            ("2026-13-45", false),
            ("2026-02-30", false),
            ("0000-00-00", false),
            ("2026-00-10", false),
            ("2026-04-31", false),
            ("2023-02-29", false),
            ("1900-02-29", false),
            ("0999-12-31", false),
            ("2024-02-29", true),
            ("2000-02-29", true),
            ("9999-12-31", true),
            ("1000-01-01", true),
        ] {
            // Act
            let facts = run(text);

            // Assert
            assert_eq!(
                !objects(&facts, "references_date").is_empty(),
                expected,
                "{text}"
            );
        }
    }

    #[test]
    fn a_date_inside_a_longer_chain_is_not_a_second_date() {
        for text in [
            "2026-01-02-03",
            "1-2026-01-02",
            "x2026-01-02",
            "2026-01-02y",
        ] {
            // Act
            let facts = run(text);

            // Assert
            assert!(objects(&facts, "references_date").is_empty(), "{text}");
        }
    }

    #[test]
    fn urls_lose_trailing_punctuation_and_brackets() {
        for (text, expected) in [
            ("https://x.y/a).", "https://x.y/a"),
            ("https://x.y/a,", "https://x.y/a"),
            ("<https://x.y/a>", "https://x.y/a"),
            ("(see https://x.y/a)", "https://x.y/a"),
            ("[https://x.y/a]", "https://x.y/a"),
            ("\"https://x.y/a\"", "https://x.y/a"),
            (
                "https://x.y/Rust_(language).",
                "https://x.y/Rust_(language)",
            ),
            ("HTTP://X.Y/A?q=1&r=2#f", "HTTP://X.Y/A?q=1&r=2#f"),
        ] {
            // Act
            let facts = run(text);

            // Assert
            assert_eq!(objects(&facts, "references_url"), vec![expected], "{text}");
        }
    }

    #[test]
    fn a_scheme_with_no_host_is_not_a_url() {
        for text in ["http://", "https://.", "see http://)", "ftp://x.y"] {
            // Act
            let facts = run(text);

            // Assert
            assert!(objects(&facts, "references_url").is_empty(), "{text}");
        }
    }

    #[test]
    fn ids_must_be_whole_tokens() {
        // Act
        let long = run("XABC-1234");
        let short = run("ABC-123");
        let glued = run("xABC-123 ABC-123x ABC-123_");
        let embedded = run("a ABC-123.");

        // Assert
        assert_eq!(objects(&long, "references_id"), vec!["XABC-1234"]);
        assert_eq!(objects(&short, "references_id"), vec!["ABC-123"]);
        assert!(objects(&glued, "references_id").is_empty());
        assert_eq!(objects(&embedded, "references_id"), vec!["ABC-123"]);
    }

    #[test]
    fn an_id_inside_a_url_is_reported_once_beside_the_url() {
        // Arrange
        let text = "https://tracker.example/browse/ABC-123";

        // Act
        let facts = run(text);

        // Assert
        assert_eq!(objects(&facts, "references_id"), vec!["ABC-123"]);
        assert_eq!(objects(&facts, "references_url"), vec![text]);
    }

    #[test]
    fn repeated_matches_are_reported_once() {
        // Act
        let facts = run("ABC-123 ABC-123 2026-03-04 2026-03-04");

        // Assert
        assert_eq!(facts.len(), 2);
    }

    #[test]
    fn entities_match_as_whole_words_only() {
        for (text, expected) in [
            ("Alice went home", true),
            ("went home with ALICE", true),
            ("Alice", true),
            ("Alice, Bob", true),
            ("malice aforethought", false),
            ("Alicex", false),
            ("alice_b", false),
        ] {
            // Act
            let facts = run_with(text, &["Alice"]);

            // Assert
            assert_eq!(!facts.is_empty(), expected, "{text}");
        }
    }

    #[test]
    fn entity_subjects_use_the_normalized_key() {
        // Act
        let facts = run_with("met Acme Corp today", &["  Acme CORP "]);

        // Assert
        assert_eq!(
            facts,
            vec![("acme corp".into(), "mentioned_in".into(), "event:e1".into())]
        );
    }

    #[test]
    fn blank_entities_are_ignored() {
        // Act
        let facts = run_with("some text", &["", "   "]);

        // Assert
        assert!(facts.is_empty());
    }

    #[test]
    fn unicode_entities_and_text_match() {
        // Act
        let facts = run_with(
            "Reunião com José em 2026-03-04 ✓ 日本語",
            &["JOSÉ", "日本語"],
        );

        // Assert
        assert_eq!(
            facts,
            vec![
                about_event("references_date", "2026-03-04"),
                ("josé".into(), "mentioned_in".into(), "event:e1".into()),
                ("日本語".into(), "mentioned_in".into(), "event:e1".into()),
            ]
        );
    }

    #[test]
    fn truncation_happens_at_the_cap_on_a_character_boundary() {
        // Arrange
        let cap = 10;
        let at = |text: &str| extract_rules(EVENT, text, &[], cap);
        let date = "2026-03-04";

        // Act
        let exactly = at(date);
        let one_over = at(&format!("{date}x"));
        let one_under = at(&date[..cap - 1]);
        let split_char = at(&format!("{}é", &date[..cap - 1]));
        let wide = at(&"é".repeat(cap + 1));

        // Assert
        assert_eq!(exactly.len(), 1);
        assert_eq!(one_over.len(), 1, "the 11th character is cut away");
        assert!(one_under.is_empty());
        assert!(split_char.is_empty());
        assert!(wide.is_empty());
    }

    #[test]
    fn a_cap_of_zero_yields_nothing() {
        // Act
        let facts = extract_rules(EVENT, "2026-03-04", &[], 0);

        // Assert
        assert!(facts.is_empty());
    }

    #[test]
    fn text_past_the_cap_is_ignored() {
        // Arrange
        let adversarial = format!("{}ABC-123", "a".repeat(CAP));
        let prefix: String = adversarial.chars().take(CAP).collect();

        // Act
        let full = extract_rules(EVENT, &adversarial, &[], CAP);
        let truncated = extract_rules(EVENT, &prefix, &[], CAP);

        // Assert
        assert_eq!(full, truncated);
        assert!(full.is_empty());
    }

    #[test]
    fn pathological_input_finishes_within_the_backstop_and_is_bounded() {
        // Arrange
        let cap = 100_000;
        let shapes = [
            "a".repeat(cap * 2),
            "https://".repeat(cap),
            "(".repeat(cap) + &")".repeat(cap),
            "2026-".repeat(cap),
            "AAAA-".repeat(cap),
            "https://x.y/".to_string() + &")".repeat(cap),
        ];
        let started = std::time::Instant::now();

        for text in &shapes {
            // Act
            let facts = extract_rules(EVENT, text, &["a".into()], cap);

            // Assert
            assert!(facts.len() <= cap, "output is bounded by the input");
        }
        assert!(started.elapsed().as_secs() < BACKSTOP_SECS);
    }

    #[test]
    fn a_one_megabyte_input_is_capped_to_the_configured_bound() {
        // Arrange
        let mut text = "x ".repeat(ONE_MB / 2);
        text.push_str(" https://late.example/after-the-cap");
        let started = std::time::Instant::now();

        // Act
        let facts = extract_rules(EVENT, &text, &["x".into()], CAP);

        // Assert
        assert_eq!(facts.len(), 1, "only the entity inside the cap");
        assert!(started.elapsed().as_secs() < BACKSTOP_SECS);
    }

    #[test]
    fn random_bytes_and_control_characters_never_panic_and_stay_ordered() {
        for round in 0..FUZZ_ROUNDS {
            // Arrange
            let bytes = pseudo_random_bytes(FUZZ_SEED + round as u64, FUZZ_LEN);
            let lossy = String::from_utf8_lossy(&bytes).into_owned();
            let controls: String = bytes.iter().map(|b| char::from(b % 0x20)).collect();
            let seeded = format!("{lossy} 2026-03-04 https://x.y/a\u{0}b ABC-1 {controls}");

            for text in [lossy, controls, seeded] {
                // Act
                let facts = extract_rules(EVENT, &text, &["a".into(), "é".into()], CAP);

                // Assert
                assert!(
                    facts.windows(2).all(|w| w[0] < w[1]),
                    "sorted, no duplicates"
                );
                assert!(facts.len() <= text.len());
            }
        }
    }

    #[test]
    fn urls_stop_at_control_characters() {
        // Act
        let facts = run("https://x.y/a\u{7}b");

        // Assert
        assert_eq!(objects(&facts, "references_url"), vec!["https://x.y/a"]);
    }

    #[test]
    fn output_is_sorted_and_stable_across_calls() {
        // Arrange
        let text = "zed ABC-9 2026-03-04 https://b.example https://a.example alice zed";
        let entities = ["zed".to_string(), "alice".to_string()];

        // Act
        let first = extract_rules(EVENT, text, &entities, CAP);
        let second = extract_rules(EVENT, text, &entities, CAP);

        // Assert
        assert_eq!(first, second);
        assert!(first.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(first[0].subject, "alice");
    }
}
