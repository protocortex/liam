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
/// Byte ranges of the fields in a `YYYY-MM-DD` candidate.
const YEAR_FIELD: std::ops::Range<usize> = 0..4;
const MONTH_FIELD: std::ops::Range<usize> = 5..7;
const DAY_FIELD: std::ops::Range<usize> = 8..10;
const FEBRUARY: u32 = 2;
const LEAP_FEBRUARY_DAYS: u32 = 29;
const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Trailing characters that end a sentence or a quotation, not a URL.
const URL_TRAILING_PUNCTUATION: [char; 7] = ['.', ',', ';', ':', '!', '?', '\''];
/// Closing quotes, sentence marks and ellipses outside ASCII, which end a
/// sentence or a quotation rather than a URL.
const URL_TRAILING_UNICODE_PUNCTUATION: [char; 17] = [
    '\u{201D}', '\u{2019}', '\u{00BB}', '\u{203A}', '\u{3002}', '\u{3001}', '\u{2026}', '\u{FF0C}',
    '\u{FF1B}', '\u{FF1A}', '\u{FF01}', '\u{FF1F}', '\u{FF09}', '\u{300D}', '\u{300F}', '\u{3011}',
    '\u{FF0E}',
];
/// Markdown emphasis that wraps a URL, as in `**https://x.y/a**`.
const URL_TRAILING_EMPHASIS: [char; 3] = ['*', '_', '~'];
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
    let scan = Scan::new(text, max_chars);
    let event_subject = format!("{EVENT_SUBJECT_PREFIX}{event_id}");
    let fact = |subject: &str, predicate: &str, object: &str| RuleFact {
        subject: subject.to_string(),
        predicate: predicate.to_string(),
        object: object.to_string(),
        source_event_id: event_id.to_string(),
    };

    let mut facts: BTreeSet<RuleFact> = BTreeSet::new();
    for (predicate, objects) in [
        (PREDICATE_DATE, find_dates(&scan)),
        (PREDICATE_URL, find_urls(&scan)),
        (PREDICATE_ID, find_ids(&scan)),
    ] {
        facts.extend(
            objects
                .iter()
                .map(|object| fact(&event_subject, predicate, object)),
        );
    }
    facts.extend(
        find_entities(&scan, known_entities)
            .iter()
            .map(|key| fact(key, PREDICATE_MENTIONED_IN, &event_subject)),
    );
    facts.into_iter().collect()
}

/// The text cut to a character bound. Matching runs on the window only, but
/// boundary checks look at the full text, so the cut never turns a longer
/// token into a shorter one that looks complete.
struct Scan<'a> {
    full: &'a str,
    cut: usize,
}

impl<'a> Scan<'a> {
    fn new(full: &'a str, max_chars: usize) -> Self {
        let cut = full
            .char_indices()
            .nth(max_chars)
            .map_or(full.len(), |(byte_index, _)| byte_index);
        Self { full, cut }
    }

    fn window(&self) -> &'a str {
        &self.full[..self.cut]
    }

    fn is_whole_word(&self, start: usize, end: usize) -> bool {
        is_whole_word(self.full, start, end)
    }

    /// The token is not part of a hyphenated digit chain such as
    /// `2026-01-02-03`, `1-2026-01-02` or `CVE-2024-12345`.
    fn is_hyphen_digit_free(&self, start: usize, end: usize) -> bool {
        let mut before = self.full[..start].chars().rev();
        let mut after = self.full[end..].chars();
        let glued_before =
            before.next() == Some('-') && before.next().is_some_and(|c| c.is_ascii_digit());
        let glued_after =
            after.next() == Some('-') && after.next().is_some_and(|c| c.is_ascii_digit());
        !glued_before && !glued_after
    }

    /// A match that reaches the cut might have continued past it.
    fn ends_at_cut_mid_text(&self, end: usize) -> bool {
        end == self.cut && self.full[end..].chars().next().is_some_and(continues_url)
    }
}

fn find_dates(scan: &Scan) -> Vec<String> {
    DATE_PATTERN
        .find_iter(scan.window())
        .filter(|m| is_whole_date_token(scan, m.start(), m.end()))
        .map(|m| m.as_str())
        .filter(|candidate| is_calendar_date(candidate))
        .map(str::to_string)
        .collect()
}

/// A date stands alone, or is followed by an ISO time (`T10:00:00Z`).
fn is_whole_date_token(scan: &Scan, start: usize, end: usize) -> bool {
    if !scan.is_hyphen_digit_free(start, end) {
        return false;
    }
    let mut after = scan.full[end..].chars();
    let time_follows =
        matches!(after.next(), Some('T' | 't')) && after.next().is_some_and(|c| c.is_ascii_digit());
    let word_before = scan.full[..start]
        .chars()
        .next_back()
        .is_some_and(is_word_char);
    !word_before && (time_follows || scan.is_whole_word(start, end))
}

/// `candidate` is `YYYY-MM-DD` by construction of the pattern.
fn is_calendar_date(candidate: &str) -> bool {
    let field = |range: std::ops::Range<usize>| candidate[range].parse::<u32>().ok();
    let (Some(year), Some(month), Some(day)) =
        (field(YEAR_FIELD), field(MONTH_FIELD), field(DAY_FIELD))
    else {
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

fn find_urls(scan: &Scan) -> Vec<String> {
    URL_PATTERN
        .find_iter(scan.window())
        .filter(|m| !scan.ends_at_cut_mid_text(m.end()))
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
        } else if !is_trailing_url_noise(last) {
            break;
        }
        url = &url[..url.len() - last.len_utf8()];
    }
    let (_, rest) = url.split_once("://")?;
    (!rest.is_empty()).then_some(url)
}

/// Whether `c` could be part of a URL match, mirroring `URL_PATTERN`.
fn continues_url(c: char) -> bool {
    !(c.is_whitespace() || c.is_control() || matches!(c, '<' | '>' | '"' | '`'))
}

fn is_trailing_url_noise(c: char) -> bool {
    URL_TRAILING_PUNCTUATION.contains(&c)
        || URL_TRAILING_UNICODE_PUNCTUATION.contains(&c)
        || URL_TRAILING_EMPHASIS.contains(&c)
}

/// `UTF-8`, `SHA-256` and `ISO-8601` match the id shape and are emitted: the
/// extractor cannot tell them from a ticket without a vocabulary.
fn find_ids(scan: &Scan) -> Vec<String> {
    ID_PATTERN
        .find_iter(scan.window())
        .filter(|m| scan.is_whole_word(m.start(), m.end()))
        .filter(|m| scan.is_hyphen_digit_free(m.start(), m.end()))
        .map(|m| m.as_str().to_string())
        .collect()
}

/// The keys of the known entities that appear as whole words, matched case
/// insensitively. A key is the entity name trimmed and lower-cased, the same
/// normalization the store applies to entity pages.
fn find_entities(scan: &Scan<'_>, known_entities: &[String]) -> BTreeSet<String> {
    if known_entities.is_empty() {
        return BTreeSet::new();
    }
    // Word boundaries are judged on the full text, so a name cut by the cap
    // is not taken for a shorter whole word; a match must still end inside
    // the window.
    let haystack = scan.full.to_lowercase();
    let window_end = scan.window().to_lowercase().len();
    known_entities
        .iter()
        .map(|name| name.trim().to_lowercase())
        .filter(|key| !key.is_empty())
        .filter(|key| {
            haystack.match_indices(key.as_str()).any(|(start, found)| {
                let end = start + found.len();
                end <= window_end && is_whole_word(&haystack, start, end)
            })
        })
        .collect()
}

/// The characters on both sides of `start..end`, if any, are not word
/// characters.
fn is_whole_word(text: &str, start: usize, end: usize) -> bool {
    !text[..start].chars().next_back().is_some_and(is_word_char)
        && !text[end..].chars().next().is_some_and(is_word_char)
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
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
            ("2026-01-00", false),
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
            "2026-01-02_",
            "_2026-01-02",
            "2026-03-04x",
        ] {
            // Act
            let facts = run(text);

            // Assert
            assert!(objects(&facts, "references_date").is_empty(), "{text}");
        }
    }

    #[test]
    fn a_date_followed_by_a_time_keeps_its_date() {
        for text in [
            "deploy at 2026-03-04T10:00:00Z",
            "2026-03-04T10:00:00+02:00",
            "2026-03-04t10:00:00",
            "2026-03-04T10",
            "2026-03-04T10:00:00.123-0300",
        ] {
            // Act
            let facts = run(text);

            // Assert
            assert_eq!(
                objects(&facts, "references_date"),
                vec!["2026-03-04"],
                "{text}"
            );
        }
    }

    #[test]
    fn a_t_not_followed_by_a_digit_still_glues_a_date() {
        for text in ["2026-03-04T", "2026-03-04Tx", "2026-03-04today"] {
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
            ("https://x.y/a;", "https://x.y/a"),
            ("https://x.y/a:", "https://x.y/a"),
            ("https://x.y/a!", "https://x.y/a"),
            ("https://x.y/a?", "https://x.y/a"),
            ("https://x.y/a'", "https://x.y/a"),
            ("\u{201C}https://x.y/a\u{201D}", "https://x.y/a"),
            ("https://x.y/a\u{3002}", "https://x.y/a"),
            ("https://x.y/a\u{2026}", "https://x.y/a"),
            ("https://x.y/a\u{2019}", "https://x.y/a"),
            ("https://x.y/a\u{00BB}", "https://x.y/a"),
            ("**https://x.y/a**", "https://x.y/a"),
            ("_https://x.y/a_", "https://x.y/a"),
            ("~~https://x.y/a~~", "https://x.y/a"),
            ("https://x.y/a**b/c", "https://x.y/a**b/c"),
            ("https://x.y/a\u{2026}b", "https://x.y/a\u{2026}b"),
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
        let glued = run("xABC-123 ABC-123x ABC-123_ _ABC-123");
        let embedded = run("a ABC-123.");

        // Assert
        assert_eq!(objects(&long, "references_id"), vec!["XABC-1234"]);
        assert_eq!(objects(&short, "references_id"), vec!["ABC-123"]);
        assert!(objects(&glued, "references_id").is_empty());
        assert_eq!(objects(&embedded, "references_id"), vec!["ABC-123"]);
    }

    #[test]
    fn an_id_inside_a_hyphen_chain_is_not_a_partial_id() {
        for text in ["CVE-2024-12345", "ABC-123-4", "1-ABC-123", "ABC-123-"] {
            // Act
            let facts = run(text);

            // Assert
            let found = objects(&facts, "references_id");
            if text == "ABC-123-" {
                assert_eq!(found, vec!["ABC-123"], "{text}");
            } else {
                assert!(found.is_empty(), "{text} gave {found:?}");
            }
        }
    }

    #[test]
    fn standard_names_with_the_id_shape_are_emitted_as_ids() {
        // Act
        let facts = run("encode as UTF-8 and hash with SHA-256 per ISO-8601");

        // Assert
        assert_eq!(
            objects(&facts, "references_id"),
            vec!["ISO-8601", "SHA-256", "UTF-8"]
        );
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
    fn an_entity_cut_by_the_cap_is_not_taken_for_a_shorter_word() {
        // Arrange: "Alicex" is cut after "Alice" by the cap
        let text = "Alicex went home";
        let cap = "Alice".chars().count();

        // Act
        let cut = extract_rules(EVENT, text, &["Alice".to_string()], cap);
        let whole = extract_rules(EVENT, "Alice went home", &["Alice".to_string()], cap);

        // Assert
        assert!(cut.is_empty(), "{cut:?}");
        assert_eq!(whole.len(), 1);
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
        let one_over_space = at(&format!("{date} x"));
        let one_under = at(&date[..cap - 1]);
        let split_char = at(&format!("{}é", &date[..cap - 1]));
        let wide = at(&"é".repeat(cap + 1));

        // Assert
        assert_eq!(exactly.len(), 1);
        assert_eq!(one_over_space.len(), 1, "the 11th character is cut away");
        assert!(one_under.is_empty());
        assert!(split_char.is_empty());
        assert!(wide.is_empty());
    }

    #[test]
    fn a_cut_never_creates_a_fact_the_full_text_lacks() {
        for (text, cap) in [
            ("2026-03-04x", 10),
            ("2026-03-04-05", 10),
            ("2026-03-0412", 10),
            ("https://x.y/abc", 14),
            ("https://x.y/abc def", 14),
            ("ABC-12345", 6),
            ("ABC-123-4", 7),
            ("ABCDEF-123", 9),
        ] {
            // Act
            let facts = extract_rules(EVENT, text, &[], cap);

            // Assert
            assert!(facts.is_empty(), "{text} cut at {cap} gave {facts:?}");
        }
    }

    #[test]
    fn a_cut_at_a_real_token_end_keeps_the_fact() {
        for (text, cap, predicate, object) in [
            ("2026-03-04 x", 10, "references_date", "2026-03-04"),
            ("2026-03-04T10:00", 10, "references_date", "2026-03-04"),
            (
                "https://x.y/abc def",
                17,
                "references_url",
                "https://x.y/abc",
            ),
            ("ABC-123 x", 7, "references_id", "ABC-123"),
        ] {
            // Act
            let facts = extract_rules(EVENT, text, &[], cap);

            // Assert
            assert_eq!(facts.len(), 1, "{text} gave {facts:?}");
            assert_eq!(
                (&*facts[0].predicate, &*facts[0].object),
                (predicate, object)
            );
        }
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
        let adversarial = format!("{}ABC-123", "a ".repeat(CAP / 2));
        let prefix: String = adversarial.chars().take(CAP).collect();

        // Act
        let full = extract_rules(EVENT, &adversarial, &[], CAP);
        let truncated = extract_rules(EVENT, &prefix, &[], CAP);

        // Assert
        assert_eq!(full, truncated);
        assert!(full.is_empty());
        assert!(!extract_rules(EVENT, &adversarial, &[], CAP + 7).is_empty());
    }

    #[test]
    fn pathological_input_finishes_within_the_backstop_and_matches_its_prefix() {
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
            let prefix: String = text.chars().take(cap).collect();

            // Act
            let facts = extract_rules(EVENT, text, &["a".into()], cap);
            let prefix_facts = extract_rules(EVENT, &prefix, &["a".into()], cap);

            // Assert
            assert!(facts.len() <= cap, "output is bounded by the input");
            let only_in_prefix = prefix_facts.iter().filter(|f| !facts.contains(f));
            assert!(
                only_in_prefix.clone().all(|f| f.predicate == PREDICATE_URL),
                "only a URL running into the cut may differ: {:?}",
                only_in_prefix.collect::<Vec<_>>()
            );
            assert!(facts.iter().all(|f| prefix_facts.contains(f)));
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
