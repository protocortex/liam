// SPDX-License-Identifier: Apache-2.0
//! Strict parsing of the model's JSON answer. The answer is untrusted: a
//! malformed whole is an error, while one bad fact is dropped and counted so a
//! single slip never costs the rest of the batch.

use std::collections::BTreeSet;
use std::fmt;

use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::Value;

use super::iso_date::parse_iso_date;

/// Longest subject, predicate or object, in characters.
pub const MAX_FIELD_CHARS: usize = 512;
/// Facts kept from one answer; the rest are counted and cut.
pub const MAX_FACTS_PER_BATCH: usize = 64;
const FACTS_KEY: &str = "facts";
const MIN_CONFIDENCE: f64 = 0.0;
const MAX_CONFIDENCE: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    Asserted,
    Ended,
}

/// A fact that passed every shape check. Whether the cited text really says it
/// is the grounding check's question.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub polarity: Polarity,
    pub valid_from: Option<String>,
    pub valid_until: Option<String>,
    pub confidence: f64,
    /// Sorted indexes into the batch, each at most once, never empty.
    pub events: Vec<usize>,
}

#[derive(Debug, PartialEq)]
pub struct ParsedBatch {
    pub facts: Vec<ParsedFact>,
    /// Facts rejected for a shape, citation, date, or vocabulary problem.
    pub dropped: usize,
    /// Facts past `MAX_FACTS_PER_BATCH`, cut unread.
    pub truncated: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ResponseError {
    NotJson(String),
    NotAnObject,
    UnknownKey(String),
    MissingFacts,
    FactsNotAnArray,
    /// A repeated key: the usual JSON reader keeps the last silently, so the
    /// answer's meaning would depend on key order.
    DuplicateKey(String),
}

impl fmt::Display for ResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotJson(why) => write!(f, "the response is not valid JSON: {why}"),
            Self::NotAnObject => write!(f, "the response is not a JSON object"),
            Self::UnknownKey(key) => write!(f, "the response has the unknown key {key:?}"),
            Self::MissingFacts => write!(f, "the response has no {FACTS_KEY:?} key"),
            Self::FactsNotAnArray => write!(f, "the {FACTS_KEY:?} value is not an array"),
            Self::DuplicateKey(why) => write!(f, "the response repeats a key: {why}"),
        }
    }
}

impl std::error::Error for ResponseError {}

/// Reads the answer's top level with serde's derived struct, which rejects a
/// repeated key, and keeps each fact unparsed so it can be checked alone.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    facts: Vec<Box<RawValue>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFact {
    subject: String,
    predicate: String,
    object: String,
    polarity: Polarity,
    valid_from: Option<String>,
    valid_until: Option<String>,
    confidence: f64,
    events: Vec<usize>,
}

/// Parses `text` for a batch of `batch_len` events. `predicates` is the closed
/// vocabulary; a fact outside it is dropped.
pub fn parse_response(
    text: &str,
    batch_len: usize,
    predicates: &[String],
) -> Result<ParsedBatch, ResponseError> {
    let value: Value =
        serde_json::from_str(text).map_err(|error| ResponseError::NotJson(error.to_string()))?;
    let Value::Object(object) = value else {
        return Err(ResponseError::NotAnObject);
    };
    if let Some(key) = object.keys().find(|key| key.as_str() != FACTS_KEY) {
        return Err(ResponseError::UnknownKey(key.clone()));
    }
    match object.get(FACTS_KEY) {
        Some(Value::Array(_)) => {}
        Some(_) => return Err(ResponseError::FactsNotAnArray),
        None => return Err(ResponseError::MissingFacts),
    }
    let Envelope { facts: entries } = serde_json::from_str(text)
        .map_err(|error| ResponseError::DuplicateKey(error.to_string()))?;

    let truncated = entries.len().saturating_sub(MAX_FACTS_PER_BATCH);
    let read = entries.len() - truncated;
    let facts: Vec<ParsedFact> = entries
        .into_iter()
        .take(MAX_FACTS_PER_BATCH)
        .filter_map(|entry| validate(&entry, batch_len, predicates))
        .collect();
    Ok(ParsedBatch {
        dropped: read - facts.len(),
        facts,
        truncated,
    })
}

fn validate(entry: &RawValue, batch_len: usize, predicates: &[String]) -> Option<ParsedFact> {
    let raw: RawFact = serde_json::from_str(entry.get()).ok()?;
    let [subject, predicate, object] =
        [raw.subject, raw.predicate, raw.object].map(|field| field.trim().to_string());
    let fields_ok = [&subject, &predicate, &object]
        .iter()
        .all(|field| !field.is_empty() && field.chars().count() <= MAX_FIELD_CHARS);
    let events: BTreeSet<usize> = raw.events.into_iter().collect();
    let cited_in_batch = !events.is_empty() && events.iter().all(|index| *index < batch_len);
    let dates_ok = validity_is_ordered(raw.valid_from.as_deref(), raw.valid_until.as_deref());
    let acceptable = fields_ok && cited_in_batch && dates_ok && predicates.contains(&predicate);
    acceptable.then(|| ParsedFact {
        subject,
        predicate,
        object,
        polarity: raw.polarity,
        valid_from: raw.valid_from,
        valid_until: raw.valid_until,
        confidence: raw.confidence.clamp(MIN_CONFIDENCE, MAX_CONFIDENCE),
        events: events.into_iter().collect(),
    })
}

/// Each date present is a real ISO date, and when both are present the span is
/// not empty or backwards.
fn validity_is_ordered(from: Option<&str>, until: Option<&str>) -> bool {
    let parse = |date: Option<&str>| date.map(parse_iso_date);
    match (parse(from), parse(until)) {
        (Some(None), _) | (_, Some(None)) => false,
        (Some(Some(from)), Some(Some(until))) => from < until,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const BATCH: usize = 3;

    fn vocabulary() -> Vec<String> {
        vec!["works_at".into(), "knows".into()]
    }

    fn fact(overrides: Value) -> Value {
        let mut base = json!({
            "subject": "alice", "predicate": "works_at", "object": "Acme",
            "polarity": "asserted", "valid_from": null, "valid_until": null,
            "confidence": 0.9, "events": [0],
        });
        base.as_object_mut()
            .unwrap()
            .extend(overrides.as_object().unwrap().clone());
        base
    }

    fn parse(facts: Vec<Value>) -> ParsedBatch {
        let text = json!({ "facts": facts }).to_string();
        parse_response(&text, BATCH, &vocabulary()).expect("a well formed response")
    }

    fn fails(text: &str) -> ResponseError {
        parse_response(text, BATCH, &vocabulary()).expect_err("a malformed response")
    }

    #[test]
    fn a_valid_response_becomes_facts_with_their_events() {
        // Arrange
        let facts = vec![
            fact(json!({})),
            fact(json!({ "predicate": "knows", "events": [2, 1] })),
            fact(json!({ "object": "Beta", "events": [1, 1] })),
        ];

        // Act
        let batch = parse(facts);

        // Assert
        assert_eq!(batch.dropped, 0);
        let events: Vec<_> = batch.facts.iter().map(|f| f.events.clone()).collect();
        assert_eq!(events, vec![vec![0], vec![1, 2], vec![1]]);
        assert_eq!(batch.facts[0].polarity, Polarity::Asserted);
    }

    #[test]
    fn a_malformed_whole_is_a_typed_error() {
        for (text, expected) in [
            (r#"{"facts": [{"subject""#, "NotJson"),
            ("not json", "NotJson"),
            (r#"{"facts": NaN}"#, "NotJson"),
            ("[]", "NotAnObject"),
            ("{}", "MissingFacts"),
            (r#"{"facts": {}}"#, "FactsNotAnArray"),
            (r#"{"facts": [], "extra": 1}"#, "UnknownKey"),
        ] {
            // Act
            let error = fails(text);

            // Assert
            assert!(
                format!("{error:?}").starts_with(expected),
                "{text}: {error:?}"
            );
        }
    }

    #[test]
    fn no_facts_is_an_empty_batch_not_an_error() {
        // Act
        let batch = parse(vec![]);

        // Assert
        assert_eq!(
            (batch.facts.len(), batch.dropped, batch.truncated),
            (0, 0, 0)
        );
    }

    #[test]
    fn a_bad_citation_drops_that_fact_only() {
        for events in [
            json!([-1]),
            json!([BATCH]),
            json!([]),
            json!([1.5]),
            json!(["0"]),
        ] {
            // Act
            let batch = parse(vec![fact(json!({ "events": events })), fact(json!({}))]);

            // Assert
            assert_eq!((batch.facts.len(), batch.dropped), (1, 1), "{events}");
        }
    }

    #[test]
    fn an_empty_batch_leaves_no_valid_citation() {
        // Arrange
        let text = json!({ "facts": [fact(json!({}))] }).to_string();

        // Act
        let batch = parse_response(&text, 0, &vocabulary()).unwrap();

        // Assert
        assert_eq!((batch.facts.len(), batch.dropped), (0, 1));
    }

    #[test]
    fn confidence_is_clamped_and_a_non_number_drops_the_fact() {
        // Act
        let batch = parse(vec![
            fact(json!({ "confidence": 7 })),
            fact(json!({ "confidence": -1 })),
            fact(json!({ "confidence": "NaN" })),
        ]);

        // Assert
        let confidences: Vec<_> = batch.facts.iter().map(|f| f.confidence).collect();
        assert_eq!(confidences, vec![1.0, 0.0]);
        assert_eq!(batch.dropped, 1);
    }

    #[test]
    fn field_length_is_bounded_in_characters_and_blank_is_rejected() {
        for field in ["subject", "predicate", "object"] {
            for (value, accepted) in [
                (String::new(), false),
                ("  ".to_string(), false),
                ("x".repeat(MAX_FIELD_CHARS), true),
                ("x".repeat(MAX_FIELD_CHARS + 1), false),
                ("é".repeat(MAX_FIELD_CHARS), true),
                ("é".repeat(MAX_FIELD_CHARS + 1), false),
                (format!("  {}  ", "x".repeat(MAX_FIELD_CHARS)), true),
            ] {
                // Arrange
                let text = json!({ "facts": [fact(json!({ field: value }))] }).to_string();
                let predicates = vec![value.trim().to_string(), "works_at".to_string()];

                // Act
                let batch = parse_response(&text, BATCH, &predicates).unwrap();

                // Assert
                assert_eq!(
                    batch.facts.len() == 1,
                    accepted,
                    "{field}: {} bytes",
                    value.len()
                );
            }
        }
    }

    #[test]
    fn a_repeated_key_rejects_the_response_or_the_fact() {
        // Arrange
        let repeated_top = r#"{"facts": [], "facts": []}"#;
        let repeated_field = r#"{"facts": [{"subject": "a", "subject": "b", "predicate": "works_at",
            "object": "Acme", "polarity": "asserted", "valid_from": null, "valid_until": null,
            "confidence": 1, "events": [0]}]}"#;

        // Act
        let top = fails(repeated_top);
        let field = parse_response(repeated_field, BATCH, &vocabulary()).unwrap();

        // Assert
        assert!(matches!(top, ResponseError::DuplicateKey(_)));
        assert_eq!((field.facts.len(), field.dropped), (0, 1));
    }

    #[test]
    fn a_span_that_is_empty_or_backwards_is_dropped() {
        for (from, until, accepted) in [
            ("2026-01-01", "2026-06-01", true),
            ("2026-06-01", "2026-06-01", false),
            ("2026-06-02", "2026-06-01", false),
            ("2025-12-31", "2026-01-01", true),
        ] {
            // Act
            let batch = parse(vec![fact(
                json!({ "valid_from": from, "valid_until": until }),
            )]);

            // Assert
            assert_eq!(batch.facts.len() == 1, accepted, "{from} to {until}");
            assert_eq!(batch.dropped, usize::from(!accepted));
        }
    }

    #[test]
    fn the_fact_count_is_capped_and_the_excess_counted() {
        for (count, truncated) in [(MAX_FACTS_PER_BATCH, 0), (MAX_FACTS_PER_BATCH + 1, 1)] {
            // Act
            let batch = parse((0..count).map(|_| fact(json!({}))).collect());

            // Assert
            assert_eq!(
                (batch.facts.len(), batch.truncated),
                (MAX_FACTS_PER_BATCH, truncated)
            );
        }
    }

    #[test]
    fn validity_dates_must_be_real_iso_dates_or_null() {
        for (date, accepted) in [
            (json!("2026-13-45"), false),
            (json!("last spring"), false),
            (json!("2026-03-04"), true),
            (json!(null), true),
        ] {
            // Act
            let from = parse(vec![fact(json!({ "valid_from": date }))]);
            let until = parse(vec![fact(json!({ "valid_until": date }))]);

            // Assert
            assert_eq!(from.facts.len() == 1, accepted, "{date}");
            assert_eq!(until.facts.len() == 1, accepted, "{date}");
        }
    }

    #[test]
    fn a_predicate_outside_the_vocabulary_is_dropped() {
        // Act
        let batch = parse(vec![
            fact(json!({ "predicate": "resides_in" })),
            fact(json!({})),
        ]);

        // Assert
        assert_eq!((batch.facts.len(), batch.dropped), (1, 1));
    }

    #[test]
    fn unknown_fields_and_bad_polarity_drop_the_fact() {
        // Act
        let batch = parse(vec![
            fact(json!({ "note": "hi" })),
            fact(json!({ "polarity": "maybe" })),
        ]);

        // Assert
        assert_eq!((batch.facts.len(), batch.dropped), (0, 2));
    }
}
