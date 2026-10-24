// SPDX-License-Identifier: Apache-2.0
//! The R10 tier that runs in normal CI against the rules extractor: its
//! output is scored per predicate against floors derived by hand from the
//! corpus.

use std::collections::BTreeMap;

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::corpus::{
    generate, LabelledEvent, Template, CORPUS_SEED, EVENT_COUNT, REFERENCES_DATE, REFERENCES_ID,
    REFERENCES_URL,
};
use super::{normalize_tuple, score, score_by_predicate, Score, Tuple};
use crate::config::ExtractionConfig;
use crate::extraction::response::Polarity;
use crate::extraction::rules::extract_rules;

/// A floor on one predicate's score, derived by hand from the corpus: it has
/// 30 events of each template, so the counts below are exact. A floor changes
/// only in a commit that updates its derivation.
struct Floor {
    predicate: &'static str,
    precision: f64,
    recall: f64,
}

const FLOORS: [Floor; 3] = [
    // 30 date events, each holding one valid date and the invalid near-miss
    // date being rejected: 30 found, 0 wrong, 0 missed.
    Floor {
        predicate: REFERENCES_DATE,
        precision: 1.0,
        recall: 1.0,
    },
    // 30 URL events found whole, trailing period trimmed; "www.example.com"
    // has no scheme and is not matched.
    Floor {
        predicate: REFERENCES_URL,
        precision: 1.0,
        recall: 1.0,
    },
    // 30 ticket events found. The 30 "SHA-256" events are also matched, since
    // the id rule cannot tell them from a ticket: 30 / (30 + 30) = 0.5. The
    // lowercase "abc-123" is not matched.
    Floor {
        predicate: REFERENCES_ID,
        precision: 0.5,
        recall: 1.0,
    },
];

fn corpus() -> Vec<LabelledEvent> {
    generate(&mut ChaCha8Rng::seed_from_u64(CORPUS_SEED))
}

fn rules_labels(events: &[LabelledEvent]) -> Vec<Tuple> {
    events
        .iter()
        .flat_map(|event| {
            event.references.iter().map(|reference| {
                normalize_tuple(
                    &event.rules_subject(),
                    reference.predicate,
                    &reference.object,
                    Polarity::Asserted,
                )
            })
        })
        .collect()
}

fn rules_found(events: &[LabelledEvent]) -> Vec<Tuple> {
    let max_chars = ExtractionConfig::default().max_event_chars();
    events
        .iter()
        .flat_map(|event| extract_rules(&event.id, &event.text, &[], max_chars))
        .map(|found| {
            normalize_tuple(
                &found.subject,
                &found.predicate,
                &found.object,
                Polarity::Asserted,
            )
        })
        .collect()
}

/// The predicates whose score is under its floor, or that were never seen.
fn floors_missed(by_predicate: &BTreeMap<String, Score>) -> Vec<&'static str> {
    FLOORS
        .iter()
        .filter(|floor| {
            by_predicate.get(floor.predicate).is_none_or(|scored| {
                scored.precision < floor.precision || scored.recall < floor.recall
            })
        })
        .map(|floor| floor.predicate)
        .collect()
}

#[test]
fn the_rules_tier_meets_every_floor() {
    // Arrange
    let events = corpus();

    // Act
    let by_predicate = score_by_predicate(&rules_found(&events), &rules_labels(&events));

    // Assert
    assert_eq!(floors_missed(&by_predicate), Vec::<&str>::new());
    let ids = by_predicate[REFERENCES_ID];
    assert_eq!((ids.tp, ids.fp, ids.fn_), (30, 30, 0));
}

#[test]
fn the_rules_tier_is_exact_on_every_template_it_has_no_distractor_for() {
    // Arrange
    let events = corpus();
    let clean: Vec<LabelledEvent> = events
        .into_iter()
        .filter(|event| event.template != Template::StandardName)
        .collect();

    // Act
    let result = score(&rules_found(&clean), &rules_labels(&clean));

    // Assert
    assert_eq!(result.f1, 1.0);
}

#[test]
fn a_degraded_predictor_fails_the_gate() {
    // Arrange: the same output with every URL lost and every id doubled up
    // with a wrong one.
    let events = corpus();
    let mut degraded: Vec<Tuple> = rules_found(&events)
        .into_iter()
        .filter(|tuple| tuple.predicate != REFERENCES_URL)
        .collect();
    degraded.extend((0..EVENT_COUNT).map(|n| {
        normalize_tuple(
            &format!("event:evt-{n:03}"),
            REFERENCES_ID,
            "ZZ-0",
            Polarity::Asserted,
        )
    }));

    // Act
    let missed = floors_missed(&score_by_predicate(&degraded, &rules_labels(&events)));

    // Assert
    assert_eq!(missed, [REFERENCES_URL, REFERENCES_ID]);
}
