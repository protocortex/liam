// SPDX-License-Identifier: Apache-2.0
//! The labelled events R10 is scored on. LIAM has no real extraction corpus,
//! so the events are synthetic: each comes from a sentence template filled
//! from fixed pools by a seeded generator, which makes its expected facts
//! known by construction.

use liam_store::Millis;
use rand_chacha::rand_core::RngCore;
use rand_chacha::ChaCha8Rng;

use crate::extraction::response::Polarity;

pub(super) const EVENT_COUNT: usize = 300;
pub(super) const CORPUS_SEED: u64 = 0x10C0_DE10;
/// Events per template group; the templates repeat in this order.
pub(super) const CYCLE_LEN: usize = 10;
pub(super) const CYCLES: usize = EVENT_COUNT / CYCLE_LEN;
/// Gap between consecutive events' observation times.
pub(super) const EVENT_GAP: Millis = Millis(3_600_000);
/// 2023-06-01 00:00:00 UTC.
pub(super) const FIRST_OBSERVED: Millis = Millis(1_685_577_600_000);

pub(super) const LIVES_IN: &str = "lives_in";
pub(super) const WORKS_AT: &str = "works_at";
pub(super) const HAS_TITLE: &str = "has_title";
pub(super) const REFERENCES_DATE: &str = "references_date";
pub(super) const REFERENCES_URL: &str = "references_url";
pub(super) const REFERENCES_ID: &str = "references_id";
const EVENT_SUBJECT_PREFIX: &str = "event:";

/// People who are the subject of a fact: one mover, one worker and one title
/// holder per cycle, all different, so no fact supersedes another cycle's.
const SUBJECTS_PER_CYCLE: usize = 3;
const FIRST_NAMES: [&str; 12] = [
    "Ana", "Bruno", "Carla", "Diego", "Elena", "Felipe", "Gabriela", "Hugo", "Iris", "Joao",
    "Karen", "Lucas",
];
const LAST_NAMES: [&str; 10] = [
    "Lima", "Souza", "Costa", "Ramos", "Nunes", "Alves", "Pinto", "Moura", "Reis", "Tavares",
];
const CITIES: [&str; 10] = [
    "Rome", "Oslo", "Lisbon", "Recife", "Berlin", "Curitiba", "Porto", "Vienna", "Quito", "Seoul",
];
const ORGS: [&str; 10] = [
    "Acme Corp",
    "Globex",
    "Initech",
    "Umbrella Labs",
    "Hooli",
    "Stark Works",
    "Wayne Systems",
    "Tyrell Group",
    "Cyberdyne",
    "Soylent Foods",
];
/// Two words each, so a wrong answer can keep only the first.
const TITLES: [&str; 6] = [
    "Staff Engineer",
    "Product Manager",
    "Design Lead",
    "Data Analyst",
    "Support Specialist",
    "Security Architect",
];
const SLUGS: [&str; 5] = ["guide", "report", "plan", "notes", "design"];
const TICKET_PREFIXES: [&str; 5] = ["OPS", "WEB", "DATA", "SEC", "INFRA"];

const FIRST_YEAR: u64 = 2020;
const YEAR_SPAN: u64 = 7;
const MONTHS: u64 = 12;
/// Every month has at least this many days, so any drawn day is real.
const SAFE_DAYS_IN_MONTH: u64 = 28;
const MAX_TICKET_NUMBER: u64 = 999;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Template {
    Relocation,
    PastResidence,
    Employment,
    Negation,
    RoleChange,
    DateReference,
    UrlReference,
    IdReference,
    /// An invalid date, a version string, a lowercase ticket and a bare domain.
    NearMiss,
    /// A hyphenated standard name the id rule cannot tell from a ticket.
    StandardName,
}

/// A fact the model should extract from an event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpectedFact {
    pub subject: String,
    pub predicate: &'static str,
    pub object: String,
    pub polarity: Polarity,
}

/// A date, URL or id the rules extractor should find in an event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Reference {
    pub predicate: &'static str,
    pub object: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LabelledEvent {
    pub index: usize,
    pub id: String,
    pub template: Template,
    pub text: String,
    pub observed_at: Millis,
    /// What a model should extract; empty for a distractor.
    pub facts: Vec<ExpectedFact>,
    /// What the rules extractor should find.
    pub references: Vec<Reference>,
}

impl LabelledEvent {
    /// The rules extractor's subject for the event's own facts.
    pub(super) fn rules_subject(&self) -> String {
        format!("{EVENT_SUBJECT_PREFIX}{}", self.id)
    }
}

/// The fixed-size corpus for `rng`'s stream, in observation order.
// Crate-visible so the trust-and-metrics work can reuse the labelled corpus.
pub(crate) fn generate(rng: &mut ChaCha8Rng) -> Vec<LabelledEvent> {
    let mut subjects = distinct_people(rng, CYCLES * SUBJECTS_PER_CYCLE).into_iter();
    let mut events = Vec::with_capacity(EVENT_COUNT);
    for _ in 0..CYCLES {
        let mover = subjects.next().expect("a mover per cycle");
        let worker = subjects.next().expect("a worker per cycle");
        let titled = subjects.next().expect("a title holder per cycle");
        events.extend(cycle(rng, &mover, &worker, &titled));
    }
    for (index, event) in events.iter_mut().enumerate() {
        event.index = index;
        event.id = format!("evt-{index:03}");
        event.observed_at = Millis(FIRST_OBSERVED.0 + EVENT_GAP.0 * index as i64);
    }
    events
}

/// One cycle in template order. Relocation and past residence, and employment
/// and negation, share subject and object, so the second of each pair ends
/// what the first asserted.
fn cycle(rng: &mut ChaCha8Rng, mover: &str, worker: &str, titled: &str) -> Vec<LabelledEvent> {
    let city = pick(rng, &CITIES);
    let org = pick(rng, &ORGS);
    let title = pick(rng, &TITLES);
    let speaker = person(rng);
    let date = date(rng);
    let slug = pick(rng, &SLUGS);
    let number = 1 + below(rng, MAX_TICKET_NUMBER);
    let url = format!("https://docs.example.com/{slug}/{number}");
    let ticket = format!("{}-{number}", pick(rng, &TICKET_PREFIXES));

    let fact = |subject: &str, predicate, object: &str, polarity| ExpectedFact {
        subject: subject.to_string(),
        predicate,
        object: object.to_string(),
        polarity,
    };
    let reference = |predicate, object: &str| Reference {
        predicate,
        object: object.to_string(),
    };
    let event = |template, text: String, facts, references| LabelledEvent {
        index: 0,
        id: String::new(),
        template,
        text,
        observed_at: Millis(0),
        facts,
        references,
    };

    let asserted = Polarity::Asserted;
    let ended = Polarity::Ended;
    vec![
        event(
            Template::Relocation,
            format!("{mover} moved to {city}."),
            vec![fact(mover, LIVES_IN, city, asserted)],
            vec![],
        ),
        event(
            Template::PastResidence,
            format!("{mover} used to live in {city}."),
            vec![fact(mover, LIVES_IN, city, ended)],
            vec![],
        ),
        event(
            Template::Employment,
            format!("{worker} works at {org}."),
            vec![fact(worker, WORKS_AT, org, asserted)],
            vec![],
        ),
        event(
            Template::Negation,
            format!("{worker} no longer works at {org}."),
            vec![fact(worker, WORKS_AT, org, ended)],
            vec![],
        ),
        event(
            Template::RoleChange,
            format!("{titled} was promoted to {title}."),
            vec![fact(titled, HAS_TITLE, title, asserted)],
            vec![],
        ),
        event(
            Template::DateReference,
            format!("{speaker} reviewed the plan on {date}."),
            vec![],
            vec![reference(REFERENCES_DATE, &date)],
        ),
        event(
            Template::UrlReference,
            format!("{speaker} wrote: see {url}."),
            vec![],
            vec![reference(REFERENCES_URL, &url)],
        ),
        event(
            Template::IdReference,
            format!("{speaker} closed ticket {ticket} yesterday."),
            vec![],
            vec![reference(REFERENCES_ID, &ticket)],
        ),
        event(
            Template::NearMiss,
            format!("{speaker} saw version 1.2.3, build 2026-13-45, abc-123 and www.example.com."),
            vec![],
            vec![],
        ),
        event(
            Template::StandardName,
            format!("{speaker} uses SHA-256 to check each file."),
            vec![],
            vec![],
        ),
    ]
}

/// A draw in `0..bound`. The modulo bias is negligible for a pool this small.
fn below(rng: &mut ChaCha8Rng, bound: u64) -> u64 {
    rng.next_u64() % bound
}

fn pick<'a>(rng: &mut ChaCha8Rng, pool: &[&'a str]) -> &'a str {
    pool[below(rng, pool.len() as u64) as usize]
}

fn person(rng: &mut ChaCha8Rng) -> String {
    format!("{} {}", pick(rng, &FIRST_NAMES), pick(rng, &LAST_NAMES))
}

fn date(rng: &mut ChaCha8Rng) -> String {
    let year = FIRST_YEAR + below(rng, YEAR_SPAN);
    let month = 1 + below(rng, MONTHS);
    let day = 1 + below(rng, SAFE_DAYS_IN_MONTH);
    format!("{year:04}-{month:02}-{day:02}")
}

/// `count` different full names, in a shuffled order (Fisher-Yates).
fn distinct_people(rng: &mut ChaCha8Rng, count: usize) -> Vec<String> {
    let mut names: Vec<String> = FIRST_NAMES
        .iter()
        .flat_map(|first| LAST_NAMES.iter().map(move |last| format!("{first} {last}")))
        .collect();
    assert!(count <= names.len(), "the name pools are too small");
    for end in (1..names.len()).rev() {
        let swap_with = below(rng, end as u64 + 1) as usize;
        names.swap(end, swap_with);
    }
    names.truncate(count);
    names
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rand_chacha::rand_core::SeedableRng;

    use super::*;

    fn corpus() -> Vec<LabelledEvent> {
        generate(&mut ChaCha8Rng::seed_from_u64(CORPUS_SEED))
    }

    #[test]
    fn the_same_seed_gives_the_same_corpus() {
        // Act
        let first = corpus();
        let second = corpus();

        // Assert
        assert_eq!(first, second);
    }

    #[test]
    fn another_seed_gives_another_corpus() {
        // Act
        let other = generate(&mut ChaCha8Rng::seed_from_u64(CORPUS_SEED + 1));

        // Assert
        assert_ne!(corpus(), other);
    }

    #[test]
    fn the_corpus_has_300_events_in_observation_order() {
        // Act
        let events = corpus();

        // Assert
        assert_eq!(events.len(), EVENT_COUNT);
        assert!(events
            .windows(2)
            .all(|pair| pair[0].observed_at < pair[1].observed_at));
        let ids: BTreeSet<&str> = events.iter().map(|event| event.id.as_str()).collect();
        assert_eq!(ids.len(), EVENT_COUNT);
    }

    #[test]
    fn every_event_has_labels_or_is_a_distractor_with_none() {
        // Act
        let events = corpus();

        // Assert
        for event in &events {
            let distractor = matches!(event.template, Template::NearMiss | Template::StandardName);
            let labelled = !event.facts.is_empty() || !event.references.is_empty();
            assert_eq!(labelled, !distractor, "{event:?}");
        }
    }

    #[test]
    fn negations_and_past_statements_expect_an_ended_fact() {
        // Act
        let events = corpus();

        // Assert
        for event in &events {
            let polarities: Vec<Polarity> = event.facts.iter().map(|f| f.polarity).collect();
            let expected = match event.template {
                Template::Negation | Template::PastResidence => vec![Polarity::Ended],
                Template::Relocation | Template::Employment | Template::RoleChange => {
                    vec![Polarity::Asserted]
                }
                _ => vec![],
            };
            assert_eq!(polarities, expected, "{event:?}");
        }
    }

    #[test]
    fn each_ended_fact_follows_the_assertion_it_ends() {
        // Act
        let events = corpus();

        // Assert
        for pair in events.chunks(CYCLE_LEN) {
            assert_eq!(pair[1].facts[0].object, pair[0].facts[0].object);
            assert_eq!(pair[1].facts[0].subject, pair[0].facts[0].subject);
            assert_eq!(pair[3].facts[0].object, pair[2].facts[0].object);
            assert_eq!(pair[3].facts[0].subject, pair[2].facts[0].subject);
        }
    }

    #[test]
    fn no_person_is_the_subject_of_two_cycles() {
        // Act
        let events = corpus();

        // Assert
        let subjects: BTreeSet<&str> = events
            .iter()
            .flat_map(|event| event.facts.iter().map(|fact| fact.subject.as_str()))
            .collect();
        assert_eq!(subjects.len(), CYCLES * SUBJECTS_PER_CYCLE);
    }
}
