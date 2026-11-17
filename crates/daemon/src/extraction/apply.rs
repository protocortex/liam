// SPDX-License-Identifier: Apache-2.0
//! Turns one parsed model answer into store writes: drops what the cited text
//! does not support, gives each fact its trust and dates, and applies the
//! facts in an order that does not depend on how the model listed them.

use liam_store::{
    ArbitrationCaps, ArbitrationOutcome, DefaultGraph, EndFact, EndOutcome, Error, ExtractorName,
    Millis, NewFact, PredicateSpec, Vocabulary, BEGINNING, FOREVER, MAX_FACTS_PER_BATCH,
};
use tracing::warn;

use super::grounding::supporting_events;
use super::inputs::ExtractionInput;
use super::iso_date::parse_iso_date;
use super::prompt::PROMPT_VERSION;
use super::response::{ParsedBatch, ParsedFact, Polarity};
use super::rules::{extract_rules, RuleFact, RULES_EXTRACTOR, RULES_EXTRACTOR_VERSION};
use crate::config::ExtractionConfig;

const MILLIS_PER_SECOND: i64 = 1000;
/// The rules match text exactly, so they are as sure as a fact can be.
const RULE_CONFIDENCE: f64 = 1.0;

/// What the store needs to know to write a batch's facts.
pub struct ApplyContext<'a> {
    pub store: &'a DefaultGraph,
    pub config: &'a ExtractionConfig,
    pub vocabulary: &'a Vocabulary,
    pub write_source: &'a str,
}

/// What applying one answer did, to be added to the run's report.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ApplyCounts {
    pub written: usize,
    pub dropped: usize,
    pub dropped_ungrounded: usize,
    pub ended_no_match: usize,
    pub ended_outranked: usize,
}

#[derive(Debug)]
pub enum ApplyError {
    /// The store refused the facts on their content, which a retry repeats.
    Rejected(String),
    /// The store failed for a reason that says nothing about the facts.
    Store(Error),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(why) => write!(f, "the store refused the facts: {why}"),
            Self::Store(error) => write!(f, "the store failed: {error}"),
        }
    }
}

impl std::error::Error for ApplyError {}

impl From<Error> for ApplyError {
    fn from(error: Error) -> Self {
        match error {
            Error::FactInBatch { source, .. } => Self::from(*source),
            Error::InvalidFactInterval { .. }
            | Error::InvalidFactTrust(_)
            | Error::FactWithoutSource
            | Error::InvalidScope(_) => Self::Rejected(error.to_string()),
            other => Self::Store(other),
        }
    }
}

/// A fact the cited text supports, with the batch events that state it.
struct Grounded {
    fact: ParsedFact,
    supporting: Vec<usize>,
}

impl Grounded {
    /// Position in the batch of the earliest event that states the fact.
    fn first_event(&self) -> usize {
        self.supporting[0]
    }

    fn sort_key(&self) -> (usize, bool, &str, &str, &str) {
        (
            self.first_event(),
            self.fact.polarity == Polarity::Ended,
            &self.fact.subject,
            &self.fact.predicate,
            &self.fact.object,
        )
    }
}

/// Writes the facts of `parsed`, whose event numbers index `batch`. A run of
/// asserted facts is one atomic store write; an ended fact is its own write.
pub async fn apply_answer(
    ctx: &ApplyContext<'_>,
    batch: &[&ExtractionInput],
    parsed: ParsedBatch,
) -> Result<ApplyCounts, ApplyError> {
    let mut counts = ApplyCounts {
        dropped: parsed.dropped + parsed.truncated,
        ..ApplyCounts::default()
    };
    let mut grounded = ground(ctx, batch, parsed.facts, &mut counts);
    grounded.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));

    let mut asserted: Vec<(NewFact, PredicateSpec)> = Vec::new();
    for item in &grounded {
        match prepare(ctx, batch, item) {
            None => counts.dropped += 1,
            Some(Prepared::Assert(fact, spec)) => asserted.push((fact, spec)),
            Some(Prepared::End(end, spec)) => {
                flush_asserted(ctx, &mut asserted, &mut counts).await?;
                let outcome = ctx.store.end_fact(end, &spec, ctx.write_source).await?;
                match outcome {
                    EndOutcome::Ended { .. } => counts.written += 1,
                    EndOutcome::NoMatch => counts.ended_no_match += 1,
                    EndOutcome::Outranked => counts.ended_outranked += 1,
                    EndOutcome::Rejected => counts.dropped += 1,
                }
            }
        }
    }
    flush_asserted(ctx, &mut asserted, &mut counts).await?;
    Ok(counts)
}

/// Writes the facts the rules find in `inputs`. One input's facts are one
/// write, so a fact the store refuses costs only that input's facts.
pub async fn apply_rules(
    ctx: &ApplyContext<'_>,
    inputs: &[ExtractionInput],
    known_entities: &[String],
) -> Result<ApplyCounts, ApplyError> {
    let mut counts = ApplyCounts::default();
    for input in inputs {
        let mut found: Vec<(NewFact, PredicateSpec)> = extract_rules(
            &input.event_id,
            &input.text,
            known_entities,
            ctx.config.max_event_chars(),
        )
        .into_iter()
        .map(|fact| rule_fact(ctx, input, fact))
        .collect();
        let attempted = found.len();
        match flush_asserted(ctx, &mut found, &mut counts).await {
            Ok(()) => {}
            Err(ApplyError::Rejected(why)) => {
                warn!(event_id = %input.event_id, %why, "the store refused an event's rule facts");
                counts.dropped += attempted;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(counts)
}

fn rule_fact(
    ctx: &ApplyContext<'_>,
    input: &ExtractionInput,
    fact: RuleFact,
) -> (NewFact, PredicateSpec) {
    let spec = ctx.vocabulary.spec(&fact.predicate);
    let new = NewFact {
        subject: fact.subject,
        predicate: fact.predicate,
        object: fact.object,
        scope: input.scope.clone(),
        valid_from: input.observed_at,
        valid_until: FOREVER,
        evidence_at: evidence_time(input),
        confidence: RULE_CONFIDENCE,
        trust_score: source_trust(ctx.config, input),
        extractor: RULES_EXTRACTOR,
        extractor_version: RULES_EXTRACTOR_VERSION.to_string(),
        source_event_ids: vec![fact.source_event_id],
    };
    (new, spec)
}

async fn flush_asserted(
    ctx: &ApplyContext<'_>,
    asserted: &mut Vec<(NewFact, PredicateSpec)>,
    counts: &mut ApplyCounts,
) -> Result<(), ApplyError> {
    let caps = ArbitrationCaps {
        max_competing_per_key: ctx.config.max_competing_per_key,
    };
    for chunk in std::mem::take(asserted)
        .chunks(MAX_FACTS_PER_BATCH)
        .map(<[_]>::to_vec)
    {
        for outcome in ctx
            .store
            .assert_facts(chunk, ctx.write_source, caps)
            .await?
        {
            match outcome {
                ArbitrationOutcome::Suppressed => counts.dropped += 1,
                ArbitrationOutcome::Corroborated { changed: false, .. } => {}
                _ => counts.written += 1,
            }
        }
    }
    Ok(())
}

fn ground(
    ctx: &ApplyContext<'_>,
    batch: &[&ExtractionInput],
    facts: Vec<ParsedFact>,
    counts: &mut ApplyCounts,
) -> Vec<Grounded> {
    let texts: Vec<&str> = batch.iter().map(|input| input.text.as_str()).collect();
    facts
        .into_iter()
        .filter_map(|fact| {
            if fact.confidence < ctx.config.min_confidence {
                counts.dropped += 1;
                return None;
            }
            let supporting = supporting_events(&fact, &texts, ctx.vocabulary, &ctx.config.end_cues);
            if supporting.is_empty() {
                counts.dropped += 1;
                counts.dropped_ungrounded += 1;
                return None;
            }
            Some(Grounded { fact, supporting })
        })
        .collect()
}

enum Prepared {
    Assert(NewFact, PredicateSpec),
    End(EndFact, PredicateSpec),
}

/// The store call a grounded fact becomes, or `None` when its dates cannot
/// make a valid fact and it is dropped.
fn prepare(
    ctx: &ApplyContext<'_>,
    batch: &[&ExtractionInput],
    item: &Grounded,
) -> Option<Prepared> {
    let sources: Vec<&ExtractionInput> = item.supporting.iter().map(|&i| batch[i]).collect();
    let fact = &item.fact;
    let spec = ctx.vocabulary.spec(&fact.predicate);
    let dates = Dates::of(fact, &sources)?;
    let trust = trust_of(ctx.config, &sources);
    let evidence_at = sources
        .iter()
        .map(|input| evidence_time(input))
        .max()
        .expect("a grounded fact has a supporting event");
    let source_event_ids: Vec<String> =
        sources.iter().map(|input| input.event_id.clone()).collect();
    let scope = sources[0].scope.clone();

    if fact.polarity == Polarity::Ended {
        let end = EndFact {
            subject: fact.subject.clone(),
            predicate: fact.predicate.clone(),
            object: fact.object.clone(),
            scope,
            valid_until: dates.until.unwrap_or(dates.earliest_observed),
            evidence_at,
            trust_score: trust,
            source_event_ids,
        };
        return Some(Prepared::End(end, spec));
    }
    let new = NewFact {
        subject: fact.subject.clone(),
        predicate: fact.predicate.clone(),
        object: fact.object.clone(),
        scope,
        valid_from: dates.from,
        valid_until: dates.until.unwrap_or(FOREVER),
        evidence_at,
        confidence: fact.confidence,
        trust_score: trust,
        extractor: ExtractorName::Llm,
        extractor_version: PROMPT_VERSION.to_string(),
        source_event_ids,
    };
    Some(Prepared::Assert(new, spec))
}

/// The strongest source trust, scaled down because a model can misread.
fn trust_of(config: &ExtractionConfig, sources: &[&ExtractionInput]) -> f64 {
    let strongest = sources
        .iter()
        .map(|input| source_trust(config, input))
        .fold(0.0, f64::max);
    strongest * config.llm_trust_factor
}

/// What the configuration trusts the event's writer to say.
fn source_trust(config: &ExtractionConfig, input: &ExtractionInput) -> f64 {
    config
        .trust
        .get(&input.source)
        .copied()
        .unwrap_or(config.default_trust)
}

/// When the event's claim was last known to hold: the earlier of when it was
/// observed and when it was ingested.
fn evidence_time(input: &ExtractionInput) -> Millis {
    input.observed_at.min(input.ingested_at)
}

struct Dates {
    from: Millis,
    until: Option<Millis>,
    earliest_observed: Millis,
}

impl Dates {
    /// `None` when a date the model gave cannot be placed on the calendar.
    fn of(fact: &ParsedFact, sources: &[&ExtractionInput]) -> Option<Self> {
        let earliest_observed = sources.iter().map(|input| input.observed_at).min()?;
        let until = match fact.valid_until.as_deref() {
            Some(text) => Some(date_millis(text)?),
            None => None,
        };
        let from = match fact.valid_from.as_deref() {
            Some(text) => date_millis(text)?,
            // A statement that only gives an end does not start at the event.
            None if until.is_some() => BEGINNING,
            None => earliest_observed,
        };
        // An empty or backwards span is a fact the model got wrong, not a
        // batch the store should refuse.
        if fact.polarity == Polarity::Asserted && from >= until.unwrap_or(FOREVER) {
            return None;
        }
        Some(Self {
            from,
            until,
            earliest_observed,
        })
    }
}

/// Midnight UTC of an ISO date.
fn date_millis(text: &str) -> Option<Millis> {
    let date = parse_iso_date(text)?;
    let month = time::Month::try_from(u8::try_from(date.month).ok()?).ok()?;
    let day = time::Date::from_calendar_date(
        i32::try_from(date.year).ok()?,
        month,
        u8::try_from(date.day).ok()?,
    )
    .ok()?;
    Some(Millis(
        day.midnight().assume_utc().unix_timestamp() * MILLIS_PER_SECOND,
    ))
}

#[cfg(test)]
mod tests;
