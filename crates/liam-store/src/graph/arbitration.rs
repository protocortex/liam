// SPDX-License-Identifier: Apache-2.0
//! The trust arbitration rules for facts as pure functions: what the live
//! versions of a key and an incoming claim decide, with no SQL. The effects
//! are a `Plan` of rows to close, insert and link, so the write path can log a
//! decision before it applies it and a test can read every rule off a table.
// The SQL effects that call these arrive in the change after this one.
#![allow(dead_code)]

use std::collections::BTreeSet;

use crate::error::{Error, Result};
use crate::ids::{FactId, Millis, FOREVER};
use crate::types::{
    ArbitrationCaps, ArbitrationOutcome, Cardinality, EndFact, EndOutcome, Fact, MatchMode,
    NewFact, PredicateSpec, RetractOutcome,
};

/// Mints the id of each version a decision inserts.
pub(super) type Mint<'a> = &'a mut dyn FnMut() -> FactId;

/// A retraction that blocks an object from coming back on old evidence.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Suppression {
    pub subject: String,
    pub predicate: String,
    pub scope: String,
    pub object_key: String,
    pub trust_score: f64,
    pub retracted_at: Millis,
}

/// Everything stored about one `(subject, predicate, scope)` key that a
/// decision reads. `versions` are the current ones (`tx_to = FOREVER`) ordered
/// by `(valid_from, id)`, each with its lineage's source events in
/// `source_event_ids`.
#[derive(Debug, Default)]
pub(super) struct KeyState {
    pub versions: Vec<Fact>,
    pub suppressions: Vec<Suppression>,
}

/// The settings a decision runs under.
pub(super) struct Rules {
    pub spec: PredicateSpec,
    pub caps: ArbitrationCaps,
    /// The transaction time every row this decision writes carries.
    pub at: Millis,
}

/// What a decision does to the store: versions to close in transaction time,
/// versions to insert, source events to link to a lineage, and suppressions to
/// record.
#[derive(Debug, Default, PartialEq)]
pub(super) struct Plan {
    pub close: Vec<FactId>,
    pub insert: Vec<Fact>,
    pub link: Vec<(FactId, Vec<String>)>,
    pub suppress: Vec<Suppression>,
}

pub(super) struct Decision<T> {
    pub plan: Plan,
    pub outcome: T,
}

/// An assertion that passed validation, with its object normalized.
#[derive(Clone, Debug)]
pub(super) struct Claim {
    pub fact: NewFact,
    pub scope: String,
    pub object_key: String,
}

/// An end statement that passed validation.
#[derive(Clone, Debug)]
pub(super) struct EndClaim {
    pub end: EndFact,
    pub scope: String,
    pub object_key: String,
}

pub(super) fn object_key(object: &str) -> String {
    object
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn tokens(key: &str) -> BTreeSet<&str> {
    key.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect()
}

fn agrees(a: &str, b: &str, mode: MatchMode) -> bool {
    if a == b {
        return true;
    }
    if mode == MatchMode::Exact {
        return false;
    }
    let (a, b) = (tokens(a), tokens(b));
    !a.is_empty() && !b.is_empty() && (a.is_subset(&b) || b.is_subset(&a))
}

fn check_trust(trust: f64) -> Result<()> {
    if trust.is_finite() && (0.0..=1.0).contains(&trust) {
        Ok(())
    } else {
        Err(Error::InvalidFactTrust(trust))
    }
}

fn check_sources(sources: &[String]) -> Result<()> {
    if sources.is_empty() {
        Err(Error::FactWithoutSource)
    } else {
        Ok(())
    }
}

impl Claim {
    pub(super) fn new(fact: NewFact, scope: String) -> Result<Self> {
        check_sources(&fact.source_event_ids)?;
        check_trust(fact.trust_score)?;
        if fact.valid_from >= fact.valid_until {
            return Err(Error::InvalidFactInterval {
                valid_from: fact.valid_from.0,
                valid_until: fact.valid_until.0,
            });
        }
        let object_key = object_key(&fact.object);
        Ok(Self {
            fact,
            scope,
            object_key,
        })
    }
}

impl EndClaim {
    pub(super) fn new(end: EndFact, scope: String) -> Result<Self> {
        check_sources(&end.source_event_ids)?;
        check_trust(end.trust_score)?;
        let object_key = object_key(&end.object);
        Ok(Self {
            end,
            scope,
            object_key,
        })
    }
}

impl Rules {
    /// Many-valued predicates only ever merge equal objects.
    fn mode(&self) -> MatchMode {
        match self.spec.cardinality {
            Cardinality::Many => MatchMode::Exact,
            Cardinality::One => self.spec.match_mode,
        }
    }
}

impl KeyState {
    /// The versions that still hold against evidence from `evidence_at`: those
    /// ending after it, whether or not something has closed them since.
    fn live(&self, evidence_at: Millis) -> Vec<&Fact> {
        self.versions
            .iter()
            .filter(|version| version.valid_until > evidence_at)
            .collect()
    }

    fn suppresses(&self, claim: &Claim, mode: MatchMode) -> bool {
        self.suppressions.iter().any(|blocked| {
            agrees(&blocked.object_key, &claim.object_key, mode)
                && claim.fact.evidence_at <= blocked.retracted_at
                && claim.fact.trust_score <= blocked.trust_score
        })
    }
}

/// The live version `object_key` agrees with. A containment match counts only
/// among fully open versions, and only when it is unambiguous.
fn live_match<'a>(live: &[&'a Fact], object_key: &str, mode: MatchMode) -> Option<&'a Fact> {
    if let Some(same) = live.iter().find(|version| version.object_key == object_key) {
        return Some(same);
    }
    let mut open = live.iter().filter(|version| {
        version.valid_until == FOREVER && agrees(&version.object_key, object_key, mode)
    });
    match (open.next(), open.next()) {
        (Some(only), None) => Some(only),
        _ => None,
    }
}

fn decision<T>(plan: Plan, outcome: T) -> Decision<T> {
    Decision { plan, outcome }
}

fn successor(version: &Fact, id: FactId, at: Millis) -> Fact {
    Fact {
        id,
        tx_from: at,
        tx_to: FOREVER,
        ..version.clone()
    }
}

/// A new lineage for `claim`, valid until `valid_until`.
fn new_lineage(claim: &Claim, id: FactId, valid_until: Millis, at: Millis) -> Fact {
    let fact = &claim.fact;
    Fact {
        id: id.clone(),
        lineage_id: id,
        subject: fact.subject.clone(),
        predicate: fact.predicate.clone(),
        object: fact.object.clone(),
        object_key: claim.object_key.clone(),
        scope: (!claim.scope.is_empty()).then(|| claim.scope.clone()),
        evidence_at: fact.evidence_at,
        valid_from: fact.valid_from,
        valid_until,
        tx_from: at,
        tx_to: FOREVER,
        closed_by: None,
        confidence: fact.confidence,
        trust_score: fact.trust_score,
        extractor_id: fact.extractor,
        extractor_version: fact.extractor_version.clone(),
        source_event_ids: fact.source_event_ids.clone(),
    }
}

fn unlinked(sources: &[String], version: &Fact) -> Vec<String> {
    sources
        .iter()
        .filter(|event| !version.source_event_ids.contains(event))
        .cloned()
        .collect()
}

fn covers(versions: &[&Fact], sources: &[String]) -> bool {
    let linked: BTreeSet<&String> = versions
        .iter()
        .flat_map(|version| &version.source_event_ids)
        .collect();
    sources.iter().all(|event| linked.contains(event))
}

impl Plan {
    fn link(&mut self, lineage: &FactId, events: Vec<String>) {
        if !events.is_empty() {
            self.link.push((lineage.clone(), events));
        }
    }

    /// Stops `version` being asserted and blocks its object from returning on
    /// evidence no newer than `at`.
    fn retire(&mut self, version: &Fact, at: Millis) {
        self.close.push(version.id.clone());
        self.suppress.push(Suppression {
            subject: version.subject.clone(),
            predicate: version.predicate.clone(),
            scope: version.scope.clone().unwrap_or_default(),
            object_key: version.object_key.clone(),
            trust_score: version.trust_score,
            retracted_at: at,
        });
    }

    /// Replaces `version` with a copy that ends at `until`, never later than it
    /// already did, and returns the id it closed. The copy carries the newest
    /// evidence behind it, so a later claim on older evidence reads as stale.
    fn end_version(
        &mut self,
        version: &Fact,
        until: Millis,
        evidence_at: Millis,
        closed_by: Option<FactId>,
        rules: &Rules,
        mint: Mint<'_>,
    ) -> FactId {
        self.close.push(version.id.clone());
        let mut ended = successor(version, mint(), rules.at);
        ended.valid_until = version.valid_until.min(until);
        ended.evidence_at = version.evidence_at.max(evidence_at);
        if closed_by.is_some() {
            ended.closed_by = closed_by;
        }
        self.insert.push(ended);
        version.id.clone()
    }
}

pub(super) fn decide_assert(
    state: &KeyState,
    claim: &Claim,
    rules: &Rules,
    mint: Mint<'_>,
) -> Result<Decision<ArbitrationOutcome>> {
    let mode = rules.mode();
    if state.suppresses(claim, mode) {
        return Ok(decision(Plan::default(), ArbitrationOutcome::Suppressed));
    }
    let same_object: Vec<&Fact> = state
        .versions
        .iter()
        .filter(|version| version.object_key == claim.object_key)
        .collect();
    if let Some(latest) = same_object.last() {
        if covers(&same_object, &claim.fact.source_event_ids) {
            return Ok(decision(
                Plan::default(),
                ArbitrationOutcome::Corroborated {
                    lineage_id: latest.lineage_id.clone(),
                    changed: false,
                },
            ));
        }
    }
    let live = state.live(claim.fact.evidence_at);
    if let Some(matched) = live_match(&live, &claim.object_key, mode) {
        return Ok(corroborate(claim, matched, &live, rules, mint));
    }
    if let Some(past) = same_object
        .iter()
        .filter(|version| !live.iter().any(|held| held.id == version.id))
        .max_by_key(|version| (version.evidence_at, version.valid_from))
    {
        // Equal evidence revives too: ties break toward the newer claim, as in `differing`.
        let revives =
            claim.fact.evidence_at >= past.evidence_at && claim.fact.valid_from != past.valid_from;
        if !revives {
            let mut plan = Plan::default();
            plan.link(
                &past.lineage_id,
                unlinked(&claim.fact.source_event_ids, past),
            );
            return Ok(decision(
                plan,
                ArbitrationOutcome::Corroborated {
                    lineage_id: past.lineage_id.clone(),
                    changed: true,
                },
            ));
        }
    }
    match rules.spec.cardinality {
        Cardinality::One => differing(claim, &live, rules, mint),
        Cardinality::Many => Ok(insert(claim, rules, mint)),
    }
}

fn insert(claim: &Claim, rules: &Rules, mint: Mint<'_>) -> Decision<ArbitrationOutcome> {
    let id = mint();
    let mut plan = Plan::default();
    plan.insert.push(new_lineage(
        claim,
        id.clone(),
        claim.fact.valid_until,
        rules.at,
    ));
    plan.link(&id, claim.fact.source_event_ids.clone());
    decision(plan, ArbitrationOutcome::Inserted { id })
}

/// The claim restates a live fact: link its events, and let better trust lift
/// the stored version and close the live facts that contradict it.
fn corroborate(
    claim: &Claim,
    matched: &Fact,
    live: &[&Fact],
    rules: &Rules,
    mint: Mint<'_>,
) -> Decision<ArbitrationOutcome> {
    let fresh = unlinked(&claim.fact.source_event_ids, matched);
    let raises = claim.fact.trust_score > matched.trust_score;
    let lineage_id = matched.lineage_id.clone();
    if fresh.is_empty() && !raises {
        return decision(
            Plan::default(),
            ArbitrationOutcome::Corroborated {
                lineage_id,
                changed: false,
            },
        );
    }
    let mut plan = Plan::default();
    plan.link(&lineage_id, fresh);
    if raises {
        plan.close.push(matched.id.clone());
        let mut lifted = successor(matched, mint(), rules.at);
        lifted.trust_score = claim.fact.trust_score;
        lifted.evidence_at = matched.evidence_at.max(claim.fact.evidence_at);
        let lifted_id = lifted.id.clone();
        plan.insert.push(lifted);
        if rules.spec.cardinality == Cardinality::One {
            let beaten = live.iter().filter(|other| {
                other.lineage_id != matched.lineage_id
                    && other.trust_score < claim.fact.trust_score
                    && claim.fact.evidence_at >= other.evidence_at
            });
            for other in beaten {
                plan.end_version(
                    other,
                    matched.valid_from.max(other.valid_from),
                    claim.fact.evidence_at,
                    Some(lifted_id.clone()),
                    rules,
                    mint,
                );
            }
        }
    }
    decision(
        plan,
        ArbitrationOutcome::Corroborated {
            lineage_id,
            changed: true,
        },
    )
}

/// The claim names a different object than the live facts of a single-valued key.
fn differing(
    claim: &Claim,
    live: &[&Fact],
    rules: &Rules,
    mint: Mint<'_>,
) -> Result<Decision<ArbitrationOutcome>> {
    let newest = live
        .iter()
        .max_by_key(|version| (version.evidence_at, version.valid_from, version.id.as_str()));
    let strongest = live.iter().max_by(|a, b| {
        a.trust_score
            .total_cmp(&b.trust_score)
            .then(a.evidence_at.cmp(&b.evidence_at))
            .then(a.id.as_str().cmp(b.id.as_str()))
    });
    let (Some(newest), Some(strongest)) = (newest, strongest) else {
        return Ok(insert(claim, rules, mint));
    };
    if claim.fact.evidence_at < newest.evidence_at {
        historical(claim, newest, rules, mint)
    } else if claim.fact.trust_score >= strongest.trust_score {
        Ok(replace(claim, live, rules, mint))
    } else {
        Ok(compete(claim, live, strongest, rules, mint))
    }
}

/// The claim rests on older evidence than the live fact, so it is recorded as
/// the past that fact followed, and the live fact stays.
fn historical(
    claim: &Claim,
    newest: &Fact,
    rules: &Rules,
    mint: Mint<'_>,
) -> Result<Decision<ArbitrationOutcome>> {
    let until = newest.valid_from.min(claim.fact.valid_until);
    if claim.fact.valid_from >= until {
        return Err(Error::InvalidFactInterval {
            valid_from: claim.fact.valid_from.0,
            valid_until: until.0,
        });
    }
    let id = mint();
    let mut plan = Plan::default();
    plan.insert
        .push(new_lineage(claim, id.clone(), until, rules.at));
    plan.link(&id, claim.fact.source_event_ids.clone());
    Ok(decision(plan, ArbitrationOutcome::Historical { id }))
}

fn replace(
    claim: &Claim,
    live: &[&Fact],
    rules: &Rules,
    mint: Mint<'_>,
) -> Decision<ArbitrationOutcome> {
    let inserted = mint();
    let mut plan = Plan::default();
    plan.insert.push(new_lineage(
        claim,
        inserted.clone(),
        claim.fact.valid_until,
        rules.at,
    ));
    plan.link(&inserted, claim.fact.source_event_ids.clone());
    let closed = live
        .iter()
        .map(|old| {
            plan.end_version(
                old,
                claim.fact.valid_from.max(old.valid_from),
                claim.fact.evidence_at,
                Some(inserted.clone()),
                rules,
                mint,
            )
        })
        .collect();
    decision(plan, ArbitrationOutcome::Replaced { closed, inserted })
}

/// A weaker claim is stored beside the stronger live facts. Past the cap the
/// weakest, oldest competing lineage is retracted, the claim itself included.
fn compete(
    claim: &Claim,
    live: &[&Fact],
    strongest: &Fact,
    rules: &Rules,
    mint: Mint<'_>,
) -> Decision<ArbitrationOutcome> {
    let mut plan = Plan::default();
    let weakest_rival = live
        .iter()
        .filter(|version| version.lineage_id != strongest.lineage_id)
        .min_by(|a, b| {
            a.trust_score
                .total_cmp(&b.trust_score)
                .then(a.tx_from.cmp(&b.tx_from))
                .then(a.id.as_str().cmp(b.id.as_str()))
        });
    // Every live version but the strongest is a rival, and the claim makes one more.
    let competing = live.len();
    if let (true, Some(weakest)) = (competing > rules.caps.max_competing_per_key, weakest_rival) {
        // The claim is the newest, so on a trust tie the older rival goes.
        if claim.fact.trust_score < weakest.trust_score {
            plan.suppress.push(Suppression {
                subject: claim.fact.subject.clone(),
                predicate: claim.fact.predicate.clone(),
                scope: claim.scope.clone(),
                object_key: claim.object_key.clone(),
                trust_score: claim.fact.trust_score,
                retracted_at: rules.at,
            });
            return decision(plan, ArbitrationOutcome::Suppressed);
        }
        plan.retire(weakest, rules.at);
    }
    let id = mint();
    plan.insert.push(new_lineage(
        claim,
        id.clone(),
        claim.fact.valid_until,
        rules.at,
    ));
    plan.link(&id, claim.fact.source_event_ids.clone());
    decision(plan, ArbitrationOutcome::Competing { id })
}

pub(super) fn decide_end(
    state: &KeyState,
    claim: &EndClaim,
    rules: &Rules,
    mint: Mint<'_>,
) -> Decision<EndOutcome> {
    let end = &claim.end;
    let no_change = || decision(Plan::default(), EndOutcome::NoMatch);
    let already_recorded = state.versions.iter().any(|version| {
        version.object_key == claim.object_key
            && version.valid_until == end.valid_until
            && unlinked(&end.source_event_ids, version).is_empty()
    });
    let live = state.live(end.evidence_at);
    let Some(target) = live_match(&live, &claim.object_key, rules.mode()) else {
        return no_change();
    };
    if already_recorded || end.valid_until >= target.valid_until {
        return no_change();
    }
    if end.valid_until <= target.valid_from {
        return decision(Plan::default(), EndOutcome::Rejected);
    }
    if end.trust_score < target.trust_score {
        return decision(Plan::default(), EndOutcome::Outranked);
    }
    let mut plan = Plan::default();
    if end.evidence_at < target.evidence_at {
        let id = mint();
        let mut past = successor(target, id.clone(), rules.at);
        past.lineage_id = id.clone();
        past.valid_until = end.valid_until;
        past.evidence_at = end.evidence_at;
        past.trust_score = end.trust_score;
        past.closed_by = None;
        past.source_event_ids = end.source_event_ids.clone();
        plan.insert.push(past);
        plan.link(&id, end.source_event_ids.clone());
        return decision(plan, EndOutcome::Ended { closed: vec![] });
    }
    let closed = plan.end_version(target, end.valid_until, end.evidence_at, None, rules, mint);
    plan.link(&target.lineage_id, unlinked(&end.source_event_ids, target));
    decision(
        plan,
        EndOutcome::Ended {
            closed: vec![closed],
        },
    )
}

/// What a retraction needs to know about the key, beyond the version it closes.
pub(super) struct RetractInput<'a> {
    pub target: &'a Fact,
    /// Valid-time closed versions the target replaced, each with the version
    /// it had before it was closed.
    pub restorable: &'a [(Fact, Fact)],
    /// Whether another live fact holds the key.
    pub key_taken: bool,
    pub restore_previous: bool,
}

pub(super) fn decide_retract(
    input: &RetractInput<'_>,
    at: Millis,
    mint: Mint<'_>,
) -> Decision<RetractOutcome> {
    if input.restore_previous && input.key_taken {
        return decision(Plan::default(), RetractOutcome::RestoreBlocked);
    }
    let mut plan = Plan::default();
    plan.retire(input.target, at);
    let mut restored = Vec::new();
    if input.restore_previous {
        for (closed, before) in input.restorable {
            plan.close.push(closed.id.clone());
            let reopened = successor(before, mint(), at);
            restored.push(reopened.id.clone());
            plan.insert.push(reopened);
        }
    }
    decision(plan, RetractOutcome::Retracted { restored })
}

#[cfg(test)]
mod tests;
