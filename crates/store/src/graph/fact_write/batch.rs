// SPDX-License-Identifier: Apache-2.0
//! Asserting many facts in one transaction and one `FactWrite`, so a batch
//! costs one fsync under the write lock and lands whole or not at all.

use std::collections::BTreeMap;

use super::{read_state, Key};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::graph::arbitration::{decide_assert, Claim, Decision, KeyState, Plan, Rules};
use crate::graph::{validate_scope, Graph};
use crate::ids::{FactId, Millis};
use crate::types::{ArbitrationCaps, ArbitrationOutcome, NewFact, PredicateSpec};

/// The most facts one batch may hold. The write lock is held for the whole
/// batch, so the bound keeps one caller from holding it for long.
pub const MAX_FACTS_PER_BATCH: usize = 64;

type StateKey = (String, String, String);

fn in_batch(index: usize) -> impl FnOnce(Error) -> Error {
    move |source| Error::FactInBatch {
        index,
        source: Box::new(source),
    }
}

impl<B: Backend> Graph<B> {
    /// Asserts `facts`, each with the rules of its predicate, as `assert_fact`
    /// would one after another, but in a single transaction with a single log
    /// record. A fact sees the effect of the ones before it. If any fact is
    /// refused, none is written and the error names its position. A version
    /// that a later fact of the batch closes is stored already closed, at the
    /// batch's one transaction time.
    pub async fn assert_facts(
        &self,
        facts: Vec<(NewFact, PredicateSpec)>,
        write_source: &str,
        caps: ArbitrationCaps,
    ) -> Result<Vec<ArbitrationOutcome>> {
        if facts.len() > MAX_FACTS_PER_BATCH {
            return Err(Error::FactBatchTooLarge {
                len: facts.len(),
                max: MAX_FACTS_PER_BATCH,
            });
        }
        if facts.is_empty() {
            return Ok(Vec::new());
        }
        let claims = facts
            .into_iter()
            .enumerate()
            .map(|(index, (fact, spec))| {
                let scope = validate_scope(&fact.scope)
                    .map_err(in_batch(index))?
                    .unwrap_or_default();
                let claim = Claim::new(fact, scope).map_err(in_batch(index))?;
                Ok((claim, spec))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut write = self.open_write().await?;
        let tx = &mut *write.tx;
        let decided = self.decide_batch(tx, &claims, caps).await;
        let outcomes = self.finish_write(write, decided, write_source).await?;
        tracing::debug!(facts = outcomes.len(), "fact batch asserted");
        Ok(outcomes)
    }

    async fn decide_batch(
        &self,
        tx: &mut dyn BackendTx,
        claims: &[(Claim, PredicateSpec)],
        caps: ArbitrationCaps,
    ) -> Result<(Decision<Vec<ArbitrationOutcome>>, Millis)> {
        let keys: Vec<Key> = claims
            .iter()
            .map(|(claim, _)| Key::new(&claim.fact.subject, &claim.fact.predicate, &claim.scope))
            .collect();
        // One time for the whole batch, no earlier than any key's latest version.
        let mut at = self.clock.now();
        for key in &keys {
            at = at.max(self.transaction_time(tx, key).await?);
        }
        let mut states: BTreeMap<StateKey, KeyState> = BTreeMap::new();
        let mut merged = Plan::default();
        let mut outcomes = Vec::with_capacity(claims.len());
        for (index, ((claim, spec), key)) in claims.iter().zip(&keys).enumerate() {
            let state_key = (
                key.subject.clone(),
                key.predicate.clone(),
                key.scope.clone(),
            );
            if !states.contains_key(&state_key) {
                states.insert(state_key.clone(), read_state(tx, key).await?);
            }
            let state = states.get_mut(&state_key).expect("state was just read");
            let rules = Rules {
                spec: *spec,
                caps,
                at,
            };
            let decision =
                decide_assert(state, claim, &rules, &mut FactId::new).map_err(in_batch(index))?;
            state.apply(&decision.plan);
            merge(&mut merged, decision.plan, at);
            outcomes.push(decision.outcome);
        }
        let decision = Decision {
            plan: merged,
            outcome: outcomes,
        };
        Ok((decision, at))
    }
}

/// Adds `next` to `merged`. A version `next` closes that `merged` inserts is
/// stored closed, since a close in the same record would run before its insert.
fn merge(merged: &mut Plan, next: Plan, at: Millis) {
    for id in next.close {
        match merged.insert.iter_mut().find(|version| version.id == id) {
            Some(inserted) => inserted.tx_to = at,
            None => merged.close.push(id),
        }
    }
    merged.insert.extend(next.insert);
    for (lineage, events) in next.link {
        match merged.link.iter_mut().find(|(known, _)| *known == lineage) {
            Some((_, known)) => {
                known.extend(events);
                known.sort();
                known.dedup();
            }
            None => merged.link.push((lineage, events)),
        }
    }
    merged.suppress.extend(next.suppress);
}
