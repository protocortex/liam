// SPDX-License-Identifier: Apache-2.0
//! Writing facts. Each operation reads the rows of one `(subject, predicate,
//! scope)` key, lets `arbitration` decide, and applies the decision, all in one
//! `begin()` transaction. The transaction holds the write lock from the read to
//! the commit, so two racing writers on a key are serialized and the second
//! reads the first's rows. It touches the rows of one key only, never a
//! caller-sized batch, and never spans a model call.
//!
//! On a logged store the decision is appended to the log as one `FactWrite`
//! before it is applied, so a replay reproduces the rows without arbitrating
//! again. A decision that changes nothing appends nothing.

use std::collections::{BTreeMap, BTreeSet};

use liam_log::event::{FactRowEffect, LogEvent, LogPayload, SuppressionEffect};

use super::arbitration::{
    decide_assert, decide_end, decide_retract, Claim, Decision, EndClaim, KeyState, Plan,
    RetractInput, Rules, Suppression,
};
use super::logged_plan::{event, STORE_TRUST};
use super::logged_write::{abandon, HeldLog};
use super::projection::steps_for;
use super::{validate_scope, Graph};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{FactId, Millis, FOREVER};
use crate::types::{
    ArbitrationCaps, ArbitrationOutcome, EndFact, EndOutcome, ExtractorName, Fact, NewFact,
    PredicateSpec, RetractOutcome,
};
use crate::value::{Row, Value};

pub(super) const FACT_COLUMNS: &str =
    "id, lineage_id, subject, predicate, object, object_key, scope,
       evidence_at, valid_from, valid_until, tx_from, tx_to, closed_by, confidence,
       trust_score, extractor_id, extractor_version";

const KEY_FILTER: &str = "subject = ?1 AND predicate = ?2 AND scope = ?3";

const LATEST_TX_SQL: &str =
    "SELECT MAX(tx_from) FROM facts WHERE subject = ?1 AND predicate = ?2 AND scope = ?3";

const WITHOUT_SOURCES_SQL: &str = "SELECT DISTINCT id FROM facts
     WHERE NOT EXISTS (SELECT 1 FROM fact_sources WHERE fact_sources.lineage_id = facts.lineage_id)
     ORDER BY id";

/// The rows a decision reads and writes, addressed by their key columns.
struct Key {
    subject: String,
    predicate: String,
    scope: String,
}

impl Key {
    fn new(subject: &str, predicate: &str, scope: &str) -> Self {
        Self {
            subject: subject.to_owned(),
            predicate: predicate.to_owned(),
            scope: scope.to_owned(),
        }
    }

    fn of(version: &Fact) -> Self {
        Self::new(
            &version.subject,
            &version.predicate,
            version.scope.as_deref().unwrap_or_default(),
        )
    }

    fn params(&self) -> [Value; 3] {
        [
            self.subject.as_str().into(),
            self.predicate.as_str().into(),
            self.scope.as_str().into(),
        ]
    }
}

pub(super) fn corrupt(id: &str, detail: &str) -> Error {
    Error::CorruptFactState(format!("fact version {id} {detail}"))
}

pub(super) fn version_from_row(row: &Row) -> Result<Fact> {
    let id = row.get_string(0)?;
    let extractor = row.get_string(15)?;
    let extractor_id = ExtractorName::from_stored(&extractor)
        .ok_or_else(|| corrupt(&id, &format!("has the unknown extractor {extractor}")))?;
    let scope = row.get_string(6)?;
    let closed_by = row.get_string(12)?;
    Ok(Fact {
        lineage_id: FactId::from_raw(row.get_string(1)?),
        subject: row.get_string(2)?,
        predicate: row.get_string(3)?,
        object: row.get_string(4)?,
        object_key: row.get_string(5)?,
        scope: (!scope.is_empty()).then_some(scope),
        evidence_at: Millis(row.get_i64(7)?),
        valid_from: Millis(row.get_i64(8)?),
        valid_until: Millis(row.get_i64(9)?),
        tx_from: Millis(row.get_i64(10)?),
        tx_to: Millis(row.get_i64(11)?),
        closed_by: (!closed_by.is_empty()).then(|| FactId::from_raw(closed_by)),
        confidence: row.get_f64(13)?,
        trust_score: row.get_f64(14)?,
        extractor_id,
        extractor_version: row.get_string(16)?,
        source_event_ids: Vec::new(),
        id: FactId::from_raw(id),
    })
}

fn version_effect(version: &Fact) -> FactRowEffect {
    FactRowEffect {
        id: version.id.as_str().to_owned(),
        lineage_id: version.lineage_id.as_str().to_owned(),
        subject: version.subject.clone(),
        predicate: version.predicate.clone(),
        object: version.object.clone(),
        object_key: version.object_key.clone(),
        scope: version.scope.clone().unwrap_or_default(),
        evidence_at: version.evidence_at.0,
        valid_from: version.valid_from.0,
        valid_until: version.valid_until.0,
        tx_from: version.tx_from.0,
        tx_to: version.tx_to.0,
        closed_by: version
            .closed_by
            .as_ref()
            .map_or_else(String::new, |id| id.as_str().to_owned()),
        confidence: version.confidence,
        trust_score: version.trust_score,
        extractor_id: version.extractor_id.as_str().to_owned(),
        extractor_version: version.extractor_version.clone(),
    }
}

async fn current_versions(tx: &mut dyn BackendTx, key: &Key) -> Result<Vec<Fact>> {
    let sql = format!(
        "SELECT {FACT_COLUMNS} FROM facts WHERE {KEY_FILTER} AND tx_to = ?4
         ORDER BY valid_from, id"
    );
    let [subject, predicate, scope] = key.params();
    let rows = tx
        .query(&sql, &[subject, predicate, scope, FOREVER.into()])
        .await?;
    rows.iter().map(version_from_row).collect()
}

async fn read_state(tx: &mut dyn BackendTx, key: &Key) -> Result<KeyState> {
    let mut versions = current_versions(tx, key).await?;
    let [subject, predicate, scope] = key.params();
    let linked = tx
        .query(
            "SELECT lineage_id, event_id FROM fact_sources
             WHERE lineage_id IN (SELECT lineage_id FROM facts
                                  WHERE subject = ?1 AND predicate = ?2 AND scope = ?3
                                    AND tx_to = ?4)
             ORDER BY lineage_id, event_id",
            &[subject, predicate, scope, FOREVER.into()],
        )
        .await?;
    let mut sources: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for row in &linked {
        sources
            .entry(row.get_string(0)?)
            .or_default()
            .push(row.get_string(1)?);
    }
    for version in &mut versions {
        version.source_event_ids = sources
            .get(version.lineage_id.as_str())
            .cloned()
            .unwrap_or_default();
    }
    let blocked = tx
        .query(
            &format!(
                "SELECT object_key, trust_score, retracted_at FROM fact_suppressions
                 WHERE {KEY_FILTER}"
            ),
            &key.params(),
        )
        .await?;
    let suppressions = blocked
        .iter()
        .map(|row| {
            Ok(Suppression {
                subject: key.subject.clone(),
                predicate: key.predicate.clone(),
                scope: key.scope.clone(),
                object_key: row.get_string(0)?,
                trust_score: row.get_f64(1)?,
                retracted_at: Millis(row.get_i64(2)?),
            })
        })
        .collect::<Result<_>>()?;
    Ok(KeyState {
        versions,
        suppressions,
    })
}

/// A fact write between its open and its finish: the log lock and the transaction.
struct OpenWrite<'a> {
    held: Option<HeldLog>,
    tx: Box<dyn BackendTx + 'a>,
}

/// The log event for a decision, or `None` when it changes nothing. It is
/// stamped by the strongest fact it inserts.
fn fact_event(plan: &Plan, at: Millis, write_source: &str) -> Option<LogEvent> {
    let inserted: Vec<FactRowEffect> = plan.insert.iter().map(version_effect).collect();
    let source_links: Vec<(String, String)> = plan
        .link
        .iter()
        .flat_map(|(lineage, events)| {
            events
                .iter()
                .map(|event| (lineage.as_str().to_owned(), event.clone()))
        })
        .collect();
    let suppressions: Vec<SuppressionEffect> = plan
        .suppress
        .iter()
        .map(|blocked| SuppressionEffect {
            subject: blocked.subject.clone(),
            predicate: blocked.predicate.clone(),
            scope: blocked.scope.clone(),
            object_key: blocked.object_key.clone(),
            trust_score: blocked.trust_score,
            retracted_at: blocked.retracted_at.0,
        })
        .collect();
    if plan.close.is_empty()
        && inserted.is_empty()
        && source_links.is_empty()
        && suppressions.is_empty()
    {
        return None;
    }
    let trust = inserted
        .iter()
        .map(|version| version.trust_score)
        .reduce(f64::max)
        .unwrap_or(STORE_TRUST);
    let observed_at = inserted
        .iter()
        .map(|version| version.evidence_at)
        .max()
        .unwrap_or(at.0);
    let payload = LogPayload::FactWrite {
        at: at.0,
        closed_version_ids: plan.close.iter().map(|id| id.as_str().to_owned()).collect(),
        inserted,
        source_links,
        suppressions,
    };
    Some(event(
        [0; 32],
        (write_source, trust),
        observed_at,
        at,
        payload,
    ))
}

impl<B: Backend> Graph<B> {
    /// Asserts a fact, arbitrating it against the live facts of its key by
    /// trust and recency (ADR-0007). A fact with no source event, a trust
    /// outside `[0.0, 1.0]`, or a validity that does not start before it ends is
    /// refused with a typed error before the store is touched. `write_source`
    /// stamps the log event, so a rolled-back run's writes can be told apart.
    pub async fn assert_fact(
        &self,
        fact: NewFact,
        spec: &PredicateSpec,
        write_source: &str,
        caps: ArbitrationCaps,
    ) -> Result<ArbitrationOutcome> {
        let scope = validate_scope(&fact.scope)?.unwrap_or_default();
        let claim = Claim::new(fact, scope)?;
        let key = Key::new(&claim.fact.subject, &claim.fact.predicate, &claim.scope);
        let outcome = self
            .write_key(&key, spec, caps, write_source, |state, rules| {
                decide_assert(state, &claim, rules, &mut FactId::new)
            })
            .await?;
        tracing::debug!(?outcome, "fact asserted");
        Ok(outcome)
    }

    /// Ends the live fact an `end` statement matches, when its trust is not
    /// below the fact's. It never inserts a live fact.
    pub async fn end_fact(
        &self,
        end: EndFact,
        spec: &PredicateSpec,
        write_source: &str,
    ) -> Result<EndOutcome> {
        let scope = validate_scope(&end.scope)?.unwrap_or_default();
        let claim = EndClaim::new(end, scope)?;
        let key = Key::new(&claim.end.subject, &claim.end.predicate, &claim.scope);
        let outcome = self
            .write_key(
                &key,
                spec,
                ArbitrationCaps::default(),
                write_source,
                |state, rules| Ok(decide_end(state, &claim, rules, &mut FactId::new)),
            )
            .await?;
        tracing::debug!(?outcome, "fact ended");
        Ok(outcome)
    }

    /// Stops asserting the lineage, as the escape hatch for a wrong fact. The
    /// retraction is a transaction-time close, so what the system believed
    /// before it stays readable, and it blocks the object from returning on
    /// evidence no newer than now. With `restore_previous` the facts the
    /// lineage had replaced become live again, unless another live fact holds
    /// the key.
    pub async fn retract_fact(
        &self,
        lineage_id: &FactId,
        restore_previous: bool,
        write_source: &str,
    ) -> Result<RetractOutcome> {
        let mut write = self.open_write().await?;
        let tx = &mut *write.tx;
        let decided = async {
            let Some(target) = lineage_head(tx, lineage_id).await? else {
                let unchanged = Decision {
                    plan: Plan::default(),
                    outcome: RetractOutcome::NotFound,
                };
                return Ok((unchanged, self.clock.now()));
            };
            let key = Key::of(&target);
            let at = self.transaction_time(tx, &key).await?;
            let restorable = restorable_versions(tx, &target).await?;
            let key_taken =
                restore_previous && key_is_taken(tx, &key, &target, &restorable).await?;
            let input = RetractInput {
                target: &target,
                restorable: &restorable,
                key_taken,
                restore_previous,
            };
            Ok((decide_retract(&input, at, &mut FactId::new), at))
        }
        .await;
        let outcome = self.finish_write(write, decided, write_source).await?;
        tracing::debug!(?outcome, "fact retracted");
        Ok(outcome)
    }

    /// Reads the key and lets `decide` plan against it, then logs and applies
    /// the plan, all in one transaction.
    async fn write_key<T>(
        &self,
        key: &Key,
        spec: &PredicateSpec,
        caps: ArbitrationCaps,
        write_source: &str,
        decide: impl FnOnce(&KeyState, &Rules) -> Result<Decision<T>>,
    ) -> Result<T> {
        let mut write = self.open_write().await?;
        let tx = &mut *write.tx;
        let decided = async {
            let state = read_state(tx, key).await?;
            let at = self.transaction_time(tx, key).await?;
            let rules = Rules {
                spec: *spec,
                caps,
                at,
            };
            Ok((decide(&state, &rules)?, at))
        }
        .await;
        self.finish_write(write, decided, write_source).await
    }

    /// Takes the log lock, when there is a log, and opens the transaction. The
    /// lock is held until `finish_write`, so the reads a decision makes cannot
    /// go stale before it is appended.
    async fn open_write(&self) -> Result<OpenWrite<'_>> {
        let held = match &self.log {
            Some(log) => Some(HeldLog::acquire_for_write(log).await?),
            None => None,
        };
        let tx = self.backend.begin().await?;
        Ok(OpenWrite { held, tx })
    }

    /// Appends the decided plan to the log and applies it, or abandons the
    /// transaction when the decision failed or changes nothing. A plan that
    /// changes nothing appends nothing.
    async fn finish_write<T>(
        &self,
        write: OpenWrite<'_>,
        decided: Result<(Decision<T>, Millis)>,
        write_source: &str,
    ) -> Result<T> {
        let OpenWrite { held, tx } = write;
        let (decision, at) = match decided {
            Ok(decided) => decided,
            Err(error) => {
                abandon(tx).await;
                return Err(error);
            }
        };
        let Some(event) = fact_event(&decision.plan, at, write_source) else {
            abandon(tx).await;
            return Ok(decision.outcome);
        };
        let steps = steps_for(&event.payload);
        match held {
            Some(mut held) => self.append_and_commit(tx, &mut held, event, &steps).await?,
            None => self.apply_and_commit(tx, &steps).await?,
        }
        Ok(decision.outcome)
    }

    /// Fact versions whose lineage has no source event, which no write of this
    /// store produces. Empty on a healthy store.
    pub async fn facts_without_sources(&self) -> Result<Vec<FactId>> {
        let rows = self.backend.query(WITHOUT_SOURCES_SQL, &[]).await?;
        rows.iter()
            .map(|row| Ok(FactId::from_raw(row.get_string(0)?)))
            .collect()
    }

    /// The time a write to `key` is stamped with: now, but never before the
    /// key's latest version, so a clock that stepped back cannot overlap history.
    async fn transaction_time(&self, tx: &mut dyn BackendTx, key: &Key) -> Result<Millis> {
        let rows = tx.query(LATEST_TX_SQL, &key.params()).await?;
        let latest = match rows.first().and_then(|row| row.0.first()) {
            Some(Value::Int(latest)) => Some(Millis(*latest)),
            _ => None,
        };
        let now = self.clock.now();
        Ok(latest.map_or(now, |latest| now.max(latest)))
    }
}

async fn lineage_head(tx: &mut dyn BackendTx, lineage_id: &FactId) -> Result<Option<Fact>> {
    let rows = tx
        .query(
            &format!("SELECT {FACT_COLUMNS} FROM facts WHERE lineage_id = ?1 AND tx_to = ?2"),
            &[lineage_id.as_str().into(), FOREVER.into()],
        )
        .await?;
    rows.first().map(version_from_row).transpose()
}

/// The valid-time closed versions the target replaced, paired with the version
/// each had before the replacement closed it.
async fn restorable_versions(tx: &mut dyn BackendTx, target: &Fact) -> Result<Vec<(Fact, Fact)>> {
    let [subject, predicate, scope] = Key::of(target).params();
    let closed = tx
        .query(
            &format!(
                "SELECT {FACT_COLUMNS} FROM facts
                 WHERE {KEY_FILTER} AND tx_to = ?4
                   AND closed_by IN (SELECT id FROM facts WHERE lineage_id = ?5)
                 ORDER BY rowid"
            ),
            &[
                subject,
                predicate,
                scope,
                FOREVER.into(),
                target.lineage_id.as_str().into(),
            ],
        )
        .await?;
    let mut pairs = Vec::new();
    for row in &closed {
        let closed = version_from_row(row)?;
        let before = tx
            .query(
                &format!(
                    "SELECT {FACT_COLUMNS} FROM facts
                     WHERE lineage_id = ?1 AND tx_to = ?2 AND id != ?3
                     ORDER BY rowid DESC LIMIT 1"
                ),
                &[
                    closed.lineage_id.as_str().into(),
                    closed.tx_from.into(),
                    closed.id.as_str().into(),
                ],
            )
            .await?;
        if let Some(before) = before.first() {
            pairs.push((closed, version_from_row(before)?));
        }
    }
    Ok(pairs)
}

/// Whether a live fact other than the target, and than the versions a restore
/// would reopen, holds the key.
async fn key_is_taken(
    tx: &mut dyn BackendTx,
    key: &Key,
    target: &Fact,
    restorable: &[(Fact, Fact)],
) -> Result<bool> {
    let reopened: BTreeSet<&str> = restorable
        .iter()
        .map(|(closed, _)| closed.lineage_id.as_str())
        .collect();
    let versions = current_versions(tx, key).await?;
    Ok(versions.iter().any(|version| {
        version.lineage_id != target.lineage_id
            && !reopened.contains(version.lineage_id.as_str())
            && version.valid_until > target.evidence_at
    }))
}
