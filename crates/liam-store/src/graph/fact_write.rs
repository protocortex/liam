// SPDX-License-Identifier: Apache-2.0
//! Writing facts. Each operation reads the rows of one `(subject, predicate,
//! scope)` key, lets `arbitration` decide, and applies the decision, all in one
//! `begin()` transaction. The transaction holds the write lock from the read to
//! the commit, so two racing writers on a key are serialized and the second
//! reads the first's rows. It touches the rows of one key only, never a
//! caller-sized batch, and never spans a model call.
//!
//! These writes go straight to the projection tables and are not in the event
//! log, so a rebuild from the log does not reproduce them.

use std::collections::{BTreeMap, BTreeSet};

use super::arbitration::{
    decide_assert, decide_end, decide_retract, Claim, Decision, EndClaim, KeyState, Plan,
    RetractInput, Rules, Suppression,
};
use super::logged_write::commit_or_abandon;
use super::{validate_scope, Graph};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{FactId, Millis, FOREVER};
use crate::types::{
    ArbitrationCaps, ArbitrationOutcome, EndFact, EndOutcome, ExtractorName, Fact, NewFact,
    PredicateSpec, RetractOutcome,
};
use crate::value::{Row, Value};

const FACT_COLUMNS: &str = "id, lineage_id, subject, predicate, object, object_key, scope,
       evidence_at, valid_from, valid_until, tx_from, tx_to, closed_by, confidence,
       trust_score, extractor_id, extractor_version";

const KEY_FILTER: &str = "subject = ?1 AND predicate = ?2 AND scope = ?3";

const INSERT_VERSION_SQL: &str = "INSERT INTO facts
       (id, lineage_id, subject, predicate, object, object_key, scope, evidence_at,
        valid_from, valid_until, tx_from, tx_to, closed_by, confidence, trust_score,
        extractor_id, extractor_version)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)";

const CLOSE_VERSION_SQL: &str = "UPDATE facts SET tx_to = ?1 WHERE id = ?2 AND tx_to = ?3";

const LINK_SOURCE_SQL: &str =
    "INSERT OR IGNORE INTO fact_sources (lineage_id, event_id) VALUES (?1, ?2)";

// A key retracted twice keeps the later retraction whole: its trust belongs to
// its time, so the two are never mixed. On a tie the higher trust stays.
const SUPPRESS_SQL: &str = "INSERT INTO fact_suppressions
       (subject, predicate, scope, object_key, trust_score, retracted_at)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
     ON CONFLICT(subject, predicate, scope, object_key) DO UPDATE SET
       trust_score = CASE
         WHEN excluded.retracted_at > retracted_at
           OR (excluded.retracted_at = retracted_at AND excluded.trust_score > trust_score)
         THEN excluded.trust_score ELSE trust_score END,
       retracted_at = MAX(retracted_at, excluded.retracted_at)";

const CLOSED_ROWS: u64 = 1;

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

fn corrupt(id: &str, detail: &str) -> Error {
    Error::CorruptFactState(format!("fact version {id} {detail}"))
}

fn version_from_row(row: &Row) -> Result<Fact> {
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

fn version_params(version: &Fact) -> Vec<Value> {
    vec![
        version.id.as_str().into(),
        version.lineage_id.as_str().into(),
        version.subject.as_str().into(),
        version.predicate.as_str().into(),
        version.object.as_str().into(),
        version.object_key.as_str().into(),
        version.scope.as_deref().unwrap_or_default().into(),
        version.evidence_at.into(),
        version.valid_from.into(),
        version.valid_until.into(),
        version.tx_from.into(),
        version.tx_to.into(),
        version.closed_by.as_ref().map_or("", FactId::as_str).into(),
        version.confidence.into(),
        version.trust_score.into(),
        version.extractor_id.as_str().into(),
        version.extractor_version.as_str().into(),
    ]
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

/// A close must hit the one current row it names, or the plan was decided on
/// state that is no longer there.
fn check_closed(id: &FactId, updated: u64) -> Result<()> {
    if updated == CLOSED_ROWS {
        Ok(())
    } else {
        Err(corrupt(
            id.as_str(),
            &format!("closed {updated} current rows instead of {CLOSED_ROWS}"),
        ))
    }
}

async fn apply(tx: &mut dyn BackendTx, plan: &Plan, at: Millis) -> Result<()> {
    for id in &plan.close {
        let updated = tx
            .execute(
                CLOSE_VERSION_SQL,
                &[at.into(), id.as_str().into(), FOREVER.into()],
            )
            .await?;
        check_closed(id, updated)?;
    }
    for version in &plan.insert {
        tx.execute(INSERT_VERSION_SQL, &version_params(version))
            .await?;
    }
    for (lineage, events) in &plan.link {
        for event in events {
            tx.execute(
                LINK_SOURCE_SQL,
                &[lineage.as_str().into(), event.as_str().into()],
            )
            .await?;
        }
    }
    for blocked in &plan.suppress {
        tx.execute(
            SUPPRESS_SQL,
            &[
                blocked.subject.as_str().into(),
                blocked.predicate.as_str().into(),
                blocked.scope.as_str().into(),
                blocked.object_key.as_str().into(),
                blocked.trust_score.into(),
                blocked.retracted_at.into(),
            ],
        )
        .await?;
    }
    Ok(())
}

impl<B: Backend> Graph<B> {
    /// Asserts a fact, arbitrating it against the live facts of its key by
    /// trust and recency (ADR-0007). A fact with no source event, a trust
    /// outside `[0.0, 1.0]`, or a validity that does not start before it ends is
    /// refused with a typed error before the store is touched.
    pub async fn assert_fact(
        &self,
        fact: NewFact,
        spec: &PredicateSpec,
        caps: ArbitrationCaps,
    ) -> Result<ArbitrationOutcome> {
        let scope = validate_scope(&fact.scope)?.unwrap_or_default();
        let claim = Claim::new(fact, scope)?;
        let key = Key::new(&claim.fact.subject, &claim.fact.predicate, &claim.scope);
        let outcome = self
            .write_key(&key, spec, caps, |state, rules| {
                decide_assert(state, &claim, rules, &mut FactId::new)
            })
            .await?;
        tracing::debug!(?outcome, "fact asserted");
        Ok(outcome)
    }

    /// Ends the live fact an `end` statement matches, when its trust is not
    /// below the fact's. It never inserts a live fact.
    pub async fn end_fact(&self, end: EndFact, spec: &PredicateSpec) -> Result<EndOutcome> {
        let scope = validate_scope(&end.scope)?.unwrap_or_default();
        let claim = EndClaim::new(end, scope)?;
        let key = Key::new(&claim.end.subject, &claim.end.predicate, &claim.scope);
        let outcome = self
            .write_key(&key, spec, ArbitrationCaps::default(), |state, rules| {
                Ok(decide_end(state, &claim, rules, &mut FactId::new))
            })
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
    ) -> Result<RetractOutcome> {
        let mut tx = self.backend.begin().await?;
        let decided = async {
            let Some(target) = lineage_head(&mut *tx, lineage_id).await? else {
                return Ok(RetractOutcome::NotFound);
            };
            let key = Key::of(&target);
            let at = self.transaction_time(&mut *tx, &key).await?;
            let restorable = restorable_versions(&mut *tx, &target).await?;
            let key_taken =
                restore_previous && key_is_taken(&mut *tx, &key, &target, &restorable).await?;
            let input = RetractInput {
                target: &target,
                restorable: &restorable,
                key_taken,
                restore_previous,
            };
            let decision = decide_retract(&input, at, &mut FactId::new);
            apply(&mut *tx, &decision.plan, at).await?;
            Ok(decision.outcome)
        }
        .await;
        let outcome = commit_or_abandon(tx, decided).await?;
        tracing::debug!(?outcome, "fact retracted");
        Ok(outcome)
    }

    /// Reads the key, lets `decide` plan against it, applies the plan and
    /// commits, all in one transaction.
    async fn write_key<T>(
        &self,
        key: &Key,
        spec: &PredicateSpec,
        caps: ArbitrationCaps,
        decide: impl FnOnce(&KeyState, &Rules) -> Result<Decision<T>>,
    ) -> Result<T> {
        let mut tx = self.backend.begin().await?;
        let decided = async {
            let state = read_state(&mut *tx, key).await?;
            let at = self.transaction_time(&mut *tx, key).await?;
            let rules = Rules {
                spec: *spec,
                caps,
                at,
            };
            let decision = decide(&state, &rules)?;
            apply(&mut *tx, &decision.plan, at).await?;
            Ok(decision.outcome)
        }
        .await;
        commit_or_abandon(tx, decided).await
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_close_that_updates_other_than_one_row_is_corrupt_state() {
        for updated in [0, 2] {
            // Arrange
            let id = FactId::from_raw("a");

            // Act
            let result = check_closed(&id, updated);

            // Assert
            assert!(
                matches!(result, Err(Error::CorruptFactState(_))),
                "{updated} rows"
            );
        }
    }

    #[test]
    fn a_close_that_updates_one_row_is_accepted() {
        // Act
        let result = check_closed(&FactId::from_raw("a"), CLOSED_ROWS);

        // Assert
        assert!(result.is_ok());
    }
}
