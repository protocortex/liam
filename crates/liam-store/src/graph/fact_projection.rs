// SPDX-License-Identifier: Apache-2.0
//! The statements a `FactWrite` projects. A live fact write and a replay of its
//! log record both run them, so the two cannot drift apart.

use liam_log::event::{FactRowEffect, LogPayload, SuppressionEffect};

use super::fact_write::corrupt;
use crate::backend::BackendTx;
use crate::error::Result;
use crate::ids::{FactId, FOREVER};
use crate::value::Value;

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

fn version_params(version: &FactRowEffect) -> Vec<Value> {
    vec![
        version.id.as_str().into(),
        version.lineage_id.as_str().into(),
        version.subject.as_str().into(),
        version.predicate.as_str().into(),
        version.object.as_str().into(),
        version.object_key.as_str().into(),
        version.scope.as_str().into(),
        version.evidence_at.into(),
        version.valid_from.into(),
        version.valid_until.into(),
        version.tx_from.into(),
        version.tx_to.into(),
        version.closed_by.as_str().into(),
        version.confidence.into(),
        version.trust_score.into(),
        version.extractor_id.as_str().into(),
        version.extractor_version.as_str().into(),
    ]
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

/// One statement of a fact write's projection.
pub(super) enum FactStep {
    Close {
        id: String,
        tx_to: i64,
    },
    Insert(FactRowEffect),
    Link {
        lineage_id: String,
        event_id: String,
    },
    Suppress(SuppressionEffect),
}

/// The statements a `FactWrite` projects, which a live write and a replay share.
pub(super) fn fact_steps(payload: &LogPayload) -> Vec<FactStep> {
    let LogPayload::FactWrite {
        at,
        closed_version_ids,
        inserted,
        source_links,
        suppressions,
    } = payload
    else {
        return Vec::new();
    };
    let closes = closed_version_ids.iter().map(|id| FactStep::Close {
        id: id.clone(),
        tx_to: *at,
    });
    let inserts = inserted.iter().cloned().map(FactStep::Insert);
    let links = source_links.iter().map(|(lineage, event)| FactStep::Link {
        lineage_id: lineage.clone(),
        event_id: event.clone(),
    });
    let blocks = suppressions.iter().cloned().map(FactStep::Suppress);
    closes.chain(inserts).chain(links).chain(blocks).collect()
}

pub(super) async fn apply_fact_step(tx: &mut dyn BackendTx, step: &FactStep) -> Result<()> {
    match step {
        FactStep::Close { id, tx_to } => close_version(tx, id, *tx_to).await,
        FactStep::Insert(version) => {
            tx.execute(INSERT_VERSION_SQL, &version_params(version))
                .await?;
            Ok(())
        }
        FactStep::Link {
            lineage_id,
            event_id,
        } => {
            let params = [lineage_id.as_str().into(), event_id.as_str().into()];
            tx.execute(LINK_SOURCE_SQL, &params).await?;
            Ok(())
        }
        FactStep::Suppress(blocked) => {
            let params = [
                blocked.subject.as_str().into(),
                blocked.predicate.as_str().into(),
                blocked.scope.as_str().into(),
                blocked.object_key.as_str().into(),
                blocked.trust_score.into(),
                blocked.retracted_at.into(),
            ];
            tx.execute(SUPPRESS_SQL, &params).await?;
            Ok(())
        }
    }
}

/// Closes a current version. One already closed at `tx_to` is a replay of this
/// close and is left as it is.
async fn close_version(tx: &mut dyn BackendTx, id: &str, tx_to: i64) -> Result<()> {
    let updated = tx
        .execute(
            CLOSE_VERSION_SQL,
            &[tx_to.into(), id.into(), FOREVER.into()],
        )
        .await?;
    if updated == 0 && closed_at(tx, id, tx_to).await? {
        return Ok(());
    }
    check_closed(&FactId::from_raw(id), updated)
}

async fn closed_at(tx: &mut dyn BackendTx, id: &str, tx_to: i64) -> Result<bool> {
    let rows = tx
        .query(
            "SELECT 1 FROM facts WHERE id = ?1 AND tx_to = ?2",
            &[id.into(), tx_to.into()],
        )
        .await?;
    Ok(!rows.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

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
