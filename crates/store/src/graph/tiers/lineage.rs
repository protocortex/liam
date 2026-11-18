// SPDX-License-Identifier: Apache-2.0
//! The lineage key of a node, written with the node so that a live write and a
//! replay of its log record agree.
//!
//! A key is computed from the stored row and the `supersedes` edge the same
//! batch wrote, so the log needs no field for it. A node that replaces another
//! copies that node's key; any other node gets its own.

use crate::backend::{Backend, BackendTx};
use crate::error::Result;
use crate::graph::logged_write::commit_or_abandon;
use crate::graph::{opt_string, Graph};
use crate::ids::{NodeId, FOREVER};
use crate::types::{relation, Derivation, LineageKey, Tier};

/// Stands for a missing scope, which no length-prefixed field can equal.
const NO_SCOPE: &str = "!";

const ENTITY_PREFIX: &str = "e";
const SUBJECT_PREFIX: &str = "s";
const ROOT_PREFIX: &str = "n";

/// `len:text`, so a field that holds the separator cannot be mistaken for two.
fn field(text: &str) -> String {
    format!("{}:{text}", text.len())
}

fn scope_field(scope: Option<&str>) -> String {
    scope.map_or_else(|| NO_SCOPE.to_string(), field)
}

/// The key of an entity page, shared by every page of one subject in a scope.
pub fn entity_key(scope: Option<&str>, subject: &str) -> LineageKey {
    let (scope, subject) = (scope_field(scope), field(subject));
    LineageKey::from_raw(format!("{ENTITY_PREFIX}:{scope}:{subject}"))
}

/// The key of a node with a subject that is not an entity page. The producer is
/// part of it because it is part of what makes two writes versions of one
/// another.
pub fn subject_key(scope: Option<&str>, subject: &str, producer: &str) -> LineageKey {
    let (scope, subject, producer) = (scope_field(scope), field(subject), field(producer));
    LineageKey::from_raw(format!("{SUBJECT_PREFIX}:{scope}:{subject}:{producer}"))
}

/// The key of a node with no subject, taken from the first version of its line.
pub fn root_key(first_version: &str) -> LineageKey {
    LineageKey::from_raw(format!("{ROOT_PREFIX}:{}", field(first_version)))
}

/// What `key_of_row` needs of a stored node.
pub struct KeyedRow<'a> {
    pub id: &'a str,
    pub scope: Option<&'a str>,
    pub subject: Option<&'a str>,
    pub producer: &'a str,
    pub derivation: Derivation,
}

/// The key a node gets when it replaces nothing.
pub fn key_of_row(row: &KeyedRow<'_>) -> LineageKey {
    match (row.subject, row.derivation) {
        (Some(subject), Derivation::Inferred) => entity_key(row.scope, subject),
        (Some(subject), _) => subject_key(row.scope, subject, row.producer),
        (None, _) => root_key(row.id),
    }
}

/// Reads the key of the node `node_id` replaced, which `Step::Lineage` copies.
const REPLACED_KEY_SQL: &str = "SELECT l.lineage_key FROM edges e
     JOIN node_lineage l ON l.node_id = e.dst
     WHERE e.src = ?1 AND e.type = ?2 ORDER BY e.id LIMIT 1";

const NODE_FIELDS_SQL: &str =
    "SELECT scope, subject, producer, derivation, tx_from FROM nodes WHERE id = ?1";

const LINEAGE_INSERT_SQL: &str =
    "INSERT OR IGNORE INTO node_lineage (node_id, lineage_key) VALUES (?1, ?2)";

/// Sets a cold lineage hot when a node is written for it, unless another live
/// node of it is still cold. A cold node has no int8 form, which is how the
/// transaction tells, because the vector table is the backend's. A lineage left
/// cold is repaired by the next reconcile, which deletes the new node's vector
/// only if it was written before the lineage went cold, so it keeps it.
const COLD_TO_HOT_SQL: &str = "UPDATE item_tier
     SET tier = ?5, changed_at = ?2, residency_until = NULL, last_event_id = NULL
     WHERE lineage_key = ?1 AND tier = ?6
       AND NOT EXISTS (
         SELECT 1 FROM node_lineage l
         JOIN nodes n ON n.id = l.node_id
         JOIN node_quant q ON q.node_id = n.id
         WHERE l.lineage_key = ?1 AND n.tx_to = ?3 AND n.id != ?4 AND q.i8 IS NULL)";

const COLD_LINEAGE_SQL: &str = "SELECT 1 FROM item_tier WHERE lineage_key = ?1 AND tier = ?2";

/// Records the lineage of `node_id`, which must already be in `tx` together
/// with the `supersedes` edge that replaced another node, if any, and wakes the
/// lineage when it was cold. A node that is gone is left alone, and so is one
/// that has a key already.
pub(in crate::graph) async fn assign_lineage(tx: &mut dyn BackendTx, node_id: &str) -> Result<()> {
    let id = [node_id.into()];
    let fields = tx.query(NODE_FIELDS_SQL, &id).await?;
    let Some(fields) = fields.first() else {
        return Ok(());
    };
    let replaced = tx
        .query(
            REPLACED_KEY_SQL,
            &[node_id.into(), relation::SUPERSEDES.into()],
        )
        .await?;
    let key = match replaced.first() {
        Some(row) => LineageKey::from_raw(row.get_string(0)?),
        None => {
            let scope = opt_string(fields, 0)?;
            let subject = opt_string(fields, 1)?;
            let producer = fields.get_string(2)?;
            let derivation = Derivation::parse(&fields.get_string(3)?)?;
            key_of_row(&KeyedRow {
                id: node_id,
                scope: scope.as_deref(),
                subject: subject.as_deref(),
                producer: &producer,
                derivation,
            })
        }
    };
    tx.execute(LINEAGE_INSERT_SQL, &[node_id.into(), key.as_str().into()])
        .await?;
    // Most lineages have no tier row, so look before writing.
    let cold = tx
        .query(
            COLD_LINEAGE_SQL,
            &[key.as_str().into(), Tier::Cold.as_str().into()],
        )
        .await?;
    if cold.is_empty() {
        return Ok(());
    }
    let params = [
        key.as_str().into(),
        fields.get_i64(4)?.into(),
        FOREVER.into(),
        node_id.into(),
        Tier::Hot.as_str().into(),
        Tier::Cold.as_str().into(),
    ];
    tx.execute(COLD_TO_HOT_SQL, &params).await?;
    Ok(())
}

/// Nodes the lineage backfill assigns in one transaction.
const BACKFILL_CHUNK: usize = 200;

impl<B: Backend> Graph<B> {
    /// A node of every lineage, by lineage key, for the migration that seals an
    /// old log and needs the key group each lineage-keyed entry belongs to.
    pub async fn one_node_per_lineage(&self) -> Result<Vec<(LineageKey, NodeId)>> {
        let rows = self
            .backend
            .query(
                "SELECT lineage_key, MIN(node_id) FROM node_lineage GROUP BY lineage_key",
                &[],
            )
            .await?;
        rows.iter()
            .map(|row| {
                Ok((
                    LineageKey::from_raw(row.get_string(0)?),
                    NodeId::from_raw(row.get_string(1)?),
                ))
            })
            .collect()
    }

    /// The lineage of `node_id`, `None` when there is no such node. A node with
    /// no stored key, which the backfill at open leaves none of, gets the key it
    /// would be written with if it replaced nothing.
    pub async fn lineage_of(&self, node_id: &NodeId) -> Result<Option<LineageKey>> {
        let id = [node_id.as_str().into()];
        let stored = self
            .backend
            .query(
                "SELECT lineage_key FROM node_lineage WHERE node_id = ?1",
                &id,
            )
            .await?;
        if let Some(row) = stored.first() {
            return Ok(Some(LineageKey::from_raw(row.get_string(0)?)));
        }
        let fields = self.backend.query(NODE_FIELDS_SQL, &id).await?;
        let Some(fields) = fields.first() else {
            return Ok(None);
        };
        let scope = opt_string(fields, 0)?;
        let subject = opt_string(fields, 1)?;
        let producer = fields.get_string(2)?;
        Ok(Some(key_of_row(&KeyedRow {
            id: node_id.as_str(),
            scope: scope.as_deref(),
            subject: subject.as_deref(),
            producer: &producer,
            derivation: Derivation::parse(&fields.get_string(3)?)?,
        })))
    }

    /// Gives every node that has no lineage one, once. Nodes go in transaction
    /// time order, so a version finds the key of the node it replaced already
    /// stored and copies it, with no walk down the chain. The same step a write
    /// runs assigns each one, so the two cannot differ. Returns how many it
    /// assigned.
    pub async fn backfill_lineage(&self) -> Result<usize> {
        let done = self
            .backend
            .query("SELECT 1 FROM lineage_backfill_state WHERE id = 1", &[])
            .await?;
        if !done.is_empty() {
            return Ok(0);
        }
        let mut assigned = 0;
        let mut after = (i64::MIN, i64::MIN);
        loop {
            let rows = self
                .backend
                .query(
                    "SELECT id, tx_from, rowid FROM nodes n
                     WHERE NOT EXISTS (SELECT 1 FROM node_lineage WHERE node_id = n.id)
                       AND (tx_from > ?1 OR (tx_from = ?1 AND rowid > ?2))
                     ORDER BY tx_from, rowid LIMIT ?3",
                    &[
                        after.0.into(),
                        after.1.into(),
                        (BACKFILL_CHUNK as i64).into(),
                    ],
                )
                .await?;
            let Some(last) = rows.last() else { break };
            after = (last.get_i64(1)?, last.get_i64(2)?);
            let mut tx = self.backend.begin().await?;
            let chunk = async {
                for row in &rows {
                    assign_lineage(&mut *tx, &row.get_string(0)?).await?;
                }
                Ok(())
            }
            .await;
            commit_or_abandon(tx, chunk).await?;
            assigned += rows.len();
        }
        self.backend
            .execute(
                "INSERT OR REPLACE INTO lineage_backfill_state (id, completed_at, filled)
                 VALUES (1, ?1, ?2)",
                &[self.clock.now().into(), (assigned as i64).into()],
            )
            .await?;
        tracing::info!(assigned, "lineage assigned to nodes that had none");
        Ok(assigned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_forms_never_collide_and_each_field_is_length_prefixed() {
        // Arrange
        let keys = [
            entity_key(Some("work"), "ada"),
            subject_key(Some("work"), "ada", "agent-a"),
            subject_key(Some("work"), "ada", "agent-b"),
            subject_key(None, "ada", "agent-a"),
            // A separator inside a field must not shift the split.
            subject_key(Some("work"), "ada:agent-a", ""),
            subject_key(Some("work:ada"), "agent-a", ""),
            root_key("01HX"),
        ];

        // Act
        let distinct: std::collections::HashSet<_> = keys.iter().collect();

        // Assert
        assert_eq!(distinct.len(), keys.len());
        assert_eq!(keys[0].as_str(), "e:4:work:3:ada");
        assert_eq!(keys[3].as_str(), "s:!:3:ada:7:agent-a");
    }

    #[test]
    fn a_row_is_keyed_by_what_makes_it_a_version_of_another() {
        // Arrange
        let row = |subject, derivation| KeyedRow {
            id: "n1",
            scope: Some("work"),
            subject,
            producer: "agent-a",
            derivation,
        };

        // Act
        let page = key_of_row(&row(Some("ada"), Derivation::Inferred));
        let fact = key_of_row(&row(Some("ada"), Derivation::Recorded));
        let extracted = key_of_row(&row(Some("ada"), Derivation::Extracted));
        let loose = key_of_row(&row(None, Derivation::Recorded));

        // Assert
        assert_eq!(page, entity_key(Some("work"), "ada"));
        assert_eq!(fact, subject_key(Some("work"), "ada", "agent-a"));
        assert_eq!(extracted, fact);
        assert_eq!(loose, root_key("n1"));
    }
}
