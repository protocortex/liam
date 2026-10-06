// SPDX-License-Identifier: Apache-2.0
//! The statements a write projects into the store. A live write and a replay of
//! its log record build them from the same payload through `steps_for`, so the
//! two cannot drift apart.

use liam_log::event::{EdgeRow, LogPayload, NodeRow, RowEffect, TombstoneTarget};

use super::logged_plan::Table;
use super::{
    edge_guard_flags, edge_refusal, node_row_insert, EDGE_INSERT_SQL, EDGE_REFUSAL_DIAGNOSTIC_SQL,
};
use crate::backend::BackendTx;
use crate::error::{Error, Result};
use crate::ids::{NodeId, FOREVER};
use crate::types::relation;
use crate::value::Value;

/// One statement of a projection, in the order it runs.
pub(super) enum Step {
    /// Ends the transaction time of a node that is still open.
    Close {
        id: String,
        tx_to: i64,
    },
    Node(NodeRow),
    Edge(EdgeRow),
    /// An edge written only while both ends are live and no live twin exists.
    GuardedEdge(EdgeRow),
    /// Deletes a row and what depends on it. Removing a row that is already
    /// gone changes nothing, so a replay can run it again.
    Remove(TombstoneTarget),
}

/// The statements that project a logged payload. A `supersedes` edge closes its
/// `dst` at the edge's `tx_from`, so a payload never carries the close itself.
pub(super) fn steps_for(payload: &LogPayload) -> Vec<Step> {
    match payload {
        LogPayload::NodeWrite(row) => vec![Step::Node(row.clone())],
        LogPayload::EdgeWrite(row) => edge_steps(row),
        LogPayload::EpisodeBatch(effects) => effects
            .iter()
            .flat_map(|effect| match effect {
                RowEffect::Node(row) => vec![Step::Node(row.clone())],
                RowEffect::Edge(row) => edge_steps(row),
            })
            .collect(),
        LogPayload::Tombstone(targets) => targets.iter().cloned().map(Step::Remove).collect(),
        LogPayload::DuplicateOf { .. } | LogPayload::Voided { .. } => Vec::new(),
    }
}

// `LoggedRows` in rebuild.rs repeats this supersede rule on purpose, to check a
// replay independently of it, so the two must change together.
fn edge_steps(row: &EdgeRow) -> Vec<Step> {
    if row.edge_type == relation::SUPERSEDES {
        let close = Step::Close {
            id: row.dst.clone(),
            tx_to: row.tx_from,
        };
        return vec![close, Step::Edge(row.clone())];
    }
    vec![Step::GuardedEdge(row.clone())]
}

/// Applies a projection's statements in order. A guarded edge that is refused
/// fails the whole projection with the reason, and so does a close that finds
/// no open node, so the write is voided as a unit rather than half applied.
/// `vector_delete` is the backend's statement for removing a node's vector, as
/// `Backend::vector_delete_sql` returns it.
pub(super) async fn apply_steps(
    tx: &mut dyn BackendTx,
    steps: &[Step],
    vector_delete: Option<&str>,
) -> Result<()> {
    for step in steps {
        match step {
            Step::Close { id, tx_to } => {
                let closed = tx
                    .execute(
                        "UPDATE nodes SET tx_to = ?1 WHERE id = ?2 AND tx_to = ?3",
                        &[(*tx_to).into(), id.as_str().into(), FOREVER.into()],
                    )
                    .await?;
                if closed != 1 {
                    return Err(Error::NodeNotFound(id.clone()));
                }
            }
            Step::Node(row) => {
                let (sql, params) = node_row_insert(row);
                tx.execute(&sql, &params).await?;
            }
            Step::Edge(row) => {
                tx.execute(EDGE_ROW_INSERT_SQL, &edge_row_params(row))
                    .await?;
            }
            Step::GuardedEdge(row) => insert_guarded_edge(tx, row).await?,
            Step::Remove(target) => remove_row(tx, target, vector_delete).await?,
        }
    }
    Ok(())
}

/// Deletes a row and what depends on it. The vector table has no cascade, so a
/// node's vector is deleted before the node. Its edges and community rows
/// cascade where foreign keys are enforced; `Table::removal_sql` deletes them
/// explicitly for a backend that does not enforce them, as `gc` does.
async fn remove_row(
    tx: &mut dyn BackendTx,
    target: &TombstoneTarget,
    vector_delete: Option<&str>,
) -> Result<()> {
    let table = Table::from(target.table);
    let id = [target.id.as_str().into()];
    if let (Table::Nodes, Some(sql)) = (table, vector_delete) {
        tx.execute(sql, &id).await?;
    }
    for sql in table.removal_sql() {
        tx.execute(sql, &id).await?;
    }
    Ok(())
}

const EDGE_ROW_INSERT_SQL: &str =
    "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)";

fn edge_row_params(row: &EdgeRow) -> Vec<Value> {
    vec![
        row.id.as_str().into(),
        row.src.as_str().into(),
        row.dst.as_str().into(),
        row.edge_type.as_str().into(),
        row.attributes.as_str().into(),
        row.tx_from.into(),
        row.tx_to.into(),
    ]
}

/// The three flags of `src -> dst` as `tx` sees them: source live, target live,
/// edge of `kind` already exists.
pub(super) async fn edge_guards(
    tx: &mut dyn BackendTx,
    src: &str,
    dst: &str,
    kind: &str,
) -> Result<[bool; 3]> {
    let rows = tx
        .query(
            EDGE_REFUSAL_DIAGNOSTIC_SQL,
            &[src.into(), dst.into(), FOREVER.into(), kind.into()],
        )
        .await?;
    match rows.first() {
        Some(guards) => edge_guard_flags(guards),
        None => Err(Error::RelateRefused("no row explains the refusal".into())),
    }
}

async fn insert_guarded_edge(tx: &mut dyn BackendTx, row: &EdgeRow) -> Result<()> {
    let mut params = edge_row_params(row);
    params.push(FOREVER.into());
    if tx.execute(EDGE_INSERT_SQL, &params).await? == 1 {
        return Ok(());
    }
    let [source_live, target_live, twin_live] =
        edge_guards(tx, &row.src, &row.dst, &row.edge_type).await?;
    Err(edge_refusal(
        source_live,
        target_live,
        twin_live,
        &NodeId::from_raw(&row.src),
        &NodeId::from_raw(&row.dst),
        &row.edge_type,
    ))
}
