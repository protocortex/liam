// SPDX-License-Identifier: Apache-2.0
//! The statements a write projects into the store. A live write and a replay of
//! its log record build them from the same payload through `steps_for`, so the
//! two cannot drift apart.

use std::iter;

use liam_log::event::{
    DerivedArtifact, EdgeRow, LogEvent, LogPayload, NodeRow, RowEffect, TombstoneTarget,
};

use super::backfill::BACKFILL_SOURCE;
use super::fact_projection::{apply_fact_step, fact_steps, FactStep};
use super::logged_plan::Table;
use super::summaries::{apply_artifact, close_if_open, store_embedding};
use super::tiers::{
    apply_feedback_in, apply_tier_in, apply_usage_in, assign_lineage, insert_quant, replace_quant,
    Quantized,
};
use super::{
    edge_guard_flags, edge_refusal, node_row_insert, EDGE_INSERT_SQL, EDGE_REFUSAL_DIAGNOSTIC_SQL,
};
use crate::backend::BackendTx;
use crate::error::{Error, Result};
use crate::ids::Millis;
use crate::ids::{NodeId, FOREVER};
use crate::types::{
    relation, EventRef, EventRole, FeedbackEvent, LineageKey, TierMove, UsageEntry,
};
use crate::value::Value;

/// One statement of a projection, in the order it runs.
pub(super) enum Step {
    /// Ends the transaction time of a node that is still open.
    Close {
        id: String,
        tx_to: i64,
    },
    Node(NodeRow),
    /// Lists a log event beside a node. Listing it twice changes nothing, so a
    /// replay can run it again.
    NodeEvent {
        node_id: String,
        event: EventRef,
        role: EventRole,
    },
    Edge(EdgeRow),
    /// An edge written only while both ends are live and no live twin exists.
    GuardedEdge(EdgeRow),
    /// Deletes a row and what depends on it. Removing a row that is already
    /// gone changes nothing, so a replay can run it again.
    Remove(TombstoneTarget),
    Fact(FactStep),
    /// Records the lineage of a node written earlier in the same projection.
    /// It follows the node and its `supersedes` edge, which it reads.
    Lineage {
        node_id: String,
    },
    /// Stores the forms of a node's embedding. A live write adds it with the
    /// node; the log does not carry embeddings, so `steps_for` never yields it
    /// and a replay leaves the node to the re-embed pass.
    Quant {
        node_id: String,
        quantized: Quantized,
    },
    /// Replaces the forms of a node's embedding, as a promotion does when it
    /// brings the vector back. Like `Quant`, only a live write adds it.
    Requant {
        node_id: String,
        quantized: Quantized,
    },
    /// Records one logged usage batch, which a replay can apply again.
    Usage {
        event_id: String,
        day: i64,
        entries: Vec<UsageEntry>,
    },
    /// Records one logged verdict, which a replay can apply again.
    Feedback(FeedbackEvent),
    /// Sets the tier of a lineage. The last move applied wins.
    Tier {
        event_id: String,
        change: TierMove,
    },
    /// Stores a derived artifact with its children and source events.
    Summary(Box<DerivedArtifact>),
    /// Ends an open summary, if it is still open.
    SummaryClose {
        id: String,
        at: i64,
    },
    /// Stores a summary's vector. A live write adds it after `Summary`; the log
    /// does not carry vectors, so `steps_for` never yields it and a replay
    /// leaves the summary to the re-embed pass.
    SummaryEmbedding {
        id: String,
        embedding: Vec<u8>,
        dims: usize,
    },
}

/// The statements that project a logged payload. A `supersedes` edge closes its
/// `dst` at the edge's `tx_from`, so a payload never carries the close itself.
pub(super) fn steps_for(payload: &LogPayload) -> Vec<Step> {
    match payload {
        LogPayload::NodeWrite(row) => with_lineage(node_steps(row)),
        LogPayload::EdgeWrite(row) => edge_steps(row),
        LogPayload::EpisodeBatch(effects) => with_lineage(
            effects
                .iter()
                .flat_map(|effect| match effect {
                    RowEffect::Node(row) => node_steps(row),
                    RowEffect::Edge(row) => edge_steps(row),
                })
                .collect(),
        ),
        LogPayload::Tombstone(targets) => targets.iter().cloned().map(Step::Remove).collect(),
        LogPayload::FactWrite { .. } => fact_steps(payload).into_iter().map(Step::Fact).collect(),
        LogPayload::DerivedArtifact(artifact) => vec![Step::Summary(Box::new(artifact.clone()))],
        LogPayload::DerivedArtifactClosed { artifact_id, at } => vec![Step::SummaryClose {
            id: artifact_id.clone(),
            at: *at,
        }],
        LogPayload::DuplicateOf { .. }
        | LogPayload::Voided { .. }
        | LogPayload::RunRolledBack { .. }
        // These payloads are applied as the event that carries them, because
        // their rows are keyed by its id; see `state_steps`.
        | LogPayload::AccessBatch { .. }
        | LogPayload::Feedback { .. }
        | LogPayload::TierMove { .. } => Vec::new(),
    }
}

/// The statements of a usage batch, a verdict or a tier move. Each row they
/// write is keyed by the id of the event, so a replay that meets the event again
/// writes nothing twice.
pub(super) fn state_steps(event: &LogEvent) -> Vec<Step> {
    let event_id = event.event_id.clone();
    match &event.payload {
        LogPayload::AccessBatch { day, entries } => vec![Step::Usage {
            event_id,
            day: *day,
            entries: entries
                .iter()
                .map(|row| UsageEntry {
                    lineage_key: LineageKey::from_raw(row.lineage_key.as_str()),
                    hits: row.hits,
                    last_access_at: Millis(row.last_access_at),
                })
                .collect(),
        }],
        LogPayload::Feedback {
            lineage_key,
            correct,
            producer,
            at,
        } => vec![Step::Feedback(FeedbackEvent {
            event_id,
            lineage_key: LineageKey::from_raw(lineage_key.as_str()),
            producer: producer.clone(),
            at: Millis(*at),
            correct: *correct,
        })],
        LogPayload::TierMove {
            lineage_key,
            to,
            run_id,
            reason,
            changed_at,
            residency_until,
        } => vec![Step::Tier {
            event_id,
            change: TierMove {
                lineage_key: LineageKey::from_raw(lineage_key.as_str()),
                to: (*to).into(),
                run_id: run_id.clone(),
                reason: reason.clone(),
                changed_at: Millis(*changed_at),
                residency_until: residency_until.map(Millis),
            },
        }],
        _ => Vec::new(),
    }
}

/// `steps` followed by the lineage of each node they write. A lineage reads the
/// `supersedes` edge of its node, so it runs after every edge of the batch.
fn with_lineage(mut steps: Vec<Step>) -> Vec<Step> {
    let lineages: Vec<Step> = steps
        .iter()
        .filter_map(|step| match step {
            Step::Node(row) => Some(Step::Lineage {
                node_id: row.id.clone(),
            }),
            _ => None,
        })
        .collect();
    steps.extend(lineages);
    steps
}

/// A node row and the basis events it lists, which a replay restores with it.
fn node_steps(row: &NodeRow) -> Vec<Step> {
    let basis = row.basis.iter().map(|event| Step::NodeEvent {
        node_id: row.id.clone(),
        event: EventRef {
            event_id: event.event_id.clone(),
            synthetic: event.synthetic,
        },
        role: EventRole::Basis,
    });
    iter::once(Step::Node(row.clone())).chain(basis).collect()
}

/// The origin of every node `steps` writes: `event`, which is synthetic when
/// the backfill logged it for a row that predates the log.
pub(super) fn origin_steps(event: &LogEvent, steps: &[Step]) -> Vec<Step> {
    let synthetic = event.source == BACKFILL_SOURCE;
    steps
        .iter()
        .filter_map(|step| match step {
            Step::Node(row) => Some(Step::NodeEvent {
                node_id: row.id.clone(),
                event: EventRef {
                    event_id: event.event_id.clone(),
                    synthetic,
                },
                role: EventRole::Origin,
            }),
            _ => None,
        })
        .collect()
}

/// `steps` for a write that no log records, where an origin event id names
/// nothing a log could produce again. A basis event keeps the flag it came with.
pub(super) fn unlogged(steps: Vec<Step>) -> Vec<Step> {
    steps
        .into_iter()
        .map(|step| match step {
            Step::NodeEvent {
                node_id,
                mut event,
                role: EventRole::Origin,
            } => {
                event.synthetic = true;
                Step::NodeEvent {
                    node_id,
                    event,
                    role: EventRole::Origin,
                }
            }
            other => other,
        })
        .collect()
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
                let (sql, params) = node_row_insert(row)?;
                tx.execute(&sql, &params).await?;
            }
            Step::NodeEvent {
                node_id,
                event,
                role,
            } => {
                tx.execute(
                    NODE_EVENT_INSERT_SQL,
                    &[
                        node_id.as_str().into(),
                        event.event_id.as_str().into(),
                        role.as_str().into(),
                        i64::from(event.synthetic).into(),
                    ],
                )
                .await?;
            }
            Step::Edge(row) => {
                tx.execute(EDGE_ROW_INSERT_SQL, &edge_row_params(row))
                    .await?;
            }
            Step::GuardedEdge(row) => insert_guarded_edge(tx, row).await?,
            Step::Remove(target) => remove_row(tx, target, vector_delete).await?,
            Step::Lineage { node_id } => assign_lineage(tx, node_id).await?,
            Step::Quant { node_id, quantized } => insert_quant(tx, node_id, quantized).await?,
            Step::Requant { node_id, quantized } => replace_quant(tx, node_id, quantized).await?,
            Step::Usage {
                event_id,
                day,
                entries,
            } => apply_usage_in(tx, event_id, *day, entries).await?,
            Step::Feedback(event) => drop(apply_feedback_in(tx, event).await?),
            Step::Tier { event_id, change } => apply_tier_in(tx, change, Some(event_id)).await?,
            Step::Fact(step) => apply_fact_step(tx, step).await?,
            Step::Summary(artifact) => apply_artifact(tx, artifact).await?,
            Step::SummaryClose { id, at } => close_if_open(tx, id, *at).await?,
            Step::SummaryEmbedding {
                id,
                embedding,
                dims,
            } => store_embedding(tx, id, *dims, embedding).await?,
        }
    }
    Ok(())
}

/// Deletes a row and what depends on it. The vector table has no cascade, so a
/// node's vector is deleted before the node. Its edges and community rows
/// cascade where foreign keys are enforced; `Table::removal_sql` deletes them
/// explicitly for a backend that does not enforce them.
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

// Skips a node that is not there instead of failing on the foreign key: a row a
// replay lost is for the rebuild check to report against the log.
const NODE_EVENT_INSERT_SQL: &str =
    "INSERT OR IGNORE INTO node_events (node_id, event_id, role, synthetic)
     SELECT ?1, ?2, ?3, ?4 WHERE EXISTS (SELECT 1 FROM nodes WHERE id = ?1)";

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
