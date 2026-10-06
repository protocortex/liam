// SPDX-License-Identifier: Apache-2.0
//! What each logged write records and applies. A `Plan` pairs the log event
//! with the statements that project it, built inside the write transaction so
//! the reads it depends on, such as the live competitor of a subject, cannot go
//! stale before the append.
//!
//! A plan only reads from the transaction. Every statement that changes the
//! store runs after the append, so a failure while applying is always voided.

use liam_log::event::{EdgeRow, LogEvent, LogPayload, NodeRow, RowEffect, CURRENT_SCHEMA_VERSION};
use liam_log::hash::{edge_row_hash, node_row_hash};
use uuid::Uuid;

use super::logged_write::{Logged, SharedLog};
use super::{
    collision_producer, edge_guard_flags, edge_refusal, edge_refusal_from_rows, live_as_of_query,
    live_by_subject_query, node_row_insert, resolve_node_row, Graph, EDGE_INSERT_SQL,
    EDGE_REFUSAL_DIAGNOSTIC_SQL,
};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{EdgeId, Millis, NodeId, FOREVER};
use crate::types::{relation, EpisodeEdge, EpisodeRef, NewNode};
use crate::value::Value;

/// Source and trust of a record no producer wrote, such as an edge.
const STORE_SOURCE: &str = "liam-store";
const STORE_TRUST: f64 = 1.0;

/// The table whose rows a content hash can point at.
#[derive(Clone, Copy)]
pub(super) enum Table {
    Nodes,
    Edges,
}

impl Table {
    pub(super) fn name(self) -> &'static str {
        match self {
            Table::Nodes => "nodes",
            Table::Edges => "edges",
        }
    }
}

/// One statement of a projection, in the order it runs.
pub(super) enum Step {
    /// Ends a live node's transaction time at `tx_to`.
    Close {
        id: String,
        tx_to: i64,
    },
    Node(NodeRow),
    Edge(EdgeRow),
    /// An edge written only while both ends are live and no live twin exists.
    GuardedEdge(EdgeRow),
}

/// A write ready to log: its event, how to apply it, and how to deduplicate it.
pub(super) struct Plan {
    pub(super) event: LogEvent,
    pub(super) steps: Vec<Step>,
    /// Where a live carrier of the event's content hash makes the write a
    /// duplicate. `None` writes unconditionally.
    pub(super) dedup: Option<Table>,
    /// Returned instead of writing when dedup finds no live carrier.
    pub(super) refusal_unless_duplicate: Option<Error>,
}

impl Plan {
    fn new(event: LogEvent, steps: Vec<Step>) -> Self {
        Self {
            event,
            steps,
            dedup: None,
            refusal_unless_duplicate: None,
        }
    }

    fn deduplicated(mut self, table: Table) -> Self {
        self.dedup = Some(table);
        self
    }

    pub(super) fn node_write(row: NodeRow, now: Millis) -> Self {
        let event = node_event(&row, now, LogPayload::NodeWrite(row.clone()));
        Self::new(event, vec![Step::Node(row)]).deduplicated(Table::Nodes)
    }

    /// Closes `old`, writes `row`, and links them. The batch carries the
    /// `supersedes` edge so the log alone says which row the new one replaced.
    ///
    /// Replay closes `dst` of a `supersedes` edge at the edge's `tx_from`, which
    /// is how the closed row is recovered without a payload of its own.
    pub(super) fn supersede(row: NodeRow, old: String, now: Millis) -> Self {
        let edge = supersedes_edge(&row.id, &old, now);
        let payload = LogPayload::EpisodeBatch(vec![
            RowEffect::Node(row.clone()),
            RowEffect::Edge(edge.clone()),
        ]);
        let event = node_event(&row, now, payload);
        let close = Step::Close {
            id: old,
            tx_to: now.0,
        };
        Self::new(event, vec![close, Step::Node(row), Step::Edge(edge)])
    }

    fn edge_write(row: EdgeRow, now: Millis) -> Self {
        let hash = edge_row_hash(&row);
        let event = event(
            hash,
            (STORE_SOURCE, STORE_TRUST),
            now.0,
            now,
            LogPayload::EdgeWrite(row.clone()),
        );
        Self::new(event, vec![Step::GuardedEdge(row)]).deduplicated(Table::Edges)
    }

    /// One event for every row `steps` writes. It is stamped by the first node,
    /// or by the store when the episode holds only edges, and is observed when
    /// it is ingested.
    fn episode(steps: Vec<Step>, now: Millis) -> Self {
        let stamp = steps
            .iter()
            .find_map(|step| match step {
                Step::Node(row) => Some((row.producer.as_str(), row.confidence)),
                _ => None,
            })
            .unwrap_or((STORE_SOURCE, STORE_TRUST));
        let payload = LogPayload::EpisodeBatch(row_effects(&steps));
        let event = event(first_hash(&steps), stamp, now.0, now, payload);
        Self::new(event, steps)
    }
}

/// The rows a projection writes, as the log carries them.
fn row_effects(steps: &[Step]) -> Vec<RowEffect> {
    steps
        .iter()
        .filter_map(|step| match step {
            Step::Node(row) => Some(RowEffect::Node(row.clone())),
            Step::Edge(row) | Step::GuardedEdge(row) => Some(RowEffect::Edge(row.clone())),
            Step::Close { .. } => None,
        })
        .collect()
}

fn first_hash(steps: &[Step]) -> [u8; 32] {
    steps
        .iter()
        .find_map(|step| match step {
            Step::Node(row) => Some(node_row_hash(row)),
            Step::Edge(row) | Step::GuardedEdge(row) => Some(edge_row_hash(row)),
            Step::Close { .. } => None,
        })
        .unwrap_or_default()
}

/// A record for a node row: the row's producer is the source, its confidence
/// the trust, and its valid time the observation time.
fn node_event(row: &NodeRow, now: Millis, payload: LogPayload) -> LogEvent {
    let stamp = (row.producer.as_str(), row.confidence);
    event(node_row_hash(row), stamp, row.valid_from, now, payload)
}

fn event(
    content_hash: [u8; 32],
    (source, trust_score): (&str, f64),
    observed_at: i64,
    now: Millis,
    payload: LogPayload,
) -> LogEvent {
    LogEvent {
        event_id: Uuid::now_v7().to_string(),
        content_hash,
        source: source.to_string(),
        trust_score,
        observed_at,
        ingested_at: now.0,
        encryption_key_id: None,
        schema_version: CURRENT_SCHEMA_VERSION,
        payload,
    }
}

fn supersedes_edge(new: &str, old: &str, now: Millis) -> EdgeRow {
    EdgeRow {
        id: EdgeId::new().as_str().to_string(),
        src: new.to_string(),
        dst: old.to_string(),
        edge_type: relation::SUPERSEDES.to_string(),
        attributes: "{}".to_string(),
        tx_from: now.0,
        tx_to: FOREVER.0,
    }
}

/// What makes a live node the one a new node replaces: the same subject and
/// scope, and the same producer for a fact or empty content for an entity page.
pub(super) struct Collision {
    subject: String,
    scope: Option<String>,
    producer: Option<String>,
}

impl Collision {
    /// `None` for a node without a subject, which collides with nothing.
    pub(super) fn of(node: &NewNode) -> Option<Self> {
        Some(Self {
            subject: node.subject.clone()?,
            scope: node.scope.clone(),
            producer: collision_producer(node).map(str::to_string),
        })
    }

    fn matches(&self, row: &NodeRow) -> bool {
        row.subject.as_deref() == Some(self.subject.as_str())
            && row.scope == self.scope
            && match &self.producer {
                Some(producer) => row.producer == *producer,
                None => row.content.is_empty(),
            }
    }

    /// The newest live row this collides with, as the transaction sees it.
    async fn find_live(&self, tx: &mut dyn BackendTx, now: Millis) -> Result<Option<String>> {
        let (sql, params) = live_by_subject_query(
            &self.subject,
            now,
            self.scope.as_deref(),
            self.producer.as_deref(),
        );
        let rows = tx.query(&sql, &params).await?;
        rows.first().map(|row| row.get_string(0)).transpose()
    }
}

fn is_live_at(row: &NodeRow, now: Millis) -> bool {
    row.tx_from <= now.0 && row.tx_to > now.0 && row.valid_from <= now.0 && row.valid_until > now.0
}

/// A logged `upsert_by` over a subject: a plain write when nothing live
/// collides, a supersede otherwise, and a duplicate when the content is
/// already live.
async fn plan_upsert(
    tx: &mut dyn BackendTx,
    row: NodeRow,
    collision: Collision,
    now: Millis,
) -> Result<Plan> {
    let plan = match collision.find_live(tx, now).await? {
        Some(old) => Plan::supersede(row, old, now).deduplicated(Table::Nodes),
        None => Plan::node_write(row, now),
    };
    Ok(plan)
}

/// A logged `supersede`. The caller asked for a state change, so it is never
/// deduplicated, only refused when `old` is not live.
async fn plan_supersede(
    tx: &mut dyn BackendTx,
    row: NodeRow,
    old: String,
    now: Millis,
) -> Result<Plan> {
    let (sql, params) = live_as_of_query(&old, now);
    if tx.query(&sql, &params).await?.is_empty() {
        return Err(Error::NodeNotFound(old));
    }
    Ok(Plan::supersede(row, old, now))
}

/// A logged `relate`. A dead endpoint is refused before anything is logged. A
/// live twin that the log never recorded is refused too, since only an indexed
/// twin can be answered as a duplicate.
async fn plan_relate(tx: &mut dyn BackendTx, row: EdgeRow, now: Millis) -> Result<Plan> {
    let rows = tx
        .query(
            EDGE_REFUSAL_DIAGNOSTIC_SQL,
            &[
                row.src.as_str().into(),
                row.dst.as_str().into(),
                FOREVER.into(),
                row.edge_type.as_str().into(),
            ],
        )
        .await?;
    let [source_live, target_live, twin_live] = match rows.first() {
        Some(guards) => edge_guard_flags(guards)?,
        None => return Err(Error::RelateRefused("no row explains the refusal".into())),
    };
    let (src, dst) = (NodeId::from_raw(&row.src), NodeId::from_raw(&row.dst));
    let refusal = edge_refusal(
        source_live,
        target_live,
        twin_live,
        &src,
        &dst,
        &row.edge_type,
    );
    if !source_live || !target_live {
        return Err(refusal);
    }
    let mut plan = Plan::edge_write(row, now);
    if twin_live {
        plan.refusal_unless_duplicate = Some(refusal);
    }
    Ok(plan)
}

/// A node of an episode and what it may replace.
pub(super) struct EpisodeNode {
    row: NodeRow,
    collision: Option<Collision>,
}

/// A logged `ingest_episode`. Nothing is written yet, so a node that replaces
/// an earlier node of the same episode is found among the staged steps, and the
/// replaced row is staged already closed.
async fn plan_episode(
    tx: &mut dyn BackendTx,
    nodes: Vec<EpisodeNode>,
    edges: Vec<EdgeRow>,
    now: Millis,
) -> Result<Plan> {
    let mut steps = Vec::new();
    for node in nodes {
        stage_node(tx, &mut steps, node, now).await?;
    }
    steps.extend(edges.into_iter().map(Step::GuardedEdge));
    Ok(Plan::episode(steps, now))
}

async fn stage_node(
    tx: &mut dyn BackendTx,
    steps: &mut Vec<Step>,
    node: EpisodeNode,
    now: Millis,
) -> Result<()> {
    let superseded = match &node.collision {
        Some(collision) => close_competitor(tx, steps, collision, now).await?,
        None => None,
    };
    let new_id = node.row.id.clone();
    steps.push(Step::Node(node.row));
    if let Some(old) = superseded {
        steps.push(Step::Edge(supersedes_edge(&new_id, &old, now)));
    }
    Ok(())
}

/// Closes the live row `collision` names, whether the store holds it or an
/// earlier node of this episode staged it, and returns its id.
async fn close_competitor(
    tx: &mut dyn BackendTx,
    steps: &mut Vec<Step>,
    collision: &Collision,
    now: Millis,
) -> Result<Option<String>> {
    if let Some(staged) = steps.iter_mut().rev().find_map(|step| match step {
        Step::Node(row) if collision.matches(row) && is_live_at(row, now) => Some(row),
        _ => None,
    }) {
        staged.tx_to = now.0;
        return Ok(Some(staged.id.clone()));
    }
    let stored = collision
        .find_live(tx, now)
        .await?
        .filter(|id| !is_closed(steps, id));
    if let Some(id) = &stored {
        steps.push(Step::Close {
            id: id.clone(),
            tx_to: now.0,
        });
    }
    Ok(stored)
}

fn is_closed(steps: &[Step], node_id: &str) -> bool {
    steps
        .iter()
        .any(|step| matches!(step, Step::Close { id, .. } if id == node_id))
}

impl<B: Backend> Graph<B> {
    /// Logs and projects one node. The vector write stays with the caller
    /// because it takes the write lock the transaction holds.
    pub(super) async fn insert_logged(
        &self,
        log: &SharedLog,
        id: &NodeId,
        node: &NewNode,
        now: Millis,
    ) -> Result<Logged> {
        let row = resolve_node_row(id, node, now)?;
        self.transact(log, move |_| {
            Box::pin(async move { Ok(Plan::node_write(row, now)) })
        })
        .await
    }

    pub(super) async fn upsert_logged(
        &self,
        log: &SharedLog,
        node: &NewNode,
        collision: Collision,
    ) -> Result<NodeId> {
        let (id, now) = (NodeId::new(), self.clock.now());
        let row = resolve_node_row(&id, node, now)?;
        let logged = self
            .transact(log, move |tx| {
                Box::pin(plan_upsert(tx, row, collision, now))
            })
            .await?;
        Ok(logged.node_id(id))
    }

    pub(super) async fn supersede_logged(
        &self,
        log: &SharedLog,
        old: &NodeId,
        node: &NewNode,
        now: Millis,
    ) -> Result<NodeId> {
        let id = NodeId::new();
        let row = resolve_node_row(&id, node, now)?;
        let old = old.as_str().to_string();
        let logged = self
            .transact(log, move |tx| Box::pin(plan_supersede(tx, row, old, now)))
            .await?;
        Ok(logged.node_id(id))
    }

    pub(super) async fn relate_logged(
        &self,
        log: &SharedLog,
        row: EdgeRow,
        now: Millis,
    ) -> Result<EdgeId> {
        let id = EdgeId::from_raw(&row.id);
        let logged = self
            .transact(log, move |tx| Box::pin(plan_relate(tx, row, now)))
            .await?;
        Ok(logged.edge_id(id))
    }

    /// Logs and projects one episode as a single event. The edge ids come back
    /// in the order of `edges`.
    pub(super) async fn episode_logged(
        &self,
        log: &SharedLog,
        ids: &[NodeId],
        nodes: &[NewNode],
        edges: &[EpisodeEdge],
        now: Millis,
    ) -> Result<Vec<EdgeId>> {
        let staged = ids
            .iter()
            .zip(nodes)
            .map(|(id, node)| {
                Ok(EpisodeNode {
                    row: resolve_node_row(id, node, now)?,
                    collision: Collision::of(node),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = edges
            .iter()
            .map(|edge| episode_edge_row(edge, ids, now))
            .collect::<Result<Vec<_>>>()?;
        let edge_ids = rows.iter().map(|row| EdgeId::from_raw(&row.id)).collect();
        if staged.is_empty() && rows.is_empty() {
            return Ok(edge_ids);
        }
        self.transact(log, move |tx| Box::pin(plan_episode(tx, staged, rows, now)))
            .await?;
        Ok(edge_ids)
    }
}

fn episode_edge_row(edge: &EpisodeEdge, ids: &[NodeId], now: Millis) -> Result<EdgeRow> {
    let endpoint = |r: &EpisodeRef| match r {
        EpisodeRef::New(i) => ids[*i].as_str().to_string(),
        EpisodeRef::Existing(id) => id.as_str().to_string(),
    };
    Ok(EdgeRow {
        id: EdgeId::new().as_str().to_string(),
        src: endpoint(&edge.from),
        dst: endpoint(&edge.to),
        edge_type: edge.kind.clone(),
        attributes: serde_json::to_string(&edge.attributes)?,
        tx_from: now.0,
        tx_to: FOREVER.0,
    })
}

/// Applies a plan's statements in order. A guarded edge that is refused fails
/// the whole projection with the reason, so the write is voided as a unit.
pub(super) async fn apply_steps(tx: &mut dyn BackendTx, steps: &[Step]) -> Result<()> {
    for step in steps {
        match step {
            Step::Close { id, tx_to } => {
                tx.execute(
                    "UPDATE nodes SET tx_to = ?1 WHERE id = ?2 AND tx_to = ?3",
                    &[(*tx_to).into(), id.as_str().into(), FOREVER.into()],
                )
                .await?;
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
        }
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

async fn insert_guarded_edge(tx: &mut dyn BackendTx, row: &EdgeRow) -> Result<()> {
    let mut params = edge_row_params(row);
    params.push(FOREVER.into());
    if tx.execute(EDGE_INSERT_SQL, &params).await? == 1 {
        return Ok(());
    }
    let rows = tx
        .query(
            EDGE_REFUSAL_DIAGNOSTIC_SQL,
            &[
                row.src.as_str().into(),
                row.dst.as_str().into(),
                FOREVER.into(),
                row.edge_type.as_str().into(),
            ],
        )
        .await?;
    let (src, dst) = (NodeId::from_raw(&row.src), NodeId::from_raw(&row.dst));
    match edge_refusal_from_rows(rows, &src, &dst, &row.edge_type) {
        Ok(refusal) | Err(refusal) => Err(refusal),
    }
}
