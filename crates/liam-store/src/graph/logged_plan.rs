// SPDX-License-Identifier: Apache-2.0
//! What each write records and applies. A `Plan` pairs the log event with the
//! statements that project it, built inside the write transaction so the reads
//! it depends on, such as the open competitor of a subject, cannot go stale
//! before the append.
//!
//! A plan only reads from the transaction. Every statement that changes the
//! store runs after the append, so a failure while applying is always voided.

use std::collections::BTreeSet;

use liam_log::event::{
    EdgeRow, LogEvent, LogPayload, NodeRow, RowEffect, TombstoneTable, TombstoneTarget,
    CURRENT_SCHEMA_VERSION,
};
use liam_log::hash::{edge_row_hash, node_row_hash};
use uuid::Uuid;

use super::logged_write::{ready, WriteOutcome};
use super::projection::{edge_guards, steps_for, Step};
use super::{
    collision_producer, edge_exists_message, edge_refusal, open_by_subject_query, open_node_query,
    resolve_episode_ref, resolve_node_row, Graph, EMPTY_ATTRIBUTES,
};
use crate::backend::{Backend, BackendTx};
use crate::error::{Error, Result};
use crate::ids::{EdgeId, Millis, NodeId, FOREVER};
use crate::types::{relation, EpisodeEdge, NewNode};
use crate::value::Value;

/// Source and trust of a record no producer wrote, such as an edge.
const STORE_SOURCE: &str = "liam-store";
const STORE_TRUST: f64 = 1.0;

/// A table whose rows a content hash or a tombstone can point at.
#[derive(Clone, Copy)]
pub(super) enum Table {
    Nodes,
    Edges,
    NodeCommunity,
}

impl From<TombstoneTable> for Table {
    fn from(table: TombstoneTable) -> Self {
        match table {
            TombstoneTable::Nodes => Table::Nodes,
            TombstoneTable::Edges => Table::Edges,
            TombstoneTable::NodeCommunity => Table::NodeCommunity,
        }
    }
}

/// The edges a removed node takes with it, with `?1` the node's id. A macro so
/// `removal_sql` can concatenate it into a literal; `gc` counts a chunk's edges
/// with the same clause.
macro_rules! edges_of_node {
    () => {
        "src = ?1 OR dst = ?1"
    };
}
pub(super) use edges_of_node;

impl Table {
    /// A query that returns a row when `row_id` is in this table, live or closed.
    pub(super) fn exists_query(self) -> &'static str {
        match self {
            Table::Nodes => "SELECT 1 FROM nodes WHERE id = ?1",
            Table::Edges => "SELECT 1 FROM edges WHERE id = ?1",
            Table::NodeCommunity => "SELECT 1 FROM node_community WHERE node_id = ?1",
        }
    }

    /// What removing a row deletes, dependents first so the foreign keys hold.
    /// A node's vector is the backend's to delete, before these run.
    ///
    /// `LoggedRows` in rebuild.rs repeats the node cascade on purpose, to check
    /// a replay independently of it, so the two must change together.
    pub(super) fn removal_sql(self) -> &'static [&'static str] {
        match self {
            // Cascades on a backend that enforces foreign keys; the explicit
            // deletes guard one that does not.
            Table::Nodes => &[
                concat!("DELETE FROM edges WHERE ", edges_of_node!()),
                "DELETE FROM node_community WHERE node_id = ?1",
                "DELETE FROM nodes WHERE id = ?1",
            ],
            Table::Edges => &["DELETE FROM edges WHERE id = ?1"],
            Table::NodeCommunity => &["DELETE FROM node_community WHERE node_id = ?1"],
        }
    }

    /// A query that returns a row when `row_id` is live in this table.
    pub(super) fn live_query(self, row_id: &str) -> (&'static str, Vec<Value>) {
        match self {
            Table::Nodes => (
                "SELECT 1 FROM nodes WHERE id = ?1 AND tx_to = ?2",
                vec![row_id.into(), FOREVER.into()],
            ),
            Table::Edges => (
                "SELECT 1 FROM edges WHERE id = ?1 AND tx_to = ?2",
                vec![row_id.into(), FOREVER.into()],
            ),
            // Its rows are deleted rather than closed, so a row that exists is live.
            Table::NodeCommunity => (
                "SELECT 1 FROM node_community WHERE node_id = ?1",
                vec![row_id.into()],
            ),
        }
    }
}

/// How a write is checked against the live carrier of its content.
pub(super) struct Dedup {
    /// Where a live carrier of the event's content hash makes the write a
    /// duplicate.
    pub(super) table: Table,
    /// Returned instead of writing when no live carrier answers the write.
    pub(super) refusal_if_absent: Option<Error>,
}

/// A write ready to log: its event, how to apply it, and how to deduplicate it.
pub(super) struct Plan {
    pub(super) event: LogEvent,
    pub(super) steps: Vec<Step>,
    /// `None` writes unconditionally.
    pub(super) dedup: Option<Dedup>,
}

impl Plan {
    fn new(event: LogEvent) -> Self {
        let steps = steps_for(&event.payload);
        Self {
            event,
            steps,
            dedup: None,
        }
    }

    fn deduplicated(mut self, table: Table, refusal_if_absent: Option<Error>) -> Self {
        self.dedup = Some(Dedup {
            table,
            refusal_if_absent,
        });
        self
    }

    fn node_write(row: NodeRow, now: Millis) -> Self {
        let event = node_event(&row, now, LogPayload::NodeWrite(row.clone()));
        Self::new(event).deduplicated(Table::Nodes, None)
    }

    /// Writes `row` and links it to `old`, which the link closes. The batch
    /// carries the `supersedes` edge so the log alone says which row the new one
    /// replaced.
    fn supersede(mut row: NodeRow, old: &OpenRow, now: Millis) -> Self {
        let edge = succeed(&mut row, old, now);
        let effects = vec![RowEffect::Node(row.clone()), RowEffect::Edge(edge)];
        Self::new(node_event(&row, now, LogPayload::EpisodeBatch(effects)))
    }

    fn edge_write(row: EdgeRow, now: Millis, refusal_if_absent: Option<Error>) -> Self {
        let stamp = (STORE_SOURCE, STORE_TRUST);
        let hash = edge_row_hash(&row);
        let event = event(hash, stamp, now.0, now, LogPayload::EdgeWrite(row));
        Self::new(event).deduplicated(Table::Edges, refusal_if_absent)
    }

    /// One event for every row of `effects`. It is stamped by the first node,
    /// or by the store when the episode holds only edges, and is observed when
    /// it is ingested.
    fn episode(effects: Vec<RowEffect>, now: Millis) -> Self {
        let (source, trust) = effects
            .iter()
            .find_map(|effect| match effect {
                RowEffect::Node(row) => Some((row.producer.clone(), row.confidence)),
                RowEffect::Edge(_) => None,
            })
            .unwrap_or_else(|| (STORE_SOURCE.to_string(), STORE_TRUST));
        let hash = effects.first().map(effect_hash).unwrap_or_default();
        let payload = LogPayload::EpisodeBatch(effects);
        Self::new(event(hash, (&source, trust), now.0, now, payload))
    }
}

fn effect_hash(effect: &RowEffect) -> [u8; 32] {
    match effect {
        RowEffect::Node(row) => node_row_hash(row),
        RowEffect::Edge(row) => edge_row_hash(row),
    }
}

/// A record that removes `targets`. It covers no content, so its hash is empty.
pub(super) fn tombstone_event(targets: Vec<TombstoneTarget>, now: Millis) -> LogEvent {
    let stamp = (STORE_SOURCE, STORE_TRUST);
    event([0; 32], stamp, now.0, now, LogPayload::Tombstone(targets))
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

/// A node that has not been closed, and when its transaction time began.
pub(super) struct OpenRow {
    id: String,
    tx_from: i64,
}

/// Starts `row` as the version after `old` and returns the `supersedes` edge
/// that closes `old`. The versions of one subject must have non-decreasing
/// transaction times, so a clock that stepped back cannot start `row` before
/// the row it closes. The row, the edge, and the close, which replay takes from
/// the edge, share that one time.
fn succeed(row: &mut NodeRow, old: &OpenRow, now: Millis) -> EdgeRow {
    row.tx_from = now.0.max(old.tx_from);
    EdgeRow {
        id: EdgeId::new().as_str().to_string(),
        src: row.id.clone(),
        dst: old.id.clone(),
        edge_type: relation::SUPERSEDES.to_string(),
        attributes: EMPTY_ATTRIBUTES.to_string(),
        tx_from: row.tx_from,
        tx_to: FOREVER.0,
    }
}

/// What makes an open node the one a new node replaces: the same subject and
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

    /// The newest open row this collides with, as the transaction sees it.
    async fn find_open(&self, tx: &mut dyn BackendTx) -> Result<Option<OpenRow>> {
        let (sql, params) = open_by_subject_query(
            &self.subject,
            self.scope.as_deref(),
            self.producer.as_deref(),
        );
        let rows = tx.query(&sql, &params).await?;
        rows.first()
            .map(|row| {
                Ok(OpenRow {
                    id: row.get_string(0)?,
                    tx_from: row.get_i64(1)?,
                })
            })
            .transpose()
    }
}

/// An `upsert_by` over a subject: a plain write when nothing open collides, a
/// supersede otherwise, and a duplicate when the content is already live.
async fn plan_upsert(
    tx: &mut dyn BackendTx,
    row: NodeRow,
    collision: Collision,
    now: Millis,
) -> Result<Plan> {
    let plan = match collision.find_open(tx).await? {
        Some(old) => Plan::supersede(row, &old, now).deduplicated(Table::Nodes, None),
        None => Plan::node_write(row, now),
    };
    Ok(plan)
}

/// A `supersede`. The caller asked for a state change, so it is never
/// deduplicated, only refused when `old` is not open.
async fn plan_supersede(
    tx: &mut dyn BackendTx,
    row: NodeRow,
    old: &NodeId,
    now: Millis,
) -> Result<Plan> {
    let (sql, params) = open_node_query(old.as_str());
    let rows = tx.query(sql, &params).await?;
    let Some(open) = rows.first() else {
        return Err(Error::NodeNotFound(old.as_str().to_string()));
    };
    let old = OpenRow {
        id: old.as_str().to_string(),
        tx_from: open.get_i64(0)?,
    };
    Ok(Plan::supersede(row, &old, now))
}

/// A guarded edge write from `relate` or `link`. A dead endpoint is refused
/// before anything is logged. So is a live twin that carries other attributes,
/// since answering it as a duplicate would drop the caller's. Any other live
/// twin is refused unless the log recorded it, since only an indexed twin can be
/// answered as a duplicate.
async fn plan_edge(tx: &mut dyn BackendTx, row: EdgeRow, now: Millis) -> Result<Plan> {
    let [source_live, target_live, twin_live] =
        edge_guards(tx, &row.src, &row.dst, &row.edge_type).await?;
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
    if twin_live {
        let stored = twin_attributes(tx, &row).await?;
        let differing = differing_attributes(&stored, &row.attributes);
        if !differing.is_empty() {
            let twin = edge_exists_message(&src, &dst, &row.edge_type);
            let keys = differing.join(", ");
            return Err(Error::RelateRefused(format!(
                "{twin} with different attributes: {keys}"
            )));
        }
    }
    Ok(Plan::edge_write(row, now, twin_live.then_some(refusal)))
}

/// The attributes of the live edge that already relates `row`'s endpoints.
async fn twin_attributes(tx: &mut dyn BackendTx, row: &EdgeRow) -> Result<String> {
    let params = [
        row.src.as_str().into(),
        row.dst.as_str().into(),
        row.edge_type.as_str().into(),
        FOREVER.into(),
    ];
    let twins = tx
        .query(
            "SELECT attributes FROM edges
             WHERE src = ?1 AND dst = ?2 AND type = ?3 AND tx_to = ?4 ORDER BY id LIMIT 1",
            &params,
        )
        .await?;
    match twins.first() {
        Some(twin) => twin.get_string(0),
        None => Err(Error::ConcurrentWrite),
    }
}

/// The keys whose values differ between two attribute documents, or a single
/// placeholder when they are not both objects or cannot be read. Names keys
/// only, so a refusal never carries a caller's values.
fn differing_attributes(stored: &str, given: &str) -> Vec<String> {
    use serde_json::Value::Object;

    if stored == given {
        return Vec::new();
    }
    let parse = |text: &str| serde_json::from_str::<serde_json::Value>(text).ok();
    match (parse(stored), parse(given)) {
        (Some(Object(old)), Some(Object(new))) => old
            .keys()
            .chain(new.keys())
            .filter(|key| old.get(*key) != new.get(*key))
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        (Some(old), Some(new)) if old == new => Vec::new(),
        _ => vec!["(whole document)".to_string()],
    }
}

/// A node of an episode and what it may replace.
struct EpisodeNode {
    row: NodeRow,
    collision: Option<Collision>,
}

/// An `ingest_episode`. Nothing is written yet, so a node that replaces an
/// earlier node of the same episode is found among the staged rows.
async fn plan_episode(
    tx: &mut dyn BackendTx,
    nodes: Vec<EpisodeNode>,
    edges: Vec<EdgeRow>,
    now: Millis,
) -> Result<Plan> {
    let mut effects = Vec::new();
    for node in nodes {
        stage_node(tx, &mut effects, node, now).await?;
    }
    effects.extend(edges.into_iter().map(RowEffect::Edge));
    Ok(Plan::episode(effects, now))
}

async fn stage_node(
    tx: &mut dyn BackendTx,
    effects: &mut Vec<RowEffect>,
    node: EpisodeNode,
    now: Millis,
) -> Result<()> {
    let superseded = match &node.collision {
        Some(collision) => find_competitor(tx, effects, collision).await?,
        None => None,
    };
    let mut row = node.row;
    let link = superseded.map(|old| succeed(&mut row, &old, now));
    effects.push(RowEffect::Node(row));
    effects.extend(link.map(RowEffect::Edge));
    Ok(())
}

/// The open row `collision` names, whether the store holds it or an earlier node
/// of this episode staged it. The `supersedes` edge staged for it closes it.
async fn find_competitor(
    tx: &mut dyn BackendTx,
    effects: &[RowEffect],
    collision: &Collision,
) -> Result<Option<OpenRow>> {
    let staged = effects.iter().rev().find_map(|effect| match effect {
        RowEffect::Node(row) if collision.matches(row) && !is_superseded(effects, &row.id) => {
            Some(OpenRow {
                id: row.id.clone(),
                tx_from: row.tx_from,
            })
        }
        _ => None,
    });
    if staged.is_some() {
        return Ok(staged);
    }
    let stored = collision.find_open(tx).await?;
    Ok(stored.filter(|open| !is_superseded(effects, &open.id)))
}

fn is_superseded(effects: &[RowEffect], node_id: &str) -> bool {
    effects.iter().any(|effect| {
        matches!(effect, RowEffect::Edge(edge)
            if edge.edge_type == relation::SUPERSEDES && edge.dst == node_id)
    })
}

impl<B: Backend> Graph<B> {
    /// Projects one node. The vector write stays with the caller because it
    /// takes the write lock the transaction holds.
    pub(super) async fn project_insert(&self, id: &NodeId, node: &NewNode) -> Result<NodeId> {
        let outcome = self
            .project(|_, now| {
                let row = resolve_node_row(id, node, now);
                ready(row.map(|row| Plan::node_write(row, now)))
            })
            .await?;
        Ok(outcome.node_id(id))
    }

    pub(super) async fn project_upsert(
        &self,
        node: &NewNode,
        collision: Collision,
    ) -> Result<NodeId> {
        let id = NodeId::new();
        let outcome = self
            .project(|tx, now| {
                let row = resolve_node_row(&id, node, now);
                Box::pin(async move { plan_upsert(tx, row?, collision, now).await })
            })
            .await?;
        Ok(outcome.node_id(&id))
    }

    pub(super) async fn project_supersede(&self, old: &NodeId, node: &NewNode) -> Result<NodeId> {
        let id = NodeId::new();
        let outcome = self
            .project(|tx, now| {
                let row = resolve_node_row(&id, node, now);
                let old = old.clone();
                Box::pin(async move { plan_supersede(tx, row?, &old, now).await })
            })
            .await?;
        Ok(outcome.node_id(&id))
    }

    /// The edge's id, which is the first edge's when the relation was already
    /// recorded, and what the write did.
    pub(super) async fn project_edge(
        &self,
        src: &NodeId,
        dst: &NodeId,
        kind: &str,
        attributes: &str,
    ) -> Result<(EdgeId, WriteOutcome)> {
        self.refuse_reserved([kind])?;
        let id = EdgeId::new();
        let outcome = self
            .project(|tx, now| {
                let row = EdgeRow {
                    id: id.as_str().to_string(),
                    src: src.as_str().to_string(),
                    dst: dst.as_str().to_string(),
                    edge_type: kind.to_string(),
                    attributes: attributes.to_string(),
                    tx_from: now.0,
                    tx_to: FOREVER.0,
                };
                Box::pin(plan_edge(tx, row, now))
            })
            .await?;
        Ok((outcome.edge_id(&id), outcome))
    }

    /// Projects one episode as a single write. The edge ids come back in the
    /// order of `edges`.
    pub(super) async fn project_episode(
        &self,
        ids: &[NodeId],
        nodes: &[NewNode],
        edges: &[EpisodeEdge],
    ) -> Result<Vec<EdgeId>> {
        self.refuse_reserved(edges.iter().map(|edge| edge.kind.as_str()))?;
        if nodes.is_empty() && edges.is_empty() {
            return Ok(Vec::new());
        }
        let edge_ids: Vec<EdgeId> = edges.iter().map(|_| EdgeId::new()).collect();
        self.project(|tx, now| {
            let staged = episode_nodes(ids, nodes, now);
            let rows = episode_edge_rows(edges, &edge_ids, ids, now);
            Box::pin(async move { plan_episode(tx, staged?, rows?, now).await })
        })
        .await?;
        Ok(edge_ids)
    }

    /// A logged store keeps the `supersedes` relation for its own version
    /// history, see `refuse_supersedes`.
    fn refuse_reserved<'a>(&self, kinds: impl IntoIterator<Item = &'a str>) -> Result<()> {
        if self.log.is_some() {
            refuse_supersedes(kinds)?;
        }
        Ok(())
    }
}

/// A caller's `supersedes` edge would replay as closing its target, which the
/// live write does not do, so the relation stays the store's own.
pub(super) fn refuse_supersedes<'a>(kinds: impl IntoIterator<Item = &'a str>) -> Result<()> {
    if kinds.into_iter().any(|kind| kind == relation::SUPERSEDES) {
        return Err(Error::RelateRefused(format!(
            "'{}' is reserved for the store's version history",
            relation::SUPERSEDES
        )));
    }
    Ok(())
}

fn episode_nodes(ids: &[NodeId], nodes: &[NewNode], now: Millis) -> Result<Vec<EpisodeNode>> {
    ids.iter()
        .zip(nodes)
        .map(|(id, node)| {
            Ok(EpisodeNode {
                row: resolve_node_row(id, node, now)?,
                collision: Collision::of(node),
            })
        })
        .collect()
}

fn episode_edge_rows(
    edges: &[EpisodeEdge],
    edge_ids: &[EdgeId],
    ids: &[NodeId],
    now: Millis,
) -> Result<Vec<EdgeRow>> {
    edges
        .iter()
        .zip(edge_ids)
        .map(|(edge, id)| {
            Ok(EdgeRow {
                id: id.as_str().to_string(),
                src: resolve_episode_ref(&edge.from, ids).as_str().to_string(),
                dst: resolve_episode_ref(&edge.to, ids).as_str().to_string(),
                edge_type: edge.kind.clone(),
                attributes: serde_json::to_string(&edge.attributes)?,
                tx_from: now.0,
                tx_to: FOREVER.0,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differing_attributes_names_the_keys_whose_values_changed_added_or_dropped() {
        // Arrange
        let stored = r#"{"kept": 1, "changed": 1, "dropped": 1}"#;
        let given = r#"{"kept": 1, "changed": 2, "added": 1}"#;

        // Act
        let differing = differing_attributes(stored, given);

        // Assert
        assert_eq!(differing, ["added", "changed", "dropped"]);
    }

    #[test]
    fn differing_attributes_ignores_key_order_and_spacing() {
        // Arrange
        let stored = r#"{"a": 1, "b": 2}"#;
        let given = r#"{"b":2,"a":1}"#;

        // Act
        let differing = differing_attributes(stored, given);

        // Assert
        assert!(differing.is_empty(), "{differing:?}");
    }

    #[test]
    fn differing_attributes_flags_documents_that_are_not_both_objects() {
        // Arrange
        let cases = [("{}", "[1]"), ("[1]", "[2]"), ("not json", "{}")];

        // Act
        let flagged: Vec<_> = cases
            .iter()
            .map(|(stored, given)| differing_attributes(stored, given))
            .collect();

        // Assert
        assert!(
            flagged.iter().all(|keys| keys == &["(whole document)"]),
            "{flagged:?}"
        );
    }

    #[test]
    fn differing_attributes_accepts_equal_documents_that_are_not_objects() {
        // Arrange
        let (stored, given) = ("[1, 2]", "[1,2]");

        // Act
        let differing = differing_attributes(stored, given);

        // Assert
        assert!(differing.is_empty(), "{differing:?}");
    }

    #[test]
    fn differing_attributes_accepts_identical_text_that_is_not_json() {
        // Arrange
        let text = "stored before attributes were json";

        // Act
        let differing = differing_attributes(text, text);

        // Assert
        assert!(differing.is_empty(), "{differing:?}");
    }
}
