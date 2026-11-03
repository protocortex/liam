// SPDX-License-Identifier: Apache-2.0
//! The tier, usage and feedback rows of the lineages, and what the planner reads
//! of them. These methods write the projection only; the log records that carry
//! the same state replay through them.

use std::collections::{BTreeMap, HashSet};

use crate::backend::{Backend, BackendTx};
use crate::error::Result;
use crate::graph::logged_write::commit_or_abandon;
use crate::graph::Graph;
use crate::ids::{Millis, NodeId, FOREVER};
use crate::types::{
    CountedFeedback, FeedbackEvent, ItemView, LineageKey, LineageNode, NodeText, ReconcileReport,
    Tier, TierMove, UsageEntry, UsageSnapshot,
};
use crate::value::{Row, Value};

pub(in crate::graph) const TIER_UPSERT_SQL: &str = "INSERT INTO item_tier
       (lineage_key, tier, changed_at, residency_until, last_event_id)
     VALUES (?1, ?2, ?3, ?4, ?5)
     ON CONFLICT(lineage_key) DO UPDATE SET
       tier = excluded.tier, changed_at = excluded.changed_at,
       residency_until = excluded.residency_until, last_event_id = excluded.last_event_id";

/// The nodes a demotion leaves without an int8 form: those of the lineage in
/// `?1` that are live at `?2` and were written no later than `?3`.
const DROP_INT8_SQL: &str = "UPDATE node_quant SET i8 = NULL WHERE i8 IS NOT NULL AND node_id IN
       (SELECT n.id FROM nodes n JOIN node_lineage l ON l.node_id = n.id
        WHERE l.lineage_key = ?1 AND n.tx_to = ?2 AND n.tx_from <= ?3)";

const USAGE_EPOCH_SQL: &str = "INSERT OR IGNORE INTO usage_epochs (event_id, day) VALUES (?1, ?2)";

const USAGE_ENTRY_SQL: &str = "INSERT OR IGNORE INTO item_usage_events
       (event_id, lineage_key, day, hits, last_access_at)
     VALUES (?1, ?2, ?3, ?4, ?5)";

/// A producer's first verdict on a lineage in a day counts; later ones that day
/// are kept and do not.
const FEEDBACK_COUNTED_SQL: &str = "SELECT 1 FROM item_feedback
     WHERE lineage_key = ?1 AND producer = ?2 AND day = ?3 AND counted = 1 AND event_id != ?4";

const FEEDBACK_INSERT_SQL: &str = "INSERT OR IGNORE INTO item_feedback
       (event_id, lineage_key, producer, at, day, correct, counted)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)";

/// The rows a demotion leaves without a vector, with `?1` the cold tier and
/// `?2` the end of an open node.
const DEMOTED_NODES: &str = "FROM nodes n
     JOIN node_lineage l ON l.node_id = n.id
     JOIN item_tier t ON t.lineage_key = l.lineage_key AND t.tier = ?1
     JOIN node_quant q ON q.node_id = n.id
     WHERE n.tx_to = ?2 AND n.tx_from <= t.changed_at";

/// What a cold lineage still holds in vectors: those to drop, and those kept
/// because the node was written after the lineage went cold or has no signature.
pub(in crate::graph) struct Held {
    pub(in crate::graph) droppable: Vec<NodeId>,
    pub(in crate::graph) late: usize,
    pub(in crate::graph) unsigned: usize,
}

impl Held {
    pub(in crate::graph) fn of(nodes: &[LineageNode], went_cold_at: Millis) -> Self {
        let mut held = Self {
            droppable: Vec::new(),
            late: 0,
            unsigned: 0,
        };
        for node in nodes.iter().filter(|node| node.has_vector) {
            if node.tx_from > went_cold_at {
                held.late += 1;
            } else if !node.has_signature {
                held.unsigned += 1;
            } else {
                held.droppable.push(node.node_id.clone());
            }
        }
        held
    }
}

fn optional_i64(row: &Row, index: usize) -> Result<Option<i64>> {
    match row.0.get(index) {
        Some(Value::Null) | None => Ok(None),
        _ => row.get_i64(index).map(Some),
    }
}

fn optional_millis(row: &Row, index: usize) -> Result<Option<Millis>> {
    Ok(optional_i64(row, index)?.map(Millis))
}

impl<B: Backend> Graph<B> {
    /// The tier of a lineage. One with no row is hot.
    pub async fn tier_of(&self, key: &LineageKey) -> Result<Tier> {
        let rows = self
            .backend
            .query(
                "SELECT tier FROM item_tier WHERE lineage_key = ?1",
                &[key.as_str().into()],
            )
            .await?;
        match rows.first() {
            Some(row) => Tier::parse(&row.get_string(0)?),
            None => Ok(Tier::Hot),
        }
    }

    /// Sets the tier of a lineage and flags its nodes in the index to match.
    /// `event_id` is the log record that made the move, if one did.
    pub async fn set_tier(&self, change: &TierMove, event_id: Option<&str>) -> Result<()> {
        let _gate = self.quant_gate.lock().await;
        self.backend
            .execute(TIER_UPSERT_SQL, &tier_params(change, event_id))
            .await?;
        self.flag_lineage(&change.lineage_key, change.to).await
    }

    /// Flags the nodes of a lineage in the index as `tier` holds them. The
    /// caller holds the index gate.
    pub(in crate::graph) async fn flag_lineage(&self, key: &LineageKey, tier: Tier) -> Result<()> {
        let nodes = self.lineage_node_ids(key).await?;
        self.quant
            .set_cold(nodes.iter().map(String::as_str), tier == Tier::Cold);
        Ok(())
    }

    async fn lineage_node_ids(&self, key: &LineageKey) -> Result<Vec<String>> {
        self.backend
            .query(
                "SELECT node_id FROM node_lineage WHERE lineage_key = ?1",
                &[key.as_str().into()],
            )
            .await?
            .iter()
            .map(|row| row.get_string(0))
            .collect()
    }

    /// Records one logged usage batch. A batch already applied changes nothing.
    /// An empty one still leaves its epoch, which is when observing began.
    pub async fn apply_usage(
        &self,
        event_id: &str,
        day: i64,
        entries: &[UsageEntry],
    ) -> Result<()> {
        let mut tx = self.backend.begin().await?;
        let applied = apply_usage_in(&mut *tx, event_id, day, entries).await;
        commit_or_abandon(tx, applied).await
    }

    /// Records one logged verdict and says whether it was newly counted. Of a
    /// producer's verdicts on a lineage in a day only the first applied counts,
    /// and replay applies in log order, so it agrees. A verdict already applied
    /// changes nothing.
    pub async fn apply_feedback(&self, event: &FeedbackEvent) -> Result<bool> {
        let mut tx = self.backend.begin().await?;
        let applied = apply_feedback_in(&mut *tx, event).await;
        commit_or_abandon(tx, applied).await
    }

    /// Whether any usage batch, even an empty one, has been recorded. Counting
    /// began when one did.
    pub async fn usage_started(&self) -> Result<bool> {
        let rows = self
            .backend
            .query("SELECT 1 FROM usage_epochs LIMIT 1", &[])
            .await?;
        Ok(!rows.is_empty())
    }

    /// The live nodes of a lineage and whether each holds a vector and a
    /// signature.
    pub async fn lineage_nodes(&self, key: &LineageKey) -> Result<Vec<LineageNode>> {
        let with_vector = self.live_vector_ids().await?;
        self.lineage_nodes_among(key, &with_vector).await
    }

    async fn lineage_nodes_among(
        &self,
        key: &LineageKey,
        with_vector: &HashSet<String>,
    ) -> Result<Vec<LineageNode>> {
        let rows = self
            .backend
            .query(
                "SELECT n.id, n.tx_from, q.node_id IS NOT NULL
                 FROM node_lineage l JOIN nodes n ON n.id = l.node_id
                 LEFT JOIN node_quant q ON q.node_id = n.id
                 WHERE l.lineage_key = ?1 AND n.tx_to = ?2 ORDER BY n.id",
                &[key.as_str().into(), FOREVER.into()],
            )
            .await?;
        rows.iter()
            .map(|row| {
                let id = row.get_string(0)?;
                Ok(LineageNode {
                    has_vector: with_vector.contains(&id),
                    node_id: NodeId::from_raw(id),
                    tx_from: Millis(row.get_i64(1)?),
                    has_signature: row.get_i64(2)? != 0,
                })
            })
            .collect()
    }

    /// The text of each live node of a lineage, in id order.
    pub async fn lineage_texts(&self, key: &LineageKey) -> Result<Vec<NodeText>> {
        let with_vector = self.live_vector_ids().await?;
        let rows = self
            .backend
            .query(
                "SELECT n.id, n.label, n.content FROM node_lineage l
                 JOIN nodes n ON n.id = l.node_id
                 WHERE l.lineage_key = ?1 AND n.tx_to = ?2 ORDER BY n.id",
                &[key.as_str().into(), FOREVER.into()],
            )
            .await?;
        rows.iter()
            .map(|row| {
                let id = row.get_string(0)?;
                Ok(NodeText {
                    has_vector: with_vector.contains(&id),
                    node_id: NodeId::from_raw(id),
                    label: row.get_string(1)?,
                    content: row.get_string(2)?,
                })
            })
            .collect()
    }

    /// The ids of the live nodes that have a vector. A node with no text to
    /// embed has none.
    async fn live_vector_ids(&self) -> Result<HashSet<String>> {
        let embeddable = self
            .backend
            .query(
                "SELECT id FROM nodes WHERE tx_to = ?1 AND trim(content) != ''",
                &[FOREVER.into()],
            )
            .await?;
        let missing: HashSet<String> = self
            .backend
            .nodes_missing_vectors()
            .await?
            .into_iter()
            .map(|id| id.as_str().to_string())
            .collect();
        let mut with_vector = HashSet::with_capacity(embeddable.len());
        for row in &embeddable {
            let id = row.get_string(0)?;
            if !missing.contains(&id) {
                with_vector.insert(id);
            }
        }
        Ok(with_vector)
    }

    /// Makes the vectors agree with the tier rows, which decide: every live node
    /// of a cold lineage that was written no later than the lineage went cold
    /// and has a signature loses its vector. A node written later, or one with a
    /// vector and no signature, keeps it and is counted. A hot lineage is left
    /// alone, because a hot node with no vector is legitimate.
    pub async fn reconcile_tier_vectors(&self) -> Result<ReconcileReport> {
        let cold = self
            .backend
            .query(
                "SELECT lineage_key, changed_at FROM item_tier WHERE tier = ?1 ORDER BY lineage_key",
                &[Tier::Cold.as_str().into()],
            )
            .await?;
        let with_vector = self.live_vector_ids().await?;
        let mut report = ReconcileReport::default();
        for row in &cold {
            let (key, changed_at) = (LineageKey::from_raw(row.get_string(0)?), row.get_i64(1)?);
            let nodes = self.lineage_nodes_among(&key, &with_vector).await?;
            let held = Held::of(&nodes, Millis(changed_at));
            for node in &held.droppable {
                self.backend.vector_delete(node.as_str()).await?;
            }
            report.deleted += held.droppable.len();
            report.skipped_late += held.late;
            report.skipped_unsigned += held.unsigned;
        }
        tracing::info!(?report, "tier rows and vectors reconciled");
        Ok(report)
    }

    /// The live nodes a demotion left without a vector: those of a cold lineage
    /// that were written no later than it went cold and have a signature.
    pub(in crate::graph) async fn demoted_node_ids(&self) -> Result<HashSet<String>> {
        let rows = self
            .backend
            .query(
                &format!("SELECT n.id {DEMOTED_NODES}"),
                &[Tier::Cold.as_str().into(), FOREVER.into()],
            )
            .await?;
        rows.iter().map(|row| row.get_string(0)).collect()
    }

    /// Makes what a replay or a re-embed brought back agree with the tier rows:
    /// a demoted node loses its int8 form and its vector again.
    pub(in crate::graph) async fn settle_cold_tiers(&self) -> Result<ReconcileReport> {
        self.backend
            .execute(
                &format!(
                    "UPDATE node_quant SET i8 = NULL WHERE i8 IS NOT NULL AND node_id IN
                     (SELECT n.id {DEMOTED_NODES})"
                ),
                &[Tier::Cold.as_str().into(), FOREVER.into()],
            )
            .await?;
        self.reconcile_tier_vectors().await
    }

    /// Drops usage older than `before_day` and every epoch but the earliest,
    /// which `observation_since` is read from, so it never moves forward.
    /// Returns how many rows went.
    pub async fn prune_usage(&self, before_day: i64) -> Result<u64> {
        let usage = self
            .backend
            .execute(
                "DELETE FROM item_usage_events WHERE day < ?1",
                &[before_day.into()],
            )
            .await?;
        let epochs = self
            .backend
            .execute(
                "DELETE FROM usage_epochs WHERE day < ?1 AND event_id != (
                   SELECT event_id FROM usage_epochs ORDER BY day, event_id LIMIT 1)",
                &[before_day.into()],
            )
            .await?;
        Ok(usage + epochs)
    }

    /// Drops the tier, usage and feedback rows of a lineage that no node has,
    /// as after the sweep deleted them all. Returns how many rows went.
    pub async fn prune_tier_state(&self) -> Result<u64> {
        let mut removed = 0;
        for table in ["item_tier", "item_usage_events", "item_feedback"] {
            let sql = format!(
                "DELETE FROM {table} WHERE NOT EXISTS (
                   SELECT 1 FROM node_lineage WHERE lineage_key = {table}.lineage_key)"
            );
            removed += self.backend.execute(&sql, &[]).await?;
        }
        Ok(removed)
    }

    /// The lineages that tiering never demotes: those with a live entity page,
    /// and those with a live node that a live summary is built from.
    pub async fn tier_exempt_lineages(&self) -> Result<HashSet<LineageKey>> {
        let rows = self
            .backend
            .query(
                "SELECT l.lineage_key FROM node_lineage l JOIN nodes n ON n.id = l.node_id
                 WHERE n.tx_to = ?1 AND n.derivation = ?2
                 UNION
                 SELECT l.lineage_key FROM node_lineage l
                 JOIN nodes n ON n.id = l.node_id AND n.tx_to = ?1
                 JOIN summary_children c ON c.child_id = n.id AND c.child_kind = 'node'
                 JOIN summaries s ON s.id = c.parent_id AND s.tx_to = ?1",
                &[
                    FOREVER.into(),
                    crate::types::Derivation::Inferred.as_str().into(),
                ],
            )
            .await?;
        rows.iter()
            .map(|row| Ok(LineageKey::from_raw(row.get_string(0)?)))
            .collect()
    }

    /// One consistent read of every lineage that has a live node: usage inside
    /// the window (a day exactly `window_days` old is inside), counted feedback,
    /// the tier row, and what the lineage's nodes hold. The read holds the
    /// store's write lock, so no batch of usage is seen in part.
    pub async fn usage_snapshot(&self, now: Millis, window_days: i64) -> Result<UsageSnapshot> {
        let with_vector = self.live_vector_ids().await?;
        let mut tx = self.backend.begin().await?;
        let read = read_snapshot(&mut *tx, now, window_days, &with_vector).await;
        // Nothing was written, so rolling back is the same as committing.
        commit_or_abandon(tx, read).await
    }
}

fn tier_params(change: &TierMove, event_id: Option<&str>) -> [Value; 5] {
    [
        change.lineage_key.as_str().into(),
        change.to.as_str().into(),
        change.changed_at.into(),
        change.residency_until.map_or(Value::Null, Value::from),
        event_id.map_or(Value::Null, Value::from),
    ]
}

/// Sets the tier row of a lineage inside `tx`. A move already applied is set
/// again to the same values, and the last move replayed wins.
pub(in crate::graph) async fn apply_tier_in(
    tx: &mut dyn BackendTx,
    change: &TierMove,
    event_id: Option<&str>,
) -> Result<()> {
    tx.execute(TIER_UPSERT_SQL, &tier_params(change, event_id))
        .await?;
    if change.to == Tier::Cold {
        // A node with no int8 form is how a write inside a transaction tells a
        // demoted node from a hot one.
        tx.execute(
            DROP_INT8_SQL,
            &[
                change.lineage_key.as_str().into(),
                FOREVER.into(),
                change.changed_at.into(),
            ],
        )
        .await?;
    }
    Ok(())
}

pub(in crate::graph) async fn apply_usage_in(
    tx: &mut dyn BackendTx,
    event_id: &str,
    day: i64,
    entries: &[UsageEntry],
) -> Result<()> {
    tx.execute(USAGE_EPOCH_SQL, &[event_id.into(), day.into()])
        .await?;
    for entry in entries {
        let params = [
            event_id.into(),
            entry.lineage_key.as_str().into(),
            day.into(),
            i64::from(entry.hits).into(),
            entry.last_access_at.into(),
        ];
        tx.execute(USAGE_ENTRY_SQL, &params).await?;
    }
    Ok(())
}

pub(in crate::graph) async fn apply_feedback_in(
    tx: &mut dyn BackendTx,
    event: &FeedbackEvent,
) -> Result<bool> {
    let day = event.at.day_number();
    let earlier = tx
        .query(
            FEEDBACK_COUNTED_SQL,
            &[
                event.lineage_key.as_str().into(),
                event.producer.as_str().into(),
                day.into(),
                event.event_id.as_str().into(),
            ],
        )
        .await?;
    let counted = earlier.is_empty();
    let stored = tx
        .execute(
            FEEDBACK_INSERT_SQL,
            &[
                event.event_id.as_str().into(),
                event.lineage_key.as_str().into(),
                event.producer.as_str().into(),
                event.at.into(),
                day.into(),
                i64::from(event.correct).into(),
                i64::from(counted).into(),
            ],
        )
        .await?;
    Ok(counted && stored == 1)
}

/// A lineage while its nodes are being read.
struct Building {
    newest: Millis,
    producer: String,
    has_vector: bool,
    unsigned_vector: bool,
}

async fn read_snapshot(
    tx: &mut dyn BackendTx,
    now: Millis,
    window_days: i64,
    with_vector: &HashSet<String>,
) -> Result<UsageSnapshot> {
    let mut lineages: BTreeMap<String, Building> = BTreeMap::new();
    let nodes = tx
        .query(
            "SELECT l.lineage_key, n.id, n.tx_from, n.producer, q.node_id IS NOT NULL
             FROM nodes n JOIN node_lineage l ON l.node_id = n.id
             LEFT JOIN node_quant q ON q.node_id = n.id
             WHERE n.tx_to = ?1 ORDER BY l.lineage_key, n.tx_from, n.id",
            &[FOREVER.into()],
        )
        .await?;
    for row in &nodes {
        let (id, tx_from) = (row.get_string(1)?, Millis(row.get_i64(2)?));
        let vector = with_vector.contains(&id);
        let signed = row.get_i64(4)? != 0;
        let entry = lineages
            .entry(row.get_string(0)?)
            .or_insert_with(|| Building {
                newest: tx_from,
                producer: String::new(),
                has_vector: false,
                unsigned_vector: false,
            });
        // Rows come oldest first, so the last one read is the newest.
        entry.newest = tx_from;
        entry.producer = row.get_string(3)?;
        entry.has_vector |= vector;
        entry.unsigned_vector |= vector && !signed;
    }

    let mut tiers = BTreeMap::new();
    let tier_rows = tx
        .query(
            "SELECT lineage_key, tier, changed_at, residency_until FROM item_tier",
            &[],
        )
        .await?;
    for row in &tier_rows {
        tiers.insert(
            row.get_string(0)?,
            (
                Tier::parse(&row.get_string(1)?)?,
                Millis(row.get_i64(2)?),
                optional_millis(row, 3)?,
            ),
        );
    }

    let first_day = now.day_number() - window_days;
    let mut hits: BTreeMap<String, Vec<(i64, i64)>> = BTreeMap::new();
    let hit_rows = tx
        .query(
            "SELECT lineage_key, day, SUM(hits) FROM item_usage_events WHERE day >= ?1
             GROUP BY lineage_key, day ORDER BY lineage_key, day",
            &[first_day.into()],
        )
        .await?;
    for row in &hit_rows {
        hits.entry(row.get_string(0)?)
            .or_default()
            .push((row.get_i64(1)?, row.get_i64(2)?));
    }

    let mut last_access = BTreeMap::new();
    let access_rows = tx
        .query(
            "SELECT lineage_key, MAX(last_access_at) FROM item_usage_events GROUP BY lineage_key",
            &[],
        )
        .await?;
    for row in &access_rows {
        last_access.insert(row.get_string(0)?, Millis(row.get_i64(1)?));
    }

    let mut feedback: BTreeMap<String, Vec<CountedFeedback>> = BTreeMap::new();
    let feedback_rows = tx
        .query(
            "SELECT lineage_key, producer, at, correct FROM item_feedback WHERE counted = 1
             ORDER BY lineage_key, at DESC, event_id DESC",
            &[],
        )
        .await?;
    for row in &feedback_rows {
        feedback
            .entry(row.get_string(0)?)
            .or_default()
            .push(CountedFeedback {
                producer: row.get_string(1)?,
                at: Millis(row.get_i64(2)?),
                correct: row.get_i64(3)? != 0,
            });
    }

    let earliest = tx.query("SELECT MIN(day) FROM usage_epochs", &[]).await?;
    let observation_since = match earliest.first() {
        Some(row) => optional_i64(row, 0)?.map(Millis::start_of_day),
        None => None,
    };

    let items = lineages
        .into_iter()
        .map(|(key, building)| {
            let (tier, changed_at, residency_until) = match tiers.get(&key) {
                Some((tier, changed_at, residency)) => (*tier, Some(*changed_at), *residency),
                None => (Tier::Hot, None, None),
            };
            ItemView {
                newest_tx_from: building.newest,
                hits_by_day: hits.remove(&key).unwrap_or_default(),
                last_access_at: last_access.get(&key).copied(),
                feedback: feedback.remove(&key).unwrap_or_default(),
                tier,
                changed_at,
                residency_until,
                has_vector: building.has_vector,
                has_signature: !building.unsigned_vector,
                producer: building.producer,
                lineage_key: LineageKey::from_raw(key),
            }
        })
        .collect();
    Ok(UsageSnapshot {
        now,
        observation_since,
        items,
    })
}
