// SPDX-License-Identifier: Apache-2.0
//! Removing what a destroyed key group left in the store's derived tables.
//!
//! A purge starts from the nodes the group sealed (`node_keys`), follows them
//! into every table that copies or cites them, and deletes those rows in one
//! transaction, with a `LineageRemoval` event appended first so a replay deletes
//! them again. The sets it deletes by are fixed before the first delete and kept
//! in the `lineage_removals` outbox, which is how a rerun after a crash removes
//! the same rows and how `verify` finds a row once the rows that named it are
//! gone.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use async_trait::async_trait;
use futures_util::StreamExt;
use liam_log::event::LogPayload;
use serde::{Deserialize, Serialize};

use super::logged_plan::{event, STORE_SOURCE, STORE_TRUST};
use super::logged_write::{abandon, commit_or_abandon, HeldLog};
use super::projection::{apply_steps, steps_for};
use super::row_keys::{record_row_keys, DERIVED_ARTIFACT, DERIVED_FACT};
use super::Graph;
use crate::backend::{Backend, BackendTx};
use crate::error::Result;
use crate::ids::{NodeId, FOREVER};
use crate::types::{LineageHits, LineageTarget, PurgeReport, Residual};
use crate::value::{Row, Value};

/// The name `Graph` answers to as a lineage store.
pub const GRAPH_STORE: &str = "graph";

/// Values bound in one `IN` list, which keeps a statement under the engine's
/// bound-variable limit and each batch of work short.
const CHUNK: usize = 500;

/// Where a statement's `IN` list goes. Every occurrence in one statement binds
/// the same numbered parameters.
const IN_LIST: &str = "{in}";

/// A store whose rows a delete job removes. `Graph` is one; the daemon adds the
/// stores that live in files and in memory.
#[async_trait]
pub trait LineageStore: Send + Sync {
    fn name(&self) -> &str;

    /// What a purge of `target` would remove, at most: a row that a surviving
    /// row still vouches for is counted here and kept by the purge.
    async fn find(&self, target: &LineageTarget) -> Result<LineageHits>;

    /// Removes the rows of `target`. A rerun removes the same rows, and one that
    /// finds nothing left appends no event.
    async fn purge(&self, target: &LineageTarget) -> Result<PurgeReport>;

    /// Every row that still refers to a deleted id.
    async fn verify(&self, target: &LineageTarget) -> Result<Vec<Residual>>;
}

/// What a purge is told to remove. A replay of the logged removal has only this.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct PurgeInput {
    pub(super) node_ids: Vec<String>,
    pub(super) event_ids: Vec<String>,
    pub(super) key_ids: Vec<String>,
}

/// The rows to delete by, worked out from a `PurgeInput` and the tables as they
/// stand. A row of a set that a surviving row still vouches for is kept by the
/// guard of the table it is in.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Doomed {
    nodes: Vec<String>,
    facts: Vec<String>,
    fact_lineages: Vec<String>,
    summaries: Vec<String>,
    summary_hashes: Vec<String>,
    lineages: Vec<String>,
    subjects: Vec<String>,
    batches: Vec<String>,
    events: Vec<String>,
    keys: Vec<String>,
}

#[derive(Clone, Copy)]
enum Set {
    Nodes,
    Facts,
    FactLineages,
    Summaries,
    SummaryHashes,
    Lineages,
    Subjects,
    Batches,
    Events,
    Keys,
}

impl Doomed {
    fn set(&self, set: Set) -> &[String] {
        match set {
            Set::Nodes => &self.nodes,
            Set::Facts => &self.facts,
            Set::FactLineages => &self.fact_lineages,
            Set::Summaries => &self.summaries,
            Set::SummaryHashes => &self.summary_hashes,
            Set::Lineages => &self.lineages,
            Set::Subjects => &self.subjects,
            Set::Batches => &self.batches,
            Set::Events => &self.events,
            Set::Keys => &self.keys,
        }
    }
}

/// One table a purge deletes from. `filter` selects the rows that belong to
/// `set`; `guard` keeps the ones a surviving row still vouches for, and only
/// applies once the nodes are gone.
struct Target {
    table: &'static str,
    id: &'static str,
    filter: &'static str,
    set: Set,
    guard: Option<&'static str>,
}

const NO_NODE_OF_LINEAGE: &str = "NOT EXISTS (SELECT 1 FROM node_lineage nl
     WHERE nl.lineage_key = {table}.lineage_key)";
const NO_NODE_OF_SUBJECT: &str = "NOT EXISTS (SELECT 1 FROM nodes n
     WHERE n.subject = {table}.subject AND COALESCE(n.scope, '') = {table}.scope)";
const NO_FACT_IN_LINEAGE: &str = "NOT EXISTS (SELECT 1 FROM facts f
     WHERE f.lineage_id = fact_sources.lineage_id)";

/// The tables a purge deletes from, in the order it deletes: a row goes before
/// the rows it points at, and a guard reads tables an earlier entry emptied.
/// `INVENTORY_EXEMPT` names the tables that are not here and why.
const TARGETS: [Target; 23] = [
    target(
        "summary_failures",
        "child_hash",
        "child_hash IN ({in})",
        Set::SummaryHashes,
    ),
    target(
        "summary_dissolving",
        "summary_id",
        "summary_id IN ({in})",
        Set::Summaries,
    ),
    target(
        "summary_embeddings",
        "summary_id",
        "summary_id IN ({in})",
        Set::Summaries,
    ),
    target(
        "summary_sources",
        "summary_id",
        "summary_id IN ({in})",
        Set::Summaries,
    ),
    target(
        "summary_children",
        "parent_id",
        "parent_id IN ({in}) OR child_id IN ({in})",
        Set::Summaries,
    ),
    target("summaries", "id", "id IN ({in})", Set::Summaries),
    target("facts", "id", "id IN ({in})", Set::Facts),
    guarded(
        "fact_sources",
        "lineage_id",
        "lineage_id IN ({in})",
        Set::FactLineages,
        NO_FACT_IN_LINEAGE,
    ),
    target("derived_keys", "row_id", "key_id IN ({in})", Set::Keys),
    target("edges", "id", "src IN ({in}) OR dst IN ({in})", Set::Nodes),
    target("node_community", "node_id", "node_id IN ({in})", Set::Nodes),
    target("node_events", "node_id", "node_id IN ({in})", Set::Nodes),
    target("node_quant", "node_id", "node_id IN ({in})", Set::Nodes),
    target("node_lineage", "node_id", "node_id IN ({in})", Set::Nodes),
    target("nodes", "id", "id IN ({in})", Set::Nodes),
    guarded(
        "item_tier",
        "lineage_key",
        "lineage_key IN ({in})",
        Set::Lineages,
        NO_NODE_OF_LINEAGE,
    ),
    guarded(
        "item_usage_events",
        "lineage_key",
        "lineage_key IN ({in})",
        Set::Lineages,
        NO_NODE_OF_LINEAGE,
    ),
    guarded(
        "item_feedback",
        "event_id",
        "lineage_key IN ({in})",
        Set::Lineages,
        NO_NODE_OF_LINEAGE,
    ),
    guarded(
        "entity_mention_state",
        "subject",
        "subject IN ({in})",
        Set::Subjects,
        NO_NODE_OF_SUBJECT,
    ),
    guarded(
        "fact_suppressions",
        "subject",
        "subject IN ({in})",
        Set::Subjects,
        NO_NODE_OF_SUBJECT,
    ),
    target(
        "extraction_attempts",
        "batch_key",
        "batch_key IN ({in})",
        Set::Batches,
    ),
    target(
        "log_hash_index",
        "first_event_id",
        "first_event_id IN ({in})",
        Set::Events,
    ),
    target(
        "log_quarantine",
        "event_id",
        "event_id IN ({in})",
        Set::Events,
    ),
];

/// Tables a purge leaves alone, each with the reason. A test checks that every
/// table is a target, here, or the vector table the backend owns.
#[cfg(test)]
const INVENTORY_EXEMPT: [(&str, &str); 13] = [
    (
        "node_keys",
        "ids only, and the key group still resolves to the nodes it sealed",
    ),
    ("lineage_removals", "ids only, the purge's own record"),
    ("log_cursor", "a log position"),
    ("log_backfill_state", "a log position"),
    ("extraction_cursor", "a log position"),
    ("cluster_state", "counts of edges and times"),
    ("provenance_repair_state", "a time"),
    ("derivation_backfill_state", "a time and a count"),
    ("quant_backfill_state", "a time and a count"),
    ("lineage_backfill_state", "a time and a count"),
    ("summary_tree_state", "counts per scope and level"),
    ("usage_epochs", "an event id and a day, no lineage"),
    ("node_vectors", "removed through the backend"),
];

#[cfg(test)]
pub(super) fn target_tables() -> Vec<&'static str> {
    TARGETS.iter().map(|target| target.table).collect()
}

#[cfg(test)]
pub(super) fn exempt_tables() -> Vec<&'static str> {
    INVENTORY_EXEMPT.iter().map(|(table, _)| *table).collect()
}

const fn target(table: &'static str, id: &'static str, filter: &'static str, set: Set) -> Target {
    Target {
        table,
        id,
        filter,
        set,
        guard: None,
    }
}

const fn guarded(
    table: &'static str,
    id: &'static str,
    filter: &'static str,
    set: Set,
    guard: &'static str,
) -> Target {
    Target {
        table,
        id,
        filter,
        set,
        guard: Some(guard),
    }
}

impl Target {
    fn clause(&self, bound: usize, guarded: bool) -> String {
        let filter = self.filter.replace(IN_LIST, &placeholders(bound));
        match self.guard.filter(|_| guarded) {
            Some(guard) => format!("({filter}) AND {}", guard.replace("{table}", self.table)),
            None => filter,
        }
    }

    fn delete_sql(&self, bound: usize) -> String {
        format!(
            "DELETE FROM {} WHERE {}",
            self.table,
            self.clause(bound, true)
        )
    }

    fn select_sql(&self, bound: usize, guarded: bool) -> String {
        format!(
            "SELECT {} FROM {} WHERE {}",
            self.id,
            self.table,
            self.clause(bound, guarded)
        )
    }
}

fn placeholders(n: usize) -> String {
    (1..=n)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn params(values: &[String]) -> Vec<Value> {
    values.iter().map(|value| value.as_str().into()).collect()
}

/// Something that answers a read: the pool, or an open transaction that sees its
/// own writes.
#[async_trait]
trait Rows: Send {
    async fn rows(&mut self, sql: &str, params: &[Value]) -> Result<Vec<Row>>;
}

struct Pool<'a, B>(&'a B);

#[async_trait]
impl<B: Backend> Rows for Pool<'_, B> {
    async fn rows(&mut self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        self.0.query(sql, params).await
    }
}

struct InTx<'a>(&'a mut dyn BackendTx);

#[async_trait]
impl Rows for InTx<'_> {
    async fn rows(&mut self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        self.0.query(sql, params).await
    }
}

/// The first column of `sql` for `values`, bound `CHUNK` at a time.
async fn column(reader: &mut dyn Rows, sql: &str, values: &[String]) -> Result<BTreeSet<String>> {
    let mut found = BTreeSet::new();
    for chunk in values.chunks(CHUNK) {
        let sql = sql.replace(IN_LIST, &placeholders(chunk.len()));
        for row in reader.rows(&sql, &params(chunk)).await? {
            found.insert(row.get_string(0)?);
        }
    }
    Ok(found)
}

fn listed(found: BTreeSet<String>) -> Vec<String> {
    found.into_iter().collect()
}

/// Works out every set a purge deletes by.
async fn resolve(reader: &mut dyn Rows, input: &PurgeInput) -> Result<Doomed> {
    let nodes = &input.node_ids;
    let fact_ids = column(
        reader,
        &format!("SELECT row_id FROM derived_keys WHERE row_kind = '{DERIVED_FACT}' AND key_id IN ({IN_LIST})"),
        &input.key_ids,
    )
    .await?;
    let artifact_ids = column(
        reader,
        &format!("SELECT row_id FROM derived_keys WHERE row_kind = '{DERIVED_ARTIFACT}' AND key_id IN ({IN_LIST})"),
        &input.key_ids,
    )
    .await?;
    let facts = listed(fact_ids);
    let fact_lineages = column(
        reader,
        "SELECT lineage_id FROM facts WHERE id IN ({in})",
        &facts,
    )
    .await?;
    let summaries = summary_closure(reader, nodes, &listed(artifact_ids)).await?;
    let summaries = listed(summaries);
    let summary_hashes = column(
        reader,
        "SELECT child_hash FROM summaries WHERE id IN ({in})",
        &summaries,
    )
    .await?;
    let lineages = column(
        reader,
        "SELECT lineage_key FROM node_lineage WHERE node_id IN ({in})",
        nodes,
    )
    .await?;
    let subjects = column(
        reader,
        "SELECT subject FROM nodes WHERE subject IS NOT NULL AND id IN ({in})",
        nodes,
    )
    .await?;
    Ok(Doomed {
        nodes: nodes.clone(),
        facts,
        fact_lineages: listed(fact_lineages),
        summaries,
        summary_hashes: listed(summary_hashes),
        lineages: listed(lineages),
        subjects: listed(subjects),
        batches: citing_batches(reader, &input.event_ids).await?,
        events: input.event_ids.clone(),
        keys: input.key_ids.clone(),
    })
}

/// Every summary built from one of `nodes` or sealed under a deleted key, and
/// every ancestor of those: a parent's text is made from its children's.
async fn summary_closure(
    reader: &mut dyn Rows,
    nodes: &[String],
    artifacts: &[String],
) -> Result<BTreeSet<String>> {
    let mut found = column(
        reader,
        "SELECT parent_id FROM summary_children WHERE child_kind = 'node' AND child_id IN ({in})",
        nodes,
    )
    .await?;
    found.extend(
        column(
            reader,
            "SELECT id FROM summaries WHERE id IN ({in})",
            artifacts,
        )
        .await?,
    );
    let mut frontier = listed(found.clone());
    while !frontier.is_empty() {
        let parents = column(
            reader,
            "SELECT parent_id FROM summary_children WHERE child_kind = 'summary' AND child_id IN ({in})",
            &frontier,
        )
        .await?;
        frontier = parents.difference(&found).cloned().collect();
        found.extend(parents);
    }
    Ok(found)
}

/// The batches in `extraction_attempts` that cite one of `event_ids`. Their ids
/// are a JSON array, so the table is read whole: it holds failures only.
async fn citing_batches(reader: &mut dyn Rows, event_ids: &[String]) -> Result<Vec<String>> {
    if event_ids.is_empty() {
        return Ok(Vec::new());
    }
    let wanted: HashSet<&str> = event_ids.iter().map(String::as_str).collect();
    let mut batches = Vec::new();
    for row in reader
        .rows("SELECT batch_key, event_ids FROM extraction_attempts", &[])
        .await?
    {
        let cited: Vec<String> = serde_json::from_str(&row.get_string(1)?)?;
        if cited.iter().any(|id| wanted.contains(id.as_str())) {
            batches.push(row.get_string(0)?);
        }
    }
    batches.sort();
    Ok(batches)
}

/// The ids of the rows each table holds for `doomed`, skipping empty ones.
async fn matching(
    reader: &mut dyn Rows,
    doomed: &Doomed,
    guarded: bool,
) -> Result<Vec<(&'static str, BTreeSet<String>)>> {
    let mut found = Vec::new();
    for target in &TARGETS {
        let mut ids = BTreeSet::new();
        for chunk in doomed.set(target.set).chunks(CHUNK) {
            let sql = target.select_sql(chunk.len(), guarded);
            for row in reader.rows(&sql, &params(chunk)).await? {
                ids.insert(row.get_string(0)?);
            }
        }
        if !ids.is_empty() {
            found.push((target.table, ids));
        }
    }
    Ok(found)
}

/// The surviving fact versions that were closed by a deleted one.
async fn dangling_closers(reader: &mut dyn Rows, doomed: &Doomed) -> Result<BTreeSet<String>> {
    column(
        reader,
        "SELECT id FROM facts WHERE closed_by IN ({in})",
        &doomed.facts,
    )
    .await
}

/// Deletes the rows of `input` inside `tx` and returns how many it removed. A
/// live purge and a replay of its record run this one function.
pub(super) async fn purge_in_tx(
    tx: &mut dyn BackendTx,
    input: &PurgeInput,
    vector_delete: Option<&str>,
) -> Result<usize> {
    let doomed = resolve(&mut InTx(&mut *tx), input).await?;
    let mut removed = 0;
    for target in &TARGETS {
        if target.table == "nodes" {
            if let Some(sql) = vector_delete {
                for id in &doomed.nodes {
                    tx.execute(sql, &[id.as_str().into()]).await?;
                }
            }
        }
        for chunk in doomed.set(target.set).chunks(CHUNK) {
            removed += tx
                .execute(&target.delete_sql(chunk.len()), &params(chunk))
                .await? as usize;
        }
        if target.table == "facts" {
            reopen_closed(tx, &doomed.facts).await?;
        }
    }
    Ok(removed)
}

/// A fact version that a deleted one had replaced is current again.
async fn reopen_closed(tx: &mut dyn BackendTx, closers: &[String]) -> Result<()> {
    for chunk in closers.chunks(CHUNK) {
        let sql = format!(
            "UPDATE facts SET tx_to = {}, closed_by = '' WHERE closed_by IN ({})",
            FOREVER.0,
            placeholders(chunk.len())
        );
        tx.execute(&sql, &params(chunk)).await?;
    }
    Ok(())
}

/// What the outbox holds for one job.
struct Outbox {
    input: PurgeInput,
    doomed: Doomed,
    done: bool,
}

const OUTBOX_INSERT_SQL: &str = "INSERT INTO lineage_removals (job_id, store, input, doomed)
     VALUES (?1, ?2, ?3, ?4) ON CONFLICT(job_id, store) DO NOTHING";
const OUTBOX_DONE_SQL: &str =
    "UPDATE lineage_removals SET done_at = ?3 WHERE job_id = ?1 AND store = ?2 AND done_at IS NULL";

impl<B: Backend> Graph<B> {
    async fn read_outbox(&self, job_id: &str) -> Result<Option<Outbox>> {
        let rows = self
            .backend
            .query(
                "SELECT input, doomed, done_at FROM lineage_removals
                 WHERE job_id = ?1 AND store = ?2",
                &[job_id.into(), GRAPH_STORE.into()],
            )
            .await?;
        rows.first()
            .map(|row| {
                Ok(Outbox {
                    input: serde_json::from_str(&row.get_string(0)?)?,
                    doomed: serde_json::from_str(&row.get_string(1)?)?,
                    done: matches!(row.0.get(2), Some(Value::Int(_))),
                })
            })
            .transpose()
    }

    /// The nodes the key groups sealed.
    async fn nodes_of_keys(&self, key_ids: &[String]) -> Result<Vec<String>> {
        let nodes = column(
            &mut Pool(&self.backend),
            "SELECT node_id FROM node_keys WHERE key_id IN ({in})",
            key_ids,
        )
        .await?;
        Ok(listed(nodes))
    }

    /// What a purge of `target` is told to remove: the nodes its key groups sealed.
    async fn input_for(&self, target: &LineageTarget) -> Result<PurgeInput> {
        Ok(PurgeInput {
            node_ids: self.nodes_of_keys(&target.key_ids).await?,
            event_ids: sorted(&target.event_ids),
            key_ids: sorted(&target.key_ids),
        })
    }

    /// The sets of a job that has not purged yet, worked out from the tables.
    async fn resolve_fresh(&self, target: &LineageTarget) -> Result<Doomed> {
        let input = self.input_for(target).await?;
        resolve(&mut Pool(&self.backend), &input).await
    }

    /// The outbox row of the job, written if this is its first purge, so the
    /// sets are fixed before anything is deleted.
    async fn open_outbox(&self, target: &LineageTarget) -> Result<Outbox> {
        if let Some(outbox) = self.read_outbox(&target.job_id).await? {
            return Ok(outbox);
        }
        let input = self.input_for(target).await?;
        let doomed = resolve(&mut Pool(&self.backend), &input).await?;
        self.backend
            .execute(
                OUTBOX_INSERT_SQL,
                &[
                    target.job_id.as_str().into(),
                    GRAPH_STORE.into(),
                    serde_json::to_string(&input)?.into(),
                    serde_json::to_string(&doomed)?.into(),
                ],
            )
            .await?;
        Ok(Outbox {
            input,
            doomed,
            done: false,
        })
    }

    async fn vectors_of(&self, node_ids: &[String]) -> Result<Vec<String>> {
        let mut stored = Vec::new();
        for id in node_ids {
            if self.backend.vector_get(id, self.dims).await?.is_some() {
                stored.push(id.clone());
            }
        }
        Ok(stored)
    }

    async fn mark_done(&self, job_id: &str) -> Result<()> {
        let params = [job_id.into(), GRAPH_STORE.into(), self.clock.now().into()];
        self.backend.execute(OUTBOX_DONE_SQL, &params).await?;
        Ok(())
    }

    /// Deletes the SQL rows in one transaction with the removal logged first.
    async fn commit_purge(&self, job_id: &str, input: PurgeInput) -> Result<()> {
        let now = self.clock.now();
        let removal = event(
            [0; 32],
            (STORE_SOURCE, STORE_TRUST),
            now.0,
            now,
            LogPayload::LineageRemoval {
                job_id: job_id.to_owned(),
                store: GRAPH_STORE.to_owned(),
                event_ids: input.event_ids.clone(),
                key_ids: input.key_ids.clone(),
                node_ids: input.node_ids.clone(),
            },
        );
        let steps = steps_for(&removal.payload);
        let mut held = match &self.log {
            Some(log) => Some(HeldLog::acquire_for_write(log).await?),
            None => None,
        };
        let mut tx = self.backend.begin().await?;
        let params = [job_id.into(), GRAPH_STORE.into(), now.into()];
        if let Err(error) = tx.execute(OUTBOX_DONE_SQL, &params).await {
            abandon(tx).await;
            return Err(error);
        }
        match held.as_mut() {
            Some(held) => self.append_and_commit(tx, held, removal, &steps).await,
            None => {
                let vector_delete = self.backend.vector_delete_sql();
                let applied = apply_steps(&mut *tx, &steps, vector_delete).await;
                commit_or_abandon(tx, applied).await
            }
        }
    }

    /// The key group that sealed `node_id`, `None` when none is recorded.
    pub async fn key_of_node(&self, node_id: &NodeId) -> Result<Option<String>> {
        let rows = self
            .backend
            .query(
                "SELECT key_id FROM node_keys WHERE node_id = ?1",
                &[node_id.as_str().into()],
            )
            .await?;
        rows.first().map(|row| row.get_string(0)).transpose()
    }

    /// `backfill_node_keys` for a store that has no key group recorded yet.
    pub async fn backfill_node_keys_if_empty(&self) -> Result<usize> {
        let held = self
            .backend
            .query("SELECT 1 FROM node_keys LIMIT 1", &[])
            .await?;
        if held.is_empty() {
            self.backfill_node_keys().await
        } else {
            Ok(0)
        }
    }

    /// Fills `node_keys` and `derived_keys` from the headers of the sealed
    /// records in the log, for a store that was built before they existed.
    /// Returns how many records named a key group. A record already there is
    /// left as it is, so a rerun changes nothing.
    pub async fn backfill_node_keys(&self) -> Result<usize> {
        let Some(log) = &self.log else {
            return Ok(0);
        };
        let held = HeldLog::acquire(log).await?;
        let Some(head) = held.head()? else {
            return Ok(0);
        };
        let reader = held.reader()?;
        let mut records = reader.scan_through(None, head);
        let mut sealed = 0;
        while let Some(record) = records.next().await {
            let record = record?;
            if record.row_keys.is_empty() {
                continue;
            }
            let mut tx = self.backend.begin().await?;
            let recorded =
                record_row_keys(&mut *tx, &record.event.event_id, &record.row_keys).await;
            commit_or_abandon(tx, recorded).await?;
            sealed += 1;
        }
        tracing::info!(sealed, "node_keys backfilled from the log headers");
        Ok(sealed)
    }
}

fn sorted(values: &[String]) -> Vec<String> {
    values
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[async_trait]
impl<B: Backend> LineageStore for Graph<B> {
    fn name(&self) -> &str {
        GRAPH_STORE
    }

    async fn find(&self, target: &LineageTarget) -> Result<LineageHits> {
        let outbox = self.read_outbox(&target.job_id).await?;
        if outbox.as_ref().is_some_and(|outbox| outbox.done) {
            return Ok(LineageHits::default());
        }
        let doomed = match outbox {
            Some(outbox) => outbox.doomed,
            None => self.resolve_fresh(target).await?,
        };
        let mut rows: BTreeMap<String, usize> = matching(&mut Pool(&self.backend), &doomed, false)
            .await?
            .into_iter()
            .map(|(table, ids)| (table.to_owned(), ids.len()))
            .collect();
        let vectors = self.vectors_of(&doomed.nodes).await?.len();
        if vectors > 0 {
            rows.insert(VECTOR_TABLE.to_owned(), vectors);
        }
        Ok(LineageHits { rows })
    }

    async fn purge(&self, target: &LineageTarget) -> Result<PurgeReport> {
        let outbox = self.open_outbox(target).await?;
        if outbox.done {
            return Ok(PurgeReport::default());
        }
        let hits = self.find(target).await?;
        if hits.total() == 0 {
            self.mark_done(&target.job_id).await?;
            return Ok(PurgeReport::default());
        }
        // The vectors go first and on their own: the backend's delete takes the
        // write lock a transaction would already hold.
        for id in &outbox.input.node_ids {
            self.backend.vector_delete(id).await?;
        }
        self.commit_purge(&target.job_id, outbox.input).await?;
        self.reload_quant_index().await?;
        tracing::info!(removed = hits.total(), job = %target.job_id, "lineage purged");
        Ok(PurgeReport {
            removed: hits.total(),
        })
    }

    async fn verify(&self, target: &LineageTarget) -> Result<Vec<Residual>> {
        let doomed = match self.read_outbox(&target.job_id).await? {
            Some(outbox) => outbox.doomed,
            None => self.resolve_fresh(target).await?,
        };
        let mut pool = Pool(&self.backend);
        let mut residuals: Vec<(String, String)> = matching(&mut pool, &doomed, true)
            .await?
            .into_iter()
            .flat_map(|(table, ids)| ids.into_iter().map(move |id| (table.to_owned(), id)))
            .collect();
        residuals.extend(
            dangling_closers(&mut pool, &doomed)
                .await?
                .into_iter()
                .map(|id| (DANGLING_CLOSER.to_owned(), id)),
        );
        residuals.extend(
            self.vectors_of(&doomed.nodes)
                .await?
                .into_iter()
                .map(|id| (VECTOR_TABLE.to_owned(), id)),
        );
        Ok(residuals
            .into_iter()
            .map(|(kind, id)| Residual {
                store: GRAPH_STORE.to_owned(),
                kind,
                id,
            })
            .collect())
    }
}

/// The backend's vector table, which the backend deletes from.
const VECTOR_TABLE: &str = "node_vectors";

/// A surviving fact version still marked closed by a deleted one.
const DANGLING_CLOSER: &str = "facts.closed_by";
