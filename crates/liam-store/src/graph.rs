// SPDX-License-Identifier: MIT OR Apache-2.0
//! The graph handle, generic over a `Backend`. Shared logic (write, read, GC,
//! clustering) routes through the backend trait; only vector storage and search
//! differ per backend.
//!
//! Visibility uses a full bitemporal predicate: a node is visible "as of" an
//! instant T when it was recorded before T, not yet superseded at T, and true
//! in the world at T. Passing a past T yields point-in-time recall.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::backend::Backend;
use crate::clock::{Clock, SystemClock};
use crate::error::{Error, Result};
use crate::ids::{EdgeId, Millis, NodeId, FOREVER};
use crate::schema::schema;
use crate::types::{
    Change, ExplainedHit, GcReport, GraphConfig, Hit, NewEdge, NewNode, Query, RetentionPolicy,
};
use crate::value::{Row, Value};

/// Bitemporal "live at T" predicate over an aliased nodes table. `?{t}` is T.
fn live_at(alias: &str, t: usize) -> String {
    format!(
        "{alias}.tx_from <= ?{t} AND {alias}.tx_to > ?{t} \
         AND {alias}.valid_from <= ?{t} AND {alias}.valid_until > ?{t}"
    )
}

fn opt_text(v: Option<String>) -> Value {
    v.map(Value::Text).unwrap_or(Value::Null)
}

fn decay_factor(valid_from: Millis, now: Millis, half_life: Option<Millis>) -> f64 {
    match half_life {
        Some(h) if h.0 > 0 => {
            let age = (now.0 - valid_from.0).max(0) as f64;
            0.5f64.powf(age / h.0 as f64)
        }
        _ => 1.0,
    }
}

struct Candidate {
    id: NodeId,
    kind: String,
    label: String,
    content: String,
    attributes: serde_json::Value,
    confidence: f64,
    valid_from: Millis,
}

pub struct Graph<B: Backend> {
    backend: B,
    clock: Arc<dyn Clock>,
    dims: usize,
    rrf_k: f64,
    expansion_weight: f64,
}

impl<B: Backend> Graph<B> {
    pub async fn open(path: &str, config: GraphConfig) -> Result<Self> {
        Self::open_with_clock(path, config, Arc::new(SystemClock)).await
    }

    pub async fn open_with_clock(
        path: &str,
        config: GraphConfig,
        clock: Arc<dyn Clock>,
    ) -> Result<Self> {
        let backend = B::open(path).await?;
        let mut ddl = schema(&config);
        ddl.push_str(&backend.vector_ddl(config.embedding_dims));
        backend.execute_batch(&ddl).await?;
        // `schema()` above is entirely `CREATE TABLE IF NOT EXISTS`, so a
        // database that already existed before `producer` was added never ran
        // it and does not have the column. This guarded ALTER TABLE is what
        // gives it one; a fresh database already has it from the DDL and this
        // call then no-ops. Both paths end in the same shape.
        crate::migrate::add_column_if_missing(
            &backend,
            "nodes",
            "producer",
            "TEXT NOT NULL DEFAULT 'unknown'",
        )
        .await?;
        Ok(Self {
            backend,
            clock,
            dims: config.embedding_dims,
            rrf_k: config.rrf_k,
            expansion_weight: config.expansion_weight,
        })
    }

    // ---- write ----

    pub async fn insert(&self, node: NewNode) -> Result<NodeId> {
        let id = NodeId::new();
        let now = self.clock.now();
        self.write_node(&id, &node, now).await?;
        if let Some(embedding) = node.embedding.as_deref() {
            self.check_dims(embedding)?;
            self.backend.vector_upsert(id.as_str(), embedding).await?;
        }
        Ok(id)
    }

    /// Insert, or supersede a competing live node with the same subject and
    /// scope. This makes contradiction handling automatic: the caller sets a
    /// subject, the library closes the prior version and links to it.
    pub async fn upsert_by(&self, node: NewNode) -> Result<NodeId> {
        let subject = match node.subject.clone() {
            Some(s) => s,
            None => return self.insert(node).await,
        };
        match self
            .find_live_by_subject(&subject, node.scope.as_deref())
            .await?
        {
            Some(existing) => self.supersede(&existing, node).await,
            None => self.insert(node).await,
        }
    }

    /// Close the old node in transaction time, insert the new one, link them
    /// with a reserved `supersedes` edge.
    pub async fn supersede(&self, old: &NodeId, node: NewNode) -> Result<NodeId> {
        let now = self.clock.now();
        if !self.exists_as_of(old, now).await? {
            return Err(Error::NodeNotFound(old.as_str().to_string()));
        }
        let new_id = NodeId::new();
        let embedding = node.embedding.clone();
        let (node_sql, node_params) = self.node_insert(&new_id, &node, now)?;

        let statements = vec![
            (
                "UPDATE nodes SET tx_to = ?1 WHERE id = ?2 AND tx_to = ?3".to_string(),
                vec![now.into(), old.as_str().into(), FOREVER.into()],
            ),
            (node_sql, node_params),
            (
                "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
                 VALUES (?1, ?2, ?3, ?4, '{}', ?5, ?6)"
                    .to_string(),
                vec![
                    EdgeId::new().as_str().into(),
                    new_id.as_str().into(),
                    old.as_str().into(),
                    crate::types::relation::SUPERSEDES.into(),
                    now.into(),
                    FOREVER.into(),
                ],
            ),
        ];
        self.backend.execute_atomic(&statements).await?;

        if let Some(embedding) = embedding.as_deref() {
            self.check_dims(embedding)?;
            self.backend
                .vector_upsert(new_id.as_str(), embedding)
                .await?;
        }
        Ok(new_id)
    }

    pub async fn link(&self, edge: NewEdge) -> Result<EdgeId> {
        let id = EdgeId::new();
        let now = self.clock.now();
        let attrs = serde_json::to_string(&edge.attributes)?;
        self.backend
            .execute(
                "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                &[
                    id.as_str().into(),
                    edge.src.as_str().into(),
                    edge.dst.as_str().into(),
                    edge.kind.into(),
                    attrs.into(),
                    now.into(),
                    FOREVER.into(),
                ],
            )
            .await?;
        Ok(id)
    }

    fn node_insert(
        &self,
        id: &NodeId,
        node: &NewNode,
        now: Millis,
    ) -> Result<(String, Vec<Value>)> {
        let attrs = serde_json::to_string(&node.attributes)?;
        let sql = "INSERT INTO nodes
             (id, kind, label, content, producer, attributes, scope, subject, confidence,
              valid_from, valid_until, tx_from, tx_to)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"
            .to_string();
        let params = vec![
            id.as_str().into(),
            node.kind.clone().into(),
            node.label.clone().into(),
            node.content.clone().into(),
            node.producer.clone().into(),
            attrs.into(),
            opt_text(node.scope.clone()),
            opt_text(node.subject.clone()),
            Value::Real(node.confidence),
            node.valid_from.unwrap_or(now).into(),
            FOREVER.into(),
            now.into(),
            FOREVER.into(),
        ];
        Ok((sql, params))
    }

    async fn write_node(&self, id: &NodeId, node: &NewNode, now: Millis) -> Result<()> {
        let (sql, params) = self.node_insert(id, node, now)?;
        self.backend.execute(&sql, &params).await?;
        Ok(())
    }

    // ---- read ----

    pub async fn query(&self, q: &Query) -> Result<Vec<Hit>> {
        Ok(self
            .query_core(q)
            .await?
            .into_iter()
            .map(|e| e.hit)
            .collect())
    }

    /// Like `query`, but each result carries the score components that produced
    /// it: lexical rank, vector rank, fused RRF, confidence, decay, expansion.
    pub async fn query_explained(&self, q: &Query) -> Result<Vec<ExplainedHit>> {
        self.query_core(q).await
    }

    async fn query_core(&self, q: &Query) -> Result<Vec<ExplainedHit>> {
        let now = q.as_of.unwrap_or_else(|| self.clock.now());
        let pool = q.k.max(1) * 3;
        let scope = q.scope.as_deref();
        let kind = q.kind.as_deref();

        let lexical = match q.text.as_deref() {
            Some(text) => self.lexical(text, kind, scope, now, pool).await?,
            None => Vec::new(),
        };
        let vector = match q.embedding.as_deref() {
            Some(embedding) => {
                self.backend
                    .vector_search(embedding, pool, kind, scope, now)
                    .await?
            }
            None => Vec::new(),
        };

        let lex_rank: HashMap<String, usize> = lexical
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str().to_string(), i))
            .collect();
        let vec_rank: HashMap<String, usize> = vector
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str().to_string(), i))
            .collect();

        let mut rrf: HashMap<String, f64> = HashMap::new();
        for (i, id) in lexical.iter().chain(vector.iter()).enumerate() {
            let rank = if i < lexical.len() {
                i
            } else {
                i - lexical.len()
            };
            *rrf.entry(id.as_str().to_string()).or_insert(0.0) +=
                1.0 / (self.rrf_k + rank as f64 + 1.0);
        }

        // Seeds: top of the fused list. Expand their neighbours, down-weighted.
        let mut seeds: Vec<(String, f64)> = rrf.iter().map(|(k, v)| (k.clone(), *v)).collect();
        seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        seeds.truncate(q.k);

        let floor = 1.0 / (self.rrf_k + pool as f64 + 1.0);
        let mut candidates: HashMap<String, (f64, bool)> = HashMap::new();
        for (id, s) in &rrf {
            candidates.insert(id.clone(), (*s, false));
        }
        let mut expanded: HashSet<String> = HashSet::new();
        for (id, _) in &seeds {
            for n in self.neighbors(&NodeId::from_raw(id.clone()), now).await? {
                expanded.insert(n.as_str().to_string());
            }
        }
        for id in expanded {
            candidates.entry(id).or_insert((floor, true));
        }

        let ids: Vec<NodeId> = candidates
            .keys()
            .map(|s| NodeId::from_raw(s.clone()))
            .collect();
        let rows = self.fetch_candidates(&ids, now, scope, kind).await?;

        let mut out = Vec::with_capacity(rows.len());
        for c in rows {
            let (base, is_expanded) = candidates
                .get(c.id.as_str())
                .copied()
                .unwrap_or((floor, true));
            let decay = decay_factor(c.valid_from, now, q.half_life);
            let weight = if is_expanded {
                self.expansion_weight
            } else {
                1.0
            };
            let score = base * c.confidence * decay * weight;
            out.push(ExplainedHit {
                hit: Hit {
                    id: c.id.clone(),
                    kind: c.kind,
                    label: c.label,
                    content: c.content,
                    attributes: c.attributes,
                    score,
                },
                lexical_rank: lex_rank.get(c.id.as_str()).copied(),
                vector_rank: vec_rank.get(c.id.as_str()).copied(),
                rrf: base,
                confidence: c.confidence,
                decay,
                valid_from: c.valid_from,
                expanded: is_expanded,
            });
        }
        out.sort_by(|a, b| {
            b.hit
                .score
                .partial_cmp(&a.hit.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(q.k);
        Ok(out)
    }

    pub async fn neighbors(&self, seed: &NodeId, as_of: Millis) -> Result<Vec<NodeId>> {
        let rows = self
            .backend
            .query(
                "SELECT dst FROM edges WHERE src = ?1 AND tx_from <= ?2 AND tx_to > ?2
                 UNION SELECT src FROM edges WHERE dst = ?1 AND tx_from <= ?2 AND tx_to > ?2",
                &[seed.as_str().into(), as_of.into()],
            )
            .await?;
        ids_from(&rows)
    }

    async fn lexical(
        &self,
        text: &str,
        kind: Option<&str>,
        scope: Option<&str>,
        now: Millis,
        k: usize,
    ) -> Result<Vec<NodeId>> {
        // FTS5 parses the MATCH value as its own query expression, so a raw
        // question (`?`, apostrophes, bare operators) is a syntax error that
        // aborts the whole hybrid query. Sanitize to a term-quoted literal.
        let match_query = fts5_query(text);
        if !match_query.chars().any(|c| c.is_alphanumeric()) {
            // Nothing searchable (empty or all-punctuation); skip the lexical
            // leg so the vector leg, if present, still runs.
            return Ok(Vec::new());
        }
        let mut params: Vec<Value> = vec![match_query.into(), now.into(), (k as i64).into()];
        let mut filters = String::new();
        let mut next = 4;
        if let Some(kind) = kind {
            filters.push_str(&format!(" AND n.kind = ?{next}"));
            params.push(kind.into());
            next += 1;
        }
        if let Some(scope) = scope {
            filters.push_str(&format!(" AND n.scope = ?{next}"));
            params.push(scope.into());
        }
        let sql = format!(
            "SELECT n.id FROM nodes_fts
             JOIN nodes n ON n.rowid = nodes_fts.rowid
             WHERE nodes_fts MATCH ?1 AND {live}{filters}
             ORDER BY bm25(nodes_fts) LIMIT ?3",
            live = live_at("n", 2),
        );
        let rows = self.backend.query(&sql, &params).await?;
        ids_from(&rows)
    }

    /// Fetch the live rows for a set of candidate ids in one query per chunk
    /// (missing or expired ids simply don't come back). Chunked to stay under
    /// SQLite's 999 bound-parameter limit; order is not meaningful, callers
    /// look candidates up by id.
    async fn fetch_candidates(
        &self,
        ids: &[NodeId],
        now: Millis,
        scope: Option<&str>,
        kind: Option<&str>,
    ) -> Result<Vec<Candidate>> {
        const CHUNK: usize = 512;
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            // `?1` is the live-at instant; `?2..` are the ids; the scope and
            // kind filters, if any, take the parameters after those.
            let mut params: Vec<Value> = vec![now.into()];
            let placeholders = chunk
                .iter()
                .enumerate()
                .map(|(i, id)| {
                    params.push(id.as_str().into());
                    format!("?{}", i + 2)
                })
                .collect::<Vec<_>>()
                .join(",");
            // Both filters are applied HERE, not only on the retrieval paths:
            // graph expansion adds a seed's neighbours to the candidate set
            // without consulting the query at all, so a neighbour of another
            // kind or scope would otherwise ride along and break the filter the
            // caller asked for.
            let mut filters = String::new();
            for (column, value) in [("scope", scope), ("kind", kind)] {
                if let Some(v) = value {
                    params.push(v.into());
                    // Just-pushed value sits at the 1-based position `len()`.
                    filters.push_str(&format!(" AND {column} = ?{}", params.len()));
                }
            }
            let sql = format!(
                "SELECT id, kind, label, content, attributes, confidence, valid_from
                 FROM nodes WHERE id IN ({placeholders}) AND {live}{filters}",
                live = live_at("nodes", 1),
            );
            let rows = self.backend.query(&sql, &params).await?;
            for row in &rows {
                out.push(Candidate {
                    id: NodeId::from_raw(row.get_string(0)?),
                    kind: row.get_string(1)?,
                    label: row.get_string(2)?,
                    content: row.get_string(3)?,
                    attributes: serde_json::from_str(&row.get_string(4)?)?,
                    confidence: row_f64(row, 5),
                    valid_from: Millis(row.get_i64(6)?),
                });
            }
        }
        Ok(out)
    }

    async fn find_live_by_subject(
        &self,
        subject: &str,
        scope: Option<&str>,
    ) -> Result<Option<NodeId>> {
        let now = self.clock.now();
        let mut params: Vec<Value> = vec![subject.into(), now.into()];
        let scope_filter = match scope {
            Some(s) => {
                params.push(s.into());
                " AND scope = ?3"
            }
            None => " AND scope IS NULL",
        };
        // If two live nodes ever share a subject+scope, supersede the newest
        // deterministically (tie-break by id) rather than an arbitrary row.
        let sql = format!(
            "SELECT id FROM nodes WHERE subject = ?1 AND {live}{scope_filter}
             ORDER BY tx_from DESC, id DESC LIMIT 1",
            live = live_at("nodes", 2),
        );
        let rows = self.backend.query(&sql, &params).await?;
        Ok(rows
            .first()
            .map(|r| NodeId::from_raw(r.get_string(0).unwrap_or_default())))
    }

    async fn exists_as_of(&self, id: &NodeId, as_of: Millis) -> Result<bool> {
        let sql = format!(
            "SELECT 1 FROM nodes WHERE id = ?1 AND {live}",
            live = live_at("nodes", 2)
        );
        let rows = self
            .backend
            .query(&sql, &[id.as_str().into(), as_of.into()])
            .await?;
        Ok(!rows.is_empty())
    }

    fn check_dims(&self, embedding: &[f32]) -> Result<()> {
        if embedding.len() == self.dims {
            return Ok(());
        }
        Err(Error::Dimension {
            expected: self.dims,
            got: embedding.len(),
        })
    }

    // ---- change cursor ----

    /// Nodes recorded or closed strictly after `cursor`, for incremental work
    /// (rebuild only new vectors, recompute only changed communities).
    pub async fn changes_since(&self, cursor: Millis) -> Result<Vec<Change>> {
        let rows = self
            .backend
            .query(
                "SELECT id, tx_from, tx_to FROM nodes
                 WHERE tx_from > ?1 OR (tx_to > ?1 AND tx_to < ?2)
                 ORDER BY tx_from",
                &[cursor.into(), FOREVER.into()],
            )
            .await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            let tx_to = row.get_i64(2)?;
            out.push(Change {
                id: NodeId::from_raw(row.get_string(0)?),
                tx_from: Millis(row.get_i64(1)?),
                closed: tx_to != FOREVER.0 && tx_to > cursor.0,
            });
        }
        Ok(out)
    }

    // ---- retention ----

    pub async fn gc(&self, policy: &RetentionPolicy) -> Result<GcReport> {
        let now = self.clock.now();
        let mut nodes_removed = 0u64;
        for rule in &policy.rules {
            let cutoff = now.0 - rule.max_age.0;
            nodes_removed += self
                .backend
                .execute(
                    "DELETE FROM nodes WHERE kind = ?1 AND valid_from < ?2",
                    &[rule.kind.as_str().into(), cutoff.into()],
                )
                .await?;
        }
        let edges_removed = self
            .backend
            .execute(
                "DELETE FROM edges
                 WHERE src NOT IN (SELECT id FROM nodes)
                    OR dst NOT IN (SELECT id FROM nodes)",
                &[],
            )
            .await?;
        self.backend.vector_sweep_orphans().await?;
        if policy.reclaim {
            self.backend
                .execute("PRAGMA incremental_vacuum", &[])
                .await?;
        }
        let report = GcReport {
            nodes_removed,
            edges_removed,
        };
        tracing::info!(?report, "gc swept");
        Ok(report)
    }
}

#[cfg(feature = "cluster")]
impl<B: Backend> Graph<B> {
    pub async fn recompute_communities(&self) -> Result<usize> {
        use crate::cluster::{detect, Edge};

        let mut index: HashMap<String, usize> = HashMap::new();
        let mut labels: Vec<String> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();

        let rows = self
            .backend
            .query(
                "SELECT src, dst FROM edges WHERE tx_to = ?1",
                &[FOREVER.into()],
            )
            .await?;
        for row in &rows {
            let u = intern(&mut index, &mut labels, row.get_string(0)?);
            let v = intern(&mut index, &mut labels, row.get_string(1)?);
            edges.push(Edge(u, v));
        }

        let assignment = detect(labels.len(), &edges);
        let now = self.clock.now();
        let mut statements: Vec<(String, Vec<Value>)> =
            vec![("DELETE FROM node_community".to_string(), Vec::new())];
        for (i, community) in assignment.iter().enumerate() {
            statements.push((
                "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, ?2, ?3)"
                    .to_string(),
                vec![
                    labels[i].as_str().into(),
                    (*community as i64).into(),
                    now.into(),
                ],
            ));
        }
        self.backend.execute_atomic(&statements).await?;

        let count = assignment.iter().collect::<HashSet<_>>().len();
        tracing::info!(
            nodes = labels.len(),
            communities = count,
            "communities recomputed"
        );
        Ok(count)
    }

    pub async fn communities(&self) -> Result<Vec<(NodeId, i64)>> {
        let rows = self
            .backend
            .query(
                "SELECT node_id, community FROM node_community ORDER BY community, node_id",
                &[],
            )
            .await?;
        let mut out = Vec::with_capacity(rows.len());
        for row in &rows {
            out.push((NodeId::from_raw(row.get_string(0)?), row.get_i64(1)?));
        }
        Ok(out)
    }
}

fn ids_from(rows: &[Row]) -> Result<Vec<NodeId>> {
    rows.iter()
        .map(|r| Ok(NodeId::from_raw(r.get_string(0)?)))
        .collect()
}

fn row_f64(row: &Row, i: usize) -> f64 {
    match row.0.get(i) {
        Some(Value::Real(v)) => *v,
        Some(Value::Int(v)) => *v as f64,
        _ => 1.0,
    }
}

/// Turn arbitrary user text into a safe FTS5 MATCH string. FTS5 parses the
/// *value* of a MATCH parameter as its own query language, so a natural-language
/// question (`?`, `'`, `:`, leading `-`, bare `AND`/`OR`/`NOT`, parentheses)
/// raises a syntax error and aborts the whole hybrid query. Wrapping each
/// whitespace token in double quotes (embedded `"` doubled) makes every operator
/// character match literally; the tokenizer still stems inside each quoted
/// phrase, and single-token quoting is equivalent to a bare term under the
/// tokenizer, so well-formed single-word queries are unchanged.
///
/// The quoted terms are joined with `OR`, not a space. A space is implicit AND
/// in FTS5, which would require every word of the query to appear in a
/// document. A stored memory is a statement; a question shares only its
/// content words with the statement that answers it and adds interrogatives
/// and auxiliaries ("when", "does") the statement never has, so AND-joining
/// made the lexical arm return zero hits for any natural-language question.
/// ORing recovers that recall: BM25 still ranks by inverse document frequency,
/// so a rare shared term dominates the score while "the" or "when" contribute
/// almost nothing, and RRF plus the reranker re-sort the fused result anyway.
fn fts5_query(text: &str) -> String {
    text.split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

#[cfg(feature = "cluster")]
fn intern(index: &mut HashMap<String, usize>, labels: &mut Vec<String>, id: String) -> usize {
    if let Some(&i) = index.get(&id) {
        return i;
    }
    let i = labels.len();
    labels.push(id.clone());
    index.insert(id, i);
    i
}

#[cfg(all(test, feature = "backend-libsql"))]
mod tests {
    use super::*;
    use crate::clock::FixedClock;
    use crate::DefaultGraph;
    use tempfile::TempDir;

    async fn graph_at(t: Millis) -> DefaultGraph {
        let clock = Arc::new(FixedClock::new(t));
        DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock)
            .await
            .unwrap()
    }

    /// `:memory:` cannot stand in for the rest of S1: WAL is a no-op on an
    /// in-memory database, and every `:memory:` connection is its own private
    /// database, so a connection pool over it would silently fan out to
    /// separate empty stores. This opens the same graph on a real file so
    /// later tests exercise real single-file database semantics. The
    /// `TempDir` guard is returned alongside the graph, not dropped here,
    /// because dropping it deletes the database file out from under a graph
    /// that still holds it open.
    async fn file_graph_at(t: Millis) -> (TempDir, DefaultGraph) {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("graph.db");
        let clock = Arc::new(FixedClock::new(t));
        let g = DefaultGraph::open_with_clock(
            path.to_str().expect("temp path is valid utf-8"),
            GraphConfig::new(8),
            clock,
        )
        .await
        .unwrap();
        (dir, g)
    }

    #[tokio::test]
    async fn file_backed_graph_inserts_and_queries() {
        // Arrange
        let (_dir, g) = file_graph_at(Millis(1000)).await;

        // Act
        g.insert(NewNode::now("decision", "Use libSQL", "single file"))
            .await
            .unwrap();
        let hits = g.query(&Query::text("libSQL")).await.unwrap();

        // Assert
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].label, "Use libSQL");
    }

    #[tokio::test]
    async fn file_backed_graph_removes_temp_dir_when_guard_drops() {
        // Arrange
        let (dir, g) = file_graph_at(Millis(1000)).await;
        let dir_path = dir.path().to_path_buf();

        // Act
        drop(g);
        drop(dir);

        // Assert
        assert!(!dir_path.exists());
    }

    #[tokio::test]
    async fn insert_then_query() {
        let g = graph_at(Millis(1000)).await;
        g.insert(NewNode::now("decision", "Use libSQL", "single file"))
            .await
            .unwrap();
        let hits = g.query(&Query::text("libSQL")).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].label, "Use libSQL");
    }

    #[tokio::test]
    async fn kind_filter_holds_for_graph_expanded_neighbours() {
        // Arrange: a matching `decision` linked to an `episode` that the query
        // text does not match. The episode enters the candidate set only through
        // graph expansion off the seed.
        let g = graph_at(Millis(1000)).await;
        let seed = g
            .insert(NewNode::now(
                "decision",
                "Rollout",
                "zorbnax rollout approved",
            ))
            .await
            .unwrap();
        let neighbour = g
            .insert(NewNode::now(
                "episode",
                "Chatter",
                "unrelated standup notes",
            ))
            .await
            .unwrap();
        g.link(NewEdge::new(&seed, &neighbour, "mentions"))
            .await
            .unwrap();

        // Act
        let hits = g
            .query(&Query::text("zorbnax rollout").with_kind("decision"))
            .await
            .unwrap();

        // Assert: `with_kind` is a filter on the result set, not merely on the
        // retrieval paths, so an expanded neighbour of another kind must not ride
        // along.
        let kinds: Vec<&str> = hits.iter().map(|h| h.kind.as_str()).collect();
        assert!(
            hits.iter().all(|h| h.kind == "decision"),
            "kind filter leaked expanded neighbours: {kinds:?}"
        );
    }

    #[tokio::test]
    async fn scope_filter_holds_for_graph_expanded_neighbours() {
        // Arrange: same shape as the kind test, partitioned by scope instead.
        // This one guards behaviour that already held, so the shared filter
        // refactor cannot regress it silently.
        let g = graph_at(Millis(1000)).await;
        let seed = g
            .insert(
                NewNode::now("decision", "Rollout", "zorbnax rollout approved")
                    .with_scope("proj-a"),
            )
            .await
            .unwrap();
        let neighbour = g
            .insert(NewNode::now("decision", "Chatter", "unrelated notes").with_scope("proj-b"))
            .await
            .unwrap();
        g.link(NewEdge::new(&seed, &neighbour, "mentions"))
            .await
            .unwrap();

        // Act
        let hits = g
            .query(&Query::text("zorbnax rollout").with_scope("proj-a"))
            .await
            .unwrap();

        // Assert
        let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
        assert!(
            hits.iter().all(|h| h.label != "Chatter"),
            "scope filter leaked expanded neighbours: {labels:?}"
        );
        assert!(
            hits.iter().any(|h| h.label == "Rollout"),
            "in-scope match missing: {labels:?}"
        );
    }

    #[tokio::test]
    async fn as_of_recovers_superseded_history() {
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let old = g
            .insert(NewNode::now("decision", "Deno", "runtime").with_valid_from(Millis(1000)))
            .await
            .unwrap();
        clock.set(Millis(2000));
        g.supersede(
            &old,
            NewNode::now("decision", "Rust", "runtime").with_valid_from(Millis(2000)),
        )
        .await
        .unwrap();

        let now_hits = g.query(&Query::text("runtime")).await.unwrap();
        assert!(now_hits.iter().any(|h| h.label == "Rust"));
        assert!(now_hits.iter().all(|h| h.label != "Deno"));

        let past = g
            .query(&Query::text("runtime").with_as_of(Millis(1500)))
            .await
            .unwrap();
        assert!(past.iter().any(|h| h.label == "Deno"));
        assert!(past.iter().all(|h| h.label != "Rust"));
    }

    #[tokio::test]
    async fn as_of_recovers_superseded_history_vector() {
        // Same contract as the text-only test above, but exercised purely
        // through the vector channel: the vector path must obey the same
        // "live at T" predicate as the lexical path.
        let e = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let old = g
            .insert(
                NewNode::now("decision", "Deno", "runtime")
                    .with_valid_from(Millis(1000))
                    .with_embedding(e.clone()),
            )
            .await
            .unwrap();
        clock.set(Millis(2000));
        g.supersede(
            &old,
            NewNode::now("decision", "Rust", "runtime")
                .with_valid_from(Millis(2000))
                .with_embedding(e.clone()),
        )
        .await
        .unwrap();

        let now_hits = g.query(&Query::vector(e.clone())).await.unwrap();
        assert!(
            now_hits.iter().any(|h| h.label == "Rust"),
            "current: Rust live"
        );
        assert!(
            now_hits.iter().all(|h| h.label != "Deno"),
            "current: Deno superseded"
        );

        let past = g
            .query(&Query::vector(e.clone()).with_as_of(Millis(1500)))
            .await
            .unwrap();
        assert!(
            past.iter().any(|h| h.label == "Deno"),
            "as-of 1500: Deno was live"
        );
        assert!(
            past.iter().all(|h| h.label != "Rust"),
            "as-of 1500: Rust not yet recorded"
        );
    }

    #[tokio::test]
    async fn upsert_by_supersedes_same_subject() {
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        g.upsert_by(NewNode::now("fact", "v1", "price is 10").with_subject("price"))
            .await
            .unwrap();
        clock.set(Millis(2000));
        g.upsert_by(NewNode::now("fact", "v2", "price is 20").with_subject("price"))
            .await
            .unwrap();

        let hits = g.query(&Query::text("price")).await.unwrap();
        assert_eq!(hits.len(), 1, "only the current version is live");
        assert_eq!(hits[0].label, "v2");
    }

    #[tokio::test]
    async fn upsert_by_supersedes_newest_competitor_deterministically() {
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        // Two live nodes share a subject (via `insert`, which does not dedup).
        g.insert(
            NewNode::now("fact", "old", "price 10")
                .with_subject("price")
                .with_valid_from(Millis(1000)),
        )
        .await
        .unwrap();
        clock.set(Millis(2000));
        g.insert(
            NewNode::now("fact", "newer", "price 15")
                .with_subject("price")
                .with_valid_from(Millis(2000)),
        )
        .await
        .unwrap();
        // upsert_by supersedes the newest competitor, deterministically.
        clock.set(Millis(3000));
        g.upsert_by(
            NewNode::now("fact", "newest", "price 20")
                .with_subject("price")
                .with_valid_from(Millis(3000)),
        )
        .await
        .unwrap();

        let hits = g.query(&Query::text("price").with_k(10)).await.unwrap();
        let labels: Vec<&str> = hits.iter().map(|h| h.label.as_str()).collect();
        assert!(labels.contains(&"newest"), "new version is live");
        assert!(
            !labels.contains(&"newer"),
            "newest competitor was superseded"
        );
        assert!(labels.contains(&"old"), "the older competitor is untouched");
    }

    #[tokio::test]
    async fn query_returns_all_matching_candidates() {
        // Guards the batched candidate fetch: several live nodes match one
        // query and all must come back.
        let g = graph_at(Millis(1000)).await;
        for i in 0..5 {
            g.insert(NewNode::now(
                "fact",
                format!("n{i}"),
                "shared keyword topic",
            ))
            .await
            .unwrap();
        }
        let hits = g.query(&Query::text("keyword").with_k(10)).await.unwrap();
        assert_eq!(hits.len(), 5);
    }

    #[tokio::test]
    async fn gc_ages_out_by_kind() {
        const DAY: i64 = 86_400_000;
        let g = graph_at(Millis(100 * DAY)).await;
        g.insert(NewNode::now("episode", "old", "x").with_valid_from(Millis(10 * DAY)))
            .await
            .unwrap();
        g.insert(NewNode::now("decision", "keep", "y").with_valid_from(Millis(10 * DAY)))
            .await
            .unwrap();
        let report = g
            .gc(&RetentionPolicy::keep("episode", Millis::days(30)).without_reclaim())
            .await
            .unwrap();
        assert_eq!(report.nodes_removed, 1);
    }

    #[test]
    fn new_node_entity_sets_kind_label_subject() {
        let n = NewNode::entity("person", "  Ada Lovelace ");
        assert_eq!(n.kind, "person");
        assert_eq!(n.label, "  Ada Lovelace ");
        assert_eq!(n.subject.as_deref(), Some("ada lovelace"));
    }

    #[tokio::test]
    async fn entity_mentions_edge_round_trips() {
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let person = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
        let fact = g
            .insert(NewNode::now(
                "fact",
                "note",
                "Ada wrote the first algorithm",
            ))
            .await
            .unwrap();
        g.link(NewEdge::new(
            &person,
            &fact,
            crate::types::relation::MENTIONS,
        ))
        .await
        .unwrap();

        // `Graph::neighbors` traverses edges in either direction without
        // filtering by type, so it can't isolate a single relation. Assert
        // directly against the backend that the MENTIONS edge row exists.
        let rows = g
            .backend
            .query(
                "SELECT dst FROM edges WHERE src = ?1 AND dst = ?2 AND type = ?3",
                &[
                    person.as_str().into(),
                    fact.as_str().into(),
                    crate::types::relation::MENTIONS.into(),
                ],
            )
            .await
            .unwrap();
        assert_eq!(ids_from(&rows).unwrap(), vec![fact]);
    }

    #[tokio::test]
    async fn query_explained_carries_valid_from() {
        // The daemon's `ask` tool renders `valid_from` as a date, so
        // `ExplainedHit` must carry the raw instant, not just the decay
        // computed from it.
        // Arrange
        let g = graph_at(Millis(5000)).await;
        g.insert(NewNode::now("fact", "label", "body").with_valid_from(Millis(1234)))
            .await
            .unwrap();

        // Act
        let hits = g.query_explained(&Query::text("body")).await.unwrap();

        // Assert
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].valid_from, Millis(1234));
    }

    #[tokio::test]
    async fn lexical_query_tolerates_question_punctuation() {
        // FTS5 parses a raw MATCH value as its own query language; a natural
        // question's trailing `?` or an embedded apostrophe previously raised
        // "fts5: syntax error" and aborted the whole hybrid query. The terms
        // must still match. Regression pin for the ask tool's primary input
        // shape. Quoting is what protects this, so it must hold regardless of
        // how the quoted terms are joined.
        // Arrange
        let g = graph_at(Millis(1000)).await;
        g.insert(NewNode::now(
            "fact",
            "Storage",
            "the gadget uses libsql for storage",
        ))
        .await
        .unwrap();

        // Act: a question-shaped query whose terms appear in the content.
        let hits = g.query(&Query::text("storage gadget?")).await.unwrap();

        // Assert
        assert!(hits.iter().any(|h| h.label == "Storage"));

        // An apostrophe also opens FTS5 syntax; a question containing one must
        // not raise a syntax error either.
        assert!(g
            .query(&Query::text("what's the gadget's storage?"))
            .await
            .is_ok());

        // A punctuation-only query hits the "no searchable term" guard: it must
        // return Ok (empty), never an FTS5 syntax error.
        assert!(g.query(&Query::text("???")).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn lexical_question_finds_the_statement_that_answers_it() {
        // Regression pin for the whole bug: a question and the statement it
        // answers share only content words. The question adds interrogatives
        // ("when", "does") that a stored statement never contains, so joining
        // the quoted terms with implicit AND required every word to appear and
        // always returned zero hits. This fails on the pre-fix separator.
        // Arrange
        let g = graph_at(Millis(1000)).await;
        g.insert(NewNode::now(
            "fact",
            "Gizmo ship date",
            "The zorbnax gizmo ships in June 2026.",
        ))
        .await
        .unwrap();

        // Act
        let hits = g
            .query(&Query::text("When does the zorbnax gizmo ship?"))
            .await
            .unwrap();

        // Assert
        assert!(hits.iter().any(|h| h.label == "Gizmo ship date"));
    }

    #[tokio::test]
    async fn lexical_rare_term_still_ranks_first_under_or() {
        // ORing the quoted terms recovers recall, but recall alone would also
        // pass if precision at the top were destroyed. This proves BM25's
        // inverse-document-frequency weighting still does its job: "zorbnax"
        // appears in only one of three otherwise-similar nodes, so it must
        // dominate the common words ("the", "ships", "June", "2026") every node
        // shares.
        // Arrange
        let g = graph_at(Millis(1000)).await;
        g.insert(NewNode::now(
            "fact",
            "Rare",
            "the zorbnax gizmo ships in June 2026",
        ))
        .await
        .unwrap();
        g.insert(NewNode::now(
            "fact",
            "Common A",
            "the gadget ships in June 2026",
        ))
        .await
        .unwrap();
        g.insert(NewNode::now(
            "fact",
            "Common B",
            "the widget ships in June 2026",
        ))
        .await
        .unwrap();

        // Act
        let hits = g
            .query(&Query::text("when does the zorbnax gizmo ship"))
            .await
            .unwrap();

        // Assert
        assert_eq!(hits[0].label, "Rare");
    }

    #[tokio::test]
    async fn producer_round_trips_through_insert() {
        // Given a node inserted with a producer
        let g = graph_at(Millis(1000)).await;
        let id = g
            .insert(NewNode::now("fact", "label", "content").with_producer("agent-a"))
            .await
            .unwrap();

        // When read back, then the producer round-trips. `producer` is
        // deliberately absent from `Hit`, so query the `nodes` table
        // directly through the backend rather than through `query`.
        let rows = g
            .backend
            .query(
                "SELECT producer FROM nodes WHERE id = ?1",
                &[id.as_str().into()],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get_string(0).unwrap(), "agent-a");
    }

    #[tokio::test]
    async fn producer_defaults_to_unknown_when_not_specified() {
        // Given a node inserted without specifying a producer
        let g = graph_at(Millis(1000)).await;
        let id = g
            .insert(NewNode::now("fact", "label", "content"))
            .await
            .unwrap();

        // When read back, then it is "unknown" rather than empty or null.
        let rows = g
            .backend
            .query(
                "SELECT producer FROM nodes WHERE id = ?1",
                &[id.as_str().into()],
            )
            .await
            .unwrap();
        assert_eq!(rows[0].get_string(0).unwrap(), "unknown");
    }

    #[tokio::test]
    async fn upsert_by_carries_producer_through_supersede() {
        // Given a live node from producer A
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let old_id = g
            .upsert_by(
                NewNode::now("fact", "v1", "price is 10")
                    .with_subject("price")
                    .with_producer("agent-a"),
            )
            .await
            .unwrap();

        // When producer B supersedes it by subject via upsert_by
        clock.set(Millis(2000));
        let new_id = g
            .upsert_by(
                NewNode::now("fact", "v2", "price is 20")
                    .with_subject("price")
                    .with_producer("agent-b"),
            )
            .await
            .unwrap();

        // Then the new version records B and the superseded version still
        // records A: history attributes each version to whoever wrote it.
        let rows = g
            .backend
            .query(
                "SELECT id, producer FROM nodes WHERE id IN (?1, ?2)",
                &[old_id.as_str().into(), new_id.as_str().into()],
            )
            .await
            .unwrap();
        let producer_of = |id: &str| -> String {
            rows.iter()
                .find(|r| r.get_string(0).unwrap() == id)
                .unwrap()
                .get_string(1)
                .unwrap()
        };
        assert_eq!(producer_of(old_id.as_str()), "agent-a");
        assert_eq!(producer_of(new_id.as_str()), "agent-b");
    }

    #[tokio::test]
    async fn opening_an_old_schema_database_adds_producer_with_no_data_loss() {
        // Given a database created with the OLD schema (no `producer` column)
        // holding a row. Built by executing an explicit old `CREATE TABLE
        // nodes (...)` statement against a temp-file backend and inserting
        // directly through it, bypassing `Graph` entirely so the row
        // genuinely predates the column rather than merely having had it
        // dropped afterward.
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("old.db");
        let path_str = path.to_str().expect("temp path is valid utf-8");
        {
            let old_backend = crate::DefaultBackend::open(path_str)
                .await
                .expect("open old-schema backend");
            old_backend
                .execute_batch(
                    "CREATE TABLE nodes (
                       rowid       INTEGER PRIMARY KEY,
                       id          TEXT    NOT NULL UNIQUE,
                       kind        TEXT    NOT NULL,
                       label       TEXT    NOT NULL,
                       content     TEXT    NOT NULL,
                       attributes  TEXT    NOT NULL DEFAULT '{}',
                       scope       TEXT,
                       subject     TEXT,
                       confidence  REAL    NOT NULL DEFAULT 1.0,
                       valid_from  INTEGER NOT NULL,
                       valid_until INTEGER NOT NULL DEFAULT 4102444800000,
                       tx_from     INTEGER NOT NULL,
                       tx_to       INTEGER NOT NULL DEFAULT 4102444800000
                     )",
                )
                .await
                .expect("create old-schema nodes table");
            old_backend
                .execute(
                    "INSERT INTO nodes
                     (id, kind, label, content, attributes, scope, subject, confidence,
                      valid_from, valid_until, tx_from, tx_to)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    &[
                        "pre-existing".into(),
                        "fact".into(),
                        "old label".into(),
                        "old content".into(),
                        "{}".into(),
                        "proj-a".into(),
                        "old-subject".into(),
                        Value::Real(0.75),
                        Millis(1000).into(),
                        FOREVER.into(),
                        Millis(1000).into(),
                        FOREVER.into(),
                    ],
                )
                .await
                .expect("insert pre-existing row");
        } // old_backend drops here, releasing its connection before reopening.

        // When it is opened with the new code (a fresh `Graph` over the same
        // path, running the new schema plus the guarded migration).
        let clock = Arc::new(FixedClock::new(Millis(2000)));
        let g = DefaultGraph::open_with_clock(path_str, GraphConfig::new(8), clock)
            .await
            .expect("open graph over old-schema database");

        // Then the column exists (this SELECT would itself error if it did
        // not), the pre-existing row reads as "unknown", and every other
        // field on that row is intact.
        let rows = g
            .backend
            .query(
                "SELECT producer, kind, label, content, scope, subject, confidence,
                        valid_from, valid_until, tx_from, tx_to
                 FROM nodes WHERE id = ?1",
                &["pre-existing".into()],
            )
            .await
            .expect("query the migrated nodes table");
        assert_eq!(rows.len(), 1, "the pre-existing row was not lost");
        let row = &rows[0];
        assert_eq!(
            row.get_string(0).unwrap(),
            "unknown",
            "producer defaults for rows written before the column existed"
        );
        assert_eq!(row.get_string(1).unwrap(), "fact");
        assert_eq!(row.get_string(2).unwrap(), "old label");
        assert_eq!(row.get_string(3).unwrap(), "old content");
        assert_eq!(row.get_string(4).unwrap(), "proj-a");
        assert_eq!(row.get_string(5).unwrap(), "old-subject");
        assert!((row_f64(row, 6) - 0.75).abs() < f64::EPSILON);
        assert_eq!(row.get_i64(7).unwrap(), 1000);
        assert_eq!(row.get_i64(8).unwrap(), FOREVER.0);
        assert_eq!(row.get_i64(9).unwrap(), 1000);
        assert_eq!(row.get_i64(10).unwrap(), FOREVER.0);
    }
}
