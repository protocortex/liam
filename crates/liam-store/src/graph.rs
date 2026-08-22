// SPDX-License-Identifier: AGPL-3.0-only
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

/// How many candidates an ambiguous handle reports back. Bounded so a
/// one-character handle answers with something a caller can act on instead of
/// every live node in the store.
const HANDLE_MATCH_LIMIT: usize = 8;

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
        let backend = B::open(path, config.read_pool_size).await?;
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

    /// Assert a semantic edge between two live nodes, idempotently.
    ///
    /// Liveness and idempotency are conditions ON the insert rather than checks
    /// before it, which is what makes this safe without a new backend
    /// capability (ADR-0001). Checking first and inserting after would let a
    /// concurrent `supersede` close an endpoint in the gap, and moving both
    /// into `execute_atomic` cannot help: that method takes a statement list
    /// built before the call and runs each with `execute`, never `query`
    /// (`backend.rs:53`, `backends/libsql.rs:228`), so nothing inside the
    /// transaction can read a row and branch on it. A single statement is
    /// atomic by itself, so the guards ride along in its `WHERE`.
    ///
    /// The idempotency guard keys on the ORDERED triple, so `relate(a, b, t)`
    /// and `relate(b, a, t)` both insert. That is correct for a directed edge
    /// and deliberately does not stop the pair reaching weight 2.0 in
    /// clustering; that guard lives on the read side, in
    /// `recompute_communities` (ADR-0001 Amendment 1, ADR-0002 Amendment 2).
    ///
    /// Endpoints are tested with `tx_to = FOREVER` alone, not the four-column
    /// `live_at` predicate this file uses everywhere else. Deliberate, not an
    /// oversight: see ADR-0001 Amendment 2 before "fixing" it, because widening
    /// it changes the guarantee rather than tightening it.
    pub async fn relate(&self, src: &NodeId, dst: &NodeId, kind: &str) -> Result<EdgeId> {
        let id = EdgeId::new();
        let now = self.clock.now();
        let written = self
            .backend
            .execute(
                "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
                 SELECT ?1, ?2, ?3, ?4, '{}', ?5, ?6
                 WHERE EXISTS     (SELECT 1 FROM nodes WHERE id = ?2 AND tx_to = ?7)
                   AND EXISTS     (SELECT 1 FROM nodes WHERE id = ?3 AND tx_to = ?7)
                   AND NOT EXISTS (SELECT 1 FROM edges
                                   WHERE src = ?2 AND dst = ?3 AND type = ?4 AND tx_to = ?7)",
                &[
                    id.as_str().into(),
                    src.as_str().into(),
                    dst.as_str().into(),
                    kind.into(),
                    now.into(),
                    FOREVER.into(),
                    FOREVER.into(),
                ],
            )
            .await?;
        if written == 1 {
            return Ok(id);
        }
        Err(self.relate_refusal(src, dst, kind).await)
    }

    /// Name the guard that refused the write. Runs only after a refusal and
    /// races nothing, because it never decides whether to write: the write
    /// already happened, or already did not, and this only shapes the message.
    async fn relate_refusal(&self, src: &NodeId, dst: &NodeId, kind: &str) -> Error {
        let rows = self
            .backend
            .query(
                "SELECT
                   EXISTS(SELECT 1 FROM nodes WHERE id = ?1 AND tx_to = ?3),
                   EXISTS(SELECT 1 FROM nodes WHERE id = ?2 AND tx_to = ?3),
                   EXISTS(SELECT 1 FROM edges
                          WHERE src = ?1 AND dst = ?2 AND type = ?4 AND tx_to = ?3)",
                &[
                    src.as_str().into(),
                    dst.as_str().into(),
                    FOREVER.into(),
                    kind.into(),
                ],
            )
            .await;
        let rows = match rows {
            Ok(rows) => rows,
            Err(e) => return e,
        };
        let Some(row) = rows.first() else {
            return Error::RelateRefused("no row explains the refusal".to_string());
        };
        // Surface a read fault as itself. `unwrap_or(0)` here would read as
        // "this guard failed", and since column 0 is tested first, every
        // backend fault would come back as a dead source node: a specific,
        // confident, wrong diagnosis that sends the client to re-resolve a
        // handle that was fine all along.
        let flags: std::result::Result<Vec<i64>, Error> = (0..3).map(|i| row.get_i64(i)).collect();
        let flags = match flags {
            Ok(flags) => flags,
            Err(e) => return e,
        };
        let truthy = |i: usize| flags[i] != 0;
        if !truthy(0) {
            return Error::RelateRefused(format!("source node {} is not live", src.as_str()));
        }
        if !truthy(1) {
            return Error::RelateRefused(format!("target node {} is not live", dst.as_str()));
        }
        if truthy(2) {
            return Error::RelateRefused(format!(
                "{} already relates to {} as '{kind}'",
                src.as_str(),
                dst.as_str()
            ));
        }
        // Every guard passes now, so one of them flipped between the insert and
        // this read. A retry would land.
        Error::RelateRefused("a concurrent write took the row, retry".to_string())
    }

    /// Resolve a client-supplied handle to a full node id. A handle is any
    /// prefix of a ULID, including the whole 26 characters; `recall` renders 13
    /// (ADR-0001 Amendment 3).
    ///
    /// Two matches is an error, never a pick. That is the property that makes
    /// prefixes acceptable where ADR-0001 rejected addressing by label: a label
    /// collision is invisible and silently writes a plausible wrong edge, while
    /// a prefix collision is visible right here. The error carries the
    /// candidates in full because the caller was only ever shown 13 characters
    /// and cannot lengthen the handle on its own.
    pub async fn resolve_handle(&self, handle: &str) -> Result<NodeId> {
        // Errors quote what the caller actually sent. Echoing the normalised
        // form back shows a model a string it never wrote, which invites it to
        // retry against the transformation instead of against its own input.
        let sent = handle.trim().to_string();
        // Crockford base32 is case-insensitive by definition, GLOB is not.
        let handle = sent.to_ascii_uppercase();
        // `*`, `?` and `[` are GLOB wildcards, so an unfiltered handle could
        // match far more than its prefix. With a single live node in the store
        // a bare `*` would even resolve successfully, to an arbitrary node.
        // Restricting to alphanumerics excludes every metacharacter; it is
        // wider than the real ULID alphabet, which costs nothing because a
        // non-ULID character simply matches no row.
        if handle.is_empty() || !handle.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(Error::HandleNotFound(sent));
        }
        // GLOB, not LIKE. SQLite's LIKE is case-insensitive for ASCII by
        // default, so it cannot use the unique index on `nodes(id)`: measured
        // with EXPLAIN QUERY PLAN, LIKE plans as SCAN and GLOB as SEARCH.
        let rows = self
            .backend
            .query(
                "SELECT id FROM nodes WHERE id GLOB ?1 AND tx_to = ?2 ORDER BY id LIMIT ?3",
                &[
                    format!("{handle}*").into(),
                    FOREVER.into(),
                    (HANDLE_MATCH_LIMIT as i64).into(),
                ],
            )
            .await?;
        match rows.len() {
            0 => Err(Error::HandleNotFound(sent)),
            1 => Ok(NodeId::from_raw(rows[0].get_string(0)?)),
            // The list is capped, so the message names no count: a truncated
            // list with a claimed total would be a lie the caller cannot check.
            _ => Err(Error::AmbiguousHandle {
                handle: sent,
                candidates: rows
                    .iter()
                    .filter_map(|r| r.get_string(0).ok())
                    .collect::<Vec<_>>(),
            }),
        }
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

    /// Nodes recorded or closed strictly after `cursor`, for rebuilding only
    /// the vectors that changed.
    ///
    /// **This cursor cannot see an edge write, so it is useless for clustering.**
    /// It queries `nodes` only, and `Change` carries a node id, a timestamp and
    /// a closed flag. A `relate` inserts into `edges` and touches no node row,
    /// so this returns nothing for exactly the event that changes communities.
    /// An earlier version of this comment promised "recompute only changed
    /// communities", which the query has never supported.
    ///
    /// Clustering staleness is detected by the edge fingerprint in
    /// `cluster_state` instead (ADR-0002). Making this cursor able to answer the
    /// question would take the append-only change ledger that ADR-0002 weighs
    /// and defers.
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

impl<B: Backend> Graph<B> {
    pub async fn recompute_communities(&self) -> Result<usize> {
        use crate::cluster::detect;

        // `supersedes` is excluded because it is structural, not semantic: it
        // is written only inside `supersede`'s transaction and describes
        // version history. Leaving it in means a version chain reads as a
        // topic (ADR-0001, Scope). `relate` refuses to assert it for the same
        // reason, so this is that decision seen from the other end.
        //
        // `ORDER BY id` fixes the row order, which `build_cluster_input` turns
        // into the dense vertex numbering. Without it the numbering varies run
        // to run on identical data, and so can Leiden's seeded partition.
        let rows = self
            .backend
            .query(
                "SELECT src, dst, type FROM edges
                 WHERE tx_to = ?1 AND type != ?2 ORDER BY id",
                &[FOREVER.into(), crate::types::relation::SUPERSEDES.into()],
            )
            .await?;
        let (labels, edges) = build_cluster_input(&rows)?;

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

/// Turn live edge rows of `(src, dst, type)` into the dense node labels and the
/// edge list Leiden runs on. Pure, and separate from `recompute_communities` so
/// the dedup below can be asserted directly. Going through the partition
/// instead would test it via a heuristic: a count of communities does not
/// discriminate between a deduped graph and a doubled one, since modularity can
/// land on the same number of groups either way.
///
/// Collapses each unordered pair to one edge. `relate` is idempotent on the
/// ORDERED triple, so a client asserting both directions writes two rows, which
/// is right for a directed edge table. `leiden-rs` then merges only consecutive
/// exact duplicates in `build_undirected_csr`, so `(a,b)` and `(b,a)` both
/// survive and the pair reaches weight 2.0 (ADR-0002 Amendment 2).
///
/// `type` stays in the key on purpose. `relate(a, b, "mentions")` and
/// `relate(a, b, "causes")` are two independent things a client said about one
/// pair, and neither collides with the other on the ordered triple, so both are
/// in the table deliberately. Keying on the pair alone would erase that and
/// weigh a pair carrying two relations like one carrying a single relation.
fn build_cluster_input(rows: &[Row]) -> Result<(Vec<String>, Vec<crate::cluster::Edge>)> {
    use crate::cluster::Edge;

    let mut index: HashMap<String, usize> = HashMap::new();
    let mut labels: Vec<String> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut seen: HashSet<(usize, usize, String)> = HashSet::new();
    for row in rows {
        let u = intern(&mut index, &mut labels, row.get_string(0)?);
        let v = intern(&mut index, &mut labels, row.get_string(1)?);
        if seen.insert((u.min(v), u.max(v), row.get_string(2)?)) {
            edges.push(Edge(u, v));
        }
    }
    Ok((labels, edges))
}

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
        // This test exercises schema migration, not read pooling, so the
        // pool size is arbitrary; 1 keeps it minimal rather than implying
        // some other value matters here.
        const ARBITRARY_POOL_SIZE: usize = 1;
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("old.db");
        let path_str = path.to_str().expect("temp path is valid utf-8");
        {
            let old_backend = crate::DefaultBackend::open(path_str, ARBITRARY_POOL_SIZE)
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

    /// `cluster_state` is a new TABLE, not a new column, so it needs no
    /// `migrate::` call: `schema()` is all `CREATE TABLE IF NOT EXISTS` and
    /// `open_with_clock` re-runs the whole batch on every open. This pins that,
    /// because the reasoning is easy to doubt later and the failure would be a
    /// panic on the first `clusters` call against any pre-existing database.
    #[tokio::test]
    async fn opening_a_database_that_predates_cluster_state_creates_it_with_no_data_loss() {
        // This test exercises schema evolution, not read pooling.
        const ARBITRARY_POOL_SIZE: usize = 1;
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("no-cluster-state.db");
        let path_str = path.to_str().expect("temp path is valid utf-8");

        // Given a database whose schema has node_community but NO
        // cluster_state, which is exactly what a build predating ADR-0002
        // left behind. Written through a bare backend so the table genuinely
        // never existed rather than having been dropped.
        {
            let old = crate::DefaultBackend::open(path_str, ARBITRARY_POOL_SIZE)
                .await
                .expect("open old-schema backend");
            old.execute_batch(
                "CREATE TABLE node_community (
                   node_id     TEXT    NOT NULL PRIMARY KEY,
                   community   INTEGER NOT NULL,
                   computed_at INTEGER NOT NULL
                 );",
            )
            .await
            .expect("create the old cluster tables");
            old.execute(
                "INSERT INTO node_community (node_id, community, computed_at)
                 VALUES (?1, ?2, ?3)",
                &["pre-existing".into(), Value::Int(7), Millis(1000).into()],
            )
            .await
            .expect("insert a pre-existing assignment");
        } // dropped, releasing the connection before reopening

        // When it is opened by the new code
        let clock = Arc::new(FixedClock::new(Millis(2000)));
        let g = DefaultGraph::open_with_clock(path_str, GraphConfig::new(8), clock)
            .await
            .expect("open graph over a database with no cluster_state");

        // Then cluster_state exists. This SELECT errors if it does not, which
        // is the assertion.
        let state = g
            .backend
            .query("SELECT COUNT(*) FROM cluster_state", &[])
            .await
            .expect("cluster_state must exist after opening");
        assert_eq!(
            state[0].get_i64(0).unwrap(),
            0,
            "a table created by this open starts empty, so the fingerprint \
             check reads 'no prior run' and forces a cold recompute"
        );

        // And the assignment that was already there is untouched. Creating a
        // table must never disturb its neighbour.
        let kept = g
            .backend
            .query(
                "SELECT community, computed_at FROM node_community WHERE node_id = ?1",
                &["pre-existing".into()],
            )
            .await
            .expect("query the pre-existing assignment");
        assert_eq!(kept.len(), 1, "the pre-existing assignment was lost");
        assert_eq!(kept[0].get_i64(0).unwrap(), 7);
        assert_eq!(kept[0].get_i64(1).unwrap(), 1000);
    }

    /// The empty state is the one every store is in on the day ADR-0002 ships,
    /// so it is worth naming rather than leaving implied.
    #[tokio::test]
    async fn a_fresh_database_has_an_empty_cluster_state() {
        let g = graph_at(Millis(1000)).await;

        let rows = g
            .backend
            .query("SELECT COUNT(*) FROM cluster_state", &[])
            .await
            .expect("cluster_state must exist on a fresh database");

        assert_eq!(rows[0].get_i64(0).unwrap(), 0);
    }

    /// Community detection over the edge graph, which had no test at all
    /// before this. "Nothing calls it and nothing covers it" was used once as
    /// an argument for dropping the feature; the approved design
    /// (`2026-07-23-always-available-clusters-design.md`, folded into M5) says
    /// the opposite, so this pins the behaviour the daemon will lean on when
    /// `recompute_communities` moves onto the maintenance tick.
    #[tokio::test]
    async fn communities_separate_two_disconnected_groups_of_nodes() {
        // Given two clusters of linked nodes with NO edge between them: three
        // notes about auth that reference each other, and two about billing.
        let g = graph_at(Millis(1000)).await;

        let mut auth = Vec::new();
        for label in ["auth: token refresh", "auth: session expiry", "auth: login"] {
            auth.push(
                g.insert(NewNode::now("fact", label, "content"))
                    .await
                    .unwrap(),
            );
        }

        let mut billing = Vec::new();
        for label in ["billing: invoices", "billing: dunning"] {
            billing.push(
                g.insert(NewNode::now("fact", label, "content"))
                    .await
                    .unwrap(),
            );
        }

        for pair in [(0, 1), (1, 2), (0, 2)] {
            g.link(NewEdge::new(&auth[pair.0], &auth[pair.1], "mentions"))
                .await
                .unwrap();
        }
        g.link(NewEdge::new(&billing[0], &billing[1], "mentions"))
            .await
            .unwrap();

        // When communities are recomputed
        let count = g.recompute_communities().await.unwrap();

        // Then every node is assigned, and the two groups land in different
        // communities: that separation is the whole point, and it is what
        // makes clusters usable for finding sections nobody declared.
        // `recompute_communities` returns how many DISTINCT communities it
        // found, not how many nodes it touched.
        assert_eq!(count, 2, "the two disconnected groups are two communities");
        let assignments = g.communities().await.unwrap();
        assert_eq!(
            assignments.len(),
            5,
            "every node that appears on a live edge should be assigned"
        );

        let community_of = |id: &NodeId| {
            assignments
                .iter()
                .find(|(n, _)| n == id)
                .map(|(_, c)| *c)
                .expect("node must have a community")
        };
        let auth_community = community_of(&auth[0]);
        let billing_community = community_of(&billing[0]);

        assert!(
            auth.iter().all(|n| community_of(n) == auth_community),
            "the three linked auth nodes belong together: {assignments:?}"
        );
        assert!(
            billing.iter().all(|n| community_of(n) == billing_community),
            "the two linked billing nodes belong together: {assignments:?}"
        );
        assert_ne!(
            auth_community, billing_community,
            "disconnected groups must not be merged into one community: {assignments:?}"
        );
    }

    #[tokio::test]
    async fn version_chains_do_not_form_communities() {
        // Before the `supersedes` filter, a store holding only version history
        // still produced communities, and they read as topics while describing
        // nothing but edit order. `supersede` is the only writer of that edge
        // type, so a store built purely by `upsert_by` has to cluster to
        // nothing at all (ADR-0001, Scope).
        let g = graph_at(Millis(1000)).await;
        for content in ["first", "second", "third"] {
            g.upsert_by(NewNode::now("fact", "Rollout status", content).with_subject("rollout"))
                .await
                .unwrap();
        }

        let count = g.recompute_communities().await.unwrap();

        assert_eq!(count, 0, "version history is not a topic");
        assert!(
            g.communities().await.unwrap().is_empty(),
            "no node should be assigned from supersedes edges alone"
        );
    }

    /// `(src, dst, type)` rows in the shape `build_cluster_input` reads, so the
    /// dedup can be checked without a database or a Leiden run.
    fn edge_rows(triples: &[(&str, &str, &str)]) -> Vec<Row> {
        triples
            .iter()
            .map(|(s, d, t)| {
                Row(vec![
                    Value::Text((*s).to_string()),
                    Value::Text((*d).to_string()),
                    Value::Text((*t).to_string()),
                ])
            })
            .collect()
    }

    #[tokio::test]
    async fn asserting_a_pair_in_both_directions_yields_one_edge() {
        // `relate` is idempotent on the ORDERED triple, so both directions
        // insert, which is right for a directed edge table. Clustering reads
        // the set as undirected and `leiden-rs` merges only consecutive exact
        // duplicates, so without this dedup the pair reaches weight 2.0
        // (ADR-0002 Amendment 2).
        //
        // Asserted on the edge list, NOT on the community count: a count does
        // not discriminate between the deduped graph and the doubled one, so
        // the earlier version of this test passed with the dedup deleted.
        let rows = edge_rows(&[("a", "b", "mentions"), ("b", "a", "mentions")]);

        let (labels, edges) = build_cluster_input(&rows).unwrap();

        assert_eq!(labels.len(), 2, "both endpoints stay in the node set");
        assert_eq!(
            edges.len(),
            1,
            "the reversed pair must collapse to one edge"
        );
    }

    #[tokio::test]
    async fn two_relation_types_on_one_pair_stay_two_edges() {
        // The dedup key carries `type` on purpose, so this pins the OTHER
        // direction of the same guard. Dropping `type` from the key would weigh
        // a pair carrying two asserted relations exactly like a pair carrying
        // one, which is the error ADR-0002 Amendment 2 calls out as second bug.
        let rows = edge_rows(&[("a", "b", "mentions"), ("a", "b", "causes")]);

        let (_, edges) = build_cluster_input(&rows).unwrap();

        assert_eq!(edges.len(), 2, "distinct types are independent evidence");
    }

    #[tokio::test]
    async fn a_repeated_triple_and_a_self_loop_each_collapse_to_one_edge() {
        // The exact-duplicate case cannot come from `relate`, whose NOT EXISTS
        // refuses it, but `link` writes rows without that guard and a `gc` can
        // leave the table in shapes nothing else produces.
        let repeated = edge_rows(&[("a", "b", "mentions"), ("a", "b", "mentions")]);
        assert_eq!(build_cluster_input(&repeated).unwrap().1.len(), 1);

        // A self-loop normalises to min == max and survives once.
        let loops = edge_rows(&[("a", "a", "mentions"), ("a", "a", "mentions")]);
        let (labels, edges) = build_cluster_input(&loops).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(labels, vec!["a".to_string()]);
    }

    #[tokio::test]
    async fn a_deduped_row_never_drops_a_node_from_the_partition() {
        // Both endpoints are interned BEFORE the dedup check. A row that
        // dedupes shares its key with an earlier row, so its endpoints were
        // already interned and no node can be lost this way. Pinned because the
        // obvious "optimisation" of skipping the intern on a duplicate would
        // silently drop nodes out of clustering entirely.
        let rows = edge_rows(&[
            ("a", "b", "mentions"),
            ("b", "a", "mentions"),
            ("c", "d", "mentions"),
        ]);

        let (labels, edges) = build_cluster_input(&rows).unwrap();

        assert_eq!(labels.len(), 4, "every endpoint keeps a dense index");
        assert_eq!(edges.len(), 2);
    }

    #[tokio::test]
    async fn the_dense_numbering_follows_row_order() {
        // `recompute_communities` orders by edge id so this numbering is a
        // function of the edge set alone. If it were not, Leiden's seeded
        // partition could differ run to run on identical data.
        let rows = edge_rows(&[("z", "y", "mentions"), ("y", "x", "mentions")]);

        let (labels, edges) = build_cluster_input(&rows).unwrap();

        assert_eq!(labels, vec!["z", "y", "x"], "interned in row order");
        assert_eq!((edges[0].0, edges[0].1), (0, 1));
        assert_eq!((edges[1].0, edges[1].1), (1, 2));
    }

    #[tokio::test]
    async fn relate_writes_a_separate_row_per_relation_type() {
        // The write-side half: `relate`'s NOT EXISTS keys on the ordered triple
        // INCLUDING type, so a second type on the same pair is not a duplicate.
        let g = graph_at(Millis(1000)).await;
        let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
        let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();

        g.relate(&a, &b, "mentions").await.unwrap();
        g.relate(&a, &b, "causes").await.unwrap();

        let rows = g
            .backend
            .query(
                "SELECT COUNT(*) FROM edges WHERE src = ?1 AND dst = ?2",
                &[a.as_str().into(), b.as_str().into()],
            )
            .await
            .unwrap();
        assert_eq!(rows[0].get_i64(0).unwrap(), 2, "both types must be stored");
    }

    #[tokio::test]
    async fn relate_refuses_an_endpoint_that_is_no_longer_live() {
        // The guarantee the unenforced foreign keys were supposed to give. The
        // check rides on the INSERT's own WHERE, so a `supersede` landing
        // between a caller's decision and this write still wins.
        let g = graph_at(Millis(1000)).await;
        let old = g
            .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
            .await
            .unwrap();
        let other = g.insert(NewNode::now("fact", "Other", "x")).await.unwrap();
        g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
            .await
            .unwrap();

        let err = g.relate(&old, &other, "mentions").await.unwrap_err();

        assert!(
            matches!(&err, Error::RelateRefused(m) if m.contains("source node")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn relate_refuses_a_destination_that_is_no_longer_live() {
        // The mirror of the test above, and not redundant: the two endpoints
        // are separate EXISTS subqueries binding separate parameters. Swapping
        // `?3` for `?2` in the second one is an easy slip in a statement that
        // mentions both four times, and with only the source case covered the
        // whole suite would stay green while `relate` wrote edges pointing at
        // superseded nodes.
        let g = graph_at(Millis(1000)).await;
        let other = g.insert(NewNode::now("fact", "Other", "x")).await.unwrap();
        let old = g
            .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
            .await
            .unwrap();
        g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
            .await
            .unwrap();

        let err = g.relate(&other, &old, "mentions").await.unwrap_err();

        assert!(
            matches!(&err, Error::RelateRefused(m) if m.contains("target node")),
            "{err}"
        );
        let rows = g
            .backend
            .query(
                "SELECT COUNT(*) FROM edges WHERE dst = ?1 AND type = ?2",
                &[old.as_str().into(), "mentions".into()],
            )
            .await
            .unwrap();
        assert_eq!(rows[0].get_i64(0).unwrap(), 0, "no edge may have landed");
    }

    #[tokio::test]
    async fn relate_is_idempotent_on_the_ordered_triple() {
        let g = graph_at(Millis(1000)).await;
        let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
        let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();

        g.relate(&a, &b, "mentions").await.unwrap();
        let err = g.relate(&a, &b, "mentions").await.unwrap_err();

        assert!(
            matches!(&err, Error::RelateRefused(m) if m.contains("already relates")),
            "{err}"
        );
        // The reverse direction is a different edge and must still land.
        g.relate(&b, &a, "mentions").await.unwrap();
    }

    #[tokio::test]
    async fn a_handle_resolves_and_an_ambiguous_one_reports_its_candidates() {
        // The whole reason prefixes are acceptable where labels were not: a
        // collision is visible here, so the store refuses instead of guessing.
        // The refusal has to carry full ids, because a caller shown 13
        // characters cannot lengthen the handle on its own.
        let g = graph_at(Millis(1000)).await;
        let a = g.insert(NewNode::now("fact", "a", "x")).await.unwrap();
        let b = g.insert(NewNode::now("fact", "b", "x")).await.unwrap();

        assert_eq!(g.resolve_handle(a.as_str()).await.unwrap(), a);
        assert_eq!(g.resolve_handle(a.handle()).await.unwrap(), a);
        // Crockford base32 is case-insensitive, GLOB is not.
        assert_eq!(
            g.resolve_handle(&a.handle().to_lowercase()).await.unwrap(),
            a
        );

        // Both ids start with `0`, so a one-character handle names neither.
        let err = g.resolve_handle("0").await.unwrap_err();
        let Error::AmbiguousHandle { candidates, .. } = &err else {
            panic!("expected an ambiguous handle, got {err}");
        };
        assert!(candidates.contains(&a.as_str().to_string()), "{err}");
        assert!(candidates.contains(&b.as_str().to_string()), "{err}");

        assert!(matches!(
            g.resolve_handle("ZZZZZZZZZZZZZ").await.unwrap_err(),
            Error::HandleNotFound(_)
        ));
    }

    #[tokio::test]
    async fn a_wildcard_handle_never_resolves_even_with_one_live_node() {
        // With a single live node, an unfiltered `*` would GLOB-match it and
        // resolve successfully, handing back an arbitrary node the caller never
        // named. The alphabet check is what stops that.
        let g = graph_at(Millis(1000)).await;
        g.insert(NewNode::now("fact", "only", "x")).await.unwrap();

        for handle in ["*", "?", "[0-9]*", "0*"] {
            assert!(
                matches!(
                    g.resolve_handle(handle).await,
                    Err(Error::HandleNotFound(_))
                ),
                "wildcard {handle:?} resolved"
            );
        }
    }

    #[tokio::test]
    async fn a_superseded_node_is_not_reachable_by_handle() {
        let g = graph_at(Millis(1000)).await;
        let old = g
            .upsert_by(NewNode::now("fact", "Rollout", "first").with_subject("rollout"))
            .await
            .unwrap();
        g.upsert_by(NewNode::now("fact", "Rollout", "second").with_subject("rollout"))
            .await
            .unwrap();

        assert!(matches!(
            g.resolve_handle(old.as_str()).await.unwrap_err(),
            Error::HandleNotFound(_)
        ));
    }
}
