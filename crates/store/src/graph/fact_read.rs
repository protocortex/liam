// SPDX-License-Identifier: Apache-2.0
//! Read-only bitemporal fact queries: what is believed now, what was true at a
//! valid time, what the store believed at a transaction time, and the versions
//! and sources of one lineage. Every query is bounded by a row cap and has a
//! total `ORDER BY`, so no map order reaches the output.

use super::fact_write::{version_from_row, FACT_COLUMNS};
use super::{scope_within, validate_scope, Graph};
use crate::backend::Backend;
use crate::error::Result;
use crate::ids::{FactId, Millis, FOREVER};
use crate::types::Fact;
use crate::value::Value;

/// Most fact versions one read returns, whatever limit the caller asks for.
pub const MAX_FACT_ROWS: usize = 200;

/// Most source events attached to one returned fact.
pub const MAX_SOURCES_PER_FACT: usize = 100;

/// The scope sits after the subject, transaction and valid time params.
const SCOPE_PARAM: usize = 4;

/// Strongest claim first within a predicate; ids settle every remaining tie.
const FACT_ORDER: &str = "predicate ASC, trust_score DESC, confidence DESC, tx_from DESC, id ASC";

/// A version never closed (`?2` is `FOREVER`), valid at `?3`. Only the
/// `subject` prefix of `facts_current` is a seek key, because `predicate`
/// sits between it and `tx_to`; the rest is filtered per subject.
pub(super) const CURRENT_FILTER: &str =
    "subject = ?1 AND tx_to = ?2 AND valid_from <= ?3 AND valid_until > ?3";

/// A version recorded by transaction time `?2`, valid at `?3`.
const BELIEVED_FILTER: &str =
    "subject = ?1 AND tx_from <= ?2 AND tx_to > ?2 AND valid_from <= ?3 AND valid_until > ?3";

const SOURCES_SQL: &str =
    "SELECT event_id FROM fact_sources WHERE lineage_id = ?1 ORDER BY event_id LIMIT ?2";

/// Facts in read order, with whether a cap cut the result short.
#[derive(Clone, Debug, PartialEq)]
pub struct FactPage {
    pub facts: Vec<Fact>,
    /// More versions matched than the caller's limit or `MAX_FACT_ROWS` allowed.
    pub truncated: bool,
    /// Some returned fact has more sources than `MAX_SOURCES_PER_FACT`.
    pub sources_truncated: bool,
}

impl<B: Backend> Graph<B> {
    /// The facts the store currently believes held at `valid_at`. `scope`
    /// matches itself and its descendants; `None` matches every scope.
    ///
    /// Sources are per lineage and carry no time, so a read of the past can
    /// list sources that were linked after it.
    pub async fn facts_valid_at(
        &self,
        subject: &str,
        scope: Option<&str>,
        valid_at: Millis,
        limit: usize,
    ) -> Result<FactPage> {
        let params = [subject.into(), FOREVER.into(), valid_at.into()];
        self.read_facts(CURRENT_FILTER, params, scope, limit).await
    }

    /// The facts the store believed at `believed_at` held at `valid_at`, which
    /// defaults to `believed_at`. Sources are as in `facts_valid_at`.
    pub async fn facts_believed_at(
        &self,
        subject: &str,
        scope: Option<&str>,
        believed_at: Millis,
        valid_at: Option<Millis>,
        limit: usize,
    ) -> Result<FactPage> {
        let valid_at = valid_at.unwrap_or(believed_at);
        let params = [subject.into(), believed_at.into(), valid_at.into()];
        self.read_facts(BELIEVED_FILTER, params, scope, limit).await
    }

    /// Every stored version of one lineage, oldest transaction first, closed
    /// and ended versions included. It does not follow `closed_by` into the
    /// lineages that replaced it.
    pub async fn fact_lineage_history(
        &self,
        lineage_id: &FactId,
        limit: usize,
    ) -> Result<FactPage> {
        let sql = format!(
            "SELECT {FACT_COLUMNS} FROM facts WHERE lineage_id = ?1
             ORDER BY tx_from ASC, id ASC LIMIT ?2"
        );
        let params = [lineage_id.as_str().into(), probe(limit)];
        self.fetch_with_sources(&sql, &params, limit).await
    }

    async fn read_facts(
        &self,
        filter: &str,
        params: [Value; 3],
        scope: Option<&str>,
        limit: usize,
    ) -> Result<FactPage> {
        let mut params = params.to_vec();
        let mut filter = filter.to_owned();
        if let Some(scope) = validate_scope(&scope.map(str::to_owned))? {
            params.push(scope.into());
            filter.push_str(&format!(" AND {}", scope_within("scope", SCOPE_PARAM)));
        }
        params.push(probe(limit));
        let sql = format!(
            "SELECT {FACT_COLUMNS} FROM facts WHERE {filter}
             ORDER BY {FACT_ORDER} LIMIT ?{}",
            params.len()
        );
        self.fetch_with_sources(&sql, &params, limit).await
    }

    async fn fetch_with_sources(
        &self,
        sql: &str,
        params: &[Value],
        limit: usize,
    ) -> Result<FactPage> {
        let allowed = limit.min(MAX_FACT_ROWS);
        let rows = self.backend.query(sql, params).await?;
        let truncated = rows.len() > allowed;
        let mut sources_truncated = false;
        let mut facts = Vec::with_capacity(rows.len().min(allowed));
        for row in rows.iter().take(allowed) {
            let mut fact = version_from_row(row)?;
            let source_params = [
                fact.lineage_id.as_str().into(),
                Value::Int(MAX_SOURCES_PER_FACT as i64 + 1),
            ];
            let sources = self.backend.query(SOURCES_SQL, &source_params).await?;
            sources_truncated |= sources.len() > MAX_SOURCES_PER_FACT;
            fact.source_event_ids = sources
                .iter()
                .take(MAX_SOURCES_PER_FACT)
                .map(|source| source.get_string(0))
                .collect::<Result<_>>()?;
            facts.push(fact);
        }
        Ok(FactPage {
            facts,
            truncated,
            sources_truncated,
        })
    }
}

/// One row past the cap, so a full result can be told from a cut one.
fn probe(limit: usize) -> Value {
    Value::Int(limit.min(MAX_FACT_ROWS) as i64 + 1)
}
