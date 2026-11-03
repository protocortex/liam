// SPDX-License-Identifier: Apache-2.0
//! Which key group seals each logged row. A write reads the groups of rows it
//! points at before the log seals it, and a logged or replayed sealed record
//! leaves its rows' groups behind in `node_keys` and `derived_keys`.

use liam_log::envelope::{named_lineages, outside_edge_sources, KeyHints, RowKeyRef, RowKind};
use liam_log::event::LogEvent;

use crate::backend::BackendTx;
use crate::error::Result;
use crate::value::Value;

/// The `row_kind` of a sealed fact write in `derived_keys`.
const DERIVED_FACT: &str = "fact";
/// The `row_kind` of a sealed derived artifact in `derived_keys`.
const DERIVED_ARTIFACT: &str = "artifact";

const NODE_KEY_SQL: &str = "SELECT key_id FROM node_keys WHERE node_id = ?1";

// Every node of a lineage shares its subject's group, so any one answers.
const LINEAGE_KEY_SQL: &str = "SELECT nk.key_id FROM node_lineage nl
     JOIN node_keys nk ON nk.node_id = nl.node_id
     WHERE nl.lineage_key = ?1 ORDER BY nk.node_id LIMIT 1";

const NODE_KEY_INSERT_SQL: &str = "INSERT INTO node_keys (node_id, key_id, source)
     VALUES (?1, ?2, ?3) ON CONFLICT(node_id) DO NOTHING";

const DERIVED_KEY_INSERT_SQL: &str = "INSERT INTO derived_keys (row_kind, row_id, key_id)
     VALUES (?1, ?2, ?3) ON CONFLICT(row_kind, row_id) DO NOTHING";

/// The groups of the rows `event` points at that it does not carry itself: the
/// source of each edge, and the nodes behind each lineage key it names.
pub(super) async fn key_hints(tx: &mut dyn BackendTx, event: &LogEvent) -> Result<KeyHints> {
    let mut hints = KeyHints::default();
    for src in outside_edge_sources(&event.payload) {
        if let Some(key_id) = first_string(tx, NODE_KEY_SQL, &src).await? {
            hints.edge_sources.insert(src, key_id);
        }
    }
    for lineage_key in named_lineages(&event.payload) {
        if let Some(key_id) = first_string(tx, LINEAGE_KEY_SQL, &lineage_key).await? {
            hints.lineages.insert(lineage_key, key_id);
        }
    }
    Ok(hints)
}

async fn first_string(tx: &mut dyn BackendTx, sql: &str, param: &str) -> Result<Option<String>> {
    let rows = tx.query(sql, &[Value::from(param)]).await?;
    rows.first().map(|row| row.get_string(0)).transpose()
}

/// Records the group of each row a sealed record held, openable or not.
pub(super) async fn record_row_keys(
    tx: &mut dyn BackendTx,
    event_id: &str,
    keys: &[RowKeyRef],
) -> Result<()> {
    for key in keys {
        let row_id = Value::from(key.row_id.as_str());
        let key_id = Value::from(key.key_id.as_str());
        match key.kind {
            RowKind::Node => {
                let params = [row_id, key_id, Value::from(event_id)];
                tx.execute(NODE_KEY_INSERT_SQL, &params).await?;
            }
            RowKind::Fact => {
                let params = [Value::from(DERIVED_FACT), row_id, key_id];
                tx.execute(DERIVED_KEY_INSERT_SQL, &params).await?;
            }
            RowKind::Artifact => {
                let params = [Value::from(DERIVED_ARTIFACT), row_id, key_id];
                tx.execute(DERIVED_KEY_INSERT_SQL, &params).await?;
            }
            RowKind::Edge | RowKind::Usage | RowKind::Feedback | RowKind::Tier => {}
        }
    }
    Ok(())
}
