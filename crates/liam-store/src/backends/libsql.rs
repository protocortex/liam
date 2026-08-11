// SPDX-License-Identifier: MIT OR Apache-2.0
//! libSQL backend. Native vector search via `F32_BLOB` and `vector_distance_cos`,
//! with embeddings in a `node_vectors` table so the search can prefilter against
//! the live node set in one query.
//!
//! VERSION CHECK: the libSQL parameter-binding call (`params_from_iter`) and the
//! row accessors (`column_count`, `get_value`) are the surface to confirm against
//! the version you pin.

use async_trait::async_trait;
use libsql::{Builder, Connection};

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::ids::{Millis, NodeId};
use crate::value::{Row, Value};

pub struct LibsqlBackend {
    conn: Connection,
}

fn err(e: libsql::Error) -> Error {
    Error::Backend(e.to_string())
}

fn to_libsql(v: &Value) -> libsql::Value {
    match v {
        Value::Null => libsql::Value::Null,
        Value::Int(i) => libsql::Value::Integer(*i),
        Value::Real(r) => libsql::Value::Real(*r),
        Value::Text(s) => libsql::Value::Text(s.clone()),
        Value::Blob(b) => libsql::Value::Blob(b.clone()),
    }
}

fn from_libsql(v: libsql::Value) -> Value {
    match v {
        libsql::Value::Null => Value::Null,
        libsql::Value::Integer(i) => Value::Int(i),
        libsql::Value::Real(r) => Value::Real(r),
        libsql::Value::Text(s) => Value::Text(s),
        libsql::Value::Blob(b) => Value::Blob(b),
    }
}

fn bind(params: &[Value]) -> Vec<libsql::Value> {
    params.iter().map(to_libsql).collect()
}

async fn read_rows(mut rows: libsql::Rows) -> Result<Vec<Row>> {
    let cols = rows.column_count();
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(err)? {
        let mut values = Vec::with_capacity(cols as usize);
        for i in 0..cols {
            values.push(from_libsql(row.get_value(i).map_err(err)?));
        }
        out.push(Row(values));
    }
    Ok(out)
}

fn le_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for x in embedding {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

fn vector_literal(embedding: &[f32]) -> String {
    let mut out = String::with_capacity(embedding.len() * 8 + 2);
    out.push('[');
    for (i, x) in embedding.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&x.to_string());
    }
    out.push(']');
    out
}

#[async_trait]
impl Backend for LibsqlBackend {
    async fn open(path: &str) -> Result<Self> {
        let db = Builder::new_local(path).build().await.map_err(err)?;
        let conn = db.connect().map_err(err)?;
        Ok(Self { conn })
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64> {
        self.conn
            .execute(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)
    }

    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>> {
        let rows = self
            .conn
            .query(sql, libsql::params_from_iter(bind(params)))
            .await
            .map_err(err)?;
        read_rows(rows).await
    }

    async fn execute_batch(&self, sql: &str) -> Result<()> {
        self.conn.execute_batch(sql).await.map(|_| ()).map_err(err)
    }

    async fn execute_atomic(&self, statements: &[(String, Vec<Value>)]) -> Result<()> {
        let tx = self.conn.transaction().await.map_err(err)?;
        for (sql, params) in statements {
            tx.execute(sql, libsql::params_from_iter(bind(params)))
                .await
                .map_err(err)?;
        }
        tx.commit().await.map_err(err)
    }

    fn vector_ddl(&self, dims: usize) -> String {
        format!(
            "CREATE TABLE IF NOT EXISTS node_vectors (
  node_id   TEXT NOT NULL PRIMARY KEY REFERENCES nodes(id),
  embedding F32_BLOB({dims}) NOT NULL
);"
        )
    }

    async fn vector_upsert(&self, node_id: &str, embedding: &[f32]) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO node_vectors (node_id, embedding) VALUES (?1, ?2)
                 ON CONFLICT(node_id) DO UPDATE SET embedding = excluded.embedding",
                libsql::params_from_iter(vec![
                    libsql::Value::Text(node_id.to_string()),
                    libsql::Value::Blob(le_bytes(embedding)),
                ]),
            )
            .await
            .map_err(err)?;
        Ok(())
    }

    async fn vector_delete(&self, node_id: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM node_vectors WHERE node_id = ?1",
                libsql::params_from_iter(vec![libsql::Value::Text(node_id.to_string())]),
            )
            .await
            .map_err(err)?;
        Ok(())
    }

    async fn vector_search(
        &self,
        query: &[f32],
        k: usize,
        kind: Option<&str>,
        scope: Option<&str>,
        as_of: Millis,
    ) -> Result<Vec<NodeId>> {
        let mut params = vec![
            libsql::Value::Text(vector_literal(query)),
            libsql::Value::Integer(as_of.0),
            libsql::Value::Integer(k as i64),
        ];
        let mut filters = String::new();
        let mut next = 4;
        if let Some(kind) = kind {
            filters.push_str(&format!(" AND n.kind = ?{next}"));
            params.push(libsql::Value::Text(kind.to_string()));
            next += 1;
        }
        if let Some(scope) = scope {
            filters.push_str(&format!(" AND n.scope = ?{next}"));
            params.push(libsql::Value::Text(scope.to_string()));
        }
        // Same four-bound "live at T" predicate the lexical path enforces
        // (see `live_at` in graph.rs): recorded before T, not yet superseded
        // at T, and true in the world at T. `?2` is `as_of`, reused for all
        // four bounds.
        let sql = format!(
            "SELECT v.node_id FROM node_vectors v
             JOIN nodes n ON n.id = v.node_id
             WHERE n.tx_from <= ?2 AND n.tx_to > ?2
               AND n.valid_from <= ?2 AND n.valid_until > ?2{filters}
             ORDER BY vector_distance_cos(v.embedding, vector(?1)) LIMIT ?3"
        );
        let rows = self
            .conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(err)?;
        let rows = read_rows(rows).await?;
        rows.iter()
            .map(|r| Ok(NodeId::from_raw(r.get_string(0)?)))
            .collect()
    }

    async fn vector_sweep_orphans(&self) -> Result<u64> {
        self.conn
            .execute(
                "DELETE FROM node_vectors WHERE node_id NOT IN (SELECT id FROM nodes)",
                libsql::params_from_iter(Vec::<libsql::Value>::new()),
            )
            .await
            .map_err(err)
    }
}
