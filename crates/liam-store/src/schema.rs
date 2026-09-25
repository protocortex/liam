// SPDX-License-Identifier: Apache-2.0
//! The core schema, shared across backends. It has no embedding column: vector
//! storage is backend-owned and appended via `Backend::vector_ddl`. Open
//! intervals use the FOREVER sentinel so every currency check is positive.

use crate::types::GraphConfig;

pub fn schema(_config: &GraphConfig) -> String {
    let mut sql = String::from(
        "PRAGMA auto_vacuum = INCREMENTAL;

CREATE TABLE IF NOT EXISTS nodes (
  rowid       INTEGER PRIMARY KEY,
  id          TEXT    NOT NULL UNIQUE,
  kind        TEXT    NOT NULL,
  label       TEXT    NOT NULL,
  content     TEXT    NOT NULL,
  -- DEFAULT 'unknown' is load-bearing, not decoration: it is what lets the
  -- guarded migration in `migrate::add_column_if_missing` add this column to
  -- an EXISTING database via ALTER TABLE without a NOT NULL failure on rows
  -- written before producer existed, so they read back as 'unknown' instead
  -- of losing data. A fresh database gets the column from this DDL directly.
  producer    TEXT    NOT NULL DEFAULT 'unknown',
  attributes  TEXT    NOT NULL DEFAULT '{}',
  scope       TEXT,
  subject     TEXT,
  confidence  REAL    NOT NULL DEFAULT 1.0,
  valid_from  INTEGER NOT NULL,
  valid_until INTEGER NOT NULL DEFAULT 4102444800000,
  tx_from     INTEGER NOT NULL,
  tx_to       INTEGER NOT NULL DEFAULT 4102444800000
);

CREATE INDEX IF NOT EXISTS nodes_live    ON nodes (kind, scope, tx_to, valid_until);
CREATE INDEX IF NOT EXISTS nodes_subject ON nodes (subject, scope, tx_to);
CREATE INDEX IF NOT EXISTS nodes_changed ON nodes (tx_from);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
  label, content,
  content = 'nodes',
  content_rowid = 'rowid',
  tokenize = 'porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
  INSERT INTO nodes_fts(rowid, label, content) VALUES (new.rowid, new.label, new.content);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
  INSERT INTO nodes_fts(nodes_fts, rowid, label, content) VALUES ('delete', old.rowid, old.label, old.content);
END;
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
  INSERT INTO nodes_fts(nodes_fts, rowid, label, content) VALUES ('delete', old.rowid, old.label, old.content);
  INSERT INTO nodes_fts(rowid, label, content) VALUES (new.rowid, new.label, new.content);
END;

-- The REFERENCES below are ENFORCED, not decoration. libSQL turns foreign keys
-- on by default, unlike stock SQLite, and no `PRAGMA foreign_keys` is needed or
-- present.
--
-- `ON DELETE CASCADE` puts the delete-ordering rule in the database instead of
-- in every caller that removes a node (ADR-0003). It reaches EXISTING databases
-- only through `migrate::ensure_cascade`: every statement here is
-- `CREATE TABLE IF NOT EXISTS`, so an existing table keeps the constraint it was
-- created with, and SQLite cannot ALTER one. Adding the clause here alone would
-- read as fixed on every fresh test database while leaving real stores broken.
--
-- `Graph::gc` still deletes referencing rows explicitly, and that is not
-- redundant: it is the guard on any backend that does not enforce foreign keys,
-- which includes the stubbed rusqlite one and stock SQLite generally.
CREATE TABLE IF NOT EXISTS edges (
  id         TEXT    NOT NULL PRIMARY KEY,
  src        TEXT    NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  dst        TEXT    NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  type       TEXT    NOT NULL,
  attributes TEXT    NOT NULL DEFAULT '{}',
  tx_from    INTEGER NOT NULL,
  tx_to      INTEGER NOT NULL DEFAULT 4102444800000
);

CREATE INDEX IF NOT EXISTS edges_out ON edges (src, type, tx_to);
CREATE INDEX IF NOT EXISTS edges_in  ON edges (dst, type, tx_to);
",
    );

    // Unconditional since ADR-0002 deleted the `cluster` feature. A database
    // created by an older build that lacked the feature simply gains these two
    // tables on its next open, because everything here is
    // `CREATE TABLE IF NOT EXISTS` and `Graph::open_with_clock` re-runs the
    // whole batch every time. That is also why neither table needs a
    // `migrate::` call: `migrate` exists for a COLUMN a fresh database gets
    // from this schema and an existing one does not, which is a different
    // problem from a missing table.
    //
    // The upgrade is self-healing rather than merely tolerable. An old database
    // arrives with `node_community` absent and gains it empty, and
    // `cluster_state` is empty too, so the fingerprint check reads "no prior
    // run" and forces a cold recompute on the first `clusters` call or GC tick.
    sql.push_str(
        "
CREATE TABLE IF NOT EXISTS node_community (
  node_id     TEXT    NOT NULL PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
  community   INTEGER NOT NULL,
  computed_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS node_community_by_community ON node_community (community);

CREATE TABLE IF NOT EXISTS cluster_state (
  edge_count         INTEGER NOT NULL,
  max_tx_from        INTEGER NOT NULL,
  computed_at        INTEGER NOT NULL,
  last_cold_start_at INTEGER NOT NULL
);

-- scope DEFAULT '' (never NULL) so (subject, scope) behaves as a normal
-- uniqueness constraint: SQLite treats NULLs as distinct in a PRIMARY KEY,
-- which would let multiple \"no scope\" rows coexist for the same subject.
CREATE TABLE IF NOT EXISTS entity_mention_state (
  subject              TEXT    NOT NULL,
  scope                TEXT    NOT NULL DEFAULT '',
  mentions_count       INTEGER NOT NULL,
  mentions_max_tx_from INTEGER NOT NULL,
  last_synthesized_at  INTEGER NOT NULL,
  PRIMARY KEY (subject, scope)
);

CREATE TABLE IF NOT EXISTS provenance_repair_state (
  id               INTEGER PRIMARY KEY CHECK (id = 1),
  last_repaired_at INTEGER NOT NULL
);
",
    );

    sql
}

#[cfg(all(test, feature = "backend-libsql"))]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::DefaultBackend;
    use tempfile::TempDir;

    async fn file_backend_at(name: &str) -> (TempDir, DefaultBackend) {
        // This test exercises schema shape, not read pooling.
        const ARBITRARY_POOL_SIZE: usize = 1;
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join(name);
        let backend = DefaultBackend::open(path.to_str().expect("utf8 path"), ARBITRARY_POOL_SIZE)
            .await
            .expect("open file-backed backend");
        (dir, backend)
    }

    async fn table_columns_with_pk(backend: &DefaultBackend, table: &str) -> Vec<(String, i64)> {
        backend
            .query(&format!("PRAGMA table_info({table})"), &[])
            .await
            .expect("query table_info")
            .iter()
            .map(|row| {
                (
                    row.get_string(1).expect("column name at index 1"),
                    row.get_i64(5).expect("pk ordinal at index 5"),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn fresh_database_gets_entity_mention_state_and_provenance_repair_state() {
        // Arrange
        let (_dir, backend) = file_backend_at("fresh.db").await;

        // Act
        backend
            .execute_batch(&schema(&GraphConfig::new(8)))
            .await
            .expect("apply schema to a fresh database");

        // Assert: entity_mention_state is keyed on (subject, scope), subject first.
        let mention_columns = table_columns_with_pk(&backend, "entity_mention_state").await;
        assert_eq!(
            mention_columns
                .iter()
                .find(|(name, _)| name == "subject")
                .map(|(_, pk)| *pk),
            Some(1),
            "subject must be the first primary key column"
        );
        assert_eq!(
            mention_columns
                .iter()
                .find(|(name, _)| name == "scope")
                .map(|(_, pk)| *pk),
            Some(2),
            "scope must be the second primary key column"
        );
        for column in [
            "mentions_count",
            "mentions_max_tx_from",
            "last_synthesized_at",
        ] {
            assert!(
                mention_columns.iter().any(|(name, _)| name == column),
                "entity_mention_state is missing column {column}"
            );
        }

        // Assert: provenance_repair_state carries the single-row watermark shape.
        let repair_columns = table_columns_with_pk(&backend, "provenance_repair_state").await;
        assert!(repair_columns.iter().any(|(name, _)| name == "id"));
        assert!(repair_columns
            .iter()
            .any(|(name, _)| name == "last_repaired_at"));
    }

    /// Applying `schema()` a second time is exactly what `Graph::open_with_clock`
    /// does on every open (see the comment above the unconditional block near
    /// the end of `schema()`), so the first application here stands in for a
    /// database created before these two tables existed, and the second for
    /// reopening it afterward.
    #[tokio::test]
    async fn reopening_a_database_that_predates_the_new_tables_creates_them_with_no_error() {
        // Arrange
        let (_dir, backend) = file_backend_at("reopen.db").await;
        backend
            .execute_batch(&schema(&GraphConfig::new(8)))
            .await
            .expect("apply schema the first time");

        // Act
        backend
            .execute_batch(&schema(&GraphConfig::new(8)))
            .await
            .expect("reapplying schema must not error");

        // Assert
        let mention_exists = backend
            .query(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='entity_mention_state'",
                &[],
            )
            .await
            .expect("query sqlite_master");
        assert_eq!(mention_exists[0].get_i64(0).unwrap(), 1);

        let repair_exists = backend
            .query(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='provenance_repair_state'",
                &[],
            )
            .await
            .expect("query sqlite_master");
        assert_eq!(repair_exists[0].get_i64(0).unwrap(), 1);
    }
}
