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
",
    );

    sql
}
