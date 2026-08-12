// SPDX-License-Identifier: AGPL-3.0-only
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

CREATE TABLE IF NOT EXISTS edges (
  id         TEXT    NOT NULL PRIMARY KEY,
  src        TEXT    NOT NULL REFERENCES nodes(id),
  dst        TEXT    NOT NULL REFERENCES nodes(id),
  type       TEXT    NOT NULL,
  attributes TEXT    NOT NULL DEFAULT '{}',
  tx_from    INTEGER NOT NULL,
  tx_to      INTEGER NOT NULL DEFAULT 4102444800000
);

CREATE INDEX IF NOT EXISTS edges_out ON edges (src, type, tx_to);
CREATE INDEX IF NOT EXISTS edges_in  ON edges (dst, type, tx_to);
",
    );

    if cfg!(feature = "cluster") {
        sql.push_str(
            "
CREATE TABLE IF NOT EXISTS node_community (
  node_id     TEXT    NOT NULL PRIMARY KEY REFERENCES nodes(id),
  community   INTEGER NOT NULL,
  computed_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS node_community_by_community ON node_community (community);
",
        );
    }

    sql
}
