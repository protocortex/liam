// SPDX-License-Identifier: Apache-2.0
//! Guarded, idempotent migration for existing databases.
//!
//! `schema.rs` is entirely `CREATE TABLE IF NOT EXISTS`, so editing a table's
//! DDL there does nothing for anyone who already has a database: the
//! statement is skipped wholesale. These helpers are how a fresh database and
//! an existing one end up with the same shape.
//!
//! Still deliberately not a migration framework: no version table, no
//! registry, no ordering. Each helper detects the state it cares about and
//! makes it true, so calling it on an already-correct database is a no-op and
//! the order they run in does not matter.
//!
//! Three exist. `add_column_if_missing` handles a column, through `ALTER TABLE`.
//! `ensure_cascade` handles a CONSTRAINT, which `ALTER TABLE` cannot change at
//! all, so it rebuilds the table. `ensure_usage_key` does the same for a primary
//! key.

use std::collections::HashSet;

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::ids::Millis;
use crate::types::{relation, Derivation};
use crate::value::Value;

/// Adds `column` to `table` with `type_and_constraints` (for example
/// `"TEXT NOT NULL DEFAULT 'unknown'"`) if it is not already present. Safe to
/// call every time a database is opened: a database that already has the
/// column is left untouched, and one that does not gets it added exactly
/// once.
///
/// Table name, column name, and type are parameters rather than baked in, so
/// this stays testable independent of any one caller's schema.
///
/// # Errors
/// Returns an error if `table` does not exist, or if `ALTER TABLE` fails for
/// any reason other than the column already existing.
///
/// Called from `Graph::open_with_clock` to add `producer` to the `nodes`
/// table on databases that predate that column.
pub async fn add_column_if_missing<B: Backend>(
    backend: &B,
    table: &str,
    column: &str,
    type_and_constraints: &str,
) -> Result<()> {
    if column_exists(backend, table, column).await? {
        return Ok(());
    }

    let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {type_and_constraints}");
    match backend.execute(&sql, &[]).await {
        Ok(_) => Ok(()),
        // Two processes can open the same store at the same moment and both
        // pass the `column_exists` check above before either runs its own
        // `ALTER TABLE`: the pre-check alone is not enough to make this
        // idempotent under that race, only the pre-check PLUS tolerating the
        // race's loser is. Without this, the losing process's `ALTER TABLE`
        // fails and it never starts.
        //
        // Every backend maps its native error into the opaque
        // `Error::Backend(String)` (see `error.rs`), specifically so this
        // crate never depends on one engine's error type. That is exactly
        // what forces the string match below: there is no typed
        // duplicate-column variant to match on instead, and there
        // deliberately will not be one, so this is not a shortcut to clean
        // up into a typed match later, it is the only match the trait as
        // designed allows.
        Err(Error::Backend(msg)) if msg.contains("duplicate column name") => Ok(()),
        Err(e) => Err(e),
    }
}

/// Gives `log_backfill_state` the `last_rowid` column the nodes phase of the
/// log backfill resumes from, and drops a node cursor an older build saved as
/// an id. The walk then starts the nodes again: a row the log already holds is
/// skipped, so nothing is logged twice, and the rows still to log follow
/// insertion order.
pub async fn add_backfill_node_rowid_cursor<B: Backend>(backend: &B) -> Result<()> {
    add_column_if_missing(backend, "log_backfill_state", "last_rowid", "INTEGER").await?;
    backend
        .execute(
            "UPDATE log_backfill_state SET last_id = NULL
             WHERE phase = 'nodes' AND last_id IS NOT NULL",
            &[],
        )
        .await?;
    Ok(())
}

/// Gives `item_usage_events` the producer in its key. A key cannot be altered in
/// place, so a table that lacks the `producer` column is rebuilt with it; its
/// rows keep the producer `''`, which counts as one legacy producer. A table
/// that has the column is left alone, so this is safe on every open.
pub async fn ensure_usage_key<B: Backend>(backend: &B) -> Result<()> {
    if column_exists(backend, "item_usage_events", "producer").await? {
        return Ok(());
    }
    backend
        .execute_batch(
            "BEGIN;
             DROP TABLE IF EXISTS item_usage_events_key_rebuild;
             CREATE TABLE item_usage_events_key_rebuild (
               event_id       TEXT    NOT NULL,
               lineage_key    TEXT    NOT NULL,
               producer       TEXT    NOT NULL DEFAULT '',
               day            INTEGER NOT NULL,
               hits           INTEGER NOT NULL,
               last_access_at INTEGER NOT NULL,
               PRIMARY KEY (event_id, lineage_key, producer)
             );
             INSERT INTO item_usage_events_key_rebuild
               (event_id, lineage_key, day, hits, last_access_at)
               SELECT event_id, lineage_key, day, hits, last_access_at FROM item_usage_events;
             DROP TABLE item_usage_events;
             ALTER TABLE item_usage_events_key_rebuild RENAME TO item_usage_events;
             CREATE INDEX IF NOT EXISTS item_usage_by_lineage ON item_usage_events (lineage_key);
             CREATE INDEX IF NOT EXISTS item_usage_by_day ON item_usage_events (day);
             COMMIT;",
        )
        .await
}

/// Tables whose `REFERENCES nodes(id)` must cascade. Fixed literals, never
/// caller input, which is what makes interpolating them into SQL below safe.
const CASCADING_TABLES: [&str; 2] = ["edges", "node_community"];

/// Give every `REFERENCES nodes(id)` on `CASCADING_TABLES` an `ON DELETE
/// CASCADE`, rebuilding any table that predates the clause (ADR-0003).
///
/// A constraint cannot be altered in place: SQLite has no `ALTER TABLE ... ALTER
/// CONSTRAINT`, and re-running the `CREATE TABLE IF NOT EXISTS` from `schema.rs`
/// silently skips an existing table. So the only way to reach a database that
/// already exists is to rebuild the table, and the only way to know whether it
/// needs one is to ask the database what constraint it currently holds.
///
/// Safe to call on every open. A database already carrying the clause is
/// detected and left alone.
pub async fn ensure_cascade<B: Backend>(backend: &B) -> Result<()> {
    for table in CASCADING_TABLES {
        if cascade_present(backend, table).await? {
            continue;
        }
        rebuild_with_cascade(backend, table).await?;
    }
    Ok(())
}

/// Whether every foreign key on `table` already cascades on delete.
///
/// `pragma_foreign_key_list` reports one row per declared constraint with an
/// `on_delete` column, so this asks the database rather than inferring from a
/// version number. A table with no foreign keys at all needs nothing, which is
/// why an empty result is `true` and not `false`.
async fn cascade_present<B: Backend>(backend: &B, table: &str) -> Result<bool> {
    let rows = backend
        .query(
            &format!("SELECT \"on_delete\" FROM pragma_foreign_key_list('{table}')"),
            &[],
        )
        .await?;
    for row in &rows {
        if row.get_string(0)? != "CASCADE" {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Rebuild `table` with `ON DELETE CASCADE` on its node references, preserving
/// rows and explicitly-created indexes.
///
/// The new DDL is derived from the table's OWN stored `CREATE` statement rather
/// than written out here, so a column added later is carried across without
/// this function knowing about it. `sqlite_master` stores that text with
/// leading comments and `IF NOT EXISTS` already stripped, confirmed by reading
/// it back (see `sqlite_master_strips_comments_and_if_not_exists`).
///
/// Indexes are filtered to `sql IS NOT NULL`: an index SQLite creates itself
/// for a PRIMARY KEY or UNIQUE has no SQL and comes back automatically with the
/// new table, while re-issuing one would be an error.
async fn rebuild_with_cascade<B: Backend>(backend: &B, table: &str) -> Result<()> {
    let create = match first_string(
        backend,
        &format!("SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'"),
    )
    .await?
    {
        Some(sql) => sql,
        // Nothing to rebuild. A caller that runs this before the schema exists
        // should not be told the database is broken.
        None => return Ok(()),
    };
    let indexes = all_strings(
        backend,
        &format!(
            "SELECT sql FROM sqlite_master
             WHERE type='index' AND tbl_name='{table}' AND sql IS NOT NULL"
        ),
    )
    .await?;

    let staging = format!("{table}_cascade_rebuild");
    let new_create = create.replacen(table, &staging, 1).replace(
        "REFERENCES nodes(id)",
        "REFERENCES nodes(id) ON DELETE CASCADE",
    );

    // `PRAGMA foreign_keys` is a no-op inside a transaction, so the toggles sit
    // OUTSIDE the BEGIN/COMMIT. Enforcement has to be off for the rebuild
    // itself, because dropping the old table would otherwise trip the very
    // constraints being replaced. `Backend::execute_batch` maps to libSQL's
    // plain batch, which does NOT wrap in a transaction (its
    // `execute_transactional_batch` is the one that does), so this sequence
    // survives intact. Confirmed by running it, not from the docs.
    //
    // `DROP TABLE IF EXISTS` on the staging name first, so a process that died
    // mid-rebuild, or lost a race to another process, does not wedge every
    // later start on a leftover table.
    let mut batch = String::from("PRAGMA foreign_keys=OFF;\nBEGIN;\n");
    batch.push_str(&format!("DROP TABLE IF EXISTS {staging};\n"));
    batch.push_str(&new_create);
    batch.push_str(";\n");
    batch.push_str(&format!("INSERT INTO {staging} SELECT * FROM {table};\n"));
    batch.push_str(&format!("DROP TABLE {table};\n"));
    batch.push_str(&format!("ALTER TABLE {staging} RENAME TO {table};\n"));
    for index in &indexes {
        batch.push_str(index);
        batch.push_str(";\n");
    }
    batch.push_str("COMMIT;\nPRAGMA foreign_keys=ON;\n");
    backend.execute_batch(&batch).await?;

    // Turns the enforcement-off window from an assumption into a check.
    // `execute_batch` discards results, so this runs as its own query: a
    // violation comes back as rows, not as an error.
    let violations = backend.query("PRAGMA foreign_key_check", &[]).await?;
    if !violations.is_empty() {
        return Err(Error::Backend(format!(
            "cascade rebuild of {table} left {} foreign key violation(s)",
            violations.len()
        )));
    }
    Ok(())
}

async fn first_string<B: Backend>(backend: &B, sql: &str) -> Result<Option<String>> {
    let rows = backend.query(sql, &[]).await?;
    match rows.first() {
        Some(row) => Ok(Some(row.get_string(0)?)),
        None => Ok(None),
    }
}

async fn all_strings<B: Backend>(backend: &B, sql: &str) -> Result<Vec<String>> {
    let rows = backend.query(sql, &[]).await?;
    rows.iter().map(|row| row.get_string(0)).collect()
}

/// Trim whitespace from every already-stored `nodes.scope` value, so a row
/// written before `validate_scope` (`graph.rs`) existed matches the same
/// trimmed value a scope-filtered query now searches for. Safe to call on
/// every open: a database with no untrimmed scope is left untouched.
///
/// Each row is read into Rust and trimmed with the same `str::trim()`
/// `validate_scope` uses internally, then only rows that actually changed
/// get their own `UPDATE`, one row at a time. SQLite's single-argument SQL
/// `TRIM()` strips only ASCII spaces, not the full Unicode whitespace set
/// (tabs, newlines, carriage returns) that `str::trim()` strips, so a bulk
/// SQL `TRIM()` would silently leave those values both untouched and
/// unlogged; reading and trimming in Rust is what makes this exactly what
/// `validate_scope` itself does on write.
///
/// Only whitespace is fixed here, mechanically. A scope that is still
/// invalid after trimming (empty, over the length cap, a disallowed
/// character, a malformed `/` shape) is NOT rewritten or dropped: guessing
/// a replacement would be a worse kind of data loss than the
/// stops-matching-a-scope-filter state this otherwise leaves it in. Each
/// such row is logged instead, so whoever operates this store can see and
/// correct it by hand.
pub async fn normalize_scope_column<B: Backend>(backend: &B) -> Result<()> {
    let rows = backend
        .query("SELECT id, scope FROM nodes WHERE scope IS NOT NULL", &[])
        .await?;
    for row in &rows {
        let id = row.get_string(0)?;
        let scope = row.get_string(1)?;
        let trimmed = scope.trim();
        if trimmed != scope {
            backend
                .execute(
                    "UPDATE nodes SET scope = ?1 WHERE id = ?2",
                    &[trimmed.into(), id.as_str().into()],
                )
                .await?;
        }
        if crate::graph::validate_scope(&Some(trimmed.to_string())).is_err() {
            tracing::warn!(
                node = %id,
                scope = %scope,
                "stored scope does not conform to the new validation rules \
                 (after whitespace trim); it will not match any future \
                 scope-filtered query until corrected"
            );
        }
    }
    Ok(())
}

/// Ids per `UPDATE`, so the write lock is held for one small statement at a
/// time and never for the whole set of pages.
const BACKFILL_CHUNK: usize = 200;

/// Marks the entity pages that predate the `derivation` column as inferred,
/// once, and records that it ran. Returns how many rows it marked, `0` when
/// the pass already ran. See `mark_inferred_pages` for the rule.
pub(crate) async fn backfill_derivation<B: Backend>(backend: &B, now: Millis) -> Result<u64> {
    let done = backend
        .query("SELECT 1 FROM derivation_backfill_state WHERE id = 1", &[])
        .await?;
    if !done.is_empty() {
        return Ok(0);
    }
    let flagged = mark_inferred_pages(backend, None).await?;
    record_derivation_backfill(backend, now, flagged).await?;
    tracing::info!(flagged, "derivation backfill: entity pages marked inferred");
    let stubs = unlinked_stubs(backend).await?;
    tracing::info!(
        stubs,
        "derivation backfill: nodes shaped like a page but with no mentions edge stay recorded"
    );
    Ok(flagged)
}

/// Records that the derivation pass ran, so the open-time pass does not run
/// again, and how many rows it marked.
pub(crate) async fn record_derivation_backfill<B: Backend>(
    backend: &B,
    now: Millis,
    flagged: u64,
) -> Result<()> {
    backend
        .execute(
            "INSERT OR IGNORE INTO derivation_backfill_state (id, completed_at, flagged)
             VALUES (1, ?1, ?2)",
            &[now.into(), Value::Int(flagged as i64)],
        )
        .await?;
    Ok(())
}

/// The rule behind the backfill, with no gate: a page is a node that is the source of any `mentions` edge, live or closed,
/// whose subject is its trimmed, lowercased label, which is what
/// `NewNode::entity` stores. The edge alone is not enough, because `relate`
/// accepts a `mentions` edge from any node.
///
/// The subject test runs in Rust: SQLite's `lower()` folds ASCII only, and the
/// constructor lowercases with Unicode rules. Only rows still `recorded` are
/// touched, so a rerun, or a race with another process, changes nothing. The
/// writes are one statement per `BACKFILL_CHUNK` ids.
///
/// `only` limits the pass to those node ids, as a replay does for the nodes of
/// events that carry no derivation; a node written with one is never touched.
pub(crate) async fn mark_inferred_pages<B: Backend>(
    backend: &B,
    only: Option<&[String]>,
) -> Result<u64> {
    let allowed: Option<HashSet<&str>> = only.map(|ids| ids.iter().map(String::as_str).collect());
    let candidates = backend
        .query(
            "SELECT n.id, n.label, n.subject FROM nodes n
             WHERE n.derivation = ?1 AND n.subject IS NOT NULL
               AND EXISTS (SELECT 1 FROM edges e WHERE e.src = n.id AND e.type = ?2)
             ORDER BY n.id",
            &[
                Derivation::Recorded.as_str().into(),
                relation::MENTIONS.into(),
            ],
        )
        .await?;
    let mut pages = Vec::new();
    for row in &candidates {
        let id = row.get_string(0)?;
        let wanted = allowed.as_ref().is_none_or(|ids| ids.contains(id.as_str()));
        if wanted && subject_is_label(&row.get_string(2)?, &row.get_string(1)?) {
            pages.push(id);
        }
    }
    let mut flagged = 0;
    for chunk in pages.chunks(BACKFILL_CHUNK) {
        let marks = (3..chunk.len() + 3)
            .map(|n| format!("?{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut params = vec![
            Derivation::Inferred.as_str().into(),
            Derivation::Recorded.as_str().into(),
        ];
        params.extend(chunk.iter().map(|id| Value::from(id.as_str())));
        flagged += backend
            .execute(
                &format!(
                    "UPDATE nodes SET derivation = ?1 WHERE derivation = ?2 AND id IN ({marks})"
                ),
                &params,
            )
            .await?;
    }
    Ok(flagged)
}

fn subject_is_label(subject: &str, label: &str) -> bool {
    subject == label.trim().to_lowercase()
}

/// How many recorded nodes have a page's subject but no `mentions` edge, such
/// as an entity that was never synthesized. They stay recorded, so the count
/// tells the operator what the rule leaves out.
async fn unlinked_stubs<B: Backend>(backend: &B) -> Result<usize> {
    let rows = backend
        .query(
            "SELECT n.label, n.subject FROM nodes n
             WHERE n.derivation = ?1 AND n.subject IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.src = n.id AND e.type = ?2)",
            &[
                Derivation::Recorded.as_str().into(),
                relation::MENTIONS.into(),
            ],
        )
        .await?;
    let mut stubs = 0;
    for row in &rows {
        if subject_is_label(&row.get_string(1)?, &row.get_string(0)?) {
            stubs += 1;
        }
    }
    Ok(stubs)
}

/// Whether `table` already has `column`, read from `PRAGMA table_info`.
///
/// Errors if `table` does not exist. `PRAGMA table_info` on a nonexistent
/// table reports zero rows rather than an error (confirmed empirically, see
/// `pragma_table_info_on_a_missing_table_returns_no_rows` below), so without
/// this check a missing table would look identical to a table that simply
/// lacks the column, and silently running `ALTER TABLE` against a table that
/// does not exist would fail confusingly instead of failing clearly here.
async fn column_exists<B: Backend>(backend: &B, table: &str, column: &str) -> Result<bool> {
    let rows = backend
        .query(&format!("PRAGMA table_info({table})"), &[])
        .await?;
    if rows.is_empty() {
        return Err(Error::Backend(format!("no such table: {table}")));
    }
    // `PRAGMA table_info` returns one row per column, shaped
    // (cid, name, type, notnull, dflt_value, pk). Confirmed against the
    // pinned libsql by querying it back rather than assumed, the same
    // way S1 confirmed `journal_mode` by reading it back: see
    // `pragma_table_info_reports_the_column_name_at_index_1` below.
    for row in &rows {
        if row.get_string(1)? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// `:memory:` would work here too, but a file keeps this consistent with
    /// the rest of S1's file-backed harness and with how `add_column_if_missing`
    /// is actually used (against a real, persistent store).
    async fn file_backend_at(name: &str) -> (TempDir, DefaultBackend) {
        // This test exercises schema migration, not read pooling, so the
        // pool size is arbitrary; 1 keeps it minimal rather than implying
        // some other value matters here.
        const ARBITRARY_POOL_SIZE: usize = 1;
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join(name);
        let backend = DefaultBackend::open(path.to_str().expect("utf8 path"), ARBITRARY_POOL_SIZE)
            .await
            .expect("open file-backed backend");
        (dir, backend)
    }

    async fn column_names(backend: &DefaultBackend, table: &str) -> Vec<String> {
        backend
            .query(&format!("PRAGMA table_info({table})"), &[])
            .await
            .expect("query table_info")
            .iter()
            .map(|row| row.get_string(1).expect("column name at index 1"))
            .collect()
    }

    /// Grounds the assumption `column_exists` relies on: the column name
    /// sits at index 1 of each `PRAGMA table_info` row, not assumed but read
    /// back from a real table with a known, distinctive column name.
    #[tokio::test]
    async fn pragma_table_info_reports_the_column_name_at_index_1() {
        // Arrange
        let (_dir, backend) = file_backend_at("shape.db").await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, distinctive_name TEXT)")
            .await
            .expect("create table");

        // Act
        let rows = backend
            .query("PRAGMA table_info(t)", &[])
            .await
            .expect("query table_info");

        // Assert: two rows, one per column, name readable as text at index 1.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get_string(1).unwrap(), "id");
        assert_eq!(rows[1].get_string(1).unwrap(), "distinctive_name");
    }

    /// Grounds the other assumption `column_exists` relies on: a pragma
    /// against a table that does not exist comes back empty rather than as
    /// an error, so `add_column_if_missing` has to check for that itself
    /// rather than relying on the pragma to fail.
    #[tokio::test]
    async fn pragma_table_info_on_a_missing_table_returns_no_rows() {
        // Arrange
        let (_dir, backend) = file_backend_at("empty.db").await;

        // Act
        let rows = backend
            .query("PRAGMA table_info(nonexistent)", &[])
            .await
            .expect("query table_info succeeds even for a missing table");

        // Assert
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn adds_the_column_when_it_is_absent() {
        // Arrange
        let (_dir, backend) = file_backend_at("add.db").await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");

        // Act
        add_column_if_missing(&backend, "t", "producer", "TEXT NOT NULL DEFAULT 'unknown'")
            .await
            .expect("add column");

        // Assert: read the pragma back rather than trusting the return value.
        let names = column_names(&backend, "t").await;
        assert!(names.contains(&"producer".to_string()));
    }

    #[tokio::test]
    async fn running_it_again_is_a_no_op() {
        // Arrange
        let (_dir, backend) = file_backend_at("noop.db").await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");
        add_column_if_missing(&backend, "t", "producer", "TEXT NOT NULL DEFAULT 'unknown'")
            .await
            .expect("first add column");

        // Act: run it again against a table that already has the column.
        let result =
            add_column_if_missing(&backend, "t", "producer", "TEXT NOT NULL DEFAULT 'unknown'")
                .await;

        // Assert: no error, and the column still exists exactly once.
        result.expect("second call is a no-op, not an error");
        let names = column_names(&backend, "t").await;
        assert_eq!(names.iter().filter(|n| *n == "producer").count(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_concurrent_migrations_both_succeed_and_the_column_exists_once() {
        // Arrange
        let (_dir, backend) = file_backend_at("race.db").await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");
        let backend = Arc::new(backend);
        const CALLERS: usize = 8;

        // Act: fire several migrate calls concurrently, from separate tasks
        // on separate threads, so the pre-check in each has a real chance to
        // run before any of the others has committed its `ALTER TABLE`.
        // Calling these sequentially would never exercise the race this
        // helper exists to survive.
        let handles: Vec<_> = (0..CALLERS)
            .map(|_| {
                let backend = backend.clone();
                tokio::spawn(async move {
                    add_column_if_missing(
                        backend.as_ref(),
                        "t",
                        "producer",
                        "TEXT NOT NULL DEFAULT 'unknown'",
                    )
                    .await
                })
            })
            .collect();

        // Assert: every caller reports success, including whichever lost the
        // race to the duplicate-column error, and the column exists exactly
        // once regardless of how many callers raced.
        for handle in handles {
            handle
                .await
                .expect("migrate task did not panic")
                .expect("every concurrent caller reports success");
        }
        let names = column_names(&backend, "t").await;
        assert_eq!(names.iter().filter(|n| *n == "producer").count(), 1);
    }

    #[tokio::test]
    async fn a_table_that_does_not_exist_is_an_error_not_a_silent_success() {
        // Arrange
        let (_dir, backend) = file_backend_at("missing_table.db").await;

        // Act
        let result =
            add_column_if_missing(&backend, "nonexistent", "producer", "TEXT NOT NULL").await;

        // Assert
        assert!(result.is_err());
    }

    /// Pins the literal wording the string match in `add_column_if_missing`
    /// depends on: confirmed empirically against the pinned libsql (not assumed
    /// generic SQLite wording) by running the same `ALTER TABLE ADD COLUMN`
    /// twice outside the helper and reading the second error back. Observed:
    /// `` backend: SQLite failure: `duplicate column name: producer` ``. If
    /// libsql ever changes this wording, this test catches it before the
    /// tolerance in `add_column_if_missing` silently stops matching.
    #[tokio::test]
    async fn duplicate_column_error_contains_the_text_the_helper_matches_on() {
        // Arrange
        let (_dir, backend) = file_backend_at("wording.db").await;
        backend
            .execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            .await
            .expect("create table");
        backend
            .execute("ALTER TABLE t ADD COLUMN producer TEXT", &[])
            .await
            .expect("first alter");

        // Act
        let err = backend
            .execute("ALTER TABLE t ADD COLUMN producer TEXT", &[])
            .await
            .expect_err("second alter should fail");

        // Assert
        assert!(matches!(err, Error::Backend(ref msg) if msg.contains("duplicate column name")));
    }

    // ---- ensure_cascade (ADR-0003) ----

    /// An "old" database: the shape `schema.rs` produced before ADR-0003, with
    /// plain REFERENCES, two explicit indexes, and a row to preserve.
    async fn legacy_shape(backend: &DefaultBackend) {
        backend
            .execute_batch(
                "CREATE TABLE nodes (rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE);
                 CREATE TABLE edges (
                   id   TEXT NOT NULL PRIMARY KEY,
                   src  TEXT NOT NULL REFERENCES nodes(id),
                   dst  TEXT NOT NULL REFERENCES nodes(id),
                   type TEXT NOT NULL
                 );
                 CREATE INDEX edges_out ON edges (src, type);
                 CREATE INDEX edges_in  ON edges (dst, type);
                 CREATE TABLE node_community (
                   node_id   TEXT NOT NULL PRIMARY KEY REFERENCES nodes(id),
                   community INTEGER NOT NULL
                 );
                 INSERT INTO nodes (id) VALUES ('a'), ('b');
                 INSERT INTO edges VALUES ('e1', 'a', 'b', 'mentions');
                 INSERT INTO node_community VALUES ('a', 7);",
            )
            .await
            .expect("legacy schema");
    }

    async fn on_delete_of(backend: &DefaultBackend, table: &str) -> Vec<String> {
        backend
            .query(
                &format!("SELECT \"on_delete\" FROM pragma_foreign_key_list('{table}')"),
                &[],
            )
            .await
            .expect("foreign_key_list")
            .iter()
            .map(|r| r.get_string(0).expect("on_delete"))
            .collect()
    }

    async fn count(backend: &DefaultBackend, sql: &str) -> i64 {
        backend.query(sql, &[]).await.expect("count")[0]
            .get_i64(0)
            .expect("integer")
    }

    /// Grounds the assumption `rebuild_with_cascade` rests on: the stored
    /// `CREATE` text has leading comments and `IF NOT EXISTS` already removed,
    /// so it can be reused directly as the staging table's DDL.
    #[tokio::test]
    async fn sqlite_master_strips_comments_and_if_not_exists() {
        let (_dir, backend) = file_backend_at("master.db").await;
        backend
            .execute_batch(
                "-- a leading comment, as schema.rs has
                 CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY)",
            )
            .await
            .expect("create");

        let sql = first_string(&backend, "SELECT sql FROM sqlite_master WHERE name='t'")
            .await
            .expect("query")
            .expect("one row");

        assert!(sql.starts_with("CREATE TABLE t"), "got: {sql}");
        assert!(!sql.contains("comment"), "comment leaked into stored sql");
        assert!(!sql.contains("IF NOT EXISTS"), "IF NOT EXISTS was kept");
    }

    /// Grounds the index filter: an index SQLite creates for a PRIMARY KEY has
    /// no SQL and must not be re-issued, while an explicit one does.
    #[tokio::test]
    async fn autoindexes_have_no_sql_but_explicit_indexes_do() {
        let (_dir, backend) = file_backend_at("idx.db").await;
        backend
            .execute_batch(
                "CREATE TABLE t (id TEXT PRIMARY KEY, v TEXT);
                 CREATE INDEX t_v ON t (v);",
            )
            .await
            .expect("create");

        let explicit = all_strings(
            &backend,
            "SELECT sql FROM sqlite_master
             WHERE type='index' AND tbl_name='t' AND sql IS NOT NULL",
        )
        .await
        .expect("query");
        let all = count(
            &backend,
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND tbl_name='t'",
        )
        .await;

        assert_eq!(explicit.len(), 1, "only the explicit index carries sql");
        assert_eq!(all, 2, "but both indexes exist");
    }

    #[tokio::test]
    async fn a_legacy_database_gains_cascade_keeping_its_rows_and_indexes() {
        let (_dir, backend) = file_backend_at("legacy.db").await;
        legacy_shape(&backend).await;
        assert_eq!(
            on_delete_of(&backend, "edges").await,
            ["NO ACTION", "NO ACTION"]
        );

        ensure_cascade(&backend).await.expect("migrate");

        assert_eq!(
            on_delete_of(&backend, "edges").await,
            ["CASCADE", "CASCADE"]
        );
        assert_eq!(on_delete_of(&backend, "node_community").await, ["CASCADE"]);
        assert_eq!(count(&backend, "SELECT COUNT(*) FROM edges").await, 1);
        assert_eq!(
            count(&backend, "SELECT COUNT(*) FROM node_community").await,
            1
        );
        assert_eq!(
            count(
                &backend,
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type='index' AND tbl_name='edges' AND sql IS NOT NULL"
            )
            .await,
            2,
            "both explicit indexes must come back"
        );
        // No staging table left behind.
        assert_eq!(
            count(
                &backend,
                "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%_cascade_rebuild'"
            )
            .await,
            0
        );
    }

    #[tokio::test]
    async fn after_migrating_a_node_delete_takes_its_references_with_it() {
        // The point of the whole exercise: the database enforces the ordering
        // that `gc` previously had to get right by hand.
        let (_dir, backend) = file_backend_at("fires.db").await;
        legacy_shape(&backend).await;
        ensure_cascade(&backend).await.expect("migrate");

        backend
            .execute("DELETE FROM nodes WHERE id = 'a'", &[])
            .await
            .expect("delete must not fail on a referenced node");

        assert_eq!(count(&backend, "SELECT COUNT(*) FROM edges").await, 0);
        assert_eq!(
            count(&backend, "SELECT COUNT(*) FROM node_community").await,
            0
        );
    }

    #[tokio::test]
    async fn running_ensure_cascade_again_changes_nothing() {
        let (_dir, backend) = file_backend_at("idempotent.db").await;
        legacy_shape(&backend).await;
        ensure_cascade(&backend).await.expect("first");

        ensure_cascade(&backend).await.expect("second is a no-op");

        assert_eq!(
            on_delete_of(&backend, "edges").await,
            ["CASCADE", "CASCADE"]
        );
        assert_eq!(count(&backend, "SELECT COUNT(*) FROM edges").await, 1);
    }

    #[tokio::test]
    async fn a_column_added_after_this_migration_was_written_still_survives_it() {
        // The new DDL is derived from the table's own stored CREATE, so a column
        // this function has never heard of is carried across. Writing the
        // columns out by hand here would silently drop it.
        let (_dir, backend) = file_backend_at("extra_col.db").await;
        legacy_shape(&backend).await;
        backend
            .execute(
                "ALTER TABLE edges ADD COLUMN future_col TEXT DEFAULT 'kept'",
                &[],
            )
            .await
            .expect("add column");

        ensure_cascade(&backend).await.expect("migrate");

        let names: Vec<String> = column_names(&backend, "edges").await;
        assert!(names.contains(&"future_col".to_string()), "{names:?}");
        assert_eq!(
            backend
                .query("SELECT future_col FROM edges", &[])
                .await
                .expect("select")[0]
                .get_string(0)
                .expect("text"),
            "kept"
        );
    }

    #[tokio::test]
    async fn a_missing_table_is_not_an_error() {
        // `ensure_cascade` runs on open, and a caller could reach it before the
        // schema exists. That should be a no-op, not a failure to start.
        let (_dir, backend) = file_backend_at("no_tables.db").await;

        ensure_cascade(&backend).await.expect("no tables is fine");
    }

    // ---- normalize_scope_column ----

    /// A backend carrying the real `nodes` table (via `crate::schema::schema`),
    /// so a raw `INSERT` below bypasses `validate_scope` (`graph.rs`) exactly
    /// the way a row written before that validation existed would have.
    async fn nodes_schema_backend_at(name: &str) -> (TempDir, DefaultBackend) {
        let (dir, backend) = file_backend_at(name).await;
        let config = crate::types::GraphConfig::new(3);
        backend
            .execute_batch(&crate::schema::schema(&config))
            .await
            .expect("create nodes schema");
        (dir, backend)
    }

    /// Inserts a `nodes` row directly with the given raw `scope`, skipping
    /// `validate_scope` entirely, to simulate a row written before this
    /// migration shipped. Every other column either has a schema default or
    /// is filled with an arbitrary valid placeholder.
    async fn insert_raw_node(backend: &DefaultBackend, id: &str, scope: Value) {
        backend
            .execute(
                "INSERT INTO nodes (id, kind, label, content, scope, valid_from, tx_from)
                 VALUES (?1, 'fact', 'label', 'content', ?2, 0, 0)",
                &[id.into(), scope],
            )
            .await
            .expect("insert raw node");
    }

    /// Reads `scope` back for `id`, as `None` for `NULL` and `Some` for text.
    /// Not `Row::get_string`, which errors on `NULL`, exactly the case this
    /// helper needs to distinguish.
    async fn scope_of(backend: &DefaultBackend, id: &str) -> Option<String> {
        let rows = backend
            .query("SELECT scope FROM nodes WHERE id = ?1", &[id.into()])
            .await
            .expect("query scope");
        match &rows[0].0[0] {
            Value::Null => None,
            Value::Text(s) => Some(s.clone()),
            other => panic!("scope is neither NULL nor text: {other:?}"),
        }
    }

    #[tokio::test]
    async fn trims_whitespace_from_a_stored_scope() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("trim.db").await;
        insert_raw_node(&backend, "n1", Value::Text(" proj-a ".to_string())).await;

        // Act
        normalize_scope_column(&backend).await.expect("normalize");

        // Assert
        assert_eq!(scope_of(&backend, "n1").await, Some("proj-a".to_string()));
    }

    #[tokio::test]
    async fn trims_a_trailing_tab_that_sql_trim_would_leave_untouched() {
        // Arrange: a trailing tab, not a space. SQLite's single-argument
        // TRIM() only strips ASCII spaces, so a bulk SQL TRIM() leaves this
        // exact value untouched; only Rust's str::trim() catches it.
        let (_dir, backend) = nodes_schema_backend_at("tab.db").await;
        insert_raw_node(&backend, "n1", Value::Text("proj-a\t".to_string())).await;

        // Act
        normalize_scope_column(&backend).await.expect("normalize");

        // Assert
        assert_eq!(scope_of(&backend, "n1").await, Some("proj-a".to_string()));
    }

    #[tokio::test]
    async fn trims_leading_and_trailing_newlines_and_carriage_returns() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("newline.db").await;
        insert_raw_node(&backend, "n1", Value::Text("\nproj-a\r\n".to_string())).await;

        // Act
        normalize_scope_column(&backend).await.expect("normalize");

        // Assert
        assert_eq!(scope_of(&backend, "n1").await, Some("proj-a".to_string()));
    }

    #[tokio::test]
    async fn an_already_clean_scope_is_left_unchanged() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("clean.db").await;
        insert_raw_node(&backend, "n1", Value::Text("proj-a".to_string())).await;

        // Act
        normalize_scope_column(&backend).await.expect("normalize");

        // Assert
        assert_eq!(scope_of(&backend, "n1").await, Some("proj-a".to_string()));
    }

    #[tokio::test]
    async fn a_scope_still_invalid_after_trim_is_left_exactly_as_stored() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("invalid.db").await;
        insert_raw_node(&backend, "n1", Value::Text("bad scope!".to_string())).await;

        // Act
        normalize_scope_column(&backend).await.expect("normalize");

        // Assert: not rewritten, not nulled.
        assert_eq!(
            scope_of(&backend, "n1").await,
            Some("bad scope!".to_string())
        );
    }

    #[tokio::test]
    async fn a_whitespace_only_scope_becomes_an_empty_string_and_stays_there() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("whitespace.db").await;
        insert_raw_node(&backend, "n1", Value::Text("   ".to_string())).await;

        // Act
        normalize_scope_column(&backend).await.expect("normalize");

        // Assert
        assert_eq!(scope_of(&backend, "n1").await, Some(String::new()));
    }

    #[tokio::test]
    async fn trim_and_the_post_trim_validity_check_both_apply_together() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("trim_and_invalid.db").await;
        insert_raw_node(&backend, "n1", Value::Text(" bad@scope ".to_string())).await;

        // Act
        normalize_scope_column(&backend).await.expect("normalize");

        // Assert: trimmed, but still left as-is (not conforming).
        assert_eq!(
            scope_of(&backend, "n1").await,
            Some("bad@scope".to_string())
        );
    }

    #[tokio::test]
    async fn a_null_scope_is_left_untouched() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("null.db").await;
        insert_raw_node(&backend, "n1", Value::Null).await;

        // Act
        let result = normalize_scope_column(&backend).await;

        // Assert
        result.expect("no error on a NULL scope");
        assert_eq!(scope_of(&backend, "n1").await, None);
    }

    #[tokio::test]
    async fn running_normalize_scope_column_again_is_a_no_op() {
        // Arrange
        let (_dir, backend) = nodes_schema_backend_at("normalize_idempotent.db").await;
        insert_raw_node(&backend, "n1", Value::Text(" proj-a ".to_string())).await;
        insert_raw_node(&backend, "n2", Value::Text("bad scope!".to_string())).await;
        normalize_scope_column(&backend).await.expect("first run");

        // Act
        normalize_scope_column(&backend)
            .await
            .expect("second run is a no-op");

        // Assert
        assert_eq!(scope_of(&backend, "n1").await, Some("proj-a".to_string()));
        assert_eq!(
            scope_of(&backend, "n2").await,
            Some("bad scope!".to_string())
        );
    }

    #[tokio::test]
    async fn running_normalize_scope_column_again_on_a_tab_padded_scope_is_a_no_op() {
        // Arrange: the first run must trim the tab a bulk SQL TRIM() would
        // have missed; the second run must then change nothing further.
        let (_dir, backend) = nodes_schema_backend_at("normalize_idempotent_tab.db").await;
        insert_raw_node(&backend, "n1", Value::Text("proj-a\t".to_string())).await;
        normalize_scope_column(&backend).await.expect("first run");
        assert_eq!(scope_of(&backend, "n1").await, Some("proj-a".to_string()));

        // Act
        normalize_scope_column(&backend)
            .await
            .expect("second run is a no-op");

        // Assert
        assert_eq!(scope_of(&backend, "n1").await, Some("proj-a".to_string()));
    }
}
