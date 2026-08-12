// SPDX-License-Identifier: AGPL-3.0-only
//! Guarded, idempotent column migration for existing databases.
//!
//! `schema.rs` is entirely `CREATE TABLE IF NOT EXISTS`, so editing a table's
//! DDL there does nothing for anyone who already has a database: the
//! statement is skipped wholesale. This is the one mechanism this crate has
//! for a column that a fresh database gets from the schema but an existing
//! one does not. It is deliberately not a migration framework: no version
//! table, no registry, no ordering of multiple migrations. One helper, called
//! explicitly wherever a column needs to exist on an old database.

use crate::backend::Backend;
use crate::error::{Error, Result};

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
    // pinned libsql 0.9.30 by querying it back rather than assumed, the same
    // way S1 confirmed `journal_mode` by reading it back: see
    // `pragma_table_info_reports_the_column_name_at_index_1` below.
    for row in &rows {
        if row.get_string(1)? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(all(test, feature = "backend-libsql"))]
mod tests {
    use super::*;
    use crate::DefaultBackend;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// `:memory:` would work here too, but a file keeps this consistent with
    /// the rest of S1's file-backed harness and with how `add_column_if_missing`
    /// is actually used (against a real, persistent store).
    async fn file_backend_at(name: &str) -> (TempDir, DefaultBackend) {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join(name);
        let backend = DefaultBackend::open(path.to_str().expect("utf8 path"))
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
    /// depends on: confirmed empirically against libsql 0.9.30 (not assumed
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
}
