// SPDX-License-Identifier: Apache-2.0
//! The database half of a delete's scrub step. The files and the dedup filter
//! are purged through the lineage registry; this pushes the deleted text out of
//! the database file and its write-ahead log.

use liam_store::{DefaultGraph, Error, ScrubReport};

/// How a scrub ended.
#[derive(Debug, PartialEq, Eq)]
pub enum ScrubOutcome {
    Done(ScrubReport),
    /// A reader still held the log. The step stays open and runs again.
    Incomplete,
}

/// Scrubs the database. A log a reader still holds is an outcome to retry, any
/// other failure is an error.
pub async fn scrub_database(graph: &DefaultGraph) -> anyhow::Result<ScrubOutcome> {
    match graph.scrub().await {
        Ok(report) => Ok(ScrubOutcome::Done(report)),
        Err(Error::ScrubIncomplete {
            busy,
            log_frames,
            checkpointed,
        }) => {
            tracing::warn!(
                busy,
                log_frames,
                checkpointed,
                "scrub left open, a reader holds the log"
            );
            Ok(ScrubOutcome::Incomplete)
        }
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deletion::test_support::sealed_subjects;

    #[tokio::test]
    async fn a_scrub_of_an_idle_database_is_done() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let subjects = sealed_subjects(dir.path()).await;

        // Act
        let outcome = scrub_database(&subjects.store).await.unwrap();

        // Assert
        assert!(matches!(outcome, ScrubOutcome::Done(report) if report.checkpoint.is_complete()));
    }

    #[tokio::test]
    async fn a_reader_holding_the_log_leaves_the_step_open() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let subjects = sealed_subjects(dir.path()).await;
        let database = libsql::Builder::new_local(dir.path().join("liam.db").to_str().unwrap())
            .build()
            .await
            .unwrap();
        let reader = database.connect().unwrap();
        reader.execute("BEGIN", ()).await.unwrap();
        let mut rows = reader
            .query("SELECT COUNT(*) FROM nodes", ())
            .await
            .unwrap();
        while rows.next().await.unwrap().is_some() {}
        subjects
            .store
            .insert(liam_store::NewNode::now(
                "fact",
                "later",
                "after the reader",
            ))
            .await
            .unwrap();

        // Act
        let held = scrub_database(&subjects.store).await.unwrap();
        drop(rows);
        drop(reader);
        let released = scrub_database(&subjects.store).await.unwrap();

        // Assert
        assert_eq!(held, ScrubOutcome::Incomplete);
        assert!(matches!(released, ScrubOutcome::Done(_)));
    }
}
