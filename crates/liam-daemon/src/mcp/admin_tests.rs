// SPDX-License-Identifier: Apache-2.0
//! The admin tools: who may call them, that `delete` only reports, and which
//! tools a client sees.

use std::collections::BTreeSet;
use std::sync::Arc;

use futures_util::StreamExt;
use liam_daemon::event_log;
use liam_log::crypto::KeyState;
use liam_store::{NewNode, BEGINNING};
use rmcp::handler::server::wrapper::Parameters;
use tempfile::TempDir;

use super::delete::DeleteArgs;
use super::MemoryServer;
use crate::config::AdminConfig;
use crate::deletion::test_support::{sealed_subjects, Subjects};
use crate::deletion::{job_file_path_for, DeletePreview};
use crate::log_startup::test_support::{database_in, embedder, log_dir_in};

const ADMIN: &str = "ops";
const OTHER_SCOPE: &str = "elsewhere";
const REFUSED: &str = "[admin].producers";

struct Fixture {
    dir: TempDir,
    subjects: Subjects,
    server: MemoryServer,
}

impl Fixture {
    async fn new(admins: &[&str]) -> Self {
        let dir = TempDir::new().unwrap();
        let subjects = sealed_subjects(dir.path()).await;
        let preview = DeletePreview::new(
            &subjects.config,
            &database_in(dir.path()),
            event_log::log_reader(&log_dir_in(dir.path()), Some(&subjects.keys)).unwrap(),
            Arc::clone(&subjects.keys),
            Vec::new(),
        )
        .unwrap();
        let server = MemoryServer::new(
            Arc::clone(&subjects.store),
            embedder(),
            Arc::new(liam_model::IdentityReranker),
            Arc::new(liam_model::MockLlm),
            30,
            false,
            8192,
            1,
        )
        .with_admin(AdminConfig {
            producers: admins.iter().map(|name| (*name).to_owned()).collect(),
        })
        .with_delete_preview(Arc::new(preview));
        Self {
            dir,
            subjects,
            server,
        }
    }

    async fn delete(&self, subject: &str, scope: Option<&str>) -> String {
        self.server
            .delete(Parameters(DeleteArgs {
                subject: Some(subject.to_owned()),
                scope: scope.map(str::to_owned),
                include_children: None,
                event_id: None,
            }))
            .await
    }

    /// What a delete would change: the log, the live rows, the state of every key.
    async fn state(&self) -> (usize, usize, Vec<Option<KeyState>>) {
        let reader =
            event_log::log_reader(&log_dir_in(self.dir.path()), Some(&self.subjects.keys)).unwrap();
        let events = reader.scan(None).count().await;
        let live = self.subjects.store.changes_since(BEGINNING).await.unwrap();
        let key = self
            .subjects
            .store
            .key_of_node(&self.subjects.alice)
            .await
            .unwrap()
            .unwrap();
        (events, live.len(), vec![self.subjects.keys.state(&key)])
    }
}

#[tokio::test]
async fn delete_reports_counts_and_changes_nothing() {
    // Arrange
    let fixture = Fixture::new(&[ADMIN]).await;
    fixture.server.set_producer(ADMIN);
    let before = fixture.state().await;

    // Act
    let report = fixture.delete("alice", None).await;

    // Assert
    assert!(report.contains("nothing was changed"), "{report}");
    assert!(report.contains("store graph"), "{report}");
    assert_eq!(fixture.state().await, before);
    assert!(
        !job_file_path_for(&database_in(fixture.dir.path())).exists(),
        "no job file"
    );
}

#[tokio::test]
async fn a_subject_in_another_scope_looks_like_one_that_does_not_exist() {
    // Arrange
    let fixture = Fixture::new(&[ADMIN]).await;
    fixture.server.set_producer(ADMIN);
    let note = NewNode::now("fact", "carol", "carol secret")
        .with_subject("carol")
        .with_scope(OTHER_SCOPE);
    fixture.subjects.store.insert(note).await.unwrap();

    // Act
    let outside = fixture.delete("carol", Some("mine")).await;
    let nonexistent = fixture.delete("nobody", Some("mine")).await;
    let inside = fixture.delete("carol", Some(OTHER_SCOPE)).await;

    // Assert
    assert_eq!(outside, nonexistent);
    assert!(!outside.contains("key groups"), "{outside}");
    assert!(inside.contains("key groups"), "{inside}");
}

#[tokio::test]
async fn a_producer_outside_the_admin_list_is_refused_and_nothing_changes() {
    // Arrange
    let fixture = Fixture::new(&[ADMIN]).await;
    let before = fixture.state().await;

    // Act: no producer set, so the unknown fallback applies
    let unknown = fixture.delete("alice", None).await;
    fixture.server.set_producer("someone-else");
    let other = fixture.delete("alice", None).await;
    let consolidate = fixture.server.consolidate().await;

    // Assert
    for refusal in [&unknown, &other, &consolidate] {
        assert!(refusal.contains("refused"), "{refusal}");
        assert!(refusal.contains(REFUSED), "{refusal}");
    }
    assert_eq!(fixture.state().await, before);
}

#[tokio::test]
async fn an_empty_admin_list_refuses_every_producer() {
    // Arrange
    let fixture = Fixture::new(&[]).await;
    fixture.server.set_producer(ADMIN);

    // Act
    let report = fixture.delete("alice", None).await;

    // Assert
    assert!(report.contains(REFUSED), "{report}");
}

#[tokio::test]
async fn delete_wants_exactly_one_target() {
    // Arrange
    let fixture = Fixture::new(&[ADMIN]).await;
    fixture.server.set_producer(ADMIN);

    // Act
    let neither = fixture
        .server
        .delete(Parameters(DeleteArgs {
            subject: None,
            scope: None,
            include_children: None,
            event_id: None,
        }))
        .await;
    let both = fixture
        .server
        .delete(Parameters(DeleteArgs {
            subject: Some("alice".into()),
            scope: None,
            include_children: None,
            event_id: Some("e".into()),
        }))
        .await;

    // Assert
    for answer in [neither, both] {
        assert!(answer.starts_with("delete failed:"), "{answer}");
    }
}

#[tokio::test]
async fn the_tool_set_is_the_existing_tools_plus_delete_consolidate_and_metrics() {
    // Arrange
    let fixture = Fixture::new(&[ADMIN]).await;

    // Act
    let tools = fixture.server.tool_router.list_all();

    // Assert
    let names: BTreeSet<&str> = tools.iter().map(|tool| &*tool.name).collect();
    let expected: BTreeSet<&str> = [
        "remember",
        "recall",
        "relate",
        "clusters",
        "timeline",
        "facts",
        "feedback",
        "summaries",
        "ask",
        "delete",
        "consolidate",
        "metrics",
    ]
    .into();
    assert_eq!(names, expected);
    let metrics = tools.iter().find(|tool| tool.name == "metrics").unwrap();
    let schema = serde_json::to_string(&metrics.input_schema).unwrap();
    for field in ["\"metric\"", "\"limit\""] {
        assert!(schema.contains(field), "{field} missing from {schema}");
    }
    let delete = tools.iter().find(|tool| tool.name == "delete").unwrap();
    let schema = serde_json::to_string(&delete.input_schema).unwrap();
    for field in [
        "\"subject\"",
        "\"scope\"",
        "\"include_children\"",
        "\"event_id\"",
    ] {
        assert!(schema.contains(field), "{field} missing from {schema}");
    }
}
