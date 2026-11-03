// SPDX-License-Identifier: Apache-2.0
//! The detectors of the deletion criterion. Each looks at one class of store or
//! file and names what it found, so a planted residual points at its detector.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::StreamExt;
use liam_daemon::event_log;
use liam_log::crypto::KeyRegistry;

use super::fixture::{files_under, Fixture, CANARY, ROLLED_BACK_CANARY};

/// A class of file the canary must not survive in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileClass {
    /// The database and its write-ahead log; the shared-memory file holds no content.
    Database,
    Snapshots,
    Cache,
    Manifest,
    Cooldown,
    Jobs,
    MetricsState,
    Registry,
}

impl FileClass {
    pub const ALL: [FileClass; 8] = [
        Self::Database,
        Self::Snapshots,
        Self::Cache,
        Self::Manifest,
        Self::Cooldown,
        Self::Jobs,
        Self::MetricsState,
        Self::Registry,
    ];

    /// What the class holds when the delete has not reached it: the state files
    /// name lineages, not content.
    fn needle(self, fixture: &Fixture) -> String {
        match self {
            Self::Manifest | Self::Cooldown => fixture.alice_lineage.as_str().to_owned(),
            _ => CANARY.to_owned(),
        }
    }

    pub fn files(self, fixture: &Fixture) -> Vec<PathBuf> {
        let database = fixture.database();
        match self {
            Self::Database => {
                let mut wal = database.clone().into_os_string();
                wal.push("-wal");
                vec![database, PathBuf::from(wal)]
            }
            Self::Snapshots => files_under(&fixture.snapshot_dir()),
            Self::Cache => vec![fixture.cache_file()],
            Self::Manifest => vec![fixture.manifest_file()],
            Self::Cooldown => vec![fixture.cooldown_file()],
            Self::Jobs => vec![crate::deletion::job_file_path_for(&database)],
            Self::MetricsState => vec![fixture.metrics_state_file()],
            Self::Registry => vec![fixture.subjects.keys.path()],
        }
    }

    /// True when a byte scan of the class's files finds its needle.
    pub fn holds_needle(self, fixture: &Fixture) -> bool {
        self.holds_token(fixture, &self.needle(fixture))
    }

    /// True when a byte scan of the class's files finds `token`.
    pub fn holds_token(self, fixture: &Fixture, token: &str) -> bool {
        scan_files(&self.files(fixture), token)
    }

    /// The same scan over copies of the class's files with the needle added to
    /// the first: the control that shows the scan can fail for this class.
    pub fn finds_the_needle_in_a_control_copy(self, fixture: &Fixture) -> bool {
        let needle = self.needle(fixture);
        let scratch = tempfile::tempdir().unwrap();
        let mut copies: Vec<PathBuf> = Vec::new();
        for (index, file) in self
            .files(fixture)
            .iter()
            .filter(|f| f.exists())
            .enumerate()
        {
            let copy = scratch.path().join(index.to_string());
            std::fs::copy(file, &copy).unwrap();
            copies.push(copy);
        }
        if copies.is_empty() {
            copies.push(scratch.path().join("empty"));
        }
        let mut handle = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&copies[0])
            .unwrap();
        handle.write_all(needle.as_bytes()).unwrap();
        scan_files(&copies, &needle)
    }
}

fn scan_files(files: &[PathBuf], token: &str) -> bool {
    files
        .iter()
        .any(|file| std::fs::read(file).is_ok_and(|bytes| contains(&bytes, token.as_bytes())))
}

pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// What the log and its archive show once opened with the keys that are left.
pub struct LogView {
    pub text: String,
    pub shredded: usize,
}

impl LogView {
    pub fn shows(&self, needle: &str) -> bool {
        self.text.contains(needle)
    }
}

/// Opens every record of the log, archived partitions included, and renders
/// what could be opened. A destroyed key leaves its rows out and counts them.
pub async fn log_view(fixture: &Fixture) -> LogView {
    log_view_of(&fixture.log_dir(), &fixture.subjects.keys).await
}

pub async fn log_view_of(log_dir: &Path, keys: &Arc<KeyRegistry>) -> LogView {
    let reader = event_log::log_reader(log_dir, Some(keys)).unwrap();
    let mut records = reader.scan(None);
    let mut view = LogView {
        text: String::new(),
        shredded: 0,
    };
    while let Some(record) = records.next().await {
        let record = record.expect("the log can be read");
        view.shredded += record.shredded.len();
        view.text.push_str(&format!("{:?}\n", record.event));
    }
    view
}

/// Columns that name an event, a node or a lineage, by what they refer to.
const REFERENCING: [&str; 6] = [
    "event_id",
    "first_event_id",
    "node_id",
    "src",
    "dst",
    "lineage_key",
];
const EVENT_LISTS: [&str; 1] = ["event_ids"];
/// Tables the scan leaves out, each with the reason: they hold no content, and
/// an id they hold is not a way back to what was deleted.
pub const EXEMPT: [(&str, &str); 12] = [
    ("log_cursor", "a log position"),
    ("log_backfill_state", "a log position"),
    ("extraction_cursor", "a log position"),
    ("cluster_state", "counts of edges and times"),
    ("provenance_repair_state", "a time"),
    ("derivation_backfill_state", "a time and a count"),
    ("quant_backfill_state", "a time and a count"),
    ("lineage_backfill_state", "a time and a count"),
    ("summary_tree_state", "counts per scope and level"),
    ("usage_epochs", "an event id and a day, no lineage"),
    ("lineage_removals", "ids only, the purge's own record"),
    (
        "node_keys",
        "ids only, the key group still resolves to the nodes it sealed",
    ),
];

/// Tables whose rows are not keyed by an event, node or lineage column, with
/// the predicate that selects a row a delete of alice must have removed: her
/// subject or a canary, or a row whose parent is gone.
fn row_filters(deleted: &Deleted) -> Vec<(&'static str, String)> {
    let nodes = deleted
        .nodes
        .iter()
        .map(|id| format!("'{id}'"))
        .collect::<Vec<_>>()
        .join(",");
    let canary = |column: &str| {
        format!("{column} LIKE '%{CANARY}%' OR {column} LIKE '%{ROLLED_BACK_CANARY}%'")
    };
    let orphan = |column: &str, parents: &str| format!("{column} NOT IN ({parents})");
    vec![
        ("nodes", format!("id IN ({nodes})")),
        (
            "entity_mention_state",
            "lower(subject) = 'alice'".to_owned(),
        ),
        ("fact_suppressions", "lower(subject) = 'alice'".to_owned()),
        (
            "facts",
            format!("lower(subject) = 'alice' OR {}", canary("object")),
        ),
        ("summaries", canary("text")),
        (
            "summary_children",
            orphan("parent_id", "SELECT id FROM summaries"),
        ),
        (
            "summary_embeddings",
            orphan("summary_id", "SELECT id FROM summaries"),
        ),
        (
            "summary_dissolving",
            orphan("summary_id", "SELECT id FROM summaries"),
        ),
        ("summary_failures", "1 = 1".to_owned()),
        (
            "derived_keys",
            orphan(
                "row_id",
                "SELECT id FROM facts UNION SELECT id FROM summaries",
            ),
        ),
    ]
}

/// The ids a delete of alice removed.
pub struct Deleted {
    pub events: BTreeSet<String>,
    pub nodes: BTreeSet<String>,
    pub lineages: BTreeSet<String>,
}

/// Rows that still name a deleted id, found in every table `sqlite_master`
/// lists, and the tables the scan could not classify.
pub struct LineageScan {
    pub referencing: Vec<String>,
    pub unclassified: Vec<String>,
}

pub async fn lineage_scan(fixture: &Fixture, deleted: &Deleted) -> LineageScan {
    let database = libsql::Builder::new_local(fixture.database().to_str().unwrap())
        .build()
        .await
        .unwrap();
    let connection = database.connect().unwrap();
    let mut scan = LineageScan {
        referencing: Vec::new(),
        unclassified: Vec::new(),
    };
    let filters = row_filters(deleted);
    for table in tables(&connection).await {
        if EXEMPT.iter().any(|(name, _)| *name == table) {
            continue;
        }
        let columns = columns_of(&connection, &table).await;
        let mut classified = false;
        if let Some((_, filter)) = filters.iter().find(|(name, _)| *name == table) {
            classified = true;
            let count = count_rows(&connection, &table, filter).await;
            if count > 0 {
                scan.referencing.push(format!("{table}: {count}"));
            }
        }
        for column in &columns {
            let ids = if REFERENCING.contains(&column.as_str()) {
                values_for(column, deleted)
            } else if EVENT_LISTS.contains(&column.as_str()) {
                deleted.events.iter().cloned().collect()
            } else {
                continue;
            };
            classified = true;
            if ids.is_empty() {
                continue;
            }
            let quoted = ids.iter().map(|id| format!("'{}'", id.replace('\'', "''")));
            let filter = if EVENT_LISTS.contains(&column.as_str()) {
                ids.iter()
                    .map(|id| format!("{column} LIKE '%{id}%'"))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            } else {
                format!("{column} IN ({})", quoted.collect::<Vec<_>>().join(","))
            };
            let count = count_rows(&connection, &table, &filter).await;
            if count > 0 {
                scan.referencing.push(format!("{table}.{column}: {count}"));
            }
        }
        if !classified {
            scan.unclassified.push(table);
        }
    }
    scan
}

fn values_for(column: &str, deleted: &Deleted) -> Vec<String> {
    let set = match column {
        "event_id" | "first_event_id" => &deleted.events,
        "lineage_key" => &deleted.lineages,
        _ => &deleted.nodes,
    };
    set.iter().cloned().collect()
}

async fn tables(connection: &libsql::Connection) -> Vec<String> {
    // A virtual table's shadow tables hold the index's own pages, which the
    // byte scan of the database file covers.
    let mut rows = connection
        .query(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
             AND sql NOT LIKE 'CREATE VIRTUAL%'",
            (),
        )
        .await
        .unwrap();
    let mut names = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        names.push(row.get::<String>(0).unwrap());
    }
    let mut virtuals = connection
        .query(
            "SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL%'",
            (),
        )
        .await
        .unwrap();
    let mut shadow_prefixes = Vec::new();
    while let Some(row) = virtuals.next().await.unwrap() {
        shadow_prefixes.push(format!("{}_", row.get::<String>(0).unwrap()));
    }
    names.retain(|name| {
        !shadow_prefixes
            .iter()
            .any(|prefix| name.starts_with(prefix))
    });
    names
}

async fn columns_of(connection: &libsql::Connection, table: &str) -> Vec<String> {
    let mut rows = connection
        .query(&format!("PRAGMA table_info({table})"), ())
        .await
        .unwrap();
    let mut columns = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        columns.push(row.get::<String>(1).unwrap());
    }
    columns
}

async fn count_rows(connection: &libsql::Connection, table: &str, filter: &str) -> i64 {
    let mut rows = connection
        .query(&format!("SELECT COUNT(*) FROM {table} WHERE {filter}"), ())
        .await
        .unwrap();
    rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
}
