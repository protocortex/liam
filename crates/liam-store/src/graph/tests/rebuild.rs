// SPDX-License-Identifier: Apache-2.0
//! Replaying a tombstone from the log. Tests run against a real WAL and reader
//! in a temp directory.

use std::path::PathBuf;

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::event::{
    LogEvent, LogPayload, TombstoneTable, TombstoneTarget, CURRENT_SCHEMA_VERSION,
};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{WalConfig, WalWriter};
use liam_log::LogWriter;

use super::support::{count, fact_at, shared, StubEmbedder};
use super::*;

const DIMS: usize = 8;

const ONE_SEGMENT: WalConfig = WalConfig {
    segment_max_bytes: 1 << 20,
    rotate_interval_secs: 3600,
};

/// A log directory and the databases that project it.
struct Env {
    dir: TempDir,
}

/// A store opened over the log.
struct Opened {
    graph: DefaultGraph,
}

impl Env {
    fn new() -> Self {
        Self {
            dir: TempDir::new().unwrap(),
        }
    }

    fn wal_dir(&self) -> PathBuf {
        self.dir.path().join("wal")
    }

    fn db(&self, name: &str) -> String {
        self.dir.path().join(name).to_str().unwrap().to_owned()
    }

    /// Appends `events` with a writer of their own that is gone afterwards, so
    /// no store has projected them.
    fn append(&self, events: &[LogEvent]) {
        let mut writer =
            WalWriter::open_with_system_clock(&self.wal_dir(), ONE_SEGMENT).expect("open wal");
        for event in events {
            writer.append(event).expect("append event");
        }
    }

    fn log(&self) -> SharedLog {
        let writer =
            WalWriter::open_with_system_clock(&self.wal_dir(), ONE_SEGMENT).expect("open wal");
        let reader = SequentialScanReader::local(&self.wal_dir()).expect("open reader");
        let bloom = HashBloom::new(BloomConfig::default());
        shared(EventLog::new(Box::new(writer), Arc::new(reader), bloom))
    }

    /// A store over the log with a stub embedder attached.
    async fn open(&self, db: &str) -> Opened {
        let clock = Arc::new(FixedClock::new(Millis(1000)));
        let graph = DefaultGraph::open_with_clock(
            &self.db(db),
            GraphConfig::new(DIMS),
            Arc::clone(&clock) as Arc<dyn Clock>,
        )
        .await
        .expect("open graph")
        .with_log(self.log())
        .await
        .expect("attach log")
        .with_embedder(StubEmbedder::new(DIMS));
        Opened { graph }
    }
}

// ---- events, built the way the write path logs them ----

fn event(event_id: &str, payload: LogPayload) -> LogEvent {
    LogEvent {
        event_id: event_id.into(),
        content_hash: [0; 32],
        source: "agent-a".into(),
        trust_score: 0.75,
        observed_at: 500,
        ingested_at: 1000,
        encryption_key_id: None,
        schema_version: CURRENT_SCHEMA_VERSION,
        payload,
    }
}

fn tombstone(targets: &[(TombstoneTable, &str)]) -> LogEvent {
    let targets = targets
        .iter()
        .map(|(table, id)| TombstoneTarget {
            table: *table,
            id: (*id).into(),
        })
        .collect();
    event("event-tombstone", LogPayload::Tombstone(targets))
}

// ---- reading the store back ----

async fn has_node<B: Backend>(g: &Graph<B>, id: &str) -> bool {
    let rows = g
        .backend
        .query("SELECT 1 FROM nodes WHERE id = ?1", &[id.into()])
        .await
        .unwrap();
    !rows.is_empty()
}

/// The record counts alone, leaving the re-embed pass out.
fn record_counts(report: CatchUpReport) -> CatchUpReport {
    CatchUpReport {
        reembedded: ReembedReport::default(),
        ..report
    }
}

// ---- tests ----

#[tokio::test]
async fn rebuild_a_tombstoned_node_takes_its_edges_vector_and_community_row_with_it() {
    // Arrange: a store with a vector and a community row, then a log that
    // tombstones the node twice behind its back
    let env = Env::new();
    let (a, b) = {
        let first = env.open("live.db").await;
        let a = first.graph.insert(fact_at("a")).await.unwrap();
        let b = first.graph.insert(fact_at("b")).await.unwrap();
        first.graph.relate(&a, &b, "mentions").await.unwrap();
        first.graph.reembed_missing().await.unwrap();
        first
            .graph
            .backend
            .execute(
                "INSERT INTO node_community (node_id, community, computed_at) VALUES (?1, 1, 1)",
                &[a.as_str().into()],
            )
            .await
            .unwrap();
        (a, b)
    };
    let gone = [(TombstoneTable::Nodes, a.as_str())];
    let mut again = tombstone(&gone);
    again.event_id = "event-tombstone-again".into();
    env.append(&[tombstone(&gone), again]);
    let live = env.open("live.db").await;

    // Act
    let report = live.graph.catch_up().await.unwrap();

    // Assert: the second tombstone finds nothing left and is not an error
    assert_eq!(record_counts(report).applied, 2, "{report:?}");
    assert!(!has_node(&live.graph, a.as_str()).await);
    assert!(has_node(&live.graph, b.as_str()).await);
    assert_eq!(count(&live.graph, "edges").await, 0);
    assert_eq!(count(&live.graph, "node_community").await, 0);
    assert_eq!(count(&live.graph, "node_vectors").await, 1);
}
