// SPDX-License-Identifier: Apache-2.0
//! Checking a node against the log before its vector is dropped.

use std::sync::atomic::{AtomicUsize, Ordering};

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::event::{LogEvent, LogPayload, RowEffect, TombstoneTable};
use liam_log::hash::node_row_hash;
use liam_log::reader::{LogReader, LogStream, SequentialScanReader};
use liam_log::wal::{WalConfig, WalWriter};
use liam_log::LogOffset;

use super::support::{event, node, node_write, shared, tombstone, Env};
use super::*;
use crate::types::{OriginClaim, OriginVerdict};
use crate::DefaultBackend;

fn claim(node_id: &str, origins: &[&str], content: &str) -> OriginClaim {
    OriginClaim {
        node_id: NodeId::from_raw(node_id),
        origin_event_ids: origins.iter().map(|id| (*id).to_string()).collect(),
        label: "label".to_string(),
        content: content.to_string(),
    }
}

async fn verdicts(env: &Env, claims: &[OriginClaim]) -> Vec<OriginVerdict> {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let (graph, _) = env.open::<DefaultBackend>("verify.db", clock).await;
    let found = graph.verify_origins(claims).await.unwrap();
    claims.iter().map(|claim| found[&claim.node_id]).collect()
}

fn voided(target: &str) -> LogEvent {
    event(
        &format!("void-{target}"),
        [0; 32],
        LogPayload::Voided {
            target_event_id: target.to_string(),
        },
    )
}

#[tokio::test]
async fn each_kind_of_origin_gets_its_own_verdict() {
    // Arrange
    let env = Env::new();
    let mut from_a_rolled_back_run = node_write("n4", "four");
    from_a_rolled_back_run.source = "consolidation:run-9".to_string();
    let batch_rows = vec![
        RowEffect::Node(node("n8", "eight")),
        RowEffect::Node(node("n9", "nine")),
    ];
    env.append(&[
        node_write("n1", "one"),
        node_write("n2", "two"),
        voided("event-n2"),
        node_write("n3", "three"),
        tombstone("gc-1", &[(TombstoneTable::Nodes, "n3")]),
        from_a_rolled_back_run,
        event(
            "rolled-back",
            [0; 32],
            LogPayload::RunRolledBack {
                run_id: "run-9".to_string(),
            },
        ),
        node_write("n5", "five"),
        event(
            "dup-of-five",
            [0; 32],
            LogPayload::DuplicateOf {
                first_event_id: "event-n5".to_string(),
            },
        ),
        event("batch", [0; 32], LogPayload::EpisodeBatch(batch_rows)),
    ]);
    let claims = [
        claim("verified", &["event-n1"], "one"),
        claim("mismatch", &["event-n1"], "uno"),
        claim("absent", &["event-absent"], "one"),
        claim("none-listed", &[], "one"),
        claim("n2", &["event-n2"], "two"),
        claim("n3", &["event-n3"], "three"),
        claim("n4", &["event-n4"], "four"),
        claim("n6", &["dup-of-five"], "five"),
        claim("n8", &["batch"], "eight"),
        claim("n9", &["batch"], "eight"),
        claim("second-origin", &["event-absent", "event-n1"], "one"),
    ];

    // Act
    let found = verdicts(&env, &claims).await;

    // Assert
    use OriginVerdict::*;
    assert_eq!(
        found,
        [
            Verified,      // the origin is in force and says the same
            Mismatch,      // in force, says something else
            OriginMissing, // not in the log
            OriginMissing, // none listed
            Excluded,      // voided
            Excluded,      // the node was tombstoned
            Excluded,      // written by a rolled back run
            Verified,      // a duplicate is judged by its first event
            Verified,      // the row of an episode batch with the node's id
            Mismatch,      // another row of the batch says something else
            Verified,      // one origin out of two is enough
        ]
    );
}

#[tokio::test]
async fn a_duplicate_whose_first_event_was_voided_is_excluded() {
    // Arrange
    let env = Env::new();
    env.append(&[
        node_write("n5", "five"),
        event(
            "dup-of-five",
            [0; 32],
            LogPayload::DuplicateOf {
                first_event_id: "event-n5".to_string(),
            },
        ),
        voided("event-n5"),
    ]);

    // Act
    let found = verdicts(&env, &[claim("n6", &["dup-of-five"], "five")]).await;

    // Assert
    assert_eq!(found, [OriginVerdict::Excluded]);
}

#[tokio::test]
async fn an_empty_log_verifies_nothing() {
    // Arrange
    let env = Env::new();

    // Act
    let found = verdicts(&env, &[claim("n1", &["event-n1"], "one")]).await;

    // Assert
    assert_eq!(found, [OriginVerdict::OriginMissing]);
}

/// Counts the scans a reader is asked for and the records they yield.
struct CountingReader {
    inner: SequentialScanReader,
    scans: Arc<AtomicUsize>,
    yielded: Arc<AtomicUsize>,
}

impl CountingReader {
    fn counted(&self, stream: LogStream) -> LogStream {
        self.scans.fetch_add(1, Ordering::SeqCst);
        let yielded = Arc::clone(&self.yielded);
        Box::pin(futures_util::StreamExt::inspect(stream, move |_| {
            yielded.fetch_add(1, Ordering::SeqCst);
        }))
    }
}

impl LogReader for CountingReader {
    fn scan(&self, from: Option<LogOffset>) -> LogStream {
        self.counted(self.inner.scan(from))
    }

    fn scan_through(&self, from: Option<LogOffset>, through: LogOffset) -> LogStream {
        self.counted(self.inner.scan_through(from, through))
    }
}

#[tokio::test]
async fn many_claims_are_checked_in_one_scan_of_the_log() {
    // Arrange: 1000 events, 100 of them claimed
    const EVENTS: usize = 1000;
    const CLAIMED: usize = 100;
    let env = Env::new();
    let events: Vec<LogEvent> = (0..EVENTS)
        .map(|i| {
            let row = node(&format!("n{i}"), &format!("text {i}"));
            event(
                &format!("event-n{i}"),
                node_row_hash(&row),
                LogPayload::NodeWrite(row),
            )
        })
        .collect();
    env.append(&events);
    let (scans, yielded) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let reader = CountingReader {
        inner: SequentialScanReader::local(&env.wal_dir()).unwrap(),
        scans: Arc::clone(&scans),
        yielded: Arc::clone(&yielded),
    };
    let writer = WalWriter::open_with_system_clock(
        &env.wal_dir(),
        WalConfig {
            segment_max_bytes: 1 << 20,
            rotate_interval_secs: 3600,
        },
    )
    .unwrap();
    let log = shared(EventLog::new(
        Box::new(writer),
        Arc::new(reader),
        HashBloom::new(BloomConfig::default()),
    ));
    let graph = Graph::<DefaultBackend>::open_with_clock(
        &env.db("counted.db"),
        GraphConfig::new(8),
        Arc::new(FixedClock::new(Millis(1000))),
    )
    .await
    .unwrap()
    .with_log(log)
    .await
    .unwrap();
    let before = scans.load(Ordering::SeqCst);
    let yielded_before = yielded.load(Ordering::SeqCst);
    let claims: Vec<OriginClaim> = (0..CLAIMED)
        .map(|i| {
            claim(
                &format!("n{i}"),
                &[&format!("event-n{i}")],
                &format!("text {i}"),
            )
        })
        .collect();

    // Act
    let found = graph.verify_origins(&claims).await.unwrap();

    // Assert
    assert!(found.values().all(|v| *v == OriginVerdict::Verified));
    assert_eq!(scans.load(Ordering::SeqCst) - before, 1);
    assert!(yielded.load(Ordering::SeqCst) - yielded_before <= EVENTS);
}
