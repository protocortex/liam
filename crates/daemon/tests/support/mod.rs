// SPDX-License-Identifier: Apache-2.0
//! Fixtures the command-line tests share.

use std::path::Path;
use std::sync::Arc;

use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{WalConfig, WalWriter};
use liam_store::{DefaultGraph, EventLog, GraphConfig, NewNode};

pub const DIMS: usize = 8;

/// A directory holding a config file, a store and the log it was written
/// through, with the store and log named the way the flags default to.
pub async fn fixture(dir: &Path) {
    std::fs::write(dir.join("liam.toml"), format!("embedding_dims = {DIMS}\n")).unwrap();
    let log_dir = dir.join("liam.log");
    let writer = WalWriter::open_with_system_clock(
        &log_dir,
        WalConfig {
            segment_max_bytes: 1 << 20,
            rotate_interval_secs: 3600,
        },
    )
    .unwrap();
    let reader = SequentialScanReader::local(&log_dir).unwrap();
    let log = Arc::new(tokio::sync::Mutex::new(EventLog::new(
        Box::new(writer),
        Arc::new(reader),
        HashBloom::new(BloomConfig::default()),
    )));
    DefaultGraph::open(
        dir.join("liam.db").to_str().unwrap(),
        GraphConfig::new(DIMS),
    )
    .await
    .unwrap()
    .with_log(log)
    .await
    .unwrap()
    .insert(NewNode::now("fact", "label", "alpha"))
    .await
    .unwrap();
}
