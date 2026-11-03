// SPDX-License-Identifier: Apache-2.0
//! Doubles and builders shared by the tiering tests.

use std::path::Path;
use std::sync::Arc;

use liam_log::dedup::BloomConfig;
use liam_store::{DefaultGraph, FixedClock, GraphConfig};

use crate::config::LogConfig;

/// An in-memory store on an event log under `dir`, read through `clock`.
pub async fn logged_store(dir: &Path, clock: Arc<FixedClock>, dims: usize) -> DefaultGraph {
    let log_dir = dir.join("events");
    let writer = liam_daemon::event_log::open_writer(&log_dir, LogConfig::default().wal()).unwrap();
    let log =
        liam_daemon::event_log::shared_log(writer, &log_dir, BloomConfig::default(), None).unwrap();
    DefaultGraph::open_with_clock(":memory:", GraphConfig::new(dims), clock)
        .await
        .unwrap()
        .with_log(log)
        .await
        .unwrap()
}
