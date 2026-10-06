// SPDX-License-Identifier: Apache-2.0
//! Helpers the log tests share: a log to hand to a graph, and reads of what the
//! store holds.

use futures_util::stream;
use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::{LogReader, LogStream};
use liam_log::{LogOffset, LogWriter};

use super::*;

/// A reader over a log nothing replays from.
struct EmptyReader;

impl LogReader for EmptyReader {
    fn scan(&self, _from: Option<LogOffset>) -> LogStream {
        Box::pin(stream::empty())
    }

    fn scan_through(&self, _from: Option<LogOffset>, _through: LogOffset) -> LogStream {
        Box::pin(stream::empty())
    }
}

pub(super) fn shared(log: EventLog) -> SharedLog {
    Arc::new(tokio::sync::Mutex::new(log))
}

/// A log over `writer` that cannot be replayed from, with the default filter.
pub(super) fn share(writer: impl LogWriter + 'static) -> SharedLog {
    share_sized(writer, BloomConfig::default())
}

/// `share` with a filter configured for `config`, as an operator would set it.
pub(super) fn share_sized(writer: impl LogWriter + 'static, config: BloomConfig) -> SharedLog {
    shared(EventLog::new(
        Box::new(writer),
        Arc::new(EmptyReader),
        HashBloom::new(config),
    ))
}

pub(super) fn fact(content: &str) -> NewNode {
    NewNode::now("fact", "label", content)
        .with_producer("agent-a")
        .with_confidence(0.75)
}

/// `fact` with a supplied valid time, so source, trust, valid time, and ingest
/// time are four different values on the log record.
pub(super) fn fact_at(content: &str) -> NewNode {
    fact(content).with_valid_from(Millis(500))
}

pub(super) async fn count<B: Backend>(g: &Graph<B>, table: &str) -> i64 {
    let rows = g
        .backend
        .query(&format!("SELECT COUNT(*) FROM {table}"), &[])
        .await
        .unwrap();
    rows[0].get_i64(0).unwrap()
}

/// `None` when the cursor row does not exist yet; otherwise its log id and
/// last applied offset (`None` while the offsets are still NULL).
pub(super) async fn cursor<B: Backend>(g: &Graph<B>) -> Option<(String, Option<(i64, i64)>)> {
    let rows = g
        .backend
        .query(
            "SELECT log_id, last_segment, last_index FROM log_cursor",
            &[],
        )
        .await
        .unwrap();
    let row = rows.first()?;
    let offset = match (&row.0[1], &row.0[2]) {
        (Value::Int(segment), Value::Int(index)) => Some((*segment, *index)),
        _ => None,
    };
    Some((row.get_string(0).unwrap(), offset))
}

pub(super) fn offset_pair(offset: LogOffset) -> (i64, i64) {
    (offset.segment as i64, offset.index as i64)
}
