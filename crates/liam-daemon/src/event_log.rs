// SPDX-License-Identifier: Apache-2.0
//! Assembling the event log over a directory, shared by `liamd` and `liam
//! rebuild` so both open a log the same way.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{WalConfig, WalWriter};
use liam_log::LogWriter;
use liam_store::{EventLog, SharedLog};

/// Opens the log directory for writing, creating it on first use.
pub fn open_writer(log_dir: &Path, wal: WalConfig) -> anyhow::Result<impl LogWriter + 'static> {
    WalWriter::open_with_system_clock(log_dir, wal)
        .with_context(|| format!("could not open the log in {}", log_dir.display()))
}

/// The log over `writer` and a reader of the same directory.
pub fn shared_log(
    writer: impl LogWriter + 'static,
    log_dir: &Path,
    bloom: BloomConfig,
) -> anyhow::Result<SharedLog> {
    let reader = SequentialScanReader::local(log_dir)
        .with_context(|| format!("could not read the log in {}", log_dir.display()))?;
    let log = EventLog::new(Box::new(writer), Arc::new(reader), HashBloom::new(bloom));
    Ok(Arc::new(tokio::sync::Mutex::new(log)))
}
