// SPDX-License-Identifier: Apache-2.0
//! Assembling the event log over a directory, shared by `liamd` and `liam
//! rebuild` so both open a log the same way.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use liam_log::dedup::{BloomConfig, HashBloom};
use liam_log::reader::SequentialScanReader;
use liam_log::wal::{SystemClock, WalConfig, WalWriter, LOCK_NAME, MANIFEST_NAME};
use liam_log::LogWriter;
use liam_store::{EventLog, PoisonHook, SharedLog};

/// Opens the log directory for writing, creating it on first use. The writer
/// keeps the directory locked until it is dropped.
pub fn open_writer(log_dir: &Path, wal: WalConfig) -> anyhow::Result<WalWriter<SystemClock>> {
    WalWriter::open_with_system_clock(log_dir, wal)
        .with_context(|| format!("could not open the log in {}", log_dir.display()))
}

/// The log over `writer` and a reader of the same directory. `on_poisoned` is
/// called whenever a write meets a poisoned log.
pub fn shared_log(
    writer: impl LogWriter + 'static,
    log_dir: &Path,
    bloom: BloomConfig,
    on_poisoned: Option<PoisonHook>,
) -> anyhow::Result<SharedLog> {
    let reader = SequentialScanReader::local(log_dir)
        .with_context(|| format!("could not read the log in {}", log_dir.display()))?;
    let mut log = EventLog::new(Box::new(writer), Arc::new(reader), HashBloom::new(bloom));
    if let Some(hook) = on_poisoned {
        log = log.with_poison_hook(hook);
    }
    Ok(Arc::new(tokio::sync::Mutex::new(log)))
}

/// Whether `dir` already is a log, which its manifest says.
pub fn holds_log(dir: &Path) -> bool {
    dir.join(MANIFEST_NAME).is_file()
}

/// Refuses a directory that holds something other than a log, before a writer
/// puts any file there. Only a missing or empty directory may become a new log;
/// the lock file a crashed first start left behind does not count as content.
pub fn ensure_usable_as_log(dir: &Path) -> anyhow::Result<()> {
    if holds_log(dir) {
        return Ok(());
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("could not read the log directory {}", dir.display()))
        }
    };
    for entry in entries {
        let entry =
            entry.with_context(|| format!("could not read the log directory {}", dir.display()))?;
        anyhow::ensure!(
            entry.file_name() == LOCK_NAME,
            "{} holds files but is not an event log, it has no {MANIFEST_NAME} file; point \
             [log] dir at an empty or new directory, or at the existing log",
            dir.display()
        );
    }
    Ok(())
}
