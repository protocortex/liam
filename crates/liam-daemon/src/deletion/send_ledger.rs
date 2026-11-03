// SPDX-License-Identifier: Apache-2.0
//! A content-free record of what was sent to a hosted provider, so a delete can
//! tell which providers hold copies of a deleted subject's text. It lives in a
//! file beside the database, outside the snapshot.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use liam_log::wal::{write_atomically, OsFileOps};
use serde::{Deserialize, Serialize};

/// Appended to the database path to name the ledger file.
const LEDGER_FILE_SUFFIX: &str = ".hosted-sends.jsonl";
const LINE_END: u8 = b'\n';
const OWNER_ONLY: u32 = 0o600;

/// Where the send ledger of the database at `database_path` lives.
pub fn send_ledger_path_for(database_path: &Path) -> PathBuf {
    let mut path = database_path.as_os_str().to_owned();
    path.push(LEDGER_FILE_SUFFIX);
    PathBuf::from(path)
}

/// One request sent to a provider, synchronously or in a batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SendRow {
    /// The provider's name, as `[hosted.models]` spells it.
    pub provider: String,
    pub cache_key: String,
    pub source_event_ids: Vec<String>,
    /// Epoch milliseconds.
    pub sent_at: i64,
}

impl SendRow {
    /// Whether the request was built from one of `event_ids`.
    pub fn cites(&self, event_ids: &HashSet<String>) -> bool {
        self.source_event_ids
            .iter()
            .any(|id| event_ids.contains(id))
    }
}

/// The rows of one database, one JSON object per line.
pub struct SendLedger {
    path: PathBuf,
}

impl SendLedger {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Every row, oldest first. A last line cut short by a crash is skipped,
    /// because it was never acknowledged as sent; any other unreadable line
    /// is an error.
    pub fn load(&self) -> anyhow::Result<Vec<SendRow>> {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => anyhow::bail!(
                "the send ledger {} could not be read: {source}",
                self.path.display()
            ),
        };
        let terminated = bytes.last().is_none_or(|last| *last == LINE_END);
        let lines: Vec<&[u8]> = bytes
            .split(|byte| *byte == LINE_END)
            .filter(|line| !line.is_empty())
            .collect();
        let mut rows = Vec::with_capacity(lines.len());
        for (index, line) in lines.iter().enumerate() {
            match serde_json::from_slice(line) {
                Ok(row) => rows.push(row),
                Err(_) if !terminated && index + 1 == lines.len() => {
                    tracing::warn!("the send ledger ends in a torn line, which was skipped");
                }
                Err(source) => anyhow::bail!(
                    "line {} of the send ledger {} could not be parsed: {source}",
                    index + 1,
                    self.path.display()
                ),
            }
        }
        Ok(rows)
    }

    /// Adds `row` and syncs it, so a send is on record before the next step.
    pub fn append(&self, row: &SendRow) -> anyhow::Result<()> {
        let mut line = serde_json::to_vec(row)?;
        line.push(LINE_END);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(OWNER_ONLY)
            .open(&self.path)?;
        // A crash can leave half a line; cut it so the new row starts clean.
        let bytes = fs::read(&self.path)?;
        if bytes.last().is_some_and(|last| *last != LINE_END) {
            let whole = bytes
                .iter()
                .rposition(|byte| *byte == LINE_END)
                .map_or(0, |at| at + 1);
            file.set_len(whole as u64)?;
        }
        file.write_all(&line)?;
        file.sync_all()?;
        Ok(())
    }

    /// The rows built from one of `event_ids`.
    pub fn citing(&self, event_ids: &HashSet<String>) -> anyhow::Result<Vec<SendRow>> {
        Ok(self
            .load()?
            .into_iter()
            .filter(|row| row.cites(event_ids))
            .collect())
    }

    /// Replaces the ledger with `rows`.
    pub fn replace(&self, rows: &[SendRow]) -> anyhow::Result<()> {
        let mut bytes = Vec::new();
        for row in rows {
            serde_json::to_writer(&mut bytes, row)?;
            bytes.push(LINE_END);
        }
        write_atomically(&self.path, &bytes, &mut OsFileOps).map_err(|source| {
            anyhow::anyhow!(
                "could not save the send ledger {}: {source}",
                self.path.display()
            )
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn row(cache_key: &str, provider: &str, events: &[&str]) -> SendRow {
        SendRow {
            provider: provider.to_owned(),
            cache_key: cache_key.to_owned(),
            source_event_ids: events.iter().map(|id| (*id).to_owned()).collect(),
            sent_at: 1,
        }
    }

    #[test]
    fn rows_append_in_order_and_replace_drops_the_rest() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let ledger = SendLedger::new(send_ledger_path_for(&dir.path().join("liam.db")));
        let first = row("k1", "anthropic", &["e1"]);
        let second = row("k2", "openai", &["e2"]);

        // Act
        let empty = ledger.load().unwrap();
        ledger.append(&first).unwrap();
        ledger.append(&second).unwrap();
        let both = ledger.load().unwrap();
        ledger.replace(std::slice::from_ref(&second)).unwrap();
        let one = ledger.load().unwrap();

        // Assert
        assert!(empty.is_empty());
        assert_eq!(both, vec![first, second.clone()]);
        assert_eq!(one, vec![second]);
    }

    #[test]
    fn citing_returns_only_the_rows_built_from_the_given_events() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let ledger = SendLedger::new(dir.path().join("sends.jsonl"));
        ledger.append(&row("k1", "anthropic", &["e1"])).unwrap();
        ledger.append(&row("k2", "anthropic", &["e2"])).unwrap();
        let asked: HashSet<String> = ["e2".to_owned()].into();

        // Act
        let found = ledger.citing(&asked).unwrap();

        // Assert
        assert_eq!(found, vec![row("k2", "anthropic", &["e2"])]);
    }

    #[test]
    fn a_torn_last_line_is_skipped_and_the_next_append_starts_a_new_line() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sends.jsonl");
        let ledger = SendLedger::new(path.clone());
        ledger.append(&row("k1", "anthropic", &["e1"])).unwrap();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(br#"{"provider":"anth"#).unwrap();

        // Act
        let torn = ledger.load().unwrap();
        ledger.append(&row("k2", "anthropic", &["e2"])).unwrap();
        let healed = ledger.load().unwrap();

        // Assert
        assert_eq!(torn, vec![row("k1", "anthropic", &["e1"])]);
        assert_eq!(healed.len(), 2);
    }

    #[test]
    fn an_unreadable_line_before_the_end_is_an_error_naming_the_line() {
        // Arrange
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sends.jsonl");
        let good = serde_json::to_string(&row("k1", "anthropic", &["e1"])).unwrap();
        fs::write(&path, format!("garbage\n{good}\n")).unwrap();

        // Act
        let error = SendLedger::new(path).load().unwrap_err().to_string();

        // Assert
        assert!(error.contains("line 1"), "{error}");
    }
}
