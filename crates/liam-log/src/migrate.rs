// SPDX-License-Identifier: Apache-2.0

//! Rewrites a plaintext log as sealed records, one partition at a time.
//!
//! A partition keeps its sequence, its event ids and its record order, so every
//! cursor into the log stays valid. The sealed copy is written beside the old
//! file and checked by opening every record; only then is the plaintext file
//! overwritten with zeros and replaced. A crash at any step is finished by the
//! next run.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::compactor::{decode_parquet, segment_parquet_bytes, CompactError};
use crate::crypto::KeyRegistry;
use crate::envelope::{
    named_lineages, open_record, outside_edge_sources, row_keys, KeyHints, OpenError, RowKind,
    SealError, Sealer,
};
use crate::event::{EventError, LogEvent};
use crate::wal::{
    encode_frame, numbered_files, read_segment, record_length, segment_paths, sync_dir, WalError,
    PARQUET_EXTENSION,
};

/// The file that names the partition whose plaintext may already be gone.
pub const STATE_FILE: &str = "migration.state";
const TEMP_SUFFIX: &str = ".migrating";
/// Bytes overwritten per write when a plaintext file is zeroed.
const ZERO_CHUNK: usize = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum MigrateError {
    #[error("log io failed: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Wal(#[from] WalError),
    #[error(transparent)]
    Compact(#[from] CompactError),
    #[error(transparent)]
    Event(#[from] EventError),
    #[error(transparent)]
    Seal(#[from] SealError),
    #[error(transparent)]
    Open(#[from] OpenError),
    #[error("the sealed copy of partition {sequence} does not match the original: {detail}")]
    Verify { sequence: u64, detail: String },
    #[error("the migration state file {} does not name a partition", path.display())]
    StateCorrupt { path: PathBuf },
}

/// Where in a partition's rewrite a run can be stopped, to stand in for a crash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrateStep {
    /// The sealed copy is on disk, the plaintext untouched.
    Written,
    /// The sealed copy verified and the state file names the partition.
    Verified,
    /// The plaintext is zeroed, the sealed copy not yet in its place.
    Zeroed,
}

pub trait MigrateFault: Send + Sync {
    fn reached(&self, step: MigrateStep, sequence: u64) -> io::Result<()>;
}

/// Never fails.
pub struct NoFault;

impl MigrateFault for NoFault {
    fn reached(&self, _step: MigrateStep, _sequence: u64) -> io::Result<()> {
        Ok(())
    }
}

/// Supplies the key hints a legacy event needs, which only the caller can read
/// from its own state. It sees every event in log order, sealed or not.
pub trait HintSource {
    fn hints(&mut self, plain: &LogEvent) -> KeyHints;
    fn learn(&mut self, event: &LogEvent);
}

/// Hints from the nodes the migration has met so far, and from the lineage keys
/// the caller read out of the store (`lineage_nodes` maps a lineage key to a
/// node of it).
#[derive(Debug, Default)]
pub struct NodeKeyHints {
    node_keys: HashMap<String, String>,
    lineage_nodes: HashMap<String, String>,
}

impl NodeKeyHints {
    pub fn new(lineage_nodes: HashMap<String, String>) -> Self {
        Self {
            node_keys: HashMap::new(),
            lineage_nodes,
        }
    }
}

impl HintSource for NodeKeyHints {
    fn hints(&mut self, plain: &LogEvent) -> KeyHints {
        let mut hints = KeyHints::default();
        for src in outside_edge_sources(&plain.payload) {
            if let Some(key_id) = self.node_keys.get(&src) {
                hints.edge_sources.insert(src, key_id.clone());
            }
        }
        for lineage in named_lineages(&plain.payload) {
            let key_id = self
                .lineage_nodes
                .get(&lineage)
                .and_then(|node_id| self.node_keys.get(node_id));
            if let Some(key_id) = key_id {
                hints.lineages.insert(lineage, key_id.clone());
            }
        }
        hints
    }

    fn learn(&mut self, event: &LogEvent) {
        for key in row_keys(event) {
            if key.kind == RowKind::Node {
                self.node_keys.insert(key.row_id, key.key_id);
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MigrateReport {
    pub partitions_rewritten: u64,
    pub events_sealed: u64,
}

/// The files of one partition.
struct Partition {
    sequence: u64,
    wal: Option<PathBuf>,
    parquet: Option<PathBuf>,
}

/// Seals every plaintext event of the log in `dir`. A log that is already
/// sealed is left as it is.
pub fn seal_log(
    dir: &Path,
    sealer: &Sealer,
    registry: &KeyRegistry,
    hints: &mut dyn HintSource,
    fault: &dyn MigrateFault,
) -> Result<MigrateReport, MigrateError> {
    finish_interrupted(dir)?;
    let mut report = MigrateReport::default();
    for partition in partitions(dir)? {
        let plain = read_events(&partition)?;
        let mut sealed = Vec::with_capacity(plain.len());
        for event in &plain {
            if matches!(event.payload, crate::event::LogPayload::Sealed(_)) {
                hints.learn(event);
                sealed.push(event.clone());
                continue;
            }
            let hinted = hints.hints(event);
            let one = sealer.seal(event, &hinted)?;
            hints.learn(&one);
            sealed.push(one);
        }
        if sealed == plain {
            continue;
        }
        rewrite(dir, &partition, &plain, &sealed, registry, fault)?;
        report.partitions_rewritten += 1;
        report.events_sealed += sealed.iter().zip(&plain).filter(|(s, p)| s != p).count() as u64;
    }
    Ok(report)
}

fn partitions(dir: &Path) -> Result<Vec<Partition>, MigrateError> {
    let mut by_sequence = std::collections::BTreeMap::new();
    for (sequence, path) in segment_paths(dir)? {
        by_sequence.insert(
            sequence,
            Partition {
                sequence,
                wal: Some(path),
                parquet: None,
            },
        );
    }
    for (sequence, path) in numbered_files(dir, PARQUET_EXTENSION)? {
        by_sequence
            .entry(sequence)
            .or_insert(Partition {
                sequence,
                wal: None,
                parquet: None,
            })
            .parquet = Some(path);
    }
    Ok(by_sequence.into_values().collect())
}

/// The WAL wins while it exists, as it does for the reader.
fn read_events(partition: &Partition) -> Result<Vec<LogEvent>, MigrateError> {
    if let Some(wal) = &partition.wal {
        if let Some(events) = read_segment(wal, false)? {
            return Ok(events);
        }
    }
    match &partition.parquet {
        Some(parquet) => Ok(decode_parquet(File::open(parquet)?, partition.sequence)?),
        None => Ok(Vec::new()),
    }
}

fn temp_of(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(TEMP_SUFFIX);
    PathBuf::from(name)
}

fn rewrite(
    dir: &Path,
    partition: &Partition,
    plain: &[LogEvent],
    sealed: &[LogEvent],
    registry: &KeyRegistry,
    fault: &dyn MigrateFault,
) -> Result<(), MigrateError> {
    let sequence = partition.sequence;
    if let Some(wal) = &partition.wal {
        let mut bytes = Vec::new();
        for event in sealed {
            let payload = event.encode()?;
            bytes.extend(encode_frame(record_length(payload.len())?, &payload));
        }
        write_synced(&temp_of(wal), &bytes)?;
    }
    if let Some(parquet) = &partition.parquet {
        write_synced(&temp_of(parquet), &segment_parquet_bytes(sequence, sealed)?)?;
    }
    sync_dir(dir)?;
    fault.reached(MigrateStep::Written, sequence)?;
    verify(partition, plain, registry)?;
    write_synced(&dir.join(STATE_FILE), sequence.to_string().as_bytes())?;
    sync_dir(dir)?;
    fault.reached(MigrateStep::Verified, sequence)?;
    install(dir, partition, Some(fault))
}

/// Checks the sealed copy on disk against the plaintext it replaces.
fn verify(
    partition: &Partition,
    plain: &[LogEvent],
    registry: &KeyRegistry,
) -> Result<(), MigrateError> {
    let sequence = partition.sequence;
    let mismatch = |detail: String| MigrateError::Verify { sequence, detail };
    let mut copies = Vec::new();
    if let Some(wal) = &partition.wal {
        let events = read_segment(&temp_of(wal), true)?;
        copies.push(events.ok_or_else(|| mismatch("the sealed WAL copy is missing".into()))?);
    }
    if let Some(parquet) = &partition.parquet {
        copies.push(decode_parquet(File::open(temp_of(parquet))?, sequence)?);
    }
    for copy in copies {
        if copy.len() != plain.len() {
            return Err(mismatch(format!(
                "{} events, expected {}",
                copy.len(),
                plain.len()
            )));
        }
        for (original, stored) in plain.iter().zip(&copy) {
            let opened = open_record(registry, stored)?;
            let same = opened.shredded.is_empty()
                && opened.event.event_id == original.event_id
                && opened.event.payload == original.payload;
            if !same {
                return Err(mismatch(format!(
                    "event {} reads back differently",
                    original.event_id
                )));
            }
        }
    }
    Ok(())
}

/// Zeroes each plaintext file and puts its sealed copy in its place.
fn install(
    dir: &Path,
    partition: &Partition,
    fault: Option<&dyn MigrateFault>,
) -> Result<(), MigrateError> {
    let files: Vec<&PathBuf> = [&partition.wal, &partition.parquet]
        .into_iter()
        .flatten()
        .collect();
    for path in &files {
        if temp_of(path).exists() && path.exists() {
            zero_file(path)?;
        }
    }
    if let Some(fault) = fault {
        fault.reached(MigrateStep::Zeroed, partition.sequence)?;
    }
    for path in files {
        let temp = temp_of(path);
        if temp.exists() {
            fs::rename(&temp, path)?;
        }
    }
    sync_dir(dir)?;
    fs::remove_file(dir.join(STATE_FILE))?;
    sync_dir(dir)?;
    Ok(())
}

/// Finishes a rewrite a crash cut short. Sealed copies that never verified are
/// discarded, since their plaintext is still whole.
fn finish_interrupted(dir: &Path) -> Result<(), MigrateError> {
    let state = dir.join(STATE_FILE);
    let verified = match fs::read_to_string(&state) {
        Ok(text) => Some(
            text.trim()
                .parse::<u64>()
                .map_err(|_| MigrateError::StateCorrupt {
                    path: state.clone(),
                })?,
        ),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    for partition in partitions(dir)? {
        if verified == Some(partition.sequence) {
            install(dir, &partition, None)?;
            continue;
        }
        for path in [&partition.wal, &partition.parquet].into_iter().flatten() {
            match fs::remove_file(temp_of(path)) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(())
}

fn write_synced(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Overwrites a file with zeros and syncs it, so its old contents are not left
/// in the blocks it held. Copy-on-write filesystems and SSDs may keep earlier
/// blocks regardless.
fn zero_file(path: &Path) -> io::Result<()> {
    let len = fs::metadata(path)?.len();
    let mut file = OpenOptions::new().write(true).open(path)?;
    let zeros = vec![0u8; ZERO_CHUNK];
    let mut left = len;
    while left > 0 {
        let chunk = left.min(ZERO_CHUNK as u64) as usize;
        file.write_all(&zeros[..chunk])?;
        left -= chunk as u64;
    }
    file.sync_all()
}

#[cfg(test)]
mod tests;
