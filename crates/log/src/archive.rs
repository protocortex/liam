// SPDX-License-Identifier: Apache-2.0

//! Moving closed partitions out of the log directory.
//!
//! A closed partition is a Parquet file whose WAL has been compacted away. Once
//! its last event is older than the configured age it is rewritten with zstd,
//! placed in the archive directory, read back and compared with the original,
//! listed in `partitions.manifest`, and only then removed from the log
//! directory. No event is ever deleted: every step is safe to run again, and a
//! run that stopped anywhere is finished by the next one.
//!
//! The manifest is what lets the reader and the writer treat an archived
//! partition as present: its sequence is not a gap, and not reusable.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::compactor::{archived_parquet_bytes, decode_parquet, parquet_name, CompactError};
use crate::event::{EventError, LogPayload};
use crate::wal::{
    numbered_files, segment_paths, sync_dir, sync_dir_for, write_atomically, OsFileOps, WalError,
    PARQUET_EXTENSION,
};
use crate::LogOffset;

/// The file in the log directory that lists the archived partitions.
pub const MANIFEST_FILE: &str = "partitions.manifest";
/// Layout version of the manifest. Any other is refused.
const MANIFEST_VERSION: u32 = 1;
const MILLIS_PER_DAY: i64 = 86_400_000;
const TEMP_SUFFIX: &str = ".archiving";

#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    #[error("archive io failed: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    Wal(#[from] WalError),
    #[error(transparent)]
    Compact(#[from] CompactError),
    #[error(transparent)]
    Event(#[from] EventError),
    #[error("the archive directory {} does not exist", path.display())]
    DirMissing { path: PathBuf },
    #[error(
        "partition {sequence} still holds a plaintext event, so nothing was archived; \
         run liamd encrypt-log first"
    )]
    PlaintextLogPresent { sequence: u64 },
    #[error("the archived copy of partition {sequence} does not match the original: {detail}")]
    CopyMismatch { sequence: u64, detail: String },
    #[error("the partition manifest {} cannot be read: {reason}", path.display())]
    ManifestUnreadable { path: PathBuf, reason: String },
}

/// One partition that was moved: where it is and how many events it holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchivedPartition {
    pub sequence: u64,
    pub events: u64,
    pub path: PathBuf,
}

impl ArchivedPartition {
    /// The offset of the first event of the partition.
    pub fn first(&self) -> LogOffset {
        LogOffset {
            segment: self.sequence,
            index: 0,
        }
    }

    /// The offset of the last event, `None` for a partition without events.
    pub fn last(&self) -> Option<LogOffset> {
        self.events.checked_sub(1).map(|index| LogOffset {
            segment: self.sequence,
            index,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct ManifestFile {
    version: u32,
    partitions: Vec<ArchivedPartition>,
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(MANIFEST_FILE)
}

/// The partitions listed by the manifest of the log in `dir`, by sequence. A
/// log without a manifest has archived nothing.
///
/// # Errors
///
/// Fails when the manifest exists and cannot be read or is another version.
pub fn read_manifest(dir: &Path) -> Result<Vec<ArchivedPartition>, ArchiveError> {
    let path = manifest_path(dir);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let unreadable = |reason: String| ArchiveError::ManifestUnreadable {
        path: path.clone(),
        reason,
    };
    let file: ManifestFile =
        postcard::from_bytes(&bytes).map_err(|error| unreadable(error.to_string()))?;
    if file.version != MANIFEST_VERSION {
        return Err(unreadable(format!(
            "it is version {}, but this build reads version {MANIFEST_VERSION}",
            file.version
        )));
    }
    Ok(file.partitions)
}

fn write_manifest(dir: &Path, partitions: &[ArchivedPartition]) -> Result<(), ArchiveError> {
    let bytes = postcard::to_stdvec(&ManifestFile {
        version: MANIFEST_VERSION,
        partitions: partitions.to_vec(),
    })
    .map_err(|error| ArchiveError::ManifestUnreadable {
        path: manifest_path(dir),
        reason: error.to_string(),
    })?;
    write_atomically(&manifest_path(dir), &bytes, &mut OsFileOps)?;
    Ok(())
}

/// Where a run stopped, to stand in for a crash in a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveStep {
    /// The compressed copy is complete beside its final name, not yet in place.
    Written,
    /// The copy is in the archive directory.
    Placed,
    /// The copy was read back and equals the original.
    Verified,
    /// The manifest lists the partition, and the original is still in the log
    /// directory.
    Recorded,
}

pub trait ArchiveFault: Send + Sync {
    fn reached(&self, step: ArchiveStep, sequence: u64) -> io::Result<()>;
}

/// Never fails.
pub struct NoFault;

impl ArchiveFault for NoFault {
    fn reached(&self, _step: ArchiveStep, _sequence: u64) -> io::Result<()> {
        Ok(())
    }
}

/// Which partitions to archive, and how to compress them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchivePolicy {
    /// A partition whose last event is at least this many days old is archived.
    pub after_days: u32,
    pub zstd_level: i32,
}

/// What a run moved.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ArchiveReport {
    pub archived: Vec<u64>,
    /// Partitions a previous run left half done, now finished.
    pub finished: Vec<u64>,
}

/// Archives the closed partitions of the log in `dir` that are old enough at
/// `now_ms`, into `archive_dir`.
///
/// # Errors
///
/// Fails before moving anything when `archive_dir` does not exist or a
/// partition due for archiving still holds a plaintext event.
pub fn archive_partitions(
    dir: &Path,
    archive_dir: &Path,
    policy: ArchivePolicy,
    now_ms: i64,
    fault: &dyn ArchiveFault,
) -> Result<ArchiveReport, ArchiveError> {
    if !archive_dir.is_dir() {
        return Err(ArchiveError::DirMissing {
            path: archive_dir.to_path_buf(),
        });
    }
    let mut manifest = read_manifest(dir)?;
    let mut report = ArchiveReport {
        finished: finish_interrupted(dir, &manifest)?,
        ..ArchiveReport::default()
    };
    let due = due_partitions(dir, &manifest, policy, now_ms)?;
    for (sequence, path) in due {
        let archived = archive_one(archive_dir, policy, sequence, &path, fault)?;
        manifest.push(archived);
        manifest.sort_by_key(|partition| partition.sequence);
        write_manifest(dir, &manifest)?;
        fault.reached(ArchiveStep::Recorded, sequence)?;
        remove_original(&path)?;
        report.archived.push(sequence);
    }
    Ok(report)
}

/// Closed partitions old enough to move, checked all together so a plaintext
/// one stops the run before any other moves.
fn due_partitions(
    dir: &Path,
    manifest: &[ArchivedPartition],
    policy: ArchivePolicy,
    now_ms: i64,
) -> Result<Vec<(u64, PathBuf)>, ArchiveError> {
    let cutoff = now_ms.saturating_sub(i64::from(policy.after_days) * MILLIS_PER_DAY);
    let open: Vec<u64> = segment_paths(dir)?
        .into_iter()
        .map(|(sequence, _)| sequence)
        .collect();
    let mut due = Vec::new();
    for (sequence, path) in numbered_files(dir, PARQUET_EXTENSION)? {
        if open.contains(&sequence) || manifest.iter().any(|done| done.sequence == sequence) {
            continue;
        }
        let events = decode_parquet(File::open(&path)?, sequence)?;
        let Some(newest) = events.iter().map(|event| event.ingested_at).max() else {
            continue;
        };
        if newest > cutoff {
            continue;
        }
        let plain = events.iter().any(|event| {
            !matches!(event.payload, LogPayload::Sealed(_)) && event.payload.holds_rows()
        });
        if plain {
            return Err(ArchiveError::PlaintextLogPresent { sequence });
        }
        due.push((sequence, path));
    }
    Ok(due)
}

/// Writes the compressed copy, places it, and checks it against the original.
fn archive_one(
    archive_dir: &Path,
    policy: ArchivePolicy,
    sequence: u64,
    original: &Path,
    fault: &dyn ArchiveFault,
) -> Result<ArchivedPartition, ArchiveError> {
    let events = decode_parquet(File::open(original)?, sequence)?;
    let placed = archive_dir.join(parquet_name(sequence));
    let mut staged = placed.as_os_str().to_owned();
    staged.push(TEMP_SUFFIX);
    let staged = PathBuf::from(staged);
    write_synced(
        &staged,
        &archived_parquet_bytes(sequence, &events, policy.zstd_level)?,
    )?;
    fault.reached(ArchiveStep::Written, sequence)?;
    fs::rename(&staged, &placed)?;
    sync_dir(archive_dir)?;
    fault.reached(ArchiveStep::Placed, sequence)?;
    let copy = decode_parquet(File::open(&placed)?, sequence).map_err(|error| {
        ArchiveError::CopyMismatch {
            sequence,
            detail: error.to_string(),
        }
    })?;
    if copy != events {
        return Err(ArchiveError::CopyMismatch {
            sequence,
            detail: "the events read back differ".to_owned(),
        });
    }
    fault.reached(ArchiveStep::Verified, sequence)?;
    Ok(ArchivedPartition {
        sequence,
        events: events.len() as u64,
        path: placed,
    })
}

/// A partition the manifest lists whose original is still in the log
/// directory: the previous run stopped before removing it. The original goes
/// only once the archived copy is read back and equal to it.
fn finish_interrupted(
    dir: &Path,
    manifest: &[ArchivedPartition],
) -> Result<Vec<u64>, ArchiveError> {
    let mut finished = Vec::new();
    for archived in manifest {
        let original = dir.join(parquet_name(archived.sequence));
        if !original.exists() {
            continue;
        }
        let sequence = archived.sequence;
        let kept = decode_parquet(File::open(&original)?, sequence)?;
        let copy = decode_parquet(File::open(&archived.path)?, sequence).map_err(|error| {
            ArchiveError::CopyMismatch {
                sequence,
                detail: error.to_string(),
            }
        })?;
        if copy != kept {
            return Err(ArchiveError::CopyMismatch {
                sequence,
                detail: "the events read back differ".to_owned(),
            });
        }
        remove_original(&original)?;
        finished.push(sequence);
    }
    Ok(finished)
}

fn remove_original(original: &Path) -> Result<(), ArchiveError> {
    match fs::remove_file(original) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    sync_dir(sync_dir_for(original))?;
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

#[cfg(test)]
mod tests;
