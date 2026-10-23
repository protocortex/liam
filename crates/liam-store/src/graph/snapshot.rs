// SPDX-License-Identifier: Apache-2.0
//! Point-in-time snapshots of the database file, and their restore.
//!
//! The database is a projection of the event log, which only grows. A snapshot
//! is the database at a moment when it agreed with the log's cursor, plus the
//! log id and the cursor and head offsets of that moment. Restoring puts the
//! database back and nothing else: the log is not rolled back, so the restored
//! store is behind the log and `catch_up` carries it forward, applying each
//! record the snapshot did not hold once. A rollback therefore returns to the
//! log's full state, not to the snapshot's.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use liam_log::LogOffset;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::log_cursor;
use super::logged_write::HeldLog;
use super::Graph;
use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::ids::Millis;

const FORMAT_VERSION: u32 = 1;
const DATABASE_EXTENSION: &str = "db";
const META_EXTENSION: &str = "json";
const PARTIAL_SUFFIX: &str = ".partial";
const STAGED_SUFFIX: &str = ".restore";
const WAL_SUFFIX: &str = "-wal";
const SHM_SUFFIX: &str = "-shm";
const COPY_CHUNK_BYTES: usize = 64 * 1024;

/// Where the store stood against its log when a snapshot was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotLog {
    pub log_id: Uuid,
    /// The last record the database had accounted for.
    pub cursor: Option<LogOffset>,
    /// The last record the log had acknowledged.
    pub head: Option<LogOffset>,
}

/// What a snapshot is: when it was taken, against which log, and the checksum
/// and size of its database file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotMeta {
    pub snapshot_id: String,
    pub created_at: Millis,
    /// `None` for a store that has no event log.
    pub log: Option<SnapshotLog>,
    /// Lowercase hex SHA-256 of the database file.
    pub sha256: String,
    pub bytes: u64,
}

/// A snapshot on disk. It exists once `meta_path` does: the database file is
/// put in place before the metadata that vouches for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub meta: SnapshotMeta,
    pub database_path: PathBuf,
    pub meta_path: PathBuf,
}

/// The log a store is attached to, as a restore must see it to decide whether
/// a snapshot still belongs to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogPosition {
    pub log_id: Uuid,
    pub head: Option<LogOffset>,
}

impl<B: Backend> Graph<B> {
    /// The attached log's id and head, `None` for a store without a log. A
    /// caller reads this before closing the store, to hand to `restore_snapshot`.
    pub async fn log_position(&self) -> Option<LogPosition> {
        let log = self.log.as_ref()?.lock().await;
        Some(LogPosition {
            log_id: log.log_id(),
            head: log.head(),
        })
    }

    /// Writes a consistent copy of the database into `dir`, with the log id and
    /// offsets of the moment it was taken. Fails on a poisoned log, whose tail
    /// is unknown.
    ///
    /// The log lock and then the write lock are held only while the database
    /// is copied (`VACUUM INTO`), so no write is half applied in the copy and
    /// the cursor read beside it matches its contents. Checksumming, syncing
    /// and publishing the files happen after both are released. The copy is
    /// written under a temporary name and renamed once whole, so a failure
    /// leaves nothing at the snapshot's final paths.
    pub async fn snapshot(&self, dir: &Path) -> Result<Snapshot> {
        let snapshot_id = ulid::Ulid::gen().to_string();
        let paths = SnapshotPaths::new(dir, &snapshot_id);
        tracing::debug!(%snapshot_id, "snapshot: copying the database under the log lock");
        let copied = async {
            fs::create_dir_all(dir).map_err(snapshot_io("create the snapshot directory", dir))?;
            self.copy_database(&paths.partial_database).await
        }
        .await;
        let log = match copied {
            Ok(log) => log,
            Err(error) => {
                remove_quietly(&paths.partial_database);
                return Err(error);
            }
        };
        tracing::debug!(%snapshot_id, "snapshot: copy taken, locks released");
        let created_at = self.clock.now();
        let published =
            tokio::task::spawn_blocking(move || publish(paths, snapshot_id, created_at, log))
                .await
                .map_err(|error| Error::Snapshot(format!("the snapshot task failed: {error}")))?;
        let snapshot = published?;
        tracing::debug!(snapshot_id = %snapshot.meta.snapshot_id, "snapshot: published");
        Ok(snapshot)
    }

    async fn copy_database(&self, target: &Path) -> Result<Option<SnapshotLog>> {
        let held = match &self.log {
            Some(log) => Some(HeldLog::acquire(log).await?),
            None => None,
        };
        let log = match &held {
            Some(held) => Some(SnapshotLog {
                log_id: held.log_id,
                cursor: log_cursor::read(&self.backend)
                    .await?
                    .and_then(|cursor| cursor.last),
                head: held.head()?,
            }),
            None => None,
        };
        let target_text = target
            .to_str()
            .ok_or_else(|| Error::Snapshot(format!("{} is not valid UTF-8", target.display())))?;
        self.backend
            .execute("VACUUM INTO ?1", &[target_text.into()])
            .await
            .map_err(|error| {
                Error::Snapshot(format!(
                    "could not copy the database to {}: {error}",
                    target.display()
                ))
            })?;
        Ok(log)
    }
}

/// Puts the snapshot described by `meta_path` in place of the database file at
/// `database_path`, after checking it against `log`, the log the store is
/// attached to now. The store must be closed first. Reopen it with its log and
/// run `catch_up` afterwards.
///
/// Refused before anything is touched: a snapshot of another log
/// (`LogIdMismatch`), one ahead of the log (`CursorBeyondLog`), one taken
/// without a log for a logged store, and one whose file does not match its
/// recorded checksum (`SnapshotCorrupt`). The copy is checked while it is
/// staged beside the live file, so a corrupt snapshot never reaches it.
/// `RestoreFailed` means the swap itself did not complete.
pub fn restore_snapshot(
    database_path: &Path,
    meta_path: &Path,
    log: Option<&LogPosition>,
) -> Result<SnapshotMeta> {
    let meta = read_meta(meta_path)?;
    check_log(&meta, log)?;
    let snapshot_file = meta_path.with_extension(DATABASE_EXTENSION);
    let staged = with_suffix(database_path, STAGED_SUFFIX);
    tracing::debug!(snapshot_id = %meta.snapshot_id, "restore: staging and verifying the snapshot");
    let swapped = stage_verified(&snapshot_file, &staged, &meta)
        .and_then(|()| swap_in(&staged, database_path));
    if let Err(error) = swapped {
        remove_quietly(&staged);
        return Err(error);
    }
    tracing::debug!(snapshot_id = %meta.snapshot_id, "restore: snapshot in place");
    Ok(meta)
}

fn check_log(meta: &SnapshotMeta, current: Option<&LogPosition>) -> Result<()> {
    match (&meta.log, current) {
        (None, None) => Ok(()),
        (Some(_), None) => Err(Error::NoEventLog),
        (None, Some(_)) => Err(Error::Snapshot(format!(
            "snapshot {} was taken without an event log, so its position in the log is unknown",
            meta.snapshot_id
        ))),
        (Some(snapshot), Some(current)) => {
            if snapshot.log_id != current.log_id {
                return Err(Error::LogIdMismatch {
                    store: snapshot.log_id,
                    log: current.log_id,
                });
            }
            match snapshot.cursor {
                Some(cursor) if current.head.is_none_or(|head| cursor > head) => {
                    Err(Error::CursorBeyondLog {
                        cursor,
                        head: current.head,
                    })
                }
                _ => Ok(()),
            }
        }
    }
}

/// Copies the snapshot's database file to `staged` and syncs it, checking size
/// and checksum on the bytes that were written.
fn stage_verified(snapshot_file: &Path, staged: &Path, meta: &SnapshotMeta) -> Result<()> {
    let mut source =
        File::open(snapshot_file).map_err(snapshot_io("open the snapshot file", snapshot_file))?;
    let mut target = File::create(staged).map_err(restore_io("create", staged))?;
    let (sha256, bytes) =
        copy_hashed(&mut source, &mut target).map_err(restore_io("copy to", staged))?;
    if bytes != meta.bytes || sha256 != meta.sha256 {
        return Err(Error::SnapshotCorrupt(format!(
            "snapshot {} does not match its recorded checksum and size",
            meta.snapshot_id
        )));
    }
    target.sync_all().map_err(restore_io("sync", staged))
}

/// Replaces the live file with `staged`. The live file's `-wal` and `-shm` are
/// removed first: a leftover WAL would be replayed onto the snapshot's pages.
/// Nothing after this point can fail without the swap, and everything before it
/// can fail without touching the live file.
fn swap_in(staged: &Path, database_path: &Path) -> Result<()> {
    for suffix in [WAL_SUFFIX, SHM_SUFFIX] {
        let sidecar = with_suffix(database_path, suffix);
        match fs::remove_file(&sidecar) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(restore_io("remove", &sidecar)(error)),
        }
    }
    fs::rename(staged, database_path).map_err(restore_io("replace", database_path))?;
    sync_parent(database_path).map_err(restore_io("sync the directory of", database_path))
}

struct SnapshotPaths {
    database: PathBuf,
    meta: PathBuf,
    partial_database: PathBuf,
    partial_meta: PathBuf,
}

impl SnapshotPaths {
    fn new(dir: &Path, snapshot_id: &str) -> Self {
        let database = dir.join(snapshot_id).with_extension(DATABASE_EXTENSION);
        let meta = dir.join(snapshot_id).with_extension(META_EXTENSION);
        Self {
            partial_database: with_suffix(&database, PARTIAL_SUFFIX),
            partial_meta: with_suffix(&meta, PARTIAL_SUFFIX),
            database,
            meta,
        }
    }
}

/// Checksums the copied database, syncs it, and renames it and its metadata
/// into place, metadata last. Cleans up after itself on failure.
fn publish(
    paths: SnapshotPaths,
    snapshot_id: String,
    created_at: Millis,
    log: Option<SnapshotLog>,
) -> Result<Snapshot> {
    let published = write_files(&paths, snapshot_id, created_at, log);
    if published.is_err() {
        for leftover in [
            &paths.partial_database,
            &paths.partial_meta,
            &paths.database,
            &paths.meta,
        ] {
            remove_quietly(leftover);
        }
    }
    published
}

fn write_files(
    paths: &SnapshotPaths,
    snapshot_id: String,
    created_at: Millis,
    log: Option<SnapshotLog>,
) -> Result<Snapshot> {
    let mut copy = File::open(&paths.partial_database)
        .map_err(snapshot_io("open the copy", &paths.partial_database))?;
    let (sha256, bytes) = copy_hashed(&mut copy, &mut io::sink())
        .map_err(snapshot_io("checksum the copy", &paths.partial_database))?;
    File::open(&paths.partial_database)
        .and_then(|file| file.sync_all())
        .map_err(snapshot_io("sync the copy", &paths.partial_database))?;
    let meta = SnapshotMeta {
        snapshot_id,
        created_at,
        log,
        sha256,
        bytes,
    };
    fs::rename(&paths.partial_database, &paths.database)
        .map_err(snapshot_io("publish the copy", &paths.database))?;
    let encoded = serde_json::to_vec_pretty(&StoredMeta::from(&meta))?;
    let mut partial = File::create(&paths.partial_meta)
        .map_err(snapshot_io("create the metadata", &paths.partial_meta))?;
    partial
        .write_all(&encoded)
        .and_then(|()| partial.sync_all())
        .map_err(snapshot_io("write the metadata", &paths.partial_meta))?;
    fs::rename(&paths.partial_meta, &paths.meta)
        .map_err(snapshot_io("publish the metadata", &paths.meta))?;
    sync_parent(&paths.meta).map_err(snapshot_io("sync the directory of", &paths.meta))?;
    Ok(Snapshot {
        meta,
        database_path: paths.database.clone(),
        meta_path: paths.meta.clone(),
    })
}

/// Streams `source` into `sink`, returning the hex SHA-256 and length of what
/// passed through.
fn copy_hashed(source: &mut impl Read, sink: &mut impl Write) -> io::Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_CHUNK_BYTES];
    let mut total = 0_u64;
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        sink.write_all(&buffer[..read])?;
        total += read as u64;
    }
    let hex = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((hex, total))
}

fn read_meta(meta_path: &Path) -> Result<SnapshotMeta> {
    let raw = fs::read(meta_path).map_err(snapshot_io("read the snapshot metadata", meta_path))?;
    let stored: StoredMeta = serde_json::from_slice(&raw).map_err(|error| {
        Error::SnapshotCorrupt(format!(
            "{} is not snapshot metadata: {error}",
            meta_path.display()
        ))
    })?;
    SnapshotMeta::try_from(stored)
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn remove_quietly(path: &Path) {
    if let Err(error) = fs::remove_file(path) {
        if error.kind() != io::ErrorKind::NotFound {
            tracing::warn!(path = %path.display(), %error, "could not remove a leftover snapshot file");
        }
    }
}

fn sync_parent(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => File::open(parent)?.sync_all(),
        _ => Ok(()),
    }
}

fn snapshot_io<'a>(action: &'a str, path: &'a Path) -> impl FnOnce(io::Error) -> Error + 'a {
    move |error| Error::Snapshot(format!("could not {action} {}: {error}", path.display()))
}

fn restore_io<'a>(action: &'a str, path: &'a Path) -> impl FnOnce(io::Error) -> Error + 'a {
    move |error| Error::RestoreFailed(format!("could not {action} {}: {error}", path.display()))
}

/// The metadata file's shape, kept apart from `SnapshotMeta` so the on-disk
/// format does not depend on how the log crate represents ids and offsets.
#[derive(Serialize, Deserialize)]
struct StoredMeta {
    format: u32,
    snapshot_id: String,
    created_at: i64,
    log: Option<StoredLog>,
    sha256: String,
    bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct StoredLog {
    log_id: String,
    cursor: Option<StoredOffset>,
    head: Option<StoredOffset>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
struct StoredOffset {
    segment: u64,
    index: u64,
}

impl From<LogOffset> for StoredOffset {
    fn from(offset: LogOffset) -> Self {
        Self {
            segment: offset.segment,
            index: offset.index,
        }
    }
}

impl From<StoredOffset> for LogOffset {
    fn from(offset: StoredOffset) -> Self {
        Self {
            segment: offset.segment,
            index: offset.index,
        }
    }
}

impl From<&SnapshotMeta> for StoredMeta {
    fn from(meta: &SnapshotMeta) -> Self {
        Self {
            format: FORMAT_VERSION,
            snapshot_id: meta.snapshot_id.clone(),
            created_at: meta.created_at.0,
            log: meta.log.map(|log| StoredLog {
                log_id: log.log_id.to_string(),
                cursor: log.cursor.map(Into::into),
                head: log.head.map(Into::into),
            }),
            sha256: meta.sha256.clone(),
            bytes: meta.bytes,
        }
    }
}

impl TryFrom<StoredMeta> for SnapshotMeta {
    type Error = Error;

    fn try_from(stored: StoredMeta) -> Result<Self> {
        if stored.format != FORMAT_VERSION {
            return Err(Error::SnapshotCorrupt(format!(
                "snapshot {} has format {}, this build reads format {FORMAT_VERSION}",
                stored.snapshot_id, stored.format
            )));
        }
        let log = stored
            .log
            .map(|log| {
                let log_id = Uuid::parse_str(&log.log_id).map_err(|_| {
                    Error::SnapshotCorrupt(format!(
                        "snapshot {} names the log {:?}, which is not a UUID",
                        stored.snapshot_id, log.log_id
                    ))
                })?;
                Ok::<_, Error>(SnapshotLog {
                    log_id,
                    cursor: log.cursor.map(Into::into),
                    head: log.head.map(Into::into),
                })
            })
            .transpose()?;
        Ok(Self {
            snapshot_id: stored.snapshot_id,
            created_at: Millis(stored.created_at),
            log,
            sha256: stored.sha256,
            bytes: stored.bytes,
        })
    }
}
