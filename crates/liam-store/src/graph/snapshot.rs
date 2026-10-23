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

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::future::Future;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use liam_log::wal::{sync_dir_for, write_atomically, FileOps};
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
const ASIDE_SUFFIX: &str = ".pre-restore";
const WAL_SUFFIX: &str = "-wal";
const SHM_SUFFIX: &str = "-shm";
const META_TEMP_SUFFIX: &str = ".tmp";
const COPY_CHUNK_BYTES: usize = 64 * 1024;

/// Snapshots in one process take turns, so the sweep of leftovers never
/// removes the files of a snapshot that is still being written.
static SNAPSHOT_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The disk operations a snapshot or restore performs beyond the log crate's
/// `FileOps`, so tests can fail each one in turn.
pub(crate) trait SnapshotOps: FileOps {
    fn remove_file(&mut self, path: &Path) -> io::Result<()>;
}

/// The real disk.
pub(crate) struct OsOps;

impl FileOps for OsOps {
    fn write_all(&mut self, file: &mut File, bytes: &[u8]) -> io::Result<()> {
        file.write_all(bytes)
    }

    fn sync_file(&mut self, file: &File) -> io::Result<()> {
        file.sync_all()
    }

    fn truncate(&mut self, file: &File, len: u64) -> io::Result<()> {
        file.set_len(len)
    }

    fn sync_dir(&mut self, dir: &Path) -> io::Result<()> {
        File::open(dir)?.sync_all()
    }
}

impl SnapshotOps for OsOps {
    fn remove_file(&mut self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }
}

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
pub(crate) struct LogPosition {
    pub log_id: Uuid,
    pub head: Option<LogOffset>,
}

impl<B: Backend> Graph<B> {
    pub(crate) async fn log_position(&self) -> Option<LogPosition> {
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
    /// The log lock is held only while the cursor is read and the database is
    /// copied (`VACUUM INTO` on a read connection), so no logged write lands
    /// between the two and the cursor matches the copy's contents. Writers that
    /// use the log wait for that copy; reads and unlogged writes do not. The
    /// copy is a consistent rewrite of the database, not a byte-for-byte copy
    /// of the live file. Checksumming, syncing and publishing happen after the
    /// lock is released, and the copy is renamed into place only once whole, so
    /// a failure leaves nothing at the snapshot's final paths. Leftovers of an
    /// earlier crashed snapshot in `dir` are swept first.
    pub async fn snapshot(&self, dir: &Path) -> Result<Snapshot> {
        self.snapshot_with(dir, OsOps, || std::future::ready(()))
            .await
    }

    /// `snapshot` over the given disk operations, with `between` awaited after
    /// the cursor is read and before the copy starts.
    pub(crate) async fn snapshot_with<O, P, F>(
        &self,
        dir: &Path,
        mut ops: O,
        between: P,
    ) -> Result<Snapshot>
    where
        O: SnapshotOps + 'static,
        P: Fn() -> F,
        F: Future<Output = ()>,
    {
        let _turn = SNAPSHOT_GATE.lock().await;
        let snapshot_id = ulid::Ulid::gen().to_string();
        let paths = SnapshotPaths::new(dir, &snapshot_id);
        tracing::debug!(%snapshot_id, "snapshot: copying the database under the log lock");
        fs::create_dir_all(dir).map_err(snapshot_io("create the snapshot directory", dir))?;
        sweep_leftovers(dir, &mut ops);
        let log = match self.copy_database(&paths.partial_database, between).await {
            Ok(log) => log,
            Err(error) => {
                remove_quietly(&mut ops, &paths.partial_database);
                return Err(error);
            }
        };
        tracing::debug!(%snapshot_id, "snapshot: copy taken, log lock released");
        let created_at = self.clock.now();
        let published = tokio::task::spawn_blocking(move || {
            publish(&mut ops, paths, snapshot_id, created_at, log)
        })
        .await
        .map_err(|error| Error::Snapshot(format!("the snapshot task failed: {error}")))?;
        let snapshot = published?;
        tracing::debug!(snapshot_id = %snapshot.meta.snapshot_id, "snapshot: published");
        Ok(snapshot)
    }

    async fn copy_database<P, F>(&self, target: &Path, between: P) -> Result<Option<SnapshotLog>>
    where
        P: Fn() -> F,
        F: Future<Output = ()>,
    {
        let held = match &self.log {
            Some(log) => Some(HeldLog::acquire(log).await?),
            None => None,
        };
        // Read out of `held` before the await: the guard is not `Sync`, so a
        // borrow of it across the await would make the snapshot unspawnable.
        let identity = held
            .as_ref()
            .map(|held| held.head().map(|head| (held.log_id, head)))
            .transpose()?;
        let log = match identity {
            Some((log_id, head)) => Some(SnapshotLog {
                log_id,
                cursor: log_cursor::read(&self.backend)
                    .await?
                    .and_then(|cursor| cursor.last),
                head,
            }),
            None => None,
        };
        between().await;
        let target_text = target
            .to_str()
            .ok_or_else(|| Error::Snapshot(format!("{} is not valid UTF-8", target.display())))?;
        self.backend
            .query("VACUUM INTO ?1", &[target_text.into()])
            .await
            .map_err(|error| {
                Error::Snapshot(format!(
                    "could not copy the database to {}: {error}",
                    target.display()
                ))
            })?;
        Ok(log)
    }

    /// Puts the snapshot described by `meta_path` in place of the database file
    /// at `database_path`, after checking it against the log this store is
    /// attached to. Reopen the store with its log and run `catch_up` afterwards.
    ///
    /// Takes the store by value so its connections are closed before the file
    /// is replaced; an open connection would keep the old file and write a WAL
    /// beside the new one. The caller must also hold the daemon's `StoreLock`,
    /// which this crate cannot take, so no other process opens the file
    /// meanwhile.
    ///
    /// Refused before anything is touched: a snapshot of another log
    /// (`SnapshotLogMismatch`), one ahead of the log (`SnapshotAheadOfLog`),
    /// one taken without a log for a logged store (`SnapshotWithoutLog`), one
    /// taken with a log for a store without one (`NoEventLog`), and one whose
    /// file does not match its recorded checksum (`SnapshotChecksumMismatch`).
    /// The copy is verified while it is staged beside the live file, so a
    /// corrupt snapshot never reaches it. `RestoreFailed` means the swap itself
    /// did not complete; see `recover_interrupted_restore`.
    pub async fn restore_snapshot(
        self,
        database_path: &Path,
        meta_path: &Path,
    ) -> Result<SnapshotMeta> {
        let position = self.log_position().await;
        drop(self);
        let (database_path, meta_path) = (database_path.to_owned(), meta_path.to_owned());
        tokio::task::spawn_blocking(move || {
            restore_files(&mut OsOps, &database_path, &meta_path, position.as_ref())
        })
        .await
        .map_err(|error| Error::RestoreFailed(format!("the restore task failed: {error}")))?
    }
}

/// Finishes or undoes a restore that a crash interrupted, so the database is
/// either the one it was before or the snapshot, with the matching `-wal`.
/// `Graph::open` calls it first; a restore does too.
///
/// A restore moves the live `-wal` and `-shm` aside, renames the staged copy
/// over the database, then deletes the asides. The staged file exists exactly
/// until the rename, which tells the cases apart: if it is still there the swap
/// never happened and the asides go back beside the old database, otherwise the
/// swap is done and the asides, which belong to the replaced file, are deleted.
pub fn recover_interrupted_restore(database_path: &Path) -> Result<()> {
    recover_with(&mut OsOps, database_path)
}

fn recover_with(ops: &mut impl SnapshotOps, database_path: &Path) -> Result<()> {
    let staged = with_suffix(database_path, STAGED_SUFFIX);
    let asides: Vec<(PathBuf, PathBuf)> = sidecars(database_path)
        .into_iter()
        .filter(|(_, aside)| aside.exists())
        .collect();
    let swap_pending = staged.exists();
    if asides.is_empty() && !swap_pending {
        return Ok(());
    }
    tracing::warn!(database = %database_path.display(), swap_pending, "restore: finishing an interrupted restore");
    if swap_pending {
        for (live, aside) in &asides {
            if live.exists() {
                return Err(Error::RestoreFailed(format!(
                    "both {} and {} exist, so neither can be put back safely; \
                     keep the one with the committed data and remove the other",
                    live.display(),
                    aside.display()
                )));
            }
            ops.rename(aside, live)
                .map_err(restore_io("put back", live))?;
        }
        ops.remove_file(&staged)
            .or_else(ignore_not_found)
            .map_err(restore_io("remove", &staged))
    } else {
        for (_, aside) in &asides {
            ops.remove_file(aside)
                .or_else(ignore_not_found)
                .map_err(restore_io("remove", aside))?;
        }
        Ok(())
    }
}

/// The live `-wal` and `-shm` of `database_path`, each with the name it is
/// moved aside to.
fn sidecars(database_path: &Path) -> Vec<(PathBuf, PathBuf)> {
    [WAL_SUFFIX, SHM_SUFFIX]
        .into_iter()
        .map(|suffix| {
            let live = with_suffix(database_path, suffix);
            let aside = with_suffix(&live, ASIDE_SUFFIX);
            (live, aside)
        })
        .collect()
}

pub(crate) fn restore_files(
    ops: &mut impl SnapshotOps,
    database_path: &Path,
    meta_path: &Path,
    log: Option<&LogPosition>,
) -> Result<SnapshotMeta> {
    recover_with(ops, database_path)?;
    let meta = read_meta(meta_path)?;
    check_log(&meta, log)?;
    let snapshot_file = meta_path.with_extension(DATABASE_EXTENSION);
    let staged = with_suffix(database_path, STAGED_SUFFIX);
    tracing::debug!(snapshot_id = %meta.snapshot_id, "restore: staging and verifying the snapshot");
    let swapped = stage_verified(ops, &snapshot_file, &staged, &meta)
        .and_then(|()| swap_in(ops, &staged, database_path));
    if let Err(error) = swapped {
        remove_quietly(ops, &staged);
        return Err(error);
    }
    tracing::debug!(snapshot_id = %meta.snapshot_id, "restore: snapshot in place");
    Ok(meta)
}

fn check_log(meta: &SnapshotMeta, current: Option<&LogPosition>) -> Result<()> {
    match (&meta.log, current) {
        (None, None) => Ok(()),
        (Some(_), None) => Err(Error::NoEventLog),
        (None, Some(_)) => Err(Error::SnapshotWithoutLog {
            snapshot_id: meta.snapshot_id.clone(),
        }),
        (Some(snapshot), Some(current)) => {
            if snapshot.log_id != current.log_id {
                return Err(Error::SnapshotLogMismatch {
                    snapshot: snapshot.log_id,
                    store: current.log_id,
                });
            }
            match snapshot.cursor {
                Some(cursor) if current.head.is_none_or(|head| cursor > head) => {
                    Err(Error::SnapshotAheadOfLog {
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
fn stage_verified(
    ops: &mut impl SnapshotOps,
    snapshot_file: &Path,
    staged: &Path,
    meta: &SnapshotMeta,
) -> Result<()> {
    let mut source =
        File::open(snapshot_file).map_err(snapshot_io("open the snapshot file", snapshot_file))?;
    let mut target = File::create(staged).map_err(restore_io("create", staged))?;
    let (sha256, bytes) = copy_hashed(&mut source, |chunk| ops.write_all(&mut target, chunk))
        .map_err(restore_io("copy to", staged))?;
    if bytes != meta.bytes || sha256 != meta.sha256 {
        return Err(Error::SnapshotChecksumMismatch {
            snapshot_id: meta.snapshot_id.clone(),
        });
    }
    ops.sync_file(&target).map_err(restore_io("sync", staged))
}

/// Replaces the live file with `staged` without ever deleting the live `-wal`:
/// it holds commits the database file does not yet, so it is moved aside, and
/// deleted only after the swap. A leftover WAL beside the new file would be
/// replayed onto the snapshot's pages, which is why it cannot stay in place.
/// Every failure before the rename puts the asides back; a crash leaves the
/// state `recover_interrupted_restore` reads.
fn swap_in(ops: &mut impl SnapshotOps, staged: &Path, database_path: &Path) -> Result<()> {
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (live, aside) in sidecars(database_path) {
        match ops.rename(&live, &aside) {
            Ok(()) => moved.push((live, aside)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                put_back(ops, &moved);
                return Err(restore_io("move aside", &live)(error));
            }
        }
    }
    if let Err(error) = ops.rename(staged, database_path) {
        put_back(ops, &moved);
        return Err(restore_io("replace", database_path)(error));
    }
    ops.sync_dir(sync_dir_for(database_path)).map_err(|error| {
        Error::RestoreFailed(format!(
            "the snapshot replaced {} but the change may not be durable yet: {error}; \
             run the restore again",
            database_path.display()
        ))
    })?;
    for (_, aside) in &moved {
        remove_quietly(ops, aside);
    }
    Ok(())
}

fn put_back(ops: &mut impl SnapshotOps, moved: &[(PathBuf, PathBuf)]) {
    for (live, aside) in moved {
        if let Err(error) = ops.rename(aside, live) {
            tracing::error!(
                live = %live.display(), aside = %aside.display(), %error,
                "could not put a moved-aside WAL file back; recover_interrupted_restore will"
            );
        }
    }
}

struct SnapshotPaths {
    database: PathBuf,
    meta: PathBuf,
    partial_database: PathBuf,
    meta_temp: PathBuf,
}

impl SnapshotPaths {
    fn new(dir: &Path, snapshot_id: &str) -> Self {
        let database = dir.join(snapshot_id).with_extension(DATABASE_EXTENSION);
        let meta = dir.join(snapshot_id).with_extension(META_EXTENSION);
        Self {
            partial_database: with_suffix(&database, PARTIAL_SUFFIX),
            meta_temp: with_suffix(&meta, META_TEMP_SUFFIX),
            database,
            meta,
        }
    }
}

/// Removes what a crashed snapshot left in `dir`: unfinished copies and
/// metadata, and a database file whose metadata was never written. Only names
/// that start with a snapshot id are touched, as the directory may hold other
/// files.
fn sweep_leftovers(dir: &Path, ops: &mut impl SnapshotOps) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let names: BTreeSet<String> = entries
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    for name in &names {
        let Some((stem, rest)) = name.split_once('.') else {
            continue;
        };
        if ulid::Ulid::from_string(stem).is_err() {
            continue;
        }
        let leftover = match rest {
            "db.partial" | "json.tmp" => true,
            "db" => !names.contains(&format!("{stem}.{META_EXTENSION}")),
            _ => false,
        };
        if leftover {
            remove_quietly(ops, &dir.join(name));
        }
    }
}

/// Checksums the copied database, syncs it, and renames it and its metadata
/// into place, metadata last. Cleans up after itself on failure, metadata
/// first so no published metadata outlives its database.
fn publish(
    ops: &mut impl SnapshotOps,
    paths: SnapshotPaths,
    snapshot_id: String,
    created_at: Millis,
    log: Option<SnapshotLog>,
) -> Result<Snapshot> {
    let published = write_files(ops, &paths, snapshot_id, created_at, log);
    if published.is_err() {
        for leftover in [
            &paths.meta,
            &paths.meta_temp,
            &paths.database,
            &paths.partial_database,
        ] {
            remove_quietly(ops, leftover);
        }
    }
    published
}

fn write_files(
    ops: &mut impl SnapshotOps,
    paths: &SnapshotPaths,
    snapshot_id: String,
    created_at: Millis,
    log: Option<SnapshotLog>,
) -> Result<Snapshot> {
    let mut copy = File::open(&paths.partial_database)
        .map_err(snapshot_io("open the copy", &paths.partial_database))?;
    let (sha256, bytes) = copy_hashed(&mut copy, |_| Ok(()))
        .map_err(snapshot_io("checksum the copy", &paths.partial_database))?;
    ops.sync_file(&copy)
        .map_err(snapshot_io("sync the copy", &paths.partial_database))?;
    let meta = SnapshotMeta {
        snapshot_id,
        created_at,
        log,
        sha256,
        bytes,
    };
    ops.rename(&paths.partial_database, &paths.database)
        .map_err(snapshot_io("publish the copy", &paths.database))?;
    let encoded = serde_json::to_vec_pretty(&StoredMeta::from(&meta))?;
    write_atomically(&paths.meta, &encoded, ops)
        .map_err(snapshot_io("publish the metadata", &paths.meta))?;
    Ok(Snapshot {
        meta,
        database_path: paths.database.clone(),
        meta_path: paths.meta.clone(),
    })
}

/// Streams `source` through `sink` in chunks, returning the hex SHA-256 and
/// length of what passed.
fn copy_hashed(
    source: &mut impl Read,
    mut sink: impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_CHUNK_BYTES];
    let mut total = 0_u64;
    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        sink(&buffer[..read])?;
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

fn ignore_not_found(error: io::Error) -> io::Result<()> {
    if error.kind() == io::ErrorKind::NotFound {
        Ok(())
    } else {
        Err(error)
    }
}

fn remove_quietly(ops: &mut impl SnapshotOps, path: &Path) {
    if let Err(error) = ops.remove_file(path).or_else(ignore_not_found) {
        tracing::warn!(path = %path.display(), %error, "could not remove a leftover snapshot file");
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
