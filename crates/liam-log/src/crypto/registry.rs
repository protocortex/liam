// SPDX-License-Identifier: Apache-2.0

//! The key registry: an append-only file of wrapped keys that lives beside the
//! log and outside every snapshot, so restoring a snapshot cannot bring a
//! destroyed key back.
//!
//! The file is a header, then records. A record is a tag, a header length, the
//! header, a CRC of those, and for a key a fixed-width slot holding the
//! wrapped data key, the wrapped dedup key, and the subject hash. Adding a key
//! appends, so nothing is ever rewritten. Destroying a key overwrites its slot
//! in place and appends a state record. The CRC deliberately skips the slot: a
//! torn overwrite must not make the file unreadable, and the slot is
//! authenticated by its own wrap.
//!
//! `keys.destroyed` beside it lists every key a delete destroyed, one line
//! each, written before the slot is overwritten. An older copy of the registry
//! that still shows such a key as active is finished off on open.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zeroize::Zeroizing;

use super::{
    hmac_sha256, random_array, CryptoError, DataKey, DedupKey, MasterKeyProvider, Wrapped, KEY_LEN,
    NONCE_LEN, TAG_LEN,
};
use crate::wal::sync_dir;

#[cfg(test)]
mod tests;

/// The registry file inside the log directory.
pub const REGISTRY_FILE: &str = "keys.registry";
/// The sidecar index of destroyed keys.
pub const DESTROYED_INDEX_FILE: &str = "keys.destroyed";

const MAGIC: &[u8; 8] = b"LIAMKEYS";
const FORMAT_VERSION: u32 = 1;
const FILE_HEADER_LEN: usize = MAGIC.len() + size_of::<u32>();

const TAG_SECRET: u8 = 1;
const TAG_KEY: u8 = 2;
const TAG_STATE: u8 = 3;

/// A wrapped 256-bit key: nonce, ciphertext, tag.
const WRAPPED_LEN: usize = NONCE_LEN + KEY_LEN + TAG_LEN;
const HASH_LEN: usize = 32;
const SLOT_LEN: usize = WRAPPED_LEN * 2 + HASH_LEN;
const DEDUP_AT: usize = WRAPPED_LEN;
const HASH_AT: usize = WRAPPED_LEN * 2;

const LENGTH_BYTES: usize = size_of::<u32>();
const CRC_BYTES: usize = size_of::<u32>();
const RECORD_PREFIX_LEN: usize = 1 + LENGTH_BYTES;
/// A header holds a few ids and one wrapped secret, so a larger length is damage.
const MAX_HEADER_LEN: usize = 1024;

/// The scope token for an entry that has none. A literal scope of this value
/// is refused so the two can never share a key group.
const NO_SCOPE_TOKEN: &str = "~";
/// Unwrapped keys kept in memory. The cache is dropped whole when it fills.
const KEY_CACHE_LIMIT: usize = 1024;

const SECRET_AAD: &[u8] = b"liam registry secret";
const DATA_AAD: &[u8] = b"data";
const DEDUP_AAD: &[u8] = b"dedup";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyState {
    Active,
    /// A delete job has marked the group and will destroy it. Writers are
    /// refused and readers treat its rows as shredded.
    Destroying,
    Destroyed,
}

/// The keys of one group, unwrapped.
#[derive(Debug, Clone)]
pub struct KeyMaterial {
    pub data: DataKey,
    pub dedup: DedupKey,
}

/// What the registry knows about a key id. `Missing` (a lost, truncated, or
/// older registry) is deliberately not `Destroyed`: it is damage, not a delete.
#[derive(Debug, Clone)]
pub enum KeyLookup {
    Active(KeyMaterial),
    Destroying,
    Destroyed,
    Missing,
}

#[derive(Serialize, Deserialize)]
struct SecretHeader {
    master_key_id: String,
    wrapped: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct KeyHeader {
    key_id: String,
    master_key_id: String,
}

#[derive(Serialize, Deserialize)]
struct StateHeader {
    key_id: String,
    state: KeyState,
}

/// Steps a test can stop at, to stand in for a crash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Point {
    DirectorySync,
    AfterDestroyingRecord,
    AfterIndexLine,
    AfterSlotOverwrite,
}

pub(crate) type FaultHook = Arc<dyn Fn(Point) -> io::Result<()> + Send + Sync>;

struct Entry {
    state: KeyState,
    slot_offset: u64,
}

struct Inner {
    file: File,
    path: PathBuf,
    index_path: PathBuf,
    end: u64,
    master: Arc<dyn MasterKeyProvider>,
    secret: Zeroizing<Vec<u8>>,
    entries: HashMap<String, Entry>,
    by_subject: HashMap<[u8; HASH_LEN], String>,
    cache: HashMap<String, KeyMaterial>,
    hook: Option<FaultHook>,
}

/// The registry of data keys. One process owns it at a time (the store lock
/// guarantees that), so there is no file lock.
pub struct KeyRegistry {
    inner: Mutex<Inner>,
    /// Held across a seal and its append, and taken to mark a group, so no
    /// append lands after the mark.
    write_gate: Mutex<()>,
}

impl KeyRegistry {
    /// Opens `dir`'s registry, creating it on first use. A torn final record
    /// is discarded, a damaged record is [`CryptoError::RegistryCorrupt`], and
    /// every key `keys.destroyed` lists is destroyed again if the registry
    /// still shows it as live.
    pub fn open(dir: &Path, master: Arc<dyn MasterKeyProvider>) -> Result<Self, CryptoError> {
        Self::open_with(dir, master, None)
    }

    pub(crate) fn open_with(
        dir: &Path,
        master: Arc<dyn MasterKeyProvider>,
        hook: Option<FaultHook>,
    ) -> Result<Self, CryptoError> {
        let mut inner = Inner::open(dir, master, hook)?;
        inner.reconcile_destroyed()?;
        Ok(Self {
            inner: Mutex::new(inner),
            write_gate: Mutex::new(()),
        })
    }

    /// Writers hold this from sealing a record until it is appended.
    pub fn write_gate(&self) -> MutexGuard<'_, ()> {
        self.write_gate
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The key group of a subject in a scope, made on first use. The subject
    /// is trimmed and lowercased, so `Alice` and `alice` share one group, and
    /// a missing scope is the token `~`. A group being destroyed is
    /// [`CryptoError::KeyGroupLocked`]; a destroyed group has no mapping left,
    /// so the same subject gets a new group.
    pub fn key_for_subject(
        &self,
        scope: Option<&str>,
        subject: &str,
    ) -> Result<String, CryptoError> {
        let mut inner = self.lock();
        let hash = inner.subject_hash(scope, subject)?;
        if let Some(key_id) = inner.by_subject.get(&hash).cloned() {
            return match inner.entries.get(&key_id).map(|entry| entry.state) {
                Some(KeyState::Active) => Ok(key_id),
                _ => Err(CryptoError::KeyGroupLocked { key_id }),
            };
        }
        inner.add_key(Some(hash))
    }

    /// A new group for rows that have no subject. It is never found again by
    /// lookup, only by the id the caller stores with the row.
    pub fn new_event_key(&self) -> Result<String, CryptoError> {
        self.lock().add_key(None)
    }

    pub fn open_key(&self, key_id: &str) -> Result<KeyLookup, CryptoError> {
        self.lock().open_key(key_id)
    }

    pub fn state(&self, key_id: &str) -> Option<KeyState> {
        self.lock().entries.get(key_id).map(|entry| entry.state)
    }

    /// Marks a group so no writer may use it and readers treat its rows as
    /// shredded. Idempotent.
    pub fn begin_destroy(&self, key_id: &str) -> Result<(), CryptoError> {
        let _gate = self.write_gate();
        let mut inner = self.lock();
        match inner.state_of(key_id)? {
            KeyState::Active => inner.set_state(key_id, KeyState::Destroying),
            KeyState::Destroying | KeyState::Destroyed => Ok(()),
        }
    }

    /// Destroys the key bytes of a marked group: the index line first, then
    /// the slot overwrite, then the `Destroyed` record. Idempotent, and a
    /// re-run after a crash at any step finishes the job.
    pub fn finish_destroy(&self, key_id: &str) -> Result<(), CryptoError> {
        let mut inner = self.lock();
        match inner.state_of(key_id)? {
            KeyState::Destroyed => Ok(()),
            KeyState::Active | KeyState::Destroying => inner.destroy(key_id),
        }
    }

    /// `begin_destroy` then `finish_destroy`.
    pub fn destroy(&self, key_id: &str) -> Result<(), CryptoError> {
        self.begin_destroy(key_id)?;
        self.finish_destroy(key_id)
    }

    /// Returns a marked group to active, for a job cancelled before it
    /// destroyed anything. A group whose destroy reached the index, or whose
    /// bytes are already gone, cannot return.
    pub fn cancel_destroy(&self, key_id: &str) -> Result<(), CryptoError> {
        let _gate = self.write_gate();
        let mut inner = self.lock();
        match inner.state_of(key_id)? {
            KeyState::Destroyed => Err(CryptoError::KeyDestroyed {
                key_id: key_id.to_owned(),
            }),
            KeyState::Active => Ok(()),
            KeyState::Destroying if inner.listed_in_index(key_id)? => {
                Err(CryptoError::KeyDestroyed {
                    key_id: key_id.to_owned(),
                })
            }
            KeyState::Destroying => inner.set_state(key_id, KeyState::Active),
        }
    }

    pub fn path(&self) -> PathBuf {
        self.lock().path.clone()
    }
}

impl Inner {
    fn open(
        dir: &Path,
        master: Arc<dyn MasterKeyProvider>,
        hook: Option<FaultHook>,
    ) -> Result<Self, CryptoError> {
        let path = dir.join(REGISTRY_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let bytes = Zeroizing::new(fs::read(&path)?);
        let mut inner = Self {
            file,
            path,
            index_path: dir.join(DESTROYED_INDEX_FILE),
            end: 0,
            master,
            secret: Zeroizing::new(Vec::new()),
            entries: HashMap::new(),
            by_subject: HashMap::new(),
            cache: HashMap::new(),
            hook,
        };
        if bytes.len() < FILE_HEADER_LEN {
            // The file is new, or a crash cut its creation short before any
            // key could exist.
            inner.file.set_len(0)?;
            inner.write_file_header()?;
            inner.add_secret()?;
            inner.checkpoint(Point::DirectorySync)?;
            sync_dir(dir)?;
        } else {
            inner.load(&bytes)?;
            if inner.secret.is_empty() {
                if !inner.entries.is_empty() {
                    return Err(
                        inner.corrupt("keys exist but the registry secret record is missing")
                    );
                }
                inner.add_secret()?;
            }
        }
        Ok(inner)
    }

    fn checkpoint(&self, point: Point) -> Result<(), CryptoError> {
        match &self.hook {
            Some(hook) => hook(point).map_err(CryptoError::Io),
            None => Ok(()),
        }
    }

    fn corrupt(&self, reason: &str) -> CryptoError {
        CryptoError::RegistryCorrupt {
            path: self.path.clone(),
            reason: reason.to_owned(),
        }
    }

    fn write_file_header(&mut self) -> Result<(), CryptoError> {
        let mut header = Vec::with_capacity(FILE_HEADER_LEN);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        self.file.write_all_at(&header, 0)?;
        self.end = FILE_HEADER_LEN as u64;
        Ok(())
    }

    /// Reads every record, rebuilding the index. A final record cut short is a
    /// torn append and is dropped; any complete record that fails its CRC is
    /// damage, wherever it sits.
    fn load(&mut self, bytes: &[u8]) -> Result<(), CryptoError> {
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(self.corrupt("the file does not start with the registry magic"));
        }
        let version = u32::from_le_bytes(
            bytes[MAGIC.len()..FILE_HEADER_LEN]
                .try_into()
                .expect("four bytes"),
        );
        if version != FORMAT_VERSION {
            return Err(self.corrupt(&format!("unsupported format version {version}")));
        }
        let mut offset = FILE_HEADER_LEN;
        while offset < bytes.len() {
            let Some(record) = self.parse_record(bytes, offset)? else {
                tracing::warn!(path = %self.path.display(), offset, "discarding a torn final registry record");
                self.file.set_len(offset as u64)?;
                self.file.sync_all()?;
                break;
            };
            self.apply(&record, offset)?;
            offset += record.total_len;
        }
        self.end = offset as u64;
        Ok(())
    }

    fn parse_record(&self, bytes: &[u8], offset: usize) -> Result<Option<Record>, CryptoError> {
        let rest = &bytes[offset..];
        if rest.len() < RECORD_PREFIX_LEN {
            return Ok(None);
        }
        let tag = rest[0];
        let header_len =
            u32::from_le_bytes(rest[1..RECORD_PREFIX_LEN].try_into().expect("four bytes")) as usize;
        if !matches!(tag, TAG_SECRET | TAG_KEY | TAG_STATE) {
            return Err(self.corrupt(&format!("unknown record tag {tag} at offset {offset}")));
        }
        if header_len > MAX_HEADER_LEN {
            return Err(self.corrupt(&format!(
                "record at offset {offset} claims a {header_len} byte header"
            )));
        }
        let slot_len = if tag == TAG_KEY { SLOT_LEN } else { 0 };
        let total_len = RECORD_PREFIX_LEN + header_len + CRC_BYTES + slot_len;
        if rest.len() < total_len {
            return Ok(None);
        }
        let covered = RECORD_PREFIX_LEN + header_len;
        let stored = u32::from_le_bytes(
            rest[covered..covered + CRC_BYTES]
                .try_into()
                .expect("four bytes"),
        );
        if crc32fast::hash(&rest[..covered]) != stored {
            return Err(self.corrupt(&format!("record at offset {offset} fails its checksum")));
        }
        Ok(Some(Record {
            tag,
            header: rest[RECORD_PREFIX_LEN..covered].to_vec(),
            slot: (slot_len > 0).then(|| rest[covered + CRC_BYTES..total_len].to_vec()),
            total_len,
        }))
    }

    fn apply(&mut self, record: &Record, offset: usize) -> Result<(), CryptoError> {
        match record.tag {
            TAG_SECRET => {
                let header: SecretHeader = decode(&record.header)?;
                self.check_master(&header.master_key_id)?;
                self.secret = self
                    .master
                    .unwrap(SECRET_AAD, &Wrapped::from_bytes(header.wrapped))?;
            }
            TAG_KEY => {
                let header: KeyHeader = decode(&record.header)?;
                self.check_master(&header.master_key_id)?;
                let slot = record.slot.as_deref().expect("a key record has a slot");
                let slot_offset =
                    (offset + RECORD_PREFIX_LEN + record.header.len() + CRC_BYTES) as u64;
                let hash: [u8; HASH_LEN] = slot[HASH_AT..].try_into().expect("a fixed-width hash");
                if hash != [0u8; HASH_LEN] {
                    self.by_subject.insert(hash, header.key_id.clone());
                }
                self.entries.insert(
                    header.key_id,
                    Entry {
                        state: KeyState::Active,
                        slot_offset,
                    },
                );
            }
            _ => {
                let header: StateHeader = decode(&record.header)?;
                let Some(entry) = self.entries.get_mut(&header.key_id) else {
                    return Err(
                        self.corrupt("a state record names a key the registry does not hold")
                    );
                };
                if entry.state == KeyState::Destroyed && header.state != KeyState::Destroyed {
                    return Err(self.corrupt("a destroyed key returns to a live state"));
                }
                entry.state = header.state;
                if header.state == KeyState::Destroyed {
                    self.by_subject.retain(|_, key_id| *key_id != header.key_id);
                }
            }
        }
        Ok(())
    }

    fn check_master(&self, found: &str) -> Result<(), CryptoError> {
        if found == self.master.master_key_id() {
            return Ok(());
        }
        Err(CryptoError::WrongMaster {
            expected: self.master.master_key_id().to_owned(),
            found: found.to_owned(),
        })
    }

    /// Finishes every destroy the sidecar index records but this copy of the
    /// registry does not show as done.
    fn reconcile_destroyed(&mut self) -> Result<(), CryptoError> {
        let listed = match fs::read_to_string(&self.index_path) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        let listed: HashSet<&str> = listed.lines().filter(|line| !line.is_empty()).collect();
        let mut pending: Vec<&str> = listed
            .into_iter()
            .filter(|key_id| {
                self.entries
                    .get(*key_id)
                    .is_some_and(|entry| entry.state != KeyState::Destroyed)
            })
            .collect();
        pending.sort_unstable();
        for key_id in pending {
            tracing::warn!(
                key = key_id,
                "finishing a destroy the registry copy did not show as done"
            );
            self.destroy(key_id)?;
        }
        Ok(())
    }

    fn state_of(&self, key_id: &str) -> Result<KeyState, CryptoError> {
        self.entries
            .get(key_id)
            .map(|entry| entry.state)
            .ok_or_else(|| CryptoError::KeyMissing {
                key_id: key_id.to_owned(),
            })
    }

    fn subject_hash(
        &self,
        scope: Option<&str>,
        subject: &str,
    ) -> Result<[u8; HASH_LEN], CryptoError> {
        let scope = match scope.map(str::trim) {
            None => NO_SCOPE_TOKEN,
            Some(NO_SCOPE_TOKEN) => {
                return Err(CryptoError::ReservedScope {
                    token: NO_SCOPE_TOKEN,
                })
            }
            Some(scope) => scope,
        };
        let mut data = Vec::with_capacity(scope.len() + 1 + subject.len());
        data.extend_from_slice(scope.as_bytes());
        data.push(0);
        data.extend_from_slice(subject.trim().to_lowercase().as_bytes());
        Ok(hmac_sha256(&self.secret, &data))
    }

    /// Appends one record and syncs it. A failed append is cut back so a
    /// later append never leaves bytes behind a shorter record.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, CryptoError> {
        let at = self.end;
        let written = self
            .file
            .write_all_at(bytes, at)
            .and_then(|()| self.file.sync_all());
        if let Err(error) = written {
            let _ = self.file.set_len(at);
            return Err(error.into());
        }
        self.end = at + bytes.len() as u64;
        Ok(at)
    }

    fn add_secret(&mut self) -> Result<(), CryptoError> {
        let secret = Zeroizing::new(random_array::<KEY_LEN>()?.to_vec());
        let wrapped = self.master.wrap(SECRET_AAD, &secret)?;
        let header = SecretHeader {
            master_key_id: self.master.master_key_id().to_owned(),
            wrapped: wrapped.as_bytes().to_vec(),
        };
        self.append(&encode_record(TAG_SECRET, &header, None)?)?;
        self.secret = secret;
        Ok(())
    }

    fn add_key(&mut self, subject_hash: Option<[u8; HASH_LEN]>) -> Result<String, CryptoError> {
        let key_id = Uuid::now_v7().to_string();
        let data = DataKey::random()?;
        let dedup = DedupKey::random()?;
        let mut slot = Vec::with_capacity(SLOT_LEN);
        slot.extend_from_slice(
            self.master
                .wrap(&slot_aad(&key_id, DATA_AAD), data.as_bytes())?
                .as_bytes(),
        );
        slot.extend_from_slice(
            self.master
                .wrap(&slot_aad(&key_id, DEDUP_AAD), dedup.as_bytes())?
                .as_bytes(),
        );
        slot.extend_from_slice(&subject_hash.unwrap_or([0u8; HASH_LEN]));
        debug_assert_eq!(slot.len(), SLOT_LEN);
        let header = KeyHeader {
            key_id: key_id.clone(),
            master_key_id: self.master.master_key_id().to_owned(),
        };
        let at = self.append(&encode_record(TAG_KEY, &header, Some(&slot))?)?;
        let slot_offset = at + (RECORD_PREFIX_LEN + postcard_len(&header)? + CRC_BYTES) as u64;
        self.entries.insert(
            key_id.clone(),
            Entry {
                state: KeyState::Active,
                slot_offset,
            },
        );
        if let Some(hash) = subject_hash {
            self.by_subject.insert(hash, key_id.clone());
        }
        self.cache_insert(&key_id, KeyMaterial { data, dedup });
        Ok(key_id)
    }

    fn cache_insert(&mut self, key_id: &str, material: KeyMaterial) {
        if self.cache.len() >= KEY_CACHE_LIMIT {
            self.cache.clear();
        }
        self.cache.insert(key_id.to_owned(), material);
    }

    fn open_key(&mut self, key_id: &str) -> Result<KeyLookup, CryptoError> {
        let Some(entry) = self.entries.get(key_id) else {
            return Ok(KeyLookup::Missing);
        };
        match entry.state {
            KeyState::Destroying => return Ok(KeyLookup::Destroying),
            KeyState::Destroyed => return Ok(KeyLookup::Destroyed),
            KeyState::Active => {}
        }
        if let Some(material) = self.cache.get(key_id) {
            return Ok(KeyLookup::Active(material.clone()));
        }
        let mut slot = [0u8; SLOT_LEN];
        self.file.read_exact_at(&mut slot, entry.slot_offset)?;
        let data = self.master.unwrap(
            &slot_aad(key_id, DATA_AAD),
            &Wrapped::from_bytes(slot[..DEDUP_AT].to_vec()),
        )?;
        let dedup = self.master.unwrap(
            &slot_aad(key_id, DEDUP_AAD),
            &Wrapped::from_bytes(slot[DEDUP_AT..HASH_AT].to_vec()),
        )?;
        let material = KeyMaterial {
            data: DataKey::from_bytes(to_key(&data)?),
            dedup: DedupKey::from_bytes(to_key(&dedup)?),
        };
        self.cache_insert(key_id, material.clone());
        Ok(KeyLookup::Active(material))
    }

    fn set_state(&mut self, key_id: &str, state: KeyState) -> Result<(), CryptoError> {
        let header = StateHeader {
            key_id: key_id.to_owned(),
            state,
        };
        self.append(&encode_record(TAG_STATE, &header, None)?)?;
        if let Some(entry) = self.entries.get_mut(key_id) {
            entry.state = state;
        }
        self.cache.remove(key_id);
        if state == KeyState::Destroying {
            self.checkpoint(Point::AfterDestroyingRecord)?;
        }
        Ok(())
    }

    /// The destroy sequence. Each step is safe to repeat, so a crash anywhere
    /// is finished by calling this again.
    fn destroy(&mut self, key_id: &str) -> Result<(), CryptoError> {
        if self.state_of(key_id)? == KeyState::Active {
            self.set_state(key_id, KeyState::Destroying)?;
        }
        self.cache.remove(key_id);
        self.record_in_index(key_id)?;
        self.checkpoint(Point::AfterIndexLine)?;
        self.wipe_slot(key_id)?;
        self.checkpoint(Point::AfterSlotOverwrite)?;
        self.set_state(key_id, KeyState::Destroyed)?;
        self.by_subject.retain(|_, mapped| mapped != key_id);
        Ok(())
    }

    /// Whether a destroy of `key_id` already reached the point of no return.
    fn listed_in_index(&self, key_id: &str) -> Result<bool, CryptoError> {
        match fs::read_to_string(&self.index_path) {
            Ok(text) => Ok(text.lines().any(|line| line == key_id)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    fn record_in_index(&self, key_id: &str) -> Result<(), CryptoError> {
        let existed = self.index_path.exists();
        let mut index = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.index_path)?;
        writeln!(index, "{key_id}")?;
        index.sync_all()?;
        if !existed {
            if let Some(dir) = self.index_path.parent() {
                sync_dir(dir)?;
            }
        }
        Ok(())
    }

    fn wipe_slot(&mut self, key_id: &str) -> Result<(), CryptoError> {
        let offset = self
            .entries
            .get(key_id)
            .map(|entry| entry.slot_offset)
            .ok_or_else(|| CryptoError::KeyMissing {
                key_id: key_id.to_owned(),
            })?;
        self.file.write_all_at(&[0u8; SLOT_LEN], offset)?;
        self.file.sync_all()?;
        Ok(())
    }
}

struct Record {
    tag: u8,
    header: Vec<u8>,
    slot: Option<Vec<u8>>,
    total_len: usize,
}

/// What a slot's wrapped keys are bound to, so one key's wrap cannot stand in
/// for another's.
fn slot_aad(key_id: &str, purpose: &[u8]) -> Vec<u8> {
    let mut aad = Vec::with_capacity(key_id.len() + 1 + purpose.len());
    aad.extend_from_slice(key_id.as_bytes());
    aad.push(0);
    aad.extend_from_slice(purpose);
    aad
}

fn encode_record<H: Serialize>(
    tag: u8,
    header: &H,
    slot: Option<&[u8]>,
) -> Result<Vec<u8>, CryptoError> {
    let header = postcard::to_stdvec(header).map_err(CryptoError::Decode)?;
    let mut bytes = Vec::with_capacity(RECORD_PREFIX_LEN + header.len() + CRC_BYTES + SLOT_LEN);
    bytes.push(tag);
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header);
    let crc = crc32fast::hash(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());
    if let Some(slot) = slot {
        bytes.extend_from_slice(slot);
    }
    Ok(bytes)
}

fn postcard_len<H: Serialize>(header: &H) -> Result<usize, CryptoError> {
    Ok(postcard::to_stdvec(header)
        .map_err(CryptoError::Decode)?
        .len())
}

fn decode<'a, H: Deserialize<'a>>(bytes: &'a [u8]) -> Result<H, CryptoError> {
    postcard::from_bytes(bytes).map_err(CryptoError::Decode)
}

fn to_key(bytes: &[u8]) -> Result<[u8; KEY_LEN], CryptoError> {
    bytes.try_into().map_err(|_| CryptoError::Tampered)
}
