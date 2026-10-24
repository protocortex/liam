// SPDX-License-Identifier: Apache-2.0
//! A response cache for the model, kept in a file beside the database so a
//! snapshot restore or a rebuild neither reverts nor deletes it.
//!
//! Each line holds extracted user text, so the file is created readable by its
//! owner only, and nothing here logs a response or a prompt.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{debug, warn};

use super::client::{ExtractError, ExtractRequest, JsonExtractor, ModelIdentity};

const FILE_SUFFIX: &str = ".extraction-cache.jsonl";
const TEMP_SUFFIX: &str = ".tmp";
const OWNER_ONLY: u32 = 0o600;
const LINE_END: u8 = b'\n';
/// A size cap the file cannot outgrow for long: far above a laptop's extraction history.
const MAX_ENTRIES: usize = 20_000;
/// How far past the cap the file may run before one rewrite trims it back.
const PRUNE_SLACK: usize = 500;

/// Where the cache for the database at `database_path` lives.
pub fn cache_path(database_path: &Path) -> PathBuf {
    let mut name = database_path.as_os_str().to_owned();
    name.push(FILE_SUFFIX);
    PathBuf::from(name)
}

/// Everything that changes what the model would answer.
pub struct KeyParts<'a> {
    pub prompt_version: &'a str,
    pub identity: &'a ModelIdentity,
    pub request: &'a ExtractRequest,
}

/// Hashes the parts with a length in front of each, so `("ab", "c")` and
/// `("a", "bc")` cannot collide. The schema hash covers the configured
/// vocabulary.
pub fn cache_key(parts: &KeyParts<'_>) -> String {
    let request = parts.request;
    let mut hasher = Sha256::new();
    let mut feed = |bytes: &[u8]| {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    feed(parts.prompt_version.as_bytes());
    feed(parts.identity.model_id.as_bytes());
    feed(parts.identity.model_sha256.as_bytes());
    feed(request.schema_hash.as_bytes());
    feed(&request.temperature.to_bits().to_le_bytes());
    feed(&request.seed.to_le_bytes());
    feed(&request.max_tokens.to_le_bytes());
    feed(request.system.as_bytes());
    feed(request.prompt.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    key: String,
    response: String,
    model_id: String,
    // Absent in no shipped file; the default only keeps an old line parseable.
    #[serde(default)]
    model_sha256: String,
    prompt_version: String,
    source_event_ids: Vec<String>,
    created_at: u64,
}

/// What `put` records beside the response.
pub struct NewEntry<'a> {
    pub key: &'a str,
    pub response: &'a str,
    pub model_id: &'a str,
    pub model_sha256: &'a str,
    pub prompt_version: &'a str,
    pub source_event_ids: &'a [String],
}

type AppendFn = fn(&Path, &[u8]) -> io::Result<()>;
type RenameFn = fn(&Path, &Path) -> io::Result<()>;
type SyncDirFn = fn(&Path) -> io::Result<()>;

/// The file writes, swappable so tests can force their failures.
#[derive(Clone, Copy)]
struct FileOps {
    append: AppendFn,
    rename: RenameFn,
    sync_dir: SyncDirFn,
}

impl FileOps {
    const REAL: Self = Self {
        append: append_line,
        rename: rename_file,
        sync_dir: sync_parent_dir,
    };
}

struct State {
    entries: HashMap<String, Entry>,
    /// A failed append can leave half a line; the next append starts a new one.
    torn: bool,
}

pub struct CacheStore {
    path: PathBuf,
    state: Mutex<State>,
    ops: FileOps,
    max_entries: usize,
}

impl CacheStore {
    /// Loads the file at `path`, or starts empty when there is none.
    pub fn open(path: PathBuf) -> io::Result<Self> {
        Self::open_with(path, FileOps::REAL, MAX_ENTRIES)
    }

    fn open_with(path: PathBuf, ops: FileOps, max_entries: usize) -> io::Result<Self> {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error),
        };
        let entries = parse_lines(&bytes);
        debug!(entries = entries.len(), "extraction cache loaded");
        Ok(Self {
            path,
            state: Mutex::new(State {
                entries,
                // A torn last line has no terminator; the first append closes it.
                torn: bytes.last().is_some_and(|last| *last != LINE_END),
            }),
            ops,
            max_entries,
        })
    }

    pub fn get(&self, key: &str) -> Option<String> {
        self.lock()
            .entries
            .get(key)
            .map(|entry| entry.response.clone())
    }

    /// Appends the entry and syncs it, replacing any entry under the same key.
    /// The in-memory copy changes only after the write lands, so a failed
    /// write leaves a clean miss. Past the size cap the oldest entries go.
    pub fn put(&self, new: NewEntry<'_>) -> io::Result<()> {
        let entry = Entry {
            key: new.key.to_owned(),
            response: new.response.to_owned(),
            model_id: new.model_id.to_owned(),
            model_sha256: new.model_sha256.to_owned(),
            prompt_version: new.prompt_version.to_owned(),
            source_event_ids: new.source_event_ids.to_vec(),
            created_at: unix_now(),
        };
        let mut line = Vec::new();
        let mut state = self.lock();
        if state.torn {
            line.push(LINE_END);
        }
        serde_json::to_writer(&mut line, &entry).map_err(io::Error::other)?;
        line.push(LINE_END);
        if let Err(error) = (self.ops.append)(&self.path, &line) {
            state.torn = true;
            return Err(error);
        }
        state.torn = false;
        state.entries.insert(entry.key.clone(), entry);
        if state.entries.len() > self.max_entries + PRUNE_SLACK {
            let kept = newest(&state.entries, self.max_entries);
            self.rewrite(&mut state, kept)?;
        }
        Ok(())
    }

    /// Drops every entry that cites one of `event_ids` and returns how many
    /// went.
    pub fn remove_by_event_ids(&self, event_ids: &HashSet<String>) -> io::Result<usize> {
        let mut state = self.lock();
        let before = state.entries.len();
        let kept: Vec<Entry> = state
            .entries
            .values()
            .filter(|entry| {
                !entry
                    .source_event_ids
                    .iter()
                    .any(|id| event_ids.contains(id))
            })
            .cloned()
            .collect();
        let removed = before - kept.len();
        if removed > 0 {
            self.rewrite(&mut state, kept)?;
        }
        Ok(removed)
    }

    /// Drops the entries a different model or prompt wrote, then the oldest
    /// ones past the size cap, and returns how many went. They can never be
    /// hit again, so keeping them only grows the file.
    pub fn retain_current(
        &self,
        model_id: &str,
        model_sha256: &str,
        prompt_version: &str,
    ) -> io::Result<usize> {
        let mut state = self.lock();
        let before = state.entries.len();
        let current: HashMap<String, Entry> = state
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.model_id == model_id
                    && entry.model_sha256 == model_sha256
                    && entry.prompt_version == prompt_version
            })
            .map(|(key, entry)| (key.clone(), entry.clone()))
            .collect();
        let kept = newest(&current, self.max_entries);
        let removed = before - kept.len();
        if removed > 0 {
            self.rewrite(&mut state, kept)?;
        }
        Ok(removed)
    }

    /// Replaces the file with `kept` through a temp file and a rename, so a
    /// crash leaves the old file or the new one, then syncs the directory so
    /// the rename itself survives a power cut.
    fn rewrite(&self, state: &mut State, mut kept: Vec<Entry>) -> io::Result<()> {
        kept.sort_by(|a, b| (a.created_at, &a.key).cmp(&(b.created_at, &b.key)));
        let mut bytes = Vec::new();
        for entry in &kept {
            serde_json::to_writer(&mut bytes, entry).map_err(io::Error::other)?;
            bytes.push(LINE_END);
        }
        let temp = temp_path(&self.path);
        let swapped =
            write_new_file(&temp, &bytes).and_then(|()| (self.ops.rename)(&temp, &self.path));
        if let Err(error) = swapped {
            let _ = fs::remove_file(&temp);
            return Err(error);
        }
        state.entries = kept
            .into_iter()
            .map(|entry| (entry.key.clone(), entry))
            .collect();
        state.torn = false;
        (self.ops.sync_dir)(&self.path)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // Each write replaces or extends the state whole, so a panic in
        // another holder cannot leave it half-written.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The newest `limit` entries; the rest are the oldest and go first.
fn newest(entries: &HashMap<String, Entry>, limit: usize) -> Vec<Entry> {
    let mut all: Vec<Entry> = entries.values().cloned().collect();
    all.sort_by(|a, b| (b.created_at, &b.key).cmp(&(a.created_at, &a.key)));
    all.truncate(limit);
    all
}

/// Later lines win. A blank, torn or corrupt line is skipped.
fn parse_lines(bytes: &[u8]) -> HashMap<String, Entry> {
    let mut entries = HashMap::new();
    for (number, line) in bytes.split(|byte| *byte == LINE_END).enumerate() {
        if line.is_empty() {
            continue;
        }
        match serde_json::from_slice::<Entry>(line) {
            Ok(entry) => {
                entries.insert(entry.key.clone(), entry);
            }
            Err(error) => warn!(line = number + 1, %error, "skipping an unreadable cache line"),
        }
    }
    entries
}

fn append_line(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(OWNER_ONLY)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn rename_file(from: &Path, to: &Path) -> io::Result<()> {
    fs::rename(from, to)
}

fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let dir = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::File::open(dir)?.sync_all()
}

fn write_new_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(OWNER_ONLY)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(TEMP_SUFFIX);
    PathBuf::from(name)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Answers from the cache when it can and stores what the model says when the
/// answer is usable. A cache failure never fails the call.
pub struct CachedExtractor {
    inner: Arc<dyn JsonExtractor>,
    store: Arc<CacheStore>,
    prompt_version: String,
    validator: fn(&str) -> bool,
}

impl CachedExtractor {
    /// `validator` is the response parse check: a reply that fails it is
    /// returned but not kept, and a stored reply that fails it is a miss, so a
    /// truncated answer is retried, not replayed. Entries from another model
    /// or prompt are pruned here.
    pub fn new(
        inner: Arc<dyn JsonExtractor>,
        store: Arc<CacheStore>,
        prompt_version: String,
        validator: fn(&str) -> bool,
    ) -> Self {
        let identity = inner.identity();
        match store.retain_current(&identity.model_id, &identity.model_sha256, &prompt_version) {
            Ok(0) => {}
            Ok(removed) => debug!(removed, "pruned stale extraction cache entries"),
            Err(error) => warn!(kind = ?error.kind(), "could not prune the extraction cache"),
        }
        Self {
            inner,
            store,
            prompt_version,
            validator,
        }
    }
}

#[async_trait]
impl JsonExtractor for CachedExtractor {
    async fn extract_json(&self, req: &ExtractRequest) -> Result<String, ExtractError> {
        let identity = self.inner.identity();
        let key = cache_key(&KeyParts {
            prompt_version: &self.prompt_version,
            identity,
            request: req,
        });
        if let Some(response) = self.store.get(&key) {
            if (self.validator)(&response) {
                return Ok(response);
            }
            warn!("a stored extraction response failed validation, asking the model again");
        }
        let response = self.inner.extract_json(req).await?;
        if (self.validator)(&response) {
            let stored = self.store.put(NewEntry {
                key: &key,
                response: &response,
                model_id: &identity.model_id,
                model_sha256: &identity.model_sha256,
                prompt_version: &self.prompt_version,
                source_event_ids: &req.source_event_ids,
            });
            if let Err(error) = stored {
                warn!(kind = ?error.kind(), "could not store an extraction response");
            }
        }
        Ok(response)
    }

    fn identity(&self) -> &ModelIdentity {
        self.inner.identity()
    }

    async fn health(&self) -> bool {
        self.inner.health().await
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicBool, Ordering};

    use serde_json::json;
    use tempfile::TempDir;

    use super::super::client::MockExtractor;
    use super::*;

    const PROMPT_VERSION: &str = "v1";
    const LARGE_RESPONSE_BYTES: usize = 5 * 1024 * 1024;
    const VALID: &str = r#"{"facts":[]}"#;

    fn identity() -> ModelIdentity {
        ModelIdentity {
            model_id: "model-a".into(),
            model_sha256: "sha-a".into(),
        }
    }

    fn request() -> ExtractRequest {
        ExtractRequest {
            system: "system".into(),
            prompt: "prompt".into(),
            schema: json!({}),
            source_event_ids: vec!["e1".into()],
            schema_hash: "schema-a".into(),
            temperature: 0.0,
            seed: 7,
            max_tokens: 256,
        }
    }

    fn key_of(version: &str, identity: &ModelIdentity, request: &ExtractRequest) -> String {
        cache_key(&KeyParts {
            prompt_version: version,
            identity,
            request,
        })
    }

    fn accepts_json(text: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(text).is_ok()
    }

    fn cache_file(dir: &TempDir) -> PathBuf {
        cache_path(&dir.path().join("liam.db"))
    }

    fn store_at(path: &Path) -> Arc<CacheStore> {
        Arc::new(CacheStore::open(path.to_path_buf()).unwrap())
    }

    fn cached(
        script: Vec<Result<String, ExtractError>>,
        store: Arc<CacheStore>,
    ) -> (CachedExtractor, Arc<MockExtractor>) {
        let mock = Arc::new(MockExtractor::new(identity(), script));
        let extractor =
            CachedExtractor::new(mock.clone(), store, PROMPT_VERSION.into(), accepts_json);
        (extractor, mock)
    }

    fn entry_line(key: &str, response: &str, events: &[&str]) -> String {
        aged_line(key, response, events, 1)
    }

    fn aged_line(key: &str, response: &str, events: &[&str], created_at: u64) -> String {
        line_for(
            key,
            response,
            events,
            created_at,
            ("model-a", "sha-a", PROMPT_VERSION),
        )
    }

    fn line_for(
        key: &str,
        response: &str,
        events: &[&str],
        created_at: u64,
        (model_id, model_sha256, prompt_version): (&str, &str, &str),
    ) -> String {
        let entry = Entry {
            key: key.into(),
            response: response.into(),
            model_id: model_id.into(),
            model_sha256: model_sha256.into(),
            prompt_version: prompt_version.into(),
            source_event_ids: events.iter().map(|id| (*id).into()).collect(),
            created_at,
        };
        format!("{}\n", serde_json::to_string(&entry).unwrap())
    }

    fn store_with(path: &Path, ops: FileOps, max_entries: usize) -> CacheStore {
        CacheStore::open_with(path.to_path_buf(), ops, max_entries).unwrap()
    }

    fn put(store: &CacheStore, key: &str, response: &str, events: &[&str]) {
        let ids: Vec<String> = events.iter().map(|id| (*id).into()).collect();
        store
            .put(NewEntry {
                key,
                response,
                model_id: "model-a",
                model_sha256: "sha-a",
                prompt_version: PROMPT_VERSION,
                source_event_ids: &ids,
            })
            .unwrap();
    }

    fn ids(values: &[&str]) -> HashSet<String> {
        values.iter().map(|id| (*id).into()).collect()
    }

    #[tokio::test]
    async fn the_same_input_is_answered_without_the_model() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let store = store_at(&cache_file(&dir));
        let (extractor, mock) = cached(vec![Ok(VALID.into())], store.clone());
        let req = request();
        // Act
        let first = extractor.extract_json(&req).await.unwrap();
        let second = extractor.extract_json(&req).await.unwrap();
        // Assert
        assert_eq!((first.as_str(), second.as_str()), (VALID, VALID));
        assert_eq!(mock.calls(), 1);
        assert_eq!(
            fs::read_to_string(cache_file(&dir))
                .unwrap()
                .lines()
                .count(),
            1
        );
    }

    #[test]
    fn each_changed_input_gives_another_key() {
        // Arrange
        let base = key_of(PROMPT_VERSION, &identity(), &request());
        type Change = fn(&mut String, &mut ModelIdentity, &mut ExtractRequest);
        let rows: [(&str, Change); 10] = [
            ("prompt version", |v, _, _| *v = "v2".into()),
            ("model id", |_, i, _| i.model_id = "model-b".into()),
            ("model sha256", |_, i, _| i.model_sha256 = "sha-b".into()),
            ("schema hash", |_, _, r| r.schema_hash = "schema-b".into()),
            ("seed", |_, _, r| r.seed = 8),
            ("temperature", |_, _, r| r.temperature = 0.5),
            ("max tokens", |_, _, r| r.max_tokens = 512),
            ("system", |_, _, r| r.system = "other".into()),
            ("prompt", |_, _, r| r.prompt = "other".into()),
            ("nothing", |_, _, _| {}),
        ];
        for (name, change) in rows {
            let (mut version, mut who, mut req) =
                (PROMPT_VERSION.to_owned(), identity(), request());
            // Act
            change(&mut version, &mut who, &mut req);
            let key = key_of(&version, &who, &req);
            // Assert
            assert_eq!(key == base, name == "nothing", "{name}");
        }
    }

    #[test]
    fn a_boundary_shift_between_parts_changes_the_key() {
        // Arrange
        let mut shifted = request();
        shifted.system = "ab".into();
        shifted.prompt = "c".into();
        let mut other = request();
        other.system = "a".into();
        other.prompt = "bc".into();
        // Act
        let keys = (
            key_of(PROMPT_VERSION, &identity(), &shifted),
            key_of(PROMPT_VERSION, &identity(), &other),
        );
        // Assert
        assert_ne!(keys.0, keys.1);
    }

    #[tokio::test]
    async fn a_rejected_response_is_returned_but_not_kept() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let (extractor, mock) = cached(
            vec![Ok("{\"facts\":[".into()), Ok(VALID.into())],
            store_at(&cache_file(&dir)),
        );
        let req = request();
        // Act
        let first = extractor.extract_json(&req).await.unwrap();
        let second = extractor.extract_json(&req).await.unwrap();
        // Assert
        assert_eq!((first.as_str(), second.as_str()), ("{\"facts\":[", VALID));
        assert_eq!(mock.calls(), 2);
    }

    #[tokio::test]
    async fn an_inner_error_passes_through_and_stores_nothing() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let (extractor, _) = cached(
            vec![Err(ExtractError::Timeout)],
            store_at(&cache_file(&dir)),
        );
        // Act
        let result = extractor.extract_json(&request()).await;
        // Assert
        assert_eq!(result, Err(ExtractError::Timeout));
        assert!(!cache_file(&dir).exists());
    }

    #[test]
    fn an_empty_and_a_large_response_round_trip() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let large = "x".repeat(LARGE_RESPONSE_BYTES);
        let store = store_at(&path);
        // Act
        put(&store, "empty", "", &[]);
        put(&store, "large", &large, &[]);
        let reopened = store_at(&path);
        // Assert
        assert_eq!(reopened.get("empty"), Some(String::new()));
        assert_eq!(reopened.get("large"), Some(large));
        assert_eq!(reopened.get("absent"), None);
    }

    #[test]
    fn the_later_of_two_duplicate_keys_wins() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let lines = entry_line("k", "old", &[]) + &entry_line("k", "new", &[]);
        fs::write(&path, lines).unwrap();
        // Act
        let store = store_at(&path);
        // Assert
        assert_eq!(store.get("k"), Some("new".into()));
    }

    #[test]
    fn damaged_lines_are_skipped_and_the_next_put_works() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let mut bytes = Vec::new();
        bytes.extend(entry_line("first", "a", &[]).bytes());
        bytes.extend(b"\n{not json}\n");
        bytes.extend(entry_line("last", "b", &[]).bytes());
        // The tail is cut inside the two-byte "é".
        let torn = entry_line("torn", "é", &[]);
        let cut = torn.find('é').unwrap() + 1;
        bytes.extend(&torn.as_bytes()[..cut]);
        fs::write(&path, bytes).unwrap();
        // Act
        let store = store_at(&path);
        put(&store, "after", "c", &[]);
        let reopened = store_at(&path);
        // Assert
        let found: Vec<_> = ["first", "last", "torn", "after"]
            .iter()
            .map(|key| reopened.get(key))
            .collect();
        assert_eq!(
            found,
            [Some("a".into()), Some("b".into()), None, Some("c".into())]
        );
    }

    #[test]
    fn a_file_with_only_a_torn_line_opens_empty() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        fs::write(&path, b"{\"key\":\"k\",\"resp").unwrap();
        // Act
        let store = store_at(&path);
        put(&store, "k", "v", &[]);
        // Assert
        assert_eq!(store_at(&path).get("k"), Some("v".into()));
    }

    #[test]
    fn the_file_is_owner_only() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        // Act
        put(&store_at(&path), "k", "v", &[]);
        // Assert
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, OWNER_ONLY);
    }

    #[tokio::test]
    async fn the_cache_outlives_the_database_file() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let database = dir.path().join("liam.db");
        fs::write(&database, b"db").unwrap();
        let (extractor, _) = cached(vec![Ok(VALID.into())], store_at(&cache_path(&database)));
        extractor.extract_json(&request()).await.unwrap();
        // Act
        fs::remove_file(&database).unwrap();
        let (replay, mock) = cached(vec![], store_at(&cache_path(&database)));
        let response = replay.extract_json(&request()).await.unwrap();
        // Assert
        assert_eq!((response.as_str(), mock.calls()), (VALID, 0));
    }

    #[tokio::test]
    async fn a_failed_append_still_returns_the_model_answer() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let ops = FileOps {
            append: |_, _| Err(io::Error::other("disk full")),
            ..FileOps::REAL
        };
        let store = Arc::new(store_with(&cache_file(&dir), ops, MAX_ENTRIES));
        let (extractor, mock) = cached(vec![Ok(VALID.into()), Ok(VALID.into())], store);
        let req = request();
        // Act
        let first = extractor.extract_json(&req).await.unwrap();
        let second = extractor.extract_json(&req).await.unwrap();
        // Assert
        assert_eq!((first.as_str(), second.as_str()), (VALID, VALID));
        assert_eq!(mock.calls(), 2);
    }

    #[test]
    fn removing_by_event_id_drops_only_matching_entries() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let store = store_at(&path);
        put(&store, "a", "1", &["e1", "e2"]);
        put(&store, "b", "2", &["e3"]);
        put(&store, "c", "3", &["e2"]);
        // Act
        let removed = store.remove_by_event_ids(&ids(&["e2"])).unwrap();
        // Assert
        assert_eq!(removed, 2);
        assert_eq!(store.get("b"), Some("2".into()));
        let reopened = store_at(&path);
        assert_eq!((reopened.get("a"), reopened.get("c")), (None, None));
        assert_eq!(reopened.get("b"), Some("2".into()));
        assert!(!temp_path(&path).exists());
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, OWNER_ONLY);
    }

    #[test]
    fn a_failed_rename_keeps_the_original_file() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let ops = FileOps {
            rename: |_, _| Err(io::Error::other("rename refused")),
            ..FileOps::REAL
        };
        let store = store_with(&path, ops, MAX_ENTRIES);
        put(&store, "a", "1", &["e1"]);
        let before = fs::read(&path).unwrap();
        // Act
        let result = store.remove_by_event_ids(&ids(&["e1"]));
        // Assert
        assert!(result.is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
        assert!(!temp_path(&path).exists());
        assert_eq!(store.get("a"), Some("1".into()));
        assert_eq!(store_at(&path).get("a"), Some("1".into()));
    }

    #[test]
    fn a_leftover_temp_file_is_not_read_as_the_cache() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        fs::write(&path, entry_line("real", "r", &[])).unwrap();
        fs::write(temp_path(&path), entry_line("stray", "s", &[])).unwrap();
        // Act
        let store = store_at(&path);
        // Assert
        assert_eq!(
            (store.get("real"), store.get("stray")),
            (Some("r".into()), None)
        );
    }

    #[tokio::test]
    async fn a_replayed_run_makes_no_model_calls() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let requests: Vec<ExtractRequest> = (0..5)
            .map(|n| ExtractRequest {
                prompt: format!("batch {n}"),
                ..request()
            })
            .collect();
        let script = (0..5)
            .map(|n| Ok(format!(r#"{{"facts":[],"n":{n}}}"#)))
            .collect();
        let (first, _) = cached(script, store_at(&path));
        let mut expected = Vec::new();
        for req in &requests {
            expected.push(first.extract_json(req).await.unwrap());
        }
        let (replay, mock) = cached(vec![], store_at(&path));
        // Act
        let mut replayed = Vec::new();
        for req in &requests {
            replayed.push(replay.extract_json(req).await.unwrap());
        }
        // Assert
        assert_eq!(replayed, expected);
        assert_eq!(mock.calls(), 0);
    }

    #[tokio::test]
    async fn concurrent_calls_all_land_in_the_file() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let store = store_at(&path);
        // Act
        let writers: Vec<_> = (0..16)
            .map(|n| {
                let store = store.clone();
                tokio::task::spawn_blocking(move || put(&store, &format!("k{n}"), "v", &["e"]))
            })
            .collect();
        for writer in writers {
            writer.await.unwrap();
        }
        // Assert
        let reopened = store_at(&path);
        assert!((0..16).all(|n| reopened.get(&format!("k{n}")).is_some()));
    }

    #[test]
    fn the_key_is_sixty_four_hex_characters() {
        // Arrange
        let req = request();
        // Act
        let key = key_of(PROMPT_VERSION, &identity(), &req);
        // Assert
        assert_eq!(key.len(), 64);
        assert!(key.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn a_put_replaces_an_existing_key() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let store = store_at(&path);
        put(&store, "k", "old", &[]);
        // Act
        put(&store, "k", "new", &[]);
        // Assert
        assert_eq!(store.get("k"), Some("new".into()));
        assert_eq!(store_at(&path).get("k"), Some("new".into()));
    }

    #[tokio::test]
    async fn another_prompt_version_or_model_asks_the_model_again() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let (first, _) = cached(vec![Ok(VALID.into())], store_at(&path));
        first.extract_json(&request()).await.unwrap();
        let other_model = ModelIdentity {
            model_id: "model-b".into(),
            ..identity()
        };
        let versions_and_identities = [("v2", identity()), (PROMPT_VERSION, other_model)];
        for (version, who) in versions_and_identities {
            let mock = Arc::new(MockExtractor::new(who, vec![Ok(VALID.into())]));
            let extractor =
                CachedExtractor::new(mock.clone(), store_at(&path), version.into(), accepts_json);
            // Act
            extractor.extract_json(&request()).await.unwrap();
            // Assert
            assert_eq!(mock.calls(), 1, "{version}");
        }
    }

    #[tokio::test]
    async fn a_stored_response_that_fails_validation_is_replaced() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let req = request();
        let key = key_of(PROMPT_VERSION, &identity(), &req);
        fs::write(&path, entry_line(&key, "{\"facts\":[", &[])).unwrap();
        let (extractor, mock) = cached(vec![Ok(VALID.into())], store_at(&path));
        // Act
        let response = extractor.extract_json(&req).await.unwrap();
        let (replay, replay_mock) = cached(vec![], store_at(&path));
        let replayed = replay.extract_json(&req).await.unwrap();
        // Assert
        assert_eq!((response.as_str(), mock.calls()), (VALID, 1));
        assert_eq!((replayed.as_str(), replay_mock.calls()), (VALID, 0));
    }

    #[test]
    fn retain_current_drops_entries_from_another_model_or_prompt() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let current = ("model-a", "sha-a", PROMPT_VERSION);
        let lines = line_for("keep", "1", &[], 1, current)
            + &line_for("model", "2", &[], 1, ("model-b", "sha-a", PROMPT_VERSION))
            + &line_for("sha", "3", &[], 1, ("model-a", "sha-b", PROMPT_VERSION))
            + &line_for("prompt", "4", &[], 1, ("model-a", "sha-a", "v0"));
        fs::write(&path, lines).unwrap();
        let store = store_at(&path);
        // Act
        let removed = store
            .retain_current("model-a", "sha-a", PROMPT_VERSION)
            .unwrap();
        // Assert
        assert_eq!(removed, 3);
        assert_eq!(store_at(&path).get("keep"), Some("1".into()));
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 1);
    }

    #[test]
    fn a_new_extractor_prunes_entries_of_an_older_prompt() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let old = line_for("old", "1", &[], 1, ("model-a", "sha-a", "v0"));
        fs::write(&path, old).unwrap();
        // Act
        let (_extractor, _) = cached(vec![], store_at(&path));
        // Assert
        assert_eq!(fs::read_to_string(&path).unwrap(), "");
    }

    #[test]
    fn pruning_past_the_cap_drops_the_oldest_first() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let lines: String = (1..=6)
            .map(|n| aged_line(&format!("k{n}"), "v", &[], n))
            .collect();
        fs::write(&path, lines).unwrap();
        let store = store_with(&path, FileOps::REAL, 3);
        // Act
        let removed = store
            .retain_current("model-a", "sha-a", PROMPT_VERSION)
            .unwrap();
        // Assert
        assert_eq!(removed, 3);
        let reopened = store_at(&path);
        let present: Vec<bool> = (1..=6)
            .map(|n| reopened.get(&format!("k{n}")).is_some())
            .collect();
        assert_eq!(present, [false, false, false, true, true, true]);
    }

    #[test]
    fn a_put_past_the_cap_trims_the_file() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let cap = 3;
        let lines: String = (1..=(cap + PRUNE_SLACK) as u64)
            .map(|n| aged_line(&format!("k{n}"), "v", &[], n))
            .collect();
        fs::write(&path, lines).unwrap();
        let store = store_with(&path, FileOps::REAL, cap);
        // Act
        put(&store, "fresh", "v", &[]);
        // Assert
        assert!(store_at(&path).get("fresh").is_some());
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), cap);
    }

    #[test]
    fn the_directory_is_synced_after_the_rename() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let ops = FileOps {
            sync_dir: |_| Err(io::Error::other("dir sync refused")),
            ..FileOps::REAL
        };
        let store = store_with(&path, ops, MAX_ENTRIES);
        put(&store, "a", "1", &["e1"]);
        // Act
        let result = store.remove_by_event_ids(&ids(&["e1"]));
        // Assert
        assert!(result.is_err());
        assert_eq!(store_at(&path).get("a"), None);
    }

    static SHORT_WRITE_PENDING: AtomicBool = AtomicBool::new(false);

    fn short_write_once(path: &Path, bytes: &[u8]) -> io::Result<()> {
        if SHORT_WRITE_PENDING.swap(false, Ordering::SeqCst) {
            append_line(path, &bytes[..bytes.len() / 2])?;
            return Err(io::Error::other("short write"));
        }
        append_line(path, bytes)
    }

    #[test]
    fn an_append_after_a_short_write_starts_a_new_line() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let ops = FileOps {
            append: short_write_once,
            ..FileOps::REAL
        };
        let store = store_with(&path, ops, MAX_ENTRIES);
        SHORT_WRITE_PENDING.store(true, Ordering::SeqCst);
        let no_events: Vec<String> = Vec::new();
        let entry = |key| NewEntry {
            key,
            response: "v",
            model_id: "model-a",
            model_sha256: "sha-a",
            prompt_version: PROMPT_VERSION,
            source_event_ids: &no_events,
        };
        // Act
        let failed = store.put(entry("torn"));
        store.put(entry("after")).unwrap();
        // Assert
        assert!(failed.is_err());
        let reopened = store_at(&path);
        assert_eq!(
            (reopened.get("torn"), reopened.get("after")),
            (None, Some("v".into()))
        );
    }

    #[test]
    fn a_put_during_a_removal_is_not_lost() {
        // Arrange
        let dir = TempDir::new().unwrap();
        let path = cache_file(&dir);
        let store = store_at(&path);
        for n in 0..20 {
            put(&store, &format!("gone{n}"), "v", &["gone"]);
        }
        // Act
        let writers: Vec<_> = (0..8)
            .map(|n| {
                let store = store.clone();
                std::thread::spawn(move || put(&store, &format!("keep{n}"), "v", &["keep"]))
            })
            .collect();
        let removed = store.remove_by_event_ids(&ids(&["gone"])).unwrap();
        for writer in writers {
            writer.join().unwrap();
        }
        // Assert
        assert_eq!(removed, 20);
        let reopened = store_at(&path);
        assert!((0..8).all(|n| reopened.get(&format!("keep{n}")).is_some()));
        assert!((0..20).all(|n| reopened.get(&format!("gone{n}")).is_none()));
    }
}
