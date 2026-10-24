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
    prompt_version: String,
    source_event_ids: Vec<String>,
    created_at: u64,
}

/// What `put` records beside the response.
pub struct NewEntry<'a> {
    pub key: &'a str,
    pub response: &'a str,
    pub model_id: &'a str,
    pub prompt_version: &'a str,
    pub source_event_ids: &'a [String],
}

type AppendFn = fn(&Path, &[u8]) -> io::Result<()>;
type RenameFn = fn(&Path, &Path) -> io::Result<()>;

pub struct CacheStore {
    path: PathBuf,
    entries: Mutex<HashMap<String, Entry>>,
    append: AppendFn,
    rename: RenameFn,
}

impl CacheStore {
    /// Loads the file at `path`, or starts empty when there is none.
    pub fn open(path: PathBuf) -> io::Result<Self> {
        Self::open_with(path, append_line, rename_file)
    }

    /// `open` with the writes swapped out, so tests can force their failures.
    fn open_with(path: PathBuf, append: AppendFn, rename: RenameFn) -> io::Result<Self> {
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error),
        };
        let entries = parse_lines(&bytes);
        // A torn last line has no terminator; without one the next append
        // would glue itself onto the fragment.
        if bytes.last().is_some_and(|last| *last != LINE_END) {
            append(&path, &[LINE_END])?;
        }
        debug!(entries = entries.len(), "extraction cache loaded");
        Ok(Self {
            path,
            entries: Mutex::new(entries),
            append,
            rename,
        })
    }

    pub fn get(&self, key: &str) -> Option<String> {
        self.lock().get(key).map(|entry| entry.response.clone())
    }

    /// Appends the entry and syncs it. The in-memory copy changes only after
    /// the write lands, so a failed write leaves a clean miss.
    pub fn put(&self, new: NewEntry<'_>) -> io::Result<()> {
        let entry = Entry {
            key: new.key.to_owned(),
            response: new.response.to_owned(),
            model_id: new.model_id.to_owned(),
            prompt_version: new.prompt_version.to_owned(),
            source_event_ids: new.source_event_ids.to_vec(),
            created_at: unix_now(),
        };
        let mut line = serde_json::to_vec(&entry).map_err(io::Error::other)?;
        line.push(LINE_END);
        let mut entries = self.lock();
        (self.append)(&self.path, &line)?;
        entries.insert(entry.key.clone(), entry);
        Ok(())
    }

    /// Drops every entry that cites one of `event_ids` and returns how many
    /// went. The file is rewritten beside itself and renamed over the old one,
    /// so a crash leaves either the old file or the new one.
    pub fn remove_by_event_ids(&self, event_ids: &HashSet<String>) -> io::Result<usize> {
        let mut entries = self.lock();
        let (removed, mut kept): (Vec<_>, Vec<_>) = entries.values().cloned().partition(|entry| {
            entry
                .source_event_ids
                .iter()
                .any(|id| event_ids.contains(id))
        });
        if removed.is_empty() {
            return Ok(0);
        }
        kept.sort_by(|a, b| (a.created_at, &a.key).cmp(&(b.created_at, &b.key)));
        let mut bytes = Vec::new();
        for entry in &kept {
            serde_json::to_writer(&mut bytes, entry).map_err(io::Error::other)?;
            bytes.push(LINE_END);
        }
        let temp = temp_path(&self.path);
        let swapped = write_new_file(&temp, &bytes).and_then(|()| (self.rename)(&temp, &self.path));
        if let Err(error) = swapped {
            let _ = fs::remove_file(&temp);
            return Err(error);
        }
        *entries = kept
            .into_iter()
            .map(|entry| (entry.key.clone(), entry))
            .collect();
        Ok(removed.len())
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<String, Entry>> {
        // The map is only ever replaced or inserted into whole, so a panic in
        // another holder cannot leave it half-written.
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
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
    /// returned but not kept, so a truncated answer is retried, not replayed.
    pub fn new(
        inner: Arc<dyn JsonExtractor>,
        store: Arc<CacheStore>,
        prompt_version: String,
        validator: fn(&str) -> bool,
    ) -> Self {
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
            return Ok(response);
        }
        let response = self.inner.extract_json(req).await?;
        if (self.validator)(&response) {
            let stored = self.store.put(NewEntry {
                key: &key,
                response: &response,
                model_id: &identity.model_id,
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
        let entry = Entry {
            key: key.into(),
            response: response.into(),
            model_id: "model-a".into(),
            prompt_version: PROMPT_VERSION.into(),
            source_event_ids: events.iter().map(|id| (*id).into()).collect(),
            created_at: 1,
        };
        format!("{}\n", serde_json::to_string(&entry).unwrap())
    }

    fn put(store: &CacheStore, key: &str, response: &str, events: &[&str]) {
        let ids: Vec<String> = events.iter().map(|id| (*id).into()).collect();
        store
            .put(NewEntry {
                key,
                response,
                model_id: "model-a",
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
        let failing: AppendFn = |_, _| Err(io::Error::other("disk full"));
        let store =
            Arc::new(CacheStore::open_with(cache_file(&dir), failing, rename_file).unwrap());
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
        let failing: RenameFn = |_, _| Err(io::Error::other("rename refused"));
        let store = CacheStore::open_with(path.clone(), append_line, failing).unwrap();
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
}
