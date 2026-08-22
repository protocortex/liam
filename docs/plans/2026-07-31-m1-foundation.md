# M1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable `Llm` trait (mock + candle-local) and an open entity dimension (conventions over existing free-form columns) as the foundation later GBrain-aligned milestones consume.

**Architecture:** Mirror the existing `Embedder`/`Reranker` abstraction for a new generative `Llm` trait in `liam-model`. Express the "entity dimension" as conventions over the already free-form `kind` (nodes) and `type` (edges) columns plus the existing `subject`/`scope` dedup, so no schema migration is needed. Wire an `Arc<dyn Llm>` into the daemon now; first consumer is M2.

**Tech Stack:** Rust (Cargo workspace), `async-trait`, `thiserror`, candle (`candle-transformers`/`tokenizers`/`hf-hub`) behind the `local` feature, `tokio`, `serde`/`toml`, `rmcp`.

## Global Constraints

- **Baseline:** implement against tag `v0.1.0`. Do not regress existing tests (`cargo test --workspace` is green at baseline: 7 tests).
- **No API LLM provider in M1** — only `MockLlm` (base build) and `CandleLlm` (feature `local`).
- **Entity types are open `kind` strings** — no enum, no taxonomy enforcement, no new nodes column.
- **candle pin:** stay on the candle line compatible with `candle-core = "0.10"` / fastembed 5.17 (see memory: dependency-constraints). Confirm any new candle crate version against that pin; add a `VERSION CHECK` comment as existing model code does.
- **Local models stay behind the `local` feature**; the base/dev build must compile and pass tests with only `MockLlm`.
- **Commits go through the `/commit-and-push` skill** (never raw `git commit` with identity overrides). Commit messages: Conventional Commits, no em/en dashes, no AI/Claude attribution of any kind.
- **Milestone naming:** "M1", not "Phase 1", in code comments, docs, and commit messages.

---

### Task 1: Entity dimension conventions (liam-store)

Reserved edge-type constants, the `supersede` refactor to use them, and a `NewNode::entity` constructor. Independent of the `Llm` work.

**Files:**
- Modify: `crates/liam-store/src/types.rs` (add `relation` module + `NewNode::entity`)
- Modify: `crates/liam-store/src/graph.rs:119-134` (supersede uses `relation::SUPERSEDES`)
- Modify: `crates/liam-store/src/lib.rs:27-30` (re-export `relation`)
- Test: `crates/liam-store/src/graph.rs` (tests module at end of file)

**Interfaces:**
- Produces: `liam_store::types::relation::{SUPERSEDES, MENTIONS}` (`&str` consts); `liam_store::relation` (re-export); `NewNode::entity(entity_type: impl Into<String>, name: impl Into<String>) -> NewNode` (sets `kind = entity_type`, `label = name`, `subject = name.trim().to_lowercase()`, empty content).

- [ ] **Step 1: Write the failing test** for the entity constructor and provenance round-trip. Append to the `#[cfg(all(test, feature = "backend-libsql"))] mod tests` block in `crates/liam-store/src/graph.rs`:

```rust
#[test]
fn new_node_entity_sets_kind_label_subject() {
    let n = NewNode::entity("person", "  Ada Lovelace ");
    assert_eq!(n.kind, "person");
    assert_eq!(n.label, "  Ada Lovelace ");
    assert_eq!(n.subject.as_deref(), Some("ada lovelace"));
}

#[tokio::test]
async fn entity_mentions_edge_round_trips() {
    let clock = Arc::new(FixedClock::new(Millis(1000)));
    let g = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
        .await
        .unwrap();
    let person = g.insert(NewNode::entity("person", "Ada")).await.unwrap();
    let fact = g.insert(NewNode::now("fact", "note", "Ada wrote the first algorithm")).await.unwrap();
    g.link(NewEdge::new(&person, &fact, crate::types::relation::MENTIONS)).await.unwrap();

    let out = g.neighbors(&person, crate::types::relation::MENTIONS).await.unwrap();
    assert_eq!(out, vec![fact]);
}
```

Note: if `Graph` exposes no `neighbors(node, edge_type)` helper, replace the last two lines with the existing traversal accessor used elsewhere in this test module (check the file for the current edge-read method name and mirror it); the assertion is that the `MENTIONS` edge from `person` reaches `fact`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p liam-store new_node_entity_sets_kind_label_subject -- --nocapture`
Expected: FAIL — `no function or associated item named 'entity' found for struct 'NewNode'`.

- [ ] **Step 3: Add the `relation` module and `NewNode::entity`** to `crates/liam-store/src/types.rs`. Add the module just below the `empty_attributes` helper (after line 11):

```rust
/// Reserved edge `type` values that carry library meaning. Use these instead of
/// string literals so provenance/versioning relations stay centralized.
pub mod relation {
    /// Links a new node to the prior version it replaced (contradiction handling).
    pub const SUPERSEDES: &str = "supersedes";
    /// Provenance: an entity node references a source fact/episode that mentions it.
    pub const MENTIONS: &str = "mentions";
}
```

Add the constructor inside `impl NewNode` (after `now`, around line 71):

```rust
    /// An entity page node. `entity_type` becomes the `kind` (e.g. "person",
    /// "company", "concept"); `name` is the label and, normalized (trimmed +
    /// lowercased), the `subject` so re-observing the same entity supersedes
    /// via `upsert_by` rather than duplicating. Content is empty until M2
    /// synthesizes the compiled truth.
    pub fn entity(entity_type: impl Into<String>, name: impl Into<String>) -> Self {
        let name = name.into();
        let subject = name.trim().to_lowercase();
        Self::now(entity_type, name, String::new()).with_subject(subject)
    }
```

- [ ] **Step 4: Refactor `supersede` to use the constant** in `crates/liam-store/src/graph.rs`. Replace the third statement of the `statements` vec (lines 125-133) with a bound `type` parameter:

```rust
            (
                "INSERT INTO edges (id, src, dst, type, attributes, tx_from, tx_to)
                 VALUES (?1, ?2, ?3, ?4, '{}', ?5, ?6)"
                    .to_string(),
                vec![
                    EdgeId::new().as_str().into(),
                    new_id.as_str().into(),
                    old.as_str().into(),
                    crate::types::relation::SUPERSEDES.into(),
                    now.into(),
                    FOREVER.into(),
                ],
            ),
```

- [ ] **Step 5: Re-export `relation`** in `crates/liam-store/src/lib.rs`. Extend the `pub use types::{...}` list (lines 27-30) to add `relation`:

```rust
pub use types::{
    relation, Change, ExplainedHit, GcReport, GraphConfig, Hit, NewEdge, NewNode, Query,
    RetentionPolicy, RetentionRule,
};
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p liam-store`
Expected: PASS — the two new tests plus all existing tests (including `upsert_by_supersedes_same_subject` and any `supersede` test, proving the refactor is behavior-preserving).

- [ ] **Step 7: Commit** via `/commit-and-push` with message:

`feat(store): add entity constructor and reserved relation constants for M1`

---

### Task 2: `Llm` trait + `MockLlm` (liam-model)

**Files:**
- Create: `crates/liam-model/src/llm.rs`
- Modify: `crates/liam-model/src/error.rs:5-12` (add `Llm` variant)
- Modify: `crates/liam-model/src/lib.rs:7-13` (module + exports)
- Modify: `crates/liam-model/Cargo.toml:19-22` (dev-dependency tokio for async tests)

**Interfaces:**
- Produces: `liam_model::Llm` (`#[async_trait]` trait with `async fn complete(&self, system: &str, prompt: &str) -> liam_model::Result<String>`); `liam_model::MockLlm` (unit struct); `ModelError::Llm(String)`.

- [ ] **Step 1: Write the failing test.** Create `crates/liam-model/src/llm.rs` containing only the test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_llm_is_deterministic_and_echoes_prompt() {
        let llm = MockLlm;
        let a = llm.complete("be terse", "hello").await.unwrap();
        let b = llm.complete("be terse", "hello").await.unwrap();
        assert_eq!(a, b, "same input yields same output");
        assert!(a.contains("hello"), "output reflects the prompt");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p liam-model mock_llm_is_deterministic_and_echoes_prompt`
Expected: FAIL — `cannot find type 'MockLlm' in this scope` / module not declared.

- [ ] **Step 3: Implement the trait and mock.** Prepend to `crates/liam-model/src/llm.rs` (above the test module):

```rust
//! Generative completion: turn a prompt into text. The store never does this;
//! the daemon uses it for synthesis (M2) and extraction (M3).

use async_trait::async_trait;

use crate::error::Result;

#[async_trait]
pub trait Llm: Send + Sync {
    /// Generate a completion for `prompt` under `system` guidance.
    async fn complete(&self, system: &str, prompt: &str) -> Result<String>;
}

/// Deterministic echo LLM for the base build and tests: no model, stable output.
pub struct MockLlm;

#[async_trait]
impl Llm for MockLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        Ok(format!("[mock] system={system} prompt={prompt}"))
    }
}
```

- [ ] **Step 4: Add the `Llm` error variant** to `crates/liam-model/src/error.rs`, after the `Rerank` variant (line 11):

```rust
    #[error("llm: {0}")]
    Llm(String),
```

- [ ] **Step 5: Declare the module and export** in `crates/liam-model/src/lib.rs`. Add `pub mod llm;` to the module list (after line 9) and extend exports (after line 13):

```rust
pub use llm::{Llm, MockLlm};
```

- [ ] **Step 6: Add the tokio dev-dependency** so async tests run. Append to `crates/liam-model/Cargo.toml`:

```toml

[dev-dependencies]
tokio = { version = "1", features = ["rt", "macros"] }
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cargo test -p liam-model mock_llm_is_deterministic_and_echoes_prompt`
Expected: PASS.

- [ ] **Step 8: Commit** via `/commit-and-push` with message:

`feat(model): add pluggable Llm trait and MockLlm for M1`

---

### Task 3: `CandleLlm` local provider (liam-model, feature `local`)

In-process generative model behind the `local` feature, following the `FastEmbedEmbedder` pattern (`Arc<Mutex<model>>`, `spawn_blocking`). No unit test (requires a model download); the deliverable is that it compiles under `--features local`.

**Files:**
- Modify: `crates/liam-model/Cargo.toml:14-22` (candle chat deps under `local`)
- Modify: `crates/liam-model/src/llm.rs` (append `CandleLlm`)
- Modify: `crates/liam-model/src/lib.rs` (feature-gated export)

**Interfaces:**
- Produces: `#[cfg(feature = "local")] liam_model::CandleLlm` with `pub fn load(model_id: &str, cache_dir: &str) -> Result<Self>` and the `Llm` impl.

- [ ] **Step 1: Add candle chat dependencies** under the `local` feature in `crates/liam-model/Cargo.toml`. Add to `[dependencies]` (optional) and extend the `local` feature list:

```toml
# Generative chat model for the Llm trait (feature `local`). VERSION CHECK:
# confirm candle-transformers tracks the same candle line as candle-core 0.10.
candle-transformers = { version = "0.10", optional = true }
tokenizers = { version = "0.20", optional = true }
hf-hub = { version = "0.3", optional = true }
```

```toml
local = ["dep:fastembed", "dep:candle-core", "dep:tokio", "dep:candle-transformers", "dep:tokenizers", "dep:hf-hub"]
```

- [ ] **Step 2: Implement `CandleLlm`.** Append to `crates/liam-model/src/llm.rs`:

```rust
/// In-process chat model over candle (feature `local`). Loads a quantized
/// instruct model by Hugging Face id; no server, offline once cached. The sync
/// generate loop runs on a blocking thread so the async runtime stays free.
///
/// VERSION CHECK: the candle-transformers generation API (model constructor,
/// `forward`, logits processing) moves across releases. Confirm the
/// quantized-model surface against the candle line pinned by candle-core 0.10
/// before relying on this in production.
#[cfg(feature = "local")]
pub struct CandleLlm {
    inner: std::sync::Arc<std::sync::Mutex<candle_chat::Session>>,
}

#[cfg(feature = "local")]
impl CandleLlm {
    /// Load a quantized instruct model by HF id (e.g.
    /// "TheBloke/Qwen1.5-0.5B-Chat-GGUF"), caching weights under `cache_dir`.
    pub fn load(model_id: &str, cache_dir: &str) -> Result<Self> {
        let session = candle_chat::Session::load(model_id, cache_dir)
            .map_err(|e| crate::error::ModelError::Llm(e.to_string()))?;
        Ok(Self { inner: std::sync::Arc::new(std::sync::Mutex::new(session)) })
    }
}

#[cfg(feature = "local")]
#[async_trait]
impl Llm for CandleLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        let inner = self.inner.clone();
        let system = system.to_string();
        let prompt = prompt.to_string();
        let out = tokio::task::spawn_blocking(move || {
            let mut s = inner
                .lock()
                .map_err(|_| crate::error::ModelError::Llm("model lock poisoned".into()))?;
            s.complete(&system, &prompt)
                .map_err(|e| crate::error::ModelError::Llm(e.to_string()))
        })
        .await
        .map_err(|e| crate::error::ModelError::Llm(e.to_string()))??;
        Ok(out)
    }
}
```

Then create the `candle_chat` submodule holding the concrete candle glue (weight download via `hf-hub`, tokenizer load, quantized-model construction, and a greedy/argmax token loop that stops at EOS or a max-token cap). Add `#[cfg(feature = "local")] mod candle_chat;` to `llm.rs` and implement `Session::load(model_id, cache_dir) -> anyhow::Result<Session>` and `Session::complete(&mut self, system, prompt) -> anyhow::Result<String>` there, mirroring `FastEmbedEmbedder::load` for the download/cache pattern. Keep the model, tokenizer, and device as `Session` fields.

Rationale for the split: it isolates the version-fragile candle API in one file so the trait/provider boundary stays stable.

- [ ] **Step 3: Add the feature-gated export** to `crates/liam-model/src/lib.rs` (near the other `#[cfg(feature = "local")]` exports):

```rust
#[cfg(feature = "local")]
pub use llm::CandleLlm;
```

- [ ] **Step 4: Verify the base build still compiles and tests pass** (no `local`):

Run: `cargo test -p liam-model`
Expected: PASS — `MockLlm` test green; `CandleLlm` not compiled.

- [ ] **Step 5: Verify the `local` build compiles**

Run: `cargo build -p liam-model --features local`
Expected: SUCCESS. If candle-transformers/tokenizers/hf-hub versions mismatch the candle-core 0.10 line, resolve per the VERSION CHECK note before proceeding (this is the known-fragile step).

- [ ] **Step 6: Commit** via `/commit-and-push` with message:

`feat(model): add candle-local CandleLlm provider behind the local feature`

---

### Task 4: Daemon wiring (`LlmConfig`, `build_llm`, injection)

**Files:**
- Modify: `crates/liam-daemon/src/config.rs` (add `LlmConfig`, `llm` field, defaults)
- Modify: `crates/liam-daemon/src/main.rs` (add `build_llm`, inject into `MemoryServer`)
- Modify: `crates/liam-daemon/src/mcp.rs` (add `llm` field + constructor param)

**Interfaces:**
- Consumes: `liam_model::{Llm, MockLlm}` (Task 2); `#[cfg(feature="local")] liam_model::CandleLlm` (Task 3).
- Produces: `Config.llm: LlmConfig` (`provider`, `model`, `cache_dir`); `build_llm(&Config) -> anyhow::Result<Arc<dyn Llm>>`; `MemoryServer::new(store, embedder, reranker, llm)`.

- [ ] **Step 1: Write the failing test** for config defaults. Append a test module to `crates/liam-daemon/src/config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_defaults_to_mock() {
        let c = Config::default();
        assert_eq!(c.llm.provider, "mock");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p liam-daemon llm_defaults_to_mock`
Expected: FAIL — `no field 'llm' on type 'Config'`.

- [ ] **Step 3: Add `LlmConfig` and the `llm` field** to `crates/liam-daemon/src/config.rs`. Add the field to `Config` (after `embedder` on line 17):

```rust
    pub llm: LlmConfig,
```

Add the struct after `EmbedderConfig` (after line 39):

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    /// "mock" (dev) or "local" (in-process candle; needs the `local` feature).
    pub provider: String,
    /// Hugging Face model id (GGUF repo) for the local provider.
    pub model: String,
    /// GGUF filename within the repo (GGUF repos host multiple quant variants,
    /// so the file must be named explicitly). Consumed by `CandleLlm::load`.
    pub gguf_file: String,
    /// Where model files live (offline after first fetch).
    pub cache_dir: String,
}
```

Add `llm: LlmConfig::default()` to `Config::default()` (in the struct literal, after `embedder`) and a `Default` impl after `EmbedderConfig`'s (after line 67):

```rust
impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "mock".into(),
            model: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
            gguf_file: "qwen2.5-0.5b-instruct-q4_k_m.gguf".into(),
            cache_dir: "~/.liam/models".into(),
        }
    }
}
```

- [ ] **Step 4: Run the config test to verify it passes**

Run: `cargo test -p liam-daemon llm_defaults_to_mock`
Expected: PASS.

- [ ] **Step 5: Add `build_llm`** to `crates/liam-daemon/src/main.rs`, mirroring `build_models` (after line 80):

```rust
/// Choose the LLM from config. Mock keeps the base build runnable; the `local`
/// provider (with the `local` feature) loads a candle chat model in-process.
fn build_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    if config.llm.provider == "local" {
        return build_local_llm(config);
    }
    Ok(Arc::new(MockLlm))
}

#[cfg(feature = "local")]
fn build_local_llm(config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    use liam_model::CandleLlm;
    Ok(Arc::new(CandleLlm::load(&config.llm.model, &config.llm.gguf_file, &config.llm.cache_dir)?))
}

#[cfg(not(feature = "local"))]
fn build_local_llm(_config: &Config) -> anyhow::Result<Arc<dyn Llm>> {
    tracing::warn!("llm.provider is 'local' but the daemon was built without the `local` feature; using mock");
    Ok(Arc::new(MockLlm))
}
```

Update the `use liam_model::...` import (line 14) to include the LLM types:

```rust
use liam_model::{Embedder, IdentityReranker, Llm, MockEmbedder, MockLlm, Reranker};
```

- [ ] **Step 6: Construct and inject the LLM** in `run()` in `crates/liam-daemon/src/main.rs`. After the `build_models` line (line 40), add:

```rust
    let llm = build_llm(&config)?;
```

Change the server construction (line 44) to pass it:

```rust
    let server = MemoryServer::new(store, embedder, reranker, llm);
```

- [ ] **Step 7: Add the `llm` field to `MemoryServer`** in `crates/liam-daemon/src/mcp.rs`. Extend the import (line 9):

```rust
use liam_model::{Embedder, Llm, Reranker};
```

Add the field to the struct (after `reranker`, line 41):

```rust
    llm: Arc<dyn Llm>,
```

Update the constructor signature and body (lines 47-49):

```rust
    pub fn new(
        store: Arc<DefaultGraph>,
        embedder: Arc<dyn Embedder>,
        reranker: Arc<dyn Reranker>,
        llm: Arc<dyn Llm>,
    ) -> Self {
        Self { store, embedder, reranker, llm, tool_router: Self::tool_router() }
    }
```

Note: the `llm` field is stored but unread in M1 (first consumed in M2). A stored-but-unread struct field does not trigger Rust's dead-code lint, so no `#[allow]` is needed; confirm at Step 8.

- [ ] **Step 8: Verify the daemon builds and the workspace is green**

Run: `cargo test --workspace`
Expected: PASS — all baseline tests plus the new `llm_defaults_to_mock`, with no new warnings.

- [ ] **Step 9: Commit** via `/commit-and-push` with message:

`feat(daemon): wire pluggable Llm provider into config and MemoryServer for M1`

---

## Self-Review

**Spec coverage:**
- `Llm` trait + `complete` + `MockLlm` + `ModelError::Llm` → Task 2. ✓
- `CandleLlm` under `local` with candle deps → Task 3. ✓
- `LlmConfig` + `build_llm` + `MemoryServer.llm` injection → Task 4. ✓
- Entity types as `kind` values (no schema change) → Task 1 (via `NewNode::entity`). ✓
- `relation` module (`SUPERSEDES`, `MENTIONS`) + `supersede` refactor → Task 1. ✓
- `NewNode::entity` with normalized (trim+lowercase) subject → Task 1. ✓
- Tests: MockLlm determinism (T2), build_llm/config dispatch (T4), entity round-trip + upsert dedup via existing test (T1), supersede unchanged (T1 Step 6). ✓
- Non-goals respected: no extraction/synthesis/retrieval change; no API provider. ✓

**Placeholder scan:** No TBD/TODO; the one judgement point (candle generation glue in Task 3 Step 2) is scoped with explicit `Session::load`/`Session::complete` signatures and a VERSION CHECK, consistent with the repo's existing model-code convention, not a placeholder.

**Type consistency:** `complete(&self, system, prompt) -> Result<String>` identical across trait, MockLlm, CandleLlm, and the daemon import. `MemoryServer::new` 4-arg form matches the `run()` call site. `relation::{SUPERSEDES, MENTIONS}` referenced consistently. `LlmConfig` fields (`provider`, `model`, `cache_dir`) match `build_llm` reads.

## Execution Handoff

Plan complete and saved to `docs/plans/2026-07-31-m1-foundation.md`.
