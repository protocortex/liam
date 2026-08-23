# M1 — Foundation: Llm trait + open entity dimension

**Date:** 2026-07-31
**Status:** Approved design, pending implementation plan (writing-plans)
**Roadmap:** first milestone of `2026-07-31-liam-gbrain-architecture-roadmap-design.md`
**Baseline:** tag `v0.1.0`

## Purpose

Deliver the two foundational primitives every later GBrain-aligned milestone
depends on, and nothing more:

1. A pluggable `Llm` trait (generative text), mirroring the existing
   `Embedder` / `Reranker` abstractions. Consumed by synthesis (M2) and
   extraction (M3); unused by any tool in M1.
2. An open entity dimension on the graph, expressed as conventions over the
   existing free-form primitives rather than new schema.

M1 is plumbing. It adds no extraction, no synthesis, no entity-page compilation,
and no retrieval change.

## Context (from v0.1.0)

- `kind` (nodes) and `type` (edges) are already free-form `TEXT` columns.
- `subject` + `scope` already drive identity/dedup via `Graph::upsert_by`
  (same subject+scope -> supersede the prior live version).
- `Embedder` / `Reranker` live in `liam-model` with a `mock` default and a
  `#[cfg(feature = "local")]` fastembed provider; the daemon selects a provider
  from config via `build_models` and injects `Arc<dyn Trait>` into
  `MemoryServer`.
- The `supersede()` path hard-codes the `'supersedes'` edge type string.

## Design

### 1. `Llm` trait (liam-model)

New file `liam-model/src/llm.rs`:

```rust
#[async_trait]
pub trait Llm: Send + Sync {
    /// Generate a completion for `prompt` under `system` guidance.
    async fn complete(&self, system: &str, prompt: &str) -> Result<String>;
}
```

- Single text-completion method. Structured/JSON extraction is a caller concern,
  deferred to M3; no `complete_json` in M1.
- `MockLlm`: deterministic, dependency-free, always compiled. Returns a
  canned/echo-style output stable enough to assert on in tests.
- `#[cfg(feature = "local")] CandleLlm`: in-process chat model via candle
  (`candle-transformers`, `tokenizers`, `hf-hub`), loaded by Hugging Face model
  id + cache dir, following the offline-first pattern of `FastEmbedEmbedder`.
- `ModelError` gains an `Llm(String)` variant.
- `liam-model/src/lib.rs` exports `Llm`, `MockLlm` unconditionally and
  `CandleLlm` under `#[cfg(feature = "local")]`.

**Cargo features:** the new candle chat deps are added under the existing
`local` feature in `liam-model/Cargo.toml`; `liam-daemon`'s `local` feature
already forwards `liam-model/local`.

### 2. Config + daemon wiring

- `LlmConfig { provider: String, model: String, cache_dir: String }` added to
  `liam-daemon/src/config.rs`, mirroring `EmbedderConfig`; add `llm: LlmConfig`
  to `Config` with serde defaults (`provider = "mock"`).
- `build_llm(config) -> anyhow::Result<Arc<dyn Llm>>` in `main.rs`, mirroring
  `build_models`: dispatch on `config.llm.provider`, with
  `#[cfg(feature = "local")]` / `#[cfg(not(feature = "local"))]` variants; the
  non-local build warns and falls back to `MockLlm` when `provider = "local"`.
- `MemoryServer` gains an `llm: Arc<dyn Llm>` field, constructed and injected in
  `run()` now. No M1 tool consumes it; M2 is the first consumer. (A stored,
  not-yet-read field does not trip Rust dead-code warnings.)

### 3. Entity dimension (conventions, near-zero schema change)

- **Entity types are `kind` values** (`person`, `company`, `concept`,
  `artifact`, ...). Any string is allowed; these are conventions, not an enum.
  No new column, no migration.
- **Reserved-name module** to stop stringly-typed constants from scattering:

  ```rust
  pub mod relation {
      pub const SUPERSEDES: &str = "supersedes";
      pub const MENTIONS: &str = "mentions"; // provenance: entity <- source fact/episode
  }
  ```

  Refactor the hard-coded `'supersedes'` in `supersede()` to reference
  `relation::SUPERSEDES` (behavior-preserving).
- **Entity identity** reuses `subject` + `scope` + `upsert_by`: an entity's
  `subject` is its normalized name, so re-observing the same entity supersedes
  rather than duplicates.
- **One ergonomic constructor** `NewNode::entity(entity_type, name)` that sets
  `kind = entity_type`, `label = name`, `subject = normalized(name)`. No
  store-layer logic change; it is sugar over existing builders.

## Testing

- `MockLlm::complete` returns deterministic output; assert stability.
- `build_llm` dispatches `provider = "mock"` -> `MockLlm`; provider selection
  matches the `build_models` test shape.
- Entity round-trip: insert a `kind = "person"` node via `NewNode::entity`, link
  a `relation::MENTIONS` provenance edge to a `fact` node, query both directions
  (entity -> sources, source -> entities).
- `upsert_by` dedups two entities sharing a normalized subject+scope (one live
  version remains), mirroring the existing `upsert_by_supersedes_same_subject`
  test.
- Existing `supersede` tests still pass after the `relation::SUPERSEDES`
  refactor (no behavior change).

## Affected files

- `crates/liam-model/src/llm.rs` (new)
- `crates/liam-model/src/lib.rs`
- `crates/liam-model/src/error.rs`
- `crates/liam-model/Cargo.toml`
- `crates/liam-store/src/types.rs` (`NewNode::entity`, `relation` module)
- `crates/liam-store/src/graph.rs` (`supersede` uses `relation::SUPERSEDES`)
- `crates/liam-daemon/src/config.rs` (`LlmConfig`)
- `crates/liam-daemon/src/main.rs` (`build_llm`, injection)
- `crates/liam-daemon/src/mcp.rs` (`MemoryServer.llm` field)
- Tests in the above crates.

## Non-goals (M1)

- No LLM consumer: no extraction, no synthesis, no entity-page compilation.
- No retrieval / RRF change.
- No provider beyond mock + candle-local (no API provider in M1).
- No entity-type taxonomy enforcement; `kind` stays open/free-form.

## Open risks

- **candle chat model weight/build cost.** Adding a generative model in-process
  enlarges the `local` build (deps, memory, model download). Mitigated by
  keeping it behind the `local` feature; base/dev build stays on `MockLlm`.
- **`relation::MENTIONS` semantics.** M1 only reserves the name and proves the
  edge round-trips; the extraction rules that actually populate it land in M3.

## Next step

Invoke writing-plans to turn this into a step-by-step implementation plan, then
implement against tag `v0.1.0`.
