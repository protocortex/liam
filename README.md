# protocortex

A local-first, bitemporal memory system for AI agents, as a Cargo workspace.

- **protocortex-store** — the storage crate. A bitemporal graph with hybrid
  retrieval (FTS + vector + graph, fused by RRF), backend-generic over libSQL or
  SQLite, with as-of queries, scope partitioning, graph-expansion down-weighting,
  confidence and temporal decay in ranking, contradiction-on-write (`upsert_by`),
  a change cursor, community detection, and GC. Model-free by design.
- **protocortex-model** — in-process embedding and reranking via `fastembed-rs`,
  behind a `local` feature. The `Embedder` and `Reranker` traits with a
  `FastEmbedEmbedder` (Qwen3, MRL-truncated) and a `FastEmbedReranker`
  (cross-encoder), plus mock/identity defaults for a light dev build. No server:
  models run inside the binary and are offline once cached.
- **protocortex-daemon** — the MCP server. Wires store and model, exposing
  `remember` and `recall` (recall embeds, retrieves, then reranks) to agents,
  with config, stderr logging, and a background GC task.

## Layers

```
agent ──MCP──▶ protocortex-daemon ──embed/rerank──▶ protocortex-model
                       │
                       └── retrieve/store ──▶ protocortex-store ──▶ SQLite/libSQL file
```

The store is the reusable core; the daemon is one consumer. A CLI or a web
graph view would be additional consumers of the same store.

## Status

The workspace compiles on the default features and the daemon builds to a binary;
`cargo test --features backend-libsql` is green. Store features are implemented
(see the store README for the per-feature verification boundaries). The model
crate wraps `fastembed-rs` behind the `local` feature (Qwen3 embedder +
cross-encoder reranker), with the fastembed v5 API surface still flagged for
confirmation; the default build uses mock/identity. The `local` and `cluster`
feature builds remain unverified.

## Packaging (no Ollama)

Models run in-process, so the only shipping task is putting the model files where
fastembed looks. The installer pre-populates `cache_dir` (it sets
`FASTEMBED_CACHE_DIR`) with the pinned model, so first run is fully offline and
there is no query-time download and no separate server. Either fetch-on-install
(bash script pulls the model, verifies a checksum) or bundle-in-release (ship the
model files in the tarball and point `cache_dir` at them). A Homebrew formula
follows once there are tagged releases with prebuilt binaries.

Next: finish the rusqlite backend, confirm the fastembed and rmcp surfaces,
add the CLI and the Unix-socket IPC, then run the daemon end to end.
