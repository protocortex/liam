# protocortex

Local-first, bitemporal memory for AI agents. Runs as one process, stores to a
single file, and speaks the Model Context Protocol (MCP) so an agent reads and
writes long-term memory with two tools: `remember` and `recall`.

## Why

Agents forget between sessions. The usual fix is a vector database, but that
loses two things a memory needs: history and contradiction. protocortex keeps
both.

Every fact tracks two timelines. Valid time is when the fact is true in the
world. Transaction time is when the store learned it. Nothing is overwritten, so
you ask "what did I believe last Tuesday" and get the answer as of then. When a
new fact contradicts an old one, the old version stays readable and the store
links them.

Retrieval blends three signals instead of relying on vectors alone: full-text
match, vector similarity, and one-hop graph expansion, fused into one ranked
list and reranked for precision.

## What you get

- Hybrid retrieval. BM25 full-text, cosine vector search, and graph expansion,
  fused with reciprocal rank fusion (RRF), then reranked by a cross-encoder.
- As-of recall. Point-in-time queries over both timelines, so history is a
  first-class read, not an audit log you dig through.
- Contradiction handling on write. `upsert_by` supersedes the prior fact with
  the same subject and records a `supersedes` edge between them.
- Ranking that fades. Confidence and a per-query half-life down-weight old or
  low-trust facts; graph-expanded neighbours rank below direct matches.
- Housekeeping built in. A change cursor for incremental work, community
  detection (Leiden), and retention GC by kind.
- A model-free core. The store takes vectors as input. Embedding and reranking
  live in a separate crate, so you swap models without touching storage.

## Workspace

Three crates, each usable on its own.

| Crate | Role |
|---|---|
| `protocortex-store` | The core. Bitemporal graph, hybrid retrieval, GC, clustering. Backend-generic over libSQL or SQLite. No ML. |
| `protocortex-model` | In-process embedding and reranking via `fastembed`, behind the `local` feature. Mock and identity defaults keep dev builds light. |
| `protocortex-daemon` | The MCP server. Wires store and model, serves `remember` and `recall` over stdio, runs GC in the background. |

```
agent ──MCP──▶ protocortex-daemon ──embed/rerank──▶ protocortex-model
                       │
                       └── store/retrieve ──▶ protocortex-store ──▶ libSQL/SQLite file
```

The store is the reusable part. The daemon is one consumer; a CLI or a web graph
view would be others.

## Build and test

Rust stable, edition 2021.

```sh
cargo build --workspace          # mock embedder, no ML deps
cargo test  --workspace          # libSQL backend, in-memory
```

The base build stays light. The in-process model stack (fastembed, candle, ONNX
runtime) pulls a large dependency tree and is opt-in:

```sh
cargo build -p protocortex-daemon --features local
```

## Run

```sh
cargo run -p protocortex-daemon
```

The daemon serves MCP over stdio, so you launch it from an MCP client rather than
talking to it by hand. Point a client at the binary and it exposes two tools.

`remember` records a fact.

| Field | Required | Notes |
|---|---|---|
| `kind` | yes | Opaque label: `decision`, `fact`, `symbol`, `episode`. |
| `label` | yes | Short title. |
| `content` | yes | The text to embed and store. |
| `scope` | no | Partition by project or agent. |
| `subject` | no | Identity. A new value with the same subject supersedes the old. |

`recall` retrieves and reranks.

| Field | Required | Notes |
|---|---|---|
| `query` | yes | Text; embedded for the vector channel too. |
| `kind` | no | Restrict to one kind. |
| `scope` | no | Restrict to one partition. |
| `k` | no | How many hits to return (default 8). |

By default the embedder is a mock, so retrieval leans on full-text and graph
signals. Set `provider = "local"` and build with `--features local` for real
embeddings.

## Configuration

The daemon reads `protocortex.toml` from the working directory. Override the path
with `PROTOCORTEX_CONFIG=/path/to/file`. A missing file or key falls back to the
default; an unknown key fails loudly.

| Key | Default | Meaning |
|---|---|---|
| `database_path` | `protocortex.db` | The libSQL file. |
| `log_filter` | `info,protocortex=debug` | tracing filter. `RUST_LOG` overrides it. |
| `embedding_dims` | `768` | Vector width. Fixed when the DB is created. |
| `gc.episode_retention_days` | `30` | Age out `episode` nodes older than this. |
| `gc.interval_hours` | `6` | Sweep interval. |
| `gc.reclaim` | `true` | Run incremental vacuum after a sweep. |
| `gc.run_on_start` | `false` | Sweep once at boot. |
| `embedder.provider` | `mock` | `mock` for dev, or `local` for in-process fastembed. |
| `embedder.model` | `Qwen/Qwen3-Embedding-0.6B` | Hugging Face model id for `local`. |
| `embedder.cache_dir` | `~/.protocortex/models` | Model files. Sets `FASTEMBED_CACHE_DIR`. |

Logs go to stderr. Stdout carries the MCP JSON-RPC stream, so it stays clean.

## Feature flags

`protocortex-store`:

- `backend-libsql` (default): libSQL with native vector search (`F32_BLOB`,
  `vector_distance_cos`).
- `cluster` (default): Leiden community detection.
- `backend-rusqlite`: stock SQLite plus sqlite-vec. Scaffold only; the methods
  are `todo!()` and panic at runtime.

`protocortex-daemon`:

- `local`: load fastembed in-process (Qwen3 embedder, cross-encoder reranker).
  Off by default, which keeps the dev build small.

## Packaging

Models run inside the binary, so shipping means putting model files where
fastembed looks. The installer pre-populates `cache_dir` with the pinned model,
so first run is offline with no query-time download and no separate server. Two
options: fetch-on-install (a script pulls the model and checks a checksum) or
bundle-in-release (ship the files in the tarball and point `cache_dir` at them).

## Status

The workspace compiles on default features and the daemon builds to a binary.
`cargo test --workspace` is green. The store abstraction, the libSQL backend, and
the shared graph logic are done.

Not yet verified against their third-party APIs: the `local` build (fastembed v5
signatures) and the `cluster` build (the leiden-rs membership accessor). The
rusqlite backend is a scaffold and panics if enabled.

## Roadmap

- Finish the rusqlite backend, or drop the feature until a second backend is
  needed.
- Confirm the fastembed and rmcp surfaces under a real end-to-end run.
- Add a CLI and Unix-socket IPC as a second consumer of the store.
- First tagged release with prebuilt binaries and a Homebrew formula.

## License

MIT or Apache-2.0, at your option.
