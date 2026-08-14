# LIAM

Layered Intelligent Agent Memory. Local-first and bitemporal, for AI agents. It
runs as one process, stores to a single file, and speaks the Model Context
Protocol (MCP) so an agent reads and writes long-term memory with two tools:
`remember` and `recall`.

## Why

Agents forget between sessions. The usual fix is a vector database, but that
loses two things a memory needs: history and contradiction. LIAM keeps both.

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
| `liam-store` | The core. Bitemporal graph, hybrid retrieval, GC, clustering. Backend-generic over libSQL or SQLite. No ML. |
| `liam-model` | In-process embedding and reranking via `fastembed`, behind the `local` feature. Mock and identity defaults keep dev builds light. |
| `liam-daemon` | The MCP server (binary `liamd`). Wires store and model, serves `remember` and `recall` over stdio, runs GC in the background. |

```
agent ──MCP──▶ liamd ──embed/rerank──▶ liam-model
                 │
                 └── store/retrieve ──▶ liam-store ──▶ libSQL/SQLite file
```

The store is the reusable part. The daemon is one consumer; a `liam` CLI or a web
graph view would be others.

## Build and test

Rust stable, edition 2021.

```sh
cargo build --workspace          # mock embedder, no ML deps
cargo test  --workspace          # libSQL backend, in-memory
```

The base build stays light. The in-process model stack (fastembed, candle, ONNX
runtime) pulls a large dependency tree and is opt-in:

```sh
cargo build -p liam-daemon --features local
```

## Run

```sh
cargo run -p liam-daemon         # builds and runs the `liamd` binary
```

`liamd` has three modes. Running it with no subcommand is the original one, so
existing MCP client configs keep working unchanged.

| Command | What it does | Opens the store |
|---|---|---|
| `liamd` | Serves MCP over this process's stdio. One client, one process. | yes |
| `liamd serve` | The shared daemon: serves many clients over a Unix socket. | yes |
| `liamd proxy` | Forwards stdio to a running daemon's socket. | no |

`--config PATH` overrides `LIAM_CONFIG` for any of them.

### One daemon, many agents

Running `liamd` per client gives each one its own process, and only one of them
can hold the database: the others fail to start on the store lock. Use the daemon
instead when more than one agent needs the same memory.

Start it once:

```sh
liamd serve
```

Then point every MCP client at the proxy rather than at `liamd` directly:

```json
{ "command": "liamd", "args": ["proxy"] }
```

Each client keeps its own identity through the proxy, because the proxy forwards
bytes untouched and the daemon reads the name from the client's own MCP
handshake. Map those names to stable producer ids in `[producers.clients]`, and
every fact records who wrote it.

On macOS, launchd can own the socket and start the daemon on the first
connection, so nothing has to run `liamd serve` by hand. See
`packaging/dev.protocortex.liamd.plist` for the job and its install steps.

### The socket

`socket_path` defaults to `~/.liam/liamd.sock` and is created owner-only (0600).
It carries full tool access with no further authentication, so anyone who can
reach the file can act as any client; owner-only is what makes "can reach it"
mean "is already you". `liamd serve` refuses to start if another daemon is
already listening, and replaces the socket only when it is stale.

Point a client at `liamd` and it exposes two tools.

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

The daemon reads `liam.toml` from the working directory. Override the path with
`LIAM_CONFIG=/path/to/file`. A missing file or key falls back to the default; an
unknown key fails loudly.

| Key | Default | Meaning |
|---|---|---|
| `database_path` | `liam.db` | The libSQL file. |
| `log_filter` | `info,liam=debug` | tracing filter. `RUST_LOG` overrides it. |
| `embedding_dims` | `768` | Vector width. Fixed when the DB is created. |
| `gc.episode_retention_days` | `30` | Age out `episode` nodes older than this. |
| `gc.interval_hours` | `6` | Sweep interval. |
| `gc.reclaim` | `true` | Run incremental vacuum after a sweep. |
| `gc.run_on_start` | `false` | Sweep once at boot. |
| `embedder.provider` | `mock` | `mock` for dev, or `local` for in-process fastembed. |
| `embedder.model` | `Qwen/Qwen3-Embedding-0.6B` | Hugging Face model id for `local`. |
| `embedder.cache_dir` | `~/.liam/models` | Model files. Sets `FASTEMBED_CACHE_DIR`. |
| `socket_path` | `~/.liam/liamd.sock` | Where `serve` listens and `proxy` connects. |
| `max_connections` | `16` | Concurrent socket sessions. Further clients wait in the kernel backlog. |
| `read_pool_size` | `4` | Read connections. Ignored for an in-memory database. |
| `producers.unknown_id` | `unknown` | Producer recorded for a client not in the table below. |
| `producers.clients` | empty | Maps a client's declared MCP name to a producer id. Matching ignores case. |

`[producers.clients]` is a plain table, and the ids are what land in the
`producer` column:

```toml
[producers.clients]
claude-code = "claude"
ai-notetaker = "notetaker"
```

Logs go to stderr. Stdout carries the MCP JSON-RPC stream, so it stays clean.

The store runs in WAL mode, so libSQL keeps `liam.db-wal` and `liam.db-shm` next
to the database. Anything that copies or backs up the store by path needs all
three: the `.db` file alone can be missing recently committed data.

## Feature flags

`liam-store`:

- `backend-libsql` (default): libSQL with native vector search (`F32_BLOB`,
  `vector_distance_cos`).
- `cluster` (default): Leiden community detection.

`liam-daemon`:

- `local`: load fastembed in-process (Qwen3 embedder, cross-encoder reranker).
  Off by default, which keeps the dev build small.

## Packaging

Models run inside the binary, so shipping means putting model files where
fastembed looks. The installer pre-populates `cache_dir` with the pinned model,
so first run is offline with no query-time download and no separate server. Two
options: fetch-on-install (a script pulls the model and checks a checksum) or
bundle-in-release (ship the files in the tarball and point `cache_dir` at them).

## Status

LIAM v1 (MVP). It re-extracts the memory core of the retired v0 (the archived
`liam-archive` repo, a larger code-intelligence engine), rebuilt smaller with RRF
fusion and a full bitemporal model.

The workspace compiles on default features and the daemon builds to a binary.
`cargo test --workspace` is green. The store abstraction, the libSQL backend, and
the shared graph logic are done.

Not yet verified against their third-party APIs: the `local` build (fastembed v5
signatures) and the `cluster` build (the leiden-rs membership accessor).

## Roadmap

- Confirm the fastembed and rmcp surfaces under a real end-to-end run.
- Add a `liam` CLI as a second consumer of the store.
- First tagged release with prebuilt binaries and a Homebrew formula.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).

Dependencies are a separate matter: LIAM takes no strong copyleft dependency. Everything in the
tree is permissive (MIT, Apache-2.0, BSD, ISC, Zlib, and similar) apart from MPL-2.0, which is
weak, file-level copyleft. `cargo deny check licenses` enforces that with an allowlist, so a new
GPL, LGPL, or AGPL dependency fails CI rather than arriving unnoticed.
