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
| `scope` | no | Partition by project or agent. Trimmed; ASCII letters, digits, `-`, `_`, `/`; up to 200 characters; no leading/trailing `/` or empty segment (`//`). |
| `subject` | no | Identity. A new value with the same subject supersedes the old. |

`recall` retrieves and reranks.

| Field | Required | Notes |
|---|---|---|
| `query` | yes | Text; embedded for the vector channel too. |
| `kind` | no | Restrict to one kind. |
| `scope` | no | Restrict to one partition. Same format as `remember`'s `scope`. |
| `k` | no | How many hits to return (default 8). |

`timeline` shows an entity's compiled page and the mentions behind it, most
recent first.

| Field | Required | Notes |
|---|---|---|
| `entity` | yes | Handle or unambiguous prefix, as shown by `recall`. |
| `limit` | no | How many mentions to show (default 50, max 200). A capped result ends with an "(more mentions exist)" trailer instead of claiming an exact total. |

By default the embedder is a mock, so retrieval leans on full-text and graph
signals. Set `provider = "local"` and build with `--features local` for real
embeddings.

### Try it with a real embedder

From your clone of this repo, build with the `local` feature, switch the
config to the `local` provider, then start the daemon as documented above:

```sh
cargo build -p liam-daemon --features local
```

Set `embedder.provider = "local"` in `liam.toml` (see the Configuration table
below for the rest of the keys). Run `liamd` as shown above, then call
`recall` from your MCP client. The vector channel now carries real semantic
similarity instead of the mock embedder's hash-derived placeholder, so
results ranked mainly by meaning (not just shared words) start showing up
alongside the full-text and graph signals `recall` always blends in.

## Configuration

The daemon reads `liam.toml` from the working directory. Override the path with
`LIAM_CONFIG=/path/to/file`. A missing file or key falls back to the default; an
unknown key fails loudly.

| Key | Default | Meaning |
|---|---|---|
| `database_path` | `liam.db` | The libSQL file. |
| `log_filter` | `info,liam=debug` | tracing filter. `RUST_LOG` overrides it. |
| `embedding_dims` | `768` | Vector width. Fixed when the DB is created; changing it needs `liam rebuild` (see The embedder fingerprint). |
| `gc.episode_retention_days` | `30` | Age out `episode` nodes older than this. |
| `gc.interval_hours` | `6` | Sweep interval. |
| `gc.reclaim` | `true` | Run incremental vacuum after a sweep. |
| `gc.run_on_start` | `false` | Sweep once at boot. |
| `gc.max_resynth_per_tick` | `5` | Ceiling on entities re-synthesized per maintenance tick. |
| `gc.full_synthesis_mention_threshold` | `8` | Live-mention count at or above which the next resynthesis uses the full-tier token budget. |
| `gc.full_synthesis_max_new_tokens` | `512` | Output token budget for a full-tier resynthesis. |
| `embedder.provider` | `mock` | `mock` for dev, or `local` for in-process fastembed. Part of the embedder fingerprint. |
| `embedder.model` | `Qwen/Qwen3-Embedding-0.6B` | Hugging Face model id for `local`. Part of the embedder fingerprint. |
| `embedder.cache_dir` | `~/.liam/models` | Files for both the reranker and the embedder, each under its own `models--Org--Name` subdirectory. |
| `socket_path` | `~/.liam/liamd.sock` | Where `serve` listens and `proxy` connects. |
| `max_connections` | `16` | Concurrent socket sessions. Further clients wait in the kernel backlog. |
| `read_pool_size` | `4` | Read connections. Ignored for an in-memory database. |
| `log.dir` | `<database stem>.log` beside the database | The event log directory. `~` is expanded, and a relative path is taken from the database's directory. Must not be empty. An existing directory must be empty or already a log. |
| `log.wal_segment_max_bytes` | `8388608` | The open WAL segment is closed by the first write after it reaches this size, then compacted to Parquet. Must be at least 1. |
| `log.wal_rotate_interval_secs` | `300` | The open WAL segment is closed by the first write after it reaches this age, then compacted to Parquet. Must be at least 1. |
| `log.bloom_fpr` | `0.01` | False positive rate of the dedup filter. At least `1e-9` and below `1`. |
| `consolidation.probe_recall_margin` | `0.02` | A probe whose recall drops by at least this much fails the run, which is then rolled back. Above `0` and at most `1`. |
| `consolidation.dedup_similarity_threshold` | `0.95` | Cosine similarity at or above which two nodes are linked as near-duplicates. Above `0` and at most `1`. |
| `consolidation.probe_count` | `20` | Most probes drawn from the store to gate a run. From `1` to `1000`. |
| `consolidation.snapshot_dir` | `<database stem>.snapshots` beside the database | Where runs keep their snapshots. `~` is expanded, and a relative path is taken from the database's directory. Must not be empty. |
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

## The event log

Every write goes into an event log before it reaches the database, so the
database is a projection of the log. `liamd` opens the log at startup, before
it serves anything, and replays any event the database has not applied yet, such
as one left by a crash between the two. The maintenance tick does the same
before each sweep.

The log lives in `log.dir`, which defaults to `<database stem>.log` beside the
database (`~/.liam/liam.db` logs to `~/.liam/liam.log`). Back it up with the
database: it is the source of truth that `liam rebuild` reads.

`liamd` compacts each closed WAL segment into a Parquet file in the background
and finishes the ones a crash left behind at startup. A segment is closed by the
first write after it reaches `log.wal_segment_max_bytes` or
`log.wal_rotate_interval_secs`, so an idle daemon keeps its last segment open.

One writer owns the log directory at a time. A second `liamd`, or a `liam
rebuild` while `liamd` runs, fails at once with an error naming the directory.
The lock needs a filesystem with advisory locks, so keep the log on a local disk.

If the log is poisoned, which a failed disk write does, `liamd` logs `the event
log is poisoned; restart liamd to replay it` at error level and exits with a
failure. Restarting replays the log into the database.

`liam rebuild --database PATH [--log-dir DIR] [--force]` recreates the nodes and
edges from the log and checks the result against it, row by row. It needs the
store to itself, so stop `liamd` first. The log directory is `--log-dir`, else
`log.dir` from the config, else the default beside `--database`.

Vectors stored before documents and queries were embedded separately carry the
`query: ` prefix on documents too. After upgrading, run `liam rebuild --database
PATH` once: it clears the stored vectors and re-embeds every live node as a
document, which makes the store consistent.

### The embedder fingerprint

A store's vectors only make sense next to others from the same embedder. Beside
the database, `liamd` keeps a record of what wrote them: `<database>.embedder.json`
(for `liam.db`, that is `liam.db.embedder.json`). It holds four fields and a
file `version`:

| Field | Source |
|---|---|
| `provider` | `embedder.provider` |
| `model` | `embedder.model` |
| `dims` | `embedding_dims` |
| `prompt_scheme` | A number built into the binary: the version of the convention that turns text into model input. Scheme 1 puts the `query: ` prefix on Qwen3 search queries only; BGE-M3 and the mock embedder get no prefix. |

Every start of `liamd` and `liamd serve` compares the record with the
configured values before the store opens. `liamd proxy` opens no store and
skips the check.

- **No record, and no database or an empty one:** a fresh install. The record
  is written after the store opens, so a start that fails leaves nothing
  behind.
- **No record, and a database that holds data:** a store from before
  fingerprints. It is taken to hold the configured provider, model and
  dimensions under prompt scheme 0, so it fails the check (the binary is on
  scheme 1). `liam rebuild` clears this, but it needs an existing event log and
  refuses without one. A store that predates the log has none, so the
  fingerprint check currently blocks its start before the log migration can
  run. This is tracked in
  [#205](https://github.com/protocortex/liam/issues/205).
- **Record matches:** the start goes on.
- **Record differs in any field:** `liamd` exits before opening the store. The
  error names the recorded and the configured values and the rebuild command.
- **Record cannot be read, parsed, or has another `version`:** `liamd` exits
  with a message distinct from the mismatch one. The file is never overwritten,
  because that could hide a change of embedder.

To change `embedder.provider`, `embedder.model` or `embedding_dims`, stop
`liamd`, edit `liam.toml`, and run:

```sh
liam rebuild --database PATH
```

It re-embeds every live node with the configured embedder, resizes the vector
table, and records the new fingerprint, but only when no embedding failed. Do
not delete the `.embedder.json` file to get past the error: the old vectors
would stay in `node_vectors`, mixed with the new ones. Deleting the file is
safe only after the vectors are cleared or re-embedded. The other way out is to
restore the previous `liam.toml` settings.

### Evaluating retrieval

```sh
liam eval retrieval --model <ID> [--json]
```

It scores one embedding model on a fixed golden set and measures the embedder's
speed. Build `liam` with `--features local`; without it the command fails with
an error saying so. The model is downloaded into `embedder.cache_dir` on first
use. BGE-M3 is cached under `HF_HOME` instead when that variable is set.

| Flag | Meaning |
|---|---|
| `--model <ID>` | Required. `Qwen/Qwen3-Embedding-0.6B` or `BAAI/bge-m3`. The match ignores case. Any other id is refused with the list of supported ones. |
| `--json` | Print the report as JSON instead of text. Latencies are in milliseconds. |
| `--config <PATH>` | The config to read, as for the other `liam` commands (`LIAM_CONFIG` also works). |

The run uses the in-process `local` embedder whatever `embedder.provider` says.
Qwen3 runs at `embedding_dims`; BGE-M3 always runs at its fixed 1024.

The golden set is 80 hand-written questions, ten in each of eight categories
(`Lexical`, `VectorSemantic`, `GraphExpansion`, `KindFilter`, `ScopeFilter`,
`Decay`, `Confidence`, `AsOf`), over a synthetic corpus about one fictional
company. LIAM has no real usage logs yet, so the set stands in for them. It is
small, and most categories are decided by full-text search, so only
`VectorSemantic` separates embedders. Treat differences of a few thousandths as
noise.

The text report has a header line (model, dimensions, facts, questions), a
per-category table (precision and recall at 4 and 8, recall@10, nDCG@10, MRR),
an `overall` line, and an `embedder:` line with the call count, item count, mean
latency and throughput in items per second. The JSON report carries the same
figures plus a row per question. The ranking metrics are deterministic: two
runs of the same model give the same values. Only the timings vary with the
machine. The measured comparison of the two models is in
[`docs/specs/2026-10-23-m-l2-d42-embedder-evaluation.md`](docs/specs/2026-10-23-m-l2-d42-embedder-evaluation.md).

### Upgrading a database that predates the log

The first start with the log copies every row the database already holds into
the log, nodes first, then edges. Writes are refused with `BackfillRequired`
until the copy is done, and since the daemon runs it before serving, clients see
nothing but a slower start on a large database. The cost grows with the number
of rows: the copy is one log write per row, so expect minutes for millions of
rows. A client that connects while it runs can time out, so retry once the
daemon logs `backfill completed`. The log reports `logged/total` at every saved
resume point. An interrupted copy resumes where it stopped, and a database that
is already logged is not copied again.

### Garbage collection

A sweep is recorded in the log as tombstones before the rows are deleted, so a
rebuild does not bring swept rows back. The log lock is held for the whole
sweep, so a very large sweep blocks every writer until it finishes. Chunking
bounds the size of each event and transaction, not the wait.

### Consolidation

Consolidation is the maintenance job that runs on the `gc.interval_hours`
schedule, and on demand with `liam consolidate`. It is not the log's compactor,
which only turns closed WAL segments into Parquet and never removes data.

A run goes through these steps, each recorded in `<database>.consolidation.json`
as it finishes:

1. Snapshot the database into `consolidation.snapshot_dir`.
2. Draw up to `consolidation.probe_count` live nodes as probes and score them.
3. Link near-duplicates, those whose embeddings reach
   `consolidation.dedup_similarity_threshold`. Both rows stay; the merge adds an edge.
4. Run the maintenance steps: replay the log, sweep, repair provenance, refresh
   clusters, and (in the daemon only) resynthesize stale entities.
5. Score the probes again. A probe whose recall drops by `consolidation.probe_recall_margin`
   or more fails the run. A probe whose node the run's own sweep removed on purpose is
   not counted, but a probe that lost its node any other way fails the run, and so does a
   sweep that leaves no probe to check against.

A passing run commits. A failing one is rolled back: a marker in the log cancels
the run's own writes, the snapshot is restored and the log is replayed, so writes
clients made meanwhile survive. A run that stops for any reason resumes at its
next step, and a rollback that was cut short is finished when `liamd` starts,
before it serves.

```
liam consolidate [--database PATH]          # run, or resume an unfinished run
liam consolidate status [--database PATH]   # show the last run
```

`liam consolidate` needs the store to itself, so stop `liamd` first. It runs no
model, so resynthesis waits for the daemon's next scheduled run. `status` reads
one file and works while the daemon runs.

| Exit code | Meaning |
|---|---|
| `0` | The run committed, or `status` printed. |
| `1` | The run was rolled back and the store is as it was, or the command failed before or during the run. |
| `2` | The restore failed. The store needs manual inspection and its snapshot is kept. |

After a failed restore no new run starts, from the command or from the daemon,
until the manifest `<database>.consolidation.json` is removed. Check the store
and the kept snapshot first.

### Refused at open

The database remembers which log it belongs to. If you delete the log directory
and keep the database, `liamd` refuses to start with an error naming both ids,
because a fresh log would not hold the rows the database has. Restore the log
directory, point `log.dir` at it, or rebuild from a log that holds the rows.

## Feature flags

`liam-store`:

- `backend-libsql` (default): libSQL with native vector search (`F32_BLOB`,
  `vector_distance_cos`).
Leiden community detection is unconditional; there is no feature to disable it.

`liam-daemon`:

- `local`: load fastembed in-process (Qwen3 embedder, cross-encoder reranker).
  Off by default, which keeps the dev build small.
- `llama`: llama.cpp text generation, which is what `ask` needs. Metal comes
  from the macOS target, not from this flag.
- `cuda`: NVIDIA support. Needs the CUDA toolkit at build time, so it is
  opt-in; Apple GPU support is automatic and needs nothing.

A release build sets `local,llama`. Without them the daemon refuses to start
against a config naming those providers, rather than falling back to mocks
that would return random vectors and invented answers.

## Packaging

Models run inside the binary, so shipping means getting weights onto the
machine before the first query. Releases take the fetch-on-install route: the
tarball carries `liamd`, `liam`, a config, and `install.sh`, and weighs a few
megabytes.

```sh
tar -xzf liam-vX.Y.Z-aarch64-apple-darwin-netinstall.tar.gz -C liam
cd liam && ./install.sh
```

`install.sh` copies both binaries to `~/.local/bin`, writes `~/.liam/liam.toml`
if you do not already have one, runs `liam fetch-models`, and registers the
launchd agent. Use `--prefix`, `--skip-models`, or `--no-launchd` to change
that. Running it again is safe.

`liam fetch-models` downloads **and loads** each model. Loading is the point: a
truncated file downloads happily and only fails when something tries to use it,
which would otherwise be a user's first `recall`, long after the install.

Both models land under `embedder.cache_dir`, each in its own `models--Org--Name`
subdirectory. That makes a bundle-in-release option possible (ship the weights
in the tarball, pointed at by that same directory), though fetch-on-install is
still the route the current release workflow takes.

## Status

LIAM v1 (MVP). It re-extracts the memory core of the retired v0 (the archived
`liam-archive` repo, a larger code-intelligence engine), rebuilt smaller with RRF
fusion and a full bitemporal model.

The workspace compiles on default features and the daemon builds to a binary.
`cargo test --workspace` is green. The store abstraction, the libSQL backend, and
the shared graph logic are done.

Not yet verified against their third-party APIs: the `local` build (fastembed v5
signatures) and the leiden-rs membership accessor in `cluster.rs`.

## Roadmap

- Confirm the fastembed and rmcp surfaces under a real end-to-end run.
- Add a `liam` CLI as a second consumer of the store.
- First tagged release with prebuilt binaries and a Homebrew formula.

## License

Apache-2.0. See [LICENSE](LICENSE).

Dependencies stay to the same standard: everything in the tree is permissive (MIT, Apache-2.0,
BSD, ISC, Zlib, and similar) apart from MPL-2.0, which is weak, file-level copyleft.
`cargo deny check licenses` enforces that with an allowlist, so a new GPL, LGPL, or AGPL
dependency fails CI rather than arriving unnoticed.
