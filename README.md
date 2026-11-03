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
- Labelled provenance. Every item says whether it was recorded, extracted or
  inferred and which log events it rests on, and strict reads never return
  model-written text.
- A model-free core. The store takes vectors as input. Embedding and reranking
  live in a separate crate, so you swap models without touching storage.

## Workspace

Four crates, each usable on its own.

| Crate | Role |
|---|---|
| `liam-store` | The core. Bitemporal graph, hybrid retrieval, GC, clustering. Backend-generic over libSQL or SQLite. No ML. |
| `liam-log` | The append-only event log: write-ahead segments, Parquet compaction, encryption and archive. The store rebuilds from it. |
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
| `encryption.master_key` | unset | Seals new log events when set: `keychain` (the macOS login keychain) or `file`. Unset keeps a plaintext log. |
| `encryption.key_file` | unset | The master key file for `master_key = "file"`. `~` is expanded. Only its owner may read it. |
| `deletion.deadline_days` | `30` | Days a delete job has before it is reported as overdue. At least `1`. |
| `deletion.secure_delete` | `true` | Zero the database pages a delete frees. |
| `admin.producers` | empty | Producers that may call the `delete` and `consolidate` tools. Empty refuses everyone. |
| `archive.enabled` | `false` | Move old closed log partitions to `archive.dir`. |
| `archive.after_days` | `90` | Archive a closed partition once its newest event is this many days old. At least `1`. |
| `archive.dir` | unset | Where archived partitions go. Required when `archive.enabled` is true, and must exist. |
| `archive.zstd_level` | `3` | The zstd level of an archived copy, from `1` to `22`. |
| `metrics.enabled` | `true` | Keep metrics and raise alerts. `false` exposes no readings. |
| `metrics.sink` | `memory` | `memory`, or `log` to also log each active alert at error level. |
| `metrics.min_samples` | `5` | Samples a metric needs before a relative or rate rule can alert. At least `1`. |
| `metrics.delta_pct` | `20.0` | Percent a recall, latency, `ece` or Bloom rate may move between runs before an alert. |
| `metrics.rollback_ceiling` | `1.0` | Rollbacks at which an alert is raised. |
| `metrics.extraction_backlog_ceiling` | `10000.0` | Events past an extraction cursor at which an alert is raised. |
| `metrics.model_calls_per_hour` | `1000.0` | Model calls an hour at which an alert is raised. |
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
| `prompt_scheme` | A number built into the binary: the version of the convention that turns text into model input. Scheme 2 wraps Qwen3 search queries as `Instruct: <task>\nQuery:<query>`, the format on the model card, and leaves stored text plain; BGE-M3 and the mock embedder get no wrapper. Scheme 1 used a `query: ` prefix instead. |

The Qwen3 query text is `Instruct: Given a web search query, retrieve relevant
passages that answer the query` and a newline, `Query:` and the question.
Documents are embedded as they are. `liam eval retrieval` scores the same on
this format as on the old `query: ` prefix (recall@10 0.975, nDCG@10 0.836, MRR
0.817 either way), because the golden set is too easy to tell them apart. The
format was adopted for matching what the model was trained on, and the scheme
bump makes every store written under scheme 1 go through `liam rebuild` once.

Every start of `liamd` and `liamd serve` compares the record with the
configured values before the store opens. `liamd proxy` opens no store and
skips the check.

- **No record, and no database or an empty one:** a fresh install. The record
  is written after the store opens, so a start that fails leaves nothing
  behind.
- **No record, and a database that holds data:** a store from before
  fingerprints. It is taken to hold the configured provider, model and
  dimensions under prompt scheme 0, so it fails the check (the binary is on
  scheme 2). `liam rebuild` clears this and needs an event log. A store that
  predates the log has none, so `liamd` copies its rows into a new log before
  it runs the check. The start is still refused, but the log now exists: run
  `liam rebuild --database PATH` and start again.
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

### Quarantined events

When replay refuses an event, the daemon appends a `Quarantined` record that
cancels it and carries the reason, and keeps the reason in the store's
`log_quarantine` table. A store restored from a snapshot, or rebuilt from the
log, gets the row back from that record. A log written before this record
existed holds a plain `Voided` record instead, which has no reason, so a replay
skips the event and recreates no row. Schema version 7 adds the record; a build
that reads only version 6 refuses a log that holds one.

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

## Reading modes and provenance

Every item a read tool returns says where it came from. There are three
derivations:

- `recorded`: text a client wrote through `remember`.
- `extracted`: rows the extractor took out of recorded content.
- `inferred`: text a model composed, which is entity pages and summaries.

Reads come in two modes. `strict` returns what was recorded or extracted and
nothing a model wrote. `infer` may add inferred content, always labelled.

| Tool | Default | `infer` |
|---|---|---|
| `recall` | `strict` | Refused. |
| `clusters` | `strict` | Refused. |
| `timeline` | `strict` | Refused. The entity page heading the output is the one inferred item, and it is labelled so. |
| `ask` | `infer` | The default. Evidence may include entity pages and summaries. `strict` returns the recorded evidence and makes no model call. |
| `summaries` | not applicable | Lists summaries, which are always inferred. |

Entity pages are labelled `inferred` and left out of every strict read, both
from the results and from the ranking.

Each item ends with a `provenance:` line:

```text
provenance: derivation: recorded; events: 01a1..., 01a2...
provenance: derivation: inferred; basis: 01a1..., 01a2...; summary: 2
provenance: derivation: inferred; basis: unknown
```

`events` are the log events that wrote the item. `basis` are the events an
inferred item rests on, `basis: unknown` marks an entity page written before
LIAM tracked that, and `summary: N` gives a summary's level. A list longer than
`summary.max_basis_ids_rendered` is cut with `(+N more)`, and an event minted by
the backfill of an older store is marked `(synthetic)`.

## Summaries

Summaries are model-written overviews of clusters of recorded memory, built level
by level into a tree: level 1 summarizes clusters of memories, and level 2 and 3
summarize clusters of the level below. They are off by default and need a real
`llm` provider, not the mock. A `summarize` step in each consolidation run writes
the missing or out of date ones for every scope, and closes the ones whose
cluster is gone. A rolled back run leaves none of its summaries behind.

A summary is kept only if its content words are mostly found in the items it
summarizes (`summary.min_grounded_share`). Each one stores the log events behind
it, which are the events of its children, so it can be traced down to the
records.

Keys, all under `[summary]` in `liam.toml`:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | `false` makes the whole step a no-op. |
| `knn` | `8` | Nearest neighbours considered for each item. |
| `min_similarity` | `0.5` | Least cosine similarity at which two items are linked. |
| `time_window_hours` | `720` | Items further apart than this are not linked. |
| `min_cluster_size` | `3` | Fewest members of a cluster that gets a summary. |
| `max_cluster_size` | `30` | A larger cluster is split. |
| `max_depth` | `3` | Levels of summaries above the nodes, at most 3. |
| `recluster_change_ratio` | `0.2` | Share of a level that may change before it is clustered again. |
| `recluster_interval_days` | `7` | Age of the last full build after which a level is clustered again. |
| `min_attach_similarity` | `0.6` | Least similarity to a cluster's centroid for a new item to join it. |
| `max_unattached` | `20` | New items that may wait unattached before a level is clustered again. |
| `max_summaries_per_run` | `20` | Most summaries one run generates. A capped level is finished by the next run. |
| `run_deadline_secs` | `600` | Bounds the generation of one run. |
| `min_grounded_share` | `0.5` | Least share of a summary's content words its items must hold. |
| `max_summary_failures` | `3` | Failures of one child set after which it is skipped. |
| `min_summary_similarity` | `0.5` | Least similarity of a question to a summary for `ask` to use it. |
| `dissolve_grace_runs` | `3` | Runs a summary whose cluster is gone waits for a replacement before it is closed. |
| `dissolve_overlap` | `0.5` | Share of its items a stored summary must hold to replace it. |
| `max_basis_ids_rendered` | `8` | Event ids a provenance line lists before it counts the rest. |

### Reading summaries

`ask` in infer mode adds up to three summaries to its evidence: the live ones in
the question's scope (that scope and its descendants) that are at least
`summary.min_summary_similarity` close to it. Strict `ask` never reads them.

`summaries` lists them, ranked and cut by `query` when one is given, and takes
`level`, `scope` and `limit` (default 20, at most 100). With `id` it shows one
summary, open or closed, then each child as it stood when the summary was
written, with the events behind it. A child that no longer exists is named.

Time travel reads live summaries only. `ask` with `as_of` uses a summary that
was stored by then and does not cover a later time, and a summary closed after
`as_of` is not eligible, because closed summaries are not read.

### What the exit criteria prove

Two checks run in the test suite over a store written through a real log, and a
real summarize pass:

- Every live summary traces to real events. Its events are exactly those of its
  children, at least one is a real log event rather than a backfill placeholder,
  and each is an event the log holds.
- No inferred content appears in a recorded field of any tool's output. A strict
  output holds no inferred text or id, apart from the labelled `timeline` page,
  and every item shown carries the derivation the store holds for it.

These prove the links hold. They do not prove a summary is faithful to what it
summarizes: the grounding gate is a lexical check on words and nothing stronger,
so a summary can still misstate its sources. Follow the `provenance:` line to
the events before relying on one.

## Tiering

Tiering decides which memories keep their vectors. It is off by default. A hot
item has its float vector, a compact int8 form and a sign-bit signature. A cold
item has dropped its vectors, and keeps its row, its text index entry, its
edges, its events and the signature. A cold item is still found by a text query,
and by a vector query through a probe of the signatures that adds a few cold
hits behind the hot ones. Nothing is deleted: the log keeps every record, and a
`liam rebuild` leaves a cold item cold.

A `tier` step in each consolidation run scores the items, demotes the ones that
have not been useful and promotes the cold ones that have. The int8 form is
stored but nothing reads it yet. Items are tracked by lineage, the chain of
versions of one subject, so a new version of an item keeps its usage and its
verdicts. A new version is written hot.

### The score

The utility of an item is the weighted mean of four terms, each from 0 to 1:

- recent hits, `log(1 + hits) / log(1 + hits_cap)` over the last `window_days`,
  weight `w1`. Each producer's hits are capped first, per item per day and in
  total per day, so one client cannot keep an item hot alone;
- recency, `exp(-days / tau_days)` since the item was last returned (or written,
  if it never was), weight `w2`;
- answer contribution, the share of recent verdicts on the item that said it was
  correct, each counted by the trust of the producer who gave it, starting from
  `feedback_prior_correct` and `feedback_prior_incorrect` and reading at most
  the latest `feedback_item_cap` verdicts, weight `w3`. It is left out for an
  item with no verdicts, and the other weights are scaled to match;
- the trust of the item's producer from `[extraction.trust]`, weight `w4`.

A hot item below `demote_threshold` is demoted; a cold item at or above
`promote_threshold` is promoted. Between the two nothing moves. A cold item that
was returned since it went cold is promoted once its residency is over. A
promoted item is protected for `min_residency_days`, and an item younger than
`min_item_age_days` is never demoted.

### What counts as use

Only `recall` and `ask` results count, and only the ranked results the client
was shown, not the neighbours pulled in beside them. Counts are kept in memory
and written to the log every `usage_flush_secs`, so a crash loses up to that
many seconds of counts. A store that has not been counting for a whole
`window_days` demotes nothing, because every item would look unused.

Clients can say whether a result was right with the `feedback` tool, using the
handle `recall` or `ask` printed for it (on `Sources:` lines and in evidence
blocks too). A handle works on the connection that was given it. One producer
may give `feedback_per_producer_per_hour` verdicts an hour, counted in memory,
so the limit starts over when the daemon restarts. Clients that do not name
themselves share one producer, and so one budget. One producer's verdicts on
one item count once a day. The tool answers that verdicts are not kept while
tiering is off.

### What is never demoted

Entity pages, items a live summary is built from, items a recall probe of the
current run expects to find, and items moved by a rolled back run for
`cooldown_runs` runs. A rollback also halves the demotion cap of the next run,
for each rollback in a row, down to one. The caps are `max_promotions_per_run`
and `max_demotions_per_run`, and `run_deadline_secs` bounds the step.

Before a vector is dropped, the log is read to check that the item's origin
events still hold what the item says. An item the log does not vouch for stays
hot. A demoted item is left out of later summary builds, because summaries are
built from items that have a vector.

If the embedder changes, `liam rebuild` clears the signatures with the vectors
and writes new ones from the new vectors. A store from before tiering gets
signatures the first time the step runs.

Keys, all under `[tiering]` in `liam.toml`:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | `false` makes the tier step a no-op and counts nothing. |
| `w1` | `0.4` | Weight of the recent hits term. |
| `w2` | `0.3` | Weight of the recency term. |
| `w3` | `0.2` | Weight of the answer contribution term, left out for an item with no verdicts. |
| `w4` | `0.1` | Weight of the producer trust term. |
| `tau_days` | `30.0` | Days over which the recency term falls to 1/e. |
| `hits_cap` | `20` | Hits in the window at which the hits term reaches 1. |
| `hits_per_producer_item_day_cap` | `10` | Most hits one producer adds to one item in a day. |
| `hits_per_producer_day_cap` | `100` | Most hits one producer adds across items in a day; past it the largest contributions are cut first. |
| `unknown_trust_ceiling` | `0.3` | Most trust the fallback producer (`producers.unknown_id`) can have, as an author and as a giver of verdicts. |
| `promote_threshold` | `0.5` | Utility at or above which a cold item is promoted. |
| `demote_threshold` | `0.15` | Utility below which a hot item is demoted. |
| `min_residency_days` | `7` | Days a promoted item is protected from demotion. |
| `min_item_age_days` | `14` | Least age of an item for it to be demoted. |
| `max_promotions_per_run` | `50` | Most items one run promotes. |
| `max_demotions_per_run` | `50` | Most items one run demotes, halved for each rollback in a row. |
| `window_days` | `30` | Days of usage that count, and of observation needed before any demotion. |
| `usage_flush_secs` | `60` | Seconds between flushes of the in-memory access counts to the log. |
| `max_tracked_items` | `10000` | Items the in-memory counts hold before the oldest is dropped. |
| `feedback_per_producer_per_hour` | `60` | Verdicts one producer may give per hour. |
| `feedback_item_cap` | `20` | Most recent verdicts on one item that the score reads. |
| `feedback_prior_correct` | `1.0` | Pseudo-verdicts of "correct" the contribution rate starts from. |
| `feedback_prior_incorrect` | `1.0` | Pseudo-verdicts of "incorrect" the contribution rate starts from. |
| `cooldown_runs` | `3` | Runs a lineage moved by a rolled back run is left alone. |
| `cold_probe_pool` | `24` | Cold items the probe contributes to one query. |
| `cold_oversample` | `4` | Candidates asked of the signature index per item wanted. |
| `cold_max_hamming_ratio` | `0.35` | Share of signature bits that may differ for a cold item to be a candidate. |
| `cold_arm_weight` | `0.5` | Factor on the cold probe's contribution to a fused score. |
| `run_deadline_secs` | `600` | Bounds the tier step of one run. |

The defaults are starting values and have not been measured on real data.

## Extraction

Extraction derives subject, predicate, object facts from the event log and
stores them beside the nodes and edges. It is off by default. Turn it on with
`extraction.enabled = true` in `liam.toml`. It runs inside the daemon's
consolidation job, so a fact appears after the next run, not at `remember` time.
`recall` is unchanged: facts are read through the `facts` tool only.

Two extractors feed it. The rules extractor needs no model: it finds dates,
links, identifiers and mentions of known entity pages. The model extractor asks
a local `llama-server` to state facts as JSON, and each fact must be grounded in
the text of the events it cites or it is dropped. Set `extraction.llm_enabled =
false` to run the rules alone.

### Keys

Every key lives under `[extraction]` in `liam.toml`. A key left out keeps its
default, and an unknown key fails at load. The shipped `liam.toml` lists them
all.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Off makes the whole hook a no-op. |
| `llm_enabled` | `true` | Off extracts with the rules only. |
| `llama_server_path` | unset | The `llama-server` binary. Required while `enabled` and `llm_enabled` are on. `~` is expanded. |
| `model_gguf` | unset | The GGUF file the server loads. Required under the same condition. `~` is expanded. |
| `extra_args` | empty | Appended to the server command line, for example thread limits. `--host`, `--port`, `--api-key` and `--alias` (also `-a`) are set by LIAM and refused here. |
| `host` | `127.0.0.1` | Where the server listens. Loopback addresses only. |
| `port` | `0` | `0` picks a free port. |
| `ctx_tokens` | `4096` | The server's context window. |
| `request_timeout_secs` | `120` | Timeout of one model request. |
| `start_timeout_secs` | `120` | How long the server has to become ready. At most `3600`. |
| `run_deadline_secs` | `600` | Bounds one extraction run. At most `86400`. |
| `ram_budget_mb` | `4096` | The server is not started while less than this much RAM is free. |
| `max_events_per_run` | `200` | Events each pass reads per run. A larger log is worked through over several runs. |
| `batch_events` | `8` | Events sent to the model in one request. |
| `min_event_chars` | `40` | Shorter events are skipped. |
| `max_batch_failures` | `3` | Failed attempts at one event before it is quarantined and the cursor moves past it. |
| `max_event_chars` | half of `ctx_tokens` in characters | Longest event text sent to the model. |
| `end_cues` | empty | Phrases that end a fact, added to the built-in English and Portuguese ones. |
| `max_competing_per_key` | `5` | Competing facts one single-valued key may hold. |
| `min_confidence` | `0.5` | The model's facts below this confidence are dropped. |
| `llm_trust_factor` | `0.8` | Share of a source's trust that a model fact keeps. |
| `default_trust` | `0.5` | Trust of a source `trust` does not name. |
| `trust` | empty | The `[extraction.trust]` table, below. |
| `predicates` | empty | The `[extraction.predicates]` table, below. |

All counts and timeouts must be at least `1`, and the three trust and confidence
values must lie in `[0.0, 1.0]`.

### Trust and arbitration

`[extraction.trust]` maps a source name, the producer that wrote the event, to a
score in `[0.0, 1.0]`. The name `backfill` stands for history backfilled from a
database that predates the log.

```toml
[extraction.trust]
notetaker = 0.9
backfill = 0.3
```

A fact's trust is the maximum over the sources that support it, multiplied by
`llm_trust_factor` when the model produced it. `LogEvent.trust_score` is ignored,
because live writes do not set it.

Arbitration applies to single-valued predicates only, and decides at write time:

- A fact with the same object as a live one only adds a source.
- A different object closes the live fact when its trust is at least the live
  one's, so a tie goes to the newer evidence. The closed fact stays readable as
  history.
- A lower-trust fact does not close a higher-trust one. It is stored beside it
  as a competing fact, and the `facts` tool marks both `competing: yes`. A key
  holds at most `max_competing_per_key` of them, and the weakest is retracted
  when it overflows.
- Recency is the evidence time of the cited events (the earlier of observed and
  ingested time), not the time of the run, so a restated old event cannot
  displace newer evidence.

Many-valued predicates never destroy each other.

This is a scoped departure from ADR-0006. That decision keys a fact's identity
on subject, scope and producer so a producer never closes another's fact. The
`facts` table has no producer column and arbitrates by trust instead. `nodes`
keeps the ADR-0006 rules.

### Predicates

The vocabulary is closed. Built in:

| Predicate | Cardinality | Notes |
|---|---|---|
| `lives_in` | one | Matches when one place contains the other ("Sydney" and "Sydney, Australia"). |
| `works_at`, `has_title`, `status` | one | |
| `born_on` | one | The object must be a date found in the text. |
| `knows`, `member_of`, `mentioned_in`, `references_date`, `references_url`, `references_id` | many | |

`[extraction.predicates]` overrides a built-in or declares a new one. Each entry
may set `cardinality` (`"one"` or `"many"`), `match` (`"exact"` or `"contains"`)
and `grounding` (`"text"` or `"date"`). Fields left out keep the built-in value,
and a predicate that is not declared at all is many-valued, which never destroys
a fact.

```toml
[extraction.predicates]
works_at = { cardinality = "many" }
reports_to = { cardinality = "one" }
```

### Reading facts: the `facts` tool

`facts` is read-only. It takes a `subject` (trimmed and lower-cased) and
optionally `scope` (that scope and its descendants), `as_of`, `believed_at` and
`limit` (default 20, at most 100). Times are epoch milliseconds.

Facts are immutable versions with two time axes. Valid time (`valid_from`,
`valid_until`) says when the fact held in the world. Transaction time (`tx_from`,
`tx_to`) says when the store believed it. A change never edits a row: it closes
the old version by setting its `tx_to` once, and inserts the new one, so earlier
belief stays exact. `open` means the end of time.

- `as_of` alone asks what is true at that instant, as believed now.
- `believed_at` asks what the store believed at that instant. With `as_of` as
  well, it asks what the store believed then about that other instant.

Each result lists the claim, `trust`, `confidence`, the extractor, the valid and
recorded ranges, the scope, `competing` when it applies, and the ids of the
supporting log events.

The schema names the columns `valid_from`, `valid_until`, `tx_from` and `tx_to`,
plus `evidence_at` (the recency above), `lineage_id` (stable across versions of
one claim), `object_key` (the normalized object that equality uses) and
`closed_by`. These differ from the names in the original brief.

### The model server

Under `extraction.enabled` and `extraction.llm_enabled`, a run starts `llama-server` and
stops it when the run ends. A server crash stops the model pass, never the memory
store. Because the port is open to any local process, the server is
started with a random API key that stays off the command line, and LIAM checks
that the process holding the port is the child it spawned before it sends the key
or any event text. A pidfile, `<database>.sidecar.pid`, lets the next daemon clear
a server a crash left behind. The server is not started when free RAM is below
`ram_budget_mb`.

`liam fetch-models` downloads the pinned `llama-server` (llama.cpp `b11860`,
macOS on Apple Silicon only) and the Qwen3.5-4B Q4_K_M model, checks the size and
sha256 of each, and prints the two lines to set. `install.sh` runs it when
`[extraction]` is enabled. Both steps need `curl` and `tar`, which ship with
macOS. On another platform, point `llama_server_path` at your own build.

### The cache and rebuilding

Model answers are cached in `<database>.extraction-cache.jsonl`, mode `0600`,
next to the database and outside it. The cache is safe to keep across a restore
or a rebuild, and deleting it costs only the model calls it saved. Facts are
rebuilt from the event log, which records every fact decision, and never from the
cache. A run that is rolled back stays gone after a rebuild.

The store can retract a fact so a restated old event does not revive it
(`retract_fact`, with an option to restore what the fact replaced). No tool
exposes it yet.

## Deleting a subject

LIAM can forget a subject. Set `[encryption]` and the log is sealed: each row
is encrypted under a key for its subject, and the keys are wrapped by a master
key. Deleting a subject destroys its keys, removes what was derived from its
rows (summaries, facts, cache entries, tier and usage rows, snapshots), and
scrubs the database. The data stays unreadable even where a copy of the log
remains.

Only `liamd delete` destroys data, and it needs the daemon stopped. The `delete`
tool is a dry run: it counts what a delete would destroy, store by store, and
changes nothing.

```sh
liamd delete --subject alice           # the counts, nothing changes
liamd delete --subject alice --yes     # start the job
liamd delete --status                  # job ids and states, never subjects
liamd delete --resume                  # finish every open job
liamd delete --cancel JOB              # only before a key was destroyed
```

A job has `deletion.deadline_days` to finish. It is warned about at 80 percent
of that time and alerted at the deadline (the `deletion_backlog_age_days`
metric). The daemon finishes open jobs when it starts, and refuses to start if
one cannot finish, because rows of a half-deleted subject would otherwise be
served.

What to know before you rely on it:

- Backups and cloud-synced copies of the database are out of reach. A copy made
  before a delete still holds the old pages.
- Losing the master key or the key registry (`keys.registry` in the log
  directory) loses all log content. There is no escrow.
- A client name is a claim, not an authenticated identity. `[admin].producers`
  guards against mistakes, and any client that claims an admin name can read the
  dry-run counts.
- An edge body is sealed under its source subject's key. Deleting only the
  destination subject does not make it unreadable.
- On Linux there is no keychain. A `0600` key file beside the data gives little
  protection against a local reader.
- Dedup applies within a key group. A subject that changed names across versions
  needs each name deleted.
- `secure_delete` lowers the risk that freed pages keep text, and does not
  remove it on an SSD or a copy-on-write file system.
- Nothing is deleted by age unless the owner sets a retention policy.

`liamd encrypt-log` seals an existing plaintext log in place, with the daemon
stopped.

### Archive

With `[archive]` enabled, a closed log partition whose newest event is older
than `after_days` is copied to `archive.dir` as zstd, checked, and removed from
the log directory. Nothing is deleted. A rebuild or a catch-up reads archived
partitions, and one that was archived and then lost fails with an error naming
the partition instead of rebuilding a store with a hole in it. Sealed bodies
barely compress, so zstd saves little on an encrypted log. The 90 days and
level 3 are starting values, not measured.

### Metrics and alerts

The `metrics` tool shows the latest reading of each metric and the active
alerts. Pass `metric` for one, and `limit` to cap the readings (200 at most).
Readings carry only a small fixed set of labels (tier, retrieval arm, provider,
model) and never a subject, a producer or a lineage key, so keeping them cannot
defeat a delete.

| Metric | Meaning |
|---|---|
| `probe_recall_at_10` | Mean recall at 10 over the consolidation probes. |
| `arm_latency_p50_ms`, `arm_latency_p95_ms` | Nearest-rank percentiles of the last 50 timings of each retrieval arm. |
| `recall_latency_*`, `ask_latency_*` | The same for the `recall` and `ask` tools. |
| `tier_bytes` | Bytes of vectors and signatures in the hot and the cold tier. |
| `log_bytes`, `archive_bytes` | Size of the log and archive directories. |
| `peak_rss_bytes` | Resident memory of the daemon and the processes it started. |
| `consolidation_seconds`, `rollback_count` | Duration of the last run, and rolled-back runs since start. |
| `items_promoted`, `items_reembedded` | What the last tier step moved, and the vectors it computed. |
| `bloom_false_positive_rate` | Filter passes the log did not confirm, over those plus rejections. |
| `ece` | Expected calibration error of item utility against feedback verdicts, in ten bins. Absent below `min_samples` verdicts. |
| `rules_extraction_backlog`, `model_extraction_backlog` | Events past each extraction cursor. |
| `model_calls`, `model_cost_micro_dollars` | Running totals per provider and model. |
| `cache_hit_rate` | Hits over lookups in the extraction response cache. |
| `deletion_backlog_count`, `deletion_backlog_age_days` | Open delete jobs, and the age of the oldest. |

After each consolidation run the readings are compared with the previous run's.
Three kinds of rule raise an alert, each inclusive of its bound: a relative
move of `delta_pct` percent (recall, latency, `ece`, Bloom rate), a ceiling
(rollbacks, extraction backlog, delete job age), and a rate of growth an hour
(model calls). A first run raises only ceiling alerts. A run that ends in a
failed restore is neither judged nor remembered. The previous values live in
`<database>.metrics-state.json`.

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

LIAM is pre-1.0. The last release is v0.1.1 (macOS on Apple Silicon); `main`
carries more since. It re-extracts the memory core of the retired v0 (the
archived `liam-archive` repo, a larger code-intelligence engine), rebuilt
smaller with RRF fusion and a full bitemporal model.

What is built:

- Storage and retrieval: the bitemporal graph, hybrid search with a cold arm,
  entity pages, scope hierarchy, producer-scoped collisions and `same_as`
  links.
- The event log, with rebuild, snapshots and restore, and consolidation runs
  that extract facts and link near-duplicates.
- Summary trees with strict and infer reading modes, and utility-based
  tiering.
- Deletion and log archive, with the first metrics. The rest of the work on
  trust caps, metrics and hosted adapters is in progress.

What is checked: the workspace builds, `cargo test --workspace` is green, the
`local` build compiles against fastembed v5, and the community-detection
wrapper in `cluster.rs` is covered by tests against `leiden-rs` 0.8.1. The
real-model tests are gated and run on demand, not in the default test run.

## Roadmap

- Finish trust caps, metrics and hosted adapters, the last event log milestone.
- Hosted deployment: HTTP transport, OAuth and per-tenant scoping. Not
  planned yet.
- Agent Client Protocol support. Not planned yet.
- A Homebrew formula for the tagged releases.

## License

Apache-2.0. See [LICENSE](LICENSE).

Dependencies stay to the same standard: everything in the tree is permissive (MIT, Apache-2.0,
BSD, ISC, Zlib, and similar) apart from MPL-2.0, which is weak, file-level copyleft.
`cargo deny check licenses` enforces that with an allowlist, so a new GPL, LGPL, or AGPL
dependency fails CI rather than arriving unnoticed.
