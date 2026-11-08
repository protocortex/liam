# LIAM

Layered Intelligent Agent Memory. Local-first and bitemporal, for AI agents. It
runs as one process, stores to a single file, and speaks the Model Context
Protocol (MCP) so an agent reads and writes long-term memory. `remember` and `recall` are the
core pair, and ten more tools cover reading, review and admin.

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

Moved to [Workspace](docs/development/workspace.md).

## Build and test

Moved to [Build and test](docs/development/build-and-test.md).

## Install with Homebrew

Moved to [Install](docs/getting-started/install.md).

## Run

Moved to [Running LIAM](docs/getting-started/running.md) and [Tools](docs/reference/tools.md).

## Configuration

Moved to [Configuration](docs/reference/configuration.md).

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
| `prompt_scheme` | A number built into the binary: the version of the convention that turns text into model input. Scheme 2 wraps Qwen3 search queries as `Instruct: <task>\nQuery:<query>`, the format on the model card, and leaves stored text plain; BGE-M3 and the mock embedder get no wrapper, and each small ONNX model gets the prefix on its model card. Scheme 1 used a `query: ` prefix instead. |

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

Moved to [Evaluating retrieval](docs/development/evaluating-retrieval.md).

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
| `provider` | `"local"` | Who writes summaries: `local`, or `anthropic`, `openai` or `gemini` (these need `[hosted]`). `ask` and entity synthesis stay local. |
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
| `provider` | `"local"` | Who answers the model half: `local`, or `anthropic`, `openai` or `gemini` (these need `[hosted]`). |
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

### Experimental status and the real-model baseline

Model extraction is experimental. Its quality on the real model has not been
measured: `crates/liam-daemon/src/extraction/r10/baseline.json` holds
`"measured": null`, so the only gates in force are an F1 floor of `0.5` and a
ceiling of `150` facts dropped as ungrounded over the 300-event corpus. The
rules and plumbing tiers run in CI. The real-model tier does not.

To measure it, run `liam fetch-models`, then, on a machine with enough free
memory:

```sh
cargo test -p liam-daemon --features llama real_model -- --ignored --nocapture
```

The run prints a JSON object with `measured`. Copy it into `baseline.json`, set
`gate.min_f1` at or below the measured F1 and `gate.max_facts_dropped_ungrounded`
at or above the measured count, and commit the file in a reviewed change. A new
prompt version or model needs a new run.

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

## Hosted providers

LIAM can send extraction and summary requests to Anthropic, OpenAI or Gemini
instead of the local model. It is off by default, the default build links no
TLS code for it, and nothing is sent until you build with `--features hosted`
and set `extraction.provider` or `summary.provider` along with every part of
`[hosted]`.

A request goes out only when all of this holds:

- `hosted.enabled` is true and the request's host is in `hosted.allowed_hosts`.
  The host must match as written: a port, a trailing dot, capitals, a user name
  or an IP address is a different string and is refused. Redirects are never
  followed, and only https is used.
- The item passes the privacy gate. The default is to deny. An item is sent
  only when its scope (or an ancestor) is in `hosted.privacy.allowed_scopes`,
  none of its sources is in `local_only_sources`, and the provider's declared
  region is attested for that scope. A summary is judged on everything behind
  it: one local-only source keeps the whole summary at home. A cached answer is
  used only while the item still passes.
- The provider has a dated model id in `hosted.models` and a price in
  `hosted.prices`. The reply must name the pinned model, or it is refused and
  nothing is cached. A provider that no longer has the model is an error, never
  a fallback.

Keys never go in `liam.toml`. Put one in a file only you can read and name it in
`hosted.key_files`, or on macOS in a Keychain item called `liam-<provider>`. A
key does not appear in logs, errors or the cache.

Extraction asks hosted providers through their batch APIs, which cost less and
answer later. A run sends the requests the cache cannot answer in one batch,
records them in `<database>.hosted-batches.json`, and a later run collects the
answers into the response cache before it writes any fact. Events whose request
is still with the provider are not passed, so the cursor waits for them. A
request that expires is sent once more, and a second expiry fails it. Summaries
are asked directly. `ask` and entity synthesis stay on the local model whatever
the setting.

Every request, batch or direct, is first noted in `<database>.hosted-sends.jsonl`
with the provider, an opaque request id, the events it was built from and the
time. The file holds no text. If a run stops where it cannot tell whether a
request reached the provider, the next run sends it again and logs a warning
that it may be billed twice.

What a delete does about it: `liamd delete` cancels the batches that hold a
deleted request and asks the provider to delete their files where it has an
endpoint for that. A provider that keeps a copy anyway, one that cannot be
reached with the key and pin you set, and every direct request, leave a notice in
the job result and in `liamd delete --status`. A provider's own retention terms
decide how long its copy lasts, and LIAM cannot reach it. Treat anything you
allow into a hosted prompt as leaving your control.

Costs are counted per provider and model in the `model_calls` and
`model_cost_micro_dollars` metrics, using the price table you set. A model with
no price counts as free and is logged.

The endpoint paths and field names of the three adapters follow each
provider's public documentation and have not been checked against a live
account. Hosted R10, the extraction quality check, runs in two tiers. The
recorded tier replays answers shaped like each provider's through the real
adapters in normal CI. The live tier is ignored, needs a key, and has not been
run yet, so `r10_results.toml` has no rows.

## Feature flags

Moved to [Feature flags](docs/reference/feature-flags.md).

## Packaging

Moved to [Install](docs/getting-started/install.md).

## Status

Moved to [Status and roadmap](docs/development/status-and-roadmap.md).

## Roadmap

Moved to [Status and roadmap](docs/development/status-and-roadmap.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

Dependencies stay to the same standard: everything in the tree is permissive (MIT, Apache-2.0,
BSD, ISC, Zlib, and similar) apart from MPL-2.0, which is weak, file-level copyleft.
`cargo deny check licenses` enforces that with an allowlist, so a new GPL, LGPL, or AGPL
dependency fails CI rather than arriving unnoticed.
