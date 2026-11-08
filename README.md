# LIAM

Layered Intelligent Agent Memory. Local-first and bitemporal, for AI agents. It
runs as one process, stores to a single file, and speaks the Model Context
Protocol (MCP) so an agent reads and writes long-term memory. `remember` and `recall` are the
core pair, and ten more tools cover reading, review and admin.

## Why

Moved to [Why LIAM](docs/concepts/why.md).

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

Moved to [The event log](docs/concepts/event-log.md).

### The embedder fingerprint

Moved to [The embedder fingerprint](docs/concepts/embedder-fingerprint.md).

### Evaluating retrieval

Moved to [Evaluating retrieval](docs/development/evaluating-retrieval.md).

### Upgrading a database that predates the log

Moved to [The event log](docs/concepts/event-log.md).

### Quarantined events

Moved to [The event log](docs/concepts/event-log.md).

### Garbage collection

Moved to [Garbage collection](docs/operations/garbage-collection.md).

### Consolidation

Moved to [Consolidation](docs/concepts/consolidation.md).

### Refused at open

Moved to [The event log](docs/concepts/event-log.md).

## Reading modes and provenance

Moved to [Reading modes and provenance](docs/concepts/reading-modes.md).

## Summaries

Moved to [Summaries](docs/concepts/summaries.md).

## Tiering

Moved to [Tiering](docs/concepts/tiering.md).

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

Moved to [Hosted providers](docs/concepts/hosted-providers.md).

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
