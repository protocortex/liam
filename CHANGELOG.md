# Changelog

All notable changes to LIAM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An embedder-only multilingual retrieval check in eight languages, with a gate on English, Portuguese
  and Spanish. It runs as an ignored test. See [Evaluating retrieval](docs/development/evaluating-retrieval.md#multilingual-embedder-check).
- `recall_timeout_secs` (default `10`) bounds how long `recall` and `ask` spend retrieving. Past it they reply
  with a timeout. Cancelling a request now stops retrieval with a `cancelled` reply; the `ask` model call that
  follows retrieval still runs until `ask_timeout_secs`.
- `recall` and `ask` now report when a reply used a fallback. Every reply carries a structured `degraded` list
  (`embed_failed`, `synthesis_unavailable`), and a degraded reply
  starts with a marker line that names each reason. A new `degraded_replies` metric counts them by tool and reason.
  See [Degraded replies](docs/reference/tools.md#degraded-replies).

### Changed

- **Breaking:** the local generation model is removed, and with it llama.cpp from the daemon. LIAM now holds one
  model, the arctic embedder. The model was Qwen2.5 1.5B on llama.cpp, and its last jobs were local summaries and
  counting tokens for `clusters`. The daemon is smaller and builds without the vendored C++ code, with no Metal or
  GPU build path. What to do:
  - Delete the `[llm]` section from `liam.toml`. An old `[llm]` section is accepted and ignored, and the daemon
    logs one warning at start.
  - Summaries need a hosted provider and are off otherwise. A config with `summary.provider = "local"`, or with
    `summary.enabled = true` and no provider, still loads: summaries are off and the daemon logs one warning at
    start. To write summaries, set `summary.provider` to `anthropic`, `openai` or `gemini` and configure `[hosted]`
    and `metrics.enabled`, as [Summaries](docs/concepts/summaries.md) describes.
  - `liam devices` is removed, because nothing chooses a device any more: the embedder runs on the CPU. The
    `docs/reference/devices.md` page is gone with it.
  - `liam fetch-models` no longer fetches the generation GGUF. It fetches the embedder, and the extraction server
    and model when `[extraction]` is on.
  - `~/.liam/models` may hold the folder `models--Qwen--Qwen2.5-1.5B-Instruct-GGUF`, the old generation model.
    LIAM no longer reads it, and you may delete it. Keep `extraction.gguf` and the embedder's own folder. LIAM
    deletes nothing itself.
  - `clusters` cuts its listing at 3,276 characters instead of a tenth of `llm.context_tokens` counted in model
    tokens. A very large listing can show a different number of clusters, and still never fewer than one.
  - Consolidation runs no longer record an LLM identity, so a run is no longer refused for a changed generation
    model. A saved run that still holds the old field loads.
  - The permit that sidecar extraction, tiering and summary calls share is now fixed at 1 (`MODEL_CALL_PERMITS`).
    The old default autotuned and could grow, so this is a lower ceiling on a machine that had tuned up: those
    calls take turns.
  - A daemon older than this version cannot resume an extraction or consolidation run saved by this version,
    because the manifest lost `llm_model_fingerprint`. Old manifests still load.
  - Summaries the local model already wrote stay stored and the `summaries` tool still returns them. No new ones
    are written unless a hosted provider is configured.
  - The extraction sidecar is unchanged: it runs its own `llama-server` and model, and is off by default.
  See [ADR 0019](docs/adr/0019-remove-the-local-generation-model.md).
- Entity pages are built from the stored sentences of an entity without a model, and `timeline` shows them as
  before. A page lists the entity's memories oldest first with their dates, drops near duplicates and keeps at
  most four lines, so it cannot state a fact no memory states. `recall` is unchanged: it still returns recorded
  memory only, because a written page is not indexed for text or vector search. A store that indexed model-written
  pages drops them from the text index once, when it opens. Pages the old model wrote are ignored (`timeline` shows no text for them) and rebuilt by the
  maintenance tick, a few per tick, which also closes any extra live page of the same entity. A scoped entity's
  page is built only from memories inside its scope. Two faults the pages exposed are fixed: a written page is
  now superseded when the entity is rebuilt, where it used to leave a second live row, and the consolidation
  probes no longer draw pages or summaries. What to do: delete `gc.full_synthesis_mention_threshold` and
  `gc.full_synthesis_max_new_tokens` from `liam.toml`. An old line for either is ignored and logs one warning at
  start. See [ADR 0018](docs/adr/0018-build-entity-pages-without-a-model-and-show-them-in-timeline.md).
- The generation tuner no longer adjusts its permits while the daemon runs, because entity pages were its only
  input. The whole tuner is removed with the local model, see the entry on removing it.
- **Breaking:** the `ask` tool is removed. In strict mode `ask` ranked recorded memory like `recall` and added a
  written answer. In infer mode, its default, it also ranked entity pages and summaries beside recorded memory.
  Use `recall` for recorded memory: it returns recorded memory only, as it did before, and the calling agent
  writes its own answer. Entity pages are now read through `timeline` and summaries through `summaries`; no
  query ranks them any more. There was no `liam ask` command to remove. The local
  1.5B model stayed for local summaries then and is removed in a later entry. `ask_sufficiency_check` is removed too: an old line is
  ignored and logs one warning. `ask_timeout_secs` stays, now bounding one batch of entity-page builds.
  The `ask_latency_*` metrics and the `synthesis_unavailable` degraded reason are gone, and `recall` no
  longer lists `synthesis_unavailable` in its output schema, which it never produced. The recall bench no
  longer measures recall during an `ask`. See [ADR 0017](docs/adr/0017-remove-the-ask-tool.md).
- **Breaking:** Snowflake arctic-embed-m-v2.0 is now the only embedder. Any other `embedder.model`, for
  example `Qwen/Qwen3-Embedding-0.6B-GGUF` or `BAAI/bge-m3`, stops `liamd` and `liam` at start with a message
  that names the value and says to delete the line, and a store built with another embedder is refused until
  `liam rebuild`. The other embedders were not the default and the owner chose to keep one, so one model
  gives a lighter daemon and a smaller build. Removing them raised line coverage from 95.69% to 96.47%. Arctic matches or beats Qwen3 on English, Portuguese and Spanish
  and is weaker on Arabic and Chinese.
  Removed with them: the `embedder.device` key (an old `device` line is ignored and logs one warning), the
  `--model` and `--device` flags of `liam eval retrieval` and `liam eval embedder`, the embedder line of
  `liam devices`, the `cuda` build feature and the candle dependency.
  See [ADR 0016](docs/adr/0016-arctic-is-the-only-embedder.md).
- `liamd` now tightens the files it owns inside `~/.liam` to owner-only (0600) when an older start left them
  wider: the sidecars named after the database, the log's WAL and Parquet files, `log.id`, `log.lock`, the key
  registry, and the snapshot files. It runs at start, and `liam rebuild`, `liam consolidate`, `liamd delete`
  and `liamd encrypt-log` do the same when they finish, so `liam.db` is no longer left at 0644 after a
  rebuild. The other files are only touched inside `~/.liam`. The database and its `-wal` and `-shm` files are
  narrowed at any path, because the user pointed the tool at them, for example with
  `liam rebuild --database /elsewhere/copy.db`, and that holds even when `HOME` is unset. Symbolic links and
  files with other names are left alone, and a hard link is narrowed like any other file the same user owns.
  `liam.toml` is never changed; it holds no secrets.
- `liam rebuild` uses `database_path` from the config when `--database` is not given. The flag still overrides it.
  Before it opens the store it prints the database it will rebuild and the embedder it will re-embed with.
- The `liam rebuild` report now says `embeddings: re-embedded N nodes and summaries with MODEL at D dimensions`
  instead of `N restored`. The warning about GC sweeps that are not in the log is now printed only when the
  database already holds rows that the rebuild will replace, and it says what to do about it.
- `liam fetch-models`, `liam devices` and `liamd` no longer print llama.cpp and Metal log lines. They go to the
  log at their own levels and only warnings show by default. Set `RUST_LOG=info,llama-cpp-2=debug` to see them.
- An MCP call that leaves out a required field now gets a message that names the field, for example
  "remember needs the field kind, the kind of memory. The common kinds are decision, fact, symbol, episode, but
  any label is stored", instead of the raw
  `failed to deserialize parameters` text. What the tools accept is unchanged.
- `liamd` now logs one info line before it downloads a model that is not in the cache, naming the model, the
  directory and, where known, the size, and pointing to `liam fetch-models` (`liam fetch-models` itself leaves
  that hint out). It checks for the files the model loads, in `HF_HOME` when that is set, so a download that was
  interrupted gets the notice again. Nothing is logged when the files are already there.
- A clean shutdown no longer waits up to 3 seconds and logs `compaction was still queued at exit`. Closed
  segments that were not compacted yet stay as WAL files, and the next start compacts them. A compaction that
  is already running still gets up to 3 seconds to finish.
- A socket path longer than the platform limit is refused before anything is created, with a message that gives
  its length and the limit and says to set `socket_path` to a shorter path.
- The warning about a relative `socket_path` now prints the path and the directory it resolves against instead
  of an empty `dir=`.
- The refusal for an embedder mismatch is now a few short sentences: what differs, the `liam rebuild` command,
  and what happens if you do nothing. It is still one log event.
- Errors from the store, the log, the models and the sidecar now keep their underlying error as a source. The
  store can report a busy or locked database apart from other engine failures, and the dedup step logs one
  warning when embedding fails.
- The toolchain is pinned to Rust 1.99.1 in `rust-toolchain.toml`, CI installs it from that file, and the
  workspace declares a minimum Rust version of 1.89.
- The answer LLM now loads on first use and unloads after 5 minutes idle, so a daemon nobody is asking holds
  no model memory (about 1.0 to 1.3 GB). The first `ask` after an idle period is slower by the load time. A new
  `llm.idle_unload_secs` key (default `300`, `0` never unloads) sets the idle period, and the default of
  `llm.warmup` is now `false`. For the old behaviour set `llm.warmup = true` and `llm.idle_unload_secs = 0`.
  With `warmup = false` a model that cannot load is reported on the first `ask`, and at start only as a
  warning when the model file is missing from the cache; before, the daemon refused to start. With
  `warmup = true` a model that fails to load still stops the daemon at start.
  See [ADR 0015](docs/adr/0015-load-the-llm-on-first-use-and-unload-it-when-idle.md).
- **Breaking:** the default `embedder.model` is now `Snowflake/snowflake-arctic-embed-m-v2.0`, an int8 ONNX
  model on the CPU (311 MB, cut to 384 dimensions). On English, Portuguese and Spanish it matches or beats the
  previous default, Qwen3 Q8 on llama.cpp (nDCG@10 0.949 against 0.932 on the English 340 question set, 0.966
  against 0.935 on Portuguese, 0.931 against 0.924 on Spanish), and answers a query in about 8 ms against 37 ms.
  Qwen3 is better on Arabic and Chinese. A store built with the old default is refused at start with a message
  that names `liam rebuild --database PATH`; run it with `liamd` stopped, then `liam fetch-models` downloads the
  new model. See [ADR 0013](docs/adr/0013-arctic-default-embedder-and-reranker-off.md).
- A store with data and no `<database>.embedder.json` is now taken to hold the vectors of the default from before fingerprints,
  `Qwen/Qwen3-Embedding-0.6B` at 768 dimensions, instead of the configured embedder's. Opened under any other
  model or width, including the new default, it is refused with the same message that names `liam rebuild`.
- The embedder fingerprint records an ONNX model under its canonical id, so `embedder.model` spelled in another
  case no longer reads as a change of embedder.
- **Breaking:** the reranker is removed. `recall` and `ask` return candidates in store order, which is what
  the default setting already gave. A `[reranker]` section in `liam.toml` is ignored and the daemon logs one
  warning at start, so you can delete it; the `rerank_*` degraded reasons and the `stage_rerank_*` metrics are
  gone. On 10,000 passages the best reranker gained 0.05 nDCG@10 but took hundreds of milliseconds to seconds per
  recall and over 1 GB, and no model under 150 MB gained anything significant. A lighter daemon is worth more than that gain.
  See [ADR 0014](docs/adr/0014-remove-the-reranker.md).
- `recall` and `ask` now set `isError` when the call fails: the retrieval deadline passes, the store fails, or the
  query is invalid. A cancelled call still returns a normal reply. `recall` and `ask` now log a failed embed and
  report `embed_failed` instead of hiding it.
- An embed that gave up waiting for the model no longer queues behind it. A call waits for the model lock before it
  starts the work, so a client that disconnects while waiting leaves no thread behind. A cancel that arrives after
  the call took the lock still runs its embed to the end.
- `recall` and `ask` now reject an empty, whitespace-only or NUL-containing query and one over 4000 characters
  with a clear error. Only the first 64 distinct words reach the lexical arm; later words reach only the vector arm.
- `recall` now fetches a wider pool of candidates (twice `k`, at least 16, at most 32) before it cuts to `k`. A `k` over 32 now returns 32, and `k = 0` now returns 1 hit.
- `ask` now fetches a wider pool of candidates before it cuts. The evidence stays `k` hits plus at most 3 summaries.
- One slow read no longer blocks other clients' reads: each read takes its own connection from the pool and waits for a
  free one when all are busy. A read that is dropped before it finishes now stops its SQLite statement and returns its
  connection to the pool. A cancelled read also stops at its next row. Single-connection stores (in-memory and `file:`
  paths) share one connection with the writer, so they now run one read at a time.
- The pinned `llama-server` is now llama.cpp `b12173` (was `b11860`). The launch flags, environment variables and the
  `/v1/chat/completions` contract the extraction sidecar relies on are unchanged.
- The `local`, `llama`, `hosted` and `backend-libsql` build flags are removed. Every build now
  includes the local models, llama.cpp and the hosted-provider code, so `cargo install` and
  `cargo build` need no `--features`. Hosted calls stay off until `[hosted]` is configured.
  The two `test-support` flags remain.
- The default `embedder.model` is now `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0 on llama.cpp, on Metal
  on macOS). In the benchmark in `docs/specs/2026-11-09-llama-embedder-benchmark.md` it answers a
  query 8 times faster than candle on the CPU (23 ms against 183 ms median, the behaviour of
  v0.1.1) and the embedder uses 59 percent less memory (1072 MB against 2596 MB peak). Against
  candle on Metal, which `auto` picks on current main, it is 3 times faster (23 ms against 72 ms)
  and uses 53 percent less embedder memory (1072 MB against 2299 MB). Retrieval quality is equal
  (nDCG@10 0.934 against 0.933). A store created with `provider = "mock"` is not affected.
  `liam fetch-models` now downloads the 0.6 GB GGUF and no longer the safetensors unless the candle
  id is configured.
- The shipped `liam.toml` now uses the local embedder and the llama-cpp LLM. `provider = "mock"`
  is still accepted for tests and dev, and an unknown provider name is now an error.

### Fixed

- `liamd serve` now catches SIGTERM and SIGINT from the moment its socket exists. Before, a stop sent right after a
  client could connect could end the process without the orderly shutdown, leaving the socket file behind.
- Atomic multi-statement database writes, checkpoints, segment compaction and the stale sidecar sweep now run on
  blocking threads instead of the async runtime's workers, so a wait on a lock no longer stalls other requests or
  their deadline timers. Transactions and single statements stay on the runtime thread, so a cancelled request cannot
  leave the event log ahead of the graph.
- State directories are now created owner-only (0700) and the files `liamd` creates (WAL segments, lock files,
  state files) owner-only (0600), instead of taking the umask. `liamd` and `liam` also set the process umask to 077
  at start, so the database and its snapshots are never wider than their owner, even while the store opens. At start
  `liamd` tightens an existing `~/.liam`, the snapshot directory and the database files that are wider, never through
  a symbolic link or a `..` out of `~/.liam`, and only warns for a directory you pointed it at. The umask applies to the whole process, so a file `liam` writes to a
  path you choose (for example `--vectors-out`) is created owner-only, and child processes inherit it.
- `liamd` now logs one warning at start when `liam.toml` is not found, naming the path and saying that mock models
  are in use. Before, it ran with mock models and said nothing.
- The daemon now refuses a `gc.episode_retention_days` outside 1 to 36500 and a `gc.interval_hours` outside 1 to 8760
  at config load. Before, `0` days deleted every episode at the next sweep and `0` hours stopped maintenance. A
  maintenance task that ends abnormally is now logged at error level at shutdown.
- Batch embedding with the default embedder failed above 16 texts. The default embedder now embeds one text per
  call, so a text gets the same vector alone or in a batch.

### Upgrading

- An existing store is refused at its first start on this version until you run `liam rebuild
  --database PATH` with `liamd` stopped. Stores written before the event log have no log, so
  `liamd` first copies their rows into a new one, then refuses the start and names the rebuild
  command. The copy and the rebuild both read every row, so a large store takes a while: the
  rebuild re-embeds every live node with the configured embedder.
- A store built with the old default embedder is refused at start with a message that names
  `liam rebuild --database PATH`. Run it with `liamd` stopped.
- The event schema is now version 7.
- The embedder prompt scheme is now 2, which wraps Qwen3 search queries in the model's
  instruction format. A store written under scheme 1 needs the same `liam rebuild` once.
- The default `embedding_dims` is now 384 (Qwen3-Embedding-0.6B truncated by MRL; nDCG@10 0.933
  against 0.953 at 768 in `docs/specs/2026-11-06-embedder-speed-evaluation.md`). A store created
  at 768 is refused at 384 by the embedder fingerprint check: run `liam rebuild --database PATH`,
  or set `embedding_dims = 768` to keep the old width.
- Do not delete `<database>.embedder.json` to get past the refusal. The old vectors would stay
  mixed with the new ones.

### Added

- `liam devices` (and `--json`) prints the platform, the performance and efficiency cores, whether
  Metal opens, the compute backends this build has, the devices llama.cpp reports, and what `auto`
  resolves to for `embedder.device` and `llm.device`. See `docs/reference/devices.md`.
- Two producers writing the same `subject` in one scope now each keep a live fact instead of
  the later write replacing the earlier one. Entity pages stay one shared page per subject and
  scope. A scope filter such as `work` also matches `work/...` descendants on read, and a
  `same_as` relation, set through `relate`, makes `recall` and `ask` follow links between
  entities across scopes.
- An append-only event log (new `liam-log` crate) that the store rebuilds from: `liam rebuild`,
  snapshots and restore, per-key-group encryption, and archive of old partitions.
- Consolidation runs (`liam consolidate`) that extract bitemporal facts with a local model server
  and a rules pass, link near-duplicate nodes, and roll back a failed run. `facts` reads them
  back by valid and transaction time.
- Summary trees, with `strict` and `infer` reading modes, labelled provenance on every item, and
  a `summaries` tool.
- Utility-based tiering, with a `feedback` tool and result handles, and a cold search arm for
  demoted memories.
- `liam eval retrieval`, a BGE-M3 embedder option, and a pinned embedder fingerprint.
- Seven small ONNX embedders as `embedder.model` options (bge-small, MiniLM L6 and L12, Arctic
  xs and s, nomic v1.5, multilingual e5 small), `liam eval retrieval --dims`, and `liam eval
  embedder`, a 340 question embedder benchmark with load time, latency and throughput. The
  default embedder is unchanged.

- `remember` accepts an optional `episode` field, so a client can write several related facts,
  the entities they mention, and the edges between them in one call. Everything in the episode
  lands together, or none of it does. Leaving `episode` out keeps `remember` working exactly as
  it always has.

- `remember` now synthesizes an entity's compiled page from its mentions whenever an episode
  newly links a fresh entity, after the episode itself has already committed. A synthesis
  failure never fails the `remember` call; it's named in the response text instead. Re-mentioning
  an entity that already existed before the call, by handle rather than creating it fresh in the
  same episode, does not yet trigger resynthesis.

- `timeline` MCP tool: given an entity handle, returns its compiled page plus its live mentions,
  most recent first. Accepts an optional `limit` (default 50, max 200); a capped result ends with
  a "(more mentions exist)" trailer instead of claiming an exact total it cannot prove.

- `liamd delete`, which destroys everything held about a subject across the database, the log and
  the caches as a resumable job, and a `delete` tool that only counts what a delete would destroy.
  The `delete` and `consolidate` tools are limited to the producers named in `admin.producers`.
- A `metrics` tool, with alert rules judged after each run and set under `[metrics]`.
- A trust ceiling for the fallback producer, `tiering.unknown_trust_ceiling`, so a writer that
  names no producer cannot outrank a known one.
- Archive of old closed log partitions to `archive.dir`, under `[archive]`. A rebuild reads
  archived partitions.
- Sealing of new log events when `encryption.master_key` is set. Unset keeps a plaintext log.
- Hosted model providers (Anthropic, OpenAI and Gemini) for extraction and summaries. They are
  off by default: they need `hosted.enabled = true` and an allow-list of hosts.

### Changed

- `llm.max_concurrent_generations` now defaults to `0`, meaning auto-tune, instead of `1`. A `0`
  used to be treated as a misconfiguration, silently clamped to `1` with a warning; it is now the
  intentional default, so that warning is gone. Setting it explicitly still works exactly as
  before.

- License changed from AGPL-3.0-only to Apache-2.0. Applies going forward only; releases already
  published under AGPL-3.0-only keep those terms for anyone who already has them.

- Clustering's staleness check is now one database read instead of two, closing
  a window where a relationship recorded at the wrong moment could leave a
  stale grouping looking current. Recorded as ADR-0002 Amendment 6.

- The garbage collector and cluster maintenance now share the daemon's own store
  connection instead of opening a second one, and a cluster refresh runs after
  every sweep. Recorded as ADR-0002 Amendment 4.

- Deleting a memory now removes its relationships and cluster assignment
  automatically, enforced by the database rather than by each caller. Stores
  created before this are upgraded in place the next time the daemon starts, so
  no action is needed. The upgrade rebuilds two internal tables and runs once,
  so the first start after updating takes a little longer on a large store.

### Fixed

- `embedder.cache_dir` now relocates the embedder's weights too, not just the reranker's.
  Each model lands under its own `models--Org--Name` subdirectory of the configured path.

- Garbage collection no longer fails on any store that holds a relationship.
  It deleted memories before the links pointing at them, which the database
  rejects, so the whole sweep aborted. The daemon logged the failure and carried
  on, so old memories were never actually removed and the file kept growing. Any
  store that had ever recorded a memory with a `subject` was affected, because
  updating one leaves a link behind. If your store has been running a while,
  the first sweep after this will remove more than usual.

### Added

- `relate` MCP tool, so a client can record a relationship between two memories
  it already wrote or recalled. Before this, the only edge in the store was the
  `supersedes` link that version history writes, so the graph half of retrieval
  carried no real data and clustering had nothing to cluster. `relate` refuses
  `supersedes`, refuses a self-loop, and refuses an endpoint that is no longer
  live. Asserting the same relation twice is a no-op rather than a second edge.

- `clusters` MCP tool, listing the memory clusters community detection found,
  largest first, in the same `[kind handle] label` shape `recall` uses so a
  handle it prints feeds straight into `relate`. Output is bounded by a token
  budget of one tenth the configured model context rather than a fixed count,
  so a machine with more memory sees more before it starts withholding groups.
  No cluster is ever named by its internal number: that number is reassigned
  on every recompute, so nothing durable would be left to compare it against.

### Changed

- `recall` now prints a handle for each hit, as `[kind handle] label`. An agent
  could read a memory but had no name to pass to anything, so nothing could be
  linked after the fact. The handle is the first 13 characters of the node id,
  and `relate` takes either it or the full id. 13 because a node id is a ULID
  and its first 10 characters are a timestamp, so shorter prefixes collide for
  everything written in the same moment rather than spreading out.
- Community detection ignores `supersedes` edges, so a chain of edits to one
  memory no longer reads as a topic. It also counts a pair of memories once even
  when a client states the relationship in both directions.

### Added

- `remember` accepts `attributes` (a JSON object), `valid_from` (a backdated instant), and
  `confidence`, so a client can write the same shape of fact the store has always been able to
  hold, not just kind, label, and content. `confidence` outside `0.0` to `1.0` and `attributes`
  that is not a JSON object are rejected rather than silently accepted. Recorded as ADR-0004.
- `recall` and `ask` accept `as_of`, an epoch-millisecond instant, so a client can ask what was
  true at a past moment instead of only now. Recorded as ADR-0004.

### Changed

- `recall` and `ask` now show `confidence` and `attributes` on a hit when either was set to
  something other than the default, as trailing lines after the memory's content. A hit with
  neither set renders exactly as before. Recorded as ADR-0004.

## [0.1.1] - 2026-08-22

First release with a published artifact. macOS on Apple Silicon only.

### Added

- `ask` MCP tool, which writes an answer from retrieved evidence instead of
  returning raw hits. A separate yes/no pre-pass decides whether the store holds
  enough to answer at all, so it refuses instead of inventing. Answers are bounded
  by one deadline and trimmed to the model's context budget.
- Local text generation through llama.cpp. GGUF architecture is read from the file,
  so qwen2, qwen3, gemma, llama, mistral and phi3 all load. Metal comes from the
  macOS target and needs no flag. CUDA is opt-in behind `--features cuda`.
- `liamd serve`, a Unix socket listener, so several local clients share one store.
  `liamd proxy` bridges a client's stdio to it. Reads run concurrently through a
  pool with an explicit write lock.
- launchd socket activation. The socket outlives the daemon, so the first client
  connection starts it and a crash costs one reconnect.
- Producer attribution. Each node records which client wrote it, mapped from the
  MCP handshake through `[producers.clients]`.
- `liam`, a command line tool separate from the daemon. `liam fetch-models`
  downloads and loads the models your config names, so the first daemon start is
  not also the first download. Loading is deliberate: a truncated file downloads
  fine and only fails when something reads it.
- `packaging/install.sh`, which installs both binaries, writes a config if you
  don't have one, fetches models, then registers the launchd agent. Safe to re-run.
- A grounding eval for `ask`, measuring how often a local model answers from the
  evidence it was given.

### Changed

- Release artifacts build with `local,llama` and ship a config with real providers.
  Earlier builds shipped mock defaults, which start cleanly and then return random
  vectors from `recall` and invented answers from `ask`.
- Release tarballs are named `liam-` rather than `liamd-`, since they carry two
  binaries and an installer.
- The database defaults to `~/.liam` instead of the working directory.
- Declared license corrected to AGPL-3.0-only.
- The daemon refuses to start when the config names a provider the binary lacks,
  rather than falling back to a mock and logging to a stderr nobody reads.

### Fixed

- FTS5 queries no longer abort on raw punctuation. A question mark or apostrophe
  used to kill the whole hybrid query.
- Prompt injection defences for `ask` across the evidence and chat layers, plus a
  grounding gate on the output.
- The KV cache is cleared on every generation call. Qwen3 models don't reset
  themselves and were answering from stale keys.
- `kind` and `scope` filters apply to graph-expanded neighbours, which previously
  slipped past them.
- `embedder.cache_dir` and `llm.cache_dir` are tilde-expanded. Left raw, a real
  build downloaded gigabytes into a directory named `~`.
- launchd no longer starts the daemon at load time, and points it at a writable
  working directory. It used to start at `/`, which is read-only, so every spawn
  died on the store lock file.

### Known limitations

- `embedder.cache_dir` places the reranker only. The Qwen3 embedder weights always
  land in `~/.cache/huggingface/hub`, because fastembed's loader builds its hub
  client without a cache directory and hf-hub's default reads no environment
  variable. Neither `FASTEMBED_CACHE_DIR` nor `HF_HOME` moves them. This also blocks
  a bundle artifact with weights included, so fetch-on-install is the only route.
- Linux and Intel macOS aren't built or tested.

## [0.1.0] - 2026-07-31

Baseline tag on the rename from protocortex to LIAM. No artifact was published for
this version, and it predates local generation, the socket transport, and a working
release workflow. It's kept as a historical marker.
