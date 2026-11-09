# Changelog

All notable changes to LIAM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The `local`, `llama`, `hosted` and `backend-libsql` build flags are removed. Every build now
  includes the local models, llama.cpp and the hosted-provider code, so `cargo install` and
  `cargo build` need no `--features`. Hosted calls stay off until `[hosted]` is configured.
  `cuda` and the two `test-support` flags remain.
- The default `embedder.model` is now `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0 on llama.cpp, on Metal
  on macOS). In the benchmark in `docs/specs/2026-11-09-llama-embedder-benchmark.md` it answers a
  query 8 times faster (23 ms against 183 ms median) and the embedder uses about half the memory
  (1072 MB against 2596 MB peak) with equal retrieval quality (nDCG@10 0.934 against 0.933).
  `liam fetch-models` now downloads the 0.6 GB GGUF and no longer the safetensors unless the candle
  id is configured. The reranker is unchanged.
- The shipped `liam.toml` now uses the local embedder and the llama-cpp LLM. `provider = "mock"`
  is still accepted for tests and dev, and an unknown provider name is now an error.

### Upgrading

- An existing store is refused at its first start on this version until you run `liam rebuild
  --database PATH` with `liamd` stopped. Stores written before the event log have no log, so
  `liamd` first copies their rows into a new one, then refuses the start and names the rebuild
  command. The copy and the rebuild both read every row, so a large store takes a while: the
  rebuild re-embeds every live node with the configured embedder.
- A store built with the old default embedder is refused at start with a message that names
  `liam rebuild --database PATH`. Run it with `liamd` stopped, or set
  `embedder.model = "Qwen/Qwen3-Embedding-0.6B"` to keep the old candle model.
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
