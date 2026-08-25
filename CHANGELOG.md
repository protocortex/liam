# Changelog

All notable changes to LIAM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
