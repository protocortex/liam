# Configuration

[Back to the docs index](../README.md)

The daemon reads `liam.toml` from the working directory. Override the path with
`LIAM_CONFIG=/path/to/file`. A missing file or key falls back to the default; an
unknown key fails loudly.

| Key | Default | Meaning |
|---|---|---|
| `database_path` | `liam.db` | The libSQL file. |
| `log_filter` | `info,liam=debug` | tracing filter. `RUST_LOG` overrides it. |
| `embedding_dims` | `384` | Vector width. Fixed when the DB is created; changing it needs `liam rebuild` (see [The embedder fingerprint](../concepts/embedder-fingerprint.md)). |
| `recall_timeout_secs` | `10` | Seconds `recall` may spend retrieving (embedding and store query). Past it `recall` replies `retrieval took longer than N s`. A value of `0` is read as `1`. |
| `ask_timeout_secs` | `30` | Seconds one batch of entity-page builds may spend embedding the mentions and the page, which the `remember` call waits for. The key keeps the name it had when `ask` used it, so an existing `liam.toml` still loads. |
| `gc.episode_retention_days` | `30` | Age out `episode` nodes older than this. Must be between `1` and `36500` (100 years): a value outside that range is refused at load, because `0` or less would delete every episode. For the longest retention set `36500`. |
| `gc.interval_hours` | `6` | Sweep interval in hours, from `1` to `8760`. `0` and larger values are refused at load. |
| `gc.reclaim` | `true` | Run incremental vacuum after a sweep. |
| `gc.run_on_start` | `false` | Sweep once at boot. |
| `gc.max_resynth_per_tick` | `5` | Ceiling on entity pages rebuilt per maintenance tick. |
| `embedder.provider` | `mock` | `local` for the in-process arctic ONNX model, or `mock` for tests and dev. The shipped `liam.toml` sets `local`; the built-in default is `mock`. Part of the embedder fingerprint. |
| `embedder.model` | `Snowflake/snowflake-arctic-embed-m-v2.0` | The embedding model, and the only one: Snowflake arctic-embed-m-v2.0 as an int8 ONNX file (311 MB, Apache-2.0) on the CPU, with CLS pooling, the `query: ` prefix on queries and 512 input tokens. It is pinned to one revision and `liam fetch-models` checks its size and sha256. Its native width is 768 and `embedding_dims` (384 by default) cuts it and normalises again. Leave the key out. Any other value stops the daemon at start with a message that names it, see [ADR 0016](../adr/0016-arctic-is-the-only-embedder.md). A store built with another model is refused at start until `liam rebuild`. Part of the embedder fingerprint. |
| `embedder.cache_dir` | `~/.liam/models` | Files for the embedder, each model under its own `models--Org--Name` subdirectory. |
| `llm.provider` | `mock` | `llama-cpp` for in-process llama.cpp, or `mock` for tests and dev. The shipped `liam.toml` sets `llama-cpp`; the built-in default is `mock`. `local`, the retired candle provider, is no longer accepted. |
| `llm.model` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | Hugging Face id of the GGUF repo the local model comes from. |
| `llm.gguf_file` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | The GGUF file to load from that repo. A repo holds several quantisations, so the file is named explicitly. |
| `llm.cache_dir` | `~/.liam/models` | Where the local model's files live. They are used offline after the first fetch. |
| `llm.device` | `auto` | Where the local model runs: `auto`, `cpu`, `metal` or `cuda`. `auto` takes the fastest backend built into the binary and falls back to the CPU. macOS builds include Metal. `cuda` is not built, so it is refused with a message. |
| `llm.warmup` | `false` | Load the local model and generate one token at start. Off by default: the model loads on the first call that needs it (an entity page or a local summary), so that first call is slower by the load time (on Metal this includes the first-call kernel compile, about 10 s). Set `true` to pay that cost at start and to stop the daemon at start when the model cannot load. With `false`, a model that cannot load is reported on the first call that needs it, and at start only as one warning when the model file is missing from the cache. See [ADR 0015](../adr/0015-load-the-llm-on-first-use-and-unload-it-when-idle.md). |
| `llm.idle_unload_secs` | `300` | Seconds without a generation after which the local model is unloaded, which returns its 1.0 to 1.3 GB to the system. The next call loads it again. `0` never unloads. An unload happens at most 30 s late. The embedder is never unloaded. For the behaviour of earlier versions set `llm.warmup = true` and `llm.idle_unload_secs = 0`. |
| `llm.context_tokens` | `8192` | The local model's context window. It sizes the llama.cpp context and the budget of `clusters`. |
| `llm.max_concurrent_generations` | `0` | Generations that may run at once. Each one costs about 110 MB of KV cache. `0` tunes the number from a one-time benchmark; a fixed number turns that off. |
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
| `hosted.enabled` | `false` | Allow calls to hosted model providers. |
| `hosted.allowed_hosts` | empty | The only hosts a request may go to, each lowercase with no scheme, port or path. Required when `hosted.enabled` is true. |
| `hosted.request_timeout_secs` | `60` | How long one request may take, answer included. At least `1`. |
| `hosted.max_response_bytes` | `4194304` | The most bytes of a response that are read. A larger one is an error. At least `1`. |
| `hosted.privacy` | empty | The gate every item passes before it goes into a hosted prompt. See [Hosted providers](../concepts/hosted-providers.md). |
| `hosted.privacy.allowed_scopes` | empty | Scopes whose items may be sent. A scope covers its descendants, and `""` is the unscoped group. Empty sends nothing. |
| `hosted.privacy.local_only_sources` | empty | Producers whose items are never sent, in any scope. |
| `hosted.privacy.provider_regions` | empty | The region each provider processes requests in, as you declare it. |
| `hosted.privacy.attested_regions` | empty | For a scope and its descendants, the regions it may be sent to. |
| `hosted.models` | empty | The dated model id each provider is pinned to. An alias is refused. |
| `hosted.prices` | empty | What each pinned model costs. |
| `hosted.prices.version` | empty | Names the price table, so a recorded cost can be tied to its rates. |
| `hosted.prices.models` | empty | Rates by pinned model id, in millionths of a dollar per million tokens: input, output, cached input, batch input and batch output. |
| `hosted.key_files` | empty | A provider's key file, an absolute path only its owner can read. Without one the key is read from the Keychain item `liam-<provider>` (macOS). |
| `producers.unknown_id` | `unknown` | Producer recorded for a client not in the table below. |
| `producers.clients` | empty | Maps a client's declared MCP name to a producer id. Matching ignores case. |

`[producers.clients]` is a plain table, and the ids are what land in the
`producer` column:

```toml
[producers.clients]
claude-code = "claude"
ai-notetaker = "notetaker"
```

The reranker was removed. A `[reranker]` section from an older `liam.toml` is ignored, and the daemon logs one warning at start. You can delete it.

`embedder.device` was removed, because the embedder always runs on the CPU. An old `device` line under `[embedder]` is ignored, and the daemon logs one warning at start. You can delete it.

`ask_sufficiency_check` was removed with the `ask` tool. An old `ask_sufficiency_check` line is ignored, and the daemon logs one warning at start. You can delete it.

`gc.full_synthesis_mention_threshold` and `gc.full_synthesis_max_new_tokens` were removed, because entity pages are no longer written by a model. An old line for either under `[gc]` is ignored, and the daemon logs one warning at start for each. You can delete them.

Logs go to stderr. Stdout carries the MCP JSON-RPC stream, so it stays clean.

The store runs in WAL mode, so libSQL keeps `liam.db-wal` and `liam.db-shm` next
to the database. Anything that copies or backs up the store by path needs all
three: the `.db` file alone can be missing recently committed data.
