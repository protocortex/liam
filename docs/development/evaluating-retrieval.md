# Evaluating retrieval

[Back to the docs index](../README.md)

```sh
liam eval retrieval --model <ID> [--json]
```

It scores one embedding model on a fixed golden set and measures the embedder's
speed. The model is downloaded into `embedder.cache_dir` on first
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
[`docs/specs/2026-10-23-m-l2-d42-embedder-evaluation.md`](../specs/2026-10-23-m-l2-d42-embedder-evaluation.md).
