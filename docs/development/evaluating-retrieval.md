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
| `--device <DEVICE>` | `auto`, `cpu`, `metal` or `cuda`, instead of `embedder.device`. Applies to the Qwen3 (candle) and GGUF (llama.cpp) models. |
| `--vectors-out <PATH>` | `liam eval embedder` only: write the fact then question vectors to this JSON file, overwriting it, so two runs can be compared. `liam eval retrieval` refuses it. |
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

## Multilingual embedder check

`crates/daemon/src/retrieval_eval/multilingual/` holds an embedder-only check in
eight languages: English, Portuguese, Spanish, Chinese, French, German, Japanese
and Arabic. Each language has 30 queries over 287 facts, and `pairs.json` names
the relevant facts for each query. English is the source. The other seven are
machine translated, so a gap between languages is a signal, not a measure of
translation quality. For each language the check embeds every fact as a
document, embeds each query as a query, ranks all facts by cosine similarity,
and reports the mean nDCG@10 over the 30 queries.

English, Portuguese and Spanish gate, because they are the languages the
project targets. Each may fall at most 0.05 below its baseline, and their mean
at most 0.02. The other five are printed with their change from the baseline
and never fail the check.

The baseline is `baseline-arctic-embed-m-v2-384.json` in the same folder. It
records the embedder id and width it was measured with. The check refuses to
run against a different default embedder or width, and says to re-measure.

The test is `#[ignore]`d because it loads the real model, about 311 MB on the
first run:

```sh
cargo test --release -p liam-daemon --lib -- --ignored --nocapture \
  multilingual::tests::the_default_embedder_meets_the_multilingual_baseline
```

Set `LIAM_EVAL_CACHE_DIR` to keep the model files out of `embedder.cache_dir`.
To re-measure the baseline, for example after changing the default embedder, run
the test and copy the printed `ndcg@10` column, with the new embedder id and
`dims`, into the baseline file.
