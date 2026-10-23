# D42: Qwen3-Embedding-0.6B vs BGE-M3, and whether an ANN index is justified

**Date:** 2026-10-23
**Status:** Measured report
**Implements:** M-L2 WU-11 (decision D42)
**Baseline:** `main` at `658977e`, plus the `liam eval retrieval` command from the same PR (branch `feat/m-l2-s7b-eval-cli`)

## Summary

- On the 80-question synthetic set the two models tie on quality. recall@10 is
  0.975 for both. nDCG@10 is 0.836 (Qwen3) against 0.837 (BGE-M3). Only the
  `VectorSemantic` category differs, by one rank swap in one question. The set
  cannot separate them.
- BGE-M3 embedded faster per call in every run (best 46 ms against 171 ms) but
  the two models run on different runtimes (ONNX for BGE-M3, candle for Qwen3),
  so that is a measurement of the stacks, not of the models.
- BGE-M3 costs more: 2.1 GB of weights against 1.1 GB, and 1024-wide vectors
  against 768-wide ones (4.64 GB against 4.13 GB on disk per million).
- Brute-force search costs about 3.2 ms per 1,000 stored vectors. It crosses a
  100 ms query budget at roughly 30,000 embedded nodes.
- **An ANN index is not justified at LIAM's current scale.** The decision rule
  is in the last section.

## Method and machine

| Item | Value |
|---|---|
| Machine | Apple M1 Pro, 16 GiB RAM |
| OS | macOS 15.8.1 |
| Toolchain | rustc 1.99.1, release build, `--features local` |
| Code | `main` at `658977e` plus the CLI in this PR |
| Date | 2026-10-23 |
| Command | `liam eval retrieval --model <id> [--json]` |
| Corpus | 134 synthetic facts about one fictional company |
| Questions | 80, ten in each of eight categories |
| Embedder calls per run | 144 (134 facts plus 10 `VectorSemantic` queries), one text per call |

**Caveat, read first.** The question set is synthetic and small. LIAM is
pre-release and has no real usage or query logs, so the set stands in for real
usage and is hand written. Most categories are decided by the lexical arm and
score identically for any embedder. Differences of a few thousandths are
noise. Do not read these numbers as a ranking of the models on real memory
content.

**Load on the machine.** The machine was shared and busy (load average between
13 and 36 during the runs), so every wall-clock number below is noisy and
biased upward. Where it matters the tables give the best run and the median
run, and the raw range.

The ranking metrics are deterministic: the Qwen3 report was byte-identical
across three runs and the BGE-M3 report across two, once the wall-clock fields
(`total_ms`, `mean_latency_ms`, `throughput_per_sec`) were removed. The
`calls` and `items` counts are part of the compared output.

## Retrieval quality

Overall, over all 80 questions:

| Metric | Qwen3-Embedding-0.6B (768 dims) | BGE-M3 (1024 dims) |
|---|---|---|
| recall@10 | 0.975 | 0.975 |
| nDCG@10 | 0.836 | 0.837 |
| MRR | 0.817 | 0.819 |
| precision@4 / @8 | 0.231 / 0.131 | 0.231 / 0.131 |
| recall@4 / @8 | 0.838 / 0.944 | 0.838 / 0.944 |

Per category, recall@10 and nDCG@10. Seven categories are identical for the
two models; `VectorSemantic` is the only row that differs:

| Category | recall@10 | nDCG@10 Qwen3 | nDCG@10 BGE-M3 |
|---|---|---|---|
| Lexical | 1.000 | 1.000 | 1.000 |
| VectorSemantic | 1.000 | 0.950 | 0.963 |
| GraphExpansion | 0.900 | 0.831 | 0.831 |
| KindFilter | 1.000 | 1.000 | 1.000 |
| ScopeFilter | 1.000 | 1.000 | 1.000 |
| Decay | 1.000 | 0.527 | 0.527 |
| Confidence | 0.900 | 0.377 | 0.377 |
| AsOf | 1.000 | 1.000 | 1.000 |

Both models find every `VectorSemantic` answer in the top 10. BGE-M3 ranks one
of them slightly higher (MRR 0.950 against 0.933). Ten questions cannot
support a claim from that. The weak rows (`Decay`, `Confidence`,
`GraphExpansion`) are the same for both models and for the text-only baseline
in `retrieval_eval.rs`, so they reflect the ranking pipeline and the small
fixed corpus, not the embedder.

## Embedder latency and throughput

Mean latency of one `embed` call (one text), after an untimed warm-up call.
Throughput is items per second over the timed calls. Runs: Qwen3 seven, BGE-M3
six. Model download and load time sit outside the timed section.

| Statistic | Qwen3 latency | Qwen3 throughput | BGE-M3 latency | BGE-M3 throughput |
|---|---|---|---|---|
| Best run | 170.6 ms | 5.9 items/s | 46.4 ms | 21.6 items/s |
| Median run | 218.5 ms | 4.6 items/s | 67.3 ms | 14.9 items/s |
| Range | 170.6 to 494.7 ms | 2.0 to 5.9 items/s | 46.4 to 271.0 ms | 3.7 to 21.6 items/s |

Reading this: BGE-M3 was 3 to 4 times faster per call in the best and median
runs. That gap is not purely the model. Qwen3 runs through the candle backend
and BGE-M3 through ONNX, and both were single-text calls, which is how the
daemon embeds a write or a query but not how a bulk import would. A batched
re-embed would behave differently and was not measured. Treat the table as the
cost of one embed call in the daemon as it ships.

Peak resident memory of the whole eval process (`/usr/bin/time -l`), three
runs each: Qwen3 1.94 to 2.71 GB (median 2.01 GB), BGE-M3 1.73 to 1.88 GB
(median 1.87 GB). This is the model plus the harness; the in-memory store of
134 facts adds little.

## Disk and RAM per million items

Which parts are measured and which are arithmetic:

| Quantity | Qwen3 (768 dims) | BGE-M3 (1024 dims) | Basis |
|---|---|---|---|
| Model weights on disk | 1.1 GB | 2.1 GB | Measured (`du` of the cache directory) |
| Raw vector bytes | 3,072 B | 4,096 B | Arithmetic: dims x 4 bytes (f32) |
| Bytes a vector adds to the database | 4,128 B | 4,640 B | Measured at 10,000 rows: size of an embedded store minus a text-only store with the same nodes, divided by 10,000 |
| Overhead over raw | 1,056 B (34%) | 544 B (13%) | Arithmetic: row above minus raw |
| Disk per million vectors | 4.13 GB | 4.64 GB | Arithmetic: measured bytes per vector x 1,000,000 |
| RAM per million vectors if fully cached | 4.13 GB | 4.64 GB | Arithmetic: assumes the page cache holds every vector page |
| Process RAM, model loaded | about 2.0 GB | about 1.9 GB | Measured, see above |

The store scans vectors through SQLite pages, so it does not hold a vector
array in process memory. The RAM row is therefore what the operating system
would need to keep to avoid disk reads during a scan, not an allocation the
daemon makes. The per-vector overhead is page packing, and it is not constant
across widths: a 768-wide row leaves slack in a 4 KiB page, and a 1024-wide
row spills into an overflow page. The 10,000 row figure is measured; the
million row figure assumes the same packing continues.

## Brute-force latency against store size

The store has no vector index. `query_explained` computes the distance to every
stored vector for each query. The probe
(`retrieval_eval/scale_probe.rs`, run with `cargo test --release -p liam-daemon
scale_probe -- --ignored --nocapture`) fills a file-backed store with
synthetic unit vectors and times the full `query_explained` call, which covers
the text arm, the vector arm, fusion and graph expansion, with `k = 10`. It
excludes the cost of embedding the query. Each cell is the mean of 30 queries
after 3 warm-up queries; the p95 is in brackets.

| Stored vectors | 768 dims, mean (p95) | 1024 dims, mean (p95) |
|---|---|---|
| 1,000 | 4.4 ms (5.1) | 3.6 ms (3.7) |
| 10,000 | 34.6 ms (44.1) | 38.6 ms (52.0) |
| 25,000 | 83.7 ms (96.7) | 128.3 ms (234.0) |
| 50,000 | 161.9 ms (169.7) | 167.5 ms (173.3) |
| 100,000 | 318.9 ms (339.3) | 338.7 ms (362.1) |

All of these were measured. The 1024-wide 25,000 row is an outlier from machine
load (its p95 is 234 ms); the neighbouring rows fit a straight line. Cost is
linear in row count at about 3.2 ms per 1,000 vectors at 768 dims and about
3.4 ms per 1,000 at 1024 dims.

**Extrapolated, not measured:** one million vectors would take about 3.2 to
3.4 seconds per query. Nothing above 100,000 was run.

## Recommendation on an ANN index

**Not justified at LIAM's current scale.** Reasons, from the numbers:

1. LIAM has no real store to measure. The only corpus on record is the 134 fact
   benchmark set, which is under 0.5% of the size where brute force becomes
   slow.
2. Brute force costs about 3.2 ms per 1,000 vectors. A query stays under
   100 ms until about 30,000 embedded nodes. The 100 ms budget is chosen
   because it sits near BGE-M3's own query embedding cost (46 to 67 ms) and
   below Qwen3's (171 to 219 ms): until the scan costs more than embedding the
   query, an index would not change what a user feels.
3. An index adds a second structure to build, keep consistent with
   `node_vectors` under writes, deletes and the model fingerprint check, and
   rebuild on a model change. That cost is not worth paying while the scan is
   cheaper than the embedder call.

**Decision rule.** Build an ANN index when either holds:

- the number of nodes with embeddings in a real store reaches about 30,000, or
  is projected to within the next planning period; or
- the p95 of `query_explained` on a real store exceeds 100 ms on an idle
  machine.

Re-run the probe on the target machine to check the first condition, since the
30,000 figure was measured on a loaded M1 Pro and an idle machine may cross
later. No follow-up issue is raised because the recommendation is not to build
an index.

## What was not done

- Nothing was run above 100,000 stored vectors; the million row latency is an
  extrapolation.
- Batched embedding throughput was not measured.
- No real user data exists, so quality on real memory content is unknown.
- Latency runs were taken on a busy machine; the best run is the closest
  estimate of the idle cost.

## Model choice

The numbers give no quality reason to change the default. BGE-M3 is faster in
this daemon and smaller in memory, and larger on disk, and it is fixed at
1024 dimensions (no truncation), which raises storage per vector by a third.
Qwen3 at 768 dimensions is already configured. Revisit when a larger, less
synthetic question set exists, since this one cannot rank the two.
