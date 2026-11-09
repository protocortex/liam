# llama.cpp embedder benchmark

[Back to the specs](.) | Related: [Embedder speed evaluation](2026-11-06-embedder-speed-evaluation.md)

## Summary

LIAM can now run Qwen3-Embedding-0.6B in process on llama.cpp from a GGUF file,
on the CPU or on the Apple GPU (Metal). On this machine every variant passes the
quality rule against the candle CPU baseline, and the speed gap is large. The
best variant answers a query in about 22 ms, against 183 ms for candle on the
CPU: eight times faster, with a third less memory.

## Machine and method

- Apple M1 Pro (8 performance cores, 2 efficiency cores), 16 GiB, macOS 15.8.1.
- llama-cpp-2 0.1.154 (llama.cpp bundled with it, Metal built in on macOS),
  candle-core 0.11.1, rustc 1.99.1, release build.
- Command per run: `liam eval embedder --model <ID> --dims 384 --device <D> --json
  --vectors-out <file>` under `/usr/bin/time -l`. 300 facts, 340 questions,
  384 dimensions (Matryoshka cut of the 1024 wide vector).
- Each of the six variants ran three times, interleaved in rounds (variant a to
  f, then again), so a burst of background load hits every variant alike. Tables
  show the median of the three runs. Quality metrics repeated exactly across
  the three runs of each variant.
- Load: the machine was not quiet. One-minute load average before each run was
  5 to 8, with spikes to 12 to 14 on a few runs. Spotlight and `mediaanalysisd`
  (system background work that cannot be stopped without privileges) kept
  cores busy. The three runs per row agree within a few percent, which shows
  the noise is small next to the gaps. The raw per-run load lines are in the
  artifacts.
- Not measured: a cold file cache start. `sudo purge` is not available, so the
  load figures are warm page cache loads. Candle loads 2.4 GB of F32
  safetensors, which a cold read would slow far more than the 0.6 to 1.2 GB
  GGUF files.
- Peak RSS is `ru_maxrss` of the benchmark process itself and includes the
  embedded benchmark data and the store, not just the model.
- Cosine: the 300 fact vectors and 340 question vectors of each variant against
  the same texts from candle CPU (first run of each).
- Batch items per second is `embed_batch` over the 300 facts. Candle embeds one
  item at a time (the trait default). llama.cpp packs up to 32 texts or 512
  tokens into one decode, so its batch figure includes a real batching gain
  and is not a like for like kernel comparison.

## Results

Decision rule: within 0.02 nDCG@10 and 0.05 paraphrase nDCG@10 of candle CPU.

| Variant | Device | nDCG@10 | Para. nDCG@10 | recall@1/5/10 | MRR | Load ms | Query median ms | Query p95 ms | Batch items/s | Peak RSS MB | Mean cosine vs candle CPU | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| candle Qwen3 F32 | cpu | 0.933 | 0.826 | 0.836/0.974/0.982 | 0.920 | 2562 | 183.3 | 196.5 | 5.5 | 2596 | 1.0000 (min 1.0000) | baseline |
| candle Qwen3 F32 | metal | 0.933 | 0.826 | 0.836/0.974/0.982 | 0.920 | 2570 | 71.8 | 76.6 | 14.4 | 2299 | 1.0000 (min 1.0000) | pass |
| llama.cpp Q8_0 | cpu | 0.934 | 0.831 | 0.838/0.971/0.982 | 0.920 | 945 | 36.5 | 45.9 | 32.1 | 1808 | 0.9993 (min 0.9982) | pass |
| llama.cpp Q8_0 | metal | 0.934 | 0.830 | 0.839/0.974/0.982 | 0.922 | 215 | 23.2 | 32.7 | 83.5 | 1072 | 0.9997 (min 0.9994) | pass |
| llama.cpp F16 | cpu | 0.933 | 0.828 | 0.835/0.974/0.982 | 0.919 | 676 | 61.6 | 73.1 | 19.3 | 1899 | 1.0000 (min 0.9999) | pass |
| llama.cpp F16 | metal | 0.933 | 0.826 | 0.836/0.974/0.982 | 0.920 | 243 | 21.8 | 30.3 | 90.1 | 1597 | 1.0000 (min 1.0000) | pass |

Per run values (three runs): query median ms and batch items/s are within 5
percent of the median in every row except candle CPU (178 to 192 ms).

Per category nDCG@10, candle CPU against llama.cpp Q8_0 Metal: config 0.994 and
0.994, confusable 0.975 and 0.975, multi 0.934 and 0.933, paraphrase 0.826 and
0.830, rationale 0.985 and 0.985, symbol 0.978 and 0.978. The stored baseline
in the speed evaluation (nDCG@10 0.933, MRR 0.920) is reproduced.

### Reading the table

- Every row passes the rule. The largest nDCG@10 gap to candle CPU is 0.001 and
  the largest paraphrase gap is 0.005.
- Numerical agreement: F16 matches candle to four decimals. Q8_0 differs by
  0.0003 to 0.0007 on average (worst vector 0.9982), which moves no ranking that
  matters here. Candle on Metal returns the same vectors as candle on the CPU.
- Lowest query latency among passing rows: llama.cpp F16 on Metal, 21.8 ms
  median and 30.3 ms p95. Q8_0 on Metal is within 1.4 ms of it, loads faster,
  and uses 525 MB less memory, so it is the practical pick and the default
  quantisation for the bare model id.
- Candle on Metal works for this model: no crash, identical vectors, 2.5 times
  faster per query and 2.6 times faster on the batch. Candle `auto` therefore
  now resolves to Metal on macOS (CPU if no Metal device opens).
- llama.cpp on the CPU beats candle on the CPU by 5 times per query for Q8_0,
  because it uses quantised kernels and eight threads on the performance cores.
- The in-process llama.cpp CPU rows use only the performance cores. With all
  ten threads a query took 62 ms instead of 37 ms, because the slow
  efficiency cores gate each parallel step.
- Memory: peak RSS of candle F32 is 2.3 to 2.6 GB. Q8_0 on Metal is 1.07 GB.

## What was built

- `LlamaEmbedder` in `crates/liam-model/src/llama_embedder.rs`: llama.cpp
  context with embeddings on and last token pooling, the Qwen3 query
  instruction, plain documents, L2 normalise then Matryoshka cut and normalise
  on the host. One context is kept and its KV cache cleared before each decode.
  Building and freeing a Metal context cost about 10 ms each, which was the
  difference between 40 ms and 23 ms per query in an early version.
- Model ids: `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0) and
  `Qwen/Qwen3-Embedding-0.6B-GGUF:F16`, with `:Q8_0` also accepted. The files
  are pinned by size and sha256, and `liam fetch-models` verifies them.
- `embedder.device`: `auto`, `cpu`, `metal`, `cuda`. Candle and llama.cpp honour
  it. It is not in the embedder fingerprint; the model id is, so switching to a
  GGUF id needs `liam rebuild`.
- `liam eval embedder --device` and `--vectors-out`; the report and JSON now
  carry backend, device and peak RSS.
- llama-cpp-2 stays at 0.1.154: its embedding API (embeddings flag, pooling type,
  `embeddings_seq_ith`) is present, so no bump was needed.

## Neural Engine experiment (results only, nothing shipped)

Setup: a scratch binary (not in the repo) running `BAAI/bge-small-en-v1.5`
through fastembed 5.17.3 and ort 2.0.0-rc.12 with the `coreml` cargo feature and
`CoreML::default().with_compute_units(..)` passed to
`InitOptions::with_execution_providers`. The CoreML provider is exposed and
works. Same benchmark data, one text per call, same prefixes as LIAM's ONNX
path, one run each (a few repeats agreed within 3 percent). The scratch scoring
code is its own small copy, so its nDCG@10 (0.906) is comparable across the
rows below but not with the table above.

| ORT provider | Query median ms | Query p95 ms | Facts items/s | user + sys CPU s | nDCG@10 | Cosine vs CPU |
|---|---|---|---|---|---|---|
| CPU | 7.2 | 8.1 | 117 | 20.1 | 0.906 | 1.0 |
| CoreML, CPU and Neural Engine | 36.2 | 44.2 | 25 | 39.6 | 0.906 | 1.0000 |
| CoreML, CPU only | 37.4 | 46.2 | 25 | 39.9 | 0.906 | n/a |
| CoreML, CPU and GPU | 44.3 | 50.5 | 20 | 42.7 | 0.906 | 0.99999999 |

Did CoreML engage: the provider loaded and ran, but the Neural Engine did not do
the work. The CPU and Neural Engine run is as slow as the CPU only run (36 ms
against 37 ms), and its vectors match the CPU to twelve decimals, where fp16
Neural Engine arithmetic would differ in the third or fourth decimal.
`powermetrics` needs sudo, and I did not capture ORT's node assignment log, so
this is indirect evidence. The likely cause is dynamic shapes: fastembed pads
each call to its own length, CoreML compiles per shape, and the provider keeps
dynamic nodes on the CPU path.

Verdict: not usable. It is 5 times slower per query and uses twice the CPU time
of plain ORT on the CPU, with no quality change. A fixed padded length with
static shapes might engage the Neural Engine, but that needs a custom
tokenizer path and was outside the time box. `onnx-community/Qwen3-Embedding-0.6B-ONNX`
was not tried: it would need last token pooling code outside fastembed, and the
dynamic shape problem would apply to it as well.

## Open points

- Cold cache load is not measured.
- Candle batches one item at a time; a batched candle path could narrow the
  batch throughput gap, which is why that column should not be read as raw
  engine speed.
- Quantised Q8_0 vectors differ slightly from F32, so a store built with candle
  should be rebuilt (`liam rebuild`) after switching model id, which the
  fingerprint already forces.

## Artifacts

Raw per run JSON, vectors, `/usr/bin/time -l` output and load lines are in
`/private/tmp/claude-501/-Users-isantos-Workspace-protocortex-liam/68ec758d-d203-418b-8aa9-9ce7dfd3e0b6/scratchpad/bench/`
(`runs/`, `summary.json`, `matrix.sh`, `analyze.py`).
