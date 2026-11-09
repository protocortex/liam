# ADR-0012: llama.cpp is the default embedder, and Linux and Windows GPUs use a pinned llama-server sidecar

- **Status:** Accepted
- **Date created:** 2026-11-10
- **Date modified:** 2026-11-10

## Context

LIAM embeds every memory and every query, so the embedder sets how fast `recall`
feels, how much memory the daemon holds and which machines get a GPU.

- The previous default ran Qwen3-Embedding-0.6B on candle. Candle supports three
  backends: the CPU, CUDA and Metal. AMD ROCm support is an open candle pull
  request (#3424) that is not released and is fragile, and Intel GPUs have no
  candle backend at all.
- In practice the candle embedder was loading on the CPU. On the benchmark
  machine (Apple M1 Pro) it answered a query in 183 ms at the median, held 2596 MB
  at peak and took 2.6 s to load.
- LIAM already ships llama.cpp in the daemon for generation. Pull request #347
  added a llama.cpp GGUF embedder and an `embedder.device` setting, and a
  benchmark (`docs/specs/2026-11-09-llama-embedder-benchmark.md`) that scored
  both against the same questions.
- The release today is macOS arm64 only, so Linux and Windows hardware is a
  future concern. Nobody on the team can test an AMD, Intel or NVIDIA card in
  this setup.

## Decision Drivers

- **Latency per query.** The embedder runs on every `recall` and `remember`.
- **Memory.** The daemon runs all day next to an editor and a browser.
- **Retrieval quality must not drop.** The benchmark rule is within 0.02 nDCG@10
  and 0.05 paraphrase nDCG@10 of the candle CPU baseline.
- **No dependence on unreleased upstream work** for a shipped path.
- **Hardware we cannot test must not ship as a silent default.**

## Evidence

Apple M1 Pro, 300 facts, 340 questions, 384 dimensions, three interleaved runs
per row, medians. From the benchmark spec:

| Variant | Device | nDCG@10 | Query median ms | Load ms | Peak RSS MB |
|---|---|---|---|---|---|
| candle F32 | cpu | 0.933 | 183.3 | 2562 | 2596 |
| candle F32 | metal | 0.933 | 71.8 | 2570 | 2299 |
| llama.cpp Q8_0 | cpu | 0.934 | 36.5 | 945 | 1808 |
| llama.cpp Q8_0 | metal | 0.934 | 23.2 | 215 | 1072 |
| llama.cpp F16 | metal | 0.933 | 21.8 | 243 | 1597 |

Q8_0 on Metal is 7.9 times faster per query than candle on the CPU, loads in 0.2 s
instead of 2.6 s and holds 59 percent less memory at the embedder level, with the
same quality (mean cosine to the candle vectors 0.9997). Candle on Metal, which
`embedder.device = auto` now selects for candle on macOS, is faster than candle
on the CPU. Against it, Q8_0 on Metal is 3 times faster (23 ms against 72 ms) and
holds 53 percent less memory at the embedder level (1072 MB against 2299 MB).

The switch was also run on a real throwaway store of 300 memories (see the
addendum in the spec). The daemon refused the old store and named `liam rebuild`.
The rebuild re-embedded 300 nodes in 17.6 s. Seven of eight fixed queries
returned the same top five before and after, and the eighth changed one hit. Start
to first successful `recall` fell from about 4.0 s to 1.7 s. The whole daemon
process peaked at about 2.6 GB both before and after, because the reranker and the
store sit in it, so the memory saving shows at the embedder and not yet at the
process.

## Considered Alternatives

### Stay on candle and accept CPU only on AMD and Intel (effort: none)

- Trade-offs: nothing to build. But AMD and Intel machines would run the slowest
  option measured, and the AMD path would wait on an unreleased upstream change.
  Rejected.

### ONNX Runtime with execution providers (effort: M)

- The CoreML provider was tried on Apple Silicon with bge-small through fastembed
  and ort. It loaded and ran, but engaged nothing on the Neural Engine: the run
  that allowed the Neural Engine took 36.2 ms per query, the same as CPU only
  (37.4 ms), and it was 5 times slower than plain ORT on the CPU (7.2 ms). Its
  vectors matched the CPU to twelve decimals. The likely cause is dynamic shapes.
- Trade-offs: other providers (DirectML, ROCm, OpenVINO) have the same shape and
  packaging problems, and each needs its own runtime build. Rejected for now.
  The follow-up to revisit the Neural Engine is #351.

### llama.cpp in process with dynamic backend libraries downloaded at runtime (effort: L)

- Trade-offs: it would give one binary GPU support everywhere. But the prebuilt
  ggml backend libraries must match the llama.cpp that `llama-cpp-sys-2` vendors
  exactly, down to the ABI, or the process crashes. Pinning two moving parts that
  must stay in lockstep is fragile. Rejected.

### llama.cpp in process on Metal and CPU, with a pinned llama-server sidecar for other GPUs (effort: M, chosen)

- Trade-offs: the in-process path is the fastest and simplest where we can test it.
  A sidecar process adds a lifecycle to manage, but it has a stable HTTP interface
  (`/v1/embeddings`), it can be pinned by sha256 and it never has to match the
  in-process ABI.

## Decision

1. **The default `embedder.model` is `Qwen/Qwen3-Embedding-0.6B-GGUF`** (Q8_0),
   run in process on llama.cpp. `embedder.device = auto` is Metal on macOS and the
   CPU elsewhere. The reranker is unchanged.
2. **Candle stays selectable** as the explicit model id `Qwen/Qwen3-Embedding-0.6B`.
   It is opt in and no longer downloaded by `liam fetch-models` unless configured.
3. **Hardware support for Linux and Windows GPUs will use a pinned `llama-server`
   sidecar** that serves embeddings and generation. The build variant is chosen by
   detection (`liam devices`) and downloaded on first run, with its sha256 pinned
   in a manifest. Vulkan comes first: one Vulkan build covers AMD, Intel and NVIDIA
   from a single download. CUDA and ROCm are optional faster builds for those
   cards.
4. **Nothing in the sidecar path ships as a default until a tester has verified it
   on real hardware** of that kind.

## Consequences

- **Fingerprint change and rebuild.** The model id is part of the embedder
  fingerprint, so a store built with the old default is refused at start. The
  message names `liam rebuild`. Users who want to keep candle set
  `embedder.model = "Qwen/Qwen3-Embedding-0.6B"`. Q8_0 vectors differ slightly from
  F32 ones, so a rebuild is needed in either direction.
- **Larger native dependency.** The daemon already carries llama.cpp for
  generation. It now also depends on it for every embedding, so a llama.cpp bug
  reaches search as well as generation.
- **Untestable here.** AMD, Intel and NVIDIA paths cannot be run in this setup.
  They ship only behind explicit verification, which needs testers (#349).
- **Smaller download.** The default fetch is a 0.6 GB GGUF instead of 2.4 GB of
  safetensors.
- **More surface to explain.** `liam devices` exists so users and bug reports can
  show what a machine and a build can do.

Follow-up issues: the sidecar backend (#348), verification on AMD, Intel and
NVIDIA hardware (#349), Linux and Windows release artifacts (#350), the Neural
Engine revisit (#351) and removing the `cuda` feature once CUDA comes from a
downloaded runtime (#352).

## Blueprint

1. Default switch and docs (done in the PR that introduced this record).
2. `liam devices` (done).
3. A sidecar manifest format with url, size and sha256 per platform and variant, and
   the download and verify code (#348).
4. An embeddings client over `/v1/embeddings` and the sidecar lifecycle (#348).
5. Variant selection from `liam devices` detection (#348).
6. Linux and Windows release artifacts (#350).
7. Tester matrix and results on real GPUs (#349).
8. Remove the `cuda` feature (#352).
