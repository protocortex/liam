# ADR-0013: arctic-embed-m-v2.0 is the default embedder, and the reranker is off by default

- **Status:** Accepted. Supersedes in part [ADR-0012](0012-llama-cpp-default-embedder-and-hardware-support.md): the llama.cpp embedder stays, and it is no longer the default.
- **Date created:** 2026-11-17
- **Date modified:** 2026-11-17

## Context

ADR-0012 made Qwen3-Embedding-0.6B at Q8_0 on llama.cpp the default embedder.
The default reranker was `bge-reranker-base`. LIAM's target languages are
English, Portuguese and Spanish, and a later benchmark compared the default
against a small multilingual ONNX encoder and against four rerankers on those
languages.

## Decision Drivers

- **Retrieval quality** in English, Portuguese and Spanish.
- **Latency per query**, since the embedder runs on every `recall` and `remember`.
- **Memory and download size**, since the daemon runs all day.
- **A free licence** for the default model.

## Evidence

Apple M1 Pro, nDCG@10, 384 dimensions.

| Set | arctic-embed-m-v2.0 int8 | Qwen3 Q8 |
|---|---|---|
| English, 340 questions | 0.942 to 0.949 | 0.932 |
| Portuguese | 0.966 | 0.935 |
| Spanish | 0.931 | 0.924 |

Arctic answers a query in about 8 ms where Qwen3 Q8 takes 37 ms, and its file is
311 MB where the Qwen3 Q8 file is about 640 MB. Qwen3 is better on Arabic and
Chinese: the eight language mean is 0.939 for Qwen3 against 0.905 to 0.913 for
arctic. Arctic is Apache-2.0.

The best reranker moved nDCG@10 on English, Portuguese and Spanish by +0.017,
-0.010 and +0.033. The benchmark noise floor is 0.03, so none of these is a
clear gain. It costs 254 ms per query and over 1 GB of memory.

Caveats. The eight language sets are machine translated and hold 30 questions
each, so a small difference in them is weak evidence. The memory figures (arctic
alone 930 to 990 MB resident, a reranker adds about 1.2 GB) come from a Rust
probe on an M1 Pro, not from a long running daemon.

## Considered Alternatives

### Keep Qwen3 Q8 on llama.cpp as the default (effort: none)

- Trade-offs: best on Arabic and Chinese, which are not target languages. It is
  4.6 times slower per query and twice the file size, with no gain in the three
  target languages. Rejected as the default, kept as a choice.

### arctic-embed-m-v2.0 int8 on ONNX, CPU (effort: S, chosen)

- Trade-offs: loads through the ONNX path that LIAM already ships for smaller
  models. fastembed has no variant for v2.0, so the files are pinned by revision,
  size and sha256 and downloaded into `embedder.cache_dir`. Weaker than Qwen3 on
  Arabic and Chinese.

### Keep the reranker on by default (effort: none)

- Trade-offs: no behaviour change. But it costs 254 ms and over 1 GB for a gain
  inside the noise on the target languages. Rejected.

## Decision

1. **The default `embedder.model` is `Snowflake/snowflake-arctic-embed-m-v2.0`**,
   the int8 ONNX file at revision `95c2741480856aa9666782eb4afe11959938017f`, on the
   CPU. It uses CLS pooling, the `query: ` prefix on queries and none on documents,
   and at most 512 input tokens (the model allows 8192; 512 bounds latency). Its
   native width is 768 and `embedding_dims` (384) cuts it by Matryoshka truncation
   and normalises again.
2. **The reranker is off by default.** `reranker.model = "none"` keeps the order
   the store returns and loads nothing. `bge-reranker-base`,
   `gte-multilingual-reranker-base` and `bge-reranker-v2-m3` stay selectable.
3. **Qwen3 stays selectable** by its model id, on llama.cpp
   (`Qwen/Qwen3-Embedding-0.6B-GGUF`) or candle (`Qwen/Qwen3-Embedding-0.6B`).
   Choose it if Arabic or Chinese matter.

## Consequences

- **Rebuild.** The model id is part of the embedder fingerprint, so a store built
  with the old default is refused at start. The message names `liam rebuild`.
- **Smaller and faster by default.** About 311 MB to download, and no reranker
  memory.
- **Weaker for Arabic and Chinese** unless the user picks Qwen3.
- **No change in recall order from a reranker.** With `none`, `recall` returns
  the retrieval order and its stage timer still runs.
- **A pinned download path.** A new model that fastembed does not list is one
  table row with its file sizes and hashes.

## Addendum, 2026-11-17: a reranker on a store of 10,000 passages

The evidence above came from small stores. A later run used 10,000 Wikipedia passages from MIRACL dev (`mteb/MIRACLRetrievalHardNegatives`, CC BY-SA 4.0), with 200 English, 100 Spanish and 100 Japanese queries. Each reranker reordered the top 16 results of the default arctic embedder. The mean nDCG@10 over the three languages was:

| Reranker | Mean nDCG@10 |
|---|---|
| none | 0.903 |
| `gte-multilingual-reranker-base` | 0.940 |
| `bge-reranker-v2-m3` | 0.954 |

For English the gain was +0.031 (gte) and +0.041 (bge). For Spanish and Japanese the mean gain was +0.040 (gte) and +0.057 (bge). Paired bootstrap 95% intervals were above zero for both rerankers in every language. `bge-reranker-v2-m3` beat gte in every language, and a pool of 16 beat a pool of 8. The embedder alone already found the right passages (recall at 16 was 0.98 to 0.99), so the reranker improves their order.

The cost, from a probe on an Apple M1 Pro on a machine that was not quiet: about 1.2 GB of resident memory, and about 254 ms per recall for gte at 16 pairs. `bge-reranker-v2-m3` took about 779 ms per 16 pairs in an earlier measurement and has a 571 MB file. The data is Wikipedia text, not agent memory, and the run did not cover Portuguese at this size.

Consequences:

- The default stays `none`, by the user's choice.
- A large store should opt in to `bge-reranker-v2-m3`. The [configuration reference](../reference/configuration.md#when-to-turn-the-reranker-on) says how.
- A reranker that is turned on must load on demand and unload when idle, so it costs memory only while in use. This is required follow-up work and is not built yet: today a configured reranker loads at daemon start.
