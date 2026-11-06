# Smaller and faster embedders against Qwen3-Embedding-0.6B

**Date:** 2026-11-06
**Status:** Measured report
**Follows:** `2026-10-23-m-l2-d42-embedder-evaluation.md` (D42)
**Baseline:** `main` at `2f78bab`, plus the code in the same series of pull requests

## Summary

- **Keep Qwen3-Embedding-0.6B as the default.** On the new 340 question set it is
  clearly ahead: nDCG@10 0.953 against 0.906 for the best small model
  (bge-small-en-v1.5). The D42 set of 80 questions still ties every model at
  recall@10 0.975, as D42 warned, so it cannot be used to choose.
- **Small models are far cheaper.** Query embedding takes 2 to 8 ms (Qwen3: 148 to
  197 ms), the process peaks at 0.2 to 0.35 GB (Qwen3: 2.2 to 3.0 GB), weights are
  87 to 128 MB (Qwen3: 1.1 GB), a warm load takes 75 to 141 ms (Qwen3: 2.8 to
  4.3 s), and a vector is 1,536 bytes (Qwen3: 3,072).
- **The quality cost is real and sits in paraphrased questions.** The best small
  models lose about 0.05 nDCG@10 and 0.05 recall@1 overall, and about 0.11 nDCG@10
  on `paraphrase`. Lookups by symbol, config value and rationale are within 0.02
  of Qwen3 or equal.
- **Recommendation: offer a `fast` profile, do not change the default.** The
  profile is `embedder.model = "BAAI/bge-small-en-v1.5"` with `embedding_dims =
  384`. This series adds the loaders; the default is untouched. Switching an
  existing store needs `liam rebuild`.
- **Qwen3 at 384 dimensions beats every small model on quality** (nDCG@10 0.933)
  and halves the vector size, but costs the same time and memory as Qwen3 at 768.
  That is the cheaper way to shrink the store, if speed is not the problem.

## Method and machine

| Item | Value |
|---|---|
| Machine | Apple M1 Pro, 16 GiB RAM |
| OS | macOS 15, Darwin 24.6.0 |
| Toolchain | release build, `--features local` |
| Commands | `liam eval embedder --model <id> [--dims N] --json`, `liam eval retrieval --model <id>` |
| Memory | `/usr/bin/time -l`, maximum resident set size of the whole process |
| Corpus | 300 facts about a fictional invoicing platform (Rust, TypeScript, Python), hand written |
| Questions | 340, six categories: paraphrase 100, config 60, symbol 50, rationale 50, confusable 50, multi 30 |
| Runs | 6 per model at native width for timing; 1 per truncated width (quality is deterministic) |
| Date | 2026-11-06 |

The new set is built to separate models. It has about 24 clusters of sibling
facts that differ in one detail (a port, a crate, a threshold), questions
written to avoid the fact's own nouns (`paraphrase`), and questions that need
two or three facts (`multi`). Two authors wrote the facts and questions; every
relevant id was checked against the fact by its author, not by a second person
or a model-independent method. Some questions may have an unlisted fact that
also answers them. That would depress every model's score alike.

The benchmark skips the store and the text arm. It embeds each fact (one
`embed_batch` call, document role), embeds each question alone (query role), and
ranks facts by cosine similarity. recall@k, nDCG@10 and MRR use the same
functions as `liam eval retrieval`. The ranking metrics are deterministic for a
given model and width.

Query prefixes follow each model card: `Represent this sentence for searching
relevant passages: ` for bge-small and Arctic, `search_query: ` and
`search_document: ` for nomic, `query: ` and `passage: ` for e5, none for MiniLM.
Qwen3 uses its `Instruct:` format, BGE-M3 none.

**Load on the machine.** The machine was shared and busy. Load average (1 minute)
at the start of each run was 5 to 14 for the small models and 14 to 47 for Qwen3
and 8 to 32 for BGE-M3, so timings are biased upward, Qwen3 the most. Tables give
the best run and the median over runs.

## Retrieval quality (340 questions)

Sorted by nDCG@10. Width is the vector width used.

| Model | Width | recall@1 | recall@5 | recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 768 | 0.866 | 0.982 | 0.994 | 0.953 | 0.942 |
| Qwen3-Embedding-0.6B | 1024 | 0.863 | 0.982 | 0.991 | 0.949 | 0.938 |
| Qwen3-Embedding-0.6B | 512 | 0.842 | 0.981 | 0.991 | 0.942 | 0.928 |
| Qwen3-Embedding-0.6B | 384 | 0.836 | 0.974 | 0.982 | 0.933 | 0.920 |
| Qwen3-Embedding-0.6B | 256 | 0.826 | 0.963 | 0.978 | 0.922 | 0.907 |
| BGE-M3 | 1024 | 0.828 | 0.950 | 0.971 | 0.919 | 0.908 |
| bge-small-en-v1.5 | 384 | 0.814 | 0.947 | 0.967 | 0.906 | 0.894 |
| nomic-embed-text-v1.5 | 768 | 0.809 | 0.945 | 0.963 | 0.905 | 0.891 |
| all-MiniLM-L12-v2 | 384 | 0.791 | 0.958 | 0.981 | 0.905 | 0.886 |
| snowflake-arctic-embed-s | 384 | 0.781 | 0.945 | 0.977 | 0.900 | 0.882 |
| snowflake-arctic-embed-xs | 384 | 0.796 | 0.941 | 0.966 | 0.899 | 0.883 |
| multilingual-e5-small | 384 | 0.793 | 0.929 | 0.963 | 0.894 | 0.878 |
| all-MiniLM-L6-v2 | 384 | 0.776 | 0.942 | 0.973 | 0.890 | 0.873 |
| nomic-embed-text-v1.5 | 384 | 0.810 | 0.939 | 0.954 | 0.901 | 0.889 |
| nomic-embed-text-v1.5 | 256 | 0.794 | 0.927 | 0.947 | 0.889 | 0.877 |
| nomic-embed-text-v1.5 | 128 | 0.762 | 0.893 | 0.926 | 0.861 | 0.849 |
| Qwen3-Embedding-0.6B | 128 | 0.757 | 0.939 | 0.967 | 0.881 | 0.860 |

nDCG@10 per category (native width; Qwen3 at 768):

| Model | paraphrase | config | symbol | rationale | confusable | multi |
|---|---|---|---|---|---|---|
| Qwen3 | 0.881 | 0.994 | 0.993 | 0.985 | 0.985 | 0.935 |
| BGE-M3 | 0.775 | 0.994 | 0.978 | 0.993 | 0.985 | 0.917 |
| bge-small | 0.767 | 0.975 | 0.973 | 0.969 | 0.993 | 0.874 |
| nomic v1.5 | 0.738 | 0.994 | 0.985 | 0.983 | 0.975 | 0.900 |
| MiniLM-L12 | 0.768 | 0.984 | 0.953 | 0.959 | 0.981 | 0.901 |
| MiniLM-L6 | 0.745 | 0.994 | 0.940 | 0.949 | 0.958 | 0.870 |
| Arctic s | 0.751 | 0.982 | 0.953 | 1.000 | 0.941 | 0.907 |
| Arctic xs | 0.725 | 0.988 | 0.968 | 0.990 | 0.973 | 0.914 |
| e5-small | 0.708 | 0.994 | 0.975 | 1.000 | 0.954 | 0.906 |

Reading this: every model finds config values, symbols and rationales almost
perfectly. The gap is paraphrase (Qwen3 0.881, best small 0.775) and multi fact
questions. That matches the use case: an agent asking in words unlike the stored
note is where a larger model helps. No confidence intervals or paired tests were
computed. With 340 questions, differences under about 0.01 nDCG@10 between the
small models should be treated as ties; the 0.05 gap to Qwen3 is five times that
but was not tested formally.

Truncation: Qwen3 loses 0.020 nDCG@10 from 768 to 384 and 0.031 to 256, a smooth
decline, and falls off at 128 (0.072). nomic loses 0.004 to 384 and 0.016 to 256.
The arctic, MiniLM, bge-small and e5 models are not Matryoshka models and are
refused below their native width. nomic is truncated without the layer norm its
card recommends before truncating, so its truncated scores may be slightly low.

## Quality on the D42 set (80 questions)

`liam eval retrieval` for each model: recall@10 is 0.975 for every model. nDCG@10
ranges from 0.824 (e5-small) to 0.837 (BGE-M3, nomic), and Qwen3 is 0.836. The
set cannot rank these models, which is what D42 said.

## Speed and memory

Query latency is the time of one single-text `embed` call (query role) over 340
questions per run, after one untimed warm-up. "Best" is the fastest single call in any of
the 6 runs; "median" is the median of the six per-run medians.
Batch throughput is one `embed_batch` of 300 facts. Load is the wall time to
construct the embedder in a fresh process; "warm" is the median of runs 2 to 6.

| Model | Query best (ms) | Query median (ms) | Batch best (items/s) | Batch median | Warm load median (ms) | Peak RSS min / median (MB) |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 2.1 | 3.2 | 414 | 398 | 75 | 218 / 224 |
| snowflake-arctic-embed-xs | 2.8 | 4.2 | 414 | 362 | 91 | 217 / 275 |
| all-MiniLM-L12-v2 | 4.2 | 6.3 | 210 | 209 | 125 | 269 / 315 |
| multilingual-e5-small | 4.8 | 6.9 | 201 | 200 | 764 | 1284 / 1308 |
| snowflake-arctic-embed-s | 5.5 | 7.6 | 225 | 211 | 140 | 262 / 349 |
| bge-small-en-v1.5 | 5.7 | 7.8 | 211 | 188 | 141 | 281 / 330 |
| nomic-embed-text-v1.5 | 12.4 | 16.2 | 50 | 43 | 407 | 943 / 984 |
| BGE-M3 | 26.3 | 41.4 | 21 | 14 | 1591 | 1701 / 1872 |
| Qwen3-Embedding-0.6B (768) | 147.9 | 196.7 | 5 | 5 | 4281 (best 2769) | 2159 / 3026 |

Disk and vectors:

| Model | Params | Weights on disk | Native width | Bytes per vector at native width | License | Max input tokens | Matryoshka |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22.7M | 87 MB | 384 | 1,536 | Apache-2.0 | 256 | no |
| snowflake-arctic-embed-xs | 22.6M | 87 MB | 384 | 1,536 | Apache-2.0 | 512 | no |
| all-MiniLM-L12-v2 | 33.4M | 128 MB | 384 | 1,536 | Apache-2.0 | 128 | no |
| bge-small-en-v1.5 | 33.4M | 128 MB | 384 | 1,536 | MIT | 512 | no |
| snowflake-arctic-embed-s | 33.2M | 128 MB | 384 | 1,536 | Apache-2.0 | 512 | no |
| multilingual-e5-small | 117.7M | 465 MB | 384 | 1,536 | MIT | 512 | no |
| nomic-embed-text-v1.5 | 136.7M | 523 MB | 768 | 3,072 | Apache-2.0 | 8192 | yes |
| BGE-M3 | about 568M | 2.1 GB | 1024 | 4,096 | MIT | 8192 | no |
| Qwen3-Embedding-0.6B | about 600M | 1.1 GB | 1024, run at 768 | 3,072 | Apache-2.0 | 8192 here | yes |

Weights are the `du` of the fastembed cache directory for the ONNX models (fp32
files plus tokenizer). Parameter counts and licenses are from the Hugging Face
model pages on 2026-11-06. Runtime: all ONNX models run through fastembed 5.17.3
(ONNX Runtime) except Qwen3, which runs on candle. Vector bytes are arithmetic
(width times 4). A 384 wide store is half the vector bytes of a 768 wide one.
The scan cost in D42 was linear in rows at about 3.2 ms per 1,000 vectors at 768
wide; scan time at 384 wide is inferred to be about half, not measured.

What the timing tables can and cannot say:

- Qwen3 (candle) and the rest (ONNX Runtime) are different stacks, so the speed
  gap mixes model size and runtime, as in D42. What matters for the choice is the
  cost of each model in the daemon as it ships.
- The numbers for Qwen3 were taken under the heaviest load (1 minute load average
  14 to 47), so its latency and RSS (D42 saw 2.0 GB median) are biased upward.
  Even its best run (148 ms) is 25 times slower than the best small model run.
- Cold load was not measured. A true cold start needs the OS file cache emptied,
  which needs `sudo purge`, not available here. The first load of each candidate
  happened right after download and includes the download time (6.6 to 18.5 s), so
  it is not a load time. The first Qwen3 and BGE-M3 loads were 3.0 s and 3.1 s with
  files that had been on disk for weeks, but the page cache state is unknown.
- multilingual-e5-small is slow to load and large in memory (1.3 GB) for its
  parameter count. Its 250,000 token vocabulary dominates: the word embedding
  matrix is most of the 118M parameters.

## Candidates and why some were not run

Runs in our stack and measured: the seven ONNX models above (fastembed 5.17.3's
supported list, read from `src/models/text_embedding.rs` in the pinned crate), BGE-M3
and Qwen3.

Not measured:

| Model | Reason |
|---|---|
| EmbeddingGemma-300m | In the pinned fastembed list, but its license is the Gemma Terms of Use, not a permissive one, so it fails the redistribution rule. Vendor blogs report MTEB English retrieval 55.7 and 2,048 tokens; not verified here. |
| EmbeddingGemma 2 | Vendor and blog pages say Apache-2.0, released October 2026. Not in the pinned fastembed, not verified, not tested. |
| granite-embedding-small-english-r2 | Apache-2.0, 47.7M parameters, 384 wide, 8,192 tokens. The Hugging Face repo has no ONNX file, so it needs an export and fastembed's user-defined model path. A strong candidate to test next. |
| Jina embeddings v5 | Weights are CC BY-NC 4.0 according to the pages found; not redistributable. |
| Snowflake arctic-embed-m-v2.0, nomic-embed-text-v2-moe | 305M and 475M parameters; not small, and v2-moe uses fastembed's candle path. Not tested. |
| gte-base-en-v1.5, quantized (`Q`) variants | In the pinned list, but not run for time. Quantized int8 variants might cut latency and size further. |
| Qwen3 smaller variants | None exist: 0.6B is the smallest Qwen3-Embedding. |
| A public benchmark slice | Not done. BEIR sets that fit this use (SciFact, NFCorpus) carry non-commercial or unclear licenses, and no clean coding memory benchmark was found in the time. |

Statements about models not run come from web pages (blogs and model cards) and
were not independently checked.

## What adding a model costs, and what changing one forces

- **Code:** one row in `ONNX_MODELS` in `crates/liam-model/src/onnx.rs`: id, native
  width, Matryoshka flag, token cap, query and document prefixes, and the
  fastembed variant. The daemon routes by id (`models.rs`), the CLI lists it in
  the unknown id error, and `fetch-models` downloads it with no other change.
- **Config:** `embedder.model` and `embedding_dims` (the native width for a fixed
  width model, any width up to native for Qwen3 and nomic).
- **Fingerprint:** the sidecar `<database>.embedder.json` records provider, model,
  width and prompt scheme. Changing model or width makes `liamd` refuse to start.
  The prompt scheme stays 2; a new model's prefix convention is part of its
  loader, and changing any prefix later must bump the scheme.
- **Migration:** stop `liamd`, edit `liam.toml`, run `liam rebuild --database PATH`.
  It re-embeds every live node. Cost is the batch throughput above: about 200 to
  400 nodes per second for the small models and about 5 per second for Qwen3, so
  100,000 nodes take roughly 4 to 8 minutes against roughly 5.5 hours. The 5.5 hour
  figure is arithmetic from the busy machine batch rate and an idle machine would
  be faster. Going from Qwen3 to a small model needs this once; so does going back.

## Decision rule

1. **Replace the default** if a candidate scores within 0.02 nDCG@10 and 0.05
   paraphrase nDCG@10 of Qwen3 on this set, and runs under the same license rule.
   No candidate does: the best is 0.047 and 0.106 behind.
2. **Offer as a `fast` profile** if it is within 0.06 nDCG@10 and at least 10 times
   lower in query latency and 5 times lower in peak memory. bge-small-en-v1.5,
   MiniLM-L12, Arctic s and xs and MiniLM-L6 all pass. bge-small is chosen because
   it has the best recall@1 and nDCG@10 of that group; MiniLM-L6 is the fastest and
   smallest (87 MB, 2 ms) with 0.016 less nDCG@10.
3. **Re-test** when real memory usage exists. The set is hand written and
   synthetic.

## Recommendation

- Keep `Qwen/Qwen3-Embedding-0.6B` at 768 as the default.
- Document `BAAI/bge-small-en-v1.5` at 384 as the fast option for a machine that is
  memory or battery constrained, or for a large store where a rebuild must be
  quick. Expect worse results when you ask in words different from the stored
  note. English only: it is an English model.
- If the aim is only smaller vectors, run Qwen3 at 384 or 512 (nDCG@10 0.933 and
  0.942): half to two thirds the vector bytes for 0.02 to 0.01 less nDCG@10.
  `embedding_dims` already supports it.
- For multilingual notes, none of the fast models is good. multilingual-e5-small
  works but scored lowest here, costs 1.3 GB and was only tested on English text.
  Qwen3 and BGE-M3 cover many languages.
- BGE-M3 is the middle option (0.919, 41 ms, 1.9 GB), as D42 found.

## What was not done

- No confidence intervals, no paired significance test, and one author pair wrote
  the whole set. Question labels were checked by their authors only.
- No cold cache load, no idle machine runs, no battery or energy measurement.
- No quantized, fp16 or int8 variants; no models outside fastembed's list were
  loaded; no model other than those named was measured.
- No non-English text, no long inputs (every fact is under 40 words), no code
  bodies, only short notes about code.
- Scan time and store size at 384 wide are inferred from D42, not measured here.
- The speed ratio between Qwen3 and the ONNX models includes the runtime
  difference (candle against ONNX Runtime). A candle run of a small model was not
  made, nor was Qwen3 run in ONNX.
- No public benchmark slice.
