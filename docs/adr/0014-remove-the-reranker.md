# ADR-0014: remove the reranker

- **Status:** Accepted. Supersedes in part [ADR-0013](0013-arctic-default-embedder-and-reranker-off.md): the embedder decision stands, and the reranker decision and its 2026-11-17 addendum are superseded.
- **Date created:** 2026-11-18
- **Date modified:** 2026-11-18

## Context

ADR-0013 turned the reranker off by default and kept three models selectable.
Its addendum asked for a reranker that loads on demand, so a large store could
opt in. Three benchmark rounds on 2026-11-17 then asked whether any reranker
earns its place in the daemon.

- **A store of 300 facts.** The best reranker moved English nDCG@10 by +0.002 to
  +0.003, which is inside the noise.
- **A store of 10,000 passages** (MIRACL dev, 200 English, 100 Spanish and 100
  Japanese queries, the top 16 of arctic-embed-m-v2.0 reranked). Mean nDCG@10
  over the three languages: none 0.9026, `gte-multilingual-reranker-base`
  0.9399, `bge-reranker-v2-m3` 0.9543. The gain is real, +0.037 and +0.052, and
  the 95% interval is above zero in every language for bge. For gte the Spanish
  lower bound is +0.0004.
- **Small models.** Every reranker under 150 MB gave no significant gain at pool
  16. The multilingual `mmarco-mMiniLMv2-L12` (119 MB) gained +0.011 with
  intervals that include zero in all three languages. The English only
  MS MARCO models lost ground, and on Japanese they lost 0.40 to 0.50.
- **Cost of the two large models.** The files are 341 MB and 571 MB. They hold
  1.0 to 1.5 GB resident after load and took 2.0 s and 4.3 s for 16 pairs of
  long Wikipedia passages. These timings came from a machine under memory
  pressure, so they are a ranking and not a benchmark. The smaller timings are
  just as unclean: TinyBERT 29 ms, MiniLM-L6 267 ms, mMiniLM 661 ms.

The user decided to give up the large store gain for a lighter daemon.

## Considered Alternatives

### Keep the reranker, load on demand (effort: M)

- Trade-offs: the follow-up that ADR-0013 asked for. It keeps the +0.04 to
  +0.05 on a large store, and costs memory and hundreds of milliseconds to
  seconds per recall while in use, plus a loader, an idle timer and the
  tokenizer for each model. Rejected.

### Ship a small reranker (effort: S)

- Trade-offs: cheap in memory. No model under 150 MB gained significantly, and
  the English only ones harm other languages. Rejected.

### Remove the reranker (effort: M, chosen)

- Trade-offs: deletes the loader, the stage timer and three
  degraded reasons. The config section is no longer read; a hidden field accepts
  and ignores it. Gives up the large store gain.

## Decision

1. **Remove the reranker from the code.** `recall` and `ask` return candidates in
   store order, which is what `reranker.model = "none"` already gave. The
   candidate pool stays twice `k`, at least 16 and at most 32, so the order of
   the first `k` is unchanged.
2. **Accept an old `[reranker]` section.** The daemon ignores it and logs one
   warning at start that says the reranker was removed and the section can be
   deleted.
3. **Remove the `rerank` stage timer and the `rerank_skipped`, `rerank_failed`
   and `rerank_timeout` degraded reasons.**

## Consequences

- **A lighter daemon.** One less model, one less tokenizer and no reranker memory
  at all.
- **The large store gain is given up.** On 10,000 passages that is about 0.04 to
  0.05 nDCG@10.
- **A breaking change for operators who set a reranker.** Their recall order
  changes to store order, and the config section does nothing.
- **A way back.** Late interaction retrieval or a reranker can return through a
  new ADR, with the numbers above as the bar to beat.
