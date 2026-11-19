# ADR-0018: build entity pages without a model and return them from recall

- **Status:** Accepted. Supersedes in part [ADR-0017](0017-remove-the-ask-tool.md) where it says entity pages use the local model and where it leaves open whether `recall` ranks pages.
- **Date created:** 2026-11-20
- **Date modified:** 2026-11-20

## Context

An entity page was a short profile the in-process 1.5B model wrote from the
memories that mention one entity. After [ADR-0017](0017-remove-the-ask-tool.md)
removed `ask`, no query could rank a page, so a page was read only through
`timeline`. The owner asked whether pages should be ranked beside recorded
memory in `recall`, and whether a model should still write them.

A measurement answered both. The sets are synthetic, hand written and small, so
the numbers describe those sets and nothing else.

**The first set.** Per language (English, Portuguese, Spanish): 15 entities
with 4 or 5 memories each, 60 questions (single fact, aggregate, latest state),
the same facts in three languages. nDCG@10 for no pages (A), pages written by
the Qwen 1.5B model (B), and pages cut from the memories without a model (C):

| Language | A | B | C |
|---|---|---|---|
| English | 0.884 | 0.902 | 0.923 |
| Portuguese | 0.874 | 0.877 | 0.944 |
| Spanish | 0.901 | 0.907 | 0.942 |

- **The model's pages failed in Portuguese and Spanish.** The prompt is
  English and the model answered in English, so the grounding gate, which wants
  half of a page's content words to appear in the memories, rejected every page
  in both languages (0 of 30 written). The B rows for those languages equal A for
  that reason, and say nothing about model pages. Letting the rejected text
  through lowered nDCG@10 below A (Portuguese 0.874 to 0.834, Spanish 0.901 to
  0.866).
- **The model's pages invented facts.** Reading all 45 pages by hand against
  their memories, 33 had at least one invented or wrong claim, including 7 of
  the 14 that passed the gate. The gate checks vocabulary and not truth. One
  reader counted, so the figure is a judgement.
- **A page cut from the memories cannot invent a fact.** Every line is a
  sentence a client recorded.
- **The caveat the owner accepted as a gate.** Those entities had 4 or 5
  memories, so the step that drops near duplicates and caps the page barely
  ran.

**The many-mention set (the gate).** Per language: 8 entities with 20 or 30
memories over many dates, with updates that supersede an earlier value and with
near-duplicate memories, and 40 questions (16 single fact, 8 aggregate, 16
latest state), hand written. The rule was fixed before the run: C must not lose
to A on nDCG@10 in any language by more than 0.01, and must beat A on the
latest-state questions overall. With arctic, through the same `recall` the daemon
serves:

| Language | nDCG@10 A | nDCG@10 C | Latest-state nDCG@10 A | Latest-state nDCG@10 C |
|---|---|---|---|---|
| English | 0.773 | 0.771 | 0.601 | 0.615 |
| Portuguese | 0.766 | 0.772 | 0.577 | 0.605 |
| Spanish | 0.790 | 0.800 | 0.620 | 0.638 |

Over the 120 questions, A scored 0.776 and C 0.781 (C won 23, tied 82 and lost
15). On the latest-state questions the share that had the newest value among the
first three results and no stale-only result above it was 0.312 for A and 0.333
for C. The gain is small and the rule passed: no language lost more than 0.01,
and C won on the latest-state questions. A page keeps at most four of 20 or 30
memories, so it cannot answer an aggregate question over many of them, and
aggregate nDCG@10 was 0.766 for A and 0.759 for C overall. The sets are
synthetic and small, and a gain of this size would not survive a different set.

## Considered Alternatives

### Keep the model's pages and keep them out of recall (effort: none)

- Trade-offs: no change. Pages stay readable only through `timeline`, still
  fail for Portuguese and Spanish, and still carry invented claims. Rejected.

### Rank the model's pages in recall (effort: S)

- Trade-offs: fixes nothing above. Rejected.

### Build pages from the memories and rank them in recall (effort: M, chosen)

- Trade-offs: no model, no invented claim, a few microseconds to build and one
  embedding of the page. A page shows several dated lines and does not say which
  is current.

### Build no pages (effort: S)

- Trade-offs: the safe baseline. It gives up a gain on latest-state questions
  that the sets show, though a small one.

## Decision

1. **A page is cut from recorded text.** For one entity: drop each memory whose
   embedding has a cosine similarity of 0.92 or more with a newer one, keep the
   two newest and the two most central of the rest (highest mean similarity to
   the others), and list them oldest first as `YYYY-MM-DD: text` under a
   `name (kind)` line, within 600 characters. The builder is a pure function of
   the memories, their dates and their vectors, so the same input gives the same
   page. The 0.92 threshold and the counts are those the first set used; the
   many-mention set raised no problem with them.
2. **Vectors.** The builder uses the vector the store holds for each memory and
   asks the embedder only for a memory with none, as a demoted one has.
3. **The page row keeps its basis and staleness.** The row records the origin
   events of the memories on the page, as before, and `stale_entities` and
   `mark_entity_synthesized` are unchanged. An entity with no recorded mention
   gets no page.
4. **`recall` returns pages.** It asks the store for inferred rows, leaves out
   summaries and entities with no page yet, and marks a page: a
   `kind: entity_page` line in the text and a `pages` list in the structured
   reply, each with the `basis` of the memories it was built from. Every
   existing field keeps its meaning, the input schema is unchanged, and
   `mode: infer` is refused as before. Summaries are not added to `recall`.
5. **Config keys.** `gc.full_synthesis_mention_threshold` and
   `gc.full_synthesis_max_new_tokens` are removed: an old line is ignored and
   logs one warning at start. `gc.max_resynth_per_tick` and `ask_timeout_secs`
   stay.
6. **Two store fixes the page needs.** A written page is found by its
   derivation and no longer by empty content, so a rebuild supersedes the page
   instead of leaving a second live row, and the staleness check can see it. The
   consolidation probes no longer draw a page or a summary, which a strict query
   cannot find and a rebuild supersedes.

## Consequences

- **A page cannot invent a fact.** It states only what a memory states.
- **A page does not merge or resolve.** Where a value changed, the reader sees
  both dated lines and takes the newer. A page can also leave out a memory that
  matters, so `recall` returns the memories too.
- **A page takes result slots.** In the first set a page was among the first
  three results for 99% of questions. A caller reading three results sees one
  page and two memories.
- **The store keeps its support for inferred rows.** `include_inferred` and the
  `as_of` argument of `summaries_within` stay.
- **Summaries still are not in `recall`.** Whether they should be is open.
- **The generation tuner has less to measure.** Entity pages were the only
  input to its runtime adjustment, which is removed. The start-up benchmark,
  the permits the summaries use, the `Llm` trait, llama.cpp and the `[llm]`
  section stay. The local model is still used for local summaries, until the
  next change removes it.
- **Old pages are rebuilt.** Entities whose pages the model wrote are rebuilt
  when a mention makes them stale. A page the model wrote stays until then.
- **A way back.** A model-written page can return through a new ADR.
