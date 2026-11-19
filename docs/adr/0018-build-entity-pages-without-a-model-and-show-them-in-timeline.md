# ADR-0018: build entity pages without a model and show them in timeline only

- **Status:** Accepted. Supersedes in part [ADR-0017](0017-remove-the-ask-tool.md) where it says entity pages use the local model and where it leaves open whether `recall` ranks pages.
- **Note, 2026-11-20:** the `Llm` trait, llama.cpp, the `[llm]` section and the start-up benchmark that this ADR keeps were removed by [ADR-0019](0019-remove-the-local-generation-model.md).
- **Date created:** 2026-11-20
- **Date modified:** 2026-11-20

## Context

An entity page was a short profile the in-process 1.5B model wrote from the
memories that mention one entity. After [ADR-0017](0017-remove-the-ask-tool.md)
removed `ask`, no query could rank a page, so a page was read only through
`timeline`. The owner asked whether a model should still write pages and
whether `recall` should rank them.

All the sets below are hand written, synthetic and small. The numbers describe
those sets and nothing else.

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
  that reason, and say nothing about model pages.
- **The model's pages invented facts.** Reading all 45 pages by hand against
  their memories, 33 had at least one invented or wrong claim, including 7 of
  the 14 that passed the gate. The gate checks vocabulary and not truth. One
  reader counted, so the figure is a judgement.
- **A page cut from the memories cannot invent a fact.** Every line is a
  sentence a client recorded.

**The many-mention set.** Per language: 8 entities with 20 or 30 memories over
many dates, with updates that supersede an earlier value and with
near-duplicate memories, and 40 questions. The rule fixed before the run was
that pages in `recall` must not lose to no pages by more than 0.01 nDCG@10 in
any language and must beat no pages on the latest-state questions overall. With
arctic, through `recall`:

| Language | A (no pages) | Pages in `recall`, first design | Pages in `recall`, placed by the joint ranking |
|---|---|---|---|
| English | 0.773 | 0.752 | 0.771 |
| Portuguese | 0.767 | 0.762 | 0.766 |
| Spanish | 0.790 | 0.763 | 0.797 |

The first design ranked pages in a query of their own, at most two per reply.
It failed the rule in English and Spanish: a page ranked alone gets the top
score whatever it is about, so both pages landed in the first three results for
all 120 questions and held the answer in 28 of them. Placing each page by a
ranking of memories and pages together passed the rule. Even so, the answer was
in the reply for 97.5% of questions with or without pages, so a reader of the
whole reply gains nothing in what is present. Pages also shifted the text index
statistics that rank memories: the order of the ten recorded memories changed
for 54 of 120 questions between a store with pages and one without, though the
page code never reorders memories. The gain in `recall` vanished at 20 to 30
memories per entity, and the owner chose not to ship pages in `recall`.

## Considered Alternatives

### Keep the model's pages (effort: none)

- Trade-offs: pages stay readable through `timeline`, still fail for
  Portuguese and Spanish, and still carry invented claims. Rejected.

### Build pages from the memories and rank them in recall (effort: M)

- Trade-offs: no model and no invented claim, but the measurement above shows
  no gain a reader can use, a change to the shape of every `recall` reply, and a
  shift in the order of recorded memories. Tried, not shipped.

### Build pages from the memories and show them in timeline only (effort: M, chosen)

- Trade-offs: no model, no invented claim, `recall` unchanged. A page shows
  several dated lines and does not say which is current.

### Build no pages (effort: S)

- Trade-offs: the safe baseline, which gives up the readable compiled page in
  `timeline`.

## Decision

1. **A page is cut from recorded text.** For one entity: drop each memory whose
   embedding has a cosine similarity of 0.92 or more with a newer one, keep the
   two newest and the two most central of the rest (highest mean similarity to
   the others), and list them oldest first as `YYYY-MM-DD: text` under a
   `name (kind)` line, within 600 characters. A mention's whitespace is
   collapsed to one space so it stays on its own line. The builder is a pure
   function of the memories, their dates and their vectors.
2. **Vectors.** The builder uses the vector the store holds for each memory and
   asks the embedder only for a memory with none, as a demoted one has. The page
   itself is not embedded.
3. **The page row keeps its basis and staleness.** The row records the origin
   events of the memories on the page. An entity with no recorded mention gets
   no page.
4. **`timeline` shows pages and `recall` is unchanged.** `recall`, its input and
   output schemas and its ranking are as before: it never returns a page or a
   summary. A written page is not indexed for text search and has no vector, so
   it cannot move the ranking statistics of recorded memory or take a place in a
   search. The store keeps its ability to rank inferred rows through
   `include_inferred`, by vector only. Whether pages come to `recall` later
   needs a new measurement and a new ADR.
5. **Config keys.** `gc.full_synthesis_mention_threshold` and
   `gc.full_synthesis_max_new_tokens` are removed: an old line is ignored and
   logs one warning at start. `gc.max_resynth_per_tick` and `ask_timeout_secs`
   stay.
6. **Two store fixes the page needs.** A written page is found by its
   derivation and no longer by empty content, so a rebuild supersedes the page
   instead of leaving a second live row. A rewrite of a page closes every live
   row of the entity. The mention edges of the entity carry to the new row
   before the staleness fingerprint is recorded, so the entity is not stale on
   every tick. The consolidation probes no longer draw a page or a summary.
7. **Legacy pages.** A page this builder writes carries the attribute
   `page_builder`. A page an earlier version wrote, which the model composed,
   lacks it. `timeline` shows no text for such a page. The maintenance tick
   treats it as stale, after the stale entities and within
   `gc.max_resynth_per_tick`, and replaces it with a new page. The old row, and
   any further live row of that entity, is closed. A legacy page of an entity
   with no recorded mention in its scope is replaced by an entity row with no
   text, so it is not tried again and cannot fill the slots of the pages that
   can be rebuilt. A failed lookup of legacy pages is logged and leaves the
   stale entities due.
8. **Scope.** A scoped entity's page is built only from mentions whose scope is
   the entity's scope or inside it, the same hierarchy a scoped read uses. An
   entity with no mention in its scope gets no page. An unscoped entity's page
   may hold mentions of any scope.

## Consequences

- **Pages are isolated from recall.** The triggers that keep the text index in
  step with `nodes` skip a row that is inferred and holds text, and the page
  write sets no vector. A store that indexed pages is migrated once at open:
  the page entries are removed from the index and the triggers are replaced.
  The store tests `a_strict_query_ranks_the_same_with_a_page_written_and_without_for_pt_and_es`
  (same ordered texts and scores) and `a_written_page_is_not_indexed_for_text_and_has_no_vector`
  check this, and `a_store_that_indexed_pages_has_them_removed_once_and_gets_the_new_triggers`
  checks the migration. An entity row with no text stays indexed by its label, as
  before.
- **Pages cannot be found by search.** A test or tool that needs the live page of
  an entity reads it by subject, not by query.
- **A page cannot invent a fact.** It states only what a memory states.
- **A page does not merge or resolve.** Where a value changed, the reader sees
  both dated lines and takes the newer. A page can also leave out a memory that
  matters, so `timeline` lists the mentions behind it.
- **The store keeps its support for inferred rows.** `include_inferred` and the
  `as_of` argument of `summaries_within` stay.
- **The generation tuner has less to measure.** Entity pages were the only
  input to its runtime adjustment, which is removed. The start-up benchmark,
  the permits the summaries use, the `Llm` trait, llama.cpp and the `[llm]`
  section stay. The local model is still used for local summaries.
- **Old pages are ignored and rebuilt.** `timeline` never shows a page the model
  wrote, and the next maintenance ticks rebuild them.
- **A way back.** A model-written page can return through a new ADR.
