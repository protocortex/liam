# ADR-0018: build entity pages without a model and return them from recall

- **Status:** Proposed. The many-mention gate passes with the placement rule below, on synthetic sets. It would supersede in part [ADR-0017](0017-remove-the-ask-tool.md) where it says entity pages use the local model and where it leaves open whether `recall` ranks pages.
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
serves, run twice.

*Run 1: pages join the fused ranking of recorded memory, count toward `k` and
have no cap.* The rule passed, with a small gain (nDCG@10 A and C: English
0.773 and 0.771, Portuguese 0.766 and 0.772, Spanish 0.790 and 0.800). Pages
could take a slot a recorded memory held on main, so the recorded results of a
query changed when a page existed.

*Run 2: pages ranked in a query of their own, merged by score, at most two.*
Recorded rows were ranked as on main. nDCG@10 A and C: English 0.773 and 0.752,
Portuguese 0.767 and 0.762, Spanish 0.790 and 0.763. The rule failed: English
lost 0.021 and Spanish 0.027.

*The cause, measured on the 120 questions.* A page ranked alone gets the top
fused score whatever it is about, because the best of a few pages is rank 1 in
every arm: the best page scored 0.03279 on average against 0.03262 for the best
recorded row, beat it on 38 questions and tied on 78. So the two pages landed in
the first three results for all 120 questions, and held the answer for 28 of
them (23%). On the worst questions a page at rank 1 or 2 without the answer
moved the gold memory down by one or two places (for example a single gold
memory from rank 2 to 3, or the three gold memories of an aggregate question
from ranks 3, 4, 5 to 5, 8, 6). The score of a page in a ranking of pages is
not comparable with the score of a memory in a ranking of memories.

Variants tried on the same data, nDCG@10 for A and the variant in English,
Portuguese and Spanish (latest-state nDCG@10 in brackets):

| Variant | en | pt | es | Passes the rule |
|---|---|---|---|---|
| A, no pages | 0.773 (0.601) | 0.767 (0.577) | 0.790 (0.620) | |
| Run 2, cap 2 by score | 0.752 (0.599) | 0.762 (0.642) | 0.763 (0.640) | no |
| V1, cap 1 by score | 0.761 (0.611) | 0.774 (0.663) | 0.773 (0.652) | no, es loses 0.017 |
| V2, cap 2 after rank 3 | 0.758 (0.593) | 0.755 (0.584) | 0.772 (0.611) | no, loses latest state |
| V3, cap 2 after all `k` memories | 0.724 (0.548) | 0.716 (0.525) | 0.738 (0.550) | no |
| Run 3, placed by the joint ranking | 0.771 (0.615) | 0.766 (0.602) | 0.797 (0.634) | yes |

V3 scores below A, not equal to it, because of how the harness counts: a page
that holds the answer joins the relevant set when it is in the reply, so the
ideal ranking grows by one, and a page after the tenth result earns nothing.
Scored on the memories alone, V3 equals A. A page placed beyond the cutoff is
therefore penalised by the harness and cannot show a gain in it. In every
variant the answer was present somewhere in the reply for 97.5% of questions
and the newest value for 100%, the same as without pages in each language, so a
reader of the whole reply sees no difference in what is present. A variant that
ranks pages only for questions about the latest state was not tried, because
telling those questions apart needs a word list per language or a model.

*Run 3, the design below.* The rule passed, with the figures above: English
loses 0.002, Portuguese 0.001, Spanish gains 0.007, and latest-state nDCG@10 is
0.617 for C against 0.599 for A. The share of latest-state questions with the
newest value among the first three results was 0.312 for A and 0.312 for C in
English, 0.250 and 0.250 in Portuguese, 0.375 and 0.312 in Spanish: pages did
not improve that figure. The recorded rows of a store with pages are not
byte-identical to those of a store without: the pages are in the text index, so
the statistics it ranks by shift, and 54 of 120 questions had a different order
of the ten memories, and 13 of those a different set. Scored on the memories alone the
two stores are equal (0.773 and 0.773, 0.767 and 0.767, 0.790 and 0.789). The
page logic itself never reorders the memories. The sets are synthetic and
small.

## Considered Alternatives

### Keep the model's pages and keep them out of recall (effort: none)

- Trade-offs: no change. Pages stay readable only through `timeline`, still
  fail for Portuguese and Spanish, and still carry invented claims. Rejected.

### Rank the model's pages in recall (effort: S)

- Trade-offs: fixes nothing above. Rejected.

### Build pages from the memories and rank them in recall (effort: M, chosen)

- Trade-offs: no model, no invented claim, and a cost of 10 to 30 microseconds
  of selection per entity plus 27 to 58 ms to embed the page with arctic (the
  first measurement). A page shows several dated lines and does not say which is
  current.

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
4. **`recall` returns pages beside recorded memory, never in place of it.**
   The memories come from a query that leaves out inferred rows from the
   ranking, from neighbour expansion and from the `same_as` closure, as before
   pages existed, so the memories the page logic returns are those of that
   query, in its order. A second query ranks recorded and inferred rows
   together. Up to `MAX_PAGES_IN_RECALL` (2) current pages are taken from it in
   its order, and each is placed after as many memories as that ranking puts
   above it. A reply therefore holds up to `k` recorded rows plus up to 2
   pages. Summaries, entities with no page yet and legacy pages are left out. A
   page is marked: a `kind: entity_page` line in the text and a `pages` list in
   the structured reply, each with the `basis` of the memories it was built
   from. Every existing field keeps its meaning, the input schema is unchanged,
   and `mode: infer` is refused as before.
5. **Config keys.** `gc.full_synthesis_mention_threshold` and
   `gc.full_synthesis_max_new_tokens` are removed: an old line is ignored and
   logs one warning at start. `gc.max_resynth_per_tick` and `ask_timeout_secs`
   stay.
6. **Two store fixes the page needs.** A written page is found by its
   derivation and no longer by empty content, so a rebuild supersedes the page
   instead of leaving a second live row, and the staleness check can see it. The
   consolidation probes no longer draw a page or a summary, which a strict query
   cannot find and a rebuild supersedes. Rewriting a page closes every live row
   of the entity, not only the newest, so a store that already holds several
   leaves one.
7. **Legacy pages.** A page this builder writes carries the attribute
   `page_builder` in its row, which `recall` hides from the reader. A page an
   earlier version wrote, which the Qwen model composed, lacks it. `recall`
   never returns a page without the attribute. The maintenance tick treats such
   a page as stale after the stale entities, up to `gc.max_resynth_per_tick`
   entities in all, and replaces it with a new page. The old row, and any
   further live row of that entity, is closed. A legacy page of an entity with
   no recorded mention in its scope stays closed to readers and is tried again
   each tick, because no new page can replace it.
8. **Scope.** A scoped entity's page is built only from mentions whose scope is
   the entity's scope or inside it, the same hierarchy a scoped read uses. An
   entity with no mention in its scope gets no page. An unscoped entity's page
   may hold mentions of any scope.
9. **Readers.** `recall` has no producer filter, for memories or for pages: the
   only filters are `kind` and `scope`, applied the same way to both. A scoped
   read sees a page only when the page's scope is inside the read's scope, and
   an unscoped read sees every page. A page is therefore visible to the readers
   who can already see the memories it quotes.

## Consequences

- **A page cannot invent a fact.** It states only what a memory states.
- **A page does not merge or resolve.** Where a value changed, the reader sees
  both dated lines and takes the newer. A page can also leave out a memory that
  matters, so `recall` returns the memories too.
- **A reply can hold extra entries.** A reply holds up to `k` recorded rows plus
  up to 2 entries of kind `entity_page`. A caller that counted on `k` entries
  exactly, or that treats every entry as a memory, must change: this is a
  breaking change to `recall`.
- **The store keeps its support for inferred rows.** `include_inferred` and the
  `as_of` argument of `summaries_within` stay.
- **Summaries still are not in `recall`.** Whether they should be is open.
- **The generation tuner has less to measure.** Entity pages were the only
  input to its runtime adjustment, which is removed. The start-up benchmark,
  the permits the summaries use, the `Llm` trait, llama.cpp and the `[llm]`
  section stay. The local model is still used for local summaries, until the
  next change removes it.
- **Old pages are ignored and rebuilt.** `recall` never returns a page the
  model wrote. The next maintenance ticks rebuild them, a few per tick.
- **A way back.** A model-written page can return through a new ADR.
