# Tiering

[Back to the docs index](../README.md)

Tiering decides which memories keep their vectors. It is off by default. A hot
item has its float vector, a compact int8 form and a sign-bit signature. A cold
item has dropped its vectors, and keeps its row, its text index entry, its
edges, its events and the signature. A cold item is still found by a text query,
and by a vector query through a probe of the signatures that adds a few cold
hits behind the hot ones. Nothing is deleted: the log keeps every record, and a
`liam rebuild` leaves a cold item cold.

A `tier` step in each consolidation run scores the items, demotes the ones that
have not been useful and promotes the cold ones that have. The int8 form is
stored but nothing reads it yet. Items are tracked by lineage, the chain of
versions of one subject, so a new version of an item keeps its usage and its
verdicts. A new version is written hot.

## The score

The utility of an item is the weighted mean of four terms, each from 0 to 1:

- recent hits, `log(1 + hits) / log(1 + hits_cap)` over the last `window_days`,
  weight `w1`. Each producer's hits are capped first, per item per day and in
  total per day, so one client cannot keep an item hot alone;
- recency, `exp(-days / tau_days)` since the item was last returned (or written,
  if it never was), weight `w2`;
- answer contribution, the share of recent verdicts on the item that said it was
  correct, each counted by the trust of the producer who gave it, starting from
  `feedback_prior_correct` and `feedback_prior_incorrect` and reading at most
  the latest `feedback_item_cap` verdicts, weight `w3`. It is left out for an
  item with no verdicts, and the other weights are scaled to match;
- the trust of the item's producer from `[extraction.trust]`, weight `w4`.

A hot item below `demote_threshold` is demoted; a cold item at or above
`promote_threshold` is promoted. Between the two nothing moves. A cold item that
was returned since it went cold is promoted once its residency is over. A
promoted item is protected for `min_residency_days`, and an item younger than
`min_item_age_days` is never demoted.

## What counts as use

Only `recall` and `ask` results count, and only the ranked results the client
was shown, not the neighbours pulled in beside them. Counts are kept in memory
and written to the log every `usage_flush_secs`, so a crash loses up to that
many seconds of counts. A store that has not been counting for a whole
`window_days` demotes nothing, because every item would look unused.

Clients can say whether a result was right with the `feedback` tool, using the
handle `recall` or `ask` printed for it (on `Sources:` lines and in evidence
blocks too). A handle works on the connection that was given it. One producer
may give `feedback_per_producer_per_hour` verdicts an hour, counted in memory,
so the limit starts over when the daemon restarts. Clients that do not name
themselves share one producer, and so one budget. One producer's verdicts on
one item count once a day. The tool answers that verdicts are not kept while
tiering is off.

## What is never demoted

Entity pages, items a live summary is built from, items a recall probe of the
current run expects to find, and items moved by a rolled back run for
`cooldown_runs` runs. A rollback also halves the demotion cap of the next run,
for each rollback in a row, down to one. The caps are `max_promotions_per_run`
and `max_demotions_per_run`, and `run_deadline_secs` bounds the step.

Before a vector is dropped, the log is read to check that the item's origin
events still hold what the item says. An item the log does not vouch for stays
hot. A demoted item is left out of later summary builds, because summaries are
built from items that have a vector.

If the embedder changes, `liam rebuild` clears the signatures with the vectors
and writes new ones from the new vectors. A store from before tiering gets
signatures the first time the step runs.

Keys, all under `[tiering]` in `liam.toml`:

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | `false` makes the tier step a no-op and counts nothing. |
| `w1` | `0.4` | Weight of the recent hits term. |
| `w2` | `0.3` | Weight of the recency term. |
| `w3` | `0.2` | Weight of the answer contribution term, left out for an item with no verdicts. |
| `w4` | `0.1` | Weight of the producer trust term. |
| `tau_days` | `30.0` | Days over which the recency term falls to 1/e. |
| `hits_cap` | `20` | Hits in the window at which the hits term reaches 1. |
| `hits_per_producer_item_day_cap` | `10` | Most hits one producer adds to one item in a day. |
| `hits_per_producer_day_cap` | `100` | Most hits one producer adds across items in a day; past it the largest contributions are cut first. |
| `unknown_trust_ceiling` | `0.3` | Most trust the fallback producer (`producers.unknown_id`) can have, as an author and as a giver of verdicts. |
| `promote_threshold` | `0.5` | Utility at or above which a cold item is promoted. |
| `demote_threshold` | `0.15` | Utility below which a hot item is demoted. |
| `min_residency_days` | `7` | Days a promoted item is protected from demotion. |
| `min_item_age_days` | `14` | Least age of an item for it to be demoted. |
| `max_promotions_per_run` | `50` | Most items one run promotes. |
| `max_demotions_per_run` | `50` | Most items one run demotes, halved for each rollback in a row. |
| `window_days` | `30` | Days of usage that count, and of observation needed before any demotion. |
| `usage_flush_secs` | `60` | Seconds between flushes of the in-memory access counts to the log. |
| `max_tracked_items` | `10000` | Items the in-memory counts hold before the oldest is dropped. |
| `feedback_per_producer_per_hour` | `60` | Verdicts one producer may give per hour. |
| `feedback_item_cap` | `20` | Most recent verdicts on one item that the score reads. |
| `feedback_prior_correct` | `1.0` | Pseudo-verdicts of "correct" the contribution rate starts from. |
| `feedback_prior_incorrect` | `1.0` | Pseudo-verdicts of "incorrect" the contribution rate starts from. |
| `cooldown_runs` | `3` | Runs a lineage moved by a rolled back run is left alone. |
| `cold_probe_pool` | `24` | Cold items the probe contributes to one query. |
| `cold_oversample` | `4` | Candidates asked of the signature index per item wanted. |
| `cold_max_hamming_ratio` | `0.35` | Share of signature bits that may differ for a cold item to be a candidate. |
| `cold_arm_weight` | `0.5` | Factor on the cold probe's contribution to a fused score. |
| `run_deadline_secs` | `600` | Bounds the tier step of one run. |

The defaults are starting values and have not been measured on real data.
