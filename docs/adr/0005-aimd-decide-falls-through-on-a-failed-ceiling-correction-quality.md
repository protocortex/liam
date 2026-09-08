# Quality gate report: ADR-0005

**Record:** `docs/adr/0005-aimd-decide-falls-through-on-a-failed-ceiling-correction.md`
**Blueprint:** `docs/adr/0005-aimd-decide-falls-through-on-a-failed-ceiling-correction-blueprint.md`
**Date:** 2026-09-07

This gate ran three iterations on Phase 2 and Phase 3 before both passed. Phase 1 ran once,
inline.

## Phase 1: Fact-Check — PASS

Ran inline (mechanical checks): every referenced file path exists; line anchors for
`AimdHandles`, `decide()`, `IMPROVEMENT_THRESHOLD`, `reconcile_capacity_after_benchmark` and its 6
tests, `clamp_ask_k`, and `NewNode::entity` all confirmed against the real source. Work Unit
dependency graph (WU-2 -> WU-3 -> WU-4 -> WU-5, WU-0/WU-1 unattached) is acyclic. No parallel
groups declared, so no disjointness violation possible.

## Phase 2: Adversarial Review (focus: decision)

**First pass: FAIL.**

| # | Severity | Finding |
|---|---|---|
| 1 | HIGH | WU-3's failed-correction test only required asserting "an observable side effect," loose enough that a fallthrough which runs but discards its result could still pass. |
| 2 | HIGH | The ADR rejected alternative B partly for "a second writer racing the existing `decide()` writer," but the chosen alternative C has the same class of race between `main.rs`'s one-time reconcile call and `decide()`'s new recurring call, undiscussed. |
| 3 | MEDIUM | No trade-off analysis recorded for why `decide()` skips the latency decision only on a successful correction, rather than always falling through regardless of outcome. |
| 4 | LOW | Consequences said `decide()` "grows a third branch" but the diagram showed four outcomes. |

**Revision:** WU-3's test now names concrete assertions (`held_permit` becomes `Some`, or
`available_permits()` drops by exactly one) after seeding a qualifying regression, with an
explicit ban on binding held permits to `_`. Alternative B's rejection reworded to cite
over-engineering, not race-uniqueness; a new "Accepted risk, pre-existing" Consequences bullet
names the real race, its mechanism, and defers a real fix to a future ADR. A trade-off paragraph
was added explaining why always-falling-through was rejected. Consequences reworded to "two new
top-level branches beyond its current two."

**Second pass: FAIL** (1 new MEDIUM). The "accepted risk" bullet wrongly listed `shrink()` as a
writer of `granted_capacity` alongside `grow()`, contradicting `shrink()`'s own doc comment
(`tuning.rs:238-239`, "without touching `granted_capacity`").

**Revision:** corrected the bullet to name only `grow()` as the second writer, with the doc
comment citation.

**Third pass: PASS.** All findings confirmed resolved; final full-document scan found nothing
else.

## Phase 3: Test Review

**First pass: FAIL** (2 FAIL, 3 WARN).

| # | Severity | Finding |
|---|---|---|
| 1 | FAIL | WU-3's failed-correction test assertion was vague (see Phase 2 finding 1, same root cause). |
| 2 | FAIL | WU-4's three-point threshold test baked `Duration::mul_f64` arithmetic directly into the test, risking a floating-point rounding mismatch against production's own computation. |
| 3 | WARN | WU-5's panic-seam mechanism was underspecified, risking a trivial/unrealistic seam. |
| 4 | WARN | WU-2's "zero test count change" Done-When had no concrete verification step. |
| 5 | WARN | WU-3/WU-4's held-permit tests risked binding acquired permits to `_`, silently defeating the test's premise. |

**Revision:** WU-3 fixed as above. WU-4 gained a shared `shrink_bar(prev)` helper called by both
`decide()` and the test, eliminating independently re-derived arithmetic. WU-5's seam now must
panic from inside `decide()`'s real call graph via `evaluate()`'s normal path, with a concrete
example. WU-2 now requires capturing `cargo test ... | grep "test result:"` before and after,
diffing the passed-count. Held-permit tests now require a named, scope-lived binding.

**Second pass: FAIL** (1 new FAIL, 1 new WARN). WU-3's "ALL excess permits held" scenario didn't
pin concrete numbers: holding exactly the excess count wouldn't make `try_acquire_many` fail, and
holding every permit would hang `shrink()`'s own acquire forever. Also, WU-3's test referenced
`SHRINK_REGRESSION_THRESHOLD`, a constant WU-4 (a later Work Unit) introduces, breaking WU-3's
independent compilability.

**Revision:** pinned `granted_capacity = 6`, `ceiling = 4`, hold 5 of 6 permits (1 stays free, so
the correction's `try_acquire_many(2)` fails without blocking `shrink()`'s own acquire). Switched
WU-3's seeding to the pre-WU-4 `window_total > prev` comparison, deferring
`SHRINK_REGRESSION_THRESHOLD` to WU-4's own scenarios.

**Third pass: PASS.** Both findings confirmed resolved; final scan of all 6 Work Units found
nothing else.

## Final Result

**PASS** on all three phases, after two full revision cycles on Phase 2 and Phase 3. Record and
blueprint approved by the user.
