# Quality gate report: ADR-0004

## Phase 1: Fact-Check — PASS (run inline)

Every `file:line` citation in the ADR record and blueprint verified against current source:
`types.rs` (`NewNode`, `Query`, `Hit`, `ExplainedHit`), `mcp.rs` (`RememberArgs`, `RecallArgs`,
`recall_renders_a_handle_relate_can_resolve`), `graph.rs` (`resolve_handle`), `ask.rs`
(`neutralize_fence`, `evidence()`). `git status --short crates/` confirmed no code changes
occurred during the ADR process, so all line-number citations stayed valid throughout.

## Phase 2: Adversarial Review (critic, focus=decision) — PASS after 2 rounds

**Round 1: FAIL.** One MEDIUM finding: `is_grounded` (`ask.rs:302-316`) never included
`e.attributes` in its grounding vocabulary set, so an answer correctly citing an attribute-only
fact (`ai-notetaker`'s own driving scenario, named in the ADR's Context) could score below
`MIN_GROUNDED_SHARE` and wrongly fall back to raw evidence display. Confirmed genuine bug, not a
false positive, verified against source directly. Three LOW findings also raised: no documented
rationale for `valid_from`'s lack of validation, no test for a future `as_of`, and a missing
"encode into content by convention" alternative in Considered Alternatives.

**Round 2: PASS.** All four fixes verified against source: `is_grounded` gains
`content_words(e.attributes.as_deref().unwrap_or(""))` in WU-4 (with one trivial type-precision
correction applied on the spot); `valid_from`'s non-validation rationale cited against
`decay_factor`'s `.max(0)` clamp; the future-`as_of` claim confirmed against `node_insert`'s
hard-coded `FOREVER` default for `valid_until`; the fifth Considered Alternative added and
confirmed consistent in tone with the other four. `query()`/`query_explained()` equivalence,
the rejected in-bracket alternative, and the blueprint's Ordering/Parallel-groups/Dependency-graph
self-consistency were all re-spot-checked and confirmed still solid.

## Phase 3: Test Review — PASS after fixes to 2 WARN findings (no FAIL on the final round)

**Round 1: FAIL.** The newly-added `is_grounded`-attributes test scenario named the behavior but
supplied no concrete test data, so as originally written it could pass by incidental
content/label/kind word overlap without the fix actually being wired in — a vacuous regression
pin for the one real bug it exists to guard. WARN: the future-`as_of` test only asserted presence
of the current version, not exclusion of the superseded one (a materially weaker pin than its
sibling scenario), and the `as_of >= FOREVER` edge case was neither tested nor documented.

**Round 2: WARN (2 findings, both fixed directly, no further FAIL).** Both fixes from round 1
confirmed sound against source (concrete test data genuinely flips the assertion from false to
true only once the fix lands; the future-`as_of` fix's `tx_to`-vs-`valid_until` distinction
confirmed against `supersede`'s actual `UPDATE` statement). Two new WARNs found on a full pass
over the rest of the blueprint: WU-3's recall-rendering tests covered confidence `0.6` but not
the `0.0` boundary (inconsistent with WU-4's own explicit `0.0` case); WU-5's `as_of`-scoped
round-trip test didn't state it uses the `FixedClock` fixture, risking a silent reintroduction of
the exact flakiness defect WU-2 already fixed once. Both fixed directly (cheap, one-line
additions) rather than deferred, since WARNs don't block but these were free to close.

## Result

**Gate passed.** One genuine bug was found beyond what the underlying `/playbook:scope` plan's
own gate caught: the `is_grounded` vocabulary gap, which would have quietly undermined this
milestone's stated purpose for the specific consumer scenario motivating it. The gate also caught
two of its own freshly-written regression tests before they shipped: one that would have been
vacuous, and one that risked reintroducing an already-fixed flakiness defect. Both artifacts
(`0004-...-adr.md`, `0004-...-blueprint.md`) are finalized at Status: Accepted.
