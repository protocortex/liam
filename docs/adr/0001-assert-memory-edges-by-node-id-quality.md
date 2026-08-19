# ADR-0001 Quality Gate Result

- **Parent ADR:** [0001-assert-memory-edges-by-node-id.md](0001-assert-memory-edges-by-node-id.md)
- **Date:** 2026-08-19

## Result

```
Fact-Check:         INCONCLUSIVE (agent unavailable)
Adversarial Review: INCONCLUSIVE (agent unavailable)
Test Review:        N/A (no execution blueprint yet)
```

The record is **Accepted** on direct approval from the maintainer, not on a passing gate.
The gate did not run, and this file exists so that is on the record rather than implied
away.

## Why the gate did not run

All three gate phases dispatch subagents (`fact-checker`, `critic`, `test-reviewer`)
through the Agent tool. On Claude Code 2.1.220 every Agent-tool subagent starts, never
returns, and yields no result. Four spawns were tested and all four hung: `Explore`
(sonnet, large prompt), `Plan` (sonnet, large prompt), `Explore` (haiku, single-`Read`
prompt), and `playbook:reviewer` (haiku, single-`Read` prompt). Forked skill agents were
unaffected in the same session.

Ruled out by direct test: plugin hooks (all run in 0.1s on a stdin payload), MCP
connectivity (all 10 servers connected), tool-set size (a 4-tool agent with no MCP hangs
the same as an all-tools agent), model tier, and prompt size.

Per the `/adr` gate rules, a phase whose agent returned nothing is INCONCLUSIVE rather than
PASS, and INCONCLUSIVE blocks finalisation the same way a FAIL does. Reporting PASS here
would have described a check that never happened.

## What was verified, and how

Self-review only, no independent adversarial pass. What it caught:

- **Corrected, material.** The draft claimed in two places that rendering node ids in
  `recall` put the 6/6 grounding eval at risk because "`ask` renders evidence from the same
  retrieval path". Not true. `recall` renders at `crates/liam-daemon/src/mcp.rs:279`;
  `ask` builds `ask::Evidence::from_hit` (`crates/liam-daemon/src/ask.rs:51`) from `kind`,
  `label`, `content`, and `valid_from_ms`, never reading `Hit::id`. The two share
  `build_query` and the store, not one line of rendering. `crates/liam-daemon/src/eval.rs:470`
  drives `ask`, and `recall`'s only callers in the tree are four tests in `mcp.rs`. The
  stated risk was therefore zero, and both passages were rewritten.
- **Added.** A consequence that was missing: nodes with no edges receive no community at
  all, because `recompute_communities` builds its node list from the endpoints its own
  query returns (`crates/liam-store/src/graph.rs:591`). Filtering `supersedes` shrinks that
  set further, since a node whose only edge is a version link drops out entirely.
- **Neutralised.** Two rejected alternatives had been credited with "no eval risk" as a
  comparative advantage. With the risk shown to be absent from the chosen option too, that
  advantage was removed from both.

## Outstanding, to verify when the gate can run

- No independent fact-check of the file:line citations in Context and Decision Drivers.
  They were written by the same session that self-reviewed them.
- No adversarial pass on the decision itself: nobody has argued for a simpler option or
  challenged the blast radius from outside the authoring context.
- No execution blueprint exists yet, so no work-unit dependency graph, file plan, or test
  plan has been checked.

Re-run the full three-phase gate before implementation begins, once Agent-tool subagents
return results again.
