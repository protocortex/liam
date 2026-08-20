# ADR-0001 and ADR-0002 Quality Gate Result

- **Parent ADRs:** [0001-assert-memory-edges-by-node-id.md](0001-assert-memory-edges-by-node-id.md),
  [0002-cluster-recompute-cadence.md](0002-cluster-recompute-cadence.md)
- **Date:** 2026-08-21

## Result

```
Fact-Check:         PASS (HIGH confidence, 1 WARN)
Adversarial Review: PASS on round 3 of 3 (FAIL, FAIL, PASS)
Test Review:        N/A (no execution blueprint yet)
```

An earlier version of this file recorded the gate as INCONCLUSIVE, because Agent-tool
subagents could not run at all. That was a harness fault, since resolved, and the gate has now
run properly. This file replaces that record.

## What the gate actually caught

The adversarial phase failed twice before passing, and both failures were substantive. This
section exists because a bare "PASS" would hide the fact that the record shipped to `main`
Accepted, on 2026-08-19, containing a correctness gap and an unargued bundle.

### Round 1: FAIL, seven findings, three HIGH

1. **Time-of-check-to-time-of-use race.** `relate` as designed checked liveness, then called
   `link`, which is a bare `INSERT` (`graph.rs:170`). A concurrent `supersede` between the two
   steps writes an edge to a node that is no longer live. The record's own Driver 5 treated
   atomicity as load-bearing for `supersedes` and then failed to apply it here.
2. **Scope creep with no alternatives weighed.** Three of six bundled changes (deleting the
   `cluster` feature, GC-tick recompute, the `clusters` tool) appeared only in Decision prose,
   never in Drivers or Alternatives.
3. **Duplicate `relate` biases Leiden.** The record called this "junk that pollutes clusters".
   `cluster.rs:11` states "A repeated pair raises weight" and `detect` calls `add_edge` once
   per row, so duplicates distort the algorithm's input rather than merely adding noise.
4. Arbitrary relation types weigh the same as `mentions`, since clustering excludes only
   `supersedes` (MEDIUM/HIGH).
5. "One community per version chain" was stated as settled fact in Decision Drivers when it is
   inference (MEDIUM), matching the fact-check's single WARN.
6. A write-time-by-id complement was never evaluated (MEDIUM).
7. The `cluster` feature deletion's blast radius was overstated (LOW).

Every load-bearing claim was verified against the source before acting on it.

### Round 2: FAIL, and the first fix was wrong

The response to finding 1 was "do the check and the insert inside one `execute_atomic`". That
described a mechanism the codebase cannot execute. `Backend::execute_atomic`
(`backend.rs:53`) takes a statement list built before the call and returns `Result<()>`, and
the libSQL implementation loops `tx.execute` rather than `query` (`libsql.rs:247`), so nothing
inside the transaction can read a row and branch on it. `supersede` was not the precedent it
was cited as: its `exists_as_of` runs before the transaction is built (`graph.rs:132`) and the
real guard is the `WHERE id = ?2 AND tx_to = ?3` clause on its UPDATE (`graph.rs:141`).

Had that revision been implemented literally, the SELECT's result would have been discarded and
the INSERT would have run unconditionally, reproducing the exact bug round 1 identified. The
corrected design is a single conditional write guarded by its own `WHERE`, which needs no new
`Backend` capability.

The same round rejected an idempotency claim resting on a constraint that does not exist:
`edges` declares only `id` as PRIMARY KEY, and both indexes are non-unique (`schema.rs:56`,
`:66`, `:67`), so `INSERT OR IGNORE` would have nothing to ignore.

Two further round-2 findings against ADR-0002 (a staleness check blind to deletion, and one
that false-positives on `supersedes` churn) had already been found independently and fixed
while that round was in flight, so it reviewed superseded text on those two. The independent
convergence is worth recording: both the author and the reviewer reached the same two flaws
separately.

### Round 3: PASS

Every code claim in both records was re-verified against source. The reviewer additionally
confirmed the proposed SQL is implementable by finding an existing precedent for its repeated
numbered parameters (`Graph::neighbors`, `graph.rs:353`, reuses `?1` and `?2` across a UNION
through the identical binding path), and exhaustively grepped the writers to `edges` to confirm
ADR-0002's claim that only `link`, `supersede`, and `gc` mutate that table.

It also stress-tested whether `gc` deleting nodes could leave `node_community` rows invisible
to an edge-only fingerprint, and found it cannot: a node reaches `node_community` only as an
edge endpoint, and `gc` always runs its edge-orphan sweep in the same call that deletes nodes,
so any node deletion that matters moves the count.

## Verified independently of the reviewer

- All 26 `file:line` citations in ADR-0001 resolve. `grep -n "\.link("` returns exactly the
  five lines the record names, all after `mod tests` at `:694`.
- All Mermaid diagrams in both records parse. One did not: a semicolon inside a
  `Note over` line acts as a mermaid statement separator and broke the ADR-0001 sequence
  diagram, which would have rendered as an error block on GitHub. Fixed.
- `ask::Evidence::from_hit` (`ask.rs:51`) never reads `Hit::id`, and `eval.rs:470` drives
  `ask`, so rendering ids in `recall` cannot regress the grounding eval.

## Outstanding

- **ADR-0002 is Status: Proposed and needs owner approval.** It passed the gate, but its
  central choice, recomputing lazily on read instead of on the GC tick, has not been approved
  by the person who owns the roadmap.
- No execution blueprint exists for either record, so no work-unit graph, file plan, or test
  plan has been reviewed. Test Review is N/A until one does.
- The fingerprint in ADR-0002 is a documented heuristic with two disclosed gaps, in-place row
  mutation and a same-millisecond tie. Neither is currently reachable, both are recorded.
