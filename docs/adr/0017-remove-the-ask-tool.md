# ADR-0017: remove the ask tool

- **Status:** Accepted. Supersedes in part [ADR-0001](0001-assert-memory-edges-by-node-id.md), [ADR-0004](0004-non-lossy-mcp-surface-for-attributes-confidence-and-as-of.md), [ADR-0014](0014-remove-the-reranker.md) and [ADR-0015](0015-load-the-llm-on-first-use-and-unload-it-when-idle.md) where they describe `ask` as part of the tool surface.
- **Note, 2026-11-20:** entity pages no longer use the local model, and only `timeline` shows them. See [ADR-0018](0018-build-entity-pages-without-a-model-and-return-them-from-recall.md).
- **Date created:** 2026-11-19
- **Date modified:** 2026-11-19

## Context

`ask` retrieved evidence, as `recall` does, and then had the in-process 1.5B
model write a short cited answer from it.

- **The caller is an agent.** Every client of LIAM is an agent with a far
  stronger model than the 1.5B one. It can write its own answer from what
  `recall` returns.
- **Mostly the same evidence.** In strict mode `ask` ranked recorded memory
  only, as `recall` does, and added a written answer on top. In infer mode, its
  default, `ask` also ranked entity pages and summaries beside recorded memory.
  `recall` is strict only and refuses infer, so it never returned them.
- **A weak last step.** The answer came from a small local model, so it needed
  a sufficiency pre-pass, a grounding check and a fallback to raw evidence to
  stay safe. Without the model, `ask` would be the same tool as `recall`.

The owner decided on 2026-11-19 to keep `recall` as the only read-by-query tool.

## Considered Alternatives

### Keep ask (effort: none)

- Trade-offs: a caller with no model of its own still gets a written answer.
  Keeps a second tool, a pre-pass, a grounding check and two code paths that
  `recall` does not need. Rejected.

### Keep ask and point it at a hosted model (effort: M)

- Trade-offs: a better answer, but it sends memory to a provider for a step the
  calling agent already does better. Rejected.

### Remove ask and keep recall (effort: M, chosen)

- Trade-offs: one read tool and less code. A caller with no model of its own
  gets evidence and no answer.

## Decision

1. **The `ask` tool is removed.** There was no `liam ask` command.
2. **`recall` is unchanged.** Same input schema, same output, same order, same
   deadline and same degraded reply. Its output schema no longer lists
   `synthesis_unavailable`, which only `ask` produced.
3. **The local model stays for now.** Entity pages and local summaries still
   use it, with the `Llm` trait, the `[llm]` section, lazy loading, the
   generation tuner and llama.cpp. What replaces it is a separate decision.
4. **Config keys.** `ask_sufficiency_check` is removed: an old line is ignored
   and logs one warning at start. `ask_timeout_secs` stays, because one batch of
   entity-page syntheses still reads it. `llm.context_tokens` stays, because it
   sizes the llama.cpp context and the mention trimming of entity synthesis.
5. **Metrics.** `ask_latency_p50_ms` and `ask_latency_p95_ms` and the `ask`
   label of `degraded_replies` are removed.

## Consequences

- **No written answer.** A caller with no model of its own gets recalled
  memories and writes nothing.
- **Entity pages and summaries are no longer ranked by a query.** `ask` in
  infer mode was the one tool that ranked them next to recorded memory. Read
  entity pages through `timeline` and summaries through `summaries`. `recall`
  returns recorded memory only, as it did before.
- **The store keeps its ranking of inferred rows.** The query option
  `include_inferred` and the `as_of` argument of `summaries_within` stay, with
  their tests, though no daemon tool sets them now.
- **Open decision, resolved on 2026-11-20.** `recall` gains no reading mode: it
  ranks neither entity pages nor summaries, see [ADR-0018](0018-build-entity-pages-without-a-model-and-return-them-from-recall.md).
- **A breaking change for clients that called `ask`.** The call returns the
  standard unknown-tool error. They use `recall`.
- **Less to test.** The answer eval, the sufficiency pre-pass and the answer
  prompt are gone, and so is the recall benchmark phase that held an `ask` in
  flight.
- **The model's future is open.** Whether the local model stays for entity pages
  and local summaries is a separate decision.
- **A way back.** An answer tool can return through a new ADR.
