# ADR-0016: arctic-embed-m-v2.0 is the only embedder

- **Status:** Accepted. Supersedes in part [ADR-0013](0013-arctic-default-embedder-and-reranker-off.md) and [ADR-0012](0012-llama-cpp-default-embedder-and-hardware-support.md): arctic stays the embedder, and the other embedders they kept selectable are removed.
- **Date created:** 2026-11-19
- **Date modified:** 2026-11-19

## Context

ADR-0013 made Snowflake arctic-embed-m-v2.0 (int8 ONNX, CPU) the default
embedder and kept the others selectable: Qwen3-Embedding-0.6B on llama.cpp (Q8_0
and F16), Qwen3 on candle, BGE-M3 on candle, and seven small ONNX models.

- **Quality and cost.** On English, Portuguese and Spanish, arctic matched or
  beat Qwen3 Q8. It answers a query in about 8 ms against 37 ms, and its file is
  311 MB against about 640 MB.
- **No user.** Nobody selects the alternatives. They stayed in the code only
  because ADR-0013 left them selectable.
- **Untested code.** Their loaders are most of the lines no default test runs,
  and their tests are most of the `#[ignore]`d ones. Those tests need
  downloads of several GB, so they rarely run.
- **A heavy build.** The candle models pull in candle and its GPU crates, which
  the one remaining embedder does not use.

The owner decided on 2026-11-19 that the daemon is lighter with one model.

## Considered Alternatives

### Keep the alternatives selectable (effort: none)

- Trade-offs: keeps Arabic and Chinese quality for the one user who might want
  it. Keeps the untested loaders, the multi-GB tests, candle in the build, and
  a device setting that only the alternatives read. Rejected.

### Keep one alternative, Qwen3 Q8 on llama.cpp (effort: S)

- Trade-offs: a single fallback for Arabic and Chinese. llama.cpp stays in the
  build for generation, so the cost is its loader, its tests and a 640 MB
  download nobody asked for. Rejected for the same reason as the rest: no user.

### Remove every embedder but arctic (effort: M, chosen)

- Trade-offs: deletes four loaders, the ONNX model table and the device code
  that only they used. Arabic and Chinese quality drops.

## Decision

1. **Arctic is the only embedder.** The loader keeps what makes it safe: the
   pinned revision, size and sha256 checks, the `query: ` prefix, the cut from
   768 to the configured width with a new normalisation, and one text per model
   call.
2. **`embedder.model` stays as a key, and only the arctic id or no key loads.**
   Any other value stops the daemon and the `liam` commands at config load. The
   message names the value, says arctic is the only embedder since this
   version, and says to delete the line.
3. **`embedder.device` is removed.** An old `device` line is ignored and logs one
   warning at start, because it only tuned the removed backends and stopping on
   it would help nobody.
4. **The fingerprint check stays as it was.** A store built with another
   embedder is refused with the `liam rebuild` message.
5. **candle and the `cuda` feature leave the build.** The Metal probe that
   `auto` used now asks llama.cpp, which still runs generation. `liam eval
   retrieval` and `liam eval embedder` lose `--model` and `--device`, and
   `liam devices` stops reporting an embedder device.

## Consequences

- **One model.** One loader, one set of tests, one download of 311 MB.
- **A smaller build.** candle, its GPU kernels and the crates only they pulled in
  are gone from the dependency tree.
- **Weaker Arabic and Chinese.** Qwen3 was better on both. Arabic and Chinese
  users get lower quality than they would have had.
- **A breaking change for operators who named another model.** `liamd` and
  `liam` stop at start with a message until the line is deleted, and a store
  built with another embedder needs `liam rebuild`.
- **A way back.** A second embedder can return through a new ADR, with the
  numbers in ADR-0013 as the bar to beat.
