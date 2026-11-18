# ADR-0015: load the LLM on first use and unload it when idle

- **Status:** Accepted
- **Date created:** 2026-11-18
- **Date modified:** 2026-11-18

## Context

The daemon runs in the background all day. It loaded the answer LLM (Qwen2.5-1.5B
Q4, 8192 context, llama.cpp) at start with `warmup = true` and kept it for the
life of the process. A memory investigation on an M1 Pro put that at about 1.0 to
1.3 GB resident, for a model that most of the day nobody uses. The embedder is
different: every `recall` and `remember` needs it, so it stays loaded.

## Considered Alternatives

### Keep the model loaded (effort: none)

- Trade-offs: no first-request delay. Holds 1.0 to 1.3 GB around the clock.
  Rejected.

### Load on first use and unload after an idle period (effort: M, chosen)

- Trade-offs: memory returns to the system when the daemon is idle. The first
  request after an idle period pays the load.

### Unload on memory pressure (effort: L)

- Trade-offs: needs an operating system signal and per platform code. Rejected
  as more than the need.

## Decision

1. **Wrap the llama.cpp model in a `LazyLlm`.** It implements the `Llm` trait and
   loads the model on the first call that needs it. Callers that arrive while a
   load runs share that one load. A failed load returns its error to those
   callers and the next call tries again. The mock provider is not wrapped.
2. **The load is its own task.** A request pays the load inside its own deadline.
   When the deadline passes first, the request takes the same fallback it takes
   for a slow model, and the load carries on so the next request finds the model
   ready.
3. **Unload when idle.** A task checks every 30 s and drops the model after
   `llm.idle_unload_secs` (default `300`) without a generation. `0` never
   unloads. The task waits for calls in flight: it never unloads while one holds
   the model, and the idle period counts from the end of the last call, even when
   a request is cancelled mid call. The task stops with the daemon's shutdown token.
4. **`llm.warmup` now defaults to `false`.** With `true` the model loads at start
   and still unloads when idle.
5. **Counting tokens needs a loaded model.** `ask` and entity synthesis load the
   model, inside the request deadline, before they count tokens and trim evidence
   and before the generation timer starts, so the count is the model's own and
   the load time stays out of the concurrency tuner's window. The character
   based estimate remains for a provider that cannot count. `liam fetch-models` still loads the model, since it exists to
   prove that it loads.

## Consequences

- **The first request after an idle period is slower** by the load time. The
  operator who wants the old behaviour sets `llm.warmup = true` and
  `llm.idle_unload_secs = 0`.
- **Memory returns to the system after the idle period.** On an M1 Pro with
  Qwen2.5-1.5B Q4, a mock embedder and a debug build, the daemon held 64 MB at
  start and 1,195 MB after one ask. After the idle limit it held 64 MB again,
  and the reload on the next ask took about 0.3 s.
- **A bad or missing model file is found later.** Start used to fail on it. With
  `warmup = false` it is found on the first ask, and start only logs one warning
  when the model file is missing from the cache. With `warmup = true` a model
  that fails to load still stops the daemon at start.
- **A cold tuning run loads the model at start.** With no cached concurrency
  result, the one time benchmark generates, so it loads the model once.
