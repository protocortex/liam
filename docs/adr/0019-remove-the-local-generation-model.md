# ADR-0019: remove the local generation model and llama.cpp from the daemon

- **Status:** Accepted. Supersedes in part [ADR-0015](0015-load-the-llm-on-first-use-and-unload-it-when-idle.md) and [ADR-0012](0012-llama-cpp-default-embedder-and-hardware-support.md) where they describe the in-process generation model, its loading and `llm.device`, and in part [ADR-0017](0017-remove-the-ask-tool.md) and [ADR-0018](0018-build-entity-pages-without-a-model-and-show-them-in-timeline.md) where they say the local model stays for local summaries.
- **Date created:** 2026-11-20
- **Date modified:** 2026-11-20

## Context

The owner decided that the daemon holds one model: the arctic embedder.

The in-process generation model was Qwen2.5 1.5B at Q4_K_M on llama.cpp. It had
three jobs. [ADR-0017](0017-remove-the-ask-tool.md) removed the first, writing
the answer of `ask`. [ADR-0018](0018-build-entity-pages-without-a-model-and-show-them-in-timeline.md)
removed the second, writing entity pages. Two uses were left:

- **Local summaries.** The consolidation job could ask the model to write
  summaries. They are off by default, and a hosted provider already writes them
  through the `[hosted]` section.
- **Counting tokens for `clusters`.** The tool cut its listing at a tenth of
  `llm.context_tokens`, counted with the model's tokenizer, and fell back to a
  four characters per token estimate when no model was loaded.

What the model cost, measured by an earlier investigation on an M1 Pro with
Qwen2.5-1.5B Q4, a mock embedder and a debug build
([ADR-0015](0015-load-the-llm-on-first-use-and-unload-it-when-idle.md)): the
daemon held 64 MB at start and 1,195 MB after one generation call, for the
model alone. The model file was an extra download. The code does not pin its
size, so this ADR gives none.

It also cost code. llama.cpp was compiled into the binary through
`llama-cpp-2`, with a Metal and GPU build path, a device probe, `liam devices`,
a lazy loader with an idle unloader, a start-up benchmark that tuned how many
generations ran at once and its cache file, and CI and release workarounds for
the CMake build.

The extraction sidecar is a different thing. It runs a separate `llama-server`
program that `liam fetch-models` downloads, with its own model, and it is off by
default. It links nothing from the deleted code: it shares no type with it and
does not use `llama-cpp-2`.

## Considered Alternatives

### Keep the model for local summaries and the cluster budget (effort: none)

- Trade-offs: a second model in the daemon, 1,195 MB resident after use, a
  large native build, for two small uses. Rejected.

### Keep the model but make it opt-in (effort: M)

- Trade-offs: the code, the build and the Metal path stay for everyone while
  few use them. Rejected.

### Remove the model and let a hosted provider write summaries (effort: M, chosen)

- Trade-offs: summaries need a hosted provider and its privacy gate, and are off
  otherwise. The default install loses nothing it used.

## Decision

1. **Remove the generation model.** Delete `llama.rs`, its shared backend, the
   lazy loader, the `Llm` trait, the mock model, the `[llm]` config, generation
   autotuning and its cache file, the device probe, `DevicePreference` and
   `liam devices`. Remove `llama-cpp-2` and the crates only it needed.
2. **Summaries are hosted or off.** `[summary] provider = "local"` no longer
   works. A config that sets it, or that turns summaries on with the old default
   provider, loads with summaries off and logs one warning at start. A hosted
   provider works as before.
3. **An old `[llm]` section is accepted and ignored.** The daemon logs one
   warning at start. The removed-key warnings share one helper.
4. **`clusters` cuts at a character budget.** The budget is a named constant of
   3,276 characters, which is a tenth of an 8,192-token context at four
   characters a token, the share the tool had by default.
5. **Consolidation runs stop recording an LLM identity.** A saved run no longer
   carries `llm_model_fingerprint`, and a run is no longer refused for a changed
   generation model. A manifest that still has the key loads, because the
   manifest reader ignores unknown keys.
6. **`liam fetch-models` fetches the embedder, and what extraction needs.** It no
   longer fetches the generation GGUF.
7. **The extraction sidecar is untouched.** It still downloads its own
   `llama-server` and model when extraction is enabled.

## Consequences

- **The daemon holds one model.** Memory after use is the embedder's, not the
  embedder's plus 1,195 MB.
- **A smaller binary and build.** No vendored C++ build, no CMake workaround in
  CI or release, no Metal or GPU build path. The embedder runs on the CPU
  through ONNX Runtime and has no device to choose.
- **Local summaries are gone.** A user who wanted them needs a hosted provider
  and `[hosted]`, and turns `metrics.enabled` on.
- **`liam devices` is gone.** Nothing chooses a device any more.
- **Old files stay on disk.** `~/.liam/models` may hold the old GGUF. LIAM no
  longer reads it, and the user may delete it.
- **`clusters` output can differ.** The old budget came from the model's
  tokenizer when one was loaded. A listing now fits 3,276 characters, so a very
  large listing can hold a different number of clusters.
- **A way back.** A local generation model can return through a new ADR.
