//! `Llm` over llama.cpp, behind the `llama` feature.
//!
//! The backend is process-global: llama.cpp allows exactly one
//! `LlamaBackend::init()` per process, so `shared_backend` initializes it once
//! and every `LlamaCppLlm` shares the result.
//!
//! Each generation opens a fresh context rather than reusing one. That is
//! deliberate: a per-call context makes stale-KV-cache bugs structurally
//! impossible instead of something a caller has to remember to clear. The
//! candle path already shipped exactly that bug, a model answering from a
//! previous question's keys, which is why this module does not take the
//! shortcut.
//!
//! The chat template comes from the GGUF file itself, not a hand-written
//! table. llama.cpp reads the template baked into the model and renders it, so
//! a new architecture needs no template code here.

use async_trait::async_trait;
use std::num::NonZeroU32;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::{ModelError, Result};
use crate::llm::{strip_reasoning, CancelFlag, Llm};

/// Cap on generated tokens so a model with no natural stop token cannot hang a
/// caller. Matches the candle path's cap so callers see the same behavior
/// regardless of which engine answered.
const MAX_NEW_TOKENS: usize = 512;

/// Floor for `context_tokens` when a caller passes 0. Large enough for the
/// one-token-in, one-token-out warmup probe; a real question's window comes
/// from config, not this fallback.
const MIN_CONTEXT_TOKENS: u32 = 512;

/// In-process llama.cpp. The model loads once; each generation gets its own
/// context and runs on a blocking thread so the async runtime stays free.
pub struct LlamaCppLlm {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    /// Context window for every per-call context, set once in `load` from the
    /// caller-supplied `context_tokens`. See `load`'s doc for why that number
    /// is not a constant here.
    context_tokens: NonZeroU32,
    device_label: &'static str,
}

impl LlamaCppLlm {
    /// Load a GGUF from a local path. `context_tokens` becomes the context
    /// window for every generation this instance runs. It is a parameter and
    /// not a constant because the daemon's `llm.context_tokens` config also
    /// sizes prompt trimming: if the two drifted apart, a full-size prompt
    /// would overflow a smaller llama.cpp context and decode would fail. A
    /// caller passing 0 gets a documented floor instead of a panic.
    pub fn load(gguf_path: &str, context_tokens: u32) -> Result<Self> {
        let backend = shared_backend()?;
        // Offload everything the backend will take; with no GPU compiled in
        // this is ignored and the model stays on CPU.
        let params = LlamaModelParams::default().with_n_gpu_layers(1000);
        let model = LlamaModel::load_from_file(&backend, gguf_path, &params)
            .map_err(|e| ModelError::Llm(format!("load {gguf_path}: {e}")))?;
        let device_label = runtime_backend(&backend);
        let context_tokens = NonZeroU32::new(context_tokens)
            .unwrap_or_else(|| NonZeroU32::new(MIN_CONTEXT_TOKENS).expect("512 is nonzero"));
        Ok(Self {
            backend,
            model: Arc::new(model),
            context_tokens,
            device_label,
        })
    }

    /// Render a system and user turn with the template embedded in the GGUF.
    /// This is what replaces `ChatArch`: the model states its own format
    /// instead of this crate maintaining a hand-written table per
    /// architecture.
    fn render_prompt(&self, system: &str, user: &str) -> Result<String> {
        let template = self
            .model
            .chat_template(None)
            .map_err(|e| ModelError::Llm(format!("this GGUF carries no chat template: {e}")))?;
        let msg = |role: &str, text: &str| {
            LlamaChatMessage::new(role.to_string(), text.to_string())
                .map_err(|e| ModelError::Llm(e.to_string()))
        };

        // Try a real system turn first, then fold the system text into the
        // user turn. Some families (Gemma) have no system role at all and the
        // C API just returns an FFI error for it, so one fallback here covers
        // every architecture instead of a per-family template.
        let with_system = vec![msg("system", system)?, msg("user", user)?];
        if let Ok(rendered) = self
            .model
            .apply_chat_template(&template, &with_system, true)
        {
            return Ok(rendered);
        }
        let merged = vec![msg("user", &format!("{system}\n\n{user}"))?];
        self.model
            .apply_chat_template(&template, &merged, true)
            .map_err(|e| ModelError::Llm(format!("apply chat template: {e}")))
    }

    /// Greedy decode: render the prompt, prefill it, then sample one token at
    /// a time until end-of-generation, the cap, or cancellation. A fresh
    /// context is built here, per call, for the reason in the module doc.
    fn generate_blocking(
        &self,
        system: &str,
        user: &str,
        max_new_tokens: usize,
        cancel: &CancelFlag,
    ) -> Result<String> {
        let prompt = self.render_prompt(system, user)?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| ModelError::Llm(format!("tokenize: {e}")))?;

        // Thread pinning to P-cores and a Q8_0 KV cache were both measured
        // against these defaults on Metal (P-core pinning cost 3 percent,
        // Q8_0 KV cost 9 percent) and both lost, so this stays at the engine
        // defaults; do not re-add either as an "optimization" without a new
        // measurement.
        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(self.context_tokens));
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| ModelError::Llm(format!("context: {e}")))?;

        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        let last = tokens.len().saturating_sub(1) as i32;
        for (i, token) in (0i32..).zip(tokens) {
            batch
                .add(token, i, &[0], i == last)
                .map_err(|e| ModelError::Llm(format!("batch: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| ModelError::Llm(format!("prefill: {e}")))?;

        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        // Absolute sequence position of the next token: generation continues
        // from the end of the prompt, so positions are prompt length plus
        // step.
        let prompt_len = batch.n_tokens();

        for step in 0..max_new_tokens as i32 {
            // Checked once per token: a blocking decode loop cannot be
            // stopped from outside except by looking at this flag, and the
            // daemon's `ask` timeout drops the caller's future without
            // stopping this thread. What was generated so far is returned
            // rather than an error, since the caller is already gone and
            // would discard either result.
            if cancel.is_cancelled() {
                break;
            }
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            out.push_str(
                &self
                    .model
                    .token_to_piece(token, &mut decoder, true, None)
                    .map_err(|e| ModelError::Llm(format!("detokenize: {e}")))?,
            );
            batch.clear();
            batch
                .add(token, prompt_len + step, &[0], true)
                .map_err(|e| ModelError::Llm(format!("batch: {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| ModelError::Llm(format!("decode: {e}")))?;
        }
        Ok(out)
    }

    /// Shared decode driver for both trait entry points, mirroring
    /// `CandleLlm::generate`. A `CancelFlag` gives a dropped caller future a
    /// way to reach into the blocking decode loop, since nothing else can
    /// stop a blocking thread from outside.
    async fn generate(&self, system: &str, user: &str, max_new_tokens: usize) -> Result<String> {
        let cancel = CancelFlag::new();
        // Dropping this future (the daemon's ask timeout firing, a client
        // going away) drops the guard, which flips the flag; the decode loop
        // notices within one token and hands the thread back.
        let _cancel_guard = cancel.cancel_on_drop();

        let backend = self.backend.clone();
        let model = self.model.clone();
        let context_tokens = self.context_tokens;
        let device_label = self.device_label;
        let system = system.to_string();
        let user = user.to_string();
        let cancel_for_worker = cancel.clone();

        // Same shape as the candle path: the sync decode loop runs off the
        // async runtime. Rebuilding `Self` here is cheap; every field is an
        // `Arc`, a `Copy` integer, or a `'static` label.
        let out = tokio::task::spawn_blocking(move || {
            let this = LlamaCppLlm {
                backend,
                model,
                context_tokens,
                device_label,
            };
            this.generate_blocking(&system, &user, max_new_tokens, &cancel_for_worker)
        })
        .await
        .map_err(|e| ModelError::Llm(e.to_string()))??;

        // Reasoning models answer after a `<think>` block even when the
        // template asks them not to; strip it so callers get the answer, not
        // the scratchpad, matching the candle path.
        Ok(strip_reasoning(&out).to_string())
    }
}

/// llama.cpp's backend is process-global: a second `LlamaBackend::init()`
/// returns `BackendAlreadyInitialized`. So it is initialized exactly once,
/// cached, and shared; a failed init stays failed rather than being retried
/// per call.
fn shared_backend() -> Result<Arc<LlamaBackend>> {
    static BACKEND: std::sync::OnceLock<std::result::Result<Arc<LlamaBackend>, String>> =
        std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| {
            LlamaBackend::init()
                .map(Arc::new)
                .map_err(|e| format!("llama backend: {e}"))
        })
        .clone()
        .map_err(ModelError::Llm)
}

/// Which llama.cpp backend this binary was built with. Compile-time, because
/// llama.cpp picks its GPU backend at build time, like candle does. Metal is
/// keyed on the TARGET, not a cargo feature: `llama-cpp-sys-2` compiles Metal
/// in for every macOS build regardless of feature flags, and this project has
/// already lost a measurement to a Metal check keyed on the wrong feature, so
/// this does not repeat it. Add a `#[cfg(...)]` arm here the day this crate
/// actually links a CUDA or Vulkan build; there is nothing to key today, so
/// nothing else is added.
fn compiled_backend() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "llama.cpp/metal"
    }
    #[cfg(not(target_os = "macos"))]
    {
        "llama.cpp/cpu"
    }
}

/// Refine the compile-time label with a runtime check. `compiled_backend` only
/// knows what llama.cpp was built with; `LlamaBackend::supports_gpu_offload`
/// (the wrapped `llama_supports_gpu_offload`) asks the ggml backend registry
/// whether a GPU device actually came up, so a macOS build that compiled
/// Metal in but found no usable device at runtime reports cpu instead of a
/// compile-time claim nothing can fall back from. A later startup assertion
/// depends on that: it can only catch a Metal fallback if this label is able
/// to say cpu.
fn runtime_backend(backend: &LlamaBackend) -> &'static str {
    if compiled_backend() == "llama.cpp/metal" && !backend.supports_gpu_offload() {
        "llama.cpp/cpu (metal unavailable)"
    } else {
        compiled_backend()
    }
}

#[async_trait]
impl Llm for LlamaCppLlm {
    async fn complete(&self, system: &str, prompt: &str) -> Result<String> {
        self.generate(system, prompt, MAX_NEW_TOKENS).await
    }

    async fn complete_capped(
        &self,
        system: &str,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String> {
        self.generate(system, prompt, max_new_tokens.max(1)).await
    }

    fn backend(&self) -> &'static str {
        self.device_label
    }

    /// Generate one token and throw it away, to pay the backend's first-call
    /// kernel-compile cost before a user is waiting, same reason as the
    /// candle path's `warmup`.
    async fn warmup(&self) -> Result<()> {
        self.generate("You are a warmup probe.", "ok", 1)
            .await
            .map(|_| ())
    }

    /// `None` on a tokenize failure rather than a wrong count, so the caller
    /// falls back to its own estimate instead of trusting a made-up number.
    fn count_tokens(&self, text: &str) -> Option<usize> {
        self.model
            .str_to_token(text, AddBos::Never)
            .ok()
            .map(|tokens| tokens.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn compiled_backend_reports_metal_on_a_macos_build() {
        // Arrange: this test only runs on a macOS build with the llama
        // feature, matching the Gherkin scenario's Given.

        // Act
        let backend = compiled_backend();

        // Assert: Metal is compiled in by TARGET (llama-cpp-sys-2), not by a
        // cargo feature, so a macOS build must report it even though no
        // `llama-metal` feature exists to key on.
        assert_eq!(backend, "llama.cpp/metal");
    }

    #[test]
    fn shared_backend_is_initialized_once_and_reused_on_later_calls() {
        // Arrange: call it once so a second call exercises the cached path.
        let first = shared_backend().expect("first backend init");

        // Act
        let second = shared_backend().expect("second backend init must not fail");

        // Assert: the same underlying instance comes back, and no
        // `BackendAlreadyInitialized` error surfaced from the second call,
        // which is the bug a fresh `LlamaBackend::init()` per call would hit.
        assert!(Arc::ptr_eq(&first, &second));
    }
}
