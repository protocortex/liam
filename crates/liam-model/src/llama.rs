//! SPIKE: `Llm` over llama.cpp (feature `llama`), to be compared against
//! `CandleLlm` before any migration decision. Not wired into the daemon.
//!
//! Three things are being measured here, in order of why they matter:
//!
//! 1. **GPU coverage.** candle has exactly two GPU backends, CUDA and Metal (its
//!    whole feature list is accelerate/cuda/cudnn/metal/mkl/nccl/ug). llama.cpp
//!    has Vulkan, ROCm/HIP, OpenCL and SYCL on top of those, and Vulkan alone
//!    covers AMD, Intel Arc and NVIDIA in one build. No amount of work makes
//!    candle run on an AMD card, so this is the only path to "every GPU".
//! 2. **Who owns the chat template.** `CandleLlm` needs a hand-written template
//!    per architecture (`ChatArch`), which is a silent-wrong-answer surface: it
//!    already produced one (see `Weights::clear_cache`) and it cannot express
//!    Gemma 4's ~18KB Jinja template at all. llama.cpp reads the template out of
//!    the GGUF and applies it, so that table stops existing.
//! 3. **Constrained output.** M3 ingestion needs schema-valid JSON. candle offers
//!    no grammar; llama.cpp converts a JSON schema to GBNF and enforces it at the
//!    token level.

use async_trait::async_trait;
use std::num::NonZeroU32;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::{ModelError, Result};
use crate::llm::Llm;

/// Matches `candle_chat::MAX_NEW_TOKENS` so latency comparisons are apples to
/// apples.
const MAX_NEW_TOKENS: usize = 512;

/// Context window for a generation. Evidence prompts are long (up to 32 items of
/// 2000 chars), so 4096 is the floor for `ask`.
const N_CTX: u32 = 4096;

/// In-process llama.cpp. The backend is global to the process, the model is
/// loaded once, and each generation gets a FRESH context: that is deliberate,
/// because a per-call context makes stale-KV-cache bugs structurally impossible
/// rather than something we have to remember to clear.
pub struct LlamaCppLlm {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    device_label: &'static str,
}

impl LlamaCppLlm {
    /// Load a GGUF from a local path. WHY a path and not an HF id: the spike
    /// reuses the weights candle already downloaded, so both engines are measured
    /// on byte-identical files.
    pub fn load(gguf_path: &str) -> Result<Self> {
        let backend = shared_backend()?;
        // Offload everything the backend will take; with no GPU compiled in this
        // is ignored and the model stays on CPU.
        let params =
            llama_cpp_2::model::params::LlamaModelParams::default().with_n_gpu_layers(1000);
        let model = LlamaModel::load_from_file(&backend, gguf_path, &params)
            .map_err(|e| ModelError::Llm(format!("load {gguf_path}: {e}")))?;
        Ok(Self {
            backend,
            model: Arc::new(model),
            device_label: compiled_backend(),
        })
    }

    /// Render a system+user exchange with the template EMBEDDED IN THE GGUF.
    /// This is the function that would delete `ChatArch`: the model tells us its
    /// own format instead of us maintaining a table of five.
    fn render_prompt(&self, system: &str, user: &str) -> Result<String> {
        let template = self
            .model
            .chat_template(None)
            .map_err(|e| ModelError::Llm(format!("this GGUF carries no chat template: {e}")))?;
        let msg = |role: &str, text: &str| {
            LlamaChatMessage::new(role.to_string(), text.to_string())
                .map_err(|e| ModelError::Llm(e.to_string()))
        };

        // Try a real system turn first, then fall back to folding the system text
        // into the user turn. WHY the fallback: Gemma's template has no system
        // role and the C API just returns `ffi error -1` for it (measured on
        // gemma-4-E2B). So engine-applied templates remove the per-architecture
        // TABLE, but not the fact that some families lack a system role. One
        // fallback here replaces five hand-written templates.
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

    /// Greedy decode, mirroring `candle_chat::Session::complete` so the two are
    /// comparable: same cap, same sampling, same stop condition.
    fn generate_blocking(&self, system: &str, user: &str, max_new_tokens: usize) -> Result<String> {
        let prompt = self.render_prompt(system, user)?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .map_err(|e| ModelError::Llm(format!("tokenize: {e}")))?;

        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(N_CTX));
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
        // Absolute sequence position of the next token: generation continues from
        // the end of the prompt, so positions are prompt length plus step.
        let prompt_len = batch.n_tokens();

        for step in 0..max_new_tokens as i32 {
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

    async fn generate(&self, system: &str, user: &str, max_new_tokens: usize) -> Result<String> {
        let model = self.model.clone();
        let backend = self.backend.clone();
        let system = system.to_string();
        let user = user.to_string();
        let device_label = self.device_label;
        // Same shape as the candle path: the sync decode loop runs off the async
        // runtime. A fresh `Self` view is cheap because both handles are Arcs.
        tokio::task::spawn_blocking(move || {
            let this = LlamaCppLlm {
                backend,
                model,
                device_label,
            };
            this.generate_blocking(&system, &user, max_new_tokens)
        })
        .await
        .map_err(|e| ModelError::Llm(e.to_string()))?
    }
}

/// llama.cpp's backend is process-global: a second `LlamaBackend::init()` returns
/// `BackendAlreadyInitialized`. Measured the hard way, by loading two models in
/// one process. So it is initialized exactly once and shared.
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
/// llama.cpp selects its backend at build time like candle does.
fn compiled_backend() -> &'static str {
    if cfg!(feature = "llama-metal") {
        "llama.cpp/metal"
    } else if cfg!(feature = "llama-cuda") {
        "llama.cpp/cuda"
    } else if cfg!(feature = "llama-vulkan") {
        "llama.cpp/vulkan"
    } else if cfg!(feature = "llama-rocm") {
        "llama.cpp/rocm"
    } else if cfg!(feature = "llama-opencl") {
        "llama.cpp/opencl"
    } else {
        "llama.cpp/cpu"
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

    async fn warmup(&self) -> Result<()> {
        self.generate("You are a warmup probe.", "ok", 1)
            .await
            .map(|_| ())
    }
}
