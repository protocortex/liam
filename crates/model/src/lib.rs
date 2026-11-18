// SPDX-License-Identifier: Apache-2.0
//! liam-model: embedding and generation for liam.
//!
//! The store is model-free by design. This crate holds the `Embedder` and
//! `Llm` traits plus adapters, so the daemon can embed before writing without
//! the store ever depending on a model runtime.

mod bge;
mod blocking;
mod download_notice;
pub mod embedder;
pub mod error;
mod gguf_verify;
pub mod hardware;
mod llama_embedder;
pub mod llm;
mod onnx;
mod onnx_pinned;

pub mod llama;

pub use bge::{BGE_M3_DIMS, BGE_M3_MODEL_ID};
pub use embedder::{Embedder, MockEmbedder, TextKind, QWEN3_EMBEDDING_MODEL_ID};
pub use error::{ModelError, Result};
pub use llm::{Llm, MockLlm};
pub use onnx::{
    onnx_model_spec, OnnxModelSpec, OnnxSource, DEFAULT_EMBEDDER_MODEL_ID, ONNX_MODELS,
};
pub use onnx_pinned::{PinnedFile, PinnedOnnxModel, ARCTIC_EMBED_M_V2};

pub use bge::BgeM3Embedder;
pub use embedder::FastEmbedEmbedder;
pub use onnx::OnnxEmbedder;

pub use gguf_verify::verify_pinned_file;
pub use llama::LlamaCppLlm;
pub use llama_embedder::{
    gguf_embedder_spec, gguf_path_from_hub, verify_gguf_file, GgufEmbedderSpec, LlamaEmbedder,
    GGUF_EMBEDDERS, QWEN3_GGUF_EMBEDDER_MODEL_ID,
};
