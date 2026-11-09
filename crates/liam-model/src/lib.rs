// SPDX-License-Identifier: Apache-2.0
//! liam-model: embedding and reranking for liam.
//!
//! The store is model-free by design. This crate holds the `Embedder` and
//! `Reranker` traits plus adapters, so the daemon can embed before writing and
//! rerank after retrieving without the store ever depending on a model runtime.

mod bge;
mod blocking;
pub mod embedder;
pub mod error;
mod gguf_verify;
mod llama_embedder;
pub mod llm;
mod onnx;
pub mod reranker;

pub mod llama;

pub use bge::{BGE_M3_DIMS, BGE_M3_MODEL_ID};
pub use embedder::{Embedder, MockEmbedder, TextKind, QWEN3_EMBEDDING_MODEL_ID};
pub use error::{ModelError, Result};
pub use llm::{Llm, MockLlm};
pub use onnx::{onnx_model_spec, OnnxModelSpec, ONNX_MODELS};
pub use reranker::{IdentityReranker, Reranker};

pub use bge::BgeM3Embedder;
pub use embedder::FastEmbedEmbedder;
pub use onnx::OnnxEmbedder;
pub use reranker::FastEmbedReranker;

pub use llama::LlamaCppLlm;
pub use llama_embedder::{
    gguf_embedder_spec, gguf_path_from_hub, verify_gguf_file, GgufEmbedderSpec, LlamaEmbedder,
    GGUF_EMBEDDERS,
};
