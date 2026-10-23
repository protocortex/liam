// SPDX-License-Identifier: Apache-2.0
//! liam-model: embedding and reranking for liam.
//!
//! The store is model-free by design. This crate holds the `Embedder` and
//! `Reranker` traits plus adapters, so the daemon can embed before writing and
//! rerank after retrieving without the store ever depending on a model runtime.

mod bge;
#[cfg(feature = "local")]
mod blocking;
pub mod embedder;
pub mod error;
pub mod llm;
pub mod reranker;

#[cfg(feature = "llama")]
pub mod llama;

pub use bge::BGE_M3_MODEL_ID;
pub use embedder::{Embedder, MockEmbedder, TextKind, QWEN3_EMBEDDING_MODEL_ID};
pub use error::{ModelError, Result};
pub use llm::{Llm, MockLlm};
pub use reranker::{IdentityReranker, Reranker};

#[cfg(feature = "local")]
pub use bge::BgeM3Embedder;
#[cfg(feature = "local")]
pub use embedder::FastEmbedEmbedder;
#[cfg(feature = "local")]
pub use reranker::FastEmbedReranker;

#[cfg(feature = "llama")]
pub use llama::LlamaCppLlm;
