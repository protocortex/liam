// SPDX-License-Identifier: Apache-2.0
//! liam-model: embedding and generation for liam.
//!
//! The store is model-free by design. This crate holds the `Embedder` and
//! `Llm` traits plus adapters, so the daemon can embed before writing without
//! the store ever depending on a model runtime.

mod arctic;
mod blocking;
mod download_notice;
pub mod embedder;
pub mod error;
pub mod hardware;
pub mod llm;
mod onnx_pinned;
mod pinned_verify;

pub mod llama;

pub use arctic::{ArcticEmbedder, EMBEDDER_MODEL_ID};
pub use download_notice::NoticeCaller;
pub use embedder::{Embedder, MockEmbedder, TextKind};
pub use error::{ModelError, Result};
pub use llama::LlamaCppLlm;
pub use llm::{Llm, MockLlm};
pub use onnx_pinned::{PinnedFile, PinnedOnnxModel, ARCTIC_EMBED_M_V2};
pub use pinned_verify::verify_pinned_file;
