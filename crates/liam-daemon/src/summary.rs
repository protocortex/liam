// SPDX-License-Identifier: Apache-2.0
//! Summaries of clusters of recorded nodes, built level by level into a tree.

pub mod file_cache;
pub mod generate;
pub mod hooks;
#[cfg(test)]
pub(crate) mod test_support;
pub mod tree;

use liam_store::{Basis, EventRef, InferredHit, Retrieved, Summary};

/// A summary as a retrieved row: inferred, resting on `sources`.
pub fn as_retrieved(summary: &Summary, sources: Vec<EventRef>) -> Retrieved {
    Retrieved::Inferred(InferredHit {
        text: summary.text.clone(),
        basis: Basis::Known(sources),
        derivation: summary.derivation,
        summary_level: Some(summary.level),
    })
}
