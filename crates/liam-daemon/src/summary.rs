// SPDX-License-Identifier: Apache-2.0
//! Summaries of clusters of recorded nodes, built level by level into a tree.
//!
//! Nothing outside this module's tests uses it yet; the allowance below goes
//! when the consolidation job calls the builder.
#![cfg_attr(not(test), allow(dead_code))]

pub mod generate;
#[cfg(test)]
mod test_support;
pub mod tree;
