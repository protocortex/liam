// SPDX-License-Identifier: Apache-2.0
//! Turning log events into subject/predicate/object facts.
//!
//! Nothing calls this module until the extraction hook is wired into
//! consolidation; the allow below goes away with that wiring, so a fact source
//! that is never reached still fails the lint afterwards.
#![cfg_attr(not(test), allow(dead_code))]

mod apply;
pub mod cache;
pub mod client;
pub mod extractor;
pub mod grounding;
mod http;
pub mod inputs;
mod iso_date;
pub mod prompt;
#[cfg(test)]
mod r10;
pub mod response;
pub mod rules;
pub mod sidecar;
#[cfg(test)]
mod test_double;
