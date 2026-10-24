// SPDX-License-Identifier: Apache-2.0
//! Turning log events into subject/predicate/object facts.

mod apply;
pub mod cache;
pub mod client;
pub mod extractor;
pub mod grounding;
pub mod hooks;
mod http;
pub mod identity;
pub mod inputs;
mod iso_date;
#[cfg(test)]
pub(crate) mod model_double;
pub mod prompt;
pub mod provider;
#[cfg(test)]
mod r10;
pub mod response;
pub mod rules;
pub mod setup;
pub mod sidecar;
#[cfg(test)]
mod test_double;
