// SPDX-License-Identifier: Apache-2.0
//! Turning log events into subject/predicate/object facts.
//!
//! Nothing calls this module until the extraction hook is wired into
//! consolidation; the allow below goes away with that wiring, so a fact source
//! that is never reached still fails the lint afterwards.
#![cfg_attr(not(test), allow(dead_code))]

mod iso_date;
pub mod response;
pub mod rules;
