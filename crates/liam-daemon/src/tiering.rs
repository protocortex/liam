// SPDX-License-Identifier: Apache-2.0
//! Moving items between the hot and cold tiers by how useful they have been.
//! The utility score and the planner are pure: they read one frozen snapshot
//! of usage and decide, and applying the decision is a separate step.

// The score and the planner are reached only from tests until the tier step
// of the consolidation job runs them.
#![cfg_attr(not(test), allow(dead_code))]

mod planner;
pub(crate) mod usage;
mod utility;
