// SPDX-License-Identifier: Apache-2.0
//! Moving items between the hot and cold tiers by how useful they have been.
//! The utility score and the planner are pure: they read one frozen snapshot
//! of usage and decide, and applying the decision is a separate step.

mod apply;
pub(crate) use apply::TierReport as AppliedMoves;
mod caps;
#[cfg(test)]
mod exit_criteria;
pub(crate) mod hooks;
mod planner;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod usage;
mod utility;
mod verify;
