// SPDX-License-Identifier: Apache-2.0
//! The privacy gate every item passes before it goes into a hosted prompt. It
//! is deterministic and denies by default: an item is sent only when its scope
//! is allowed, none of its sources is local only, and the provider's region is
//! attested for the scope. A denied item is left out and counted by reason, and
//! the count never carries the item's content.

use std::collections::BTreeMap;
use std::fmt;

use liam_store::{scope_is_within, SummaryInput};

use super::cache::CacheStore;
use super::inputs::ExtractionInput;
use crate::config::PrivacyConfig;

/// What the gate looks at in one item. The unscoped group has no scope.
pub struct GateItem<'a> {
    pub scope: Option<&'a str>,
    pub sources: &'a [String],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DenyReason {
    ScopeNotAllowed,
    LocalOnlySource,
    RegionNotAttested,
}

impl fmt::Display for DenyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ScopeNotAllowed => "scope_not_allowed",
            Self::LocalOnlySource => "local_only_source",
            Self::RegionNotAttested => "region_not_attested",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny { reason: DenyReason },
}

/// How many items were denied, by reason.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GateCounts(BTreeMap<DenyReason, usize>);

impl GateCounts {
    pub fn of(&self, reason: DenyReason) -> usize {
        self.0.get(&reason).copied().unwrap_or(0)
    }

    pub fn total(&self) -> usize {
        self.0.values().sum()
    }

    fn record(&mut self, reason: DenyReason) {
        *self.0.entry(reason).or_default() += 1;
    }
}

impl fmt::Display for GateCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .0
            .iter()
            .map(|(reason, count)| format!("{reason}={count}"))
            .collect();
        f.write_str(&parts.join(" "))
    }
}

pub struct PrivacyGate {
    policy: PrivacyConfig,
}

impl PrivacyGate {
    pub fn new(policy: PrivacyConfig) -> Self {
        Self { policy }
    }

    /// Whether `item` may go to `provider`. With no policy every item is denied.
    pub fn decide(&self, item: &GateItem<'_>, provider: &str) -> Decision {
        let scope = item.scope.unwrap_or_default();
        let allowed = self
            .policy
            .allowed_scopes
            .iter()
            .any(|allowed| scope_is_within(scope, allowed));
        if !allowed {
            return Decision::Deny {
                reason: DenyReason::ScopeNotAllowed,
            };
        }
        let local_only = item
            .sources
            .iter()
            .any(|source| self.policy.local_only_sources.contains(source));
        if local_only {
            return Decision::Deny {
                reason: DenyReason::LocalOnlySource,
            };
        }
        if self.region_attested(scope, provider) {
            Decision::Allow
        } else {
            Decision::Deny {
                reason: DenyReason::RegionNotAttested,
            }
        }
    }

    /// True when the provider's region is attested for `scope` or an ancestor.
    fn region_attested(&self, scope: &str, provider: &str) -> bool {
        let Some(region) = self.policy.provider_regions.get(provider) else {
            return false;
        };
        self.policy
            .attested_regions
            .iter()
            .any(|(attested, regions)| scope_is_within(scope, attested) && regions.contains(region))
    }

    /// The inputs `provider` may be sent, and a count of the rest. The local
    /// path keeps all of them: the caller filters only the hosted prompt.
    pub fn filter_inputs(
        &self,
        provider: &str,
        inputs: &[ExtractionInput],
    ) -> (Vec<ExtractionInput>, GateCounts) {
        let mut counts = GateCounts::default();
        let mut allowed = Vec::with_capacity(inputs.len());
        for input in inputs {
            let item = GateItem {
                scope: input.scope.as_deref(),
                sources: std::slice::from_ref(&input.source),
            };
            match self.decide(&item, provider) {
                Decision::Allow => allowed.push(input.clone()),
                Decision::Deny { reason } => counts.record(reason),
            }
        }
        report(provider, inputs.len(), &counts);
        (allowed, counts)
    }

    /// Whether a cluster of nodes may be summarised by `provider`. One
    /// local-only source among the nodes keeps the whole summary at home, and
    /// so does one node in a scope that is not allowed.
    pub fn decide_summary(&self, provider: &str, items: &[SummaryInput]) -> Decision {
        let sources: Vec<String> = items
            .iter()
            .flat_map(|item| item.sources.iter().cloned())
            .collect();
        for item in items {
            let gate_item = GateItem {
                scope: item.scope.as_deref(),
                sources: &sources,
            };
            if let deny @ Decision::Deny { .. } = self.decide(&gate_item, provider) {
                return deny;
            }
        }
        Decision::Allow
    }

    /// A cached hosted answer, served only while its item still passes the
    /// gate, so a scope taken off the allow list stops its cached answers too.
    pub fn cached(
        &self,
        provider: &str,
        item: &GateItem<'_>,
        cache: &CacheStore,
        key: &str,
    ) -> Option<String> {
        match self.decide(item, provider) {
            Decision::Allow => cache.get(key),
            Decision::Deny { .. } => None,
        }
    }
}

/// Logs how many items were held back and why, never what they said.
fn report(provider: &str, total: usize, counts: &GateCounts) {
    if counts.total() > 0 {
        tracing::info!(
            provider,
            items = total,
            denied = counts.total(),
            reasons = %counts,
            "privacy gate held items back from a hosted prompt"
        );
    }
}

#[cfg(test)]
mod tests;
