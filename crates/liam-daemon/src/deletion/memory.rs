// SPDX-License-Identifier: Apache-2.0
//! State a running process keeps about nodes: the usage counts waiting for a
//! flush and the handles a connection was issued. A delete runs with the
//! daemon stopped, so these are empty then; a daemon that resumes a delete
//! before it serves clears them for the nodes that were just removed.

use std::collections::HashSet;
use std::sync::Arc;

use crate::mcp::feedback::IssuedHandles;
use crate::tiering::usage::UsageRecorder;
use async_trait::async_trait;
use liam_store::{LineageHits, LineageStore, LineageTarget, PurgeReport, Residual, Result};

const MEMORY_STORE: &str = "memory";
const USAGE_STATE: &str = "usage_counts";
const HANDLE_STATE: &str = "issued_handles";

/// One piece of in-memory state that can name and drop the nodes it holds.
pub trait NodeState: Send + Sync {
    /// What the state is called in a residual.
    fn kind(&self) -> &str;
    /// The ids among `nodes` that the state holds.
    fn holds(&self, nodes: &HashSet<String>) -> Vec<String>;
    /// Drops `nodes` and says how many it held.
    fn forget(&self, nodes: &HashSet<String>) -> usize;
}

pub struct InMemoryLineage {
    states: Vec<Arc<dyn NodeState>>,
}

impl InMemoryLineage {
    pub fn new(states: Vec<Arc<dyn NodeState>>) -> Self {
        Self { states }
    }
}

fn nodes_of(target: &LineageTarget) -> HashSet<String> {
    target.node_ids.iter().cloned().collect()
}

#[async_trait]
impl LineageStore for InMemoryLineage {
    fn name(&self) -> &str {
        MEMORY_STORE
    }

    async fn find(&self, target: &LineageTarget) -> Result<LineageHits> {
        let nodes = nodes_of(target);
        let mut found = LineageHits::default();
        for state in &self.states {
            let held = state.holds(&nodes).len();
            if held > 0 {
                found.rows.insert(state.kind().to_owned(), held);
            }
        }
        Ok(found)
    }

    async fn purge(&self, target: &LineageTarget) -> Result<PurgeReport> {
        let nodes = nodes_of(target);
        let removed = self.states.iter().map(|state| state.forget(&nodes)).sum();
        Ok(PurgeReport { removed })
    }

    async fn verify(&self, target: &LineageTarget) -> Result<Vec<Residual>> {
        let nodes = nodes_of(target);
        Ok(self
            .states
            .iter()
            .flat_map(|state| {
                state.holds(&nodes).into_iter().map(|id| Residual {
                    store: MEMORY_STORE.to_owned(),
                    kind: state.kind().to_owned(),
                    id,
                })
            })
            .collect())
    }
}

impl NodeState for UsageRecorder {
    fn kind(&self) -> &str {
        USAGE_STATE
    }

    fn holds(&self, nodes: &HashSet<String>) -> Vec<String> {
        UsageRecorder::holds(self, nodes)
    }

    fn forget(&self, nodes: &HashSet<String>) -> usize {
        UsageRecorder::forget(self, nodes)
    }
}

impl NodeState for IssuedHandles {
    fn kind(&self) -> &str {
        HANDLE_STATE
    }

    fn holds(&self, nodes: &HashSet<String>) -> Vec<String> {
        IssuedHandles::holds(self, nodes)
    }

    fn forget(&self, nodes: &HashSet<String>) -> usize {
        IssuedHandles::forget(self, nodes)
    }
}

#[cfg(test)]
mod tests {
    use liam_store::{Millis, NodeId};

    use super::*;

    const CAPACITY: usize = 16;

    fn target_of(ids: &[&NodeId]) -> LineageTarget {
        LineageTarget {
            node_ids: ids.iter().map(|id| id.as_str().to_owned()).collect(),
            ..LineageTarget::default()
        }
    }

    #[tokio::test]
    async fn counts_and_handles_for_a_deleted_node_are_dropped_and_the_others_kept() {
        // Arrange
        let (doomed, kept) = (NodeId::new(), NodeId::new());
        let recorder = Arc::new(UsageRecorder::new(CAPACITY));
        recorder.record([&doomed, &kept], Millis(1));
        let handles = Arc::new(IssuedHandles::new(CAPACITY));
        handles.issue([&doomed, &kept]);
        let states: Vec<Arc<dyn NodeState>> = vec![recorder.clone(), handles.clone()];
        let state = InMemoryLineage::new(states);
        let target = target_of(&[&doomed]);

        // Act
        let found = state.find(&target).await.unwrap();
        let before = state.verify(&target).await.unwrap();
        let report = state.purge(&target).await.unwrap();
        let again = state.purge(&target).await.unwrap();

        // Assert
        assert_eq!(found.total(), 2);
        assert_eq!(before.len(), 2);
        assert_eq!((report.removed, again.removed), (2, 0));
        assert!(state.verify(&target).await.unwrap().is_empty());
        assert_eq!(recorder.hits().len(), 1);
        assert_eq!(handles.holds(&[kept.as_str().to_owned()].into()).len(), 1);
    }
}
