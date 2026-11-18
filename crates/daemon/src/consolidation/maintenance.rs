// SPDX-License-Identifier: Apache-2.0
//! The maintenance steps a consolidation run wraps: replay the log, sweep,
//! repair provenance, refresh clusters. `liamd` adds resynthesis after them.

use async_trait::async_trait;
use liam_store::{DefaultGraph, RetentionPolicy};
use tokio_util::sync::CancellationToken;

use super::run::Maintenance;

/// Runs the steps in order, each stamped with `write_source` when given, and
/// stops after the step that left `stopping` cancelled, as a poisoned log does:
/// the later steps would only be refused. Returns whether every step ran.
pub async fn standard_steps(
    store: &DefaultGraph,
    policy: &RetentionPolicy,
    write_source: Option<&str>,
    stopping: &CancellationToken,
) -> bool {
    // A leftover event applied after the sweep would outlive it.
    catch_up(store).await;
    if stopping.is_cancelled() {
        return false;
    }
    sweep(store, policy, write_source).await;
    if stopping.is_cancelled() {
        return false;
    }
    repair_mentions(store, write_source).await;
    if stopping.is_cancelled() {
        return false;
    }
    refresh_clusters(store).await;
    !stopping.is_cancelled()
}

/// The steps with no model behind them, for a run made without a daemon.
pub struct StoreMaintenance {
    pub policy: RetentionPolicy,
}

#[async_trait]
impl Maintenance for StoreMaintenance {
    async fn run(&self, store: &DefaultGraph, source: &str) -> anyhow::Result<()> {
        // Nothing here can stop mid-way, so the token is never cancelled.
        standard_steps(store, &self.policy, Some(source), &CancellationToken::new()).await;
        Ok(())
    }
}

/// Repairs `mentions` edges left dangling by a supersession, so citing
/// entities keep pointing at the current fact. A failure here must not block
/// the rest of the tick, matching `sweep`'s and `refresh_clusters`'s own
/// error-doesn't-abort-the-tick pattern.
pub async fn repair_mentions(store: &DefaultGraph, write_source: Option<&str>) {
    let repaired = match write_source {
        Some(source) => store.repair_superseded_mentions_from(source).await,
        None => store.repair_superseded_mentions().await,
    };
    match repaired {
        Ok(repaired) => tracing::info!(repaired, "provenance repair completed"),
        Err(e) => tracing::warn!(error = %e, "provenance repair failed"),
    }
}

pub async fn catch_up(store: &DefaultGraph) {
    match store.catch_up().await {
        Ok(report) if report.applied > 0 || report.quarantined > 0 => {
            tracing::info!(?report, "event log caught up")
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(error = %e, "event log catch up failed"),
    }
}

pub async fn sweep(
    store: &DefaultGraph,
    policy: &liam_store::RetentionPolicy,
    write_source: Option<&str>,
) {
    let swept = match write_source {
        Some(source) => store.gc_from(policy, source).await,
        None => store.gc(policy).await,
    };
    match swept {
        Ok(report) => tracing::info!(?report, "gc completed"),
        Err(e) => tracing::warn!(error = %e, "gc failed"),
    }
}

/// Not serving stale: the tick serves no one. A skipped or failed refresh
/// just means the next read re-runs the check and recomputes then.
pub async fn refresh_clusters(store: &DefaultGraph) -> bool {
    match store.refresh_communities().await {
        Ok(true) => {
            tracing::info!("clusters refreshed");
            true
        }
        Ok(false) => {
            tracing::debug!("clusters already current");
            false
        }
        Err(e) => {
            tracing::warn!(error = %e, "cluster refresh failed");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use liam_store::{FixedClock, GraphConfig, Millis, NewNode};

    use super::*;

    const SOURCE: &str = "consolidation:test";

    async fn store_with_an_old_fact() -> (DefaultGraph, liam_store::NodeId) {
        let clock = Arc::new(FixedClock::new(Millis(10)));
        let store = DefaultGraph::open_with_clock(":memory:", GraphConfig::new(8), clock.clone())
            .await
            .unwrap();
        let old = store.insert(NewNode::now("fact", "a", "x")).await.unwrap();
        clock.set(Millis(10_000));
        (store, old)
    }

    fn sweep_facts() -> RetentionPolicy {
        RetentionPolicy::keep("fact", Millis(1)).without_reclaim()
    }

    #[tokio::test]
    async fn the_store_maintenance_sweeps_what_the_policy_ages_out() {
        // Arrange
        let (store, old) = store_with_an_old_fact().await;
        let maintenance = StoreMaintenance {
            policy: sweep_facts(),
        };

        // Act
        maintenance.run(&store, SOURCE).await.unwrap();

        // Assert
        assert!(store.get(&old, Millis(10_000)).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn the_steps_stop_once_the_daemon_is_stopping() {
        // Arrange
        let (store, old) = store_with_an_old_fact().await;
        let stopping = CancellationToken::new();
        stopping.cancel();

        // Act
        let finished = standard_steps(&store, &sweep_facts(), Some(SOURCE), &stopping).await;

        // Assert
        assert!(!finished);
        assert!(
            store.get(&old, Millis(10_000)).await.unwrap().is_some(),
            "the sweep after the replay never ran"
        );
    }

    #[tokio::test]
    async fn the_steps_report_completion_when_nothing_stops_them() {
        let (store, _old) = store_with_an_old_fact().await;

        let finished = standard_steps(
            &store,
            &sweep_facts(),
            Some(SOURCE),
            &CancellationToken::new(),
        )
        .await;

        assert!(finished);
    }
}
