// SPDX-License-Identifier: Apache-2.0
//! What stops the daemon from the inside. A poisoned event log refuses every
//! write until it is reopened and replayed, so the daemon stops with a failure
//! and whatever starts it next replays the log at startup.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use liam_store::PoisonHook;
use tokio_util::sync::CancellationToken;

pub(crate) const POISONED_MESSAGE: &str = "the event log is poisoned; restart liamd to replay it";
pub(crate) const STORE_LOST_MESSAGE: &str =
    "a consolidation rollback could not reopen the store; restart liamd to open it again";

/// The token every long running task watches, plus whether the log's poison
/// is what cancelled it. Clones share both.
#[derive(Clone, Default)]
pub(crate) struct Shutdown {
    token: CancellationToken,
    poisoned: Arc<AtomicBool>,
    store_lost: Arc<AtomicBool>,
}

impl Shutdown {
    pub(crate) fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// For the event log to call each time a write meets its poison.
    pub(crate) fn poison_hook(&self) -> PoisonHook {
        let shutdown = self.clone();
        Arc::new(move || shutdown.log_poisoned())
    }

    fn log_poisoned(&self) {
        // The log reports every refused write; only the first one logs.
        if !self.poisoned.swap(true, Ordering::SeqCst) {
            tracing::error!("{POISONED_MESSAGE}");
        }
        self.token.cancel();
    }

    /// For a rollback that left the shared slot without a store: no request
    /// can be served, so the daemon stops with a failure and its supervisor
    /// starts it again.
    pub(crate) fn store_lost(&self) {
        if !self.store_lost.swap(true, Ordering::SeqCst) {
            tracing::error!("{STORE_LOST_MESSAGE}");
        }
        self.token.cancel();
    }

    /// The failure the process exits with when the log's poison or a lost
    /// store stopped it.
    pub(crate) fn failure(&self) -> Option<anyhow::Error> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Some(anyhow::anyhow!(POISONED_MESSAGE));
        }
        self.store_lost
            .load(Ordering::SeqCst)
            .then(|| anyhow::anyhow!(STORE_LOST_MESSAGE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_healthy_shutdown_has_no_failure_and_is_not_cancelled() {
        // Arrange
        let shutdown = Shutdown::default();

        // Act
        let failure = shutdown.failure();

        // Assert
        assert!(failure.is_none());
        assert!(!shutdown.token().is_cancelled());
    }

    #[test]
    fn a_lost_store_cancels_the_token_and_makes_the_exit_a_failure() {
        // Arrange
        let shutdown = Shutdown::default();

        // Act
        shutdown.store_lost();

        // Assert
        assert!(shutdown.token().is_cancelled());
        let failure = shutdown.failure().expect("a lost store must fail the exit");
        assert_eq!(failure.to_string(), STORE_LOST_MESSAGE);
    }

    #[test]
    fn the_poison_hook_cancels_the_token_and_makes_the_exit_a_failure() {
        // Arrange
        let shutdown = Shutdown::default();
        let hook = shutdown.poison_hook();

        // Act
        hook();

        // Assert
        assert!(shutdown.token().is_cancelled());
        let failure = shutdown.failure().expect("poison must fail the exit");
        assert_eq!(failure.to_string(), POISONED_MESSAGE);
    }

    #[test]
    fn only_the_first_report_is_logged_and_at_error_level() {
        // Arrange
        crate::test_logs::capture();
        let hook = Shutdown::default().poison_hook();

        // Act
        hook();
        hook();

        // Assert
        let text = crate::test_logs::captured();
        assert_eq!(text.matches(POISONED_MESSAGE).count(), 1, "{text}");
        assert!(text.contains("ERROR"), "{text}");
    }

    #[test]
    fn a_plain_cancel_is_not_a_failure() {
        // Arrange
        let shutdown = Shutdown::default();

        // Act
        shutdown.token().cancel();

        // Assert
        assert!(shutdown.failure().is_none());
    }
}
