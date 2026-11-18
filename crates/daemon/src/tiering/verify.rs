// SPDX-License-Identifier: Apache-2.0
//! The check that stands between a planned demotion and the loss of a vector:
//! the event log must still hold what each node says.

use std::collections::HashMap;

use async_trait::async_trait;
use liam_daemon::consolidation::shared_store::SharedStore;
use liam_store::{Backend, Graph, NodeId, OriginClaim, OriginVerdict};

/// Judges many nodes against the log in one read of it.
#[async_trait]
pub trait TierVerifier: Send + Sync {
    async fn verify(
        &self,
        claims: &[OriginClaim],
    ) -> liam_store::Result<HashMap<NodeId, OriginVerdict>>;
}

#[async_trait]
impl<B: Backend> TierVerifier for Graph<B> {
    async fn verify(
        &self,
        claims: &[OriginClaim],
    ) -> liam_store::Result<HashMap<NodeId, OriginVerdict>> {
        self.verify_origins(claims).await
    }
}

/// The shared slot, so the check reads the log of the store a rollback last
/// reopened.
#[async_trait]
impl TierVerifier for SharedStore {
    async fn verify(
        &self,
        claims: &[OriginClaim],
    ) -> liam_store::Result<HashMap<NodeId, OriginVerdict>> {
        let store = self
            .get()
            .await
            .map_err(|error| liam_store::Error::Backend(error.to_string()))?;
        store.verify_origins(claims).await
    }
}
