// SPDX-License-Identifier: Apache-2.0
//! The store handle a run and its clients share. A restore must own the store
//! to close it before the snapshot replaces its file, which a plain
//! `Arc<DefaultGraph>` clone held by a client forbids. Here every user takes a
//! read guard for the span of one request or one run step, and a rollback takes
//! the write guard: it waits for the requests in flight, holds new ones back for
//! the length of the restore, then puts the reopened store in the slot.
//!
//! A run holds no guard between its steps, so clients are never stopped while
//! it works. Only the rollback's restore and replay pause them.

use liam_store::DefaultGraph;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub struct SharedStore {
    slot: RwLock<Option<DefaultGraph>>,
}

impl SharedStore {
    pub fn new(store: DefaultGraph) -> Self {
        Self {
            slot: RwLock::new(Some(store)),
        }
    }

    /// The store, for as long as the guard lives. Hold it for one request,
    /// never across requests: a rollback waits for it.
    ///
    /// # Errors
    ///
    /// Fails when a rollback could not reopen the store, so the slot is empty.
    pub async fn get(&self) -> anyhow::Result<RwLockReadGuard<'_, DefaultGraph>> {
        RwLockReadGuard::try_map(self.slot.read().await, Option::as_ref).map_err(|_| {
            anyhow::anyhow!("the store is unavailable: a rollback could not reopen it")
        })
    }

    /// Exclusive access to the slot, for a restore that replaces the store.
    pub(super) async fn exclusive(&self) -> RwLockWriteGuard<'_, Option<DefaultGraph>> {
        self.slot.write().await
    }
}
