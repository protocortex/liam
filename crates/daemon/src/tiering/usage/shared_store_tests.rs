// SPDX-License-Identifier: Apache-2.0
//! Counting through the slot a rollback can empty.

use std::sync::Arc;

use liam_store::{FixedClock, NewNode};

use super::super::test_support::logged_store;
use super::*;

const DIMS: usize = 8;
const START: Millis = Millis(5 * 86_400_000);
const UNAVAILABLE: &str = "backend: the store is unavailable: a rollback could not reopen it";

#[tokio::test]
async fn the_slot_hands_each_call_to_the_store_it_holds() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let store = logged_store(dir.path(), Arc::new(FixedClock::new(START)), DIMS).await;
    let node = store
        .insert(NewNode::now("fact", "label", "content").with_subject("alice"))
        .await
        .unwrap();
    let slot = SharedStore::new(store);

    // Act
    let lineage = UsageSink::lineage_of(&slot, &node).await.unwrap();
    let before = UsageSink::usage_started(&slot).await.unwrap();
    UsageSink::put_usage_logged(&slot, START.day_number(), &[])
        .await
        .unwrap();
    let after = UsageSink::usage_started(&slot).await.unwrap();

    // Assert
    assert!(lineage.is_some());
    assert!(!before);
    assert!(after);
}

#[tokio::test]
async fn with_the_slot_empty_every_call_says_the_store_is_unavailable() {
    // Arrange
    let dir = tempfile::tempdir().unwrap();
    let store = logged_store(dir.path(), Arc::new(FixedClock::new(START)), DIMS).await;
    let slot = SharedStore::new(store);
    drop(slot.exclusive().await.take());
    let node = NodeId::from_raw("any-node");

    // Act
    let lineage = UsageSink::lineage_of(&slot, &node).await;
    let started = UsageSink::usage_started(&slot).await;
    let written = UsageSink::put_usage_logged(&slot, START.day_number(), &[]).await;

    // Assert
    for result in [lineage.map(|_| ()), started.map(|_| ()), written] {
        assert_eq!(result.unwrap_err().to_string(), UNAVAILABLE);
    }
}
