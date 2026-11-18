// SPDX-License-Identifier: Apache-2.0
//! Requests that meet a store swap.

use super::*;

/// Yields until the shared slot is held or awaited by a writer.
async fn until_contended(shared: &SharedStore) {
    for _ in 0..YIELDS {
        if shared.is_contended() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("no writer ever queued");
}

/// Yields enough times for a spawned task to run to its first wait.
async fn settle() {
    for _ in 0..YIELDS {
        tokio::task::yield_now().await;
    }
}

const YIELDS: usize = 200;
const BOUND: Duration = Duration::from_secs(10);

#[tokio::test]
async fn resynthesis_on_the_callers_store_finishes_while_a_restore_is_queued() {
    // Arrange: the caller holds a read guard and a writer queues behind it
    let (server, _embedder) =
        server_with_recording_embedder_and_llm(Arc::new(CountingLlm::new())).await;
    let shared = server.shared_store();
    let guard = server.store().await.unwrap();
    let entity_id = guard
        .upsert_by(NewNode::entity("person", "Kim"))
        .await
        .unwrap();
    let queued = {
        let shared = shared.clone();
        tokio::spawn(async move { drop(shared.exclusive().await) })
    };
    until_contended(&shared).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    // Act: a second read behind the queued writer would never return
    let resynthesized = tokio::time::timeout(
        BOUND,
        server.resynthesize_entity_as(&guard, entity_id, deadline, 64, None),
    )
    .await;
    drop(guard);

    // Assert
    assert!(matches!(resynthesized, Ok(Ok(()))), "{resynthesized:?}");
    tokio::time::timeout(BOUND, queued).await.unwrap().unwrap();
}

#[tokio::test]
async fn a_request_during_a_swap_waits_and_is_then_served() {
    // Arrange: a restore holds the slot
    let (server, _embedder) = server_with_recording_embedder().await;
    let shared = server.shared_store();
    let swap = shared.exclusive().await;
    let request = {
        let server = server.clone();
        tokio::spawn(async move { server.recall(Parameters(recall_args())).await })
    };

    // Act
    settle().await;
    let waiting = !request.is_finished();
    drop(swap);
    let answer = tokio::time::timeout(BOUND, request).await.unwrap().unwrap();

    // Assert
    assert!(waiting, "the request must wait for the swap");
    assert!(!answer.contains("failed"), "{answer}");
}

#[tokio::test]
async fn a_request_after_a_swap_that_lost_the_store_gets_a_clean_error() {
    // Arrange: the restore could not reopen the store
    let (server, _embedder) = server_with_recording_embedder().await;
    let shared = server.shared_store();
    let mut swap = shared.exclusive().await;
    *swap = None;
    let request = {
        let server = server.clone();
        tokio::spawn(async move { server.recall(Parameters(recall_args())).await })
    };
    settle().await;

    // Act
    drop(swap);
    let answer = tokio::time::timeout(BOUND, request).await.unwrap().unwrap();

    // Assert
    assert!(answer.starts_with("recall failed"), "{answer}");
    assert!(answer.contains("store is unavailable"), "{answer}");
}
