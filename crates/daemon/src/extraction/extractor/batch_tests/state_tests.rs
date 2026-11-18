// SPDX-License-Identifier: Apache-2.0
//! What the table records for each way a provider can end a batch.

use super::*;

/// Submits one window, then lets the provider answer as `state` says.
async fn submitted_then(state: RemoteState) -> (Env, Arc<Remote>) {
    let env = Env::new().await;
    let remote = Remote::new();
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();
    remote.set_state(state);
    (env, remote)
}

fn states(env: &Env) -> Vec<(BatchState, Option<EntryFailure>)> {
    env.table()
        .into_iter()
        .map(|entry| (entry.state, entry.failure))
        .collect()
}

#[tokio::test]
async fn a_batch_the_provider_failed_is_refused_for_every_request_and_nothing_is_cached() {
    // Arrange
    let (env, remote) = submitted_then(RemoteState::Failed).await;
    let key = remote.requests_in(0)[0].clone();

    // Act
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert!(states(&env)
        .iter()
        .all(|state| *state == (BatchState::Failed, Some(EntryFailure::Refused))));
    assert!(!env.cache.contains(&key));
    assert_eq!(remote.collects.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_batch_the_provider_cancelled_is_sent_again_as_a_new_first_send() {
    // Arrange
    let (env, remote) = submitted_then(RemoteState::Cancelled).await;

    // Act
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert_eq!(remote.collects.load(Ordering::SeqCst), 0);
    assert_eq!(remote.submits(), 2);
    assert_eq!(states(&env), vec![(BatchState::Submitted, None)]);
    assert_eq!(env.table()[0].attempts, 1);
}

#[tokio::test]
async fn a_request_the_provider_cancelled_inside_an_ended_batch_is_sent_again() {
    // Arrange
    let (env, remote) = submitted_then(RemoteState::Ended).await;
    remote.answer_with(|_, _| Some(RemoteResult::Cancelled));

    // Act
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert_eq!(remote.submits(), 2);
    assert_eq!(states(&env), vec![(BatchState::Submitted, None)]);
    assert_eq!(env.table()[0].attempts, 1);
}

#[tokio::test]
async fn a_request_that_expired_inside_an_ended_batch_is_sent_once_more() {
    // Arrange
    let (env, remote) = submitted_then(RemoteState::Ended).await;
    remote.answer_with(|_, _| Some(RemoteResult::Expired));

    // Act
    env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert_eq!(remote.submits(), 2, "the expired request is sent again");
    assert_eq!(
        states(&env),
        vec![(BatchState::Submitted, None)],
        "the second send is the request's second attempt"
    );
    assert_eq!(env.table()[0].attempts, 2);
}

#[tokio::test]
async fn an_ended_batch_that_returns_nothing_for_a_request_drops_it_and_sends_nothing_new() {
    // Arrange
    let (env, remote) = submitted_then(RemoteState::Ended).await;
    let lost = remote.requests_in(0)[0].clone();
    remote.answer_with(|_, _| None);

    // Act
    let report = env.run(&env.flow(&remote), &inputs(2)).await.unwrap();

    // Assert
    assert!(!env.cache.contains(&lost));
    assert!(env.table().is_empty());
    assert_eq!(remote.submits(), 1);
    assert_eq!(report.completed_offset, None);
}
