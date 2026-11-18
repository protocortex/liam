// SPDX-License-Identifier: Apache-2.0
//! Moves that cannot be made: what the report counts and what stays put.

use super::*;

struct UnreachableLog;

#[async_trait::async_trait]
impl TierVerifier for UnreachableLog {
    async fn verify(
        &self,
        _claims: &[OriginClaim],
    ) -> liam_store::Result<HashMap<NodeId, OriginVerdict>> {
        Err(liam_store::Error::Backend("log unreadable".into()))
    }
}

#[tokio::test]
async fn demotions_are_all_refused_when_the_log_cannot_vouch_for_any() {
    // Arrange
    let fixture = fixture().await;
    let (_, first) = add(&fixture.store, "a", 0b01).await;
    let (_, second) = add(&fixture.store, "b", 0b10).await;
    let embedder = ScriptedEmbedder::new(DIMS, &[]);
    let unreachable = UnreachableLog;
    let mut moves_deps = deps(&fixture, &embedder, far());
    moves_deps.verifier = &unreachable;

    // Act
    let report = apply_moves(
        &plan(
            vec![],
            vec![planned(&first, Tier::Cold), planned(&second, Tier::Cold)],
        ),
        &moves_deps,
    )
    .await;

    // Assert
    assert_eq!(report.planned, 2);
    assert_eq!(report.failed, 2);
    assert_eq!(report.demoted, 0);
    assert!(holds_vector(&fixture.store, &first).await);
    assert!(holds_vector(&fixture.store, &second).await);
}

#[tokio::test]
async fn promoting_a_lineage_that_is_already_hot_is_stale_and_embeds_nothing() {
    // Arrange
    let fixture = fixture().await;
    let (_, hot) = add(&fixture.store, "b", 0b10).await;
    let embedder = ScriptedEmbedder::new(DIMS, &[]);

    // Act
    let report = apply_moves(
        &plan(vec![planned(&hot, Tier::Hot)], vec![]),
        &deps(&fixture, &embedder, far()),
    )
    .await;

    // Assert
    assert_eq!(report.skipped_stale, 1);
    assert_eq!(report.promoted, 0);
    assert_eq!(report.failed, 0);
    assert_eq!(embedder.calls(), 0);
}
