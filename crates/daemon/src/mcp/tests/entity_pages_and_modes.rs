// SPDX-License-Identifier: Apache-2.0
//! Entity pages in the reads, and the refusal of infer mode.

use super::*;

const PAGE_LABEL: &str = "Zorbnax";

/// A recorded fact and an entity page that both match `zorbnax`, the page
/// mentioning the fact. Returns the page's id.
async fn fact_and_page(server: &MemoryServer) -> liam_store::NodeId {
    let fact = seed(server, "fact", "Preference", "zorbnax likes green tea").await;
    let store = server.store().await.unwrap();
    let mut page = NewNode::entity("person", PAGE_LABEL)
        .with_attributes(crate::entity_page::stamped(json!({})));
    page.content = "zorbnax profile: likes green tea".to_string();
    let page = store.upsert_by(page).await.unwrap();
    let fact = store.resolve_handle(&fact).await.unwrap();
    store
        .relate(&page, &fact, relation::MENTIONS)
        .await
        .unwrap();
    page
}

#[tokio::test]
async fn recall_leaves_out_an_entity_page_and_refuses_infer_mode() {
    // Arrange
    let server = plain_server().await;
    fact_and_page(&server).await;
    let args = |mode| RecallArgs {
        query: "zorbnax".to_string(),
        mode,
        ..recall_args()
    };

    // Act
    let default = server.recall(Parameters(args(None))).await;
    let infer = server.recall(Parameters(args(Some(Mode::Infer)))).await;

    // Assert
    assert!(default.contains("green tea"), "{default}");
    assert!(!default.contains("profile"), "{default}");
    assert!(
        infer.starts_with("recall failed") && infer.contains("mode infer"),
        "{infer}"
    );
}

#[tokio::test]
async fn clusters_leave_out_a_page_that_mentions_recorded_memory() {
    // Arrange
    let server = plain_server().await;
    let page = fact_and_page(&server).await;
    let other = seed(&server, "fact", "Rollout owner", "y").await;
    let store = server.store().await.unwrap();
    let other = store.resolve_handle(&other).await.unwrap();
    let fact = store
        .mentions(&page, liam_store::Millis::now(), 10)
        .await
        .unwrap()
        .remove(0)
        .id;
    store
        .relate(&fact, &other, relation::MENTIONS)
        .await
        .unwrap();
    drop(store);

    // Act
    let out = server
        .clusters(Parameters(ClustersArgs {
            k: None,
            members: None,
            mode: None,
        }))
        .await;
    let infer = server
        .clusters(Parameters(ClustersArgs {
            k: None,
            members: None,
            mode: Some(Mode::Infer),
        }))
        .await;

    // Assert
    assert!(out.contains("Preference"), "{out}");
    assert!(out.contains("Rollout owner"), "{out}");
    assert!(!out.contains(PAGE_LABEL), "{out}");
    assert!(infer.contains("mode infer"), "{infer}");
}

#[tokio::test]
async fn timeline_shows_the_page_labelled_inferred_and_refuses_infer_mode() {
    // Arrange
    let server = plain_server().await;
    let page = fact_and_page(&server).await;
    let args = |mode| TimelineArgs {
        entity: page.as_str().to_string(),
        limit: None,
        mode,
    };

    // Act
    let out = server.timeline(Parameters(args(None))).await;
    let infer = server.timeline(Parameters(args(Some(Mode::Infer)))).await;

    // Assert
    let header: Vec<&str> = out.lines().take(3).collect();
    assert!(
        header[0].starts_with(&format!("[person {}", page.handle())),
        "{out}"
    );
    assert!(
        header[2].starts_with("provenance: derivation: inferred"),
        "{out}"
    );
    assert!(out.contains("zorbnax likes green tea"), "{out}");
    assert!(infer.contains("mode infer"), "{infer}");
}
