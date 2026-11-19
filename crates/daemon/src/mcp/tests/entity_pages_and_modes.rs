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
async fn recall_returns_an_entity_page_marked_as_one_and_still_refuses_infer_mode() {
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
    let page = default
        .split("\n\n")
        .find(|block| block.contains("zorbnax profile"))
        .unwrap_or_else(|| panic!("no page in {default}"));
    assert!(page.contains("kind: entity_page"), "{page}");
    assert!(page.contains("derivation: inferred"), "{page}");
    assert!(
        infer.starts_with("recall failed") && infer.contains("mode infer"),
        "{infer}"
    );
}

const AXES: usize = 8;

fn axis(axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; AXES];
    vector[axis] = 1.0;
    vector
}

/// The recall text split into its result blocks.
fn blocks(text: &str) -> Vec<&str> {
    text.split("\n\n").collect()
}

#[tokio::test]
async fn a_page_outranks_the_stale_memory_for_a_latest_state_question_and_names_its_sources() {
    // Arrange: an older and a newer memory of one entity, a page built from both
    let server = plain_server().await;
    let store = server.store().await.unwrap();
    let kim = store
        .upsert_by(NewNode::entity("person", "Kim"))
        .await
        .unwrap();
    let day = 86_400_000;
    let mut origins = Vec::new();
    for (index, (text, days)) in [
        ("Kim lives in Lisbon.", 1),
        ("Kim moved to Porto in May.", 2),
    ]
    .into_iter()
    .enumerate()
    {
        let id = store
            .insert(
                NewNode::now("fact", "Home", text)
                    .with_valid_from(liam_store::Millis(days * day))
                    .with_embedding(axis(index)),
            )
            .await
            .unwrap();
        store.relate(&kim, &id, relation::MENTIONS).await.unwrap();
        origins.extend(
            store
                .node_event_ids(&id, liam_store::EventRole::Origin)
                .await
                .unwrap(),
        );
    }
    server
        .resynthesize_entity_as(
            &store,
            kim,
            tokio::time::Instant::now() + Duration::from_secs(5),
            None,
        )
        .await
        .unwrap();
    drop(store);

    // Act
    let reply = server
        .recall_with(
            RecallArgs {
                query: "Kim lives Lisbon moved Porto May".to_string(),
                ..recall_args()
            },
            CancellationToken::new(),
        )
        .await;
    let result = reply.into_call_result();

    // Assert: the page comes before the stale memory and says what it is
    let text = result.content[0].as_text().unwrap().text.clone();
    let shown = blocks(&text);
    let page = shown
        .iter()
        .position(|b| b.contains("kind: entity_page"))
        .unwrap_or_else(|| panic!("no page in {text}"));
    let stale = shown
        .iter()
        .position(|b| b.lines().nth(1) == Some("Kim lives in Lisbon."))
        .unwrap_or_else(|| panic!("no stale memory in {text}"));
    assert!(page < stale, "{text}");
    assert!(
        shown[page].contains("1970-01-02: Kim lives in Lisbon."),
        "{text}"
    );
    assert!(
        shown[page].contains("1970-01-03: Kim moved to Porto in May."),
        "{text}"
    );
    let structured = result.structured_content.unwrap();
    let pages = structured["pages"].as_array().expect("pages");
    assert_eq!(pages.len(), 1, "{structured}");
    assert_eq!(pages[0]["kind"], "entity_page");
    assert_eq!(
        pages[0]["basis"],
        json!(origins
            .iter()
            .map(|e| e.event_id.clone())
            .collect::<Vec<_>>())
    );
}

#[tokio::test]
async fn a_recall_of_memories_alone_keeps_its_text_and_structure() {
    // Arrange
    let server = plain_server().await;
    seed(&server, "fact", "Tea", "recall shape probe likes green tea").await;

    // Act
    let result = server
        .recall_with(
            RecallArgs {
                query: "green tea".to_string(),
                ..recall_args()
            },
            CancellationToken::new(),
        )
        .await
        .into_call_result();

    // Assert
    let text = result.content[0].as_text().unwrap().text.clone();
    assert!(!text.contains("kind:"), "{text}");
    assert_eq!(result.structured_content, Some(json!({"degraded": []})));
}

#[tokio::test]
async fn an_entity_with_no_page_yet_is_not_shown_by_recall() {
    // Arrange: an entity row with no text, found by its label
    let server = plain_server().await;
    server
        .store()
        .await
        .unwrap()
        .upsert_by(NewNode::entity("person", "Zorbnax"))
        .await
        .unwrap();

    // Act
    let out = server
        .recall(Parameters(RecallArgs {
            query: "zorbnax".to_string(),
            ..recall_args()
        }))
        .await;

    // Assert
    assert_eq!(out, "no relevant memory");
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
