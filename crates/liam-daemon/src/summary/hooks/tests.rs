// SPDX-License-Identifier: Apache-2.0
//! The summarize step over a store in memory and a scripted model: which
//! summaries one pass writes, keeps and closes, and what a failure leaves.

use std::collections::BTreeSet;

use liam_store::{EmbedError, GraphConfig, NewNode, NodeId};

use super::*;
use crate::summary::test_support::{MemoryCache, ScriptedLlm};

const DIMS: usize = 8;
const GROUP_NAMES: [&str; 3] = ["alpha", "beta", "gamma"];
const GROUP_SIZE: usize = 4;
const LEVEL_ONE_REPLY: &str = "These ledger notes cover quarterly forecasts for the team.";
const LEVEL_TWO_REPLY: &str = "Together the ledger notes cover quarterly forecasts for the team.";
const SOURCE: &str = "consolidation:test";

fn axis(index: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMS];
    vector[index] = 1.0;
    vector
}

struct FixedEmbedder;

#[async_trait]
impl ContentEmbedder for FixedEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(axis(0))
    }
}

struct Rig {
    store: DefaultGraph,
    llm: Arc<ScriptedLlm>,
    cache: Arc<MemoryCache>,
}

impl Rig {
    async fn new(llm: ScriptedLlm) -> Self {
        Self {
            store: DefaultGraph::open(":memory:", GraphConfig::new(DIMS))
                .await
                .unwrap(),
            llm: Arc::new(llm),
            cache: Arc::new(MemoryCache::default()),
        }
    }

    /// The model answers three level one summaries and then the one above.
    async fn answering_a_tree() -> Self {
        Self::new(ScriptedLlm::replying(&[
            LEVEL_ONE_REPLY,
            LEVEL_ONE_REPLY,
            LEVEL_ONE_REPLY,
            LEVEL_TWO_REPLY,
        ]))
        .await
    }

    async fn add(&self, group: usize, n: usize, scope: Option<&str>) -> NodeId {
        let content = format!(
            "ledger notes number {n} cover quarterly forecasts for the {} team",
            GROUP_NAMES[group]
        );
        let mut node = NewNode::now("fact", format!("Note {n}"), content)
            .with_embedding(axis(group))
            .with_valid_from(Millis(1000 + (group * 100 + n) as i64));
        if let Some(scope) = scope {
            node = node.with_scope(scope);
        }
        self.store.insert(node).await.unwrap()
    }

    async fn add_groups(&self, groups: usize) -> Vec<Vec<NodeId>> {
        let mut all = Vec::new();
        for group in 0..groups {
            let mut ids = Vec::new();
            for n in 0..GROUP_SIZE {
                ids.push(self.add(group, n, None).await);
            }
            all.push(ids);
        }
        all
    }

    fn hooks(&self, config: SummaryConfig) -> SummaryHooks {
        SummaryHooks {
            config,
            llm: self.llm.clone(),
            permits: Arc::new(Semaphore::new(1)),
            cache: self.cache.clone(),
            embedder: Arc::new(FixedEmbedder),
            identity: ModelIdentity {
                model_id: "model".to_string(),
                model_sha256: String::new(),
            },
            context_tokens: 8192,
        }
    }

    async fn run(&self, config: SummaryConfig) -> SummaryReport {
        self.hooks(config)
            .summarize(&SummaryCtx {
                store: &self.store,
                source: SOURCE.to_string(),
            })
            .await
            .unwrap()
    }

    async fn live(&self, level: u8) -> Vec<Summary> {
        self.store.live_summaries(level, "").await.unwrap()
    }

    async fn live_ids(&self) -> BTreeSet<String> {
        let mut ids = BTreeSet::new();
        for level in 1..=MAX_LEVELS {
            ids.extend(self.live(level).await.into_iter().map(|s| s.id));
        }
        ids
    }
}

const MAX_LEVELS: u8 = 3;

fn enabled() -> SummaryConfig {
    SummaryConfig {
        enabled: true,
        ..SummaryConfig::default()
    }
}

#[tokio::test]
async fn twelve_items_in_three_groups_get_three_summaries_and_one_above_them() {
    // Arrange
    let rig = Rig::answering_a_tree().await;
    let groups = rig.add_groups(3).await;

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!(report.written, 4);
    assert!(!report.partial());
    let level_one = rig.live(1).await;
    assert_eq!(level_one.len(), 3);
    let mut covered: Vec<String> = Vec::new();
    for summary in &level_one {
        let children = rig.store.summary_children(&summary.id).await.unwrap();
        assert_eq!(children.len(), GROUP_SIZE);
        assert!(children.iter().all(|c| c.kind == ChildKind::Node));
        assert!(rig
            .store
            .check_closure(&summary.id)
            .await
            .unwrap()
            .is_closed());
        covered.extend(children.into_iter().map(|c| c.id));
    }
    covered.sort();
    let mut expected: Vec<String> = groups
        .iter()
        .flatten()
        .map(|id| id.as_str().to_string())
        .collect();
    expected.sort();
    assert_eq!(covered, expected);
    let level_two = rig.live(2).await;
    assert_eq!(level_two.len(), 1);
    assert_eq!(level_two[0].text.as_str(), LEVEL_TWO_REPLY);
    let above = rig.store.summary_children(&level_two[0].id).await.unwrap();
    assert_eq!(above.len(), 3);
    assert!(above.iter().all(|c| c.kind == ChildKind::Summary));
    assert!(rig
        .store
        .check_closure(&level_two[0].id)
        .await
        .unwrap()
        .is_closed());
    assert_eq!(rig.llm.calls(), 4);
}

#[tokio::test]
async fn each_scope_gets_summaries_of_its_own() {
    // Arrange
    let rig = Rig::new(ScriptedLlm::replying(&[LEVEL_ONE_REPLY, LEVEL_ONE_REPLY])).await;
    for n in 0..GROUP_SIZE {
        rig.add(0, n, Some("work")).await;
        rig.add(1, n, Some("home")).await;
    }

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!(report.written, 2);
    for scope in ["work", "home"] {
        let live = rig.store.live_summaries(1, scope).await.unwrap();
        assert_eq!(live.len(), 1, "{scope}");
        assert_eq!(live[0].scope, scope);
    }
    assert!(rig.live(1).await.is_empty());
}

#[tokio::test]
async fn a_disabled_step_writes_nothing_and_never_calls_the_model() {
    // Arrange
    let rig = Rig::answering_a_tree().await;
    rig.add_groups(3).await;

    // Act
    let report = rig.run(SummaryConfig::default()).await;

    // Assert
    assert_eq!(report, SummaryReport::default());
    assert_eq!(rig.llm.calls(), 0);
    assert!(rig.live_ids().await.is_empty());
}

#[tokio::test]
async fn a_second_pass_over_the_same_items_asks_the_model_nothing_and_writes_nothing() {
    // Arrange
    let rig = Rig::answering_a_tree().await;
    rig.add_groups(3).await;
    rig.run(enabled()).await;
    let before = rig.live_ids().await;

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!(report.written, 0);
    assert!(!report.partial());
    assert_eq!(rig.llm.calls(), 4);
    assert_eq!(rig.live_ids().await, before);
}

#[tokio::test]
async fn a_failed_summary_is_reported_partial_and_the_others_are_still_written() {
    // Arrange
    let rig = Rig::new(ScriptedLlm::scripted(vec![
        Ok(LEVEL_ONE_REPLY.to_string()),
        Err("the model fell over".to_string()),
        Ok(LEVEL_ONE_REPLY.to_string()),
    ]))
    .await;
    rig.add_groups(3).await;

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!(report.written, 2);
    assert_eq!(report.unwritten, 1);
    assert!(report.partial());
    assert_eq!(rig.live(1).await.len(), 2);
    assert!(
        rig.live(2).await.is_empty(),
        "a level above a partial one is not built"
    );
    assert_eq!(rig.llm.calls(), 3);
}

#[tokio::test]
async fn a_reply_that_is_not_grounded_in_its_items_is_dropped() {
    // Arrange
    let rig = Rig::new(ScriptedLlm::replying(&["Paris sounds lovely in spring."])).await;
    for n in 0..GROUP_SIZE {
        rig.add(0, n, None).await;
    }

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!(report.written, 0);
    assert_eq!(report.unwritten, 1);
    assert!(rig.live_ids().await.is_empty());
}

#[tokio::test]
async fn a_cap_below_the_cluster_count_stops_the_level_and_the_next_pass_finishes_it() {
    // Arrange
    let rig = Rig::answering_a_tree().await;
    rig.add_groups(3).await;
    let capped = SummaryConfig {
        max_summaries_per_run: 2,
        ..enabled()
    };

    // Act
    let first = rig.run(capped).await;

    // Assert
    assert_eq!(first.written, 2);
    assert_eq!(first.unwritten, 1);
    assert!(first.partial());
    assert!(
        rig.live(2).await.is_empty(),
        "a capped level does not feed the one above"
    );

    // Act
    let second = rig.run(enabled()).await;

    // Assert
    assert_eq!(second.written, 2);
    assert!(!second.partial());
    assert_eq!(rig.live(1).await.len(), 3);
    assert_eq!(rig.live(2).await.len(), 1);
}

#[tokio::test]
async fn a_cluster_that_gained_a_member_replaces_its_summary_in_one_step() {
    // Arrange
    let rig = Rig::answering_a_tree().await;
    rig.add_groups(3).await;
    rig.run(enabled()).await;
    let before = rig.live_ids().await;
    let replies = [LEVEL_ONE_REPLY, LEVEL_TWO_REPLY];
    let rig_again = Rig {
        llm: Arc::new(ScriptedLlm::replying(&replies)),
        ..rig
    };
    rig_again.add(1, GROUP_SIZE, None).await;

    // Act
    let report = rig_again.run(enabled()).await;

    // Assert
    assert_eq!(report.written, 2);
    assert!(!report.partial());
    let after = rig_again.live_ids().await;
    assert_eq!(after.len(), before.len());
    assert_eq!(
        before.intersection(&after).count(),
        2,
        "the two untouched level one summaries stay"
    );
    let grown = rig_again
        .live(1)
        .await
        .into_iter()
        .find(|s| !before.contains(&s.id))
        .expect("a new level one summary");
    let children = rig_again.store.summary_children(&grown.id).await.unwrap();
    assert_eq!(children.len(), GROUP_SIZE + 1);
}

#[tokio::test]
async fn a_cluster_whose_items_are_gone_has_its_summary_closed() {
    // Arrange
    let rig = Rig::answering_a_tree().await;
    let groups = rig.add_groups(3).await;
    rig.run(enabled()).await;
    for id in &groups[1] {
        let bare = NewNode::now("fact", "gone", "left without a vector");
        rig.store.supersede(id, bare).await.unwrap();
    }
    let rig = Rig {
        llm: Arc::new(ScriptedLlm::replying(&[])),
        ..rig
    };

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!(report.closed, 1);
    assert_eq!(rig.live(1).await.len(), 2);
    assert!(!report.partial());
}

#[tokio::test]
async fn a_summary_text_that_cannot_be_embedded_is_not_written() {
    // Arrange
    struct Broken;

    #[async_trait]
    impl ContentEmbedder for Broken {
        async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
            Err(EmbedError::from("the embedder is down".to_string()))
        }
    }
    let rig = Rig::new(ScriptedLlm::replying(&[LEVEL_ONE_REPLY])).await;
    for n in 0..GROUP_SIZE {
        rig.add(0, n, None).await;
    }
    let hooks = SummaryHooks {
        embedder: Arc::new(Broken),
        ..rig.hooks(enabled())
    };

    // Act
    let report = hooks
        .summarize(&SummaryCtx {
            store: &rig.store,
            source: SOURCE.to_string(),
        })
        .await
        .unwrap();

    // Assert
    assert_eq!(report.written, 0);
    assert!(report.partial());
    assert!(rig.live_ids().await.is_empty());
}
