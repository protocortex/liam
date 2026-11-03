// SPDX-License-Identifier: Apache-2.0
//! The summarize step over a store in memory and a scripted model: which
//! summaries one pass writes, keeps and closes, and what a failure leaves.

use std::collections::BTreeSet;

use liam_store::{EmbedError, GraphConfig, NewNode, NodeId};

use super::*;
use crate::extraction::client::ModelIdentity;
use crate::summary::summariser::LocalSummariser;
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

    /// A node of `group`'s text that sits on the first axis whatever its group.
    async fn add_on_first_axis(&self, group: usize, n: usize) -> NodeId {
        let content = format!(
            "ledger notes number {n} cover quarterly forecasts for the {} team",
            GROUP_NAMES[group]
        );
        let node = NewNode::now("fact", format!("Note {n}"), content)
            .with_embedding(axis(0))
            .with_valid_from(Millis(1000 + (group * 100 + n) as i64));
        self.store.insert(node).await.unwrap()
    }

    /// Stores a summary of `children` as an earlier pass would have.
    async fn seed_summary(&self, children: &[NodeId]) -> String {
        let ids: Vec<SummaryChild> = children
            .iter()
            .map(|id| SummaryChild {
                id: id.as_str().to_string(),
                kind: ChildKind::Node,
            })
            .collect();
        let source = EventRef {
            event_id: "event-seed".to_string(),
            synthetic: false,
        };
        let summary = NewSummary {
            level: 1,
            scope: String::new(),
            text: LEVEL_ONE_REPLY.to_string(),
            events_fed: 1,
            valid_min: Millis(1000),
            valid_max: Millis(2000),
            model_id: "model".to_string(),
            prompt_version: SUMMARY_PROMPT_VERSION.to_string(),
        };
        match self
            .store
            .put_summary(summary, &ids, &[source], &axis(0), &[], SOURCE)
            .await
            .unwrap()
        {
            PutOutcome::Inserted(id) | PutOutcome::Unchanged(id) => id,
        }
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
            summariser: Arc::new(LocalSummariser::new(
                self.llm.clone(),
                ModelIdentity {
                    model_id: "model".to_string(),
                    model_sha256: String::new(),
                },
            )),
            permits: Arc::new(Semaphore::new(1)),
            cache: self.cache.clone(),
            embedder: Arc::new(FixedEmbedder),
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

/// Supersedes every item of the second group, so its summary has no cluster.
async fn rig_with_a_gone_group(grace: u32) -> (Rig, SummaryConfig) {
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
    let config = SummaryConfig {
        dissolve_grace_runs: grace,
        ..enabled()
    };
    (rig, config)
}

#[tokio::test]
async fn a_cluster_whose_items_are_gone_has_its_summary_closed_when_its_grace_is_one_run() {
    // Arrange
    let (rig, config) = rig_with_a_gone_group(1).await;

    // Act
    let report = rig.run(config).await;

    // Assert
    assert_eq!(report.closed, 1);
    assert_eq!(rig.live(1).await.len(), 2);
    assert!(!report.partial());
}

#[tokio::test]
async fn a_summary_with_no_replacement_stays_open_until_its_grace_runs_out() {
    // Arrange
    let (rig, config) = rig_with_a_gone_group(2).await;

    // Act
    let first = rig.run(config.clone()).await;
    let after_first = rig.live(1).await.len();
    let second = rig.run(config).await;

    // Assert
    assert_eq!((first.closed, after_first), (0, 3));
    assert_eq!((second.closed, rig.live(1).await.len()), (1, 2));
}

/// Eight items on one axis, summarized earlier as two halves.
async fn rig_with_two_summaries_of_one_cluster(
    model: ScriptedLlm,
) -> (Rig, [String; 2], [Vec<NodeId>; 2]) {
    let rig = Rig::new(model).await;
    let mut halves = [Vec::new(), Vec::new()];
    for (half, group) in halves.iter_mut().zip(0..) {
        for n in 0..GROUP_SIZE {
            half.push(rig.add_on_first_axis(group, n).await);
        }
    }
    let old = [
        rig.seed_summary(&halves[0]).await,
        rig.seed_summary(&halves[1]).await,
    ];
    (rig, old, halves)
}

#[tokio::test]
async fn a_merge_closes_the_second_old_summary_once_the_new_one_is_stored() {
    // Arrange
    let (rig, old, _) =
        rig_with_two_summaries_of_one_cluster(ScriptedLlm::replying(&[LEVEL_ONE_REPLY])).await;

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!((report.written, report.closed), (1, 1));
    let live = rig.live(1).await;
    assert_eq!(live.len(), 1);
    assert!(old.iter().all(|id| *id != live[0].id));
}

#[tokio::test]
async fn a_merge_whose_summary_cannot_be_written_leaves_both_old_summaries_open() {
    // Arrange
    let (rig, old, _) = rig_with_two_summaries_of_one_cluster(ScriptedLlm::failing("down")).await;

    // Act
    let report = rig.run(enabled()).await;

    // Assert
    assert_eq!((report.written, report.closed), (0, 0));
    let live: BTreeSet<String> = rig.live(1).await.into_iter().map(|s| s.id).collect();
    assert_eq!(live, BTreeSet::from(old));
}

#[tokio::test]
async fn a_dissolved_summary_is_closed_when_its_grace_runs_out_without_a_replacement() {
    // Arrange
    let (rig, old, _) = rig_with_two_summaries_of_one_cluster(ScriptedLlm::scripted(vec![
        Err("down".to_string()),
        Err("down".to_string()),
    ]))
    .await;
    let config = SummaryConfig {
        dissolve_grace_runs: 2,
        ..enabled()
    };

    // Act
    let first = rig.run(config.clone()).await;
    let second = rig.run(config).await;

    // Assert
    assert_eq!((first.closed, second.closed), (0, 1));
    assert_eq!(rig.live(1).await.len(), 1, "{old:?}");
}

#[tokio::test]
async fn a_replacement_closes_an_old_summary_only_when_it_holds_enough_of_its_items() {
    for (overlap, closed) in [(0.5, 0), (0.25, 1)] {
        // Arrange: of the first half's four items only one is left in the new cluster
        let (rig, _old, halves) =
            rig_with_two_summaries_of_one_cluster(ScriptedLlm::replying(&[LEVEL_ONE_REPLY])).await;
        for id in &halves[0][1..] {
            let bare = NewNode::now("fact", "gone", "left without a vector");
            rig.store.supersede(id, bare).await.unwrap();
        }
        let config = SummaryConfig {
            dissolve_overlap: overlap,
            ..enabled()
        };

        // Act
        let report = rig.run(config).await;

        // Assert: the second half's summary is replaced in the same step either way
        assert_eq!(report.written, 1, "overlap {overlap}");
        assert_eq!(report.closed, closed, "overlap {overlap}");
    }
}

#[tokio::test]
async fn the_prompt_of_a_level_above_holds_its_summaries_then_a_few_of_their_items() {
    // Arrange
    let rig = Rig::answering_a_tree().await;
    rig.add_groups(3).await;

    // Act
    rig.run(enabled()).await;

    // Assert
    let prompts = rig.llm.prompts();
    let (_, user) = &prompts[3];
    for name in GROUP_NAMES {
        let sampled = user.matches(&format!("for the {name} team")).count();
        assert_eq!(sampled, 2, "{name}");
    }
    let last_summary = user.rfind(CHILD_SUMMARY_LABEL).unwrap();
    let first_item = user.find("ledger notes number").unwrap();
    assert!(first_item > last_summary, "{user}");
    assert_eq!(user.matches(CHILD_SUMMARY_LABEL).count(), 3);
}

fn prompt_items(ids: &[&str]) -> HashMap<String, PromptItem> {
    ids.iter()
        .map(|id| {
            let item = PromptItem {
                label: (*id).to_string(),
                text: format!("text of {id}"),
                event_ids: Vec::new(),
            };
            ((*id).to_string(), item)
        })
        .collect()
}

fn labels(items: &[PromptItem]) -> Vec<&str> {
    items.iter().map(|item| item.label.as_str()).collect()
}

#[test]
fn the_sample_is_spread_over_the_members_in_id_order_whatever_the_order_they_came_in() {
    // Arrange
    let nodes = prompt_items(&["a", "b", "c", "d", "e", "f"]);
    let forward: BTreeSet<String> = ["a", "b", "c", "d", "e", "f"].map(String::from).into();
    let backward: BTreeSet<String> = ["f", "e", "d", "c", "b", "a"].map(String::from).into();

    // Act
    let from_forward = sample_of(&forward, &nodes);
    let from_backward = sample_of(&backward, &nodes);

    // Assert
    assert_eq!(labels(&from_forward), ["a", "d"]);
    assert_eq!(from_forward, from_backward);
}

#[test]
fn a_summary_with_few_members_samples_all_of_those_still_in_the_store() {
    // Arrange
    let nodes = prompt_items(&["a", "c"]);
    let members: BTreeSet<String> = ["a", "b", "c"].map(String::from).into();

    // Act
    let sample = sample_of(&members, &nodes);

    // Assert
    assert_eq!(labels(&sample), ["a", "c"]);
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
