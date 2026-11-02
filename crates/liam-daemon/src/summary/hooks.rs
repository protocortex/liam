// SPDX-License-Identifier: Apache-2.0
//! The summarize step of the consolidation job: for each scope, plans every
//! level of the tree, writes the summaries that are missing or out of date and
//! closes those whose cluster is gone.
//!
//! A level is built from the one below it only when the one below is complete,
//! so a cap, a deadline or a model failure cannot leave a higher level
//! summarizing a part of the tree as if it were the whole.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use async_trait::async_trait;
use liam_daemon::config::SummaryConfig;
use liam_daemon::consolidation::hooks::{SummaryCtx, SummaryHook, SummaryReport};
use liam_model::Llm;
use liam_store::{
    ChildKind, ContentEmbedder, DefaultGraph, EventRef, Millis, NewSummary, PutOutcome, Summary,
    SummaryChild, SummaryInput,
};
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::{debug, info, warn};

use super::generate::{
    GenerateSettings, Generation, Generator, PromptItem, SummaryCache, SUMMARY_PROMPT_VERSION,
};
use super::tree::{
    detect_cold, plan_level, should_descend, LevelRequest, Point, PreviousCluster, Regeneration,
    TreeParams,
};
use crate::extraction::client::ModelIdentity;

/// Tokens a summary may run to: two to four sentences, with room to spare.
const SUMMARY_MAX_NEW_TOKENS: usize = 512;
const MILLIS_PER_HOUR: i64 = 3_600_000;
/// What a child summary is called in the prompt of the level above.
const CHILD_SUMMARY_LABEL: &str = "Summary of earlier items";

pub struct SummaryHooks {
    pub config: SummaryConfig,
    pub llm: Arc<dyn Llm>,
    /// The daemon's generation permits, shared with the other model users.
    pub permits: Arc<Semaphore>,
    pub cache: Arc<dyn SummaryCache>,
    pub embedder: Arc<dyn ContentEmbedder>,
    pub identity: ModelIdentity,
    /// The model's context window, in tokens.
    pub context_tokens: usize,
}

/// An item of one level, with what a summary of it needs.
struct Item {
    point: Point,
    prompt: PromptItem,
    sources: Vec<EventRef>,
    valid_min: Millis,
    valid_max: Millis,
}

fn node_item(input: &SummaryInput) -> Item {
    Item {
        point: Point::from(input),
        prompt: PromptItem {
            label: input.label.clone(),
            text: input.content.as_str().to_string(),
            event_ids: input
                .origin_events
                .iter()
                .map(|event| event.event_id.clone())
                .collect(),
        },
        sources: input.origin_events.clone(),
        valid_min: input.valid_from,
        valid_max: input.valid_from,
    }
}

fn summary_item(summary: &Summary, vector: Vec<f32>, sources: Vec<EventRef>) -> Item {
    let half_span = summary.valid_max.0.saturating_sub(summary.valid_min.0) / 2;
    Item {
        point: Point {
            id: summary.id.clone(),
            vector,
            time: Millis(summary.valid_min.0.saturating_add(half_span)),
        },
        prompt: PromptItem {
            label: CHILD_SUMMARY_LABEL.to_string(),
            text: summary.text.as_str().to_string(),
            event_ids: sources.iter().map(|event| event.event_id.clone()).collect(),
        },
        sources,
        valid_min: summary.valid_min,
        valid_max: summary.valid_max,
    }
}

const fn child_kind(level: u8) -> ChildKind {
    if level == 1 {
        ChildKind::Node
    } else {
        ChildKind::Summary
    }
}

#[async_trait]
impl SummaryHook for SummaryHooks {
    async fn summarize(&self, ctx: &SummaryCtx<'_>) -> anyhow::Result<SummaryReport> {
        if !self.config.enabled {
            return Ok(SummaryReport::default());
        }
        let mut run = Run::new(self, ctx);
        let scopes = ctx
            .store
            .summary_scopes()
            .await
            .context("could not list the scopes to summarize")?;
        for scope in scopes {
            run.build_scope(&scope).await?;
        }
        info!(
            written = run.report.written,
            kept = run.report.kept,
            closed = run.report.closed,
            unwritten = run.report.unwritten,
            "summary: pass finished"
        );
        Ok(run.report)
    }
}

/// One pass over the store, with what it has left to spend.
struct Run<'a> {
    hooks: &'a SummaryHooks,
    store: &'a DefaultGraph,
    source: &'a str,
    params: TreeParams,
    settings: GenerateSettings,
    deadline: Instant,
    /// Summaries this pass may still generate.
    budget: usize,
    report: SummaryReport,
}

impl<'a> Run<'a> {
    fn new(hooks: &'a SummaryHooks, ctx: &'a SummaryCtx<'_>) -> Self {
        let config = &hooks.config;
        Self {
            hooks,
            store: ctx.store,
            source: &ctx.source,
            params: TreeParams {
                knn: config.knn,
                min_similarity: config.min_similarity,
                time_window: Millis(
                    i64::try_from(config.time_window_hours)
                        .unwrap_or(i64::MAX)
                        .saturating_mul(MILLIS_PER_HOUR),
                ),
                min_cluster_size: config.min_cluster_size,
                max_cluster_size: config.max_cluster_size,
                max_depth: config.max_depth,
                recluster_change_ratio: config.recluster_change_ratio,
                recluster_interval: Millis::days(
                    i64::try_from(config.recluster_interval_days).unwrap_or(i64::MAX),
                ),
                min_attach_similarity: config.min_attach_similarity,
                max_unattached: config.max_unattached,
            },
            settings: GenerateSettings {
                context_tokens: hooks.context_tokens.saturating_sub(SUMMARY_MAX_NEW_TOKENS),
                max_new_tokens: SUMMARY_MAX_NEW_TOKENS,
                min_grounded_share: config.min_grounded_share,
                max_summary_failures: config.max_summary_failures,
            },
            deadline: Instant::now() + Duration::from_secs(config.run_deadline_secs),
            budget: config.max_summaries_per_run,
            report: SummaryReport::default(),
        }
    }

    async fn build_scope(&mut self, scope: &str) -> anyhow::Result<()> {
        let inputs = self
            .store
            .summary_inputs(scope)
            .await
            .context("could not read the nodes to summarize")?;
        if inputs.skipped_no_vector > 0 {
            warn!(
                skipped = inputs.skipped_no_vector,
                "summary: nodes without a vector are left out until they are embedded"
            );
        }
        let mut items: Vec<Item> = inputs.items.iter().map(node_item).collect();
        for level in 1..=self.params.max_depth {
            if !self.build_level(scope, level, &items).await? {
                break;
            }
            items = self.summary_items(scope, level).await?;
        }
        Ok(())
    }

    /// Plans and builds one level. Says whether the level above should be.
    async fn build_level(
        &mut self,
        scope: &str,
        level: u8,
        items: &[Item],
    ) -> anyhow::Result<bool> {
        let previous = self.previous_clusters(scope, level).await?;
        if items.is_empty() && previous.is_empty() {
            return Ok(false);
        }
        let state = self.store.tree_state(scope, level).await?;
        let points: Vec<Point> = items.iter().map(|item| item.point.clone()).collect();
        let request = LevelRequest {
            scope,
            level,
            points: &points,
            previous: &previous,
            state: state.as_ref(),
            now: Millis::now(),
        };
        let plan = match plan_level(&request, &self.params, &detect_cold) {
            Ok(plan) => plan,
            Err(error) => {
                warn!(level, %error, "summary: the level is left as it was");
                self.report.unwritten += 1;
                return Ok(false);
            }
        };
        self.report.kept += plan.keep.len();
        let by_id: HashMap<&str, &Item> = items
            .iter()
            .map(|item| (item.point.id.as_str(), item))
            .collect();
        let mut complete = true;
        for regeneration in &plan.regenerate {
            complete &= self.regenerate(scope, level, regeneration, &by_id).await?;
        }
        for id in &plan.dissolved {
            self.store
                .close_summary(id, Millis::now(), self.source)
                .await
                .context("could not close a dissolved summary")?;
            self.report.closed += 1;
        }
        if complete {
            self.store.put_tree_state(&plan.next_state).await?;
        }
        Ok(should_descend(
            level,
            items.len(),
            &plan,
            complete,
            &self.params,
        ))
    }

    /// Writes the summary of one cluster. Says whether it is in the store.
    async fn regenerate(
        &mut self,
        scope: &str,
        level: u8,
        regeneration: &Regeneration,
        by_id: &HashMap<&str, &Item>,
    ) -> anyhow::Result<bool> {
        if self.budget == 0 || Instant::now() >= self.deadline {
            self.report.unwritten += 1;
            return Ok(false);
        }
        self.budget -= 1;
        let members = regeneration
            .cluster
            .members
            .iter()
            .map(|id| {
                by_id.get(id.as_str()).copied().ok_or_else(|| {
                    anyhow!("a planned cluster names an item {id} that is not there")
                })
            })
            .collect::<anyhow::Result<Vec<&Item>>>()?;
        let prompt: Vec<PromptItem> = members.iter().map(|item| item.prompt.clone()).collect();
        let generation = self
            .generator()
            .generate(&regeneration.cluster.child_hash, &prompt, self.deadline)
            .await;
        let generated = match generation {
            Generation::Generated(generated) => generated,
            other => {
                debug!(level, outcome = ?other, "summary: no summary for a cluster this pass");
                self.report.unwritten += 1;
                return Ok(false);
            }
        };
        let embedding = match self.hooks.embedder.embed(&generated.text).await {
            Ok(embedding) => embedding,
            Err(error) => {
                warn!(level, %error, "summary: the text could not be embedded");
                self.report.unwritten += 1;
                return Ok(false);
            }
        };
        let (valid_min, valid_max) = members
            .iter()
            .map(|item| (item.valid_min, item.valid_max))
            .reduce(|(min, max), (lo, hi)| (min.min(lo), max.max(hi)))
            .ok_or_else(|| anyhow!("a planned cluster has no members"))?;
        let summary = NewSummary {
            level,
            scope: scope.to_string(),
            text: generated.text,
            events_fed: generated.events_fed,
            valid_min,
            valid_max,
            model_id: self.hooks.identity.model_id.clone(),
            prompt_version: SUMMARY_PROMPT_VERSION.to_string(),
        };
        let children: Vec<SummaryChild> = regeneration
            .cluster
            .members
            .iter()
            .map(|id| SummaryChild {
                id: id.clone(),
                kind: child_kind(level),
            })
            .collect();
        let closes: Vec<String> = regeneration.replaces.iter().cloned().collect();
        let outcome = self
            .store
            .put_summary(
                summary,
                &children,
                &union_of_sources(&members),
                &embedding,
                &closes,
                self.source,
            )
            .await
            .context("could not store a summary")?;
        match outcome {
            PutOutcome::Inserted(_) => self.report.written += 1,
            PutOutcome::Unchanged(_) => self.report.kept += 1,
        }
        Ok(true)
    }

    fn generator(&self) -> Generator<'_> {
        Generator {
            llm: self.hooks.llm.as_ref(),
            permits: &self.hooks.permits,
            cache: self.hooks.cache.as_ref(),
            store: self.store,
            identity: &self.hooks.identity,
            settings: &self.settings,
        }
    }

    async fn previous_clusters(
        &self,
        scope: &str,
        level: u8,
    ) -> anyhow::Result<Vec<PreviousCluster>> {
        let mut previous = Vec::new();
        for summary in self.store.live_summaries(level, scope).await? {
            let children = self.store.summary_children(&summary.id).await?;
            previous.push(PreviousCluster {
                summary_id: summary.id,
                child_hash: summary.child_hash,
                children: children.into_iter().map(|child| child.id).collect(),
            });
        }
        Ok(previous)
    }

    /// The open summaries of `level`, as the items of the level above. One
    /// with no vector, because its embedding failed, is left out.
    async fn summary_items(&self, scope: &str, level: u8) -> anyhow::Result<Vec<Item>> {
        let mut vectors: HashMap<String, Vec<f32>> = self
            .store
            .summary_embeddings(scope, level)
            .await?
            .into_iter()
            .collect();
        let mut items = Vec::new();
        for summary in self.store.live_summaries(level, scope).await? {
            let Some(vector) = vectors.remove(&summary.id) else {
                continue;
            };
            let sources = self.store.summary_sources(&summary.id).await?;
            items.push(summary_item(&summary, vector, sources));
        }
        Ok(items)
    }
}

/// The events behind any member, once each. A real event stays real when a
/// synthetic one shares its id.
fn union_of_sources(members: &[&Item]) -> Vec<EventRef> {
    let mut by_id: BTreeMap<&str, bool> = BTreeMap::new();
    for source in members.iter().flat_map(|item| &item.sources) {
        let synthetic = by_id.entry(&source.event_id).or_insert(source.synthetic);
        *synthetic &= source.synthetic;
    }
    by_id
        .into_iter()
        .map(|(event_id, synthetic)| EventRef {
            event_id: event_id.to_string(),
            synthetic,
        })
        .collect()
}

#[cfg(test)]
mod tests;
