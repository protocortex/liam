// SPDX-License-Identifier: Apache-2.0
//! Extension points the consolidation job calls after its maintenance steps.
//! Each part is optional; a part left out is a no-op that reports nothing done.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

/// Input to the extraction part. Fields arrive with the extraction pipeline.
#[derive(Debug, Default)]
pub struct ExtractCtx;

/// What the extraction part did. Fields arrive with the extraction pipeline.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ExtractionReport;

/// Input to the summary part. Fields arrive with the summary tree.
#[derive(Debug, Default)]
pub struct SummaryCtx;

/// What the summary part did. Fields arrive with the summary tree.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SummaryReport;

/// Input to the tiering part. Fields arrive with utility-based tiering.
#[derive(Debug, Default)]
pub struct TierCtx;

/// What the tiering part did. Fields arrive with utility-based tiering.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TierReport;

#[async_trait]
pub trait ExtractionHook: Send + Sync {
    async fn extract(&self, ctx: &ExtractCtx) -> Result<ExtractionReport>;
}

#[async_trait]
pub trait SummaryHook: Send + Sync {
    async fn summarize(&self, ctx: &SummaryCtx) -> Result<SummaryReport>;
}

#[async_trait]
pub trait TierHook: Send + Sync {
    async fn tier(&self, ctx: &TierCtx) -> Result<TierReport>;
}

/// The hook parts a run calls. The default set has none, so every step is a
/// no-op.
#[derive(Clone, Default)]
pub struct ConsolidationHookSet {
    pub extraction: Option<Arc<dyn ExtractionHook>>,
    pub summary: Option<Arc<dyn SummaryHook>>,
    pub tiering: Option<Arc<dyn TierHook>>,
}

impl ConsolidationHookSet {
    pub async fn extract(&self, ctx: &ExtractCtx) -> Result<ExtractionReport> {
        match &self.extraction {
            Some(hook) => hook.extract(ctx).await,
            None => Ok(ExtractionReport),
        }
    }

    pub async fn summarize(&self, ctx: &SummaryCtx) -> Result<SummaryReport> {
        match &self.summary {
            Some(hook) => hook.summarize(ctx).await,
            None => Ok(SummaryReport),
        }
    }

    pub async fn tier(&self, ctx: &TierCtx) -> Result<TierReport> {
        match &self.tiering {
            Some(hook) => hook.tier(ctx).await,
            None => Ok(TierReport),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingExtraction(Arc<AtomicUsize>);

    #[async_trait]
    impl ExtractionHook for CountingExtraction {
        async fn extract(&self, _ctx: &ExtractCtx) -> Result<ExtractionReport> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(ExtractionReport)
        }
    }

    struct FailingTier;

    #[async_trait]
    impl TierHook for FailingTier {
        async fn tier(&self, _ctx: &TierCtx) -> Result<TierReport> {
            anyhow::bail!("tier part failed")
        }
    }

    #[tokio::test]
    async fn empty_set_runs_every_part_as_a_no_op() {
        // Arrange
        let hooks = ConsolidationHookSet::default();

        // Act
        let extracted = hooks.extract(&ExtractCtx).await;
        let summarized = hooks.summarize(&SummaryCtx).await;
        let tiered = hooks.tier(&TierCtx).await;

        // Assert
        assert_eq!(extracted.unwrap(), ExtractionReport);
        assert_eq!(summarized.unwrap(), SummaryReport);
        assert_eq!(tiered.unwrap(), TierReport);
    }

    #[tokio::test]
    async fn present_part_is_called_and_absent_parts_stay_no_ops() {
        // Arrange
        let calls = Arc::new(AtomicUsize::new(0));
        let hooks = ConsolidationHookSet {
            extraction: Some(Arc::new(CountingExtraction(calls.clone()))),
            ..Default::default()
        };

        // Act
        hooks.extract(&ExtractCtx).await.unwrap();
        let summarized = hooks.summarize(&SummaryCtx).await;

        // Assert
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(summarized.unwrap(), SummaryReport);
    }

    #[tokio::test]
    async fn part_error_reaches_the_caller() {
        // Arrange
        let hooks = ConsolidationHookSet {
            tiering: Some(Arc::new(FailingTier)),
            ..Default::default()
        };

        // Act
        let result = hooks.tier(&TierCtx).await;

        // Assert
        assert_eq!(result.unwrap_err().to_string(), "tier part failed");
    }
}
