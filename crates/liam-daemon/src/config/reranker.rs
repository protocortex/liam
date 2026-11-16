// SPDX-License-Identifier: Apache-2.0
//! The `[reranker]` section: which cross-encoder reorders `recall` candidates,
//! how many threads it uses and how many candidates it scores.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

pub const DEFAULT_POOL_CAP: usize = 16;
const DEFAULT_MAX_TOKENS: usize = 512;
/// Zero asks the loader to use the performance cores.
const DEFAULT_THREADS: usize = 0;

/// The reranker models the daemon accepts. An unknown name fails at config load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
pub enum RerankerModel {
    /// The legacy fastembed model.
    #[serde(rename = "bge-reranker-base")]
    BgeRerankerBase,
    #[serde(rename = "gte-multilingual-reranker-base")]
    GteMultilingualRerankerBase,
    #[serde(rename = "bge-reranker-v2-m3")]
    BgeRerankerV2M3,
}

impl RerankerModel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BgeRerankerBase => "bge-reranker-base",
            Self::GteMultilingualRerankerBase => "gte-multilingual-reranker-base",
            Self::BgeRerankerV2M3 => "bge-reranker-v2-m3",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct RerankerConfig {
    /// The cross-encoder that scores recall candidates.
    pub model: RerankerModel,
    /// Threads the model runs on. `0` means the performance cores.
    pub threads: usize,
    /// Most candidates `recall` reranks. Lower it to cut latency.
    pub pool_cap: usize,
    /// Longest query plus document pair, in tokens.
    pub max_tokens: usize,
    /// A local directory of model files used instead of the download cache.
    pub model_dir: Option<String>,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            model: RerankerModel::BgeRerankerBase,
            threads: DEFAULT_THREADS,
            pool_cap: DEFAULT_POOL_CAP,
            max_tokens: DEFAULT_MAX_TOKENS,
            model_dir: None,
        }
    }
}

impl RerankerConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.pool_cap > 0, "reranker.pool_cap must be at least 1");
        anyhow::ensure!(
            self.max_tokens > 0,
            "reranker.max_tokens must be at least 1"
        );
        anyhow::ensure!(
            self.model_dir.is_none() || self.model != RerankerModel::BgeRerankerBase,
            "reranker.model_dir cannot be used with {}: that model loads through fastembed, which has no model directory",
            RerankerModel::BgeRerankerBase.as_str()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn load(toml: &str) -> anyhow::Result<Config> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liam.toml");
        std::fs::write(&path, toml).unwrap();
        Config::load(&path)
    }

    #[test]
    fn a_config_without_the_section_uses_the_defaults() {
        // Act
        let config = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(config.reranker, RerankerConfig::default());
        assert_eq!(config.reranker.model.as_str(), "bge-reranker-base");
        assert_eq!(config.reranker.pool_cap, 16);
        assert_eq!(config.reranker.threads, 0);
        assert_eq!(config.reranker.max_tokens, 512);
        assert_eq!(config.reranker.model_dir, None);
    }

    #[test]
    fn an_unknown_key_or_model_name_fails_the_load() {
        // Act
        let key = load("[reranker]\nbackend = \"x\"\n").unwrap_err();
        let model = load("[reranker]\nmodel = \"nope\"\n").unwrap_err();

        // Assert
        assert!(format!("{key:#}").contains("backend"));
        assert!(format!("{model:#}").contains("nope"));
    }

    #[test]
    fn every_model_name_round_trips_through_the_config() {
        for model in [
            RerankerModel::BgeRerankerBase,
            RerankerModel::GteMultilingualRerankerBase,
            RerankerModel::BgeRerankerV2M3,
        ] {
            // Act
            let config = load(&format!("[reranker]\nmodel = \"{}\"\n", model.as_str())).unwrap();

            // Assert
            assert_eq!(config.reranker.model, model);
        }
    }

    #[test]
    fn a_model_dir_is_rejected_for_the_legacy_model_only() {
        // Act
        let legacy = load("[reranker]\nmodel_dir = \"/m\"\n").unwrap_err();
        let legacy_named =
            load("[reranker]\nmodel = \"bge-reranker-base\"\nmodel_dir = \"/m\"\n").unwrap_err();
        let v2 = load("[reranker]\nmodel = \"bge-reranker-v2-m3\"\nmodel_dir = \"/m\"\n").unwrap();
        let gte =
            load("[reranker]\nmodel = \"gte-multilingual-reranker-base\"\nmodel_dir = \"/m\"\n")
                .unwrap();

        // Assert
        assert!(format!("{legacy:#}").contains("reranker.model_dir"));
        assert!(format!("{legacy_named:#}").contains("reranker.model_dir"));
        assert_eq!(v2.reranker.model_dir.as_deref(), Some("/m"));
        assert_eq!(gte.reranker.model_dir.as_deref(), Some("/m"));
    }

    #[test]
    fn a_zero_pool_cap_or_token_limit_fails_the_load() {
        // Act
        let pool = load("[reranker]\npool_cap = 0\n").unwrap_err();
        let tokens = load("[reranker]\nmax_tokens = 0\n").unwrap_err();

        // Assert
        assert!(format!("{pool:#}").contains("reranker.pool_cap"));
        assert!(format!("{tokens:#}").contains("reranker.max_tokens"));
    }
}
