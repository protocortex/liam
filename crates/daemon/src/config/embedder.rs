// SPDX-License-Identifier: Apache-2.0
//! The `[embedder]` section: which provider embeds text, which model it loads
//! and where the model's files live.

use serde::Deserialize;
#[cfg(test)]
use serde::Serialize;

/// The warning for a `device` key left in an old config.
const REMOVED_DEVICE_WARNING: &str =
    "embedder.device was removed and is ignored, the embedder always runs on the CPU: \
     delete it from your liam.toml";

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(test, derive(Serialize))]
#[serde(default, deny_unknown_fields)]
pub struct EmbedderConfig {
    /// "mock" (tests and dev) or "local" (in-process ONNX).
    pub provider: String,
    /// The Hugging Face id of the embedding model. Only arctic-embed-m-v2.0
    /// loads, so this key can be left out.
    pub model: String,
    /// Where the model's files live, under their own `models--Org--Name`
    /// subdirectory.
    pub cache_dir: String,
    /// The removed `embedder.device` key. Kept so an old `liam.toml` still
    /// loads, since `deny_unknown_fields` would reject it. Its value is
    /// ignored, see `removed_device_warning`.
    #[doc(hidden)]
    pub device: Option<toml::Value>,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            provider: "mock".into(),
            model: liam_model::EMBEDDER_MODEL_ID.into(),
            cache_dir: "~/.liam/models".into(),
            device: None,
        }
    }
}

impl EmbedderConfig {
    /// Refuses a model other than arctic-embed-m-v2.0, and stores the id
    /// spelled the way the model library spells it, since Hugging Face ids are
    /// case-insensitive.
    pub fn validate(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.model
                .eq_ignore_ascii_case(liam_model::EMBEDDER_MODEL_ID),
            "embedder.model = {:?} is not supported: {} is the only embedder since this \
             version. Delete the embedder.model line from your liam.toml.",
            self.model,
            liam_model::EMBEDDER_MODEL_ID
        );
        self.model = liam_model::EMBEDDER_MODEL_ID.into();
        Ok(())
    }

    /// The warning to log at start when the config still has the removed
    /// `device` key.
    pub fn removed_device_warning(&self) -> Option<&'static str> {
        self.device.as_ref().map(|_| REMOVED_DEVICE_WARNING)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::Config;

    const OLD_MODEL_IDS: [&str; 4] = [
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
        "BAAI/bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2",
    ];

    fn load(toml: &str) -> anyhow::Result<Config> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("liam.toml");
        std::fs::write(&path, toml).unwrap();
        Config::load(&path)
    }

    #[test]
    fn a_config_without_the_key_embeds_with_arctic() {
        // Act
        let loaded = load("[gc]\n").unwrap();

        // Assert
        assert_eq!(loaded.embedder.model, liam_model::EMBEDDER_MODEL_ID);
        assert_eq!(
            Config::default().embedder.model,
            "Snowflake/snowflake-arctic-embed-m-v2.0"
        );
    }

    #[test]
    fn the_arctic_id_loads_in_any_letter_case_and_is_stored_in_the_library_spelling() {
        for spelling in [
            liam_model::EMBEDDER_MODEL_ID.to_string(),
            liam_model::EMBEDDER_MODEL_ID.to_lowercase(),
        ] {
            // Arrange
            let text = format!("[embedder]\nmodel = \"{spelling}\"\n");

            // Act
            let loaded = load(&text).unwrap();

            // Assert
            assert_eq!(loaded.embedder.model, liam_model::EMBEDDER_MODEL_ID);
        }
    }

    #[test]
    fn any_other_model_stops_the_load_naming_the_value_and_the_fix() {
        for model in OLD_MODEL_IDS {
            // Arrange
            let text = format!("[embedder]\nmodel = \"{model}\"\n");

            // Act
            let error = load(&text).unwrap_err().to_string();

            // Assert
            assert!(error.contains(model), "{error}");
            assert!(
                error.contains("only embedder since this version"),
                "{error}"
            );
            assert!(error.contains("Delete the embedder.model line"), "{error}");
        }
    }

    #[test]
    fn another_model_is_refused_under_the_mock_provider_too() {
        // Arrange
        let text = "[embedder]\nprovider = \"mock\"\nmodel = \"BAAI/bge-m3\"\n";

        // Act
        let error = load(text).unwrap_err().to_string();

        // Assert
        assert!(error.contains("BAAI/bge-m3"), "{error}");
    }

    #[test]
    fn a_removed_device_key_loads_and_yields_one_warning() {
        for device in ["auto", "metal", "cuda", "cpu"] {
            // Arrange
            let text = format!("[embedder]\ndevice = \"{device}\"\n");

            // Act
            let loaded = load(&text).unwrap();

            // Assert
            let warning = loaded.embedder.removed_device_warning().unwrap();
            assert!(warning.contains("embedder.device was removed"), "{warning}");
            assert!(warning.contains("delete it"), "{warning}");
        }
    }

    #[test]
    fn a_config_without_the_device_key_has_no_warning() {
        // Act
        let loaded = load("[embedder]\nprovider = \"mock\"\n").unwrap();

        // Assert
        assert_eq!(loaded.embedder.removed_device_warning(), None);
        assert_eq!(Config::default().embedder.removed_device_warning(), None);
    }
}
