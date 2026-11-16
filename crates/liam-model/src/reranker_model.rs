// SPDX-License-Identifier: Apache-2.0
//! Pinned ONNX cross-encoder rerankers that load from a local directory.
//!
//! Each spec in `RERANKER_MODELS` names a Hugging Face repo at one revision and
//! lists every file the loader needs with its size and sha256. A downloader
//! fetches those files into a directory; this module only reads them.

/// One file of a pinned model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedFile {
    /// The file name inside the model directory.
    pub name: &'static str,
    /// The path of the file inside the Hugging Face repo.
    pub remote_path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// A reranker pinned to one Hugging Face revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankerModelSpec {
    /// The id config selects the model with.
    pub id: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [PinnedFile],
}

pub const ONNX_FILE_NAME: &str = "model_int8.onnx";
pub const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";
pub const SPECIAL_TOKENS_FILE_NAME: &str = "special_tokens_map.json";
pub const CONFIG_FILE_NAME: &str = "config.json";

const ONNX_REMOTE_PATH: &str = "onnx/model_int8.onnx";

pub const GTE_RERANKER_ID: &str = "gte-multilingual-reranker-base";
pub const BGE_RERANKER_V2_M3_ID: &str = "bge-reranker-v2-m3";

// Both repos ship the same special tokens map.
const SPECIAL_TOKENS_SIZE: u64 = 964;
const SPECIAL_TOKENS_SHA256: &str =
    "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835";

const fn file(
    name: &'static str,
    remote_path: &'static str,
    size: u64,
    sha256: &'static str,
) -> PinnedFile {
    PinnedFile {
        name,
        remote_path,
        size,
        sha256,
    }
}

const GTE_FILES: &[PinnedFile] = &[
    file(
        ONNX_FILE_NAME,
        ONNX_REMOTE_PATH,
        340_858_200,
        "ccf51dba7f8aa9205753761cfaa68c55f741792501463a3bf25d7e5bcdac7c35",
    ),
    file(
        TOKENIZER_FILE_NAME,
        TOKENIZER_FILE_NAME,
        17_082_999,
        "3ffb37461c391f096759f4a9bbbc329da0f36952f88bab061fcf84940c022e98",
    ),
    file(
        TOKENIZER_CONFIG_FILE_NAME,
        TOKENIZER_CONFIG_FILE_NAME,
        1_340,
        "6f00514620aff01ba8b7291b2394e98daca5be264cb743805232d9ae27494b2a",
    ),
    file(
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_SIZE,
        SPECIAL_TOKENS_SHA256,
    ),
    file(
        CONFIG_FILE_NAME,
        CONFIG_FILE_NAME,
        1_578,
        "dfa5713436ecb4616eaa576795c8d3efd1f03122031a1ad4973d0b6b7e7edfd3",
    ),
];

const BGE_FILES: &[PinnedFile] = &[
    file(
        ONNX_FILE_NAME,
        ONNX_REMOTE_PATH,
        570_727_094,
        "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
    ),
    file(
        TOKENIZER_FILE_NAME,
        TOKENIZER_FILE_NAME,
        17_082_900,
        "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
    ),
    file(
        TOKENIZER_CONFIG_FILE_NAME,
        TOKENIZER_CONFIG_FILE_NAME,
        1_203,
        "b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80",
    ),
    file(
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_FILE_NAME,
        SPECIAL_TOKENS_SIZE,
        SPECIAL_TOKENS_SHA256,
    ),
    file(
        CONFIG_FILE_NAME,
        CONFIG_FILE_NAME,
        848,
        "122e922dcfed6503c8721e6fe1daf090340c3d95ca7f3aa3a72730b321a51cfd",
    ),
];

/// The rerankers this module can load.
pub const RERANKER_MODELS: &[RerankerModelSpec] = &[
    RerankerModelSpec {
        id: GTE_RERANKER_ID,
        repo: "onnx-community/gte-multilingual-reranker-base",
        revision: "ee64367e35a2db0da46bb6497e13a18f8bd585cb",
        files: GTE_FILES,
    },
    RerankerModelSpec {
        id: BGE_RERANKER_V2_M3_ID,
        repo: "onnx-community/bge-reranker-v2-m3-ONNX",
        revision: "6f5ff65298512715a1e669753bc754d2bc8f367b",
        files: BGE_FILES,
    },
];

/// The spec for a config model id, `None` when this module does not load it.
pub fn reranker_model_spec(id: &str) -> Option<&'static RerankerModelSpec> {
    RERANKER_MODELS.iter().find(|spec| spec.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA256_HEX_LEN: usize = 64;

    #[test]
    fn known_ids_resolve_and_an_unknown_id_does_not() {
        // Arrange
        let ids = [GTE_RERANKER_ID, BGE_RERANKER_V2_M3_ID];

        // Act
        let found: Vec<_> = ids.iter().map(|id| reranker_model_spec(id)).collect();
        let missing = reranker_model_spec("bge-reranker-base");

        // Assert
        for (id, spec) in ids.iter().zip(found) {
            assert_eq!(spec.unwrap().id, *id);
        }
        assert!(missing.is_none());
    }

    #[test]
    fn every_pinned_file_is_well_formed() {
        for spec in RERANKER_MODELS {
            // Arrange
            let names: Vec<_> = spec.files.iter().map(|f| f.name).collect();

            // Act and Assert
            assert!(names.contains(&ONNX_FILE_NAME), "{} has no model", spec.id);
            for f in spec.files {
                assert!(
                    !f.name.is_empty() && !f.name.contains('/') && !f.name.contains(".."),
                    "{} is not a plain file name",
                    f.name
                );
                assert!(f.size > 0, "{} has no size", f.name);
                assert_eq!(f.sha256.len(), SHA256_HEX_LEN, "{}", f.name);
                assert!(
                    f.sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "{} hash is not lowercase hex",
                    f.name
                );
            }
        }
    }
}
