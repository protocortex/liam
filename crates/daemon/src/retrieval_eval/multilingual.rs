// SPDX-License-Identifier: Apache-2.0
//! Multilingual embedder check: nDCG@10 in eight languages, with a gate on the
//! three languages the project targets.
//!
//! The set holds 30 queries and 287 facts per language. `pairs.json` names the
//! relevant facts for each query and is the same for every language; each
//! language file maps the same query and fact ids to text. English is the
//! source and the other seven are machine translated, so a gap between
//! languages is a signal, not a measure of translation quality.
//!
//! For one language, every fact is embedded as a document, every query as a
//! query, and all facts are ranked by cosine similarity. English, Portuguese
//! and Spanish gate against the committed baseline; the rest are reported.

use std::collections::{BTreeMap, HashSet};

use anyhow::Context as _;
use liam_model::{Embedder, TextKind};
use serde::Deserialize;

use super::embedder_bench::cosine;
use super::ndcg_at_10;

const PAIRS_JSON: &str = include_str!("multilingual/pairs.json");
const BASELINE_JSON: &str = include_str!("multilingual/baseline-arctic-embed-m-v2-384.json");

const QUERIES_PER_LANGUAGE: usize = 30;
#[cfg(test)]
const FACTS_PER_LANGUAGE: usize = 287;

/// A gating language may fall this far below its baseline.
const LANGUAGE_TOLERANCE: f64 = 0.05;
/// The mean of the gating languages may fall this far below its baseline.
const MEAN_TOLERANCE: f64 = 0.02;
/// Absorbs float error so a drop of exactly the tolerance passes.
const FLOAT_SLACK: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Language {
    En,
    Pt,
    Es,
    Zh,
    Fr,
    De,
    Ja,
    Ar,
}

impl Language {
    const ALL: [Language; 8] = [
        Language::En,
        Language::Pt,
        Language::Es,
        Language::Zh,
        Language::Fr,
        Language::De,
        Language::Ja,
        Language::Ar,
    ];

    /// The languages whose score can fail the check.
    const GATING: [Language; 3] = [Language::En, Language::Pt, Language::Es];

    fn code(self) -> &'static str {
        match self {
            Language::En => "en",
            Language::Pt => "pt",
            Language::Es => "es",
            Language::Zh => "zh",
            Language::Fr => "fr",
            Language::De => "de",
            Language::Ja => "ja",
            Language::Ar => "ar",
        }
    }

    fn data_json(self) -> &'static str {
        match self {
            Language::En => include_str!("multilingual/en.json"),
            Language::Pt => include_str!("multilingual/pt.json"),
            Language::Es => include_str!("multilingual/es.json"),
            Language::Zh => include_str!("multilingual/zh.json"),
            Language::Fr => include_str!("multilingual/fr.json"),
            Language::De => include_str!("multilingual/de.json"),
            Language::Ja => include_str!("multilingual/ja.json"),
            Language::Ar => include_str!("multilingual/ar.json"),
        }
    }

    fn is_gating(self) -> bool {
        Self::GATING.contains(&self)
    }
}

#[derive(Debug, Deserialize)]
struct Pair {
    qid: String,
    relevant: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PairsFile {
    pairs: Vec<Pair>,
}

#[derive(Debug, Deserialize)]
struct LanguageData {
    language: Language,
    machine_translated: bool,
    queries: BTreeMap<String, String>,
    facts: BTreeMap<String, String>,
}

fn load_pairs() -> anyhow::Result<Vec<Pair>> {
    let file: PairsFile = serde_json::from_str(PAIRS_JSON).context("parsing pairs.json")?;
    Ok(file.pairs)
}

fn load_language(language: Language) -> anyhow::Result<LanguageData> {
    serde_json::from_str(language.data_json())
        .with_context(|| format!("parsing {}.json", language.code()))
}

/// nDCG@10 for each query, in `pairs.json` order, and their mean.
#[derive(Debug, Clone, PartialEq)]
struct LanguageScore {
    per_query: Vec<(String, f64)>,
    mean: f64,
}

/// Fact ids best first. Equal scores keep id order, so the ranking is
/// deterministic.
fn rank(query: &[f32], fact_ids: &[&str], vectors: &[Vec<f32>]) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| (index, cosine(query, vector)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored
        .into_iter()
        .map(|(index, _)| fact_ids[index].to_string())
        .collect()
}

/// Scores `language` with `embedder`.
///
/// # Errors
///
/// Fails on malformed data, an id without text, or an embedder error.
async fn score_language(
    embedder: &dyn Embedder,
    language: Language,
) -> anyhow::Result<LanguageScore> {
    let pairs = load_pairs()?;
    let data = load_language(language)?;
    let fact_ids: Vec<&str> = data.facts.keys().map(String::as_str).collect();
    // One text per call: the default model is dynamically quantised, and the
    // ONNX runtime refuses to batch more than one run of such a model.
    let mut vectors = Vec::with_capacity(data.facts.len());
    for (id, text) in &data.facts {
        let vector = embedder
            .embed(text, TextKind::Document)
            .await
            .with_context(|| format!("embedding {} fact {id}", language.code()))?;
        vectors.push(vector);
    }

    let mut per_query = Vec::with_capacity(pairs.len());
    for pair in &pairs {
        let text = data
            .queries
            .get(&pair.qid)
            .with_context(|| format!("{} has no text for query {}", language.code(), pair.qid))?;
        let query = embedder
            .embed(text, TextKind::Query)
            .await
            .with_context(|| format!("embedding query {}", pair.qid))?;
        let relevant: HashSet<&str> = pair.relevant.iter().map(String::as_str).collect();
        let ranked = rank(&query, &fact_ids, &vectors);
        per_query.push((pair.qid.clone(), ndcg_at_10(&ranked, &relevant)));
    }
    let mean = mean(per_query.iter().map(|(_, score)| *score));
    Ok(LanguageScore { per_query, mean })
}

fn mean(values: impl ExactSizeIterator<Item = f64>) -> f64 {
    let count = values.len().max(1) as f64;
    values.sum::<f64>() / count
}

/// The committed per-language scores of one embedder.
#[derive(Debug, Deserialize)]
struct Baseline {
    embedder: String,
    dims: usize,
    ndcg_at_10: BTreeMap<Language, f64>,
}

impl Baseline {
    fn load() -> anyhow::Result<Self> {
        serde_json::from_str(BASELINE_JSON).context("parsing the multilingual baseline")
    }

    /// Fails when the baseline was measured with a different embedder or width
    /// than the one under test, so a changed default cannot pass on old numbers.
    fn check_matches(&self, embedder: &str, dims: usize) -> Result<(), String> {
        if self.embedder == embedder && self.dims == dims {
            return Ok(());
        }
        Err(format!(
            "the baseline was measured with {} at {} dims but the configured embedder is \
             {embedder} at {dims} dims: re-measure the baseline and commit it",
            self.embedder, self.dims
        ))
    }
}

/// The gate's verdict. `reasons` lists every failure and is empty on a pass.
#[derive(Debug, PartialEq, Eq)]
struct GateOutcome {
    reasons: Vec<String>,
}

impl GateOutcome {
    fn passed(&self) -> bool {
        self.reasons.is_empty()
    }
}

/// Each gating language must stay within [`LANGUAGE_TOLERANCE`] of its baseline
/// and their mean within [`MEAN_TOLERANCE`]. Other languages never fail it.
fn gate(baseline: &BTreeMap<Language, f64>, measured: &BTreeMap<Language, f64>) -> GateOutcome {
    let mut reasons = Vec::new();
    let mut baseline_gating = Vec::new();
    let mut measured_gating = Vec::new();
    for language in Language::GATING {
        let (Some(&base), Some(&now)) = (baseline.get(&language), measured.get(&language)) else {
            reasons.push(format!("{}: no baseline or no score", language.code()));
            continue;
        };
        if !now.is_finite() {
            reasons.push(format!(
                "{}: the score {now} is not a number",
                language.code()
            ));
            continue;
        }
        baseline_gating.push(base);
        measured_gating.push(now);
        if now < base - LANGUAGE_TOLERANCE - FLOAT_SLACK {
            reasons.push(format!(
                "{}: {now:.3} is more than {LANGUAGE_TOLERANCE} below the baseline {base:.3}",
                language.code()
            ));
        }
    }
    if !baseline_gating.is_empty() {
        let base = mean(baseline_gating.into_iter());
        let now = mean(measured_gating.into_iter());
        if now < base - MEAN_TOLERANCE - FLOAT_SLACK {
            reasons.push(format!(
                "mean of the gating languages: {now:.3} is more than {MEAN_TOLERANCE} below \
                 the baseline {base:.3}"
            ));
        }
    }
    GateOutcome { reasons }
}

/// One line per language: score, baseline, delta, and whether it gates.
fn render_table(baseline: &BTreeMap<Language, f64>, measured: &BTreeMap<Language, f64>) -> String {
    let mut out = String::from("lang  ndcg@10  baseline  delta   role\n");
    for language in Language::ALL {
        let (Some(now), Some(base)) = (measured.get(&language), baseline.get(&language)) else {
            continue;
        };
        let role = if language.is_gating() {
            "gates"
        } else {
            "reported"
        };
        out.push_str(&format!(
            "{:<4}  {now:.3}    {base:.3}     {:+.3}  {role}\n",
            language.code(),
            now - base
        ));
    }
    out
}

#[cfg(test)]
mod table_tests;

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;

    const QUERY_COUNT_PER_LANGUAGE: usize = QUERIES_PER_LANGUAGE;
    const BASE_SCORE: f64 = 0.9;
    const CONSTANT_VECTOR_DIMS: usize = 4;
    const SCORE_EPSILON: f64 = 1e-9;
    /// A constant embedder ties every fact, so it can only score by luck.
    const CONSTANT_SCORE_CEILING: f64 = 0.5;

    fn uniform(value: f64) -> BTreeMap<Language, f64> {
        Language::ALL.iter().map(|l| (*l, value)).collect()
    }

    fn with(
        mut scores: BTreeMap<Language, f64>,
        language: Language,
        value: f64,
    ) -> BTreeMap<Language, f64> {
        scores.insert(language, value);
        scores
    }

    /// Gives each document and each query the vector it was registered with.
    struct TableEmbedder {
        documents: BTreeMap<String, Vec<f32>>,
        queries: BTreeMap<String, Vec<f32>>,
        dims: usize,
    }

    #[async_trait]
    impl Embedder for TableEmbedder {
        fn dims(&self) -> usize {
            self.dims
        }

        async fn embed(&self, text: &str, kind: TextKind) -> liam_model::Result<Vec<f32>> {
            let table = if kind == TextKind::Query {
                &self.queries
            } else {
                &self.documents
            };
            Ok(table
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.dims]))
        }
    }

    /// Each fact is its own axis; each query points at its relevant facts.
    fn oracle_for(language: Language) -> TableEmbedder {
        let data = load_language(language).unwrap();
        let pairs = load_pairs().unwrap();
        let dims = data.facts.len();
        let axis: BTreeMap<&str, usize> = data
            .facts
            .keys()
            .enumerate()
            .map(|(index, id)| (id.as_str(), index))
            .collect();
        let documents = data
            .facts
            .iter()
            .map(|(id, text)| {
                let mut vector = vec![0.0; dims];
                vector[axis[id.as_str()]] = 1.0;
                (text.clone(), vector)
            })
            .collect();
        let queries = pairs
            .iter()
            .map(|pair| {
                let mut vector = vec![0.0; dims];
                for id in &pair.relevant {
                    vector[axis[id.as_str()]] = 1.0;
                }
                (data.queries[&pair.qid].clone(), vector)
            })
            .collect();
        TableEmbedder {
            documents,
            queries,
            dims,
        }
    }

    fn constant_embedder() -> TableEmbedder {
        TableEmbedder {
            documents: BTreeMap::new(),
            queries: BTreeMap::new(),
            dims: CONSTANT_VECTOR_DIMS,
        }
    }

    // Data

    #[test]
    fn every_language_file_parses_and_names_its_language() {
        for language in Language::ALL {
            // Act
            let data = load_language(language).unwrap();

            // Assert
            assert_eq!(data.language, language);
            assert_eq!(data.machine_translated, language != Language::En);
        }
    }

    #[test]
    fn every_language_has_the_ids_of_pairs_json() {
        // Arrange
        let pairs = load_pairs().unwrap();
        assert_eq!(pairs.len(), QUERY_COUNT_PER_LANGUAGE);
        let reference = load_language(Language::En).unwrap();

        for language in Language::ALL {
            // Act
            let data = load_language(language).unwrap();

            // Assert
            assert!(data.facts.keys().eq(reference.facts.keys()), "{language:?}");
            assert_eq!(data.facts.len(), FACTS_PER_LANGUAGE, "{language:?}");
            let query_ids: HashSet<&String> = data.queries.keys().collect();
            let pair_ids: HashSet<&String> = pairs.iter().map(|p| &p.qid).collect();
            assert_eq!(query_ids, pair_ids, "{language:?}");
        }
    }

    #[test]
    fn every_relevant_id_exists_and_every_query_has_one() {
        // Arrange
        let pairs = load_pairs().unwrap();
        let data = load_language(Language::En).unwrap();

        // Act / Assert
        for pair in &pairs {
            assert!(!pair.relevant.is_empty(), "{}", pair.qid);
            for id in &pair.relevant {
                assert!(
                    data.facts.contains_key(id),
                    "{} lists unknown {id}",
                    pair.qid
                );
            }
        }
    }

    #[test]
    fn no_text_is_empty() {
        for language in Language::ALL {
            // Act
            let data = load_language(language).unwrap();

            // Assert
            let texts = data.facts.values().chain(data.queries.values());
            assert!(
                texts.into_iter().all(|t| !t.trim().is_empty()),
                "{language:?}"
            );
        }
    }

    #[test]
    fn the_data_uses_no_em_or_en_dashes() {
        for language in Language::ALL {
            let raw = language.data_json();
            assert!(
                !raw.contains('\u{2013}') && !raw.contains('\u{2014}'),
                "{language:?}"
            );
        }
    }

    // Scorer

    #[tokio::test]
    async fn an_oracle_embedder_scores_one_in_every_language() {
        for language in Language::ALL {
            // Arrange
            let embedder = oracle_for(language);

            // Act
            let score = score_language(&embedder, language).await.unwrap();

            // Assert
            assert_eq!(score.per_query.len(), QUERY_COUNT_PER_LANGUAGE);
            assert!(
                (score.mean - 1.0).abs() < SCORE_EPSILON,
                "{language:?}: {}",
                score.mean
            );
        }
    }

    #[tokio::test]
    async fn a_constant_embedder_scores_low() {
        // Arrange
        let embedder = constant_embedder();

        // Act
        let score = score_language(&embedder, Language::En).await.unwrap();

        // Assert
        assert!(score.mean < CONSTANT_SCORE_CEILING, "{}", score.mean);
    }

    // Gate

    #[test]
    fn the_gate_passes_at_the_baseline() {
        // Act
        let outcome = gate(&uniform(BASE_SCORE), &uniform(BASE_SCORE));

        // Assert
        assert!(outcome.passed(), "{:?}", outcome.reasons);
    }

    #[test]
    fn the_gate_passes_a_drop_of_exactly_the_tolerance() {
        // Arrange
        let measured = with(
            uniform(BASE_SCORE),
            Language::Pt,
            BASE_SCORE - LANGUAGE_TOLERANCE,
        );

        // Act
        let outcome = gate(&uniform(BASE_SCORE), &measured);

        // Assert: the mean drops by less than its own tolerance.
        assert!(outcome.passed(), "{:?}", outcome.reasons);
    }

    #[test]
    fn the_gate_fails_on_a_score_that_is_not_a_number() {
        for language in Language::GATING {
            // Arrange
            let measured = with(uniform(BASE_SCORE), language, f64::NAN);

            // Act
            let outcome = gate(&uniform(BASE_SCORE), &measured);

            // Assert
            assert!(!outcome.passed(), "{language:?}");
            assert!(
                outcome.reasons[0].starts_with(language.code()),
                "{:?}",
                outcome.reasons
            );
        }
    }

    #[test]
    fn the_gate_passes_a_mean_drop_of_exactly_the_mean_tolerance() {
        // Arrange: every language drops by the mean tolerance, below its own.
        let mut measured = uniform(BASE_SCORE);
        for language in Language::GATING {
            measured.insert(language, BASE_SCORE - MEAN_TOLERANCE);
        }

        // Act
        let outcome = gate(&uniform(BASE_SCORE), &measured);

        // Assert
        assert!(outcome.passed(), "{:?}", outcome.reasons);
    }

    #[test]
    fn the_gate_fails_when_one_gating_language_drops_past_the_tolerance() {
        for language in Language::GATING {
            // Arrange
            let dropped = BASE_SCORE - LANGUAGE_TOLERANCE - 0.01;
            let measured = with(uniform(BASE_SCORE), language, dropped);

            // Act
            let outcome = gate(&uniform(BASE_SCORE), &measured);

            // Assert
            assert!(!outcome.passed(), "{language:?}");
            assert!(
                outcome.reasons[0].starts_with(language.code()),
                "{:?}",
                outcome.reasons
            );
        }
    }

    #[test]
    fn the_gate_ignores_a_non_gating_language_that_collapses() {
        for language in Language::ALL.into_iter().filter(|l| !l.is_gating()) {
            // Arrange
            let measured = with(uniform(BASE_SCORE), language, 0.0);

            // Act
            let outcome = gate(&uniform(BASE_SCORE), &measured);

            // Assert
            assert!(outcome.passed(), "{language:?}: {:?}", outcome.reasons);
        }
    }

    #[test]
    fn the_gate_fails_when_the_gating_mean_falls_past_its_tolerance() {
        // Arrange: each language is inside its own tolerance, the mean is not.
        let drop = MEAN_TOLERANCE + 0.01;
        let mut measured = uniform(BASE_SCORE);
        for language in Language::GATING {
            measured.insert(language, BASE_SCORE - drop);
        }

        // Act
        let outcome = gate(&uniform(BASE_SCORE), &measured);

        // Assert
        assert_eq!(outcome.reasons.len(), 1, "{:?}", outcome.reasons);
        assert!(
            outcome.reasons[0].starts_with("mean"),
            "{:?}",
            outcome.reasons
        );
    }

    #[test]
    fn the_gate_fails_when_a_gating_language_has_no_score() {
        // Arrange
        let mut measured = uniform(BASE_SCORE);
        measured.remove(&Language::Es);

        // Act
        let outcome = gate(&uniform(BASE_SCORE), &measured);

        // Assert
        assert!(!outcome.passed());
    }

    // Baseline

    #[test]
    fn the_committed_baseline_covers_all_eight_languages() {
        // Act
        let baseline = Baseline::load().unwrap();

        // Assert
        assert!(Language::ALL
            .iter()
            .all(|l| baseline.ndcg_at_10.contains_key(l)));
    }

    #[test]
    fn a_baseline_for_another_embedder_or_width_is_refused_with_a_re_measure_message() {
        // Arrange
        let baseline = Baseline {
            embedder: "model-a".into(),
            dims: 384,
            ndcg_at_10: uniform(BASE_SCORE),
        };

        // Act
        let other_model = baseline.check_matches("model-b", 384).unwrap_err();
        let other_dims = baseline.check_matches("model-a", 1024).unwrap_err();

        // Assert
        assert!(other_model.contains("re-measure"), "{other_model}");
        assert!(other_dims.contains("re-measure"), "{other_dims}");
        assert!(baseline.check_matches("model-a", 384).is_ok());
    }

    // Real model

    const CACHE_DIR_ENV: &str = "LIAM_EVAL_CACHE_DIR";

    /// Scores all eight languages with the default embedder and applies the
    /// gate. The first run downloads the model, about 311 MB. Set
    /// `LIAM_EVAL_CACHE_DIR` to keep the files out of `embedder.cache_dir`.
    /// Run with `cargo test --release -p liam-daemon --lib -- --ignored
    /// --nocapture multilingual::tests::the_default_embedder_meets_the_multilingual_baseline`.
    /// To re-measure after a deliberate change, run it and copy the printed
    /// `ndcg@10` column into `multilingual/baseline-arctic-embed-m-v2-384.json`.
    #[tokio::test]
    #[ignore = "downloads embedder weights; see the doc comment for the run command"]
    async fn the_default_embedder_meets_the_multilingual_baseline() {
        // Arrange
        let mut config = crate::config::Config::default();
        config.embedder.provider = crate::models::LOCAL_PROVIDER.into();
        if let Some(dir) = std::env::var_os(CACHE_DIR_ENV) {
            config.embedder.cache_dir = dir
                .into_string()
                .unwrap_or_else(|raw| panic!("{CACHE_DIR_ENV} is not valid UTF-8: {raw:?}"));
        }
        let baseline = Baseline::load().unwrap();
        if let Err(message) = baseline.check_matches(&config.embedder.model, config.embedding_dims)
        {
            panic!("{message}");
        }
        let embedder = crate::models::build_embedder(&config).expect("load the real embedder");

        // Act
        let mut measured = BTreeMap::new();
        for language in Language::ALL {
            let score = score_language(embedder.as_ref(), language).await.unwrap();
            measured.insert(language, score.mean);
        }
        println!("{}", render_table(&baseline.ndcg_at_10, &measured));
        let outcome = gate(&baseline.ndcg_at_10, &measured);

        // Assert
        assert!(outcome.passed(), "{}", outcome.reasons.join("\n"));
    }
}
