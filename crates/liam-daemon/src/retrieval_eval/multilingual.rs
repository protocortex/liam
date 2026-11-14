// SPDX-License-Identifier: Apache-2.0
//! Multilingual reranker quality set, report and regression gate.
//!
//! The set holds 30 queries, each with 32 candidate facts and at least one
//! relevant fact, in eight languages. `pairs.json` is language independent;
//! each language file maps the same query and fact ids to text. English is the
//! source and the other seven are machine translated, so a gap between
//! languages is a signal, not a measurement of translation quality.
//!
//! A reranker run produces a [`ScoreFile`]: one score per candidate, aligned
//! with the canonical candidate order in `pairs.json`. [`Report::from_scores`]
//! turns it into per-language nDCG@10, and [`gate`] compares a candidate report
//! against the baseline report.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _};
use serde::{Deserialize, Serialize};

use super::ndcg_at_10;

const PAIRS_JSON: &str = include_str!("multilingual/pairs.json");

/// Language codes in report order, each with its embedded data file.
const LANGUAGE_FILES: [(&str, &str); 8] = [
    ("en", include_str!("multilingual/en.json")),
    ("pt", include_str!("multilingual/pt.json")),
    ("es", include_str!("multilingual/es.json")),
    ("zh", include_str!("multilingual/zh.json")),
    ("fr", include_str!("multilingual/fr.json")),
    ("de", include_str!("multilingual/de.json")),
    ("ja", include_str!("multilingual/ja.json")),
    ("ar", include_str!("multilingual/ar.json")),
];

pub const QUERIES_PER_LANGUAGE: usize = 30;
pub const CANDIDATES_PER_QUERY: usize = 32;

/// Model whose scores form the baseline.
pub const DEFAULT_RERANKER_MODEL: &str = "bge-reranker-base";
const DATA_DIR: &str = "src/retrieval_eval/multilingual";

/// Scores are compared after rounding to this many thousandths of a point.
const MILLI: f64 = 1000.0;
/// The mean nDCG@10 may fall by at most this much, in thousandths.
const MAX_MEAN_DROP_MILLI: i64 = 20;
/// No single language may fall by more than this, in thousandths.
const MAX_LANGUAGE_DROP_MILLI: i64 = 50;

/// Text shown for a language that has no scores.
const MISSING: &str = "missing";
/// Reason text for a language the baseline does not cover.
const NO_BASELINE: &str = "no baseline";

#[derive(Debug, Clone, Deserialize)]
pub struct Pair {
    pub qid: String,
    pub category: String,
    pub relevant: Vec<String>,
    pub candidates: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PairsFile {
    pairs: Vec<Pair>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LanguageData {
    pub language: String,
    pub machine_translated: bool,
    pub queries: BTreeMap<String, String>,
    pub facts: BTreeMap<String, String>,
}

/// The shared pairs and the text of every query and fact in each language.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub pairs: Vec<Pair>,
    pub languages: Vec<LanguageData>,
}

impl Dataset {
    /// Parses the embedded files and checks them with [`Dataset::validate`].
    pub fn load() -> anyhow::Result<Self> {
        let pairs: PairsFile = serde_json::from_str(PAIRS_JSON).context("parsing pairs.json")?;
        let languages = LANGUAGE_FILES
            .iter()
            .map(|(code, raw)| {
                serde_json::from_str::<LanguageData>(raw)
                    .with_context(|| format!("parsing {code}.json"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let dataset = Self {
            pairs: pairs.pairs,
            languages,
        };
        dataset.validate()?;
        Ok(dataset)
    }

    /// Every language has all queries, every query has the full candidate list
    /// and a relevant fact, and every id used has text in every language.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.pairs.len() != QUERIES_PER_LANGUAGE {
            bail!(
                "expected {QUERIES_PER_LANGUAGE} queries, found {}",
                self.pairs.len()
            );
        }
        for pair in &self.pairs {
            if pair.candidates.len() != CANDIDATES_PER_QUERY {
                bail!(
                    "query {} has {} candidates, expected {CANDIDATES_PER_QUERY}",
                    pair.qid,
                    pair.candidates.len()
                );
            }
            let mut seen = HashSet::new();
            if let Some(repeat) = pair.candidates.iter().find(|c| !seen.insert(*c)) {
                bail!(
                    "query {} lists candidate {repeat} twice (duplicate)",
                    pair.qid
                );
            }
            if pair.relevant.is_empty() {
                bail!("query {} has no relevant fact", pair.qid);
            }
            if let Some(stray) = pair.relevant.iter().find(|r| !pair.candidates.contains(r)) {
                bail!(
                    "query {} lists relevant {stray} outside its candidates",
                    pair.qid
                );
            }
        }
        for lang in &self.languages {
            if lang.queries.len() != QUERIES_PER_LANGUAGE {
                bail!(
                    "{} has {} queries, expected {QUERIES_PER_LANGUAGE}",
                    lang.language,
                    lang.queries.len()
                );
            }
            for pair in &self.pairs {
                if !lang.queries.contains_key(&pair.qid) {
                    bail!("{} has no text for query {}", lang.language, pair.qid);
                }
                if let Some(id) = pair
                    .candidates
                    .iter()
                    .find(|c| !lang.facts.contains_key(*c))
                {
                    bail!("{} has no text for fact {id}", lang.language);
                }
            }
        }
        Ok(())
    }
}

/// The scores one model gave: language, then query id, then one score per
/// candidate in the canonical order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreFile {
    pub model: String,
    pub scores: BTreeMap<String, BTreeMap<String, Vec<f32>>>,
}

impl ScoreFile {
    pub fn from_json(raw: &str) -> anyhow::Result<Self> {
        serde_json::from_str(raw).context("parsing score file")
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self).context("serialising score file")
    }
}

/// Loads the committed baseline score file for the default reranker.
pub fn load_baseline() -> anyhow::Result<ScoreFile> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(DATA_DIR)
        .join(format!("baseline-{DEFAULT_RERANKER_MODEL}.json"));
    load_baseline_from(&path)
}

/// Loads a baseline score file, or an error naming the file that failed.
pub fn load_baseline_from(path: &Path) -> anyhow::Result<ScoreFile> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading baseline score file {}", path.display()))?;
    ScoreFile::from_json(&raw).with_context(|| format!("in {}", path.display()))
}

/// Candidate ids ordered by descending score. Equal scores keep the canonical
/// order, so a model that cannot tell candidates apart gets no credit for luck.
pub fn rank_candidates(candidates: &[String], scores: &[f32]) -> anyhow::Result<Vec<String>> {
    if scores.len() != candidates.len() {
        bail!(
            "expected {} scores, found {}",
            candidates.len(),
            scores.len()
        );
    }
    if scores.iter().any(|s| s.is_nan()) {
        bail!("scores contain NaN");
    }
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    // NaN is rejected above, so `partial_cmp` always orders; unlike
    // `total_cmp` it treats -0.0 and 0.0 as a tie.
    order.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(Ordering::Equal)
    });
    Ok(order.into_iter().map(|i| candidates[i].clone()).collect())
}

/// nDCG@10 of one query's scores against its relevant set.
pub fn query_ndcg(pair: &Pair, scores: &[f32]) -> anyhow::Result<f64> {
    let ranked =
        rank_candidates(&pair.candidates, scores).with_context(|| format!("query {}", pair.qid))?;
    let relevant: HashSet<&str> = pair.relevant.iter().map(String::as_str).collect();
    Ok(ndcg_at_10(&ranked, &relevant))
}

/// nDCG@10 per language, in dataset order, and the mean over the languages
/// that have scores. A language without scores is `None`, never `0.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub per_language: Vec<(String, Option<f64>)>,
    pub mean: Option<f64>,
}

impl Report {
    pub fn new(per_language: Vec<(String, Option<f64>)>) -> Self {
        let present: Vec<f64> = per_language.iter().filter_map(|(_, v)| *v).collect();
        let mean =
            (!present.is_empty()).then(|| present.iter().sum::<f64>() / present.len() as f64);
        Self { per_language, mean }
    }

    pub fn from_scores(dataset: &Dataset, file: &ScoreFile) -> anyhow::Result<Self> {
        if let Some(unknown) = file
            .scores
            .keys()
            .find(|code| !dataset.languages.iter().any(|l| &l.language == *code))
        {
            bail!("score file has unknown language {unknown}");
        }
        for (code, by_query) in &file.scores {
            if let Some(stray) = by_query
                .keys()
                .find(|qid| !dataset.pairs.iter().any(|p| &p.qid == *qid))
            {
                bail!("score file has unknown query {stray} for language {code}");
            }
        }
        let per_language = dataset
            .languages
            .iter()
            .map(|lang| {
                let value = match file.scores.get(&lang.language) {
                    Some(by_query) => Some(
                        language_mean(&dataset.pairs, by_query)
                            .with_context(|| format!("language {}", lang.language))?,
                    ),
                    None => None,
                };
                Ok((lang.language.clone(), value))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self::new(per_language))
    }

    /// Codes of the languages that have a value.
    fn covered(&self) -> BTreeSet<&str> {
        self.per_language
            .iter()
            .filter(|(_, v)| v.is_some())
            .map(|(code, _)| code.as_str())
            .collect()
    }

    fn value(&self, language: &str) -> Option<f64> {
        self.per_language
            .iter()
            .find(|(code, _)| code == language)
            .and_then(|(_, v)| *v)
    }
}

fn language_mean(pairs: &[Pair], by_query: &BTreeMap<String, Vec<f32>>) -> anyhow::Result<f64> {
    let mut total = 0.0;
    for pair in pairs {
        let scores = by_query
            .get(&pair.qid)
            .with_context(|| format!("no scores for query {}", pair.qid))?;
        total += query_ndcg(pair, scores)?;
    }
    Ok(total / pairs.len() as f64)
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "language  ndcg@10")?;
        for (code, value) in &self.per_language {
            writeln!(f, "{code:<8}  {}", cell(*value))?;
        }
        write!(f, "{:<8}  {}", "mean", cell(self.mean))
    }
}

fn cell(value: Option<f64>) -> String {
    value.map_or_else(|| MISSING.to_string(), |v| format!("{v:.3}"))
}

/// A value in whole thousandths, so the comparison is exact.
fn milli(value: f64) -> i64 {
    (value * MILLI).round() as i64
}

/// The gate's verdict. `reasons` lists every failure; it is empty on a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateOutcome {
    pub passed: bool,
    pub reasons: Vec<String>,
}

impl fmt::Display for GateOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed {
            return write!(f, "pass");
        }
        write!(f, "fail")?;
        for reason in &self.reasons {
            write!(f, "\n- {reason}")?;
        }
        Ok(())
    }
}

/// Passes only when the mean drop is within [`MAX_MEAN_DROP_MILLI`] and no
/// language drops by more than [`MAX_LANGUAGE_DROP_MILLI`]. A language with no
/// baseline cannot be judged, so it fails the gate.
pub fn gate(baseline: &Report, candidate: &Report) -> GateOutcome {
    let mut reasons = Vec::new();
    match (baseline.mean, candidate.mean) {
        (None, _) => reasons.push(format!("mean: {NO_BASELINE}")),
        (_, None) => reasons.push(format!("mean: candidate {MISSING}")),
        (Some(_), Some(_)) if baseline.covered() != candidate.covered() => {
            reasons.push("mean: baseline and candidate cover different languages".to_string());
        }
        (Some(base), Some(cand)) => {
            let drop = milli(base - cand);
            if drop > MAX_MEAN_DROP_MILLI {
                reasons.push(format!(
                    "mean fell {drop} thousandths, the limit is {MAX_MEAN_DROP_MILLI}"
                ));
            }
        }
    }
    let codes: BTreeSet<&str> = baseline
        .per_language
        .iter()
        .chain(&candidate.per_language)
        .map(|(code, _)| code.as_str())
        .collect();
    for code in codes {
        match (baseline.value(code), candidate.value(code)) {
            (None, _) => reasons.push(format!("{code}: {NO_BASELINE}")),
            (Some(_), None) => reasons.push(format!("{code}: candidate {MISSING}")),
            (Some(base), Some(cand)) => {
                let drop = milli(base - cand);
                if drop > MAX_LANGUAGE_DROP_MILLI {
                    reasons.push(format!(
                        "{code}: fell {drop} thousandths, the limit is {MAX_LANGUAGE_DROP_MILLI}"
                    ));
                }
            }
        }
    }
    GateOutcome {
        passed: reasons.is_empty(),
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use liam_model::{FastEmbedReranker, Reranker};

    use super::*;

    const SCORES_OUT_ENV: &str = "LIAM_MULTILINGUAL_SCORES_OUT";
    const TOLERANCE: f64 = 0.0005;
    const STUB_MODEL: &str = "stub";
    const ABSENT_BASELINE_PATH: &str = "/nonexistent-liam-dir/no-such-baseline.json";

    fn ids(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn pair(qid: &str, relevant: &[&str], candidates: &[&str]) -> Pair {
        Pair {
            qid: qid.into(),
            category: "test".into(),
            relevant: ids(relevant),
            candidates: ids(candidates),
        }
    }

    fn language(code: &str) -> LanguageData {
        LanguageData {
            language: code.into(),
            machine_translated: false,
            queries: BTreeMap::new(),
            facts: BTreeMap::new(),
        }
    }

    fn small_dataset() -> Dataset {
        Dataset {
            pairs: vec![
                pair("q1", &["c2"], &["c0", "c1", "c2", "c3"]),
                pair("q2", &["c1", "c3"], &["c0", "c1", "c2", "c3"]),
            ],
            languages: vec![language("en"), language("pt")],
        }
    }

    type QueryScores<'a> = &'a [(&'a str, &'a [f32])];

    fn score_file(entries: &[(&str, QueryScores)]) -> ScoreFile {
        let scores = entries
            .iter()
            .map(|(code, queries)| {
                let by_query = queries
                    .iter()
                    .map(|(qid, s)| (qid.to_string(), s.to_vec()))
                    .collect();
                (code.to_string(), by_query)
            })
            .collect();
        ScoreFile {
            model: STUB_MODEL.into(),
            scores,
        }
    }

    fn report(values: &[(&str, Option<f64>)]) -> Report {
        Report::new(values.iter().map(|(c, v)| (c.to_string(), *v)).collect())
    }

    fn uniform(value: f64) -> Report {
        Report::new(
            LANGUAGE_FILES
                .iter()
                .map(|(code, _)| (code.to_string(), Some(value)))
                .collect(),
        )
    }

    /// `uniform(base)` with one language moved to `value`.
    fn with_language(base: f64, code: &str, value: f64) -> Report {
        let mut r = uniform(base);
        for (c, v) in &mut r.per_language {
            if c == code {
                *v = Some(value);
            }
        }
        Report::new(r.per_language)
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("a value");
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "expected {expected}, got {actual}"
        );
    }

    // Data

    #[test]
    fn the_embedded_set_loads_and_validates() {
        // Act
        let dataset = Dataset::load().expect("embedded data is valid");

        // Assert
        assert_eq!(dataset.pairs.len(), QUERIES_PER_LANGUAGE);
        let codes: Vec<&str> = dataset
            .languages
            .iter()
            .map(|l| l.language.as_str())
            .collect();
        assert_eq!(codes, ["en", "pt", "es", "zh", "fr", "de", "ja", "ar"]);
        for pair in &dataset.pairs {
            assert_eq!(pair.candidates.len(), CANDIDATES_PER_QUERY);
            assert!(!pair.relevant.is_empty());
        }
    }

    #[test]
    fn only_english_is_flagged_as_not_machine_translated() {
        // Act
        let dataset = Dataset::load().expect("embedded data is valid");

        // Assert
        for lang in &dataset.languages {
            assert_eq!(
                lang.machine_translated,
                lang.language != "en",
                "{}",
                lang.language
            );
        }
    }

    #[test]
    fn validation_rejects_a_candidate_with_no_text() {
        // Arrange
        let mut dataset = Dataset::load().expect("embedded data is valid");
        let victim = dataset.pairs[0].candidates[0].clone();
        dataset.languages[3].facts.remove(&victim);

        // Act
        let error = dataset.validate().expect_err("missing fact text");

        // Assert
        let message = error.to_string();
        assert!(
            message.contains("zh") && message.contains(&victim),
            "{message}"
        );
    }

    // Report

    #[test]
    fn ranking_sorts_by_descending_score() {
        // Arrange
        let candidates = ids(&["c0", "c1", "c2", "c3"]);

        // Act
        let ranked = rank_candidates(&candidates, &[0.0, 0.5, 0.0, 1.0]).expect("ranks");

        // Assert
        assert_eq!(ranked, ids(&["c3", "c1", "c0", "c2"]));
    }

    #[test]
    fn ranking_keeps_the_canonical_order_for_equal_scores() {
        // Arrange
        let candidates = ids(&["c0", "c1", "c2", "c3"]);

        // Act
        let ranked = rank_candidates(&candidates, &[0.5; 4]).expect("ranks");

        // Assert
        assert_eq!(ranked, candidates);
    }

    #[test]
    fn the_report_gives_hand_computed_per_language_and_mean_values() {
        // Arrange
        let dataset = small_dataset();
        let file = score_file(&[
            (
                "en",
                &[("q1", &[0.0, 0.0, 1.0, 0.0]), ("q2", &[0.0, 1.0, 0.0, 1.0])],
            ),
            ("pt", &[("q1", &[0.0; 4]), ("q2", &[0.0; 4])]),
        ]);

        // Act
        let report = Report::from_scores(&dataset, &file).expect("report");

        // Assert: pt ties keep canonical order. q1 puts c2 third: 1/log2(4) = 0.5.
        // q2 puts c1 second and c3 fourth:
        // (1/log2(3) + 1/log2(5)) / (1 + 1/log2(3)) = 0.6509.
        assert_close(report.value("en"), 1.0);
        assert_close(report.value("pt"), 0.5755);
        assert_close(report.mean, 0.7877);
    }

    #[test]
    fn a_language_without_scores_shows_as_missing_not_zero() {
        // Arrange
        let dataset = small_dataset();
        let file = score_file(&[(
            "en",
            &[("q1", &[0.0, 0.0, 1.0, 0.0]), ("q2", &[0.0, 1.0, 0.0, 1.0])],
        )]);

        // Act
        let report = Report::from_scores(&dataset, &file).expect("report");
        let text = report.to_string();

        // Assert
        assert_eq!(report.value("pt"), None);
        assert_close(report.mean, 1.0);
        assert!(text.contains("pt") && text.contains(MISSING), "{text}");
        assert!(text.contains("1.000"), "{text}");
    }

    // Score file

    #[test]
    fn a_score_file_survives_a_serialise_and_parse_round_trip() {
        // Arrange
        let file = score_file(&[("en", &[("q1", &[0.25, -1.5, 3.0, 0.0])])]);

        // Act
        let parsed = ScoreFile::from_json(&file.to_json().expect("serialises")).expect("parses");

        // Assert
        assert_eq!(parsed, file);
    }

    #[test]
    fn malformed_json_is_an_error() {
        // Act
        let result = ScoreFile::from_json("{\"model\": 3}");

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn a_short_score_list_is_an_error_naming_the_query() {
        // Arrange
        let dataset = small_dataset();
        let file = score_file(&[("en", &[("q1", &[0.0, 1.0]), ("q2", &[0.0, 1.0, 0.0, 1.0])])]);

        // Act
        let error = Report::from_scores(&dataset, &file).expect_err("short list");

        // Assert
        let message = format!("{error:#}");
        assert!(
            message.contains("q1") && message.contains("expected 4"),
            "{message}"
        );
    }

    #[test]
    fn a_missing_query_or_unknown_language_is_an_error() {
        // Arrange
        let dataset = small_dataset();
        let missing_query = score_file(&[("en", &[("q1", &[0.0; 4])])]);
        let unknown = score_file(&[("xx", &[])]);

        // Act
        let missing = Report::from_scores(&dataset, &missing_query).expect_err("no q2");
        let stray = Report::from_scores(&dataset, &unknown).expect_err("unknown language");

        // Assert
        assert!(format!("{missing:#}").contains("q2"));
        assert!(stray.to_string().contains("xx"));
    }

    #[test]
    fn the_missing_baseline_error_names_the_file() {
        // Act
        let error = load_baseline_from(Path::new(ABSENT_BASELINE_PATH)).expect_err("no file");

        // Assert
        assert!(
            format!("{error:#}").contains("no-such-baseline.json"),
            "{error:#}"
        );
    }

    // Gate

    #[test]
    fn a_mean_drop_of_twenty_thousandths_passes_and_twenty_one_fails() {
        // Arrange
        let baseline = uniform(0.800);

        // Act
        let at_limit = gate(&baseline, &uniform(0.780));
        let over_limit = gate(&baseline, &uniform(0.779));

        // Assert
        assert!(at_limit.passed, "{at_limit}");
        assert!(!over_limit.passed);
        assert!(over_limit.reasons.iter().any(|r| r.starts_with("mean")));
    }

    #[test]
    fn a_language_drop_of_fifty_thousandths_passes_and_fifty_one_fails() {
        // Arrange
        let baseline = uniform(0.800);

        // Act
        let at_limit = gate(&baseline, &with_language(0.800, "ja", 0.750));
        let over_limit = gate(&baseline, &with_language(0.800, "ja", 0.749));

        // Assert
        assert!(at_limit.passed, "{at_limit}");
        assert!(!over_limit.passed);
        assert!(over_limit.reasons.iter().any(|r| r.starts_with("ja")));
    }

    #[test]
    fn a_passing_mean_does_not_rescue_one_language_that_fell_too_far() {
        // Arrange
        let baseline = uniform(0.800);
        let candidate = with_language(0.800, "ar", 0.749);

        // Act
        let outcome = gate(&baseline, &candidate);

        // Assert
        assert!(
            milli(baseline.mean.unwrap()) - milli(candidate.mean.unwrap()) <= MAX_MEAN_DROP_MILLI
        );
        assert!(!outcome.passed);
        assert_eq!(outcome.reasons.len(), 1, "{outcome}");
    }

    #[test]
    fn a_language_with_no_baseline_fails_the_gate_with_no_baseline_text() {
        // Arrange
        let baseline = report(&[("en", Some(0.8)), ("pt", None)]);
        let candidate = report(&[("en", Some(0.8)), ("pt", Some(0.8))]);

        // Act
        let outcome = gate(&baseline, &candidate);

        // Assert
        assert!(!outcome.passed);
        assert!(
            outcome.reasons.iter().any(|r| r == "pt: no baseline"),
            "{outcome}"
        );
        assert!(outcome.to_string().contains("no baseline"));
    }

    #[test]
    fn rounding_to_three_decimals_decides_a_borderline_drop() {
        // Arrange: raw drop is 0.0204, which rounds to 0.020 and passes.
        let baseline = uniform(0.8004);
        let candidate = uniform(0.7800);

        // Act
        let outcome = gate(&baseline, &candidate);

        // Assert
        assert!(outcome.passed, "{outcome}");
    }

    #[test]
    fn a_baseline_language_absent_from_the_candidate_fails() {
        // Arrange
        let baseline = report(&[("en", Some(0.8)), ("pt", Some(0.8))]);
        let candidate = report(&[("en", Some(0.8))]);

        // Act
        let outcome = gate(&baseline, &candidate);

        // Assert
        assert!(!outcome.passed);
        assert!(
            outcome.reasons.iter().any(|r| r == "pt: candidate missing"),
            "{outcome}"
        );
    }

    #[test]
    fn a_baseline_value_with_no_candidate_value_fails() {
        // Arrange
        let baseline = report(&[("en", Some(0.8)), ("pt", Some(0.8))]);
        let candidate = report(&[("en", Some(0.8)), ("pt", None)]);

        // Act
        let outcome = gate(&baseline, &candidate);

        // Assert
        assert!(!outcome.passed);
        assert!(
            outcome.reasons.iter().any(|r| r == "pt: candidate missing"),
            "{outcome}"
        );
    }

    #[test]
    fn a_missing_mean_on_either_side_fails() {
        // Arrange
        let full = uniform(0.8);
        let no_mean = Report {
            mean: None,
            ..uniform(0.8)
        };

        // Act
        let baseline_less = gate(&no_mean, &full);
        let candidate_less = gate(&full, &no_mean);

        // Assert
        assert!(baseline_less
            .reasons
            .contains(&"mean: no baseline".to_string()));
        assert!(candidate_less
            .reasons
            .contains(&"mean: candidate missing".to_string()));
        assert!(!baseline_less.passed && !candidate_less.passed);
    }

    #[test]
    fn a_mean_over_different_language_sets_fails_with_one_mean_reason() {
        // Arrange
        let baseline = report(&[("en", Some(0.8)), ("pt", Some(0.8))]);
        let candidate = report(&[("en", Some(0.8)), ("pt", None)]);

        // Act
        let outcome = gate(&baseline, &candidate);

        // Assert
        let mean_reasons = outcome
            .reasons
            .iter()
            .filter(|r| r.starts_with("mean"))
            .count();
        assert_eq!(mean_reasons, 1, "{outcome}");
    }

    #[test]
    fn a_drop_that_rounds_to_twenty_one_fails_even_when_the_rounded_values_differ_by_twenty() {
        // Arrange: raw drop is 0.0208, but 0.800 and 0.780 differ by 0.020.
        let baseline = uniform(0.8004);
        let candidate = uniform(0.7796);

        // Act
        let outcome = gate(&baseline, &candidate);

        // Assert
        assert!(!outcome.passed);
    }

    #[test]
    fn float_noise_does_not_flip_a_drop_of_twenty_thousandths() {
        // Arrange
        let cases = [(0.9, 0.88), (0.5, 0.48)];

        for (base, cand) in cases {
            // Act
            let outcome = gate(&uniform(base), &uniform(cand));

            // Assert
            assert!(outcome.passed, "{base} vs {cand}: {outcome}");
        }
    }

    #[test]
    fn negative_and_positive_zero_tie_in_canonical_order() {
        // Arrange
        let candidates = ids(&["c0", "c1"]);

        // Act
        let ranked = rank_candidates(&candidates, &[-0.0, 0.0]).expect("ranks");

        // Assert
        assert_eq!(ranked, candidates);
    }

    #[test]
    fn validation_rejects_duplicate_candidates_within_a_pair() {
        // Arrange
        let mut dataset = Dataset::load().expect("embedded data is valid");
        let first = dataset.pairs[0].candidates[0].clone();
        dataset.pairs[0].candidates[1] = first.clone();

        // Act
        let error = dataset.validate().expect_err("duplicate candidate");

        // Assert
        let message = error.to_string();
        assert!(
            message.contains("duplicate") && message.contains(&first),
            "{message}"
        );
    }

    #[test]
    fn a_score_file_with_a_query_outside_the_dataset_is_an_error() {
        // Arrange
        let dataset = small_dataset();
        let file = score_file(&[(
            "en",
            &[("q1", &[0.0; 4]), ("q2", &[0.0; 4]), ("q9", &[0.0; 4])],
        )]);

        // Act
        let error = Report::from_scores(&dataset, &file).expect_err("unknown query");

        // Assert
        assert!(format!("{error:#}").contains("q9"));
    }

    // Score producing harness

    /// Scores every query and candidate pair in every language with the
    /// default reranker and writes a score file to the path in
    /// `LIAM_MULTILINGUAL_SCORES_OUT`. It loads model weights, so it only runs
    /// on request:
    ///
    /// ```text
    /// LIAM_MULTILINGUAL_SCORES_OUT=/tmp/scores.json \
    ///   cargo test -p liam-daemon --release -- --ignored \
    ///   write_scores_for_the_default_reranker
    /// ```
    #[tokio::test]
    #[ignore = "loads reranker weights; see the doc comment for the run command"]
    async fn write_scores_for_the_default_reranker() {
        // Arrange
        let out = std::env::var(SCORES_OUT_ENV)
            .unwrap_or_else(|_| panic!("set {SCORES_OUT_ENV} to the score file path"));
        let dataset = Dataset::load().expect("embedded data is valid");
        let reranker = FastEmbedReranker::load().expect("default reranker loads");

        // Act
        let mut scores = BTreeMap::new();
        for lang in &dataset.languages {
            let mut by_query = BTreeMap::new();
            for pair in &dataset.pairs {
                let docs: Vec<String> = pair
                    .candidates
                    .iter()
                    .map(|id| lang.facts[id].clone())
                    .collect();
                let query_scores = reranker
                    .scores(&lang.queries[&pair.qid], &docs)
                    .await
                    .expect("reranker scores the candidates");
                by_query.insert(pair.qid.clone(), query_scores);
            }
            scores.insert(lang.language.clone(), by_query);
        }
        let file = ScoreFile {
            model: DEFAULT_RERANKER_MODEL.into(),
            scores,
        };
        std::fs::write(&out, file.to_json().expect("serialises")).expect("writes the score file");

        // Assert
        Report::from_scores(&dataset, &file).expect("the written scores form a report");
    }
}
