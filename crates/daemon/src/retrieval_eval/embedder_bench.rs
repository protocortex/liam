// SPDX-License-Identifier: Apache-2.0
//! Embedder-only benchmark: vector retrieval quality and speed on a larger,
//! harder set than the golden set in `golden_set.rs`.
//!
//! The golden set runs the whole retrieval pipeline, so most of its categories
//! are decided by the text arm and score the same for any embedder. This set
//! takes the store and the text arm out: every fact is embedded, every question
//! is embedded as a query, and facts are ranked by cosine similarity. Whatever
//! separates two models here is the embedder.
//!
//! The data is two tab separated files, `facts.tsv` and `questions.tsv`, about
//! a fictional invoicing platform. Questions come in categories (paraphrase,
//! symbol, config, rationale, confusable, multi) and name one to three relevant
//! facts. The files are hand written; nothing in them comes from real users.

use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use liam_model::{Embedder, TextKind};
use serde::Serialize;

use super::{ndcg_at_10, recall_at_k, reciprocal_rank, MS_PER_SEC, WARM_UP_TEXT};
use crate::config::Config;

const FACTS_TSV: &str = include_str!("embedder_bench/facts.tsv");
const QUESTIONS_TSV: &str = include_str!("embedder_bench/questions.tsv");

/// Separates the columns of both data files.
const COLUMN_SEPARATOR: char = '\t';
/// Separates the relevant fact ids in the questions file.
const ID_SEPARATOR: char = ',';
/// The recall cutoffs reported besides the headline `recall@10`.
const RECALL_SMALL_K: usize = 1;
const RECALL_MID_K: usize = 5;
const RECALL_HEADLINE_K: usize = 10;
/// The latency percentile reported beside best, median and mean.
const LATENCY_PERCENTILE: f64 = 0.95;
const BYTES_PER_F32: usize = std::mem::size_of::<f32>();

#[derive(Debug, Clone, Copy)]
struct BenchFact {
    id: &'static str,
    text: &'static str,
}

#[derive(Debug, Clone)]
struct BenchQuestion {
    category: &'static str,
    text: &'static str,
    relevant: Vec<&'static str>,
}

/// Rows of a data file without its header line. A row with the wrong number of
/// columns is an error naming the line.
fn rows(raw: &'static str, columns: usize) -> Result<Vec<Vec<&'static str>>, String> {
    raw.lines()
        .enumerate()
        .skip(1)
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            let fields: Vec<&str> = line.split(COLUMN_SEPARATOR).collect();
            if fields.len() == columns {
                Ok(fields)
            } else {
                Err(format!(
                    "line {}: expected {columns} columns, found {}",
                    index + 1,
                    fields.len()
                ))
            }
        })
        .collect()
}

fn parse_facts(raw: &'static str) -> Result<Vec<BenchFact>, String> {
    Ok(rows(raw, 2)?
        .into_iter()
        .map(|f| BenchFact {
            id: f[0],
            text: f[1],
        })
        .collect())
}

fn parse_questions(raw: &'static str) -> Result<Vec<BenchQuestion>, String> {
    Ok(rows(raw, 4)?
        .into_iter()
        .map(|f| BenchQuestion {
            category: f[1],
            text: f[2],
            relevant: f[3].split(ID_SEPARATOR).collect(),
        })
        .collect())
}

/// The ranking metrics for one question, or the mean of several.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BenchMetrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
}

fn score(retrieved: &[String], relevant: &HashSet<&str>) -> BenchMetrics {
    BenchMetrics {
        recall_at_1: recall_at_k(retrieved, relevant, RECALL_SMALL_K),
        recall_at_5: recall_at_k(retrieved, relevant, RECALL_MID_K),
        recall_at_10: recall_at_k(retrieved, relevant, RECALL_HEADLINE_K),
        ndcg_at_10: ndcg_at_10(retrieved, relevant),
        mrr: reciprocal_rank(retrieved, relevant),
    }
}

fn mean(scores: &[BenchMetrics]) -> Option<BenchMetrics> {
    let n = scores.len() as f64;
    if scores.is_empty() {
        return None;
    }
    let avg = |pick: fn(&BenchMetrics) -> f64| scores.iter().map(pick).sum::<f64>() / n;
    Some(BenchMetrics {
        recall_at_1: avg(|m| m.recall_at_1),
        recall_at_5: avg(|m| m.recall_at_5),
        recall_at_10: avg(|m| m.recall_at_10),
        ndcg_at_10: avg(|m| m.ndcg_at_10),
        mrr: avg(|m| m.mrr),
    })
}

/// Latency of single-text query embeds, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LatencyStats {
    pub calls: usize,
    pub best_ms: f64,
    pub median_ms: f64,
    pub mean_ms: f64,
    pub p95_ms: f64,
}

fn latency_stats(calls: &[Duration]) -> Option<LatencyStats> {
    if calls.is_empty() {
        return None;
    }
    let mut ms: Vec<f64> = calls.iter().map(|d| d.as_secs_f64() * MS_PER_SEC).collect();
    ms.sort_by(f64::total_cmp);
    let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
    Some(LatencyStats {
        calls: ms.len(),
        best_ms: ms[0],
        median_ms: at(0.5),
        mean_ms: ms.iter().sum::<f64>() / ms.len() as f64,
        p95_ms: at(LATENCY_PERCENTILE),
    })
}

/// One model's result on the embedder benchmark. Timing fields are wall clock.
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub model_id: String,
    /// The engine and device that ran the model, as the embedder reports it.
    pub backend: String,
    /// The `embedder.device` value the run asked for.
    pub device: String,
    pub dims: usize,
    pub vector_bytes: usize,
    pub facts: usize,
    pub questions: usize,
    pub load_ms: f64,
    pub overall: Option<BenchMetrics>,
    pub per_category: BTreeMap<String, BenchMetrics>,
    pub query_latency: Option<LatencyStats>,
    pub batch_items: usize,
    pub batch_ms: f64,
    pub batch_items_per_sec: f64,
    /// Peak resident memory of the process so far, in MiB. It includes the
    /// loaded model and everything the run touched.
    pub peak_rss_mb: f64,
    /// Every fact vector then every question vector, for comparing runs.
    #[serde(skip)]
    pub vectors: Vec<Vec<f32>>,
}

impl BenchReport {
    /// The report as printable text.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;

        let mut out = String::new();
        // `write!` into a `String` never fails.
        let _ = writeln!(
            out,
            "model: {} on {} (device {}), {} dims, {} B per vector, {} facts, {} questions, \
             load {:.0} ms, peak RSS {:.0} MiB",
            self.model_id,
            self.backend,
            self.device,
            self.dims,
            self.vector_bytes,
            self.facts,
            self.questions,
            self.load_ms,
            self.peak_rss_mb
        );
        let line = |name: &str, m: &BenchMetrics| {
            format!(
                "{name:<12} recall@1 {:.3} @5 {:.3} @10 {:.3}  nDCG@10 {:.3}  MRR {:.3}\n",
                m.recall_at_1, m.recall_at_5, m.recall_at_10, m.ndcg_at_10, m.mrr
            )
        };
        for (category, metrics) in &self.per_category {
            out.push_str(&line(category, metrics));
        }
        if let Some(m) = &self.overall {
            out.push_str(&line("overall", m));
        }
        if let Some(l) = &self.query_latency {
            let _ = writeln!(
                out,
                "query embed: {} calls, best {:.1} ms, median {:.1} ms, mean {:.1} ms, p95 {:.1} ms",
                l.calls, l.best_ms, l.median_ms, l.mean_ms, l.p95_ms
            );
        }
        let _ = writeln!(
            out,
            "fact batch: {} items in {:.0} ms, {:.1} items/s",
            self.batch_items, self.batch_ms, self.batch_items_per_sec
        );
        out
    }
}

/// Cosine similarity, safe for vectors that are not unit length.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denominator = norm(a) * norm(b);
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}

/// Fact ids ordered best first. Equal scores keep file order, so the ranking
/// is deterministic.
fn rank(query: &[f32], facts: &[BenchFact], vectors: &[Vec<f32>]) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored
        .into_iter()
        .map(|(i, _)| facts[i].id.to_string())
        .collect()
}

/// Loads `model` (timing the load), then runs the benchmark with it.
///
/// # Errors
///
/// Fails when the model cannot be loaded, or on an embedder error.
pub async fn run_embedder_bench(
    model: super::EvalModel,
    config: &Config,
) -> anyhow::Result<BenchReport> {
    let started = Instant::now();
    let embedder = super::load_embedder(model, config)?;
    let mut report = bench_embedder(embedder.as_ref(), model.id(), started.elapsed()).await?;
    report.device = config.embedder.device.clone();
    Ok(report)
}

/// Bytes in a MiB.
const BYTES_PER_MIB: f64 = 1_048_576.0;

/// The most resident memory this process has used, in MiB. `ru_maxrss` is in
/// bytes on macOS and in KiB on Linux.
fn peak_rss_mb() -> f64 {
    // SAFETY: `getrusage` writes into the zeroed struct we own.
    let max_rss = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0.0;
        }
        usage.ru_maxrss as f64
    };
    if cfg!(target_os = "macos") {
        max_rss / BYTES_PER_MIB
    } else {
        max_rss / 1024.0
    }
}

/// Runs the benchmark over any `Embedder`. `load` is how long loading took.
///
/// # Errors
///
/// Fails on an embedder error or malformed data files.
pub async fn bench_embedder(
    embedder: &dyn Embedder,
    model_id: &str,
    load: Duration,
) -> anyhow::Result<BenchReport> {
    let facts = parse_facts(FACTS_TSV).map_err(anyhow::Error::msg)?;
    let questions = parse_questions(QUESTIONS_TSV).map_err(anyhow::Error::msg)?;
    embedder
        .embed(WARM_UP_TEXT, TextKind::Document)
        .await
        .context("warm-up embed")?;

    let texts: Vec<String> = facts.iter().map(|f| f.text.to_string()).collect();
    let started = Instant::now();
    let vectors = embedder
        .embed_batch(&texts, TextKind::Document)
        .await
        .context("embedding the facts")?;
    let batch = started.elapsed();

    let mut calls = Vec::with_capacity(questions.len());
    let mut by_category: BTreeMap<&str, Vec<BenchMetrics>> = BTreeMap::new();
    let mut all = Vec::with_capacity(questions.len());
    let mut query_vectors = Vec::with_capacity(questions.len());
    for question in &questions {
        let started = Instant::now();
        let query = embedder
            .embed(question.text, TextKind::Query)
            .await
            .with_context(|| format!("embedding query {:?}", question.text))?;
        calls.push(started.elapsed());
        let retrieved = rank(&query, &facts, &vectors);
        query_vectors.push(query);
        let relevant: HashSet<&str> = question.relevant.iter().copied().collect();
        let metrics = score(&retrieved, &relevant);
        by_category
            .entry(question.category)
            .or_default()
            .push(metrics);
        all.push(metrics);
    }

    let dims = embedder.dims();
    let mut vectors = vectors;
    vectors.extend(query_vectors);
    Ok(BenchReport {
        model_id: model_id.to_string(),
        backend: embedder.backend().to_string(),
        device: String::new(),
        peak_rss_mb: peak_rss_mb(),
        vectors,
        dims,
        vector_bytes: dims * BYTES_PER_F32,
        facts: facts.len(),
        questions: questions.len(),
        load_ms: load.as_secs_f64() * MS_PER_SEC,
        overall: mean(&all),
        per_category: by_category
            .into_iter()
            .filter_map(|(category, scores)| Some((category.to_string(), mean(&scores)?)))
            .collect(),
        query_latency: latency_stats(&calls),
        batch_items: texts.len(),
        batch_ms: batch.as_secs_f64() * MS_PER_SEC,
        batch_items_per_sec: if batch.is_zero() {
            0.0
        } else {
            texts.len() as f64 / batch.as_secs_f64()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use liam_model::MockEmbedder;

    const MOCK_DIMS: usize = 32;

    #[test]
    fn the_data_files_parse_and_are_consistent() {
        // Arrange
        let facts = parse_facts(FACTS_TSV).expect("facts parse");
        let questions = parse_questions(QUESTIONS_TSV).expect("questions parse");
        let ids: HashSet<&str> = facts.iter().map(|f| f.id).collect();

        // Act / Assert
        assert_eq!(ids.len(), facts.len(), "fact ids are unique");
        assert!(facts.len() >= 250 && questions.len() >= 300);
        for question in &questions {
            assert!(!question.relevant.is_empty(), "{}", question.text);
            for id in &question.relevant {
                assert!(ids.contains(id), "unknown fact {id} for {}", question.text);
            }
        }
    }

    #[test]
    fn the_data_files_use_no_em_or_en_dashes() {
        for raw in [FACTS_TSV, QUESTIONS_TSV] {
            assert!(!raw.contains('\u{2013}') && !raw.contains('\u{2014}'));
        }
    }

    #[test]
    fn a_row_with_the_wrong_column_count_names_its_line() {
        // Act
        let err = rows("id\ttext\nonly-one-column\n", 2).unwrap_err();

        // Assert
        assert!(
            err.contains("line 2") && err.contains("expected 2"),
            "{err}"
        );
    }

    #[test]
    fn rank_orders_by_cosine_and_keeps_file_order_on_ties() {
        // Arrange
        let facts = [
            BenchFact { id: "a", text: "" },
            BenchFact { id: "b", text: "" },
            BenchFact { id: "c", text: "" },
        ];
        let vectors = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 0.0]];

        // Act
        let order = rank(&[2.0, 0.0], &facts, &vectors);

        // Assert
        assert_eq!(order, ["b", "c", "a"]);
    }

    #[test]
    fn cosine_of_a_zero_vector_is_zero() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert!((cosine(&[3.0, 0.0], &[5.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn latency_stats_report_best_median_and_tail() {
        // Arrange
        let calls: Vec<Duration> = [30, 10, 20, 40, 50]
            .into_iter()
            .map(Duration::from_millis)
            .collect();

        // Act
        let stats = latency_stats(&calls).expect("some calls");

        // Assert
        assert_eq!(stats.calls, 5);
        assert!((stats.best_ms - 10.0).abs() < 1e-9);
        assert!((stats.median_ms - 30.0).abs() < 1e-9);
        assert!((stats.mean_ms - 30.0).abs() < 1e-9);
        assert!((stats.p95_ms - 50.0).abs() < 1e-9);
        assert!(latency_stats(&[]).is_none());
        assert!(mean(&[]).is_none());
    }

    #[tokio::test]
    async fn the_mock_embedder_run_is_deterministic_and_in_range() {
        // Arrange
        let embedder = MockEmbedder::new(MOCK_DIMS);

        // Act
        let first = bench_embedder(&embedder, "mock", Duration::from_millis(7))
            .await
            .expect("run");
        let second = bench_embedder(&embedder, "mock", Duration::from_millis(7))
            .await
            .expect("run");

        // Assert
        assert_eq!(first.overall, second.overall);
        assert_eq!(first.per_category, second.per_category);
        assert_eq!(first.dims, MOCK_DIMS);
        assert_eq!(first.vector_bytes, MOCK_DIMS * BYTES_PER_F32);
        assert_eq!(first.batch_items, first.facts);
        assert!((first.load_ms - 7.0).abs() < 1e-6);
        let overall = first.overall.expect("scored");
        for value in [
            overall.recall_at_1,
            overall.recall_at_5,
            overall.recall_at_10,
            overall.ndcg_at_10,
            overall.mrr,
        ] {
            assert!((0.0..=1.0).contains(&value), "{value}");
        }
        assert!(overall.recall_at_1 <= overall.recall_at_5);
        assert!(overall.recall_at_5 <= overall.recall_at_10);
        let text = first.render();
        assert!(
            text.contains("overall") && text.contains("fact batch"),
            "{text}"
        );
    }
}
