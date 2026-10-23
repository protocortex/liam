// SPDX-License-Identifier: Apache-2.0
//! Brute-force retrieval cost against store size, for the ANN decision.
//!
//! The store scans every vector for each query, so latency grows linearly with
//! the row count. This probe measures that growth and the on-disk cost of a
//! vector, using synthetic unit vectors (the scan cost does not depend on what
//! the numbers mean). It is `#[ignore]`d because the largest sizes take
//! minutes:
//!
//! ```text
//! cargo test --release -p liam-daemon scale_probe -- --ignored --nocapture
//! ```

use std::path::Path;
use std::time::{Duration, Instant};

use liam_store::{Backend, DefaultBackend, DefaultGraph, GraphConfig, NewNode, Query};

/// Embedding widths to probe: LIAM's configured Qwen3 width and BGE-M3's.
const PROBE_DIMS: [usize; 2] = [768, 1024];

/// Store sizes to probe; the last is the ceiling the ANN decision cares about.
const PROBE_SIZES: [usize; 5] = [1_000, 10_000, 25_000, 50_000, 100_000];

/// Size at which an embedded store is compared with a text-only one, to
/// isolate the bytes the vectors add.
const DISK_COMPARISON_SIZE: usize = 10_000;

/// Queries timed per size, after the warm-up queries.
const TIMED_QUERIES: usize = 30;
const WARM_UP_QUERIES: usize = 3;

/// Result width of the probed query; the pipeline's usual headline cutoff.
const PROBE_K: usize = 10;

/// Percentile reported beside the mean.
const TAIL_PERCENTILE: f64 = 0.95;

const BYTES_PER_F32: usize = 4;
const MS_PER_SEC: f64 = 1_000.0;

/// Seed of the vector generator, so every run probes the same vectors.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// xorshift64: enough randomness for a spread of unit vectors, no dependency.
struct VectorSource(u64);

impl VectorSource {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit_vector(&mut self, dims: usize) -> Vec<f32> {
        let mut vector: Vec<f32> = (0..dims)
            .map(|_| (self.next_u64() as f64 / u64::MAX as f64 - 0.5) as f32)
            .collect();
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        vector.iter_mut().for_each(|x| *x /= norm);
        vector
    }
}

/// Database file plus its write-ahead log, which is where fresh rows live
/// until a checkpoint.
fn on_disk_bytes(database: &Path) -> u64 {
    let wal = database.with_extension("db-wal");
    [database, wal.as_path()]
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|meta| meta.len())
        .sum()
}

/// Folds the write-ahead log into the database file, so the byte counts
/// compare stored data and not however much log each store happened to hold.
async fn checkpoint(database: &Path) {
    let backend = DefaultBackend::open(database.to_str().expect("utf-8 temp path"), 1)
        .await
        .expect("open for checkpoint");
    backend
        .query("PRAGMA wal_checkpoint(TRUNCATE)", &[])
        .await
        .expect("checkpoint");
}

async fn filled_store(
    database: &Path,
    dims: usize,
    size: usize,
    with_vectors: bool,
) -> DefaultGraph {
    let store = DefaultGraph::open(
        database.to_str().expect("utf-8 temp path"),
        GraphConfig::new(dims),
    )
    .await
    .expect("open store");
    let mut source = VectorSource(SEED);
    for i in 0..size {
        let mut node = NewNode::now("fact", format!("fact-{i}"), format!("synthetic fact {i}"));
        if with_vectors {
            node = node.with_embedding(source.unit_vector(dims));
        }
        store.insert(node).await.expect("insert");
    }
    store
}

struct LatencyMs {
    mean: f64,
    tail: f64,
}

async fn time_queries(store: &DefaultGraph, dims: usize) -> LatencyMs {
    let mut source = VectorSource(SEED ^ 1);
    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_QUERIES);
    for round in 0..WARM_UP_QUERIES + TIMED_QUERIES {
        let query = Query::text(format!("synthetic fact {round}"))
            .with_embedding(source.unit_vector(dims))
            .with_k(PROBE_K);
        let started = Instant::now();
        store.query_explained(&query).await.expect("query");
        if round >= WARM_UP_QUERIES {
            samples.push(started.elapsed());
        }
    }
    samples.sort();
    let mean = samples.iter().sum::<Duration>() / TIMED_QUERIES as u32;
    let tail_index = ((samples.len() as f64 * TAIL_PERCENTILE) as usize).min(samples.len() - 1);
    LatencyMs {
        mean: mean.as_secs_f64() * MS_PER_SEC,
        tail: samples[tail_index].as_secs_f64() * MS_PER_SEC,
    }
}

#[tokio::test]
#[ignore = "inserts up to 100k vectors; see the module doc for the run command"]
async fn scale_probe() {
    println!("dims,rows,query_mean_ms,query_p95_ms,fill_secs");
    for dims in PROBE_DIMS {
        for size in PROBE_SIZES {
            let dir = tempfile::tempdir().expect("temp dir");
            let database = dir.path().join("probe.db");
            let filling = Instant::now();
            let store = filled_store(&database, dims, size, true).await;
            let fill_secs = filling.elapsed().as_secs_f64();
            let latency = time_queries(&store, dims).await;
            println!(
                "{dims},{size},{:.2},{:.2},{fill_secs:.1}",
                latency.mean, latency.tail
            );
        }

        let dir = tempfile::tempdir().expect("temp dir");
        let embedded_db = dir.path().join("embedded.db");
        let text_db = dir.path().join("text.db");
        let embedded = filled_store(&embedded_db, dims, DISK_COMPARISON_SIZE, true).await;
        let text_only = filled_store(&text_db, dims, DISK_COMPARISON_SIZE, false).await;
        checkpoint(&embedded_db).await;
        checkpoint(&text_db).await;
        let added = on_disk_bytes(&embedded_db).saturating_sub(on_disk_bytes(&text_db));
        let per_vector = added as f64 / DISK_COMPARISON_SIZE as f64;
        println!(
            "disk dims={dims}: raw {} bytes/vector, measured {per_vector:.0} bytes/vector added",
            dims * BYTES_PER_F32
        );
        drop((embedded, text_only));
    }
}
