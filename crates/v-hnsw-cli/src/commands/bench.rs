//! Bench command — recall + latency benchmark with brute-force ground truth.
//!
//! Compares: brute-force (exact) → HNSW f32 → HNSW SQ8 two-stage.
//! Reports recall@k and QPS for each method at multiple ef values.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_core::{DistanceMetric, PointId, VectorStore};
use v_hnsw_graph::{HnswGraph, HnswSnapshot, NormalizedCosineDistance};
use v_hnsw_storage::sq8::Sq8Params;
use v_hnsw_storage::sq8_store::Sq8VectorStore;
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;

/// Run the bench command.
pub fn run(path: PathBuf, queries: usize, k: usize) -> Result<()> {
    super::common::require_db(&path)?;
    let config = DbConfig::load(&path)?;

    if config.metric.as_str() != "cosine" {
        anyhow::bail!("bench currently only supports cosine metric (got: {})", config.metric);
    }

    let engine = StorageEngine::open(&path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let doc_count = engine.len();
    if doc_count == 0 {
        anyhow::bail!("Database is empty");
    }

    let vector_store = engine.vector_store();
    let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();
    let dim = vector_store.dim();

    // Sample query vectors (every N-th vector for diversity)
    let step = ids.len().max(1) / queries.min(ids.len()).max(1);
    let query_ids: Vec<u64> = ids.iter().step_by(step.max(1)).take(queries).copied().collect();
    let query_vectors: Vec<Vec<f32>> = query_ids
        .iter()
        .filter_map(|&id| vector_store.get(id).ok().map(|v| v.to_vec()))
        .collect();
    let num_queries = query_vectors.len();

    println!("Database: {} vectors, dim={}", doc_count, dim);
    println!("Queries:  {} sampled vectors", num_queries);
    println!("k={}", k);
    println!();

    // 1. Brute-force ground truth
    println!("Computing brute-force ground truth...");
    let bf_start = Instant::now();
    let ground_truth: Vec<Vec<PointId>> = query_vectors
        .iter()
        .map(|q| brute_force_topk(vector_store, &ids, q, k))
        .collect();
    let bf_elapsed = bf_start.elapsed();
    println!(
        "  Brute-force: {:.1}ms total, {:.1}us/query",
        bf_elapsed.as_secs_f64() * 1000.0,
        bf_elapsed.as_micros() as f64 / num_queries as f64,
    );
    println!();

    // 2. HNSW f32 search (via snapshot or heap)
    println!("=== HNSW f32 ===");
    let snap_path = path.join("hnsw.snap");
    let bin_path = path.join("hnsw.bin");

    let ef_values = [50, 100, 200, 400];

    if snap_path.exists() {
        let snap = HnswSnapshot::open(&snap_path).context("Failed to open snapshot")?;
        for ef in ef_values {
            let (recall, qps, latency) = bench_f32_snapshot(
                &snap, vector_store, &query_vectors, &ground_truth, k, ef,
            );
            println!(
                "  ef={ef:3}: recall@{k}={recall:.4}  {qps:8.1} QPS  {latency:7.1}us/q",
            );
        }
    } else if bin_path.exists() {
        let hnsw: HnswGraph<NormalizedCosineDistance> =
            HnswGraph::load(&bin_path, NormalizedCosineDistance)
                .context("Failed to load HNSW")?;
        for ef in ef_values {
            let (recall, qps, latency) = bench_f32_heap(
                &hnsw, vector_store, &query_vectors, &ground_truth, k, ef,
            );
            println!(
                "  ef={ef:3}: recall@{k}={recall:.4}  {qps:8.1} QPS  {latency:7.1}us/q",
            );
        }
    } else {
        println!("  (no HNSW index found, skipping)");
    }

    // 3. HNSW SQ8 two-stage search
    let params_path = path.join("sq8_params.bin");
    let sq8_path = path.join("sq8_vectors.bin");

    if params_path.exists() && sq8_path.exists() {
        println!();
        println!("=== HNSW SQ8 (two-stage) ===");

        let params = Sq8Params::load(&params_path).context("Failed to load SQ8 params")?;
        let mut sq8_store = Sq8VectorStore::open(&sq8_path).context("Failed to open SQ8 store")?;
        sq8_store.restore_id_map(vector_store.id_map());

        let sq8_size = ids.len() * dim;
        let f32_size = ids.len() * dim * 4;
        println!(
            "  Memory: {:.2}MB (SQ8) vs {:.2}MB (f32) = {:.1}x compression",
            sq8_size as f64 / 1_048_576.0,
            f32_size as f64 / 1_048_576.0,
            f32_size as f64 / sq8_size as f64,
        );

        if snap_path.exists() {
            let snap = HnswSnapshot::open(&snap_path).context("Failed to open snapshot")?;
            for ef in ef_values {
                let (recall, qps, latency) = bench_sq8_snapshot(
                    &snap, vector_store, &params, &sq8_store, &query_vectors, &ground_truth, k, ef,
                );
                println!(
                    "  ef={ef:3}: recall@{k}={recall:.4}  {qps:8.1} QPS  {latency:7.1}us/q",
                );
            }
        } else if bin_path.exists() {
            let hnsw: HnswGraph<NormalizedCosineDistance> =
                HnswGraph::load(&bin_path, NormalizedCosineDistance)
                    .context("Failed to load HNSW")?;
            for ef in ef_values {
                let (recall, qps, latency) = bench_sq8_heap(
                    &hnsw, vector_store, &params, &sq8_store, &query_vectors, &ground_truth, k, ef,
                );
                println!(
                    "  ef={ef:3}: recall@{k}={recall:.4}  {qps:8.1} QPS  {latency:7.1}us/q",
                );
            }
        }
    } else {
        println!();
        println!("=== SQ8 not found (run `v-hnsw build-index` to generate) ===");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Brute-force ground truth
// ---------------------------------------------------------------------------

fn brute_force_topk(
    store: &dyn VectorStore,
    ids: &[u64],
    query: &[f32],
    k: usize,
) -> Vec<PointId> {
    let mut dists: Vec<(PointId, f32)> = ids
        .iter()
        .filter_map(|&id| {
            store.get(id).ok().map(|vec| {
                (id, NormalizedCosineDistance.distance(query, vec))
            })
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists.into_iter().map(|(id, _)| id).collect()
}

// ---------------------------------------------------------------------------
// Common benchmark harness
// ---------------------------------------------------------------------------

/// Run a search function over queries and measure recall/QPS/latency.
fn bench_search<F>(
    search_fn: F,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<PointId>],
) -> (f64, f64, f64)
where
    F: Fn(&[f32]) -> Option<Vec<(PointId, f32)>>,
{
    let start = Instant::now();
    let mut total_recall = 0.0_f64;

    for (i, query) in queries.iter().enumerate() {
        if let Some(results) = search_fn(query) {
            total_recall += recall_at_k(&results, &ground_truth[i]);
        }
    }

    let elapsed = start.elapsed();
    let avg_recall = total_recall / queries.len() as f64;
    let qps = queries.len() as f64 / elapsed.as_secs_f64();
    let latency = elapsed.as_micros() as f64 / queries.len() as f64;
    (avg_recall, qps, latency)
}

// ---------------------------------------------------------------------------
// f32 benchmarks
// ---------------------------------------------------------------------------

fn bench_f32_snapshot(
    snap: &HnswSnapshot,
    store: &dyn VectorStore,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<PointId>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    bench_search(
        |q| snap.search_ext(&NormalizedCosineDistance, store, q, k, ef).ok(),
        queries,
        ground_truth,
    )
}

fn bench_f32_heap(
    hnsw: &HnswGraph<NormalizedCosineDistance>,
    store: &dyn VectorStore,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<PointId>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    bench_search(
        |q| hnsw.search_ext(store, q, k, ef).ok(),
        queries,
        ground_truth,
    )
}

// ---------------------------------------------------------------------------
// SQ8 benchmarks
// ---------------------------------------------------------------------------

use super::common::{F32Dc, Sq8Dc};

#[expect(clippy::too_many_arguments)]
fn bench_sq8_snapshot(
    snap: &HnswSnapshot,
    f32_store: &dyn VectorStore,
    params: &Sq8Params,
    sq8_store: &Sq8VectorStore,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<PointId>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    let approx = Sq8Dc { params, store: sq8_store };
    let exact = F32Dc { store: f32_store };
    bench_search(
        |q| snap.search_two_stage(&approx, &exact, q, k, ef).ok(),
        queries,
        ground_truth,
    )
}

#[expect(clippy::too_many_arguments)]
fn bench_sq8_heap(
    hnsw: &HnswGraph<NormalizedCosineDistance>,
    f32_store: &dyn VectorStore,
    params: &Sq8Params,
    sq8_store: &Sq8VectorStore,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<PointId>],
    k: usize,
    ef: usize,
) -> (f64, f64, f64) {
    let approx = Sq8Dc { params, store: sq8_store };
    let exact = F32Dc { store: f32_store };
    bench_search(
        |q| hnsw.search_two_stage(&approx, &exact, q, k, ef).ok(),
        queries,
        ground_truth,
    )
}

// ---------------------------------------------------------------------------
// Recall computation
// ---------------------------------------------------------------------------

fn recall_at_k(results: &[(PointId, f32)], truth: &[PointId]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_set: HashSet<PointId> = truth.iter().copied().collect();
    let hits = results.iter().filter(|(id, _)| truth_set.contains(id)).count();
    hits as f64 / truth.len() as f64
}
