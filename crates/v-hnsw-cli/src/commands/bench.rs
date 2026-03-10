//! Bench command - Run a benchmark.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_core::{DistanceMetric, VectorIndex, VectorStore};
use v_hnsw_graph::{DotProductDistance, HnswGraph, L2Distance, NormalizedCosineDistance};
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;
use crate::is_interrupted;

/// Run the bench command.
pub fn run(path: PathBuf, queries: usize, k: usize) -> Result<()> {
    super::common::require_db(&path)?;

    // Load config
    let config = DbConfig::load(&path)?;

    // Execute benchmark based on metric
    match config.metric.as_str() {
        "cosine" => bench_with_metric::<NormalizedCosineDistance>(
            &path,
            &config,
            queries,
            k,
            NormalizedCosineDistance,
        )?,
        "l2" => bench_with_metric::<L2Distance>(
            &path,
            &config,
            queries,
            k,
            L2Distance,
        )?,
        "dot" => bench_with_metric::<DotProductDistance>(
            &path,
            &config,
            queries,
            k,
            DotProductDistance,
        )?,
        other => anyhow::bail!("Unknown metric: {other}"),
    };

    Ok(())
}

/// Benchmark with a specific distance metric.
fn bench_with_metric<D: DistanceMetric + Clone>(
    path: &PathBuf,
    config: &DbConfig,
    num_queries: usize,
    k: usize,
    distance: D,
) -> Result<()> {
    // Open storage
    let engine = StorageEngine::open(path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let doc_count = engine.len();
    if doc_count == 0 {
        anyhow::bail!("Database is empty");
    }

    // Create HNSW config
    let hnsw_config = config.to_hnsw_config()?;

    // Build HNSW graph from storage
    println!("Loading {} vectors into HNSW graph...", doc_count);
    let load_start = Instant::now();
    let mut hnsw: HnswGraph<D> = HnswGraph::new(hnsw_config, distance);

    // Load vectors from storage into graph
    let vector_store = engine.vector_store();
    let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();
    let mut query_vectors: Vec<Vec<f32>> = Vec::new();

    for id in &ids {
        if is_interrupted() {
            println!("Interrupted during load");
            return Ok(());
        }

        if let Ok(vec) = vector_store.get(*id) {
            let _ = hnsw.insert(*id, vec);

            // Collect some vectors for queries
            if query_vectors.len() < num_queries {
                query_vectors.push(vec.to_vec());
            }
        }
    }
    let load_elapsed = load_start.elapsed();
    println!("Load time: {:.2}s", load_elapsed.as_secs_f64());

    if query_vectors.is_empty() {
        anyhow::bail!("No vectors loaded");
    }

    // If we don't have enough unique vectors, repeat them
    while query_vectors.len() < num_queries {
        let idx = query_vectors.len() % ids.len();
        if let Ok(vec) = vector_store.get(ids[idx]) {
            query_vectors.push(vec.to_vec());
        }
    }

    // Run benchmark
    println!("\nRunning {} queries (k={})...", num_queries, k);
    let ef_values = [50, 100, 200];

    for ef in ef_values {
        if is_interrupted() {
            println!("Interrupted");
            return Ok(());
        }

        let start = Instant::now();
        let mut total_results = 0usize;

        for query in &query_vectors {
            if is_interrupted() {
                println!("Interrupted");
                return Ok(());
            }

            if let Ok(results) = hnsw.search(query, k, ef) {
                total_results += results.len();
            }
        }

        let elapsed = start.elapsed();
        let qps = num_queries as f64 / elapsed.as_secs_f64();
        let avg_latency_us = elapsed.as_micros() as f64 / num_queries as f64;

        println!(
            "  ef={:3}: {:8.1} QPS, {:7.1} us/query, {:.1} avg results",
            ef,
            qps,
            avg_latency_us,
            total_results as f64 / num_queries as f64
        );
    }

    Ok(())
}
