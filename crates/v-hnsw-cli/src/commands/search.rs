//! Search command - Search for nearest neighbors.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use v_hnsw_core::{DistanceMetric, PayloadStore, VectorIndex, VectorStore};
use v_hnsw_distance::{CosineDistance, DotProductDistance, L2Distance};
use v_hnsw_graph::{HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, HybridSearchConfig, SimpleHybridSearcher, WhitespaceTokenizer};
use v_hnsw_storage::StorageEngine;

use super::create::DbConfig;

/// Search result for JSON output.
#[derive(Debug, Serialize)]
struct SearchOutput {
    results: Vec<SearchResult>,
    elapsed_ms: f64,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    id: u64,
    score: f32,
}

/// Parse a comma-separated vector string.
fn parse_vector(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .with_context(|| format!("invalid number: '{part}'"))
        })
        .collect()
}

/// Run the search command.
pub fn run(
    path: PathBuf,
    vector: Option<String>,
    text: Option<String>,
    k: usize,
    ef: usize,
) -> Result<()> {
    // At least one of vector or text must be provided
    if vector.is_none() && text.is_none() {
        anyhow::bail!("At least one of --vector or --text must be provided");
    }

    // Check database exists
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }

    // Load config
    let config = DbConfig::load(&path)?;

    // Parse vector if provided
    let query_vector = if let Some(ref v) = vector {
        let vec = parse_vector(v)?;
        if vec.len() != config.dim {
            anyhow::bail!(
                "Vector dimension mismatch: expected {}, got {}",
                config.dim,
                vec.len()
            );
        }
        Some(vec)
    } else {
        None
    };

    // Execute search based on metric
    let start = Instant::now();
    let results = match config.metric.as_str() {
        "cosine" => search_with_metric::<CosineDistance>(
            &path,
            &config,
            query_vector,
            text,
            k,
            ef,
            CosineDistance,
        )?,
        "l2" => search_with_metric::<L2Distance>(
            &path,
            &config,
            query_vector,
            text,
            k,
            ef,
            L2Distance,
        )?,
        "dot" => search_with_metric::<DotProductDistance>(
            &path,
            &config,
            query_vector,
            text,
            k,
            ef,
            DotProductDistance,
        )?,
        other => anyhow::bail!("Unknown metric: {other}"),
    };
    let elapsed = start.elapsed();

    // Output as JSON
    let output = SearchOutput {
        results: results
            .into_iter()
            .map(|(id, score)| SearchResult { id, score })
            .collect(),
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    let json = serde_json::to_string_pretty(&output)
        .with_context(|| "failed to serialize output")?;
    println!("{json}");

    Ok(())
}

/// Search with a specific distance metric.
fn search_with_metric<D: DistanceMetric + Clone>(
    path: &PathBuf,
    config: &DbConfig,
    query_vector: Option<Vec<f32>>,
    query_text: Option<String>,
    k: usize,
    ef: usize,
    distance: D,
) -> Result<Vec<(u64, f32)>> {
    // Open storage
    let engine = StorageEngine::open(path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    // Create HNSW config
    let hnsw_config = HnswConfig::builder()
        .dim(config.dim)
        .m(config.m)
        .ef_construction(config.ef_construction)
        .build()
        .with_context(|| "failed to create HNSW config")?;

    // Build HNSW graph from storage
    let mut hnsw: HnswGraph<D> = HnswGraph::new(hnsw_config, distance);

    // Load vectors from storage into graph
    let vector_store = engine.vector_store();
    let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();

    for id in &ids {
        if let Ok(vec) = vector_store.get(*id) {
            // Ignore errors during load - point might be deleted
            let _ = hnsw.insert(*id, vec);
        }
    }

    match (query_vector, query_text) {
        // Dense-only search
        (Some(vec), None) => {
            let results = hnsw.search(&vec, k, ef)
                .with_context(|| "search failed")?;
            Ok(results)
        }
        // Sparse-only search (BM25)
        (None, Some(text)) => {
            // Build BM25 index from stored texts
            let mut bm25: Bm25Index<WhitespaceTokenizer> = Bm25Index::new(WhitespaceTokenizer::new());

            let payload_store = engine.payload_store();
            for id in &ids {
                if let Ok(Some(doc_text)) = payload_store.get_text(*id) {
                    bm25.add_document(*id, &doc_text);
                }
            }

            let results = bm25.search(&text, k);
            Ok(results)
        }
        // Hybrid search
        (Some(vec), Some(text)) => {
            // Build BM25 index
            let bm25: Bm25Index<WhitespaceTokenizer> = Bm25Index::new(WhitespaceTokenizer::new());

            // Create hybrid searcher config
            let hybrid_config = HybridSearchConfig::builder()
                .ef_search(ef)
                .build();

            let mut searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);

            // Load documents into BM25
            let payload_store = engine.payload_store();
            for id in &ids {
                let text_data = payload_store
                    .get_text(*id)
                    .ok()
                    .flatten()
                    .unwrap_or_default();
                // Add to sparse index only (dense already added above)
                searcher.sparse_index_mut().add_document(*id, &text_data);
            }

            let results = searcher.search(&vec, &text, k)
                .with_context(|| "hybrid search failed")?;
            Ok(results)
        }
        (None, None) => unreachable!("validated above"),
    }
}
