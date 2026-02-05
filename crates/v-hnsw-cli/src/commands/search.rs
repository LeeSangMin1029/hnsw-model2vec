//! Search command - Search for nearest neighbors.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use v_hnsw_core::{DistanceMetric, PayloadStore, VectorIndex, VectorStore};
use v_hnsw_distance::{CosineDistance, DotProductDistance, L2Distance};
use v_hnsw_graph::{HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, HybridSearchConfig, Reranker, SimpleHybridSearcher, WhitespaceTokenizer};
use v_hnsw_storage::StorageEngine;
use v_hnsw_rerank::{CrossEncoderConfig, CrossEncoderReranker, RerankerModel};

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

/// Search command parameters.
pub struct SearchParams {
    pub path: PathBuf,
    pub vector: Option<String>,
    pub text: Option<String>,
    pub k: usize,
    pub ef: usize,
    pub collection: String,
    pub rerank: bool,
    pub rerank_model: String,
    pub rerank_top: Option<usize>,
}

/// Run the search command.
pub fn run(params: SearchParams) -> Result<()> {
    let SearchParams {
        path,
        vector,
        text,
        k,
        ef,
        collection: _collection,
        rerank,
        rerank_model,
        rerank_top,
    } = params;
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

    // Validate reranking only works with hybrid search
    if rerank && (vector.is_none() || text.is_none()) {
        anyhow::bail!("Reranking requires both --vector and --text for hybrid search");
    }

    // Determine rerank candidate count
    let rerank_candidates = rerank_top.unwrap_or(k * 3);

    // Execute search based on metric
    let start = Instant::now();
    let mut results = match config.metric.as_str() {
        "cosine" => search_with_metric::<CosineDistance>(
            &path,
            &config,
            query_vector.clone(),
            text.clone(),
            if rerank { rerank_candidates } else { k },
            ef,
            CosineDistance,
        )?,
        "l2" => search_with_metric::<L2Distance>(
            &path,
            &config,
            query_vector.clone(),
            text.clone(),
            if rerank { rerank_candidates } else { k },
            ef,
            L2Distance,
        )?,
        "dot" => search_with_metric::<DotProductDistance>(
            &path,
            &config,
            query_vector.clone(),
            text.clone(),
            if rerank { rerank_candidates } else { k },
            ef,
            DotProductDistance,
        )?,
        other => anyhow::bail!("Unknown metric: {other}"),
    };

    // Apply reranking if enabled
    if rerank {
        let query_text = text.as_ref().ok_or_else(|| anyhow::anyhow!("Text query required for reranking"))?;

        // Open storage to get document texts
        let engine = StorageEngine::open(&path)
            .with_context(|| format!("failed to open database at {}", path.display()))?;
        let payload_store = engine.payload_store();

        // Build candidates for reranking
        let mut candidates = Vec::new();
        for (id, score) in &results {
            if let Ok(Some(doc_text)) = payload_store.get_text(*id) {
                candidates.push((*id, *score, doc_text));
            }
        }

        // Create reranker
        let model = match rerank_model.as_str() {
            "minilm" => RerankerModel::MsMiniLM,
            "bge" => RerankerModel::BgeBase,
            other => anyhow::bail!("Unknown rerank model: {other}. Use 'minilm' or 'bge'"),
        };

        let config = CrossEncoderConfig::new(model);
        let reranker = CrossEncoderReranker::new(config)
            .with_context(|| "failed to create reranker")?;

        // Rerank
        let reranked = reranker.rerank(query_text, &candidates)
            .with_context(|| "reranking failed")?;

        // Use reranked results
        results = reranked;

        // Take top k
        results.truncate(k);
    }

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

    // Check for pre-built index files
    let hnsw_path = path.join("hnsw.bin");
    let bm25_path = path.join("bm25.bin");

    // Load or build HNSW graph
    let hnsw: HnswGraph<D> = if hnsw_path.exists() {
        HnswGraph::load(&hnsw_path, distance)
            .with_context(|| format!("failed to load HNSW index from {}", hnsw_path.display()))?
    } else {
        // Fallback: build from storage (slow)
        let hnsw_config = HnswConfig::builder()
            .dim(config.dim)
            .m(config.m)
            .ef_construction(config.ef_construction)
            .build()
            .with_context(|| "failed to create HNSW config")?;

        let mut hnsw = HnswGraph::new(hnsw_config, distance);

        // Load vectors from storage into graph
        let vector_store = engine.vector_store();
        let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();

        for id in &ids {
            if let Ok(vec) = vector_store.get(*id) {
                // Ignore errors during load - point might be deleted
                let _ = hnsw.insert(*id, vec);
            }
        }

        hnsw
    };

    match (query_vector, query_text) {
        // Dense-only search
        (Some(vec), None) => {
            let results = hnsw.search(&vec, k, ef)
                .with_context(|| "search failed")?;
            Ok(results)
        }
        // Sparse-only search (BM25)
        (None, Some(text)) => {
            // Load or build BM25 index
            let bm25: Bm25Index<WhitespaceTokenizer> = if bm25_path.exists() {
                Bm25Index::load(&bm25_path)
                    .with_context(|| format!("failed to load BM25 index from {}", bm25_path.display()))?
            } else {
                // Fallback: build from storage (slow)
                let mut bm25 = Bm25Index::new(WhitespaceTokenizer::new());

                let vector_store = engine.vector_store();
                let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();
                let payload_store = engine.payload_store();

                for id in &ids {
                    if let Ok(Some(doc_text)) = payload_store.get_text(*id) {
                        bm25.add_document(*id, &doc_text);
                    }
                }

                bm25
            };

            let results = bm25.search(&text, k);
            Ok(results)
        }
        // Hybrid search
        (Some(vec), Some(text)) => {
            // Load or build BM25 index
            let bm25: Bm25Index<WhitespaceTokenizer> = if bm25_path.exists() {
                Bm25Index::load(&bm25_path)
                    .with_context(|| format!("failed to load BM25 index from {}", bm25_path.display()))?
            } else {
                // Fallback: build from storage (slow)
                let mut bm25 = Bm25Index::new(WhitespaceTokenizer::new());

                let vector_store = engine.vector_store();
                let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();
                let payload_store = engine.payload_store();

                for id in &ids {
                    if let Ok(Some(doc_text)) = payload_store.get_text(*id) {
                        bm25.add_document(*id, &doc_text);
                    }
                }

                bm25
            };

            // Create hybrid searcher config
            let hybrid_config = HybridSearchConfig::builder()
                .ef_search(ef)
                .build();

            let searcher = SimpleHybridSearcher::new(hnsw, bm25, hybrid_config);

            let results = searcher.search(&vec, &text, k)
                .with_context(|| "hybrid search failed")?;
            Ok(results)
        }
        (None, None) => unreachable!("validated above"),
    }
}
