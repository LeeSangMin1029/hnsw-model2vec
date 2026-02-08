//! Daemon state: model, indexes, search execution.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_core::VectorIndex;
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{CosineDistance, HnswGraph};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer, RrfFusion};
use v_hnsw_storage::StorageEngine;

use crate::commands::common::{self, SearchResultItem};
use super::super::create::DbConfig;

/// Search response from daemon.
#[derive(Debug, serde::Serialize)]
pub(crate) struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub elapsed_ms: f64,
}

/// Loaded daemon state: model + indexes + storage.
///
/// Indexes are stored separately so we can create a per-query search config
/// with dynamic `dense_limit` (= k * 2), consistent with direct search.
pub(crate) struct DaemonState {
    pub embed_model: Model2VecModel,
    hnsw: HnswGraph<CosineDistance>,
    bm25: Bm25Index<KoreanBm25Tokenizer>,
    pub engine: StorageEngine,
}

impl DaemonState {
    pub fn new(db_path: &Path) -> Result<Self> {
        let config = DbConfig::load(db_path)?;

        let hnsw_path = db_path.join("hnsw.bin");
        let bm25_path = db_path.join("bm25.bin");

        if !hnsw_path.exists() {
            anyhow::bail!("HNSW index not found. Run 'v-hnsw add' first.");
        }
        if !bm25_path.exists() {
            anyhow::bail!("BM25 index not found. Run 'v-hnsw add' first.");
        }

        let model_name = config
            .embed_model
            .ok_or_else(|| anyhow::anyhow!("No embedding model specified in database config."))?;

        eprintln!("[daemon] Loading model2vec: {}", model_name);
        let t0 = Instant::now();
        let embed_model = Model2VecModel::from_pretrained(&model_name)
            .context("Failed to initialize model2vec model")?;
        eprintln!("[daemon] Model loaded: {:.0}ms", t0.elapsed().as_millis());

        if embed_model.dim() != config.dim {
            anyhow::bail!(
                "Model dimension ({}) doesn't match database dimension ({})",
                embed_model.dim(),
                config.dim
            );
        }

        eprintln!("[daemon] Warming up model...");
        let _ = embed_model.embed(&["warmup"]);
        eprintln!("[daemon] Warmup complete");

        eprintln!("[daemon] Loading HNSW index...");
        let hnsw: HnswGraph<CosineDistance> = HnswGraph::load(&hnsw_path, CosineDistance)
            .context("Failed to load HNSW index")?;
        eprintln!("[daemon] HNSW loaded: {} vectors", hnsw.len());

        eprintln!("[daemon] Loading BM25 index...");
        common::ensure_korean_dict()?;
        let bm25: Bm25Index<KoreanBm25Tokenizer> =
            Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;
        eprintln!("[daemon] BM25 loaded");

        let engine = StorageEngine::open(db_path).context("Failed to open storage")?;

        Ok(Self {
            embed_model,
            hnsw,
            bm25,
            engine,
        })
    }

    /// Reload HNSW, BM25 indexes and reopen storage from disk.
    pub fn reload(&mut self, db_path: &Path) -> Result<()> {
        let hnsw_path = db_path.join("hnsw.bin");
        let bm25_path = db_path.join("bm25.bin");

        eprintln!("[daemon] Reloading indexes...");
        let t0 = Instant::now();

        self.hnsw =
            HnswGraph::load(&hnsw_path, CosineDistance).context("Failed to reload HNSW index")?;
        eprintln!("[daemon] HNSW reloaded: {} vectors", self.hnsw.len());

        self.bm25 = Bm25Index::load(&bm25_path).context("Failed to reload BM25 index")?;
        eprintln!("[daemon] BM25 reloaded");

        self.engine = StorageEngine::open(db_path).context("Failed to reopen storage")?;

        eprintln!(
            "[daemon] Reload complete: {:.0}ms",
            t0.elapsed().as_millis()
        );
        Ok(())
    }

    /// Execute a search query with dynamic dense_limit = k * 2 for consistency with find.
    pub fn search(&self, query: &str, k: usize, tags: Vec<String>) -> Result<SearchResponse> {
        let start = Instant::now();

        let query_embedding = self
            .embed_model
            .embed(&[query])
            .context("Failed to embed query")?
            .into_iter()
            .next()
            .context("No embedding returned")?;

        let payload_store = self.engine.payload_store();

        // Dynamic limits consistent with find/direct.rs (k * 2)
        let dense_limit = k * 2;
        let sparse_limit = k * 2;

        let dense_results = self
            .hnsw
            .search(&query_embedding, dense_limit, 200)
            .context("HNSW search failed")?;

        let sparse_results = self.bm25.search(query, sparse_limit);

        // Fuse with weighted RRF
        let fusion = RrfFusion::new();
        let fetch_k = if tags.is_empty() { k } else { k * 10 };
        let weighted_lists = vec![
            (0.7_f32, dense_results),
            (0.3_f32, sparse_results),
        ];
        let mut results = fusion.fuse_weighted(&weighted_lists, fetch_k);

        // Apply tag filtering
        if !tags.is_empty() {
            let allowed_ids = payload_store.points_by_tags(&tags);
            let allowed_set: std::collections::HashSet<_> = allowed_ids.into_iter().collect();
            results.retain(|(id, _)| allowed_set.contains(id));
            results.truncate(k);
        }

        let results_with_text = common::build_results(&results, payload_store);

        let elapsed = start.elapsed();

        Ok(SearchResponse {
            results: results_with_text,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        })
    }
}
