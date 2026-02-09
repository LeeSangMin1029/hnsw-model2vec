//! Daemon state: model, indexes, search execution.
//!
//! The embedding model is loaded lazily on first use and unloaded after
//! `MODEL_IDLE_SECS` of inactivity to reduce steady-state memory (~488 MB).
//!
//! Indexes use mmap snapshots when available (hnsw.snap / bm25.snap),
//! falling back to heap-loaded bincode (hnsw.bin / bm25.bin).

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_core::{PointId, VectorIndex, VectorStore};
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{HnswGraph, HnswSnapshot, NormalizedCosineDistance};
use v_hnsw_search::{Bm25Index, Bm25Snapshot, ConvexFusion, KoreanBm25Tokenizer};
use v_hnsw_storage::StorageEngine;

use crate::commands::common::{self, QueryCache, SearchResultItem};
use super::super::create::DbConfig;

/// Seconds of idle time before unloading the embedding model.
const MODEL_IDLE_SECS: u64 = 300;

/// Dense index: mmap snapshot or heap graph.
enum DenseIndex {
    Snapshot(HnswSnapshot),
    Heap(HnswGraph<NormalizedCosineDistance>),
}

impl DenseIndex {
    fn search(&self, store: &dyn VectorStore, query: &[f32], k: usize, ef: usize) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        match self {
            Self::Snapshot(s) => s.search_ext(&NormalizedCosineDistance, store, query, k, ef),
            Self::Heap(h) => h.search_ext(store, query, k, ef),
        }
    }
    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self { Self::Snapshot(s) => s.len(), Self::Heap(h) => h.len() }
    }
}

/// Sparse index: mmap snapshot or heap index.
enum SparseIndex {
    Snapshot(Bm25Snapshot),
    Heap(Bm25Index<KoreanBm25Tokenizer>),
}

impl SparseIndex {
    fn search(&self, query: &str, limit: usize) -> Vec<(PointId, f32)> {
        match self {
            Self::Snapshot(s) => s.search(&KoreanBm25Tokenizer::new(), query, limit),
            Self::Heap(h) => h.search(query, limit),
        }
    }
    fn score_documents(&self, query: &str, doc_ids: &[PointId]) -> Vec<(PointId, f32)> {
        match self {
            Self::Snapshot(s) => s.score_documents(&KoreanBm25Tokenizer::new(), query, doc_ids),
            Self::Heap(h) => h.score_documents(query, doc_ids),
        }
    }
}

/// Search response from daemon.
#[derive(Debug, serde::Serialize)]
pub(crate) struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub elapsed_ms: f64,
}

/// Loaded daemon state: indexes + storage + lazy model.
pub(crate) struct DaemonState {
    model: Option<Model2VecModel>,
    model_name: String,
    model_dim: usize,
    last_embed_at: Instant,
    dense: DenseIndex,
    sparse: SparseIndex,
    pub engine: StorageEngine,
    query_cache: QueryCache,
}

impl DaemonState {
    pub fn new(db_path: &Path) -> Result<Self> {
        let config = DbConfig::load(db_path)?;
        let model_name = config
            .embed_model
            .ok_or_else(|| anyhow::anyhow!("No embedding model specified in database config."))?;

        let engine = StorageEngine::open(db_path).context("Failed to open storage")?;

        common::ensure_korean_dict()?;

        let dense = load_dense(db_path)?;
        let sparse = load_sparse(db_path)?;

        let query_cache = QueryCache::load(db_path);
        eprintln!("[daemon] Query cache: {} entries", query_cache.len());
        eprintln!("[daemon] Model deferred: {} (loads on first query)", model_name);

        Ok(Self {
            model: None,
            model_name,
            model_dim: config.dim,
            last_embed_at: Instant::now(),
            dense,
            sparse,
            engine,
            query_cache,
        })
    }

    /// Load model if not present, return a reference.
    fn ensure_model(&mut self) -> Result<&Model2VecModel> {
        if self.model.is_none() {
            eprintln!("[daemon] Loading model2vec: {}", self.model_name);
            let t0 = Instant::now();
            let m = Model2VecModel::from_pretrained(&self.model_name)
                .context("Failed to load model2vec model")?;

            if m.dim() != self.model_dim {
                anyhow::bail!(
                    "Model dimension ({}) doesn't match database dimension ({})",
                    m.dim(),
                    self.model_dim
                );
            }

            eprintln!("[daemon] Model loaded: {:.0}ms", t0.elapsed().as_millis());
            self.model = Some(m);
        }
        self.last_embed_at = Instant::now();
        #[allow(clippy::unwrap_used)]
        Ok(self.model.as_ref().unwrap())
    }

    /// Unload model if idle longer than threshold. Call periodically.
    pub fn maybe_unload_model(&mut self) {
        if self.model.is_some() && self.last_embed_at.elapsed().as_secs() > MODEL_IDLE_SECS {
            self.model = None;
            eprintln!("[daemon] Model unloaded (idle > {}s)", MODEL_IDLE_SECS);
        }
    }

    /// Embed texts using the lazy-loaded model.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let model = self.ensure_model()?;
        model.embed(texts).context("Failed to embed")
    }

    /// Reload indexes and reopen storage from disk.
    pub fn reload(&mut self, db_path: &Path) -> Result<()> {
        eprintln!("[daemon] Reloading indexes...");
        let t0 = Instant::now();
        self.engine = StorageEngine::open(db_path).context("Failed to reopen storage")?;
        self.dense = load_dense(db_path)?;
        self.sparse = load_sparse(db_path)?;
        eprintln!("[daemon] Reload complete: {:.0}ms", t0.elapsed().as_millis());
        Ok(())
    }

    /// Save query cache to disk (call on shutdown).
    pub fn save_cache(&self) -> Result<()> {
        self.query_cache.save()
    }

    /// Execute a search query with dynamic dense_limit = k * 2.
    pub fn search(&mut self, query: &str, k: usize, tags: Vec<String>) -> Result<SearchResponse> {
        let start = Instant::now();

        let query_embedding = if let Some(cached) = self.query_cache.get(query) {
            cached.clone()
        } else {
            let model = self.ensure_model()?;
            let emb = model
                .embed(&[query])
                .context("Failed to embed query")?
                .into_iter()
                .next()
                .context("No embedding returned")?;
            self.query_cache.insert(query.to_string(), emb.clone());
            emb
        };

        let payload_store = self.engine.payload_store();
        let store = self.engine.vector_store();

        let dense_limit = k * 2;
        let sparse_limit = k * 2;

        let dense_results = self.dense
            .search(store, &query_embedding, dense_limit, 200)
            .context("HNSW search failed")?;

        let sparse_results = self.sparse.search(query, sparse_limit);

        // Dense-Guided BM25: enrich sparse results with scores for dense candidates
        let all_sparse = enrich_sparse(&self.sparse, query, &dense_results, sparse_results);

        let alpha = common::fusion_alpha(query);
        let fusion = ConvexFusion::with_alpha(alpha);
        let fetch_k = if tags.is_empty() { k } else { k * 10 };
        let mut results = fusion.fuse(&dense_results, &all_sparse, fetch_k);

        // Apply tag filtering
        if !tags.is_empty() {
            let allowed_ids = payload_store.points_by_tags(&tags);
            let allowed_set: HashSet<_> = allowed_ids.into_iter().collect();
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

/// Load dense index: prefer snapshot (mmap), fallback to heap (bincode).
fn load_dense(db_path: &Path) -> Result<DenseIndex> {
    let snap_path = db_path.join("hnsw.snap");
    if snap_path.exists() {
        let snap = HnswSnapshot::open(&snap_path).context("Failed to open HNSW snapshot")?;
        eprintln!("[daemon] HNSW snapshot loaded: {} vectors (mmap)", snap.len());
        return Ok(DenseIndex::Snapshot(snap));
    }
    let hnsw_path = db_path.join("hnsw.bin");
    let hnsw = HnswGraph::load(&hnsw_path, NormalizedCosineDistance)
        .context("Failed to load HNSW index")?;
    eprintln!("[daemon] HNSW heap loaded: {} vectors", hnsw.len());
    Ok(DenseIndex::Heap(hnsw))
}

/// Load sparse index: prefer snapshot (mmap+FST), fallback to heap (bincode/FST).
fn load_sparse(db_path: &Path) -> Result<SparseIndex> {
    let snap_path = db_path.join("bm25.snap");
    let fst_path = db_path.join("bm25_terms.fst");
    if snap_path.exists() && fst_path.exists() {
        let snap = Bm25Snapshot::open(db_path).context("Failed to open BM25 snapshot")?;
        eprintln!("[daemon] BM25 snapshot loaded: {} docs (mmap)", snap.total_docs());
        return Ok(SparseIndex::Snapshot(snap));
    }
    let bm25_path = db_path.join("bm25.bin");
    let bm25 = Bm25Index::load(&bm25_path).context("Failed to load BM25 index")?;
    eprintln!("[daemon] BM25 heap loaded");
    Ok(SparseIndex::Heap(bm25))
}

/// Enrich BM25 results with scores for dense candidates missing from BM25 top-k.
fn enrich_sparse(
    sparse: &SparseIndex,
    query: &str,
    dense_results: &[(PointId, f32)],
    mut sparse_results: Vec<(PointId, f32)>,
) -> Vec<(PointId, f32)> {
    if dense_results.is_empty() {
        return sparse_results;
    }

    let sparse_ids: HashSet<PointId> = sparse_results.iter().map(|&(id, _)| id).collect();
    let missing: Vec<PointId> = dense_results
        .iter()
        .filter(|&&(id, _)| !sparse_ids.contains(&id))
        .map(|&(id, _)| id)
        .collect();

    if !missing.is_empty() {
        let extra = sparse.score_documents(query, &missing);
        sparse_results.extend(extra);
    }

    sparse_results
}
