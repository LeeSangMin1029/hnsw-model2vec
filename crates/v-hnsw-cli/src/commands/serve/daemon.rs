//! Daemon state: model, indexes, search execution.
//!
//! Single daemon serves multiple databases. The embedding model is loaded
//! lazily on first use and unloaded after `MODEL_IDLE_SECS` of inactivity
//! to reduce steady-state memory (~263 MB for f16 potion-multilingual-128M).
//!
//! Databases are loaded on-demand when first requested and kept in memory.
//! Indexes use mmap snapshots when available (hnsw.snap / bm25.snap),
//! falling back to heap-loaded bincode (hnsw.bin / bm25.bin).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
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

/// Seconds of idle time before evicting a database from memory.
const DB_IDLE_SECS: u64 = 15;

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

/// Indexes + storage for a single database.
struct DbIndexes {
    dense: DenseIndex,
    sparse: SparseIndex,
    engine: StorageEngine,
    last_used: Instant,
}

impl DbIndexes {
    /// Run hybrid search (dense + sparse fusion) with a precomputed embedding.
    fn search(&mut self, embedding: &[f32], query: &str, k: usize, tags: Vec<String>) -> Result<Vec<SearchResultItem>> {
        let store = self.engine.vector_store();
        let dense_limit = k * 2;
        let sparse_limit = k * 2;

        let dense_results = self.dense
            .search(store, embedding, dense_limit, 200)
            .context("HNSW search failed")?;
        let sparse_results = self.sparse.search(query, sparse_limit);

        let all_sparse = enrich_sparse(&self.sparse, query, &dense_results, sparse_results);

        let alpha = common::fusion_alpha(query);
        let fusion = ConvexFusion::with_alpha(alpha);
        let fetch_k = if tags.is_empty() { k } else { k * 10 };
        let mut results = fusion.fuse(&dense_results, &all_sparse, fetch_k);

        if !tags.is_empty() {
            let payload_store = self.engine.payload_store();
            let allowed: HashSet<_> = payload_store.points_by_tags(&tags).into_iter().collect();
            results.retain(|(id, _)| allowed.contains(id));
            results.truncate(k);
        }

        Ok(common::build_results(&results, self.engine.payload_store()))
    }
}

/// Search response from daemon.
#[derive(Debug, serde::Serialize)]
pub(crate) struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub elapsed_ms: f64,
}

/// Loaded daemon state: shared model + per-DB indexes.
pub(crate) struct DaemonState {
    model: Option<Model2VecModel>,
    model_name: Option<String>,
    model_dim: Option<usize>,
    last_embed_at: Instant,
    databases: HashMap<PathBuf, DbIndexes>,
    query_cache: QueryCache,
}

impl DaemonState {
    /// Create a new daemon state, optionally preloading one database.
    pub fn new(initial_db: Option<&Path>) -> Result<Self> {
        common::ensure_korean_dict()?;

        let mut state = Self {
            model: None,
            model_name: None,
            model_dim: None,
            last_embed_at: Instant::now(),
            databases: HashMap::new(),
            query_cache: QueryCache::global(),
        };

        if let Some(db_path) = initial_db {
            state.ensure_db(db_path)?;
        }

        eprintln!("[daemon] Model deferred (loads on first query)");
        Ok(state)
    }

    /// Load (or reload) a DB into the map. `key` must be canonicalized.
    fn register_db(&mut self, key: &Path) -> Result<()> {
        let indexes = load_db(key)?;
        if self.model_name.is_none() {
            let config = DbConfig::load(key)?;
            if let Some(name) = config.embed_model {
                self.model_name = Some(name);
                self.model_dim = Some(config.dim);
            }
        }
        self.databases.insert(key.to_path_buf(), indexes);
        eprintln!("[daemon] DB registered: {} ({} total)", key.display(), self.databases.len());
        Ok(())
    }

    /// Ensure a database is loaded on-demand, return mutable ref.
    fn ensure_db(&mut self, db_path: &Path) -> Result<&mut DbIndexes> {
        let key = canonicalize(db_path)?;
        if !self.databases.contains_key(&key) {
            self.register_db(&key)?;
        }
        #[allow(clippy::unwrap_used)]
        let db = self.databases.get_mut(&key).unwrap();
        db.last_used = Instant::now();
        Ok(db)
    }

    /// Ensure model is loaded, return reference.
    fn ensure_model(&mut self) -> Result<&Model2VecModel> {
        if self.model.is_none() {
            let name = self.model_name.as_deref()
                .ok_or_else(|| anyhow::anyhow!("No model name set — load a DB first"))?;
            eprintln!("[daemon] Loading model2vec: {}", name);
            let t0 = Instant::now();
            let m = Model2VecModel::from_pretrained(name)
                .context("Failed to load model2vec model")?;
            if let Some(dim) = self.model_dim {
                anyhow::ensure!(m.dim() == dim,
                    "Model dim ({}) != DB dim ({})", m.dim(), dim);
            }
            eprintln!("[daemon] Model loaded: {:.0}ms", t0.elapsed().as_millis());
            self.model = Some(m);
        }
        self.last_embed_at = Instant::now();
        #[allow(clippy::unwrap_used)]
        Ok(self.model.as_ref().unwrap())
    }

    /// Resolve query → embedding (cache hit or model inference).
    fn resolve_embedding(&mut self, query: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.query_cache.get(query) {
            return Ok(cached.clone());
        }
        let model = self.ensure_model()?;
        let emb = model.embed(&[query]).context("Failed to embed query")?
            .into_iter().next().context("No embedding returned")?;
        self.query_cache.insert(query.to_string(), emb.clone());
        Ok(emb)
    }

    /// Unload model if idle longer than threshold.
    pub fn maybe_unload_model(&mut self) {
        if self.model.is_some() && self.last_embed_at.elapsed().as_secs() > MODEL_IDLE_SECS {
            self.model = None;
            eprintln!("[daemon] Model unloaded (idle > {}s)", MODEL_IDLE_SECS);
            trim_working_set();
        }
    }

    /// Evict databases idle longer than DB_IDLE_SECS.
    pub fn maybe_evict_databases(&mut self) {
        let before = self.databases.len();
        self.databases.retain(|path, db| {
            if db.last_used.elapsed().as_secs() > DB_IDLE_SECS {
                eprintln!("[daemon] DB evicted (idle > {}s): {}", DB_IDLE_SECS, path.display());
                false
            } else {
                true
            }
        });
        if self.databases.len() < before {
            eprintln!("[daemon] {} DB(s) in memory", self.databases.len());
            trim_working_set();
        }
    }

    /// Embed texts using the lazy-loaded model.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let model = self.ensure_model()?;
        model.embed(texts).context("Failed to embed")
    }

    /// Reload indexes for a specific database.
    pub fn reload(&mut self, db_path: &Path) -> Result<()> {
        let t0 = Instant::now();
        let key = canonicalize(db_path)?;
        self.register_db(&key)?;
        eprintln!("[daemon] Reload complete: {:.0}ms", t0.elapsed().as_millis());
        Ok(())
    }

    pub fn save_cache(&self) -> Result<()> { self.query_cache.save() }

    /// Run incremental update using the daemon's shared model.
    ///
    /// Drops cached read-only engine, runs update with exclusive lock,
    /// then reloads indexes for subsequent searches.
    pub fn update(&mut self, db_path: &Path, input_path: &Path, exclude: &[String]) -> Result<crate::commands::update::UpdateStats> {
        let key = canonicalize(db_path)?;
        let t0 = Instant::now();

        // Drop cached read-only engine so exclusive lock can be acquired
        self.databases.remove(&key);

        // Ensure model is loaded before borrowing self immutably
        self.ensure_model()?;
        let model = self.model.as_ref();

        // Run core update with the daemon's shared model (no 1GB reload)
        let stats = crate::commands::update::run_core(&key, input_path, model, exclude)?;

        // Reload indexes so subsequent searches see the new data
        self.register_db(&key)?;

        eprintln!(
            "[daemon] Update complete: new={} mod={} del={} unchanged={} ({:.0}ms)",
            stats.new, stats.modified, stats.deleted, stats.unchanged,
            t0.elapsed().as_millis()
        );

        Ok(stats)
    }

    /// Search: resolve embedding → delegate to DbIndexes::search.
    pub fn search(&mut self, db_path: &Path, query: &str, k: usize, tags: Vec<String>) -> Result<SearchResponse> {
        let start = Instant::now();
        let embedding = self.resolve_embedding(query)?;
        let db = self.ensure_db(db_path)?;
        let results = db.search(&embedding, query, k, tags)?;
        Ok(SearchResponse { results, elapsed_ms: start.elapsed().as_secs_f64() * 1000.0 })
    }
}

/// Load all indexes + storage for a database.
fn load_db(db_path: &Path) -> Result<DbIndexes> {
    let engine = StorageEngine::open(db_path)
        .with_context(|| format!("Failed to open storage: {}", db_path.display()))?;
    let dense = load_dense(db_path)?;
    let sparse = load_sparse(db_path)?;
    Ok(DbIndexes { dense, sparse, engine, last_used: Instant::now() })
}

/// Check if `a` is newer than `b` by file modification time.
fn is_newer(a: &Path, b: &Path) -> bool {
    let Ok(ma) = std::fs::metadata(a) else { return false };
    let Ok(mb) = std::fs::metadata(b) else { return true };
    let Ok(ta) = ma.modified() else { return false };
    let Ok(tb) = mb.modified() else { return true };
    ta > tb
}

/// Load dense index: prefer the freshest source (snapshot vs heap).
fn load_dense(db_path: &Path) -> Result<DenseIndex> {
    let snap_path = db_path.join("hnsw.snap");
    let bin_path = db_path.join("hnsw.bin");

    // Use snapshot only if it exists AND is not stale vs .bin
    if snap_path.exists() && !is_newer(&bin_path, &snap_path) {
        let snap = HnswSnapshot::open(&snap_path).context("Failed to open HNSW snapshot")?;
        eprintln!("[daemon] HNSW snapshot loaded: {} vectors (mmap)", snap.len());
        return Ok(DenseIndex::Snapshot(snap));
    }
    let hnsw = HnswGraph::load(&bin_path, NormalizedCosineDistance)
        .context("Failed to load HNSW index")?;
    eprintln!("[daemon] HNSW heap loaded: {} vectors", hnsw.len());
    Ok(DenseIndex::Heap(hnsw))
}

/// Load sparse index: prefer the freshest source (snapshot vs heap).
fn load_sparse(db_path: &Path) -> Result<SparseIndex> {
    let snap_path = db_path.join("bm25.snap");
    let bin_path = db_path.join("bm25.bin");
    let fst_path = db_path.join("bm25_terms.fst");

    // Use snapshot only if it exists AND is not stale vs .bin
    if snap_path.exists() && fst_path.exists() && !is_newer(&bin_path, &snap_path) {
        let snap = Bm25Snapshot::open(db_path).context("Failed to open BM25 snapshot")?;
        eprintln!("[daemon] BM25 snapshot loaded: {} docs (mmap)", snap.total_docs());
        return Ok(SparseIndex::Snapshot(snap));
    }
    let bm25 = Bm25Index::load(&bin_path).context("Failed to load BM25 index")?;
    eprintln!("[daemon] BM25 heap loaded");
    Ok(SparseIndex::Heap(bm25))
}

/// Canonicalize path with a context message.
fn canonicalize(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("Database not found: {}", path.display()))
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

/// Ask the OS to trim the process working set, releasing unmapped pages.
#[cfg(windows)]
fn trim_working_set() {
    #[allow(unsafe_code)]
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
            fn SetProcessWorkingSetSize(process: isize, min: usize, max: usize) -> i32;
        }
        SetProcessWorkingSetSize(GetCurrentProcess(), usize::MAX, usize::MAX);
    }
    eprintln!("[daemon] Working set trimmed");
}

#[cfg(not(windows))]
fn trim_working_set() {}
