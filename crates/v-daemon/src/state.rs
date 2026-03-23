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
use v_hnsw_search::{Bm25Index, Bm25Snapshot, CodeTokenizer, ConvexFusion, KoreanBm25Tokenizer};
use v_hnsw_storage::sq8::Sq8Params;
use v_hnsw_storage::sq8_store::Sq8VectorStore;
use v_hnsw_storage::StorageEngine;

use v_hnsw_search::{QueryCache, SearchResultItem, fusion_alpha, build_results};
use v_hnsw_storage::db_config::DbConfig;
use v_hnsw_storage::distance::{F32Dc, Sq8LutDc};

/// Seconds of idle time before unloading the embedding model.
const MODEL_IDLE_SECS: u64 = 300;

/// Seconds of idle time before evicting a database from memory.
const DB_IDLE_SECS: u64 = 600;

/// Dense index: mmap snapshot or heap graph.
enum DenseIndex {
    Snapshot(HnswSnapshot),
    Heap(HnswGraph<NormalizedCosineDistance>),
}

/// SQ8 quantization data for accelerated search.
struct Sq8Data {
    params: Sq8Params,
    store: Sq8VectorStore,
}

impl DenseIndex {
    fn search(
        &self,
        store: &dyn VectorStore,
        sq8: Option<&Sq8Data>,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        if let Some(sq8) = sq8 {
            let approx = Sq8LutDc::new(&sq8.params, &sq8.store, query);
            let exact = F32Dc { store };
            return match self {
                Self::Snapshot(s) => s.search_two_stage(&approx, &exact, query, k, ef),
                Self::Heap(h) => h.search_two_stage(&approx, &exact, query, k, ef),
            };
        }
        match self {
            Self::Snapshot(s) => s.search_ext(&NormalizedCosineDistance, store, query, k, ef),
            Self::Heap(h) => h.search_ext(store, query, k, ef),
        }
    }
}

/// Sparse index: mmap snapshot or heap index, with content-type-aware tokenizer.
enum SparseIndex {
    Snapshot(Bm25Snapshot, bool),
    HeapDoc(Bm25Index<KoreanBm25Tokenizer>),
    HeapCode(Bm25Index<CodeTokenizer>),
}

macro_rules! sparse_dispatch {
    ($self:expr, $method:ident, $($arg:expr),* $(,)?) => {
        match $self {
            SparseIndex::Snapshot(s, true) => s.$method(&CodeTokenizer::new(), $($arg),*),
            SparseIndex::Snapshot(s, false) => s.$method(&KoreanBm25Tokenizer::new(), $($arg),*),
            SparseIndex::HeapDoc(h) => h.$method($($arg),*),
            SparseIndex::HeapCode(h) => h.$method($($arg),*),
        }
    };
}

impl SparseIndex {
    fn search(&self, query: &str, limit: usize) -> Vec<(PointId, f32)> {
        sparse_dispatch!(self, search, query, limit)
    }

    fn score_documents(&self, query: &str, doc_ids: &[PointId]) -> Vec<(PointId, f32)> {
        sparse_dispatch!(self, score_documents, query, doc_ids)
    }
}

/// Indexes + storage for a single database.
struct DbIndexes {
    dense: DenseIndex,
    sparse: SparseIndex,
    sq8: Option<Sq8Data>,
    engine: StorageEngine,
    last_used: Instant,
}

impl DbIndexes {
    fn search(&mut self, embedding: &[f32], query: &str, k: usize, tags: Vec<String>) -> Result<Vec<SearchResultItem>> {
        let store = self.engine.vector_store();
        let dense_limit = k * 2;
        let sparse_limit = k * 2;

        let dense_results = self.dense
            .search(store, self.sq8.as_ref(), embedding, dense_limit, 200)
            .context("HNSW search failed")?;
        let sparse_results = self.sparse.search(query, sparse_limit);

        let all_sparse = v_hnsw_search::enrich_sparse(&dense_results, sparse_results, |ids| {
            self.sparse.score_documents(query, ids)
        });

        let alpha = fusion_alpha(query);
        let fusion = ConvexFusion::with_alpha(alpha);
        let fetch_k = if tags.is_empty() { k } else { k * 10 };
        let mut results = fusion.fuse(&dense_results, &all_sparse, fetch_k);

        if !tags.is_empty() {
            let payload_store = self.engine.payload_store();
            let allowed: HashSet<_> = payload_store.points_by_tags(&tags).into_iter().collect();
            results.retain(|(id, _)| allowed.contains(id));
            results.truncate(k);
        }

        Ok(build_results(&results, self.engine.payload_store()))
    }
}

/// Search response from daemon.
#[derive(Debug, serde::Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub elapsed_ms: f64,
}

/// Loaded daemon state: shared model + per-DB indexes.
pub struct DaemonState {
    model: Option<Model2VecModel>,
    model_name: Option<String>,
    model_dim: Option<usize>,
    last_embed_at: Instant,
    databases: HashMap<PathBuf, DbIndexes>,
    query_cache: QueryCache,
    /// In-process rust-analyzer instance (spawned once, reused across requests).
    #[cfg(feature = "ra")]
    pub ra: Option<v_lsp::instance::RaInstance>,
}

impl DaemonState {
    pub fn new(initial_db: Option<&Path>) -> Result<Self> {
        let mut state = Self {
            model: None,
            model_name: None,
            model_dim: None,
            last_embed_at: Instant::now(),
            databases: HashMap::new(),
            query_cache: QueryCache::global(),
            #[cfg(feature = "ra")]
            ra: None,
        };

        // Store the initial DB path but don't load it yet.
        // Indexes are loaded lazily on first request (ensure_db).
        if let Some(db_path) = initial_db
            && let Ok(key) = db_path.canonicalize() {
                let config = DbConfig::load(&key).ok();
                if let Some(ref c) = config
                    && let Some(ref name) = c.embed_model {
                        state.model_name = Some(name.clone());
                        state.model_dim = Some(c.dim);
                    }
            }

        eprintln!("[daemon] Idle — indexes load on first request");
        Ok(state)
    }

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

    fn ensure_db(&mut self, db_path: &Path) -> Result<&mut DbIndexes> {
        let key = canonicalize(db_path)?;
        if !self.databases.contains_key(&key) {
            self.register_db(&key)?;
        }
        let db = self.databases.get_mut(&key)
            .context("DB disappeared after register")?;
        db.last_used = Instant::now();
        Ok(db)
    }

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
        self.model.as_ref().context("model not loaded after ensure")
    }

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

    pub fn maybe_unload_model(&mut self) {
        if self.model.is_some() && self.last_embed_at.elapsed().as_secs() > MODEL_IDLE_SECS {
            self.model = None;
            eprintln!("[daemon] Model unloaded (idle > {}s)", MODEL_IDLE_SECS);
            trim_working_set();
        }
    }

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

    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let model = self.ensure_model()?;
        model.embed(texts).context("Failed to embed")
    }

    pub fn reload(&mut self, db_path: &Path) -> Result<()> {
        let t0 = Instant::now();
        let key = canonicalize(db_path)?;
        self.register_db(&key)?;
        eprintln!("[daemon] Reload complete: {:.0}ms", t0.elapsed().as_millis());
        Ok(())
    }

    pub fn save_cache(&self) -> Result<()> { self.query_cache.save() }

    pub fn evict_db(&mut self, db_path: &Path) -> Result<PathBuf> {
        let key = canonicalize(db_path)?;
        self.databases.remove(&key);
        Ok(key)
    }

    pub fn search(&mut self, db_path: &Path, query: &str, k: usize, tags: Vec<String>) -> Result<SearchResponse> {
        let start = Instant::now();
        let embedding = self.resolve_embedding(query)?;
        let db = self.ensure_db(db_path)?;
        let results = db.search(&embedding, query, k, tags)?;
        Ok(SearchResponse { results, elapsed_ms: start.elapsed().as_secs_f64() * 1000.0 })
    }
}

fn load_db(db_path: &Path) -> Result<DbIndexes> {
    let config = DbConfig::load(db_path)?;
    let engine = StorageEngine::open(db_path)
        .with_context(|| format!("Failed to open storage: {}", db_path.display()))?;
    let dense = load_dense(db_path)?;
    let sparse = load_sparse(db_path, config.code)?;
    let sq8 = load_sq8(db_path, engine.vector_store());
    Ok(DbIndexes { dense, sparse, sq8, engine, last_used: Instant::now() })
}

pub fn is_newer(a: &Path, b: &Path) -> bool {
    let Ok(ma) = std::fs::metadata(a) else { return false };
    let Ok(mb) = std::fs::metadata(b) else { return true };
    let Ok(ta) = ma.modified() else { return false };
    let Ok(tb) = mb.modified() else { return true };
    ta > tb
}

fn load_dense(db_path: &Path) -> Result<DenseIndex> {
    let snap_path = db_path.join("hnsw.snap");
    let bin_path = db_path.join("hnsw.bin");

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

fn load_sparse(db_path: &Path, code: bool) -> Result<SparseIndex> {
    let snap_path = db_path.join("bm25.snap");
    let bin_path = db_path.join("bm25.bin");
    let fst_path = db_path.join("bm25_terms.fst");

    if snap_path.exists() && fst_path.exists() && !is_newer(&bin_path, &snap_path) {
        if !code {
            ensure_korean_dict()?;
        }
        let snap = Bm25Snapshot::open(db_path).context("Failed to open BM25 snapshot")?;
        eprintln!("[daemon] BM25 snapshot loaded: {} docs (mmap, code={})", snap.total_docs(), code);
        return Ok(SparseIndex::Snapshot(snap, code));
    }
    if code {
        let bm25: Bm25Index<CodeTokenizer> = Bm25Index::load(&bin_path).context("Failed to load BM25 index")?;
        eprintln!("[daemon] BM25 heap loaded (code)");
        Ok(SparseIndex::HeapCode(bm25))
    } else {
        ensure_korean_dict()?;
        let bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::load(&bin_path).context("Failed to load BM25 index")?;
        eprintln!("[daemon] BM25 heap loaded (document)");
        Ok(SparseIndex::HeapDoc(bm25))
    }
}

fn load_sq8(db_path: &Path, vector_store: &v_hnsw_storage::MmapVectorStore) -> Option<Sq8Data> {
    let params_path = db_path.join("sq8_params.bin");
    let store_path = db_path.join("sq8_vectors.bin");

    if !params_path.exists() || !store_path.exists() {
        return None;
    }

    let params = Sq8Params::load(&params_path).ok()?;
    let mut store = Sq8VectorStore::open(&store_path).ok()?;
    store.restore_id_map(vector_store.id_map());

    eprintln!(
        "[daemon] SQ8 loaded: {} vectors ({}x compression)",
        store.len(),
        if store.dim() > 0 { 4 } else { 1 },
    );

    Some(Sq8Data { params, store })
}

/// Initialize Korean tokenizer if the dictionary is already available.
///
/// Unlike the CLI version, the daemon does not download the dictionary;
/// it simply initializes the tokenizer when the dict exists on disk.
fn ensure_korean_dict() -> Result<()> {
    let dict_path = v_hnsw_core::ko_dic_dir();
    if dict_path.join("dict.da").exists() {
        v_hnsw_search::init_korean_tokenizer(&dict_path)
            .map_err(|e| anyhow::anyhow!("Failed to initialize Korean tokenizer: {e}"))?;
    }
    Ok(())
}

fn canonicalize(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("Database not found: {}", path.display()))
}

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
