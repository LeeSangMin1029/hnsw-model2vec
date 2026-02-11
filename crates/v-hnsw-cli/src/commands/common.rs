//! Shared utilities for CLI commands.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use lru::LruCache;
use v_hnsw_core::{Payload, PayloadStore, PayloadValue, VectorIndex};
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{NormalizedCosineDistance, HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
use v_hnsw_storage::{StorageConfig, StorageEngine};

use super::create::DbConfig;
use super::dict;
use crate::is_interrupted;

/// Platform-aware cache directory for v-hnsw.
pub fn cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local).join("v-hnsw").join("cache");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(cache).join("v-hnsw");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache").join("v-hnsw");
        }
    }
    std::env::temp_dir().join("v-hnsw")
}

/// Get a file path inside the cache directory, creating it if needed.
pub fn cache_file(name: &str) -> PathBuf {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).ok();
    dir.join(name)
}

/// Spawn a detached background process with stdin/stdout/stderr suppressed.
pub fn spawn_detached(args: &[&str]) -> Result<()> {
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x00000200 | 0x08000000);
    }

    cmd.spawn().context("Failed to spawn detached process")?;
    Ok(())
}

/// Ensure Korean dictionary is available and initialize the tokenizer.
///
/// Downloads and builds the ko-dic dictionary on first run, then
/// initializes the global Korean tokenizer. Safe to call multiple times.
pub fn ensure_korean_dict() -> Result<()> {
    let dict_path = dict::ensure_ko_dic()?;
    v_hnsw_search::init_korean_tokenizer(&dict_path)
        .map_err(|e| anyhow::anyhow!("Failed to initialize Korean tokenizer: {e}"))
}

/// Default model2vec model ID.
pub const DEFAULT_MODEL: &str = "minishlab/potion-multilingual-128M";

/// Create embedding model (model2vec).
pub fn create_model() -> Result<Model2VecModel> {
    tracing::info!(model = DEFAULT_MODEL, "Loading model2vec model");
    println!("Loading model2vec model: {}", DEFAULT_MODEL);

    let model = Model2VecModel::from_pretrained(DEFAULT_MODEL)
        .context("Failed to load model2vec model")?;

    tracing::info!(dim = model.dim(), "Model loaded");
    println!("Model loaded (dim={}).", model.dim());
    Ok(model)
}

/// Build progress bar with standard template.
pub fn make_progress_bar(total: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
            .map_err(|e| anyhow::anyhow!("Invalid progress template: {e}"))?
            .progress_chars("#>-"),
    );
    Ok(pb)
}

/// Auto-create database if it doesn't exist, or open existing with exclusive lock.
pub fn ensure_database(
    path: &Path,
    dim: usize,
    model_name: &str,
    korean: bool,
) -> Result<StorageEngine> {
    if path.exists() {
        let config = DbConfig::load(path)?;
        if config.dim != dim {
            anyhow::bail!(
                "Dimension mismatch: database has dim={}, but model produces dim={}",
                config.dim, dim
            );
        }
        // Update embed_model if not set
        if config.embed_model.is_none() {
            let mut config = config;
            config.embed_model = Some(model_name.to_string());
            config.save(path)?;
        }
        StorageEngine::open_exclusive(path)
            .with_context(|| format!("Failed to open database at {}", path.display()))
    } else {
        tracing::info!(path = %path.display(), dim, "Creating new database");
        println!("Creating new database at {}", path.display());

        let storage_config = StorageConfig {
            dim,
            initial_capacity: 10_000,
            checkpoint_threshold: 50_000,
        };

        let engine = StorageEngine::create(path, storage_config)
            .with_context(|| format!("Failed to create storage at {}", path.display()))?;

        let db_config = DbConfig {
            version: DbConfig::CURRENT_VERSION,
            dim,
            metric: "cosine".to_string(),
            m: 16,
            ef_construction: 200,
            korean,
            embed_model: Some(model_name.to_string()),
        };
        db_config.save(path)?;

        println!("  Dimension:  {dim}");
        println!("  Metric:     cosine");
        println!("  M:          16");
        println!("  ef:         200");
        println!("  Model:      {model_name}");
        println!();

        Ok(engine)
    }
}

/// Build payload from source info.
pub fn make_payload(
    source: &str,
    title: Option<&str>,
    tags: &[String],
    chunk_index: usize,
    chunk_total: usize,
    source_modified_at: u64,
) -> Payload {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut custom = HashMap::new();
    if let Some(t) = title {
        custom.insert("title".to_string(), PayloadValue::String(t.to_string()));
    }

    Payload {
        source: source.to_string(),
        tags: tags.to_vec(),
        created_at: now,
        source_modified_at,
        chunk_index: chunk_index as u32,
        chunk_total: chunk_total as u32,
        custom,
    }
}

/// Normalize a file path to a canonical forward-slash form.
///
/// Windows produces mixed separators (`C:/foo\bar\baz.md`) depending on
/// how the path was constructed. This normalizes to forward slashes so
/// that the same file always gets the same source string and ID.
pub fn normalize_source(path: &Path) -> String {
    // canonicalize resolves symlinks and produces \\?\ prefix on Windows
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let s = abs.to_string_lossy();
    // Strip Windows \\?\ prefix, normalize separators
    let s = s.strip_prefix(r"\\?\").unwrap_or(&s);
    s.replace('\\', "/")
}

/// Generate a stable ID from source path and chunk index.
pub fn generate_id(source: &str, chunk_index: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    chunk_index.hash(&mut hasher);
    hasher.finish()
}

/// Embed texts with length-sorted batching to minimize padding waste.
pub fn embed_sorted(model: &dyn EmbeddingModel, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    // Sort indices by text length
    let mut indices: Vec<usize> = (0..texts.len()).collect();
    indices.sort_by_key(|&i| texts[i].len());

    let sorted: Vec<&str> = indices.iter().map(|&i| texts[i].as_str()).collect();
    let sorted_embs = model
        .embed(&sorted)
        .map_err(|e| anyhow::anyhow!("Embedding failed: {e}"))?;

    // Restore original order (consume sorted_embs to avoid clone)
    let mut embeddings = vec![Vec::new(); texts.len()];
    for (emb, &orig_idx) in sorted_embs.into_iter().zip(indices.iter()) {
        embeddings[orig_idx] = emb;
    }
    Ok(embeddings)
}

/// Build and save HNSW and BM25 indexes from storage data.
pub fn build_indexes(path: &Path, engine: &StorageEngine, config: &DbConfig) -> Result<()> {
    if engine.is_empty() {
        println!("No vectors to index, skipping index building.");
        return Ok(());
    }

    tracing::info!("Building indexes");
    println!();
    println!("Building indexes...");

    // Build HNSW graph
    let hnsw_config = HnswConfig::builder()
        .dim(config.dim)
        .m(config.m)
        .ef_construction(config.ef_construction)
        .build()
        .with_context(|| "Failed to create HNSW config")?;

    let hnsw_path = path.join("hnsw.bin");
    let vector_store = engine.vector_store();

    println!(
        "  Building HNSW graph (M={}, ef_construction={})...",
        config.m, config.ef_construction
    );

    let pb = make_progress_bar(vector_store.id_map().keys().len() as u64)?;

    let mut hnsw = HnswGraph::new(hnsw_config, NormalizedCosineDistance);
    for id in vector_store.id_map().keys() {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted during HNSW build");
            return Ok(());
        }
        let _ = hnsw.build_insert(vector_store, *id);
        pb.inc(1);
    }

    pb.finish_with_message("HNSW build complete");

    hnsw.save(&hnsw_path)
        .with_context(|| format!("Failed to save HNSW graph to {}", hnsw_path.display()))?;
    println!("  HNSW graph saved: {}", hnsw_path.display());

    // Build BM25 index
    println!("  Building BM25 index...");
    ensure_korean_dict()?;
    let bm25_path = path.join("bm25.bin");
    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::new(KoreanBm25Tokenizer::new());
    let payload_store = engine.payload_store();

    let pb = make_progress_bar(vector_store.id_map().keys().len() as u64)?;

    for id in vector_store.id_map().keys() {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted during BM25 build");
            return Ok(());
        }
        if let Ok(Some(text)) = payload_store.get_text(*id) {
            bm25.add_document(*id, &text);
        }
        pb.inc(1);
    }

    pb.finish_with_message("BM25 build complete");

    bm25.save(&bm25_path)
        .with_context(|| format!("Failed to save BM25 index to {}", bm25_path.display()))?;
    println!("  BM25 index saved: {}", bm25_path.display());

    tracing::info!("Index building completed");
    println!("Index building completed.");
    Ok(())
}

/// Incrementally update HNSW and BM25 indexes for changed IDs only.
///
/// Falls back to full rebuild if index files don't exist yet.
pub fn update_indexes_incremental(
    path: &Path,
    engine: &StorageEngine,
    config: &DbConfig,
    added_ids: &[u64],
    removed_ids: &[u64],
) -> Result<()> {
    let hnsw_path = path.join("hnsw.bin");
    let bm25_path = path.join("bm25.bin");

    // Fallback to full rebuild if index files missing
    if !hnsw_path.exists() || !bm25_path.exists() {
        tracing::info!("Index files missing, falling back to full rebuild");
        println!("Index files not found, performing full rebuild...");
        return build_indexes(path, engine, config);
    }

    let total_changes = added_ids.len() + removed_ids.len();
    tracing::info!(added = added_ids.len(), removed = removed_ids.len(), "Incremental index update");
    println!("Updating indexes incrementally ({} additions, {} removals)...", added_ids.len(), removed_ids.len());

    // --- HNSW incremental update ---
    let mut hnsw: HnswGraph<NormalizedCosineDistance> = HnswGraph::load(&hnsw_path, NormalizedCosineDistance)
        .with_context(|| "Failed to load HNSW graph")?;

    let vector_store = engine.vector_store();

    for &id in removed_ids {
        // soft-delete; ignore PointNotFound (already removed from storage)
        let _ = hnsw.delete(id);
    }

    for &id in added_ids {
        let _ = hnsw.build_insert(vector_store, id);
    }

    hnsw.save(&hnsw_path)
        .with_context(|| "Failed to save HNSW graph")?;
    println!("  HNSW graph updated ({total_changes} changes).");

    // --- BM25 incremental update ---
    // Only init Korean dict if there are BM25 changes
    if total_changes > 0 {
        ensure_korean_dict()?;
    }

    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::load_mutable(&bm25_path)
        .with_context(|| "Failed to load BM25 index")?;

    let payload_store = engine.payload_store();

    for &id in removed_ids {
        bm25.remove_document(id);
    }

    for &id in added_ids {
        if let Ok(Some(text)) = payload_store.get_text(id) {
            bm25.add_document(id, &text);
        }
    }

    bm25.save(&bm25_path)
        .with_context(|| "Failed to save BM25 index")?;
    println!("  BM25 index updated ({total_changes} changes).");

    tracing::info!("Incremental index update completed");
    Ok(())
}

/// Search result common to both find and serve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResultItem {
    pub id: u64,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Build search result items from raw (id, score) pairs.
pub fn build_results(
    results: &[(u64, f32)],
    payload_store: &dyn PayloadStore,
) -> Vec<SearchResultItem> {
    let max_score = results.first().map(|(_, s)| *s).unwrap_or(1.0);

    results
        .iter()
        .map(|&(id, score)| {
            let text = payload_store.get_text(id).ok().flatten();
            let payload = payload_store.get_payload(id).ok().flatten();
            let source = payload
                .as_ref()
                .map(|p| p.source.clone())
                .filter(|s: &String| !s.is_empty());
            let title = payload
                .as_ref()
                .and_then(|p| p.custom.get("title"))
                .and_then(|v| match v {
                    PayloadValue::String(s) => Some(s.clone()),
                    _ => None,
                });
            let url = payload
                .as_ref()
                .and_then(|p| p.custom.get("url"))
                .and_then(|v| match v {
                    PayloadValue::String(s) => Some(s.clone()),
                    _ => None,
                });

            SearchResultItem {
                id,
                score: if max_score > 0.0 {
                    score / max_score
                } else {
                    0.0
                },
                text,
                source,
                title,
                url,
            }
        })
        .collect()
}

/// Check if query contains Korean (Hangul) characters.
///
/// Returns `true` if any character is in the Hangul Syllables or Jamo range.
pub fn has_korean(text: &str) -> bool {
    text.chars().any(|c| {
        ('\u{AC00}'..='\u{D7AF}').contains(&c)  // Hangul Syllables
            || ('\u{1100}'..='\u{11FF}').contains(&c) // Hangul Jamo
            || ('\u{3130}'..='\u{318F}').contains(&c) // Hangul Compatibility Jamo
    })
}

/// Fusion alpha for convex combination, adjusted by query language.
///
/// Returns alpha in [0, 1]: higher = more weight on dense (vector) search.
pub fn fusion_alpha(query: &str) -> f32 {
    if has_korean(query) {
        0.4 // Korean: BM25 형태소 매칭이 더 중요
    } else {
        0.7 // English: 벡터 유사도 우선
    }
}

/// Max character length for text sent to embedding model.
const EMBED_MAX_CHARS: usize = 8000;

/// Truncate text at a char boundary for embedding.
///
/// Returns a new string if truncation occurs, otherwise the original slice.
pub fn truncate_for_embed(text: &str) -> &str {
    if text.len() <= EMBED_MAX_CHARS {
        return text;
    }
    let mut end = EMBED_MAX_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// A record for batch ingestion (shared by add and update commands).
#[derive(Clone)]
pub struct IngestRecord {
    pub id: u64,
    pub text: String,
    pub source: String,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub chunk_index: usize,
    pub chunk_total: usize,
    pub source_modified_at: u64,
}

/// Get the file modification time as seconds since UNIX epoch.
pub fn get_file_mtime(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}

/// Build an HNSW index from a slice of embeddings.
pub fn build_hnsw_index(
    embeddings: &[Vec<f32>],
    dim: usize,
) -> Result<HnswGraph<NormalizedCosineDistance>> {
    let config = HnswConfig::builder()
        .dim(dim)
        .m(16)
        .ef_construction(200)
        .build()
        .context("Failed to create HNSW config")?;

    let mut hnsw: HnswGraph<NormalizedCosineDistance> = HnswGraph::new(config, NormalizedCosineDistance);

    let pb = ProgressBar::new(embeddings.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} vectors")
            .ok()
            .unwrap_or_else(ProgressStyle::default_bar)
            .progress_chars("#>-"),
    );

    for (id, embedding) in embeddings.iter().enumerate() {
        if is_interrupted() {
            pb.finish_with_message("Interrupted");
            break;
        }

        hnsw.insert(id as u64, embedding)
            .context("Failed to insert vector")?;
        pb.inc(1);
    }

    pb.finish_with_message("Done");
    Ok(hnsw)
}

// ============================================================================
// Query Embedding Cache
// ============================================================================

const QUERY_CACHE_MAX: usize = 1000;
const QUERY_CACHE_FILE: &str = "query_cache.bin";

/// LRU cache for query embeddings, with disk persistence.
///
/// Stores query text → embedding vector mappings to skip model inference
/// on repeated queries. Persisted to disk so cache survives daemon restarts.
pub struct QueryCache {
    cache: LruCache<String, Vec<f32>>,
    db_path: PathBuf,
}

/// On-disk format: Vec of (query, embedding) pairs in LRU order.
#[derive(serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    query: String,
    embedding: Vec<f32>,
}

impl QueryCache {
    /// Global cache (not tied to a specific DB).
    pub fn global() -> Self {
        Self::load(&cache_dir())
    }

    /// Load cache from disk, or create empty if not found.
    pub fn load(db_path: &Path) -> Self {
        let cap = NonZeroUsize::new(QUERY_CACHE_MAX)
            .unwrap_or(NonZeroUsize::MIN);
        let mut cache = LruCache::new(cap);

        let cache_path = db_path.join(QUERY_CACHE_FILE);
        if let Ok(data) = std::fs::read(&cache_path) {
            if let Ok((entries, _)) = bincode::serde::decode_from_slice::<Vec<CacheEntry>, _>(
                &data,
                bincode::config::standard(),
            ) {
                // Insert in reverse to preserve LRU order (most recent last)
                for entry in entries.into_iter().rev() {
                    cache.put(entry.query, entry.embedding);
                }
                tracing::debug!(count = cache.len(), "Query cache loaded");
            }
        }

        Self {
            cache,
            db_path: db_path.to_path_buf(),
        }
    }

    /// Get cached embedding for a query (promotes to most-recently-used).
    pub fn get(&mut self, query: &str) -> Option<&Vec<f32>> {
        self.cache.get(query)
    }

    /// Insert a query→embedding mapping.
    pub fn insert(&mut self, query: String, embedding: Vec<f32>) {
        self.cache.put(query, embedding);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Save cache to disk.
    pub fn save(&self) -> Result<()> {
        let entries: Vec<CacheEntry> = self.cache.iter()
            .map(|(q, e)| CacheEntry {
                query: q.clone(),
                embedding: e.clone(),
            })
            .collect();

        let encoded = bincode::serde::encode_to_vec(&entries, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("Failed to encode query cache: {e}"))?;

        let cache_path = self.db_path.join(QUERY_CACHE_FILE);
        std::fs::write(&cache_path, encoded)
            .with_context(|| format!("Failed to write query cache to {}", cache_path.display()))?;

        tracing::debug!(count = self.len(), "Query cache saved");
        Ok(())
    }
}
