//! Shared utilities for CLI commands.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use v_hnsw_core::{Payload, PayloadStore, PayloadValue, VectorIndex, VectorStore};
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{CosineDistance, HnswConfig, HnswGraph};
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

/// Auto-create database if it doesn't exist, or open existing.
pub fn ensure_database(path: &Path, dim: usize, model_name: &str) -> Result<StorageEngine> {
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
        StorageEngine::open(path)
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
            korean: true,
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

    // Restore original order
    let mut embeddings = vec![Vec::new(); texts.len()];
    for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
        embeddings[orig_idx] = sorted_embs[sorted_idx].clone();
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

    let mut hnsw = HnswGraph::new(hnsw_config, CosineDistance);
    for id in vector_store.id_map().keys() {
        if is_interrupted() {
            pb.abandon_with_message("Interrupted during HNSW build");
            return Ok(());
        }
        if let Ok(vec) = vector_store.get(*id) {
            let _ = hnsw.insert(*id, vec);
        }
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
    let mut hnsw: HnswGraph<CosineDistance> = HnswGraph::load(&hnsw_path, CosineDistance)
        .with_context(|| "Failed to load HNSW graph")?;

    let vector_store = engine.vector_store();

    for &id in removed_ids {
        // soft-delete; ignore PointNotFound (already removed from storage)
        let _ = hnsw.delete(id);
    }

    for &id in added_ids {
        if let Ok(vec) = vector_store.get(id) {
            let _ = hnsw.insert(id, vec);
        }
    }

    hnsw.save(&hnsw_path)
        .with_context(|| "Failed to save HNSW graph")?;
    println!("  HNSW graph updated ({total_changes} changes).");

    // --- BM25 incremental update ---
    // Only init Korean dict if there are BM25 changes
    if total_changes > 0 {
        ensure_korean_dict()?;
    }

    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::load(&bm25_path)
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
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SearchResultItem {
    pub id: u64,
    pub score: f32,
    pub text: Option<String>,
    pub source: Option<String>,
    pub title: Option<String>,
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

/// Get the file modification time as seconds since UNIX epoch.
pub fn get_file_mtime(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}
