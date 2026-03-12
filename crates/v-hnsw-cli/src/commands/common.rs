//! Shared utilities for CLI commands.
//!
//! Database infrastructure, model management, and progress bar.
//! Domain-specific utilities are in dedicated modules:
//! - [`super::file_utils`]: path normalization, hashing, file scanning
//! - [`super::ingest`]: embedding pipeline, payload building, batch insert
//! - [`super::search_result`]: search result formatting, language detection

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_storage::{StorageConfig, StorageEngine};

use super::db_config::DbConfig;
use super::dict;

// ── Re-exports for backwards compatibility ──────────────────────────────

pub use super::file_utils::{
    content_hash, generate_id, get_file_mtime, normalize_source, scan_files, should_skip_dir,
};
// Used only by tests (via `use common::*`)
#[cfg(test)]
pub use super::file_utils::content_hash_bytes;
pub use super::indexing::{build_indexes, update_indexes_incremental};
pub use super::ingest::{
    IngestRecord, embed_and_insert, embed_sorted, make_payload, truncate_for_embed,
};
pub use super::query_cache::QueryCache;
pub use super::search_result::{build_results, fusion_alpha, SearchResultItem};
// Used only by tests (via `use common::*`)
#[cfg(test)]
pub use super::search_result::has_korean;

// ── Cache / path utilities ──────────────────────────────────────────────

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

/// Validate that a database directory exists; bail with a standard message if not.
pub fn require_db(path: &Path) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }
    Ok(())
}

// ── Process management ──────────────────────────────────────────────────

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

// ── Model / Korean dict ─────────────────────────────────────────────────

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
    println!("Model: {} (dim={})", DEFAULT_MODEL, "loading...");

    let model = Model2VecModel::from_pretrained(DEFAULT_MODEL)
        .context("Failed to load model2vec model")?;

    tracing::info!(dim = model.dim(), "Model loaded");
    Ok(model)
}

// ── Progress bar ────────────────────────────────────────────────────────

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

// ── Database management ─────────────────────────────────────────────────

/// Auto-create database if it doesn't exist, or open existing with exclusive lock.
///
/// `code` marks the database as a code-intelligence DB (uses `CodeTokenizer` for BM25).
pub fn ensure_database(
    path: &Path,
    dim: usize,
    model_name: &str,
    korean: bool,
    code: bool,
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
        println!("New database: {} (dim={dim})", path.display());

        let storage_config = StorageConfig {
            dim,
            initial_capacity: 10_000,
            checkpoint_threshold: 50_000,
        };

        let engine = StorageEngine::create(path, storage_config)
            .with_context(|| format!("Failed to create storage at {}", path.display()))?;

        let db_config = DbConfig {
            dim,
            korean,
            code,
            embed_model: Some(model_name.to_string()),
            ..DbConfig::default()
        };
        db_config.save(path)?;

        Ok(engine)
    }
}
