//! Update command - Incremental indexing of markdown and code files.
//!
//! Compares modification times and file sizes against a stored file index
//! to detect new, modified, and deleted files. Re-processes only changed files.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use crate::chunk::{ChunkConfig, MarkdownChunker};
use v_hnsw_embed::Model2VecModel;
use v_hnsw_storage::StorageEngine;

use super::common;
use super::create::DbConfig;
use super::file_index;
use crate::is_interrupted;

/// Collected IDs for incremental index updates.
#[derive(Default)]
struct ChangeSet {
    added: Vec<u64>,
    removed: Vec<u64>,
}

/// Statistics for the update operation.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct UpdateStats {
    pub new: usize,
    pub modified: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub hash_skipped: usize,
}

/// Run the update command - incremental indexing (CLI entry point).
pub fn run(db_path: PathBuf, input_path: Option<PathBuf>, exclude: &[String]) -> Result<()> {
    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Use 'add' command to create a new database.",
            db_path.display()
        );
    }

    // Resolve input_path: explicit arg > config.input_path
    let input_path = match input_path {
        Some(p) => p,
        None => {
            let config = super::create::DbConfig::load(&db_path)?;
            match config.input_path {
                Some(ref p) => PathBuf::from(p),
                None => anyhow::bail!(
                    "No input path specified and none stored in config.\n\
                     Usage: v-hnsw update <DB> <INPUT>"
                ),
            }
        }
    };

    if !input_path.is_dir() {
        anyhow::bail!(
            "Input path must be a directory containing markdown files: {}",
            input_path.display()
        );
    }

    // Try daemon first (avoids 1GB model reload)
    if let Ok(stats) = try_daemon_update(&db_path, &input_path, exclude) {
        print_stats(&stats, std::time::Instant::now());
        return Ok(());
    }

    tracing::info!(
        db = %db_path.display(),
        input = %input_path.display(),
        "Starting update command"
    );
    println!("Updating database: {}", db_path.display());
    println!("Input folder:      {}", input_path.display());
    println!();

    let start = Instant::now();

    // No shared model — run_core will load on demand
    let stats = run_core(&db_path, &input_path, None, exclude)?;

    // Notify daemon to reload if running
    if let Ok(()) = super::serve::notify_daemon_reload(&db_path) {
        println!("Daemon notified to reload indexes.");
    }

    print_stats(&stats, start);
    Ok(())
}

/// Core update logic shared by CLI and daemon.
///
/// Pass `shared_model` to reuse an existing model (daemon path).
/// Pass `None` to load a fresh model on demand (CLI fallback path).
pub(crate) fn run_core(
    db_path: &Path,
    input_path: &Path,
    shared_model: Option<&Model2VecModel>,
    exclude: &[String],
) -> Result<UpdateStats> {
    let mut file_index = file_index::load_file_index(db_path)?;
    eprintln!(
        "Loaded file index with {} tracked files",
        file_index.files.len()
    );

    let mut engine = StorageEngine::open_exclusive(db_path)
        .with_context(|| format!("Failed to open database at {}", db_path.display()))?;

    // Scan input folder for all supported files (markdown + code)
    let all_files = common::scan_files(input_path, exclude, |ext| {
        ext == "md"
            || ext == "markdown"
            || is_supported_text_file(ext)
    });

    if all_files.is_empty() {
        eprintln!("No supported files found in {}", input_path.display());
        return Ok(UpdateStats::default());
    }

    eprintln!("Scanning {} files (markdown + code)...", all_files.len());

    let mut stats = UpdateStats::default();
    let mut seen_files = std::collections::HashSet::new();
    let mut changes = ChangeSet::default();

    let chunker = MarkdownChunker::new(ChunkConfig::default());

    // First pass: detect changes (mtime/size → content hash two-stage)
    let mut pending_files: Vec<(PathBuf, String, u64, u64, u64)> = Vec::new();

    for file_path in &all_files {
        let source = common::normalize_source(file_path);
        seen_files.insert(source.clone());

        let mtime = common::get_file_mtime(file_path).unwrap_or(0);
        let size = file_index::get_file_size(file_path).unwrap_or(0);

        if file_index.is_modified(&source, mtime, size) {
            let hash = common::content_hash(file_path).unwrap_or(0);
            let old_hash = file_index.get_file(&source).and_then(|m| m.content_hash);

            if old_hash == Some(hash) {
                if let Some(meta) = file_index.get_file(&source) {
                    file_index.update_file_with_hash(
                        source,
                        mtime,
                        size,
                        meta.chunk_ids.clone(),
                        hash,
                    );
                }
                stats.hash_skipped += 1;
            } else {
                pending_files.push((file_path.clone(), source, mtime, size, hash));
            }
        } else {
            stats.unchanged += 1;
        }
    }

    // Detect deleted files
    let input_prefix = common::normalize_source(input_path);
    let deleted_files: Vec<String> = file_index
        .files
        .keys()
        .filter(|path| path.starts_with(&input_prefix) && !seen_files.contains(*path))
        .cloned()
        .collect();

    // Early exit: no changes at all
    if pending_files.is_empty() && deleted_files.is_empty() {
        file_index::save_file_index(db_path, &file_index)?;
        return Ok(stats);
    }

    // Load model only when there are files to process
    let owned_model;
    let model: Option<&Model2VecModel> = if !pending_files.is_empty() {
        if let Some(m) = shared_model {
            Some(m)
        } else {
            owned_model = common::create_model()?;
            Some(&owned_model)
        }
    } else {
        None
    };

    // Process changed files
    for (file_path, source, mtime, size, hash) in &pending_files {
        if is_interrupted() {
            break;
        }

        let old_meta = file_index.get_file(source);

        if let Some(meta) = old_meta {
            eprintln!("Modified: {}", file_path.display());
            let old_ids = meta.chunk_ids.clone();
            for id in &old_ids {
                let _ = engine.remove(*id);
            }
            changes.removed.extend_from_slice(&old_ids);
            stats.modified += 1;
        } else {
            eprintln!("New:      {}", file_path.display());
            stats.new += 1;
        }

        let (records, chunk_ids) =
            super::add::ingest::chunk_single_file(file_path, source, *mtime, &chunker);

        if !records.is_empty() {
            #[allow(clippy::unwrap_used)]
            common::embed_and_insert(&records, model.unwrap(), &mut engine)?;
            changes.added.extend(chunk_ids.iter());
        }

        file_index.update_file_with_hash(source.clone(), *mtime, *size, chunk_ids, *hash);
    }

    // Process deleted files
    if !deleted_files.is_empty() {
        eprintln!("Deleting {} removed files:", deleted_files.len());
        for path in &deleted_files {
            eprintln!("Deleted:  {path}");
            if let Some(meta) = file_index.get_file(path) {
                for id in &meta.chunk_ids {
                    let _ = engine.remove(*id);
                }
                changes.removed.extend_from_slice(&meta.chunk_ids);
            }
            file_index.files.remove(path);
            stats.deleted += 1;
        }
    }

    engine
        .checkpoint()
        .with_context(|| "Failed to checkpoint database")?;

    file_index::save_file_index(db_path, &file_index)?;

    // Incremental index update instead of full rebuild
    let config = DbConfig::load(db_path)?;
    common::update_indexes_incremental(
        db_path,
        &engine,
        &config,
        &changes.added,
        &changes.removed,
    )?;

    Ok(stats)
}

/// Check if a file extension is a supported text-based source file
/// (not Rust code, but still worth indexing as plain text chunks).
pub(crate) fn is_supported_text_file(ext: &str) -> bool {
    matches!(
        ext,
        "ts" | "tsx" | "js" | "jsx" | "svelte" | "vue"
            | "py" | "go" | "java" | "c" | "cpp" | "h" | "hpp"
            | "toml" | "yaml" | "yml" | "json"
    )
}

/// Print update statistics.
fn print_stats(stats: &UpdateStats, start: Instant) {
    let elapsed = start.elapsed();
    tracing::info!(
        new = stats.new,
        modified = stats.modified,
        deleted = stats.deleted,
        unchanged = stats.unchanged,
        hash_skipped = stats.hash_skipped,
        elapsed_secs = elapsed.as_secs_f64(),
        "Update completed"
    );
    println!();
    println!("Update completed:");
    println!("  New files:       {}", stats.new);
    println!("  Modified files:  {}", stats.modified);
    println!("  Deleted files:   {}", stats.deleted);
    println!("  Unchanged files: {}", stats.unchanged);
    if stats.hash_skipped > 0 {
        println!("  Hash-skipped:    {} (mtime changed, content identical)", stats.hash_skipped);
    }
    println!("  Elapsed:         {:.2}s", elapsed.as_secs_f64());
}

/// Try to delegate update to running daemon (avoids 1GB model reload).
fn try_daemon_update(db_path: &Path, input_path: &Path, exclude: &[String]) -> Result<UpdateStats> {
    let canonical_db = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;
    let canonical_input = input_path
        .canonicalize()
        .with_context(|| format!("Input not found: {}", input_path.display()))?;

    let params = serde_json::json!({
        "db": canonical_db.to_str().unwrap_or(""),
        "input": canonical_input.to_str().unwrap_or(""),
        "exclude": exclude
    });

    // Update can take a long time — generous read timeout (300s)
    let result = super::serve::daemon_rpc("update", params, 300)?;

    let stats: UpdateStats = serde_json::from_value(result)
        .context("Failed to parse update stats")?;

    println!("Update completed via daemon (no extra process).");
    Ok(stats)
}

