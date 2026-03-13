//! Code-specific add/update command.
//!
//! Chunks code files via tree-sitter, embeds with Model2Vec,
//! and stores in a v-hnsw database. Uses shared infrastructure
//! from v-hnsw-cli for ingestion, payload building, and file indexing.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};

use v_code_chunk as chunk_code;
use v_hnsw_cli::commands::common;
use v_hnsw_cli::commands::db_config::DbConfig;
use v_hnsw_cli::commands::file_index;
use v_hnsw_cli::commands::file_utils::scan_files;
use v_hnsw_cli::commands::ingest::{CodeChunkEntry, entries_to_records};
use v_hnsw_cli::commands::pipeline::process_records;
use v_hnsw_cli::is_interrupted;

// ── Model loading ────────────────────────────────────────────────────────

/// Load the embedding model (Model2Vec / potion).
fn load_code_model() -> Result<Model2VecModel> {
    let model = Model2VecModel::new()
        .context("Failed to load Model2Vec embedding model")?;

    println!("Embed model: {} (dim={})", model.name(), model.dim());
    Ok(model)
}


// ── Public entry points ──────────────────────────────────────────────────

/// Run the v-code add command.
pub fn run(db_path: PathBuf, input_path: PathBuf, exclude: &[String]) -> Result<()> {
    println!("Indexing code: {}", input_path.display());
    println!("Database:      {}", db_path.display());

    // Scan for code files
    let code_files = scan_files(&input_path, exclude, chunk_code::is_supported_code_file);
    if code_files.is_empty() {
        anyhow::bail!(
            "No supported code files found in {}",
            input_path.display()
        );
    }

    // Collect language stats
    let mut lang_counts: HashMap<&str, usize> = HashMap::new();
    for f in &code_files {
        let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("");
        let lang = chunk_code::lang_for_extension(ext).unwrap_or("other");
        *lang_counts.entry(lang).or_default() += 1;
    }
    let mut lang_summary: Vec<_> = lang_counts.iter().collect();
    lang_summary.sort_by(|a, b| b.1.cmp(a.1));
    let summary: Vec<String> = lang_summary.iter().map(|(l, n)| format!("{l}:{n}")).collect();
    println!("Files: {} ({})", code_files.len(), summary.join(", "));

    // Load model
    let model = load_code_model()?;

    // Open/create database
    let mut engine = common::ensure_database(&db_path, model.dim(), model.name(), false, true)?;

    // Update config
    if let Ok(mut config) = DbConfig::load(&db_path) {
        config.code = true;
        if let Ok(canonical) = input_path.canonicalize() {
            config.input_path = Some(canonical.to_string_lossy().into_owned());
        }
        let _ = config.save(&db_path);
    }

    // === Pass 1: Chunk all files ===
    let mut entries: Vec<CodeChunkEntry> = Vec::new();
    let mut file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new();
    v_hnsw_cli::commands::ingest::chunk_code_files(
        &code_files,
        is_interrupted,
        &mut entries,
        &mut file_metadata_map,
    );

    // === Pass 2+3: Build called_by index + generate IngestRecords ===
    let records = entries_to_records(&entries);
    println!("Symbols: {} (functions, structs, enums, ...)", records.len());

    // === Remove stale chunks for re-added files ===
    let file_idx = file_index::load_file_index(&db_path)?;
    let mut removed_ids = Vec::new();
    for (path, (_, _, new_ids)) in &file_metadata_map {
        if let Some(existing) = file_idx.get_file(path) {
            for &old_id in &existing.chunk_ids {
                if !new_ids.contains(&old_id) {
                    let _ = engine.remove(old_id);
                    removed_ids.push(old_id);
                }
            }
        }
    }
    if !removed_ids.is_empty() {
        engine.checkpoint().ok();
        println!("Removed {} stale symbols", removed_ids.len());
    }

    // === Embed + insert ===
    let (inserted, _errors, _inserted_ids) = process_records(records, &model, &mut engine)?;

    // === Update file index ===
    let mut file_idx = file_index::load_file_index(&db_path)?;
    for (path, (mtime, size, chunk_ids)) in file_metadata_map {
        file_idx.update_file(path, mtime, size, chunk_ids);
    }
    file_index::save_file_index(&db_path, &file_idx)?;

    if is_interrupted() {
        println!();
        println!("Operation interrupted. Partial data may have been inserted.");
        return Ok(());
    }

    if inserted == 0 {
        println!("No symbols to index.");
    } else {
        // Build HNSW + BM25 indexes
        let config = DbConfig::load(&db_path)?;
        v_hnsw_cli::commands::indexing::update_indexes_incremental(
            &db_path, &engine, &config, &_inserted_ids, &removed_ids,
        )?;

        // Notify daemon to reload if running
        if v_hnsw_cli::commands::serve::notify_daemon_reload(&db_path).is_ok() {
            println!("Daemon notified to reload indexes.");
        }

        println!();
        println!("Done! Code DB ready: {}", db_path.display());
        println!("Use: v-code find/symbols/def/refs/impact/gather/dupes {}", db_path.display());
    }

    Ok(())
}

/// Run the v-code update command (incremental: only re-processes changed files).
pub fn run_update(db_path: PathBuf, input_path: PathBuf, exclude: &[String]) -> Result<()> {
    use v_hnsw_cli::commands::file_utils::get_file_mtime;

    let all_files = scan_files(&input_path, exclude, chunk_code::is_supported_code_file);
    if all_files.is_empty() {
        anyhow::bail!("No supported code files found in {}", input_path.display());
    }

    let file_idx = file_index::load_file_index(&db_path)?;

    let changed_files: Vec<_> = all_files
        .into_iter()
        .filter(|f| {
            let source = v_hnsw_cli::commands::file_utils::normalize_source(f);
            match file_idx.get_file(&source) {
                Some(entry) => {
                    get_file_mtime(f).is_none_or(|m| m != entry.mtime)
                }
                None => true, // new file
            }
        })
        .collect();

    if changed_files.is_empty() {
        println!("No files changed since last index. Nothing to update.");
        return Ok(());
    }

    println!("{} files changed, re-indexing...", changed_files.len());
    run(db_path, input_path, exclude)
}
