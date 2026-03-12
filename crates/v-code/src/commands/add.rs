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
use v_hnsw_cli::commands::file_utils::{
    generate_id, get_file_mtime, normalize_source, scan_files,
};
use v_hnsw_cli::commands::ingest::{
    CodeChunkEntry, IngestRecord, build_called_by_index, code_chunk_to_record,
    lookup_called_by,
};
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

    for code_path in &code_files {
        if is_interrupted() {
            break;
        }

        let source_code = match std::fs::read_to_string(code_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {e}", code_path.display());
                continue;
            }
        };

        let ext = code_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let chunks = match chunk_code::chunk_for_language(ext, &source_code) {
            Some(c) => c,
            None => continue,
        };
        if chunks.is_empty() {
            continue;
        }

        let source = normalize_source(code_path);
        let file_path_str = code_path.to_string_lossy().to_string();
        let mtime = get_file_mtime(code_path).unwrap_or(0);
        let size = file_index::get_file_size(code_path).unwrap_or(0);
        let mut chunk_ids = Vec::new();

        let lang = chunk_code::lang_for_extension(ext).unwrap_or("unknown");
        for chunk in chunks {
            let id = generate_id(&source, chunk.chunk_index);
            chunk_ids.push(id);
            entries.push(CodeChunkEntry {
                chunk,
                source: source.clone(),
                file_path_str: file_path_str.clone(),
                mtime,
                lang,
            });
        }

        file_metadata_map.insert(source, (mtime, size, chunk_ids));
    }

    // === Pass 2: Build called_by reverse index ===
    let reverse_index = build_called_by_index(&entries);
    let called_by_count: usize = reverse_index.values().map(Vec::len).sum();
    println!(
        "Call graph: {} callees, {} caller edges",
        reverse_index.len(),
        called_by_count
    );

    // === Pass 3: Generate IngestRecords ===
    let chunk_total_map: HashMap<&str, usize> = {
        let mut m: HashMap<&str, usize> = HashMap::new();
        for entry in &entries {
            *m.entry(&entry.source).or_default() += 1;
        }
        m
    };

    let mut records: Vec<IngestRecord> = Vec::with_capacity(entries.len());

    for entry in &entries {
        let chunk_total = chunk_total_map
            .get(entry.source.as_str())
            .copied()
            .unwrap_or(1);

        let called_by_refs = lookup_called_by(&reverse_index, &entry.chunk.name);
        let called_by: Vec<String> = called_by_refs.iter().map(|s| (*s).to_owned()).collect();

        records.push(code_chunk_to_record(
            &entry.chunk,
            &entry.source,
            &entry.file_path_str,
            entry.lang,
            entry.mtime,
            chunk_total,
            &called_by,
        ));
    }

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

/// Run the v-code update command (re-runs add for now).
pub fn run_update(db_path: PathBuf, input_path: PathBuf, exclude: &[String]) -> Result<()> {
    run(db_path, input_path, exclude)
}
