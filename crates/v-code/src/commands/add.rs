//! Code-specific add/update command.
//!
//! Chunks code files via tree-sitter, stores text + payload only.
//! No embedding or index building — those are deferred to `v-code embed`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

use v_code_chunk as chunk_code;
use v_hnsw_cli::commands::db_config::DbConfig;
use v_hnsw_cli::commands::file_index;
use v_hnsw_cli::commands::file_utils::scan_files;
use super::ingest::{CodeChunkEntry, entries_to_records};
use v_hnsw_cli::commands::add::ingest::finalize_ingest_text_only;
use v_hnsw_cli::is_interrupted;
use v_hnsw_core::VectorStore;
use v_hnsw_storage::{StorageConfig, StorageEngine};

/// Placeholder dimension for text-only storage (no real vectors).
const TEXT_ONLY_DIM: usize = 1;
/// Model name stored in config for later `v-code embed` to detect.
const TEXT_ONLY_MODEL: &str = "text-only";


// ── Public entry points ──────────────────────────────────────────────────

/// Run the v-code add command (auto-incremental: only re-processes changed files).
pub fn run(db_path: PathBuf, input_path: PathBuf, exclude: &[String]) -> Result<()> {
    use v_hnsw_cli::commands::file_utils::get_file_mtime;

    println!("Indexing code: {}", input_path.display());
    println!("Database:      {}", db_path.display());

    // Scan for code files
    let all_files = scan_files(&input_path, exclude, chunk_code::is_supported_code_file);
    if all_files.is_empty() {
        anyhow::bail!(
            "No supported code files found in {}",
            input_path.display()
        );
    }

    // Build set of current source paths for deleted-file detection.
    let current_sources: std::collections::HashSet<String> = all_files
        .iter()
        .map(|f| v_hnsw_cli::commands::file_utils::normalize_source(f))
        .collect();

    // Filter to changed files only (mtime check)
    let file_idx = file_index::load_file_index(&db_path)?;
    let code_files: Vec<_> = all_files
        .into_iter()
        .filter(|f| {
            let source = v_hnsw_cli::commands::file_utils::normalize_source(f);
            match file_idx.get_file(&source) {
                Some(entry) => get_file_mtime(f).is_none_or(|m| m != entry.mtime),
                None => true,
            }
        })
        .collect();

    if code_files.is_empty() {
        println!("No files changed. Nothing to update.");
        return Ok(());
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

    // Open/create database (text-only, dim=1 placeholder)
    let mut engine = if db_path.exists() {
        StorageEngine::open_exclusive(&db_path)
            .with_context(|| format!("Failed to open database at {}", db_path.display()))?
    } else {
        println!("New database: {} (dim={TEXT_ONLY_DIM})", db_path.display());
        let config = StorageConfig {
            dim: TEXT_ONLY_DIM,
            initial_capacity: 10_000,
            checkpoint_threshold: 50_000,
        };
        let engine = StorageEngine::create(&db_path, config)
            .with_context(|| format!("Failed to create database at {}", db_path.display()))?;
        DbConfig {
            dim: TEXT_ONLY_DIM,
            code: true,
            embedded: false,
            embed_model: Some(TEXT_ONLY_MODEL.to_owned()),
            ..DbConfig::default()
        }.save(&db_path)?;
        engine
    };

    // Update config
    if let Ok(mut config) = DbConfig::load(&db_path) {
        config.code = true;
        config.embedded = false;
        if let Ok(canonical) = input_path.canonicalize() {
            config.input_path = Some(canonical.to_string_lossy().into_owned());
        }
        let _ = config.save(&db_path);
    }

    // === Pass 1: Chunk all files ===
    let mut entries: Vec<CodeChunkEntry> = Vec::new();
    let mut file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new();
    super::ingest::chunk_code_files(
        &code_files,
        is_interrupted,
        &mut entries,
        &mut file_metadata_map,
    );

    // === Pass 2+3: Build called_by index + generate IngestRecords ===
    let records = entries_to_records(&entries);
    println!("Symbols: {} (functions, structs, enums, ...)", records.len());

    // === Remove stale + insert text-only (no embedding) + update file index ===
    let dim = engine.vector_store().dim();
    let result = finalize_ingest_text_only(&db_path, records, dim, &mut engine, file_metadata_map)?;
    let inserted = result.inserted;

    // === Remove chunks from deleted files ===
    let mut file_idx = file_index::load_file_index(&db_path)?;
    let deleted: Vec<String> = file_idx.files.keys()
        .filter(|p| !current_sources.contains(p.as_str()))
        .cloned()
        .collect();
    if !deleted.is_empty() {
        let mut del_count = 0usize;
        for path in &deleted {
            if let Some(entry) = file_idx.files.remove(path) {
                for id in &entry.chunk_ids {
                    let _ = engine.remove(*id);
                    del_count += 1;
                }
            }
        }
        if del_count > 0 {
            engine.checkpoint().ok();
            file_index::save_file_index(&db_path, &file_idx)?;
            eprintln!("Removed {del_count} chunks from {n} deleted file(s)", n = deleted.len());
        }
    }

    if is_interrupted() {
        println!();
        println!("Operation interrupted. Partial data may have been inserted.");
        return Ok(());
    }

    if inserted == 0 {
        println!("No symbols to index.");
    } else {
        // Notify daemon to reload if running
        if v_hnsw_storage::daemon_client::notify_reload(&db_path).is_ok() {
            println!("Daemon notified to reload indexes.");
        }

        println!();
        println!("Done! Code DB ready: {}", db_path.display());
        println!("Use: v-code context/blast/jump/symbols/dupes {}", db_path.display());
    }

    Ok(())
}

