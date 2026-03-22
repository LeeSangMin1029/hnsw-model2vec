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
use super::ingest::CodeChunkEntry;
use v_hnsw_cli::is_interrupted;
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

    // Scan for code files — prefer `git ls-files` (instant) over walkdir (slow fs walk).
    let t_scan = std::time::Instant::now();
    let all_files = scan_files_fast(&input_path, exclude);
    eprintln!("  scan: {:.1}ms ({} files)", t_scan.elapsed().as_secs_f64() * 1000.0, all_files.len());
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
            let path_str = canonical.to_string_lossy();
            config.input_path = Some(v_hnsw_core::strip_unc_prefix(&path_str).to_owned());
        }
        let _ = config.save(&db_path);
    }

    // === Pass 1: Chunk all files via daemon RA ===
    eprintln!("  [rss] start: {:.0}MB", v_code_intel::graph::current_rss_mb());
    let t0 = std::time::Instant::now();
    let mut entries: Vec<CodeChunkEntry> = Vec::new();
    let mut file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new();

    // Ensure daemon is running for RA-based chunking.
    if !v_hnsw_storage::daemon_client::is_running() {
        eprintln!("  [daemon] not running, starting...");
        if !v_hnsw_storage::daemon_client::spawn_daemon_and_wait(&db_path) {
            anyhow::bail!("daemon failed to start — run `v-daemon --db {}` first", db_path.display());
        }
    }

    super::ingest::chunk_via_daemon(&code_files, &db_path, &mut entries, &mut file_metadata_map)?;
    eprintln!("  chunk: {:.1}s  RSS: {:.0}MB", t0.elapsed().as_secs_f64(), v_code_intel::graph::current_rss_mb());

    // === Build called_by + direct bulk write (zero-copy path) ===
    println!("Symbols: {} (functions, structs, enums, ...)", entries.len());
    let t1 = std::time::Instant::now();
    let inserted = direct_bulk_write(&db_path, &entries, &mut engine, &file_metadata_map)?;
    eprintln!("  ingest: {:.1}s  RSS: {:.0}MB",
        t1.elapsed().as_secs_f64(), v_code_intel::graph::current_rss_mb());


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

    let has_changes = inserted > 0 || !deleted.is_empty();

    if !has_changes {
        println!("No changes. Database is up to date.");
    } else {
        // Notify daemon to reload if running (non-blocking).
        if v_hnsw_storage::daemon_client::is_running() {
            v_hnsw_storage::daemon_client::daemon_rpc_fire_and_forget(
                "reload",
                serde_json::json!({"db": db_path.canonicalize()
                    .unwrap_or_else(|_| db_path.clone())
                    .to_string_lossy().as_ref()}),
            );
            println!("Daemon notified to reload indexes.");
        }

        println!();
        println!("Done! Code DB ready: {}", db_path.display());
        println!("Use: v-code context/blast/jump/symbols/dupes {}", db_path.display());

        // Pre-build chunks.bin + graph.bin caches directly from in-memory entries.
            drop(engine); // Release exclusive lock.
        let t_cache = std::time::Instant::now();
        prebuild_caches(&db_path, &entries, &current_sources);
        eprintln!("  cache: {:.1}s", t_cache.elapsed().as_secs_f64());
    }

    Ok(())
}

/// Build chunks.bin + graph.bin caches directly from in-memory entries.
///
/// Single source of truth: `from_code_chunk` → chunks.bin → graph.bin.
/// verify/context/blast use these caches without any rebuilding.
fn prebuild_caches(
    db_path: &std::path::Path,
    new_entries: &[CodeChunkEntry],
    current_sources: &std::collections::HashSet<String>,
) {
    use v_code_intel::parse::ParsedChunk;

    let t_total = std::time::Instant::now();

    let cache = v_code_intel::loader::cache_path(db_path);
    if let Some(parent) = cache.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Collect sources that were just re-chunked.
    let t0 = std::time::Instant::now();
    let new_sources: std::collections::HashSet<&str> = new_entries
        .iter()
        .map(|e| e.source.as_str())
        .collect();

    // Convert new entries → ParsedChunks (no DB round-trip).
    // Use file_path_str (matches to_embed_text's File: line used in DB text parsing).
    let mut chunks: Vec<ParsedChunk> = new_entries
        .iter()
        .map(|e| ParsedChunk::from_code_chunk(&e.chunk, &e.file_path_str, e.chunk.imports.clone()))
        .collect();

    // For incremental adds: merge with existing cache (keep chunks from unchanged files).
    // Only attempt if a cache file already exists (skip on full rebuild).
    //
    // chunk.file is a relative path (e.g. "crates/v-code/src/lib.rs") while
    // new_sources uses file_path_str and current_sources uses normalize_source
    // (absolute path). Build a suffix set from current_sources for matching.
    let current_suffixes: std::collections::HashSet<&str> = current_sources
        .iter()
        .filter_map(|abs| {
            // Match suffix: the chunk.file relative path is a suffix of the absolute path.
            abs.find("crates/").map(|pos| &abs[pos..])
                .or_else(|| abs.find("src/").map(|pos| &abs[pos..]))
        })
        .collect();
    let new_suffixes: std::collections::HashSet<&str> = new_sources
        .iter()
        .filter_map(|abs| {
            abs.find("crates/").map(|pos| &abs[pos..])
                .or_else(|| abs.find("src/").map(|pos| &abs[pos..]))
        })
        .collect();

    if cache.exists() {
        if let Ok(existing) = v_code_intel::loader::load_chunks(db_path) {
            let existing_len = existing.len();
            let mut kept = 0usize;
            let mut skipped_new = 0usize;
            let mut skipped_deleted = 0usize;
            for c in existing {
                let in_new = new_suffixes.contains(c.file.as_str());
                let in_current = current_suffixes.contains(c.file.as_str());
                if !in_new && in_current {
                    chunks.push(c);
                    kept += 1;
                } else if in_new {
                    skipped_new += 1;
                } else {
                    skipped_deleted += 1;
                }
            }
            eprintln!("    [merge] existing={existing_len}, kept={kept}, skipped(new={skipped_new}, del={skipped_deleted})");
        }
    }
    eprintln!("    [cache] chunks convert: {:.1}ms ({} chunks)", t0.elapsed().as_secs_f64() * 1000.0, chunks.len());

    // Save chunks.bin
    let t1 = std::time::Instant::now();
    v_code_intel::loader::save_chunks_cache(&cache, &chunks);
    if let Ok(file) = std::fs::OpenOptions::new().write(true).open(&cache) {
        let _ = file.set_modified(std::time::SystemTime::now());
    }
    eprintln!("    [cache] chunks.bin save: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

    // Build graph directly from chunks (calls already resolved by RA in code/chunk).
    // No need for daemon graph/build RPC — avoids duplicate RA work.
    let t3 = std::time::Instant::now();
    let graph = v_code_intel::graph::CallGraph::build_full(&chunks);
    eprintln!("    [cache] graph build: {:.1}ms ({} chunks)", t3.elapsed().as_secs_f64() * 1000.0, chunks.len());

    let t4 = std::time::Instant::now();
    let _ = graph.save(db_path);
    eprintln!("    [cache] graph.bin save: {:.1}ms", t4.elapsed().as_secs_f64() * 1000.0);
    eprintln!("    [cache] total: {:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0);
}

/// Zero-copy ingest: CodeChunkEntry → Payload bincode → disk.
///
/// Skips IngestRecord and make_payload intermediates. Builds Payload inline
/// from entry references, encodes directly to contiguous buffer, single I/O.
fn direct_bulk_write(
    db_path: &std::path::Path,
    entries: &[super::ingest::CodeChunkEntry],
    engine: &mut StorageEngine,
    file_metadata_map: &HashMap<String, (u64, u64, Vec<u64>)>,
) -> Result<u64> {
    use v_hnsw_cli::commands::file_utils::generate_id;
    use v_hnsw_core::{Payload, PayloadValue};

    // Remove stale chunks for files being re-added.
    let file_index_data = file_index::load_file_index(db_path)?;
    for (path, (_, _, new_ids)) in file_metadata_map {
        if let Some(existing) = file_index_data.get_file(path) {
            for &old_id in &existing.chunk_ids {
                if !new_ids.contains(&old_id) {
                    let _ = engine.remove(old_id);
                }
            }
        }
    }

    let start = std::time::Instant::now();

    // Build called_by reverse index (needed for embed_text + tags).
    let t_idx = std::time::Instant::now();
    let reverse_index = super::ingest::build_called_by_index(entries);
    let chunk_total_map: HashMap<&str, usize> = {
        let mut m: HashMap<&str, usize> = HashMap::new();
        for entry in entries {
            *m.entry(&entry.source).or_default() += 1;
        }
        m
    };
    let idx_ms = t_idx.elapsed().as_secs_f64() * 1000.0;

    // Encode payloads + texts directly into contiguous buffers (zero-copy path).
    // No IngestRecord or intermediate Payload allocation per record.
    let t_enc = std::time::Instant::now();
    let config = bincode::config::standard();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Parallel encode: each entry independently produces (id, payload_bytes, text_bytes).
    use rayon::prelude::*;
    let encoded: Vec<(u64, Vec<u8>, String)> = entries
        .par_iter()
        .map(|entry| {
            let chunk = &entry.chunk;
            let id = generate_id(&entry.source, chunk.chunk_index);
            let chunk_total = chunk_total_map.get(entry.source.as_str()).copied().unwrap_or(1);
            let called_by_refs = super::ingest::lookup_called_by(&reverse_index, &chunk.name);

            let is_test = v_code_intel::graph::is_test_path(&entry.source)
                || chunk.name.starts_with("test_");

            let mut tags = Vec::with_capacity(4 + called_by_refs.len());
            tags.push(format!("kind:{}", chunk.kind.as_str()));
            tags.push(format!("lang:{}", entry.lang));
            tags.push(format!("role:{}", if is_test { "test" } else { "prod" }));
            if !chunk.visibility.is_empty() {
                tags.push(format!("vis:{}", chunk.visibility));
            }
            for caller in &called_by_refs {
                tags.push(format!("caller:{caller}"));
            }

            let called_by_strings: Vec<String> = called_by_refs.iter().map(|s| (*s).to_owned()).collect();
            let custom = chunk.to_custom_fields(&called_by_strings);

            let mut custom_with_title = custom;
            custom_with_title.insert("title".into(), PayloadValue::String(chunk.name.clone()));

            let payload = Payload {
                source: entry.source.clone(),
                tags,
                created_at: now,
                source_modified_at: entry.mtime,
                chunk_index: chunk.chunk_index as u32,
                chunk_total: chunk_total as u32,
                custom: custom_with_title,
            };

            let payload_bytes = bincode::encode_to_vec(&payload, config)
                .unwrap_or_default();

            let embed_text = chunk.to_embed_text(&entry.file_path_str, &called_by_strings);

            (id, payload_bytes, embed_text)
        })
        .collect();

    // Sequential merge into contiguous buffers.
    let mut payload_buf: Vec<u8> = Vec::with_capacity(entries.len() * 256);
    let mut text_buf: Vec<u8> = Vec::with_capacity(entries.len() * 512);
    let mut ids: Vec<u64> = Vec::with_capacity(encoded.len());
    let mut payload_offsets: Vec<(u64, u32)> = Vec::with_capacity(encoded.len());
    let mut text_offsets: Vec<(u64, u32)> = Vec::with_capacity(encoded.len());

    for (id, p_bytes, embed_text) in &encoded {
        let p_start = payload_buf.len() as u64;
        payload_buf.extend_from_slice(p_bytes);
        payload_offsets.push((p_start, p_bytes.len() as u32));

        let t_start = text_buf.len() as u64;
        let t_bytes = embed_text.as_bytes();
        text_buf.extend_from_slice(t_bytes);
        text_offsets.push((t_start, t_bytes.len() as u32));

        ids.push(*id);
    }
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    // Write to engine using raw bulk API.
    let t_write = std::time::Instant::now();
    engine.bulk_load_raw(&ids, &payload_buf, &payload_offsets, &text_buf, &text_offsets)
        .context("Failed to bulk load")?;
    let write_ms = t_write.elapsed().as_secs_f64() * 1000.0;

    // Update file index.
    let t_fi = std::time::Instant::now();
    let mut file_idx = file_index::load_file_index(db_path)?;
    for (path, (mtime, size, chunk_ids)) in file_metadata_map {
        file_idx.update_file(path.to_string(), *mtime, *size, chunk_ids.clone());
    }
    file_index::save_file_index(db_path, &file_idx)?;
    let fi_ms = t_fi.elapsed().as_secs_f64() * 1000.0;

    let inserted = entries.len() as u64;
    println!("\nInserted {inserted} chunks in {:.2}s (idx={:.0}ms enc={:.0}ms write={:.0}ms fidx={:.0}ms)",
        start.elapsed().as_secs_f64(), idx_ms, enc_ms, write_ms, fi_ms);

    Ok(inserted)
}


/// Fast file scan: `git ls-files` (instant from index) with walkdir fallback.
fn scan_files_fast(input_path: &std::path::Path, exclude: &[String]) -> Vec<PathBuf> {
    if let Ok(output) = std::process::Command::new("git")
        .args(["ls-files", "--cached", "--others", "--exclude-standard"])
        .current_dir(input_path)
        .output()
    {
        if output.status.success() {
            let files: Vec<PathBuf> = String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| {
                    let path = input_path.join(line);
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if chunk_code::is_supported_code_file(ext) {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect();
            if !files.is_empty() {
                return files;
            }
        }
    }
    // Fallback to walkdir.
    scan_files(input_path, exclude, chunk_code::is_supported_code_file)
}


