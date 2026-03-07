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

use super::common::{self, IngestRecord};
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
pub fn run(db_path: PathBuf, input_path: PathBuf) -> Result<()> {
    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Use 'add' command to create a new database.",
            db_path.display()
        );
    }

    if !input_path.is_dir() {
        anyhow::bail!(
            "Input path must be a directory containing markdown files: {}",
            input_path.display()
        );
    }

    // Try daemon first (avoids 1GB model reload)
    if let Ok(stats) = try_daemon_update(&db_path, &input_path) {
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
    let stats = run_core(&db_path, &input_path, None)?;

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
) -> Result<UpdateStats> {
    let mut file_index = file_index::load_file_index(db_path)?;
    eprintln!(
        "Loaded file index with {} tracked files",
        file_index.files.len()
    );

    let mut engine = StorageEngine::open_exclusive(db_path)
        .with_context(|| format!("Failed to open database at {}", db_path.display()))?;

    // Scan input folder for all supported files (markdown + code)
    let all_files: Vec<PathBuf> = walkdir::WalkDir::new(input_path)
        .into_iter()
        .filter_entry(|e| {
            // Skip build/cache directories
            if e.file_type().is_dir() {
                let name = e.file_name().to_string_lossy();
                !matches!(
                    name.as_ref(),
                    "target" | "node_modules" | ".git" | ".swarm"
                        | "__pycache__" | ".venv" | "dist" | "vendor"
                        | ".tox" | ".mypy_cache" | ".pytest_cache"
                )
            } else {
                true
            }
        })
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    ext == "md"
                        || ext == "markdown"
                        || crate::chunk_code::is_supported_code_file(ext)
                })
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if all_files.is_empty() {
        eprintln!("No supported files found in {}", input_path.display());
        return Ok(UpdateStats::default());
    }

    eprintln!("Scanning {} files (markdown + code)...", all_files.len());

    let mut stats = UpdateStats::default();
    let mut seen_files = std::collections::HashSet::new();
    let mut changes = ChangeSet::default();

    let chunker = MarkdownChunker::new(ChunkConfig {
        target_size: 1000,
        overlap: 200,
        min_size: 100,
        include_heading_context: true,
    });

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

    // Code chunker for .rs files
    let code_chunker = crate::chunk_code::RustCodeChunker::new(
        crate::chunk_code::CodeChunkConfig::default(),
    );

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

        let is_code = file_path
            .extension()
            .and_then(|e| e.to_str())
            .map(crate::chunk_code::is_supported_code_file)
            .unwrap_or(false);

        let mut records: Vec<IngestRecord> = Vec::new();
        let mut chunk_ids = Vec::new();

        if is_code {
            // Code file: tree-sitter chunking
            let source_code = match std::fs::read_to_string(file_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error reading {}: {e}", file_path.display());
                    continue;
                }
            };
            let chunks = code_chunker.chunk(&source_code);
            let file_path_str = file_path.to_string_lossy().to_string();
            let chunk_total = chunks.len();

            for chunk in &chunks {
                let id = common::generate_id(source, chunk.chunk_index);
                chunk_ids.push(id);
                let embed_text = chunk.to_embed_text(&file_path_str, &[]);
                let mut tags = vec![
                    format!("kind:{}", chunk.kind.as_str()),
                    "lang:rust".to_owned(),
                ];
                if !chunk.visibility.is_empty() {
                    tags.push(format!("vis:{}", chunk.visibility));
                }
                records.push(IngestRecord {
                    id,
                    text: embed_text,
                    source: source.clone(),
                    title: Some(chunk.name.clone()),
                    tags,
                    chunk_index: chunk.chunk_index,
                    chunk_total,
                    source_modified_at: *mtime,
                });
            }
        } else {
            // Markdown file: heading-aware chunking
            let (frontmatter, chunks) = match chunker.chunk_file(file_path) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Error processing {}: {e}", file_path.display());
                    continue;
                }
            };

            let title = frontmatter.as_ref().and_then(|f| f.title.clone());
            let tags = frontmatter
                .as_ref()
                .map(|f| f.tags.clone())
                .unwrap_or_default();
            let chunk_total = chunks.len();

            for chunk in chunks {
                let id = common::generate_id(source, chunk.chunk_index);
                chunk_ids.push(id);
                records.push(IngestRecord {
                    id,
                    text: chunk.text,
                    source: source.clone(),
                    title: title.clone(),
                    tags: tags.clone(),
                    chunk_index: chunk.chunk_index,
                    chunk_total,
                    source_modified_at: *mtime,
                });
            }
        }

        if !records.is_empty() {
            #[allow(clippy::unwrap_used)]
            process_records(&records, model.unwrap(), &mut engine)?;
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
fn try_daemon_update(db_path: &Path, input_path: &Path) -> Result<UpdateStats> {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;

    let port = super::serve::read_port_file()
        .ok_or_else(|| anyhow::anyhow!("Daemon not running"))?;

    let canonical_db = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;
    let canonical_input = input_path
        .canonicalize()
        .with_context(|| format!("Input not found: {}", input_path.display()))?;

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .context("Failed to parse socket address")?;
    let mut stream = TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(2))
        .context("Failed to connect to daemon")?;

    // Update can take a long time — generous timeouts
    stream.set_read_timeout(Some(std::time::Duration::from_secs(300)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;

    let request = serde_json::json!({
        "id": 0,
        "method": "update",
        "params": {
            "db": canonical_db.to_str().unwrap_or(""),
            "input": canonical_input.to_str().unwrap_or("")
        }
    });
    writeln!(stream, "{}", serde_json::to_string(&request)?)?;
    stream.flush()?;

    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;

    let response: serde_json::Value = serde_json::from_str(&response_line)
        .context("Failed to parse daemon response")?;

    if let Some(err) = response.get("error") {
        anyhow::bail!("Daemon update failed: {}", err);
    }

    let result = response.get("result")
        .ok_or_else(|| anyhow::anyhow!("No result in daemon response"))?;

    let stats: UpdateStats = serde_json::from_value(result.clone())
        .context("Failed to parse update stats")?;

    println!("Update completed via daemon (no extra process).");
    Ok(stats)
}

/// Process a batch of records: embed and insert.
fn process_records(
    records: &[IngestRecord],
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<()> {
    let texts: Vec<String> = records
        .iter()
        .map(|r| common::truncate_for_embed(&r.text).to_string())
        .collect();

    let embeddings = common::embed_sorted(model, &texts).context("Embedding failed")?;

    let items: Vec<(u64, &[f32], _, &str)> = records
        .iter()
        .zip(embeddings.iter())
        .map(|(rec, emb)| {
            let payload = common::make_payload(
                &rec.source,
                rec.title.as_deref(),
                &rec.tags,
                rec.chunk_index,
                rec.chunk_total,
                rec.source_modified_at,
            );
            (rec.id, emb.as_slice(), payload, rec.text.as_str())
        })
        .collect();

    engine
        .insert_batch(&items)
        .context("Failed to insert batch")?;

    Ok(())
}
