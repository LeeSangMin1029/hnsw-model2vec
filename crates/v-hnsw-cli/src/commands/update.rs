//! Update command - Incremental indexing of markdown files.
//!
//! Compares modification times and file sizes against a stored file index
//! to detect new, modified, and deleted files. Re-processes only changed files.

use std::path::PathBuf;
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
#[derive(Debug, Default)]
struct UpdateStats {
    new: usize,
    modified: usize,
    deleted: usize,
    unchanged: usize,
}

/// A record for batch processing.
struct UpdateRecord {
    id: u64,
    text: String,
    source: String,
    title: Option<String>,
    tags: Vec<String>,
    chunk_index: usize,
    chunk_total: usize,
    source_modified_at: u64,
}

/// Run the update command - incremental indexing.
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

    tracing::info!(
        db = %db_path.display(),
        input = %input_path.display(),
        "Starting update command"
    );
    println!("Updating database: {}", db_path.display());
    println!("Input folder:      {}", input_path.display());
    println!();

    let start = Instant::now();

    let mut file_index = file_index::load_file_index(&db_path)?;
    println!(
        "Loaded file index with {} tracked files",
        file_index.files.len()
    );

    let mut engine = StorageEngine::open(&db_path)
        .with_context(|| format!("Failed to open database at {}", db_path.display()))?;

    // Scan input folder for all markdown files
    let md_files: Vec<PathBuf> = walkdir::WalkDir::new(&input_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "md" || ext == "markdown")
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if md_files.is_empty() {
        println!("No markdown files found in {}", input_path.display());
        return Ok(());
    }

    println!("Scanning {} markdown files...", md_files.len());
    println!();

    let mut stats = UpdateStats::default();
    let mut seen_files = std::collections::HashSet::new();
    let mut changes = ChangeSet::default();

    let chunker = MarkdownChunker::new(ChunkConfig {
        target_size: 1000,
        overlap: 200,
        min_size: 100,
        include_heading_context: true,
    });

    // First pass: detect changes without embedding
    let mut pending_files: Vec<(PathBuf, String, u64, u64)> = Vec::new();

    for md_path in &md_files {
        let source = common::normalize_source(md_path);
        seen_files.insert(source.clone());

        let mtime = common::get_file_mtime(md_path).unwrap_or(0);
        let size = file_index::get_file_size(md_path).unwrap_or(0);

        if file_index.is_modified(&source, mtime, size) {
            pending_files.push((md_path.clone(), source, mtime, size));
        } else {
            stats.unchanged += 1;
        }
    }

    // Detect deleted files
    let input_prefix = common::normalize_source(&input_path);
    let deleted_files: Vec<String> = file_index
        .files
        .keys()
        .filter(|path| path.starts_with(&input_prefix) && !seen_files.contains(*path))
        .cloned()
        .collect();

    // Early exit: no changes at all
    if pending_files.is_empty() && deleted_files.is_empty() {
        file_index::save_file_index(&db_path, &file_index)?;
        let elapsed = start.elapsed();
        println!("No changes detected ({} files unchanged).", stats.unchanged);
        println!("Elapsed: {:.2}s", elapsed.as_secs_f64());
        return Ok(());
    }

    // Load model only when there are files to process
    let model = if !pending_files.is_empty() {
        Some(common::create_model()?)
    } else {
        None
    };

    // Process changed files
    for (md_path, source, mtime, size) in &pending_files {
        if is_interrupted() {
            break;
        }

        let old_meta = file_index.get_file(source);

        if let Some(meta) = old_meta {
            println!("Modified: {}", md_path.display());
            let old_ids = meta.chunk_ids.clone();
            for id in &old_ids {
                let _ = engine.remove(*id);
            }
            changes.removed.extend_from_slice(&old_ids);
            stats.modified += 1;
        } else {
            println!("New:      {}", md_path.display());
            stats.new += 1;
        }

        let (frontmatter, chunks) = match chunker.chunk_file(md_path) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error processing {}: {e}", md_path.display());
                continue;
            }
        };

        let title = frontmatter.as_ref().and_then(|f| f.title.clone());
        let tags = frontmatter
            .as_ref()
            .map(|f| f.tags.clone())
            .unwrap_or_default();
        let chunk_total = chunks.len();

        let mut records: Vec<UpdateRecord> = Vec::new();
        let mut chunk_ids = Vec::new();

        for chunk in chunks {
            let id = common::generate_id(source, chunk.chunk_index);
            chunk_ids.push(id);
            records.push(UpdateRecord {
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

        if !records.is_empty() {
            process_records(&records, model.as_ref().unwrap(), &mut engine)?;
            changes.added.extend(chunk_ids.iter());
        }

        file_index.update_file(source.clone(), *mtime, *size, chunk_ids);
    }

    // Process deleted files
    if !deleted_files.is_empty() {
        println!();
        println!("Deleting {} removed files:", deleted_files.len());
        for path in &deleted_files {
            println!("Deleted:  {}", path);
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

    file_index::save_file_index(&db_path, &file_index)?;

    // Incremental index update instead of full rebuild
    println!();
    let config = DbConfig::load(&db_path)?;
    common::update_indexes_incremental(
        &db_path,
        &engine,
        &config,
        &changes.added,
        &changes.removed,
    )?;

    // Notify daemon to reload if running
    if let Ok(()) = super::serve::notify_daemon_reload(&db_path) {
        println!("Daemon notified to reload indexes.");
    }

    let elapsed = start.elapsed();

    tracing::info!(
        new = stats.new,
        modified = stats.modified,
        deleted = stats.deleted,
        unchanged = stats.unchanged,
        elapsed_secs = elapsed.as_secs_f64(),
        "Update completed"
    );
    println!();
    println!("Update completed:");
    println!("  New files:       {}", stats.new);
    println!("  Modified files:  {}", stats.modified);
    println!("  Deleted files:   {}", stats.deleted);
    println!("  Unchanged files: {}", stats.unchanged);
    println!("  Elapsed:         {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

/// Process a batch of records: embed and insert.
fn process_records(
    records: &[UpdateRecord],
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<()> {
    let max_chars = 8000;

    let texts: Vec<String> = records
        .iter()
        .map(|r| {
            if r.text.len() > max_chars {
                let mut end = max_chars;
                while !r.text.is_char_boundary(end) {
                    end -= 1;
                }
                r.text[..end].to_string()
            } else {
                r.text.clone()
            }
        })
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
