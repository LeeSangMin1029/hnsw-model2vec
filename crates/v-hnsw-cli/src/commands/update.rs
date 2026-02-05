//! Update command - Incremental indexing of markdown files.
//!
//! Compares modification times and file sizes against a stored file index
//! to detect new, modified, and deleted files. Re-processes only changed files.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use v_hnsw_chunk::{ChunkConfig, MarkdownChunker};
use v_hnsw_core::{PayloadStore, VectorIndex, VectorStore};
use v_hnsw_distance::CosineDistance;
use v_hnsw_embed::Model2VecModel;
use v_hnsw_graph::{HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
use v_hnsw_storage::StorageEngine;

use super::add::{create_model, embed_sorted, generate_id, make_payload, make_progress_bar};
use super::create::DbConfig;
use super::file_index;
use crate::is_interrupted;

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

    println!("Updating database: {}", db_path.display());
    println!("Input folder:      {}", input_path.display());
    println!();

    let start = Instant::now();

    // Load existing file index
    let mut file_index = file_index::load_file_index(&db_path)?;
    println!("Loaded file index with {} tracked files", file_index.files.len());

    // Open storage engine
    let mut engine = StorageEngine::open(&db_path)
        .with_context(|| format!("Failed to open database at {}", db_path.display()))?;

    // Create model
    let model = create_model()?;

    // Scan input folder for all markdown files
    let md_files: Vec<PathBuf> = walkdir::WalkDir::new(&input_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
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

    let chunker = MarkdownChunker::new(ChunkConfig {
        target_size: 1000,
        overlap: 200,
        min_size: 100,
        include_heading_context: true,
    });

    // Process each file
    for md_path in &md_files {
        if is_interrupted() {
            break;
        }

        let source = md_path.to_string_lossy().to_string();
        seen_files.insert(source.clone());

        // Get current file metadata
        let mtime = file_index::get_file_mtime(md_path).unwrap_or(0);
        let size = file_index::get_file_size(md_path).unwrap_or(0);

        if file_index.is_modified(&source, mtime, size) {
            // File is new or modified
            let old_meta = file_index.get_file(&source);

            if old_meta.is_some() {
                // MODIFIED: delete old chunks
                println!("Modified: {}", md_path.display());
                let old_ids = old_meta.unwrap().chunk_ids.clone();
                for id in old_ids {
                    let _ = engine.remove(id);
                }
                stats.modified += 1;
            } else {
                // NEW file
                println!("New:      {}", md_path.display());
                stats.new += 1;
            }

            // Re-process the file
            let (frontmatter, chunks) = match chunker.chunk_file(md_path) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Error processing {}: {e}", md_path.display());
                    continue;
                }
            };

            let title = frontmatter.as_ref().and_then(|f| f.title.clone());
            let tags = frontmatter.as_ref().map(|f| f.tags.clone()).unwrap_or_default();
            let chunk_total = chunks.len();

            let mut records: Vec<UpdateRecord> = Vec::new();
            let mut chunk_ids = Vec::new();

            for chunk in chunks {
                let id = generate_id(&source, chunk.chunk_index);
                chunk_ids.push(id);
                records.push(UpdateRecord {
                    id,
                    text: chunk.text,
                    source: source.clone(),
                    title: title.clone(),
                    tags: tags.clone(),
                    chunk_index: chunk.chunk_index,
                    chunk_total,
                });
            }

            // Embed and insert
            if !records.is_empty() {
                process_records(&records, &model, &mut engine)?;
            }

            // Update file index
            file_index.update_file(source, mtime, size, chunk_ids);
        } else {
            // Unchanged file
            stats.unchanged += 1;
        }
    }

    // Find deleted files (in index but not on disk)
    let deleted_files: Vec<String> = file_index
        .files
        .keys()
        .filter(|path| !seen_files.contains(*path))
        .cloned()
        .collect();

    if !deleted_files.is_empty() {
        println!();
        println!("Deleting {} removed files:", deleted_files.len());
        for path in &deleted_files {
            println!("Deleted:  {}", path);
            if let Some(meta) = file_index.get_file(path) {
                for id in &meta.chunk_ids {
                    let _ = engine.remove(*id);
                }
            }
            file_index.files.remove(path);
            stats.deleted += 1;
        }
    }

    // Checkpoint the storage
    engine.checkpoint()
        .with_context(|| "Failed to checkpoint database")?;

    // Save updated file index
    file_index::save_file_index(&db_path, &file_index)?;

    println!();
    println!("Rebuilding indexes...");

    // Rebuild HNSW and BM25 indexes
    let config = DbConfig::load(&db_path)?;
    rebuild_indexes(&db_path, &engine, &config)?;

    let elapsed = start.elapsed();

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
    let max_chars = 8000; // model2vec uses simple tokenization

    // Prepare texts for embedding (truncate if needed)
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

    // Embed batch
    let embeddings = embed_sorted(model, &texts)
        .context("Embedding failed")?;

    // Insert into storage
    let items: Vec<(u64, &[f32], _, &str)> = records
        .iter()
        .zip(embeddings.iter())
        .map(|(rec, emb)| {
            let payload = make_payload(
                &rec.source,
                rec.title.as_deref(),
                &rec.tags,
                rec.chunk_index,
                rec.chunk_total,
            );
            (rec.id, emb.as_slice(), payload, rec.text.as_str())
        })
        .collect();

    engine.insert_batch(&items)
        .context("Failed to insert batch")?;

    Ok(())
}

/// Rebuild HNSW and BM25 indexes from storage.
fn rebuild_indexes(path: &Path, engine: &StorageEngine, config: &DbConfig) -> Result<()> {
    if engine.is_empty() {
        println!("No vectors to index, skipping index rebuilding.");
        return Ok(());
    }

    // Build HNSW graph
    let hnsw_config = HnswConfig::builder()
        .dim(config.dim)
        .m(config.m)
        .ef_construction(config.ef_construction)
        .build()
        .context("Failed to create HNSW config")?;

    let hnsw_path = path.join("hnsw.bin");
    let vector_store = engine.vector_store();

    println!("  Building HNSW graph (M={}, ef_construction={})...", config.m, config.ef_construction);

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

    println!("Index rebuilding completed.");
    Ok(())
}
