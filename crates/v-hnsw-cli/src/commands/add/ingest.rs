//! Input format processors: markdown folders and JSONL files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, PayloadValue};
use v_hnsw_embed::Model2VecModel;
use v_hnsw_storage::StorageEngine;

use crate::chunk::{ChunkConfig, MarkdownChunker};
use crate::commands::pipeline::process_records;
use crate::commands::common;
use crate::commands::common::IngestRecord;
use crate::commands::file_index;
use crate::is_interrupted;

/// Result of an ingest operation.
pub struct IngestResult {
    pub inserted: u64,
    pub errors: u64,
    pub added_ids: Vec<u64>,
    pub removed_ids: Vec<u64>,
}

/// Process markdown folder: chunk files, embed, insert.
pub fn process_markdown_folder(
    db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
    exclude: &[String],
) -> Result<IngestResult> {
    let md_files = common::scan_files(input_path, exclude, |ext| {
        ext == "md" || ext == "markdown"
    });

    if md_files.is_empty() {
        anyhow::bail!("No markdown files found in {}", input_path.display());
    }

    process_markdown_files(db_path, &md_files, model, engine)
}

/// Process a list of markdown files: chunk, embed, insert.
///
/// Shared implementation for both folder and single-file ingestion.
pub fn process_markdown_files(
    db_path: &Path,
    md_files: &[PathBuf],
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<IngestResult> {
    let chunker = MarkdownChunker::new(ChunkConfig::default());

    println!("Files: {} markdown", md_files.len());

    let mut records: Vec<IngestRecord> = Vec::new();
    let mut file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new();

    for md_path in md_files {
        if is_interrupted() {
            break;
        }

        let (frontmatter, chunks) = match chunker.chunk_file(md_path) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading {}: {e}", md_path.display());
                continue;
            }
        };

        let source = common::normalize_source(md_path);
        let title = frontmatter.as_ref().and_then(|f| f.title.clone());
        let tags = frontmatter
            .as_ref()
            .map(|f| f.tags.clone())
            .unwrap_or_default();
        let chunk_total = chunks.len();

        let mtime = common::get_file_mtime(md_path).unwrap_or(0);
        let size = file_index::get_file_size(md_path).unwrap_or(0);
        let mut chunk_ids = Vec::new();

        for chunk in chunks {
            let id = common::generate_id(&source, chunk.chunk_index);
            chunk_ids.push(id);
            records.push(IngestRecord {
                id,
                text: chunk.text,
                source: source.clone(),
                title: title.clone(),
                tags: tags.clone(),
                chunk_index: chunk.chunk_index,
                chunk_total,
                source_modified_at: mtime,
                custom: HashMap::new(),
            });
        }

        file_metadata_map.insert(source, (mtime, size, chunk_ids));
    }

    println!("Chunks: {}", records.len());

    finalize_ingest(db_path, records, model, engine, file_metadata_map)
}

/// Shared tail logic: remove stale chunks, filter unchanged, run processor,
/// update file index.
///
/// `processor` receives the filtered records and returns `(inserted, errors, inserted_ids)`.
fn finalize_ingest_core(
    db_path: &Path,
    records: Vec<IngestRecord>,
    engine: &mut StorageEngine,
    file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)>,
    processor: impl FnOnce(Vec<IngestRecord>, &mut StorageEngine) -> Result<(u64, u64, Vec<u64>)>,
) -> Result<IngestResult> {
    // Remove stale chunks for files being re-added
    let mut removed_ids = Vec::new();
    let file_index = file_index::load_file_index(db_path)?;
    for (path, (_, _, new_ids)) in &file_metadata_map {
        if let Some(existing) = file_index.get_file(path) {
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
        eprintln!(
            "Removed {} stale chunks from re-added files",
            removed_ids.len()
        );
    }

    // Early cutoff: skip records whose body_hash matches existing payload.
    let (records, skipped) = filter_unchanged_records(records, engine);
    if skipped > 0 {
        eprintln!("Skipped {skipped} unchanged symbols (body_hash match)");
    }

    let (inserted, errors, inserted_ids) = processor(records, engine)?;

    let mut file_index = file_index::load_file_index(db_path)?;
    for (path, (mtime, size, chunk_ids)) in file_metadata_map {
        file_index.update_file(path, mtime, size, chunk_ids);
    }
    file_index::save_file_index(db_path, &file_index)?;

    Ok(IngestResult {
        inserted,
        errors,
        added_ids: inserted_ids,
        removed_ids,
    })
}

/// Process records and update the file metadata index.
///
/// Shared tail logic for `process_markdown_files`.
pub fn finalize_ingest(
    db_path: &Path,
    records: Vec<IngestRecord>,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
    file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)>,
) -> Result<IngestResult> {
    finalize_ingest_core(db_path, records, engine, file_metadata_map, |recs, eng| {
        process_records(recs, model, eng)
    })
}


/// Extract `body_hash` (i64) from a payload's custom fields.
fn payload_body_hash(payload: &v_hnsw_core::Payload) -> Option<i64> {
    payload.custom.get("body_hash").and_then(|v| {
        if let PayloadValue::Integer(h) = v { Some(*h) } else { None }
    })
}

/// Filter out records whose `body_hash` matches the existing payload in the DB.
///
/// Returns `(records_to_process, skipped_count)`.
fn filter_unchanged_records(
    records: Vec<IngestRecord>,
    engine: &StorageEngine,
) -> (Vec<IngestRecord>, usize) {
    let store = engine.payload_store();
    let mut changed = Vec::with_capacity(records.len());
    let mut skipped = 0usize;

    for rec in records {
        let new_hash = rec.custom.get("body_hash").and_then(|v| {
            if let PayloadValue::Integer(h) = v { Some(*h) } else { None }
        });

        // If body_hash matches AND text is unchanged, skip re-embedding.
        // Text includes line numbers, so even if body_hash is the same,
        // line shifts (from edits above) will change the text and trigger re-indexing.
        if let Some(new_h) = new_hash
            && let Ok(Some(existing)) = store.get_payload(rec.id)
            && let Some(old_h) = payload_body_hash(&existing)
            && old_h == new_h
            && store.get_text(rec.id).ok().flatten().as_deref() == Some(&rec.text)
        {
            skipped += 1;
            continue;
        }

        changed.push(rec);
    }

    (changed, skipped)
}

/// Chunk a single markdown file into [`IngestRecord`]s.
///
/// Returns `(records, chunk_ids)`.
/// Returns empty results for parse failures (does not propagate errors).
pub(crate) fn chunk_single_file(
    file_path: &std::path::Path,
    source: &str,
    mtime: u64,
    chunker: &crate::chunk::MarkdownChunker,
) -> (Vec<IngestRecord>, Vec<u64>) {
    let mut records = Vec::new();
    let mut chunk_ids = Vec::new();

    let (frontmatter, chunks) = match chunker.chunk_file(file_path) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error processing {}: {e}", file_path.display());
            return (records, chunk_ids);
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
            source: source.to_owned(),
            title: title.clone(),
            tags: tags.clone(),
            chunk_index: chunk.chunk_index,
            chunk_total,
            source_modified_at: mtime,
            custom: HashMap::new(),
        });
    }

    (records, chunk_ids)
}


/// Process JSONL file: parse records, embed, insert.
pub fn process_jsonl(
    _db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<IngestResult> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(input_path)
        .with_context(|| format!("Failed to open {}", input_path.display()))?;
    let reader = BufReader::new(file);

    let mut records: Vec<IngestRecord> = Vec::new();
    let source = common::normalize_source(input_path);

    for (line_num, line_result) in reader.lines().enumerate() {
        if is_interrupted() {
            break;
        }

        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Line {}: read error: {e}", line_num + 1);
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Line {}: parse error: {e}", line_num + 1);
                continue;
            }
        };

        // Extract text field
        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let text = match text {
            Some(t) if !t.is_empty() => t,
            _ => continue,
        };

        let id = json
            .get("id")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| common::generate_id(&source, line_num));

        let title = json
            .get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let tags = json
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let item_source = json
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or(&source)
            .to_string();

        records.push(IngestRecord {
            id,
            text,
            source: item_source,
            title,
            tags,
            chunk_index: 0,
            chunk_total: 1,
            source_modified_at: 0,
            custom: HashMap::new(),
        });
    }

    println!("Records: {}", records.len());

    let (inserted, errors, added_ids) = process_records(records, model, engine)?;
    Ok(IngestResult {
        inserted,
        errors,
        added_ids,
        removed_ids: Vec::new(),
    })
}

