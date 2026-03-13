//! Input format processors: markdown folders and JSONL files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use crate::chunk::{ChunkConfig, MarkdownChunker};
use v_hnsw_embed::Model2VecModel;
use v_hnsw_storage::StorageEngine;

use crate::commands::pipeline::process_records;
use crate::commands::common;
use crate::commands::common::IngestRecord;
use crate::commands::file_index;
pub use crate::commands::ingest::{
    CodeChunkEntry, build_called_by_index, code_chunk_to_record, lookup_called_by,
};
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

/// Process records and update the file metadata index.
///
/// Shared tail logic for `process_markdown_files` and `process_code_files`.
/// Returns `(inserted, errors, added_ids, removed_ids)`.
pub(crate) fn finalize_ingest(
    db_path: &Path,
    records: Vec<IngestRecord>,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
    file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)>,
) -> Result<IngestResult> {
    // Remove stale chunks for files being re-added
    let mut removed_ids = Vec::new();
    let file_index = file_index::load_file_index(db_path)?;
    for (path, (_, _, new_ids)) in &file_metadata_map {
        if let Some(existing) = file_index.get_file(path) {
            // Remove old IDs that are NOT in the new set (chunk_index shifted)
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

    let (inserted, errors, inserted_ids) = process_records(records, model, engine)?;

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

/// Chunk a single file (code or markdown) into [`IngestRecord`]s.
///
/// Returns `(records, chunk_ids)`. For code files without `called_by` data
/// (e.g., incremental update), pass empty slice.
///
/// Returns `Err` only for I/O errors; parse failures return empty results.
pub(crate) fn chunk_single_file(
    file_path: &std::path::Path,
    source: &str,
    mtime: u64,
    chunker: &crate::chunk::MarkdownChunker,
) -> (Vec<IngestRecord>, Vec<u64>) {
    let is_code = file_path
        .extension()
        .and_then(|e| e.to_str())
        .map(crate::chunk_code::is_supported_code_file)
        .unwrap_or(false);

    let mut records = Vec::new();
    let mut chunk_ids = Vec::new();

    if is_code {
        let source_code = match std::fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {e}", file_path.display());
                return (records, chunk_ids);
            }
        };
        let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let chunks = crate::chunk_code::chunk_for_language(ext, &source_code)
            .unwrap_or_default();
        let file_path_str = file_path.to_string_lossy().to_string();
        let chunk_total = chunks.len();
        let lang = crate::chunk_code::lang_for_extension(ext).unwrap_or("unknown");

        for chunk in &chunks {
            let id = common::generate_id(source, chunk.chunk_index);
            chunk_ids.push(id);
            records.push(code_chunk_to_record(
                chunk, source, &file_path_str, lang, mtime, chunk_total, &[],
            ));
        }
    } else {
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
    }

    (records, chunk_ids)
}


/// Process code folder: chunk .rs files via tree-sitter, embed, insert.
///
/// Two-pass approach:
/// 1. Chunk all files and collect `calls` data
/// 2. Build `called_by` reverse index, then generate embed text with it
pub fn process_code_folder(
    db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
    exclude: &[String],
) -> Result<IngestResult> {
    let code_files = common::scan_files(input_path, exclude, crate::chunk_code::is_supported_code_file);

    if code_files.is_empty() {
        anyhow::bail!("No supported code files found in {}", input_path.display());
    }

    process_code_files(db_path, &code_files, model, engine)
}

/// Process a list of code files: chunk via tree-sitter, embed, insert.
///
/// Shared implementation for both folder and single-file ingestion.
pub fn process_code_files(
    db_path: &Path,
    code_files: &[PathBuf],
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<IngestResult> {
    println!("Files: {} code", code_files.len());

    // === Pass 1: Chunk all files, collect CodeChunk + metadata ===
    let mut entries: Vec<CodeChunkEntry> = Vec::new();
    let mut file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new();
    crate::commands::ingest::chunk_code_files(
        code_files,
        is_interrupted,
        &mut entries,
        &mut file_metadata_map,
    );

    // === Pass 2: Build called_by reverse index ===
    let reverse_index = build_called_by_index(&entries);

    let called_by_count: usize = reverse_index.values().map(Vec::len).sum();
    println!("Call graph: {} targets, {} edges", reverse_index.len(), called_by_count);

    // === Pass 3: Generate IngestRecords with called_by data ===
    let chunk_total_map: HashMap<&str, usize> = {
        let mut m: HashMap<&str, usize> = HashMap::new();
        for entry in &entries {
            *m.entry(&entry.source).or_default() += 1;
        }
        m
    };

    let mut records: Vec<IngestRecord> = Vec::with_capacity(entries.len());

    for entry in &entries {
        let chunk = &entry.chunk;
        let chunk_total = chunk_total_map
            .get(entry.source.as_str())
            .copied()
            .unwrap_or(1);

        // Resolve called_by for this chunk
        let called_by_refs = lookup_called_by(&reverse_index, &chunk.name);
        let called_by: Vec<String> = called_by_refs.iter().map(|s| (*s).to_owned()).collect();

        records.push(code_chunk_to_record(
            chunk,
            &entry.source,
            &entry.file_path_str,
            entry.lang,
            entry.mtime,
            chunk_total,
            &called_by,
        ));
    }

    println!("Symbols: {}", records.len());

    finalize_ingest(db_path, records, model, engine, file_metadata_map)
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

