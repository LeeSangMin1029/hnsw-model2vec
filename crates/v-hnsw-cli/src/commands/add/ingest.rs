//! Input format processors: markdown folders, JSONL, and Parquet files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use crate::chunk::{ChunkConfig, MarkdownChunker};
use v_hnsw_embed::Model2VecModel;
use v_hnsw_storage::StorageEngine;

use super::pipeline::{AddRecord, process_records};
use crate::commands::common;
use crate::commands::file_index;
use crate::is_interrupted;

/// Process markdown folder: chunk files, embed, insert.
pub fn process_markdown_folder(
    db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64)> {
    let chunker = MarkdownChunker::new(ChunkConfig {
        target_size: 1000,
        overlap: 200,
        min_size: 100,
        include_heading_context: true,
    });

    // Collect all markdown files recursively
    let md_files: Vec<PathBuf> = walkdir::WalkDir::new(input_path)
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
        anyhow::bail!("No markdown files found in {}", input_path.display());
    }

    println!("Found {} markdown files", md_files.len());

    // First pass: collect all chunks and track file metadata
    let mut records: Vec<AddRecord> = Vec::new();
    let mut file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new();

    for md_path in &md_files {
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

        // Get file metadata
        let mtime = common::get_file_mtime(md_path).unwrap_or(0);
        let size = file_index::get_file_size(md_path).unwrap_or(0);
        let mut chunk_ids = Vec::new();

        for chunk in chunks {
            let id = common::generate_id(&source, chunk.chunk_index);
            chunk_ids.push(id);
            records.push(AddRecord {
                id,
                text: chunk.text,
                source: source.clone(),
                title: title.clone(),
                tags: tags.clone(),
                chunk_index: chunk.chunk_index,
                chunk_total,
                source_modified_at: mtime,
            });
        }

        file_metadata_map.insert(source, (mtime, size, chunk_ids));
    }

    println!("Total chunks to process: {}", records.len());

    // Process in batches
    let result = process_records(records, model, engine)?;

    // Save file metadata index
    let mut file_index = file_index::load_file_index(db_path)?;
    for (path, (mtime, size, chunk_ids)) in file_metadata_map {
        file_index.update_file(path, mtime, size, chunk_ids);
    }
    file_index::save_file_index(db_path, &file_index)?;

    Ok(result)
}

/// Process JSONL file: parse records, embed, insert.
pub fn process_jsonl(
    _db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64)> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(input_path)
        .with_context(|| format!("Failed to open {}", input_path.display()))?;
    let reader = BufReader::new(file);

    let mut records: Vec<AddRecord> = Vec::new();
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

        records.push(AddRecord {
            id,
            text,
            source: item_source,
            title,
            tags,
            chunk_index: 0,
            chunk_total: 1,
            source_modified_at: 0,
        });
    }

    println!("Total records to process: {}", records.len());

    process_records(records, model, engine)
}

/// Process Parquet file: read rows, embed, insert.
pub fn process_parquet(
    _db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64)> {
    use arrow::array::{Array, StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(input_path)
        .with_context(|| format!("Failed to open {}", input_path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| "Failed to create Parquet reader")?;

    let reader = builder
        .build()
        .with_context(|| "Failed to build Parquet reader")?;

    let mut records: Vec<AddRecord> = Vec::new();
    let source = common::normalize_source(input_path);
    let mut row_idx = 0u64;

    for batch_result in reader {
        if is_interrupted() {
            break;
        }

        let batch = batch_result.with_context(|| "Failed to read Parquet batch")?;
        let schema = batch.schema();

        // Find text column
        let text_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "text" || f.name() == "content")
            .ok_or_else(|| {
                anyhow::anyhow!("No 'text' or 'content' column found in Parquet file")
            })?;

        let text_array = batch
            .column(text_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Text column is not a string array"))?;

        // Optional id column
        let id_col_idx = schema.fields().iter().position(|f| f.name() == "id");
        let id_array = id_col_idx.and_then(|idx| {
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<UInt64Array>()
        });

        // Optional title column
        let title_col_idx = schema.fields().iter().position(|f| f.name() == "title");
        let title_array = title_col_idx.and_then(|idx| {
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<StringArray>()
        });

        for i in 0..batch.num_rows() {
            if text_array.is_null(i) {
                row_idx += 1;
                continue;
            }

            let text = text_array.value(i).to_string();
            if text.is_empty() {
                row_idx += 1;
                continue;
            }

            let id = id_array
                .and_then(|arr| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .unwrap_or_else(|| common::generate_id(&source, row_idx as usize));

            let title = title_array.and_then(|arr| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i).to_string())
                }
            });

            records.push(AddRecord {
                id,
                text,
                source: source.clone(),
                title,
                tags: Vec::new(),
                chunk_index: 0,
                chunk_total: 1,
                source_modified_at: 0,
            });

            row_idx += 1;
        }
    }

    println!("Total records to process: {}", records.len());

    process_records(records, model, engine)
}
