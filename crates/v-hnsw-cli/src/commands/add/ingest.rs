//! Input format processors: markdown folders and JSONL files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use crate::chunk::{ChunkConfig, MarkdownChunker};
use v_hnsw_embed::Model2VecModel;
use v_hnsw_storage::StorageEngine;

use super::pipeline::process_records;
use crate::commands::common::IngestRecord;
use crate::commands::common;
use crate::commands::file_index;
use crate::is_interrupted;

/// Process markdown folder: chunk files, embed, insert.
pub fn process_markdown_folder(
    db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
    exclude: &[String],
) -> Result<(u64, u64, Vec<u64>)> {
    let chunker = MarkdownChunker::new(ChunkConfig {
        target_size: 1000,
        overlap: 200,
        min_size: 100,
        include_heading_context: true,
    });

    // Collect all markdown files recursively
    let md_files: Vec<PathBuf> = walkdir::WalkDir::new(input_path)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                !common::should_skip_dir(e.file_name(), exclude)
            } else {
                true
            }
        })
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
    let mut records: Vec<IngestRecord> = Vec::new();
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
            records.push(IngestRecord {
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
    let (inserted, errors, inserted_ids) = process_records(records, model, engine)?;

    // Save file metadata index
    let mut file_index = file_index::load_file_index(db_path)?;
    for (path, (mtime, size, chunk_ids)) in file_metadata_map {
        file_index.update_file(path, mtime, size, chunk_ids);
    }
    file_index::save_file_index(db_path, &file_index)?;

    Ok((inserted, errors, inserted_ids))
}

/// Intermediate data collected per code chunk before `called_by` resolution.
struct CodeChunkEntry {
    chunk: crate::chunk_code::CodeChunk,
    source: String,
    file_path_str: String,
    mtime: u64,
}

/// Build `called_by` reverse index from all chunks' `calls` data.
///
/// For each call target, extracts the bare function name (last segment after
/// `::` or `.`) and maps it to the set of callers (qualified chunk names).
/// Returns `HashMap<bare_fn_name, Vec<caller_name>>`.
fn build_called_by_index(entries: &[CodeChunkEntry]) -> HashMap<String, Vec<String>> {
    let mut reverse: HashMap<String, Vec<String>> = HashMap::new();

    for entry in entries {
        let caller = &entry.chunk.name;
        for call in &entry.chunk.calls {
            // Extract bare function name: "self.method" → "method",
            // "Module::func" → "func", "validate_amount" → "validate_amount"
            let bare = call
                .rsplit_once("::")
                .map(|(_, name)| name)
                .or_else(|| call.rsplit_once('.').map(|(_, name)| name))
                .unwrap_or(call);

            reverse
                .entry(bare.to_owned())
                .or_default()
                .push(caller.clone());
        }
    }

    // Deduplicate callers per target
    for callers in reverse.values_mut() {
        callers.sort();
        callers.dedup();
    }

    reverse
}

/// Look up `called_by` entries for a given chunk name.
///
/// Checks both the full qualified name and the bare (last segment) name
/// against the reverse index built by [`build_called_by_index`].
fn lookup_called_by<'a>(
    reverse: &'a HashMap<String, Vec<String>>,
    chunk_name: &str,
) -> Vec<&'a str> {
    // Bare name of this chunk: "PaymentIntent::new" → "new"
    let bare = chunk_name
        .rsplit_once("::")
        .map(|(_, name)| name)
        .unwrap_or(chunk_name);

    let mut result: Vec<&str> = Vec::new();

    if let Some(callers) = reverse.get(bare) {
        for c in callers {
            // Don't list self-calls
            if c != chunk_name {
                result.push(c.as_str());
            }
        }
    }

    // Also check the full qualified name as a key (e.g., someone calls "Foo::bar")
    if bare != chunk_name
        && let Some(callers) = reverse.get(chunk_name)
    {
        for c in callers {
            if c != chunk_name && !result.contains(&c.as_str()) {
                result.push(c.as_str());
            }
        }
    }

    result.sort();
    result
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
) -> Result<(u64, u64, Vec<u64>)> {
    use crate::chunk_code::{CodeChunkConfig, RustCodeChunker};

    let chunker = RustCodeChunker::new(CodeChunkConfig::default());

    // Collect all supported code files recursively
    let code_files: Vec<PathBuf> = walkdir::WalkDir::new(input_path)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                !common::should_skip_dir(e.file_name(), exclude)
            } else {
                true
            }
        })
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(crate::chunk_code::is_supported_code_file)
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if code_files.is_empty() {
        anyhow::bail!("No supported code files found in {}", input_path.display());
    }

    println!("Found {} code files", code_files.len());

    // === Pass 1: Chunk all files, collect CodeChunk + metadata ===
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

        let chunks = chunker.chunk(&source_code);
        if chunks.is_empty() {
            continue;
        }

        let source = common::normalize_source(code_path);
        let file_path_str = code_path.to_string_lossy().to_string();
        let mtime = common::get_file_mtime(code_path).unwrap_or(0);
        let size = file_index::get_file_size(code_path).unwrap_or(0);
        let mut chunk_ids = Vec::new();

        for chunk in chunks {
            let id = common::generate_id(&source, chunk.chunk_index);
            chunk_ids.push(id);
            entries.push(CodeChunkEntry {
                chunk,
                source: source.clone(),
                file_path_str: file_path_str.clone(),
                mtime,
            });
        }

        file_metadata_map.insert(source, (mtime, size, chunk_ids));
    }

    // === Pass 2: Build called_by reverse index ===
    let reverse_index = build_called_by_index(&entries);

    let called_by_count: usize = reverse_index.values().map(Vec::len).sum();
    println!(
        "Built called_by index: {} targets, {} reverse edges",
        reverse_index.len(),
        called_by_count
    );

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

        let id = common::generate_id(&entry.source, chunk.chunk_index);
        let embed_text = chunk.to_embed_text(&entry.file_path_str, &called_by);

        // Build tags from code metadata
        let mut tags = vec![
            format!("kind:{}", chunk.kind.as_str()),
            format!("lang:rust"),
        ];
        if !chunk.visibility.is_empty() {
            tags.push(format!("vis:{}", chunk.visibility));
        }
        // Add caller tags for tag-based filtering
        for caller in &called_by {
            tags.push(format!("caller:{caller}"));
        }

        records.push(IngestRecord {
            id,
            text: embed_text,
            source: entry.source.clone(),
            title: Some(chunk.name.clone()),
            tags,
            chunk_index: chunk.chunk_index,
            chunk_total,
            source_modified_at: entry.mtime,
        });
    }

    println!("Total code chunks to process: {}", records.len());

    let (inserted, errors, inserted_ids) = process_records(records, model, engine)?;

    // Save file metadata index
    let mut file_index = file_index::load_file_index(db_path)?;
    for (path, (mtime, size, chunk_ids)) in file_metadata_map {
        file_index.update_file(path, mtime, size, chunk_ids);
    }
    file_index::save_file_index(db_path, &file_index)?;

    Ok((inserted, errors, inserted_ids))
}

/// Process JSONL file: parse records, embed, insert.
pub fn process_jsonl(
    _db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64, Vec<u64>)> {
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
        });
    }

    println!("Total records to process: {}", records.len());

    process_records(records, model, engine)
}

