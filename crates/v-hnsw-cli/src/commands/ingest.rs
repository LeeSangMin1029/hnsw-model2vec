//! Embedding + ingestion pipeline utilities.
//!
//! Shared types and functions for the `add` and `update` commands:
//! record construction, payload building, batched embedding with
//! length-sorted padding optimization, and batch insert.

use std::collections::HashMap;

use anyhow::{Context, Result};
use v_hnsw_core::{Payload, PayloadValue};
use v_hnsw_embed::EmbeddingModel;
use v_hnsw_storage::StorageEngine;

use std::path::Path;

use super::file_index;
use super::file_utils::{generate_id, get_file_mtime, normalize_source};

/// A record for batch ingestion (shared by add and update commands).
#[derive(Clone)]
pub struct IngestRecord {
    pub id: u64,
    pub text: String,
    pub source: String,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub chunk_index: usize,
    pub chunk_total: usize,
    pub source_modified_at: u64,
    /// Extra custom fields to merge into payload (e.g., ast_hash).
    pub custom: HashMap<String, PayloadValue>,
}

/// Build payload from source info.
pub fn make_payload(
    source: &str,
    title: Option<&str>,
    tags: &[String],
    chunk_index: usize,
    chunk_total: usize,
    source_modified_at: u64,
    extra_custom: &HashMap<String, PayloadValue>,
) -> Payload {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut custom = extra_custom.clone();
    if let Some(t) = title {
        custom.insert("title".to_string(), PayloadValue::String(t.to_string()));
    }

    Payload {
        source: source.to_string(),
        tags: tags.to_vec(),
        created_at: now,
        source_modified_at,
        chunk_index: chunk_index as u32,
        chunk_total: chunk_total as u32,
        custom,
    }
}

/// Max character length for text sent to embedding model.
const EMBED_MAX_CHARS: usize = 8000;

/// Truncate text at a char boundary for embedding.
///
/// Returns a new string if truncation occurs, otherwise the original slice.
pub fn truncate_for_embed(text: &str) -> &str {
    if text.len() <= EMBED_MAX_CHARS {
        return text;
    }
    let mut end = EMBED_MAX_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Embed texts with length-sorted batching to minimize padding waste.
pub fn embed_sorted(model: &dyn EmbeddingModel, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    // Sort indices by text length
    let mut indices: Vec<usize> = (0..texts.len()).collect();
    indices.sort_by_key(|&i| texts[i].len());

    let sorted: Vec<&str> = indices.iter().map(|&i| texts[i].as_str()).collect();
    let sorted_embs = model
        .embed(&sorted)
        .map_err(|e| anyhow::anyhow!("Embedding failed: {e}"))?;

    // Restore original order (consume sorted_embs to avoid clone)
    let mut embeddings = vec![Vec::new(); texts.len()];
    for (emb, &orig_idx) in sorted_embs.into_iter().zip(indices.iter()) {
        embeddings[orig_idx] = emb;
    }
    Ok(embeddings)
}

/// Embed a batch of records and insert into storage (no progress bar).
///
/// Used by `update::run_core` for per-file incremental ingestion.
/// For bulk ingestion with pipeline parallelism, use `pipeline::process_records`.
pub fn embed_and_insert(
    records: &[IngestRecord],
    model: &dyn EmbeddingModel,
    engine: &mut StorageEngine,
) -> Result<()> {
    let texts: Vec<String> = records
        .iter()
        .map(|r| truncate_for_embed(&r.text).to_string())
        .collect();

    let embeddings = embed_sorted(model, &texts).context("Embedding failed")?;

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
                rec.source_modified_at,
                &rec.custom,
            );
            (rec.id, emb.as_slice(), payload, rec.text.as_str())
        })
        .collect();

    engine
        .insert_batch(&items)
        .context("Failed to insert batch")?;

    Ok(())
}

// ── Code chunk utilities ────────────────────────────────────────────────

/// Intermediate data collected per code chunk before `called_by` resolution.
pub struct CodeChunkEntry {
    pub chunk: crate::chunk_code::CodeChunk,
    pub source: String,
    pub file_path_str: String,
    pub mtime: u64,
    pub lang: &'static str,
}

/// Build `called_by` reverse index from all chunks' `calls` data.
///
/// For each call target, extracts the bare function name (last segment after
/// `::` or `.`) and maps it to the set of callers (qualified chunk names).
pub fn build_called_by_index(entries: &[CodeChunkEntry]) -> HashMap<String, Vec<String>> {
    let mut reverse: HashMap<String, Vec<String>> = HashMap::new();

    for entry in entries {
        let caller = &entry.chunk.name;
        for call in &entry.chunk.calls {
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
pub fn lookup_called_by<'a>(
    reverse: &'a HashMap<String, Vec<String>>,
    chunk_name: &str,
) -> Vec<&'a str> {
    let bare = chunk_name
        .rsplit_once("::")
        .map(|(_, name)| name)
        .unwrap_or(chunk_name);

    let mut result: Vec<&str> = Vec::new();

    if let Some(callers) = reverse.get(bare) {
        for c in callers {
            if c != chunk_name {
                result.push(c.as_str());
            }
        }
    }

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

/// Convert a single code chunk to an [`IngestRecord`].
///
/// Pass `called_by` slice for reverse-reference enrichment (empty slice if unavailable).
pub fn code_chunk_to_record(
    chunk: &crate::chunk_code::CodeChunk,
    source: &str,
    file_path_str: &str,
    lang: &str,
    mtime: u64,
    chunk_total: usize,
    called_by: &[String],
) -> IngestRecord {
    let id = generate_id(source, chunk.chunk_index);
    let embed_text = chunk.to_embed_text(file_path_str, called_by);

    let is_test = source.contains("/tests/")
        || source.contains("\\tests\\")
        || source.contains("/test/")
        || source.contains("\\test\\")
        || source.ends_with("_test.rs")
        || chunk.name.starts_with("test_");

    let mut tags = vec![
        format!("kind:{}", chunk.kind.as_str()),
        format!("lang:{lang}"),
        format!("role:{}", if is_test { "test" } else { "prod" }),
    ];
    if !chunk.visibility.is_empty() {
        tags.push(format!("vis:{}", chunk.visibility));
    }
    for caller in called_by {
        tags.push(format!("caller:{caller}"));
    }

    IngestRecord {
        id,
        text: embed_text,
        source: source.to_owned(),
        title: Some(chunk.name.clone()),
        tags,
        chunk_index: chunk.chunk_index,
        chunk_total,
        source_modified_at: mtime,
        custom: chunk.to_custom_fields(called_by),
    }
}

/// Convert chunked entries into `IngestRecord`s (Pass 2 + Pass 3 combined).
///
/// Builds the called_by reverse index and generates records with caller data.
pub fn entries_to_records(entries: &[CodeChunkEntry]) -> Vec<IngestRecord> {
    let reverse_index = build_called_by_index(entries);

    let chunk_total_map: HashMap<&str, usize> = {
        let mut m: HashMap<&str, usize> = HashMap::new();
        for entry in entries {
            *m.entry(&entry.source).or_default() += 1;
        }
        m
    };

    let mut records: Vec<IngestRecord> = Vec::with_capacity(entries.len());

    for entry in entries {
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

    records
}

/// Chunk code files via tree-sitter and collect entries with file metadata.
///
/// Iterates over `code_files`, reads each file, chunks it, and populates
/// `entries` and `file_metadata_map`. Skips files that cannot be read or
/// have no supported language / empty chunks.
///
/// `is_interrupted` is polled before each file to support graceful shutdown.
pub fn chunk_code_files(
    code_files: &[impl AsRef<Path>],
    is_interrupted: impl Fn() -> bool,
    entries: &mut Vec<CodeChunkEntry>,
    file_metadata_map: &mut HashMap<String, (u64, u64, Vec<u64>)>,
) {
    for code_path in code_files {
        if is_interrupted() {
            break;
        }
        let code_path = code_path.as_ref();

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
        let chunks = match crate::chunk_code::chunk_for_language(ext, &source_code) {
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

        let lang = crate::chunk_code::lang_for_extension(ext).unwrap_or("unknown");
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
}
