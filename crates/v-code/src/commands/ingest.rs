//! Code-specific ingestion utilities.
//!
//! Moved from `v-hnsw-cli::commands::ingest` to decouple tree-sitter
//! from the shared CLI library (v-hnsw binary doesn't need code parsing).

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;
use v_code_chunk as chunk_code;
use v_hnsw_cli::commands::file_index;
use v_hnsw_cli::commands::file_utils::{generate_id, get_file_mtime, normalize_source};
use v_hnsw_cli::commands::ingest::IngestRecord;

/// Intermediate data collected per code chunk before `called_by` resolution.
pub struct CodeChunkEntry {
    pub chunk: chunk_code::CodeChunk,
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
    chunk: &chunk_code::CodeChunk,
    source: &str,
    file_path_str: &str,
    lang: &str,
    mtime: u64,
    chunk_total: usize,
    called_by: &[String],
) -> IngestRecord {
    let id = generate_id(source, chunk.chunk_index);
    let embed_text = chunk.to_embed_text(file_path_str, called_by);

    let is_test = v_code_intel::graph::is_test_path(source)
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
    code_files: &[impl AsRef<Path> + Sync],
    is_interrupted: impl Fn() -> bool,
    entries: &mut Vec<CodeChunkEntry>,
    file_metadata_map: &mut HashMap<String, (u64, u64, Vec<u64>)>,
) {
    if is_interrupted() {
        return;
    }

    // Parallel phase: read + parse all files concurrently.
    let per_file: Vec<_> = code_files
        .par_iter()
        .filter_map(|code_path| {
            let code_path = code_path.as_ref();
            let source_code = std::fs::read_to_string(code_path).ok()?;
            let ext = code_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let chunks = chunk_code::chunk_for_language(ext, &source_code)?;
            if chunks.is_empty() {
                return None;
            }
            let source = normalize_source(code_path);
            let file_path_str = code_path.to_string_lossy().to_string();
            let mtime = get_file_mtime(code_path).unwrap_or(0);
            let size = file_index::get_file_size(code_path).unwrap_or(0);
            let lang = chunk_code::lang_for_extension(ext).unwrap_or("unknown");
            Some((chunks, source, file_path_str, mtime, size, lang))
        })
        .collect();

    // Sequential merge (cheap: just moves data).
    for (chunks, source, file_path_str, mtime, size, lang) in per_file {
        let mut chunk_ids = Vec::with_capacity(chunks.len());
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


#[cfg(test)]
#[path = "tests/ingest.rs"]
mod tests;
