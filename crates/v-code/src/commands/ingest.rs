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
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    let ns_read = AtomicU64::new(0);
    let ns_parse = AtomicU64::new(0);
    let ns_meta = AtomicU64::new(0);
    let max_parse = AtomicU64::new(0);
    let t_par = std::time::Instant::now();
    let per_file: Vec<_> = code_files
        .par_iter()
        .filter_map(|code_path| {
            let code_path = code_path.as_ref();
            let t_r = std::time::Instant::now();
            let source_code = std::fs::read_to_string(code_path).ok()?;
            ns_read.fetch_add(t_r.elapsed().as_nanos() as u64, Relaxed);
            let ext = code_path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let t_p = std::time::Instant::now();
            let chunks = chunk_code::chunk_for_language(ext, &source_code)?;
            let parse_ns = t_p.elapsed().as_nanos() as u64;
            ns_parse.fetch_add(parse_ns, Relaxed);
            max_parse.fetch_max(parse_ns, Relaxed);
            if chunks.is_empty() {
                return None;
            }
            let t_m = std::time::Instant::now();
            let source = normalize_source(code_path);
            let file_path_str = code_path.to_string_lossy().to_string();
            let mtime = get_file_mtime(code_path).unwrap_or(0);
            let size = file_index::get_file_size(code_path).unwrap_or(0);
            let lang = chunk_code::lang_for_extension(ext).unwrap_or("unknown");
            ns_meta.fetch_add(t_m.elapsed().as_nanos() as u64, Relaxed);
            Some((chunks, source, file_path_str, mtime, size, lang))
        })
        .collect();
    let par_wall = t_par.elapsed();
    eprintln!("    chunk thread-sum: read={:.1}s  parse={:.1}s  meta={:.1}s  max_parse={:.0}ms  (/{} cores)  par_wall={:.1}s",
        ns_read.load(Relaxed) as f64 / 1e9,
        ns_parse.load(Relaxed) as f64 / 1e9,
        ns_meta.load(Relaxed) as f64 / 1e9,
        max_parse.load(Relaxed) as f64 / 1e6,
        rayon::current_num_threads(),
        par_wall.as_secs_f64());

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
