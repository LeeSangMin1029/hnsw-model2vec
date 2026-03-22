//! Code-specific ingestion utilities.
//!
//! Chunks code files via daemon RA, converts to CodeChunkEntry for DB storage.

use std::collections::HashMap;
use std::path::Path;

use v_code_intel::chunk_types as chunk_code;
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

/// Chunk files via daemon RA instead of tree-sitter.
/// Calls daemon "code/chunk" RPC and converts RaChunks → CodeChunkEntry.
pub fn chunk_via_daemon(
    code_files: &[impl AsRef<Path> + Sync],
    db_path: &Path,
    entries: &mut Vec<CodeChunkEntry>,
    file_metadata_map: &mut HashMap<String, (u64, u64, Vec<u64>)>,
) -> Result<(), anyhow::Error> {
    // Collect relative file paths (matching RA's file_map keys).
    let db_parent = db_path.parent().filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let workspace_root = db_parent.canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| db_parent.to_path_buf()));
    let mut root_str = v_hnsw_core::strip_unc_prefix(&workspace_root.to_string_lossy())
        .replace('\\', "/");
    if !root_str.ends_with('/') { root_str.push('/'); }

    let mut file_keys: Vec<String> = Vec::new();
    let mut path_map: HashMap<String, std::path::PathBuf> = HashMap::new();
    for code_path in code_files {
        let code_path = code_path.as_ref();
        let abs = code_path.canonicalize().unwrap_or_else(|_| code_path.to_path_buf());
        let abs_str = v_hnsw_core::strip_unc_prefix(&abs.to_string_lossy())
            .replace('\\', "/");
        let rel = abs_str.strip_prefix(&root_str)
            .unwrap_or(&abs_str)
            .to_owned();
        path_map.insert(rel.clone(), code_path.to_path_buf());
        file_keys.push(rel);
    }

    if let Some(first) = file_keys.first() {
        eprintln!("  [chunk] root_str: {root_str}");
        eprintln!("  [chunk] first file key: {first}");
    }
    let t0 = std::time::Instant::now();
    let params = serde_json::json!({ "files": file_keys });

    let result = v_hnsw_storage::daemon_client::daemon_rpc("code/chunk", params, 300)
        .map_err(|e| anyhow::anyhow!("daemon code/chunk failed: {e}"))?;

    let ra_chunks: Vec<serde_json::Value> = serde_json::from_value(result)
        .map_err(|e| anyhow::anyhow!("failed to parse code/chunk response: {e}"))?;

    eprintln!("  chunk (daemon): {:.1}s ({} chunks)", t0.elapsed().as_secs_f64(), ra_chunks.len());

    // Group by file for metadata.
    let mut file_chunks: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
    for chunk in ra_chunks {
        let file = chunk.get("file").and_then(|v| v.as_str()).unwrap_or("").to_owned();
        file_chunks.entry(file).or_default().push(chunk);
    }

    for (file_key, chunks) in &file_chunks {
        let code_path = match path_map.get(file_key) {
            Some(p) => p,
            None => continue,
        };
        let source = normalize_source(code_path);
        let file_path_str = code_path.to_string_lossy().to_string();
        let mtime = get_file_mtime(code_path).unwrap_or(0);
        let size = file_index::get_file_size(code_path).unwrap_or(0);
        let ext = code_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let lang = v_hnsw_core::lang_for_ext(ext);

        let mut chunk_ids = Vec::with_capacity(chunks.len());
        for (idx, chunk_val) in chunks.iter().enumerate() {
            let code_chunk = ra_chunk_to_code_chunk(chunk_val, idx);
            let id = generate_id(&source, idx);
            chunk_ids.push(id);
            entries.push(CodeChunkEntry {
                chunk: code_chunk,
                source: source.clone(),
                file_path_str: file_path_str.clone(),
                mtime,
                lang,
            });
        }
        file_metadata_map.insert(source, (mtime, size, chunk_ids));
    }

    Ok(())
}

/// Convert a daemon RaChunk JSON value → CodeChunk.
fn ra_chunk_to_code_chunk(val: &serde_json::Value, index: usize) -> chunk_code::CodeChunk {
    let kind_str = val.get("kind").and_then(|v| v.as_str()).unwrap_or("function");
    let kind = match kind_str {
        "function" => chunk_code::CodeNodeKind::Function,
        "struct" => chunk_code::CodeNodeKind::Struct,
        "enum" => chunk_code::CodeNodeKind::Enum,
        "trait" => chunk_code::CodeNodeKind::Trait,
        "impl" => chunk_code::CodeNodeKind::Impl,
        _ => chunk_code::CodeNodeKind::Function,
    };

    let lines = val.get("lines").and_then(|v| v.as_array());
    let start_line = lines.and_then(|a| a.first()).and_then(|v| v.as_u64()).unwrap_or(1) as usize - 1;
    let end_line = lines.and_then(|a| a.get(1)).and_then(|v| v.as_u64()).unwrap_or(1) as usize - 1;

    let calls: Vec<String> = val.get("calls").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_owned())).collect())
        .unwrap_or_default();
    let call_lines: Vec<u32> = val.get("call_lines").and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
        .unwrap_or_default();

    let str_vec = |key: &str| -> Vec<String> {
        val.get(key).and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_owned())).collect())
            .unwrap_or_default()
    };
    let str_pair_vec = |key: &str| -> Vec<(String, String)> {
        val.get(key).and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| {
                let arr = v.as_array()?;
                Some((arr.first()?.as_str()?.to_owned(), arr.get(1)?.as_str()?.to_owned()))
            }).collect())
            .unwrap_or_default()
    };

    chunk_code::CodeChunk {
        text: val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
        kind,
        name: val.get("name").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
        signature: val.get("signature").and_then(|v| v.as_str().map(|s| s.to_owned())),
        doc_comment: None,
        visibility: val.get("visibility").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
        start_line,
        end_line,
        start_byte: val.get("start_byte").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        end_byte: val.get("end_byte").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        chunk_index: index,
        imports: str_vec("imports"),
        calls,
        call_lines,
        type_refs: str_vec("types"),
        param_types: str_pair_vec("param_types"),
        field_types: str_pair_vec("field_types"),
        return_type: val.get("return_type").and_then(|v| v.as_str().map(|s| s.to_owned())),
        ast_hash: 0,
        body_hash: 0,
        sub_blocks: Vec::new(),
        string_args: Vec::new(),
        param_flows: Vec::new(),
        local_types: str_pair_vec("local_types"),
        let_call_bindings: str_pair_vec("let_call_bindings"),
        field_accesses: str_pair_vec("field_accesses"),
        enum_variants: str_vec("enum_variants"),
        is_test: val.get("is_test").and_then(|v| v.as_bool()).unwrap_or(false),
    }
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

#[cfg(test)]
#[path = "tests/ingest.rs"]
mod tests;
