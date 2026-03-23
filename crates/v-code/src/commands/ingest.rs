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


/// Create code chunks from MIR chunk definitions + source files.
/// Reads source text from files using MIR-provided line ranges.
pub fn chunk_from_mir(
    mir_chunks: &[v_code_intel::mir_edges::MirChunk],
    db_path: &Path,
    entries: &mut Vec<CodeChunkEntry>,
    file_metadata_map: &mut HashMap<String, (u64, u64, Vec<u64>)>,
    changed_sources: Option<&std::collections::HashSet<String>>,
) -> Result<(), anyhow::Error> {
    use v_hnsw_cli::commands::file_utils::{generate_id, get_file_mtime, normalize_source};

    let db_parent = db_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let workspace_root = db_parent
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| db_parent.to_path_buf()));
    let mut root_str =
        v_hnsw_core::strip_unc_prefix(&workspace_root.to_string_lossy()).replace('\\', "/");
    if !root_str.ends_with('/') {
        root_str.push('/');
    }

    // Group MIR chunks by file — skip external crate files
    let mut by_file: HashMap<String, Vec<&v_code_intel::mir_edges::MirChunk>> = HashMap::new();
    for mc in mir_chunks {
        // Skip external crate files (.cargo/registry, rustup toolchain, etc.)
        if mc.file.contains(".cargo") || mc.file.contains("registry") || mc.file.contains("rustup") {
            continue;
        }
        by_file.entry(mc.file.clone()).or_default().push(mc);
    }

    for (file_key, chunks) in &by_file {
        // Deduplicate: same name in same file — prefer prod (is_test=false) over test
        let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        let mut deduped: Vec<&v_code_intel::mir_edges::MirChunk> = Vec::new();
        for mc in chunks {
            if let Some(&existing_idx) = seen.get(mc.name.as_str()) {
                // If existing is test and new is prod, replace
                if deduped[existing_idx].is_test && !mc.is_test {
                    deduped[existing_idx] = mc;
                }
                // Otherwise keep existing (first wins)
            } else {
                seen.insert(&mc.name, deduped.len());
                deduped.push(mc);
            }
        }
        let chunks = &deduped;
        // Resolve file path: try relative to workspace root
        let file_path = {
            let candidate = workspace_root.join(file_key);
            if candidate.exists() {
                candidate
            } else {
                std::path::PathBuf::from(file_key)
            }
        };
        if !file_path.exists() {
            continue;
        }

        let file_text = match std::fs::read_to_string(&file_path) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let all_lines: Vec<&str> = file_text.lines().collect();

        let source = normalize_source(&file_path);

        // Skip unchanged files if filter is provided
        if let Some(changed) = changed_sources {
            if !changed.contains(&source) {
                continue;
            }
        }

        let file_path_str = file_path.to_string_lossy().to_string();
        let mtime = get_file_mtime(&file_path).unwrap_or(0);
        let size = file_index::get_file_size(&file_path).unwrap_or(0);
        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let lang = v_hnsw_core::lang_for_ext(ext);

        // Extract imports once per file (top-level `use` statements)
        let imports: Vec<String> = all_lines
            .iter()
            .take_while(|line| {
                let trimmed = line.trim();
                trimmed.is_empty()
                    || trimmed.starts_with("//")
                    || trimmed.starts_with("use ")
                    || trimmed.starts_with("pub use ")
                    || trimmed.starts_with('#')
                    || trimmed.starts_with("mod ")
                    || trimmed.starts_with("pub mod ")
                    || trimmed.starts_with("extern ")
            })
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with("use ") || trimmed.starts_with("pub use ")
            })
            .map(|line| line.trim().to_owned())
            .collect();

        let mut chunk_ids = Vec::with_capacity(chunks.len());

        for (idx, mc) in chunks.iter().enumerate() {
            // Extract text from line range (1-based start_line, end_line)
            let start = mc.start_line.saturating_sub(1);
            let end = mc.end_line.min(all_lines.len());
            let chunk_lines: Vec<&str> = if start < end {
                all_lines[start..end].to_vec()
            } else {
                Vec::new()
            };
            let text = chunk_lines.join("\n");

            // Parse kind
            let kind = match mc.kind.as_str() {
                "fn" | "method" => chunk_code::CodeNodeKind::Function,
                "struct" => chunk_code::CodeNodeKind::Struct,
                "enum" => chunk_code::CodeNodeKind::Enum,
                "trait" => chunk_code::CodeNodeKind::Trait,
                "impl" => chunk_code::CodeNodeKind::Impl,
                _ => chunk_code::CodeNodeKind::Function,
            };

            // Extract signature: fn declaration up to `{`
            let signature = mc.signature.clone().or_else(|| {
                let first = chunk_lines.first()?;
                let sig_line = first.split('{').next()?.trim();
                if sig_line.is_empty() {
                    None
                } else {
                    Some(sig_line.to_owned())
                }
            });

            // Parse param_types from signature: `name: Type` patterns
            let param_types: Vec<(String, String)> = signature
                .as_deref()
                .and_then(|sig| {
                    let paren_start = sig.find('(')?;
                    let paren_end = sig.rfind(')')?;
                    if paren_start >= paren_end {
                        return None;
                    }
                    let params_str = &sig[paren_start + 1..paren_end];
                    let pairs: Vec<(String, String)> = params_str
                        .split(',')
                        .filter_map(|p| {
                            let p = p.trim();
                            if p == "self" || p == "&self" || p == "&mut self" || p.is_empty() {
                                return None;
                            }
                            let (name, ty) = p.split_once(':')?;
                            Some((name.trim().to_owned(), ty.trim().to_owned()))
                        })
                        .collect();
                    Some(pairs)
                })
                .unwrap_or_default();

            // Parse return_type from signature: after `->`
            let return_type: Option<String> = signature.as_deref().and_then(|sig| {
                let after_arrow = sig.split("->").nth(1)?;
                let rt = after_arrow.trim().trim_end_matches('{').trim();
                if rt.is_empty() {
                    None
                } else {
                    Some(rt.to_owned())
                }
            });

            // Parse field_types for structs: `name: Type,` patterns in body
            let field_types: Vec<(String, String)> =
                if kind == chunk_code::CodeNodeKind::Struct && chunk_lines.len() > 1 {
                    chunk_lines[1..]
                        .iter()
                        .filter_map(|line| {
                            let trimmed = line.trim().trim_end_matches(',');
                            if trimmed.starts_with("//") || trimmed.is_empty() || trimmed == "}" {
                                return None;
                            }
                            // Strip visibility prefix
                            let stripped = trimmed
                                .strip_prefix("pub(crate) ")
                                .or_else(|| trimmed.strip_prefix("pub(super) "))
                                .or_else(|| trimmed.strip_prefix("pub "))
                                .unwrap_or(trimmed);
                            let (name, ty) = stripped.split_once(':')?;
                            Some((name.trim().to_owned(), ty.trim().to_owned()))
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            // Parse enum_variants
            let enum_variants: Vec<String> =
                if kind == chunk_code::CodeNodeKind::Enum && chunk_lines.len() > 1 {
                    chunk_lines[1..]
                        .iter()
                        .filter_map(|line| {
                            let trimmed = line.trim().trim_end_matches(',');
                            if trimmed.starts_with("//")
                                || trimmed.is_empty()
                                || trimmed == "}"
                                || trimmed.starts_with('#')
                            {
                                return None;
                            }
                            let name = trimmed
                                .split(|c: char| c == '(' || c == '{' || c == ' ')
                                .next()?;
                            if name.is_empty() {
                                None
                            } else {
                                Some(name.to_owned())
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            // Determine visibility
            let visibility = mc
                .visibility
                .clone()
                .unwrap_or_else(|| {
                    let first_line = chunk_lines.first().map(|l| l.trim()).unwrap_or("");
                    if first_line.starts_with("pub(crate)") {
                        "pub(crate)".to_owned()
                    } else if first_line.starts_with("pub(super)") {
                        "pub(super)".to_owned()
                    } else if first_line.starts_with("pub") {
                        "pub".to_owned()
                    } else {
                        String::new()
                    }
                });

            // Compute byte offsets
            let start_byte: usize = all_lines
                .iter()
                .take(start)
                .map(|l| l.len() + 1) // +1 for newline
                .sum();
            let end_byte: usize = start_byte
                + chunk_lines
                    .iter()
                    .map(|l| l.len() + 1)
                    .sum::<usize>();

            let id = generate_id(&source, idx);
            chunk_ids.push(id);

            let code_chunk = chunk_code::CodeChunk {
                text,
                kind,
                name: mc.name.clone(),
                signature,
                doc_comment: None,
                visibility,
                start_line: start,
                end_line: end.saturating_sub(1),
                start_byte,
                end_byte,
                chunk_index: idx,
                imports: imports.clone(),
                calls: Vec::new(),
                call_lines: Vec::new(),
                type_refs: Vec::new(),
                param_types,
                field_types,
                return_type,
                ast_hash: 0,
                body_hash: 0,
                sub_blocks: Vec::new(),
                string_args: Vec::new(),
                param_flows: Vec::new(),
                local_types: Vec::new(),
                let_call_bindings: Vec::new(),
                field_accesses: Vec::new(),
                enum_variants,
                is_test: mc.is_test
                    || mc.file.contains("/tests/") || mc.file.contains("\\tests\\")
                    || mc.name.contains("::test_") || mc.name.starts_with("test_")
                    || chunk_lines
                        .first()
                        .is_some_and(|l| l.contains("#[test]") || l.contains("#[cfg(test)]")),
            };

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

#[cfg(test)]
#[path = "tests/ingest.rs"]
mod tests;
