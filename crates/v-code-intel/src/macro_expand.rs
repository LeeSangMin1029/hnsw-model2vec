//! Extract macro-generated function chunks via `cargo expand`.
//!
//! Runs `cargo expand -p <crate>` for each workspace crate, parses the expanded
//! output with tree-sitter, and returns synthetic `ParsedChunk`s for functions
//! that only exist in expanded form (e.g., `define_chunker!` → `fn new()`).

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

use crate::parse::ParsedChunk;

/// Result of macro expansion for a single crate.
struct ExpandedFunc {
    /// Qualified name: `OwnerType::method` or bare `func_name`.
    name: String,
    /// Function signature (e.g., `pub fn new(config: CodeChunkConfig) -> Self`).
    signature: String,
    /// Return type extracted from signature.
    return_type: Option<String>,
}

/// Discover workspace crate names from `Cargo.toml` members.
fn discover_crates(project_root: &Path) -> Vec<String> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version=1"])
        .current_dir(project_root)
        .output()
        .ok();
    let output = match output {
        Some(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let json: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    json["packages"]
        .as_array()
        .map(|pkgs| {
            pkgs.iter()
                .filter_map(|p| p["name"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Run `cargo expand -p <crate>` and return expanded source code.
fn expand_crate(project_root: &Path, crate_name: &str) -> Option<String> {
    let output = Command::new("cargo")
        .args(["expand", "-p", crate_name])
        .current_dir(project_root)
        .output()
        .ok()?;
    if output.status.success() {
        String::from_utf8(output.stdout).ok()
    } else {
        // Try with --lib for crates that have both lib+bin targets.
        let output = Command::new("cargo")
            .args(["expand", "-p", crate_name, "--lib"])
            .current_dir(project_root)
            .output()
            .ok()?;
        if output.status.success() {
            String::from_utf8(output.stdout).ok()
        } else {
            None
        }
    }
}

/// Parse expanded Rust source and extract function definitions with their
/// owning impl type.
fn extract_functions_from_expanded(source: &str) -> Vec<ExpandedFunc> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .ok();
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let src = source.as_bytes();
    let root = tree.root_node();
    let mut funcs = Vec::new();
    collect_functions(&root, src, None, &mut funcs);
    funcs
}

fn collect_functions(
    node: &tree_sitter::Node,
    src: &[u8],
    current_impl_type: Option<&str>,
    out: &mut Vec<ExpandedFunc>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "impl_item" => {
                // Extract the type name from `impl TypeName { ... }`
                let impl_type = child
                    .child_by_field_name("type")
                    .and_then(|t| t.utf8_text(src).ok())
                    .map(|s| s.to_owned());
                if let Some(body) = child.child_by_field_name("body") {
                    collect_functions(
                        &body,
                        src,
                        impl_type.as_deref(),
                        out,
                    );
                }
            }
            "function_item" => {
                let name = child
                    .child_by_field_name("name")
                    .and_then(|n| n.utf8_text(src).ok())
                    .unwrap_or("");
                if name.is_empty() {
                    continue;
                }
                let qualified = if let Some(owner) = current_impl_type {
                    format!("{owner}::{name}")
                } else {
                    name.to_owned()
                };

                let sig = child
                    .utf8_text(src)
                    .ok()
                    .and_then(|text| {
                        // Extract just the signature (up to the body block).
                        text.find('{').map(|pos| text[..pos].trim().to_owned())
                    })
                    .unwrap_or_default();

                let return_type = extract_return_type(&child, src);

                out.push(ExpandedFunc {
                    name: qualified,
                    signature: sig,
                    return_type,
                });
            }
            "mod_item" => {
                // Recurse into module items.
                if let Some(body) = child.child_by_field_name("body") {
                    collect_functions(&body, src, None, out);
                }
            }
            _ => {
                collect_functions(&child, src, current_impl_type, out);
            }
        }
    }
}

/// Extract return type from a function_item node.
fn extract_return_type(func: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    let text = func.utf8_text(src).ok()?;
    let sig = text.split('{').next()?;
    let ret = sig.rsplit_once("->")?;
    let ret_type = ret.1.trim().to_owned();
    if ret_type.is_empty() {
        None
    } else {
        Some(ret_type)
    }
}

/// Run macro expansion for all workspace crates and return synthetic chunks
/// for functions that don't exist in the original chunk set.
///
/// `existing_names` should contain lowercased qualified names of all known chunks.
pub fn expand_macro_chunks(
    project_root: &Path,
    existing_names: &HashSet<String>,
) -> Vec<ParsedChunk> {
    let crates = discover_crates(project_root);
    if crates.is_empty() {
        return Vec::new();
    }

    // Parallel expand: run cargo expand for all crates concurrently (max 4).
    let expanded: Vec<(String, Vec<ExpandedFunc>)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for batch in crates.chunks(4) {
            for krate in batch {
                let root = project_root;
                let name = krate.clone();
                handles.push(s.spawn(move || {
                    let source = expand_crate(root, &name)?;
                    let funcs = extract_functions_from_expanded(&source);
                    Some((name, funcs))
                }));
            }
        }
        handles
            .into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .collect()
    });

    let mut result = Vec::new();
    for (krate, funcs) in expanded {
        for f in funcs {
            let lower_name = f.name.to_lowercase();
            if existing_names.contains(&lower_name) {
                continue;
            }
            // Resolve `Self` in return type to owner type.
            let return_type = f.return_type.map(|rt| {
                if let Some((owner, _method)) = f.name.rsplit_once("::") {
                    rt.replace("Self", owner)
                } else {
                    rt
                }
            });

            result.push(ParsedChunk {
                kind: "function".to_owned(),
                name: f.name.clone(),
                file: format!("<macro-expand:{krate}>"),
                lines: None,
                signature: if f.signature.is_empty() {
                    None
                } else {
                    Some(f.signature)
                },
                calls: Vec::new(),
                call_lines: Vec::new(),
                types: Vec::new(),
                imports: Vec::new(),
                string_args: Vec::new(),
                param_flows: Vec::new(),
                param_types: Vec::new(),
                field_types: Vec::new(),
                local_types: Vec::new(),
                let_call_bindings: Vec::new(),
                return_type,
                field_accesses: Vec::new(),
                enum_variants: Vec::new(),
            });
        }
    }

    result
}

/// Cached version: check db/cache for expanded chunks, regenerate if stale.
pub fn expand_macro_chunks_cached(
    project_root: &Path,
    db_path: &Path,
    existing_names: &HashSet<String>,
) -> Vec<ParsedChunk> {
    let cache_path = db_path
        .parent()
        .unwrap_or(db_path)
        .join(db_path.file_name().unwrap_or_default())
        .with_extension("macro_expand.bin");

    // Cache invalidation: use Cargo.lock mtime as proxy for code changes.
    let lock_mtime = project_root
        .join("Cargo.lock")
        .metadata()
        .and_then(|m| m.modified())
        .ok();
    let cache_mtime = cache_path
        .metadata()
        .and_then(|m| m.modified())
        .ok();

    // If cache exists and is newer than Cargo.lock, load it.
    if let (Some(lock_t), Some(cache_t)) = (lock_mtime, cache_mtime) {
        if cache_t > lock_t {
            if let Ok(data) = std::fs::read(&cache_path) {
                if let Ok(chunks) = bincode::serde::decode_from_slice::<Vec<ParsedChunk>, _>(
                    &data,
                    bincode::config::standard(),
                ) {
                    let filtered: Vec<ParsedChunk> = chunks
                        .0
                        .into_iter()
                        .filter(|c| !existing_names.contains(&c.name.to_lowercase()))
                        .collect();
                    if !filtered.is_empty() {
                        eprintln!(
                            "    [macro] cache hit: {} macro-expanded chunks",
                            filtered.len()
                        );
                        return filtered;
                    }
                }
            }
        }
    }

    let chunks = expand_macro_chunks(project_root, existing_names);
    if !chunks.is_empty() {
        // Save to cache.
        if let Ok(data) =
            bincode::serde::encode_to_vec(&chunks, bincode::config::standard())
        {
            let _ = std::fs::write(&cache_path, data);
        }
        eprintln!(
            "    [macro] expanded {} macro-generated chunks",
            chunks.len()
        );
    }

    chunks
}
