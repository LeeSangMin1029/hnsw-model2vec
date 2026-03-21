//! Type inference via in-process `RaInstance` hover queries.
//!
//! Collects receiver types and return types from `let x = foo()` bindings
//! by hovering on variable declarations with rust-analyzer.

use std::collections::HashMap;

/// LSP-resolved type information for the call graph.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LspTypes {
    /// fn_short_name → leaf_type (for return_type_map overlay).
    pub return_types: HashMap<String, String>,
    /// chunk_index → {var_name → leaf_type} (direct receiver_type injection).
    pub receiver_types: HashMap<usize, HashMap<String, String>>,
}

/// Batch hover query: file path + (line, col) → type string.
#[derive(Debug, Clone)]
pub struct HoverQuery {
    pub file: String,
    pub line: u32,
    pub col: u32,
    pub var_name: String,
    pub chunk_idx: usize,
}

/// Collect LSP type information using an in-process `RaInstance`.
///
/// For each `let x = foo()` binding in chunks, hovers on `x` to get its type.
/// Returns both return_types (fn→type) and receiver_types (chunk→{var→type}).
pub fn collect_types_via_ra(
    chunks: &[crate::parse::ParsedChunk],
    ra: &v_lsp::instance::RaInstance,
) -> LspTypes {
    let queries = build_hover_queries_direct(chunks);
    if queries.is_empty() {
        return LspTypes::default();
    }
    eprintln!("    [ra-hover] querying {} hover positions...", queries.len());

    // Map var_name → callee_name for return type extraction.
    let var_to_callee: HashMap<(usize, String), Vec<String>> = {
        let mut m: HashMap<(usize, String), Vec<String>> = HashMap::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            for (var_name, callee_name) in &chunk.let_call_bindings {
                m.entry((idx, var_name.to_lowercase()))
                    .or_default()
                    .push(callee_name.to_lowercase());
            }
        }
        m
    };

    let mut lsp_types = LspTypes::default();
    let mut null_count = 0usize;
    let mut hover_ok = 0usize;
    let mut hover_err = 0usize;

    for q in &queries {
        match ra.hover(&q.file, q.line, q.col) {
            Ok(Some(type_str)) => {
                hover_ok += 1;
                let leaf = extract_leaf_type_from_rust(&type_str);
                if !leaf.is_empty() && leaf != "unknown" {
                    let var_lower = q.var_name.to_lowercase();

                    lsp_types.receiver_types
                        .entry(q.chunk_idx)
                        .or_default()
                        .entry(var_lower.clone())
                        .or_insert_with(|| leaf.clone());

                    let key = (q.chunk_idx, var_lower);
                    if let Some(callees) = var_to_callee.get(&key) {
                        for callee in callees {
                            lsp_types.return_types.entry(callee.clone())
                                .or_insert_with(|| leaf.clone());
                        }
                    }
                }
            }
            Ok(None) => null_count += 1,
            Err(_) => hover_err += 1,
        }
    }

    let receiver_count: usize = lsp_types.receiver_types.values().map(|m| m.len()).sum();
    eprintln!(
        "    [ra-hover] {} ok, {} null, {} err | receivers: {}, return_types: {}",
        hover_ok, null_count, hover_err, receiver_count, lsp_types.return_types.len(),
    );
    lsp_types
}

/// Extract leaf type from a full Rust type.
/// `Vec<String>` → `vec`, `HashMap<K, V>` → `hashmap`, `MyStruct` → `mystruct`
fn extract_leaf_type_from_rust(full_type: &str) -> String {
    let base = full_type.split('<').next().unwrap_or(full_type);
    let leaf = base.rsplit("::").next().unwrap_or(base);
    leaf.trim().to_lowercase()
}

/// Build hover queries using real source file paths (no stub workspace).
fn build_hover_queries_direct(
    chunks: &[crate::parse::ParsedChunk],
) -> Vec<HoverQuery> {
    let mut queries = Vec::new();
    let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if chunk.let_call_bindings.is_empty() {
            continue;
        }
        let Some((start_line, _end_line)) = chunk.lines else {
            continue;
        };

        let file_path = std::path::PathBuf::from(&chunk.file);
        if !file_path.exists() {
            continue;
        }

        let lines = file_cache.entry(chunk.file.clone()).or_insert_with(|| {
            std::fs::read_to_string(&file_path)
                .unwrap_or_default()
                .lines()
                .map(String::from)
                .collect()
        });

        for (var_name, _callee) in &chunk.let_call_bindings {
            let search_start = start_line.saturating_sub(1); // 1-based → 0-based
            let search_end = lines.len().min(start_line + 500);

            for line_idx in search_start..search_end {
                let line = &lines[line_idx];
                if let Some(col) = find_let_var_column(line, var_name) {
                    queries.push(HoverQuery {
                        file: chunk.file.clone(),
                        line: line_idx as u32,
                        col,
                        var_name: var_name.clone(),
                        chunk_idx,
                    });
                    break;
                }
            }
        }
    }

    queries.sort_by(|a, b| {
        a.file.cmp(&b.file)
            .then(a.line.cmp(&b.line))
            .then(a.col.cmp(&b.col))
    });
    queries.dedup_by(|a, b| a.file == b.file && a.line == b.line && a.col == b.col);

    queries
}

/// Find the column of a variable name in a `let` binding line.
///
/// Matches `let var_name` or `let mut var_name` patterns.
/// Returns the 0-based column of the variable name.
fn find_let_var_column(line: &str, var_name: &str) -> Option<u32> {
    // Pattern 1: `let var_name`
    let pat1 = format!("let {var_name}");
    if let Some(pos) = line.find(&pat1) {
        return Some((pos + 4) as u32); // skip "let "
    }
    // Pattern 2: `let mut var_name`
    let pat2 = format!("let mut {var_name}");
    if let Some(pos) = line.find(&pat2) {
        return Some((pos + 8) as u32); // skip "let mut "
    }
    None
}
