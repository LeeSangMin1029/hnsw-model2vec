//! Call graph adjacency list built from `CodeChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.
//!
//! ## HOW TO EXTEND
//! - Add new traversal methods (e.g., shortest path) as `impl CallGraph` methods.
//! - Add new fields to `CallGraph` and update `build()` + bump cache version.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::mir::MirCallMap;
use crate::parse::CodeChunk;

/// Cache format version — bump when struct layout changes.
const CACHE_VERSION: u8 = 2;

/// Pre-built call graph with bidirectional adjacency lists.
#[derive(bincode::Encode, bincode::Decode)]
pub struct CallGraph {
    /// Cache format version for invalidation.
    version: u8,
    /// chunk index -> chunk name.
    pub names: Vec<String>,
    /// chunk index -> file path.
    pub files: Vec<String>,
    /// chunk index -> kind (function, struct, etc.).
    pub kinds: Vec<String>,
    /// chunk index -> line range.
    pub lines: Vec<Option<(usize, usize)>>,
    /// chunk index -> signature.
    pub signatures: Vec<Option<String>>,
    /// Sorted (lowercase_name, chunk_index) pairs for binary search lookup.
    pub name_index: Vec<(String, u32)>,
    /// chunk index -> callee chunk indices.
    pub callees: Vec<Vec<u32>>,
    /// chunk index -> caller chunk indices.
    pub callers: Vec<Vec<u32>>,
    /// chunk index -> is_test flag.
    pub is_test: Vec<bool>,
}

impl CallGraph {
    /// Build the call graph from parsed code chunks.
    ///
    /// Resolution strategy: exact match on lowercase name first, then
    /// short name (last `::` segment) fallback.
    pub fn build(chunks: &[CodeChunk]) -> Self {
        let len = chunks.len();

        // Build name -> index maps for call resolution.
        let mut exact: BTreeMap<String, u32> = BTreeMap::new();
        let mut short: BTreeMap<String, u32> = BTreeMap::new();

        let mut names = Vec::with_capacity(len);
        let mut files = Vec::with_capacity(len);
        let mut kinds = Vec::with_capacity(len);
        let mut lines_vec = Vec::with_capacity(len);
        let mut signatures = Vec::with_capacity(len);
        let mut is_test = Vec::with_capacity(len);
        let mut name_index = Vec::with_capacity(len);

        for (i, c) in chunks.iter().enumerate() {
            let idx = i as u32;
            let lower = c.name.to_lowercase();

            exact.insert(lower.clone(), idx);
            if let Some(s) = c.name.rsplit("::").next() {
                short.entry(s.to_lowercase()).or_insert(idx);
            }

            name_index.push((lower, idx));
            names.push(c.name.clone());
            files.push(c.file.clone());
            kinds.push(c.kind.clone());
            lines_vec.push(c.lines);
            signatures.push(c.signature.clone());
            is_test.push(is_test_chunk(c));
        }

        name_index.sort_by(|a, b| a.0.cmp(&b.0));

        // Build adjacency lists.
        let mut callees: Vec<Vec<u32>> = vec![Vec::new(); len];
        let mut callers: Vec<Vec<u32>> = vec![Vec::new(); len];

        for (src, c) in chunks.iter().enumerate() {
            let imports = build_import_map(&c.imports);
            let self_type = owning_type(&c.name);
            let call_types = extract_call_types(&c.calls);
            for call in &c.calls {
                if let Some(tgt) = resolve_with_imports(call, &exact, &short, &imports, self_type.as_deref(), &c.types, &call_types) {
                    let tgt_usize = tgt as usize;
                    if tgt_usize != src {
                        callees[src].push(tgt);
                        callers[tgt_usize].push(src as u32);
                    }
                }
            }
        }

        // Deduplicate.
        for list in &mut callees {
            list.sort_unstable();
            list.dedup();
        }
        for list in &mut callers {
            list.sort_unstable();
            list.dedup();
        }

        Self {
            version: CACHE_VERSION,
            names,
            files,
            kinds,
            lines: lines_vec,
            signatures,
            name_index,
            callees,
            callers,
            is_test,
        }
    }

    /// Build the call graph using language-resolved calls (MIR, Python AST, etc.)
    /// when available, falling back to tree-sitter heuristics for unmatched calls.
    pub fn build_with_resolved_calls(chunks: &[CodeChunk], mir_calls: &MirCallMap) -> Self {
        let len = chunks.len();

        // Multi-map: name → all chunk indices with that name (for MIR matching).
        let mut exact: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        // Single-value maps for tree-sitter fallback (non-Rust files).
        let mut exact_single: BTreeMap<String, u32> = BTreeMap::new();
        let mut short: BTreeMap<String, u32> = BTreeMap::new();

        let mut names = Vec::with_capacity(len);
        let mut files = Vec::with_capacity(len);
        let mut kinds = Vec::with_capacity(len);
        let mut lines_vec = Vec::with_capacity(len);
        let mut signatures = Vec::with_capacity(len);
        let mut is_test = Vec::with_capacity(len);
        let mut name_index = Vec::with_capacity(len);

        for (i, c) in chunks.iter().enumerate() {
            let idx = i as u32;
            let lower = c.name.to_lowercase();

            exact.entry(lower.clone()).or_default().push(idx);
            exact_single.insert(lower.clone(), idx);
            // Also index with generic params stripped: `hnswgraph<d>::insert` → `hnswgraph::insert`.
            let stripped = strip_chunk_generics(&lower);
            if stripped != lower {
                exact.entry(stripped.clone()).or_default().push(idx);
                exact_single.insert(stripped, idx);
            }
            if let Some(s) = c.name.rsplit("::").next() {
                short.entry(s.to_lowercase()).or_insert(idx);
            }

            name_index.push((lower, idx));
            names.push(c.name.clone());
            files.push(c.file.clone());
            kinds.push(c.kind.clone());
            lines_vec.push(c.lines);
            signatures.push(c.signature.clone());
            is_test.push(is_test_chunk(c));
        }

        name_index.sort_by(|a, b| a.0.cmp(&b.0));

        let mut callees: Vec<Vec<u32>> = vec![Vec::new(); len];
        let mut callers: Vec<Vec<u32>> = vec![Vec::new(); len];

        // Build chunk-index → MIR callee list lookup.
        // Each MIR function maps to exactly one chunk (by name + module/file match).
        let mir_for_chunk: Vec<Option<Vec<String>>> = {
            let mut per_chunk: Vec<Option<Vec<String>>> = vec![None; len];
            for (mir_name, callees_list) in mir_calls {
                let mir_lower = mir_name.to_lowercase();
                let mir_module = mir_lower.rsplit_once("::").map(|(prefix, _)| prefix);

                // Find the best matching chunk for this MIR function.
                let matched_idx = find_best_chunk_match(
                    &mir_lower, mir_module, &exact, chunks,
                );
                if let Some(idx) = matched_idx {
                    per_chunk[idx as usize]
                        .get_or_insert_with(Vec::new)
                        .extend(callees_list.iter().cloned());
                }
            }
            per_chunk
        };

        for (src, c) in chunks.iter().enumerate() {
            // Try MIR-resolved calls first.
            // MIR callees are fully qualified — match by exact name or qualified suffix.
            let mut mir_used = false;
            if let Some(mir_callees) = mir_for_chunk[src].as_ref() {
                mir_used = true;
                for callee_name in mir_callees {
                    let callee_lower = callee_name.to_lowercase();
                    let callee_module = callee_lower.rsplit_once("::").map(|(p, _)| p);
                    // Resolve callee to a chunk index.
                    if let Some(tgt) = find_best_chunk_match(
                        &callee_lower, callee_module, &exact, chunks,
                    ) {
                        let tgt_usize = tgt as usize;
                        if tgt_usize != src {
                            callees[src].push(tgt);
                            callers[tgt_usize].push(src as u32);
                        }
                    }
                }
            }

            // Fallback to tree-sitter heuristics if resolved calls had no data for this chunk.
            // Skip fallback for Rust files — MIR is authoritative for those.
            let is_rust = c.file.ends_with(".rs");
            if !mir_used && !is_rust {
                let imports = build_import_map(&c.imports);
                let self_type = owning_type(&c.name);
                let call_types = extract_call_types(&c.calls);
                for call in &c.calls {
                    if let Some(tgt) = resolve_with_imports(
                        call, &exact_single, &short, &imports,
                        self_type.as_deref(), &c.types, &call_types,
                    ) {
                        let tgt_usize = tgt as usize;
                        if tgt_usize != src {
                            callees[src].push(tgt);
                            callers[tgt_usize].push(src as u32);
                        }
                    }
                }
            }
        }

        for list in &mut callees {
            list.sort_unstable();
            list.dedup();
        }
        for list in &mut callers {
            list.sort_unstable();
            list.dedup();
        }

        Self {
            version: CACHE_VERSION,
            names,
            files,
            kinds,
            lines: lines_vec,
            signatures,
            name_index,
            callees,
            callers,
            is_test,
        }
    }

    /// Resolve a symbol name to matching chunk indices.
    ///
    /// Returns all indices whose lowercase name equals `name` (case-insensitive)
    /// or ends with `::<name>`.
    pub fn resolve(&self, name: &str) -> Vec<u32> {
        let lower = name.to_lowercase();
        let mut results = Vec::new();

        // Binary search for exact match range.
        let start = self.name_index.partition_point(|(n, _)| n.as_str() < lower.as_str());
        for entry in &self.name_index[start..] {
            if entry.0 == lower {
                results.push(entry.1);
            } else {
                break;
            }
        }

        // Also match entries ending with `::<name>`.
        if results.is_empty() {
            let suffix = format!("::{lower}");
            for (n, idx) in &self.name_index {
                if n.ends_with(&suffix) {
                    results.push(*idx);
                }
            }
        }

        results
    }

    /// Save the graph to `<db>/cache/graph.bin`.
    pub fn save(&self, db: &Path) -> Result<()> {
        let path = graph_cache_path(db);
        let _ = fs::create_dir_all(path.parent().unwrap_or(Path::new(".")));
        let config = bincode::config::standard();
        let bytes = bincode::encode_to_vec(self, config)
            .context("failed to encode call graph")?;
        fs::write(&path, bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    /// Load the graph from `<db>/cache/graph.bin`, returning `None` on
    /// cache miss or version mismatch.
    pub fn load(db: &Path) -> Option<Self> {
        let path = graph_cache_path(db);
        let db_mtime = fs::metadata(db).and_then(|m| m.modified()).ok()?;
        let cache_meta = fs::metadata(&path).ok()?;
        let cache_mtime = cache_meta.modified().ok()?;

        if cache_mtime < db_mtime {
            return None;
        }

        let bytes = fs::read(&path).ok()?;
        let config = bincode::config::standard();
        let (graph, _): (Self, _) = bincode::decode_from_slice(&bytes, config).ok()?;

        if graph.version != CACHE_VERSION {
            return None;
        }

        Some(graph)
    }

    /// Total number of chunks (nodes) in the graph.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn graph_cache_path(db: &Path) -> std::path::PathBuf {
    db.join("cache").join("graph.bin")
}

/// Resolve a call using import-qualified lookup, then exact, then short fallback.
///
/// Resolution order:
/// 1. Exact match on full lowercase name
/// 2. `self.method` → `OwningType::method` (from chunk name)
/// 3. Import-qualified: `Foo::bar` → look up `foo` in imports
/// 4. Direct import match for bare names
/// 5. Type-reference heuristic: `var.method` where source chunk references
///    a type that has `::method` → resolve to that type's method
/// 6. Short name fallback for `::` calls; receiver-type gated for `.` calls
fn resolve_with_imports(
    call: &str,
    exact: &BTreeMap<String, u32>,
    short: &BTreeMap<String, u32>,
    imports: &BTreeMap<String, String>,
    self_type: Option<&str>,
    source_types: &[String],
    call_types: &[String],
) -> Option<u32> {
    let lower = call.to_lowercase();

    // 1. Exact match.
    if let Some(&idx) = exact.get(&lower) {
        return Some(idx);
    }

    // 2. self.method → OwningType::method.
    if let Some(method) = lower.strip_prefix("self.") {
        // Strip chained field access: self.foo.bar.method → method (last segment)
        let leaf_method = method.rsplit_once('.').map_or(method, |p| p.1);
        if let Some(owner) = self_type {
            let qualified = format!("{owner}::{leaf_method}");
            if let Some(&idx) = exact.get(&qualified) {
                return Some(idx);
            }
        }
    }

    // 3. Import-qualified: "Foo::bar" → look up "foo" in imports → "mod::foo::bar".
    if let Some((prefix, suffix)) = lower.split_once("::") {
        let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
        if let Some(qualified) = imports.get(leaf) {
            let qualified_call = format!("{qualified}::{suffix}");
            if let Some(&idx) = exact.get(&qualified_call) {
                return Some(idx);
            }
        }
    }

    // 4. Direct import match: bare name like "HashMap".
    if let Some(qualified) = imports.get(&lower) {
        if let Some(&idx) = exact.get(qualified) {
            return Some(idx);
        }
    }

    // 5. Type-reference heuristic for `.method()` calls.
    //    Check source chunk's type refs AND types inferred from `::` calls in the same chunk.
    if let Some((_, method)) = lower.rsplit_once('.') {
        let leaf_method = method.rsplit_once('.').map_or(method, |p| p.1);
        for ty in source_types.iter().chain(call_types.iter()) {
            let ty_lower = ty.to_lowercase();
            let candidate = format!("{ty_lower}::{leaf_method}");
            if let Some(&idx) = exact.get(&candidate) {
                return Some(idx);
            }
        }
    }

    // 6. Short name fallback.
    //    `Foo::bar` → try short match on "bar" (qualified = likely a type).
    //    `receiver.method` → only if receiver is a known type (in imports or exact map).
    //    Bare names → skip (too ambiguous).
    if let Some((_, suffix)) = lower.rsplit_once("::") {
        return short.get(suffix).copied();
    }
    if let Some((receiver, method)) = lower.rsplit_once('.') {
        let receiver_leaf = receiver.rsplit_once('.').map_or(receiver, |p| p.1);
        if imports.contains_key(receiver_leaf) || exact.contains_key(receiver_leaf) {
            return short.get(method).copied();
        }
        return None;
    }
    None
}

/// Extract type names from `::` call prefixes.
///
/// E.g. `["DeltaNeighbors::from_ids", "Vec::new"]` → `["deltaneighbors", "vec"]`.
fn extract_call_types(calls: &[String]) -> Vec<String> {
    let mut types = Vec::new();
    for call in calls {
        if let Some((prefix, _)) = call.split_once("::") {
            // Take the last segment: "std::collections::HashMap::new" → "HashMap"
            let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
            let lower = leaf.to_lowercase();
            if !types.contains(&lower) {
                types.push(lower);
            }
        }
    }
    types
}

/// Extract the owning type from a chunk name like "Foo::bar" → "foo".
fn owning_type(name: &str) -> Option<String> {
    let (prefix, _) = name.rsplit_once("::")?;
    // Take the last segment for nested paths: "mod::Foo::bar" → "Foo"
    let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
    Some(leaf.to_lowercase())
}

/// Parse `use` declarations into a short_name → qualified_path map.
fn build_import_map(imports: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for imp in imports {
        let s = imp.trim().trim_start_matches("use ").trim_end_matches(';').trim();
        let s = s.trim_start_matches("crate::");
        if let Some(brace) = s.find('{') {
            let prefix = s[..brace].trim_end_matches("::");
            if let Some(end) = s.rfind('}') {
                for part in s[brace + 1..end].split(',') {
                    let name = part.trim().split_whitespace().next().unwrap_or("");
                    if !name.is_empty() && name != "self" {
                        map.insert(name.to_lowercase(), format!("{}::{}", prefix, name).to_lowercase());
                    }
                }
            }
        } else if let Some(last) = s.rsplit("::").next() {
            let short = last.trim().split_whitespace().next().unwrap_or("");
            if !short.is_empty() {
                map.insert(short.to_lowercase(), s.to_lowercase());
            }
        }
    }
    map
}

/// Determine if a chunk is test code based on file path and name.
/// Find the best chunk match for a MIR function name.
///
/// Tries exact match first, then suffix match with module-to-file disambiguation.
/// Returns `None` if no match found.
fn find_best_chunk_match(
    mir_lower: &str,
    mir_module: Option<&str>,
    exact: &BTreeMap<String, Vec<u32>>,
    chunks: &[CodeChunk],
) -> Option<u32> {
    // Exact match: MIR name == chunk name.
    if let Some(indices) = exact.get(mir_lower) {
        if indices.len() == 1 {
            return Some(indices[0]);
        }
        // Multiple chunks with same name — disambiguate by module/file.
        // Prefer .rs files over non-Rust files.
        if let Some(mod_prefix) = mir_module {
            let last_mod = mod_prefix.rsplit("::").next().unwrap_or(mod_prefix);
            let last_mod_no_us = last_mod.replace('_', "");
            let mut fallback: Option<u32> = None;
            for &idx in indices {
                let file_lower = chunks[idx as usize].file.to_lowercase();
                let file_no_us = file_lower.replace('_', "");
                if file_lower.contains(last_mod) || file_no_us.contains(&last_mod_no_us) {
                    if file_lower.ends_with(".rs") {
                        return Some(idx);
                    }
                    if fallback.is_none() {
                        fallback = Some(idx);
                    }
                }
            }
            if let Some(fb) = fallback {
                return Some(fb);
            }
        }
        return Some(indices[0]);
    }

    // Suffix match: MIR "graph::CallGraph::build" → chunk "CallGraph::build".
    // Only match qualified chunk names (containing "::") to avoid bare method ambiguity.
    // E.g., callee "HashMap::insert" must NOT match bare chunk "insert".
    let mut best: Option<u32> = None;
    for (chunk_name, indices) in exact {
        if !chunk_name.contains("::") {
            // Bare names require module-to-file disambiguation.
            if let Some(mod_prefix) = mir_module {
                let last_mod = mod_prefix.rsplit("::").next().unwrap_or(mod_prefix);
                let last_mod_no_us = last_mod.replace('_', "");
                if mir_lower.ends_with(chunk_name.as_str()) {
                    let prefix_len = mir_lower.len() - chunk_name.len();
                    if prefix_len == 0 || mir_lower.as_bytes()[prefix_len - 1] == b':' {
                        // Prefer .rs files (Rust sources) over non-Rust files.
                        let mut fallback: Option<u32> = None;
                        for &idx in indices {
                            let file_lower = chunks[idx as usize].file.to_lowercase();
                            let file_no_us = file_lower.replace('_', "");
                            if file_lower.contains(last_mod) || file_no_us.contains(&last_mod_no_us) {
                                if file_lower.ends_with(".rs") {
                                    return Some(idx);
                                }
                                if fallback.is_none() {
                                    fallback = Some(idx);
                                }
                            }
                        }
                        if let Some(fb) = fallback {
                            return Some(fb);
                        }
                    }
                }
            }
            continue;
        }
        if mir_lower.ends_with(chunk_name.as_str()) {
            let prefix_len = mir_lower.len() - chunk_name.len();
            if prefix_len == 0 || mir_lower.as_bytes()[prefix_len - 1] == b':' {
                if let Some(mod_prefix) = mir_module {
                    let last_mod = mod_prefix.rsplit("::").next().unwrap_or(mod_prefix);
                    let last_mod_no_us = last_mod.replace('_', "");
                    for &idx in indices {
                        let file_lower = chunks[idx as usize].file.to_lowercase();
                        let file_no_us = file_lower.replace('_', "");
                        if file_lower.contains(last_mod) || file_no_us.contains(&last_mod_no_us) {
                            return Some(idx);
                        }
                    }
                }
                if best.is_none() {
                    best = Some(indices[0]);
                }
            }
        }
    }
    best
}

/// Strip generic params from chunk names: `hnswgraph<d>::insert` → `hnswgraph::insert`.
fn strip_chunk_generics(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    let mut depth = 0u32;
    for c in name.chars() {
        match c {
            '<' => depth += 1,
            '>' => { depth = depth.saturating_sub(1); }
            _ if depth == 0 => result.push(c),
            _ => {}
        }
    }
    result
}

fn is_test_chunk(c: &CodeChunk) -> bool {
    c.file.contains("/tests/")
        || c.file.contains("\\tests\\")
        || c.file.ends_with("_test.rs")
        || c.name.starts_with("test_")
        || c.file.contains("/test_")
}

