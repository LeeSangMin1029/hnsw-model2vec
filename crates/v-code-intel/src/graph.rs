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

use crate::lsp::CallMap;
use crate::parse::CodeChunk;

/// Cache format version — bump when struct layout changes.
const CACHE_VERSION: u8 = 3;

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
    /// trait chunk index -> implementing chunk indices.
    /// Only populated for chunks whose kind is "trait".
    pub trait_impls: Vec<Vec<u32>>,
    /// caller chunk → list of (callee_idx, call_line).
    /// call_line is 1-based source line where the call occurs.
    pub call_sites: Vec<Vec<(u32, u32)>>,
}

/// Shared chunk metadata collected during graph construction.
struct ChunkMeta {
    names: Vec<String>,
    files: Vec<String>,
    kinds: Vec<String>,
    lines: Vec<Option<(usize, usize)>>,
    signatures: Vec<Option<String>>,
    is_test: Vec<bool>,
    name_index: Vec<(String, u32)>,
    /// Single-value name → index map (tree-sitter resolution).
    exact: BTreeMap<String, u32>,
    /// Short (last `::` segment) → index map.
    short: BTreeMap<String, u32>,
}

impl ChunkMeta {
    fn collect(chunks: &[CodeChunk]) -> Self {
        let len = chunks.len();
        let mut exact = BTreeMap::new();
        let mut short = BTreeMap::new();
        let mut names = Vec::with_capacity(len);
        let mut files = Vec::with_capacity(len);
        let mut kinds = Vec::with_capacity(len);
        let mut lines = Vec::with_capacity(len);
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
            lines.push(c.lines);
            signatures.push(c.signature.clone());
            is_test.push(is_test_chunk(c));
        }

        name_index.sort_by(|a, b| a.0.cmp(&b.0));

        Self { names, files, kinds, lines, signatures, is_test, name_index, exact, short }
    }
}

/// Mutable adjacency state accumulated during edge resolution.
struct AdjState {
    callees: Vec<Vec<u32>>,
    callers: Vec<Vec<u32>>,
    call_sites: Vec<Vec<(u32, u32)>>,
}

impl AdjState {
    fn new(len: usize) -> Self {
        Self {
            callees: vec![Vec::new(); len],
            callers: vec![Vec::new(); len],
            call_sites: vec![Vec::new(); len],
        }
    }

    /// Add a directed edge from `src` to `tgt` with the given call line.
    fn add_edge(&mut self, src: usize, tgt: u32, call_line: u32) {
        let tgt_usize = tgt as usize;
        if tgt_usize != src {
            self.callees[src].push(tgt);
            self.callers[tgt_usize].push(src as u32);
            self.call_sites[src].push((tgt, call_line));
        }
    }

    /// Deduplicate all adjacency lists and call sites.
    fn dedup(&mut self) {
        for list in &mut self.callees {
            list.sort_unstable();
            list.dedup();
        }
        for list in &mut self.callers {
            list.sort_unstable();
            list.dedup();
        }
        for sites in &mut self.call_sites {
            sites.sort_by_key(|&(tgt, _)| tgt);
            sites.dedup_by_key(|e| e.0);
        }
    }
}

impl CallGraph {
    /// Build the call graph from parsed code chunks.
    ///
    /// Resolution strategy: exact match on lowercase name first, then
    /// short name (last `::` segment) fallback.
    pub fn build(chunks: &[CodeChunk]) -> Self {
        let meta = ChunkMeta::collect(chunks);
        let mut adj = AdjState::new(chunks.len());

        for (src, c) in chunks.iter().enumerate() {
            let imports = build_import_map(&c.imports);
            let self_type = owning_type(&c.name);
            let call_types = extract_call_types(&c.calls);
            for (call_idx, call) in c.calls.iter().enumerate() {
                if let Some(tgt) = resolve_with_imports(call, &meta.exact, &meta.short, &imports, self_type.as_deref(), &c.types, &call_types) {
                    let call_line = c.call_lines.get(call_idx).copied().unwrap_or(0);
                    adj.add_edge(src, tgt, call_line);
                }
            }
            resolve_type_ref_edges(src, &c.types, &meta.exact, &meta.short, &imports, &mut adj.callees, &mut adj.callers);
        }

        adj.dedup();
        Self::from_parts(meta, adj)
    }

    /// Build the call graph using language-resolved calls (LSP definition, etc.)
    /// when available, falling back to tree-sitter heuristics for unmatched calls.
    pub fn build_with_resolved_calls(chunks: &[CodeChunk], resolved_calls: &CallMap) -> Self {
        let meta = ChunkMeta::collect(chunks);
        let len = chunks.len();

        // Multi-map: name → all chunk indices (for LSP resolved call matching).
        let mut exact_multi: BTreeMap<String, Vec<u32>> = BTreeMap::new();
        for (i, c) in chunks.iter().enumerate() {
            let idx = i as u32;
            let lower = c.name.to_lowercase();
            exact_multi.entry(lower.clone()).or_default().push(idx);
            let stripped = strip_chunk_generics(&lower);
            if stripped != lower {
                exact_multi.entry(stripped.clone()).or_default().push(idx);
                // Also add stripped to single-value map (meta.exact doesn't have it).
            }
        }

        let mut adj = AdjState::new(len);

        // Build chunk-index → resolved callee list lookup.
        let resolved_for_chunk: Vec<Option<Vec<String>>> = {
            let mut per_chunk: Vec<Option<Vec<String>>> = vec![None; len];
            for (fn_name, callees_list) in resolved_calls {
                let fn_lower = fn_name.to_lowercase();
                let fn_module = fn_lower.rsplit_once("::").map(|(prefix, _)| prefix);
                if let Some(idx) = find_best_chunk_match(&fn_lower, fn_module, &exact_multi, chunks) {
                    per_chunk[idx as usize]
                        .get_or_insert_with(Vec::new)
                        .extend(callees_list.iter().cloned());
                }
            }
            per_chunk
        };

        // Also build exact_single with stripped generics for tree-sitter fallback.
        let mut exact_single = meta.exact.clone();
        for (i, c) in chunks.iter().enumerate() {
            let lower = c.name.to_lowercase();
            let stripped = strip_chunk_generics(&lower);
            if stripped != lower {
                exact_single.insert(stripped, i as u32);
            }
        }

        for (src, c) in chunks.iter().enumerate() {
            let mut resolved_names: std::collections::HashSet<String> = std::collections::HashSet::new();

            // Build a short-name → call_line lookup from tree-sitter data.
            let ts_call_lines: BTreeMap<String, u32> = c.calls.iter().enumerate().map(|(i, name)| {
                let short = name.rsplit("::").next().unwrap_or(name).to_lowercase();
                let line = c.call_lines.get(i).copied().unwrap_or(0);
                (short, line)
            }).collect();

            // Try LSP-resolved calls first.
            if let Some(resolved_callees) = resolved_for_chunk[src].as_ref() {
                for callee_name in resolved_callees {
                    let callee_lower = callee_name.to_lowercase();
                    let callee_module = callee_lower.rsplit_once("::").map(|(p, _)| p);
                    if let Some(tgt) = find_best_chunk_match(&callee_lower, callee_module, &exact_multi, chunks) {
                        let short_name = callee_lower.rsplit("::").next().unwrap_or(&callee_lower);
                        let call_line = ts_call_lines.get(short_name).copied().unwrap_or(0);
                        adj.add_edge(src, tgt, call_line);
                    }
                    let short_name = callee_lower.rsplit("::").next().unwrap_or(&callee_lower);
                    resolved_names.insert(short_name.to_owned());
                }
            }

            // Tree-sitter fallback for calls NOT covered by LSP resolution.
            let imports = build_import_map(&c.imports);
            let self_type = owning_type(&c.name);
            let call_types = extract_call_types(&c.calls);
            for (call_idx, call) in c.calls.iter().enumerate() {
                let call_short = call.to_lowercase();
                let call_leaf = call_short.rsplit("::").next().unwrap_or(&call_short);
                if resolved_names.contains(call_leaf) {
                    continue;
                }
                if let Some(tgt) = resolve_with_imports(
                    call, &exact_single, &meta.short, &imports,
                    self_type.as_deref(), &c.types, &call_types,
                ) {
                    let call_line = c.call_lines.get(call_idx).copied().unwrap_or(0);
                    adj.add_edge(src, tgt, call_line);
                }
            }
            resolve_type_ref_edges(src, &c.types, &exact_single, &meta.short, &imports, &mut adj.callees, &mut adj.callers);
        }

        adj.dedup();
        Self::from_parts(meta, adj)
    }

    /// Assemble a `CallGraph` from pre-built metadata and adjacency state.
    fn from_parts(meta: ChunkMeta, adj: AdjState) -> Self {
        let trait_impls = build_trait_impls(&meta.names, &meta.kinds, &meta.exact, &meta.short);
        Self {
            version: CACHE_VERSION,
            names: meta.names,
            files: meta.files,
            kinds: meta.kinds,
            lines: meta.lines,
            signatures: meta.signatures,
            name_index: meta.name_index,
            callees: adj.callees,
            callers: adj.callers,
            is_test: meta.is_test,
            trait_impls,
            call_sites: adj.call_sites,
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

    /// Look up the source line where `caller_idx` calls `callee_idx`.
    /// Returns 0 if no call site info is available.
    pub fn call_site_line(&self, caller_idx: u32, callee_idx: u32) -> u32 {
        let sites = &self.call_sites[caller_idx as usize];
        for &(tgt, line) in sites {
            if tgt == callee_idx {
                return line;
            }
        }
        0
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
    if let Some(qualified) = imports.get(&lower)
        && let Some(&idx) = exact.get(qualified) {
            return Some(idx);
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

/// Resolve type references (struct/enum/trait usage) into caller→callee edges.
///
/// For each type in `types`, looks up the corresponding chunk (struct, enum, trait)
/// and adds an edge from `src` to that chunk.
fn resolve_type_ref_edges(
    src: usize,
    types: &[String],
    exact: &BTreeMap<String, u32>,
    short: &BTreeMap<String, u32>,
    imports: &BTreeMap<String, String>,
    callees: &mut [Vec<u32>],
    callers: &mut [Vec<u32>],
) {
    for ty in types {
        let lower = ty.to_lowercase();

        // 1. Import-qualified lookup.
        let tgt = if let Some(qualified) = imports.get(&lower) {
            exact.get(qualified).copied()
        } else {
            None
        };

        // 2. Exact name match.
        let tgt = tgt.or_else(|| exact.get(&lower).copied());

        // 3. Short name fallback.
        let tgt = tgt.or_else(|| short.get(&lower).copied());

        if let Some(tgt) = tgt {
            let tgt_usize = tgt as usize;
            if tgt_usize != src {
                callees[src].push(tgt);
                callers[tgt_usize].push(src as u32);
            }
        }
    }
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
                    let name = part.split_whitespace().next().unwrap_or("");
                    if !name.is_empty() && name != "self" {
                        map.insert(name.to_lowercase(), format!("{}::{}", prefix, name).to_lowercase());
                    }
                }
            }
        } else if let Some(last) = s.rsplit("::").next() {
            let short = last.split_whitespace().next().unwrap_or("");
            if !short.is_empty() {
                map.insert(short.to_lowercase(), s.to_lowercase());
            }
        }
    }
    map
}

/// Pick the best index from `indices` by matching module name against file path.
/// Prefers `.rs` files; returns first file-match or `None`.
fn disambiguate_by_module(
    indices: &[u32],
    mod_prefix: &str,
    chunks: &[CodeChunk],
) -> Option<u32> {
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
    fallback
}

/// Find the best chunk match for a resolved function name.
///
/// Tries exact match first, then suffix match with module-to-file disambiguation.
fn find_best_chunk_match(
    fn_lower: &str,
    fn_module: Option<&str>,
    exact: &BTreeMap<String, Vec<u32>>,
    chunks: &[CodeChunk],
) -> Option<u32> {
    // Exact match: resolved name == chunk name.
    if let Some(indices) = exact.get(fn_lower) {
        if indices.len() == 1 {
            return Some(indices[0]);
        }
        if let Some(mod_prefix) = fn_module
            && let Some(idx) = disambiguate_by_module(indices, mod_prefix, chunks) {
            return Some(idx);
        }
        return Some(indices[0]);
    }

    // Suffix match: "graph::CallGraph::build" → chunk "CallGraph::build".
    let mut best: Option<u32> = None;
    for (chunk_name, indices) in exact {
        if !fn_lower.ends_with(chunk_name.as_str()) {
            continue;
        }
        let prefix_len = fn_lower.len() - chunk_name.len();
        if prefix_len > 0 && fn_lower.as_bytes()[prefix_len - 1] != b':' {
            continue;
        }

        if let Some(mod_prefix) = fn_module {
            if let Some(idx) = disambiguate_by_module(indices, mod_prefix, chunks) {
                return Some(idx);
            }
        }
        // Bare names without "::" skip unless module-disambiguated above.
        if !chunk_name.contains("::") {
            continue;
        }
        if best.is_none() {
            best = Some(indices[0]);
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

/// Check if a file path looks like a test file.
pub fn is_test_path(path: &str) -> bool {
    path.contains("/tests/")
        || path.contains("\\tests\\")
        || path.contains("/test/")
        || path.contains("\\test\\")
        || path.ends_with("_test.rs")
        || path.ends_with("_test.go")
        || path.contains("/test_")
}

pub fn is_test_chunk(c: &CodeChunk) -> bool {
    is_test_path(&c.file) || c.name.starts_with("test_")
}

/// Build trait → impl mapping from impl chunk names.
///
/// Impl chunks have names like `"VectorIndex for HnswGraph<D>"`.
/// We split on `" for "` to find the trait name, then resolve it to a chunk index.
fn build_trait_impls(
    names: &[String],
    kinds: &[String],
    exact: &BTreeMap<String, u32>,
    short: &BTreeMap<String, u32>,
) -> Vec<Vec<u32>> {
    let len = names.len();
    let mut trait_impls: Vec<Vec<u32>> = vec![Vec::new(); len];

    for (i, (name, kind)) in names.iter().zip(kinds.iter()).enumerate() {
        if kind != "impl" {
            continue;
        }
        // Look for "Trait for Type" pattern.
        let lower = name.to_lowercase();
        if let Some(pos) = lower.find(" for ") {
            let trait_name = &lower[..pos];
            // Resolve trait name to chunk index.
            let trait_idx = exact.get(trait_name).copied()
                .or_else(|| short.get(trait_name).copied());
            if let Some(tidx) = trait_idx {
                trait_impls[tidx as usize].push(i as u32);
            }
        }
    }

    // Deduplicate.
    for list in &mut trait_impls {
        list.sort_unstable();
        list.dedup();
    }

    trait_impls
}

