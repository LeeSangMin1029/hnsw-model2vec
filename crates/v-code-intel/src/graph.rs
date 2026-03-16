//! Call graph adjacency list built from `CodeChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.
//!
//! ## HOW TO EXTEND
//! - Add new traversal methods (e.g., shortest path) as `impl CallGraph` methods.
//! - Add new fields to `CallGraph` and update `build()` + bump cache version.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::lsp::CallMap;
use crate::parse::CodeChunk;

/// Cache format version — bump when struct layout changes.
const CACHE_VERSION: u8 = 6;

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
    /// chunk index → string literal arguments [(callee, value, line, arg_pos)].
    pub string_args: Vec<Vec<(String, String, u32, u8)>>,
    /// Sorted (lowercase_value, [(chunk_idx, line)]) for binary search lookup.
    pub string_index: Vec<(String, Vec<(u32, u32)>)>,
    /// chunk index → parameter-to-callee flows [(param_name, param_pos, callee, callee_arg, line)].
    #[expect(clippy::type_complexity, reason = "tuple layout mirrors string_args")]
    pub param_flows: Vec<Vec<(String, u8, String, u8, u32)>>,
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
            // Also insert generic-stripped alias: "foo<t>::bar" → "foo::bar"
            let stripped = strip_generics_from_key(&lower);
            if stripped != lower {
                exact.entry(stripped).or_insert(idx);
            }
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
        let owner_types = collect_owner_types(chunks, &meta);
        let owner_field_types = collect_owner_field_types(chunks);
        let return_type_map = build_return_type_map(chunks);

        for (src, c) in chunks.iter().enumerate() {
            Self::resolve_chunk_edges(src, c, &meta.exact, &meta.short, &owner_types, &owner_field_types, &return_type_map, &mut adj, &HashSet::new());
        }

        adj.dedup();
        Self::from_parts(meta, adj, chunks)
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
            let stripped = strip_generics_from_key(&lower);
            if stripped != lower {
                exact_multi.entry(stripped.clone()).or_default().push(idx);
                // Also add stripped to single-value map (meta.exact doesn't have it).
            }
        }

        let mut adj = AdjState::new(len);
        let owner_types = collect_owner_types(chunks, &meta);
        let owner_field_types = collect_owner_field_types(chunks);
        let return_type_map = build_return_type_map(chunks);

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
            let stripped = strip_generics_from_key(&lower);
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
            Self::resolve_chunk_edges(src, c, &exact_single, &meta.short, &owner_types, &owner_field_types, &return_type_map, &mut adj, &resolved_names);
        }

        adj.dedup();
        Self::from_parts(meta, adj, chunks)
    }

    /// Resolve tree-sitter edges for a single chunk.
    ///
    /// Shared by `build()` and `build_with_resolved_calls()` (fallback path).
    /// `skip_calls` contains short names already resolved by LSP — those are skipped.
    fn resolve_chunk_edges(
        src: usize,
        chunk: &CodeChunk,
        exact: &BTreeMap<String, u32>,
        short: &BTreeMap<String, u32>,
        owner_types: &BTreeMap<String, Vec<String>>,
        owner_field_types: &BTreeMap<String, BTreeMap<String, String>>,
        return_type_map: &BTreeMap<String, String>,
        adj: &mut AdjState,
        skip_calls: &HashSet<String>,
    ) {
        let imports = build_import_map(&chunk.imports);
        let self_type = owning_type(&chunk.name);
        let call_types = extract_call_types(&chunk.calls);
        let mut enriched_types = chunk.types.clone();
        if let Some(ref owner) = self_type {
            if let Some(extra) = owner_types.get(owner.as_str()) {
                for t in extra {
                    if !enriched_types.contains(t) {
                        enriched_types.push(t.clone());
                    }
                }
            }
        }
        // Build receiver name → type lookup from param/local/field types.
        let mut receiver_types = build_receiver_type_map(chunk);
        // Enrich with struct field types for self.field resolution.
        if let Some(ref owner) = self_type {
            if let Some(fields) = owner_field_types.get(owner.as_str()) {
                for (field_name, field_type) in fields {
                    receiver_types.entry(field_name.clone()).or_insert_with(|| field_type.clone());
                }
            }
        }
        // Infer local variable types from return_type_map.
        // `let x = Foo::new()` → if return_type_map["foo::new"] = "foo", then x: foo.
        infer_local_types_from_calls(&chunk.calls, return_type_map, &mut receiver_types);
        for (call_idx, call) in chunk.calls.iter().enumerate() {
            if !skip_calls.is_empty() {
                let call_short = call.to_lowercase();
                let call_leaf = call_short.rsplit("::").next().unwrap_or(&call_short);
                if skip_calls.contains(call_leaf) {
                    continue;
                }
            }
            if let Some(tgt) = resolve_with_imports(
                call, exact, short, &imports,
                self_type.as_deref(), &enriched_types, &call_types,
                &receiver_types,
            ) {
                let call_line = chunk.call_lines.get(call_idx).copied().unwrap_or(0);
                adj.add_edge(src, tgt, call_line);
            }
        }
        resolve_type_ref_edges(src, &chunk.types, exact, short, &imports, &mut adj.callees, &mut adj.callers);
    }

    /// Assemble a `CallGraph` from pre-built metadata and adjacency state.
    fn from_parts(meta: ChunkMeta, adj: AdjState, chunks: &[CodeChunk]) -> Self {
        let trait_impls = build_trait_impls(&meta.names, &meta.kinds, &meta.exact, &meta.short);
        let string_args = collect_string_args(chunks);
        let string_index = build_string_index(&string_args);
        #[expect(clippy::type_complexity, reason = "tuple layout mirrors string_args")]
        let param_flows: Vec<Vec<(String, u8, String, u8, u32)>> =
            chunks.iter().map(|c| c.param_flows.clone()).collect();
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
            string_args,
            string_index,
            param_flows,
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

    /// Find chunks that use a specific string literal value (exact match, case-insensitive).
    pub fn find_string(&self, value: &str) -> Vec<(u32, u32)> {
        let lower = value.to_lowercase();
        match self.string_index.binary_search_by_key(&&*lower, |(k, _)| k.as_str()) {
            Ok(i) => self.string_index[i].1.clone(),
            Err(_) => Vec::new(),
        }
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
/// 6. Short name fallback for `::` calls; type-gated for `.` calls (no hardcoded blocklist)
fn resolve_with_imports(
    call: &str,
    exact: &BTreeMap<String, u32>,
    short: &BTreeMap<String, u32>,
    imports: &BTreeMap<String, String>,
    self_type: Option<&str>,
    source_types: &[String],
    call_types: &[String],
    receiver_types: &BTreeMap<String, String>,
) -> Option<u32> {
    let lower = call.to_lowercase();

    // 1. Exact match.
    if let Some(&idx) = exact.get(&lower) {
        return Some(idx);
    }

    // 2. self.method → OwningType::method.
    //    self.field.method → try field name as type hint against source_types.
    if let Some(method) = lower.strip_prefix("self.") {
        let leaf_method = method.rsplit_once('.').map_or(method, |p| p.1);
        // 2a. Try owning type first.
        if let Some(owner) = self_type {
            let qualified = format!("{owner}::{leaf_method}");
            if let Some(&idx) = exact.get(&qualified) {
                return Some(idx);
            }
        }
        // 2b. self.field.method → look up field type from receiver_types (populated
        //     from owner_field_types), then resolve FieldType::method.
        //     e.g. self.tokenizer.encode → receiver_types["tokenizer"] = "tokenizer" → Tokenizer::encode
        if let Some((field_path, _)) = method.rsplit_once('.') {
            let field_leaf = field_path.rsplit_once('.').map_or(field_path, |p| p.1);
            // Direct field type lookup (from struct definition)
            if let Some(field_type) = receiver_types.get(field_leaf) {
                let candidate = format!("{field_type}::{leaf_method}");
                if let Some(&idx) = exact.get(&candidate) {
                    return Some(idx);
                }
                // Field type exists in project → allow short fallback
                if short.contains_key(field_type) || exact.contains_key(field_type) {
                    if let Some(&idx) = short.get(leaf_method) {
                        return Some(idx);
                    }
                }
                // Field type is external → skip to avoid false positive
                return None;
            }
            // Fallback: try type_refs heuristic
            for ty in source_types.iter().chain(call_types.iter()) {
                let ty_lower = ty.to_lowercase();
                let candidate = format!("{ty_lower}::{leaf_method}");
                if let Some(&idx) = exact.get(&candidate) {
                    return Some(idx);
                }
            }
            // Try field_leaf directly as a type name via imports
            if let Some(qualified) = imports.get(field_leaf) {
                let candidate = format!("{qualified}::{leaf_method}");
                if let Some(&idx) = exact.get(&candidate) {
                    return Some(idx);
                }
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
    //    Prefer call_types (from `Foo::bar` calls in the same chunk) over source_types,
    //    because `let x = Foo::new(); x.method()` means `.method()` is likely on Foo.
    if let Some((_, method)) = lower.rsplit_once('.') {
        let leaf_method = method.rsplit_once('.').map_or(method, |p| p.1);
        for ty in call_types.iter().chain(source_types.iter()) {
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
        // 6b. Type-aware receiver check: if we know the receiver's type from
        // param_types/local_types/field_types, only match if that type is a
        // project-internal type (exists in exact or short map).
        if let Some(recv_type) = receiver_types.get(receiver_leaf) {
            let ty_lower = recv_type.to_lowercase();
            // Type exists in project → resolve via Type::method or short fallback
            let qualified = format!("{ty_lower}::{method}");
            if let Some(&idx) = exact.get(&qualified) {
                return Some(idx);
            }
            // Type is in project (exact or short map) → allow short fallback
            if short.contains_key(&ty_lower) || exact.contains_key(&ty_lower) {
                return short.get(method).copied();
            }
            // Type is external (File, OpenOptions, etc.) → skip
            return None;
        }
        // 6c. Unknown receiver, unknown type → don't match.
        // As type inference improves, more receivers become known and
        // true positives recover automatically — no hardcoded blocklist needed.
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

/// Collect type_refs from struct/impl chunks keyed by owning type name (lowercase).
///
/// For `impl SimpleHybridSearcher<D, T>` with type_refs `[Bm25Index, HnswGraph, ...]`,
/// produces `{"simplehybridsearcher": ["Bm25Index", "HnswGraph", ...]}`.
/// Methods of that type can then use these types to resolve `self.field.method()` calls.
fn collect_owner_types(chunks: &[CodeChunk], _meta: &ChunkMeta) -> BTreeMap<String, Vec<String>> {
    let mut result: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in chunks {
        if !matches!(c.kind.as_str(), "struct" | "impl") {
            continue;
        }
        // owning_type uses lowercase leaf before generic params
        // e.g. "SimpleHybridSearcher<D, T>" → key should match owning_type output
        let lower = c.name.to_lowercase();
        let leaf = lower.rsplit("::").next().unwrap_or(&lower);
        // Strip generic params for the key: "simplehybridsearcher<d, t>" → "simplehybridsearcher"
        let key = leaf.split('<').next().unwrap_or(leaf);
        let entry = result.entry(key.to_owned()).or_default();
        for ty in &c.types {
            if !entry.contains(ty) {
                entry.push(ty.clone());
            }
        }
    }
    result
}

/// Collect struct field name → type mappings keyed by owning type (lowercase).
///
/// E.g. struct `MmapStaticModel { tokenizer: Tokenizer, weights: Vec }` →
/// `{"mmapmstaticmodel": {"tokenizer": "tokenizer", "weights": "vec"}}`.
fn collect_owner_field_types(chunks: &[CodeChunk]) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut result: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for c in chunks {
        if c.kind != "struct" {
            continue;
        }
        let lower = c.name.to_lowercase();
        let leaf = lower.rsplit("::").next().unwrap_or(&lower);
        let key = leaf.split('<').next().unwrap_or(leaf);
        let entry = result.entry(key.to_owned()).or_default();
        for (field_name, field_type) in &c.field_types {
            entry.insert(field_name.to_lowercase(), field_type.to_lowercase());
        }
    }
    result
}

/// Build function name → return type map (lowercase → lowercase leaf type).
///
/// Resolves `Self` to the owning type: `Foo::new → Self` becomes `foo::new → foo`.
fn build_return_type_map(chunks: &[CodeChunk]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        let Some(ref ret) = c.return_type else { continue };
        let ret_lower = ret.to_lowercase();
        // Extract leaf type: "Result<Vec<Item>>" → "result", "Option<String>" → "option"
        // For Self resolution, check the raw type
        let leaf = extract_leaf_type(&ret_lower);
        let resolved = if leaf == "self" || leaf == "&self" || leaf == "&mut self" {
            // Resolve Self to owning type
            owning_type(&c.name).unwrap_or(leaf.to_owned())
        } else {
            leaf.to_owned()
        };
        let name_lower = c.name.to_lowercase();
        map.insert(name_lower, resolved);
    }
    map
}

/// Extract the leaf type name from a possibly generic/reference type string.
/// `"result<vec<item>>"` → `"result"`, `"&mut foo"` → `"foo"`, `"Self"` → `"self"`
fn extract_leaf_type(ty: &str) -> &str {
    let ty = ty.strip_prefix('&').unwrap_or(ty);
    let ty = ty.strip_prefix("mut ").unwrap_or(ty);
    ty.split('<').next().unwrap_or(ty).trim()
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
/// Strips generic parameters: "Foo<T>::bar" → "foo".
/// Strip generic params from each `::` segment of a key.
/// `"foo<t>::bar"` → `"foo::bar"`, `"foo<d, t>::search_ext"` → `"foo::search_ext"`.
fn strip_generics_from_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut depth = 0u32;
    for ch in key.chars() {
        match ch {
            '<' => depth += 1,
            '>' => { depth = depth.saturating_sub(1); }
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Build a receiver name → type map from param_types, local_types, field_types.
///
/// E.g. `param_types = [("handle", "File")]` → `{"handle": "file"}` (lowercase).
/// For `self.field` lookups, field_types are keyed as-is.
fn build_receiver_type_map(chunk: &CodeChunk) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for (name, ty) in &chunk.param_types {
        let name_lower = name.to_lowercase();
        let ty_lower = ty.to_lowercase();
        if name_lower != "self" && !ty_lower.is_empty() {
            map.insert(name_lower, ty_lower);
        }
    }
    for (name, ty) in &chunk.local_types {
        let name_lower = name.to_lowercase();
        let ty_lower = ty.to_lowercase();
        if !ty_lower.is_empty() {
            map.insert(name_lower, ty_lower);
        }
    }
    for (name, ty) in &chunk.field_types {
        let name_lower = name.to_lowercase();
        let ty_lower = ty.to_lowercase();
        if !ty_lower.is_empty() {
            map.insert(name_lower, ty_lower);
        }
    }
    map
}

/// Infer local variable types from `Foo::bar()` calls + return type map.
///
/// Scans calls for `Foo::new`, `Foo::open`, etc. patterns and uses the return_type_map
/// to infer that the result variable is of type Foo (or the actual return type).
/// This enables resolution of `x.method()` when `x = Foo::new()`.
fn infer_local_types_from_calls(
    calls: &[String],
    return_type_map: &BTreeMap<String, String>,
    receiver_types: &mut BTreeMap<String, String>,
) {
    for call in calls {
        // Only process `Type::method` patterns (constructors/static methods)
        let Some((prefix, _suffix)) = call.split_once("::") else { continue };
        let leaf_type = prefix.rsplit("::").next().unwrap_or(prefix);
        let call_lower = call.to_lowercase();
        // Look up return type of this function call
        if let Some(ret_type) = return_type_map.get(&call_lower) {
            // The return type is the inferred type for any variable assigned from this call.
            // We use the type prefix as a potential receiver name hint.
            // E.g. `let engine = StorageEngine::open()` → engine: storageengine
            let type_lower = leaf_type.to_lowercase();
            // Add the type itself as a known "callable type" — this helps when
            // the variable name matches the type pattern (engine ~ StorageEngine).
            receiver_types.entry(type_lower).or_insert_with(|| ret_type.clone());
        }
    }
}

fn owning_type(name: &str) -> Option<String> {
    let (prefix, _) = name.rsplit_once("::")?;
    let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
    // Strip generic params: "Foo<T>" → "Foo"
    let stripped = leaf.split('<').next().unwrap_or(leaf);
    Some(stripped.to_lowercase())
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

/// Collect string_args from each chunk.
fn collect_string_args(chunks: &[CodeChunk]) -> Vec<Vec<(String, String, u32, u8)>> {
    chunks.iter().map(|c| c.string_args.clone()).collect()
}

/// Build a sorted index from lowercase string value → [(chunk_idx, line)].
fn build_string_index(all: &[Vec<(String, String, u32, u8)>]) -> Vec<(String, Vec<(u32, u32)>)> {
    let mut map: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    for (chunk_idx, args) in all.iter().enumerate() {
        for (_, value, line, _) in args {
            map.entry(value.to_lowercase())
                .or_default()
                .push((chunk_idx as u32, *line));
        }
    }
    map.into_iter().collect()
}

