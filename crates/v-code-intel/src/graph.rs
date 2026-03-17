//! Call graph adjacency list built from `ParsedChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.
//!
//! ## HOW TO EXTEND
//! - Add new traversal methods (e.g., shortest path) as `impl CallGraph` methods.
//! - Add new fields to `CallGraph` and update `build()` + bump cache version.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::parse::ParsedChunk;
use crate::rustdoc::RustdocTypes;

/// Cache format version — bump when struct layout changes.
const CACHE_VERSION: u8 = 7;

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
    /// Sorted (type::field, [chunk_indices]) for field-level blast radius.
    /// Key format: "typename::fieldname" (lowercase).
    pub field_access_index: Vec<(String, Vec<u32>)>,
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
    /// Enum type short names (lowercase). Used to identify enum variant
    /// constructors (`Type::Variant(args)`) that look like function calls
    /// to tree-sitter but are not actual call edges.
    enum_types: HashSet<String>,
    /// Known enum variant qualified names (lowercase): `"cfgexpr::any"`, `"hltag::symbol"`.
    /// Extracted from enum definition text.  Calls matching these are skipped.
    enum_variants: HashSet<String>,
}

impl ChunkMeta {
    fn collect(chunks: &[ParsedChunk]) -> Self {
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
            // Insert Owner::method alias for method_owners resolution.
            // e.g. "mod::Owner::method" → also insert "owner::method"
            if let Some((prefix, method_name)) = lower.rsplit_once("::") {
                if let Some(owner_leaf) = prefix.rsplit_once("::").map(|p| p.1) {
                    let alias = format!("{owner_leaf}::{method_name}");
                    if alias != lower {
                        exact.entry(alias).or_insert(idx);
                    }
                }
                // "impl Trait for Type::method" → insert "type::method" alias
                // Enables trait dispatch: receiver typed as `&impl Trait` or `&dyn Trait`
                // can resolve to the concrete impl's method.
                if let Some(for_pos) = prefix.find(" for ") {
                    let concrete_part = &prefix[for_pos + 5..];
                    let concrete_leaf = concrete_part.rsplit("::").next().unwrap_or(concrete_part);
                    let concrete_leaf = concrete_leaf.split('<').next().unwrap_or(concrete_leaf);
                    if !concrete_leaf.is_empty() {
                        let alias = format!("{concrete_leaf}::{method_name}");
                        exact.entry(alias).or_insert(idx);
                    }
                }
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

        // Collect enum type short names and variant names for variant detection.
        let mut enum_types = HashSet::new();
        let mut enum_variants = HashSet::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if kinds[i] == "enum" {
                let leaf = names[i].rsplit("::").next().unwrap_or(&names[i]).to_lowercase();
                enum_types.insert(leaf.clone());
                // Use pre-extracted variant names from tree-sitter (via Variants: line).
                for v in &chunk.enum_variants {
                    let qualified = format!("{leaf}::{v}");
                    enum_variants.insert(qualified);
                }
            }
        }

        Self { names, files, kinds, lines, signatures, is_test, name_index, exact, short, enum_types, enum_variants }
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
    pub fn build(chunks: &[ParsedChunk]) -> Self {
        Self::build_with_rustdoc(chunks, None)
    }

    /// Build the call graph, optionally enriched with rustdoc JSON type info.
    pub fn build_with_rustdoc(chunks: &[ParsedChunk], rustdoc: Option<&RustdocTypes>) -> Self {
        let meta = ChunkMeta::collect(chunks);
        let mut adj = AdjState::new(chunks.len());
        let owner_types = collect_owner_types(chunks, &meta);
        let owner_field_types = collect_owner_field_types(chunks);
        let return_type_map = build_return_type_map(chunks);
        let trait_methods = collect_trait_methods(chunks);
        let method_owners = build_method_owner_index(chunks);
        let instantiated = collect_instantiated_types(chunks);
        let trait_impl_methods = build_trait_impl_method_map(chunks);

        // Pass 1: resolve with static type info.
        for (src, c) in chunks.iter().enumerate() {
            Self::resolve_chunk_edges(src, c, &meta.exact, &meta.short, &meta.kinds, &owner_types, &owner_field_types, &return_type_map, &trait_methods, &method_owners, &instantiated, &trait_impl_methods, rustdoc, &mut adj, &HashSet::new(), &meta.enum_types, &meta.enum_variants);
        }

        // Pass 2+: iterative convergence.
        // Use resolved callees to infer more receiver types via return_type_map,
        // then re-resolve previously unresolved calls.
        // Accumulate extra receiver types across iterations so earlier discoveries persist.
        let mut extra_receiver_types: Vec<BTreeMap<String, String>> = vec![BTreeMap::new(); chunks.len()];
        for _pass in 0..3 {
            let prev_total: usize = adj.callees.iter().map(|c| c.len()).sum();
            // Build enriched receiver types from two sources:
            // (A) Forward: let_call_bindings + return_type_map (existing)
            // (B) Reverse argument propagation: callee param_types → caller variables

            // Build callee param_types index: chunk_idx → param_pos → type (lowercase)
            let callee_param_index: Vec<BTreeMap<u8, String>> = chunks.iter().map(|c| {
                c.param_types.iter().enumerate().filter_map(|(i, (name, ty))| {
                    if name.to_lowercase() == "self" || ty.is_empty() { return None; }
                    let ty_lower = extract_leaf_type(&ty.to_lowercase()).to_owned();
                    Some((i as u8, ty_lower))
                }).collect()
            }).collect();

            for (src, chunk) in chunks.iter().enumerate() {
                // (A) Forward: let_call_bindings → return type propagation
                for (var_name, callee_name) in &chunk.let_call_bindings {
                    let callee_lower = callee_name.to_lowercase();
                    for &callee_idx in &adj.callees[src] {
                        let ci = callee_idx as usize;
                        let resolved_lower = meta.names[ci].to_lowercase();
                        let resolved_short = resolved_lower.rsplit("::").next().unwrap_or(&resolved_lower);
                        let callee_short = callee_lower.rsplit("::").next().unwrap_or(&callee_lower);
                        let callee_leaf = callee_short.rsplit('.').next().unwrap_or(callee_short);
                        if resolved_short == callee_leaf
                            && let Some(ret) = return_type_map.get(&resolved_lower) {
                                extra_receiver_types[src]
                                    .entry(var_name.to_lowercase())
                                    .or_insert_with(|| ret.clone());
                            }
                    }
                }

                // (B) Reverse argument propagation (Andersen/PoTo style):
                // For each param_flow (param_name, _, callee_raw, callee_arg, _),
                // if callee_raw resolved to a known chunk, use that chunk's
                // param_types[callee_arg] to infer param_name's type.
                for (param_name, _param_pos, callee_raw, callee_arg, _line) in &chunk.param_flows {
                    let callee_lower = callee_raw.to_lowercase();
                    let callee_leaf = callee_lower.rsplit("::").next().unwrap_or(&callee_lower);
                    let callee_leaf = callee_leaf.rsplit('.').next().unwrap_or(callee_leaf);
                    for &callee_idx in &adj.callees[src] {
                        let ci = callee_idx as usize;
                        let resolved_lower = meta.names[ci].to_lowercase();
                        let resolved_short = resolved_lower.rsplit("::").next().unwrap_or(&resolved_lower);
                        if resolved_short != callee_leaf { continue; }
                        // callee_arg is 0-based; +1 to skip self for methods
                        let param_idx = *callee_arg;
                        if let Some(ty) = callee_param_index[ci].get(&param_idx)
                            && !ty.is_empty() {
                                extra_receiver_types[src]
                                    .entry(param_name.to_lowercase())
                                    .or_insert_with(|| ty.clone());
                            }
                    }
                }
            }

            // Re-resolve only chunks that got new receiver types.
            let mut new_adj = AdjState::new(chunks.len());
            for (src, c) in chunks.iter().enumerate() {
                if extra_receiver_types[src].is_empty() {
                    // Copy existing edges.
                    new_adj.callees[src] = adj.callees[src].clone();
                    new_adj.callers[src] = adj.callers[src].clone();
                    new_adj.call_sites[src] = adj.call_sites[src].clone();
                    continue;
                }
                // Build skip set from already-resolved callee short names.
                let skip: HashSet<String> = adj.callees[src].iter().map(|&idx| {
                    let name = &meta.names[idx as usize];
                    name.rsplit("::").next().unwrap_or(name).to_lowercase()
                }).collect();
                // Merge extra receiver types into chunk's let_call_bindings context.
                // We pass extra types via a temporary enriched chunk.
                let mut enriched = c.clone();
                for (var, ty) in &extra_receiver_types[src] {
                    enriched.local_types.push((var.clone(), ty.clone()));
                }
                Self::resolve_chunk_edges(src, &enriched, &meta.exact, &meta.short, &meta.kinds, &owner_types, &owner_field_types, &return_type_map, &trait_methods, &method_owners, &instantiated, &trait_impl_methods, rustdoc, &mut new_adj, &skip, &meta.enum_types, &meta.enum_variants);
                // Merge with existing edges.
                for &callee in &adj.callees[src] {
                    if !new_adj.callees[src].contains(&callee) {
                        new_adj.callees[src].push(callee);
                    }
                }
                for &caller in &adj.callers[src] {
                    if !new_adj.callers[src].contains(&caller) {
                        new_adj.callers[src].push(caller);
                    }
                }
                for &site in &adj.call_sites[src] {
                    if !new_adj.call_sites[src].contains(&site) {
                        new_adj.call_sites[src].push(site);
                    }
                }
            }
            adj = new_adj;
            let new_total: usize = adj.callees.iter().map(|c| c.len()).sum();
            if new_total == prev_total {
                break; // Converged
            }
        }

        adj.dedup();
        Self::from_parts(meta, adj, chunks)
    }

    /// Resolve tree-sitter edges for a single chunk.
    ///
    /// `skip_calls` contains short names already resolved — those are skipped.
    #[expect(clippy::too_many_arguments, reason = "graph resolution needs all lookup tables")]
    fn resolve_chunk_edges(
        src: usize,
        chunk: &ParsedChunk,
        exact: &BTreeMap<String, u32>,
        short: &BTreeMap<String, u32>,
        kinds: &[String],
        owner_types: &BTreeMap<String, Vec<String>>,
        owner_field_types: &BTreeMap<String, BTreeMap<String, String>>,
        return_type_map: &BTreeMap<String, String>,
        trait_methods: &BTreeSet<String>,
        method_owners: &BTreeMap<String, Vec<String>>,
        instantiated: &HashSet<String>,
        trait_impl_methods: &BTreeMap<String, Vec<String>>,
        rustdoc: Option<&RustdocTypes>,
        adj: &mut AdjState,
        skip_calls: &HashSet<String>,
        enum_types: &HashSet<String>,
        enum_variants: &HashSet<String>,
    ) {
        let imports = build_import_map(&chunk.imports);
        let self_type = owning_type(&chunk.name);
        let call_types = extract_call_types(&chunk.calls);
        let mut enriched_types = chunk.types.clone();
        if let Some(ref owner) = self_type
            && let Some(extra) = owner_types.get(owner.as_str()) {
                for t in extra {
                    if !enriched_types.contains(t) {
                        enriched_types.push(t.clone());
                    }
                }
            }
        // Build receiver name → type lookup from param/local/field types.
        let mut receiver_types = build_receiver_type_map(chunk);
        // Register self type so chained access (self.field.method) can resolve.
        if let Some(ref owner) = self_type {
            receiver_types.entry("self".to_owned()).or_insert_with(|| owner.clone());
            // Enrich with struct field types for self.field resolution.
            if let Some(fields) = owner_field_types.get(owner.as_str()) {
                for (field_name, field_type) in fields {
                    receiver_types.entry(field_name.clone()).or_insert_with(|| field_type.clone());
                }
            }
        }
        // Infer local variable types from return_type_map.
        // `let x = Foo::new()` → if return_type_map["foo::new"] = "foo", then x: foo.
        infer_local_types_from_calls(&chunk.calls, &chunk.let_call_bindings, return_type_map, &mut receiver_types);
        // Generic bound resolution: `fn foo<T: Trait>(x: T)` → x's type = trait name.
        // This allows `x.method()` to resolve to `Trait::method`.
        if let Some(ref sig) = chunk.signature {
            let bounds = extract_generic_bounds(sig);
            for (type_param, trait_bound) in &bounds {
                // Find params whose type is this generic param (e.g., param type "T")
                for (param_name, param_type) in &chunk.param_types {
                    if param_type.to_lowercase() == *type_param {
                        receiver_types
                            .entry(param_name.to_lowercase())
                            .or_insert_with(|| trait_bound.clone());
                    }
                }
            }
        }
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
                &receiver_types, trait_methods, method_owners, instantiated, trait_impl_methods, rustdoc, kinds, enum_types, enum_variants,
            ) {
                let call_line = chunk.call_lines.get(call_idx).copied().unwrap_or(0);
                adj.add_edge(src, tgt, call_line);
            }
        }
        resolve_type_ref_edges(src, &chunk.types, exact, short, &imports, &mut adj.callees, &mut adj.callers);
    }

    /// Assemble a `CallGraph` from pre-built metadata and adjacency state.
    fn from_parts(meta: ChunkMeta, adj: AdjState, chunks: &[ParsedChunk]) -> Self {
        let trait_impls = build_trait_impls(&meta.names, &meta.kinds, &meta.exact, &meta.short);
        let string_args = collect_string_args(chunks);
        let string_index = build_string_index(&string_args);
        #[expect(clippy::type_complexity, reason = "tuple layout mirrors string_args")]
        let param_flows: Vec<Vec<(String, u8, String, u8, u32)>> =
            chunks.iter().map(|c| c.param_flows.clone()).collect();
        let field_access_index = build_field_access_index(chunks, &meta);
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
            field_access_index,
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

    /// Find chunks that access a specific type::field (exact match, case-insensitive).
    ///
    /// `key` should be in `"typename::fieldname"` format (lowercase).
    pub fn find_field_access(&self, key: &str) -> Vec<u32> {
        let lower = key.to_lowercase();
        match self.field_access_index.binary_search_by_key(&&*lower, |(k, _)| k.as_str()) {
            Ok(i) => self.field_access_index[i].1.clone(),
            Err(_) => Vec::new(),
        }
    }

    /// Find all field access entries for a given type (prefix match).
    ///
    /// Returns `(field_name, chunk_indices)` pairs.
    pub fn find_field_accesses_for_type(&self, type_name: &str) -> Vec<(&str, &[u32])> {
        let prefix = format!("{}::", type_name.to_lowercase());
        let start = self.field_access_index.partition_point(|(k, _)| k.as_str() < prefix.as_str());
        let mut results = Vec::new();
        for (key, indices) in &self.field_access_index[start..] {
            if let Some(field) = key.strip_prefix(&prefix) {
                results.push((field, indices.as_slice()));
            } else {
                break;
            }
        }
        results
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
    ///
    /// Uses `payload.dat` mtime (not directory mtime) for invalidation —
    /// directory mtime doesn't update on Windows when files inside change.
    pub fn load(db: &Path) -> Option<Self> {
        let path = graph_cache_path(db);
        let db_mtime = fs::metadata(db.join("payload.dat"))
            .and_then(|m| m.modified())
            .ok()?;
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
#[expect(clippy::too_many_arguments, reason = "graph resolution needs all lookup tables")]
fn resolve_with_imports(
    call: &str,
    exact: &BTreeMap<String, u32>,
    short: &BTreeMap<String, u32>,
    imports: &BTreeMap<String, String>,
    self_type: Option<&str>,
    source_types: &[String],
    call_types: &[String],
    receiver_types: &BTreeMap<String, String>,
    _trait_methods: &BTreeSet<String>,
    method_owners: &BTreeMap<String, Vec<String>>,
    instantiated: &HashSet<String>,
    trait_impl_methods: &BTreeMap<String, Vec<String>>,
    rustdoc: Option<&RustdocTypes>,
    kinds: &[String],
    _enum_types: &HashSet<String>,
    enum_variants: &HashSet<String>,
) -> Option<u32> {
    let lower = call.to_lowercase();

    // Skip Rust prelude enum variant constructors that tree-sitter sees as calls.
    // `Ok(v)`, `Err(e)`, `Some(v)` are bare calls starting with uppercase.
    // Only skip when original call starts with uppercase (preserves real `ok()` functions).
    static PRELUDE_VARIANTS: &[&str] = &["ok", "err", "some", "none"];
    if call.starts_with(char::is_uppercase) && PRELUDE_VARIANTS.contains(&lower.as_str()) {
        return None;
    }

    // Skip enum variant constructors: `Type::Variant(args)` looks like a call
    // to tree-sitter but isn't a function call.  Check against the extracted
    // enum_variants set which contains `"type::variant"` pairs parsed from
    // enum definitions.  Also require the original call's last segment to start
    // with uppercase to avoid skipping real `Type::method()` calls.
    if let Some((prefix, _name)) = lower.rsplit_once("::") {
        let orig_last = call.rsplit_once("::").map_or(call, |p| p.1);
        if orig_last.starts_with(char::is_uppercase) {
            let type_leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
            let variant_key = format!("{type_leaf}::{}", _name);
            if enum_variants.contains(&variant_key) {
                return None;
            }
        }
    }

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
            // Direct field type lookup (from struct definition or rustdoc field_types)
            let field_type_from_receiver = receiver_types.get(field_leaf).cloned();
            let field_type_resolved = field_type_from_receiver.or_else(|| {
                // Fallback: try rustdoc field_types for self.field resolution.
                // Key format: "owner_type.field_name" → field_type
                let owner = self_type?;
                let rdoc = rustdoc?;
                let key = format!("{owner}.{field_leaf}");
                rdoc.field_types.get(&key).cloned()
            });
            if let Some(ref field_type) = field_type_resolved {
                let candidate = format!("{field_type}::{leaf_method}");
                if let Some(&idx) = exact.get(&candidate) {
                    return Some(idx);
                }
                // Field type exists in project → allow short fallback
                if (short.contains_key(field_type) || exact.contains_key(field_type))
                    && let Some(&idx) = short.get(leaf_method) {
                        return Some(idx);
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
            // Try method_owners for self.field.method — RTA-filtered unique resolve
            if let Some(owners) = method_owners.get(leaf_method) {
                let rta: Vec<&String> = if !instantiated.is_empty() {
                    let f: Vec<&String> = owners.iter().filter(|o| instantiated.contains(o.as_str())).collect();
                    if f.is_empty() { owners.iter().collect() } else { f }
                } else {
                    owners.iter().collect()
                };
                if rta.len() == 1 {
                    let qualified = format!("{}::{}", rta[0], leaf_method);
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
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
    if let Some((prefix, suffix)) = lower.rsplit_once("::") {
        // Skip enum variant constructors: `CliError::Embed(...)` is not a function call.
        let prefix_leaf = prefix.rsplit("::").next().unwrap_or(prefix);
        if let Some(&idx) = exact.get(prefix_leaf)
            && let Some(kind) = kinds.get(idx as usize)
                && kind == "enum" {
                    return None;
                }
        // Skip external enum variant constructors: `Cow::Borrowed(...)` where
        // "cow" is not a project type and "Borrowed" starts with uppercase.
        // This catches external enum variants without hardcoding specific types.
        if !exact.contains_key(prefix_leaf) && !short.contains_key(prefix_leaf) {
            let orig_suffix = call.rsplit_once("::").map_or(call, |p| p.1);
            if orig_suffix.starts_with(char::is_uppercase) {
                return None;
            }
        }
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
            // Trait dispatch: receiver is a trait → look up concrete impl via trait_impl_methods.
            // e.g. pstore: &dyn PayloadStore → trait_impl_methods["get_payload"] → ["filepayloadstore"]
            if let Some(impl_types) = trait_impl_methods.get(method) {
                if impl_types.len() == 1 {
                    let qualified = format!("{}::{}", impl_types[0], method);
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                } else {
                    // Multiple impls — narrow by type context or receiver name similarity
                    let matched = impl_types.iter().find(|t| {
                        source_types.iter().chain(call_types.iter())
                            .any(|st| st.to_lowercase() == **t)
                    }).or_else(|| impl_types.iter().find(|t| {
                        t.contains(receiver_leaf) || receiver_leaf.contains(t.as_str())
                    }));
                    if let Some(t) = matched {
                        let qualified = format!("{t}::{method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    }
                }
            }
            // Also check method_owners (non-trait methods)
            if let Some(owners) = method_owners.get(method)
                && let Some(owner) = owners.iter().find(|o| **o == ty_lower) {
                    let qualified = format!("{owner}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                }
            // Type is in project (exact or short map) → allow short fallback
            if short.contains_key(&ty_lower) || exact.contains_key(&ty_lower) {
                return short.get(method).copied();
            }
            // Type is external (File, OpenOptions, etc.) → skip
            return None;
        }
        // 6c. Unknown receiver → try rustdoc method→owner disambiguation.
        //     Only resolve if the owner type appears in the function's type context
        //     (source_types or call_types). This prevents false positives like
        //     resolving HashMap::get() to Sq8VectorStore::get() just because
        //     Sq8VectorStore is the only project type with a `get` method.
        if let Some(rdoc) = rustdoc
            && let Some(owners) = rdoc.owners_of(method) {
                // Check if any owner type appears in the function's type context.
                let ctx_owner = owners.iter().find(|o| {
                    let ol = o.to_lowercase();
                    source_types.iter().chain(call_types.iter())
                        .any(|t| t.to_lowercase() == ol)
                });
                if let Some(owner) = ctx_owner {
                    let qualified = format!("{owner}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                }
                // Rustdoc knows this method exists on project types but none in context.
                // Fall through to 6d — method_owners has its own guards (unique owner,
                // receiver name match, self_type, receiver_types, type context).
            }
        // 6d. Project-internal method→owner disambiguation.
        //     method_name → [owner_types] reverse index from all project functions.
        //     Resolution priority:
        //     (1) Unique owner → resolve immediately
        //     (2) Receiver name contains owner type → resolve (e.g. engine ~ storageengine)
        //     (3) Self type match → resolve to own type's method
        //     (4) Known receiver type from receiver_types → resolve
        //     (5) Type context (source_types + call_types + imports) → narrow
        if let Some(owners) = method_owners.get(method) {
            // RTA filter: if instantiated set is non-empty, narrow owners
            // to only those types that are actually instantiated.
            let rta_filtered: Vec<&String> = if !instantiated.is_empty() {
                let filtered: Vec<&String> = owners.iter().filter(|o| instantiated.contains(o.as_str())).collect();
                if filtered.is_empty() { owners.iter().collect() } else { filtered }
            } else {
                owners.iter().collect()
            };
            if rta_filtered.len() == 1 {
                let qualified = format!("{}::{}", rta_filtered[0], method);
                if let Some(&idx) = exact.get(&qualified) {
                    return Some(idx);
                }
            } else if owners.len() > 1 {
                // (2) Receiver name similarity — "engine" matches "storageengine"
                let matched_owner = owners.iter().find(|o| {
                    o.contains(receiver_leaf) || receiver_leaf.contains(o.as_str())
                });
                if let Some(owner) = matched_owner {
                    let qualified = format!("{owner}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                }
                // (3) Self type — method on the same struct we're in
                if let Some(st) = self_type
                    && owners.contains(&st.to_owned()) {
                        let qualified = format!("{st}::{method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    }
                // (4) Known receiver type from receiver_types
                if let Some(recv_type) = receiver_types.get(receiver_leaf)
                    && owners.contains(recv_type) {
                        let qualified = format!("{recv_type}::{method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    }
                // (5) Type context — source_types, call_types, imports
                let ctx_owner = owners.iter().find(|o| {
                    source_types.iter().chain(call_types.iter())
                        .any(|t| t.to_lowercase() == **o)
                    || imports.values().any(|v| {
                        let leaf = v.rsplit("::").next().unwrap_or(v);
                        leaf == o.as_str()
                    })
                });
                if let Some(owner) = ctx_owner {
                    let qualified = format!("{owner}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                }
            }
        }
        // 6e. Trait impl method fallback: if method is known only through
        //      trait impls (not in method_owners), try unique impl resolution.
        if method_owners.get(method).is_none()
            && let Some(impl_types) = trait_impl_methods.get(method) {
                if impl_types.len() == 1 {
                    let qualified = format!("{}::{}", impl_types[0], method);
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                } else {
                    // Narrow by receiver name similarity or type context
                    let matched = impl_types.iter().find(|t| {
                        t.contains(receiver_leaf) || receiver_leaf.contains(t.as_str())
                    }).or_else(|| impl_types.iter().find(|t| {
                        source_types.iter().chain(call_types.iter())
                            .any(|st| st.to_lowercase() == **t)
                    }));
                    if let Some(t) = matched {
                        let qualified = format!("{t}::{method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    }
                }
            }
        // Unknown receiver + unknown method → too ambiguous → skip.
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
fn collect_owner_types(chunks: &[ParsedChunk], _meta: &ChunkMeta) -> BTreeMap<String, Vec<String>> {
    let mut result: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in chunks {
        if !matches!(c.kind.as_str(), "struct" | "impl") {
            continue;
        }
        // owning_type uses lowercase leaf before generic params
        // e.g. "SimpleHybridSearcher<D, T>" → key should match owning_type output
        let lower = c.name.to_lowercase();
        let leaf = lower.rsplit("::").next().unwrap_or(&lower);
        // Handle trait impl: "sometrait for concretetype" → "concretetype"
        let leaf = leaf.rsplit_once(" for ").map_or(leaf, |(_, concrete)| concrete);
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
fn collect_owner_field_types(chunks: &[ParsedChunk]) -> BTreeMap<String, BTreeMap<String, String>> {
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
fn build_return_type_map(chunks: &[ParsedChunk]) -> BTreeMap<String, String> {
    // Collect all project type names (structs, enums, traits, etc.)
    let project_types: std::collections::HashSet<String> = chunks
        .iter()
        .filter(|c| matches!(c.kind.as_str(), "struct" | "enum" | "trait" | "class" | "interface"))
        .map(|c| {
            let short = c.name.rsplit("::").next().unwrap_or(&c.name);
            short.to_lowercase()
        })
        .collect();

    let mut map = BTreeMap::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        let Some(ref ret) = c.return_type else { continue };
        let ret_lower = ret.to_lowercase();
        let leaf = extract_leaf_type(&ret_lower);
        let resolved = if leaf == "self" || leaf == "&self" || leaf == "&mut self" {
            owning_type(&c.name).unwrap_or(leaf.to_owned())
        } else if project_types.contains(leaf) {
            leaf.to_owned()
        } else {
            // Try deeper unwrap for nested wrappers: Result<Option<Foo>> → Foo
            extract_project_type_from_return(&ret_lower, &project_types)
                .unwrap_or_else(|| leaf.to_owned())
        };
        let name_lower = c.name.to_lowercase();
        map.insert(name_lower, resolved);
    }
    map
}

/// Build a reverse index: method_name → [owner_type1, owner_type2, ...].
///
/// Scans all `Type::method` function chunks and groups by method short name.
/// Used for resolving `receiver.method()` when receiver type is unknown.
fn build_method_owner_index(chunks: &[ParsedChunk]) -> BTreeMap<String, Vec<String>> {
    let mut index: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        // Only methods (Type::method pattern)
        let Some((prefix, method)) = c.name.rsplit_once("::") else { continue };
        let owner = prefix.rsplit("::").next().unwrap_or(prefix);
        // Strip generics: "Foo<T>" → "Foo"
        let owner_clean = owner.split('<').next().unwrap_or(owner).to_lowercase();
        let method_lower = method.to_lowercase();
        let entry = index.entry(method_lower).or_default();
        if !entry.contains(&owner_clean) {
            entry.push(owner_clean);
        }
    }
    index
}

/// Build trait method → concrete impl type map from `impl Trait for Type` chunks.
///
/// For `impl Search for Engine` with a method `search()`, this produces
/// `{"search": ["engine"]}` — meaning `x.search()` could resolve to `Engine::search`.
/// This complements method_owners by adding trait-based disambiguation.
fn build_trait_impl_method_map(
    chunks: &[ParsedChunk],
) -> BTreeMap<String, Vec<String>> {
    // Step 1: Find all "impl Trait for Type" chunks → (trait_name, concrete_type)
    let mut trait_to_types: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in chunks {
        if c.kind != "impl" {
            continue;
        }
        let lower = c.name.to_lowercase();
        let Some(pos) = lower.find(" for ") else { continue };
        let trait_name = lower[..pos].trim();
        let concrete = lower[pos + 5..].trim();
        // Strip generics: "hnsw<d>" → "hnsw"
        let trait_clean = trait_name.rsplit("::").next().unwrap_or(trait_name);
        let trait_clean = trait_clean.split('<').next().unwrap_or(trait_clean);
        let concrete_clean = concrete.rsplit("::").next().unwrap_or(concrete);
        let concrete_clean = concrete_clean.split('<').next().unwrap_or(concrete_clean);
        if !trait_clean.is_empty() && !concrete_clean.is_empty() {
            trait_to_types
                .entry(trait_clean.to_owned())
                .or_default()
                .push(concrete_clean.to_owned());
        }
    }

    // Step 2: Find trait methods (Trait::method chunks) → map method → concrete types
    let mut method_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        let lower = c.name.to_lowercase();
        let Some((prefix, method)) = lower.rsplit_once("::") else { continue };
        let leaf_prefix = prefix.rsplit("::").next().unwrap_or(prefix);
        let leaf_clean = leaf_prefix.split('<').next().unwrap_or(leaf_prefix);
        // Handle "trait for type::method" pattern: extract trait name
        let trait_key = if let Some(pos) = leaf_clean.find(" for ") {
            &leaf_clean[..pos]
        } else {
            leaf_clean
        };
        if let Some(concrete_types) = trait_to_types.get(trait_key) {
            let entry = method_map.entry(method.to_owned()).or_default();
            for ct in concrete_types {
                if !entry.contains(ct) {
                    entry.push(ct.clone());
                }
            }
        }
    }
    method_map
}

/// Collect instantiated types from constructor calls (RTA — Rapid Type Analysis).
///
/// Scans all `Type::new()`, `Type::default()`, `Type::from()`, `Type::builder()`,
/// `Type::with_*()`, `Type::create()`, `Type::open()`, `Type::connect()` calls
/// to identify types that are actually instantiated in the project.
fn collect_instantiated_types(chunks: &[ParsedChunk]) -> HashSet<String> {
    let constructors = [
        "new", "default", "from", "builder", "create", "open", "connect",
        "init", "build", "with_capacity", "with_config", "with_options",
    ];
    let mut instantiated = HashSet::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        for call in &c.calls {
            let Some((prefix, method)) = call.rsplit_once("::") else { continue };
            let method_lower = method.to_lowercase();
            if constructors.contains(&method_lower.as_str())
                || method_lower.starts_with("with_")
                || method_lower.starts_with("from_")
            {
                let leaf = prefix.rsplit("::").next().unwrap_or(prefix);
                let clean = leaf.split('<').next().unwrap_or(leaf).to_lowercase();
                if !clean.is_empty() {
                    instantiated.insert(clean);
                }
            }
        }
    }
    instantiated
}

/// Try to find a project type inside a return type string.
/// Handles nested wrappers: `Result<Option<StorageEngine>>` → `storageengine`.
fn extract_project_type_from_return(
    ret: &str,
    project_types: &std::collections::HashSet<String>,
) -> Option<String> {
    // Split by common delimiters and look for project types
    for token in ret.split(['<', '>', ',', '(', ')', '&', ' ']) {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        // Take the leaf: "std::path::PathBuf" → "pathbuf"
        let leaf = token.rsplit("::").next().unwrap_or(token);
        if project_types.contains(leaf) {
            return Some(leaf.to_owned());
        }
    }
    None
}

/// Collect trait method names from chunks.
///
/// Finds all trait chunks, then collects short method names from function chunks
/// whose name starts with `TraitName::` (e.g., `Iterator::next` → `"next"`).
fn collect_trait_methods(chunks: &[ParsedChunk]) -> BTreeSet<String> {
    let trait_names: BTreeSet<String> = chunks
        .iter()
        .filter(|c| c.kind == "trait")
        .map(|c| {
            let lower = c.name.to_lowercase();
            lower.rsplit("::").next().unwrap_or(&lower).to_owned()
        })
        .collect();

    let mut methods = BTreeSet::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        let lower = c.name.to_lowercase();
        if let Some((prefix, method)) = lower.rsplit_once("::") {
            let leaf_prefix = prefix.rsplit("::").next().unwrap_or(prefix);
            // Strip generics: "iterator<item>" → "iterator"
            let leaf_clean = leaf_prefix.split('<').next().unwrap_or(leaf_prefix);
            if trait_names.contains(leaf_clean) {
                methods.insert(method.to_owned());
            }
        }
    }
    methods
}

/// Extract the leaf type name from a possibly generic/reference type string.
/// `"result<vec<item>>"` → `"result"`, `"&mut foo"` → `"foo"`, `"Self"` → `"self"`
pub fn extract_leaf_type(ty: &str) -> &str {
    let ty = ty.strip_prefix('&').unwrap_or(ty);
    // Strip lifetime: 'a , 'db , 'static  etc.
    let ty = if ty.starts_with('\'') {
        ty.find(' ').map_or(ty, |i| &ty[i + 1..])
    } else {
        ty
    };
    let ty = ty.strip_prefix("mut ").unwrap_or(ty);
    let ty = ty.strip_prefix("dyn ").unwrap_or(ty);
    let ty = ty.strip_prefix("impl ").unwrap_or(ty);
    let outer = ty.split('<').next().unwrap_or(ty).trim();
    // Unwrap common wrapper types to get the inner type
    if matches!(outer, "result" | "option" | "box" | "arc" | "rc" | "vec"
        | "Result" | "Option" | "Box" | "Arc" | "Rc" | "Vec")
    {
        // Extract first generic param: "Result<Foo, Error>" → "Foo"
        if let Some(start) = ty.find('<') {
            let inner = &ty[start + 1..];
            // Strip trailing '>' and take up to first ',' (for Result<T, E>)
            let inner = inner.trim_end_matches('>');
            let inner = inner.split(',').next().unwrap_or(inner).trim();
            let inner = inner.strip_prefix('&').unwrap_or(inner);
            let inner = inner.strip_prefix("mut ").unwrap_or(inner);
            let inner_leaf = inner.split('<').next().unwrap_or(inner).trim();
            if !inner_leaf.is_empty() && inner_leaf != outer {
                return inner_leaf;
            }
        }
    }
    outer
}

/// Extract generic type parameter → trait bound mappings from a signature.
///
/// `"fn foo<T: Search, U: Clone + Display>(x: T, y: U)"` →
/// `[("t", "search"), ("u", "clone")]` (first trait bound only).
pub fn extract_generic_bounds(sig: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    // Find generic params between first < and matching >
    let Some(start) = sig.find('<') else { return result };
    let mut depth = 0u32;
    let mut end = start;
    for (i, ch) in sig[start..].char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    end = start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    if end <= start { return result; }
    let generics = &sig[start + 1..end];
    // Split by comma (but not within nested <>)
    let mut params = Vec::new();
    let mut current = String::new();
    let mut d = 0u32;
    for ch in generics.chars() {
        match ch {
            '<' => { d += 1; current.push(ch); }
            '>' => { d = d.saturating_sub(1); current.push(ch); }
            ',' if d == 0 => { params.push(std::mem::take(&mut current)); }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() { params.push(current); }
    for param in &params {
        let param = param.trim();
        // "T: Search + Clone" → type_param="T", first_bound="Search"
        if let Some(colon) = param.find(':') {
            let type_param = param[..colon].trim().to_lowercase();
            let bounds = param[colon + 1..].trim();
            let first_bound = bounds.split('+').next().unwrap_or(bounds).trim();
            let first_bound = first_bound.split('<').next().unwrap_or(first_bound);
            if !type_param.is_empty() && !first_bound.is_empty() {
                result.push((type_param, first_bound.to_lowercase()));
            }
        }
    }
    // Also check where clause: "where T: Trait" (simple heuristic)
    if let Some(where_pos) = sig.to_lowercase().find("where ") {
        let where_clause = &sig[where_pos + 6..];
        for clause in where_clause.split(',') {
            let clause = clause.trim().trim_end_matches('{').trim();
            if let Some(colon) = clause.find(':') {
                let tp = clause[..colon].trim().to_lowercase();
                let bounds = clause[colon + 1..].trim();
                let first = bounds.split('+').next().unwrap_or(bounds).trim();
                let first = first.split('<').next().unwrap_or(first);
                if !tp.is_empty() && !first.is_empty() && !result.iter().any(|(t, _)| t == &tp) {
                    result.push((tp, first.to_lowercase()));
                }
            }
        }
    }
    result
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
/// Build a receiver→type map from a chunk's param/local/field types.
///
/// Populates variable names → leaf type names for method resolution.
pub fn build_receiver_type_map(chunk: &ParsedChunk) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for (name, ty) in &chunk.param_types {
        let name_lower = name.to_lowercase();
        let leaf = extract_leaf_type(&ty.to_lowercase()).to_owned();
        if name_lower != "self" && !leaf.is_empty() {
            map.insert(name_lower, leaf);
        }
    }
    for (name, ty) in &chunk.local_types {
        let name_lower = name.to_lowercase();
        let leaf = extract_leaf_type(&ty.to_lowercase()).to_owned();
        if !leaf.is_empty() {
            map.insert(name_lower, leaf);
        }
    }
    for (name, ty) in &chunk.field_types {
        let name_lower = name.to_lowercase();
        let leaf = extract_leaf_type(&ty.to_lowercase()).to_owned();
        if !leaf.is_empty() {
            map.insert(name_lower, leaf);
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
    let_call_bindings: &[(String, String)],
    return_type_map: &BTreeMap<String, String>,
    receiver_types: &mut BTreeMap<String, String>,
) {
    // 1) From `Type::method` patterns (constructors/static methods)
    for call in calls {
        let Some((prefix, _suffix)) = call.split_once("::") else { continue };
        let leaf_type = prefix.rsplit("::").next().unwrap_or(prefix);
        let call_lower = call.to_lowercase();
        if let Some(ret_type) = return_type_map.get(&call_lower) {
            let type_lower = leaf_type.to_lowercase();
            receiver_types.entry(type_lower).or_insert_with(|| ret_type.clone());
        }
    }

    // 2) From let_call_bindings: `let var = callee()` → var: return_type_map[callee]
    for (var_name, callee_name) in let_call_bindings {
        let callee_lower = callee_name.to_lowercase();
        // Try direct lookup first
        if let Some(ret_type) = return_type_map.get(&callee_lower) {
            receiver_types
                .entry(var_name.to_lowercase())
                .or_insert_with(|| ret_type.clone());
            continue;
        }
        // For `receiver.method`, try looking up just the method part
        // with known receiver type prefix: e.g. `self.method` → `SelfType::method`
        if let Some(dot) = callee_name.find('.') {
            let recv = &callee_name[..dot];
            let method = &callee_name[dot + 1..];
            let recv_lower = recv.to_lowercase();
            // If the receiver has a known type, try Type::method
            if let Some(recv_type) = receiver_types.get(&recv_lower).cloned() {
                let qualified = format!("{}::{}", recv_type, method.to_lowercase());
                if let Some(ret_type) = return_type_map.get(&qualified) {
                    receiver_types
                        .entry(var_name.to_lowercase())
                        .or_insert_with(|| ret_type.clone());
                    continue;
                }
            }
        }
    }
}

/// Extract owning type from a qualified name (e.g., `"Foo::bar"` → `"foo"`).
pub fn owning_type(name: &str) -> Option<String> {
    let (prefix, _) = name.rsplit_once("::")?;
    let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
    // Handle trait impl: "SomeTrait for ConcreteType" → "ConcreteType"
    let leaf = leaf.rsplit_once(" for ").map_or(leaf, |(_, concrete)| concrete);
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

pub fn is_test_chunk(c: &ParsedChunk) -> bool {
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
fn collect_string_args(chunks: &[ParsedChunk]) -> Vec<Vec<(String, String, u32, u8)>> {
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

/// Build field access index: `type::field` → chunk indices.
///
/// For each chunk's `field_accesses`, resolve the receiver variable to a type
/// using the same sources as edge resolution (self type, param types, local types,
/// field types), then index as `"typename::fieldname"`.
fn build_field_access_index(chunks: &[ParsedChunk], _meta: &ChunkMeta) -> Vec<(String, Vec<u32>)> {
    let mut map: BTreeMap<String, Vec<u32>> = BTreeMap::new();

    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.field_accesses.is_empty() {
            continue;
        }

        // Build receiver→type mapping for this chunk
        let mut receiver_types: BTreeMap<String, String> = BTreeMap::new();

        // self → owning type
        if let Some(owner) = owning_type(&chunk.name) {
            receiver_types.insert("self".to_owned(), owner);
        }

        // param_types
        for (pname, pty) in &chunk.param_types {
            let leaf = extract_leaf_type(&pty.to_lowercase()).to_owned();
            if !leaf.is_empty() && pname.to_lowercase() != "self" {
                receiver_types.entry(pname.to_lowercase()).or_insert(leaf);
            }
        }

        // local_types
        for (vname, vty) in &chunk.local_types {
            let leaf = extract_leaf_type(&vty.to_lowercase()).to_owned();
            if !leaf.is_empty() {
                receiver_types.entry(vname.to_lowercase()).or_insert(leaf);
            }
        }

        // self.field → field type (for struct fields accessed via self)
        if let Some(owner) = owning_type(&chunk.name) {
            // Find struct chunk's field_types
            let owner_lower = owner.to_lowercase();
            for c in chunks {
                let name_lower = c.name.to_lowercase();
                if (name_lower == owner_lower || name_lower.ends_with(&format!("::{owner_lower}")))
                    && c.kind == "struct"
                {
                    for (fname, fty) in &c.field_types {
                        let key = format!("self.{}", fname.to_lowercase());
                        let leaf = extract_leaf_type(&fty.to_lowercase()).to_owned();
                        receiver_types.entry(key).or_insert(leaf);
                    }
                    break;
                }
            }
        }

        for (recv, field) in &chunk.field_accesses {
            let recv_lower = recv.to_lowercase();
            let field_lower = field.to_lowercase();

            if let Some(ty) = receiver_types.get(&recv_lower) {
                let key = format!("{ty}::{field_lower}");
                map.entry(key).or_default().push(idx as u32);
            }
            // Also store unresolved as receiver_var::field for partial matches
            // (blast can match on just the field name portion)
        }
    }

    // Dedup each entry
    for list in map.values_mut() {
        list.sort_unstable();
        list.dedup();
    }

    map.into_iter().collect()
}

