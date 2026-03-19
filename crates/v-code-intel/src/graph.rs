//! Call graph adjacency list built from `ParsedChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.
//!
//! ## HOW TO EXTEND
//! - Add new traversal methods (e.g., shortest path) as `impl CallGraph` methods.
//! - Add new fields to `CallGraph` and update `build()` (cache auto-invalidates via source hash).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::parse::ParsedChunk;
use crate::rustdoc::RustdocTypes;

/// Source hash of graph.rs — auto-computed by build.rs.
/// Invalidates cache whenever resolve logic changes, no manual version bump needed.
const GRAPH_SOURCE_HASH: &str = env!("GRAPH_SOURCE_HASH");

/// Get current process RSS in MB (Windows).
pub fn current_rss_mb() -> f64 {
    #[cfg(windows)]
    {
        use std::mem::MaybeUninit;
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(
                process: *mut std::ffi::c_void,
                ppsmemcounters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        #[expect(unsafe_code, reason = "FFI for memory profiling")]
        // SAFETY: GetCurrentProcess returns a pseudo-handle, K32GetProcessMemoryInfo reads only.
        unsafe {
            let mut pmc = MaybeUninit::<ProcessMemoryCounters>::zeroed().assume_init();
            pmc.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                return pmc.working_set_size as f64 / (1024.0 * 1024.0);
            }
        }
        0.0
    }
    #[cfg(not(windows))]
    {
        0.0
    }
}

/// Pre-built call graph with bidirectional adjacency lists.
#[derive(bincode::Encode, bincode::Decode)]
pub struct CallGraph {
    /// Source hash for cache invalidation.
    version: String,
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
    /// chunk index → extern calls [(call_name, reason)].
    /// Populated during graph build when ExternMethodIndex is available.
    pub extern_calls: Vec<Vec<(String, String)>>,
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
    exact: HashMap<String, u32>,
    /// Short (last `::` segment) → index map.
    short: HashMap<String, u32>,
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
        let mut exact = HashMap::new();
        let mut short = HashMap::new();
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

    /// Build the call graph, optionally enriched with rustdoc JSON type info
    /// and extern method index for internal/external call classification.
    pub fn build_with_rustdoc(chunks: &[ParsedChunk], rustdoc: Option<&RustdocTypes>) -> Self {
        Self::build_full(chunks, rustdoc, None)
    }

    /// Build the call graph with all available type information.
    pub fn build_full(
        chunks: &[ParsedChunk],
        rustdoc: Option<&RustdocTypes>,
        extern_index: Option<&crate::extern_types::ExternMethodIndex>,
    ) -> Self {
        Self::build_full_inner(chunks, rustdoc, || extern_index.cloned())
    }

    /// Build the call graph with deferred extern index (runs in parallel).
    ///
    /// Resolves all graph edges first, then joins the extern index thread
    /// for classify_extern_calls. This overlaps extern parsing with graph build.
    pub fn build_full_deferred(
        chunks: &[ParsedChunk],
        rustdoc: Option<&RustdocTypes>,
        extern_handle: std::thread::JoinHandle<Option<crate::extern_types::ExternMethodIndex>>,
    ) -> Self {
        Self::build_full_inner(chunks, rustdoc, || extern_handle.join().ok().flatten())
    }


    /// Core graph build logic. `get_extern` is called after edge resolution
    /// to obtain the extern index for classification.
    fn build_full_inner(
        chunks: &[ParsedChunk],
        rustdoc: Option<&RustdocTypes>,
        get_extern: impl FnOnce() -> Option<crate::extern_types::ExternMethodIndex>,
    ) -> Self {
        let rss0 = current_rss_mb();
        eprintln!("      [graph] RSS baseline: {rss0:.1}MB");

        let t0 = std::time::Instant::now();
        let meta = ChunkMeta::collect(chunks);
        eprintln!("      [graph] meta collect: {:.1}ms  RSS: {:.1}MB (+{:.1})", t0.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        let t1 = std::time::Instant::now();
        let mut adj = AdjState::new(chunks.len());
        // Build index tables in parallel (all read-only on chunks).
        let (left, right) = rayon::join(
            || rayon::join(
                || rayon::join(
                    || collect_owner_types(chunks, &meta),
                    || collect_owner_field_types(chunks),
                ),
                || rayon::join(
                    || build_return_type_map(chunks),
                    || collect_trait_methods(chunks),
                ),
            ),
            || rayon::join(
                || build_method_owner_index(chunks),
                || rayon::join(
                    || collect_instantiated_types(chunks),
                    || build_trait_impl_method_map(chunks),
                ),
            ),
        );
        let ((owner_types, owner_field_types), (mut return_type_map, trait_methods)) = left;
        let (mut method_owners, (instantiated, trait_impl_methods)) = right;

        // Enrich with rustdoc compiler-resolved types (overlay on tree-sitter heuristics).
        if let Some(rdoc) = rustdoc {
            // 1. Merge rustdoc fn_return_types into return_type_map.
            //    Rustdoc has compiler-verified return types — prefer over tree-sitter guesses.
            let mut rdoc_added = 0usize;
            for (fn_name, ret_type) in &rdoc.fn_return_types {
                use std::collections::hash_map::Entry;
                match return_type_map.entry(fn_name.clone()) {
                    Entry::Vacant(e) => { e.insert(ret_type.clone()); rdoc_added += 1; }
                    Entry::Occupied(mut e) => {
                        // Only override if tree-sitter result is a generic placeholder
                        let current = e.get();
                        if current.len() <= 1 || current == "self" {
                            e.insert(ret_type.clone());
                            rdoc_added += 1;
                        }
                    }
                }
            }
            // 2. Merge rustdoc method_owner into method_owners.
            //    Adds owner information for methods that tree-sitter couldn't resolve.
            let mut owner_added = 0usize;
            for (method, owners) in &rdoc.method_owner {
                let entry = method_owners.entry(method.clone()).or_default();
                for owner in owners {
                    if !entry.contains(owner) {
                        entry.push(owner.clone());
                        owner_added += 1;
                    }
                }
            }
            eprintln!("      [graph] rustdoc overlay: +{rdoc_added} return types, +{owner_added} method owners");
        }

        eprintln!("      [graph] index tables: {:.1}ms  RSS: {:.1}MB (+{:.1})", t1.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        // Get extern index early (before resolve) so extern_all_methods can be used
        // to avoid resolving std methods (len, push, get) to project types.
        let t_ext = std::time::Instant::now();
        let extern_index = get_extern();
        let extern_all_methods = extern_index.as_ref()
            .map(|ext| ext.all_method_set())
            .unwrap_or_default();
        eprintln!("      [graph] extern join (early): {:.1}ms ({} methods)", t_ext.elapsed().as_secs_f64() * 1000.0, extern_all_methods.len());

        // Pre-compute per-chunk context (import_map, receiver_types, etc.) — done once.
        let t2 = std::time::Instant::now();
        use rayon::prelude::*;
        let mut ctxs: Vec<ChunkCtx> = chunks.par_iter()
            .map(|c| build_chunk_ctx(c, &owner_types, &owner_field_types, &return_type_map, &method_owners))
            .collect();

        // Pass 1: parallel resolve — each chunk independently collects edges.
        let par_results: Vec<(Vec<(u32, u32)>, Vec<(u32, u32)>, HashMap<String, String>, Vec<bool>)> = chunks.par_iter()
            .zip(ctxs.par_iter_mut())
            .enumerate()
            .map(|(src, (c, ctx))| {
                let empty_skip = HashSet::new();
                resolve_collect(src, c, ctx, &meta.exact, &meta.short, &meta.kinds, &meta.names, &return_type_map, &trait_methods, &method_owners, &instantiated, &trait_impl_methods, rustdoc, &empty_skip, &meta.enum_types, &meta.enum_variants, &extern_all_methods)
            })
            .collect();

        // Per-chunk resolved call indices from pass 1.
        let mut resolved_indices: Vec<Vec<bool>> = Vec::with_capacity(chunks.len());

        // Merge parallel results into AdjState.
        for (src, (edges, type_edges, _receiver_updates, _resolved)) in par_results.iter().enumerate() {
            for &(tgt, line) in edges {
                adj.add_edge(src, tgt, line);
            }
            for &(tgt, _) in type_edges {
                let tgt_usize = tgt as usize;
                if tgt_usize != src {
                    adj.callees[src].push(tgt);
                    adj.callers[tgt_usize].push(src as u32);
                }
            }
        }
        // Apply receiver_type updates and collect resolved indices from pass 1.
        for (src, (_, _, receiver_updates, resolved)) in par_results.into_iter().enumerate() {
            for (var, ty) in receiver_updates {
                ctxs[src].receiver_types.entry(var).or_insert(ty);
            }
            debug_assert_eq!(src, resolved_indices.len());
            resolved_indices.push(resolved);
        }
        eprintln!("      [graph] pass 1 resolve: {:.1}ms ({} chunks)  RSS: {:.1}MB (+{:.1})", t2.elapsed().as_secs_f64() * 1000.0, chunks.len(), current_rss_mb(), current_rss_mb() - rss0);

        // Pre-compute lowercase name + short (last :: segment) for each chunk.
        let names_lower: Vec<String> = meta.names.iter().map(|n| n.to_lowercase()).collect();
        let names_short: Vec<&str> = names_lower.iter().map(|n| n.rsplit("::").next().unwrap_or(n)).collect();

        // Pre-compute callee param index (unchanged across passes).
        let callee_param_index: Vec<HashMap<u8, String>> = chunks.iter().map(|c| {
            c.param_types.iter().enumerate().filter_map(|(i, (name, ty))| {
                if name.eq_ignore_ascii_case("self") || ty.is_empty() { return None; }
                let ty_lower = extract_leaf_type(&ty.to_lowercase()).to_owned();
                Some((i as u8, ty_lower))
            }).collect()
        }).collect();

        // Pass 2+: iterative convergence.
        let t_iter = std::time::Instant::now();
        let mut iter_passes = 0u32;
        for _pass in 0..3 {
            let prev_total: usize = adj.callees.iter().map(|c| c.len()).sum();

            // Parallel: compute extra_receiver_types per chunk (read-only on adj).
            let extra_receiver_types: Vec<HashMap<String, String>> = chunks.par_iter()
                .enumerate()
                .map(|(src, chunk)| {
                    let mut extra = HashMap::new();
                    for (var_name, callee_name) in &chunk.let_call_bindings {
                        let callee_lower = callee_name.to_lowercase();
                        let callee_short = callee_lower.rsplit("::").next().unwrap_or(&callee_lower);
                        let callee_leaf = callee_short.rsplit('.').next().unwrap_or(callee_short);
                        for &callee_idx in &adj.callees[src] {
                            let ci = callee_idx as usize;
                            if names_short[ci] == callee_leaf {
                                if let Some(ret) = return_type_map.get(&names_lower[ci]) {
                                    extra.entry(var_name.to_lowercase())
                                        .or_insert_with(|| ret.clone());
                                }
                            }
                        }
                    }
                    for (param_name, _param_pos, callee_raw, callee_arg, _line) in &chunk.param_flows {
                        let callee_lower = callee_raw.to_lowercase();
                        let callee_leaf_full = callee_lower.rsplit("::").next().unwrap_or(&callee_lower);
                        let callee_leaf = callee_leaf_full.rsplit('.').next().unwrap_or(callee_leaf_full);
                        for &callee_idx in &adj.callees[src] {
                            let ci = callee_idx as usize;
                            if names_short[ci] != callee_leaf { continue; }
                            let param_idx = *callee_arg;
                            if let Some(ty) = callee_param_index[ci].get(&param_idx) {
                                if !ty.is_empty() {
                                    extra.entry(param_name.to_lowercase())
                                        .or_insert_with(|| ty.clone());
                                }
                            }
                        }
                    }
                    extra
                })
                .collect();

            // Merge extra receiver types into ctxs.
            for (src, extra) in extra_receiver_types.iter().enumerate() {
                for (var, ty) in extra {
                    ctxs[src].receiver_types.entry(var.clone()).or_insert_with(|| ty.clone());
                }
            }

            // Parallel delta resolve: only enriched chunks, skip already-resolved call indices.
            let enriched_indices: Vec<usize> = (0..chunks.len())
                .filter(|&src| !extra_receiver_types[src].is_empty())
                .collect();
            let delta_results: Vec<(usize, Vec<(u32, u32)>)> = enriched_indices.par_iter()
                .map(|&src| {
                    let edges = resolve_delta(src, &chunks[src], &ctxs[src], &meta.exact, &meta.short, &return_type_map, &trait_methods, &method_owners, &instantiated, &trait_impl_methods, rustdoc, &resolved_indices[src], &meta.enum_types, &meta.enum_variants, &meta.kinds, &meta.names, &extern_all_methods);
                    (src, edges)
                })
                .collect();

            // Merge delta into adj.
            for (src, edges) in &delta_results {
                for &(tgt, line) in edges {
                    let tgt_usize = tgt as usize;
                    if tgt_usize != *src && !adj.callees[*src].contains(&tgt) {
                        adj.callees[*src].push(tgt);
                        adj.callers[tgt_usize].push(*src as u32);
                        adj.call_sites[*src].push((tgt, line));
                    }
                }
            }
            let new_total: usize = adj.callees.iter().map(|c| c.len()).sum();
            iter_passes += 1;
            let enriched_count = extra_receiver_types.iter().filter(|m| !m.is_empty()).count();
            eprintln!("        [iter] pass {iter_passes}: edges {prev_total}->{new_total} (+{}), enriched chunks: {enriched_count}, RSS: {:.1}MB", new_total - prev_total, current_rss_mb());
            if new_total == prev_total {
                break;
            }
        }
        eprintln!("      [graph] pass 2+ iterative: {:.1}ms ({iter_passes} passes)  RSS: {:.1}MB (+{:.1})", t_iter.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        let t3 = std::time::Instant::now();
        adj.dedup();
        eprintln!("      [graph] dedup: {:.1}ms", t3.elapsed().as_secs_f64() * 1000.0);

        let t5 = std::time::Instant::now();
        let extern_calls = Self::classify_extern_calls(
            chunks, &meta, &adj, &return_type_map, &owner_field_types, extern_index.as_ref(),
            &method_owners,
        );
        eprintln!("      [graph] classify extern: {:.1}ms  RSS: {:.1}MB (+{:.1})", t5.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        Self::from_parts(meta, adj, extern_calls, chunks, &owner_field_types)
    }

    // resolve_with_ctx is a free function defined outside impl CallGraph.


    /// Classify unresolved calls as extern or truly unresolved.
    ///
    /// For each function chunk, finds calls not resolved by the graph and checks
    /// if they match extern types (std/deps). Returns per-chunk extern call lists.
    fn classify_extern_calls(
        chunks: &[ParsedChunk],
        meta: &ChunkMeta,
        adj: &AdjState,
        return_type_map: &HashMap<String, String>,
        owner_field_types: &HashMap<String, HashMap<String, String>>,
        extern_index: Option<&crate::extern_types::ExternMethodIndex>,
        method_owners: &HashMap<String, Vec<String>>,
    ) -> Vec<Vec<(String, String)>> {
        let n = chunks.len();
        let Some(ext) = extern_index else {
            return vec![Vec::new(); n];
        };

        let project_type_shorts = collect_project_type_shorts(chunks);

        // Build project function short names for bare-fn extern classification.
        let project_fn_shorts: HashSet<String> = {
            let mut set = HashSet::new();
            for (i, kind) in meta.kinds.iter().enumerate() {
                if kind == "function" {
                    let leaf = meta.names[i].rsplit("::").next().unwrap_or(&meta.names[i]).to_lowercase();
                    set.insert(leaf);
                }
            }
            set
        };

        // Flat set for O(1) "any extern type has this method?" lookup.
        let extern_all_methods = ext.all_method_set();

        use rayon::prelude::*;
        chunks.par_iter().enumerate().map(|(i, chunk)| {
            if meta.kinds[i] != "function" {
                return Vec::new();
            }

            let resolved_shorts: HashSet<String> = adj.callees[i].iter().map(|&idx| {
                let name = &meta.names[idx as usize];
                name.rsplit("::").next().unwrap_or(name).to_lowercase()
            }).collect();

            let mut receiver_types = build_receiver_type_map(chunk);
            infer_local_types_from_calls(&chunk.calls, &chunk.let_call_bindings, return_type_map, &mut receiver_types);
            infer_receiver_types_by_co_methods(&chunk.calls, method_owners, &mut receiver_types);
            let mut extern_calls = Vec::new();

            for call in &chunk.calls {
                let call_lower = call.to_lowercase();
                let short = if let Some((_, s)) = call_lower.rsplit_once("::") {
                    s
                } else if let Some((_, s)) = call_lower.rsplit_once('.') {
                    s
                } else {
                    &call_lower
                };

                if resolved_shorts.contains(short) {
                    continue;
                }

                if let Some(reason) = check_extern(
                    &call_lower, &receiver_types, ext, &project_type_shorts,
                    return_type_map, &extern_all_methods, owner_field_types,
                    Some(&project_fn_shorts), method_owners,
                ) {
                    extern_calls.push((call.clone(), reason));
                }
            }
            extern_calls
        }).collect()
    }

    /// Assemble a `CallGraph` from pre-built metadata and adjacency state.
    fn from_parts(meta: ChunkMeta, adj: AdjState, extern_calls: Vec<Vec<(String, String)>>, chunks: &[ParsedChunk], owner_field_types: &HashMap<String, HashMap<String, String>>) -> Self {
        let t = std::time::Instant::now();
        let ((trait_impls, string_args), (field_access_index, param_flows)) = rayon::join(
            || rayon::join(
                || build_trait_impls(&meta.names, &meta.kinds, &meta.exact, &meta.short),
                || collect_string_args(chunks),
            ),
            || rayon::join(
                || build_field_access_index(chunks, owner_field_types),
                || {
                    #[expect(clippy::type_complexity, reason = "tuple layout mirrors string_args")]
                    let pf: Vec<Vec<(String, u8, String, u8, u32)>> =
                        chunks.iter().map(|c| c.param_flows.clone()).collect();
                    pf
                },
            ),
        );
        let string_index = build_string_index(&string_args);
        eprintln!("      [graph] from_parts: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
        Self {
            version: GRAPH_SOURCE_HASH.to_owned(),
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
            extern_calls,
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

        if graph.version != GRAPH_SOURCE_HASH {
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
/// Pre-computed per-chunk context for resolve. Built once, reused across passes.
#[derive(Clone)]
struct ChunkCtx {
    imports: HashMap<String, String>,
    self_type: Option<String>,
    call_types_set: HashSet<String>,
    import_leaves: HashSet<String>,
    source_types_set: HashSet<String>,
    receiver_types: HashMap<String, String>,
}

/// Build per-chunk context (expensive preprocessing done once).
fn build_chunk_ctx(
    chunk: &ParsedChunk,
    owner_types: &HashMap<String, Vec<String>>,
    owner_field_types: &HashMap<String, HashMap<String, String>>,
    return_type_map: &HashMap<String, String>,
    method_owners: &HashMap<String, Vec<String>>,
) -> ChunkCtx {
    let imports = build_import_map(&chunk.imports);
    let self_type = owning_type(&chunk.name);
    let call_types = extract_call_types(&chunk.calls);
    let call_types_set: HashSet<String> = call_types.iter().map(|t| t.to_lowercase()).collect();
    let import_leaves: HashSet<String> = imports.values()
        .map(|v| v.rsplit("::").next().unwrap_or(v).to_owned())
        .collect();
    let mut enriched_types = chunk.types.clone();
    if let Some(ref owner) = self_type
        && let Some(extra) = owner_types.get(owner.as_str()) {
            for t in extra {
                if !enriched_types.contains(t) {
                    enriched_types.push(t.clone());
                }
            }
        }
    let mut receiver_types = build_receiver_type_map(chunk);
    if let Some(ref owner) = self_type {
        receiver_types.entry("self".to_owned()).or_insert_with(|| owner.clone());
        if let Some(fields) = owner_field_types.get(owner.as_str()) {
            for (field_name, field_type) in fields {
                receiver_types.entry(field_name.clone()).or_insert_with(|| field_type.clone());
            }
        }
    }
    infer_local_types_from_calls(&chunk.calls, &chunk.let_call_bindings, return_type_map, &mut receiver_types);
    infer_receiver_types_by_co_methods(&chunk.calls, method_owners, &mut receiver_types);
    if let Some(ref sig) = chunk.signature {
        let bounds = extract_generic_bounds(sig);
        for (type_param, trait_bound) in &bounds {
            for (param_name, param_type) in &chunk.param_types {
                if param_type.to_lowercase() == *type_param {
                    receiver_types
                        .entry(param_name.to_lowercase())
                        .or_insert_with(|| trait_bound.clone());
                }
            }
        }
    }
    let source_types_set: HashSet<String> = enriched_types.iter().map(|t| t.to_lowercase()).collect();
    ChunkCtx { imports, self_type, call_types_set, import_leaves, source_types_set, receiver_types }
}

/// Parallel-friendly resolve: returns (call_edges, type_edges, receiver_updates) instead of mutating AdjState.
#[expect(clippy::too_many_arguments, reason = "graph resolution needs all lookup tables")]
fn resolve_collect(
    src: usize,
    chunk: &ParsedChunk,
    ctx: &mut ChunkCtx,
    exact: &HashMap<String, u32>,
    short: &HashMap<String, u32>,
    kinds: &[String],
    chunk_names: &[String],
    return_type_map: &HashMap<String, String>,
    trait_methods: &BTreeSet<String>,
    method_owners: &HashMap<String, Vec<String>>,
    instantiated: &HashSet<String>,
    trait_impl_methods: &HashMap<String, Vec<String>>,
    rustdoc: Option<&RustdocTypes>,
    skip_calls: &HashSet<String>,
    enum_types: &HashSet<String>,
    enum_variants: &HashSet<String>,
    extern_all_methods: &HashSet<String>,
) -> (Vec<(u32, u32)>, Vec<(u32, u32)>, HashMap<String, String>, Vec<bool>) {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut receiver_updates: HashMap<String, String> = HashMap::new();
    let mut resolved_set = vec![false; chunk.calls.len()];
    for pass in 0..2 {
        for (call_idx, call) in chunk.calls.iter().enumerate() {
            if resolved_set[call_idx] {
                continue;
            }
            if !skip_calls.is_empty() {
                let call_short = call.to_lowercase();
                let call_leaf = call_short.rsplit("::").next().unwrap_or(&call_short);
                if skip_calls.contains(call_leaf) {
                    resolved_set[call_idx] = true;
                    continue;
                }
            }
            if let Some(tgt) = resolve_with_imports(
                call, exact, short, &ctx.imports,
                ctx.self_type.as_deref(), &ctx.source_types_set, &ctx.call_types_set,
                &ctx.receiver_types, trait_methods, method_owners, instantiated, trait_impl_methods, rustdoc, kinds, chunk_names, enum_types, enum_variants, return_type_map, &ctx.import_leaves, extern_all_methods,
            ) {
                let call_line = chunk.call_lines.get(call_idx).copied().unwrap_or(0);
                let tgt_usize = tgt as usize;
                if tgt_usize != src {
                    edges.push((tgt, call_line));
                }
                trait_fan_out(call, tgt, src, call_line, trait_impl_methods, exact, extern_all_methods, &mut edges);
                resolved_set[call_idx] = true;

                if pass == 0 {
                    let call_lower = call.to_lowercase();
                    if let Some(ret_type) = return_type_map.get(&call_lower).or_else(|| {
                        let (recv, method) = call_lower.rsplit_once('.')?;
                        let recv_leaf = recv.rsplit_once('.').map_or(recv, |p| p.1);
                        let recv_type = ctx.receiver_types.get(recv_leaf)?;
                        let qualified = format!("{recv_type}::{method}");
                        return_type_map.get(&qualified)
                    }) {
                        for (var_name, callee_name) in &chunk.let_call_bindings {
                            if callee_name.to_lowercase() == call_lower {
                                let key = var_name.to_lowercase();
                                receiver_updates.entry(key.clone()).or_insert_with(|| ret_type.clone());
                                ctx.receiver_types
                                    .entry(key)
                                    .or_insert_with(|| ret_type.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    // Type ref edges.
    let mut type_edges: Vec<(u32, u32)> = Vec::new();
    for ty in &chunk.types {
        let lower = ty.to_lowercase();
        let tgt = if let Some(qualified) = ctx.imports.get(&lower) {
            exact.get(qualified).copied()
        } else {
            None
        };
        let tgt = tgt.or_else(|| exact.get(&lower).copied());
        let tgt = tgt.or_else(|| short.get(&lower).copied());
        if let Some(tgt) = tgt {
            type_edges.push((tgt, 0));
        }
    }
    (edges, type_edges, receiver_updates, resolved_set)
}

/// Read-only delta resolve for iterative passes (no clone, no receiver_type updates).
#[expect(clippy::too_many_arguments, reason = "graph resolution needs all lookup tables")]
fn resolve_delta(
    src: usize,
    chunk: &ParsedChunk,
    ctx: &ChunkCtx,
    exact: &HashMap<String, u32>,
    short: &HashMap<String, u32>,
    return_type_map: &HashMap<String, String>,
    trait_methods: &BTreeSet<String>,
    method_owners: &HashMap<String, Vec<String>>,
    instantiated: &HashSet<String>,
    trait_impl_methods: &HashMap<String, Vec<String>>,
    rustdoc: Option<&RustdocTypes>,
    resolved_indices: &[bool],
    enum_types: &HashSet<String>,
    enum_variants: &HashSet<String>,
    kinds: &[String],
    chunk_names: &[String],
    extern_all_methods: &HashSet<String>,
) -> Vec<(u32, u32)> {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for (call_idx, call) in chunk.calls.iter().enumerate() {
        // Skip calls already resolved in pass 1.
        if call_idx < resolved_indices.len() && resolved_indices[call_idx] {
            continue;
        }
        if let Some(tgt) = resolve_with_imports(
            call, exact, short, &ctx.imports,
            ctx.self_type.as_deref(), &ctx.source_types_set, &ctx.call_types_set,
            &ctx.receiver_types, trait_methods, method_owners, instantiated, trait_impl_methods, rustdoc, kinds, chunk_names, enum_types, enum_variants, return_type_map, &ctx.import_leaves, extern_all_methods,
        ) {
            let tgt_usize = tgt as usize;
            let call_line = chunk.call_lines.get(call_idx).copied().unwrap_or(0);
            if tgt_usize != src {
                edges.push((tgt, call_line));
            }
            trait_fan_out(call, tgt, src, call_line, trait_impl_methods, exact, extern_all_methods, &mut edges);
        }
    }
    edges
}

/// Add edges to all concrete trait impls for project-specific trait methods.
///
/// When a call resolves to one impl of a trait method, also add edges to
/// all other impls. Skips std/dep trait methods (Default, Clone, etc.)
/// to avoid edge explosion.
fn trait_fan_out(
    call: &str,
    resolved_tgt: u32,
    src: usize,
    call_line: u32,
    trait_impl_methods: &HashMap<String, Vec<String>>,
    exact: &HashMap<String, u32>,
    extern_all_methods: &HashSet<String>,
    edges: &mut Vec<(u32, u32)>,
) {
    let call_lower = call.to_lowercase();
    // Only fan out for receiver.method() calls (dynamic dispatch).
    // Type::method() calls have an explicit type — fan out would add
    // every trait impl as a callee (e.g. SparseBlockCodec::serialize
    // would add all BinarySerializable::serialize impls).
    // Bare function calls (no receiver, no ::) also skip — the resolved
    // target is already the correct function.
    let Some(method_leaf) = call_lower.rsplit_once('.').map(|p| p.1) else {
        return;
    };
    if extern_all_methods.contains(method_leaf) {
        return;
    }
    let Some(impl_types) = trait_impl_methods.get(method_leaf) else { return };
    if impl_types.len() <= 1 {
        return;
    }
    for impl_type in impl_types {
        let qualified = format!("{impl_type}::{method_leaf}");
        if let Some(&alt_idx) = exact.get(&qualified) {
            if alt_idx != resolved_tgt && (alt_idx as usize) != src {
                edges.push((alt_idx, call_line));
            }
        }
    }
}

fn resolve_with_imports(
    call: &str,
    exact: &HashMap<String, u32>,
    short: &HashMap<String, u32>,
    imports: &HashMap<String, String>,
    self_type: Option<&str>,
    source_types: &HashSet<String>,
    call_types: &HashSet<String>,
    receiver_types: &HashMap<String, String>,
    _trait_methods: &BTreeSet<String>,
    method_owners: &HashMap<String, Vec<String>>,
    instantiated: &HashSet<String>,
    trait_impl_methods: &HashMap<String, Vec<String>>,
    rustdoc: Option<&RustdocTypes>,
    kinds: &[String],
    chunk_names: &[String],
    _enum_types: &HashSet<String>,
    enum_variants: &HashSet<String>,
    return_type_map: &HashMap<String, String>,
    _import_leaves: &HashSet<String>,
    extern_all_methods: &HashSet<String>,
) -> Option<u32> {
    // Normalize UFCS / turbofish syntax before resolution:
    //   `<Foo>::func`         → `Foo::func`
    //   `<Foo as Trait>::func` → `Foo::func`
    let call_owned;
    let call: &str = if let Some(rest) = call.strip_prefix('<') {
        if let Some((inner, suffix)) = rest.split_once(">::") {
            let type_part = inner.split_once(" as ").map_or(inner, |p| p.0);
            call_owned = format!("{type_part}::{suffix}");
            &call_owned
        } else {
            call
        }
    } else {
        call
    };
    let lower = call.to_lowercase();

    // Guard: fallback lookups must not resolve to enum variant chunks.
    // Enum variants like `Expression::Sum` should not match `.sum()` method calls.
    // Also check if the target chunk's method name starts with uppercase (variant constructor).
    let names = chunk_names;
    let is_callable = |idx: u32| -> bool {
        let kind = kinds.get(idx as usize).map(|s| s.as_str()).unwrap_or("");
        if kind == "enum" || kind == "struct" || kind == "trait" {
            return false;
        }
        // Skip function chunks whose method name starts with uppercase (enum variant constructors
        // from Python .pyi, TypeScript, etc.)
        if let Some(name) = names.get(idx as usize) {
            if let Some(method) = name.rsplit("::").next() {
                if method.starts_with(|c: char| c.is_uppercase()) {
                    return false;
                }
            }
        }
        true
    };
    let short_fn = |key: &str| -> Option<u32> {
        let idx = short.get(key).copied()?;
        if is_callable(idx) { Some(idx) } else { None }
    };
    let _exact_fn = |key: &str| -> Option<u32> {
        let idx = exact.get(key).copied()?;
        if is_callable(idx) { Some(idx) } else { None }
    };

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

    // 1b. Self::method → OwningType::method (Rust associated function syntax).
    if let Some(method) = lower.strip_prefix("self::") {
        if let Some(owner) = self_type {
            let qualified = format!("{owner}::{method}");
            if let Some(&idx) = exact.get(&qualified) {
                return Some(idx);
            }
        }
    }

    // 1c. super::func → strip prefix, resolve via short lookup.
    //     super:: refers to parent module — the function should be findable by short name.
    if let Some(rest) = lower.strip_prefix("super::") {
        let leaf = rest.rsplit_once("::").map_or(rest, |p| p.1);
        if let Some(idx) = short_fn(leaf) {
            return Some(idx);
        }
    }

    // 2. self.method → OwningType::method.
    //    self.field.method → try field name as type hint against source_types.
    if let Some(method) = lower.strip_prefix("self.") {
        let leaf_method = method.rsplit_once('.').map_or(method, |p| p.1);
        let is_field_chain = method.contains('.');
        // 2a. Try owning type first — only for direct self.method() calls,
        //     not self.field.method() chains where the receiver is the field.
        if !is_field_chain {
            if let Some(owner) = self_type {
                let qualified = format!("{owner}::{leaf_method}");
                if let Some(&idx) = exact.get(&qualified) {
                    return Some(idx);
                }
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
                // Field type exists in project → try qualified lookup only.
                // short_fn fallback would pick an arbitrary chunk (e.g. DeltaNeighbors::push
                // for Vec::push) — never use short_fn for receiver.method() calls.
                // Field type is external → skip to avoid false positive.
                // But if field type is a likely generic param (single lowercase letter
                // like "t", "u", "k", "v"), fall through to heuristic matching.
                if !(field_type.len() == 1 && field_type.as_bytes()[0].is_ascii_lowercase()) {
                    return None;
                }
            }
            // Fallback: field name as type hint via underscore-stripped matching.
            // e.g. self.token_tree.method() → "token_tree" → "tokentree" ↔ "TokenTree"
            let field_stripped: String = field_leaf.chars().filter(|&c| c != '_').collect();
            if let Some(owners) = method_owners.get(leaf_method) {
                if let Some(owner) = owners.iter().find(|o| {
                    o.contains(&field_stripped) || field_stripped.contains(o.as_str())
                }) {
                    let candidate = format!("{owner}::{leaf_method}");
                    if let Some(&idx) = exact.get(&candidate) {
                        return Some(idx);
                    }
                }
            }
            if let Some(impl_types) = trait_impl_methods.get(leaf_method) {
                if let Some(t) = impl_types.iter().find(|t| {
                    t.contains(&field_stripped) || field_stripped.contains(t.as_str())
                }) {
                    let candidate = format!("{t}::{leaf_method}");
                    if let Some(&idx) = exact.get(&candidate) {
                        return Some(idx);
                    }
                }
            }
            // (removed) type_refs heuristic for self.field.method — too aggressive.
            // Field type must be known (from struct definition or receiver_types).

            // Try field_leaf directly as a type name via imports
            if let Some(qualified) = imports.get(field_leaf) {
                let candidate = format!("{qualified}::{leaf_method}");
                if let Some(&idx) = exact.get(&candidate) {
                    return Some(idx);
                }
            }
            // Try method_owners for self.field.method — RTA-filtered unique resolve
            // Only resolve if field name is similar to the owner type.
            // Prevents `self.callees.push()` → `DeltaNeighbors::push`.
            if let Some(owners) = method_owners.get(leaf_method) {
                let rta: Vec<&String> = if !instantiated.is_empty() {
                    let f: Vec<&String> = owners.iter().filter(|o| instantiated.contains(o.as_str())).collect();
                    if f.is_empty() { owners.iter().collect() } else { f }
                } else {
                    owners.iter().collect()
                };
                if rta.len() == 1 {
                    let owner = &rta[0];
                    let field_match = field_stripped.len() >= 3
                        && (owner.contains(field_leaf) || field_leaf.contains(owner.as_str())
                            || owner.contains(&field_stripped) || field_stripped.contains(owner.as_str()));
                    if field_match {
                        let qualified = format!("{owner}::{leaf_method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
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

    // 5. (removed) Type-reference heuristic was too aggressive — connecting
    //    `receiver.method()` to any type mentioned in the same function caused
    //    false positives (e.g. `items.push()` → `DeltaNeighbors::push`).
    //    Receiver type evidence is now required (steps 6b-6d).

    // 5a. Return-type chain: receiver is a function name whose return type is known.
    //     e.g. `type_of_expr.adjusted` where return_type_map["type_of_expr"] = "typeinfo"
    //     → try exact["typeinfo::adjusted"]
    //     Only trust return_type_map when the method actually belongs to the inferred type
    //     (prevents `map.collect()` → `chunkmeta::collect` via return_type_map["map"]).
    if let Some((receiver, method)) = lower.rsplit_once('.') {
        let receiver_leaf = receiver.rsplit_once('.').map_or(receiver, |p| p.1);
        if let Some(ret_type) = return_type_map.get(receiver_leaf) {
            // Only trust return_type_map when receiver is a known project entity,
            // not just a bare method name that happens to match (e.g. "map", "filter").
            // For chained builder calls (e.g. `ef_construction.max_elements`), the receiver
            // IS a unique project method name registered in return_type_map as a bare key.
            let receiver_is_known = exact.contains_key(receiver_leaf)
                || imports.contains_key(receiver_leaf)
                || return_type_map.contains_key(receiver_leaf);
            if receiver_is_known {
                let candidate = format!("{ret_type}::{method}");
                if let Some(&idx) = exact.get(&candidate) {
                    // Verify: method must belong to ret_type (or method_owners unknown)
                    let valid = method_owners.get(method).map_or(true, |owners| {
                        owners.iter().any(|o| *o == *ret_type
                            || o.split(" for ").last().map_or(false, |c| c == *ret_type))
                    });
                    if valid {
                        return Some(idx);
                    }
                }
            }
        }
    }

    // 6. Short name fallback.
    //    `Foo::bar` → try short match on "bar" (qualified = likely a type).
    //    `receiver.method` → only if receiver is a known type (in imports or exact map).
    //    Bare names → skip (too ambiguous).
    if let Some((prefix, suffix)) = lower.rsplit_once("::") {
        // Skip enum variant constructors: `CliError::Embed(...)` is not a function call.
        // Only skip when original suffix starts with uppercase — lowercase suffix
        // like `Expr::cast()` or `Expr::from()` is a method call, not a variant.
        let prefix_leaf = prefix.rsplit("::").next().unwrap_or(prefix);
        let orig_suffix = call.rsplit_once("::").map_or(call, |p| p.1);
        if orig_suffix.starts_with(char::is_uppercase) {
            // Uppercase suffix = likely enum variant constructor, not a method call.
            // Skip entirely: both project enums and external enum variants.
            return None;
        }
        // Try exact match with type leaf: `ast::Expr::cast` → exact["expr::cast"]
        // (populated by trait impl alias: "Trait for Type::method" → "type::method").
        let qualified = format!("{prefix_leaf}::{suffix}");
        if let Some(&idx) = exact.get(&qualified) {
            return Some(idx);
        }
        // Only use short fallback if the type prefix is a known project type
        // (exists in exact map = has a chunk definition in the project).
        // Prevents `Vec::new()` → project's `Scanner::new` false positive.
        if exact.contains_key(prefix_leaf) {
            // Validate that short_fn result's owner matches prefix_leaf.
            // Prevents `CCodeChunker::new` → `Scanner::new` when macro-generated
            // method is not in exact map but short_fn("new") returns first-registered.
            if let Some(idx) = short_fn(suffix) {
                let resolved_name = &chunk_names[idx as usize];
                let resolved_owner = resolved_name.rsplit_once("::")
                    .map(|(p, _)| {
                        let leaf = p.rsplit("::").next().unwrap_or(p);
                        let after_for = leaf.split(" for ").last().unwrap_or(leaf);
                        after_for.split('<').next().unwrap_or(after_for)
                    })
                    .unwrap_or("");
                if resolved_owner == prefix_leaf
                    || resolved_owner.contains(prefix_leaf)
                    || prefix_leaf.contains(resolved_owner) {
                    return Some(idx);
                }
            }
        }
        return None;
    }
    if let Some((receiver, method)) = lower.rsplit_once('.') {
        let receiver_leaf = receiver.rsplit_once('.').map_or(receiver, |p| p.1);
        let recv_stripped: String = receiver_leaf.chars().filter(|&c| c != '_').collect();
        // 6a. Name-based receiver matching (receiver name matches a type/import).
        // Skip this when receiver_types has a known type — prefer the precise
        // type from let bindings / param types (6b) over name similarity.
        // Prevents `let tokenizer = WhitespaceTokenizer::new(); tokenizer.tokenize()`
        // from matching `KoreanBm25Tokenizer` because "tokenizer" contains "tokenizer".
        if !receiver_types.contains_key(receiver_leaf)
            && (imports.contains_key(receiver_leaf) || exact.contains_key(receiver_leaf)) {
            // If method has known owners, verify receiver matches one of them.
            // Prevents `scanner.new()` → `CallGraph::new` when receiver "scanner"
            // is a known import but not related to the method's owner.
            if let Some(owners) = method_owners.get(method) {
                // Find the best matching owner for the receiver.
                let matched_owner = owners.iter().find(|o| {
                    // Strip "trait for " prefix to match concrete type against receiver.
                    let concrete = o.split(" for ").last().unwrap_or(o);
                    concrete.contains(receiver_leaf) || receiver_leaf.contains(concrete)
                    || concrete.contains(recv_stripped.as_str()) || recv_stripped.contains(concrete)
                });
                if let Some(owner) = matched_owner {
                    // Try exact lookup with concrete type (strip "trait for " prefix).
                    let concrete = owner.split(" for ").last().unwrap_or(owner);
                    let qualified = format!("{concrete}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                    // Also try full owner key (for trait impl chunks).
                    let qualified_full = format!("{owner}::{method}");
                    if let Some(&idx) = exact.get(&qualified_full) {
                        return Some(idx);
                    }
                    // Owner matched but not in exact — don't fall to short_fn
                    // which would pick an arbitrary chunk.
                }
                // No owner matches receiver — fall through to 6b+ for more context.
            } else {
                // method_owners has no entry — may be a trait method.
                // Check trait_impl_methods for receiver-specific dispatch first.
                // Prevents `l2distance.distance()` → `MockL2::distance` via short_fn.
                if let Some(impl_types) = trait_impl_methods.get(method) {
                    let matched = impl_types.iter().find(|t| {
                        t.contains(receiver_leaf) || receiver_leaf.contains(t.as_str())
                        || t.contains(recv_stripped.as_str()) || recv_stripped.contains(t.as_str())
                    });
                    if let Some(t) = matched {
                        let qualified = format!("{t}::{method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    }
                    // No receiver match in trait impls — don't fall to short_fn
                    // as it would pick an arbitrary impl.
                    // Fall through to 6b+ for more context regardless of count.
                } else {
                    // No method_owners, no trait_impl_methods — no project evidence.
                    // Fall through to 6b+ rather than short_fn blind fallback.
                }
            }
        }
        // 6b. Type-aware receiver check: if we know the receiver's type from
        // param_types/local_types/field_types, only match if that type is a
        // project-internal type (exists in exact or short map).
        if let Some(recv_type) = receiver_types.get(receiver_leaf) {
            // Extern-voted receiver: all methods on this receiver are extern-only.
            if recv_type == "<extern>" {
                return None;
            }
            let ty_lower = recv_type.to_lowercase();
            // Validate: the method must actually belong to this type.
            // Prevents `map.collect()` → `ChunkMeta::collect` when `map` was
            // wrongly inferred as `chunkmeta` via return_type_map["map"].
            // Check if this type actually owns the method.
            // Also allow trait types (trait dispatch handles resolution).
            let recv_is_trait = exact.get(&ty_lower)
                .and_then(|&idx| kinds.get(idx as usize))
                .map_or(false, |k| k == "trait");
            let type_has_method = recv_is_trait
                || method_owners.get(method)
                    .map_or(true, |owners| owners.iter().any(|o| {
                        *o == ty_lower
                        // Handle trait impl owners: "runnable for task" contains "task"
                        || o.split(" for ").last().map_or(false, |concrete| concrete == ty_lower)
                    }));
            if type_has_method {
                // Type exists in project and owns this method → resolve via Type::method
                let qualified = format!("{ty_lower}::{method}");
                if let Some(&idx) = exact.get(&qualified) {
                    return Some(idx);
                }
                // Trait dispatch: receiver is a trait → look up concrete impl
                if let Some(impl_types) = trait_impl_methods.get(method) {
                    if impl_types.len() == 1 {
                        let qualified = format!("{}::{}", impl_types[0], method);
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    } else {
                        let matched = impl_types.iter().find(|t| {
                            source_types.contains(t.as_str()) || call_types.contains(t.as_str())
                        }).or_else(|| impl_types.iter().find(|t| {
                            t.contains(receiver_leaf) || receiver_leaf.contains(t.as_str())
                            || t.contains(recv_stripped.as_str()) || recv_stripped.contains(t.as_str())
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
                let ty_stripped: String = ty_lower.chars().filter(|&c| c != '_').collect();
                if let Some(owners) = method_owners.get(method)
                    && let Some(owner) = owners.iter().find(|o| {
                        **o == ty_lower || **o == ty_stripped
                        || o.contains(ty_stripped.as_str()) || ty_stripped.contains(o.as_str())
                    }) {
                        let qualified = format!("{owner}::{method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    }
                // Exact lookup with recv_type as owner — no short_fn fallback.
                // short_fn would pick an arbitrary chunk, ignoring the known type.
                if let Some(owners) = method_owners.get(method) {
                    let matched = owners.iter().find(|o| **o == ty_lower || **o == ty_stripped);
                    if let Some(owner) = matched {
                        let qualified = format!("{owner}::{method}");
                        if let Some(&idx) = exact.get(&qualified) {
                            return Some(idx);
                        }
                    }
                }
                // Trait receiver fallback: if receiver is a trait type and all
                // qualified lookups failed (common with salsa/macro-generated DB
                // methods), try short_fn. The method is likely a bare function
                // generated by a proc macro that belongs to this trait.
                if recv_is_trait {
                    if let Some(idx) = short_fn(method) {
                        return Some(idx);
                    }
                }
            }
            // recv_type is known but doesn't own this method.
            // If the type is not a project type (no struct/enum/trait chunk)
            // AND the method exists in extern (std/deps), this is an extern call.
            // e.g. receiver_types["all_files"] = "vec" → vec is not a project type
            // → all_files.is_empty() should be extern, not StorageEngine::is_empty.
            if !exact.contains_key(&ty_lower) && extern_all_methods.contains(method) {
                return None;
            }
            // Otherwise fall through to 6c/6d for disambiguation.
        }
        // 6c. (disabled for receiver.method() calls)
        // Type context alone is insufficient evidence for receiver.method() disambiguation.
        // e.g. lut.tf_norm() should not resolve to Bm25Params::tf_norm just because
        // Bm25Params appears in the function's type context.
        // Resolution for receiver.method() relies on receiver_types (6b) and
        // method_owners with receiver name matching (6d).
        // 6d. Project-internal method→owner disambiguation.
        //     method_name → [owner_types] reverse index from all project functions.
        //     Resolution priority:
        //     (1) Unique owner → resolve immediately
        //     (2) Receiver name contains owner type → resolve (e.g. engine ~ storageengine)
        //         Also underscore-stripped: "bin_expr" → "binexpr" ~ "BinExpr"
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
            // Minimum length for name similarity — 1-2 char names like "w", "db"
            // match too many types ("w" ⊂ "widget", "db" ⊂ "database").
            let min_sim_len = 3;
            // If method also exists on extern types (std/deps), require stronger
            // evidence — receiver name must match owner. Prevents `.len()`, `.push()`
            // on Vec/HashMap from resolving to a project type with the same method.
            let method_is_extern = extern_all_methods.contains(method);
            if rta_filtered.len() == 1 {
                let owner = &rta_filtered[0];
                let owner_match = recv_stripped.len() >= min_sim_len
                    && (owner.contains(receiver_leaf)
                    || receiver_leaf.contains(owner.as_str())
                    || owner.contains(&recv_stripped)
                    || recv_stripped.contains(owner.as_str()));
                if owner_match {
                    let qualified = format!("{owner}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                }
                // Unique project owner but no receiver match — if method is also extern,
                // it's likely a std method call. Skip entirely.
                if method_is_extern {
                    return None;
                }
            } else if owners.len() > 1 {
                // (2) Receiver name similarity — "engine" matches "storageengine"
                //     Also underscore-stripped: "bin_expr" → "binexpr" ~ "binexpr"
                let matched_owner = if recv_stripped.len() >= min_sim_len {
                    owners.iter().find(|o| {
                        o.contains(receiver_leaf) || receiver_leaf.contains(o.as_str())
                        || o.contains(&recv_stripped) || recv_stripped.contains(o.as_str())
                    })
                } else {
                    None
                };
                if let Some(owner) = matched_owner {
                    let qualified = format!("{owner}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
                    }
                }
                // (3) Self type — method on the same struct we're in
                // Only resolve if receiver name is related to self_type.
                // Prevents `chunks.len()` → `CallGraph::len` inside CallGraph methods.
                if let Some(st) = self_type
                    && owners.contains(&st.to_owned()) {
                        let self_match = recv_stripped.len() >= 3
                            && (st.contains(receiver_leaf) || receiver_leaf.contains(st)
                                || st.contains(recv_stripped.as_str()) || recv_stripped.contains(st));
                        if self_match {
                            let qualified = format!("{st}::{method}");
                            if let Some(&idx) = exact.get(&qualified) {
                                return Some(idx);
                            }
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
                // (5) (removed) Type context fallback was too weak — just because
                //     a type is mentioned in the same function doesn't mean the
                //     receiver is that type. Causes false positives.
            }
        }
        // 6e. Trait impl method fallback: only resolve if receiver name
        //      matches the impl type (evidence of receiver identity).
        if method_owners.get(method).is_none()
            && let Some(impl_types) = trait_impl_methods.get(method) {
                let matched = impl_types.iter().find(|t| {
                    t.contains(receiver_leaf) || receiver_leaf.contains(t.as_str())
                    || t.contains(recv_stripped.as_str()) || recv_stripped.contains(t.as_str())
                });
                if let Some(t) = matched {
                    let qualified = format!("{t}::{method}");
                    if let Some(&idx) = exact.get(&qualified) {
                        return Some(idx);
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

/// Collect type_refs from struct/impl chunks keyed by owning type name (lowercase).
///
/// For `impl SimpleHybridSearcher<D, T>` with type_refs `[Bm25Index, HnswGraph, ...]`,
/// produces `{"simplehybridsearcher": ["Bm25Index", "HnswGraph", ...]}`.
/// Methods of that type can then use these types to resolve `self.field.method()` calls.
fn collect_owner_types(chunks: &[ParsedChunk], _meta: &ChunkMeta) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
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
pub fn collect_owner_field_types(chunks: &[ParsedChunk]) -> HashMap<String, HashMap<String, String>> {
    let mut result: HashMap<String, HashMap<String, String>> = HashMap::new();
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

/// Collect project type short names (struct, enum, trait, impl) for extern classification.
pub fn collect_project_type_shorts(chunks: &[ParsedChunk]) -> HashSet<String> {
    let mut set = HashSet::new();
    for c in chunks {
        if matches!(c.kind.as_str(), "struct" | "enum" | "trait" | "impl") {
            let leaf = c.name.rsplit("::").next().unwrap_or(&c.name).to_lowercase();
            let clean = leaf.split('<').next().unwrap_or(&leaf);
            set.insert(clean.to_owned());
        }
    }
    set
}

/// Build function name → return type map (lowercase → lowercase leaf type).
///
/// Resolves `Self` to the owning type: `Foo::new → Self` becomes `foo::new → foo`.
pub fn build_return_type_map(chunks: &[ParsedChunk]) -> HashMap<String, String> {
    // Collect all project type names (structs, enums, traits, etc.)
    let project_types: std::collections::HashSet<String> = chunks
        .iter()
        .filter(|c| matches!(c.kind.as_str(), "struct" | "enum" | "trait" | "class" | "interface"))
        .map(|c| {
            let short = c.name.rsplit("::").next().unwrap_or(&c.name);
            short.to_lowercase()
        })
        .collect();

    let mut map = HashMap::new();
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
        // Also insert bare method name for chained-call resolution,
        // but ONLY if the method name is unique across all project functions.
        // Common names like "collect", "map", "len" would cause false positives
        // by polluting return_type_map with wrong type associations.
        // Deferred to a second pass below.

        map.insert(name_lower, resolved);
    }
    // Second pass: insert bare method names only if they're unique in the project.
    // Count how many times each bare method name appears across all functions.
    let mut method_count: HashMap<String, u32> = HashMap::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        if let Some(method) = c.name.rsplit_once("::").map(|p| p.1) {
            *method_count.entry(method.to_lowercase()).or_default() += 1;
        }
    }
    for c in chunks {
        if c.kind != "function" || c.return_type.is_none() {
            continue;
        }
        let name_lower = c.name.to_lowercase();
        if let Some(method) = name_lower.rsplit_once("::").map(|p| p.1) {
            if method_count.get(method).copied().unwrap_or(0) == 1 {
                let resolved = map.get(&name_lower).cloned();
                if let Some(resolved) = resolved {
                    map.entry(method.to_owned()).or_insert(resolved);
                }
            }
        }
    }
    map
}

/// Build a reverse index: method_name → [owner_type1, owner_type2, ...].
///
/// Scans all `Type::method` function chunks and groups by method short name.
/// Used for resolving `receiver.method()` when receiver type is unknown.
pub fn build_method_owner_index(chunks: &[ParsedChunk]) -> HashMap<String, Vec<String>> {
    let mut index: HashMap<String, Vec<String>> = HashMap::new();
    for c in chunks {
        if c.kind != "function" {
            continue;
        }
        // Only methods (Type::method pattern)
        let Some((prefix, method)) = c.name.rsplit_once("::") else { continue };
        // Skip enum variant constructors: `Expression::Sum`, `Error::NotFound`
        // These start with uppercase and should not match `.sum()`, `.not_found()` calls.
        if method.starts_with(|c: char| c.is_uppercase()) {
            continue;
        }
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
) -> HashMap<String, Vec<String>> {
    // Step 1: Find all "impl Trait for Type" chunks → (trait_name, concrete_type)
    let mut trait_to_types: HashMap<String, Vec<String>> = HashMap::new();
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
    let mut method_map: HashMap<String, Vec<String>> = HashMap::new();
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
pub(crate) fn strip_generics_from_key(key: &str) -> String {
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

/// Parse generic trait bounds from a function signature string.
///
/// E.g. `"fn foo<N: NodeGraph, T: Tokenizer + Clone>(nodes: &N)"` →
///       `{"n": "nodegraph", "t": "tokenizer"}` (first trait bound, lowercase).
fn parse_generic_trait_bounds(signature: &str) -> HashMap<String, String> {
    let mut bounds = HashMap::new();
    // Extract the `<...>` section from the signature.
    let Some(start) = signature.find('<') else { return bounds };
    let mut depth = 0;
    let mut end = start;
    for (i, ch) in signature[start..].char_indices() {
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
    if end <= start + 1 { return bounds; }
    let generics = &signature[start + 1..end];

    // Split by ',' but respect nested <> depth.
    let mut params = Vec::new();
    let mut current_start = 0;
    let mut nest = 0;
    for (i, ch) in generics.char_indices() {
        match ch {
            '<' => nest += 1,
            '>' => nest -= 1,
            ',' if nest == 0 => {
                params.push(generics[current_start..i].trim());
                current_start = i + 1;
            }
            _ => {}
        }
    }
    params.push(generics[current_start..].trim());

    // Parse each param: "N: NodeGraph + Clone" → ("n", "nodegraph")
    // Also handle where-clause style: just "N" with no bound → skip.
    for param in params {
        // Skip lifetime params ('a, 'b)
        if param.starts_with('\'') { continue; }
        // Skip const generics (const N: usize)
        if param.starts_with("const ") { continue; }
        let Some((name, bound_str)) = param.split_once(':') else { continue };
        let name = name.trim().to_lowercase();
        // Take the first trait bound (before any '+').
        let first_bound = bound_str.split('+').next().unwrap_or("").trim();
        // Strip generic params from bound: Trait<Item=Foo> → Trait
        let first_bound = first_bound.split('<').next().unwrap_or(first_bound).trim();
        if !first_bound.is_empty() && first_bound != "?" {
            bounds.insert(name, first_bound.to_lowercase());
        }
    }
    bounds
}

/// Build a receiver name → type map from param_types, local_types, field_types.
///
/// E.g. `param_types = [("handle", "File")]` → `{"handle": "file"}` (lowercase).
/// For `self.field` lookups, field_types are keyed as-is.
/// Build a receiver→type map from a chunk's param/local/field types.
///
/// Populates variable names → leaf type names for method resolution.
pub fn build_receiver_type_map(chunk: &ParsedChunk) -> HashMap<String, String> {
    let mut map = HashMap::new();
    // Parse generic trait bounds from signature for substitution.
    let generic_bounds = chunk.signature.as_deref()
        .map(|s| parse_generic_trait_bounds(s))
        .unwrap_or_default();
    for (name, ty) in &chunk.param_types {
        let name_lower = name.to_lowercase();
        let leaf = extract_leaf_type(&ty.to_lowercase()).to_owned();
        if name_lower != "self" && !leaf.is_empty() {
            // If the leaf type is a single-letter generic parameter (e.g. "n"),
            // substitute with the trait bound (e.g. "nodegraph").
            let resolved = if leaf.len() == 1 && leaf.as_bytes()[0].is_ascii_lowercase() {
                generic_bounds.get(&leaf).cloned().unwrap_or(leaf)
            } else {
                leaf
            };
            map.insert(name_lower, resolved);
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
pub fn infer_local_types_from_calls(
    calls: &[String],
    let_call_bindings: &[(String, String)],
    return_type_map: &HashMap<String, String>,
    receiver_types: &mut HashMap<String, String>,
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
        // Fallback: `let x = Type::new(...)` → infer x: type (constructor pattern).
        // Works even for macro-generated types not in return_type_map.
        if let Some((prefix, method)) = callee_name.rsplit_once("::") {
            let method_lower = method.to_lowercase();
            if matches!(method_lower.as_str(), "new" | "default" | "from" | "create" | "builder" | "open" | "init") {
                let leaf = prefix.rsplit("::").next().unwrap_or(prefix);
                receiver_types
                    .entry(var_name.to_lowercase())
                    .or_insert_with(|| leaf.to_lowercase());
                continue;
            }
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

/// Infer receiver types by analyzing method calls on each receiver.
///
/// Three strategies (applied in order):
/// 1. **Co-method intersection**: If `x.foo()` and `x.bar()` are both called,
///    and only `MyType` has both → `x: MyType`.
/// 2. **Single-method unique owner**: If `x.foo()` and `foo` has exactly one
///    project owner → `x: ThatOwner`.
/// 3. **Extern voting**: If ALL methods on receiver exist in NO project type
///    → receiver is extern, marked as `"<extern>"` to aid extern classification.
pub fn infer_receiver_types_by_co_methods(
    calls: &[String],
    method_owners: &HashMap<String, Vec<String>>,
    receiver_types: &mut HashMap<String, String>,
) {
    // Group method names by receiver.
    let mut receiver_methods: HashMap<String, Vec<String>> = HashMap::new();
    for call in calls {
        let Some(dot) = call.find('.') else { continue };
        let receiver = &call[..dot];
        let method = &call[dot + 1..];
        if method.is_empty() || receiver.is_empty() { continue; }
        let recv_lower = receiver.to_lowercase();
        if receiver_types.contains_key(&recv_lower) { continue; }
        // Only bare receiver names (no dots/colons).
        if receiver.contains('.') || receiver.contains(':') { continue; }
        receiver_methods.entry(recv_lower).or_default().push(method.to_lowercase());
    }

    for (recv, methods) in &receiver_methods {
        // Count how many methods are project-only vs extern-only.
        let mut project_methods: Vec<&str> = Vec::new();
        let mut extern_only_count = 0u32;
        for method in methods {
            if method_owners.contains_key(method.as_str()) {
                project_methods.push(method.as_str());
            } else {
                extern_only_count += 1;
            }
        }

        // Strategy 3: ALL methods are extern-only → receiver is extern type.
        if project_methods.is_empty() && extern_only_count > 0 {
            receiver_types.entry(recv.clone()).or_insert_with(|| "<extern>".to_owned());
            continue;
        }

        // Strategy 1: intersect owners of project methods (2+ methods).
        if project_methods.len() >= 2 {
            let mut candidates: Option<Vec<String>> = None;
            for method in &project_methods {
                let Some(owners) = method_owners.get(*method) else { continue };
                candidates = Some(match candidates {
                    None => owners.clone(),
                    Some(prev) => prev.into_iter().filter(|o| owners.contains(o)).collect(),
                });
            }
            if let Some(ref c) = candidates
                && c.len() == 1
            {
                // Require receiver-owner name similarity to avoid false inference.
                // e.g. `entry.push()` + `entry.contains()` → DeltaNeighbors is the
                // only project type with both, but "entry" ≠ "deltaneighbors".
                // These are actually Vec methods on a HashMap entry.
                let owner = &c[0];
                let recv_stripped: String = recv.chars().filter(|&c| c != '_').collect();
                let name_match = recv_stripped.len() >= 3
                    && (owner.contains(recv_stripped.as_str())
                        || recv_stripped.contains(owner.as_str()));
                if name_match {
                    receiver_types.entry(recv.clone()).or_insert_with(|| c[0].clone());
                    continue;
                }
            }
        }

        // Strategy 2: single project method with unique owner.
        // Only infer when there are NO extern methods on the same receiver —
        // mixed evidence (project + extern) suggests the receiver is an external
        // type that happens to share a method name with a project function.
        // e.g. iterator "map" calling .collect() → project_methods=["collect"],
        // extern_only_count=0 because "map" is the receiver, not a method.
        // Also require receiver name similarity to the inferred owner.
        if project_methods.len() == 1 && extern_only_count == 0 {
            if let Some(owners) = method_owners.get(project_methods[0]) {
                if owners.len() == 1 {
                    let owner = &owners[0];
                    let recv_stripped: String = recv.chars().filter(|&c| c != '_').collect();
                    let name_match = recv_stripped.len() >= 3
                        && (owner.contains(recv_stripped.as_str())
                            || recv_stripped.contains(owner.as_str()));
                    if name_match {
                        receiver_types.entry(recv.clone()).or_insert_with(|| owner.clone());
                    }
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
fn build_import_map(imports: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
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
    exact: &HashMap<String, u32>,
    short: &HashMap<String, u32>,
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
    let mut map: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
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
fn build_field_access_index(
    chunks: &[ParsedChunk],
    owner_field_types: &HashMap<String, HashMap<String, String>>,
) -> Vec<(String, Vec<u32>)> {
    let mut map: HashMap<String, Vec<u32>> = HashMap::new();

    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.field_accesses.is_empty() {
            continue;
        }

        let mut receiver_types: HashMap<String, String> = HashMap::new();

        if let Some(owner) = owning_type(&chunk.name) {
            receiver_types.insert("self".to_owned(), owner.clone());
            // Reuse pre-built owner_field_types (O(1) lookup, no inner loop).
            if let Some(fields) = owner_field_types.get(&owner) {
                for (fname, fty) in fields {
                    let key = format!("self.{fname}");
                    let leaf = extract_leaf_type(fty).to_owned();
                    receiver_types.entry(key).or_insert(leaf);
                }
            }
        }

        for (pname, pty) in &chunk.param_types {
            let leaf = extract_leaf_type(&pty.to_lowercase()).to_owned();
            if !leaf.is_empty() && !pname.eq_ignore_ascii_case("self") {
                receiver_types.entry(pname.to_lowercase()).or_insert(leaf);
            }
        }

        for (vname, vty) in &chunk.local_types {
            let leaf = extract_leaf_type(&vty.to_lowercase()).to_owned();
            if !leaf.is_empty() {
                receiver_types.entry(vname.to_lowercase()).or_insert(leaf);
            }
        }

        for (recv, field) in &chunk.field_accesses {
            let recv_lower = recv.to_lowercase();
            let field_lower = field.to_lowercase();

            if let Some(ty) = receiver_types.get(&recv_lower) {
                let key = format!("{ty}::{field_lower}");
                map.entry(key).or_default().push(idx as u32);
            }
        }
    }

    for list in map.values_mut() {
        list.sort_unstable();
        list.dedup();
    }

    let mut result: Vec<_> = map.into_iter().collect();
    result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Check if a call is to an extern (std/deps) function.
///
/// Unified logic for extern classification — used by both graph build and verify.
pub fn check_extern(
    call_lower: &str,
    receiver_types: &HashMap<String, String>,
    extern_index: &crate::extern_types::ExternMethodIndex,
    project_type_shorts: &HashSet<String>,
    return_type_map: &HashMap<String, String>,
    extern_all_methods: &HashSet<String>,
    owner_field_types: &HashMap<String, HashMap<String, String>>,
    project_fn_shorts: Option<&HashSet<String>>,
    method_owners: &HashMap<String, Vec<String>>,
) -> Option<String> {
    // Bare function (no receiver): `len`, `drop`, etc.
    if !call_lower.contains('.') && !call_lower.contains("::") {
        if extern_all_methods.contains(call_lower) {
            return Some(format!("bare-extern: {call_lower}"));
        }
        // Bare name that's also not a project function → extern.
        if let Some(fns) = project_fn_shorts {
            if !fns.contains(call_lower) {
                return Some(format!("bare-extern: {call_lower}"));
            }
        }
        return None;
    }

    // Qualified call: `Type::method` — if Type is not a project type, it's extern.
    if let Some((prefix, _method)) = call_lower.rsplit_once("::") {
        let type_leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
        let type_clean = type_leaf.split('<').next().unwrap_or(type_leaf);
        if !type_clean.is_empty() && !project_type_shorts.contains(type_clean) {
            return Some(format!("qualified-extern: {call_lower}"));
        }
    }

    let (receiver, method) = call_lower.rsplit_once('.')?;
    let receiver_leaf = receiver.rsplit_once('.').map_or(receiver, |p| p.1);

    // self.field.method → field type lookup
    if receiver.starts_with("self.") {
        let field = receiver.strip_prefix("self.").unwrap_or(receiver);
        let field_leaf = field.rsplit_once('.').map_or(field, |p| p.1);
        // Try receiver_types (param/local types) first
        if let Some(field_type) = receiver_types.get(field_leaf) {
            let lowered = field_type.to_lowercase();
            let leaf = extract_leaf_type(&lowered);
            if extern_index.has_method(leaf, method) {
                return Some(format!("self.field: {field_leaf}:{leaf}.{method}"));
            }
        }
        // Try owner_field_types (struct field definitions)
        for fields in owner_field_types.values() {
            if let Some(field_type) = fields.get(field_leaf) {
                let lowered = field_type.to_lowercase();
                let leaf = extract_leaf_type(&lowered);
                if extern_index.has_method(leaf, method) {
                    return Some(format!("self.field: {field_leaf}:{leaf}.{method}"));
                }
            }
        }
    }

    // Direct receiver lookup (param, local, inferred types).
    if let Some(recv_type) = receiver_types.get(receiver_leaf) {
        // Co-method extern voting: all methods on this receiver are extern-only.
        if recv_type == "<extern>" {
            return Some(format!("co-extern: {receiver_leaf}.{method}"));
        }
        let lowered = recv_type.to_lowercase();
        let leaf = extract_leaf_type(&lowered);
        if extern_index.has_method(leaf, method) {
            return Some(format!("receiver: {receiver_leaf}:{leaf}.{method}"));
        }
    }

    // Return-type chain: receiver_leaf is a method whose return type is known.
    if let Some(ret_type) = return_type_map.get(receiver_leaf) {
        if extern_index.has_method(ret_type, method) {
            return Some(format!("return-chain: {receiver_leaf}->{ret_type}.{method}"));
        }
    }

    // Fallback: method exists in extern types AND receiver is not a project type.
    if extern_all_methods.contains(method)
        && !project_type_shorts.contains(receiver_leaf)
    {
        return Some(format!("untyped-extern: {receiver_leaf}.{method}"));
    }

    // Receiver is an extern method name (iterator chain: `filter_map.collect`,
    // `into_iter.partition`). The receiver itself is a chained call result from
    // an extern method, so the outer call is also extern.
    if extern_all_methods.contains(receiver_leaf)
        && !project_type_shorts.contains(receiver_leaf)
    {
        return Some(format!("chain-extern: {receiver_leaf}.{method}"));
    }

    // Method not defined in any project type AND receiver is not a project type
    // → must be extern (e.g. `client.put`, `profile.encode`, `self.repr`).
    if !method_owners.contains_key(method)
        && !project_type_shorts.contains(receiver_leaf)
    {
        return Some(format!("no-project-method: {receiver_leaf}.{method}"));
    }

    None
}

