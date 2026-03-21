//! Call graph adjacency list built from `ParsedChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::index_tables::{self, extract_leaf_type, strip_generics_from_key};
use crate::parse::ParsedChunk;
use crate::resolve::{ChunkCtx, ResolveCtx, resolve_calls};

/// Source hash — auto-computed by build.rs for cache invalidation.
const GRAPH_SOURCE_HASH: &str = env!("GRAPH_SOURCE_HASH");

/// Get current process RSS in MB.
pub fn current_rss_mb() -> f64 {
    #[cfg(windows)]
    {
        use std::mem::MaybeUninit;
        #[repr(C)]
        struct Pmc { cb: u32, pf: u32, peak: usize, wss: usize, qpp: usize, qp: usize, qnpp: usize, qnp: usize, pfu: usize, ppfu: usize }
        #[expect(unsafe_code, reason = "FFI for Windows memory profiling")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(h: *mut std::ffi::c_void, p: *mut Pmc, cb: u32) -> i32;
        }
        #[expect(unsafe_code, reason = "FFI for memory profiling")]
        unsafe {
            let mut pmc = MaybeUninit::<Pmc>::zeroed().assume_init();
            pmc.cb = std::mem::size_of::<Pmc>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                return pmc.wss as f64 / (1024.0 * 1024.0);
            }
        }
        0.0
    }
    #[cfg(not(windows))]
    { 0.0 }
}

// Re-export for external consumers (used by v-code, stats, clones).
pub use crate::index_tables::{is_test_path, is_test_chunk};

// ── CallGraph struct ────────────────────────────────────────────────

/// Pre-built call graph with bidirectional adjacency lists.
#[derive(bincode::Encode, bincode::Decode)]
pub struct CallGraph {
    version: String,
    pub names: Vec<String>,
    pub files: Vec<String>,
    pub kinds: Vec<String>,
    pub lines: Vec<Option<(usize, usize)>>,
    pub signatures: Vec<Option<String>>,
    pub name_index: Vec<(String, u32)>,
    pub callees: Vec<Vec<u32>>,
    pub callers: Vec<Vec<u32>>,
    pub is_test: Vec<bool>,
    pub trait_impls: Vec<Vec<u32>>,
    pub call_sites: Vec<Vec<(u32, u32)>>,
    pub string_args: Vec<Vec<(String, String, u32, u8)>>,
    pub string_index: Vec<(String, Vec<(u32, u32)>)>,
    #[expect(clippy::type_complexity, reason = "tuple layout mirrors string_args")]
    pub param_flows: Vec<Vec<(String, u8, String, u8, u32)>>,
    pub field_access_index: Vec<(String, Vec<u32>)>,
    pub extern_calls: Vec<Vec<(String, String)>>,
}

// ── ChunkMeta ───────────────────────────────────────────────────────

/// Shared chunk metadata collected during graph construction.
struct ChunkMeta {
    names: Vec<String>,
    files: Vec<String>,
    kinds: Vec<String>,
    lines: Vec<Option<(usize, usize)>>,
    signatures: Vec<Option<String>>,
    is_test: Vec<bool>,
    name_index: Vec<(String, u32)>,
    exact: HashMap<String, u32>,
    short: HashMap<String, u32>,
    enum_variants: HashSet<String>,
}

impl ChunkMeta {
    fn collect(chunks: &[ParsedChunk]) -> Self {
        let len = chunks.len();
        let mut exact = HashMap::new();
        let mut short = HashMap::new();
        let (mut names, mut files, mut kinds) = (Vec::with_capacity(len), Vec::with_capacity(len), Vec::with_capacity(len));
        let (mut lines, mut signatures, mut is_test) = (Vec::with_capacity(len), Vec::with_capacity(len), Vec::with_capacity(len));
        let mut name_index = Vec::with_capacity(len);

        for (i, c) in chunks.iter().enumerate() {
            let idx = i as u32;
            let lower = c.name.to_lowercase();

            // Exact match + generic-stripped alias
            exact.insert(lower.clone(), idx);
            let stripped = strip_generics_from_key(&lower);
            if stripped != lower { exact.entry(stripped).or_insert(idx); }

            // Short name (last :: segment)
            if let Some(s) = c.name.rsplit("::").next() {
                short.entry(s.to_lowercase()).or_insert(idx);
            }

            // Owner::method alias for method_owners resolution
            if let Some((prefix, method_name)) = lower.rsplit_once("::") {
                if let Some(owner_leaf) = prefix.rsplit_once("::").map(|p| p.1) {
                    let alias = format!("{owner_leaf}::{method_name}");
                    if alias != lower { exact.entry(alias).or_insert(idx); }
                }
                // "impl Trait for Type::method" → "type::method" alias
                if let Some(for_pos) = prefix.find(" for ") {
                    let concrete = &prefix[for_pos + 5..];
                    let leaf = concrete.rsplit("::").next().unwrap_or(concrete)
                        .split('<').next().unwrap_or("");
                    if !leaf.is_empty() {
                        exact.entry(format!("{leaf}::{method_name}")).or_insert(idx);
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

        // Collect enum variant names for variant detection
        let mut enum_variants = HashSet::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if kinds[i] == "enum" {
                let leaf = names[i].rsplit("::").next().unwrap_or(&names[i]).to_lowercase();
                for v in &chunk.enum_variants {
                    enum_variants.insert(format!("{leaf}::{v}"));
                }
            }
        }

        Self { names, files, kinds, lines, signatures, is_test, name_index, exact, short, enum_variants }
    }
}

// ── AdjState ────────────────────────────────────────────────────────

/// Mutable adjacency state accumulated during edge resolution.
struct AdjState {
    callees: Vec<Vec<u32>>,
    callers: Vec<Vec<u32>>,
    call_sites: Vec<Vec<(u32, u32)>>,
}

impl AdjState {
    fn new(len: usize) -> Self {
        Self { callees: vec![Vec::new(); len], callers: vec![Vec::new(); len], call_sites: vec![Vec::new(); len] }
    }

    fn add_edge(&mut self, src: usize, tgt: u32, call_line: u32) {
        let tgt_usize = tgt as usize;
        if tgt_usize != src {
            self.callees[src].push(tgt);
            self.callers[tgt_usize].push(src as u32);
            self.call_sites[src].push((tgt, call_line));
        }
    }

    fn dedup(&mut self) {
        for list in &mut self.callees { list.sort_unstable(); list.dedup(); }
        for list in &mut self.callers { list.sort_unstable(); list.dedup(); }
        for sites in &mut self.call_sites { sites.sort_by_key(|&(tgt, _)| tgt); sites.dedup_by_key(|e| e.0); }
    }
}

// ── Build ───────────────────────────────────────────────────────────

impl CallGraph {
    pub fn build(chunks: &[ParsedChunk]) -> Self { Self::build_full(chunks) }

    pub fn build_full(chunks: &[ParsedChunk]) -> Self {
        Self::build_with_lsp(chunks, &crate::lsp_client::LspTypes::default(), None)
    }

    pub fn build_with_lsp(
        chunks: &[ParsedChunk],
        lsp_types: &crate::lsp_client::LspTypes,
        ra: Option<&v_lsp::instance::RaInstance>,
    ) -> Self {
        Self::build_with_lsp_filtered(chunks, lsp_types, ra, None)
    }

    pub fn build_with_lsp_filtered(
        chunks: &[ParsedChunk],
        lsp_types: &crate::lsp_client::LspTypes,
        ra: Option<&v_lsp::instance::RaInstance>,
        ra_skip_files: Option<&HashSet<&str>>,
    ) -> Self {
        let rss0 = current_rss_mb();
        eprintln!("      [graph] RSS baseline: {rss0:.1}MB");

        let t0 = std::time::Instant::now();
        let meta = ChunkMeta::collect(chunks);
        eprintln!("      [graph] meta collect: {:.1}ms  RSS: {:.1}MB (+{:.1})", t0.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        // Build index tables in parallel.
        let t1 = std::time::Instant::now();
        let mut adj = AdjState::new(chunks.len());
        let (left, right) = rayon::join(
            || rayon::join(
                || rayon::join(
                    || index_tables::collect_owner_types(chunks),
                    || index_tables::collect_owner_field_types(chunks),
                ),
                || rayon::join(
                    || index_tables::build_return_type_map(chunks),
                    || index_tables::collect_trait_methods(chunks),
                ),
            ),
            || rayon::join(
                || index_tables::build_method_owner_index(chunks),
                || rayon::join(
                    || index_tables::collect_instantiated_types(chunks),
                    || index_tables::build_trait_impl_method_map(chunks),
                ),
            ),
        );
        let ((owner_types, owner_field_types), (mut return_type_map, _trait_methods)) = left;
        let (method_owners, (instantiated, trait_impl_methods)) = right;

        // LSP return type overlay
        merge_lsp_return_types(&mut return_type_map, &lsp_types.return_types, chunks);
        eprintln!("      [graph] index tables: {:.1}ms  RSS: {:.1}MB (+{:.1})", t1.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        let extern_all_methods: HashSet<String> = HashSet::new();

        // Build ResolveCtx (shared across all chunks)
        let rctx = ResolveCtx {
            exact: &meta.exact, short: &meta.short,
            kinds: &meta.kinds, names: &meta.names,
            return_type_map: &return_type_map,
            method_owners: &method_owners,
            instantiated: &instantiated,
            trait_impl_methods: &trait_impl_methods,
            enum_variants: &meta.enum_variants,
            extern_all_methods: &extern_all_methods,
        };

        // Pre-compute per-chunk contexts.
        let t2 = std::time::Instant::now();
        use rayon::prelude::*;
        let mut ctxs: Vec<ChunkCtx> = chunks.par_iter()
            .map(|c| build_chunk_ctx(c, &owner_types, &owner_field_types))
            .collect();

        // Inject LSP-resolved receiver types.
        inject_lsp_receivers(&mut ctxs, &lsp_types.receiver_types, &meta);

        // Pass 1: parallel resolve.
        let par_results: Vec<_> = chunks.par_iter()
            .zip(ctxs.par_iter_mut())
            .enumerate()
            .map(|(src, (c, ctx))| resolve_calls(src, c, ctx, &rctx, None))
            .collect();

        let mut resolved_indices: Vec<Vec<bool>> = Vec::with_capacity(chunks.len());
        for (src, (edges, type_edges, _, _)) in par_results.iter().enumerate() {
            for &(tgt, line) in edges { adj.add_edge(src, tgt, line); }
            for &(tgt, _) in type_edges {
                if tgt as usize != src {
                    adj.callees[src].push(tgt);
                    adj.callers[tgt as usize].push(src as u32);
                }
            }
        }
        for (src, (_, _, receiver_updates, resolved)) in par_results.into_iter().enumerate() {
            for (var, ty) in receiver_updates {
                ctxs[src].receiver_types.entry(var).or_insert(ty);
            }
            debug_assert_eq!(src, resolved_indices.len());
            resolved_indices.push(resolved);
        }
        eprintln!("      [graph] pass 1 resolve: {:.1}ms ({} chunks)  RSS: {:.1}MB (+{:.1})", t2.elapsed().as_secs_f64() * 1000.0, chunks.len(), current_rss_mb(), current_rss_mb() - rss0);

        // Pass 2+: iterative convergence.
        let names_lower: Vec<String> = meta.names.iter().map(|n| n.to_lowercase()).collect();
        let names_short: Vec<&str> = names_lower.iter().map(|n| n.rsplit("::").next().unwrap_or(n)).collect();
        let callee_param_index: Vec<HashMap<u8, String>> = chunks.iter().map(|c| {
            c.param_types.iter().enumerate().filter_map(|(i, (name, ty))| {
                if name.eq_ignore_ascii_case("self") || ty.is_empty() { return None; }
                Some((i as u8, extract_leaf_type(&ty.to_lowercase()).to_owned()))
            }).collect()
        }).collect();

        let t_iter = std::time::Instant::now();
        let mut iter_passes = 0u32;
        for _ in 0..3 {
            let prev_total: usize = adj.callees.iter().map(|c| c.len()).sum();

            // Compute extra receiver types from resolved callees.
            let extra: Vec<HashMap<String, String>> = chunks.par_iter().enumerate()
                .map(|(src, chunk)| {
                    let mut extra = HashMap::new();
                    for (var, callee_name) in &chunk.let_call_bindings {
                        let cl = callee_name.to_lowercase();
                        let leaf = cl.rsplit("::").next().unwrap_or(&cl).rsplit('.').next().unwrap_or(&cl);
                        for &ci in &adj.callees[src] {
                            if names_short[ci as usize] == leaf {
                                if let Some(ret) = return_type_map.get(&names_lower[ci as usize]) {
                                    extra.entry(var.to_lowercase()).or_insert_with(|| ret.clone());
                                }
                            }
                        }
                    }
                    for (pname, _, callee_raw, callee_arg, _) in &chunk.param_flows {
                        let cl = callee_raw.to_lowercase();
                        let leaf = cl.rsplit("::").next().unwrap_or(&cl).rsplit('.').next().unwrap_or(&cl);
                        for &ci in &adj.callees[src] {
                            if names_short[ci as usize] != leaf { continue; }
                            if let Some(ty) = callee_param_index[ci as usize].get(callee_arg) {
                                if !ty.is_empty() {
                                    extra.entry(pname.to_lowercase()).or_insert_with(|| ty.clone());
                                }
                            }
                        }
                    }
                    extra
                }).collect();

            for (src, e) in extra.iter().enumerate() {
                for (var, ty) in e { ctxs[src].receiver_types.entry(var.clone()).or_insert_with(|| ty.clone()); }
            }

            // Delta resolve: only enriched chunks.
            let enriched: Vec<usize> = (0..chunks.len()).filter(|&i| !extra[i].is_empty()).collect();
            let delta: Vec<(usize, Vec<(u32, u32)>)> = enriched.par_iter()
                .map(|&src| {
                    let (edges, _, _, _) = resolve_calls(src, &chunks[src], &mut ctxs[src].clone_for_delta(), &rctx, Some(&resolved_indices[src]));
                    (src, edges)
                }).collect();

            for (src, edges) in &delta {
                for &(tgt, line) in edges {
                    if tgt as usize != *src && !adj.callees[*src].contains(&tgt) {
                        adj.callees[*src].push(tgt);
                        adj.callers[tgt as usize].push(*src as u32);
                        adj.call_sites[*src].push((tgt, line));
                    }
                }
            }
            let new_total: usize = adj.callees.iter().map(|c| c.len()).sum();
            iter_passes += 1;
            let enriched_count = extra.iter().filter(|m| !m.is_empty()).count();
            eprintln!("        [iter] pass {iter_passes}: edges {prev_total}->{new_total} (+{}), enriched chunks: {enriched_count}, RSS: {:.1}MB", new_total - prev_total, current_rss_mb());
            if new_total == prev_total { break; }
        }
        eprintln!("      [graph] pass 2+ iterative: {:.1}ms ({iter_passes} passes)  RSS: {:.1}MB (+{:.1})", t_iter.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        // RA call hierarchy edges.
        let t_ra = std::time::Instant::now();
        let ra_result = crate::ra_direct::resolve_via_ra_filtered(chunks, ra, ra_skip_files);
        if !ra_result.edges.is_empty() {
            let mut ra_added = 0usize;
            for &(src, tgt, line) in &ra_result.edges {
                if src < chunks.len() && tgt < chunks.len() && src != tgt {
                    let tgt_u32 = tgt as u32;
                    if !adj.callees[src].contains(&tgt_u32) {
                        adj.callees[src].push(tgt_u32);
                        adj.callers[tgt].push(src as u32);
                        adj.call_sites[src].push((tgt_u32, line));
                        ra_added += 1;
                    }
                }
            }
            eprintln!("      [graph] RA edges: +{ra_added} new ({} total from RA, {:.1}s)", ra_result.edges.len(), t_ra.elapsed().as_secs_f64());
        }

        adj.dedup();
        let extern_calls = vec![Vec::new(); chunks.len()];
        Self::from_parts(meta, adj, extern_calls, chunks, &owner_field_types)
    }

    fn from_parts(meta: ChunkMeta, adj: AdjState, extern_calls: Vec<Vec<(String, String)>>, chunks: &[ParsedChunk], owner_field_types: &HashMap<String, HashMap<String, String>>) -> Self {
        let t = std::time::Instant::now();
        let (trait_impls, field_access_index) = rayon::join(
            || index_tables::build_trait_impls(&meta.names, &meta.kinds, &meta.exact, &meta.short),
            || index_tables::build_field_access_index(chunks, owner_field_types),
        );
        let string_index = index_tables::build_string_index(chunks);
        let param_flows: Vec<Vec<(String, u8, String, u8, u32)>> = chunks.iter().map(|c| c.param_flows.clone()).collect();
        let string_args: Vec<Vec<(String, String, u32, u8)>> = chunks.iter().map(|c| c.string_args.clone()).collect();
        eprintln!("      [graph] from_parts: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
        Self {
            version: GRAPH_SOURCE_HASH.to_owned(),
            names: meta.names, files: meta.files, kinds: meta.kinds,
            lines: meta.lines, signatures: meta.signatures,
            name_index: meta.name_index,
            callees: adj.callees, callers: adj.callers,
            is_test: meta.is_test, trait_impls,
            call_sites: adj.call_sites,
            string_args, string_index, param_flows,
            field_access_index, extern_calls,
        }
    }

    // ── Query API ───────────────────────────────────────────────────

    pub fn resolve(&self, name: &str) -> Vec<u32> {
        let lower = name.to_lowercase();
        let start = self.name_index.partition_point(|(n, _)| n.as_str() < lower.as_str());
        let mut results: Vec<u32> = self.name_index[start..].iter()
            .take_while(|(n, _)| n == &lower).map(|(_, idx)| *idx).collect();
        if results.is_empty() {
            let suffix = format!("::{lower}");
            results = self.name_index.iter().filter(|(n, _)| n.ends_with(&suffix)).map(|(_, idx)| *idx).collect();
        }
        results
    }

    pub fn call_site_line(&self, caller_idx: u32, callee_idx: u32) -> u32 {
        self.call_sites[caller_idx as usize].iter()
            .find(|&&(tgt, _)| tgt == callee_idx).map(|&(_, line)| line).unwrap_or(0)
    }

    pub fn find_field_access(&self, key: &str) -> Vec<u32> {
        let lower = key.to_lowercase();
        self.field_access_index.binary_search_by_key(&&*lower, |(k, _)| k.as_str())
            .ok().map(|i| self.field_access_index[i].1.clone()).unwrap_or_default()
    }

    pub fn find_field_accesses_for_type(&self, type_name: &str) -> Vec<(&str, &[u32])> {
        let prefix = format!("{}::", type_name.to_lowercase());
        let start = self.field_access_index.partition_point(|(k, _)| k.as_str() < prefix.as_str());
        self.field_access_index[start..].iter()
            .take_while(|(k, _)| k.starts_with(&prefix))
            .map(|(k, v)| (&k[prefix.len()..], v.as_slice())).collect()
    }

    pub fn find_string(&self, value: &str) -> Vec<(u32, u32)> {
        let lower = value.to_lowercase();
        self.string_index.binary_search_by_key(&&*lower, |(k, _)| k.as_str())
            .ok().map(|i| self.string_index[i].1.clone()).unwrap_or_default()
    }

    pub fn save(&self, db: &Path) -> Result<()> {
        let path = graph_cache_path(db);
        let _ = fs::create_dir_all(path.parent().unwrap_or(Path::new(".")));
        let bytes = bincode::encode_to_vec(self, bincode::config::standard())
            .context("failed to encode call graph")?;
        fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn load(db: &Path) -> Option<Self> {
        let path = graph_cache_path(db);
        let db_mtime = fs::metadata(db.join("payload.dat")).and_then(|m| m.modified()).ok()?;
        let cache_mtime = fs::metadata(&path).and_then(|m| m.modified()).ok()?;
        if cache_mtime < db_mtime { return None; }
        let bytes = fs::read(&path).ok()?;
        let (graph, _): (Self, _) = bincode::decode_from_slice(&bytes, bincode::config::standard()).ok()?;
        if graph.version != GRAPH_SOURCE_HASH { return None; }
        Some(graph)
    }

    pub fn len(&self) -> usize { self.names.len() }
    pub fn is_empty(&self) -> bool { self.names.is_empty() }

    pub fn global_aliases(&self) -> (std::collections::BTreeMap<String, String>, Vec<(String, String)>) {
        let all: Vec<&str> = self.files.iter().map(|f| crate::helpers::relative_path(f)).collect();
        crate::helpers::build_path_aliases(&all)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn graph_cache_path(db: &Path) -> std::path::PathBuf {
    db.join("cache").join("graph.bin")
}

/// Build per-chunk context for resolution.
fn build_chunk_ctx(
    chunk: &ParsedChunk,
    owner_types: &HashMap<String, Vec<String>>,
    owner_field_types: &HashMap<String, HashMap<String, String>>,
) -> ChunkCtx {
    let imports = index_tables::build_import_map(&chunk.imports);
    let self_type = index_tables::owning_type(&chunk.name);
    let call_types = index_tables::extract_call_types(&chunk.calls);
    let mut receiver_types = index_tables::build_receiver_type_map(chunk);

    // Generic trait bounds: T: Search → receiver_types["t"] = "search"
    let generic_bounds = chunk.signature.as_deref()
        .map(index_tables::parse_generic_trait_bounds)
        .unwrap_or_default();
    for (param, bound) in &generic_bounds {
        receiver_types.entry(param.clone()).or_insert_with(|| bound.clone());
    }

    // Owner field types → receiver_types for self.field resolution
    if let Some(ref owner) = self_type {
        if let Some(fields) = owner_field_types.get(owner.as_str()) {
            for (fname, fty) in fields {
                let leaf = extract_leaf_type(fty).to_owned();
                receiver_types.entry(fname.clone()).or_insert(leaf);
            }
        }
        // Owner type_refs → source_types
        if let Some(type_refs) = owner_types.get(owner.as_str()) {
            let source_types: HashSet<String> = type_refs.iter().map(|t| t.to_lowercase()).collect();
            return ChunkCtx {
                imports, self_type, receiver_types,
                call_types_set: call_types.into_iter().collect(),
                import_leaves: HashSet::new(),
                source_types_set: source_types,
            };
        }
    }

    ChunkCtx {
        imports, self_type, receiver_types,
        call_types_set: call_types.into_iter().collect(),
        import_leaves: HashSet::new(),
        source_types_set: HashSet::new(),
    }
}

/// Merge LSP-resolved return types into tree-sitter return_type_map.
fn merge_lsp_return_types(
    return_type_map: &mut HashMap<String, String>,
    lsp_return_types: &HashMap<String, String>,
    chunks: &[ParsedChunk],
) {
    if lsp_return_types.is_empty() { return; }

    let mut short_to_full: HashMap<String, Vec<String>> = HashMap::new();
    for key in return_type_map.keys() {
        let short = key.rsplit("::").next().unwrap_or(key);
        short_to_full.entry(short.to_owned()).or_default().push(key.clone());
    }
    for c in chunks.iter().filter(|c| c.kind == "function") {
        let name_lower = c.name.to_lowercase();
        let short = name_lower.rsplit("::").next().unwrap_or(&name_lower);
        short_to_full.entry(short.to_owned()).or_default().push(name_lower);
    }
    for names in short_to_full.values_mut() { names.sort(); names.dedup(); }

    let mut lsp_added = 0usize;
    for (fn_name, ret_type) in lsp_return_types {
        let full_names = short_to_full.get(fn_name.as_str())
            .cloned().unwrap_or_else(|| vec![fn_name.clone()]);
        for full_name in &full_names {
            use std::collections::hash_map::Entry;
            match return_type_map.entry(full_name.clone()) {
                Entry::Vacant(e) => { e.insert(ret_type.clone()); lsp_added += 1; }
                Entry::Occupied(mut e) => {
                    if e.get().len() <= 1 || e.get() == "self" {
                        e.insert(ret_type.clone()); lsp_added += 1;
                    }
                }
            }
        }
    }
    if lsp_added > 0 { eprintln!("      [graph] LSP overlay: +{lsp_added} return types"); }
}

/// Inject LSP-resolved receiver types into chunk contexts.
fn inject_lsp_receivers(
    ctxs: &mut [ChunkCtx],
    lsp_receivers: &HashMap<usize, HashMap<String, String>>,
    meta: &ChunkMeta,
) {
    if lsp_receivers.is_empty() { return; }
    let mut lsp_injected = 0usize;
    let (mut project, mut ext) = (0usize, 0usize);
    for (&chunk_idx, types) in lsp_receivers {
        if chunk_idx < ctxs.len() {
            for (var, ty) in types {
                ctxs[chunk_idx].receiver_types.entry(var.clone()).or_insert_with(|| {
                    lsp_injected += 1;
                    if meta.exact.contains_key(ty.as_str()) || meta.short.contains_key(ty.as_str()) {
                        project += 1;
                    } else { ext += 1; }
                    ty.clone()
                });
            }
        }
    }
    if lsp_injected > 0 {
        eprintln!("      [graph] LSP receiver injection: +{lsp_injected} var types ({project} project, {ext} extern)");
    }
}
