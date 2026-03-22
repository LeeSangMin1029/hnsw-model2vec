//! Call graph adjacency list built from `ParsedChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::index_tables::{self, strip_generics_from_key};
use crate::parse::ParsedChunk;

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

        Self { names, files, kinds, lines, signatures, is_test, name_index, exact, short }
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

    /// Build the call graph from parsed chunks.
    ///
    /// Resolves calls by simple name matching (exact → short fallback).
    /// Chunks should already contain accurate call data from RA via daemon.
    pub fn build_full(chunks: &[ParsedChunk]) -> Self {
        let rss0 = current_rss_mb();
        eprintln!("      [graph] RSS baseline: {rss0:.1}MB");

        let t0 = std::time::Instant::now();
        let meta = ChunkMeta::collect(chunks);
        eprintln!("      [graph] meta collect: {:.1}ms  RSS: {:.1}MB (+{:.1})", t0.elapsed().as_secs_f64() * 1000.0, current_rss_mb(), current_rss_mb() - rss0);

        let t1 = std::time::Instant::now();
        let owner_field_types = index_tables::collect_owner_field_types(chunks);
        let mut adj = AdjState::new(chunks.len());

        // Resolve calls by name matching: exact (lowercase) → short (last :: segment).
        for (src, chunk) in chunks.iter().enumerate() {
            for (call_idx, call) in chunk.calls.iter().enumerate() {
                let lower = call.to_lowercase();
                let tgt = meta.exact.get(&lower).copied()
                    .or_else(|| {
                        let short = lower.rsplit("::").next().unwrap_or(&lower);
                        meta.short.get(short).copied()
                    });
                if let Some(tgt) = tgt {
                    let line = chunk.call_lines.get(call_idx).copied().unwrap_or(0);
                    adj.add_edge(src, tgt, line);
                }
            }
            // Type ref edges.
            for ty in &chunk.types {
                let lower = ty.to_lowercase();
                if let Some(&tgt) = meta.exact.get(&lower).or_else(|| meta.short.get(&lower)) {
                    if tgt as usize != src {
                        adj.callees[src].push(tgt);
                        adj.callers[tgt as usize].push(src as u32);
                    }
                }
            }
        }
        eprintln!("      [graph] resolve: {:.1}ms ({} chunks)  RSS: {:.1}MB (+{:.1})", t1.elapsed().as_secs_f64() * 1000.0, chunks.len(), current_rss_mb(), current_rss_mb() - rss0);

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

