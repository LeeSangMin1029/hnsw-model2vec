//! Call graph adjacency list built from `ParsedChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.
//!
//! Edge resolution (how calls are connected) is handled by [`crate::edge_resolve`].

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::edge_resolve::{self, ChunkIndex, ResolvedEdges};
use crate::index_tables;
use crate::mir_edges::MirEdgeMap;
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
}

// ── Build ───────────────────────────────────────────────────────────

impl CallGraph {
    /// Build call graph using name matching (legacy).
    pub fn build(chunks: &[ParsedChunk]) -> Self {
        let t0 = std::time::Instant::now();
        let index = ChunkIndex::build(chunks);
        let adj = edge_resolve::resolve_by_name(chunks, &index);
        eprintln!("      [graph] name-resolve: {:.1}ms ({} chunks)", t0.elapsed().as_secs_f64() * 1000.0, chunks.len());
        Self::assemble(chunks, &index, adj)
    }

    /// Build call graph using MIR-resolved edges (100% accurate).
    /// Falls back to name matching for edges not found in MIR.
    pub fn build_with_mir(chunks: &[ParsedChunk], mir_edges: &MirEdgeMap) -> Self {
        let t0 = std::time::Instant::now();
        let index = ChunkIndex::build(chunks);
        let adj = edge_resolve::resolve_with_mir(chunks, &index, mir_edges);
        eprintln!("      [graph] mir-resolve: {:.1}ms ({} chunks)", t0.elapsed().as_secs_f64() * 1000.0, chunks.len());
        Self::assemble(chunks, &index, adj)
    }

    /// Assemble CallGraph from resolved edges + chunk metadata.
    fn assemble(chunks: &[ParsedChunk], index: &ChunkIndex, adj: ResolvedEdges, ) -> Self {
        let t = std::time::Instant::now();
        let owner_field_types = index_tables::collect_owner_field_types(chunks);

        let len = chunks.len();
        let mut names = Vec::with_capacity(len);
        let mut files = Vec::with_capacity(len);
        let mut kinds = Vec::with_capacity(len);
        let mut lines_vec = Vec::with_capacity(len);
        let mut signatures = Vec::with_capacity(len);
        let mut is_test = Vec::with_capacity(len);
        let mut name_index = Vec::with_capacity(len);

        for (i, c) in chunks.iter().enumerate() {
            name_index.push((c.name.to_lowercase(), i as u32));
            names.push(c.name.clone());
            files.push(c.file.clone());
            kinds.push(c.kind.clone());
            lines_vec.push(c.lines);
            signatures.push(c.signature.clone());
            is_test.push(is_test_chunk(c));
        }
        name_index.sort_by(|a, b| a.0.cmp(&b.0));

        let (trait_impls, field_access_index) = rayon::join(
            || index_tables::build_trait_impls(&names, &kinds, &index.exact, &index.short),
            || index_tables::build_field_access_index(chunks, &owner_field_types),
        );
        let string_index = index_tables::build_string_index(chunks);
        let param_flows: Vec<Vec<(String, u8, String, u8, u32)>> = chunks.iter().map(|c| c.param_flows.clone()).collect();
        let string_args: Vec<Vec<(String, String, u32, u8)>> = chunks.iter().map(|c| c.string_args.clone()).collect();
        eprintln!("      [graph] assemble: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);

        Self {
            version: GRAPH_SOURCE_HASH.to_owned(),
            names, files, kinds,
            lines: lines_vec, signatures, name_index,
            callees: adj.callees, callers: adj.callers,
            is_test, trait_impls,
            call_sites: adj.call_sites,
            string_args, string_index, param_flows,
            field_access_index,
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

    // ── Persistence ─────────────────────────────────────────────────

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

fn graph_cache_path(db: &Path) -> std::path::PathBuf {
    db.join("cache").join("graph.bin")
}
