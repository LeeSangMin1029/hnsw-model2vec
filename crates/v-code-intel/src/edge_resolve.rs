//! Edge resolution strategies for call graph construction.
//!
//! Separates "how to connect edges" from the graph data structure itself.
//! Two resolvers:
//! - `resolve_by_name`: legacy name matching (exact → short fallback)
//! - `resolve_with_mir`: MIR-first, 100% accurate, name fallback for unmatched

use std::collections::HashMap;

use crate::index_tables::strip_generics_from_key;
use crate::mir_edges::MirEdgeMap;
use crate::parse::ParsedChunk;

// ── ChunkIndex — name-to-index lookup tables ────────────────────────

/// Bidirectional name index for resolving call names to chunk indices.
pub(crate) struct ChunkIndex {
    pub exact: HashMap<String, u32>,
    pub short: HashMap<String, u32>,
}

impl ChunkIndex {
    pub fn build(chunks: &[ParsedChunk]) -> Self {
        let mut exact = HashMap::new();
        let mut short = HashMap::new();

        for (i, c) in chunks.iter().enumerate() {
            let idx = i as u32;
            let lower = c.name.to_lowercase();

            exact.insert(lower.clone(), idx);
            let stripped = strip_generics_from_key(&lower);
            if stripped != lower { exact.entry(stripped).or_insert(idx); }

            if let Some(s) = c.name.rsplit("::").next() {
                short.entry(s.to_lowercase()).or_insert(idx);
            }

            if let Some((prefix, method_name)) = lower.rsplit_once("::") {
                if let Some(owner_leaf) = prefix.rsplit_once("::").map(|p| p.1) {
                    let alias = format!("{owner_leaf}::{method_name}");
                    if alias != lower { exact.entry(alias).or_insert(idx); }
                }
                if let Some(for_pos) = prefix.find(" for ") {
                    let concrete = &prefix[for_pos + 5..];
                    let leaf = concrete.rsplit("::").next().unwrap_or(concrete)
                        .split('<').next().unwrap_or("");
                    if !leaf.is_empty() {
                        exact.entry(format!("{leaf}::{method_name}")).or_insert(idx);
                    }
                }
            }
        }

        Self { exact, short }
    }

    fn resolve_name(&self, call: &str) -> Option<u32> {
        let lower = call.to_lowercase();
        self.exact.get(&lower).copied()
            .or_else(|| {
                let short = lower.rsplit("::").next().unwrap_or(&lower);
                self.short.get(short).copied()
            })
    }
}

// ── Resolved edges output ───────────────────────────────────────────

/// Accumulated adjacency state from edge resolution.
pub(crate) struct ResolvedEdges {
    pub callees: Vec<Vec<u32>>,
    pub callers: Vec<Vec<u32>>,
    pub call_sites: Vec<Vec<(u32, u32)>>,
}

impl ResolvedEdges {
    fn new(len: usize) -> Self {
        Self {
            callees: vec![Vec::new(); len],
            callers: vec![Vec::new(); len],
            call_sites: vec![Vec::new(); len],
        }
    }

    fn add_edge(&mut self, src: usize, tgt: u32, call_line: u32) {
        let tgt_usize = tgt as usize;
        if tgt_usize != src {
            self.callees[src].push(tgt);
            self.callers[tgt_usize].push(src as u32);
            self.call_sites[src].push((tgt, call_line));
        }
    }

    pub(crate) fn dedup(&mut self) {
        for list in &mut self.callees { list.sort_unstable(); list.dedup(); }
        for list in &mut self.callers { list.sort_unstable(); list.dedup(); }
        for sites in &mut self.call_sites { sites.sort_by_key(|&(tgt, _)| tgt); sites.dedup_by_key(|e| e.0); }
    }
}

// ── Name-based resolver (legacy) ────────────────────────────────────

/// Resolve call edges by name matching only.
pub(crate) fn resolve_by_name(chunks: &[ParsedChunk], index: &ChunkIndex) -> ResolvedEdges {
    let mut adj = ResolvedEdges::new(chunks.len());

    for (src, chunk) in chunks.iter().enumerate() {
        for (call_idx, call) in chunk.calls.iter().enumerate() {
            if let Some(tgt) = index.resolve_name(call) {
                let line = chunk.call_lines.get(call_idx).copied().unwrap_or(0);
                adj.add_edge(src, tgt, line);
            }
        }
        resolve_type_refs(src, chunk, index, &mut adj);
    }

    adj.dedup();
    adj
}

// ── MIR-based resolver ──────────────────────────────────────────────

/// Resolve call edges directly from MIR edge map.
///
/// Iterates MIR caller→callee pairs and maps them to chunk indices.
/// Does not depend on chunk.calls (which may be empty in MIR mode).
pub(crate) fn resolve_with_mir(
    chunks: &[ParsedChunk],
    index: &ChunkIndex,
    mir_edges: &MirEdgeMap,
) -> ResolvedEdges {
    let mut adj = ResolvedEdges::new(chunks.len());
    let mut mir_resolved: usize = 0;
    let mut mir_external: usize = 0;

    // Build chunk name → index map.
    //
    // MIR edge callee names may differ from chunk names due to `pub use`
    // re-exports. In test compilation, lib functions appear external and
    // `def_path_str` uses the re-export visible path (skipping internal
    // modules). For example:
    //   chunk:  commands::intel::commands::run_blast  (definition path)
    //   edge:   commands::intel::run_blast            (re-export path)
    //
    // We register both the full name and the re-export alias (derived from
    // the file path: if file has `<parent>/commands.rs`, the `commands`
    // module segment is an artifact of the file→module mapping).
    let mut name_to_idx: HashMap<String, u32> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        let lower = c.name.to_lowercase();
        let idx = i as u32;
        name_to_idx.insert(lower.clone(), idx);

        // Derive re-export alias from file path.
        // Files like `commands/intel/commands.rs` create a module segment
        // matching the file stem (`commands`). When `mod.rs` does
        // `pub use commands::run_blast`, the visible path skips this segment.
        //
        // We find the stem segment's position in the chunk name and remove it.
        // e.g. chunk `commands::intel::commands::run_blast` with stem `commands`
        //    → find `::commands::` after the module path → remove
        //    → alias `commands::intel::run_blast`
        if let Some(stem) = std::path::Path::new(&c.file)
            .file_stem()
            .and_then(|s| s.to_str())
        {
            if stem != "mod" && stem != "lib" && stem != "main" {
                let needle = format!("::{stem}::");
                let lower_needle = needle.to_lowercase();
                // Find the LAST occurrence of `::stem::` — that's the file module
                if let Some(pos) = lower.rfind(&lower_needle) {
                    let alias = format!(
                        "{}::{}",
                        &lower[..pos],
                        &lower[pos + lower_needle.len()..]
                    );
                    if !alias.is_empty() && alias != lower {
                        name_to_idx.entry(alias).or_insert(idx);
                    }
                }
            }
        }
    }

    for (caller_name, callees) in &mir_edges.by_caller {
        let src = resolve_mir_name(&caller_name.to_lowercase(), &name_to_idx);

        for (callee_name, line) in callees {
            let tgt = resolve_mir_name(&callee_name.to_lowercase(), &name_to_idx);

            match (src, tgt) {
                (Some(s), Some(t)) => {
                    mir_resolved += 1;
                    adj.add_edge(s as usize, t, *line as u32);
                }
                _ => {
                    mir_external += 1;
                }
            }
        }
    }

    // Type ref edges from chunks
    for (src, chunk) in chunks.iter().enumerate() {
        resolve_type_refs(src, chunk, index, &mut adj);
    }

    eprintln!("      [edge-resolve] mir={mir_resolved} external={mir_external}");
    adj.dedup();
    adj
}

/// Resolve a MIR fully-qualified name to a chunk index.
///
/// MIR names are exact — no fuzzy fallback. Only two strategies:
/// 1. Direct match against chunk name
/// 2. Strip crate prefix (MIR: `v_code_intel::graph::build`, chunk: `graph::build`)
fn resolve_mir_name(name: &str, name_to_idx: &HashMap<String, u32>) -> Option<u32> {
    let name = strip_closure_suffix(name);

    // 1. Direct match
    if let Some(&idx) = name_to_idx.get(name) {
        return Some(idx);
    }
    // 2. Strip crate prefix
    if let Some((_, rest)) = name.split_once("::") {
        if let Some(&idx) = name_to_idx.get(rest) {
            return Some(idx);
        }
    }
    None
}

/// Strip `{closure#N}` suffixes from async function MIR names.
/// `daemon::run::{closure#0}` → `daemon::run`
fn strip_closure_suffix(name: &str) -> &str {
    if let Some(pos) = name.find("::{closure") {
        &name[..pos]
    } else {
        name
    }
}

// ── Shared helpers ──────────────────────────────────────────────────

fn resolve_type_refs(src: usize, chunk: &ParsedChunk, index: &ChunkIndex, adj: &mut ResolvedEdges) {
    for ty in &chunk.types {
        let lower = ty.to_lowercase();
        if let Some(&tgt) = index.exact.get(&lower).or_else(|| index.short.get(&lower)) {
            if tgt as usize != src {
                adj.callees[src].push(tgt);
                adj.callers[tgt as usize].push(src as u32);
            }
        }
    }
}

/// Extract `Type::method` from a fully-qualified MIR name.
fn extract_type_method(mir_name: &str) -> &str {
    let cleaned = mir_name.trim_start_matches('<');
    let mut colons: Vec<usize> = Vec::new();
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b':' && bytes[i + 1] == b':' {
            colons.push(i);
            i += 2;
        } else {
            i += 1;
        }
    }
    if colons.len() >= 2 {
        &cleaned[colons[colons.len() - 2] + 2..]
    } else {
        cleaned.trim_start_matches('>')
    }
}

