//! Edge resolution strategies for call graph construction.
//!
//! Separates "how to connect edges" from the graph data structure itself.
//! Two resolvers:
//! - `resolve_by_name`: legacy name matching (exact → short fallback)
//! - `resolve_with_mir`: MIR-first, 100% accurate, name fallback for unmatched

use std::collections::HashMap;

use crate::index_tables::{self, strip_generics_from_key};
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

    // Build name → chunk_index map for fast lookup
    let mut name_to_idx: HashMap<String, u32> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        let lower = c.name.to_lowercase();
        name_to_idx.insert(lower.clone(), i as u32);
        let short = extract_type_method(&lower);
        if short != lower {
            name_to_idx.entry(short.to_owned()).or_insert(i as u32);
        }
    }

    // Resolve from MIR edge map directly
    for (caller_name, callees) in &mir_edges.by_caller {
        let caller_lower = caller_name.to_lowercase();
        let src = resolve_mir_name(&caller_lower, &name_to_idx, index);
        let Some(src) = src else { continue };

        for (callee_name, line) in callees {
            let callee_lower = callee_name.to_lowercase();
            let tgt = resolve_mir_name(&callee_lower, &name_to_idx, index);
            if let Some(tgt) = tgt {
                mir_resolved += 1;
                adj.add_edge(src as usize, tgt, *line as u32);
            } else {
                mir_external += 1;
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
/// Tries progressively shorter name forms:
/// 1. Full name: `onttothree_agent::daemon::check_estop`
/// 2. Strip crate prefix: `daemon::check_estop`
/// 3. Type::method extract: `check_estop`
/// 4. ChunkIndex name matching (exact → short fallback)
fn resolve_mir_name(name: &str, name_to_idx: &HashMap<String, u32>, index: &ChunkIndex) -> Option<u32> {
    // Strip closure suffixes: `fn::{closure#0}` → `fn`
    let name = strip_closure_suffix(name);

    // 1. Direct match
    if let Some(&idx) = name_to_idx.get(name) {
        return Some(idx);
    }
    // 2. Strip crate prefix (first :: segment)
    if let Some(rest) = name.split_once("::").map(|(_, r)| r) {
        if let Some(&idx) = name_to_idx.get(rest) {
            return Some(idx);
        }
    }
    // 3. Type::method (last two :: segments)
    let short = extract_type_method(name);
    if let Some(&idx) = name_to_idx.get(short) {
        return Some(idx);
    }
    // 4. Just the last segment
    if let Some(last) = name.rsplit("::").next() {
        if let Some(&idx) = name_to_idx.get(last) {
            return Some(idx);
        }
    }
    // 5. ChunkIndex fallback
    index.resolve_name(name)
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

fn callee_matches_call(callee: &str, call: &str) -> bool {
    callee == call || callee.ends_with(&format!("::{call}"))
        || extract_type_method(callee) == call
}

fn find_mir_caller<'a>(
    mir_edges: &'a MirEdgeMap,
    chunk_name_lower: &str,
) -> Option<&'a [(String, usize)]> {
    if let Some(v) = mir_edges.by_caller.get(chunk_name_lower) {
        return Some(v.as_slice());
    }
    for (caller, edges) in &mir_edges.by_caller {
        let caller_lower = caller.to_lowercase();
        let suffix = extract_type_method(&caller_lower);
        if suffix == chunk_name_lower || caller_lower.ends_with(&format!("::{chunk_name_lower}")) {
            return Some(edges.as_slice());
        }
    }
    None
}
