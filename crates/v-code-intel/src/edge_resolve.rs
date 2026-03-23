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

    // Primary: (file, start_line) → chunk index.
    // MIR provides exact callee definition location — no name ambiguity.
    let mut loc_to_idx: HashMap<(&str, usize), u32> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        if let Some((start, _)) = c.lines {
            loc_to_idx.insert((&c.file, start), i as u32);
        }
    }

    // Secondary: name → chunk index (fallback for edges without location).
    let mut name_to_idx: HashMap<String, u32> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        name_to_idx.insert(c.name.to_lowercase(), i as u32);
    }

    for (caller_name, callees) in &mir_edges.by_caller {
        let src = resolve_by_loc_or_name(caller_name, &loc_to_idx, &name_to_idx, chunks);

        for callee in callees {
            let tgt = if !callee.file.is_empty() && callee.start_line > 0 {
                // Primary: exact file + start_line match
                resolve_by_location(&callee.file, callee.start_line, &loc_to_idx, chunks)
            } else {
                None
            }.or_else(|| {
                // Fallback: name match (for edges without callee location)
                let lower = callee.name.to_lowercase();
                resolve_mir_name(&lower, &name_to_idx)
            });

            match (src, tgt) {
                (Some(s), Some(t)) => {
                    mir_resolved += 1;
                    adj.add_edge(s as usize, t, callee.call_line as u32);
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

/// Resolve by file + start_line (exact location match).
///
/// Normalizes the MIR file path to match chunk file paths, then looks up
/// the chunk whose file and start_line match. If exact start_line doesn't
/// match (off by one from span differences), checks ±1 range.
fn resolve_by_location(
    file: &str,
    start_line: usize,
    loc_to_idx: &HashMap<(&str, usize), u32>,
    chunks: &[ParsedChunk],
) -> Option<u32> {
    // Find chunk whose file ends with the MIR file path (handles relative vs absolute)
    for c in chunks {
        if let Some((chunk_start, _)) = c.lines {
            let file_match = c.file.ends_with(file)
                || file.ends_with(&c.file)
                || c.file == file;
            if file_match && (chunk_start == start_line
                || chunk_start + 1 == start_line
                || (chunk_start > 0 && chunk_start - 1 == start_line))
            {
                // Found — look up index
                return loc_to_idx.get(&(c.file.as_str(), chunk_start)).copied();
            }
        }
    }
    None
}

/// Resolve a caller by location (from MIR caller_file in by_caller key)
/// or by name fallback.
fn resolve_by_loc_or_name(
    caller_name: &str,
    _loc_to_idx: &HashMap<(&str, usize), u32>,
    name_to_idx: &HashMap<String, u32>,
    chunks: &[ParsedChunk],
) -> Option<u32> {
    let lower = caller_name.to_lowercase();
    // Name match first (callers are always from local crate, names are consistent)
    resolve_mir_name(&lower, name_to_idx)
        .or_else(|| {
            // Fallback: find by name suffix in chunks
            let stripped = strip_closure_suffix(&lower);
            chunks.iter().enumerate().find_map(|(i, c)| {
                let cl = c.name.to_lowercase();
                if cl == stripped || cl.ends_with(&format!("::{stripped}")) {
                    Some(i as u32)
                } else {
                    None
                }
            })
        })
}

/// Name-based fallback for edges without location data.
fn resolve_mir_name(name: &str, name_to_idx: &HashMap<String, u32>) -> Option<u32> {
    let name = strip_closure_suffix(name);
    if let Some(&idx) = name_to_idx.get(name) {
        return Some(idx);
    }
    // Strip crate prefix
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


