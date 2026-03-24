//! Edge resolution strategies for call graph construction.
//!
//! Separates "how to connect edges" from the graph data structure itself.
//! Three resolvers:
//! - `resolve_by_name`: legacy name matching (exact → short fallback)
//! - `resolve_with_mir`: MIR-first, 100% accurate, name fallback for unmatched
//! - `resolve_incremental`: per-crate caching, only re-resolves changed crates

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

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
        if tgt_usize != src && src < self.callees.len() && tgt_usize < self.callers.len() {
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

// ── Per-crate resolved edge cache ───────────────────────────────────

/// Cached resolved edges for a single crate.
///
/// Stores edges as `(src_key, tgt_key, call_line)` where keys are
/// `"file:start_line"` strings — independent of chunk array ordering.
/// This allows caches to survive chunk additions/removals/reordering.
#[derive(bincode::Encode, bincode::Decode)]
pub(crate) struct CrateEdgeCache {
    /// Resolved edges: (src_key, tgt_key, call_line).
    /// Keys are "file:start_line" for stable identity across index changes.
    pub edges: Vec<(String, String, u32)>,
}

/// Build a chunk identity key from file + start_line.
fn chunk_key(c: &ParsedChunk) -> String {
    match c.lines {
        Some((start, _)) => format!("{}:{start}", c.file),
        None => c.name.clone(),
    }
}

/// Build a reverse lookup: key → chunk index (for applying cached edges).
fn build_key_to_idx(chunks: &[ParsedChunk]) -> HashMap<String, u32> {
    let mut map = HashMap::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        map.insert(chunk_key(c), i as u32);
    }
    map
}

/// Directory for per-crate edge caches.
fn edge_cache_dir(db_path: &Path) -> std::path::PathBuf {
    db_path.join("cache").join("graph-edges")
}

/// Path to a specific crate's edge cache file.
fn crate_cache_path(db_path: &Path, crate_name: &str) -> std::path::PathBuf {
    edge_cache_dir(db_path).join(format!("{crate_name}.bin"))
}

/// Save per-crate resolved edges to cache.
fn save_crate_cache(db_path: &Path, crate_name: &str, cache: &CrateEdgeCache) -> Result<()> {
    let dir = edge_cache_dir(db_path);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create edge cache dir: {}", dir.display()))?;
    let path = crate_cache_path(db_path, crate_name);
    let bytes = bincode::encode_to_vec(cache, bincode::config::standard())
        .context("failed to encode crate edge cache")?;
    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to write edge cache: {}", path.display()))
}

/// Load per-crate resolved edges from cache.
fn load_crate_cache(db_path: &Path, crate_name: &str) -> Option<CrateEdgeCache> {
    let path = crate_cache_path(db_path, crate_name);
    let bytes = std::fs::read(&path).ok()?;
    let (cache, _): (CrateEdgeCache, _) =
        bincode::decode_from_slice(&bytes, bincode::config::standard()).ok()?;
    Some(cache)
}

/// Check if a crate's edge cache is stale by comparing mtime of
/// the edge JSONL source file against the cache file.
fn is_crate_cache_stale(db_path: &Path, mir_edge_dir: &Path, crate_name: &str) -> bool {
    let cache_path = crate_cache_path(db_path, crate_name);
    let edge_file = mir_edge_dir.join(format!("{crate_name}.edges.jsonl"));

    let cache_mtime = match std::fs::metadata(&cache_path).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return true, // no cache file → stale
    };

    let edge_mtime = match std::fs::metadata(&edge_file).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false, // no edge file → cache is still valid (crate wasn't re-analyzed)
    };

    edge_mtime > cache_mtime
}

/// Resolve edges for a single crate's callers and return the edge triples.
fn resolve_crate_edges(
    crate_name: &str,
    mir_edges: &MirEdgeMap,
    loc_to_idx: &HashMap<(&str, usize), u32>,
    name_to_idx: &HashMap<String, u32>,
    chunks: &[ParsedChunk],
) -> Vec<(u32, u32, u32)> {
    let mut edges = Vec::new();
    let callers = mir_edges.callers_for_crate(crate_name);

    for caller_name in &callers {
        let src = resolve_by_loc_or_name(caller_name, loc_to_idx, name_to_idx, chunks);
        let Some(callees) = mir_edges.by_caller.get(*caller_name) else { continue };

        for callee in callees {
            let tgt = if !callee.file.is_empty() && callee.start_line > 0 {
                resolve_by_location(&callee.file, callee.start_line, loc_to_idx, chunks)
            } else {
                None
            }.or_else(|| {
                let lower = callee.name.to_lowercase();
                resolve_mir_name(&lower, name_to_idx)
            });

            if let (Some(s), Some(t)) = (src, tgt) {
                edges.push((s, t, callee.call_line as u32));
            }
        }
    }
    edges
}

/// Incremental MIR edge resolve with per-crate caching.
///
/// Only re-resolves edges for `changed_crates` (or stale crates).
/// Caches store edges as `(src_key, tgt_key, call_line)` using stable
/// identity keys (file:line), so they survive chunk reordering/add/remove.
pub(crate) fn resolve_incremental(
    chunks: &[ParsedChunk],
    index: &ChunkIndex,
    mir_edges: &MirEdgeMap,
    changed_crates: &[String],
    db_path: &Path,
    mir_edge_dir: &Path,
) -> ResolvedEdges {
    let mut adj = ResolvedEdges::new(chunks.len());
    let mut mir_resolved: usize = 0;
    let mut cache_loaded: usize = 0;
    let mut re_resolved_crates: usize = 0;

    // Build lookup tables (same as resolve_with_mir)
    let mut loc_to_idx: HashMap<(&str, usize), u32> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        if let Some((start, _)) = c.lines {
            loc_to_idx.insert((&c.file, start), i as u32);
        }
    }
    let mut name_to_idx: HashMap<String, u32> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        let clean = strip_visibility_prefix(&c.name);
        name_to_idx.insert(clean.to_lowercase(), i as u32);
    }

    // Key-to-index map for applying cached edges (stable across reordering)
    let key_to_idx = build_key_to_idx(chunks);

    // Collect all crate names from MIR edges
    let all_crate_names = mir_edges.crate_names();
    let changed_set: std::collections::HashSet<&str> =
        changed_crates.iter().map(String::as_str).collect();

    for crate_name in &all_crate_names {
        let needs_resolve = changed_set.contains(crate_name)
            || is_crate_cache_stale(db_path, mir_edge_dir, crate_name);

        if needs_resolve {
            // Re-resolve this crate's edges
            let idx_edges = resolve_crate_edges(crate_name, mir_edges, &loc_to_idx, &name_to_idx, chunks);
            re_resolved_crates += 1;
            mir_resolved += idx_edges.len();

            // Convert to key-based edges for cache
            let key_edges: Vec<(String, String, u32)> = idx_edges.iter()
                .map(|&(s, t, line)| {
                    let sk = chunks.get(s as usize).map(chunk_key).unwrap_or_default();
                    let tk = chunks.get(t as usize).map(chunk_key).unwrap_or_default();
                    (sk, tk, line)
                })
                .collect();
            let cache = CrateEdgeCache { edges: key_edges };
            let _ = save_crate_cache(db_path, crate_name, &cache);

            // Apply edges (use current indices)
            for &(s, t, line) in &idx_edges {
                adj.add_edge(s as usize, t, line);
            }
        } else {
            // Load key-based cache and resolve to current indices
            match load_crate_cache(db_path, crate_name) {
                Some(cache) => {
                    for (sk, tk, line) in &cache.edges {
                        if let (Some(&s), Some(&t)) = (key_to_idx.get(sk.as_str()), key_to_idx.get(tk.as_str())) {
                            adj.add_edge(s as usize, t, *line);
                            cache_loaded += 1;
                        }
                        // Silently skip edges whose chunks no longer exist
                    }
                }
                None => {
                    // No cache — resolve fresh
                    let edges = resolve_crate_edges(crate_name, mir_edges, &loc_to_idx, &name_to_idx, chunks);
                    re_resolved_crates += 1;
                    mir_resolved += edges.len();
                    let key_edges: Vec<(String, String, u32)> = edges.iter()
                        .map(|&(s, t, line)| {
                            let sk = chunks.get(s as usize).map(chunk_key).unwrap_or_default();
                            let tk = chunks.get(t as usize).map(chunk_key).unwrap_or_default();
                            (sk, tk, line)
                        })
                        .collect();
                    let _ = save_crate_cache(db_path, crate_name, &CrateEdgeCache { edges: key_edges });
                    for &(s, t, line) in &edges {
                        adj.add_edge(s as usize, t, line);
                    }
                }
            }
        }
    }

    // Type ref edges from chunks (always re-resolved, cheap)
    for (src, chunk) in chunks.iter().enumerate() {
        resolve_type_refs(src, chunk, index, &mut adj);
    }

    eprintln!(
        "      [edge-resolve] incremental: resolved={mir_resolved} cached={cache_loaded} re-resolved_crates={re_resolved_crates}/{}",
        all_crate_names.len()
    );
    adj.dedup();
    adj
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
    // Strip visibility prefixes (e.g. "pub(in foo) bar::baz" → "bar::baz")
    // so MIR names like "bar::baz" match chunks whose name includes visibility.
    let mut name_to_idx: HashMap<String, u32> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        let clean = strip_visibility_prefix(&c.name);
        name_to_idx.insert(clean.to_lowercase(), i as u32);
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

/// Strip visibility prefix from chunk names.
/// `pub(in fusion) fusion::convex::normalize` → `fusion::convex::normalize`
/// `pub(crate) edge_resolve::resolve_with_mir` → `edge_resolve::resolve_with_mir`
fn strip_visibility_prefix(name: &str) -> &str {
    if let Some(rest) = name.strip_prefix("pub(") {
        // Find closing `)` then skip the space after it
        if let Some(close) = rest.find(") ") {
            return &rest[close + 2..];
        }
    }
    if let Some(rest) = name.strip_prefix("pub ") {
        return rest;
    }
    name
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


