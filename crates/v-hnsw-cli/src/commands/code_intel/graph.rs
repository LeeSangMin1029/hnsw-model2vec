//! Call graph adjacency list built from `CodeChunk` data.
//!
//! Provides `CallGraph` — a pre-built, bincode-cached graph that maps
//! chunk indices to their callees and callers for fast BFS traversal.
//!
//! ## HOW TO EXTEND
//! - Add new traversal methods (e.g., shortest path) as `impl CallGraph` methods.
//! - Add new fields to `CallGraph` and update `build()` + bump cache version.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use super::parse::CodeChunk;

/// Cache format version — bump when struct layout changes.
const CACHE_VERSION: u8 = 1;

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
}

impl CallGraph {
    /// Build the call graph from parsed code chunks.
    ///
    /// Resolution strategy: exact match on lowercase name first, then
    /// short name (last `::` segment) fallback.
    pub fn build(chunks: &[CodeChunk]) -> Self {
        let len = chunks.len();

        // Build name -> index maps for call resolution.
        let mut exact: BTreeMap<String, u32> = BTreeMap::new();
        let mut short: BTreeMap<String, u32> = BTreeMap::new();

        let mut names = Vec::with_capacity(len);
        let mut files = Vec::with_capacity(len);
        let mut kinds = Vec::with_capacity(len);
        let mut lines_vec = Vec::with_capacity(len);
        let mut signatures = Vec::with_capacity(len);
        let mut is_test = Vec::with_capacity(len);
        let mut name_index = Vec::with_capacity(len);

        for (i, c) in chunks.iter().enumerate() {
            let idx = i as u32;
            let lower = c.name.to_lowercase();

            exact.insert(lower.clone(), idx);
            if let Some(s) = c.name.rsplit("::").next() {
                short.entry(s.to_lowercase()).or_insert(idx);
            }

            name_index.push((lower, idx));
            names.push(c.name.clone());
            files.push(c.file.clone());
            kinds.push(c.kind.clone());
            lines_vec.push(c.lines);
            signatures.push(c.signature.clone());
            is_test.push(is_test_chunk(c));
        }

        name_index.sort_by(|a, b| a.0.cmp(&b.0));

        // Build adjacency lists.
        let mut callees: Vec<Vec<u32>> = vec![Vec::new(); len];
        let mut callers: Vec<Vec<u32>> = vec![Vec::new(); len];

        for (src, c) in chunks.iter().enumerate() {
            for call in &c.calls {
                if let Some(tgt) = resolve(call, &exact, &short) {
                    let tgt_usize = tgt as usize;
                    if tgt_usize != src {
                        callees[src].push(tgt);
                        callers[tgt_usize].push(src as u32);
                    }
                }
            }
        }

        // Deduplicate.
        for list in &mut callees {
            list.sort_unstable();
            list.dedup();
        }
        for list in &mut callers {
            list.sort_unstable();
            list.dedup();
        }

        Self {
            version: CACHE_VERSION,
            names,
            files,
            kinds,
            lines: lines_vec,
            signatures,
            name_index,
            callees,
            callers,
            is_test,
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
    #[expect(dead_code, reason = "required by clippy::len_without_is_empty")]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn graph_cache_path(db: &Path) -> std::path::PathBuf {
    db.join("cache").join("graph.bin")
}

/// Resolve a call string to a chunk index.
fn resolve(call: &str, exact: &BTreeMap<String, u32>, short: &BTreeMap<String, u32>) -> Option<u32> {
    let lower = call.to_lowercase();

    if let Some(&idx) = exact.get(&lower) {
        return Some(idx);
    }

    // Extract short name: "Foo::bar" -> "bar", "self.method" -> "method"
    let short_name = lower
        .rsplit_once("::")
        .map_or_else(|| lower.rsplit_once('.').map_or(lower.as_str(), |p| p.1), |p| p.1);

    short.get(short_name).copied()
}

/// Determine if a chunk is test code based on file path and name.
fn is_test_chunk(c: &CodeChunk) -> bool {
    c.file.contains("/tests/")
        || c.file.contains("\\tests\\")
        || c.file.ends_with("_test.rs")
        || c.name.starts_with("test_")
        || c.file.contains("/test_")
}
