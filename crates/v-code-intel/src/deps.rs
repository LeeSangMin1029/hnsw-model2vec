//! Bidirectional file-level dependency graph.
//!
//! Analyses `calls` and `types` fields to build a complete dependency graph
//! showing both outgoing (uses) and incoming (used by) edges per file.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::parse::ParsedChunk;

/// A dependency edge: target file + relationship type.
pub type DepSet = BTreeSet<(String, &'static str)>;

/// Complete bidirectional graph for a single file.
pub struct FileNode {
    /// Files this file depends on (outgoing).
    pub outgoing: DepSet,
    /// Files that depend on this file (incoming).
    pub incoming: DepSet,
}

/// Full bidirectional dependency graph.
pub struct DepGraph {
    pub nodes: BTreeMap<String, FileNode>,
}

impl DepGraph {
    /// Build from code chunks.
    pub fn build(chunks: &[ParsedChunk]) -> Self {
        // Build symbol -> file index.
        let mut sym_to_file: BTreeMap<String, String> = BTreeMap::new();
        for c in chunks {
            sym_to_file.insert(c.name.to_lowercase(), c.file.clone());
            if let Some(short) = c.name.rsplit("::").next() {
                sym_to_file
                    .entry(short.to_lowercase())
                    .or_insert_with(|| c.file.clone());
            }
        }

        let mut nodes: BTreeMap<String, FileNode> = BTreeMap::new();

        for c in chunks {
            let src = &c.file;

            // Ensure source file has an entry.
            nodes.entry(src.clone()).or_insert_with(|| FileNode {
                outgoing: DepSet::new(),
                incoming: DepSet::new(),
            });

            // Resolve calls and types -> bidirectional edges.
            add_edges(&mut nodes, src, &c.calls, &sym_to_file, "calls");
            add_edges(&mut nodes, src, &c.types, &sym_to_file, "types");
        }

        Self { nodes }
    }

    /// Find a file by suffix match.
    pub fn match_file(&self, query: &str) -> Vec<&str> {
        self.nodes
            .keys()
            .filter(|k| k.ends_with(query) || *k == query)
            .map(String::as_str)
            .collect()
    }

    /// Count total edges (outgoing only, to avoid double-counting).
    pub fn total_edges(&self) -> usize {
        self.nodes.values().map(|n| n.outgoing.len()).sum()
    }
}

/// Add bidirectional edges for a list of symbols (calls or types).
fn add_edges(
    nodes: &mut BTreeMap<String, FileNode>,
    src: &str,
    symbols: &[String],
    sym_to_file: &BTreeMap<String, String>,
    label: &'static str,
) {
    for sym in symbols {
        if let Some(target) = resolve_symbol(sym, sym_to_file)
            && target != *src
        {
            nodes
                .entry(src.to_owned())
                .or_insert_with(|| FileNode {
                    outgoing: DepSet::new(),
                    incoming: DepSet::new(),
                })
                .outgoing
                .insert((target.clone(), label));

            nodes
                .entry(target)
                .or_insert_with(|| FileNode {
                    outgoing: DepSet::new(),
                    incoming: DepSet::new(),
                })
                .incoming
                .insert((src.to_owned(), label));
        }
    }
}

/// Resolve a symbol name to its defining file.
pub fn resolve_symbol(symbol: &str, index: &BTreeMap<String, String>) -> Option<String> {
    let lower = symbol.to_lowercase();

    if let Some(f) = index.get(&lower) {
        return Some(f.clone());
    }

    let short = lower
        .rsplit_once("::")
        .map_or_else(|| lower.rsplit_once('.').map_or(lower.as_str(), |p| p.1), |p| p.1);

    index.get(short).cloned()
}

/// Merge dep edges by file, joining via labels.
pub fn merge_deps(deps: &DepSet) -> BTreeMap<&str, Vec<&str>> {
    let mut merged: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (file, via) in deps {
        merged.entry(file.as_str()).or_default().push(via);
    }
    merged
}

/// Collect transitive files reachable from `start` up to `depth` levels.
/// Traverses BOTH directions.
pub fn collect_transitive_files(graph: &DepGraph, start: &str, depth: usize) -> BTreeSet<String> {
    let mut visited = BTreeSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    for file in graph.match_file(start) {
        queue.push_back((file.to_owned(), 0));
    }

    while let Some((file, level)) = queue.pop_front() {
        if visited.contains(&file) {
            continue;
        }
        visited.insert(file.clone());

        // Only expand neighbors if we haven't reached the depth limit.
        if level < depth
            && let Some(node) = graph.nodes.get(&file) {
                for (target, _) in &node.outgoing {
                    if !visited.contains(target) {
                        queue.push_back((target.clone(), level + 1));
                    }
                }
                for (source, _) in &node.incoming {
                    if !visited.contains(source) {
                        queue.push_back((source.clone(), level + 1));
                    }
                }
            }
    }

    visited
}

/// Extract crate name from path for color grouping.
pub fn crate_group(path: &str) -> &str {
    if let Some(start) = path.find("crates/") {
        let rest = &path[start + 7..];
        if let Some(end) = rest.find('/') {
            return &path[start..start + 7 + end];
        }
    }
    if let Some(start) = path.find("src/") {
        return &path[..start + 3];
    }
    path
}

/// Find the length of the common path prefix among all file paths.
pub fn common_prefix_len(paths: &[&str]) -> usize {
    if paths.is_empty() {
        return 0;
    }
    let first = paths[0].as_bytes();
    let mut len = first.len();
    for p in &paths[1..] {
        let b = p.as_bytes();
        len = len.min(b.len());
        for i in 0..len {
            if first[i] != b[i] {
                len = i;
                break;
            }
        }
    }
    // Snap back to last '/' boundary.
    let first_str = &paths[0][..len];
    first_str.rfind('/').map_or(0, |i| i + 1)
}
