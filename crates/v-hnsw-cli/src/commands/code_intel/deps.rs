//! `v-hnsw deps` — bidirectional file-level dependency graph.
//!
//! Analyses `calls` and `types` fields to build a complete dependency graph
//! showing both outgoing (→ uses) and incoming (← used by) edges per file.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use anyhow::Result;

use super::parse::CodeChunk;
use super::{cached_json, load_chunks, OutputFormat};

/// Schema descriptor for deps JSON output.
const DEPS_SCHEMA: &str = "o=outgoing(uses),i=incoming(used_by),n=name,v=via";

/// A dependency edge: target file + relationship type.
type DepSet = BTreeSet<(String, &'static str)>;

/// Complete bidirectional graph for a single file.
struct FileNode {
    /// Files this file depends on (outgoing).
    outgoing: DepSet,
    /// Files that depend on this file (incoming).
    incoming: DepSet,
}

/// Full bidirectional dependency graph.
struct DepGraph {
    nodes: BTreeMap<String, FileNode>,
}

impl DepGraph {
    /// Build from code chunks.
    fn build(chunks: &[CodeChunk]) -> Self {
        // Build symbol → file index.
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

            // Resolve calls → outgoing edges.
            for call in &c.calls {
                if let Some(target) = resolve_symbol(call, &sym_to_file)
                    && target != *src {
                        nodes
                            .entry(src.clone())
                            .or_insert_with(|| FileNode {
                                outgoing: DepSet::new(),
                                incoming: DepSet::new(),
                            })
                            .outgoing
                            .insert((target.clone(), "calls"));

                        // Reverse edge: target ← src.
                        nodes
                            .entry(target)
                            .or_insert_with(|| FileNode {
                                outgoing: DepSet::new(),
                                incoming: DepSet::new(),
                            })
                            .incoming
                            .insert((src.clone(), "calls"));
                    }
            }

            // Resolve types → outgoing edges.
            for ty in &c.types {
                if let Some(target) = resolve_symbol(ty, &sym_to_file)
                    && target != *src {
                        nodes
                            .entry(src.clone())
                            .or_insert_with(|| FileNode {
                                outgoing: DepSet::new(),
                                incoming: DepSet::new(),
                            })
                            .outgoing
                            .insert((target.clone(), "types"));

                        nodes
                            .entry(target)
                            .or_insert_with(|| FileNode {
                                outgoing: DepSet::new(),
                                incoming: DepSet::new(),
                            })
                            .incoming
                            .insert((src.clone(), "types"));
                    }
            }
        }

        Self { nodes }
    }

    /// Find a file by suffix match.
    fn match_file(&self, query: &str) -> Vec<&str> {
        self.nodes
            .keys()
            .filter(|k| k.ends_with(query) || *k == query)
            .map(String::as_str)
            .collect()
    }

    /// Count total edges (outgoing only, to avoid double-counting).
    fn total_edges(&self) -> usize {
        self.nodes.values().map(|n| n.outgoing.len()).sum()
    }
}

/// Run the `deps` subcommand.
pub fn run_deps(
    db: std::path::PathBuf,
    file: Option<String>,
    format: OutputFormat,
    depth: usize,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("deps:{}:{depth}", file.as_deref().unwrap_or("*"));
        return cached_json(&db, &key, || compute_deps_json(&db, file.as_deref(), depth));
    }

    let chunks = load_chunks(&db)?;
    let graph = DepGraph::build(&chunks);

    if matches!(format, OutputFormat::Html) {
        let html = generate_func_html(&chunks);
        let out_path = db.parent().unwrap_or(Path::new(".")).join("deps-graph.html");
        std::fs::write(&out_path, html)?;
        println!("Graph written to {}", out_path.display());
        return Ok(());
    }

    if let Some(ref target) = file {
        print_file_deps(&graph, target, depth);
    } else {
        print_full_graph(&graph, depth);
    }
    Ok(())
}

/// Resolve a symbol name to its defining file.
fn resolve_symbol(symbol: &str, index: &BTreeMap<String, String>) -> Option<String> {
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
fn merge_deps(deps: &DepSet) -> BTreeMap<&str, Vec<&str>> {
    let mut merged: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (file, via) in deps {
        merged.entry(file.as_str()).or_default().push(via);
    }
    merged
}

/// Print a single file node (both directions).
fn print_node(file: &str, node: &FileNode) {
    let out_count = node.outgoing.len();
    let in_count = node.incoming.len();
    println!("{file}  (→{out_count} ←{in_count})");

    if !node.outgoing.is_empty() {
        for (dep, vias) in &merge_deps(&node.outgoing) {
            let via_str = vias.join(", ");
            println!("  → {dep} (via {via_str})");
        }
    }
    if !node.incoming.is_empty() {
        for (dep, vias) in &merge_deps(&node.incoming) {
            let via_str = vias.join(", ");
            println!("  ← {dep} (via {via_str})");
        }
    }
}

/// Collect transitive files reachable from `start` up to `depth` levels.
/// Traverses BOTH directions.
fn collect_transitive_files(graph: &DepGraph, start: &str, depth: usize) -> BTreeSet<String> {
    let mut visited = BTreeSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();

    for file in graph.match_file(start) {
        queue.push_back((file.to_owned(), 0));
    }

    while let Some((file, level)) = queue.pop_front() {
        if visited.contains(&file) || level >= depth {
            continue;
        }
        visited.insert(file.clone());

        if let Some(node) = graph.nodes.get(&file) {
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

/// Print deps for a single file (bidirectional).
fn print_file_deps(graph: &DepGraph, target: &str, depth: usize) {
    let files = collect_transitive_files(graph, target, depth);

    if files.is_empty() {
        println!("No dependencies found for \"{target}\".");
        return;
    }

    for file in &files {
        if let Some(node) = graph.nodes.get(file) {
            print_node(file, node);
        }
    }
}

/// Print the full dependency graph (all files, both directions).
fn print_full_graph(graph: &DepGraph, _depth: usize) {
    if graph.nodes.is_empty() {
        println!("No dependencies found.");
        return;
    }

    let total_files = graph.nodes.len();
    let total_edges = graph.total_edges();
    println!("{total_files} files, {total_edges} dependency edges:\n");

    for (file, node) in &graph.nodes {
        if node.outgoing.is_empty() && node.incoming.is_empty() {
            continue;
        }
        print_node(file, node);
    }
}

/// Compute deps JSON output (bidirectional).
fn compute_deps_json(db: &Path, file: Option<&str>, depth: usize) -> Result<String> {
    let chunks = load_chunks(db)?;
    let graph = DepGraph::build(&chunks);

    let files: BTreeSet<String> = if let Some(target) = file {
        collect_transitive_files(&graph, target, depth)
    } else {
        graph.nodes.keys().cloned().collect()
    };

    let mut map = serde_json::Map::new();
    map.insert(
        "_s".to_owned(),
        serde_json::Value::String(DEPS_SCHEMA.to_owned()),
    );

    for f in &files {
        if let Some(node) = graph.nodes.get(f) {
            if node.outgoing.is_empty() && node.incoming.is_empty() {
                continue;
            }

            let outgoing: Vec<serde_json::Value> = merge_deps(&node.outgoing)
                .into_iter()
                .map(|(dep, vias)| serde_json::json!({"n": dep, "v": vias}))
                .collect();

            let incoming: Vec<serde_json::Value> = merge_deps(&node.incoming)
                .into_iter()
                .map(|(dep, vias)| serde_json::json!({"n": dep, "v": vias}))
                .collect();

            map.insert(
                f.clone(),
                serde_json::json!({"o": outgoing, "i": incoming}),
            );
        }
    }

    Ok(serde_json::to_string(&serde_json::Value::Object(map))?)
}

// ── HTML visualization ──────────────────────────────────────────────────

/// Extract crate name from path for color grouping.
fn crate_group(path: &str) -> &str {
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

/// Build function-level call graph data for HTML visualization.
///
/// Compact format: short keys (`n`=name, `f`=file, `k`=kind, `s`=sig,
/// `l`=lines, `g`=group), common path prefix stripped, isolated nodes removed.
/// Returns `(nodes_json, links_json, groups_json)`.
fn build_func_graph_json(chunks: &[CodeChunk]) -> (String, String, String) {
    // Build name → chunk-index map for call resolution.
    let mut name_to_idx: BTreeMap<String, usize> = BTreeMap::new();
    let mut short_to_idx: BTreeMap<String, usize> = BTreeMap::new();

    for (i, c) in chunks.iter().enumerate() {
        name_to_idx.insert(c.name.to_lowercase(), i);
        if let Some(short) = c.name.rsplit("::").next() {
            short_to_idx.entry(short.to_lowercase()).or_insert(i);
        }
    }

    let resolve = |call: &str| -> Option<usize> {
        let lower = call.to_lowercase();
        if let Some(&idx) = name_to_idx.get(&lower) {
            return Some(idx);
        }
        let short = lower
            .rsplit_once("::")
            .map_or_else(|| lower.rsplit_once('.').map_or(lower.as_str(), |p| p.1), |p| p.1);
        short_to_idx.get(short).copied()
    };

    // Build raw edges first to identify connected nodes.
    let mut edges: Vec<(usize, usize, &'static str)> = Vec::new();
    let mut connected: BTreeSet<usize> = BTreeSet::new();

    for (src, c) in chunks.iter().enumerate() {
        for call in &c.calls {
            if let Some(tgt) = resolve(call)
                && tgt != src {
                    edges.push((src, tgt, "c"));
                    connected.insert(src);
                    connected.insert(tgt);
                }
        }
        for ty in &c.types {
            if let Some(tgt) = resolve(ty)
                && tgt != src {
                    edges.push((src, tgt, "t"));
                    connected.insert(src);
                    connected.insert(tgt);
                }
        }
    }

    // Build compact index: old chunk idx → new compact idx.
    let connected_vec: Vec<usize> = connected.into_iter().collect();
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    for (new_idx, &old_idx) in connected_vec.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }

    // Strip common path prefix from all files.
    let files: Vec<&str> = connected_vec
        .iter()
        .map(|&i| chunks[i].file.as_str())
        .collect();
    let prefix_len = common_prefix_len(&files);

    // Groups from connected nodes only.
    let mut groups_set: BTreeSet<String> = BTreeSet::new();
    for &i in &connected_vec {
        let f = &chunks[i].file[prefix_len..];
        groups_set.insert(crate_group(f).to_owned());
    }
    let group_list: Vec<String> = groups_set.into_iter().collect();

    // Build compact nodes: {n,f,k,s,l,g}
    let esc = |s: &str| s.replace('\\', "/").replace('"', r#"\""#);
    let mut nodes_parts = Vec::with_capacity(connected_vec.len());
    for &old in &connected_vec {
        let c = &chunks[old];
        let file = esc(&c.file[prefix_len..]);
        let name = esc(&c.name);
        let kind = &c.kind;
        let sig = c.signature.as_deref().map_or(String::new(), |s| {
            let s = esc(s);
            if s.len() > 80 { format!("{}...", &s[..77]) } else { s }
        });
        let lines = c.lines.map_or(String::new(), |(s, e)| format!("{s}-{e}"));
        let g = group_list
            .iter()
            .position(|grp| grp == crate_group(&c.file[prefix_len..]))
            .unwrap_or(0);

        nodes_parts.push(format!(
            r#"{{"n":"{name}","f":"{file}","k":"{kind}","s":"{sig}","l":"{lines}","g":{g}}}"#,
        ));
    }

    // Build compact links: [src,tgt,"c"|"t"]
    let mut links_parts = Vec::with_capacity(edges.len());
    for (src, tgt, via) in &edges {
        if let (Some(&s), Some(&t)) = (old_to_new.get(src), old_to_new.get(tgt)) {
            links_parts.push(format!(r#"[{s},{t},"{via}"]"#));
        }
    }

    let groups_json: Vec<String> = group_list
        .iter()
        .map(|g| format!(r#""{}""#, esc(g)))
        .collect();

    (
        nodes_parts.join(","),
        links_parts.join(","),
        groups_json.join(","),
    )
}

/// Find the length of the common path prefix among all file paths.
fn common_prefix_len(paths: &[&str]) -> usize {
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

/// Generate function-level call graph HTML from raw chunks.
fn generate_func_html(chunks: &[CodeChunk]) -> String {
    let (nodes_json, links_json, groups_json) = build_func_graph_json(chunks);
    super::deps_html::render(&nodes_json, &links_json, &groups_json)
}
