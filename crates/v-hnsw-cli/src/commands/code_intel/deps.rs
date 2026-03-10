//! `v-hnsw deps` — bidirectional file-level dependency graph.
//!
//! CLI command handler for the deps subcommand. Analysis logic lives in
//! `v_hnsw_intel::deps`.

use std::path::Path;

use anyhow::Result;

use super::{cached_json, load_chunks, OutputFormat};
pub use v_hnsw_intel::deps::{DepGraph, collect_transitive_files, merge_deps, crate_group, common_prefix_len};
// Re-exported for tests only.
#[cfg(test)]
pub use v_hnsw_intel::deps::{DepSet, resolve_symbol};
use v_hnsw_intel::parse::CodeChunk;

use std::collections::{BTreeMap, BTreeSet};

/// Schema descriptor for deps JSON output.
const DEPS_SCHEMA: &str = "o=outgoing(uses),i=incoming(used_by),n=name,v=via";

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

/// Print a single file node (both directions).
fn print_node(file: &str, node: &v_hnsw_intel::deps::FileNode) {
    let out_count = node.outgoing.len();
    let in_count = node.incoming.len();
    println!("{file}  (\u{2192}{out_count} \u{2190}{in_count})");

    if !node.outgoing.is_empty() {
        for (dep, vias) in &merge_deps(&node.outgoing) {
            let via_str = vias.join(", ");
            println!("  \u{2192} {dep} (via {via_str})");
        }
    }
    if !node.incoming.is_empty() {
        for (dep, vias) in &merge_deps(&node.incoming) {
            let via_str = vias.join(", ");
            println!("  \u{2190} {dep} (via {via_str})");
        }
    }
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

/// Build function-level call graph data for HTML visualization.
///
/// Compact format: short keys (`n`=name, `f`=file, `k`=kind, `s`=sig,
/// `l`=lines, `g`=group), common path prefix stripped, isolated nodes removed.
/// Returns `(nodes_json, links_json, groups_json)`.
fn build_func_graph_json(chunks: &[CodeChunk]) -> (String, String, String) {
    // Build name -> chunk-index map for call resolution.
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

    // Build compact index: old chunk idx -> new compact idx.
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

/// Generate function-level call graph HTML from raw chunks.
fn generate_func_html(chunks: &[CodeChunk]) -> String {
    let (nodes_json, links_json, groups_json) = build_func_graph_json(chunks);
    super::deps_html::render(&nodes_json, &links_json, &groups_json)
}
