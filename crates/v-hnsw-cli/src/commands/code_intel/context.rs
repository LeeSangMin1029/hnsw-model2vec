//! `v-hnsw context` — BFS forward traversal from a symbol.
//!
//! Shows the "context" of a symbol: what it calls (callees), what those
//! call, etc., up to a configurable depth. Useful for understanding the
//! dependency cone of a function before modifying it.
//!
//! ## HOW TO EXTEND
//! - Add scoring strategies by modifying `score_fn`.
//! - Add output formats by adding branches in `run_context`.

use std::path::Path;

use anyhow::Result;

use super::graph::CallGraph;
use super::{load_chunks, OutputFormat, cached_json, HasIdx, bfs_generic, BfsDirection, BfsEntryExt, build_bfs_json, print_bfs_grouped};

/// BFS result entry with depth and score.
pub(crate) struct BfsEntry {
    pub(crate) idx: u32,
    pub(crate) depth: u32,
    pub(crate) score: f64,
}

impl HasIdx for BfsEntry {
    fn idx(&self) -> u32 { self.idx }
}

impl BfsEntryExt for BfsEntry {
    fn depth(&self) -> u32 { self.depth }

    fn is_test(&self, graph: &CallGraph) -> bool {
        graph.is_test[self.idx as usize]
    }

    fn extra_json_fields(&self) -> Vec<(&'static str, serde_json::Value)> {
        vec![("sc", serde_json::Value::String(format!("{:.2}", self.score)))]
    }

    fn schema_suffix() -> &'static str { "sc=score" }

    fn extra_text_suffix(&self) -> String {
        format!("  (score={:.2})", self.score)
    }
}

/// Run depth-limited BFS on the callees direction.
pub(crate) fn bfs_forward(graph: &CallGraph, seeds: &[u32], max_depth: u32) -> Vec<BfsEntry> {
    bfs_generic(graph, seeds, max_depth, BfsDirection::Forward, |idx, depth| {
        let is_test = graph.is_test[idx as usize];
        let score = (1.0 / f64::from(depth + 1)) * if is_test { 0.1 } else { 1.0 };
        Some(BfsEntry { idx, depth, score })
    })
}

/// `v-hnsw context <db> <symbol> --depth N --k K [--include-tests] [--detail]`
pub fn run_context(
    db: std::path::PathBuf,
    symbol: String,
    depth: u32,
    k: usize,
    format: OutputFormat,
    include_tests: bool,
    detail: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("context:{symbol}:{depth}:{k}:{include_tests}");
        return cached_json(&db, &key, || compute_context_json(&db, &symbol, depth, k, include_tests));
    }

    let graph = load_or_build_graph(&db)?;
    let seeds = graph.resolve(&symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    let mut entries = bfs_forward(&graph, &seeds, depth);
    if !include_tests {
        entries.retain(|e| !graph.is_test[e.idx as usize]);
    }
    entries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(k);

    println!("Context of \"{symbol}\" (depth={depth}, top {k}):\n");
    print_bfs_grouped(&graph, &entries);

    if detail {
        super::print_detail_annotations(&db, &graph, &entries);
    }

    Ok(())
}

fn compute_context_json(db: &Path, symbol: &str, depth: u32, k: usize, include_tests: bool) -> Result<String> {
    let graph = load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);

    let mut entries = bfs_forward(&graph, &seeds, depth);
    if !include_tests {
        entries.retain(|e| !graph.is_test[e.idx as usize]);
    }
    entries.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    entries.truncate(k);

    let json = build_bfs_json(&graph, &entries);
    Ok(serde_json::to_string(&json)?)
}

// ── Shared helpers ───────────────────────────────────────────────────────

/// Load graph from cache or build from chunks.
pub(crate) fn load_or_build_graph(db: &Path) -> Result<CallGraph> {
    if let Some(g) = CallGraph::load(db) {
        return Ok(g);
    }

    let chunks = load_chunks(db)?;
    let g = CallGraph::build(&chunks);
    let _ = g.save(db);
    Ok(g)
}


