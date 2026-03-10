//! JSON-returning functions for daemon/handler reuse.
//!
//! Each function returns a `serde_json::Value` without printing,
//! suitable for embedding in JSON-RPC responses.

use std::path::Path;

use anyhow::Result;

use super::{
    build_bfs_json, build_stats, grouped_json, grouped_json_refs,
    gather, load_chunks, load_or_build_graph, reason, stats_to_json, trace,
    find_refs, print_grouped, CodeChunk,
};
use super::commands::compute_impact_entries;

/// Return stats as a JSON value without printing.
pub fn stats_as_json(db: &Path) -> Result<serde_json::Value> {
    let chunks = load_chunks(db)?;
    let stats = build_stats(&chunks);
    Ok(stats_to_json(&stats))
}

/// Return definition lookup as a JSON value without printing.
pub fn def_as_json(db: &Path, name: &str) -> Result<serde_json::Value> {
    let name_lower = name.to_lowercase();
    let chunks = load_chunks(db)?;
    let matches: Vec<&CodeChunk> = chunks
        .iter()
        .filter(|c| {
            c.name.to_lowercase() == name_lower
                || c.name.to_lowercase().ends_with(&format!("::{name_lower}"))
        })
        .collect();
    Ok(grouped_json(&matches))
}

/// Return references lookup as a JSON value without printing.
pub fn refs_as_json(db: &Path, name: &str) -> Result<serde_json::Value> {
    let chunks = load_chunks(db)?;
    let refs = find_refs(&chunks, name);
    Ok(grouped_json_refs(&refs))
}

/// Return symbols listing as a JSON value without printing.
pub fn symbols_as_json(
    db: &Path,
    name: Option<&str>,
    kind: Option<&str>,
) -> Result<serde_json::Value> {
    let chunks = load_chunks(db)?;
    let filtered: Vec<&CodeChunk> = chunks
        .iter()
        .filter(|c| {
            if let Some(n) = name
                && !c.name.to_lowercase().contains(&n.to_lowercase())
            {
                return false;
            }
            if let Some(k) = kind
                && c.kind.to_lowercase() != k.to_lowercase()
            {
                return false;
            }
            true
        })
        .collect();
    Ok(grouped_json(&filtered))
}

/// Return gather results as a JSON value without printing.
pub fn gather_as_json(
    db: &Path,
    symbol: &str,
    depth: u32,
    k: usize,
    include_tests: bool,
) -> Result<serde_json::Value> {
    let graph = load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    if seeds.is_empty() {
        return Ok(serde_json::json!({"results": [], "message": format!("No symbol found matching \"{symbol}\"")}));
    }
    let forward = gather::bfs_directed(&graph, &seeds, depth, include_tests, gather::Direction::Forward);
    let reverse = gather::bfs_directed(&graph, &seeds, depth, include_tests, gather::Direction::Reverse);
    let mut entries = gather::merge_entries(forward, reverse);
    entries.truncate(k);
    Ok(build_bfs_json(&graph, &entries))
}

/// Return impact results as a JSON value without printing.
pub fn impact_as_json(
    db: &Path,
    symbol: &str,
    depth: u32,
    include_tests: bool,
) -> Result<serde_json::Value> {
    let graph = load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    if seeds.is_empty() {
        return Ok(serde_json::json!({"results": [], "message": format!("No symbol found matching \"{symbol}\"")}));
    }
    let entries = compute_impact_entries(&graph, &seeds, depth, include_tests);
    Ok(build_bfs_json(&graph, &entries))
}

/// Return trace results as a JSON value without printing.
pub fn trace_as_json(db: &Path, from: &str, to: &str) -> Result<serde_json::Value> {
    let graph = load_or_build_graph(db)?;
    let sources = graph.resolve(from);
    let targets = graph.resolve(to);
    if sources.is_empty() {
        return Ok(serde_json::json!({"path": null, "message": format!("No symbol found matching \"{from}\"")}));
    }
    if targets.is_empty() {
        return Ok(serde_json::json!({"path": null, "message": format!("No symbol found matching \"{to}\"")}));
    }
    match trace::bfs_shortest_path(&graph, &sources, &targets) {
        Some(path) => Ok(trace::build_json(&graph, &path)),
        None => Ok(serde_json::json!({"path": null, "hops": null})),
    }
}

/// Try inline code-intel lookup for find command.
///
/// Attempts def match → callers, then symbol partial match.
/// Returns `true` if results were found and printed.
pub(crate) fn try_inline_lookup(db: &Path, query: &str) -> Result<bool> {
    let chunks = load_chunks(db)?;
    let query_lower = query.to_lowercase();

    // Try def first
    let defs: Vec<&CodeChunk> = chunks
        .iter()
        .filter(|c| {
            c.name.to_lowercase() == query_lower
                || c.name.to_lowercase().ends_with(&format!("::{query_lower}"))
        })
        .collect();

    if !defs.is_empty() {
        eprintln!("[code-intel] Found {} definition(s) for \"{query}\"", defs.len());
        println!("Definition of \"{query}\":\n");
        print_grouped(&defs, None);

        // Also show callers
        let callers: Vec<&CodeChunk> = chunks
            .iter()
            .filter(|c| {
                c.calls.iter().any(|call| {
                    let call_lower = call.to_lowercase();
                    call_lower == query_lower
                        || call_lower.ends_with(&format!("::{query_lower}"))
                        || call_lower.contains(&query_lower)
                })
            })
            .collect();
        if !callers.is_empty() {
            println!("\n{} caller(s):\n", callers.len());
            print_grouped(&callers, None);
        }
        return Ok(true);
    }

    // Try symbol name search (partial match)
    let symbols: Vec<&CodeChunk> = chunks
        .iter()
        .filter(|c| c.name.to_lowercase().contains(&query_lower))
        .collect();

    if !symbols.is_empty() {
        eprintln!("[code-intel] Found {} symbol(s) matching \"{query}\"", symbols.len());
        println!("Symbols matching \"{query}\":\n");
        print_grouped(&symbols, None);
        return Ok(true);
    }

    Ok(false)
}

/// Return detail/reason as a JSON value without printing.
pub fn detail_as_json(db: &Path, symbol: &str) -> Result<serde_json::Value> {
    match reason::load_reason(db, symbol)? {
        Some(entry) => Ok(serde_json::to_value(entry)?),
        None => Ok(serde_json::json!({"symbol": symbol, "found": false})),
    }
}
