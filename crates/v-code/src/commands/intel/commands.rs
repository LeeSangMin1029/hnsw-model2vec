//! CLI command handlers for code-intel subcommands.
//!
//! Each `run_*` function corresponds to a CLI subcommand (stats, symbols,
//! def, refs, gather, impact, trace). These handle text/JSON output and
//! caching; pure analysis logic lives in `v-code-intel`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;

use super::{
    OutputFormat, cached_json, find_refs, parent_dir, file_name, print_detail_annotations,
    print_grouped, print_grouped_compact,
};
use super::{
    build_bfs_json, print_bfs_grouped, BfsEntryExt, HasIdx,
    build_stats, stats_to_json, load_chunks, load_or_build_graph,
    format_lines_opt, grouped_json, grouped_json_refs,
    gather, graph, impact, trace, CodeChunk,
};

// ── Commands ─────────────────────────────────────────────────────────────

/// `v-hnsw stats` — per-crate summary of code symbols.
pub fn run_stats(db: PathBuf, format: OutputFormat) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        return cached_json(&db, "stats", || compute_stats_json(&db));
    }
    let chunks = load_chunks(&db)?;
    let stats = build_stats(&chunks);
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8}",
        "crate", "prod_fn", "test_fn", "struct", "enum"
    );
    println!("{}", "-".repeat(60));
    let mut totals = [0usize; 4];
    for (name, row) in &stats {
        println!(
            "{:<24} {:>8} {:>8} {:>8} {:>8}",
            name, row[0], row[1], row[2], row[3]
        );
        for (i, v) in row.iter().enumerate() {
            totals[i] += v;
        }
    }
    println!("{}", "-".repeat(60));
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8}",
        "total", totals[0], totals[1], totals[2], totals[3]
    );
    Ok(())
}

fn compute_stats_json(db: &std::path::Path) -> Result<String> {
    let chunks = load_chunks(db)?;
    let stats = build_stats(&chunks);
    Ok(serde_json::to_string(&stats_to_json(&stats))?)
}

/// `v-hnsw symbols` — list symbols matching filters.
pub fn run_symbols(
    db: PathBuf,
    name: Option<String>,
    kind: Option<String>,
    format: OutputFormat,
    include_tests: bool,
    limit: Option<usize>,
    compact: bool,
) -> Result<()> {
    let key = format!(
        "symbols:{}:{}:{}:{}",
        name.as_deref().unwrap_or(""),
        kind.as_deref().unwrap_or(""),
        if include_tests { "t" } else { "" },
        limit.map_or(String::new(), |l| l.to_string()),
    );
    run_chunk_query(&db, format, &key,
        |c| {
            if let Some(ref n) = name
                && !c.name.to_lowercase().contains(&n.to_lowercase()) { return false; }
            if let Some(ref k) = kind
                && c.kind.to_lowercase() != k.to_lowercase() { return false; }
            if !include_tests && v_code_intel::graph::is_test_chunk(c) { return false; }
            true
        },
        "No symbols found.",
        |n| format!("{n} symbols found:\n"),
        limit,
        compact,
    )
}

/// `v-hnsw def` — find definition location of a symbol.
pub fn run_def(db: PathBuf, name: String, format: OutputFormat, compact: bool) -> Result<()> {
    let key = format!("def:{name}");
    let name_lower = name.to_lowercase();
    run_chunk_query(&db, format, &key,
        |c| {
            c.name.to_lowercase() == name_lower
                || c.name.to_lowercase().ends_with(&format!("::{name_lower}"))
        },
        &format!("No definition found for \"{name}\"."),
        |_| format!("Definition of \"{name}\":\n"),
        None,
        compact,
    )
}

/// `v-hnsw refs` — find all references to a symbol.
pub fn run_refs(db: PathBuf, name: String, format: OutputFormat, compact: bool) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("refs:{name}");
        return cached_json(&db, &key, || {
            let chunks = load_chunks(&db)?;
            let refs = find_refs(&chunks, &name);
            Ok(serde_json::to_string(&grouped_json_refs(&refs))?)
        });
    }
    // Text path — no caching needed.
    let chunks = load_chunks(&db)?;
    let refs = find_refs(&chunks, &name);
    if refs.is_empty() {
        println!("No references found for \"{name}\".");
        return Ok(());
    }
    let mut groups: BTreeMap<String, Vec<(&CodeChunk, &[&str])>> = BTreeMap::new();
    for (chunk, via) in &refs {
        let dir = parent_dir(&chunk.file);
        groups.entry(dir).or_default().push((chunk, via));
    }
    println!("{} references to \"{name}\":\n", refs.len());
    for (dir, items) in &groups {
        println!("  {dir}/");
        for (c, via) in items {
            let filename = file_name(&c.file);
            let lines = format_lines_opt(c.lines);
            if compact {
                println!("    {filename}{lines}  [{kind}] {name}",
                    kind = c.kind, name = c.name);
            } else {
                let via_str = via.join(", ");
                println!("    {filename}{lines}  [{kind}] {name} (via {via_str})",
                    kind = c.kind, name = c.name);
            }
        }
        if !compact { println!(); }
    }
    Ok(())
}

/// `v-hnsw gather <db> <symbol> --depth N --k K [--include-tests] [--detail]`
pub fn run_gather(
    db: PathBuf,
    symbol: String,
    depth: u32,
    k: usize,
    format: OutputFormat,
    include_tests: bool,
    detail: bool,
) -> Result<()> {
    run_bfs_command(&db, &symbol, format, detail, &format!("gather:{symbol}:{depth}:{k}:{include_tests}"),
        &format!("Gather for \"{symbol}\" (depth={depth}, top {k}):\n"),
        |graph, seeds| {
            let forward = gather::bfs_directed(graph, seeds, depth, include_tests, gather::Direction::Forward);
            let reverse = gather::bfs_directed(graph, seeds, depth, include_tests, gather::Direction::Reverse);
            let mut entries = gather::merge_entries(forward, reverse);
            entries.truncate(k);
            entries
        },
    )
}

/// `v-hnsw impact <db> <symbol> --depth N [--include-tests] [--detail]`
pub fn run_impact(
    db: PathBuf,
    symbol: String,
    depth: u32,
    format: OutputFormat,
    include_tests: bool,
    detail: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("impact:{symbol}:{depth}:{include_tests}");
        return cached_json(&db, &key, || {
            let graph = load_or_build_graph(&db, None)?;
            let seeds = graph.resolve(&symbol);
            let entries = compute_impact_entries(&graph, &seeds, depth, include_tests);
            let json = build_bfs_json(&graph, &entries);
            Ok(serde_json::to_string(&json)?)
        });
    }

    let graph = load_or_build_graph(&db, None)?;
    let seeds = graph.resolve(&symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    let all_entries = impact::bfs_reverse(&graph, &seeds, depth);
    let prod_count = all_entries.iter().filter(|e| e.depth > 0 && !e.is_test).count();
    let test_count = all_entries.iter().filter(|e| e.depth > 0 && e.is_test).count();

    let entries = filter_test_entries(all_entries, include_tests);

    println!("Impact of \"{symbol}\" (depth={depth}):");
    println!("  {prod_count} production callers, {test_count} test callers\n");
    print_bfs_grouped(&graph, &entries);

    if detail {
        print_detail_annotations(&db, &graph, &entries);
    }

    Ok(())
}

pub(super) fn compute_impact_entries(
    graph: &graph::CallGraph,
    seeds: &[u32],
    depth: u32,
    include_tests: bool,
) -> Vec<impact::BfsEntry> {
    let all = impact::bfs_reverse(graph, seeds, depth);
    filter_test_entries(all, include_tests)
}

fn filter_test_entries(entries: Vec<impact::BfsEntry>, include_tests: bool) -> Vec<impact::BfsEntry> {
    if include_tests {
        entries
    } else {
        entries.into_iter().filter(|e| !e.is_test).collect()
    }
}

/// `v-hnsw trace <db> <from> <to>`
pub fn run_trace(
    db: PathBuf,
    from: String,
    to: String,
    format: OutputFormat,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("trace:{from}:{to}");
        return cached_json(&db, &key, || {
            let graph = load_or_build_graph(&db, None)?;
            let sources = graph.resolve(&from);
            let targets = graph.resolve(&to);
            let json = match trace::bfs_shortest_path(&graph, &sources, &targets) {
                Some(path) => trace::build_json(&graph, &path),
                None => serde_json::json!({ "path": null, "hops": null }),
            };
            Ok(serde_json::to_string(&json)?)
        });
    }

    let graph = load_or_build_graph(&db, None)?;
    let sources = graph.resolve(&from);
    let targets = graph.resolve(&to);

    if sources.is_empty() {
        println!("No symbol found matching \"{from}\".");
        return Ok(());
    }
    if targets.is_empty() {
        println!("No symbol found matching \"{to}\".");
        return Ok(());
    }

    match trace::bfs_shortest_path(&graph, &sources, &targets) {
        Some(path) => {
            println!("Call path from \"{from}\" to \"{to}\" ({} hops):\n", path.len() - 1);
            print_trace_path(&graph, &path);
        }
        None => {
            println!("No call path found from \"{from}\" to \"{to}\".");
        }
    }

    Ok(())
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Shared runner for chunk-filter commands (symbols, def).
fn run_chunk_query(
    db: &std::path::Path,
    format: OutputFormat,
    cache_key: &str,
    filter: impl Fn(&CodeChunk) -> bool,
    empty_msg: &str,
    header: impl FnOnce(usize) -> String,
    limit: Option<usize>,
    compact: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        return cached_json(db, cache_key, || {
            let chunks = load_chunks(db)?;
            let mut filtered: Vec<&CodeChunk> = chunks.iter().filter(|c| filter(c)).collect();
            if let Some(n) = limit { filtered.truncate(n); }
            Ok(serde_json::to_string(&grouped_json(&filtered))?)
        });
    }
    let chunks = load_chunks(db)?;
    let filtered: Vec<&CodeChunk> = chunks.iter().filter(|c| filter(c)).collect();
    let total = filtered.len();
    let display: Vec<&CodeChunk> = if let Some(n) = limit {
        filtered.into_iter().take(n).collect()
    } else {
        filtered
    };
    if display.is_empty() {
        println!("{empty_msg}");
    } else {
        let suffix = if let Some(n) = limit { format!(" (showing {}/{})", display.len().min(n), total) } else { String::new() };
        println!("{}{suffix}", header(total));
        if compact {
            print_grouped_compact(&display);
        } else {
            print_grouped(&display, None);
        }
    }
    Ok(())
}

/// Shared runner for BFS-based commands (gather).
fn run_bfs_command<E, F>(
    db: &std::path::Path,
    symbol: &str,
    format: OutputFormat,
    detail: bool,
    cache_key: &str,
    header: &str,
    compute: F,
) -> Result<()>
where
    E: BfsEntryExt + HasIdx,
    F: Fn(&graph::CallGraph, &[u32]) -> Vec<E>,
{
    if matches!(format, OutputFormat::Json) {
        return cached_json(db, cache_key, || {
            let graph = load_or_build_graph(db, None)?;
            let seeds = graph.resolve(symbol);
            let entries = compute(&graph, &seeds);
            let json = build_bfs_json(&graph, &entries);
            Ok(serde_json::to_string(&json)?)
        });
    }

    let graph = load_or_build_graph(db, None)?;
    let seeds = graph.resolve(symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    let entries = compute(&graph, &seeds);
    print!("{header}");
    print_bfs_grouped(&graph, &entries);

    if detail {
        print_detail_annotations(db, &graph, &entries);
    }

    Ok(())
}

fn print_trace_path(graph: &graph::CallGraph, path: &[u32]) {
    for (step, &idx) in path.iter().enumerate() {
        let i = idx as usize;
        let file = super::relative_path(&graph.files[i]);
        let name = &graph.names[i];
        let kind = &graph.kinds[i];
        let lines = format_lines_opt(graph.lines[i]);
        let test_marker = if graph.is_test[i] { " [test]" } else { "" };

        let arrow = if step == 0 { "  " } else { "-> " };
        let indent = if step == 0 { "" } else { &"   ".repeat(step) };
        println!("  {indent}{arrow}{file}{lines}  [{kind}] {name}{test_marker}");
    }
    println!();
}
