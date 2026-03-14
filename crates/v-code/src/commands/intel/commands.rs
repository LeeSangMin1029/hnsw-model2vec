//! CLI command handlers for code-intel subcommands.
//!
//! Each `run_*` function corresponds to a CLI subcommand (stats, symbols,
//! def, refs, gather, impact, trace). These handle text/JSON output and
//! caching; pure analysis logic lives in `v-code-intel`.

use std::path::PathBuf;

use anyhow::Result;

use super::{
    OutputFormat, cached_json, print_grouped,
};
use super::{
    build_bfs_json,
    build_stats, stats_to_json, load_chunks, load_or_build_graph,
    format_lines_opt, grouped_json,
    graph, impact, trace, CodeChunk,
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
    )?;

    // Show trait implementations if any trait was in the results.
    if !matches!(format, OutputFormat::Json) {
        print_trait_impls_if_relevant(&db, name.as_deref())?;
    }
    Ok(())
}

/// If the symbol search matched any trait, print its implementations.
fn print_trait_impls_if_relevant(db: &std::path::Path, name: Option<&str>) -> Result<()> {
    let name = match name {
        Some(n) => n,
        None => return Ok(()),
    };
    let graph = load_or_build_graph(db, None)?;
    let indices = graph.resolve(name);
    for idx in indices {
        let i = idx as usize;
        if graph.kinds[i] != "trait" {
            continue;
        }
        let impls = &graph.trait_impls[i];
        if impls.is_empty() {
            continue;
        }
        println!("  implementations of {}:", graph.names[i]);
        for &impl_idx in impls {
            let ii = impl_idx as usize;
            let file = super::relative_path(&graph.files[ii]);
            let lines = format_lines_opt(graph.lines[ii]);
            println!("    {file}{lines}  [{}] {}", graph.kinds[ii], graph.names[ii]);
        }
        println!();
    }
    Ok(())
}

/// `v-code context <db> <symbol> --depth N`
pub fn run_context(
    db: PathBuf,
    symbol: String,
    depth: u32,
    format: OutputFormat,
) -> Result<()> {
    use v_code_intel::context_cmd;

    let chunks = load_chunks(&db)?;
    let graph = load_or_build_graph(&db, None)?;
    let result = context_cmd::build_context(&graph, &chunks, &symbol, depth);

    if result.seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    if matches!(format, OutputFormat::Json) {
        let json = build_context_json(&graph, &result);
        println!("{}", serde_json::to_string(&json)?);
        return Ok(());
    }

    // Build tagged entries for file-grouped output
    let mut entries: Vec<TaggedEntry> = Vec::new();
    for &idx in &result.seeds {
        entries.push(TaggedEntry { idx, tag: "def", sig: true, call_line: 0 });
    }
    for e in &result.callers {
        // Caller calls the seed — look up call site line
        let cl = result.seeds.first().map_or(0, |&seed| graph.call_site_line(e.idx, seed));
        entries.push(TaggedEntry { idx: e.idx, tag: "caller", sig: false, call_line: cl });
    }
    for e in &result.callees {
        // Seed calls this callee — look up call site line from seed
        let cl = result.seeds.first().map_or(0, |&seed| graph.call_site_line(seed, e.idx));
        entries.push(TaggedEntry { idx: e.idx, tag: "callee", sig: false, call_line: cl });
    }
    for &idx in &result.types {
        entries.push(TaggedEntry { idx, tag: "type", sig: false, call_line: 0 });
    }
    for &idx in &result.tests {
        let cl = result.seeds.first().map_or(0, |&seed| graph.call_site_line(idx, seed));
        entries.push(TaggedEntry { idx, tag: "test", sig: false, call_line: cl });
    }

    // Header with counts
    let ext_count = result.unresolved_calls.len();
    let counts = format!(
        "{} caller, {} callee, {} type, {} test{}",
        result.callers.len(), result.callees.len(),
        result.types.len(), result.tests.len(),
        if ext_count > 0 { format!(", {} extern", ext_count) } else { String::new() },
    );
    println!("=== context: {symbol} ({counts}) ===\n");
    print_file_grouped(&graph, &entries);

    // Show unresolved/external calls if any.
    if !result.unresolved_calls.is_empty() {
        println!("@ [extern]");
        for call in &result.unresolved_calls {
            println!("  {call}");
        }
        println!();
    }

    Ok(())
}

fn build_context_json(
    graph: &graph::CallGraph,
    result: &v_code_intel::context_cmd::ContextResult,
) -> serde_json::Value {
    let entry_json = |idx: u32| -> serde_json::Value {
        let i = idx as usize;
        serde_json::json!({
            "f": super::relative_path(&graph.files[i]),
            "l": super::format_lines_str_opt(graph.lines[i]),
            "k": &graph.kinds[i],
            "n": &graph.names[i],
            "sig": graph.signatures[i].as_deref().unwrap_or(""),
            "t": graph.is_test[i],
        })
    };

    let bfs_entry_json = |idx: u32, depth: u32| -> serde_json::Value {
        let i = idx as usize;
        serde_json::json!({
            "f": super::relative_path(&graph.files[i]),
            "l": super::format_lines_str_opt(graph.lines[i]),
            "k": &graph.kinds[i],
            "n": &graph.names[i],
            "d": depth,
            "t": graph.is_test[i],
        })
    };

    serde_json::json!({
        "definition": result.seeds.iter().map(|&idx| entry_json(idx)).collect::<Vec<_>>(),
        "callers": result.callers.iter().map(|e| bfs_entry_json(e.idx, e.depth)).collect::<Vec<_>>(),
        "callees": result.callees.iter().map(|e| bfs_entry_json(e.idx, e.depth)).collect::<Vec<_>>(),
        "types": result.types.iter().map(|&idx| entry_json(idx)).collect::<Vec<_>>(),
        "tests": result.tests.iter().map(|&idx| entry_json(idx)).collect::<Vec<_>>(),
    })
}

/// `v-code blast <db> <symbol> --depth N [--include-tests]`
pub fn run_blast(
    db: PathBuf,
    symbol: String,
    depth: u32,
    format: OutputFormat,
    include_tests: bool,
) -> Result<()> {
    use v_code_intel::blast;

    if matches!(format, OutputFormat::Json) {
        let key = format!("blast:{symbol}:{depth}:{include_tests}");
        return cached_json(&db, &key, || {
            let graph = load_or_build_graph(&db, None)?;
            let seeds = graph.resolve(&symbol);
            let all_entries = impact::bfs_reverse(&graph, &seeds, depth);
            let summary = blast::summarize_blast(&graph, &all_entries);
            let entries = filter_test_entries(all_entries, include_tests);
            let mut json = build_bfs_json(&graph, &entries);
            if let serde_json::Value::Object(ref mut map) = json {
                map.insert("total_affected".to_owned(), serde_json::json!(summary.total_affected));
                map.insert("affected_files".to_owned(), serde_json::json!(summary.affected_files));
                map.insert("prod_count".to_owned(), serde_json::json!(summary.prod_count));
                map.insert("test_count".to_owned(), serde_json::json!(summary.test_count));
            }
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
    let summary = blast::summarize_blast(&graph, &all_entries);
    let entries = filter_test_entries(all_entries, include_tests);

    // Build tagged entries with depth labels
    let mut tagged: Vec<TaggedEntry> = Vec::new();
    for e in &entries {
        let tag = match e.depth {
            0 => "target",
            1 => "d1",
            2 => "d2",
            _ => "d3+",
        };
        tagged.push(TaggedEntry { idx: e.idx, tag, sig: false, call_line: 0 });
    }

    println!("=== blast: {symbol} ({} affected, {} prod, {} test) ===\n",
        summary.total_affected, summary.prod_count, summary.test_count);
    print_file_grouped(&graph, &tagged);

    Ok(())
}

/// `v-code jump <db> <symbol> --depth N`
pub fn run_jump(
    db: PathBuf,
    symbol: String,
    depth: u32,
    format: OutputFormat,
) -> Result<()> {
    use v_code_intel::jump;

    if matches!(format, OutputFormat::Json) {
        let key = format!("jump:{symbol}:{depth}");
        return cached_json(&db, &key, || {
            let graph = load_or_build_graph(&db, None)?;
            let seeds = graph.resolve(&symbol);
            let tree = jump::build_flow_tree(&graph, &seeds, depth);
            let json = jump::tree_to_json(&graph, &tree);
            Ok(serde_json::to_string(&json)?)
        });
    }

    let graph = load_or_build_graph(&db, None)?;
    let seeds = graph.resolve(&symbol);

    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        return Ok(());
    }

    println!("=== Execution Flow: {symbol} ===\n");
    let tree = jump::build_flow_tree(&graph, &seeds, depth);
    print!("{}", jump::render_tree(&graph, &tree));

    Ok(())
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
#[expect(clippy::too_many_arguments)]
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
        print_grouped(&display, compact);
    }
    Ok(())
}

// ── File-grouped output (shared by context, blast) ──────────────────────

/// A tagged graph entry: index + role tag + whether to show signature.
struct TaggedEntry {
    idx: u32,
    tag: &'static str,
    sig: bool,
    /// Source line where this entry calls/is-called-by the seed (0 = unknown).
    call_line: u32,
}

/// Print entries grouped by file, with multi-base path aliases.
///
/// Output format:
/// ```text
/// @ [A]scorer.rs
///   [def] :128-171 fn accumulate_and_rank
///   [callee] :107-118 fn ScoringCtx<'_>::score
/// @ [B]index.rs
///   [caller] :276-303 fn Bm25Index<T>::search
///
/// [A] = crates/v-hnsw-search/src/bm25/
/// [B] = crates/v-hnsw-search/src/bm25/
/// ```
fn print_file_grouped(graph: &graph::CallGraph, entries: &[TaggedEntry]) {
    use std::collections::BTreeMap;
    use v_code_intel::helpers::{apply_alias, build_path_aliases};

    if entries.is_empty() {
        return;
    }

    // Collect all file paths
    let files: Vec<&str> = entries
        .iter()
        .map(|e| super::relative_path(&graph.files[e.idx as usize]))
        .collect();

    let (alias_map, legend) = build_path_aliases(&files);

    // Group entries by file, preserving insertion order per file
    let mut groups: BTreeMap<&str, Vec<&TaggedEntry>> = BTreeMap::new();
    for (entry, file) in entries.iter().zip(files.iter()) {
        groups.entry(file).or_default().push(entry);
    }

    // Print each file group
    for (file, items) in &groups {
        let short = apply_alias(file, &alias_map);
        println!("@ {short}");
        for e in items {
            let i = e.idx as usize;
            let lines = format_lines_opt(graph.lines[i]);
            let kind = &graph.kinds[i];
            let name = &graph.names[i];
            let test_marker = if graph.is_test[i] { " [test]" } else { "" };
            let call_site = if e.call_line > 0 {
                format!("  → :{}", e.call_line)
            } else {
                String::new()
            };
            println!("  [{}] {lines} {kind} {name}{test_marker}{call_site}", e.tag);
            if e.sig {
                if let Some(s) = &graph.signatures[i] {
                    println!("    {s}");
                }
            }
        }
        println!();
    }

    if !legend.is_empty() {
        for (alias, dir) in &legend {
            println!("{alias} = {dir}");
        }
    }
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
