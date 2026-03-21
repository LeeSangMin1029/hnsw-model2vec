//! CLI command handlers for code-intel subcommands.
//!
//! Each `run_*` function corresponds to a CLI subcommand (stats, symbols,
//! def, refs, impact, trace). These handle text/JSON output and
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
    graph, impact, trace, ParsedChunk,
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
    let is_file_query = name.as_deref().is_some_and(looks_like_file_path);

    let key = format!(
        "symbols:{}:{}:{}:{}:{}",
        name.as_deref().unwrap_or(""),
        kind.as_deref().unwrap_or(""),
        if include_tests { "t" } else { "" },
        limit.map_or(String::new(), |l| l.to_string()),
        if is_file_query { "f" } else { "" },
    );
    run_chunk_query(&db, format, &key,
        |c| {
            if let Some(ref n) = name {
                if is_file_query {
                    // File mode: match file path suffix
                    if !c.file.ends_with(n) && !c.file.ends_with(&n.replace('\\', "/")) {
                        return false;
                    }
                } else {
                    // Name mode: substring match
                    if !c.name.to_lowercase().contains(&n.to_lowercase()) { return false; }
                }
            }
            if let Some(ref k) = kind
                && c.kind.to_lowercase() != k.to_lowercase() { return false; }
            if !include_tests && v_code_intel::graph::is_test_chunk(c) { return false; }
            true
        },
        "No symbols found.",
        |n| {
            if is_file_query {
                format!("{n} symbols in file:\n")
            } else {
                format!("{n} symbols found:\n")
            }
        },
        limit,
        compact,
    )?;

    // Show trait implementations if any trait was in the results (name mode only).
    if !is_file_query && !matches!(format, OutputFormat::Json) {
        print_trait_impls_if_relevant(&db, name.as_deref())?;
    }
    Ok(())
}

/// Check if a string looks like a file path rather than a symbol name.
///
/// Heuristic: contains a known source file extension or path separator.
fn looks_like_file_path(s: &str) -> bool {
    const EXTENSIONS: &[&str] = &[
        ".rs", ".go", ".py", ".js", ".ts", ".tsx", ".jsx",
        ".c", ".cpp", ".cc", ".h", ".hpp",
        ".java", ".kt", ".cs", ".rb", ".swift",
    ];
    EXTENSIONS.iter().any(|ext| s.ends_with(ext)) || s.contains('/')
}

/// If the symbol search matched any trait, print its implementations.
fn print_trait_impls_if_relevant(db: &std::path::Path, name: Option<&str>) -> Result<()> {
    let name = match name {
        Some(n) => n,
        None => return Ok(()),
    };
    // Use cached graph only — don't trigger a full build just for trait impls.
    let graph = match graph::CallGraph::load(db) {
        Some(g) => g,
        None => return Ok(()),
    };
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
    source: bool,
    include_tests: bool,
) -> Result<()> {
    use v_code_intel::context_cmd;

    let chunks = load_chunks(&db)?;
    let graph = load_or_build_graph(&db)?;
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
    if include_tests {
        for &idx in &result.tests {
            let cl = result.seeds.first().map_or(0, |&seed| graph.call_site_line(idx, seed));
            entries.push(TaggedEntry { idx, tag: "test", sig: false, call_line: cl });
        }
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
    print_file_grouped(&graph, &entries, source);

    if !include_tests && !result.tests.is_empty() {
        println!("  {} tests (use --include-tests to show)\n", result.tests.len());
    }

    // Show unresolved/external calls if any.
    if !result.unresolved_calls.is_empty() {
        println!("@ [extern]");
        for call in &result.unresolved_calls {
            println!("  {call}");
        }
        println!();
    }

    // Show string args from seed chunks.
    let all_string_args: Vec<_> = result.seeds.iter()
        .flat_map(|&s| graph.string_args.get(s as usize).into_iter().flatten())
        .collect();
    if !all_string_args.is_empty() {
        println!("@ [strings]");
        for (callee, value, line, pos) in &all_string_args {
            let line_display = if *line > 0 { format!(":{line}") } else { String::new() };
            println!("  {callee}(\"{value}\"){line_display}  arg[{pos}]");
        }
        println!();
    }

    // Show field access summary for struct seeds.
    let is_struct_seed = result.seeds.iter().any(|&s| graph.kinds[s as usize] == "struct");
    if is_struct_seed {
        // Use the struct name (lowercase) to look up field accesses
        let type_name = result.seeds.iter()
            .find(|&&s| graph.kinds[s as usize] == "struct")
            .map(|&s| &graph.names[s as usize]);
        if let Some(tn) = type_name {
            let field_entries = graph.find_field_accesses_for_type(tn);
            if !field_entries.is_empty() {
                println!("@ [field accesses]");
                for (field, indices) in &field_entries {
                    let accessors: Vec<&str> = indices.iter()
                        .map(|&i| graph.names[i as usize].as_str())
                        .collect();
                    println!("  .{field} ← {}", accessors.join(", "));
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Build a JSON object for a graph node, optionally including depth/signature.
fn node_json(graph: &graph::CallGraph, idx: u32, depth: Option<u32>, sig: bool) -> serde_json::Value {
    let i = idx as usize;
    let mut obj = serde_json::json!({
        "f": super::relative_path(&graph.files[i]),
        "l": super::format_lines_str_opt(graph.lines[i]),
        "k": &graph.kinds[i],
        "n": &graph.names[i],
        "t": graph.is_test[i],
    });
    if let Some(d) = depth {
        obj["d"] = serde_json::json!(d);
    }
    if sig {
        obj["sig"] = serde_json::json!(graph.signatures[i].as_deref().unwrap_or(""));
    }
    obj
}

fn build_context_json(
    graph: &graph::CallGraph,
    result: &v_code_intel::context_cmd::ContextResult,
) -> serde_json::Value {
    let string_args: Vec<_> = result.seeds.iter()
        .flat_map(|&s| graph.string_args.get(s as usize).into_iter().flatten())
        .map(|(callee, value, line, pos)| serde_json::json!({
            "callee": callee, "value": value, "line": line, "pos": pos,
        }))
        .collect();
    serde_json::json!({
        "definition": result.seeds.iter().map(|&idx| node_json(graph, idx, None, true)).collect::<Vec<_>>(),
        "callers": result.callers.iter().map(|e| node_json(graph, e.idx, Some(e.depth), false)).collect::<Vec<_>>(),
        "callees": result.callees.iter().map(|e| node_json(graph, e.idx, Some(e.depth), false)).collect::<Vec<_>>(),
        "types": result.types.iter().map(|&idx| node_json(graph, idx, None, false)).collect::<Vec<_>>(),
        "tests": result.tests.iter().map(|&idx| node_json(graph, idx, None, false)).collect::<Vec<_>>(),
        "string_args": string_args,
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
            let graph = load_or_build_graph(&db)?;
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

    let graph = load_or_build_graph(&db)?;

    // Field-level blast: "Type.field" notation
    if let Some(dot) = symbol.find('.') {
        let type_name = &symbol[..dot];
        let field_name = &symbol[dot + 1..];
        let key = format!("{}::{}", type_name.to_lowercase(), field_name.to_lowercase());
        let field_chunks = graph.find_field_access(&key);
        if field_chunks.is_empty() {
            println!("No field accesses found for {symbol}");
            return Ok(());
        }
        // BFS reverse from field-accessing chunks
        let all_entries = impact::bfs_reverse(&graph, &field_chunks, depth);
        let summary = blast::summarize_blast(&graph, &all_entries);
        let entries = filter_test_entries(all_entries, include_tests);
        let mut tagged: Vec<TaggedEntry> = Vec::new();
        for e in &entries {
            let tag = if field_chunks.contains(&e.idx) {
                "field"
            } else {
                match e.depth {
                    0 => "target",
                    1 => "d1",
                    2 => "d2",
                    _ => "d3+",
                }
            };
            tagged.push(TaggedEntry { idx: e.idx, tag, sig: false, call_line: 0 });
        }
        println!("=== blast: {symbol} ({} field accessors, {} affected, {} prod, {} test) ===\n",
            field_chunks.len(), summary.total_affected, summary.prod_count, summary.test_count);
        print_file_grouped(&graph, &tagged, false);
        return Ok(());
    }

    let Some(seeds) = resolve_symbol(&graph, &symbol) else { return Ok(()) };

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
    print_file_grouped(&graph, &tagged, false);

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
            let graph = load_or_build_graph(&db)?;
            let seeds = graph.resolve(&symbol);
            let tree = jump::build_flow_tree(&graph, &seeds, depth);
            let json = jump::tree_to_json(&graph, &tree);
            Ok(serde_json::to_string(&json)?)
        });
    }

    let graph = load_or_build_graph(&db)?;
    let Some(seeds) = resolve_symbol(&graph, &symbol) else { return Ok(()) };

    println!("=== Execution Flow: {symbol} ===\n");
    let tree = jump::build_flow_tree(&graph, &seeds, depth);
    print!("{}", jump::render_tree(&graph, &tree));

    Ok(())
}

/// Resolve a symbol name to graph indices, printing a message if not found.
/// Returns `None` (and prints) when resolution yields no results.
fn resolve_symbol(graph: &graph::CallGraph, symbol: &str) -> Option<Vec<u32>> {
    let seeds = graph.resolve(symbol);
    if seeds.is_empty() {
        println!("No symbol found matching \"{symbol}\".");
        None
    } else {
        Some(seeds)
    }
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
            let graph = load_or_build_graph(&db)?;
            let sources = graph.resolve(&from);
            let targets = graph.resolve(&to);
            let json = match trace::bfs_shortest_path(&graph, &sources, &targets) {
                Some(path) => trace::build_json(&graph, &path),
                None => serde_json::json!({ "path": null, "hops": null }),
            };
            Ok(serde_json::to_string(&json)?)
        });
    }

    let graph = load_or_build_graph(&db)?;
    let Some(sources) = resolve_symbol(&graph, &from) else { return Ok(()) };
    let Some(targets) = resolve_symbol(&graph, &to) else { return Ok(()) };

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

/// `v-code strings <db> <query> [--callee filter]`
pub fn run_strings(
    db: PathBuf,
    query: String,
    callee_filter: Option<String>,
) -> Result<()> {
    use std::collections::BTreeMap;
    use v_code_intel::helpers::{apply_alias, build_path_aliases};

    let graph = load_or_build_graph(&db)?;
    let lower = query.to_lowercase();
    let callee_lower = callee_filter.as_ref().map(|f| f.to_lowercase());

    let mut matches: Vec<(u32, &str, &str, u32, u8)> = Vec::new();

    for (chunk_idx, args) in graph.string_args.iter().enumerate() {
        for (callee, value, line, pos) in args {
            if !value.to_lowercase().contains(&lower) {
                continue;
            }
            if let Some(ref filter) = callee_lower
                && !callee.to_lowercase().contains(filter) {
                    continue;
                }
            matches.push((chunk_idx as u32, callee, value, *line, *pos));
        }
    }

    if matches.is_empty() {
        println!("No string arguments matching \"{query}\" found.");
        return Ok(());
    }

    println!("=== strings: \"{query}\" ({} matches) ===\n", matches.len());

    let files: Vec<&str> = matches
        .iter()
        .map(|(idx, ..)| super::relative_path(&graph.files[*idx as usize]))
        .collect();
    let (alias_map, legend) = build_path_aliases(&files);

    type MatchRef<'a> = &'a (u32, &'a str, &'a str, u32, u8);
    let mut groups: BTreeMap<&str, Vec<MatchRef<'_>>> = BTreeMap::new();
    for (m, file) in matches.iter().zip(files.iter()) {
        groups.entry(file).or_default().push(m);
    }

    for (file, items) in &groups {
        let short = apply_alias(file, &alias_map);
        println!("@ {short}");
        for (idx, callee, value, _line, pos) in items.iter().copied() {
            let i = *idx as usize;
            let func = &graph.names[i];
            let lines = format_lines_opt(graph.lines[i]);
            println!("  {lines} {func}  {callee}(\"{value}\")  arg[{pos}]");
        }
        println!();
    }

    if !legend.is_empty() {
        for (alias, dir) in &legend {
            println!("{alias} = {dir}");
        }
    }
    Ok(())
}

/// `v-code flow <db> <query> --depth N`
pub fn run_flow(
    db: PathBuf,
    query: String,
    depth: u32,
) -> Result<()> {
    use std::collections::BTreeMap;
    use v_code_intel::flow_cmd;
    use v_code_intel::helpers::{apply_alias, build_path_aliases};

    let graph = load_or_build_graph(&db)?;
    let paths = flow_cmd::trace_string_flow(&graph, &query, depth);

    if paths.is_empty() {
        println!("No string flow matching \"{query}\" found.");
        return Ok(());
    }

    // Collect all files across all paths for alias building
    let all_files: Vec<&str> = paths
        .iter()
        .flat_map(|p| p.iter().map(|e| super::relative_path(&graph.files[e.chunk_idx as usize])))
        .collect();
    let (alias_map, legend) = build_path_aliases(&all_files);

    // Group flows by their origin file (first step's file)
    let mut groups: BTreeMap<&str, Vec<(usize, &Vec<flow_cmd::FlowStep>)>> = BTreeMap::new();
    for (i, path) in paths.iter().enumerate() {
        if let Some(first) = path.first() {
            let file = super::relative_path(&graph.files[first.chunk_idx as usize]);
            groups.entry(file).or_default().push((i, path));
        }
    }

    println!("=== flow: \"{query}\" ({} path(s)) ===\n", paths.len());

    for (file, items) in &groups {
        let short = apply_alias(file, &alias_map);
        println!("@ {short}");
        for (_, path) in items {
            for (step, entry) in path.iter().enumerate() {
                let i = entry.chunk_idx as usize;
                let func = &graph.names[i];
                let lines = format_lines_opt(graph.lines[i]);
                let marker = if entry.is_direct { "direct" } else { "relay" };
                if step == 0 {
                    println!(
                        "  {lines} {func}  [{marker}] {callee}(\"{value}\")",
                        callee = entry.callee, value = entry.value,
                    );
                } else {
                    let ef = super::relative_path(&graph.files[i]);
                    let eshort = apply_alias(ef, &alias_map);
                    println!(
                        "    -> {eshort}{lines}  {func}  [{marker}] {callee}(\"{value}\")",
                        callee = entry.callee, value = entry.value,
                    );
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
    Ok(())
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Shared runner for chunk-filter commands (symbols, def).
#[expect(clippy::too_many_arguments)]
fn run_chunk_query(
    db: &std::path::Path,
    format: OutputFormat,
    cache_key: &str,
    filter: impl Fn(&ParsedChunk) -> bool,
    empty_msg: &str,
    header: impl FnOnce(usize) -> String,
    limit: Option<usize>,
    compact: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        return cached_json(db, cache_key, || {
            let chunks = load_chunks(db)?;
            let mut filtered: Vec<&ParsedChunk> = chunks.iter().filter(|c| filter(c)).collect();
            if let Some(n) = limit { filtered.truncate(n); }
            Ok(serde_json::to_string(&grouped_json(&filtered))?)
        });
    }
    let chunks = load_chunks(db)?;
    let filtered: Vec<&ParsedChunk> = chunks.iter().filter(|c| filter(c)).collect();
    let total = filtered.len();
    let display: Vec<&ParsedChunk> = if let Some(n) = limit {
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
fn print_file_grouped(graph: &graph::CallGraph, entries: &[TaggedEntry], show_source: bool) {
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
            if e.sig
                && let Some(s) = &graph.signatures[i] {
                    println!("    {s}");
                }
            if show_source && (e.tag == "def" || e.sig)
                && let Some((start, end)) = graph.lines[i] {
                    print_source_lines(&graph.files[i], start, end);
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

/// Read and print source lines from a file.
fn print_source_lines(file_path: &str, start: usize, end: usize) {
    let Ok(content) = std::fs::read_to_string(file_path) else {
        return;
    };
    let lines: Vec<&str> = content.lines().collect();
    let start = start.saturating_sub(1); // 1-based → 0-based
    let end = end.min(lines.len());
    if start >= end {
        return;
    }
    println!("    ```");
    for (i, line) in lines[start..end].iter().enumerate() {
        println!("    {:>4}│ {line}", start + i + 1);
    }
    println!("    ```");
}

fn print_trace_path(graph: &graph::CallGraph, path: &[u32]) {
    use v_code_intel::helpers::{apply_alias, build_path_aliases};
    let files: Vec<&str> = path.iter()
        .map(|&idx| super::relative_path(&graph.files[idx as usize]))
        .collect();
    let (alias_map, legend) = build_path_aliases(&files);

    for (step, &idx) in path.iter().enumerate() {
        let i = idx as usize;
        let file = super::relative_path(&graph.files[i]);
        let short_file = apply_alias(file, &alias_map);
        let name = &graph.names[i];
        let lines = format_lines_opt(graph.lines[i]);

        let arrow = if step == 0 { "  " } else { "→ " };
        let indent = if step == 0 { "" } else { &"  ".repeat(step) };
        println!("  {indent}{arrow}{short_file}{lines}  {name}");
    }
    println!();
    if !legend.is_empty() {
        for (alias, dir) in &legend {
            println!("{alias} = {dir}");
        }
        println!();
    }
}


/// `v-code coverage` — per-crate test coverage with per-function test counts.
pub fn run_coverage(
    db: PathBuf,
    depth: u32,
    file_filter: Option<String>,
    format: OutputFormat,
) -> Result<()> {
    use std::collections::{BTreeMap, VecDeque};
    use v_code_intel::helpers::extract_crate_name;

    let graph = load_or_build_graph(&db)?;
    let n = graph.names.len();

    // For each function, count how many distinct test functions reach it via BFS callees.
    let mut test_counts = vec![0u32; n];

    for test_idx in 0..n {
        if !graph.is_test[test_idx] {
            continue;
        }
        // BFS from this test, following callees.
        let mut visited = vec![false; n];
        visited[test_idx] = true;
        let mut queue: VecDeque<(usize, u32)> = VecDeque::new();
        queue.push_back((test_idx, 0));

        while let Some((idx, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            for &callee in &graph.callees[idx] {
                let c = callee as usize;
                if !visited[c] {
                    visited[c] = true;
                    test_counts[c] += 1;
                    queue.push_back((c, d + 1));
                }
            }
        }
    }

    // Group by crate.
    let mut crate_data: BTreeMap<String, CrateStats> = BTreeMap::new();

    for i in 0..n {
        if let Some(ref filter) = file_filter {
            if !graph.files[i].contains(filter.as_str()) {
                continue;
            }
        }
        let crate_name = extract_crate_name(&graph.files[i]);
        let entry = crate_data.entry(crate_name).or_default();

        if graph.is_test[i] && graph.kinds[i] == "function" {
            entry.test_fn += 1;
            continue;
        }
        if graph.kinds[i] != "function" {
            continue;
        }

        entry.prod_fn += 1;
        if test_counts[i] > 0 {
            entry.tested += 1;
        } else {
            entry.untested += 1;
        }
        entry.functions.push(FnCoverage {
            name: graph.names[i].clone(),
            file: graph.files[i].clone(),
            lines: graph.lines[i],
            test_count: test_counts[i],
        });
    }

    if matches!(format, OutputFormat::Json) {
        let mut crates_json = serde_json::Map::new();
        for (name, stats) in &crate_data {
            let fns: Vec<serde_json::Value> = stats.functions.iter().map(|f| {
                serde_json::json!({
                    "name": f.name,
                    "file": super::relative_path(&f.file),
                    "lines": format_lines_opt(f.lines),
                    "tests": f.test_count,
                })
            }).collect();
            crates_json.insert(name.clone(), serde_json::json!({
                "prod_fn": stats.prod_fn,
                "test_fn": stats.test_fn,
                "tested": stats.tested,
                "untested": stats.untested,
                "coverage": if stats.prod_fn > 0 {
                    format!("{:.1}%", stats.tested as f64 / stats.prod_fn as f64 * 100.0)
                } else {
                    "N/A".to_owned()
                },
                "functions": fns,
            }));
        }
        println!("{}", serde_json::to_string_pretty(&serde_json::Value::Object(crates_json))?);
        return Ok(());
    }

    // Text output: per-crate summary table.
    println!("=== coverage (BFS depth {depth}) ===\n");
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "crate", "prod_fn", "test_fn", "tested", "untested", "coverage"
    );
    println!("{}", "-".repeat(73));

    let mut total = CrateStats::default();
    for (name, stats) in &crate_data {
        let pct = if stats.prod_fn > 0 {
            format!("{:.1}%", stats.tested as f64 / stats.prod_fn as f64 * 100.0)
        } else {
            "N/A".to_owned()
        };
        println!(
            "{:<24} {:>8} {:>8} {:>8} {:>8} {:>9}",
            name, stats.prod_fn, stats.test_fn, stats.tested, stats.untested, pct
        );
        total.prod_fn += stats.prod_fn;
        total.test_fn += stats.test_fn;
        total.tested += stats.tested;
        total.untested += stats.untested;
    }
    println!("{}", "-".repeat(73));
    let total_pct = if total.prod_fn > 0 {
        format!("{:.1}%", total.tested as f64 / total.prod_fn as f64 * 100.0)
    } else {
        "N/A".to_owned()
    };
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "total", total.prod_fn, total.test_fn, total.tested, total.untested, total_pct
    );

    // Detail: untested functions grouped by crate.
    let any_untested = crate_data.values().any(|s| s.untested > 0);
    if any_untested {
        println!("\n--- untested functions ---\n");
        for (crate_name, stats) in &crate_data {
            let untested: Vec<&FnCoverage> = stats.functions.iter()
                .filter(|f| f.test_count == 0)
                .collect();
            if untested.is_empty() {
                continue;
            }
            println!("[{}] {} untested:", crate_name, untested.len());
            for f in &untested {
                let loc = format_lines_opt(f.lines);
                let rel = super::relative_path(&f.file);
                println!("  {} ({}:{})", f.name, rel, loc);
            }
            println!();
        }
    }

    Ok(())
}

#[derive(Default)]
struct CrateStats {
    prod_fn: usize,
    test_fn: usize,
    tested: usize,
    untested: usize,
    functions: Vec<FnCoverage>,
}

struct FnCoverage {
    name: String,
    file: String,
    lines: Option<(usize, usize)>,
    test_count: u32,
}
