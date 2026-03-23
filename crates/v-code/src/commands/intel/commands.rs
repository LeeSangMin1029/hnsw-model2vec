//! CLI command handlers for code-intel subcommands.
//!
//! Each `run_*` function corresponds to a CLI subcommand (stats, symbols,
//! context, blast, jump, trace, etc.). Pure analysis logic lives in `v-code-intel`.

use std::path::{Path, PathBuf};

use anyhow::Result;

use super::print_grouped;
use super::{
    build_stats, load_chunks, load_or_build_graph,
    format_lines_opt,
    graph, impact, trace, ParsedChunk,
};

// ── Commands ─────────────────────────────────────────────────────────────

/// `v-code aliases` — print global path alias mapping.
pub fn run_aliases(db: PathBuf) -> Result<()> {
    let graph = load_or_build_graph(&db)?;
    let (_alias_map, legend) = graph.global_aliases();
    for (alias, dir) in &legend {
        println!("{alias} = {dir}");
    }
    Ok(())
}

/// `v-hnsw stats` — per-crate summary of code symbols.
pub fn run_stats(db: PathBuf) -> Result<()> {
    let chunks = load_chunks(&db)?;
    let stats = build_stats(&chunks);
    println!("=== stats: {} crates ===\n", stats.len());
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

/// `v-hnsw symbols` — list symbols matching filters.
pub fn run_symbols(
    db: PathBuf,
    name: Option<String>,
    kind: Option<String>,
    include_tests: bool,
    limit: Option<usize>,
    compact: bool,
) -> Result<()> {
    let is_file_query = name.as_deref().is_some_and(looks_like_file_path);

    run_chunk_query(&db,
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
                format!("=== symbols: {n} in file ===\n")
            } else {
                format!("=== symbols: {n} found ===\n")
            }
        },
        limit,
        compact,
    )?;

    // Show trait implementations if any trait was in the results (name mode only).
    if !is_file_query {
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
    let counts = format!(
        "{} caller, {} callee, {} type, {} test",
        result.callers.len(), result.callees.len(),
        result.types.len(), result.tests.len(),
    );
    let (alias_map, _) = graph.global_aliases();
    println!("=== context: {symbol} ({counts}) ===\n");
    print_file_grouped(&graph, &entries, source, &alias_map);

    if !include_tests && !result.tests.is_empty() {
        println!("  {} tests (use --include-tests to show)\n", result.tests.len());
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

/// `v-code blast <db> <symbol> --depth N [--include-tests]`
pub fn run_blast(
    db: PathBuf,
    symbol: String,
    depth: u32,
    include_tests: bool,
) -> Result<()> {
    use v_code_intel::blast;

    let graph = load_or_build_graph(&db)?;
    let (alias_map, _) = graph.global_aliases();

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
        print_file_grouped(&graph, &tagged, false, &alias_map);
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
    print_file_grouped(&graph, &tagged, false, &alias_map);

    Ok(())
}

/// `v-code jump <db> <symbol> --depth N`
pub fn run_jump(
    db: PathBuf,
    symbol: String,
    depth: u32,
) -> Result<()> {
    use v_code_intel::jump;

    let graph = load_or_build_graph(&db)?;
    let Some(seeds) = resolve_symbol(&graph, &symbol) else { return Ok(()) };
    let (alias_map, _legend) = graph.global_aliases();

    println!("=== jump: {symbol} ===\n");
    let tree = jump::build_flow_tree(&graph, &seeds, depth);
    print!("{}", jump::render_tree(&graph, &tree, &alias_map));

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
) -> Result<()> {
    let graph = load_or_build_graph(&db)?;
    let (alias_map, _) = graph.global_aliases();
    let Some(sources) = resolve_symbol(&graph, &from) else { return Ok(()) };
    let Some(targets) = resolve_symbol(&graph, &to) else { return Ok(()) };

    match trace::bfs_shortest_path(&graph, &sources, &targets) {
        Some(path) => {
            println!("=== trace: {from} \u{2192} {to} ({} hops) ===\n", path.len() - 1);
            print_trace_path(&graph, &path, &alias_map);
        }
        None => {
            println!("No call path found from \"{from}\" to \"{to}\".");
        }
    }

    Ok(())
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Shared runner for chunk-filter commands (symbols).
fn run_chunk_query(
    db: &std::path::Path,
    filter: impl Fn(&ParsedChunk) -> bool,
    empty_msg: &str,
    header: impl FnOnce(usize) -> String,
    limit: Option<usize>,
    compact: bool,
) -> Result<()> {
    let chunks = load_chunks(db)?;
    // Compute aliases from ALL chunks (not just filtered) for global consistency.
    let all_files: Vec<&str> = chunks.iter().map(|c| v_code_intel::helpers::relative_path(&c.file)).collect();
    let (alias_map, _legend) = v_code_intel::helpers::build_path_aliases(&all_files);

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
        print_grouped(&display, compact, &alias_map);
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
fn print_file_grouped(
    graph: &graph::CallGraph,
    entries: &[TaggedEntry],
    show_source: bool,
    alias_map: &std::collections::BTreeMap<String, String>,
) {
    use std::collections::BTreeMap;
    use v_code_intel::helpers::apply_alias;

    if entries.is_empty() {
        return;
    }

    // Deduplicate: if same idx appears in multiple roles, keep highest priority.
    // Priority: def > caller > callee > type > test
    let role_priority = |tag: &str| -> u8 {
        match tag {
            "def" => 0, "target" | "field" => 0,
            "caller" | "d1" | "d2" | "d3+" => 1,
            "callee" => 2,
            "type" => 3,
            "test" => 4,
            _ => 5,
        }
    };
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut deduped: Vec<&TaggedEntry> = Vec::new();
    // Sort by priority first, then deduplicate
    let mut all_entries: Vec<&TaggedEntry> = entries.iter().collect();
    all_entries.sort_by_key(|e| role_priority(e.tag));
    for e in &all_entries {
        if seen.insert(e.idx) {
            deduped.push(e);
        }
    }

    // Re-collect files and re-group after dedup
    let files: Vec<&str> = deduped
        .iter()
        .map(|e| super::relative_path(&graph.files[e.idx as usize]))
        .collect();
    let mut groups: BTreeMap<&str, Vec<&TaggedEntry>> = BTreeMap::new();
    for (entry, file) in deduped.iter().zip(files.iter()) {
        groups.entry(file).or_default().push(entry);
    }

    // Print each file group
    for (file, items) in &groups {
        let short = apply_alias(file, alias_map);
        println!("@ {short}");
        for e in items {
            let i = e.idx as usize;
            let lines = format_lines_opt(graph.lines[i]);
            let kind = &graph.kinds[i];
            let name = &graph.names[i];
            let test_marker = if graph.is_test[i] { " [test]" } else { "" };
            let call_site = if e.call_line > 0 {
                format!(" → :{}", e.call_line)
            } else {
                String::new()
            };
            let kind_tag = if *kind == "function" { String::new() } else { format!("[{kind}] ") };
            println!("  [{}] {lines} {kind_tag}{name}{test_marker}{call_site}", e.tag);
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

fn print_trace_path(
    graph: &graph::CallGraph,
    path: &[u32],
    alias_map: &std::collections::BTreeMap<String, String>,
) {
    use v_code_intel::helpers::apply_alias;

    for (step, &idx) in path.iter().enumerate() {
        let i = idx as usize;
        let file = super::relative_path(&graph.files[i]);
        let short_file = apply_alias(file, alias_map);
        let name = &graph.names[i];
        let lines = format_lines_opt(graph.lines[i]);

        let test_marker = if graph.is_test[i] { " [test]" } else { "" };
        let arrow = if step == 0 { "  " } else { "→ " };
        let indent = if step == 0 { "" } else { &"  ".repeat(step) };
        println!("  {indent}{arrow}{short_file}{lines}  {name}{test_marker}");
    }
    println!();
}


/// `v-code coverage` — test coverage via `cargo llvm-cov` with call-graph supplement.
pub fn run_coverage(
    db: PathBuf,
    depth: u32,
    file_filter: Option<String>,
) -> Result<()> {
    use std::collections::{BTreeMap, VecDeque};
    use v_code_intel::helpers::extract_crate_name;

    let graph = load_or_build_graph(&db)?;
    let n = graph.names.len();

    // BFS: for each function, count how many distinct test functions reach it.
    // depth=0 means unlimited (full reachability).
    let mut test_counts = vec![0u32; n];

    for test_idx in 0..n {
        if !graph.is_test[test_idx] {
            continue;
        }
        let mut visited = vec![false; n];
        visited[test_idx] = true;
        let mut queue: VecDeque<(usize, u32)> = VecDeque::new();
        queue.push_back((test_idx, 0));

        while let Some((idx, d)) = queue.pop_front() {
            if depth > 0 && d >= depth {
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

    // Try llvm-cov first (actual execution-based coverage)
    let llvm_cov_result = run_llvm_cov(&db);

    if let Some(cov) = llvm_cov_result {
        // ── llvm-cov succeeded: show actual coverage + unreached functions ──
        println!("=== test coverage (cargo llvm-cov) ===\n");
        println!("  functions: {}/{} ({:.1}%)", cov.fn_covered, cov.fn_total, cov.fn_percent);
        println!("  lines:     {}/{} ({:.1}%)", cov.line_covered, cov.line_total, cov.line_percent);
        println!();

        // Show functions with no test call path from static analysis
        let unreached: Vec<usize> = (0..n)
            .filter(|&i| {
                if graph.is_test[i] || graph.kinds[i] != "function" {
                    return false;
                }
                if let Some(ref filter) = file_filter {
                    if !graph.files[i].contains(filter.as_str()) {
                        return false;
                    }
                }
                test_counts[i] == 0
            })
            .collect();

        if !unreached.is_empty() {
            let (alias_map, _) = graph.global_aliases();
            println!("--- {} functions with no test call path ---\n", unreached.len());
            for &i in unreached.iter().take(30) {
                let loc = format_lines_opt(graph.lines[i]);
                let rel = super::relative_path(&graph.files[i]);
                let short = v_code_intel::helpers::apply_alias(rel, &alias_map);
                println!("  {short}{loc}  {}", graph.names[i]);
            }
            if unreached.len() > 30 {
                println!("  ... and {} more", unreached.len() - 30);
            }
            println!();
        }
    } else {
        // ── Fallback: static reachability (no llvm-cov available) ──
        eprintln!("  [coverage] cargo llvm-cov not available, using static reachability");

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

        let depth_str = if depth == 0 { "unlimited".to_owned() } else { format!("{depth}") };
        println!("=== static reachability (depth {depth_str}) ===\n");
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

        // Detail: untested functions grouped by crate
        let any_untested = crate_data.values().any(|s| s.untested > 0);
        if any_untested {
            let (alias_map, _) = graph.global_aliases();
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
                    let short = v_code_intel::helpers::apply_alias(rel, &alias_map);
                    println!("  {short}{loc}  {}", f.name);
                }
                println!();
            }
        }
    }

    Ok(())
}

// ── llvm-cov integration ─────────────────────────────────────────────────

struct LlvmCovResult {
    fn_total: usize,
    fn_covered: usize,
    fn_percent: f64,
    line_total: usize,
    line_covered: usize,
    line_percent: f64,
}

/// Run `cargo llvm-cov --json --ignore-run-fail` and parse totals.
/// Returns `None` if the tool is not installed or the command fails.
fn run_llvm_cov(db: &Path) -> Option<LlvmCovResult> {
    let project_root = db.parent()?;

    let output = std::process::Command::new("cargo")
        .arg("llvm-cov")
        .arg("--json")
        .arg("--ignore-run-fail")
        .current_dir(project_root)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    let data = json.get("data")?.get(0)?;
    let totals = data.get("totals")?;

    let functions = totals.get("functions")?;
    let lines = totals.get("lines")?;

    Some(LlvmCovResult {
        fn_total: functions.get("count")?.as_u64()? as usize,
        fn_covered: functions.get("covered")?.as_u64()? as usize,
        fn_percent: functions.get("percent")?.as_f64()?,
        line_total: lines.get("count")?.as_u64()? as usize,
        line_covered: lines.get("covered")?.as_u64()? as usize,
        line_percent: lines.get("percent")?.as_f64()?,
    })
}

// ── Coverage helper types ────────────────────────────────────────────────

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
