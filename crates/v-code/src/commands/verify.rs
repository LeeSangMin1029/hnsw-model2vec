//! Call graph accuracy verification (precision + recall).
//!
//! **Precision**: For every callee edge with a call-site line, does the callee's
//! short name actually appear on that source line?
//!
//! **Recall**: For every function call that tree-sitter extracts from the source,
//! did the graph resolve it to some callee? (Unresolved = "extern" in the graph.)

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;

use v_code_intel::extern_types::ExternMethodIndex;
use v_code_intel::graph::{extract_leaf_type, extract_generic_bounds, owning_type, build_receiver_type_map};
use v_code_intel::parse::ParsedChunk;
use v_hnsw_cli::commands::db_config::DbConfig;

/// Extract the short (leaf) name from a qualified name: `"Foo::bar"` → `"bar"`.
pub(crate) fn short_name(full: &str) -> &str {
    full.rsplit("::").next().unwrap_or(full)
}

/// Verify call-site accuracy for all function chunks in the database.
#[expect(clippy::needless_range_loop, clippy::type_complexity, reason = "multiple parallel arrays indexed together")]
pub fn run(db: PathBuf) -> Result<()> {
    let start = std::time::Instant::now();
    // Load graph + chunks in one pass to avoid double deserialization.
    let (graph, cached_chunks) = super::intel::load_or_build_graph_with_chunks(&db)?;

    let project_root = DbConfig::load(&db)
        .ok()
        .and_then(|c| c.input_path)
        .map(PathBuf::from);

    // Reuse chunks from graph build if available, otherwise load separately.
    let chunks = match cached_chunks {
        Some(c) => c,
        None => v_code_intel::loader::load_chunks(&db)?,
    };
    let t_load = start.elapsed();

    let n = graph.names.len();

    // ── Single pass: build all per-chunk indexes ─────────────────────
    let mut name_counts: HashMap<&str, usize> = HashMap::new();
    let mut project_shorts: HashSet<String> = HashSet::new();
    let mut project_type_shorts: HashSet<String> = HashSet::new();
    let mut chunk_callee_shorts: Vec<HashSet<String>> = vec![HashSet::new(); n];

    for (i, kind) in graph.kinds.iter().enumerate().take(n) {
        if kind == "function" {
            let short = short_name(&graph.names[i]);
            *name_counts.entry(short).or_default() += 1;
            project_shorts.insert(short.to_lowercase());
            for &callee_idx in &graph.callees[i] {
                let ci = callee_idx as usize;
                if ci < n {
                    chunk_callee_shorts[i].insert(short_name(&graph.names[ci]).to_lowercase());
                }
            }
        } else if matches!(kind.as_str(), "struct" | "enum" | "trait" | "impl") {
            let short = short_name(&graph.names[i]).to_lowercase();
            // Strip generic params: "vec<t>" → "vec"
            let clean = short.split('<').next().unwrap_or(&short);
            project_type_shorts.insert(clean.to_owned());
        }
    }

    // Build owner_field_types + return_type_map from DB chunks (no file I/O).
    let mut owner_field_types: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut return_type_map: HashMap<String, String> = HashMap::new();
    for c in &chunks {
        if c.kind == "struct" {
            let lower = c.name.to_lowercase();
            let leaf = lower.rsplit("::").next().unwrap_or(&lower);
            let key = leaf.split('<').next().unwrap_or(leaf);
            let entry = owner_field_types.entry(key.to_owned()).or_default();
            for (fname, ftype) in &c.field_types {
                entry.insert(fname.to_lowercase(), ftype.to_lowercase());
            }
        }
        if c.kind == "function"
            && let Some(ref ret) = c.return_type {
                let ret_lower = ret.to_lowercase();
                let leaf = extract_leaf_type(&ret_lower);
                let resolved_type = if leaf == "self" || leaf == "&self" {
                    owning_type(&c.name).unwrap_or_else(|| leaf.to_owned())
                } else {
                    leaf.to_owned()
                };
                let name_lower = c.name.to_lowercase();
                // Also insert bare method name for chained-call resolution
                if let Some(method) = name_lower.rsplit_once("::").map(|p| p.1) {
                    return_type_map.entry(method.to_owned()).or_insert_with(|| resolved_type.clone());
                }
                return_type_map.insert(name_lower, resolved_type);
            }
    }

    // Use graph-embedded extern info if available, otherwise build ExternMethodIndex.
    let has_graph_extern = !graph.extern_calls.is_empty()
        && graph.extern_calls.iter().any(|v| !v.is_empty());

    let extern_index = if has_graph_extern {
        None // Graph already has extern classification
    } else {
        let extern_root = project_root.as_deref().unwrap_or(Path::new("."));
        Some(ExternMethodIndex::build(extern_root))
    };

    if let Some(ref ext) = extern_index {
        // Merge extern return types into return_type_map (fallback path).
        let before = return_type_map.len();
        for (key, ret_type) in &ext.return_types {
            return_type_map.entry(key.clone()).or_insert_with(|| ret_type.clone());
            if let Some(method) = key.rsplit_once("::").map(|p| p.1) {
                return_type_map.entry(method.to_owned()).or_insert_with(|| ret_type.clone());
            }
        }
        let extern_ret_count = return_type_map.len() - before;
        if extern_ret_count > 0 {
            println!("  Extern return types merged: {extern_ret_count} (total return_type_map: {})", return_type_map.len());
        }
    }

    // Pre-build flat method set for O(1) lookups (only for fallback path).
    let extern_all_methods = extern_index.as_ref().map(|e| e.all_method_set());

    let t_index = start.elapsed();

    // ── Precision ────────────────────────────────────────────────────
    // Only precision needs source files (to verify call-site lines).
    let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();
    let mut prec_total = 0usize;
    let mut prec_ok = 0usize;
    let mut prec_wrong = 0usize;
    let mut ambig_total = 0usize;
    let mut ambig_wrong = 0usize;
    let mut wrong_unique: Vec<(String, Vec<String>)> = Vec::new();
    let mut wrong_ambig: Vec<(String, Vec<String>)> = Vec::new();

    for i in 0..n {
        if graph.kinds[i] != "function" {
            continue;
        }

        let sname = short_name(&graph.names[i]);
        let is_ambiguous = name_counts.get(sname).copied().unwrap_or(0) > 1;
        let mut errors = Vec::new();

        for &(callee_idx, call_line) in &graph.call_sites[i] {
            if call_line == 0 {
                continue;
            }
            let ci = callee_idx as usize;
            if ci >= n || graph.kinds[ci] != "function" {
                continue;
            }

            let callee_short = short_name(&graph.names[ci]);

            prec_total += 1;
            if is_ambiguous {
                ambig_total += 1;
            }

            let resolved = resolve_path(&graph.files[i], project_root.as_deref());
            let lines = file_cache
                .entry(resolved.clone())
                .or_insert_with(|| load_lines(&resolved));

            let ln = call_line as usize;
            if ln == 0 || ln > lines.len() {
                prec_wrong += 1;
                if is_ambiguous {
                    ambig_wrong += 1;
                }
                errors.push(format!(
                    "    {callee_short} \u{2192} L{call_line}: OUT OF RANGE"
                ));
                continue;
            }

            let src = &lines[ln - 1];
            if src.contains(callee_short) {
                prec_ok += 1;
            } else {
                prec_wrong += 1;
                if is_ambiguous {
                    ambig_wrong += 1;
                }
                let truncated: String = src.trim().chars().take(70).collect();
                errors.push(format!(
                    "    {} \u{2192} L{call_line}: '{truncated}'",
                    graph.names[ci]
                ));
            }
        }

        if !errors.is_empty() {
            let label = format!(
                "{}:{}) {}",
                graph.files[i],
                graph.lines[i].map_or(0, |l| l.0),
                graph.names[i],
            );
            if is_ambiguous {
                wrong_ambig.push((label, errors));
            } else {
                wrong_unique.push((label, errors));
            }
        }
    }

    let t_precision = start.elapsed();

    // ── Recall ───────────────────────────────────────────────────────
    // Uses DB chunks directly — no tree-sitter re-parsing needed.
    // Match DB chunks to graph indices by (name, lines).
    let chunk_map: HashMap<(&str, Option<(usize, usize)>), &ParsedChunk> = chunks
        .iter()
        .map(|c| ((c.name.as_str(), c.lines), c))
        .collect();

    // Build enum variant set for recall exclusion: "type::variant" (lowercase).
    let enum_variant_set: HashSet<String> = {
        let mut set = HashSet::new();
        for c in &chunks {
            if c.kind == "enum" {
                let leaf = c.name.rsplit("::").next().unwrap_or(&c.name).to_lowercase();
                for v in &c.enum_variants {
                    set.insert(format!("{leaf}::{v}"));
                }
            }
        }
        set
    };

    let mut recall_total = 0usize;
    let mut recall_resolved = 0usize;
    let mut recall_unresolved = 0usize;
    let mut recall_extern = 0usize;
    let mut miss_categories: HashMap<&str, usize> = HashMap::new();
    let mut unresolved_samples: Vec<String> = Vec::new();
    let mut extern_samples: Vec<String> = Vec::new();

    // Pre-build graph extern call lookup: chunk_idx → set of extern call names (lowercase).
    let graph_extern_set: Vec<HashSet<String>> = if has_graph_extern {
        graph.extern_calls.iter().map(|calls| {
            calls.iter().map(|(name, _)| name.to_lowercase()).collect()
        }).collect()
    } else {
        vec![HashSet::new(); n]
    };
    // Also build extern reason lookup for samples.
    let graph_extern_reasons: Vec<HashMap<String, String>> = if has_graph_extern {
        graph.extern_calls.iter().map(|calls| {
            calls.iter().map(|(name, reason)| (name.to_lowercase(), reason.clone())).collect()
        }).collect()
    } else {
        vec![HashMap::new(); n]
    };

    for i in 0..n {
        if graph.kinds[i] != "function" {
            continue;
        }
        let key = (graph.names[i].as_str(), graph.lines[i]);
        let Some(chunk) = chunk_map.get(&key) else { continue };

        let resolved_shorts = &chunk_callee_shorts[i];
        let self_short = short_name(&graph.names[i]).to_lowercase();

        // Build receiver types only for fallback path.
        let receiver_types = if !has_graph_extern {
            build_db_chunk_receiver_types(chunk, &owner_field_types, &return_type_map)
        } else {
            HashMap::new()
        };

        for call in &chunk.calls {
            let call_lower = call.to_lowercase();
            let short = if let Some((_, s)) = call_lower.rsplit_once("::") {
                s
            } else if let Some((_, s)) = call_lower.rsplit_once('.') {
                s
            } else {
                &call_lower
            };

            if !project_shorts.contains(short) {
                continue;
            }

            // Skip enum variant constructors.
            static PRELUDE_VARIANTS: &[&str] = &["ok", "err", "some", "none"];
            if call.starts_with(char::is_uppercase) && PRELUDE_VARIANTS.contains(&call_lower.as_str()) {
                continue;
            }
            if let Some((prefix, last)) = call.rsplit_once("::") {
                if last.starts_with(char::is_uppercase) {
                    let prefix_lower = prefix.to_lowercase();
                    let type_leaf = prefix_lower.rsplit_once("::").map_or(prefix_lower.as_str(), |p| p.1);
                    let variant_key = format!("{type_leaf}::{}", last.to_lowercase());
                    if enum_variant_set.contains(&variant_key) {
                        continue;
                    }
                    let prefix_leaf = prefix.rsplit("::").next().unwrap_or(prefix).to_lowercase();
                    if !project_shorts.contains(&prefix_leaf) && !project_type_shorts.contains(&prefix_leaf) {
                        continue;
                    }
                }
            }

            recall_total += 1;

            if resolved_shorts.contains(short) || short == self_short {
                recall_resolved += 1;
            } else if has_graph_extern && graph_extern_set[i].contains(&call_lower) {
                // Graph already classified this call as extern.
                recall_extern += 1;
                if extern_samples.len() < 2000 {
                    let reason = graph_extern_reasons[i].get(&call_lower)
                        .map_or("graph-extern", |r| r.as_str());
                    extern_samples.push(format!("{call}  [{reason}]"));
                }
            } else if !has_graph_extern {
                // Fallback: use ExternMethodIndex directly.
                if let Some(ref ext) = extern_index {
                    if let Some(ref all_methods) = extern_all_methods {
                        if let Some(reason) = v_code_intel::graph::check_extern(
                            &call_lower, &receiver_types, ext, &project_type_shorts,
                            &return_type_map, all_methods, &owner_field_types,
                        ) {
                            recall_extern += 1;
                            if extern_samples.len() < 2000 {
                                extern_samples.push(format!("{call}  [{reason}]"));
                            }
                            continue;
                        }
                    }
                }
                recall_unresolved += 1;
                let cat = categorize_miss(call);
                *miss_categories.entry(cat).or_default() += 1;
                if unresolved_samples.len() < 2000 {
                    unresolved_samples.push(call.clone());
                }
            } else {
                recall_unresolved += 1;
                let cat = categorize_miss(call);
                *miss_categories.entry(cat).or_default() += 1;
                if unresolved_samples.len() < 2000 {
                    unresolved_samples.push(call.clone());
                }
            }
        }
    }

    let t_recall = start.elapsed();

    // ── Output ───────────────────────────────────────────────────────
    let uniq_total = prec_total - ambig_total;
    let uniq_ok = prec_ok - (ambig_total - ambig_wrong);
    let uniq_wrong = prec_wrong - ambig_wrong;

    println!("{}", "=".repeat(60));
    println!("  Call graph accuracy verification");
    println!("{}", "=".repeat(60));

    println!("\n  [Precision] reported edges with correct call-site line");
    if prec_total > 0 {
        let pct = prec_ok as f64 / prec_total as f64 * 100.0;
        println!("  All edges:    {prec_total}  correct={prec_ok} ({pct:.1}%)  wrong={prec_wrong}");
    }
    if uniq_total > 0 {
        let pct = uniq_ok as f64 / uniq_total as f64 * 100.0;
        println!("  Unique names: {uniq_total}  correct={uniq_ok} ({pct:.1}%)  wrong={uniq_wrong}");
    }
    println!("  Ambiguous:    {ambig_total}  wrong={ambig_wrong} (same-name functions, unreliable)");

    println!("\n  [Recall] tree-sitter calls resolved by graph");
    if recall_total > 0 {
        let internal_resolved = recall_resolved + recall_extern;
        let pct = internal_resolved as f64 / recall_total as f64 * 100.0;
        let pct_graph = recall_resolved as f64 / recall_total as f64 * 100.0;
        println!(
            "  Total calls:  {recall_total}  resolved={internal_resolved} ({pct:.1}%)  unresolved={recall_unresolved}"
        );
        println!(
            "    graph:      {recall_resolved} ({pct_graph:.1}%)  extern: {recall_extern}  (std/deps type match)"
        );
    }
    if let Some(ref ext) = extern_index {
        if ext.type_count() > 0 {
            println!(
                "  Extern index: {} types, {} methods",
                ext.type_count(),
                ext.total_methods()
            );
        }
    } else if has_graph_extern {
        let total_extern: usize = graph.extern_calls.iter().map(|v| v.len()).sum();
        println!("  Extern calls (from graph): {total_extern}");
    }

    if !miss_categories.is_empty() {
        let mut cats: Vec<_> = miss_categories.into_iter().collect();
        cats.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\n  Unresolved by category:");
        for (cat, count) in &cats {
            println!("    {cat:30} {count:>5}");
        }
    }

    println!("\n  Elapsed: {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);
    println!("    load:        {:.0}ms", t_load.as_secs_f64() * 1000.0);
    println!("    index:       {:.0}ms", (t_index - t_load).as_secs_f64() * 1000.0);
    println!("    precision:   {:.0}ms", (t_precision - t_index).as_secs_f64() * 1000.0);
    println!("    recall:      {:.0}ms", (t_recall - t_precision).as_secs_f64() * 1000.0);

    // ── Wrong lines detail ───────────────────────────────────────────
    wrong_unique.sort_by(|a, b| a.0.cmp(&b.0));
    wrong_ambig.sort_by(|a, b| a.0.cmp(&b.0));

    if !wrong_unique.is_empty() {
        println!(
            "\n--- Wrong lines: unique names ({} functions) ---",
            wrong_unique.len()
        );
        for (label, errs) in &wrong_unique {
            println!("\n  {label}:");
            for e in errs {
                println!("  {e}");
            }
        }
    }
    if !wrong_ambig.is_empty() {
        println!(
            "\n--- Wrong lines: ambiguous names ({} functions, unreliable) ---",
            wrong_ambig.len()
        );
        for (label, errs) in &wrong_ambig {
            println!("\n  {label}:");
            for e in errs {
                println!("  {e}");
            }
        }
    }

    // ── Extern-classified calls (for audit) ─────────────────────────
    if !extern_samples.is_empty() {
        println!("\n--- Extern-classified calls ({} total) ---", extern_samples.len());
        for s in &extern_samples {
            println!("    {s}");
        }
    }

    // ── Top unresolved calls ─────────────────────────────────────────
    if !unresolved_samples.is_empty() {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for s in &unresolved_samples {
            *freq.entry(s.as_str()).or_default() += 1;
        }
        let mut sorted: Vec<_> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\n--- Top unresolved calls ({} unique, {} total samples) ---", sorted.len(), unresolved_samples.len());
        for (call, count) in sorted.iter().take(80) {
            println!("    {count:>4}x  {call}");
        }
    }

    Ok(())
}

/// Categorize an unresolved call for reporting.
pub(crate) fn categorize_miss(call: &str) -> &'static str {
    if call.starts_with("self.") {
        if call.matches('.').count() > 1 {
            "self.field.method"
        } else {
            "self.method"
        }
    } else if call.contains("::") {
        "Type::method (qualified)"
    } else if call.contains('.') {
        "receiver.method"
    } else {
        "bare function"
    }
}

/// Build receiver_types from a DB `ParsedChunk`.
///
/// Extends `build_receiver_type_map` with owning type (self/fields),
/// let-call bindings, and generic bound resolution.
fn build_db_chunk_receiver_types(
    chunk: &ParsedChunk,
    owner_field_types: &HashMap<String, HashMap<String, String>>,
    return_type_map: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut map = build_receiver_type_map(chunk);
    if let Some(owner) = owning_type(&chunk.name) {
        map.entry("self".to_owned()).or_insert_with(|| owner.clone());
        if let Some(fields) = owner_field_types.get(owner.as_str()) {
            for (field_name, field_type) in fields {
                map.entry(field_name.clone()).or_insert_with(|| field_type.clone());
            }
        }
    }
    for (var, callee) in &chunk.let_call_bindings {
        let callee_lower = callee.to_lowercase();
        if let Some(ret_type) = return_type_map.get(&callee_lower) {
            map.entry(var.to_lowercase()).or_insert_with(|| ret_type.clone());
        }
    }
    if let Some(ref sig) = chunk.signature {
        for (type_param, trait_bound) in &extract_generic_bounds(sig) {
            for (param_name, param_type) in &chunk.param_types {
                if param_type.to_lowercase() == *type_param {
                    map.entry(param_name.to_lowercase())
                        .or_insert_with(|| trait_bound.clone());
                }
            }
        }
    }
    map
}

fn resolve_path(rel: &str, project_root: Option<&Path>) -> String {
    if let Some(root) = project_root {
        let full = root.join(rel);
        if full.exists() {
            return full.to_string_lossy().into_owned();
        }
    }
    rel.to_owned()
}

fn load_lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().map(String::from).collect())
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "tests/verify.rs"]
mod tests;
