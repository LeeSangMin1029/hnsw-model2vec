//! Call graph accuracy verification (precision + recall).
//!
//! **Precision**: For every callee edge with a call-site line, does the callee's
//! short name actually appear on that source line?
//!
//! **Recall**: For every function call that tree-sitter extracts from the source,
//! did the graph resolve it to some callee? (Unresolved = "extern" in the graph.)

use std::collections::{BTreeMap, HashMap, HashSet};
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
    let graph = super::intel::load_or_build_graph(&db)?;

    let project_root = DbConfig::load(&db)
        .ok()
        .and_then(|c| c.input_path)
        .map(PathBuf::from);

    // Load DB chunks (already parsed, no tree-sitter needed for recall).
    let chunks = v_code_intel::loader::load_chunks(&db)?;
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
    let mut owner_field_types: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let mut return_type_map: BTreeMap<String, String> = BTreeMap::new();
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
                return_type_map.insert(c.name.to_lowercase(), resolved_type);
            }
    }

    // Build extern index (cached — fast on repeat runs).
    let extern_root = project_root.as_deref().unwrap_or(Path::new("."));
    let extern_index = ExternMethodIndex::build(extern_root);

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

    for i in 0..n {
        if graph.kinds[i] != "function" {
            continue;
        }
        let key = (graph.names[i].as_str(), graph.lines[i]);
        let Some(chunk) = chunk_map.get(&key) else { continue };

        let receiver_types = build_db_chunk_receiver_types(
            chunk, &owner_field_types, &return_type_map,
        );
        let resolved_shorts = &chunk_callee_shorts[i];
        // Self-call (recursion) is excluded from graph edges (no self-loops),
        // so we consider the chunk's own short name as resolved.
        let self_short = short_name(&graph.names[i]).to_lowercase();

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

            // Skip enum variant constructors: these look like calls to tree-sitter
            // but are not function calls.  Prelude variants (Ok, Err, Some, None)
            // and project-defined variants are excluded from recall counting.
            static PRELUDE_VARIANTS: &[&str] = &["ok", "err", "some", "none"];
            if call.starts_with(char::is_uppercase) && PRELUDE_VARIANTS.contains(&call_lower.as_str()) {
                continue;
            }
            // Qualified enum variant: `Type::Variant(args)` where Variant starts uppercase.
            // Covers both project enums (via enum_variant_set) and external enums
            // (prefix type not in project + suffix uppercase).
            if let Some((prefix, last)) = call.rsplit_once("::") {
                if last.starts_with(char::is_uppercase) {
                    if enum_variant_set.contains(&call_lower) {
                        continue;
                    }
                    // External enum variant: prefix type not in project index
                    let prefix_leaf = prefix.rsplit("::").next().unwrap_or(prefix).to_lowercase();
                    if !project_shorts.contains(&prefix_leaf) && !project_type_shorts.contains(&prefix_leaf) {
                        continue;
                    }
                }
            }

            recall_total += 1;

            if resolved_shorts.contains(short) || short == self_short {
                recall_resolved += 1;
            } else if let Some(reason) = check_extern_reason(
                &call_lower, &receiver_types, &extern_index, &project_type_shorts,
            ) {
                recall_extern += 1;
                if extern_samples.len() < 2000 {
                    extern_samples.push(format!("{call}  [{reason}]"));
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
    if extern_index.type_count() > 0 {
        println!(
            "  Extern index: {} types, {} methods",
            extern_index.type_count(),
            extern_index.total_methods()
        );
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

/// Check if a call is to an external type method. Returns the reason if extern.
///
/// Resolves receiver type via pre-built `receiver_types` map (populated from
/// param_types, local_types, owner_field_types, let_call_bindings, generic bounds).
/// Also classifies calls as extern when the method exists only in external types
/// (not in project), even without knowing the receiver type.
fn check_extern_reason(
    call_lower: &str,
    receiver_types: &BTreeMap<String, String>,
    extern_index: &ExternMethodIndex,
    project_type_shorts: &HashSet<String>,
) -> Option<String> {
    // Handle bare function calls (no receiver): `len`, `drop`, `push`, etc.
    if !call_lower.contains('.') && !call_lower.contains("::") {
        if extern_index.any_type_has_method(call_lower) {
            return Some(format!("bare-extern: {call_lower}"));
        }
        return None;
    }

    // Handle qualified path calls: `ast::Expr::cast`, `Command::new`, etc.
    if let Some((prefix, method)) = call_lower.rsplit_once("::") {
        let leaf_type = prefix.rsplit_once("::").map_or(prefix, |p| p.1);

        if extern_index.has_method(leaf_type, method) {
            return Some(format!("qualified-extern: {leaf_type}::{method}"));
        }
        if extern_index.any_type_has_method(method)
            && !project_type_shorts.contains(leaf_type)
        {
            return Some(format!("qualified-untyped-extern: {leaf_type}::{method}"));
        }
        return None;
    }

    let (receiver, method) = call_lower.rsplit_once('.')?;
    let receiver_leaf = receiver.rsplit_once('.').map_or(receiver, |p| p.1);

    // self.field.method → field_leaf lookup (field types injected from owner_field_types).
    if receiver.starts_with("self.") {
        let field = receiver.strip_prefix("self.").unwrap_or(receiver);
        let field_leaf = field.rsplit_once('.').map_or(field, |p| p.1);
        if let Some(field_type) = receiver_types.get(field_leaf) {
            let lowered = field_type.to_lowercase();
            let leaf = extract_leaf_type(&lowered);
            if extern_index.has_method(leaf, method) {
                return Some(format!("self.field: {field_leaf}:{leaf}.{method}"));
            }
        }
    }

    // Direct receiver lookup (param, local, inferred types).
    if let Some(recv_type) = receiver_types.get(receiver_leaf) {
        let lowered = recv_type.to_lowercase();
        let leaf = extract_leaf_type(&lowered);
        if extern_index.has_method(leaf, method) {
            return Some(format!("receiver: {receiver_leaf}:{leaf}.{method}"));
        }
    }

    // Fallback: if receiver type is unknown and receiver name is not a known
    // project type, but the method exists in extern types, classify as extern.
    // This handles cases like `data.len`, `name.is_empty` where the receiver
    // is clearly a variable (not a project type), and the method exists on
    // std/dep types. This avoids false classification when the receiver could
    // be a project type (e.g., `graph.len`).
    if extern_index.any_type_has_method(method)
        && !project_type_shorts.contains(receiver_leaf)
    {
        return Some(format!("untyped-extern: {receiver_leaf}.{method}"));
    }

    None
}

/// Build receiver_types from a DB `ParsedChunk`.
///
/// Extends `build_receiver_type_map` with owning type (self/fields),
/// let-call bindings, and generic bound resolution.
fn build_db_chunk_receiver_types(
    chunk: &ParsedChunk,
    owner_field_types: &BTreeMap<String, BTreeMap<String, String>>,
    return_type_map: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
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
