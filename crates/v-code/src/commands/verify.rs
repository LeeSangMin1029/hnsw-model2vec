//! Call graph accuracy verification (precision + recall).
//!
//! **Precision**: For every callee edge with a call-site line, does the callee's
//! short name actually appear on that source line?
//!
//! **Recall**: For every function call that tree-sitter extracts from the source,
//! did the graph resolve it to some callee? (Unresolved = "extern" in the graph.)

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Owner verification result for a callee edge.
enum OwnerVerdict {
    /// Owner confirmed in source (explicit type or receiver type match).
    Confirmed,
    /// `.method(` exists in source but owner cannot be confirmed.
    Unverified,
    /// Method name not found in source at all — likely wrong edge.
    Wrong,
}

use anyhow::Result;

use v_code_intel::extern_types::ExternMethodIndex;
use v_code_intel::graph::{extract_generic_bounds, owning_type, build_receiver_type_map, collect_owner_field_types, build_return_type_map, collect_project_type_shorts, infer_local_types_from_calls, infer_receiver_types_by_co_methods, build_method_owner_index};
use v_code_intel::parse::ParsedChunk;
use v_hnsw_cli::commands::db_config::DbConfig;

/// Extract the short (leaf) name from a qualified name: `"Foo::bar"` → `"bar"`.
pub(crate) fn short_name(full: &str) -> &str {
    full.rsplit("::").next().unwrap_or(full)
}

/// Verify call-site accuracy for all function chunks in the database.
#[expect(clippy::needless_range_loop, clippy::type_complexity, reason = "multiple parallel arrays indexed together")]
pub fn run(db: PathBuf, verbose: bool) -> Result<()> {
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
        }
    }
    let project_type_shorts = collect_project_type_shorts(&chunks);

    // Build owner_field_types + return_type_map from DB chunks (no file I/O).
    let owner_field_types = collect_owner_field_types(&chunks);
    let mut return_type_map = build_return_type_map(&chunks);
    let method_owners = build_method_owner_index(&chunks);

    // Build per-chunk receiver_types for precision verification.
    let chunk_receiver_types: Vec<HashMap<String, String>> = chunks.iter().map(|c| {
        let mut recv_types = build_receiver_type_map(c);
        // Enrich with owner_field_types for self.field resolution
        if let Some(owner) = owning_type(&c.name) {
            if let Some(fields) = owner_field_types.get(&owner) {
                for (field, ftype) in fields {
                    recv_types.entry(field.clone()).or_insert_with(|| ftype.clone());
                }
            }
        }
        infer_local_types_from_calls(&c.calls, &c.let_call_bindings, &return_type_map, &mut recv_types);
        infer_receiver_types_by_co_methods(&c.calls, &method_owners, &mut recv_types);
        recv_types
    }).collect();

    // Use graph-embedded extern info if available, otherwise build ExternMethodIndex.
    let has_graph_extern = !graph.extern_calls.is_empty()
        && graph.extern_calls.iter().any(|v| !v.is_empty());

    let extern_index = if has_graph_extern {
        None // Graph already has extern classification
    } else {
        Some(ExternMethodIndex::build(&db))
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
        if verbose && extern_ret_count > 0 {
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
    let mut prec_unverified = 0usize;
    let mut ambig_total = 0usize;
    let mut ambig_wrong = 0usize;
    let mut ambig_unverified = 0usize;

    // Per-language counters: (prec_total, prec_ok, recall_total, recall_resolved, recall_extern)
    let mut lang_stats: HashMap<String, (usize, usize, usize, usize, usize)> = HashMap::new();
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
            let lang = ext_to_lang(&graph.files[i]);

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
                { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.0 += 1; }
                errors.push(format!(
                    "    {callee_short} \u{2192} L{call_line}: OUT OF RANGE"
                ));
                continue;
            }

            let src = &lines[ln - 1];
            if src.contains(callee_short) {
                // Verify that the callee's owning type matches the call-site context.
                // For `Type::method()` calls the type is explicit; for `receiver.method()`
                // calls we check whether the owner appears on the source line or is a
                // self-call within the same impl block.
                let callee_name = &graph.names[ci];
                let owner_verdict = if let Some((owner_path, _)) = callee_name.rsplit_once("::") {
                    // Handle trait impl names: "Trait for Type" → extract "Type"
                    // Strip generic params: "Bm25Index<T>" → "Bm25Index"
                    let raw_leaf = owner_path.rsplit("::").next().unwrap_or(owner_path);
                    let after_for = if let Some(pos) = raw_leaf.find(" for ") {
                        &raw_leaf[pos + 5..]
                    } else {
                        raw_leaf
                    };
                    let owner_leaf = after_for.split('<').next().unwrap_or(after_for);
                    let src_lower = src.to_lowercase();
                    let owner_lower = owner_leaf.to_lowercase();
                    // self.method() calls: owner won't appear in source.
                    // Valid if caller's owning type matches callee's owner.
                    let caller_owner = graph.names[i].rsplit_once("::")
                        .map(|(p, _)| {
                            let leaf = p.rsplit("::").next().unwrap_or(p);
                            let after_for = leaf.split(" for ").last().unwrap_or(leaf);
                            after_for.split('<').next().unwrap_or(after_for).to_lowercase()
                        });
                    let is_self_call = (src_lower.contains("self.") || src_lower.contains("self::"))
                        && caller_owner.as_deref() == Some(&owner_lower);
                    // For trait impl names ("Trait for Type"), also accept if the
                    // trait name appears in the receiver's type context.
                    // e.g. dc.distance() → dc's type is the trait itself.
                    let trait_name_lower = if raw_leaf.contains(" for ") {
                        Some(raw_leaf.split(" for ").next().unwrap_or("")
                            .split('<').next().unwrap_or("").to_lowercase())
                    } else {
                        None
                    };
                    let trait_match = trait_name_lower.as_ref().map_or(false, |tn| {
                        if tn.is_empty() { return false; }
                        // Direct trait name in source or caller is the trait
                        if src_lower.contains(tn.as_str())
                            || caller_owner.as_deref() == Some(tn.as_str()) {
                            return true;
                        }
                        false
                    });
                    if is_self_call || src_lower.contains(&owner_lower) || trait_match {
                        OwnerVerdict::Confirmed
                    } else {
                        let method_lower = callee_name.rsplit_once("::").map_or("", |(_, m)| m).to_lowercase();
                        let recv_types = &chunk_receiver_types[i];

                        // Helper: check if recv_type matches owner
                        let type_matches = |recv_type: &str| -> bool {
                            let rt = recv_type.split('<').next().unwrap_or(recv_type);
                            rt == owner_lower
                                || rt.contains(&owner_lower)
                                || owner_lower.contains(rt)
                        };
                        let name_matches = |name: &str| -> bool {
                            let stripped: String = name.chars().filter(|&c| c != '_').collect();
                            stripped.len() >= 3
                                && (owner_lower.contains(&stripped)
                                    || stripped.contains(&*owner_lower))
                        };

                        // Strategy A: Use chunk's call expression for precise type checking.
                        let call_expr = chunks[i].calls.iter().zip(chunks[i].call_lines.iter())
                            .find(|(call, ln)| {
                                **ln == call_line
                                && call.to_lowercase().ends_with(&method_lower)
                            })
                            .map(|(call, _)| call.to_lowercase());

                        let mut matched = false;
                        if let Some(ref expr) = call_expr {
                            if let Some((recv_part, _)) = expr.rsplit_once('.') {
                                let recv_leaf = recv_part.rsplit_once('.').map_or(recv_part, |p| p.1);
                                if let Some(recv_type) = recv_types.get(recv_leaf) {
                                    matched = type_matches(recv_type);
                                } else {
                                    matched = name_matches(recv_leaf);
                                }
                            } else if let Some((prefix, _)) = expr.rsplit_once("::") {
                                let type_leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
                                let type_clean = type_leaf.split('<').next().unwrap_or(type_leaf);
                                matched = owner_lower.contains(type_clean)
                                    || type_clean.contains(&*owner_lower);
                            }
                        }
                        // Strategy B: Source text fallback when call_expr didn't conclude.
                        if !matched {
                            let dot_pat = format!(".{method_lower}(");
                            if let Some(pos) = src_lower.find(&dot_pat) {
                                let before = &src_lower[..pos];
                                let recv: String = before.chars().rev()
                                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                                    .collect::<String>().chars().rev().collect();
                                if name_matches(&recv) {
                                    matched = true;
                                } else if !recv.is_empty() {
                                    if let Some(rt) = recv_types.get(recv.as_str()) {
                                        matched = type_matches(rt);
                                    }
                                }
                            }
                        }
                        // Strategy C: return_type_map chain + word scan for chained calls.
                        if !matched {
                            // Try return_type_map chain: find `Type::method()` patterns in
                            // call expressions where method's return type matches callee owner.
                            if let Some(ref expr) = call_expr {
                                // e.g. expr = "hnswconfig::builder" for chained call
                                // return_type_map["hnswconfig::builder"] = "hnswconfigbuilder"
                                if let Some(ret) = return_type_map.get(expr.as_str()) {
                                    let rt = ret.split('<').next().unwrap_or(ret);
                                    matched = type_matches(rt);
                                }
                            }
                        }
                        // Strategy D: method_owners unique owner confirmation.
                        // Only accept if the method also does NOT exist on extern types,
                        // to avoid circular validation (graph uses same method_owners).
                        if !matched {
                            if let Some(owners) = method_owners.get(&method_lower) {
                                let concrete_owner = owner_lower.split(" for ").last().unwrap_or(&owner_lower);
                                let owner_has_method = owners.iter().any(|o| {
                                    let co = o.split(" for ").last().unwrap_or(o);
                                    co == concrete_owner
                                });
                                let is_extern_method = extern_all_methods.as_ref()
                                    .map_or(false, |m| m.contains(&method_lower));
                                if owner_has_method && owners.len() == 1 && !is_extern_method {
                                    matched = true;
                                }
                            }
                        }
                        // Strategy E: co-method on same receiver confirms owner.
                        if !matched {
                            if let Some(ref expr) = call_expr {
                                if let Some((recv_part, _)) = expr.rsplit_once('.') {
                                    let recv_leaf = recv_part.rsplit_once('.').map_or(recv_part, |p| p.1);
                                    let concrete_owner = owner_lower.split(" for ").last().unwrap_or(&owner_lower);
                                    for other_call in &chunks[i].calls {
                                        let other_lower = other_call.to_lowercase();
                                        if let Some((other_recv, _)) = other_lower.rsplit_once('.') {
                                            let other_recv_leaf = other_recv.rsplit_once('.').map_or(other_recv, |p| p.1);
                                            if other_recv_leaf == recv_leaf && other_lower != *expr {
                                                if let Some(other_method) = other_lower.rsplit_once('.').map(|p| p.1) {
                                                    let other_is_extern = extern_all_methods.as_ref()
                                                        .map_or(false, |m| m.contains(other_method));
                                                    if !other_is_extern {
                                                        if let Some(owners) = method_owners.get(other_method) {
                                                            if owners.iter().any(|o| {
                                                                let co = o.split(" for ").last().unwrap_or(o);
                                                                co == concrete_owner
                                                            }) {
                                                                matched = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Strategy F: trait impl callee — receiver type matches trait.
                        if !matched && trait_name_lower.is_some() {
                            let method_lower = callee_name.rsplit_once("::").map_or("", |(_, m)| m).to_lowercase();
                            let dot_method = format!(".{method_lower}(");
                            if src_lower.contains(&dot_method) {
                                if let Some(ref expr) = call_expr {
                                    if let Some((recv_part, _)) = expr.rsplit_once('.') {
                                        let recv_leaf = recv_part.rsplit_once('.').map_or(recv_part, |p| p.1);
                                        let tn = trait_name_lower.as_ref().expect("checked above");
                                        if let Some(rt) = recv_types.get(recv_leaf) {
                                            let rt_leaf = rt.split('<').next().unwrap_or(rt);
                                            if rt_leaf == tn.as_str()
                                                || rt_leaf == owner_lower
                                                || type_matches(rt_leaf) {
                                                matched = true;
                                            }
                                        }
                                        if !matched && name_matches(recv_leaf) {
                                            matched = true;
                                        }
                                    }
                                }
                            }
                        }
                        if matched {
                            OwnerVerdict::Confirmed
                        } else {
                            // Check if `.method(` exists in source — if so, the call
                            // site has a method invocation but we can't confirm the owner.
                            let method_lower = callee_name.rsplit_once("::").map_or("", |(_, m)| m).to_lowercase();
                            let dot_method = format!(".{method_lower}(");
                            let colon_method = format!("::{method_lower}(");
                            if src_lower.contains(&dot_method) || src_lower.contains(&colon_method) {
                                OwnerVerdict::Unverified
                            } else {
                                OwnerVerdict::Wrong
                            }
                        }
                    }
                } else {
                    OwnerVerdict::Confirmed // bare function, no owner to check
                };

                match owner_verdict {
                    OwnerVerdict::Wrong => {
                        prec_wrong += 1;
                        if is_ambiguous { ambig_wrong += 1; }
                        { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.0 += 1; }
                        let truncated: String = src.trim().chars().take(70).collect();
                        errors.push(format!(
                            "    {} \u{2192} L{call_line}: wrong '{truncated}'",
                            callee_name,
                        ));
                    }
                    OwnerVerdict::Unverified => {
                        prec_unverified += 1;
                        if is_ambiguous { ambig_unverified += 1; }
                        { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.0 += 1; }
                        let truncated: String = src.trim().chars().take(70).collect();
                        errors.push(format!(
                            "    {} \u{2192} L{call_line}: unverified '{truncated}'",
                            callee_name,
                        ));
                    }
                    OwnerVerdict::Confirmed => {
                        prec_ok += 1;
                        let ls = lang_stats.entry(lang.to_owned()).or_default();
                        ls.0 += 1; ls.1 += 1;
                    }
                }
            } else {
                // callee short name not found in source line at all
                prec_wrong += 1;
                if is_ambiguous {
                    ambig_wrong += 1;
                }
                { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.0 += 1; }
                let truncated: String = src.trim().chars().take(70).collect();
                errors.push(format!(
                    "    {} \u{2192} L{call_line}: wrong '{truncated}'",
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
                }
                // For qualified calls Type::method, if Type is not a project type
                // and not an internal path (self/super/crate), it's extern.
                let prefix_leaf = prefix.rsplit("::").next().unwrap_or(prefix).to_lowercase();
                let is_internal_path = prefix_leaf == "super" || prefix_leaf == "crate"
                    || prefix_leaf == "self";
                if !is_internal_path
                    && !project_type_shorts.contains(&prefix_leaf)
                {
                    recall_total += 1;
                    recall_extern += 1;
                    let lang = ext_to_lang(&graph.files[i]);
                    { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.2 += 1; ls.4 += 1; }
                    continue;
                }
            }

            recall_total += 1;
            let lang = ext_to_lang(&graph.files[i]);

            if resolved_shorts.contains(short) || short == self_short {
                recall_resolved += 1;
                { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.2 += 1; ls.3 += 1; }
            } else if has_graph_extern && graph_extern_set[i].contains(&call_lower) {
                // Graph already classified this call as extern.
                recall_extern += 1;
                { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.2 += 1; ls.4 += 1; }
                if extern_samples.len() < 2000 {
                    let reason = graph_extern_reasons[i].get(&call_lower)
                        .map_or("graph-extern", |r| r.as_str());
                    extern_samples.push(format!("{call}  [{reason}]"));
                }
            } else if !has_graph_extern {
                // Fallback: use ExternMethodIndex directly.
                if let Some(ref ext) = extern_index {
                    if let Some(ref all_methods) = extern_all_methods {
                        let empty_owners = std::collections::HashMap::new();
                        if let Some(reason) = v_code_intel::graph::check_extern(
                            &call_lower, &receiver_types, ext, &project_type_shorts,
                            &return_type_map, all_methods, &owner_field_types,
                            None, &empty_owners,
                        ) {
                            recall_extern += 1;
                            { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.2 += 1; ls.4 += 1; }
                            if extern_samples.len() < 2000 {
                                extern_samples.push(format!("{call}  [{reason}]"));
                            }
                            continue;
                        }
                    }
                }
                recall_unresolved += 1;
                { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.2 += 1; }
                let cat = categorize_miss(call);
                *miss_categories.entry(cat).or_default() += 1;
                if unresolved_samples.len() < 2000 {
                    unresolved_samples.push(call.clone());
                }
            } else {
                // Graph has extern data but this call wasn't classified.
                // Try ExternMethodIndex as fallback before marking unresolved.
                let mut is_extern = false;
                if let Some(ref ext) = extern_index {
                    if let Some(ref all_methods) = extern_all_methods {
                        let empty_owners = std::collections::HashMap::new();
                        if let Some(reason) = v_code_intel::graph::check_extern(
                            &call_lower, &receiver_types, ext, &project_type_shorts,
                            &return_type_map, all_methods, &owner_field_types,
                            None, &empty_owners,
                        ) {
                            recall_extern += 1;
                            { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.2 += 1; ls.4 += 1; }
                            if extern_samples.len() < 2000 {
                                extern_samples.push(format!("{call}  [{reason}]"));
                            }
                            is_extern = true;
                        }
                    }
                }
                if !is_extern {
                    recall_unresolved += 1;
                    { let ls = lang_stats.entry(lang.to_owned()).or_default(); ls.2 += 1; }
                    let cat = categorize_miss(call);
                    *miss_categories.entry(cat).or_default() += 1;
                    if unresolved_samples.len() < 2000 {
                        unresolved_samples.push(format!("{call}  [in {}]", graph.names[i]));
                    }
                }
            }
        }
    }

    let t_recall = start.elapsed();

    // ── Output ───────────────────────────────────────────────────────
    let uniq_total = prec_total - ambig_total;
    let uniq_ok = prec_ok - (ambig_total - ambig_wrong - ambig_unverified);
    let uniq_wrong = prec_wrong - ambig_wrong;
    let uniq_unverified = prec_unverified - ambig_unverified;

    // ── Precision summary ────────────────────────────────────────
    // confirmed = owner verified in source.  unverified = .method( exists but owner unconfirmed.
    let verifiable = prec_ok + prec_wrong;
    let prec_pct = if verifiable > 0 { prec_ok as f64 / verifiable as f64 * 100.0 } else { 100.0 };
    let recall_internal = recall_resolved + recall_extern;
    let recall_pct = if recall_total > 0 { recall_internal as f64 / recall_total as f64 * 100.0 } else { 100.0 };
    println!(
        "=== verify: precision {prec_pct:.1}% ({prec_ok}/{verifiable}), recall {recall_pct:.1}% ({recall_internal}/{recall_total}) ===\n"
    );

    println!("  [Precision] reported edges with correct call-site line");
    if prec_total > 0 {
        println!("  {:30} {:>8}  {:>8}  {:>8}  {:>10}", "scope", "total", "confirmed", "wrong", "unverified");
        println!("  {}", "-".repeat(72));
        println!(
            "  {:30} {:>8}  {:>8}  {:>8}  {:>10}",
            "all edges", prec_total, prec_ok, prec_wrong, prec_unverified
        );
        if uniq_total > 0 {
            println!(
                "  {:30} {:>8}  {:>8}  {:>8}  {:>10}",
                "unique names", uniq_total, uniq_ok, uniq_wrong, uniq_unverified
            );
        }
        println!(
            "  {:30} {:>8}  {:>8}  {:>8}  {:>10}",
            "ambiguous (unreliable)", ambig_total, ambig_total - ambig_wrong - ambig_unverified, ambig_wrong, ambig_unverified
        );
    }

    println!("\n  [Recall] tree-sitter calls resolved by graph");
    if recall_total > 0 {
        let pct_graph = recall_resolved as f64 / recall_total as f64 * 100.0;
        println!("  {:30} {:>8}  {:>8}", "category", "count", "%");
        println!("  {}", "-".repeat(50));
        println!("  {:30} {:>8}  {:>7.1}%", "total calls", recall_total, recall_pct);
        println!("  {:30} {:>8}  {:>7.1}%", "  graph resolved", recall_resolved, pct_graph);
        println!("  {:30} {:>8}", "  extern (std/deps)", recall_extern);
        println!("  {:30} {:>8}", "  unresolved", recall_unresolved);
    }
    if let Some(ref ext) = extern_index {
        if ext.type_count() > 0 {
            println!(
                "  {:30} {} types, {} methods",
                "extern index", ext.type_count(), ext.total_methods()
            );
        }
    } else if has_graph_extern {
        let total_extern: usize = graph.extern_calls.iter().map(|v| v.len()).sum();
        println!("  {:30} {total_extern}", "extern calls (from graph)");
    }

    if !miss_categories.is_empty() {
        let mut cats: Vec<_> = miss_categories.into_iter().collect();
        cats.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\n  [Unresolved by category]");
        for (cat, count) in &cats {
            println!("    {cat:30} {count:>5}");
        }
    }

    // ── Per-language breakdown ─────────────────────────────────────
    if lang_stats.len() > 1 {
        let mut langs: Vec<_> = lang_stats.iter().collect();
        langs.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(b.1.2.cmp(&a.1.2)));
        println!("\n  [Per-language]");
        println!("  {:12} {:>8} {:>8} {:>8}  {:>8} {:>8} {:>8}", "lang", "p_total", "p_ok", "p_%", "r_total", "r_res", "r_%");
        println!("  {}", "-".repeat(70));
        for (lang, (pt, po, rt, rr, re)) in &langs {
            let pp = if *pt > 0 { *po as f64 / *pt as f64 * 100.0 } else { 100.0 };
            let rp = if *rt > 0 { (*rr + *re) as f64 / *rt as f64 * 100.0 } else { 100.0 };
            println!("  {:12} {:>8} {:>8} {:>7.1}%  {:>8} {:>8} {:>7.1}%", lang, pt, po, pp, rt, rr + re, rp);
        }
    }

    println!(
        "\n  [Elapsed] {:.1}ms  (load {:.0}, index {:.0}, prec {:.0}, recall {:.0})",
        start.elapsed().as_secs_f64() * 1000.0,
        t_load.as_secs_f64() * 1000.0,
        (t_index - t_load).as_secs_f64() * 1000.0,
        (t_precision - t_index).as_secs_f64() * 1000.0,
        (t_recall - t_precision).as_secs_f64() * 1000.0,
    );

    // ── Verbose detail sections ─────────────────────────────────────
    if verbose {
        wrong_unique.sort_by(|a, b| a.0.cmp(&b.0));
        wrong_ambig.sort_by(|a, b| a.0.cmp(&b.0));

        if !wrong_unique.is_empty() {
            println!(
                "\n  [Wrong lines] unique names ({} functions)",
                wrong_unique.len()
            );
            for (label, errs) in &wrong_unique {
                println!("\n  {label}:");
                for e in errs {
                    println!("      {e}");
                }
            }
        }
        if !wrong_ambig.is_empty() {
            println!(
                "\n  [Wrong lines] ambiguous names ({} functions, unreliable)",
                wrong_ambig.len()
            );
            for (label, errs) in &wrong_ambig {
                println!("\n  {label}:");
                for e in errs {
                    println!("      {e}");
                }
            }
        }

        if !extern_samples.is_empty() {
            println!("\n  [Extern-classified] {} total", extern_samples.len());
            for s in &extern_samples {
                println!("    {s}");
            }
        }

        if !unresolved_samples.is_empty() {
            let mut freq: HashMap<&str, usize> = HashMap::new();
            for s in &unresolved_samples {
                *freq.entry(s.as_str()).or_default() += 1;
            }
            let mut sorted: Vec<_> = freq.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            println!("\n  [Top unresolved] {} unique, {} total", sorted.len(), unresolved_samples.len());
            for (call, count) in &sorted {
                println!("    {count:>4}x  {call}");
            }
        }
    }

    Ok(())
}

/// Map file extension to language name for per-language stats.
fn ext_to_lang(file: &str) -> &'static str {
    let ext = file.rsplit_once('.').map_or("", |p| p.1);
    match ext {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" => "javascript",
        "java" => "java",
        "go" => "go",
        "c" => "c",
        "cpp" | "cc" | "cxx" | "h" | "hpp" => "c++",
        "cs" => "c#",
        "kt" | "kts" => "kotlin",
        "swift" => "swift",
        "rb" => "ruby",
        _ => "other",
    }
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
