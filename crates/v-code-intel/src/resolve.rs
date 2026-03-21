//! Call resolution engine — resolves call strings to chunk indices.
//!
//! Core of the call graph builder: given a call like `"self.engine.search"`,
//! determines which chunk (function) it refers to using type inference,
//! import resolution, and heuristic matching.

use std::collections::{HashMap, HashSet};
use crate::parse::ParsedChunk;

// ── Shared context ──────────────────────────────────────────────────

/// Read-only context shared across all resolution calls within a graph build.
pub(crate) struct ResolveCtx<'a> {
    pub exact: &'a HashMap<String, u32>,
    pub short: &'a HashMap<String, u32>,
    pub kinds: &'a [String],
    pub names: &'a [String],
    pub return_type_map: &'a HashMap<String, String>,
    pub method_owners: &'a HashMap<String, Vec<String>>,
    pub instantiated: &'a HashSet<String>,
    pub trait_impl_methods: &'a HashMap<String, Vec<String>>,
    pub enum_variants: &'a HashSet<String>,
    pub extern_all_methods: &'a HashSet<String>,
}

/// Per-chunk context (imports, receiver types, etc.)
pub(crate) struct ChunkCtx {
    pub imports: HashMap<String, String>,
    pub self_type: Option<String>,
    pub call_types_set: HashSet<String>,
    pub import_leaves: HashSet<String>,
    pub source_types_set: HashSet<String>,
    pub receiver_types: HashMap<String, String>,
}

impl ChunkCtx {
    /// Shallow clone for delta pass (receiver_types may have been enriched).
    pub fn clone_for_delta(&self) -> Self {
        Self {
            imports: self.imports.clone(),
            self_type: self.self_type.clone(),
            call_types_set: self.call_types_set.clone(),
            import_leaves: self.import_leaves.clone(),
            source_types_set: self.source_types_set.clone(),
            receiver_types: self.receiver_types.clone(),
        }
    }
}

// ── Unified resolve entry point ─────────────────────────────────────

/// Resolve all calls in a chunk. Combines the old resolve_collect + resolve_delta.
///
/// When `skip_resolved` is None: full pass (pass 1) — also returns receiver_updates + type_edges.
/// When `skip_resolved` is Some: delta pass — only resolves previously-unresolved calls.
pub(crate) fn resolve_calls(
    src: usize,
    chunk: &ParsedChunk,
    ctx: &mut ChunkCtx,
    rctx: &ResolveCtx<'_>,
    skip_resolved: Option<&[bool]>,
) -> (Vec<(u32, u32)>, Vec<(u32, u32)>, HashMap<String, String>, Vec<bool>) {
    let is_full = skip_resolved.is_none();
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut receiver_updates: HashMap<String, String> = HashMap::new();
    let mut resolved_set = vec![false; chunk.calls.len()];

    let passes = if is_full { 2 } else { 1 };
    for pass in 0..passes {
        for (call_idx, call) in chunk.calls.iter().enumerate() {
            if let Some(skip) = skip_resolved {
                if call_idx < skip.len() && skip[call_idx] { continue; }
            }
            if resolved_set[call_idx] { continue; }

            if let Some(tgt) = resolve_with_imports(call, ctx, rctx) {
                if tgt as usize != src {
                    let line = chunk.call_lines.get(call_idx).copied().unwrap_or(0);
                    edges.push((tgt, line));
                }
                resolved_set[call_idx] = true;

                // On first pass, track return types for let bindings.
                if is_full && pass == 0 {
                    update_receiver_from_call(call, chunk, ctx, rctx, &mut receiver_updates);
                }
            }
        }
    }

    // Type ref edges (only on full pass).
    let mut type_edges: Vec<(u32, u32)> = Vec::new();
    if is_full {
        for ty in &chunk.types {
            let lower = ty.to_lowercase();
            let tgt = ctx.imports.get(&lower).and_then(|q| rctx.exact.get(q).copied())
                .or_else(|| rctx.exact.get(&lower).copied())
                .or_else(|| rctx.short.get(&lower).copied());
            if let Some(tgt) = tgt { type_edges.push((tgt, 0)); }
        }
    }

    (edges, type_edges, receiver_updates, resolved_set)
}

/// Track return type info from resolved calls for receiver type inference.
fn update_receiver_from_call(
    call: &str,
    chunk: &ParsedChunk,
    ctx: &mut ChunkCtx,
    rctx: &ResolveCtx<'_>,
    updates: &mut HashMap<String, String>,
) {
    let call_lower = call.to_lowercase();
    let ret_type = rctx.return_type_map.get(&call_lower).or_else(|| {
        let (recv, method) = call_lower.rsplit_once('.')?;
        let recv_leaf = recv.rsplit_once('.').map_or(recv, |p| p.1);
        let recv_type = ctx.receiver_types.get(recv_leaf)?;
        rctx.return_type_map.get(&format!("{recv_type}::{method}"))
    });
    if let Some(ret_type) = ret_type {
        for (var_name, callee_name) in &chunk.let_call_bindings {
            if callee_name.to_lowercase() == call_lower {
                let key = var_name.to_lowercase();
                updates.entry(key.clone()).or_insert_with(|| ret_type.clone());
                ctx.receiver_types.entry(key).or_insert_with(|| ret_type.clone());
            }
        }
    }
}

// ── Core resolution function ────────────────────────────────────────

/// Resolve a single call string to a chunk index.
///
/// Resolution cascade:
/// 1. Exact match (fully qualified name)
/// 2. Self::method / super::func
/// 3. self.method / self.field.method (owner type + field type lookup)
/// 4. Import-qualified (Foo::bar via use)
/// 5. Return-type chain (receiver.method via return_type_map)
/// 6. Short name fallback (Type::method / receiver.method heuristics)
fn resolve_with_imports(
    call: &str,
    ctx: &ChunkCtx,
    rctx: &ResolveCtx<'_>,
) -> Option<u32> {
    // Normalize UFCS: `<Foo as Trait>::func` → `Foo::func`
    let call_owned;
    let call: &str = if let Some(rest) = call.strip_prefix('<') {
        if let Some((inner, suffix)) = rest.split_once(">::") {
            let type_part = inner.split_once(" as ").map_or(inner, |p| p.0);
            call_owned = format!("{type_part}::{suffix}");
            &call_owned
        } else { call }
    } else { call };
    let lower = call.to_lowercase();

    // Skip prelude variants: Ok(), Err(), Some(), None
    static PRELUDE: &[&str] = &["ok", "err", "some", "none"];
    if call.starts_with(char::is_uppercase) && PRELUDE.contains(&lower.as_str()) {
        return None;
    }

    // Skip enum variant constructors: `Type::Variant(args)`
    if let Some((prefix, _name)) = lower.rsplit_once("::") {
        let orig_last = call.rsplit_once("::").map_or(call, |p| p.1);
        if orig_last.starts_with(char::is_uppercase) {
            let type_leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
            if rctx.enum_variants.contains(&format!("{type_leaf}::{_name}")) {
                return None;
            }
        }
    }

    // 1. Exact match
    if let Some(&idx) = rctx.exact.get(&lower) { return Some(idx); }

    // 2. Self::method → OwningType::method
    if let Some(method) = lower.strip_prefix("self::") {
        if let Some(owner) = ctx.self_type.as_deref() {
            if let Some(idx) = try_qualified(rctx.exact, owner, method) { return Some(idx); }
        }
    }

    // 2b. super::func → short lookup
    if let Some(rest) = lower.strip_prefix("super::") {
        let leaf = rest.rsplit_once("::").map_or(rest, |p| p.1);
        if let Some(idx) = short_fn(rctx, leaf) { return Some(idx); }
    }

    // 3. self.method / self.field.method
    if let Some(method) = lower.strip_prefix("self.") {
        return resolve_self_method(method, ctx, rctx);
    }

    // 4. Import-qualified: "Foo::bar" → imports["foo"] + "::bar"
    if let Some((prefix, suffix)) = lower.split_once("::") {
        let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
        if let Some(qualified) = ctx.imports.get(leaf) {
            if let Some(&idx) = rctx.exact.get(&format!("{qualified}::{suffix}")) {
                return Some(idx);
            }
        }
    }

    // 4b. Direct import match
    if let Some(qualified) = ctx.imports.get(&lower) {
        if let Some(&idx) = rctx.exact.get(qualified) { return Some(idx); }
    }

    // 5. Return-type chain: receiver.method via return_type_map
    if let Some((receiver, method)) = lower.rsplit_once('.') {
        let recv_leaf = receiver.rsplit_once('.').map_or(receiver, |p| p.1);
        if let Some(ret_type) = rctx.return_type_map.get(recv_leaf) {
            let receiver_is_known = rctx.exact.contains_key(recv_leaf)
                || ctx.imports.contains_key(recv_leaf)
                || rctx.return_type_map.contains_key(recv_leaf);
            if receiver_is_known {
                if let Some(idx) = try_qualified_validated(rctx, ret_type, method) {
                    return Some(idx);
                }
            }
        }
    }

    // 6. Fallback: Type::method or receiver.method
    if let Some((prefix, suffix)) = lower.rsplit_once("::") {
        return resolve_qualified_fallback(call, prefix, suffix, rctx);
    }
    if let Some((receiver, method)) = lower.rsplit_once('.') {
        return resolve_receiver_method(receiver, method, ctx, rctx);
    }
    None
}

// ── Resolution sub-routines ─────────────────────────────────────────

/// Resolve self.method or self.field.method calls.
fn resolve_self_method(method: &str, ctx: &ChunkCtx, rctx: &ResolveCtx<'_>) -> Option<u32> {
    let leaf_method = method.rsplit_once('.').map_or(method, |p| p.1);
    let is_field_chain = method.contains('.');

    // Direct self.method() → OwningType::method
    if !is_field_chain {
        if let Some(owner) = ctx.self_type.as_deref() {
            if let Some(idx) = try_qualified(rctx.exact, owner, leaf_method) { return Some(idx); }
        }
    }

    // self.field.method() → look up field type, then Type::method
    if let Some((field_path, _)) = method.rsplit_once('.') {
        let field_leaf = field_path.rsplit_once('.').map_or(field_path, |p| p.1);

        // Direct field type lookup
        if let Some(field_type) = ctx.receiver_types.get(field_leaf) {
            if let Some(idx) = try_qualified(rctx.exact, field_type, leaf_method) { return Some(idx); }
            // Known type but no match — skip unless generic param
            if !(field_type.len() == 1 && field_type.as_bytes()[0].is_ascii_lowercase()) {
                return None;
            }
        }

        // Heuristic: field name similarity with method owners
        let field_stripped = strip_underscores(field_leaf);
        if let Some(idx) = find_owner_by_name(rctx, leaf_method, field_leaf, &field_stripped) {
            return Some(idx);
        }

        // Try field_leaf as import
        if let Some(qualified) = ctx.imports.get(field_leaf) {
            if let Some(idx) = try_qualified(rctx.exact, qualified, leaf_method) { return Some(idx); }
        }

        // RTA-filtered unique owner with field name match
        if let Some(owners) = rctx.method_owners.get(leaf_method) {
            let rta = rta_filter(owners, rctx);
            if rta.len() == 1 && name_matches(field_leaf, &field_stripped, rta[0]) {
                if let Some(idx) = try_qualified(rctx.exact, rta[0], leaf_method) { return Some(idx); }
            }
        }
    }
    None
}

/// Resolve Type::method with short name fallback.
fn resolve_qualified_fallback(
    orig_call: &str,
    prefix: &str,
    suffix: &str,
    rctx: &ResolveCtx<'_>,
) -> Option<u32> {
    // Skip uppercase suffix (enum variant constructor)
    let orig_suffix = orig_call.rsplit_once("::").map_or(orig_call, |p| p.1);
    if orig_suffix.starts_with(char::is_uppercase) { return None; }

    let prefix_leaf = prefix.rsplit("::").next().unwrap_or(prefix);
    // Try exact with type leaf: "expr::cast"
    if let Some(&idx) = rctx.exact.get(&format!("{prefix_leaf}::{suffix}")) {
        return Some(idx);
    }

    // Short fallback only if prefix is a known project type
    if rctx.exact.contains_key(prefix_leaf) {
        if let Some(idx) = short_fn(rctx, suffix) {
            // Validate resolved owner matches prefix
            let resolved_owner = rctx.names.get(idx as usize)
                .and_then(|n| n.rsplit_once("::"))
                .map(|(p, _)| {
                    let leaf = p.rsplit("::").next().unwrap_or(p);
                    leaf.split(" for ").last().unwrap_or(leaf).split('<').next().unwrap_or(leaf)
                })
                .unwrap_or("");
            if name_matches(prefix_leaf, &strip_underscores(prefix_leaf), resolved_owner) {
                return Some(idx);
            }
        }
    }
    None
}

/// Resolve receiver.method() calls via type evidence.
fn resolve_receiver_method(
    receiver: &str,
    method: &str,
    ctx: &ChunkCtx,
    rctx: &ResolveCtx<'_>,
) -> Option<u32> {
    let recv_leaf = receiver.rsplit_once('.').map_or(receiver, |p| p.1);
    let recv_stripped = strip_underscores(recv_leaf);

    // A. Known receiver type from receiver_types
    if let Some(recv_type) = ctx.receiver_types.get(recv_leaf) {
        if recv_type == "<extern>" { return None; }
        return resolve_typed_receiver(recv_leaf, &recv_stripped, recv_type, method, ctx, rctx);
    }

    // B. Name-based: receiver name matches a known type/import
    if ctx.imports.contains_key(recv_leaf) || rctx.exact.contains_key(recv_leaf) {
        if let Some(idx) = resolve_by_owner_match(recv_leaf, &recv_stripped, method, rctx) {
            return Some(idx);
        }
    }

    // C. Method owner disambiguation (no receiver type info)
    if let Some(idx) = resolve_by_method_owners(recv_leaf, &recv_stripped, method, ctx, rctx) {
        return Some(idx);
    }

    // D. Trait impl fallback (method_owners has no entry)
    if rctx.method_owners.get(method).is_none() {
        if let Some(idx) = resolve_trait_impl(recv_leaf, &recv_stripped, method, rctx) {
            return Some(idx);
        }
    }

    None
}

/// Resolve when receiver type is known (from param_types, local_types, LSP).
fn resolve_typed_receiver(
    recv_leaf: &str,
    recv_stripped: &str,
    recv_type: &str,
    method: &str,
    ctx: &ChunkCtx,
    rctx: &ResolveCtx<'_>,
) -> Option<u32> {
    let ty_lower = recv_type.to_lowercase();

    // Check if receiver type is a trait
    let recv_is_trait = rctx.exact.get(&ty_lower)
        .and_then(|&idx| rctx.kinds.get(idx as usize))
        .map_or(false, |k| k == "trait");

    // Verify type actually owns this method
    let type_has_method = recv_is_trait
        || rctx.method_owners.get(method).map_or(true, |owners|
            owners.iter().any(|o| *o == ty_lower
                || o.split(" for ").last().map_or(false, |c| c == ty_lower)));

    if type_has_method {
        // Direct type::method
        if let Some(idx) = try_qualified(rctx.exact, &ty_lower, method) { return Some(idx); }

        // Trait dispatch → concrete impl
        if let Some(impl_types) = rctx.trait_impl_methods.get(method) {
            if impl_types.len() == 1 {
                if let Some(idx) = try_qualified(rctx.exact, &impl_types[0], method) { return Some(idx); }
            } else {
                // Prefer source/call type match, then name similarity
                let matched = impl_types.iter()
                    .find(|t| ctx.source_types_set.contains(t.as_str()) || ctx.call_types_set.contains(t.as_str()))
                    .or_else(|| impl_types.iter().find(|t| name_matches(recv_leaf, recv_stripped, t)));
                if let Some(t) = matched {
                    if let Some(idx) = try_qualified(rctx.exact, t, method) { return Some(idx); }
                }
            }
        }

        // method_owners with type name matching
        let ty_stripped = strip_underscores(&ty_lower);
        if let Some(owners) = rctx.method_owners.get(method) {
            if let Some(owner) = owners.iter().find(|o| name_matches(&ty_lower, &ty_stripped, o)) {
                if let Some(idx) = try_qualified(rctx.exact, owner, method) { return Some(idx); }
            }
        }

        // Trait receiver → short_fn fallback (proc-macro generated methods)
        if recv_is_trait {
            if let Some(idx) = short_fn(rctx, method) { return Some(idx); }
        }
    }

    // Known type not in project + method is extern → skip
    if !rctx.exact.contains_key(&ty_lower) && rctx.extern_all_methods.contains(method) {
        return None;
    }

    // Fall through to method_owners disambiguation
    resolve_by_method_owners(recv_leaf, recv_stripped, method, &ChunkCtx {
        imports: HashMap::new(), self_type: None,
        call_types_set: HashSet::new(), import_leaves: HashSet::new(),
        source_types_set: ctx.source_types_set.clone(),
        receiver_types: ctx.receiver_types.clone(),
    }, rctx)
}

/// Resolve receiver.method by matching receiver name to method owners.
fn resolve_by_owner_match(
    recv_leaf: &str,
    recv_stripped: &str,
    method: &str,
    rctx: &ResolveCtx<'_>,
) -> Option<u32> {
    if let Some(owners) = rctx.method_owners.get(method) {
        let matched = owners.iter().find(|o| {
            let concrete = o.split(" for ").last().unwrap_or(o);
            name_matches(recv_leaf, recv_stripped, concrete)
        });
        if let Some(owner) = matched {
            let concrete = owner.split(" for ").last().unwrap_or(owner);
            if let Some(idx) = try_qualified(rctx.exact, concrete, method) { return Some(idx); }
            if let Some(idx) = try_qualified(rctx.exact, owner, method) { return Some(idx); }
        }
    } else if let Some(impl_types) = rctx.trait_impl_methods.get(method) {
        let matched = impl_types.iter().find(|t| name_matches(recv_leaf, recv_stripped, t));
        if let Some(t) = matched {
            if let Some(idx) = try_qualified(rctx.exact, t, method) { return Some(idx); }
        }
    }
    None
}

/// Resolve using method_owners with RTA + name similarity + self_type.
fn resolve_by_method_owners(
    recv_leaf: &str,
    recv_stripped: &str,
    method: &str,
    ctx: &ChunkCtx,
    rctx: &ResolveCtx<'_>,
) -> Option<u32> {
    let owners = rctx.method_owners.get(method)?;
    let rta = rta_filter(owners, rctx);
    let method_is_extern = rctx.extern_all_methods.contains(method);

    if rta.len() == 1 {
        let owner = rta[0];
        if name_matches(recv_leaf, recv_stripped, owner) {
            if let Some(idx) = try_qualified(rctx.exact, owner, method) { return Some(idx); }
        }
        if method_is_extern { return None; }
    } else if owners.len() > 1 {
        // Name similarity match
        if recv_stripped.len() >= 3 {
            if let Some(owner) = owners.iter().find(|o| name_matches(recv_leaf, recv_stripped, o)) {
                if let Some(idx) = try_qualified(rctx.exact, owner, method) { return Some(idx); }
            }
        }
        // Self type match (only if receiver name matches self_type)
        if let Some(st) = ctx.self_type.as_deref() {
            if owners.contains(&st.to_owned())
                && recv_stripped.len() >= 3
                && name_matches(recv_leaf, recv_stripped, st)
            {
                if let Some(idx) = try_qualified(rctx.exact, st, method) { return Some(idx); }
            }
        }
        // Known receiver type from receiver_types
        if let Some(recv_type) = ctx.receiver_types.get(recv_leaf) {
            if owners.contains(recv_type) {
                if let Some(idx) = try_qualified(rctx.exact, recv_type, method) { return Some(idx); }
            }
        }
    }
    None
}

/// Resolve via trait_impl_methods when method_owners has no entry.
fn resolve_trait_impl(
    recv_leaf: &str,
    recv_stripped: &str,
    method: &str,
    rctx: &ResolveCtx<'_>,
) -> Option<u32> {
    let impl_types = rctx.trait_impl_methods.get(method)?;
    let matched = impl_types.iter().find(|t| name_matches(recv_leaf, recv_stripped, t));
    if let Some(t) = matched {
        return try_qualified(rctx.exact, t, method);
    }
    None
}

// ── Shared helpers ──────────────────────────────────────────────────

/// Check if a callable chunk (not enum/struct/trait, not uppercase method).
fn is_callable(idx: u32, rctx: &ResolveCtx<'_>) -> bool {
    let kind = rctx.kinds.get(idx as usize).map(|s| s.as_str()).unwrap_or("");
    if matches!(kind, "enum" | "struct" | "trait") { return false; }
    if let Some(name) = rctx.names.get(idx as usize) {
        if let Some(method) = name.rsplit("::").next() {
            if method.starts_with(|c: char| c.is_uppercase()) { return false; }
        }
    }
    true
}

/// Short name lookup with callable filter.
fn short_fn(rctx: &ResolveCtx<'_>, key: &str) -> Option<u32> {
    let idx = rctx.short.get(key).copied()?;
    if is_callable(idx, rctx) { Some(idx) } else { None }
}

/// Try exact lookup: `format!("{owner}::{method}")`.
fn try_qualified(exact: &HashMap<String, u32>, owner: &str, method: &str) -> Option<u32> {
    exact.get(&format!("{owner}::{method}")).copied()
}

/// Try qualified lookup with method_owners validation (prevents false positives).
fn try_qualified_validated(rctx: &ResolveCtx<'_>, owner: &str, method: &str) -> Option<u32> {
    let idx = try_qualified(rctx.exact, owner, method)?;
    let valid = rctx.method_owners.get(method).map_or(true, |owners|
        owners.iter().any(|o| o == owner
            || o.split(" for ").last().map_or(false, |c| c == owner)));
    if valid { Some(idx) } else { None }
}

/// Bidirectional name similarity: a contains b or b contains a.
/// Requires minimum 3 chars to avoid false matches ("w" ⊂ "widget").
fn name_matches(name: &str, name_stripped: &str, candidate: &str) -> bool {
    name_stripped.len() >= 3
        && (candidate.contains(name) || name.contains(candidate)
            || candidate.contains(name_stripped) || name_stripped.contains(candidate))
}

/// Strip underscores: "token_tree" → "tokentree" for fuzzy type matching.
fn strip_underscores(s: &str) -> String {
    s.chars().filter(|&c| c != '_').collect()
}

/// RTA filter: narrow owners to instantiated types if available.
fn rta_filter<'a>(owners: &'a [String], rctx: &ResolveCtx<'_>) -> Vec<&'a String> {
    if rctx.instantiated.is_empty() { return owners.iter().collect(); }
    let filtered: Vec<&String> = owners.iter()
        .filter(|o| rctx.instantiated.contains(o.as_str())).collect();
    if filtered.is_empty() { owners.iter().collect() } else { filtered }
}

/// Find matching owner by name similarity in method_owners + trait_impl_methods.
fn find_owner_by_name(rctx: &ResolveCtx<'_>, method: &str, name: &str, stripped: &str) -> Option<u32> {
    if let Some(owners) = rctx.method_owners.get(method) {
        if let Some(owner) = owners.iter().find(|o| name_matches(name, stripped, o)) {
            if let Some(idx) = try_qualified(rctx.exact, owner, method) { return Some(idx); }
        }
    }
    if let Some(impl_types) = rctx.trait_impl_methods.get(method) {
        if let Some(t) = impl_types.iter().find(|t| name_matches(name, stripped, t)) {
            if let Some(idx) = try_qualified(rctx.exact, t, method) { return Some(idx); }
        }
    }
    None
}
