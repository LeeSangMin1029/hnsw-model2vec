//! Index table builders for call graph construction.
//!
//! Pure collection functions: iterate chunks once, produce lookup tables.
//! All functions are `pub(crate)` — used only by `graph::build_with_lsp_filtered`.

use std::collections::{BTreeSet, HashMap, HashSet};
use crate::parse::ParsedChunk;

// ── Owner / field types ─────────────────────────────────────────────

/// Collect type_refs from struct/impl chunks keyed by owning type (lowercase leaf).
pub(crate) fn collect_owner_types(chunks: &[ParsedChunk]) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    for c in chunks.iter().filter(|c| matches!(c.kind.as_str(), "struct" | "impl")) {
        let key = owner_leaf(&c.name);
        let entry = result.entry(key).or_default();
        for ty in &c.types {
            if !entry.contains(ty) {
                entry.push(ty.clone());
            }
        }
    }
    result
}

/// Collect struct field name → type mappings keyed by owning type (lowercase).
pub(crate) fn collect_owner_field_types(chunks: &[ParsedChunk]) -> HashMap<String, HashMap<String, String>> {
    let mut result: HashMap<String, HashMap<String, String>> = HashMap::new();
    for c in chunks.iter().filter(|c| c.kind == "struct") {
        let key = owner_leaf(&c.name);
        let entry = result.entry(key).or_default();
        for (fname, ftype) in &c.field_types {
            entry.insert(fname.to_lowercase(), ftype.to_lowercase());
        }
    }
    result
}

// ── Return type map ─────────────────────────────────────────────────

/// Build function name → return type map (lowercase → lowercase leaf type).
/// Resolves `Self` to the owning type. Inserts bare method names only if unique.
pub(crate) fn build_return_type_map(chunks: &[ParsedChunk]) -> HashMap<String, String> {
    let project_types: HashSet<String> = chunks.iter()
        .filter(|c| matches!(c.kind.as_str(), "struct" | "enum" | "trait" | "class" | "interface"))
        .map(|c| c.name.rsplit("::").next().unwrap_or(&c.name).to_lowercase())
        .collect();

    let mut map = HashMap::new();
    for c in chunks.iter().filter(|c| c.kind == "function") {
        let Some(ref ret) = c.return_type else { continue };
        let ret_lower = ret.to_lowercase();
        let leaf = extract_leaf_type(&ret_lower);
        let resolved = if leaf == "self" || leaf == "&self" || leaf == "&mut self" {
            owning_type(&c.name).unwrap_or_else(|| leaf.to_owned())
        } else if project_types.contains(leaf) {
            leaf.to_owned()
        } else {
            extract_project_type_from_return(&ret_lower, &project_types)
                .unwrap_or_else(|| leaf.to_owned())
        };
        map.insert(c.name.to_lowercase(), resolved);
    }

    // Insert bare method names only if unique across the project.
    let mut method_count: HashMap<String, u32> = HashMap::new();
    for c in chunks.iter().filter(|c| c.kind == "function") {
        if let Some(method) = c.name.rsplit_once("::").map(|p| p.1) {
            *method_count.entry(method.to_lowercase()).or_default() += 1;
        }
    }
    for c in chunks.iter().filter(|c| c.kind == "function" && c.return_type.is_some()) {
        let name_lower = c.name.to_lowercase();
        if let Some(method) = name_lower.rsplit_once("::").map(|p| p.1) {
            if method_count.get(method).copied().unwrap_or(0) == 1 {
                if let Some(resolved) = map.get(&name_lower).cloned() {
                    map.entry(method.to_owned()).or_insert(resolved);
                }
            }
        }
    }
    map
}

// ── Method / trait indexes ──────────────────────────────────────────

/// Reverse index: method_name → [owner_type1, owner_type2, ...].
/// Used for resolving `receiver.method()` when receiver type is unknown.
pub(crate) fn build_method_owner_index(chunks: &[ParsedChunk]) -> HashMap<String, Vec<String>> {
    let mut index: HashMap<String, Vec<String>> = HashMap::new();
    for c in chunks.iter().filter(|c| c.kind == "function") {
        let Some((prefix, method)) = c.name.rsplit_once("::") else { continue };
        // Skip enum variant constructors (uppercase first char).
        if method.starts_with(|c: char| c.is_uppercase()) { continue; }
        let owner = prefix.rsplit("::").next().unwrap_or(prefix);
        let owner_clean = owner.split('<').next().unwrap_or(owner).to_lowercase();
        let method_lower = method.to_lowercase();
        let entry = index.entry(method_lower).or_default();
        if !entry.contains(&owner_clean) {
            entry.push(owner_clean);
        }
    }
    index
}

/// Build trait method → concrete impl type map from `impl Trait for Type` chunks.
pub(crate) fn build_trait_impl_method_map(chunks: &[ParsedChunk]) -> HashMap<String, Vec<String>> {
    // Step 1: trait_name → [concrete_types]
    let mut trait_to_types: HashMap<String, Vec<String>> = HashMap::new();
    for c in chunks.iter().filter(|c| c.kind == "impl") {
        let lower = c.name.to_lowercase();
        let Some(pos) = lower.find(" for ") else { continue };
        let trait_clean = lower[..pos].trim().rsplit("::").next().unwrap_or("")
            .split('<').next().unwrap_or("");
        let concrete_clean = lower[pos + 5..].trim().rsplit("::").next().unwrap_or("")
            .split('<').next().unwrap_or("");
        if !trait_clean.is_empty() && !concrete_clean.is_empty() {
            trait_to_types.entry(trait_clean.to_owned()).or_default()
                .push(concrete_clean.to_owned());
        }
    }

    // Step 2: method → concrete types (via trait→type mapping)
    let mut method_map: HashMap<String, Vec<String>> = HashMap::new();
    for c in chunks.iter().filter(|c| c.kind == "function") {
        let lower = c.name.to_lowercase();
        let Some((prefix, method)) = lower.rsplit_once("::") else { continue };
        let leaf = prefix.rsplit("::").next().unwrap_or(prefix).split('<').next().unwrap_or("");
        let trait_key = if let Some(pos) = leaf.find(" for ") { &leaf[..pos] } else { leaf };
        if let Some(concrete_types) = trait_to_types.get(trait_key) {
            let entry = method_map.entry(method.to_owned()).or_default();
            for ct in concrete_types {
                if !entry.contains(ct) { entry.push(ct.clone()); }
            }
        }
    }
    method_map
}

/// Collect instantiated types from constructor calls (RTA — Rapid Type Analysis).
pub(crate) fn collect_instantiated_types(chunks: &[ParsedChunk]) -> HashSet<String> {
    const CONSTRUCTORS: &[&str] = &[
        "new", "default", "from", "builder", "create", "open", "connect",
        "init", "build", "with_capacity", "with_config", "with_options",
    ];
    let mut instantiated = HashSet::new();
    for c in chunks.iter().filter(|c| c.kind == "function") {
        for call in &c.calls {
            let Some((prefix, method)) = call.rsplit_once("::") else { continue };
            let method_lower = method.to_lowercase();
            if CONSTRUCTORS.contains(&method_lower.as_str())
                || method_lower.starts_with("with_")
                || method_lower.starts_with("from_")
            {
                let leaf = prefix.rsplit("::").next().unwrap_or(prefix)
                    .split('<').next().unwrap_or("").to_lowercase();
                if !leaf.is_empty() { instantiated.insert(leaf); }
            }
        }
    }
    instantiated
}

/// Collect trait method names (methods defined on trait chunks).
pub(crate) fn collect_trait_methods(chunks: &[ParsedChunk]) -> BTreeSet<String> {
    let trait_names: BTreeSet<String> = chunks.iter()
        .filter(|c| c.kind == "trait")
        .map(|c| c.name.to_lowercase().rsplit("::").next().unwrap_or("").to_owned())
        .collect();

    let mut methods = BTreeSet::new();
    for c in chunks.iter().filter(|c| c.kind == "function") {
        let lower = c.name.to_lowercase();
        if let Some((prefix, method)) = lower.rsplit_once("::") {
            let leaf = prefix.rsplit("::").next().unwrap_or(prefix)
                .split('<').next().unwrap_or("");
            if trait_names.contains(leaf) {
                methods.insert(method.to_owned());
            }
        }
    }
    methods
}

// ── String / field access indexes ───────────────────────────────────

/// Build sorted index: lowercase string value → [(chunk_idx, line)].
pub(crate) fn build_string_index(chunks: &[ParsedChunk]) -> Vec<(String, Vec<(u32, u32)>)> {
    let mut map: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
    for (idx, c) in chunks.iter().enumerate() {
        for (_, value, line, _) in &c.string_args {
            map.entry(value.to_lowercase()).or_default().push((idx as u32, *line));
        }
    }
    map.into_iter().collect()
}

/// Build field access index: `type::field` → chunk indices.
/// Resolves receiver variables to types via param/local/field type info.
pub(crate) fn build_field_access_index(
    chunks: &[ParsedChunk],
    owner_field_types: &HashMap<String, HashMap<String, String>>,
) -> Vec<(String, Vec<u32>)> {
    let mut map: HashMap<String, Vec<u32>> = HashMap::new();
    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.field_accesses.is_empty() { continue; }
        let recv_types = build_receiver_type_map_with_fields(chunk, owner_field_types);
        for (recv, field) in &chunk.field_accesses {
            if let Some(ty) = recv_types.get(&recv.to_lowercase()) {
                map.entry(format!("{ty}::{}", field.to_lowercase()))
                    .or_default().push(idx as u32);
            }
        }
    }
    for list in map.values_mut() { list.sort_unstable(); list.dedup(); }
    let mut result: Vec<_> = map.into_iter().collect();
    result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    result
}

/// Build trait → impl mapping from impl chunk names.
pub(crate) fn build_trait_impls(
    names: &[String],
    kinds: &[String],
    exact: &HashMap<String, u32>,
    short: &HashMap<String, u32>,
) -> Vec<Vec<u32>> {
    let mut trait_impls: Vec<Vec<u32>> = vec![Vec::new(); names.len()];
    for (i, (name, kind)) in names.iter().zip(kinds.iter()).enumerate() {
        if kind != "impl" { continue; }
        let lower = name.to_lowercase();
        if let Some(pos) = lower.find(" for ") {
            let trait_idx = exact.get(&lower[..pos]).copied()
                .or_else(|| short.get(&lower[..pos]).copied());
            if let Some(tidx) = trait_idx {
                trait_impls[tidx as usize].push(i as u32);
            }
        }
    }
    for list in &mut trait_impls { list.sort_unstable(); list.dedup(); }
    trait_impls
}

// ── Receiver type map ───────────────────────────────────────────────

/// Build receiver→type map from a chunk's param/local/field types.
pub(crate) fn build_receiver_type_map(chunk: &ParsedChunk) -> HashMap<String, String> {
    let generic_bounds = chunk.signature.as_deref()
        .map(parse_generic_trait_bounds)
        .unwrap_or_default();
    let mut map = HashMap::new();
    for (name, ty) in &chunk.param_types {
        let name_lower = name.to_lowercase();
        if name_lower == "self" { continue; }
        let leaf = extract_leaf_type(&ty.to_lowercase()).to_owned();
        if !leaf.is_empty() {
            let resolved = if leaf.len() == 1 && leaf.as_bytes()[0].is_ascii_lowercase() {
                generic_bounds.get(&leaf).cloned().unwrap_or(leaf)
            } else { leaf };
            map.insert(name_lower, resolved);
        }
    }
    for (name, ty) in &chunk.local_types {
        let leaf = extract_leaf_type(&ty.to_lowercase()).to_owned();
        if !leaf.is_empty() { map.insert(name.to_lowercase(), leaf); }
    }
    for (name, ty) in &chunk.field_types {
        let leaf = extract_leaf_type(&ty.to_lowercase()).to_owned();
        if !leaf.is_empty() { map.insert(name.to_lowercase(), leaf); }
    }
    map
}

/// Like `build_receiver_type_map` but also includes self.field entries
/// from owner_field_types. Used by `build_field_access_index`.
fn build_receiver_type_map_with_fields(
    chunk: &ParsedChunk,
    owner_field_types: &HashMap<String, HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut map = build_receiver_type_map(chunk);
    if let Some(owner) = owning_type(&chunk.name) {
        map.entry("self".to_owned()).or_insert_with(|| owner.clone());
        if let Some(fields) = owner_field_types.get(&owner) {
            for (fname, fty) in fields {
                let leaf = extract_leaf_type(fty).to_owned();
                map.entry(format!("self.{fname}")).or_insert(leaf);
            }
        }
    }
    map
}

// ── Type utilities (shared across modules) ──────────────────────────

/// Extract leaf type from possibly generic/reference type.
/// `"result<vec<item>>"` → `"item"`, `"&mut foo"` → `"foo"`
pub fn extract_leaf_type(ty: &str) -> &str {
    let ty = ty.strip_prefix('&').unwrap_or(ty);
    let ty = if ty.starts_with('\'') { ty.find(' ').map_or(ty, |i| &ty[i + 1..]) } else { ty };
    let ty = ty.strip_prefix("mut ").unwrap_or(ty);
    let ty = ty.strip_prefix("dyn ").unwrap_or(ty);
    let ty = ty.strip_prefix("impl ").unwrap_or(ty);
    let outer = ty.split('<').next().unwrap_or(ty).trim();
    // Unwrap common wrappers: Result<Foo> → Foo
    if matches!(outer, "result" | "option" | "box" | "arc" | "rc" | "vec"
        | "Result" | "Option" | "Box" | "Arc" | "Rc" | "Vec")
    {
        if let Some(start) = ty.find('<') {
            let raw = ty[start + 1..].trim_end_matches('>');
            let first = raw.split(',').next().unwrap_or("").trim();
            let first = first.strip_prefix('&').unwrap_or(first);
            let first = first.strip_prefix("mut ").unwrap_or(first);
            let inner_leaf = first.split('<').next().unwrap_or(first).trim();
            if !inner_leaf.is_empty() && inner_leaf != outer { return inner_leaf; }
        }
    }
    outer
}

/// Extract owning type: `"Foo::bar"` → `"foo"`, handles `impl Trait for Type`.
pub fn owning_type(name: &str) -> Option<String> {
    let (prefix, _) = name.rsplit_once("::")?;
    let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1);
    let leaf = leaf.rsplit_once(" for ").map_or(leaf, |(_, c)| c);
    Some(leaf.split('<').next().unwrap_or(leaf).to_lowercase())
}

/// Strip generic params from each `::` segment: `"foo<t>::bar"` → `"foo::bar"`.
pub(crate) fn strip_generics_from_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut depth = 0u32;
    for ch in key.chars() {
        match ch {
            '<' => depth += 1,
            '>' => { depth = depth.saturating_sub(1); }
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

/// Extract type names from `::` call prefixes.
/// `["DeltaNeighbors::from_ids"]` → `["deltaneighbors"]`
pub(crate) fn extract_call_types(calls: &[String]) -> Vec<String> {
    let mut types = Vec::new();
    for call in calls {
        if let Some((prefix, _)) = call.split_once("::") {
            let leaf = prefix.rsplit_once("::").map_or(prefix, |p| p.1).to_lowercase();
            if !types.contains(&leaf) { types.push(leaf); }
        }
    }
    types
}

/// Parse generic trait bounds from signature: `<T: Search>` → `{"t": "search"}`.
/// Handles both inline bounds and where clauses.
pub(crate) fn parse_generic_trait_bounds(signature: &str) -> HashMap<String, String> {
    let mut bounds = HashMap::new();
    let Some(start) = signature.find('<') else { return bounds };
    let mut depth = 0u32;
    let mut end = start;
    for (i, ch) in signature[start..].char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => { depth -= 1; if depth == 0 { end = start + i; break; } }
            _ => {}
        }
    }
    if end <= start + 1 { return bounds; }

    // Split params by comma (respecting nested <>)
    for param in split_at_commas(&signature[start + 1..end]) {
        let param = param.trim();
        if param.starts_with('\'') || param.starts_with("const ") { continue; }
        if let Some((name, bound_str)) = param.split_once(':') {
            let first = bound_str.split('+').next().unwrap_or("").trim()
                .split('<').next().unwrap_or("").trim();
            if !first.is_empty() && first != "?" {
                bounds.insert(name.trim().to_lowercase(), first.to_lowercase());
            }
        }
    }

    // Also check where clause
    if let Some(where_pos) = signature.to_lowercase().find("where ") {
        for clause in signature[where_pos + 6..].split(',') {
            let clause = clause.trim().trim_end_matches('{').trim();
            if let Some((tp, bound_str)) = clause.split_once(':') {
                let tp = tp.trim().to_lowercase();
                let first = bound_str.split('+').next().unwrap_or("").trim()
                    .split('<').next().unwrap_or("").trim();
                if !first.is_empty() && !bounds.contains_key(&tp) {
                    bounds.insert(tp, first.to_lowercase());
                }
            }
        }
    }
    bounds
}

/// Extract generic bounds as Vec (convenience wrapper).
pub fn extract_generic_bounds(sig: &str) -> Vec<(String, String)> {
    parse_generic_trait_bounds(sig).into_iter().collect()
}

/// Try to find a project type inside a nested return type.
/// `Result<Option<StorageEngine>>` → `"storageengine"`.
fn extract_project_type_from_return(ret: &str, project_types: &HashSet<String>) -> Option<String> {
    for token in ret.split(['<', '>', ',', '(', ')', '&', ' ']) {
        let leaf = token.trim().rsplit("::").next().unwrap_or(token.trim());
        if !leaf.is_empty() && project_types.contains(leaf) {
            return Some(leaf.to_owned());
        }
    }
    None
}

// ── Test helpers ────────────────────────────────────────────────────

pub fn is_test_path(path: &str) -> bool {
    path.contains("/tests/") || path.contains("\\tests\\")
        || path.contains("/test/") || path.contains("\\test\\")
        || path.ends_with("_test.rs") || path.ends_with("_test.go")
        || path.contains("/test_")
}

pub fn is_test_chunk(c: &ParsedChunk) -> bool {
    is_test_path(&c.file) || c.name.starts_with("test_")
}

// ── Import map ──────────────────────────────────────────────────────

/// Parse `use` declarations into short_name → qualified_path map.
pub(crate) fn build_import_map(imports: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for imp in imports {
        let s = imp.trim().trim_start_matches("use ").trim_end_matches(';').trim()
            .trim_start_matches("crate::");
        if let Some(brace) = s.find('{') {
            let prefix = s[..brace].trim_end_matches("::");
            if let Some(end) = s.rfind('}') {
                for part in s[brace + 1..end].split(',') {
                    let name = part.split_whitespace().next().unwrap_or("");
                    if !name.is_empty() && name != "self" {
                        map.insert(name.to_lowercase(), format!("{prefix}::{name}").to_lowercase());
                    }
                }
            }
        } else if let Some(last) = s.rsplit("::").next() {
            let short = last.split_whitespace().next().unwrap_or("");
            if !short.is_empty() {
                map.insert(short.to_lowercase(), s.to_lowercase());
            }
        }
    }
    map
}

// ── Internal helpers ────────────────────────────────────────────────

/// Extract lowercase leaf from a qualified name, handling `impl Trait for Type`.
fn owner_leaf(name: &str) -> String {
    let lower = name.to_lowercase();
    let leaf = lower.rsplit("::").next().unwrap_or(&lower);
    let leaf = leaf.rsplit_once(" for ").map_or(leaf, |(_, c)| c);
    leaf.split('<').next().unwrap_or(leaf).to_owned()
}

/// Split a string at top-level commas (not inside `<>`).
fn split_at_commas(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0u32;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => { result.push(&s[start..i]); start = i + 1; }
            _ => {}
        }
    }
    result.push(&s[start..]);
    result
}
