//! Lightweight type inference via compiler-assisted probes.
//!
//! Generates minimal Rust probe code from parsed chunks, runs `cargo check`
//! with a cfg gate, and parses "mismatched types" errors to extract receiver
//! type information that tree-sitter alone cannot determine.
//!
//! Probe code is appended to each crate's `lib.rs` (or `main.rs`) under
//! `#[cfg(vcode_type_probe)]` so normal builds are unaffected.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::parse::ParsedChunk;

/// Cache version — bump when probe format changes.
const TYPE_MAP_CACHE_VERSION: u8 = 1;

/// Result of type probe inference.
#[derive(Debug, Clone, Default, bincode::Encode, bincode::Decode)]
pub struct TypeMap {
    version: u8,
    fingerprint: String,
    /// `"function_name::var_name"` → `"inferred_type"` (lowercase)
    pub types: HashMap<String, String>,
}

/// Try to build a type map via cargo check probes.
///
/// Returns `None` if probing is not possible (no Cargo.toml, etc.).
/// Results are cached in `db/cache/type_map.bin`.
pub fn try_build_type_map(
    db: &Path,
    chunks: &[ParsedChunk],
) -> Option<TypeMap> {
    let project_root = crate::helpers::find_project_root(db)?;
    let fingerprint = compute_fingerprint(&project_root);

    // Try cache first.
    if let Some(cached) = load_cache(db, &fingerprint) {
        eprintln!("    [type-probe] cache hit: {} types", cached.types.len());
        return Some(cached);
    }

    // Find all crate entry points in the project (workspace-aware).
    let entry_points = find_entry_points(&project_root);
    if entry_points.is_empty() {
        eprintln!("    [type-probe] no entry points found");
        return None;
    }

    // Generate probes from chunks, grouped by crate.
    let probes = generate_probes(chunks);
    if probes.is_empty() {
        eprintln!("    [type-probe] no probes to generate");
        let map = TypeMap { version: TYPE_MAP_CACHE_VERSION, fingerprint, types: HashMap::new() };
        save_cache(db, &map);
        return Some(map);
    }

    // Group probes by which crate entry point they belong to.
    let grouped = group_probes_by_crate(&probes, chunks, &entry_points, &project_root);
    let total_probes: usize = grouped.values().map(|v| v.len()).sum();
    eprintln!("    [type-probe] {} probes across {} crates", total_probes, grouped.len());

    if total_probes == 0 {
        let map = TypeMap { version: TYPE_MAP_CACHE_VERSION, fingerprint, types: HashMap::new() };
        save_cache(db, &map);
        return Some(map);
    }

    // Inject probes into each entry point.
    let mut originals: Vec<(PathBuf, String)> = Vec::new();
    for (entry, probe_indices) in &grouped {
        let Ok(original) = std::fs::read_to_string(entry) else { continue };
        originals.push((entry.clone(), original.clone()));

        let crate_probes: Vec<&Probe> = probe_indices.iter().map(|&i| &probes[i]).collect();
        let probe_code = build_probe_module(&crate_probes);
        let modified = format!("{original}\n\n{probe_code}");
        let _ = std::fs::write(entry, &modified);
    }

    // Run cargo check once for all crates.
    let types = run_cargo_check(&project_root);

    // Restore all entry points immediately.
    for (path, content) in &originals {
        let _ = std::fs::write(path, content);
    }

    eprintln!("    [type-probe] inferred {} types", types.len());

    let map = TypeMap { version: TYPE_MAP_CACHE_VERSION, fingerprint, types };
    save_cache(db, &map);
    Some(map)
}

/// A single type probe: exercise a let-binding chain to discover receiver type.
struct Probe {
    /// Unique ID for the probe function name.
    id: usize,
    /// Index into chunks array (for crate grouping).
    chunk_idx: usize,
    /// Parameter declarations: `(param_name, Type)`
    params: Vec<(String, String)>,
    /// Statements to probe.
    stmts: Vec<ProbeStmt>,
}

struct ProbeStmt {
    /// Variable being assigned.
    var: String,
    /// Expression (e.g., `param.method()` or `Type::new()`).
    expr: String,
    /// Fully qualified key for the type map: `"function::var"`.
    key: String,
}

/// Generate probes from chunks that have unresolved let-call-binding chains.
fn generate_probes(chunks: &[ParsedChunk]) -> Vec<Probe> {
    let project_types: HashSet<String> = chunks
        .iter()
        .filter(|c| matches!(c.kind.as_str(), "struct" | "enum" | "trait"))
        .map(|c| c.name.rsplit("::").next().unwrap_or(&c.name).to_lowercase())
        .collect();

    let return_types: HashMap<String, String> = chunks
        .iter()
        .filter(|c| c.kind == "function" && c.return_type.is_some())
        .map(|c| (c.name.to_lowercase(), c.return_type.as_deref().unwrap_or("").to_owned()))
        .collect();

    let mut probes = Vec::new();
    let mut probe_id = 0usize;

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        if chunk.kind != "function" || chunk.let_call_bindings.is_empty() {
            continue;
        }

        let known_params: Vec<(String, String)> = chunk.param_types.iter()
            .filter(|(name, _)| !name.eq_ignore_ascii_case("self"))
            .filter(|(_, ty)| {
                let ty_lower = ty.to_lowercase();
                let leaf = crate::graph::extract_leaf_type(&ty_lower);
                project_types.contains(leaf)
            })
            .map(|(name, ty)| (name.to_lowercase(), ty.clone()))
            .collect();

        let self_type = chunk.name.rsplit_once("::").and_then(|(prefix, _)| {
            let leaf = prefix.rsplit("::").next().unwrap_or(prefix);
            let leaf_lower = leaf.split('<').next().unwrap_or(leaf).to_lowercase();
            project_types.contains(&leaf_lower).then(|| leaf.to_owned())
        });

        let mut known_types: HashMap<String, String> = HashMap::new();
        for (name, ty) in &known_params {
            known_types.insert(name.clone(), ty.clone());
        }
        if let Some(ref st) = self_type {
            known_types.insert("self".to_owned(), st.clone());
        }

        // Track variables declared in earlier stmts (for chaining).
        let mut declared_vars: HashSet<String> = HashSet::new();
        let mut stmts = Vec::new();
        for (var, callee) in &chunk.let_call_bindings {
            let var_lower = var.to_lowercase();
            if known_types.contains_key(&var_lower) {
                continue;
            }
            let callee_lower = callee.to_lowercase();
            if return_types.contains_key(&callee_lower) {
                continue;
            }

            if let Some(expr) = build_probe_expr(callee, &known_types, &declared_vars) {
                let key = format!("{}::{}", chunk.name.to_lowercase(), var_lower);
                stmts.push(ProbeStmt { var: var_lower.clone(), expr, key });
                declared_vars.insert(var_lower);
            }
        }

        if stmts.is_empty() {
            continue;
        }

        let mut params: Vec<(String, String)> = known_params;
        if let Some(ref st) = self_type {
            params.insert(0, ("_self".to_owned(), st.clone()));
        }

        probes.push(Probe { id: probe_id, chunk_idx, params, stmts });
        probe_id += 1;
    }

    probes
}

/// Build a Rust expression for a callee call.
///
/// `declared_vars` contains variables from prior probe stmts in the same
/// function, allowing chained calls: `let a = self.x(); let b = a.y();`
fn build_probe_expr(
    callee: &str,
    known_types: &HashMap<String, String>,
    declared_vars: &HashSet<String>,
) -> Option<String> {
    if let Some(dot) = callee.find('.') {
        let receiver = &callee[..dot];
        let method = &callee[dot + 1..];
        let recv_lower = receiver.to_lowercase();

        if recv_lower == "self" {
            return Some(format!("_self.{method}()"));
        }
        if known_types.contains_key(&recv_lower) || declared_vars.contains(&recv_lower) {
            return Some(format!("{receiver}.{method}()"));
        }
        return None;
    }

    if callee.contains("::") {
        return Some(format!("{callee}()"));
    }

    None
}

/// Group probes by their crate entry point.
fn group_probes_by_crate(
    probes: &[Probe],
    chunks: &[ParsedChunk],
    entry_points: &[PathBuf],
    project_root: &Path,
) -> HashMap<PathBuf, Vec<usize>> {
    let mut grouped: HashMap<PathBuf, Vec<usize>> = HashMap::new();

    // Build a map: crate_prefix → entry_point
    // e.g., "crates/v-code-intel/src/" → "crates/v-code-intel/src/lib.rs"
    let crate_prefixes: Vec<(String, PathBuf)> = entry_points.iter().filter_map(|ep| {
        let parent = ep.parent()?;
        let rel = parent.strip_prefix(project_root).ok()?;
        let prefix = rel.to_string_lossy().replace('\\', "/");
        Some((prefix, ep.clone()))
    }).collect();

    for (probe_idx, probe) in probes.iter().enumerate() {
        let chunk_file = &chunks[probe.chunk_idx].file;
        let file_normalized = chunk_file.replace('\\', "/");

        // Find which crate this chunk belongs to.
        let mut best_match: Option<&PathBuf> = None;
        let mut best_len = 0;
        for (prefix, ep) in &crate_prefixes {
            if file_normalized.starts_with(prefix) && prefix.len() > best_len {
                best_match = Some(ep);
                best_len = prefix.len();
            }
        }

        if let Some(ep) = best_match {
            grouped.entry(ep.clone()).or_default().push(probe_idx);
        } else if let Some(ep) = entry_points.first() {
            // Fallback: use first entry point.
            grouped.entry(ep.clone()).or_default().push(probe_idx);
        }
    }

    grouped
}

/// Build the `#[cfg(vcode_type_probe)]` module code.
fn build_probe_module(probes: &[&Probe]) -> String {
    let mut code = String::new();
    code.push_str("#[cfg(vcode_type_probe)]\n");
    code.push_str("#[allow(unused, dead_code, unreachable_code, clippy::all)]\n");
    code.push_str("mod _vcode_type_probes {\n");
    code.push_str("    use super::*;\n\n");

    for probe in probes {
        code.push_str(&format!("    fn _vcode_probe_{}(\n", probe.id));
        for (name, ty) in &probe.params {
            code.push_str(&format!("        {name}: {ty},\n"));
        }
        code.push_str("    ) {\n");

        for (i, stmt) in probe.stmts.iter().enumerate() {
            code.push_str(&format!("        let {} = {};\n", stmt.var, stmt.expr));
            code.push_str(&format!(
                "        let _vcode_t{i}: () = {}; // VCODE_KEY:{}\n",
                stmt.var, stmt.key
            ));
        }

        code.push_str("    }\n\n");
    }

    code.push_str("}\n");
    code
}

/// Run `cargo check` with cfg flag and parse type errors.
///
/// Uses `CARGO_ENCODED_RUSTFLAGS` to pass cfg only to workspace crates,
/// avoiding full rebuild of external dependencies.
fn run_cargo_check(project_root: &Path) -> HashMap<String, String> {
    // Use cargo rustc per-package to avoid rebuilding all deps.
    // CARGO_ENCODED_RUSTFLAGS only applies to the current package.
    let result = std::process::Command::new("cargo")
        .arg("check")
        .arg("--workspace")
        .arg("--message-format=json")
        .arg("--quiet")
        .env("CARGO_ENCODED_RUSTFLAGS", "--cfg\x1fvcode_type_probe")
        .current_dir(project_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    match result {
        Ok(output) => parse_type_errors(&output.stdout),
        Err(_) => HashMap::new(),
    }
}

/// Parse `cargo check --message-format=json` output for mismatched type errors.
fn parse_type_errors(stdout: &[u8]) -> HashMap<String, String> {
    let mut types = HashMap::new();
    let stdout_str = String::from_utf8_lossy(stdout);

    for line in stdout_str.lines() {
        if !line.contains("mismatched types") || !line.contains("VCODE_KEY") {
            continue;
        }

        let Ok(val) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let Some(message) = val.get("message") else { continue };

        let msg_text = message.get("message").and_then(|m| m.as_str()).unwrap_or("");
        if !msg_text.contains("mismatched types") {
            continue;
        }

        let key = extract_vcode_key(message);
        if key.is_empty() {
            continue;
        }

        if let Some(found_type) = extract_found_type(message) {
            let leaf = crate::graph::extract_leaf_type(&found_type.to_lowercase()).to_owned();
            if !leaf.is_empty() && leaf != "()" {
                types.insert(key, leaf);
            }
        }
    }

    types
}

/// Extract VCODE_KEY from span text in a diagnostic.
fn extract_vcode_key(message: &serde_json::Value) -> String {
    if let Some(spans) = message.get("spans").and_then(|s| s.as_array()) {
        for span in spans {
            if let Some(text_arr) = span.get("text").and_then(|t| t.as_array()) {
                for text_obj in text_arr {
                    if let Some(text) = text_obj.get("text").and_then(|t| t.as_str()) {
                        if let Some(idx) = text.find("VCODE_KEY:") {
                            return text[idx + 10..].trim().to_owned();
                        }
                    }
                }
            }
        }
    }
    String::new()
}

/// Extract the "found" type from a mismatched types error.
///
/// Checks (in order): children messages, span labels, rendered text.
fn extract_found_type(message: &serde_json::Value) -> Option<String> {
    // 1. Check children (some rustc versions put "found `T`" here).
    if let Some(children) = message.get("children").and_then(|c| c.as_array()) {
        for child in children {
            if let Some(msg) = child.get("message").and_then(|m| m.as_str()) {
                if let Some(ty) = parse_found_backtick(msg) {
                    return Some(ty);
                }
            }
        }
    }

    // 2. Check span labels (common format: `expected `()`, found `VhnswError``).
    if let Some(spans) = message.get("spans").and_then(|s| s.as_array()) {
        for span in spans {
            if let Some(label) = span.get("label").and_then(|l| l.as_str()) {
                if let Some(ty) = parse_found_backtick(label) {
                    return Some(ty);
                }
            }
        }
    }

    // 3. Check rendered text as fallback.
    if let Some(rendered) = message.get("rendered").and_then(|r| r.as_str()) {
        if let Some(ty) = parse_found_backtick(rendered) {
            return Some(ty);
        }
    }

    // 4. Check top-level message.
    let msg = message.get("message").and_then(|m| m.as_str())?;
    parse_found_backtick(msg)
}

/// Parse "found `Type`" from a string, handling common prefixes.
fn parse_found_backtick(text: &str) -> Option<String> {
    for prefix in ["found `", "found struct `", "found enum `"] {
        if let Some(idx) = text.find(prefix) {
            let rest = &text[idx + prefix.len()..];
            if let Some(end) = rest.find('`') {
                let ty = rest[..end].to_owned();
                if !ty.is_empty() {
                    return Some(ty);
                }
            }
        }
    }
    None
}

/// Find all crate entry points (lib.rs / main.rs) in the project.
///
/// Supports both single-crate and workspace layouts.
fn find_entry_points(project_root: &Path) -> Vec<PathBuf> {
    let mut entry_points = Vec::new();
    let cargo_toml = project_root.join("Cargo.toml");
    let Ok(content) = std::fs::read_to_string(&cargo_toml) else {
        return entry_points;
    };

    // Single crate: check src/lib.rs or src/main.rs directly.
    let src = project_root.join("src");
    for name in ["lib.rs", "main.rs"] {
        let path = src.join(name);
        if path.exists() {
            entry_points.push(path);
        }
    }

    // Workspace: parse member paths.
    if content.contains("[workspace]") {
        // Extract member globs from Cargo.toml.
        let members = parse_workspace_members(&content);
        for member in members {
            // Handle glob patterns like "crates/*".
            if member.contains('*') {
                let prefix = member.split('*').next().unwrap_or("");
                let base = project_root.join(prefix);
                if let Ok(entries) = std::fs::read_dir(&base) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let path = entry.path();
                        if path.is_dir() {
                            for name in ["lib.rs", "main.rs"] {
                                let ep = path.join("src").join(name);
                                if ep.exists() {
                                    entry_points.push(ep);
                                }
                            }
                        }
                    }
                }
            } else {
                let member_dir = project_root.join(&member);
                for name in ["lib.rs", "main.rs"] {
                    let ep = member_dir.join("src").join(name);
                    if ep.exists() {
                        entry_points.push(ep);
                    }
                }
            }
        }
    }

    // Deduplicate.
    entry_points.sort();
    entry_points.dedup();
    entry_points
}

/// Parse workspace members from Cargo.toml content.
fn parse_workspace_members(content: &str) -> Vec<String> {
    let mut members = Vec::new();
    let mut in_members = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("members") && trimmed.contains('[') {
            in_members = true;
            // Handle inline: members = ["a", "b"]
            if let Some(bracket_start) = trimmed.find('[') {
                let rest = &trimmed[bracket_start + 1..];
                if let Some(bracket_end) = rest.find(']') {
                    for item in rest[..bracket_end].split(',') {
                        let m = item.trim().trim_matches('"').trim_matches('\'');
                        if !m.is_empty() {
                            members.push(m.to_owned());
                        }
                    }
                    in_members = false;
                }
            }
            continue;
        }
        if in_members {
            if trimmed == "]" {
                in_members = false;
                continue;
            }
            let m = trimmed.trim_end_matches(',').trim().trim_matches('"').trim_matches('\'').trim();
            if !m.is_empty() && !m.starts_with('#') {
                members.push(m.to_owned());
            }
        }
    }
    members
}

/// Compute fingerprint for cache invalidation.
fn compute_fingerprint(project_root: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for name in ["Cargo.toml", "Cargo.lock"] {
        if let Ok(content) = std::fs::read_to_string(project_root.join(name)) {
            content.hash(&mut hasher);
        }
    }
    if let Ok(meta) = std::fs::metadata(project_root.join("src")) {
        if let Ok(mtime) = meta.modified() {
            mtime.hash(&mut hasher);
        }
    }
    format!("{:016x}", hasher.finish())
}

fn cache_path(db: &Path) -> PathBuf {
    db.join("cache").join("type_map.bin")
}

fn load_cache(db: &Path, fingerprint: &str) -> Option<TypeMap> {
    let path = cache_path(db);
    let data = std::fs::read(&path).ok()?;
    let config = bincode::config::standard();
    let (map, _): (TypeMap, _) = bincode::decode_from_slice(&data, config).ok()?;
    (map.version == TYPE_MAP_CACHE_VERSION && map.fingerprint == fingerprint).then_some(map)
}

fn save_cache(db: &Path, map: &TypeMap) {
    let path = cache_path(db);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let config = bincode::config::standard();
    if let Ok(data) = bincode::encode_to_vec(map, config) {
        let _ = std::fs::write(&path, data);
    }
}
