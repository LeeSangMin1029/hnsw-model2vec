//! MIR text parser — extracts fully-resolved call graphs from `--emit=mir` output.
//!
//! `cargo rustc -p <crate> -- --emit=mir` produces text MIR where every method
//! call is fully resolved (e.g. `<Vec<u32> as Default>::default`).  This module
//! parses that output into caller→callee pairs that `CallGraph::build_with_resolved_calls`
//! can consume.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

// ── Types ────────────────────────────────────────────────────────────

/// A single function's resolved calls extracted from MIR text.
#[derive(Debug, Clone)]
pub struct MirFunction {
    /// Normalized function name (crate prefix stripped, `<impl at …>` resolved).
    pub name: String,
    /// Fully-resolved callee names (normalized the same way).
    pub calls: Vec<String>,
}

/// Caller→callee edges keyed by normalized caller name.
pub type MirCallMap = BTreeMap<String, Vec<String>>;

// ── MIR Text Parser ─────────────────────────────────────────────────

/// Parse MIR text into a list of functions with their resolved calls.
///
/// `project_root` is used to resolve `<impl at path:line:col>` blocks by reading
/// the source file to extract the actual type name.
pub fn parse_mir(mir_text: &str, crate_prefix: &str, project_root: &Path) -> Vec<MirFunction> {
    let mut functions = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_calls: Vec<String> = Vec::new();

    for line in mir_text.lines() {
        let trimmed = line.trim();

        // Function definition: `fn <name>(...) -> ... {`
        if trimmed.starts_with("fn ") && trimmed.ends_with('{') {
            // Flush previous function.
            if let Some(name) = current_name.take() {
                current_calls.sort_unstable();
                current_calls.dedup();
                functions.push(MirFunction { name, calls: current_calls });
                current_calls = Vec::new();
            }
            current_name = parse_fn_name(trimmed, crate_prefix, project_root);
        }

        // Call site: `_N = <target>(args) -> [return: bbN, ...]`
        // or:        `_N = module::func(args) -> [return: bbN, ...]`
        if current_name.is_some() {
            if let Some(callee) = extract_call_target(trimmed, crate_prefix) {
                current_calls.push(callee);
            }
        }
    }

    // Flush last function.
    if let Some(name) = current_name {
        current_calls.sort_unstable();
        current_calls.dedup();
        functions.push(MirFunction { name, calls: current_calls });
    }

    functions
}

/// Build a `MirCallMap` from parsed MIR functions.
pub fn build_mir_call_map(functions: &[MirFunction]) -> MirCallMap {
    let mut map = BTreeMap::new();
    for f in functions {
        if !f.calls.is_empty() {
            map.insert(f.name.clone(), f.calls.clone());
        }
    }
    map
}

// ── MIR Collection ──────────────────────────────────────────────────

/// Collect MIR for all workspace crates by running `cargo rustc -p <crate> -- --emit=mir`.
///
/// Returns parsed `MirFunction`s for the entire workspace.
pub fn collect_workspace_mir(project_root: &Path) -> Result<Vec<MirFunction>> {
    let crate_names = workspace_crate_names(project_root)?;
    let mut all_functions = Vec::new();

    for (pkg_name, crate_prefix) in &crate_names {
        // emit_mir_for_crate reads MIR text immediately (before next clean can delete it).
        let mir_texts = emit_mir_for_crate(project_root, pkg_name)?;
        for text in &mir_texts {
            let functions = parse_mir(text, crate_prefix, project_root);
            all_functions.extend(functions);
        }
    }

    Ok(all_functions)
}

/// Get workspace crate names and their MIR prefix (underscored name).
fn workspace_crate_names(project_root: &Path) -> Result<Vec<(String, String)>> {
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--no-deps")
        .arg("--format-version=1")
        .current_dir(project_root)
        .output()
        .context("failed to run cargo metadata")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Simple JSON parsing for package names — avoid pulling in serde_json.
    let mut crates = Vec::new();
    for segment in stdout.split("\"name\":\"") {
        if let Some(end) = segment.find('"') {
            let name = &segment[..end];
            // Skip non-workspace entries (they'd be in deps).
            if !name.is_empty() && !name.contains('/') {
                let prefix = name.replace('-', "_");
                crates.push((name.to_owned(), prefix));
            }
        }
    }
    Ok(crates)
}

/// Collect MIR text for a single crate.
///
/// Strategy: reuse existing `.mir` files if present; only invoke `cargo rustc`
/// (with `cargo clean -p` to force recompilation) when no MIR file is found.
fn emit_mir_for_crate(project_root: &Path, pkg_name: &str) -> Result<Vec<String>> {
    let deps_dir = project_root.join("target").join("debug").join("deps");
    let prefix = pkg_name.replace('-', "_");

    // Check for existing MIR files first.
    let mut mir_files = find_mir_files(&deps_dir, &prefix);

    // If no MIR files exist, compile with --emit=mir.
    if mir_files.is_empty() {
        // Clean to force recompilation (cached builds skip --emit=mir).
        let _ = Command::new("cargo")
            .args(["clean", "-p", pkg_name])
            .current_dir(project_root)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();

        // Try --lib first (most crates are libraries).
        let status = Command::new("cargo")
            .args(["rustc", "-p", pkg_name, "--lib", "--", "--emit=mir"])
            .current_dir(project_root)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .with_context(|| format!("failed to run cargo rustc for {pkg_name}"))?;

        if !status.success() {
            // --lib failed — try without --lib for binary-only crates.
            let bin_status = Command::new("cargo")
                .args(["rustc", "-p", pkg_name, "--", "--emit=mir"])
                .current_dir(project_root)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .with_context(|| format!("failed to run cargo rustc (bin) for {pkg_name}"))?;

            if !bin_status.success() {
                return Ok(Vec::new());
            }
        }

        mir_files = find_mir_files(&deps_dir, &prefix);
    }

    // Read MIR text immediately into memory (before next clean can delete).
    let mut texts = Vec::new();
    for mir_path in &mir_files {
        if let Ok(text) = std::fs::read_to_string(mir_path) {
            texts.push(text);
        }
    }

    Ok(texts)
}

/// Find `.mir` files in deps_dir matching `prefix-HASH.mir` or `prefix.mir`.
fn find_mir_files(deps_dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut mir_files: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(deps_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".mir")
                && name_str.starts_with(prefix)
                && name_str[prefix.len()..].starts_with(|c: char| c == '-' || c == '.')
            {
                mir_files.push(entry.path());
            }
        }
    }

    // Keep only the newest .mir file per crate (there may be stale ones).
    if mir_files.len() > 1 {
        mir_files.sort_by_key(|p| {
            std::fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        mir_files = vec![mir_files.pop().unwrap_or_default()];
    }

    mir_files
}

// ── Internal Parsers ────────────────────────────────────────────────

/// Extract and normalize function name from a MIR `fn` line.
///
/// Input:  `fn l2::<impl at crates/…>::distance(_1: &L2Distance, ...) -> f32 {`
/// Output: `l2::L2Distance::distance`
///
/// The `<impl at …>` block is resolved using the first parameter's type name.
fn parse_fn_name(line: &str, crate_prefix: &str, project_root: &Path) -> Option<String> {
    // Strip "fn " prefix.
    let rest = line.strip_prefix("fn ")?;
    // Find the parameter list start.
    let paren = rest.find('(')?;
    let raw_name = &rest[..paren];
    let params = &rest[paren..];

    // If the name contains `<impl at …>`, resolve using the first param's type.
    let resolved_name = if raw_name.contains("<impl at ") {
        resolve_impl_at_with_param(raw_name, params, project_root)
    } else {
        raw_name.to_owned()
    };

    let normalized = normalize_mir_name(&resolved_name, crate_prefix);

    // Skip closures.
    if normalized.contains("{closure") {
        return None;
    }
    // Skip derive-generated functions.
    if is_derive_fn(&normalized) {
        return None;
    }

    Some(normalized)
}

/// Replace `<impl at …>` with the type name extracted from `_1: &TypeName`.
///
/// `"l2::<impl at path:11:10>::distance"` + `"(_1: &L2Distance, …)"` → `"l2::L2Distance::distance"`
fn resolve_impl_at_with_param(name: &str, params: &str, project_root: &Path) -> String {
    // Extract type from first param: `(_1: &L2Distance, …)` or `(_1: &mut Node, …)`
    let type_name = extract_self_type_from_params(params);

    let mut result = name.to_owned();
    while let Some(start) = result.find("<impl at ") {
        if let Some(end_rel) = result[start..].find(">::") {
            let end = start + end_rel;
            let impl_block = &result[start + 9..end]; // path:line:col
            let replacement = type_name
                .as_deref()
                .or_else(|| extract_type_from_impl_path(impl_block, project_root))
                .unwrap_or("");
            result = format!("{}{}::{}", &result[..start], replacement, &result[end + 3..]);
        } else {
            break;
        }
    }
    result
}

/// Extract the type name from `<impl at path:line:col>` by reading the source file.
///
/// Reads the source file at the given line and extracts the type from `impl TypeName`.
/// Falls back to PascalCase conversion of the file stem.
///
/// `"crates/.../search_context.rs:17:1: 17:19"` (line 17: `impl SearchContext`) → `"SearchContext"`
/// `"crates/.../engine.rs:54:1: 54:19"` (line 54: `impl StorageEngine`) → `"StorageEngine"`
fn extract_type_from_impl_path(impl_block: &str, project_root: &Path) -> Option<&'static str> {
    // Parse "path:line:col" or "path:line:col: line:col"
    let parts: Vec<&str> = impl_block.splitn(3, ':').collect();
    if parts.len() < 2 {
        return None;
    }
    let path = parts[0];
    let line_num: usize = parts[1].trim().parse().ok()?;

    // Try to read the source file and extract `impl TypeName` from that line.
    // Path is relative to project root (where cargo rustc was invoked).
    let full_path = project_root.join(path);
    if let Ok(content) = std::fs::read_to_string(&full_path) {
        if let Some(line) = content.lines().nth(line_num.saturating_sub(1)) {
            let trimmed = line.trim();
            // Match patterns: `impl TypeName`, `impl TypeName {`, `impl<T> TypeName<T>`
            if let Some(after_impl) = trimmed.strip_prefix("impl") {
                let mut rest = after_impl.trim_start();
                // Skip optional generic params: `<T>`, `<'a, T: Trait>`
                if rest.starts_with('<') {
                    if let Some(close) = find_closing_angle(rest, 0) {
                        rest = rest[close + 1..].trim_start();
                    }
                }
                // Take the type name until whitespace, `{`, `<`, or end.
                let type_end = rest.find(|c: char| c.is_whitespace() || c == '{' || c == '<');
                let type_name = match type_end {
                    Some(e) => &rest[..e],
                    None => rest,
                };
                if !type_name.is_empty() && type_name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    return Some(Box::leak(type_name.to_owned().into_boxed_str()));
                }
            }
        }
    }

    // Fallback: convert file stem to PascalCase.
    let filename = path.rsplit(['/', '\\']).next()?;
    let stem = filename.strip_suffix(".rs").or_else(|| filename.strip_suffix(".py"))?;
    let pascal: String = stem
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    upper + chars.as_str()
                }
                None => String::new(),
            }
        })
        .collect();
    if pascal.is_empty() {
        return None;
    }
    Some(Box::leak(pascal.into_boxed_str()))
}

/// Extract the self type from MIR parameter list.
/// `(_1: &L2Distance, _2: &[f32])` → `Some("L2Distance")`
/// `(_1: &mut Node, _2: u8)` → `Some("Node")`
///
/// Returns `None` for common std types (Path, str, etc.) — these indicate
/// static methods where the first param is NOT `self`.
fn extract_self_type_from_params(params: &str) -> Option<String> {
    // Find `_1: ` or first param.
    let colon = params.find(": ")?;
    let after_colon = &params[colon + 2..];
    // Strip `&` and `&mut `.
    let trimmed = after_colon
        .trim_start_matches("&mut ")
        .trim_start_matches('&');
    // Skip `impl Trait` params — these are generic bounds, not self.
    // e.g., `_1: impl AsRef<Path>` is not a method receiver.
    if trimmed.starts_with("impl ") {
        return None;
    }
    // Take until `,` or `)` or `<`.
    let end = trimmed.find(|c: char| c == ',' || c == ')' || c == '<')?;
    let ty = trimmed[..end].trim();
    if ty.is_empty() || ty.starts_with('[') || ty.starts_with('(') {
        return None;
    }
    // Skip common std types — these are likely static method params, not self.
    if is_std_type(ty) {
        return None;
    }
    Some(ty.to_owned())
}

/// Check if a type name is a common std/core/alloc type (not a project type).
fn is_std_type(ty: &str) -> bool {
    matches!(
        ty,
        "Path" | "PathBuf" | "str" | "String" | "OsStr" | "OsString"
            | "bool" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
            | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
            | "f32" | "f64" | "char"
    )
}

/// Extract a call target from a MIR statement line.
///
/// Patterns:
/// - `_N = <Type as Trait>::method(args) -> …`
/// - `_N = module::func(args) -> …`
/// - `_N = func(args) -> …`
fn extract_call_target(line: &str, crate_prefix: &str) -> Option<String> {
    // Must contain `-> [return:` or `-> unwind` (marks a call, not an assignment).
    if !line.contains("-> [return:") && !line.contains("-> unwind") {
        return None;
    }
    // Find the `= ` assignment.
    let eq_pos = line.find("= ")?;
    let after_eq = &line[eq_pos + 2..];

    // Find the call target: everything before the first `(`.
    let paren = after_eq.find('(')?;
    let raw_target = after_eq[..paren].trim();

    // Skip non-call assignments (e.g., `Eq(`, `PtrMetadata(`).
    if raw_target.is_empty() {
        return None;
    }
    // Skip MIR builtins (first char is uppercase, no `::` or `<`).
    let first = raw_target.chars().next()?;
    if first.is_uppercase() && !raw_target.contains("::") && !raw_target.contains('<') {
        return None;
    }

    let normalized = normalize_mir_name(raw_target, crate_prefix);

    // Skip std/core/alloc calls.
    if is_external_call(&normalized) {
        return None;
    }
    // Skip closures.
    if normalized.contains("{closure") || normalized.contains("closure@") {
        return None;
    }
    // Skip derive-generated functions.
    if is_derive_fn(&normalized) {
        return None;
    }

    Some(normalized)
}

/// Normalize a MIR name:
/// 1. Strip crate prefix (`v_hnsw_graph::` → ``)
/// 2. Resolve `<impl at path:line>` → type name from context
/// 3. Strip `<Type as Trait>::` → `Type::`
/// 4. Strip generic parameters
fn normalize_mir_name(raw: &str, crate_prefix: &str) -> String {
    let mut s = raw.to_owned();

    // Strip crate prefix.
    let prefix_with_sep = format!("{crate_prefix}::");
    if let Some(rest) = s.strip_prefix(&prefix_with_sep) {
        s = rest.to_owned();
    }

    // Resolve `<impl at path:line:col>` → extract type from surrounding context.
    // MIR format: `module::<impl at crates/foo/src/bar.rs:14:1: 14:35>::method`
    // We replace `<impl at ...>` with the type name extracted from surrounding module path.
    while let Some(impl_start) = s.find("<impl at ") {
        if let Some(impl_end) = s[impl_start..].find(">::") {
            let before = &s[..impl_start];
            let after = &s[impl_start + impl_end + 3..];
            // The module path before `<impl at …>` often contains the type.
            // E.g., `distance::l2::<impl at …>::distance` → `distance::l2::distance`
            // We need to find the actual type from the impl block.
            // For now, use the module prefix + method name.
            s = format!("{before}{after}");
        } else {
            break;
        }
    }

    // Strip generic parameters FIRST: `func::<T>` → `func`, `Vec::<u64>::push` → `Vec::push`.
    // Must come before `<Type as Trait>` handling to avoid `::<D>` being misinterpreted.
    loop {
        if let Some(start) = s.find("::<") {
            if let Some(end) = find_closing_angle(&s, start + 2) {
                s = format!("{}{}", &s[..start], &s[end + 1..]);
                continue;
            }
        }
        break;
    }

    // Strip `<Type as Trait>::` → `Type::`.
    // E.g., `<Vec<u32> as Default>::default` → `Vec::default`
    while let Some(start) = s.find('<') {
        if let Some(as_pos) = find_balanced_as(&s, start) {
            if let Some(gt_pos) = find_closing_angle(&s, start) {
                let type_part = &s[start + 1..as_pos].trim();
                let type_name = extract_base_type(type_part);
                let after = &s[gt_pos + 1..];
                let before = &s[..start];
                s = format!("{before}{type_name}{after}");
                continue;
            }
        }
        // `<Type>::method` (no `as`).
        if let Some(gt_pos) = find_closing_angle(&s, start) {
            let type_part = &s[start + 1..gt_pos].trim();
            let type_name = extract_base_type(type_part);
            let after = &s[gt_pos + 1..];
            let before = &s[..start];
            s = format!("{before}{type_name}{after}");
            continue;
        }
        break;
    }

    // Clean up double colons from removals.
    while s.contains("::::") {
        s = s.replace("::::", "::");
    }
    s = s.trim_start_matches("::").to_owned();

    s
}

/// Check if a normalized name is a derive-generated function.
fn is_derive_fn(name: &str) -> bool {
    let leaf = name.rsplit("::").next().unwrap_or(name);
    matches!(leaf, "fmt" | "clone" | "eq" | "ne" | "partial_cmp" | "cmp"
        | "hash" | "encode" | "decode" | "borrow_decode" | "default"
        | "serialize" | "deserialize")
}

/// Check if a call target is to an external (std/core/alloc/third-party) crate.
fn is_external_call(name: &str) -> bool {
    let prefixes = [
        "std::", "core::", "alloc::", "serde::", "bincode::", "anyhow::",
        "assert_failed", "panicking::", "begin_panic",
    ];
    if prefixes.iter().any(|p| name.starts_with(p)) {
        return true;
    }
    // Common std/alloc types that appear as `Type::method` after normalization.
    let std_types = [
        "Vec::", "HashMap::", "BTreeMap::", "BTreeSet::", "HashSet::",
        "BinaryHeap::", "VecDeque::", "LinkedList::",
        "String::", "Box::", "Rc::", "Arc::", "Cell::", "RefCell::",
        "Mutex::", "RwLock::", "Option::", "Result::",
        "File::", "Cow::", "Weak::", "PhantomData::",
        "Iterator::", "IntoIterator::",
    ];
    std_types.iter().any(|p| name.starts_with(p))
}

/// Find ` as ` at the correct nesting level within angle brackets starting at `start`.
fn find_balanced_as(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return None; // Closed without finding ` as `.
                }
            }
            b' ' if depth == 1 => {
                if s[i..].starts_with(" as ") {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Find the closing `>` that matches the `<` at position `start`.
fn find_closing_angle(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Extract the base type name from a possibly generic type.
/// `Vec<u32>` → `Vec`, `std::vec::Vec<T>` → `Vec`, `&[f32]` → skip.
fn extract_base_type(ty: &str) -> String {
    let t = ty.trim().trim_start_matches('&').trim();
    // Find the last `::` segment before any `<`.
    let before_generic = if let Some(lt) = t.find('<') { &t[..lt] } else { t };
    let base = before_generic.rsplit("::").next().unwrap_or(before_generic);
    base.trim().to_owned()
}
