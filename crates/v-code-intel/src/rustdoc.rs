//! Parse rustdoc JSON output to extract type information for call graph enrichment.
//!
//! Provides [`RustdocTypes`] — a set of lookup tables extracted from
//! `cargo rustdoc --output-format json` output that supplements tree-sitter
//! type inference with compiler-verified type information.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use rayon::prelude::*;

/// Type information extracted from rustdoc JSON output.
///
/// All keys are lowercase for case-insensitive matching.
#[derive(Default, Debug, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct RustdocTypes {
    /// `function_name → return_type` (fully resolved, leaf only).
    /// E.g. `"callgraph::build" → "callgraph"`, `"graph_cache_path" → "pathbuf"`
    pub fn_return_types: BTreeMap<String, String>,

    /// `method_name → owner_type` for disambiguating short name collisions.
    /// E.g. `"build" → "callgraph"`, `"resolve" → "callgraph"`
    /// Note: last-write-wins for duplicate method names across types.
    pub method_owner: BTreeMap<String, Vec<String>>,

    /// `struct_name.field_name → field_type` (lowercase).
    /// E.g. `"callgraph.names" → "vec"`, `"codechunk.kind" → "string"`
    pub field_types: BTreeMap<String, String>,
}

impl RustdocTypes {
    /// Parse a rustdoc JSON file and extract type tables.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read rustdoc JSON: {}", path.display()))?;
        Self::parse(&content)
    }

    /// Parse rustdoc JSON from a string.
    pub fn parse(json: &str) -> Result<Self> {
        let doc: serde_json::Value =
            serde_json::from_str(json).context("failed to parse rustdoc JSON")?;

        let index = doc
            .get("index")
            .and_then(|v| v.as_object())
            .context("missing 'index' in rustdoc JSON")?;

        let mut result = Self::default();

        // First pass: collect impl block method→owner mappings and struct fields.
        for item in index.values() {
            let inner = match item.get("inner").and_then(|v| v.as_object()) {
                Some(v) => v,
                None => continue,
            };

            // impl blocks: extract method→owner type
            if let Some(impl_data) = inner.get("impl").and_then(|v| v.as_object()) {
                let for_type = impl_data.get("for");
                let owner = for_type.and_then(extract_type_name);
                if let Some(owner) = owner {
                    let owner_lower = leaf_lower(&owner);
                    for method_id in impl_data
                        .get("items")
                        .and_then(|v| v.as_array())
                        .into_iter()
                        .flatten()
                    {
                        let mid = match method_id {
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::String(s) => s.clone(),
                            _ => continue,
                        };
                        if let Some(method_item) = index.get(&mid) {
                            if let Some(name) = method_item.get("name").and_then(|v| v.as_str()) {
                                let name_lower = name.to_lowercase();
                                // Skip blanket trait impls (clone, from, into, etc.)
                                if !is_blanket_trait_method(&name_lower) {
                                    result
                                        .method_owner
                                        .entry(name_lower)
                                        .or_default()
                                        .push(owner_lower.clone());
                                }
                            }
                            // Also extract return type for methods
                            extract_fn_return(method_item, &owner_lower, &mut result.fn_return_types);
                        }
                    }
                }
            }

            // struct fields
            if let Some(struct_data) = inner.get("struct").and_then(|v| v.as_object()) {
                let struct_name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_lowercase();
                let fields = struct_data
                    .get("kind")
                    .and_then(|v| v.get("plain"))
                    .and_then(|v| v.get("fields"))
                    .and_then(|v| v.as_array());
                if let Some(fields) = fields {
                    for fid in fields {
                        let fid_str = match fid {
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::String(s) => s.clone(),
                            _ => continue,
                        };
                        if let Some(field_item) = index.get(&fid_str) {
                            let field_name = field_item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_lowercase();
                            if let Some(field_type) = field_item
                                .get("inner")
                                .and_then(|v| v.get("struct_field"))
                                .and_then(extract_type_name)
                            {
                                let key = format!("{struct_name}.{field_name}");
                                result.field_types.insert(key, leaf_lower(&field_type));
                            }
                        }
                    }
                }
            }

            // Free functions (not in impl blocks)
            if inner.contains_key("function") {
                extract_fn_return(item, "", &mut result.fn_return_types);
            }
        }

        Ok(result)
    }

    /// Look up the return type for a function name (lowercase).
    pub fn return_type(&self, fn_name: &str) -> Option<&str> {
        self.fn_return_types.get(fn_name).map(String::as_str)
    }

    /// Look up possible owner types for a method name.
    pub fn owners_of(&self, method: &str) -> Option<&[String]> {
        self.method_owner.get(method).map(Vec::as_slice)
    }

    /// Look up a struct field type.
    pub fn field_type(&self, struct_name: &str, field: &str) -> Option<&str> {
        let key = format!("{struct_name}.{field}");
        self.field_types.get(&key).map(String::as_str)
    }

    /// Merge another `RustdocTypes` into this one.
    pub fn merge(&mut self, other: Self) {
        for (k, v) in other.fn_return_types {
            self.fn_return_types.entry(k).or_insert(v);
        }
        for (k, mut v) in other.method_owner {
            self.method_owner.entry(k).or_default().append(&mut v);
        }
        for (k, v) in other.field_types {
            self.field_types.entry(k).or_insert(v);
        }
    }

    /// Load and merge all `*.json` files from a directory (parallel).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read dir: {}", dir.display()))?;
        let paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();

        let parts: Vec<Self> = paths
            .par_iter()
            .filter_map(|path| Self::from_file(path).ok())
            .collect();

        let mut combined = Self::default();
        for part in parts {
            combined.merge(part);
        }
        Ok(combined)
    }
}

// ── Internal helpers ──────────────────────────────────────────────────

/// Extract a human-readable type name from a rustdoc JSON type node.
fn extract_type_name(ty: &serde_json::Value) -> Option<String> {
    if let Some(obj) = ty.as_object() {
        if let Some(rp) = obj.get("resolved_path").and_then(|v| v.as_object()) {
            return rp.get("path").and_then(|v| v.as_str()).map(String::from);
        }
        if let Some(p) = obj.get("primitive").and_then(|v| v.as_str()) {
            return Some(p.to_owned());
        }
        if let Some(g) = obj.get("generic").and_then(|v| v.as_str()) {
            return Some(g.to_owned());
        }
        if let Some(br) = obj.get("borrowed_ref").and_then(|v| v.as_object()) {
            return br.get("type").and_then(extract_type_name);
        }
        if let Some(sl) = obj.get("slice") {
            return extract_type_name(sl);
        }
        if let Some(arr) = obj.get("array").and_then(|v| v.as_object()) {
            return arr.get("type").and_then(extract_type_name);
        }
        if let Some(tup) = obj.get("tuple").and_then(|v| v.as_array())
            && tup.is_empty() {
                return Some("()".to_owned());
            }
    }
    None
}

/// Extract function return type and insert into the map.
fn extract_fn_return(
    item: &serde_json::Value,
    owner: &str,
    map: &mut BTreeMap<String, String>,
) {
    let fn_data = item
        .get("inner")
        .and_then(|v| v.as_object())
        .and_then(|v| v.get("function"))
        .and_then(|v| v.as_object());
    let fn_data = match fn_data {
        Some(v) => v,
        None => return,
    };

    let ret = fn_data
        .get("sig")
        .and_then(|v| v.as_object())
        .and_then(|v| v.get("output"));
    let ret_type = ret.and_then(extract_type_name);
    let Some(ret_type) = ret_type else { return };

    let name = item
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if name.is_empty() {
        return;
    }

    let mut resolved = leaf_lower(&ret_type);

    // Resolve Self → owner type
    if (resolved == "self" || resolved == "&self") && !owner.is_empty() {
        resolved = owner.to_owned();
    }

    let name_lower = name.to_lowercase();

    // Insert as both bare name and qualified name
    if !owner.is_empty() {
        let qualified = format!("{owner}::{name_lower}");
        map.insert(qualified, resolved.clone());
    }
    map.entry(name_lower).or_insert(resolved);
}

/// Extract the leaf (last segment, lowercase, no generics) from a type path.
/// `"crate::parse::CodeChunk"` → `"codechunk"`, `"Option"` → `"option"`
fn leaf_lower(path: &str) -> String {
    let leaf = path.rsplit("::").next().unwrap_or(path);
    let leaf = leaf.split('<').next().unwrap_or(leaf);
    leaf.to_lowercase()
}

/// Check if a method name is a blanket trait impl (From, Clone, etc.)
/// These are auto-generated for nearly every type and not useful for disambiguation.
fn is_blanket_trait_method(name: &str) -> bool {
    matches!(
        name,
        "from"
            | "into"
            | "try_from"
            | "try_into"
            | "clone"
            | "clone_into"
            | "clone_to_uninit"
            | "to_owned"
            | "borrow"
            | "borrow_mut"
            | "type_id"
            | "eq"
            | "ne"
            | "fmt"
            | "encode"
            | "decode"
            | "borrow_decode"
            | "default"
            | "hash"
            | "partial_cmp"
            | "cmp"
            | "serialize"
            | "deserialize"
            // Auto-generated compiler intrinsics
            | "error"
            | "align"
            | "init"
            | "deref"
            | "deref_mut"
            | "drop"
            | "owned"
    )
}

// ── Cargo rustdoc runner ──────────────────────────────────────────────

/// Run `cargo rustdoc --output-format json` for all workspace lib crates
/// and merge the results.
///
/// Requires nightly toolchain. Returns `None` on any failure (nightly missing,
/// build error, etc.) — callers should treat this as a best-effort enrichment.
///
/// `project_root` should contain `Cargo.toml`.
pub fn generate_and_load(project_root: &Path) -> Option<RustdocTypes> {
    let manifest = project_root.join("Cargo.toml");
    if !manifest.exists() {
        eprintln!("[rustdoc] No Cargo.toml found in {}", project_root.display());
        return None;
    }

    let crates = list_workspace_lib_crates(project_root);
    if crates.is_empty() {
        eprintln!("[rustdoc] No lib crates found in workspace");
        return None;
    }

    eprintln!("[rustdoc] Generating rustdoc JSON for {} crate(s)", crates.len());
    let start = std::time::Instant::now();

    let mut succeeded = 0u32;
    for krate in &crates {
        let output = Command::new("cargo")
            .args([
                "+nightly", "rustdoc", "--lib",
                "-p", krate,
                "--manifest-path", &manifest.to_string_lossy(),
                "--", "-Z", "unstable-options", "--output-format", "json",
                "--document-private-items",
            ])
            .env("RUSTDOCFLAGS", "")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .current_dir(project_root)
            .output()
            .ok();

        match output {
            Some(o) if o.status.success() => succeeded += 1,
            Some(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                let short: String = stderr.lines().take(2).collect::<Vec<_>>().join(" ");
                eprintln!("[rustdoc] {krate}: failed ({short})");
            }
            None => eprintln!("[rustdoc] {krate}: failed to spawn"),
        }
    }

    let elapsed = start.elapsed();
    eprintln!("[rustdoc] {succeeded}/{} crates completed in {:.1}s",
        crates.len(), elapsed.as_secs_f64());

    // Load and merge all JSON files from target/doc/
    let doc_dir = project_root.join("target").join("doc");
    match RustdocTypes::from_dir(&doc_dir) {
        Ok(types) if !types.fn_return_types.is_empty() => {
            eprintln!(
                "[rustdoc] Merged: {} fn returns, {} method owners, {} field types",
                types.fn_return_types.len(),
                types.method_owner.len(),
                types.field_types.len(),
            );
            Some(types)
        }
        Ok(_) => {
            eprintln!("[rustdoc] No type information extracted");
            None
        }
        Err(e) => {
            eprintln!("[rustdoc] Failed to load JSON: {e}");
            None
        }
    }
}

/// List workspace member crate names that have a `lib` target.
fn list_workspace_lib_crates(project_root: &Path) -> Vec<String> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version=1"])
        .current_dir(project_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok();

    let Some(output) = output else { return Vec::new() };
    if !output.status.success() {
        return Vec::new();
    }

    let Ok(meta) = serde_json::from_slice::<serde_json::Value>(&output.stdout) else {
        return Vec::new();
    };

    let Some(packages) = meta.get("packages").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    packages
        .iter()
        .filter(|pkg| {
            pkg.get("targets")
                .and_then(|t| t.as_array())
                .is_some_and(|targets| {
                    targets.iter().any(|t| {
                        t.get("kind")
                            .and_then(|k| k.as_array())
                            .is_some_and(|kinds| {
                                kinds.iter().any(|k| k.as_str() == Some("lib"))
                            })
                    })
                })
        })
        .filter_map(|pkg| pkg.get("name").and_then(|n| n.as_str()).map(String::from))
        .collect()
}

/// Load rustdoc types from cached sources.
///
/// Tries in order:
/// 1. `<db>/cache/rustdoc/` directory (multiple JSON files)
/// 2. `<db>/cache/rustdoc.json` (single file, legacy)
/// 3. `<project_root>/target/doc/` (directly from cargo rustdoc output)
pub fn load_cached(db_path: &Path) -> Option<RustdocTypes> {
    let bin_cache = db_path.join("cache").join("rustdoc_types.bin");

    // 1. Bincode cache (fastest — single file, pre-parsed)
    if let Some(types) = load_bincode_cache(&bin_cache) {
        return Some(types);
    }

    // 2. Multi-file JSON cache directory
    let cache_dir = db_path.join("cache").join("rustdoc");
    if cache_dir.is_dir()
        && let Ok(types) = RustdocTypes::from_dir(&cache_dir)
            && !types.fn_return_types.is_empty() {
                save_bincode_cache(&bin_cache, &types);
                return Some(types);
            }

    // 3. Single file cache (legacy)
    let cache_path = db_path.join("cache").join("rustdoc.json");
    if cache_path.exists()
        && let Ok(types) = RustdocTypes::from_file(&cache_path) {
            save_bincode_cache(&bin_cache, &types);
            return Some(types);
        }

    // 4. Try target/doc/ from project root
    if let Some(project_root) = crate::helpers::find_project_root(db_path) {
        let doc_dir = project_root.join("target").join("doc");
        if doc_dir.is_dir()
            && let Ok(types) = RustdocTypes::from_dir(&doc_dir)
                && !types.fn_return_types.is_empty() {
                    save_bincode_cache(&bin_cache, &types);
                    return Some(types);
                }
    }

    None
}

/// Cache format version for `rustdoc_types.bin` — bump when `RustdocTypes` layout changes.
const RUSTDOC_CACHE_VERSION: u8 = 1;

/// Load `RustdocTypes` from a bincode cache file.
///
/// Uses a version prefix byte to detect stale caches.
fn load_bincode_cache(path: &Path) -> Option<RustdocTypes> {
    let data = std::fs::read(path).ok()?;
    if data.first() != Some(&RUSTDOC_CACHE_VERSION) {
        return None;
    }
    let config = bincode::config::standard();
    let (types, _): (RustdocTypes, _) = bincode::decode_from_slice(&data[1..], config).ok()?;
    if types.fn_return_types.is_empty() { None } else { Some(types) }
}

/// Save `RustdocTypes` to a bincode cache file.
fn save_bincode_cache(path: &Path, types: &RustdocTypes) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let config = bincode::config::standard();
    if let Ok(data) = bincode::encode_to_vec(types, config) {
        let mut bytes = vec![RUSTDOC_CACHE_VERSION];
        bytes.extend_from_slice(&data);
        let _ = std::fs::write(path, bytes);
    }
}

/// Save all rustdoc JSON files from `target/doc/` to the cache directory.
pub fn save_to_cache(db_path: &Path, project_root: &Path) -> Option<()> {
    let doc_dir = project_root.join("target").join("doc");
    if !doc_dir.is_dir() {
        return None;
    }
    let cache_dir = db_path.join("cache").join("rustdoc");
    std::fs::create_dir_all(&cache_dir).ok()?;

    let mut count = 0u32;
    let entries = std::fs::read_dir(&doc_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            let dest = cache_dir.join(path.file_name()?);
            if std::fs::copy(&path, &dest).is_ok() {
                count += 1;
            }
        }
    }

    if count > 0 {
        eprintln!("[rustdoc] Cached {count} JSON file(s) to {}", cache_dir.display());
    }
    Some(())
}

/// Find the rustdoc JSON output file in `target/doc/`.
///
/// Looks for `*.json` files, preferring the most recently modified one.
#[expect(dead_code, reason = "reserved for future rustdoc integration")]
fn find_rustdoc_json(project_root: &Path) -> Option<PathBuf> {
    let doc_dir = project_root.join("target").join("doc");
    if !doc_dir.exists() {
        return None;
    }

    let mut best: Option<(PathBuf, std::time::SystemTime)> = None;
    let entries = std::fs::read_dir(&doc_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            let mtime = entry.metadata().ok()?.modified().ok()?;
            if best.as_ref().is_none_or(|(_, t)| mtime > *t) {
                best = Some((path, mtime));
            }
        }
    }

    best.map(|(p, _)| p)
}
