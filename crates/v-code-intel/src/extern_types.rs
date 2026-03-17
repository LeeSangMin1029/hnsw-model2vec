//! External type→method index built from std library and cargo dependency sources.
//!
//! Discovers Rust std source via `rustc --print sysroot` and cargo dependency
//! sources via `Cargo.toml` + `~/.cargo/registry/src/`. Parses all impl blocks
//! using tree-sitter (parallelized with rayon) to build a type→methods mapping.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

/// Cache version — bump when struct layout changes.
const EXTERN_CACHE_VERSION: u8 = 3;

/// Std library subdirectories that contain useful type definitions.
/// Skips test-only, profiler, backtrace, and platform-specific crates.
const STD_DIRS: &[&str] = &["core", "alloc", "std"];

/// Maps lowercase type names to their known method names.
///
/// Built from std library and cargo dependency sources.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode)]
pub struct ExternMethodIndex {
    /// Cache format version.
    version: u8,
    /// Fingerprint for cache invalidation.
    fingerprint: String,
    /// type_name (lowercase) → set of method_names (lowercase)
    pub types: BTreeMap<String, BTreeSet<String>>,
}

impl ExternMethodIndex {
    /// Check if a type has a specific method.
    pub fn has_method(&self, type_name: &str, method: &str) -> bool {
        self.types
            .get(type_name)
            .is_some_and(|methods| methods.contains(method))
    }

    /// Check if any extern type has this method (type-agnostic lookup).
    pub fn any_type_has_method(&self, method: &str) -> bool {
        self.types.values().any(|methods| methods.contains(method))
    }

    /// Total number of type→method entries.
    pub fn total_methods(&self) -> usize {
        self.types.values().map(|m| m.len()).sum()
    }

    /// Number of types indexed.
    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    /// Build the index from std + cargo dependency sources.
    ///
    /// Uses a cache file to avoid re-parsing on subsequent runs.
    /// Cache is invalidated when `Cargo.toml` or `rustc --version` changes.
    pub fn build(project_root: &Path) -> Self {
        let fingerprint = compute_fingerprint(project_root);

        // Try loading from cache first.
        if let Some(cached) = Self::load_cache(project_root, &fingerprint) {
            return cached;
        }

        // Collect all .rs file paths first, then parse in parallel.
        let mut all_files: Vec<PathBuf> = Vec::new();

        if let Some(std_src) = discover_std_src() {
            for dir_name in STD_DIRS {
                let sub = std_src.join(dir_name);
                if sub.is_dir() {
                    collect_rs_files(&sub, &mut all_files);
                }
            }
        }

        for dep_dir in discover_cargo_deps(project_root) {
            collect_rs_files(&dep_dir, &mut all_files);
        }

        // Parallel parse: read + quick impl check + tree-sitter parse.
        let per_file_results: Vec<Vec<(String, String)>> = all_files
            .par_iter()
            .filter_map(|path| {
                let src = std::fs::read(path).ok()?;
                // Quick pre-filter: skip files without "impl " to avoid tree-sitter overhead.
                if !has_impl_keyword(&src) {
                    return None;
                }
                let methods = v_code_chunk::extern_impl::extract_impl_methods(&src);
                if methods.is_empty() {
                    None
                } else {
                    Some(methods)
                }
            })
            .collect();

        // Merge results.
        let mut types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for methods in per_file_results {
            for (type_name, method_name) in methods {
                types.entry(type_name).or_default().insert(method_name);
            }
        }

        let index = Self {
            version: EXTERN_CACHE_VERSION,
            fingerprint,
            types,
        };

        index.save_cache(project_root);
        index
    }

    /// Try to load a cached extern index.
    fn load_cache(project_root: &Path, fingerprint: &str) -> Option<Self> {
        let path = cache_path(project_root);
        let data = std::fs::read(&path).ok()?;
        let config = bincode::config::standard();
        let (index, _): (Self, _) = bincode::decode_from_slice(&data, config).ok()?;
        if index.version == EXTERN_CACHE_VERSION && index.fingerprint == fingerprint {
            Some(index)
        } else {
            None
        }
    }

    /// Save the index to cache.
    fn save_cache(&self, project_root: &Path) {
        let path = cache_path(project_root);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let config = bincode::config::standard();
        if let Ok(data) = bincode::encode_to_vec(self, config) {
            let _ = std::fs::write(&path, data);
        }
    }
}

/// Cache file path for the extern type index.
fn cache_path(project_root: &Path) -> PathBuf {
    project_root.join(".code.db").join("cache").join("extern_types.bin")
}

/// Compute a fingerprint from Cargo.toml content + rustc version.
fn compute_fingerprint(project_root: &Path) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Hash Cargo.toml (direct deps) + Cargo.lock (pinned versions)
    for name in &["Cargo.toml", "Cargo.lock"] {
        if let Ok(content) = std::fs::read_to_string(project_root.join(name)) {
            content.hash(&mut hasher);
        }
    }

    // Hash rustc version
    if let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        output.stdout.hash(&mut hasher);
    }

    format!("{:016x}", hasher.finish())
}

/// Quick byte scan for `impl ` keyword — avoids tree-sitter parse for files without impl blocks.
fn has_impl_keyword(src: &[u8]) -> bool {
    src.windows(5).any(|w| w == b"impl ")
}

/// Find the std library source directory.
fn discover_std_src() -> Option<PathBuf> {
    let output = std::process::Command::new("rustc")
        .arg("--print")
        .arg("sysroot")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sysroot = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    let lib_path = PathBuf::from(&sysroot)
        .join("lib")
        .join("rustlib")
        .join("src")
        .join("rust")
        .join("library");
    if lib_path.is_dir() {
        Some(lib_path)
    } else {
        None
    }
}

/// Find cargo dependency source directories from Cargo.lock (all deps including transitive).
///
/// Parses `Cargo.lock` for all package (name, version) pairs, then locates
/// their sources in `~/.cargo/registry/src/`.
fn discover_cargo_deps(project_root: &Path) -> Vec<PathBuf> {
    // Get all pinned packages from Cargo.lock (direct + transitive).
    let lock_path = project_root.join("Cargo.lock");
    let Ok(lock_content) = std::fs::read_to_string(&lock_path) else {
        return Vec::new();
    };
    let all_deps = parse_cargo_lock_deps(&lock_content);
    if all_deps.is_empty() {
        return Vec::new();
    }

    // Find registry source directory.
    let home = std::env::var("CARGO_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs_fallback().map(|h| h.join(".cargo")));
    let Some(cargo_home) = home else {
        return Vec::new();
    };
    let registry_src = cargo_home.join("registry").join("src");
    if !registry_src.is_dir() {
        return Vec::new();
    }

    let Ok(entries) = std::fs::read_dir(&registry_src) else {
        return Vec::new();
    };
    let index_dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();

    let mut result = Vec::new();
    for (name, version) in &all_deps {
        let dir_name = format!("{name}-{version}");
        for index_dir in &index_dirs {
            let dep_path = index_dir.join(&dir_name);
            if dep_path.is_dir() {
                result.push(dep_path);
                break;
            }
        }
    }
    result
}

/// Parse Cargo.lock to extract (name, version) pairs.
fn parse_cargo_lock_deps(content: &str) -> Vec<(String, String)> {
    let mut deps = Vec::new();
    let mut in_package = false;
    let mut name = String::new();
    let mut version = String::new();

    for line in content.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            if !name.is_empty() && !version.is_empty() {
                deps.push((
                    std::mem::take(&mut name),
                    std::mem::take(&mut version),
                ));
            }
            in_package = true;
            continue;
        }
        if in_package {
            if let Some(v) = line.strip_prefix("name = ") {
                name = v.trim_matches('"').to_owned();
            } else if let Some(v) = line.strip_prefix("version = ") {
                version = v.trim_matches('"').to_owned();
            }
        }
    }
    if !name.is_empty() && !version.is_empty() {
        deps.push((name, version));
    }
    deps
}

/// Recursively collect all `.rs` file paths (skipping test/bench directories).
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            // Skip test, bench, and example directories.
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, "tests" | "test" | "benches" | "examples" | "target") {
                continue;
            }
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Fallback home directory discovery.
fn dirs_fallback() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}
