//! External type→method index built from std library and cargo dependency sources.
//!
//! Discovers Rust std source via `rustc --print sysroot` and cargo dependency
//! sources via `Cargo.toml` + `~/.cargo/registry/src/`. Parses all impl blocks
//! using tree-sitter (parallelized with rayon) to build a type→methods mapping.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

/// Cache version — bump when struct layout changes.
const EXTERN_CACHE_VERSION: u8 = 4;

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
    /// "type::method" (lowercase) → return_type_leaf (lowercase)
    /// Only populated for methods with non-primitive, non-void return types.
    pub return_types: BTreeMap<String, String>,
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

    /// Build a flat set of all method names for O(1) lookup.
    /// Use this instead of `any_type_has_method` in hot loops.
    pub fn all_method_set(&self) -> std::collections::HashSet<String> {
        self.types.values()
            .flat_map(|methods| methods.iter().cloned())
            .collect()
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
    /// Cache stored in `<db>/cache/extern_types.bin`.
    pub fn build(db_path: &Path) -> Self {
        let project_root = crate::helpers::find_project_root(db_path)
            .unwrap_or_else(|| db_path.parent().unwrap_or(Path::new(".")).to_path_buf());
        let fingerprint = compute_fingerprint(&project_root);

        // Try loading from project-local cache first.
        if let Some(cached) = Self::load_cache(db_path, &fingerprint) {
            return cached;
        }

        let global_cache = global_cache_dir();
        let t0 = std::time::Instant::now();

        let mut types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut return_types: BTreeMap<String, String> = BTreeMap::new();
        let mut cache_hits = 0u32;
        let mut parsed_groups = 0u32;

        // ── std library ──
        if let Some(std_src) = discover_std_src() {
            let rustc_hash = rustc_version_hash();
            let cache_key = format!("std-{rustc_hash}");

            if let Some(cached) = load_global_unit(&global_cache, &cache_key) {
                merge_methods(&cached, &mut types, &mut return_types);
                cache_hits += 1;
            } else {
                let mut std_files = Vec::new();
                for dir_name in STD_DIRS {
                    let sub = std_src.join(dir_name);
                    if sub.is_dir() {
                        collect_rs_files(&sub, &mut std_files);
                    }
                }
                let results = parse_files_parallel(&std_files);
                save_global_unit(&global_cache, &cache_key, &results);
                merge_methods(&results, &mut types, &mut return_types);
                parsed_groups += 1;
            }
        }

        // ── cargo deps (per crate-version) ──
        let lock_path = project_root.join("Cargo.lock");
        let dep_dirs = discover_cargo_deps(&project_root);
        let dep_versions = if lock_path.exists() {
            std::fs::read_to_string(&lock_path).ok()
                .map(|c| parse_cargo_lock_deps(&c))
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Build a name→version lookup from Cargo.lock.
        let _version_map: std::collections::HashMap<String, String> = dep_versions
            .into_iter()
            .collect();

        // Group dep_dirs by crate name-version for global caching.
        let mut uncached_files: Vec<PathBuf> = Vec::new();
        for dep_dir in &dep_dirs {
            // dep_dir is like ~/.cargo/registry/src/.../serde-1.0.210
            let dir_name = dep_dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            // Extract crate name and version from directory name.
            let cache_key = if let Some((name, ver)) = split_crate_dir_name(dir_name) {
                format!("{name}-{ver}")
            } else {
                String::new()
            };

            if !cache_key.is_empty() {
                if let Some(cached) = load_global_unit(&global_cache, &cache_key) {
                    merge_methods(&cached, &mut types, &mut return_types);
                    cache_hits += 1;
                    continue;
                }
            }

            // Parse this crate's files and cache individually.
            let mut crate_files = Vec::new();
            collect_rs_files(dep_dir, &mut crate_files);
            if !cache_key.is_empty() && !crate_files.is_empty() {
                let results = parse_files_parallel(&crate_files);
                save_global_unit(&global_cache, &cache_key, &results);
                merge_methods(&results, &mut types, &mut return_types);
                parsed_groups += 1;
            } else {
                uncached_files.extend(crate_files);
            }
        }

        // Parse any remaining files that couldn't be cached individually.
        if !uncached_files.is_empty() {
            let results = parse_files_parallel(&uncached_files);
            merge_methods(&results, &mut types, &mut return_types);
            parsed_groups += 1;
        }

        eprintln!("    [extern] {:.1}s — {} global cache hits, {} groups parsed, {} types, {} methods",
            t0.elapsed().as_secs_f64(), cache_hits, parsed_groups,
            types.len(), types.values().map(|m| m.len()).sum::<usize>());

        let index = Self {
            version: EXTERN_CACHE_VERSION,
            fingerprint,
            types,
            return_types,
        };

        index.save_cache(db_path);
        index
    }

    /// Try to load a cached extern index.
    fn load_cache(db_path: &Path, fingerprint: &str) -> Option<Self> {
        let path = cache_path(db_path);
        let data = std::fs::read(&path).ok()?;
        let config = bincode::config::standard();
        let (index, _): (Self, _) = bincode::decode_from_slice(&data, config).ok()?;
        if index.version == EXTERN_CACHE_VERSION && index.fingerprint == fingerprint {
            Some(index)
        } else {
            None
        }
    }

    /// Try to load a cached extern index from the DB directory.
    ///
    /// Returns `None` if no cache exists (does NOT rebuild).
    pub fn try_load_cached(db_path: &Path) -> Option<Self> {
        let project_root = crate::helpers::find_project_root(db_path)?;
        let fingerprint = compute_fingerprint(&project_root);
        Self::load_cache(db_path, &fingerprint)
    }

    /// Save the index to cache.
    fn save_cache(&self, db_path: &Path) {
        let path = cache_path(db_path);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let config = bincode::config::standard();
        if let Ok(data) = bincode::encode_to_vec(self, config) {
            let _ = std::fs::write(&path, data);
        }
    }
}

/// Cache file path for the extern type index — stored inside the DB directory.
fn cache_path(db_path: &Path) -> PathBuf {
    db_path.join("cache").join("extern_types.bin")
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

/// Find cargo dependency source directories from Cargo.lock.
///
/// Parses `Cargo.lock` for all package (name, version) pairs, then locates
/// their sources in `~/.cargo/registry/src/`.
/// Includes direct + transitive deps to ensure complete extern coverage.
/// Speed is managed by caching — this only runs on first build.
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

// ── Global per-crate cache ──────────────────────────────────────────

/// Per-crate parsed methods: Vec<(type_name, method_name, Option<return_type>)>.
type CrateMethodVec = Vec<(String, String, Option<String>)>;

/// Global cache directory: `~/.cache/v-code/extern/`.
fn global_cache_dir() -> Option<PathBuf> {
    let home = dirs_fallback()?;
    let dir = home.join(".cache").join("v-code").join("extern");
    Some(dir)
}

/// Cache version for per-crate global cache files.
const GLOBAL_UNIT_VERSION: u8 = 1;

/// Load a per-crate unit from the global cache.
fn load_global_unit(cache_dir: &Option<PathBuf>, key: &str) -> Option<CrateMethodVec> {
    let dir = cache_dir.as_ref()?;
    let path = dir.join(format!("{key}.bin"));
    let data = std::fs::read(&path).ok()?;
    if data.first() != Some(&GLOBAL_UNIT_VERSION) {
        return None;
    }
    let config = bincode::config::standard();
    let (methods, _): (CrateMethodVec, _) = bincode::decode_from_slice(&data[1..], config).ok()?;
    Some(methods)
}

/// Save a per-crate unit to the global cache.
fn save_global_unit(cache_dir: &Option<PathBuf>, key: &str, methods: &CrateMethodVec) {
    let Some(dir) = cache_dir.as_ref() else { return };
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join(format!("{key}.bin"));
    let config = bincode::config::standard();
    if let Ok(data) = bincode::encode_to_vec(methods, config) {
        let mut bytes = vec![GLOBAL_UNIT_VERSION];
        bytes.extend_from_slice(&data);
        let _ = std::fs::write(&path, bytes);
    }
}

/// Merge per-crate method results into the index tables.
fn merge_methods(
    methods: &CrateMethodVec,
    types: &mut BTreeMap<String, BTreeSet<String>>,
    return_types: &mut BTreeMap<String, String>,
) {
    for (type_name, method_name, ret_type) in methods {
        types.entry(type_name.clone()).or_default().insert(method_name.clone());
        if let Some(ret) = ret_type {
            let key = format!("{type_name}::{method_name}");
            return_types.entry(key).or_insert_with(|| ret.clone());
        }
    }
}

/// Parse a list of .rs files in parallel, returning flat method tuples.
fn parse_files_parallel(files: &[PathBuf]) -> CrateMethodVec {
    use std::sync::atomic::AtomicU64;
    let ns_parse = AtomicU64::new(0);
    let ns_walk = AtomicU64::new(0);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .thread_name(|i| format!("extern-{i}"))
        .build();

    let per_file: Vec<CrateMethodVec> = match pool {
        Ok(ref pool) => pool.install(|| {
            files
                .par_iter()
                .filter_map(|path| {
                    let src = std::fs::read(path).ok()?;
                    if !has_impl_keyword(&src) {
                        return None;
                    }
                    let methods = v_code_chunk::extern_impl::extract_impl_methods_timed(
                        &src, &ns_parse, &ns_walk,
                    );
                    if methods.is_empty() { None } else { Some(methods) }
                })
                .collect()
        }),
        Err(_) => Vec::new(),
    };

    per_file.into_iter().flatten().collect()
}

/// Compute a short hash of the rustc version string.
fn rustc_version_hash() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    if let Ok(output) = std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        output.stdout.hash(&mut hasher);
    }
    format!("{:08x}", hasher.finish() as u32)
}

/// Split a cargo registry directory name into (crate_name, version).
///
/// E.g. `"serde-1.0.210"` → `Some(("serde", "1.0.210"))`,
///      `"proc-macro2-1.0.86"` → `Some(("proc-macro2", "1.0.86"))`.
fn split_crate_dir_name(dir_name: &str) -> Option<(&str, &str)> {
    // Version starts at the last `-` followed by a digit.
    let mut last_dash = None;
    for (i, b) in dir_name.bytes().enumerate() {
        if b == b'-' {
            // Check if next char is a digit.
            if dir_name.as_bytes().get(i + 1).is_some_and(u8::is_ascii_digit) {
                last_dash = Some(i);
            }
        }
    }
    let pos = last_dash?;
    let name = &dir_name[..pos];
    let version = &dir_name[pos + 1..];
    if name.is_empty() || version.is_empty() {
        None
    } else {
        Some((name, version))
    }
}
