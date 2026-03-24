//! MIR-based call edge extraction via `mir-callgraph` subprocess.
//!
//! Parses JSONL output from the `mir-callgraph` tool and provides
//! resolved call edges for accurate graph construction.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// A single call edge extracted from MIR.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MirCallEdge {
    pub caller: String,
    pub caller_file: String,
    pub callee: String,
    #[serde(default)]
    pub callee_file: String,
    #[serde(default)]
    pub callee_start_line: usize,
    pub line: usize,
    #[serde(default)]
    pub is_local: bool,
}

/// Collection of MIR edges indexed for fast lookup by caller.
#[derive(Debug, Default)]
pub struct MirEdgeMap {
    /// (caller_file_normalized, line) → Vec<callee_name>
    pub by_location: HashMap<(String, usize), Vec<String>>,
    /// caller_name → Vec<(callee_name, callee_file, callee_start_line, call_line)>
    pub by_caller: HashMap<String, Vec<CalleeInfo>>,
    /// Total edge count
    pub total: usize,
    /// caller_name → crate_name (tracks which crate each caller belongs to)
    pub caller_crate: HashMap<String, String>,
    /// Reverse index: crate_name → Vec<caller_name> (for O(1) crate lookup)
    pub crate_to_callers: HashMap<String, Vec<String>>,
}

/// Callee information from a MIR edge.
#[derive(Debug, Clone)]
pub struct CalleeInfo {
    pub name: String,
    pub file: String,
    pub start_line: usize,
    pub call_line: usize,
}

impl MirEdgeMap {
    /// Load MIR edges from a JSONL file.
    pub fn from_jsonl(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open MIR edges: {}", path.display()))?;
        let reader = std::io::BufReader::new(file);
        let mut map = Self::default();

        for line in reader.lines() {
            let line = line.with_context(|| "failed to read MIR edge line")?;
            if line.trim().is_empty() { continue; }
            let edge: MirCallEdge = serde_json::from_str(&line)
                .with_context(|| format!("failed to parse MIR edge: {line}"))?;

            let file_normalized = normalize_path(&edge.caller_file);
            map.by_location
                .entry((file_normalized, edge.line))
                .or_default()
                .push(edge.callee.clone());

            let callee_file_normalized = normalize_path(&edge.callee_file);
            map.by_caller
                .entry(edge.caller.clone())
                .or_default()
                .push(CalleeInfo {
                    name: edge.callee,
                    file: callee_file_normalized,
                    start_line: edge.callee_start_line,
                    call_line: edge.line,
                });

            map.total += 1;
        }

        Ok(map)
    }

    /// Load all `.edges.jsonl` files from a directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let mut combined = Self::default();
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read MIR edge dir: {}", dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let name = path.to_string_lossy();
            if name.ends_with(".edges.jsonl") {
                // Extract crate name from filename: e.g. "v_hnsw_cli.edges.jsonl" → "v_hnsw_cli"
                let crate_name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_suffix(".edges"))
                    .unwrap_or("")
                    .to_owned();

                let partial = Self::from_jsonl(&path)?;
                for (k, v) in partial.by_location {
                    combined.by_location.entry(k).or_default().extend(v);
                }
                for (k, _v) in &partial.by_caller {
                    if !crate_name.is_empty() {
                        combined.caller_crate.insert(k.clone(), crate_name.clone());
                    }
                }
                for (k, v) in partial.by_caller {
                    combined.by_caller.entry(k).or_default().extend(v);
                }
                combined.total += partial.total;
            }
        }

        // Build reverse index: crate → callers
        for (caller, crate_name) in &combined.caller_crate {
            combined.crate_to_callers
                .entry(crate_name.clone())
                .or_default()
                .push(caller.clone());
        }

        Ok(combined)
    }

    /// Resolve a call from a specific file and line to its callee names.
    pub fn resolve_at(&self, file: &str, line: usize) -> Option<&[String]> {
        let key = (normalize_path(file), line);
        self.by_location.get(&key).map(|v| v.as_slice())
    }

    /// Get all callees for a given caller function name.
    pub fn callees_of(&self, caller: &str) -> Option<&[CalleeInfo]> {
        self.by_caller.get(caller).map(|v| v.as_slice())
    }

    /// Get the set of all unique crate names present in this edge map.
    pub fn crate_names(&self) -> std::collections::HashSet<&str> {
        self.caller_crate.values().map(String::as_str).collect()
    }

    /// Get callers belonging to a specific crate.
    /// Uses a pre-built reverse index for O(1) lookup instead of O(N) scan.
    pub fn callers_for_crate<'a>(&'a self, crate_name: &str) -> Vec<&'a str> {
        self.crate_to_callers.get(crate_name)
            .map(|callers| callers.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }
}

// Embedded mir-callgraph source files for auto-build.
const MIR_CALLGRAPH_MAIN_RS: &str =
    include_str!("../../../tools/mir-callgraph/src/main.rs");
const MIR_CALLGRAPH_CARGO_TOML: &str =
    include_str!("../../../tools/mir-callgraph/Cargo.toml");
const MIR_CALLGRAPH_RUST_TOOLCHAIN: &str =
    include_str!("../../../tools/mir-callgraph/rust-toolchain.toml");

/// Binary name for mir-callgraph (platform-dependent).
fn mir_callgraph_bin_name() -> &'static str {
    if cfg!(windows) { "mir-callgraph.exe" } else { "mir-callgraph" }
}

/// Base directory for v-code data: `~/.v-code/`.
fn v_code_home() -> Result<PathBuf> {
    let home = v_hnsw_core::home_dir()
        .context("cannot determine home directory")?;
    Ok(home.join(".v-code"))
}

/// Get the current nightly rustc version string, or None if nightly is not installed.
fn nightly_rustc_version() -> Option<String> {
    Command::new("rustc")
        .args(["+nightly", "--version"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
}

/// Extract embedded mir-callgraph source to the build directory.
fn extract_mir_callgraph_source(build_dir: &Path) -> Result<()> {
    let src_dir = build_dir.join("src");
    std::fs::create_dir_all(&src_dir)
        .with_context(|| format!("failed to create build dir: {}", src_dir.display()))?;

    std::fs::write(build_dir.join("Cargo.toml"), MIR_CALLGRAPH_CARGO_TOML)
        .context("failed to write Cargo.toml")?;
    std::fs::write(build_dir.join("rust-toolchain.toml"), MIR_CALLGRAPH_RUST_TOOLCHAIN)
        .context("failed to write rust-toolchain.toml")?;
    std::fs::write(src_dir.join("main.rs"), MIR_CALLGRAPH_MAIN_RS)
        .context("failed to write main.rs")?;

    Ok(())
}

/// Build mir-callgraph from embedded source using nightly toolchain.
fn build_mir_callgraph(base: &Path) -> Result<PathBuf> {
    let nightly_ver = nightly_rustc_version().ok_or_else(|| {
        anyhow::anyhow!(
            "nightly Rust toolchain required for v-code add.\n\
             Run: rustup toolchain install nightly --component rust-src rustc-dev llvm-tools-preview"
        )
    })?;

    let bin_dir = base.join("bin");
    let cached_bin = bin_dir.join(mir_callgraph_bin_name());
    let version_file = bin_dir.join(".nightly-version");

    // Check if cached binary exists and nightly version matches
    if cached_bin.exists() {
        if let Ok(saved_ver) = std::fs::read_to_string(&version_file) {
            if saved_ver.trim() == nightly_ver {
                return Ok(cached_bin);
            }
            eprintln!("  [mir] nightly version changed, rebuilding mir-callgraph...");
        }
    } else {
        eprintln!("  [mir] building mir-callgraph (first run)...");
    }

    // Extract source
    let build_dir = base.join("build").join("mir-callgraph");
    extract_mir_callgraph_source(&build_dir)?;

    // Build with nightly
    eprintln!("  [mir] cargo +nightly build --release (this may take a minute)...");
    let status = Command::new("cargo")
        .args(["+nightly", "build", "--release"])
        .current_dir(&build_dir)
        .status()
        .context("failed to run cargo +nightly build")?;

    if !status.success() {
        bail!("mir-callgraph build failed (exit code: {status})");
    }

    // Copy built binary to bin/
    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed to create bin dir: {}", bin_dir.display()))?;

    let built_bin = build_dir
        .join("target")
        .join("release")
        .join(mir_callgraph_bin_name());

    std::fs::copy(&built_bin, &cached_bin).with_context(|| {
        format!(
            "failed to copy built binary from {} to {}",
            built_bin.display(),
            cached_bin.display()
        )
    })?;

    // Save nightly version
    std::fs::write(&version_file, &nightly_ver)
        .context("failed to save nightly version")?;

    eprintln!("  [mir] mir-callgraph ready: {}", cached_bin.display());
    Ok(cached_bin)
}

/// Find the mir-callgraph binary.
///
/// Search order:
/// 1. Override path (explicit)
/// 2. Sibling to current exe
/// 3. Cached build in `~/.v-code/bin/`
/// 4. Auto-build from embedded source
fn find_mir_callgraph_bin(override_path: Option<&Path>) -> Result<PathBuf> {
    // 1. Explicit override
    if let Some(p) = override_path {
        return Ok(p.to_path_buf());
    }

    // 2. Sibling to current exe
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.with_file_name(mir_callgraph_bin_name());
        if sibling.exists() {
            return Ok(sibling);
        }
    }

    // 3 & 4. Cached build or auto-build
    let base = v_code_home()?;
    build_mir_callgraph(&base)
}

/// Run mir-callgraph on the entire workspace.
pub fn run_mir_callgraph(project_root: &Path, mir_callgraph_bin: Option<&Path>) -> Result<MirEdgeMap> {
    run_mir_callgraph_for(project_root, mir_callgraph_bin, &[], false)
}

/// Run mir-callgraph for specific crates only (or all if `crates` is empty).
///
/// When `rust_only` is true, Python/TypeScript extractors are skipped entirely.
pub fn run_mir_callgraph_for(
    project_root: &Path,
    mir_callgraph_bin: Option<&Path>,
    crates: &[&str],
    rust_only: bool,
) -> Result<MirEdgeMap> {
    let out_dir = project_root.join("target").join("mir-edges");
    // No pre-deletion: RUSTC_WRAPPER now truncates on lib build and appends on test.
    // Crates that cargo skips (already built) keep their existing edge files intact.
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create MIR edge dir: {}", out_dir.display()))?;

    let bin = find_mir_callgraph_bin(mir_callgraph_bin)?;

    let mut cmd = Command::new(&bin);
    cmd.current_dir(project_root)
        .arg("--keep-going")
        .env("MIR_CALLGRAPH_OUT", &out_dir)
        .env("MIR_CALLGRAPH_JSON", "1");

    for krate in crates {
        cmd.arg("-p").arg(krate);
    }

    let status = cmd.status()
        .with_context(|| format!("failed to run mir-callgraph: {}", bin.display()))?;

    if !status.success() {
        eprintln!("  [mir] mir-callgraph exited with {status} (partial results may be available)");
    }

    // Run language-specific extractors (Python, TypeScript)
    run_language_extractors(project_root, &out_dir, rust_only);

    // Ensure rustc-args nightly version is saved so next run isn't stale.
    let args_dir = out_dir.join("rustc-args");
    if args_dir.exists() {
        let ver_file = args_dir.join(".nightly-version");
        if !ver_file.exists() {
            if let Some(ver) = nightly_rustc_version() {
                let _ = std::fs::write(&ver_file, &ver);
            }
        }
    }

    MirEdgeMap::from_dir(&out_dir)
}

/// Run mir-callgraph in direct mode — always.
///
/// If prerequisites are missing (no cache, stale cache, missing artifacts),
/// runs cargo once to satisfy them, then executes direct mode.
/// Cargo is never used for MIR extraction itself, only for preparation.
/// When `rust_only` is true, Python/TypeScript extractors are skipped entirely,
/// saving ~0.3s of directory-walk overhead.
pub fn run_mir_direct(
    project_root: &Path,
    mir_callgraph_bin: Option<&Path>,
    crates: &[&str],
    rust_only: bool,
) -> Result<MirEdgeMap> {
    let out_dir = project_root.join("target").join("mir-edges");
    let args_dir = out_dir.join("rustc-args");

    // ── Phase 1: Ensure args cache is fresh ──────────────────────────
    let needs_cache_refresh = !args_dir.exists() || is_args_cache_stale(project_root, &args_dir);
    let needs_deps_rebuild = needs_cache_refresh || !all_extern_paths_valid(crates, &args_dir);

    if needs_deps_rebuild {
        if needs_cache_refresh {
            eprintln!("  [mir] refreshing rustc-args cache via cargo...");
        } else {
            eprintln!("  [mir] rebuilding deps (stale --extern artifacts)...");
        }
        // Run cargo with RUSTC_WRAPPER to (re)generate args cache + build deps.
        // This is preparation only — MIR extraction still happens via direct.
        run_mir_callgraph_for(project_root, mir_callgraph_bin, crates, rust_only)?;

        // After cargo run, args cache is fresh. Continue to direct mode below
        // to ensure consistent extraction path.
    }

    // ── Phase 2: Collect args files ──────────────────────────────────
    // Collect both lib and test args files per crate for parallel execution.
    let mut args_files = Vec::new();
    for krate in crates {
        let crate_underscore = krate.replace('-', "_");
        let lib_file = args_dir.join(format!("{crate_underscore}.lib.rustc-args.json"));
        let test_file = args_dir.join(format!("{crate_underscore}.test.rustc-args.json"));
        if lib_file.exists() { args_files.push(lib_file); }
        if test_file.exists() { args_files.push(test_file); }
    }

    // If cargo just ran (Phase 1) and we still have no args, there's nothing
    // more we can do — the crate might not be a local crate.
    if args_files.is_empty() {
        return MirEdgeMap::from_dir(&out_dir);
    }

    // If deps were just rebuilt via cargo (Phase 1), the edge files are already
    // generated by RUSTC_WRAPPER. Skip redundant direct extraction.
    if needs_deps_rebuild {
        run_language_extractors(project_root, &out_dir, rust_only);
        return MirEdgeMap::from_dir(&out_dir);
    }

    // ── Phase 3: Direct mode — lib and test in parallel per crate ──────
    let bin = find_mir_callgraph_bin(mir_callgraph_bin)?;

    // Pre-truncate edge/chunk files for crates we're about to rebuild.
    // This allows RUSTC_WRAPPER to use append mode for both lib and test.
    for krate in crates {
        let u = krate.replace('-', "_");
        for ext in [".edges.jsonl", ".chunks.jsonl"] {
            let p = out_dir.join(format!("{u}{ext}"));
            if p.exists() { let _ = std::fs::write(&p, ""); }
        }
    }

    let mut children: Vec<(std::path::PathBuf, std::process::Child)> = Vec::new();
    let mut had_error = false;

    // Launch all builds (lib + test) in parallel
    for args_file in &args_files {
        let mut cmd = Command::new(&bin);
        cmd.current_dir(project_root)
            .arg("--direct")
            .arg("--args-file").arg(args_file)
            .env("MIR_CALLGRAPH_OUT", &out_dir)
            .env("MIR_CALLGRAPH_JSON", "1");
        match cmd.spawn() {
            Ok(child) => children.push((args_file.clone(), child)),
            Err(e) => {
                eprintln!("  [mir] failed to spawn direct: {e}");
                had_error = true;
            }
        }
    }

    for (path, mut child) in children {
        if let Ok(status) = child.wait() {
            if !status.success() {
                eprintln!("  [mir] direct failed for {}: {status}", path.display());
                had_error = true;
            }
        }
    }

    if had_error {
        eprintln!("  [mir] some direct builds failed, refreshing via cargo...");
        run_mir_callgraph_for(project_root, mir_callgraph_bin, crates, rust_only)?;
    }

    run_language_extractors(project_root, &out_dir, rust_only);
    MirEdgeMap::from_dir(&out_dir)
}

/// Check if all requested crates have valid --extern artifact paths.
fn all_extern_paths_valid(crates: &[&str], args_dir: &Path) -> bool {
    for krate in crates {
        let crate_underscore = krate.replace('-', "_");
        for suffix in [".lib", ".test"] {
            let f = args_dir.join(format!("{crate_underscore}{suffix}.rustc-args.json"));
            if f.exists() && !validate_extern_paths(&f) {
                return false;
            }
        }
    }
    true
}

// Edge file cleanup removed: RUSTC_WRAPPER now truncates on lib build.
// Crates skipped by cargo keep their existing (valid) edge files.

/// Check if the rustc-args cache is stale.
///
/// Stale conditions:
/// 1. Cargo.toml or Cargo.lock modified after the cache directory
/// 2. Nightly rustc version changed
/// Check if --extern artifact paths in a cached args file still exist.
/// If any .rlib/.rmeta/.dll/.so is missing, direct mode will fail.
fn validate_extern_paths(args_file: &Path) -> bool {
    let content = match std::fs::read_to_string(args_file) {
        Ok(c) => c,
        Err(_) => return false,
    };
    // Parse JSON to get the args array with proper string unescaping.
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
    let args = match parsed {
        Ok(v) => v.get("args")
            .and_then(|a| a.as_array())
            .cloned()
            .unwrap_or_default(),
        Err(_) => return false,
    };
    for arg in &args {
        let Some(s) = arg.as_str() else { continue };
        let is_artifact = s.ends_with(".rlib") || s.ends_with(".rmeta")
            || s.ends_with(".dll") || s.ends_with(".so") || s.ends_with(".dylib");
        if !is_artifact { continue; }
        // --extern name=path or just a path
        let path_str = s.rsplit('=').next().unwrap_or(s);
        if !std::path::Path::new(path_str).exists() {
            return false;
        }
    }
    true
}

fn is_args_cache_stale(project_root: &Path, args_dir: &Path) -> bool {
    // Get the oldest cache file mtime as reference
    let cache_mtime = match args_dir_oldest_mtime(args_dir) {
        Some(t) => t,
        None => return true, // no cache files
    };

    // Check Cargo.toml
    let cargo_toml = project_root.join("Cargo.toml");
    if let Ok(meta) = std::fs::metadata(&cargo_toml) {
        if let Ok(mtime) = meta.modified() {
            if mtime > cache_mtime {
                return true;
            }
        }
    }

    // Check Cargo.lock (dependency changes, feature changes via lock update)
    let cargo_lock = project_root.join("Cargo.lock");
    if let Ok(meta) = std::fs::metadata(&cargo_lock) {
        if let Ok(mtime) = meta.modified() {
            if mtime > cache_mtime {
                return true;
            }
        }
    }

    // Check all workspace member Cargo.toml files (feature flag changes, dep edits).
    // For single-crate projects, parse_workspace_members returns empty — the root
    // Cargo.toml is already checked above.
    if let Ok(root_content) = std::fs::read_to_string(&cargo_toml) {
        for member_dir in parse_workspace_members(&root_content) {
            let member_toml = project_root.join(&member_dir).join("Cargo.toml");
            if let Ok(meta) = std::fs::metadata(&member_toml) {
                if let Ok(mtime) = meta.modified() {
                    if mtime > cache_mtime {
                        return true;
                    }
                }
            }
        }
    }

    // Check .cargo/config.toml (target changes, rustflags, etc.)
    for config_name in [".cargo/config.toml", ".cargo/config"] {
        let config_path = project_root.join(config_name);
        if let Ok(meta) = std::fs::metadata(&config_path) {
            if let Ok(mtime) = meta.modified() {
                if mtime > cache_mtime {
                    return true;
                }
            }
        }
    }

    // Check nightly version
    let version_file = args_dir.join(".nightly-version");
    if let Some(current_ver) = nightly_rustc_version() {
        match std::fs::read_to_string(&version_file) {
            Ok(saved) if saved.trim() == current_ver => {}
            _ => {
                // Save current version for next check
                let _ = std::fs::write(&version_file, &current_ver);
                return true;
            }
        }
    }

    false
}

/// Get the oldest modification time among `.rustc-args.json` files in a directory.
fn args_dir_oldest_mtime(dir: &Path) -> Option<std::time::SystemTime> {
    let mut oldest: Option<std::time::SystemTime> = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.to_string_lossy().ends_with(".rustc-args.json") {
                if let Ok(meta) = std::fs::metadata(&path) {
                    if let Ok(mtime) = meta.modified() {
                        oldest = Some(match oldest {
                            Some(prev) if prev < mtime => prev,
                            _ => mtime,
                        });
                    }
                }
            }
        }
    }
    oldest
}

/// Run Python and TypeScript call graph extractors if available.
///
/// Skips each extractor when no source files of that language exist,
/// or when existing edge files are newer than all source files (mtime-based).
///
/// When `rust_only` is true, skips entirely — avoids the ~0.3s directory walk
/// cost when only Rust files have changed.
fn run_language_extractors(project_root: &Path, out_dir: &Path, rust_only: bool) {
    if rust_only {
        return;
    }
    // Python extractor — only if .py sources are newer than cached edges
    if has_newer_sources(project_root, out_dir, &["py"], "py_callgraph") {
        let py_script = find_extractor_script("py-callgraph", "py_callgraph.py");
        if let Some(script) = py_script {
            let status = Command::new("python3")
                .arg(&script)
                .arg(project_root)
                .arg("--out-dir").arg(out_dir)
                .status();
            if let Err(e) = status {
                // python3 not available, try python
                let _ = Command::new("python")
                    .arg(&script)
                    .arg(project_root)
                    .arg("--out-dir").arg(out_dir)
                    .status()
                    .map_err(|_| eprintln!("  [py-callgraph] python not available: {e}"));
            }
        }
    }

    // TypeScript/JavaScript extractor — only if .ts/.js/.tsx/.jsx sources are newer
    if has_newer_sources(project_root, out_dir, &["ts", "js", "tsx", "jsx"], "ts_callgraph") {
        let ts_script = find_extractor_script("ts-callgraph", "ts_callgraph.js");
        if let Some(script) = ts_script {
            let _ = Command::new("node")
                .arg(&script)
                .arg(project_root)
                .arg("--out-dir").arg(out_dir)
                .status()
                .map_err(|e| eprintln!("  [ts-callgraph] node not available: {e}"));
        }
    }
}

/// Check if any source files with the given extensions are newer than the
/// corresponding edge file in `out_dir`.
///
/// Returns `true` (= should run extractor) when:
/// - The edge file does not exist, OR
/// - Any source file has an mtime newer than the edge file.
///
/// Returns `false` (= skip) when:
/// - No source files with the given extensions exist in the project, OR
/// - All source files are older than the existing edge file.
fn has_newer_sources(
    project_root: &Path,
    out_dir: &Path,
    extensions: &[&str],
    edge_stem: &str,
) -> bool {
    let edge_file = out_dir.join(format!("{edge_stem}.edges.jsonl"));
    let edge_mtime = match std::fs::metadata(&edge_file)
        .ok()
        .and_then(|m| m.modified().ok())
    {
        Some(t) => t,
        None => return true, // no cached edges → must run
    };

    // Walk the project tree looking for source files with matching extensions.
    // Skip common non-source directories for speed.
    let mut stack = vec![project_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                // Skip directories that never contain user source
                if matches!(
                    name.as_ref(),
                    "target" | "node_modules" | ".git" | "__pycache__" | ".venv" | "venv"
                ) {
                    continue;
                }
                stack.push(path);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.iter().any(|&e| e.eq_ignore_ascii_case(ext)) {
                    if let Ok(m) = std::fs::metadata(&path) {
                        if let Ok(mtime) = m.modified() {
                            if mtime > edge_mtime {
                                return true; // newer source found → must run
                            }
                        }
                    }
                }
            }
        }
    }

    // No source files found, or all are older than the edge file → skip.
    false
}

/// Find a language extractor script next to the current executable or in tools/.
fn find_extractor_script(tool_dir: &str, script_name: &str) -> Option<std::path::PathBuf> {
    // 1. Next to current exe
    if let Ok(exe) = std::env::current_exe() {
        let sibling = exe.parent()?.join(script_name);
        if sibling.exists() { return Some(sibling); }
    }
    // 2. In tools/ relative to project root (for development)
    let candidates = [
        std::path::PathBuf::from(format!("tools/{tool_dir}/{script_name}")),
        std::env::current_dir().ok()?.join(format!("tools/{tool_dir}/{script_name}")),
    ];
    for c in &candidates {
        if c.exists() { return Some(c.clone()); }
    }
    None
}

/// Detect which crates contain the given changed files.
///
/// Walks up from each file to find the nearest Cargo.toml, then extracts the package name.
pub fn detect_changed_crates(project_root: &Path, changed_files: &[impl AsRef<Path>]) -> Vec<String> {
    let mut crates = std::collections::HashSet::new();
    for file in changed_files {
        let file = file.as_ref();
        let abs = if file.is_absolute() {
            file.to_path_buf()
        } else {
            project_root.join(file)
        };
        // Walk up to find Cargo.toml
        let mut dir = abs.parent();
        while let Some(d) = dir {
            let cargo_toml = d.join("Cargo.toml");
            if cargo_toml.exists() {
                // Extract package name from Cargo.toml
                if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                    for line in content.lines() {
                        let trimmed = line.trim();
                        if trimmed.starts_with("name") {
                            if let Some(name) = trimmed.split('"').nth(1) {
                                crates.insert(name.to_owned());
                            }
                            break;
                        }
                    }
                }
                break;
            }
            dir = d.parent();
        }
    }
    crates.into_iter().collect()
}

/// A chunk definition extracted from MIR — function/struct/enum with location info.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MirChunk {
    pub name: String,
    pub file: String,
    pub kind: String,
    pub start_line: usize,
    pub end_line: usize,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(default)]
    pub visibility: Option<String>,
    #[serde(default)]
    pub is_test: bool,
}

/// Load MIR chunks from a JSONL file.
pub fn load_mir_chunks(path: &Path) -> Result<Vec<MirChunk>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open MIR chunks: {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut chunks = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: MirChunk = serde_json::from_str(&line)
            .with_context(|| format!("failed to parse MIR chunk: {line}"))?;
        chunks.push(chunk);
    }
    Ok(chunks)
}

/// Load all chunks from `.chunks.jsonl` files in a directory.
pub fn load_all_mir_chunks(dir: &Path) -> Result<Vec<MirChunk>> {
    let mut all = Vec::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read MIR chunks dir: {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.to_string_lossy().ends_with(".chunks.jsonl") {
            all.extend(load_mir_chunks(&path)?);
        }
    }
    Ok(all)
}

/// Detect workspace crates whose `.edges.jsonl` files are missing from the MIR output dir.
///
/// Reads the root `Cargo.toml` `[workspace] members` list, extracts each member's
/// package name, then checks if `target/mir-edges/{crate_name}.edges.jsonl` exists.
/// Returns the names of crates with missing edge files.
pub fn detect_missing_edge_crates(project_root: &Path) -> Vec<String> {
    let edge_dir = project_root.join("target").join("mir-edges");
    let workspace_toml = project_root.join("Cargo.toml");

    let content = match std::fs::read_to_string(&workspace_toml) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    // Parse workspace members; for single-crate (non-workspace) projects,
    // treat the project root as the sole member.
    let member_dirs = parse_workspace_members(&content);
    let is_single_crate = member_dirs.is_empty();

    let check_list: Vec<(std::path::PathBuf, std::path::PathBuf)> = if is_single_crate {
        // Single crate: root Cargo.toml + root src/
        vec![(workspace_toml.clone(), project_root.join("src"))]
    } else {
        member_dirs.iter()
            .map(|d| (project_root.join(d).join("Cargo.toml"), project_root.join(d).join("src")))
            .collect()
    };

    let mut missing = Vec::new();
    for (toml_path, src_dir) in &check_list {
        let pkg_name = match std::fs::read_to_string(toml_path) {
            Ok(c) => extract_package_name(&c),
            Err(_) => continue,
        };
        let Some(name) = pkg_name else { continue };

        if !src_dir.exists() {
            continue;
        }

        let edge_file = edge_dir.join(format!("{}.edges.jsonl", name.replace('-', "_")));
        if !edge_file.exists() {
            missing.push(name);
        }
    }
    missing
}

/// Parse `[workspace] members = [...]` from a Cargo.toml string.
fn parse_workspace_members(content: &str) -> Vec<String> {
    let mut members = Vec::new();
    let mut in_members = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("members") && trimmed.contains('[') {
            in_members = true;
            // Handle inline items on the same line as `members = [`
            for part in trimmed.split('[').nth(1).into_iter() {
                for item in part.split(']').next().into_iter() {
                    for entry in item.split(',') {
                        let entry = entry.trim().trim_matches('"').trim();
                        if !entry.is_empty() {
                            members.push(entry.to_owned());
                        }
                    }
                }
            }
            if trimmed.contains(']') {
                in_members = false;
            }
            continue;
        }
        if in_members {
            if trimmed.contains(']') {
                // Last line of the array
                let before_bracket = trimmed.split(']').next().unwrap_or("");
                let entry = before_bracket.trim().trim_matches(',').trim().trim_matches('"').trim();
                if !entry.is_empty() {
                    members.push(entry.to_owned());
                }
                in_members = false;
                continue;
            }
            // Strip comments
            let no_comment = trimmed.split('#').next().unwrap_or(trimmed);
            let entry = no_comment.trim().trim_matches(',').trim().trim_matches('"').trim();
            if !entry.is_empty() {
                members.push(entry.to_owned());
            }
        }
    }
    members
}

/// Extract `name = "..."` from the `[package]` section of a Cargo.toml.
fn extract_package_name(content: &str) -> Option<String> {
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_package = trimmed == "[package]";
            continue;
        }
        if in_package && trimmed.starts_with("name") {
            return trimmed.split('"').nth(1).map(|s| s.to_owned());
        }
    }
    None
}

/// Normalize a file path for consistent lookup.
fn normalize_path(path: &str) -> String {
    path.replace('\\', "/").to_lowercase()
}
