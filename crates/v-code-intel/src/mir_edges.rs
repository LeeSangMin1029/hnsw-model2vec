//! MIR-based call edge extraction via `mir-callgraph` subprocess.
//!
//! Parses JSONL output from the `mir-callgraph` tool and provides
//! resolved call edges for accurate graph construction.

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};

/// A single call edge extracted from MIR.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MirCallEdge {
    pub caller: String,
    pub caller_file: String,
    pub callee: String,
    pub line: usize,
    #[serde(default)]
    pub is_local: bool,
}

/// Collection of MIR edges indexed for fast lookup by caller.
#[derive(Debug, Default)]
pub struct MirEdgeMap {
    /// (caller_file_normalized, line) → Vec<callee_name>
    pub by_location: HashMap<(String, usize), Vec<String>>,
    /// caller_name → Vec<(callee_name, line)>
    pub by_caller: HashMap<String, Vec<(String, usize)>>,
    /// Total edge count
    pub total: usize,
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

            map.by_caller
                .entry(edge.caller.clone())
                .or_default()
                .push((edge.callee, edge.line));

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
            if path.to_string_lossy().ends_with(".edges.jsonl") {
                let partial = Self::from_jsonl(&path)?;
                for (k, v) in partial.by_location {
                    combined.by_location.entry(k).or_default().extend(v);
                }
                for (k, v) in partial.by_caller {
                    combined.by_caller.entry(k).or_default().extend(v);
                }
                combined.total += partial.total;
            }
        }

        Ok(combined)
    }

    /// Resolve a call from a specific file and line to its callee names.
    pub fn resolve_at(&self, file: &str, line: usize) -> Option<&[String]> {
        let key = (normalize_path(file), line);
        self.by_location.get(&key).map(|v| v.as_slice())
    }

    /// Get all callees for a given caller function name.
    pub fn callees_of(&self, caller: &str) -> Option<&[(String, usize)]> {
        self.by_caller.get(caller).map(|v| v.as_slice())
    }
}

/// Find the mir-callgraph binary.
fn find_mir_callgraph_bin(override_path: Option<&Path>) -> std::path::PathBuf {
    override_path
        .map(|p| p.to_path_buf())
        .or_else(|| {
            std::env::current_exe().ok()
                .and_then(|exe| {
                    let sibling = exe.with_file_name(if cfg!(windows) { "mir-callgraph.exe" } else { "mir-callgraph" });
                    sibling.exists().then_some(sibling)
                })
        })
        .unwrap_or_else(|| std::path::PathBuf::from("mir-callgraph"))
}

/// Run mir-callgraph on the entire workspace.
pub fn run_mir_callgraph(project_root: &Path, mir_callgraph_bin: Option<&Path>) -> Result<MirEdgeMap> {
    run_mir_callgraph_for(project_root, mir_callgraph_bin, &[])
}

/// Run mir-callgraph for specific crates only (or all if `crates` is empty).
pub fn run_mir_callgraph_for(
    project_root: &Path,
    mir_callgraph_bin: Option<&Path>,
    crates: &[&str],
) -> Result<MirEdgeMap> {
    let out_dir = project_root.join("target").join("mir-edges");
    // Clear existing output files before run (append mode in mir-callgraph)
    if out_dir.exists() {
        for entry in std::fs::read_dir(&out_dir).into_iter().flatten().flatten() {
            let p = entry.path();
            let name = p.to_string_lossy();
            if name.ends_with(".edges.jsonl") || name.ends_with(".chunks.jsonl") {
                // Only clear files for crates we're about to re-analyze
                let should_clear = crates.is_empty() || crates.iter().any(|c| {
                    let crate_underscore = c.replace('-', "_");
                    name.contains(&crate_underscore)
                });
                if should_clear {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
    }
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create MIR edge dir: {}", out_dir.display()))?;

    let bin = find_mir_callgraph_bin(mir_callgraph_bin);

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
    run_language_extractors(project_root, &out_dir);

    MirEdgeMap::from_dir(&out_dir)
}

/// Run Python and TypeScript call graph extractors if available.
fn run_language_extractors(project_root: &Path, out_dir: &Path) {
    // Python extractor
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

    // TypeScript/JavaScript extractor
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

/// Normalize a file path for consistent lookup.
fn normalize_path(path: &str) -> String {
    path.replace('\\', "/").to_lowercase()
}
