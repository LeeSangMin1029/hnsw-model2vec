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

    /// Load all .jsonl files from a directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let mut combined = Self::default();
        let entries = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read MIR edge dir: {}", dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
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

/// Run mir-callgraph tool and return edge map.
///
/// `project_root`: path to the Cargo project
/// `mir_callgraph_bin`: path to the mir-callgraph binary (if None, look in PATH)
pub fn run_mir_callgraph(project_root: &Path, mir_callgraph_bin: Option<&Path>) -> Result<MirEdgeMap> {
    let out_dir = project_root.join("target").join("mir-edges");
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create MIR edge dir: {}", out_dir.display()))?;

    let bin = mir_callgraph_bin
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("mir-callgraph"));

    let status = Command::new(&bin)
        .current_dir(project_root)
        .env("MIR_CALLGRAPH_OUT", &out_dir)
        .env("MIR_CALLGRAPH_JSON", "1")
        .status()
        .with_context(|| format!("failed to run mir-callgraph: {}", bin.display()))?;

    if !status.success() {
        anyhow::bail!("mir-callgraph exited with {status}");
    }

    MirEdgeMap::from_dir(&out_dir)
}

/// Normalize a file path for consistent lookup.
fn normalize_path(path: &str) -> String {
    path.replace('\\', "/").to_lowercase()
}
