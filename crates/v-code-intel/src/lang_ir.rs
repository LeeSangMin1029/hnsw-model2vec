//! Language-specific IR/AST call extraction for non-Rust languages.
//!
//! Each language provides a `collect_*_calls()` function that returns
//! a `MirCallMap` (caller→callees map) compatible with
//! `CallGraph::build_with_resolved_calls`.
//!
//! Currently supported:
//! - **Python**: uses `python3` subprocess with `ast` module

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};

use crate::mir::MirCallMap;

// ── Python ──────────────────────────────────────────────────────────

/// Path to the bundled Python call-extraction script (relative to crate root).
const PYTHON_SCRIPT: &str = include_str!("../scripts/python_calls.py");

/// Collect Python call edges by running `python3` with the bundled AST script.
///
/// Returns a `MirCallMap` where keys are `module::Class::method` or
/// `module::func` and values are lists of callee references.
///
/// Returns `Ok(empty)` if `python3` is not available or the project has no `.py` files.
pub fn collect_python_calls(project_root: &Path) -> Result<MirCallMap> {
    // Check that python3 is available.
    let python = find_python();
    let Some(python_cmd) = python else {
        return Ok(BTreeMap::new());
    };

    // Write the script to a temp file.
    let script_path = std::env::temp_dir().join("v_code_python_calls.py");
    std::fs::write(&script_path, PYTHON_SCRIPT)
        .context("failed to write Python call extraction script")?;

    let output = Command::new(&python_cmd)
        .arg(&script_path)
        .arg(project_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .context("failed to run python3 for call extraction")?;

    if !output.status.success() {
        // Non-fatal: return empty map.
        return Ok(BTreeMap::new());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_call_json(&stdout)
}

/// Parse the JSON output from `python_calls.py` into a `MirCallMap`.
///
/// Expected format: `{"caller": ["callee1", "callee2"], ...}`
fn parse_call_json(json_str: &str) -> Result<MirCallMap> {
    let trimmed = json_str.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return Ok(BTreeMap::new());
    }

    // Minimal JSON parser — avoids serde_json dependency.
    // Format: {"key": ["val1", "val2"], ...}
    let mut map = BTreeMap::new();
    let inner = trimmed
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(trimmed);

    // Split by top-level entries: "key": [...]
    let mut chars = inner.chars().peekable();
    let mut key = String::new();
    let mut values: Vec<String> = Vec::new();

    #[derive(PartialEq)]
    enum State {
        SeekKey,
        InKey,
        SeekColon,
        SeekArray,
        InArray,
        InValue,
    }

    let mut state = State::SeekKey;
    let mut current_val = String::new();

    while let Some(&ch) = chars.peek() {
        chars.next();
        match state {
            State::SeekKey => {
                if ch == '"' {
                    key.clear();
                    state = State::InKey;
                }
            }
            State::InKey => {
                if ch == '"' {
                    state = State::SeekColon;
                } else {
                    key.push(ch);
                }
            }
            State::SeekColon => {
                if ch == ':' {
                    state = State::SeekArray;
                }
            }
            State::SeekArray => {
                if ch == '[' {
                    values.clear();
                    state = State::InArray;
                }
            }
            State::InArray => {
                if ch == '"' {
                    current_val.clear();
                    state = State::InValue;
                } else if ch == ']' {
                    if !key.is_empty() && !values.is_empty() {
                        map.insert(key.clone(), values.clone());
                    }
                    state = State::SeekKey;
                }
            }
            State::InValue => {
                if ch == '"' {
                    values.push(current_val.clone());
                    state = State::InArray;
                } else if ch == '\\' {
                    // Handle escaped chars.
                    if let Some(&next) = chars.peek() {
                        chars.next();
                        current_val.push(next);
                    }
                } else {
                    current_val.push(ch);
                }
            }
        }
    }

    Ok(map)
}

// ── Go (stub) ───────────────────────────────────────────────────────

/// Placeholder for Go call extraction.
///
/// Future: use `go vet` analysis or `go/packages` + `go/callgraph`.
pub fn collect_go_calls(_project_root: &Path) -> Result<MirCallMap> {
    Ok(BTreeMap::new())
}

// ── TypeScript (stub) ───────────────────────────────────────────────

/// Placeholder for TypeScript/JavaScript call extraction.
///
/// Future: use `tsc --declaration` or a custom TS script with the compiler API.
pub fn collect_ts_calls(_project_root: &Path) -> Result<MirCallMap> {
    Ok(BTreeMap::new())
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Find a working Python 3 command.
fn find_python() -> Option<String> {
    for cmd in &["python3", "python"] {
        if let Ok(output) = Command::new(cmd)
            .arg("--version")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
        {
            let ver = String::from_utf8_lossy(&output.stdout);
            let ver_err = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{ver}{ver_err}");
            if combined.contains("Python 3") {
                return Some((*cmd).to_owned());
            }
        }
    }
    None
}

// ── Project detection ───────────────────────────────────────────────

/// Detect if a directory contains Python source files worth analyzing.
pub fn has_python_files(root: &Path) -> bool {
    has_file_with_extension(root, "py", 3)
}

/// Detect if a directory has a `go.mod` (Go project).
pub fn has_go_mod(root: &Path) -> bool {
    root.join("go.mod").exists()
}

/// Detect if a directory has a `package.json` (Node/TS project).
pub fn has_package_json(root: &Path) -> bool {
    root.join("package.json").exists()
        || root.join("tsconfig.json").exists()
}

/// Check if a directory tree contains at least `min_count` files with the given extension.
fn has_file_with_extension(root: &Path, ext: &str, min_count: usize) -> bool {
    let mut count = 0;
    let mut stack = vec![root.to_path_buf()];
    let skip_dirs = [
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".tox", ".mypy_cache", "dist", "build",
    ];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let ft = entry.file_type();
            let Ok(ft) = ft else { continue };
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if ft.is_dir() {
                if !skip_dirs.contains(&name_str.as_ref()) {
                    stack.push(entry.path());
                }
            } else if name_str.ends_with(&format!(".{ext}")) {
                count += 1;
                if count >= min_count {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_json() {
        let map = parse_call_json("{}").unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn parse_simple_json() {
        let input = r#"{"main::run": ["main::Processor", "utils::helper"], "utils::helper": ["utils::transform"]}"#;
        let map = parse_call_json(input).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get("main::run").unwrap(),
            &vec!["main::Processor".to_owned(), "utils::helper".to_owned()]
        );
        assert_eq!(
            map.get("utils::helper").unwrap(),
            &vec!["utils::transform".to_owned()]
        );
    }

    #[test]
    fn parse_escaped_json() {
        let input = r#"{"mod::func": ["mod::other\"quoted"]}"#;
        let map = parse_call_json(input).unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get("mod::func").unwrap(),
            &vec!["mod::other\"quoted".to_owned()]
        );
    }
}
