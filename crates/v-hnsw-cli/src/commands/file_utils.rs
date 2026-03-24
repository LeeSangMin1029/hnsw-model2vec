//! File path, hashing, and scanning utilities.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

// Re-export core UNC utilities for backward compatibility.
pub use v_hnsw_core::{strip_unc_prefix, strip_unc_prefix_path};

/// Normalize a file path to a canonical forward-slash form.
///
/// Resolves symlinks, strips Windows `\\?\` prefix, normalizes separators.
pub fn normalize_source(path: &Path) -> String {
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let s = abs.to_string_lossy();
    v_hnsw_core::strip_unc_prefix(&s).replace('\\', "/")
}

/// Lightweight path normalization without filesystem syscalls.
/// Strips UNC prefix and normalizes slashes only. No canonicalize.
pub fn normalize_source_light(path: &Path) -> String {
    let s = path.to_string_lossy();
    v_hnsw_core::strip_unc_prefix(&s).replace('\\', "/")
}

/// Generate a stable ID from source path and chunk index.
pub fn generate_id(source: &str, chunk_index: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    chunk_index.hash(&mut hasher);
    hasher.finish()
}

/// Compute a content hash (MD5 → u64) for a file's raw bytes.
///
/// Used for change detection: if mtime/size changed but content hash
/// is identical, we skip expensive re-embedding.
pub fn content_hash(path: &Path) -> Result<u64> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read file for hashing: {}", path.display()))?;
    Ok(content_hash_bytes(&bytes))
}

/// Compute content hash from raw bytes (MD5 truncated to u64).
pub fn content_hash_bytes(bytes: &[u8]) -> u64 {
    let digest = md5::compute(bytes);
    #[expect(clippy::unwrap_used, reason = "MD5 digest is always 16 bytes")]
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

/// Built-in directory names always skipped during file scanning.
const BUILTIN_SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    ".git",
    ".swarm",
    "__pycache__",
    ".venv",
    "dist",
    "vendor",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".claude",
    "build",
    "mutants.out",
];

/// Check if a directory entry should be skipped during walkdir scanning.
///
/// Skips built-in cache/build directories and any user-specified `--exclude` dirs.
pub fn should_skip_dir(dir_name: &OsStr, exclude: &[String]) -> bool {
    let name = dir_name.to_string_lossy();
    if BUILTIN_SKIP_DIRS.iter().any(|s| *s == name.as_ref()) {
        return true;
    }
    // Skip v-hnsw database directories (e.g., .v-hnsw-code.db, .v-hnsw-sessions.db)
    if name.starts_with(".v-hnsw") {
        return true;
    }
    exclude.iter().any(|e| e == name.as_ref())
}

/// Recursively scan `input` for files whose extension passes `ext_filter`,
/// skipping built-in + user-specified directories.
pub fn scan_files(
    input: &Path,
    exclude: &[String],
    ext_filter: impl Fn(&str) -> bool,
) -> Vec<PathBuf> {
    walkdir::WalkDir::new(input)
        .into_iter()
        .filter_entry(|e| {
            if e.file_type().is_dir() {
                !should_skip_dir(e.file_name(), exclude)
            } else {
                true
            }
        })
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(&ext_filter)
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

/// Get the file modification time as seconds since UNIX epoch.
pub fn get_file_mtime(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
}
