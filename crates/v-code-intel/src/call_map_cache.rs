//! Persistent, file-level CallMap cache for incremental graph rebuilds.
//!
//! Stores LSP-resolved caller→callee edges per source file alongside
//! file metadata (mtime, size) for staleness detection.
//! On graph rebuild, only changed files need LSP re-resolution.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::lsp::CallMap;

/// Cache format version — bump when struct layout changes.
const CACHE_VERSION: u8 = 1;

/// Per-file metadata for change detection.
#[derive(bincode::Encode, bincode::Decode, Clone)]
struct FileMeta {
    mtime_secs: u64,
    size: u64,
}

/// Persistent CallMap cache stored at `<db>/cache/call_map.bin`.
#[derive(bincode::Encode, bincode::Decode)]
pub struct CallMapCache {
    version: u8,
    /// Source file → metadata at time of LSP resolution.
    file_meta: HashMap<String, FileMeta>,
    /// Source file → resolved call entries [(caller_name, [callee_names])].
    per_file: HashMap<String, Vec<(String, Vec<String>)>>,
}

impl CallMapCache {
    /// Load cache from `<db>/cache/call_map.bin`.
    pub fn load(db: &Path) -> Option<Self> {
        let path = cache_path(db);
        let bytes = fs::read(&path).ok()?;
        let config = bincode::config::standard();
        let (cache, _) = bincode::decode_from_slice::<Self, _>(&bytes, config).ok()?;
        if cache.version != CACHE_VERSION {
            return None;
        }
        Some(cache)
    }

    /// Save cache to `<db>/cache/call_map.bin`.
    pub fn save(&self, db: &Path) {
        let path = cache_path(db);
        let _ = fs::create_dir_all(path.parent().unwrap_or(Path::new(".")));
        let config = bincode::config::standard();
        if let Ok(bytes) = bincode::encode_to_vec(self, config) {
            let _ = fs::write(&path, bytes);
        }
    }

    /// Determine which files have changed since last resolution.
    ///
    /// Returns the set of file paths (from chunk data) that need LSP re-resolution.
    pub fn changed_files<'a>(
        &self,
        chunks: &'a [crate::parse::CodeChunk],
        project_root: &Path,
    ) -> HashSet<&'a str> {
        let mut changed = HashSet::new();
        let mut seen = HashSet::new();

        for chunk in chunks {
            if !seen.insert(chunk.file.as_str()) {
                continue;
            }

            let fpath = project_root.join(&chunk.file);
            let current_meta = file_meta(&fpath);

            match (self.file_meta.get(&chunk.file), current_meta) {
                (Some(cached), Some(current))
                    if cached.mtime_secs == current.mtime_secs
                        && cached.size == current.size => {}
                _ => {
                    changed.insert(chunk.file.as_str());
                }
            }
        }

        // Also mark files that were in cache but no longer in chunks (deleted)
        for cached_file in self.file_meta.keys() {
            if !seen.contains(cached_file.as_str()) {
                // File was removed — its entries will be dropped naturally
            }
        }

        changed
    }

    /// Get cached CallMap entries for unchanged files.
    pub fn cached_call_map(&self, changed_files: &HashSet<&str>) -> CallMap {
        let mut map = BTreeMap::new();
        for (file, entries) in &self.per_file {
            if changed_files.contains(file.as_str()) {
                continue;
            }
            for (caller, callees) in entries {
                map.entry(caller.clone())
                    .or_insert_with(Vec::new)
                    .extend(callees.iter().cloned());
            }
        }
        map
    }

    /// Build a new cache from chunks + resolved CallMap + project root.
    ///
    /// `old_cache` provides cached entries for unchanged files.
    /// `new_calls` provides freshly resolved entries for changed files.
    /// `changed_files` indicates which files were re-resolved.
    pub fn build(
        chunks: &[crate::parse::CodeChunk],
        old_cache: Option<&Self>,
        new_calls: &CallMap,
        changed_files: &HashSet<&str>,
        project_root: &Path,
    ) -> Self {
        let mut file_meta_map: HashMap<String, FileMeta> = HashMap::new();
        let mut per_file: HashMap<String, Vec<(String, Vec<String>)>> = HashMap::new();

        // Group chunks by file
        let mut chunks_by_file: HashMap<&str, Vec<&crate::parse::CodeChunk>> = HashMap::new();
        for chunk in chunks {
            chunks_by_file
                .entry(chunk.file.as_str())
                .or_default()
                .push(chunk);
        }

        for (file, _file_chunks) in &chunks_by_file {
            let fpath = project_root.join(file);

            // Update file metadata
            if let Some(meta) = file_meta(&fpath) {
                file_meta_map.insert(file.to_string(), meta);
            }

            if changed_files.contains(file) {
                // Use freshly resolved entries for changed files
                let mut entries = Vec::new();
                for chunk in _file_chunks {
                    let caller = chunk.name.to_lowercase();
                    if let Some(callees) = new_calls.get(&caller) {
                        entries.push((caller, callees.clone()));
                    }
                }
                per_file.insert(file.to_string(), entries);
            } else if let Some(old) = old_cache {
                // Reuse cached entries for unchanged files
                if let Some(old_entries) = old.per_file.get(*file) {
                    per_file.insert(file.to_string(), old_entries.clone());
                }
                if let Some(old_meta) = old.file_meta.get(*file) {
                    file_meta_map.insert(file.to_string(), old_meta.clone());
                }
            }
        }

        Self {
            version: CACHE_VERSION,
            file_meta: file_meta_map,
            per_file,
        }
    }

    /// Merge cached and new CallMap entries into a single CallMap.
    pub fn to_call_map(&self) -> CallMap {
        let mut map = BTreeMap::new();
        for entries in self.per_file.values() {
            for (caller, callees) in entries {
                let entry = map.entry(caller.clone()).or_insert_with(Vec::new);
                entry.extend(callees.iter().cloned());
                entry.sort();
                entry.dedup();
            }
        }
        map
    }
}

fn cache_path(db: &Path) -> std::path::PathBuf {
    db.join("cache").join("call_map.bin")
}

fn file_meta(path: &Path) -> Option<FileMeta> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(FileMeta {
        mtime_secs: mtime,
        size: meta.len(),
    })
}
