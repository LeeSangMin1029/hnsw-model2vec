//! Code intelligence commands — structural queries on code-chunked databases.
//!
//! Provides `symbols`, `def`, `callers`, and `refs` subcommands that parse
//! the structured text field of code chunks (produced by `chunk_code`) and
//! answer structural navigation queries.
//!
//! These commands are read-only and do not modify the database.

pub mod context;
pub mod deps;
mod deps_html;
pub mod detail;
pub mod gather;
pub mod graph;
pub mod impact;
pub(crate) mod parse;
pub mod reason;
pub mod trace;

use std::collections::BTreeMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::PayloadStore;
use v_hnsw_storage::StorageEngine;

use parse::CodeChunk;

// ── Load all code chunks (with cache) ────────────────────────────────────

fn load_chunks_from_db(path: &Path) -> Result<Vec<CodeChunk>> {
    let engine = StorageEngine::open(path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let vector_store = engine.vector_store();
    let payload_store = engine.payload_store();
    let ids: Vec<u64> = vector_store.id_map().keys().copied().collect();

    let mut chunks = Vec::new();
    for id in ids {
        if let Ok(Some(text)) = payload_store.get_text(id)
            && let Some(chunk) = parse::parse_chunk(&text)
        {
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

fn cache_path(db: &Path) -> PathBuf {
    db.join("cache").join("chunks.bin")
}

pub(crate) fn load_chunks(path: &Path) -> Result<Vec<CodeChunk>> {
    let cache = cache_path(path);
    let db_mtime = fs::metadata(path)
        .and_then(|m| m.modified())
        .ok();

    // Try cache hit
    if let Some(db_t) = db_mtime {
        if let Ok(cache_meta) = fs::metadata(&cache) {
            if let Ok(cache_t) = cache_meta.modified() {
                if cache_t >= db_t {
                    if let Ok(bytes) = fs::read(&cache) {
                        let config = bincode::config::standard();
                        if let Ok((chunks, _)) =
                            bincode::decode_from_slice::<Vec<CodeChunk>, _>(&bytes, config)
                        {
                            return Ok(chunks);
                        }
                    }
                }
            }
        }
    }

    // Cache miss — load from DB and save
    let chunks = load_chunks_from_db(path)?;
    let config = bincode::config::standard();
    if let Ok(bytes) = bincode::encode_to_vec(&chunks, config) {
        let _ = fs::write(&cache, bytes);
    }
    Ok(chunks)
}

// ── Query result cache ───────────────────────────────────────────────────

fn query_cache_dir(db: &Path) -> PathBuf {
    db.join("cache")
}

/// Print cached JSON if DB unchanged, otherwise compute and cache.
pub(super) fn cached_json(db: &Path, cache_key: &str, compute: impl FnOnce() -> Result<String>) -> Result<()> {
    let cache_dir = query_cache_dir(db);
    let mut hasher = std::hash::DefaultHasher::new();
    cache_key.hash(&mut hasher);
    let hash = hasher.finish();
    let cache_file = cache_dir.join(format!("{hash:x}.json"));

    let db_mtime = fs::metadata(db).and_then(|m| m.modified()).ok();
    if let Some(db_t) = db_mtime {
        if let Ok(meta) = fs::metadata(&cache_file) {
            if let Ok(cache_t) = meta.modified() {
                if cache_t >= db_t {
                    if let Ok(content) = fs::read_to_string(&cache_file) {
                        println!("{content}");
                        return Ok(());
                    }
                }
            }
        }
    }

    let output = compute()?;
    let _ = fs::create_dir_all(&cache_dir);
    let _ = fs::write(&cache_file, &output);
    println!("{output}");
    Ok(())
}

// ── Output format ────────────────────────────────────────────────────────

/// Output format for code-intel commands.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    /// Standalone HTML with interactive D3.js force-directed graph.
    Html,
}

// ── Commands ─────────────────────────────────────────────────────────────

/// `v-hnsw stats` — per-crate summary of code symbols.
pub fn run_stats(db: PathBuf, format: OutputFormat) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        return cached_json(&db, "stats", || compute_stats_json(&db));
    }
    let chunks = load_chunks(&db)?;
    let stats = build_stats(&chunks);
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8}",
        "crate", "prod_fn", "test_fn", "struct", "enum"
    );
    println!("{}", "-".repeat(60));
    let mut totals = [0usize; 4];
    for (name, row) in &stats {
        println!(
            "{:<24} {:>8} {:>8} {:>8} {:>8}",
            name, row[0], row[1], row[2], row[3]
        );
        for (i, v) in row.iter().enumerate() {
            totals[i] += v;
        }
    }
    println!("{}", "-".repeat(60));
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>8}",
        "total", totals[0], totals[1], totals[2], totals[3]
    );
    Ok(())
}

fn build_stats(chunks: &[CodeChunk]) -> BTreeMap<String, [usize; 4]> {
    let mut stats: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    for c in chunks {
        let crate_name = extract_crate_name(&c.file);
        let row = stats.entry(crate_name).or_insert([0; 4]);
        let is_test = c.file.contains("/tests/") || c.name.starts_with("test_");
        match (c.kind.as_str(), is_test) {
            ("function", false) => row[0] += 1,
            ("function", true) => row[1] += 1,
            ("struct", _) => row[2] += 1,
            ("enum", _) => row[3] += 1,
            _ => {}
        }
    }
    stats
}

fn compute_stats_json(db: &Path) -> Result<String> {
    let chunks = load_chunks(db)?;
    let stats = build_stats(&chunks);
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::Value::String(STATS_SCHEMA.to_owned()));
    for (name, row) in &stats {
        map.insert(name.clone(), serde_json::json!({"p":row[0],"t":row[1],"s":row[2],"e":row[3]}));
    }
    Ok(serde_json::to_string(&serde_json::Value::Object(map))?)
}

/// `v-hnsw symbols` — list symbols matching filters.
pub fn run_symbols(
    db: PathBuf,
    name: Option<String>,
    kind: Option<String>,
    format: OutputFormat,
) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("symbols:{}:{}", name.as_deref().unwrap_or(""), kind.as_deref().unwrap_or(""));
        return cached_json(&db, &key, || {
            let chunks = load_chunks(&db)?;
            let filtered: Vec<&CodeChunk> = chunks.iter().filter(|c| {
                if let Some(ref n) = name {
                    if !c.name.to_lowercase().contains(&n.to_lowercase()) { return false; }
                }
                if let Some(ref k) = kind {
                    if c.kind.to_lowercase() != k.to_lowercase() { return false; }
                }
                true
            }).collect();
            Ok(serde_json::to_string(&grouped_json(&filtered))?)
        });
    }
    let chunks = load_chunks(&db)?;
    let filtered: Vec<&CodeChunk> = chunks.iter().filter(|c| {
        if let Some(ref n) = name {
            if !c.name.to_lowercase().contains(&n.to_lowercase()) { return false; }
        }
        if let Some(ref k) = kind {
            if c.kind.to_lowercase() != k.to_lowercase() { return false; }
        }
        true
    }).collect();
    if filtered.is_empty() {
        println!("No symbols found.");
    } else {
        println!("{} symbols found:\n", filtered.len());
        print_grouped(&filtered, None);
    }
    Ok(())
}

/// `v-hnsw def` — find definition location of a symbol.
pub fn run_def(db: PathBuf, name: String, format: OutputFormat) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("def:{name}");
        return cached_json(&db, &key, || {
            let chunks = load_chunks(&db)?;
            let name_lower = name.to_lowercase();
            let matches: Vec<&CodeChunk> = chunks.iter().filter(|c| {
                c.name.to_lowercase() == name_lower
                    || c.name.to_lowercase().ends_with(&format!("::{name_lower}"))
            }).collect();
            Ok(serde_json::to_string(&grouped_json(&matches))?)
        });
    }
    let chunks = load_chunks(&db)?;
    let name_lower = name.to_lowercase();
    let matches: Vec<&CodeChunk> = chunks.iter().filter(|c| {
        c.name.to_lowercase() == name_lower
            || c.name.to_lowercase().ends_with(&format!("::{name_lower}"))
    }).collect();
    if matches.is_empty() {
        println!("No definition found for \"{name}\".");
    } else {
        println!("Definition of \"{name}\":\n");
        print_grouped(&matches, None);
    }
    Ok(())
}

/// `v-hnsw callers` — find all callers of a function.
pub fn run_callers(db: PathBuf, function: String, format: OutputFormat) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("callers:{function}");
        return cached_json(&db, &key, || {
            let chunks = load_chunks(&db)?;
            let fn_lower = function.to_lowercase();
            let callers: Vec<&CodeChunk> = chunks.iter().filter(|c| {
                c.calls.iter().any(|call| {
                    let call_lower = call.to_lowercase();
                    call_lower == fn_lower
                        || call_lower.ends_with(&format!("::{fn_lower}"))
                        || call_lower.contains(&fn_lower)
                })
            }).collect();
            Ok(serde_json::to_string(&grouped_json(&callers))?)
        });
    }
    let chunks = load_chunks(&db)?;
    let fn_lower = function.to_lowercase();
    let callers: Vec<&CodeChunk> = chunks.iter().filter(|c| {
        c.calls.iter().any(|call| {
            let call_lower = call.to_lowercase();
            call_lower == fn_lower
                || call_lower.ends_with(&format!("::{fn_lower}"))
                || call_lower.contains(&fn_lower)
        })
    }).collect();
    if callers.is_empty() {
        println!("No callers found for \"{function}\".");
    } else {
        println!("{} callers of \"{function}\":\n", callers.len());
        print_grouped(&callers, None);
    }
    Ok(())
}

/// `v-hnsw refs` — find all references to a symbol.
pub fn run_refs(db: PathBuf, name: String, format: OutputFormat) -> Result<()> {
    if matches!(format, OutputFormat::Json) {
        let key = format!("refs:{name}");
        return cached_json(&db, &key, || {
            let chunks = load_chunks(&db)?;
            let refs = find_refs(&chunks, &name);
            Ok(serde_json::to_string(&grouped_json_refs(&refs))?)
        });
    }
    let chunks = load_chunks(&db)?;
    let refs = find_refs(&chunks, &name);
    if refs.is_empty() {
        println!("No references found for \"{name}\".");
        return Ok(());
    }
    let mut groups: BTreeMap<String, Vec<(&CodeChunk, &[&str])>> = BTreeMap::new();
    for (chunk, via) in &refs {
        let dir = parent_dir(&chunk.file);
        groups.entry(dir).or_default().push((chunk, via));
    }
    println!("{} references to \"{name}\":\n", refs.len());
    for (dir, items) in &groups {
        println!("  {dir}/");
        for (c, via) in items {
            let filename = file_name(&c.file);
            let lines = format_lines(c);
            let via_str = via.join(", ");
            println!("    {filename}{lines}  [{kind}] {name} (via {via_str})",
                kind = c.kind, name = c.name);
        }
        println!();
    }
    Ok(())
}

fn find_refs<'a>(chunks: &'a [CodeChunk], name: &str) -> Vec<(&'a CodeChunk, Vec<&'static str>)> {
    let name_lower = name.to_lowercase();
    chunks
        .iter()
        .filter_map(|c| {
            let mut via = Vec::new();
            if c.calls.iter().any(|s| s.to_lowercase().contains(&name_lower)) {
                via.push("calls");
            }
            if c.types.iter().any(|s| s.to_lowercase().contains(&name_lower)) {
                via.push("types");
            }
            if c.signature.as_ref().is_some_and(|s| s.to_lowercase().contains(&name_lower)) {
                via.push("signature");
            }
            if c.name.to_lowercase().contains(&name_lower) {
                via.push("name");
            }
            if via.is_empty() { None } else { Some((c, via)) }
        })
        .collect()
}

// ── Grouped output ───────────────────────────────────────────────────────

/// Print chunks grouped by parent directory.
fn print_grouped(chunks: &[&CodeChunk], _label: Option<&str>) {
    let mut groups: BTreeMap<String, Vec<&CodeChunk>> = BTreeMap::new();
    for c in chunks {
        let dir = parent_dir(&c.file);
        groups.entry(dir).or_default().push(c);
    }

    for (dir, items) in &groups {
        println!("  {dir}/");
        for c in items {
            let filename = file_name(&c.file);
            let lines = format_lines(c);
            let sig = c.signature.as_deref().unwrap_or("");
            println!("    {filename}{lines}  [{kind}] {name}",
                kind = c.kind, name = c.name);
            if !sig.is_empty() {
                println!("      {sig}");
            }
        }
        println!();
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn parent_dir(path: &str) -> String {
    if let Some(idx) = path.rfind('/') {
        path[..idx].to_owned()
    } else {
        ".".to_owned()
    }
}

fn file_name(path: &str) -> &str {
    if let Some(idx) = path.rfind('/') {
        &path[idx + 1..]
    } else {
        path
    }
}

fn format_lines(c: &CodeChunk) -> String {
    if let Some((start, end)) = c.lines {
        format!(":{start}-{end}")
    } else {
        String::new()
    }
}

/// Strip common absolute prefixes to produce a relative path.
///
/// Looks for `crates/` as the project-relative anchor. Falls back to the
/// original path when no anchor is found.
fn relative_path(path: &str) -> &str {
    // Normalise backslashes for matching.
    if let Some(idx) = path.find("crates/") {
        &path[idx..]
    } else if let Some(idx) = path.find("src/") {
        &path[idx..]
    } else {
        path
    }
}

/// Format line range as `"start-end"` or empty string.
fn lines_str(c: &CodeChunk) -> String {
    if let Some((start, end)) = c.lines {
        format!("{start}-{end}")
    } else {
        String::new()
    }
}

/// Schema descriptor included in every JSON output.
const SCHEMA: &str = "f=file,l=lines,k=kind,n=name,v=via";
const STATS_SCHEMA: &str = "p=prod_fn,t=test_fn,s=struct,e=enum";

/// Build file-grouped JSON with `_s` schema header.
///
/// Output: `{"_s":"...","crates/foo/src/bar.rs":[{"l":"1-10","k":"fn","n":"run"}]}`
fn grouped_json(chunks: &[&CodeChunk]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::Value::String(SCHEMA.to_owned()));

    let mut groups: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    for c in chunks {
        let path = relative_path(&c.file);
        groups.entry(path).or_default().push(serde_json::json!({
            "l": lines_str(c),
            "k": &c.kind,
            "n": &c.name,
        }));
    }
    for (path, items) in groups {
        map.insert(path.to_owned(), serde_json::Value::Array(items));
    }
    serde_json::Value::Object(map)
}

/// Build file-grouped JSON for refs (includes `v` field).
fn grouped_json_refs(refs: &[(&CodeChunk, Vec<&str>)]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::Value::String(SCHEMA.to_owned()));

    let mut groups: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    for (c, via) in refs {
        let path = relative_path(&c.file);
        groups.entry(path).or_default().push(serde_json::json!({
            "l": lines_str(c),
            "k": &c.kind,
            "n": &c.name,
            "v": via,
        }));
    }
    for (path, items) in groups {
        map.insert(path.to_owned(), serde_json::Value::Array(items));
    }
    serde_json::Value::Object(map)
}

/// Extract crate name from file path: `crates/foo-bar/src/...` → `foo-bar`.
fn extract_crate_name(path: &str) -> String {
    if let Some(start) = path.find("crates/") {
        let rest = &path[start + 7..];
        if let Some(slash) = rest.find('/') {
            return rest[..slash].to_owned();
        }
    }
    "(root)".to_owned()
}
