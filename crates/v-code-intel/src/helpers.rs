//! Shared helper functions for code intelligence.
//!
//! Format utilities, path normalization, and JSON grouping helpers
//! used across multiple modules.

use std::collections::BTreeMap;

use crate::parse::CodeChunk;

/// Format an optional line range as `"start-end"` or empty string.
pub fn format_lines_str_opt(lines: Option<(usize, usize)>) -> String {
    if let Some((s, e)) = lines {
        format!("{s}-{e}")
    } else {
        String::new()
    }
}

/// Format an optional line range as `":start-end"` or empty string.
pub fn format_lines_opt(lines: Option<(usize, usize)>) -> String {
    let s = format_lines_str_opt(lines);
    if s.is_empty() { s } else { format!(":{s}") }
}

/// Strip common absolute prefixes to produce a relative path.
///
/// Looks for `crates/` as the project-relative anchor. Falls back to the
/// original path when no anchor is found.
pub fn relative_path(path: &str) -> &str {
    if let Some(idx) = path.find("crates/") {
        &path[idx..]
    } else if let Some(idx) = path.find("src/") {
        &path[idx..]
    } else {
        path
    }
}

/// Build a multi-base alias map from a set of paths.
///
/// Groups paths by their directory prefix, assigns short aliases (`[A]`, `[B]`, …)
/// to directories containing 2+ files, and returns (alias_map, legend).
///
/// ```text
/// [A] = crates/v-hnsw-cli/src/commands/
/// [B] = crates/v-hnsw-storage/src/
/// ```
pub fn build_path_aliases(paths: &[&str]) -> (std::collections::BTreeMap<String, String>, Vec<(String, String)>) {
    use std::collections::BTreeMap;

    // Count files per directory.
    let mut dir_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for p in paths {
        let dir = match p.rfind('/') {
            Some(i) => &p[..=i],
            None => "",
        };
        *dir_counts.entry(dir).or_default() += 1;
    }

    // Assign aliases to directories with 1+ files (all dirs get aliases for consistency).
    let mut alias_map: BTreeMap<String, String> = BTreeMap::new();
    let mut legend: Vec<(String, String)> = Vec::new();
    let mut label = b'A';

    for (&dir, &_count) in &dir_counts {
        if dir.is_empty() {
            continue;
        }
        let alias = format!("[{}]", label as char);
        alias_map.insert(dir.to_owned(), alias.clone());
        legend.push((alias, dir.to_owned()));
        label += 1;
        if label > b'Z' {
            break;
        }
    }

    (alias_map, legend)
}

/// Shorten a path using the alias map: replace the directory with its alias.
pub fn apply_alias(path: &str, alias_map: &std::collections::BTreeMap<String, String>) -> String {
    // Find the longest matching directory prefix.
    let dir = match path.rfind('/') {
        Some(i) => &path[..=i],
        None => return path.to_owned(),
    };
    if let Some(alias) = alias_map.get(dir) {
        let file = &path[dir.len()..];
        format!("{alias}{file}")
    } else {
        path.to_owned()
    }
}

/// Format line range as `"start-end"` or empty string.
pub fn lines_str(c: &CodeChunk) -> String {
    if let Some((start, end)) = c.lines {
        format!("{start}-{end}")
    } else {
        String::new()
    }
}

/// Extract crate name from file path: `crates/foo-bar/src/...` -> `foo-bar`.
pub fn extract_crate_name(path: &str) -> String {
    if let Some(start) = path.find("crates/") {
        let rest = &path[start + 7..];
        if let Some(slash) = rest.find('/') {
            return rest[..slash].to_owned();
        }
    }
    "(root)".to_owned()
}

/// Schema descriptor included in every JSON output.
const SCHEMA: &str = "f=file,l=lines,k=kind,n=name,v=via";

/// Build file-grouped JSON from chunks, applying `extra_fields` to each entry.
///
/// The closure receives a chunk and returns additional key-value pairs to merge.
fn build_grouped_json_with<'a, F>(chunks: impl Iterator<Item = &'a CodeChunk>, extra_fields: F) -> serde_json::Value
where
    F: Fn(&CodeChunk) -> Vec<(&'static str, serde_json::Value)>,
{
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::Value::String(SCHEMA.to_owned()));

    let mut groups: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    for c in chunks {
        let path = relative_path(&c.file);
        let mut obj = serde_json::json!({
            "l": lines_str(c),
            "k": &c.kind,
            "n": &c.name,
        });
        for (key, val) in extra_fields(c) {
            obj[key] = val;
        }
        groups.entry(path).or_default().push(obj);
    }
    for (path, items) in groups {
        map.insert(path.to_owned(), serde_json::Value::Array(items));
    }
    serde_json::Value::Object(map)
}

/// Build file-grouped JSON with `_s` schema header.
///
/// Output: `{"_s":"...","crates/foo/src/bar.rs":[{"l":"1-10","k":"fn","n":"run"}]}`
pub fn grouped_json(chunks: &[&CodeChunk]) -> serde_json::Value {
    build_grouped_json_with(chunks.iter().copied(), |_| Vec::new())
}

/// Build file-grouped JSON for refs (includes `v` field).
pub fn grouped_json_refs(refs: &[(&CodeChunk, Vec<&str>)]) -> serde_json::Value {
    // Build a lookup map from chunk pointer to via list.
    let via_map: std::collections::HashMap<*const CodeChunk, &Vec<&str>> = refs
        .iter()
        .map(|(c, v)| (*c as *const CodeChunk, v))
        .collect();
    build_grouped_json_with(refs.iter().map(|(c, _)| *c), |c| {
        if let Some(via) = via_map.get(&(c as *const CodeChunk)) {
            vec![("v", serde_json::json!(via))]
        } else {
            Vec::new()
        }
    })
}
