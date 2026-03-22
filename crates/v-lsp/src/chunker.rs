//! RA-based code chunker — replaces tree-sitter for chunk extraction.
//!
//! Uses `file_structure` for symbol discovery, `outgoing_calls` for call edges,
//! `discover_tests_in_file` for test detection, and source text for signatures.

use std::collections::{HashMap, HashSet};

use ra_ap_ide::{FileId, StructureNodeKind, FileStructureConfig, SymbolKind, TextSize};

use crate::instance::RaInstance;

/// A code chunk extracted via RA — equivalent to `v-code-chunk::CodeChunk` + `v-code-intel::ParsedChunk`.
/// Code chunk extracted via RA. Serialized as JSON over daemon RPC.
///
/// Split into identity + source + relations for clarity,
/// but kept as one struct for serde simplicity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RaChunk {
    // ── identity ──
    pub kind: String,
    pub name: String,
    pub file: String,
    pub lines: Option<(usize, usize)>,
    pub visibility: String,
    pub is_test: bool,
    pub chunk_index: usize,
    // ── source ──
    pub text: String,
    pub signature: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
    // ── relations (call graph) ──
    pub calls: Vec<String>,
    pub call_lines: Vec<u32>,
    pub types: Vec<String>,
    pub imports: Vec<String>,
    pub param_types: Vec<(String, String)>,
    pub field_types: Vec<(String, String)>,
    pub return_type: Option<String>,
    pub enum_variants: Vec<String>,
}

impl RaInstance {
    /// Extract all code chunks from project files using RA APIs.
    ///
    /// This replaces tree-sitter chunking entirely:
    /// - `file_structure` → symbol names, kinds, ranges
    /// - `outgoing_calls` → call edges with line numbers
    /// - `discover_tests_in_file` → test function detection
    /// - source text → signatures, imports, visibility
    pub fn chunk_files(&self, files: &[String]) -> Vec<RaChunk> {
        let mut all_chunks = Vec::new();
        let mut test_ranges: HashMap<FileId, HashSet<u32>> = HashMap::new();

        // Pre-collect test functions per file via runnables (safer than discover_tests).
        for file_key in files {
            if let Some(fi) = self.file_map().get(file_key) {
                if let Ok(runnables) = self.analysis().runnables(fi.file_id) {
                    let file_len = TextSize::from(fi.source.len() as u32);
                    let ranges: HashSet<u32> = runnables.iter()
                        .filter(|r| format!("{:?}", r.kind).contains("Test"))
                        .filter_map(|r| {
                            let start = r.nav.focus_range.unwrap_or(r.nav.full_range).start();
                            if start < file_len {
                                Some(fi.line_index.line_col(start).line)
                            } else { None }
                        })
                        .collect();
                    if !ranges.is_empty() {
                        test_ranges.insert(fi.file_id, ranges);
                    }
                }
            }
        }

        for file_key in files {
            let Some(fi) = self.file_map().get(file_key) else { continue };
            let source = &fi.source;
            let lines: Vec<&str> = source.lines().collect();

            // Get file structure (symbols).
            let config = FileStructureConfig { exclude_locals: true };
            let Ok(structure) = self.analysis().file_structure(&config, fi.file_id) else { continue };

            // Extract imports from source.
            let imports = extract_imports(source);

            let file_tests = test_ranges.get(&fi.file_id);
            let mut chunk_index = 0;

            for node in &structure {
                let (kind, include) = match node.kind {
                    StructureNodeKind::SymbolKind(SymbolKind::Function) => ("function", true),
                    StructureNodeKind::SymbolKind(SymbolKind::Method) => ("function", true),
                    StructureNodeKind::SymbolKind(SymbolKind::Struct) => ("struct", true),
                    StructureNodeKind::SymbolKind(SymbolKind::Enum) => ("enum", true),
                    StructureNodeKind::SymbolKind(SymbolKind::Trait) => ("trait", true),
                    StructureNodeKind::SymbolKind(SymbolKind::Impl) => ("impl", true),
                    _ => ("", false),
                };
                if !include { continue; }

                let file_len = TextSize::from(source.len() as u32);
                if node.node_range.end() > file_len {
                    eprintln!("[chunker] skip {}: range {:?} > file_len {} in {file_key}",
                        node.label, node.node_range, source.len());
                    continue;
                }
                let start_line_0 = fi.line_index.line_col(node.node_range.start()).line as usize;
                let end_line_0 = fi.line_index.line_col(node.node_range.end()).line as usize;

                // Skip empty chunks (0 lines).
                if end_line_0 < start_line_0 { continue; }

                // Build full name with parent (impl block).
                let name = if let Some(parent_idx) = node.parent {
                    let parent = &structure[parent_idx];
                    format!("{}::{}", parent.label, node.label)
                } else {
                    node.label.clone()
                };

                // Extract text.
                let text = lines[start_line_0..=end_line_0.min(lines.len() - 1)].join("\n");

                // Visibility from first line.
                let first_line = lines.get(start_line_0).unwrap_or(&"").trim();
                let visibility = if first_line.starts_with("pub(crate)") {
                    "pub(crate)".to_owned()
                } else if first_line.starts_with("pub ") || first_line.starts_with("pub fn") {
                    "pub".to_owned()
                } else {
                    String::new()
                };

                // Signature from detail.
                let signature = node.detail.clone();

                // Is test?
                let is_test = file_tests.map_or(false, |tests| tests.contains(&(start_line_0 as u32)));

                // For functions: get outgoing calls via RA using navigation_range.
                let (calls, call_lines) = if kind == "function" {
                    let nav_line = fi.line_index.line_col(node.navigation_range.start()).line;
                    let nav_col = fi.line_index.line_col(node.navigation_range.start()).col;
                    self.extract_calls_at(file_key, nav_line, nav_col)
                } else {
                    (Vec::new(), Vec::new())
                };

                // Return type from signature.
                let return_type = signature.as_ref().and_then(|s| extract_return_type(s));

                // Param types from signature.
                let param_types = if kind == "function" {
                    signature.as_ref().map(|s| extract_param_types(s)).unwrap_or_default()
                } else {
                    Vec::new()
                };

                // Enum variants.
                let enum_variants = if kind == "enum" {
                    extract_enum_variants_from_text(&text)
                } else {
                    Vec::new()
                };

                // Field types for structs.
                let field_types = if kind == "struct" {
                    extract_field_types_from_text(&text)
                } else {
                    Vec::new()
                };

                // Type refs from text (simplified: uppercase identifiers after :: or in type positions).
                let types = extract_type_refs(&text);

                all_chunks.push(RaChunk {
                    kind: kind.to_owned(),
                    name,
                    file: file_key.clone(),
                    lines: Some((start_line_0 + 1, end_line_0 + 1)),
                    visibility,
                    is_test,
                    chunk_index: { let idx = chunk_index; chunk_index += 1; idx },
                    text,
                    signature,
                    start_byte: node.node_range.start().into(),
                    end_byte: node.node_range.end().into(),
                    calls,
                    call_lines,
                    types,
                    imports: imports.clone(),
                    param_types,
                    field_types,
                    return_type,
                    enum_variants,
                });
            }
        }

        all_chunks
    }

    /// Extract calls from a function using `outgoing_calls` at exact position.
    fn extract_calls_at(&self, file: &str, line: u32, col: u32) -> (Vec<String>, Vec<u32>) {
        match self.outgoing_calls(file, line, col) {
            Ok(callees) => {
                let mut calls = Vec::new();
                let mut lines = Vec::new();
                for callee in &callees {
                    let name = if callee.file.is_empty() {
                        callee.name.clone()
                    } else {
                        callee.name.clone()
                    };
                    for &line in &callee.call_site_lines {
                        calls.push(name.clone());
                        lines.push(line);
                    }
                }
                (calls, lines)
            }
            Err(_) => (Vec::new(), Vec::new()),
        }
    }
}

// ── Source text helpers ──────────────────────────────────────────────

/// Extract `use` statements from source text.
fn extract_imports(source: &str) -> Vec<String> {
    source.lines()
        .filter(|l| {
            let trimmed = l.trim();
            trimmed.starts_with("use ") && trimmed.ends_with(';')
        })
        .map(|l| l.trim().to_owned())
        .collect()
}

/// Extract return type from signature string.
/// `"fn foo(x: i32) -> Result<String>"` → `"Result<String>"`
fn extract_return_type(sig: &str) -> Option<String> {
    let arrow = sig.find("->")?;
    let ret = sig[arrow + 2..].trim();
    if ret.is_empty() { None } else { Some(ret.to_owned()) }
}

/// Extract parameter types from signature.
/// `"fn foo(x: i32, y: &str)"` → `[("x", "i32"), ("y", "&str")]`
fn extract_param_types(sig: &str) -> Vec<(String, String)> {
    let Some(open) = sig.find('(') else { return Vec::new() };
    let Some(close) = sig.rfind(')') else { return Vec::new() };
    let params_str = &sig[open + 1..close];
    let mut result = Vec::new();
    let mut depth = 0u32;
    let mut start = 0;
    for (i, ch) in params_str.char_indices() {
        match ch {
            '<' | '(' => depth += 1,
            '>' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                if let Some(pair) = parse_param(&params_str[start..i]) {
                    result.push(pair);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    if let Some(pair) = parse_param(&params_str[start..]) {
        result.push(pair);
    }
    return result;

    fn parse_param(s: &str) -> Option<(String, String)> {
        let s = s.trim();
        if s.is_empty() || s == "self" || s == "&self" || s == "&mut self" {
            return None;
        }
        let colon = s.find(':')?;
        let name = s[..colon].trim().to_owned();
        let ty = s[colon + 1..].trim().to_owned();
        Some((name, ty))
    }
}

/// Extract enum variant names from enum text.
fn extract_enum_variants_from_text(text: &str) -> Vec<String> {
    let mut variants = Vec::new();
    let mut in_body = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains('{') { in_body = true; continue; }
        if trimmed.starts_with('}') { break; }
        if !in_body { continue; }
        // "VariantName," or "VariantName(..." or "VariantName {"
        let name = trimmed.split(|c: char| !c.is_alphanumeric() && c != '_')
            .next().unwrap_or("");
        if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_uppercase()) {
            variants.push(name.to_lowercase());
        }
    }
    variants
}

/// Extract struct field types from text.
fn extract_field_types_from_text(text: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    let mut in_body = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains('{') { in_body = true; continue; }
        if trimmed.starts_with('}') { break; }
        if !in_body { continue; }
        // "field_name: Type,"
        let trimmed = trimmed.trim_start_matches("pub ");
        let trimmed = trimmed.trim_start_matches("pub(crate) ");
        if let Some(colon) = trimmed.find(':') {
            let name = trimmed[..colon].trim();
            let ty = trimmed[colon + 1..].trim().trim_end_matches(',');
            if !name.is_empty() && !ty.is_empty() {
                fields.push((name.to_owned(), ty.to_owned()));
            }
        }
    }
    fields
}

/// Extract type references from source text (uppercase identifiers).
fn extract_type_refs(text: &str) -> Vec<String> {
    let mut types = HashSet::new();
    for word in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.len() >= 2 && word.chars().next().map_or(false, |c| c.is_uppercase()) {
            // Skip common Rust keywords/types.
            if !matches!(word, "Self" | "Some" | "None" | "Ok" | "Err" | "true" | "false"
                | "String" | "Vec" | "Box" | "Option" | "Result" | "Arc" | "Rc"
                | "HashMap" | "HashSet" | "BTreeMap" | "BTreeSet" | "Path" | "PathBuf") {
                types.insert(word.to_owned());
            }
        }
    }
    types.into_iter().collect()
}
