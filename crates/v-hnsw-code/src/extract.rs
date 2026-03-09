//! Tree-sitter node extraction helpers.
//!
//! HOW TO EXTEND: Add new extraction functions here when adding support
//! for new code metadata (e.g., generic parameters, lifetime annotations).

use std::hash::{Hash, Hasher};

use super::{CodeChunk, CodeChunkConfig, CodeNodeKind};

// ---------------------------------------------------------------------------
// Common parse helper
// ---------------------------------------------------------------------------

/// Result of parsing source code with tree-sitter.
pub struct ParsedSource {
    /// The parsed syntax tree.
    pub tree: tree_sitter::Tree,
    /// File-level import statements (empty when import extraction is disabled).
    pub imports: Vec<String>,
}

/// Parse source code with the given tree-sitter language and extract imports.
///
/// Returns `None` if the language cannot be set or parsing fails, allowing
/// callers to return an empty `Vec` early.
pub fn parse_source(
    language: tree_sitter::Language,
    source: &str,
    extract_imports: bool,
    import_kinds: &[&str],
) -> Option<ParsedSource> {
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&language).is_err() {
        return None;
    }
    let tree = parser.parse(source, None)?;
    let root = tree.root_node();
    let src = source.as_bytes();
    let imports = if extract_imports {
        extract_imports_by_kind(&root, src, import_kinds)
    } else {
        Vec::new()
    };
    Some(ParsedSource { tree, imports })
}

// ---------------------------------------------------------------------------
// AST structural hash — for code clone detection
// ---------------------------------------------------------------------------

/// Compute a structural hash of a tree-sitter node subtree.
///
/// Identifier/literal *values* are ignored (normalized), so two functions
/// that differ only in variable names produce the same hash (Type-2 clones).
/// The hash captures: node kinds, child structure, and operator tokens.
pub fn ast_structure_hash(node: &tree_sitter::Node, src: &[u8]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    hash_node_recursive(node, src, &mut hasher);
    hasher.finish()
}

fn hash_node_recursive(node: &tree_sitter::Node, src: &[u8], h: &mut impl Hasher) {
    let kind = node.kind();
    kind.hash(h);

    // For operators and keywords, include the actual text (e.g., "+", "return").
    // For identifiers and literals, hash only the *kind* (already done above)
    // so that renamed variables / different constants still match.
    if !is_identifier_or_literal(kind) && node.child_count() == 0 {
        if let Ok(text) = node.utf8_text(src) {
            text.hash(h);
        }
    }

    // Hash child count to distinguish `f(a)` from `f(a, b)`
    node.child_count().hash(h);

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        hash_node_recursive(&child, src, h);
    }
}

/// Node kinds whose text values should be normalized (not hashed).
fn is_identifier_or_literal(kind: &str) -> bool {
    matches!(
        kind,
        "identifier"
            | "type_identifier"
            | "field_identifier"
            | "property_identifier"
            | "shorthand_property_identifier"
            | "string_literal"
            | "string"
            | "string_content"
            | "template_string"
            | "number"
            | "integer_literal"
            | "float_literal"
            | "boolean_literal"
            | "char_literal"
            | "comment"
            | "line_comment"
            | "block_comment"
            | "interpreted_string_literal"
            | "raw_string_literal"
            | "rune_literal"
    )
}

/// Hash the actual code body text with normalization.
///
/// Unlike `ast_structure_hash` (which ignores identifiers), this preserves
/// variable/function names but strips comments and normalizes whitespace.
/// This catches truly duplicated logic — `fn len()` and `fn dim()` will
/// differ because identifiers are preserved.
pub fn body_hash(text: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Skip empty lines and single-line comments
        if trimmed.is_empty()
            || trimmed.starts_with("//")
            || trimmed.starts_with('#')
            || trimmed.starts_with("///")
        {
            continue;
        }
        // Strip inline comments
        let code = trimmed
            .find("//")
            .map_or(trimmed, |pos| trimmed[..pos].trim_end());
        if !code.is_empty() {
            code.hash(&mut hasher);
        }
    }
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Token-based MinHash — for near-duplicate (Type-3) clone detection
// ---------------------------------------------------------------------------

/// Tokenize code body for clone detection.
///
/// Strips comments, normalizes number literals to `$N`, keeps identifiers
/// and keywords. Returns unigram tokens.
pub fn code_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut in_block_comment = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Block comment tracking
        if in_block_comment {
            if let Some(pos) = trimmed.find("*/") {
                let rest = &trimmed[pos + 2..];
                in_block_comment = false;
                tokenize_line(rest, &mut tokens);
            }
            continue;
        }
        if trimmed.starts_with("/*") {
            if !trimmed.contains("*/") {
                in_block_comment = true;
            }
            continue;
        }

        // Skip line comments
        if trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }

        // Strip inline comments
        let code = trimmed.find("//").map_or(trimmed, |pos| &trimmed[..pos]);
        tokenize_line(code, &mut tokens);
    }
    tokens
}

fn tokenize_line(code: &str, tokens: &mut Vec<String>) {
    for word in code.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.is_empty() {
            continue;
        }
        if word.chars().all(|c| c.is_ascii_digit()) {
            tokens.push("$N".to_owned());
        } else {
            tokens.push(word.to_owned());
        }
    }
}

/// Compute MinHash signature from tokens (unigrams + bigrams).
///
/// Returns `k` minimum hash values, one per hash function.
/// Jaccard similarity ≈ fraction of matching positions between two signatures.
pub fn minhash_signature(tokens: &[String], k: usize) -> Vec<u64> {
    // Build feature set: unigrams + bigrams for sequence sensitivity
    let n_features = tokens.len() + tokens.len().saturating_sub(1);
    let mut features = Vec::with_capacity(n_features);
    for t in tokens {
        features.push(t.as_str());
    }

    // Bigrams stored as owned strings
    let bigrams: Vec<String> = tokens
        .windows(2)
        .map(|w| format!("{}_{}", w[0], w[1]))
        .collect();
    for b in &bigrams {
        features.push(b.as_str());
    }

    (0..k)
        .map(|seed| {
            features
                .iter()
                .map(|feature| {
                    let mut hasher = std::hash::DefaultHasher::new();
                    (seed as u64).hash(&mut hasher);
                    feature.hash(&mut hasher);
                    hasher.finish()
                })
                .min()
                .unwrap_or(u64::MAX)
        })
        .collect()
}

/// Estimate Jaccard similarity from two MinHash signatures.
pub fn jaccard_from_minhash(a: &[u64], b: &[u64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matches as f64 / a.len() as f64
}

/// Encode MinHash signature as compact hex string.
pub fn minhash_to_hex(sig: &[u64]) -> String {
    let mut hex = String::with_capacity(sig.len() * 16);
    for h in sig {
        use std::fmt::Write;
        let _ = write!(hex, "{h:016x}");
    }
    hex
}

/// Decode MinHash signature from hex string.
pub fn minhash_from_hex(hex: &str) -> Option<Vec<u64>> {
    if hex.len() % 16 != 0 {
        return None;
    }
    let k = hex.len() / 16;
    let mut sig = Vec::with_capacity(k);
    for i in 0..k {
        let chunk = &hex[i * 16..(i + 1) * 16];
        sig.push(u64::from_str_radix(chunk, 16).ok()?);
    }
    Some(sig)
}

/// Number of MinHash functions to use.
pub const MINHASH_K: usize = 64;

/// Extract the symbol name from a node.
pub fn extract_name(node: &tree_sitter::Node, src: &[u8]) -> String {
    // Most items have a `name` field.
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node.utf8_text(src).unwrap_or_default().to_owned();
    }

    // impl blocks: `impl Trait for Type` or `impl Type`.
    if node.kind() == "impl_item" {
        let type_name = node
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or_default();

        if let Some(trait_node) = node.child_by_field_name("trait") {
            let trait_name = trait_node.utf8_text(src).unwrap_or_default();
            return format!("{trait_name} for {type_name}");
        }
        return type_name.to_owned();
    }

    String::new()
}

/// Extract visibility modifier (`pub`, `pub(crate)`, etc.).
pub fn extract_visibility(node: &tree_sitter::Node, src: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            return child.utf8_text(src).unwrap_or_default().to_owned();
        }
    }
    String::new()
}

/// Extract function signature (everything before the body block).
pub fn extract_function_signature(node: &tree_sitter::Node, src: &[u8]) -> String {
    if let Some(body) = node.child_by_field_name("body") {
        let sig_start = node.start_byte();
        let sig_end = body.start_byte();
        if sig_end > sig_start
            && let Ok(sig) = std::str::from_utf8(&src[sig_start..sig_end]) {
                return sig.trim().to_owned();
            }
    }
    node.utf8_text(src).unwrap_or_default().to_owned()
}

/// Extract all `use` declarations from the root node.
pub fn extract_imports(root: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    extract_imports_by_kind(root, src, &["use_declaration"])
}

/// Extract function call names from a node's subtree.
pub fn extract_calls(node: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    let mut calls = Vec::new();
    walk_for_calls(node, src, &mut calls);
    calls.sort();
    calls.dedup();
    calls
}

/// Recursively walk to find call nodes across languages.
///
/// Handles:
/// - `call_expression` (Rust, TypeScript, Go, C, C++)
/// - `call` (Python)
/// - `method_invocation` (Java)
pub fn walk_for_calls(node: &tree_sitter::Node, src: &[u8], calls: &mut Vec<String>) {
    match node.kind() {
        "call_expression" => {
            if let Some(func) = node.child_by_field_name("function")
                && let Ok(text) = func.utf8_text(src) {
                    calls.push(text.to_owned());
                }
        }
        "call" => {
            // Python: call node has a "function" field
            if let Some(func) = node.child_by_field_name("function")
                && let Ok(text) = func.utf8_text(src) {
                    calls.push(text.to_owned());
                }
        }
        "method_invocation" => {
            // Java: method_invocation has "name" field and optional "object" field
            if let Some(name_node) = node.child_by_field_name("name")
                && let Ok(name) = name_node.utf8_text(src) {
                    if let Some(obj) = node.child_by_field_name("object")
                        && let Ok(obj_text) = obj.utf8_text(src) {
                            calls.push(format!("{obj_text}.{name}"));
                        } else {
                            calls.push(name.to_owned());
                        }
                }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_calls(&child, src, calls);
    }
}

/// Extract doc comments (`///` or `//!`) preceding a node.
/// Collect doc-comment lines immediately before `target` under `root`.
///
/// `comment_kind` is the tree-sitter node kind for comments (e.g. `"line_comment"`,
/// `"comment"`). `prefixes` lists the prefix strings to strip (e.g. `&["///", "//!"]`).
/// Only comments that match at least one prefix are kept.
fn collect_doc_comment_lines(
    root: &tree_sitter::Node,
    target: &tree_sitter::Node,
    src: &[u8],
    comment_kind: &str,
    prefixes: &[&str],
) -> Option<String> {
    let target_start = target.start_position().row;
    let mut doc_lines: Vec<String> = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        if child.start_position().row >= target_start {
            break;
        }

        if child.kind() == comment_kind {
            if let Ok(text) = child.utf8_text(src) {
                let stripped = prefixes
                    .iter()
                    .find_map(|p| text.strip_prefix(p));
                if let Some(doc) = stripped {
                    doc_lines.push(doc.trim().to_owned());
                }
            }
        } else {
            // Non-comment node between previous comments and target — reset
            doc_lines.clear();
        }
    }

    if doc_lines.is_empty() {
        None
    } else {
        Some(doc_lines.join("\n"))
    }
}

pub fn extract_doc_comment_before(
    root: &tree_sitter::Node,
    target: &tree_sitter::Node,
    src: &[u8],
) -> Option<String> {
    collect_doc_comment_lines(root, target, src, "line_comment", &["///", "//!"])
}

/// Recursively collect all `type_identifier` node texts from a subtree.
///
/// Returns deduplicated, sorted type names referenced in the node.
pub fn extract_type_refs(node: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    let mut refs = Vec::new();
    walk_for_type_ids(node, src, &mut refs);
    refs.sort();
    refs.dedup();
    refs
}

/// Walk recursively to collect `type_identifier` texts.
fn walk_for_type_ids(node: &tree_sitter::Node, src: &[u8], refs: &mut Vec<String>) {
    if node.kind() == "type_identifier"
        && let Ok(text) = node.utf8_text(src) {
            refs.push(text.to_owned());
        }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_type_ids(&child, src, refs);
    }
}

/// Generic helper: extract parameter name-type pairs from a node's `parameters` field.
///
/// For each child whose kind is in `param_kinds`, calls `extractor` to produce
/// zero or more `(name, type)` pairs. This avoids duplicating the
/// "get parameters → filter by kind → collect" boilerplate across languages.
fn extract_params_generic(
    node: &tree_sitter::Node,
    src: &[u8],
    param_kinds: &[&str],
    extractor: impl Fn(&tree_sitter::Node, &[u8]) -> Vec<(String, String)>,
) -> Vec<(String, String)> {
    let Some(params) = node.child_by_field_name("parameters") else {
        return Vec::new();
    };

    let mut result = Vec::new();
    let mut cursor = params.walk();

    for child in params.children(&mut cursor) {
        if !param_kinds.contains(&child.kind()) {
            continue;
        }
        result.extend(extractor(&child, src));
    }

    result
}

/// Helper: extract a single `(name, type)` pair via field names on a parameter node.
///
/// Returns a one-element vec on success, empty vec otherwise.
fn field_name_type_pair(
    child: &tree_sitter::Node,
    src: &[u8],
    name_field: &str,
    type_field: &str,
) -> Vec<(String, String)> {
    let name = child
        .child_by_field_name(name_field)
        .and_then(|n| n.utf8_text(src).ok())
        .unwrap_or_default();
    let ty = child
        .child_by_field_name(type_field)
        .and_then(|n| n.utf8_text(src).ok())
        .unwrap_or_default();

    if !name.is_empty() && !ty.is_empty() {
        vec![(name.to_owned(), ty.to_owned())]
    } else {
        Vec::new()
    }
}

/// Extract parameter name-type pairs from a function's `parameters` field.
///
/// For each parameter, extracts the `pattern` (name) and `type` (type text).
/// Skips `self` parameters.
pub fn extract_param_types(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Vec<(String, String)> {
    extract_params_generic(node, src, &["parameter"], |child, s| {
        field_name_type_pair(child, s, "pattern", "type")
    })
}

/// Extract the return type string from a function's `return_type` field.
pub fn extract_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    let ret = node.child_by_field_name("return_type")?;

    // return_type is `-> Type`; get the type child (skip `->` token)
    let mut cursor = ret.walk();
    for child in ret.children(&mut cursor) {
        if child.kind() != "->" {
            return child.utf8_text(src).ok().map(|s| s.to_owned());
        }
    }

    // Fallback: use full text minus the `-> ` prefix
    ret.utf8_text(src)
        .ok()
        .map(|s| s.strip_prefix("-> ").unwrap_or(s).to_owned())
}

/// Extract the function name from a C/C++ function_definition node.
///
/// Navigates the declarator chain (function_declarator, pointer_declarator)
/// to find the identifier.
pub fn extract_c_func_name(node: &tree_sitter::Node, src: &[u8]) -> String {
    if let Some(declarator) = node.child_by_field_name("declarator") {
        return find_func_name_in_declarator(&declarator, src);
    }
    extract_name(node, src)
}

/// Recursively search a declarator chain for the function identifier.
fn find_func_name_in_declarator(node: &tree_sitter::Node, src: &[u8]) -> String {
    if node.kind() == "function_declarator" {
        if let Some(decl) = node.child_by_field_name("declarator") {
            return decl.utf8_text(src).unwrap_or_default().to_owned();
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "function_declarator" || child.kind() == "pointer_declarator" {
            return find_func_name_in_declarator(&child, src);
        }
        if child.kind() == "identifier" {
            return child.utf8_text(src).unwrap_or_default().to_owned();
        }
    }
    node.utf8_text(src).unwrap_or_default().to_owned()
}

/// Extract block-style doc comments (/** ... */) preceding a node.
/// Used by TypeScript, Java, C, C++.
pub fn extract_block_doc_comment_before(
    root: &tree_sitter::Node,
    target: &tree_sitter::Node,
    src: &[u8],
) -> Option<String> {
    let target_start = target.start_position().row;
    let mut last_comment: Option<String> = None;
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        if child.start_position().row >= target_start {
            break;
        }

        // Different tree-sitter grammars use different node kinds for comments:
        // - C/C++/TypeScript: "comment"
        // - Java: "block_comment" or "line_comment"
        if child.kind() == "comment" || child.kind() == "block_comment" {
            if let Ok(text) = child.utf8_text(src) {
                if text.starts_with("/**") {
                    let inner = text
                        .strip_prefix("/**")
                        .and_then(|s| s.strip_suffix("*/"))
                        .unwrap_or(text);
                    let cleaned: Vec<&str> = inner
                        .lines()
                        .map(|l| l.trim().trim_start_matches('*').trim())
                        .filter(|l| !l.is_empty())
                        .collect();
                    last_comment = Some(cleaned.join("\n"));
                } else {
                    last_comment = None;
                }
            }
        } else {
            last_comment = None;
        }
    }

    last_comment
}

/// Extract Go-style doc comments (// lines preceding a node).
pub fn extract_go_doc_comment_before(
    root: &tree_sitter::Node,
    target: &tree_sitter::Node,
    src: &[u8],
) -> Option<String> {
    collect_doc_comment_lines(root, target, src, "comment", &["//"])
}

/// Extract Python docstring from the first statement in a function/class body.
pub fn extract_python_docstring(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Option<String> {
    let body = node.child_by_field_name("body")?;
    let mut cursor = body.walk();

    for child in body.children(&mut cursor) {
        if child.kind() == "expression_statement" {
            let mut inner_cursor = child.walk();
            for inner in child.children(&mut inner_cursor) {
                if inner.kind() == "string" || inner.kind() == "concatenated_string" {
                    if let Ok(text) = inner.utf8_text(src) {
                        let stripped = text
                            .trim_start_matches("\"\"\"")
                            .trim_start_matches("'''")
                            .trim_end_matches("\"\"\"")
                            .trim_end_matches("'''")
                            .trim();
                        if !stripped.is_empty() {
                            return Some(stripped.to_owned());
                        }
                    }
                }
            }
            break;
        } else if child.kind() != "comment" {
            break;
        }
    }
    None
}

/// Extract imports generically by node kind name(s).
pub fn extract_imports_by_kind(
    root: &tree_sitter::Node,
    src: &[u8],
    import_kinds: &[&str],
) -> Vec<String> {
    let mut imports = Vec::new();
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if import_kinds.contains(&child.kind()) {
            if let Ok(text) = child.utf8_text(src) {
                imports.push(text.to_owned());
            }
        }
    }
    imports
}

/// Extract visibility from Java-style modifiers (public/private/protected).
pub fn extract_java_visibility(node: &tree_sitter::Node, src: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "modifiers" {
            let mut mod_cursor = child.walk();
            for modifier in child.children(&mut mod_cursor) {
                if let Ok(text) = modifier.utf8_text(src) {
                    match text {
                        "public" | "private" | "protected" => return text.to_owned(),
                        _ => {}
                    }
                }
            }
        }
    }
    String::new()
}

/// Check if a Go name is exported (starts with uppercase).
pub fn extract_go_visibility(name: &str) -> String {
    if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        "pub".to_owned()
    } else {
        String::new()
    }
}

/// Extract return type from a TypeScript type annotation.
pub fn extract_ts_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    node.child_by_field_name("return_type")
        .and_then(|n| {
            let mut cursor = n.walk();
            for child in n.children(&mut cursor) {
                if child.kind() != ":" {
                    return child.utf8_text(src).ok().map(|s| s.to_owned());
                }
            }
            n.utf8_text(src).ok().map(|s| {
                s.strip_prefix(": ").unwrap_or(s).to_owned()
            })
        })
}

/// Extract text from a named field on a tree-sitter node.
///
/// Common helper for return-type extractors that simply read a field's text
/// without any prefix-stripping logic (e.g., C `"type"`, Python `"return_type"`,
/// Go `"result"`, Java `"type"`).
fn extract_field_text(node: &tree_sitter::Node, field_name: &str, src: &[u8]) -> Option<String> {
    node.child_by_field_name(field_name)
        .and_then(|n| n.utf8_text(src).ok())
        .map(|s| s.to_owned())
}

/// Extract return type from Python type annotation ("-> Type").
pub fn extract_python_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    extract_field_text(node, "return_type", src)
}

/// Extract Go function return type(s).
pub fn extract_go_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    extract_field_text(node, "result", src)
}

/// Build a `CodeChunk` for a simple named type declaration (struct/enum).
///
/// Shared by C and C++ chunkers for `struct_specifier` and `enum_specifier`.
pub fn simple_type_chunk(
    node: &tree_sitter::Node,
    src: &[u8],
    kind: CodeNodeKind,
    doc: Option<String>,
    imports: &[String],
    chunk_index: usize,
    min_lines: usize,
) -> Option<CodeChunk> {
    let name = extract_name(node, src);
    if name.is_empty() {
        return None;
    }
    let text = node.utf8_text(src).ok()?.to_owned();
    if text.lines().count() < min_lines {
        return None;
    }
    let type_refs = if matches!(kind, CodeNodeKind::Struct) {
        extract_type_refs(node, src)
    } else {
        Vec::new()
    };
    Some(CodeChunk {
        text,
        kind,
        name,
        signature: None,
        doc_comment: doc,
        visibility: String::new(),
        start_line: node.start_position().row,
        end_line: node.end_position().row,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        chunk_index,
        imports: imports.to_vec(),
        calls: Vec::new(),
        type_refs,
        param_types: Vec::new(),
        return_type: None,
        ast_hash: 0,
        body_hash: 0,
    })
}

// ---------------------------------------------------------------------------
// Language-agnostic chunk builders (shared by all chunkers)
// ---------------------------------------------------------------------------

/// Language-specific extraction callbacks.
///
/// Each chunker defines a `const` instance mapping tree-sitter node kinds
/// to `CodeNodeKind` and providing extraction functions for name, visibility,
/// params, return type, and doc comments.
pub struct LangExtractors {
    /// `(tree-sitter node kind, CodeNodeKind)` pairs for top-level nodes.
    pub kind_map: &'static [(&'static str, CodeNodeKind)],
    /// Extract symbol name from a node.
    pub extract_name_fn: fn(&tree_sitter::Node, &[u8]) -> String,
    /// Extract visibility string (e.g. `"pub"`, `"export"`, `""`).
    pub extract_vis_fn: fn(&tree_sitter::Node, &[u8]) -> String,
    /// Extract parameter name-type pairs.
    pub extract_params_fn: fn(&tree_sitter::Node, &[u8]) -> Vec<(String, String)>,
    /// Extract return type.
    pub extract_return_fn: fn(&tree_sitter::Node, &[u8]) -> Option<String>,
    /// Extract doc comment given (parent, child, src).
    pub extract_doc_fn: fn(&tree_sitter::Node, &tree_sitter::Node, &[u8]) -> Option<String>,
    /// Node kinds considered methods inside a class/impl body.
    pub method_kinds: &'static [&'static str],
}

/// Build a `CodeChunk` for a function/type node using language-specific extractors.
pub fn build_chunk(
    config: &CodeChunkConfig,
    lang: &LangExtractors,
    node: &tree_sitter::Node,
    src: &[u8],
    imports: &[String],
    index: usize,
) -> Option<CodeChunk> {
    let kind = lang
        .kind_map
        .iter()
        .find(|(k, _)| *k == node.kind())
        .map(|(_, v)| *v)?;

    let text = node.utf8_text(src).ok()?.to_owned();
    if text.lines().count() < config.min_lines {
        return None;
    }

    let name = (lang.extract_name_fn)(node, src);
    let visibility = (lang.extract_vis_fn)(node, src);

    let is_func = kind == CodeNodeKind::Function;
    let signature = if is_func {
        Some(extract_function_signature(node, src))
    } else {
        None
    };
    let calls = if config.extract_calls && is_func {
        extract_calls(node, src)
    } else {
        Vec::new()
    };
    let type_refs = extract_type_refs(node, src);
    let param_types = (lang.extract_params_fn)(node, src);
    let return_type = (lang.extract_return_fn)(node, src);

    Some(CodeChunk {
        text,
        kind,
        name,
        signature,
        doc_comment: None, // filled by caller
        visibility,
        start_line: node.start_position().row,
        end_line: node.end_position().row,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        chunk_index: index,
        imports: imports.to_vec(),
        calls,
        type_refs,
        param_types,
        return_type,
        ast_hash: 0,
        body_hash: 0,
    })
}

/// Extract methods from a class/impl body using language-specific extractors.
pub fn extract_methods(
    config: &CodeChunkConfig,
    lang: &LangExtractors,
    parent_node: &tree_sitter::Node,
    src: &[u8],
    imports: &[String],
    parent_name: &str,
    chunks: &mut Vec<CodeChunk>,
) {
    let Some(body) = parent_node.child_by_field_name("body") else {
        return;
    };

    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if !lang.method_kinds.contains(&child.kind()) {
            continue;
        }

        let text = match child.utf8_text(src) {
            Ok(t) => t.to_owned(),
            Err(_) => continue,
        };

        if text.lines().count() < config.min_lines {
            continue;
        }

        let method_name = (lang.extract_name_fn)(&child, src);
        let full_name = if parent_name.is_empty() {
            method_name
        } else {
            format!("{parent_name}::{method_name}")
        };

        let visibility = (lang.extract_vis_fn)(&child, src);
        let signature = extract_function_signature(&child, src);
        let calls = if config.extract_calls {
            extract_calls(&child, src)
        } else {
            Vec::new()
        };
        let doc = (lang.extract_doc_fn)(&body, &child, src);
        let type_refs = extract_type_refs(&child, src);
        let param_types = (lang.extract_params_fn)(&child, src);
        let return_type = (lang.extract_return_fn)(&child, src);

        chunks.push(CodeChunk {
            text,
            kind: CodeNodeKind::Function,
            name: full_name,
            signature: Some(signature),
            doc_comment: doc,
            visibility,
            start_line: child.start_position().row,
            end_line: child.end_position().row,
            start_byte: child.start_byte(),
            end_byte: child.end_byte(),
            chunk_index: chunks.len(),
            imports: imports.to_vec(),
            calls,
            type_refs,
            param_types,
            return_type,
            ast_hash: 0,
            body_hash: 0,
        });
    }
}

// ---------------------------------------------------------------------------
// C/C++ shared helpers (moved from c_lang.rs / cpp.rs)
// ---------------------------------------------------------------------------

/// Extract visibility from C storage class specifier (`static`).
pub fn extract_c_visibility(node: &tree_sitter::Node, src: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "storage_class_specifier"
            && let Ok(text) = child.utf8_text(src)
            && text == "static"
        {
            return "static".to_owned();
        }
    }
    String::new()
}

/// Extract C/C++ parameter name-type pairs from a function_definition node.
pub fn extract_c_params(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
    let Some(declarator) = node.child_by_field_name("declarator") else {
        return Vec::new();
    };
    let func_decl = if declarator.kind() == "function_declarator" {
        declarator
    } else {
        let mut cursor = declarator.walk();
        match declarator
            .children(&mut cursor)
            .find(|c| c.kind() == "function_declarator")
        {
            Some(fd) => fd,
            None => return Vec::new(),
        }
    };

    let Some(params) = func_decl.child_by_field_name("parameters") else {
        return Vec::new();
    };

    let mut result = Vec::new();
    let mut cursor = params.walk();

    for child in params.children(&mut cursor) {
        if child.kind() != "parameter_declaration" {
            continue;
        }

        let ty = child
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or_default();
        let name = child
            .child_by_field_name("declarator")
            .and_then(|n| {
                if n.kind() == "identifier" {
                    n.utf8_text(src).ok()
                } else {
                    let mut c = n.walk();
                    n.children(&mut c)
                        .find(|ch| ch.kind() == "identifier")
                        .and_then(|ch| ch.utf8_text(src).ok())
                }
            })
            .unwrap_or_default();

        if !name.is_empty() && !ty.is_empty() {
            result.push((name.to_owned(), ty.to_owned()));
        }
    }

    result
}

/// Extract return type from C/C++ function_definition (child field "type").
pub fn extract_c_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    extract_field_text(node, "type", src)
}

/// No visibility (returns empty string). Used for languages without visibility modifiers.
pub fn no_visibility(_node: &tree_sitter::Node, _src: &[u8]) -> String {
    String::new()
}

/// Go visibility wrapper for `LangExtractors` fn pointer (extracts name first).
pub fn extract_go_visibility_from_node(node: &tree_sitter::Node, src: &[u8]) -> String {
    let name = extract_name(node, src);
    extract_go_visibility(&name)
}

// ---------------------------------------------------------------------------
// TypeScript param extraction (moved from typescript.rs)
// ---------------------------------------------------------------------------

/// Extract TypeScript parameter name-type pairs.
pub fn extract_ts_params(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
    extract_params_generic(
        node,
        src,
        &["required_parameter", "optional_parameter"],
        |child, s| {
            let name = child
                .child_by_field_name("pattern")
                .and_then(|n| n.utf8_text(s).ok())
                .unwrap_or_default();
            // TS type annotation is `type: ": Type"` — skip the `:` token
            let ty = child
                .child_by_field_name("type")
                .and_then(|n| {
                    let mut c = n.walk();
                    for inner in n.children(&mut c) {
                        if inner.kind() != ":" {
                            return inner.utf8_text(s).ok();
                        }
                    }
                    n.utf8_text(s).ok()
                })
                .unwrap_or_default();

            if !name.is_empty() && !ty.is_empty() {
                vec![(name.to_owned(), ty.to_owned())]
            } else {
                Vec::new()
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Java param extraction (moved from java.rs)
// ---------------------------------------------------------------------------

/// Extract Java parameter name-type pairs.
pub fn extract_java_params(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
    extract_params_generic(
        node,
        src,
        &["formal_parameter", "spread_parameter"],
        |child, s| field_name_type_pair(child, s, "name", "type"),
    )
}

/// Extract Java return type from a method's "type" field.
pub fn extract_java_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    extract_field_text(node, "type", src)
}

// ---------------------------------------------------------------------------
// Go param extraction (moved from go_lang.rs)
// ---------------------------------------------------------------------------

/// Extract Go parameter name-type pairs.
pub fn extract_go_params(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
    extract_params_generic(node, src, &["parameter_declaration"], |child, s| {
        let ty = child
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(s).ok())
            .unwrap_or_default();

        // Go allows multiple names per parameter_declaration (e.g. `a, b int`)
        let mut pairs = Vec::new();
        let mut name_cursor = child.walk();
        for name_child in child.children(&mut name_cursor) {
            if name_child.kind() == "identifier"
                && let Ok(name) = name_child.utf8_text(s)
                && !name.is_empty()
                && !ty.is_empty()
            {
                pairs.push((name.to_owned(), ty.to_owned()));
            }
        }
        pairs
    })
}

// ---------------------------------------------------------------------------
// Python param extraction (moved from python.rs)
// ---------------------------------------------------------------------------

/// Extract Python parameter name-type pairs (skips bare self/cls).
pub fn extract_py_params(node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
    extract_params_generic(
        node,
        src,
        &["typed_parameter", "typed_default_parameter"],
        |child, s| {
            let name = child
                .child_by_field_name("name")
                .or_else(|| {
                    let mut c = child.walk();
                    child.children(&mut c).find(|n| n.kind() == "identifier")
                })
                .and_then(|n| n.utf8_text(s).ok())
                .unwrap_or_default();
            let ty = child
                .child_by_field_name("type")
                .and_then(|n| n.utf8_text(s).ok())
                .unwrap_or_default();

            if !name.is_empty() && !ty.is_empty() && name != "self" && name != "cls" {
                vec![(name.to_owned(), ty.to_owned())]
            } else {
                Vec::new()
            }
        },
    )
}
