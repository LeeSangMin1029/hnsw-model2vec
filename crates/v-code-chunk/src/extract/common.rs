//! Language-agnostic AST extraction helpers.
//!
//! Functions that work across all tree-sitter grammars: name, visibility,
//! signature, imports, calls, type refs, params, return type, doc comments.

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
    super::lang::extract_imports_by_kind(root, src, &["use_declaration"])
}

/// Collect items via a walker, then sort and deduplicate.
pub fn collect_sorted_unique(
    node: &tree_sitter::Node,
    src: &[u8],
    walker: fn(&tree_sitter::Node, &[u8], &mut Vec<String>),
) -> Vec<String> {
    let mut items = Vec::new();
    walker(node, src, &mut items);
    items.sort();
    items.dedup();
    items
}

/// Recursively walk to find call nodes across languages (without line info).
///
/// Thin wrapper over `walk_for_calls_with_lines` — discards line data.
pub fn walk_for_calls(node: &tree_sitter::Node, src: &[u8], calls: &mut Vec<String>) {
    let mut lines = Vec::new();
    walk_for_calls_with_lines(node, src, calls, &mut lines);
}

/// Recursively walk to find call nodes, recording the 0-based source line of each call.
pub fn walk_for_calls_with_lines(
    node: &tree_sitter::Node,
    src: &[u8],
    calls: &mut Vec<String>,
    lines: &mut Vec<u32>,
) {
    let call_name = match node.kind() {
        "call_expression" | "call" => {
            node.child_by_field_name("function")
                .and_then(|f| f.utf8_text(src).ok())
                .map(|t| t.to_owned())
        }
        "method_invocation" => {
            node.child_by_field_name("name").and_then(|n| {
                n.utf8_text(src).ok().map(|name| {
                    if let Some(obj) = node.child_by_field_name("object")
                        && let Ok(obj_text) = obj.utf8_text(src) {
                            format!("{obj_text}.{name}")
                        } else {
                            name.to_owned()
                        }
                })
            })
        }
        _ => None,
    };

    if let Some(name) = call_name {
        // Normalize multiline call expressions: collapse whitespace to single space
        // e.g. "self\n            .request" → "self.request"
        let name = if name.contains('\n') {
            name.split_whitespace().collect::<Vec<_>>().join("")
        } else {
            name
        };
        lines.push(node.start_position().row as u32);
        calls.push(name);
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_calls_with_lines(&child, src, calls, lines);
    }
}

/// Extract doc comments (`///` or `//!`) preceding a node.
/// Collect doc-comment lines immediately before `target` under `root`.
///
/// `comment_kind` is the tree-sitter node kind for comments (e.g. `"line_comment"`,
/// `"comment"`). `prefixes` lists the prefix strings to strip (e.g. `&["///", "//!"]`).
/// Only comments that match at least one prefix are kept.
pub(super) fn collect_doc_comment_lines(
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

/// Walk recursively to collect `type_identifier` texts.
pub fn walk_for_type_ids(node: &tree_sitter::Node, src: &[u8], refs: &mut Vec<String>) {
    if node.kind() == "type_identifier"
        && let Ok(text) = node.utf8_text(src) {
            refs.push(text.to_owned());
        }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_type_ids(&child, src, refs);
    }
}

/// Extract struct/class field declarations as a compact signature string.
///
/// For Rust structs, returns `"field1: Type1, field2: Type2"`.
/// Returns `None` if the node has no `field_declaration_list` body.
pub fn extract_struct_fields(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    // Rust: "field_declaration_list", Go/TS: "body"
    let body = node.child_by_field_name("body")
        .or_else(|| {
            let mut c = node.walk();
            node.children(&mut c).find(|n| n.kind() == "field_declaration_list")
        })?;
    let mut fields = Vec::new();
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "field_declaration" {
            let name = child.child_by_field_name("name")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();
            let ty = child.child_by_field_name("type")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();
            if !name.is_empty() && !ty.is_empty() {
                fields.push(format!("{name}: {ty}"));
            }
        }
    }
    if fields.is_empty() { None } else { Some(fields.join(", ")) }
}

/// Generic helper: extract parameter name-type pairs from a node's `parameters` field.
///
/// For each child whose kind is in `param_kinds`, calls `extractor` to produce
/// zero or more `(name, type)` pairs. This avoids duplicating the
/// "get parameters → filter by kind → collect" boilerplate across languages.
pub(super) fn extract_params_generic(
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
pub(super) fn field_name_type_pair(
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
    extract_return_type_skip(node, src, "->", "-> ")
}

/// Extract return type from a field node, skipping a delimiter token.
///
/// Used by Rust (`"->"`, `"-> "`) and TypeScript (`":"`, `": "`).
pub(super) fn extract_return_type_skip(
    node: &tree_sitter::Node,
    src: &[u8],
    skip_kind: &str,
    strip_prefix: &str,
) -> Option<String> {
    let ret = node.child_by_field_name("return_type")?;

    let mut cursor = ret.walk();
    for child in ret.children(&mut cursor) {
        if child.kind() != skip_kind {
            return child.utf8_text(src).ok().map(|s| s.to_owned());
        }
    }

    ret.utf8_text(src)
        .ok()
        .map(|s| s.strip_prefix(strip_prefix).unwrap_or(s).to_owned())
}
