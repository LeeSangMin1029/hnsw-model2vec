//! Tree-sitter node extraction helpers.
//!
//! HOW TO EXTEND: Add new extraction functions here when adding support
//! for new code metadata (e.g., generic parameters, lifetime annotations).

use super::{CodeChunk, CodeChunkConfig, CodeNodeKind};

/// Extract the symbol name from a node.
pub(super) fn extract_name(node: &tree_sitter::Node, src: &[u8]) -> String {
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
pub(super) fn extract_visibility(node: &tree_sitter::Node, src: &[u8]) -> String {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            return child.utf8_text(src).unwrap_or_default().to_owned();
        }
    }
    String::new()
}

/// Extract function signature (everything before the body block).
pub(super) fn extract_function_signature(node: &tree_sitter::Node, src: &[u8]) -> String {
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
pub(super) fn extract_imports(root: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    let mut imports = Vec::new();
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "use_declaration"
            && let Ok(text) = child.utf8_text(src) {
                imports.push(text.to_owned());
            }
    }
    imports
}

/// Extract function call names from a node's subtree.
pub(super) fn extract_calls(node: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
    let mut calls = Vec::new();
    walk_for_calls(node, src, &mut calls);
    calls.sort();
    calls.dedup();
    calls
}

/// Recursively walk to find `call_expression` nodes.
pub(super) fn walk_for_calls(node: &tree_sitter::Node, src: &[u8], calls: &mut Vec<String>) {
    if node.kind() == "call_expression"
        && let Some(func) = node.child_by_field_name("function")
        && let Ok(text) = func.utf8_text(src) {
            calls.push(text.to_owned());
        }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_calls(&child, src, calls);
    }
}

/// Extract doc comments (`///` or `//!`) preceding a node.
pub(super) fn extract_doc_comment_before(
    root: &tree_sitter::Node,
    target: &tree_sitter::Node,
    src: &[u8],
) -> Option<String> {
    let target_start = target.start_position().row;
    let mut doc_lines = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        if child.start_position().row >= target_start {
            break;
        }

        if child.kind() == "line_comment" {
            if let Ok(text) = child.utf8_text(src) {
                if let Some(doc) = text.strip_prefix("///") {
                    doc_lines.push(doc.trim().to_owned());
                } else if let Some(doc) = text.strip_prefix("//!") {
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

/// Recursively collect all `type_identifier` node texts from a subtree.
///
/// Returns deduplicated, sorted type names referenced in the node.
pub(super) fn extract_type_refs(node: &tree_sitter::Node, src: &[u8]) -> Vec<String> {
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

/// Extract parameter name-type pairs from a function's `parameters` field.
///
/// For each parameter, extracts the `pattern` (name) and `type` (type text).
/// Skips `self` parameters.
pub(super) fn extract_param_types(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Vec<(String, String)> {
    let Some(params) = node.child_by_field_name("parameters") else {
        return Vec::new();
    };

    let mut result = Vec::new();
    let mut cursor = params.walk();

    for child in params.children(&mut cursor) {
        if child.kind() != "parameter" {
            continue;
        }

        let name = child
            .child_by_field_name("pattern")
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or_default();
        let ty = child
            .child_by_field_name("type")
            .and_then(|n| n.utf8_text(src).ok())
            .unwrap_or_default();

        if !name.is_empty() && !ty.is_empty() {
            result.push((name.to_owned(), ty.to_owned()));
        }
    }

    result
}

/// Extract the return type string from a function's `return_type` field.
pub(super) fn extract_return_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
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

/// Extract individual methods from impl/trait body as separate chunks.
pub(super) fn extract_body_methods(
    config: &CodeChunkConfig,
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
        if child.kind() != "function_item" {
            continue;
        }

        let text = match child.utf8_text(src) {
            Ok(t) => t.to_owned(),
            Err(_) => continue,
        };

        let line_count = text.lines().count();
        if line_count < config.min_lines {
            continue;
        }

        let method_name = extract_name(&child, src);
        let full_name = if parent_name.is_empty() {
            method_name
        } else {
            format!("{parent_name}::{method_name}")
        };

        let visibility = extract_visibility(&child, src);
        let signature = extract_function_signature(&child, src);
        let calls = if config.extract_calls {
            extract_calls(&child, src)
        } else {
            Vec::new()
        };
        let doc = extract_doc_comment_before(&body, &child, src);
        let type_refs = extract_type_refs(&child, src);
        let param_types = extract_param_types(&child, src);
        let return_type = extract_return_type(&child, src);

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
        });
    }
}
