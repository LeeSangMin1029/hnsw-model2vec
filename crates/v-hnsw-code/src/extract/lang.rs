//! Language-specific AST extraction helpers.
//!
//! C/C++, Go, Java, Python, TypeScript extractors for visibility,
//! params, return types, doc comments, and imports.

use super::common::{extract_name, extract_return_type_skip, extract_params_generic, field_name_type_pair, collect_doc_comment_lines};

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
    extract_return_type_skip(node, src, ":", ": ")
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

/// Python docstring extractor with `extract_doc_fn` signature.
///
/// Adapts `extract_python_docstring` to the `(parent, child, src)` signature
/// used by `LangExtractors::extract_doc_fn`.
pub fn extract_python_doc_wrapper(
    _parent: &tree_sitter::Node,
    child: &tree_sitter::Node,
    src: &[u8],
) -> Option<String> {
    extract_python_docstring(child, src)
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
