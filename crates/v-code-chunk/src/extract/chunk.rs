//! Language-agnostic chunk builders shared by all chunkers.

use crate::{CodeChunk, CodeChunkConfig, CodeNodeKind};
use super::ParsedSource;
use super::common::{extract_enum_variant_names, extract_function_signature, extract_struct_fields, extract_struct_field_types, collect_sorted_unique, walk_for_calls_with_lines, walk_for_field_accesses, walk_for_let_call_bindings, walk_for_let_types, walk_for_param_flows, walk_for_string_args, walk_for_type_ids, extract_name};

/// Walk calls with line info and deduplicate, preserving first occurrence's line.
pub(crate) fn extract_calls_deduped(
    node: &tree_sitter::Node,
    src: &[u8],
) -> (Vec<String>, Vec<u32>) {
    let mut raw_calls = Vec::new();
    let mut raw_lines = Vec::new();
    walk_for_calls_with_lines(node, src, &mut raw_calls, &mut raw_lines);
    let mut seen = std::collections::HashSet::new();
    let mut calls = Vec::new();
    let mut lines = Vec::new();
    for (c, l) in raw_calls.into_iter().zip(raw_lines) {
        if seen.insert(c.clone()) {
            calls.push(c);
            lines.push(l);
        }
    }
    (calls, lines)
}

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
        collect_sorted_unique(node, src, walk_for_type_ids)
    } else {
        Vec::new()
    };
    let signature = if matches!(kind, CodeNodeKind::Struct | CodeNodeKind::Enum) {
        extract_struct_fields(node, src)
    } else {
        None
    };
    let field_types = if matches!(kind, CodeNodeKind::Struct) {
        extract_struct_field_types(node, src)
    } else {
        Vec::new()
    };
    Some(CodeChunk {
        text,
        kind,
        name,
        signature,
        doc_comment: doc,
        visibility: String::new(),
        start_line: node.start_position().row,
        end_line: node.end_position().row,
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        chunk_index,
        imports: imports.to_vec(),
        calls: Vec::new(),
        call_lines: Vec::new(),
        type_refs,
        param_types: Vec::new(),
        field_types,
        return_type: None,
        ast_hash: 0,
        body_hash: 0,
        sub_blocks: Vec::new(),
        string_args: Vec::new(),
        param_flows: Vec::new(),
        local_types: Vec::new(),
        let_call_bindings: Vec::new(),
        field_accesses: Vec::new(),
        enum_variants: Vec::new(),
        is_test: false,
    })
}

// ---------------------------------------------------------------------------
// Language-agnostic chunk builders (shared by all chunkers)
// ---------------------------------------------------------------------------

/// Function pointer extracting parameter name-type pairs from a tree-sitter node.
pub type ExtractParamsFn = fn(&tree_sitter::Node, &[u8]) -> Vec<(String, String)>;

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
    pub extract_params_fn: ExtractParamsFn,
    /// Extract return type.
    pub extract_return_fn: fn(&tree_sitter::Node, &[u8]) -> Option<String>,
    /// Extract doc comment given (parent, child, src).
    pub extract_doc_fn: fn(&tree_sitter::Node, &tree_sitter::Node, &[u8]) -> Option<String>,
    /// Node kinds considered methods inside a class/impl body.
    pub method_kinds: &'static [&'static str],
    /// `(tree-sitter node kind, CodeNodeKind)` pairs for type declarations
    /// handled via `simple_type_chunk` (e.g. struct_specifier, enum_specifier).
    pub type_chunk_kinds: &'static [(&'static str, CodeNodeKind)],
    /// Node kinds whose body contains methods to extract (e.g. impl_item, class_declaration).
    pub method_parent_kinds: &'static [&'static str],
    /// Optional wrapper node to unwrap: `(wrapper_kind, visibility_to_set)`.
    /// E.g. Python's `("decorated_definition", "")`, TS's `("export_statement", "export")`.
    pub wrapper_kind: Option<(&'static str, &'static str)>,
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
    } else if matches!(kind, CodeNodeKind::Struct | CodeNodeKind::Enum) {
        extract_struct_fields(node, src)
    } else {
        None
    };
    let (calls, call_lines) = if config.extract_calls && is_func {
        extract_calls_deduped(node, src)
    } else {
        (Vec::new(), Vec::new())
    };
    let string_args = if config.extract_calls && is_func {
        walk_for_string_args(node, src)
            .into_iter()
            .map(|sa| (sa.callee, sa.value, sa.line, sa.arg_position))
            .collect()
    } else {
        Vec::new()
    };
    let param_flows = if config.extract_calls && is_func {
        walk_for_param_flows(node, src)
            .into_iter()
            .map(|pf| (pf.param_name, pf.param_position, pf.callee, pf.callee_arg, pf.line))
            .collect()
    } else {
        Vec::new()
    };
    let type_refs = collect_sorted_unique(node, src, walk_for_type_ids);
    let param_types = (lang.extract_params_fn)(node, src);
    let return_type = (lang.extract_return_fn)(node, src);
    let field_types = if matches!(kind, CodeNodeKind::Struct) {
        extract_struct_field_types(node, src)
    } else {
        Vec::new()
    };
    let local_types = if is_func {
        walk_for_let_types(node, src)
    } else {
        Vec::new()
    };
    let let_call_bindings = if is_func {
        walk_for_let_call_bindings(node, src)
    } else {
        Vec::new()
    };
    let field_accesses = if is_func {
        walk_for_field_accesses(node, src)
    } else {
        Vec::new()
    };
    let enum_variants = if kind == CodeNodeKind::Enum {
        extract_enum_variant_names(node, src)
    } else {
        Vec::new()
    };

    let is_test = is_func && has_test_attribute_with_src(node, src);

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
        call_lines,
        type_refs,
        param_types,
        field_types,
        return_type,
        ast_hash: 0,
        body_hash: 0,
        sub_blocks: Vec::new(),
        string_args,
        param_flows,
        local_types,
        let_call_bindings,
        field_accesses,
        enum_variants,
        is_test,
    })
}

/// Extract methods from a class/impl body using language-specific extractors.
///
/// Handles wrapper nodes (e.g. Python's `decorated_definition`) via
/// `LangExtractors::wrapper_kind`.
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
        // Unwrap wrapper nodes inside class/impl bodies
        let actual_child = if let Some((wrapper, _)) = lang.wrapper_kind {
            if child.kind() == wrapper {
                let mut ic = child.walk();
                match child.children(&mut ic).find(|n| lang.method_kinds.contains(&n.kind())) {
                    Some(inner) => inner,
                    None => continue,
                }
            } else {
                child
            }
        } else {
            child
        };

        if !lang.method_kinds.contains(&actual_child.kind()) {
            continue;
        }

        let text = match actual_child.utf8_text(src) {
            Ok(t) => t.to_owned(),
            Err(_) => continue,
        };

        if text.lines().count() < config.min_lines {
            continue;
        }

        let method_name = (lang.extract_name_fn)(&actual_child, src);
        let full_name = if parent_name.is_empty() {
            method_name
        } else {
            format!("{parent_name}::{method_name}")
        };

        let visibility = (lang.extract_vis_fn)(&actual_child, src);
        let signature = extract_function_signature(&actual_child, src);
        let (calls, call_lines) = if config.extract_calls {
            extract_calls_deduped(&actual_child, src)
        } else {
            (Vec::new(), Vec::new())
        };
        let string_args = if config.extract_calls {
            walk_for_string_args(&actual_child, src)
                .into_iter()
                .map(|sa| (sa.callee, sa.value, sa.line, sa.arg_position))
                .collect()
        } else {
            Vec::new()
        };
        let param_flows = if config.extract_calls {
            walk_for_param_flows(&actual_child, src)
                .into_iter()
                .map(|pf| (pf.param_name, pf.param_position, pf.callee, pf.callee_arg, pf.line))
                .collect()
        } else {
            Vec::new()
        };
        let doc = (lang.extract_doc_fn)(&body, &actual_child, src);
        let type_refs = collect_sorted_unique(&actual_child, src, walk_for_type_ids);
        let param_types = (lang.extract_params_fn)(&actual_child, src);
        let return_type = (lang.extract_return_fn)(&actual_child, src);
        let local_types = walk_for_let_types(&actual_child, src);
        let let_call_bindings = walk_for_let_call_bindings(&actual_child, src);
        let field_accesses = walk_for_field_accesses(&actual_child, src);
        let is_test = has_test_attribute_with_src(&actual_child, src);

        chunks.push(CodeChunk {
            text,
            kind: CodeNodeKind::Function,
            name: full_name,
            signature: Some(signature),
            doc_comment: doc,
            visibility,
            start_line: actual_child.start_position().row,
            end_line: actual_child.end_position().row,
            start_byte: actual_child.start_byte(),
            end_byte: actual_child.end_byte(),
            chunk_index: chunks.len(),
            imports: imports.to_vec(),
            calls,
            call_lines,
            type_refs,
            param_types,
            field_types: Vec::new(),
            return_type,
            ast_hash: 0,
            body_hash: 0,
            sub_blocks: Vec::new(),
            string_args,
            param_flows,
            local_types,
            let_call_bindings,
            field_accesses,
            enum_variants: Vec::new(),
            is_test,
        });
    }
}

/// Standard chunking loop shared by all language chunkers.
///
/// Processes top-level declarations from the parsed syntax tree:
/// 1. Nodes matching `kind_map` → `build_chunk()`
/// 2. Nodes matching `type_chunk_kinds` → `simple_type_chunk()` with visibility/type_refs
/// 3. Nodes matching `method_parent_kinds` → `extract_methods()`
///
/// Handles wrapper nodes (e.g. `decorated_definition`, `export_statement`)
/// via `LangExtractors::wrapper_kind`.
pub fn chunk_standard(
    config: &CodeChunkConfig,
    lang: &LangExtractors,
    parsed: &ParsedSource,
    source: &str,
) -> Vec<CodeChunk> {
    let root = parsed.tree.root_node();
    let src = source.as_bytes();
    let imports = &parsed.imports;

    let mut chunks = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        // Unwrap wrapper nodes (e.g. decorated_definition, export_statement)
        let (actual_node, extra_vis) = unwrap_wrapper(lang, &child, src);

        let doc = (lang.extract_doc_fn)(&root, &actual_node, src);

        // 1. kind_map → build_chunk
        if let Some(mut chunk) = build_chunk(config, lang, &actual_node, src, imports, chunks.len()) {
            if !extra_vis.is_empty() {
                chunk.visibility = extra_vis.to_owned();
            }
            chunk.doc_comment = doc;
            chunks.push(chunk);
        }
        // 2. type_chunk_kinds → simple_type_chunk with lang-specific visibility/type_refs
        else if let Some(&(_, kind)) = lang.type_chunk_kinds.iter().find(|(k, _)| *k == actual_node.kind())
            && let Some(mut chunk) = simple_type_chunk(
                &actual_node, src, kind, doc, imports, chunks.len(), 0,
            )
        {
            chunk.visibility = (lang.extract_vis_fn)(&actual_node, src);
            chunk.type_refs = collect_sorted_unique(&actual_node, src, walk_for_type_ids);
            if !extra_vis.is_empty() {
                chunk.visibility = extra_vis.to_owned();
            }
            chunks.push(chunk);
        }

        // 3. method_parent_kinds → extract_methods
        if lang.method_parent_kinds.contains(&actual_node.kind()) {
            let parent_name = (lang.extract_name_fn)(&actual_node, src);
            extract_methods(config, lang, &actual_node, src, imports, &parent_name, &mut chunks);
        }
    }

    chunks
}

/// Unwrap a wrapper node if it matches `LangExtractors::wrapper_kind`.
///
/// Returns `(actual_node, visibility_string)`. If no wrapper or no inner match,
/// returns the original node with empty visibility.
pub(crate) fn unwrap_wrapper<'a>(
    lang: &LangExtractors,
    child: &tree_sitter::Node<'a>,
    _src: &[u8],
) -> (tree_sitter::Node<'a>, &'static str) {
    if let Some((wrapper, vis)) = lang.wrapper_kind
        && child.kind() == wrapper
    {
        let mut inner_cursor = child.walk();
        let found = child.children(&mut inner_cursor).find(|inner| {
            lang.kind_map.iter().any(|(k, _)| *k == inner.kind())
                || lang.type_chunk_kinds.iter().any(|(k, _)| *k == inner.kind())
        });
        if let Some(n) = found {
            return (n, vis);
        }
    }
    (*child, "")
}

/// Check if a function node has a test attribute (`#[test]`, `#[tokio::test]`, etc.).
///
/// Checks prev siblings for attribute/decorator/annotation nodes.
/// Works for Rust (`attribute_item`), Python (`decorator`), Java (`annotation`).
pub(crate) fn has_test_attribute_with_src(node: &tree_sitter::Node, src: &[u8]) -> bool {
    let mut sibling = node.prev_named_sibling();
    while let Some(s) = sibling {
        match s.kind() {
            "attribute_item" | "attribute" => {
                if let Ok(text) = s.utf8_text(src) {
                    let lower = text.to_lowercase();
                    if lower.contains("test") {
                        return true;
                    }
                }
            }
            "decorator" | "decorator_list" => {
                if let Ok(text) = s.utf8_text(src) {
                    let lower = text.to_lowercase();
                    if lower.contains("test") || lower.contains("pytest") {
                        return true;
                    }
                }
            }
            "annotation" | "modifiers" => {
                if let Ok(text) = s.utf8_text(src) {
                    let lower = text.to_lowercase();
                    if lower.contains("test") {
                        return true;
                    }
                }
            }
            _ => break,
        }
        sibling = s.prev_named_sibling();
    }
    false
}
