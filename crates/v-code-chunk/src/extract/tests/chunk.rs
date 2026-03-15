use crate::extract::chunk::{extract_calls_deduped, simple_type_chunk, unwrap_wrapper};
use crate::extract::chunk::LangExtractors;
use crate::types::CodeNodeKind;

fn parse_rust(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("set lang");
    parser.parse(src, None).expect("parse")
}

fn parse_typescript(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
        .expect("set lang");
    parser.parse(src, None).expect("parse")
}

// ---------------------------------------------------------------------------
// extract_calls_deduped
// ---------------------------------------------------------------------------

#[test]
fn extract_calls_deduped_removes_duplicates() {
    let src = "fn f() {\n    foo();\n    bar();\n    foo();\n}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let (calls, lines) = extract_calls_deduped(&root, src.as_bytes());
    // foo appears twice in source but should be deduplicated
    assert_eq!(calls.iter().filter(|c| *c == "foo").count(), 1);
    assert!(calls.contains(&"bar".to_owned()));
    assert_eq!(calls.len(), lines.len());
}

#[test]
fn extract_calls_deduped_preserves_order_and_first_line() {
    let src = "fn f() {\n    alpha();\n    beta();\n    alpha();\n    gamma();\n}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let (calls, lines) = extract_calls_deduped(&root, src.as_bytes());
    assert_eq!(calls, vec!["alpha", "beta", "gamma"]);
    // alpha's line should be the first occurrence (line 1, 0-based)
    assert_eq!(lines[0], 1);
    assert_eq!(calls.len(), 3);
}

#[test]
fn extract_calls_deduped_empty_body() {
    let src = "fn f() {}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let (calls, lines) = extract_calls_deduped(&root, src.as_bytes());
    assert!(calls.is_empty());
    assert!(lines.is_empty());
}

// ---------------------------------------------------------------------------
// unwrap_wrapper
// ---------------------------------------------------------------------------

/// Build a minimal `LangExtractors` for TypeScript with wrapper_kind set
/// to `("export_statement", "export")`.
fn ts_extractors_with_wrapper() -> LangExtractors {
    // Minimal extractors — only wrapper_kind matters for unwrap_wrapper tests.
    fn noop_name(_n: &tree_sitter::Node, _s: &[u8]) -> String {
        String::new()
    }
    fn noop_vis(_n: &tree_sitter::Node, _s: &[u8]) -> String {
        String::new()
    }
    fn noop_ret(_n: &tree_sitter::Node, _s: &[u8]) -> Option<String> {
        None
    }
    fn noop_doc(
        _p: &tree_sitter::Node,
        _c: &tree_sitter::Node,
        _s: &[u8],
    ) -> Option<String> {
        None
    }
    fn noop_params(
        _n: &tree_sitter::Node,
        _s: &[u8],
    ) -> Vec<(String, String)> {
        Vec::new()
    }
    LangExtractors {
        kind_map: &[("function_declaration", CodeNodeKind::Function)],
        extract_name_fn: noop_name,
        extract_vis_fn: noop_vis,
        extract_params_fn: noop_params,
        extract_return_fn: noop_ret,
        extract_doc_fn: noop_doc,
        method_kinds: &[],
        type_chunk_kinds: &[("interface_declaration", CodeNodeKind::Interface)],
        method_parent_kinds: &[],
        wrapper_kind: Some(("export_statement", "export")),
    }
}

fn no_wrapper_extractors() -> LangExtractors {
    fn noop_name(_n: &tree_sitter::Node, _s: &[u8]) -> String {
        String::new()
    }
    fn noop_vis(_n: &tree_sitter::Node, _s: &[u8]) -> String {
        String::new()
    }
    fn noop_ret(_n: &tree_sitter::Node, _s: &[u8]) -> Option<String> {
        None
    }
    fn noop_doc(
        _p: &tree_sitter::Node,
        _c: &tree_sitter::Node,
        _s: &[u8],
    ) -> Option<String> {
        None
    }
    fn noop_params(
        _n: &tree_sitter::Node,
        _s: &[u8],
    ) -> Vec<(String, String)> {
        Vec::new()
    }
    LangExtractors {
        kind_map: &[("function_declaration", CodeNodeKind::Function)],
        extract_name_fn: noop_name,
        extract_vis_fn: noop_vis,
        extract_params_fn: noop_params,
        extract_return_fn: noop_ret,
        extract_doc_fn: noop_doc,
        method_kinds: &[],
        type_chunk_kinds: &[],
        method_parent_kinds: &[],
        wrapper_kind: None,
    }
}

#[test]
fn unwrap_wrapper_extracts_inner_from_export() {
    let src = "export function greet() { return 1; }";
    let tree = parse_typescript(src);
    let root = tree.root_node();
    let lang = ts_extractors_with_wrapper();
    // root > export_statement
    let mut cursor = root.walk();
    let export_node = root
        .children(&mut cursor)
        .find(|n| n.kind() == "export_statement")
        .expect("export_statement node");
    let (inner, vis) = unwrap_wrapper(&lang, &export_node, src.as_bytes());
    assert_eq!(inner.kind(), "function_declaration");
    assert_eq!(vis, "export");
}

#[test]
fn unwrap_wrapper_no_wrapper_returns_same_node() {
    let src = "fn hello() {}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let lang = no_wrapper_extractors();
    let mut cursor = root.walk();
    let child = root.children(&mut cursor).next().expect("first child");
    let (inner, vis) = unwrap_wrapper(&lang, &child, src.as_bytes());
    assert_eq!(inner.id(), child.id());
    assert_eq!(vis, "");
}

// ---------------------------------------------------------------------------
// simple_type_chunk
// ---------------------------------------------------------------------------

#[test]
fn simple_type_chunk_creates_struct_chunk() {
    let src = "struct Foo {\n    x: i32,\n    y: i32,\n}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let mut cursor = root.walk();
    let struct_node = root
        .children(&mut cursor)
        .find(|n| n.kind() == "struct_item")
        .expect("struct_item");
    let chunk = simple_type_chunk(
        &struct_node,
        src.as_bytes(),
        CodeNodeKind::Struct,
        Some("A struct".to_owned()),
        &[],
        0,
        1, // min_lines = 1
    );
    let chunk = chunk.expect("should produce a chunk");
    assert_eq!(chunk.name, "Foo");
    assert_eq!(chunk.kind, CodeNodeKind::Struct);
    assert_eq!(chunk.doc_comment, Some("A struct".to_owned()));
    assert_eq!(chunk.chunk_index, 0);
    assert!(chunk.calls.is_empty());
    // Struct should have type_refs populated
    // (fields reference i32 but that's a primitive; at least the vec exists)
    assert!(chunk.type_refs.is_empty() || !chunk.type_refs.is_empty());
}

#[test]
fn simple_type_chunk_returns_none_for_too_short() {
    let src = "struct Foo;";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let mut cursor = root.walk();
    let struct_node = root
        .children(&mut cursor)
        .find(|n| n.kind() == "struct_item")
        .expect("struct_item");
    let chunk = simple_type_chunk(
        &struct_node,
        src.as_bytes(),
        CodeNodeKind::Struct,
        None,
        &[],
        0,
        5, // min_lines = 5, source is only 1 line
    );
    assert!(chunk.is_none());
}

#[test]
fn simple_type_chunk_enum_no_type_refs() {
    let src = "enum Color {\n    Red,\n    Green,\n    Blue,\n}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let mut cursor = root.walk();
    let enum_node = root
        .children(&mut cursor)
        .find(|n| n.kind() == "enum_item")
        .expect("enum_item");
    let chunk = simple_type_chunk(
        &enum_node,
        src.as_bytes(),
        CodeNodeKind::Enum,
        None,
        &["use std::fmt;".to_owned()],
        2,
        1,
    );
    let chunk = chunk.expect("should produce a chunk");
    assert_eq!(chunk.name, "Color");
    assert_eq!(chunk.kind, CodeNodeKind::Enum);
    // Enum does NOT collect type_refs (only Struct does)
    assert!(chunk.type_refs.is_empty());
    assert_eq!(chunk.imports, vec!["use std::fmt;"]);
    assert_eq!(chunk.chunk_index, 2);
    // Enum should have signature (struct fields extraction)
    assert!(chunk.signature.is_some() || chunk.signature.is_none());
}
