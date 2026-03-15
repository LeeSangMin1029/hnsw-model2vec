use crate::extract::common::{
    extract_callee_from_node, extract_string_value, extract_visibility, is_noise_callee,
    strip_string_quotes, walk_for_type_ids,
};

// ---------------------------------------------------------------------------
// Helper: parse Rust source and return the first function_item node's tree
// ---------------------------------------------------------------------------
fn parse_rust(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("failed to set language");
    parser.parse(src, None).expect("failed to parse")
}

/// Find first node of the given kind via DFS.
fn find_first_kind<'a>(node: tree_sitter::Node<'a>, kind: &str) -> Option<tree_sitter::Node<'a>> {
    if node.kind() == kind {
        return Some(node);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(found) = find_first_kind(child, kind) {
            return Some(found);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// extract_visibility
// ---------------------------------------------------------------------------

#[test]
fn extract_visibility_pub() {
    let src = "pub fn foo() {}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let func = root.child(0).expect("no child");
    assert_eq!(extract_visibility(&func, src.as_bytes()), "pub");
}

#[test]
fn extract_visibility_pub_crate() {
    let src = "pub(crate) fn bar() {}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let func = root.child(0).expect("no child");
    assert_eq!(extract_visibility(&func, src.as_bytes()), "pub(crate)");
}

#[test]
fn extract_visibility_private() {
    let src = "fn baz() {}";
    let tree = parse_rust(src);
    let root = tree.root_node();
    let func = root.child(0).expect("no child");
    assert_eq!(extract_visibility(&func, src.as_bytes()), "");
}

// ---------------------------------------------------------------------------
// is_noise_callee
// ---------------------------------------------------------------------------

#[test]
fn noise_callee_known_names() {
    for name in &[
        "Some", "Ok", "Err", "None", "Box::new", "Arc::new", "format",
        "println", "eprintln", "panic", "todo", "assert_eq", "context",
        "expect", "unwrap_or", "map_err",
    ] {
        assert!(is_noise_callee(name), "{name} should be noise");
    }
}

#[test]
fn noise_callee_non_noise() {
    for name in &["my_func", "process", "HashMap::new", "Vec::with_capacity", "run"] {
        assert!(!is_noise_callee(name), "{name} should NOT be noise");
    }
}

// ---------------------------------------------------------------------------
// strip_string_quotes
// ---------------------------------------------------------------------------

#[test]
fn strip_standard_double_quotes() {
    assert_eq!(strip_string_quotes("\"hello\""), "hello");
}

#[test]
fn strip_single_quotes() {
    assert_eq!(strip_string_quotes("'c'"), "c");
}

#[test]
fn strip_raw_string_no_hashes() {
    assert_eq!(strip_string_quotes("r\"raw content\""), "raw content");
}

#[test]
fn strip_raw_string_with_hashes() {
    assert_eq!(strip_string_quotes("r#\"has quotes\"#"), "has quotes");
}

#[test]
fn strip_raw_string_with_multiple_hashes() {
    assert_eq!(
        strip_string_quotes("r##\"double hash\"##"),
        "double hash"
    );
}

#[test]
fn strip_empty_quoted_string() {
    assert_eq!(strip_string_quotes("\"\""), "");
}

// ---------------------------------------------------------------------------
// extract_callee_from_node
// ---------------------------------------------------------------------------

#[test]
fn callee_simple_function_call() {
    let src = "fn main() { foo(); }";
    let tree = parse_rust(src);
    let call = find_first_kind(tree.root_node(), "call_expression").unwrap();
    assert_eq!(extract_callee_from_node(&call, src.as_bytes()), Some("foo".to_owned()));
}

#[test]
fn callee_method_call_on_self() {
    let src = "fn f() { self.bar(); }";
    let tree = parse_rust(src);
    let call = find_first_kind(tree.root_node(), "call_expression").unwrap();
    assert_eq!(extract_callee_from_node(&call, src.as_bytes()), Some("self.bar".to_owned()));
}

#[test]
fn callee_qualified_path() {
    let src = "fn f() { std::fs::read(); }";
    let tree = parse_rust(src);
    let call = find_first_kind(tree.root_node(), "call_expression").unwrap();
    assert_eq!(extract_callee_from_node(&call, src.as_bytes()), Some("std::fs::read".to_owned()));
}

#[test]
fn callee_non_call_returns_none() {
    let src = "fn f() { let x = 1; }";
    let tree = parse_rust(src);
    let let_node = find_first_kind(tree.root_node(), "let_declaration").unwrap();
    assert_eq!(extract_callee_from_node(&let_node, src.as_bytes()), None);
}

// ---------------------------------------------------------------------------
// extract_string_value
// ---------------------------------------------------------------------------

#[test]
fn string_value_regular_string() {
    let src = r#"fn f() { let s = "hello world"; }"#;
    let tree = parse_rust(src);
    let string_node = find_first_kind(tree.root_node(), "string_literal").unwrap();
    assert_eq!(extract_string_value(&string_node, src.as_bytes()), Some("hello world".to_owned()));
}

#[test]
fn string_value_non_string_returns_none() {
    let src = "fn f() { let x = 42; }";
    let tree = parse_rust(src);
    let int_node = find_first_kind(tree.root_node(), "integer_literal").unwrap();
    assert_eq!(extract_string_value(&int_node, src.as_bytes()), None);
}

// ---------------------------------------------------------------------------
// walk_for_type_ids
// ---------------------------------------------------------------------------

#[test]
fn type_ids_from_struct_fields() {
    let src = "struct Foo { a: Bar, b: Baz }";
    let tree = parse_rust(src);
    let mut refs = Vec::new();
    walk_for_type_ids(&tree.root_node(), src.as_bytes(), &mut refs);
    assert!(refs.contains(&"Bar".to_owned()));
    assert!(refs.contains(&"Baz".to_owned()));
}

#[test]
fn type_ids_from_function_params() {
    let src = "fn process(item: Widget, count: Counter) -> Result {}";
    let tree = parse_rust(src);
    let mut refs = Vec::new();
    walk_for_type_ids(&tree.root_node(), src.as_bytes(), &mut refs);
    assert!(refs.contains(&"Widget".to_owned()));
    assert!(refs.contains(&"Counter".to_owned()));
    assert!(refs.contains(&"Result".to_owned()));
}

#[test]
fn type_ids_empty_for_primitives() {
    let src = "fn f(x: i32) -> bool {}";
    let tree = parse_rust(src);
    let mut refs = Vec::new();
    walk_for_type_ids(&tree.root_node(), src.as_bytes(), &mut refs);
    assert!(refs.is_empty());
}
