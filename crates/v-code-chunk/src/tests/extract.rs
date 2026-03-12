//! Unit tests for `chunk_code::extract` helper functions.

use crate::extract;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse Rust source with tree-sitter and return (tree, source bytes).
fn parse_rust(src: &str) -> (tree_sitter::Tree, Vec<u8>) {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("failed to set Rust language");
    let bytes = src.as_bytes().to_vec();
    let tree = parser.parse(&bytes, None).expect("failed to parse");
    (tree, bytes)
}

/// Parse Python source with tree-sitter and return (tree, source bytes).
fn parse_python(src: &str) -> (tree_sitter::Tree, Vec<u8>) {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("failed to set Python language");
    let bytes = src.as_bytes().to_vec();
    let tree = parser.parse(&bytes, None).expect("failed to parse");
    (tree, bytes)
}

/// Parse Java source with tree-sitter and return (tree, source bytes).
fn parse_java(src: &str) -> (tree_sitter::Tree, Vec<u8>) {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_java::LANGUAGE.into())
        .expect("failed to set Java language");
    let bytes = src.as_bytes().to_vec();
    let tree = parser.parse(&bytes, None).expect("failed to parse");
    (tree, bytes)
}

/// Walk a tree to find the first node matching `kind`.
fn find_first<'a>(node: &tree_sitter::Node<'a>, kind: &str) -> Option<tree_sitter::Node<'a>> {
    if node.kind() == kind {
        return Some(*node);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(found) = find_first(&child, kind) {
            return Some(found);
        }
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_go_visibility
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn go_visibility_uppercase_is_pub() {
    assert_eq!(extract::extract_go_visibility("Handler"), "pub");
    assert_eq!(extract::extract_go_visibility("A"), "pub");
}

#[test]
fn go_visibility_lowercase_is_empty() {
    assert_eq!(extract::extract_go_visibility("handler"), "");
    assert_eq!(extract::extract_go_visibility("a"), "");
}

#[test]
fn go_visibility_empty_string() {
    assert_eq!(extract::extract_go_visibility(""), "");
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_name
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_name_function() {
    let src = "fn hello_world() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    assert_eq!(extract::extract_name(&func, &bytes), "hello_world");
}

#[test]
fn extract_name_struct() {
    let src = "struct MyStruct { field: i32 }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let node = find_first(&root, "struct_item").expect("no struct_item");
    assert_eq!(extract::extract_name(&node, &bytes), "MyStruct");
}

#[test]
fn extract_name_impl_block() {
    let src = "impl MyStruct { fn foo(&self) {} }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let node = find_first(&root, "impl_item").expect("no impl_item");
    assert_eq!(extract::extract_name(&node, &bytes), "MyStruct");
}

#[test]
fn extract_name_impl_trait_for_type() {
    let src = "impl Display for MyStruct { fn fmt(&self, f: &mut Formatter) -> Result { Ok(()) } }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let node = find_first(&root, "impl_item").expect("no impl_item");
    assert_eq!(extract::extract_name(&node, &bytes), "Display for MyStruct");
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_visibility
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_visibility_pub() {
    let src = "pub fn foo() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    assert_eq!(extract::extract_visibility(&func, &bytes), "pub");
}

#[test]
fn extract_visibility_pub_crate() {
    let src = "pub(crate) fn bar() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    assert_eq!(extract::extract_visibility(&func, &bytes), "pub(crate)");
}

#[test]
fn extract_visibility_private() {
    let src = "fn baz() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    assert_eq!(extract::extract_visibility(&func, &bytes), "");
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_function_signature
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_function_signature_simple() {
    let src = "fn add(a: i32, b: i32) -> i32 { a + b }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let sig = extract::extract_function_signature(&func, &bytes);
    assert!(sig.contains("fn add"), "signature should contain 'fn add': {sig}");
    assert!(sig.contains("-> i32"), "signature should contain return type: {sig}");
    // The body should NOT be in the signature
    assert!(!sig.contains("a + b"), "signature should not include body: {sig}");
}

#[test]
fn extract_function_signature_no_body() {
    // trait method declaration without body
    let src = "trait Foo { fn bar(&self) -> bool; }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    // In trait declarations, methods are function_signature_item
    let func = find_first(&root, "function_signature_item");
    if let Some(func) = func {
        let sig = extract::extract_function_signature(&func, &bytes);
        assert!(sig.contains("fn bar"), "should contain fn bar: {sig}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_imports
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_imports_finds_use_declarations() {
    let src = "use std::collections::HashMap;\nuse std::io;\n\nfn main() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let imports = extract::extract_imports(&root, &bytes);
    assert_eq!(imports.len(), 2);
    assert!(imports[0].contains("HashMap"));
    assert!(imports[1].contains("std::io"));
}

#[test]
fn extract_imports_empty_when_none() {
    let src = "fn main() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let imports = extract::extract_imports(&root, &bytes);
    assert!(imports.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_calls
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_calls_finds_function_calls() {
    let src = r#"fn main() {
    let x = foo();
    let y = bar(x);
    baz(x, y);
}"#;
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let calls = extract::collect_sorted_unique(&func, &bytes, extract::walk_for_calls);
    assert!(calls.contains(&"foo".to_owned()), "should find foo: {calls:?}");
    assert!(calls.contains(&"bar".to_owned()), "should find bar: {calls:?}");
    assert!(calls.contains(&"baz".to_owned()), "should find baz: {calls:?}");
}

#[test]
fn extract_calls_deduplicates() {
    let src = r#"fn main() {
    foo();
    foo();
    foo();
}"#;
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let calls = extract::collect_sorted_unique(&func, &bytes, extract::walk_for_calls);
    assert_eq!(calls.iter().filter(|c| *c == "foo").count(), 1, "should deduplicate");
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_doc_comment_before
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_doc_comment_before_triple_slash() {
    let src = "/// This is a doc comment.\n/// Second line.\nfn documented() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let doc = extract::extract_doc_comment_before(&root, &func, &bytes);
    assert!(doc.is_some(), "should find doc comment");
    let doc = doc.unwrap();
    assert!(doc.contains("This is a doc comment"), "doc: {doc}");
    assert!(doc.contains("Second line"), "doc: {doc}");
}

#[test]
fn extract_doc_comment_before_no_comment() {
    let src = "fn undocumented() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let doc = extract::extract_doc_comment_before(&root, &func, &bytes);
    assert!(doc.is_none(), "should not find doc comment");
}

#[test]
fn extract_doc_comment_reset_on_intervening_node() {
    // The doc comment belongs to bar, not foo — there's a non-comment node
    // (another fn) between the comment and the target.
    let src = "/// This is for bar.\nfn bar() {}\nfn foo() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    // foo is the second function
    let mut cursor = root.walk();
    let funcs: Vec<_> = root
        .children(&mut cursor)
        .filter(|c| c.kind() == "function_item")
        .collect();
    assert_eq!(funcs.len(), 2);
    let doc = extract::extract_doc_comment_before(&root, &funcs[1], &bytes);
    assert!(doc.is_none(), "doc should be None for foo (reset by bar node)");
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_type_refs
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_type_refs_finds_types() {
    let src = "fn process(a: HashMap, b: Vec) -> Result { todo!() }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let refs = extract::collect_sorted_unique(&func, &bytes, extract::walk_for_type_ids);
    assert!(refs.contains(&"HashMap".to_owned()), "refs: {refs:?}");
    assert!(refs.contains(&"Vec".to_owned()), "refs: {refs:?}");
    assert!(refs.contains(&"Result".to_owned()), "refs: {refs:?}");
}

#[test]
fn extract_type_refs_sorted_deduped() {
    let src = "fn process(a: Vec, b: Vec) -> Vec { todo!() }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let refs = extract::collect_sorted_unique(&func, &bytes, extract::walk_for_type_ids);
    assert_eq!(refs.iter().filter(|r| *r == "Vec").count(), 1, "should deduplicate");
    // Should be sorted
    let mut sorted = refs.clone();
    sorted.sort();
    assert_eq!(refs, sorted, "should be sorted");
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_param_types
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_param_types_basic() {
    let src = "fn add(x: i32, y: f64) -> f64 { x as f64 + y }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let params = extract::extract_param_types(&func, &bytes);
    assert_eq!(params.len(), 2);
    assert_eq!(params[0], ("x".to_owned(), "i32".to_owned()));
    assert_eq!(params[1], ("y".to_owned(), "f64".to_owned()));
}

#[test]
fn extract_param_types_no_params() {
    let src = "fn noop() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let params = extract::extract_param_types(&func, &bytes);
    assert!(params.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_return_type
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_return_type_simple() {
    let src = "fn get_value() -> i32 { 42 }";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let ret = extract::extract_return_type(&func, &bytes);
    assert_eq!(ret, Some("i32".to_owned()));
}

#[test]
fn extract_return_type_none() {
    let src = "fn noop() {}";
    let (tree, bytes) = parse_rust(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_item").expect("no function_item");
    let ret = extract::extract_return_type(&func, &bytes);
    assert!(ret.is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_python_docstring
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_python_docstring_triple_quotes() {
    let src = r#"def greet(name):
    """Say hello to someone."""
    print(f"Hello, {name}")
"#;
    let (tree, bytes) = parse_python(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_definition").expect("no function_definition");
    let doc = extract::extract_python_docstring(&func, &bytes);
    assert!(doc.is_some(), "should find python docstring");
    assert!(doc.unwrap().contains("Say hello"), "should contain docstring text");
}

#[test]
fn extract_python_docstring_none() {
    let src = r#"def add(a, b):
    return a + b
"#;
    let (tree, bytes) = parse_python(src);
    let root = tree.root_node();
    let func = find_first(&root, "function_definition").expect("no function_definition");
    let doc = extract::extract_python_docstring(&func, &bytes);
    assert!(doc.is_none(), "should not find docstring");
}

// ═══════════════════════════════════════════════════════════════════════════
// extract_java_visibility
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extract_java_visibility_public() {
    let src = "class Foo { public void bar() {} }";
    let (tree, bytes) = parse_java(src);
    let root = tree.root_node();
    let method = find_first(&root, "method_declaration").expect("no method_declaration");
    let vis = extract::extract_java_visibility(&method, &bytes);
    assert_eq!(vis, "public");
}

#[test]
fn extract_java_visibility_private() {
    let src = "class Foo { private int getValue() { return 0; } }";
    let (tree, bytes) = parse_java(src);
    let root = tree.root_node();
    let method = find_first(&root, "method_declaration").expect("no method_declaration");
    let vis = extract::extract_java_visibility(&method, &bytes);
    assert_eq!(vis, "private");
}

#[test]
fn extract_java_visibility_default() {
    let src = "class Foo { void bar() {} }";
    let (tree, bytes) = parse_java(src);
    let root = tree.root_node();
    let method = find_first(&root, "method_declaration").expect("no method_declaration");
    let vis = extract::extract_java_visibility(&method, &bytes);
    assert_eq!(vis, "", "package-private should return empty string");
}
