use crate::extract::common::{extract_callee_name, extract_field_receiver, walk_for_calls_with_lines};

fn parse_rust(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("set lang");
    parser.parse(src, None).expect("parse")
}

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

// ── extract_field_receiver ──

#[test]
fn field_receiver_simple_identifier() {
    let src = "fn f() { obj.field; }";
    let tree = parse_rust(src);
    let node = find_first_kind(tree.root_node(), "field_expression").unwrap();
    let result = extract_field_receiver(node, src.as_bytes());
    assert_eq!(result, Some("obj.field".to_owned()));
}

#[test]
fn field_receiver_self() {
    let src = "fn f() { self.foo; }";
    let tree = parse_rust(src);
    let node = find_first_kind(tree.root_node(), "field_expression").unwrap();
    let result = extract_field_receiver(node, src.as_bytes());
    assert_eq!(result, Some("self.foo".to_owned()));
}

#[test]
fn field_receiver_nested() {
    let src = "fn f() { self.foo.bar; }";
    let tree = parse_rust(src);
    // The outermost field_expression is `self.foo.bar`
    let node = find_first_kind(tree.root_node(), "field_expression").unwrap();
    let result = extract_field_receiver(node, src.as_bytes());
    assert_eq!(result, Some("self.foo.bar".to_owned()));
}

// ── extract_callee_name ──

#[test]
fn callee_name_simple_function() {
    let src = "fn f() { foo(); }";
    let tree = parse_rust(src);
    let call = find_first_kind(tree.root_node(), "call_expression").unwrap();
    let func_node = call.child_by_field_name("function").unwrap();
    let result = extract_callee_name(func_node, src.as_bytes());
    assert_eq!(result, Some("foo".to_owned()));
}

#[test]
fn callee_name_method_on_self() {
    let src = "fn f() { self.method(); }";
    let tree = parse_rust(src);
    let call = find_first_kind(tree.root_node(), "call_expression").unwrap();
    let func_node = call.child_by_field_name("function").unwrap();
    let result = extract_callee_name(func_node, src.as_bytes());
    assert_eq!(result, Some("self.method".to_owned()));
}

#[test]
fn callee_name_method_on_identifier() {
    let src = "fn f() { obj.do_thing(); }";
    let tree = parse_rust(src);
    let call = find_first_kind(tree.root_node(), "call_expression").unwrap();
    let func_node = call.child_by_field_name("function").unwrap();
    let result = extract_callee_name(func_node, src.as_bytes());
    assert_eq!(result, Some("obj.do_thing".to_owned()));
}

#[test]
fn callee_name_chained_field() {
    let src = "fn f() { self.foo.bar(); }";
    let tree = parse_rust(src);
    let call = find_first_kind(tree.root_node(), "call_expression").unwrap();
    let func_node = call.child_by_field_name("function").unwrap();
    let result = extract_callee_name(func_node, src.as_bytes());
    assert_eq!(result, Some("self.foo.bar".to_owned()));
}

// ── walk_for_calls_with_lines ──

#[test]
fn walk_collects_calls_and_lines() {
    let src = "fn f() {\n    foo();\n    bar();\n}";
    let tree = parse_rust(src);
    let mut calls = Vec::new();
    let mut lines = Vec::new();
    walk_for_calls_with_lines(&tree.root_node(), src.as_bytes(), &mut calls, &mut lines);
    assert_eq!(calls.len(), lines.len());
    assert!(calls.contains(&"foo".to_owned()));
    assert!(calls.contains(&"bar".to_owned()));
    assert!(calls.len() >= 2);
}

#[test]
fn walk_collects_method_calls() {
    let src = "fn f() {\n    self.run();\n}";
    let tree = parse_rust(src);
    let mut calls = Vec::new();
    let mut lines = Vec::new();
    walk_for_calls_with_lines(&tree.root_node(), src.as_bytes(), &mut calls, &mut lines);
    assert!(calls.contains(&"self.run".to_owned()));
    assert_eq!(calls.len(), lines.len());
}

#[test]
fn walk_empty_function_no_calls() {
    let src = "fn f() {}";
    let tree = parse_rust(src);
    let mut calls = Vec::new();
    let mut lines = Vec::new();
    walk_for_calls_with_lines(&tree.root_node(), src.as_bytes(), &mut calls, &mut lines);
    assert!(calls.is_empty());
    assert!(lines.is_empty());
}
