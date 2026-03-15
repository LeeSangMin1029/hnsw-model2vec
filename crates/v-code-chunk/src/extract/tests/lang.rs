use crate::extract::lang::{
    extract_c_func_name, extract_c_return_type, extract_c_visibility, extract_go_visibility,
    extract_python_return_type, no_visibility,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_python(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .expect("failed to set language");
    parser.parse(src, None).expect("failed to parse")
}

fn parse_go(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_go::LANGUAGE.into())
        .expect("failed to set language");
    parser.parse(src, None).expect("failed to parse")
}

fn parse_c(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_c::LANGUAGE.into())
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
// extract_go_visibility
// ---------------------------------------------------------------------------

#[test]
fn go_visibility_exported_name() {
    assert_eq!(extract_go_visibility("Hello"), "pub");
}

#[test]
fn go_visibility_unexported_name() {
    assert_eq!(extract_go_visibility("hello"), "");
}

#[test]
fn go_visibility_empty_string() {
    assert_eq!(extract_go_visibility(""), "");
}

// ---------------------------------------------------------------------------
// no_visibility
// ---------------------------------------------------------------------------

#[test]
fn no_visibility_returns_empty_for_go_node() {
    let src = "package main\nfunc Foo() {}";
    let tree = parse_go(src);
    let root = tree.root_node();
    let func_node = find_first_kind(root, "function_declaration").expect("no function node");
    assert_eq!(no_visibility(&func_node, src.as_bytes()), "");
}

#[test]
fn no_visibility_returns_empty_for_python_node() {
    let src = "def bar(): pass";
    let tree = parse_python(src);
    let root = tree.root_node();
    let func_node = find_first_kind(root, "function_definition").expect("no function node");
    assert_eq!(no_visibility(&func_node, src.as_bytes()), "");
}

// ---------------------------------------------------------------------------
// extract_python_return_type
// ---------------------------------------------------------------------------

#[test]
fn python_return_type_present() {
    let src = "def foo() -> int: pass";
    let tree = parse_python(src);
    let root = tree.root_node();
    let func_node = find_first_kind(root, "function_definition").expect("no function node");
    let ret = extract_python_return_type(&func_node, src.as_bytes());
    assert_eq!(ret, Some("int".to_owned()));
}

#[test]
fn python_return_type_absent() {
    let src = "def foo(): pass";
    let tree = parse_python(src);
    let root = tree.root_node();
    let func_node = find_first_kind(root, "function_definition").expect("no function node");
    let ret = extract_python_return_type(&func_node, src.as_bytes());
    assert_eq!(ret, None);
}

#[test]
fn python_return_type_complex() {
    let src = "def foo() -> List[str]: pass";
    let tree = parse_python(src);
    let root = tree.root_node();
    let func_node = find_first_kind(root, "function_definition").expect("no function node");
    let ret = extract_python_return_type(&func_node, src.as_bytes());
    assert!(ret.is_some());
    assert!(ret.unwrap().contains("List"));
}

// ---------------------------------------------------------------------------
// extract_c_func_name
// ---------------------------------------------------------------------------

#[test]
fn c_func_name_simple() {
    let src = "int main() { return 0; }";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_func_name(&node, src.as_bytes()), "main");
}

#[test]
fn c_func_name_pointer_return() {
    let src = "char *get_name(int id) { return 0; }";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_func_name(&node, src.as_bytes()), "get_name");
}

#[test]
fn c_func_name_with_params() {
    let src = "void process(int a, float b) {}";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_func_name(&node, src.as_bytes()), "process");
}

// ---------------------------------------------------------------------------
// extract_c_visibility
// ---------------------------------------------------------------------------

#[test]
fn c_visibility_static() {
    let src = "static int helper() { return 1; }";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_visibility(&node, src.as_bytes()), "static");
}

#[test]
fn c_visibility_non_static() {
    let src = "int public_func() { return 0; }";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_visibility(&node, src.as_bytes()), "");
}

#[test]
fn c_visibility_extern_not_static() {
    let src = "extern int ext_func() { return 0; }";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_visibility(&node, src.as_bytes()), "");
}

// ---------------------------------------------------------------------------
// extract_c_return_type
// ---------------------------------------------------------------------------

#[test]
fn c_return_type_int() {
    let src = "int foo() { return 0; }";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_return_type(&node, src.as_bytes()), Some("int".to_owned()));
}

#[test]
fn c_return_type_void() {
    let src = "void bar() {}";
    let tree = parse_c(src);
    let node = find_first_kind(tree.root_node(), "function_definition").expect("no func node");
    assert_eq!(extract_c_return_type(&node, src.as_bytes()), Some("void".to_owned()));
}
