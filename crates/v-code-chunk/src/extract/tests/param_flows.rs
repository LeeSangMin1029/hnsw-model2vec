use crate::extract::common::{collect_param_names, walk_for_param_flows, walk_param_flows_inner};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
fn parse_rust(src: &str) -> tree_sitter::Tree {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("failed to set language");
    parser.parse(src, None).expect("failed to parse")
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

// ===========================================================================
// collect_param_names
// ===========================================================================

#[test]
fn collect_param_names_simple() {
    let src = "fn foo(a: i32, b: String) {}";
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let params = collect_param_names(&func, src.as_bytes());
    assert_eq!(params, vec![("a".to_owned(), 0), ("b".to_owned(), 1)]);
}

#[test]
fn collect_param_names_with_self() {
    let src = r#"
impl Foo {
    fn bar(&self, x: i32, y: u8) {}
}
"#;
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let params = collect_param_names(&func, src.as_bytes());
    // self occupies position 0, so x=1, y=2
    assert_eq!(params, vec![("x".to_owned(), 1), ("y".to_owned(), 2)]);
}

#[test]
fn collect_param_names_no_params() {
    let src = "fn empty() {}";
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let params = collect_param_names(&func, src.as_bytes());
    assert!(params.is_empty());
}

// ===========================================================================
// walk_param_flows_inner
// ===========================================================================

#[test]
fn walk_param_flows_inner_finds_flow() {
    let src = "fn foo(x: i32) { bar(x); }";
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let params = vec![("x".to_owned(), 0u8)];
    let mut flows = Vec::new();
    walk_param_flows_inner(&func, src.as_bytes(), &params, &mut flows);
    assert_eq!(flows.len(), 1);
    assert_eq!(flows[0].param_name, "x");
    assert_eq!(flows[0].param_position, 0);
    assert_eq!(flows[0].callee, "bar");
    assert_eq!(flows[0].callee_arg, 0);
}

#[test]
fn walk_param_flows_inner_skips_noise() {
    // Some/Ok/Err are noise callees and should be skipped
    let src = "fn foo(x: i32) { Some(x); }";
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let params = vec![("x".to_owned(), 0u8)];
    let mut flows = Vec::new();
    walk_param_flows_inner(&func, src.as_bytes(), &params, &mut flows);
    assert!(flows.is_empty(), "noise callees should be filtered out");
}

#[test]
fn walk_param_flows_inner_multiple_args() {
    let src = "fn foo(a: i32, b: i32) { baz(a, b); }";
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let params = vec![("a".to_owned(), 0u8), ("b".to_owned(), 1u8)];
    let mut flows = Vec::new();
    walk_param_flows_inner(&func, src.as_bytes(), &params, &mut flows);
    assert_eq!(flows.len(), 2);
    assert_eq!(flows[0].callee_arg, 0);
    assert_eq!(flows[1].callee_arg, 1);
}

// ===========================================================================
// walk_for_param_flows (integration of collect + inner)
// ===========================================================================

#[test]
fn walk_for_param_flows_basic() {
    let src = "fn process(data: Vec<u8>) { send(data); }";
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let flows = walk_for_param_flows(&func, src.as_bytes());
    assert_eq!(flows.len(), 1);
    assert_eq!(flows[0].param_name, "data");
    assert_eq!(flows[0].callee, "send");
    assert_eq!(flows[0].param_position, 0);
    assert_eq!(flows[0].callee_arg, 0);
}

#[test]
fn walk_for_param_flows_no_params_no_flows() {
    let src = "fn noop() { foo(42); }";
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let flows = walk_for_param_flows(&func, src.as_bytes());
    assert!(flows.is_empty());
}

#[test]
fn walk_for_param_flows_multiple_calls() {
    let src = r#"
fn handler(a: i32, b: String) {
    alpha(a);
    beta(b, a);
}
"#;
    let tree = parse_rust(src);
    let func = find_first_kind(tree.root_node(), "function_item").unwrap();
    let flows = walk_for_param_flows(&func, src.as_bytes());
    // alpha(a) → a at callee_arg 0
    // beta(b, a) → b at callee_arg 0, a at callee_arg 1
    assert_eq!(flows.len(), 3);

    let alpha_flows: Vec<_> = flows.iter().filter(|f| f.callee == "alpha").collect();
    assert_eq!(alpha_flows.len(), 1);
    assert_eq!(alpha_flows[0].param_name, "a");

    let beta_flows: Vec<_> = flows.iter().filter(|f| f.callee == "beta").collect();
    assert_eq!(beta_flows.len(), 2);
    assert_eq!(beta_flows[0].param_name, "b");
    assert_eq!(beta_flows[0].callee_arg, 0);
    assert_eq!(beta_flows[1].param_name, "a");
    assert_eq!(beta_flows[1].callee_arg, 1);
}
