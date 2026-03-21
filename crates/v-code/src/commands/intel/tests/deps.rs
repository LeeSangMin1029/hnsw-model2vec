use std::collections::BTreeMap;

use crate::commands::intel::deps::{merge_deps, resolve_symbol, DepGraph, DepSet};
use crate::commands::intel::parse::ParsedChunk;

fn chunk(name: &str, file: &str, calls: &[&str], types: &[&str]) -> ParsedChunk {
    ParsedChunk {
        kind: "function".to_owned(),
        name: name.to_owned(),
        file: file.to_owned(),
        lines: Some((1, 10)),
        signature: None,
        calls: calls.iter().map(|s| s.to_string()).collect(),
        call_lines: vec![],
        types: types.iter().map(|s| s.to_string()).collect(),
        imports: vec![],
        string_args: vec![],
        param_flows: vec![],
        param_types: vec![],
        field_types: vec![],
        local_types: vec![],
        let_call_bindings: vec![],
        field_accesses: vec![],
        return_type: None,
        enum_variants: vec![],
        is_test: false,
    }
}

// ── resolve_symbol ───────────────────────────────────────────────────

#[test]
fn resolve_exact_match() {
    let mut idx = BTreeMap::new();
    idx.insert("foo::bar".to_owned(), "src/foo.rs".to_owned());

    assert_eq!(
        resolve_symbol("Foo::Bar", &idx),
        Some("src/foo.rs".to_owned())
    );
}

#[test]
fn resolve_short_name_fallback() {
    let mut idx = BTreeMap::new();
    idx.insert("bar".to_owned(), "src/bar.rs".to_owned());

    // "Mod::bar" should resolve via short name "bar"
    assert_eq!(
        resolve_symbol("Mod::bar", &idx),
        Some("src/bar.rs".to_owned())
    );
}

#[test]
fn resolve_dot_notation() {
    let mut idx = BTreeMap::new();
    idx.insert("method".to_owned(), "src/x.rs".to_owned());

    assert_eq!(
        resolve_symbol("self.method", &idx),
        Some("src/x.rs".to_owned())
    );
}

#[test]
fn resolve_missing_returns_none() {
    let idx = BTreeMap::new();
    assert_eq!(resolve_symbol("nonexistent", &idx), None);
}

// ── merge_deps ───────────────────────────────────────────────────────

#[test]
fn merge_deps_combines_vias() {
    let mut deps: DepSet = DepSet::new();
    deps.insert(("a.rs".to_owned(), "calls"));
    deps.insert(("a.rs".to_owned(), "types"));
    deps.insert(("b.rs".to_owned(), "calls"));

    let merged = merge_deps(&deps);
    assert_eq!(merged.len(), 2);

    let a_vias = &merged["a.rs"];
    assert!(a_vias.contains(&"calls"));
    assert!(a_vias.contains(&"types"));

    let b_vias = &merged["b.rs"];
    assert_eq!(b_vias, &vec!["calls"]);
}

// ── DepGraph::build ──────────────────────────────────────────────────

#[test]
fn dep_graph_basic_edges() {
    let chunks = vec![
        chunk("Foo::run", "src/foo.rs", &["Bar::exec"], &[]),
        chunk("Bar::exec", "src/bar.rs", &[], &[]),
    ];

    let graph = DepGraph::build(&chunks);
    assert_eq!(graph.nodes.len(), 2);

    let foo_node = &graph.nodes["src/foo.rs"];
    // foo calls bar → outgoing edge
    assert!(!foo_node.outgoing.is_empty());
    assert!(foo_node.incoming.is_empty());

    let bar_node = &graph.nodes["src/bar.rs"];
    // bar is called by foo → incoming edge
    assert!(bar_node.outgoing.is_empty());
    assert!(!bar_node.incoming.is_empty());
}

#[test]
fn dep_graph_self_calls_ignored() {
    let chunks = vec![
        chunk("Foo::a", "src/foo.rs", &["Foo::b"], &[]),
        chunk("Foo::b", "src/foo.rs", &[], &[]),
    ];

    let graph = DepGraph::build(&chunks);
    let foo_node = &graph.nodes["src/foo.rs"];
    // Same file calls should NOT create edges
    assert!(foo_node.outgoing.is_empty());
    assert!(foo_node.incoming.is_empty());
}

#[test]
fn dep_graph_type_edges() {
    let chunks = vec![
        chunk("Foo::run", "src/foo.rs", &[], &["Bar"]),
        chunk("Bar", "src/bar.rs", &[], &[]),
    ];

    let graph = DepGraph::build(&chunks);
    let foo_node = &graph.nodes["src/foo.rs"];
    assert!(foo_node.outgoing.iter().any(|(_, via)| *via == "types"));
}

#[test]
fn dep_graph_match_file() {
    let chunks = vec![
        chunk("A", "crates/core/src/lib.rs", &[], &[]),
        chunk("B", "crates/core/src/util.rs", &[], &[]),
    ];
    let graph = DepGraph::build(&chunks);

    let matches = graph.match_file("lib.rs");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0], "crates/core/src/lib.rs");
}

#[test]
fn dep_graph_total_edges() {
    let chunks = vec![
        chunk("Foo::run", "src/foo.rs", &["Bar::exec"], &[]),
        chunk("Bar::exec", "src/bar.rs", &[], &[]),
    ];

    let graph = DepGraph::build(&chunks);
    assert_eq!(graph.total_edges(), 1);
}

// ── collect_transitive_files ─────────────────────────────────────────

use v_code_intel::deps::collect_transitive_files;

#[test]
fn transitive_files_depth_1() {
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"], &[]),
        chunk("B", "src/b.rs", &["C"], &[]),
        chunk("C", "src/c.rs", &[], &[]),
    ];
    let graph = DepGraph::build(&chunks);
    let files = collect_transitive_files(&graph, "a.rs", 1);
    // depth 1: a.rs itself + direct neighbors (b.rs)
    assert!(files.contains("src/a.rs"), "should include start file");
    assert!(files.contains("src/b.rs"), "should include direct dep");
    assert!(!files.contains("src/c.rs"), "should not include depth-2 dep");
}

#[test]
fn transitive_files_depth_2() {
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"], &[]),
        chunk("B", "src/b.rs", &["C"], &[]),
        chunk("C", "src/c.rs", &[], &[]),
    ];
    let graph = DepGraph::build(&chunks);
    let files = collect_transitive_files(&graph, "a.rs", 2);
    assert!(files.contains("src/a.rs"));
    assert!(files.contains("src/b.rs"));
    assert!(files.contains("src/c.rs"), "should include depth-2 dep");
}

#[test]
fn transitive_files_bidirectional() {
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"], &[]),
        chunk("B", "src/b.rs", &[], &[]),
        chunk("C", "src/c.rs", &["B"], &[]),
    ];
    let graph = DepGraph::build(&chunks);
    // Starting from b.rs, both a.rs and c.rs should be reachable (incoming edges)
    let files = collect_transitive_files(&graph, "b.rs", 1);
    assert!(files.contains("src/b.rs"));
    assert!(files.contains("src/a.rs"), "should include incoming dep");
    assert!(files.contains("src/c.rs"), "should include incoming dep");
}
