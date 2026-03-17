use std::collections::BTreeMap;

use crate::commands::intel::deps::{merge_deps, resolve_symbol, DepGraph, DepSet};
use v_code_intel::deps::{common_prefix_len, crate_group};
use crate::commands::intel::parse::CodeChunk;

fn chunk(name: &str, file: &str, calls: &[&str], types: &[&str]) -> CodeChunk {
    CodeChunk {
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

// ── crate_group ──────────────────────────────────────────────────────

#[test]
fn crate_group_extracts_crate() {
    assert_eq!(
        crate_group("crates/v-hnsw-core/src/lib.rs"),
        "crates/v-hnsw-core"
    );
}

#[test]
fn crate_group_falls_back_to_src() {
    assert_eq!(crate_group("src/main.rs"), "src");
}

#[test]
fn crate_group_no_anchor() {
    assert_eq!(crate_group("lib.rs"), "lib.rs");
}

// ── common_prefix_len ────────────────────────────────────────────────

#[test]
fn common_prefix_empty() {
    assert_eq!(common_prefix_len(&[]), 0);
}

#[test]
fn common_prefix_single() {
    assert_eq!(common_prefix_len(&["crates/foo/src/lib.rs"]), "crates/foo/src/".len());
}

#[test]
fn common_prefix_multiple() {
    let paths = &[
        "crates/foo/src/a.rs",
        "crates/foo/src/b.rs",
        "crates/foo/src/sub/c.rs",
    ];
    assert_eq!(common_prefix_len(paths), "crates/foo/src/".len());
}

#[test]
fn common_prefix_no_common() {
    let paths = &["alpha/a.rs", "beta/b.rs"];
    // No common '/' prefix
    assert_eq!(common_prefix_len(paths), 0);
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
