//! Unit tests for `commands::intel::context` helper functions.

use crate::commands::intel::context::bfs_forward;
use crate::commands::intel::{format_lines_opt as format_lines, format_lines_str_opt as format_lines_str};
use crate::commands::intel::graph::CallGraph;
use crate::commands::intel::parse::ParsedChunk;

// ---------------------------------------------------------------------------
// format_lines
// ---------------------------------------------------------------------------

#[test]
fn format_lines_some() {
    assert_eq!(format_lines(Some((10, 20))), ":10-20");
    assert_eq!(format_lines(Some((1, 1))), ":1-1");
}

#[test]
fn format_lines_none_returns_empty() {
    assert_eq!(format_lines(None), "");
}

// ---------------------------------------------------------------------------
// format_lines_str
// ---------------------------------------------------------------------------

#[test]
fn format_lines_str_some() {
    assert_eq!(format_lines_str(Some((5, 15))), "5-15");
    assert_eq!(format_lines_str(Some((100, 200))), "100-200");
}

#[test]
fn format_lines_str_none_returns_empty() {
    assert_eq!(format_lines_str(None), "");
}

// ---------------------------------------------------------------------------
// bfs_forward
// ---------------------------------------------------------------------------

/// Build a small in-memory CallGraph for testing BFS.
///
/// Graph topology:
///   0 (main) --calls--> 1 (foo) --calls--> 2 (bar)
///                                           2 (bar) --calls--> 3 (baz)
///   4 (test_main) is a test node, no edges from main
fn build_test_graph() -> CallGraph {
    let chunks = vec![
        ParsedChunk {
            kind: "function".to_owned(),
            name: "main".to_owned(),
            file: "src/main.rs".to_owned(),
            lines: Some((1, 10)),
            signature: Some("fn main()".to_owned()),
            calls: vec!["foo".to_owned()],
            call_lines: vec![],
            types: vec![],
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
        },
        ParsedChunk {
            kind: "function".to_owned(),
            name: "foo".to_owned(),
            file: "src/lib.rs".to_owned(),
            lines: Some((5, 20)),
            signature: Some("fn foo() -> i32".to_owned()),
            calls: vec!["bar".to_owned()],
            call_lines: vec![],
            types: vec![],
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
        },
        ParsedChunk {
            kind: "function".to_owned(),
            name: "bar".to_owned(),
            file: "src/lib.rs".to_owned(),
            lines: Some((25, 35)),
            signature: Some("fn bar() -> bool".to_owned()),
            calls: vec!["baz".to_owned()],
            call_lines: vec![],
            types: vec![],
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
        },
        ParsedChunk {
            kind: "function".to_owned(),
            name: "baz".to_owned(),
            file: "src/util.rs".to_owned(),
            lines: Some((1, 5)),
            signature: None,
            calls: vec![],
            call_lines: vec![],
            types: vec![],
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
        },
        ParsedChunk {
            kind: "function".to_owned(),
            name: "test_main".to_owned(),
            file: "src/tests/test_main.rs".to_owned(),
            lines: Some((1, 15)),
            signature: None,
            calls: vec!["main".to_owned()],
            call_lines: vec![],
            types: vec![],
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
        },
    ];
    CallGraph::build(&chunks)
}

#[test]
fn bfs_forward_depth_zero_returns_seed_only() {
    let graph = build_test_graph();
    let results = bfs_forward(&graph, &[0], 0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].idx, 0);
    assert_eq!(results[0].depth, 0);
}

#[test]
fn bfs_forward_depth_one() {
    let graph = build_test_graph();
    let results = bfs_forward(&graph, &[0], 1);
    let indices: Vec<u32> = results.iter().map(|e| e.idx).collect();
    // Should include main (0) and foo (1)
    assert!(indices.contains(&0), "should include seed: {indices:?}");
    assert!(indices.contains(&1), "should include direct callee foo: {indices:?}");
    // Should NOT include bar (depth 2)
    assert!(!indices.contains(&2), "should not include bar at depth 2: {indices:?}");
}

#[test]
fn bfs_forward_depth_two_reaches_bar() {
    let graph = build_test_graph();
    let results = bfs_forward(&graph, &[0], 2);
    let indices: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(indices.contains(&2), "should reach bar at depth 2: {indices:?}");
    // baz is at depth 3
    assert!(!indices.contains(&3), "should not reach baz: {indices:?}");
}

#[test]
fn bfs_forward_full_depth() {
    let graph = build_test_graph();
    let results = bfs_forward(&graph, &[0], 10);
    let indices: Vec<u32> = results.iter().map(|e| e.idx).collect();
    // Should reach all nodes reachable from main: 0, 1, 2, 3
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(indices.contains(&2));
    assert!(indices.contains(&3));
    // test_main (4) calls main, but is not reachable FROM main
    assert!(!indices.contains(&4), "test_main should not be reachable from main: {indices:?}");
}

#[test]
fn bfs_forward_multiple_seeds() {
    let graph = build_test_graph();
    let results = bfs_forward(&graph, &[0, 3], 0);
    let indices: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(indices.contains(&0));
    assert!(indices.contains(&3));
    assert_eq!(results.len(), 2);
}

#[test]
fn bfs_forward_no_duplicate_visits() {
    let graph = build_test_graph();
    let results = bfs_forward(&graph, &[0], 10);
    let mut indices: Vec<u32> = results.iter().map(|e| e.idx).collect();
    let len_before = indices.len();
    indices.sort();
    indices.dedup();
    assert_eq!(indices.len(), len_before, "BFS should not visit the same node twice");
}

#[test]
fn bfs_forward_test_node_gets_lower_score() {
    let graph = build_test_graph();
    // Start from test_main (4), which calls main (0)
    let results = bfs_forward(&graph, &[4], 1);
    let test_entry = results.iter().find(|e| e.idx == 4).expect("should include seed");
    let main_entry = results.iter().find(|e| e.idx == 0).expect("should include main");
    // test_main is test -> score = (1.0 / 1) * 0.1 = 0.1
    assert!(test_entry.score < 0.5, "test node score should be low: {}", test_entry.score);
    // main is not test -> score = (1.0 / 2) * 1.0 = 0.5
    assert!(main_entry.score > 0.1, "non-test node score at depth 1: {}", main_entry.score);
}

#[test]
fn bfs_forward_empty_seeds() {
    let graph = build_test_graph();
    let results = bfs_forward(&graph, &[], 10);
    assert!(results.is_empty(), "no seeds should yield no results");
}
