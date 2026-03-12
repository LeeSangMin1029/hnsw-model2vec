use crate::commands::intel::gather::{
    bfs_directed, merge_entries, GatherEntry, Direction,
};
use crate::commands::intel::{format_lines_opt as format_lines, format_lines_str_opt as format_lines_str};
use crate::commands::intel::gather::Direction::{Forward, Reverse};
use crate::commands::intel::graph::CallGraph;
use crate::commands::intel::parse::CodeChunk;

/// Convenience wrapper for tests: forward BFS.
fn bfs_forward(graph: &CallGraph, seeds: &[u32], depth: u32, include_tests: bool) -> Vec<GatherEntry> {
    bfs_directed(graph, seeds, depth, include_tests, Direction::Forward)
}

/// Convenience wrapper for tests: reverse BFS.
fn bfs_reverse(graph: &CallGraph, seeds: &[u32], depth: u32, include_tests: bool) -> Vec<GatherEntry> {
    bfs_directed(graph, seeds, depth, include_tests, Direction::Reverse)
}

fn chunk(name: &str, file: &str, calls: &[&str]) -> CodeChunk {
    CodeChunk {
        kind: "function".to_owned(),
        name: name.to_owned(),
        file: file.to_owned(),
        lines: Some((1, 10)),
        signature: Some(format!("fn {name}()")),
        calls: calls.iter().map(|s| s.to_string()).collect(),
        types: vec![],
    }
}

fn test_chunk(name: &str, file: &str, calls: &[&str]) -> CodeChunk {
    CodeChunk {
        kind: "function".to_owned(),
        name: name.to_owned(),
        file: format!("src/tests/{file}"),
        lines: Some((1, 10)),
        signature: Some(format!("fn {name}()")),
        calls: calls.iter().map(|s| s.to_string()).collect(),
        types: vec![],
    }
}

// ── Direction::label ────────────────────────────────────────────────

#[test]
fn direction_label_forward() {
    assert_eq!(Forward.label(), "callee");
}

#[test]
fn direction_label_reverse() {
    assert_eq!(Reverse.label(), "caller");
}

// ── bfs_forward ─────────────────────────────────────────────────────

#[test]
fn bfs_forward_empty_graph() {
    let graph = CallGraph::build(&[]);
    let results = bfs_forward(&graph, &[], 3, true);
    assert!(results.is_empty());
}

#[test]
fn bfs_forward_single_node() {
    let chunks = vec![chunk("A", "src/a.rs", &[])];
    let graph = CallGraph::build(&chunks);
    let results = bfs_forward(&graph, &[0], 3, true);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].idx, 0);
    assert_eq!(results[0].depth, 0);
}

#[test]
fn bfs_forward_chain() {
    // A -> B -> C
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["C"]),
        chunk("C", "src/c.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);
    let results = bfs_forward(&graph, &[0], 5, true);

    assert_eq!(results.len(), 3);
    let idxs: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(idxs.contains(&0));
    assert!(idxs.contains(&1));
    assert!(idxs.contains(&2));
}

#[test]
fn bfs_forward_depth_limit() {
    // A -> B -> C, depth=1 should not reach C
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["C"]),
        chunk("C", "src/c.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);
    let results = bfs_forward(&graph, &[0], 1, true);

    let idxs: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(idxs.contains(&0));
    assert!(idxs.contains(&1));
    assert!(!idxs.contains(&2), "C should not be reached at depth 1");
}

#[test]
fn bfs_forward_filters_test_nodes() {
    // A -> T (test node)
    let chunks = vec![
        chunk("A", "src/a.rs", &["T"]),
        test_chunk("T", "t.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    // include_tests = false: test node visited but skipped
    let results = bfs_forward(&graph, &[0], 3, false);
    let idxs: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(idxs.contains(&0));
    assert!(!idxs.contains(&1), "test node should be filtered out");

    // include_tests = true: test node included
    let results = bfs_forward(&graph, &[0], 3, true);
    assert_eq!(results.len(), 2);
}

#[test]
fn bfs_forward_scores_decrease_with_depth() {
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["C"]),
        chunk("C", "src/c.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);
    let results = bfs_forward(&graph, &[0], 5, true);

    // depth 0 -> score 1.0, depth 1 -> 0.5, depth 2 -> 0.333...
    let d0 = results.iter().find(|e| e.depth == 0).unwrap();
    let d1 = results.iter().find(|e| e.depth == 1).unwrap();
    let d2 = results.iter().find(|e| e.depth == 2).unwrap();
    assert!(d0.score > d1.score);
    assert!(d1.score > d2.score);
}

#[test]
fn bfs_forward_test_nodes_get_lower_score() {
    // A -> T (test), A -> B (not test), both at depth 1
    let chunks = vec![
        chunk("A", "src/a.rs", &["B", "T"]),
        chunk("B", "src/b.rs", &[]),
        test_chunk("T", "t.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);
    let results = bfs_forward(&graph, &[0], 3, true);

    let b_entry = results.iter().find(|e| e.idx == 1).unwrap();
    let t_entry = results.iter().find(|e| e.idx == 2).unwrap();
    assert!(b_entry.score > t_entry.score, "test node should have lower score");
}

// ── bfs_reverse ─────────────────────────────────────────────────────

#[test]
fn bfs_reverse_follows_callers() {
    // A -> B -> C, reverse from C should find B and A
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["C"]),
        chunk("C", "src/c.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);
    let results = bfs_reverse(&graph, &[2], 5, true);

    let idxs: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(idxs.contains(&2)); // seed
    assert!(idxs.contains(&1)); // B calls C
    assert!(idxs.contains(&0)); // A calls B
}

#[test]
fn bfs_reverse_depth_limit() {
    // A -> B -> C, reverse from C with depth=1 should reach B but not A
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["C"]),
        chunk("C", "src/c.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);
    let results = bfs_reverse(&graph, &[2], 1, true);

    let idxs: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(idxs.contains(&2));
    assert!(idxs.contains(&1));
    assert!(!idxs.contains(&0), "A should not be reached at depth 1");
}

#[test]
fn bfs_reverse_filters_test_nodes() {
    // T (test) -> A, reverse from A
    let chunks = vec![
        test_chunk("T", "t.rs", &["A"]),
        chunk("A", "src/a.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    let results = bfs_reverse(&graph, &[1], 3, false);
    let idxs: Vec<u32> = results.iter().map(|e| e.idx).collect();
    assert!(idxs.contains(&1));
    assert!(!idxs.contains(&0), "test caller should be filtered");
}

#[test]
fn bfs_reverse_direction_is_reverse() {
    let chunks = vec![chunk("A", "src/a.rs", &[])];
    let graph = CallGraph::build(&chunks);
    let results = bfs_reverse(&graph, &[0], 1, true);
    assert_eq!(results[0].direction, Reverse);
}

// ── merge_entries ───────────────────────────────────────────────────

#[test]
fn merge_deduplicates_keeping_higher_score() {
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["A"]),
    ];
    let graph = CallGraph::build(&chunks);

    let forward = bfs_forward(&graph, &[0], 1, true);
    let reverse = bfs_reverse(&graph, &[0], 1, true);

    // Both contain idx 0 (seed, score 1.0) and idx 1 (depth 1, score 0.5)
    let merged = merge_entries(forward, reverse);
    assert_eq!(merged.len(), 2);

    // Each idx appears exactly once
    let idxs: Vec<u32> = merged.iter().map(|e| e.idx).collect();
    assert!(idxs.contains(&0));
    assert!(idxs.contains(&1));
}

#[test]
fn merge_sorted_by_score_descending() {
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["C"]),
        chunk("C", "src/c.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    let forward = bfs_forward(&graph, &[0], 5, true);
    let merged = merge_entries(forward, vec![]);

    for w in merged.windows(2) {
        assert!(w[0].score >= w[1].score, "should be sorted by score descending");
    }
}

#[test]
fn merge_empty_inputs() {
    let merged = merge_entries(vec![], vec![]);
    assert!(merged.is_empty());
}

// ── format_lines / format_lines_str ─────────────────────────────────

#[test]
fn format_lines_with_range() {
    assert_eq!(format_lines(Some((10, 20))), ":10-20");
}

#[test]
fn format_lines_none() {
    assert_eq!(format_lines(None), "");
}

#[test]
fn format_lines_str_with_range() {
    assert_eq!(format_lines_str(Some((5, 15))), "5-15");
}

#[test]
fn format_lines_str_none() {
    assert_eq!(format_lines_str(None), "");
}

// ── cycle handling ──────────────────────────────────────────────────

#[test]
fn bfs_forward_handles_cycle() {
    // A -> B -> C -> A (cycle)
    let chunks = vec![
        chunk("A", "src/a.rs", &["B"]),
        chunk("B", "src/b.rs", &["C"]),
        chunk("C", "src/c.rs", &["A"]),
    ];
    let graph = CallGraph::build(&chunks);
    let results = bfs_forward(&graph, &[0], 10, true);

    // Should visit each node exactly once despite cycle
    assert_eq!(results.len(), 3);
}
