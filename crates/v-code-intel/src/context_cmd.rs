//! Unified context gathering for a symbol.
//! Collects definition, callers, callees, related types, and tests in one pass.

use crate::bfs::{bfs_generic, BfsDirection};
use crate::graph::CallGraph;
use crate::parse::CodeChunk;

/// A single entry in the context result (definition, caller, callee, type, or test).
pub struct ContextEntry {
    pub idx: u32,
    pub depth: u32,
}

/// Complete context for a symbol: definition + callers + callees + types + tests.
pub struct ContextResult {
    /// Seed chunk indices (the symbol's own definitions).
    pub seeds: Vec<u32>,
    /// Callers found via reverse BFS (excludes seeds).
    pub callers: Vec<ContextEntry>,
    /// Callees found via forward BFS (excludes seeds).
    pub callees: Vec<ContextEntry>,
    /// Chunk indices of types referenced by the seed chunks.
    pub types: Vec<u32>,
    /// Chunk indices of test functions that call the symbol.
    pub tests: Vec<u32>,
}

/// Build unified context for a symbol.
///
/// Resolves `symbol` via `graph.resolve()`, then collects callers (reverse BFS),
/// callees (forward BFS), referenced types, and test functions — all in one pass.
pub fn build_context(
    graph: &CallGraph,
    chunks: &[CodeChunk],
    symbol: &str,
    depth: u32,
) -> ContextResult {
    let seeds = graph.resolve(symbol);

    // Callers: reverse BFS, exclude seeds themselves.
    let callers = {
        let all = bfs_generic(graph, &seeds, depth, BfsDirection::Reverse, |idx, d| {
            Some(ContextEntry { idx, depth: d })
        });
        all.into_iter().filter(|e| e.depth > 0).collect()
    };

    // Callees: forward BFS, exclude seeds themselves.
    let callees = {
        let all = bfs_generic(graph, &seeds, depth, BfsDirection::Forward, |idx, d| {
            Some(ContextEntry { idx, depth: d })
        });
        all.into_iter().filter(|e| e.depth > 0).collect()
    };

    // Types: collect type names from seed chunks' `types` field, resolve to chunk indices.
    let types = collect_types(graph, chunks, &seeds);

    // Tests: find test chunks that call the symbol.
    let tests = collect_tests(graph, &seeds);

    ContextResult { seeds, callers, callees, types, tests }
}

/// Resolve type names from seed chunks to chunk indices.
fn collect_types(graph: &CallGraph, chunks: &[CodeChunk], seeds: &[u32]) -> Vec<u32> {
    let mut type_indices = Vec::new();
    let mut seen = vec![false; graph.len()];

    // Mark seeds as seen so they don't appear in types.
    for &s in seeds {
        if (s as usize) < seen.len() {
            seen[s as usize] = true;
        }
    }

    for &seed in seeds {
        let seed_usize = seed as usize;
        if seed_usize >= chunks.len() {
            continue;
        }
        let chunk = &chunks[seed_usize];
        for type_name in &chunk.types {
            let resolved = graph.resolve(type_name);
            for idx in resolved {
                let idx_usize = idx as usize;
                if idx_usize < seen.len() && !seen[idx_usize] {
                    seen[idx_usize] = true;
                    type_indices.push(idx);
                }
            }
        }
    }

    type_indices
}

/// Find test chunks that directly call any seed.
fn collect_tests(graph: &CallGraph, seeds: &[u32]) -> Vec<u32> {
    let mut test_indices = Vec::new();
    let mut seen = vec![false; graph.len()];

    for &seed in seeds {
        if (seed as usize) < seen.len() {
            seen[seed as usize] = true;
        }
    }

    for &seed in seeds {
        let seed_usize = seed as usize;
        if seed_usize >= graph.callers.len() {
            continue;
        }
        for &caller_idx in &graph.callers[seed_usize] {
            let caller_usize = caller_idx as usize;
            if caller_usize < graph.is_test.len()
                && graph.is_test[caller_usize]
                && !seen[caller_usize]
            {
                seen[caller_usize] = true;
                test_indices.push(caller_idx);
            }
        }
    }

    test_indices
}
