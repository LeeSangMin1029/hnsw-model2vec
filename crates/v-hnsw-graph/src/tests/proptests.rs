//! Property-based tests for HnswGraph using proptest.

use proptest::prelude::*;
use v_hnsw_core::VectorIndex;
use crate::distance::L2Distance;

use crate::{HnswConfig, HnswGraph};

/// Generate a deterministic test vector from point id and dimension.
fn make_vector(id: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| (id as f32 * 0.1 + j as f32 * 0.3).sin())
        .collect()
}

/// Build a small graph with `count` vectors of dimension `dim`.
fn build_graph(dim: usize, count: usize, seed: u64) -> (HnswGraph<L2Distance>, Vec<u64>) {
    let config = HnswConfig::builder()
        .dim(dim)
        .m(4)
        .ef_construction(32)
        .max_elements(count + 10)
        .build()
        .expect("valid config in test");

    let mut graph = HnswGraph::with_seed(config, L2Distance, if seed == 0 { 1 } else { seed });
    let mut ids = Vec::with_capacity(count);

    for i in 0..count as u64 {
        graph.insert(i, &make_vector(i, dim)).expect("insert ok");
        ids.push(i);
    }

    (graph, ids)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Inserting a vector then searching for it should find it as nearest (distance ≈ 0).
    #[test]
    fn insert_then_find_self(
        dim in 4usize..=32,
        count in 1usize..=50,
        seed in 1u64..=u64::MAX,
    ) {
        let (graph, ids) = build_graph(dim, count, seed);

        for &id in &ids {
            let query = make_vector(id, dim);
            let results = graph.search(&query, 1, 50).expect("search ok");

            prop_assert!(!results.is_empty(), "search returned no results for id={}", id);
            prop_assert_eq!(
                results[0].0, id,
                "expected id={} as nearest, got id={}",
                id, results[0].0
            );
            prop_assert!(
                results[0].1 < 1e-4,
                "distance to self should be ≈0, got {}",
                results[0].1
            );
        }
    }

    /// Search results must always be sorted by ascending distance.
    #[test]
    fn search_results_sorted(
        dim in 4usize..=32,
        count in 2usize..=50,
        seed in 1u64..=u64::MAX,
    ) {
        let (graph, _) = build_graph(dim, count, seed);

        let query = make_vector(count as u64 + 1, dim); // query not in index
        let k = count.min(10);
        let results = graph.search(&query, k, 50).expect("search ok");

        for i in 0..results.len().saturating_sub(1) {
            prop_assert!(
                results[i].1 <= results[i + 1].1,
                "not sorted: dist[{i}]={} > dist[{}]={}",
                results[i].1,
                i + 1,
                results[i + 1].1
            );
        }
    }

    /// Number of results <= min(k, graph.len()).
    #[test]
    fn search_k_bounded(
        dim in 4usize..=16,
        count in 1usize..=30,
        k in 1usize..=100,
        seed in 1u64..=u64::MAX,
    ) {
        let (graph, _) = build_graph(dim, count, seed);

        let query = make_vector(999, dim);
        let results = graph.search(&query, k, 50).expect("search ok");
        let expected_max = k.min(graph.len());

        prop_assert!(
            results.len() <= expected_max,
            "got {} results but expected <= min(k={k}, len={})={expected_max}",
            results.len(),
            graph.len()
        );
    }

    /// Deleted vectors never appear in search results.
    #[test]
    fn delete_excludes_from_results(
        dim in 4usize..=16,
        count in 3usize..=30,
        seed in 1u64..=u64::MAX,
    ) {
        let (mut graph, ids) = build_graph(dim, count, seed);

        // Delete the first half
        let delete_count = count / 2;
        let deleted: std::collections::HashSet<u64> =
            ids.iter().take(delete_count).copied().collect();

        for &id in &deleted {
            graph.delete(id).expect("delete ok");
        }

        let query = make_vector(0, dim);
        let results = graph.search(&query, count, 100).expect("search ok");

        for (id, _) in &results {
            prop_assert!(
                !deleted.contains(id),
                "deleted id={id} appeared in results"
            );
        }
    }

    /// len() increments by 1 after each insert.
    #[test]
    fn insert_increments_len(
        dim in 4usize..=16,
        count in 1usize..=50,
        seed in 1u64..=u64::MAX,
    ) {
        let config = HnswConfig::builder()
            .dim(dim)
            .m(4)
            .ef_construction(16)
            .max_elements(count + 10)
            .build()
            .expect("valid config");

        let mut graph = HnswGraph::with_seed(config, L2Distance, seed);
        prop_assert_eq!(graph.len(), 0);

        for i in 0..count as u64 {
            let vec = make_vector(i, dim);
            graph.insert(i, &vec).expect("insert ok");
            prop_assert_eq!(
                graph.len(),
                i as usize + 1,
                "len after {} inserts",
                i + 1
            );
        }
    }
}
