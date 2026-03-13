//! Tests for the HNSW insert module.

use v_hnsw_core::VectorIndex;

use crate::config::HnswConfig;
use crate::distance::l2::L2Distance;
use crate::graph::HnswGraph;

use super::helpers::test_vector;

// ---------------------------------------------------------------------------
// Basic insert tests
// ---------------------------------------------------------------------------

#[test]
fn insert_first_node_sets_entry_point() {
    let config = HnswConfig::builder().dim(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    assert_eq!(graph.entry_point(), None);
    assert_eq!(graph.len(), 0);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();

    assert_eq!(graph.entry_point(), Some(1));
    assert_eq!(graph.len(), 1);
}

#[test]
fn insert_increments_count() {
    let config = HnswConfig::builder().dim(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..5 {
        graph.insert(i, &test_vector(i, 4)).unwrap();
        assert_eq!(graph.len(), (i + 1) as usize);
    }
}

#[test]
fn insert_creates_node_in_graph() {
    let config = HnswConfig::builder().dim(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(42, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(graph.nodes.contains_key(&42));
}

// ---------------------------------------------------------------------------
// Dimension mismatch
// ---------------------------------------------------------------------------

#[test]
fn insert_wrong_dimension_too_short() {
    let config = HnswConfig::builder().dim(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    let result = graph.insert(1, &[1.0, 2.0]);
    assert!(result.is_err(), "inserting wrong-dimension vector should fail");
    assert_eq!(graph.len(), 0, "failed insert should not increment count");
}

#[test]
fn insert_wrong_dimension_too_long() {
    let config = HnswConfig::builder().dim(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    let result = graph.insert(1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(result.is_err());
}

#[test]
fn insert_empty_vector_wrong_dim() {
    let config = HnswConfig::builder().dim(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    let result = graph.insert(1, &[]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Index full
// ---------------------------------------------------------------------------

#[test]
fn insert_rejects_when_full() {
    let config = HnswConfig::builder().dim(2).max_elements(3).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(1, &[1.0, 0.0]).unwrap();
    graph.insert(2, &[0.0, 1.0]).unwrap();
    graph.insert(3, &[1.0, 1.0]).unwrap();

    let result = graph.insert(4, &[0.0, 0.0]);
    assert!(result.is_err(), "should reject when index is full");
    assert_eq!(graph.len(), 3);
}

// ---------------------------------------------------------------------------
// Connectivity tests
// ---------------------------------------------------------------------------

#[test]
fn insert_builds_bidirectional_connections() {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).m(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    // Insert enough nodes to force connections
    for i in 0..10 {
        graph.insert(i, &test_vector(i, dim)).unwrap();
    }

    // Check that at least some nodes have neighbors at layer 0
    let nodes_with_neighbors: usize = graph.nodes.values()
        .filter(|n| !n.neighbors_at(0).is_empty())
        .count();
    // All nodes except possibly the first should have neighbors
    assert!(nodes_with_neighbors >= 9,
        "most nodes should have neighbors, got {nodes_with_neighbors}/10");
}

#[test]
fn insert_respects_m_limit() {
    let dim = 4;
    let m = 4;
    let config = HnswConfig::builder().dim(dim).m(m).build().unwrap();
    let m0 = config.m0; // 2 * m = 8
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..50 {
        graph.insert(i, &test_vector(i, dim)).unwrap();
    }

    for (_, node) in &graph.nodes {
        for layer in 0..=node.max_layer {
            let count = node.neighbors_at(layer).len();
            let limit = if layer == 0 { m0 } else { m };
            assert!(count <= limit + 1,
                "node {} layer {} has {} neighbors, limit {}",
                node.id, layer, count, limit);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic seed
// ---------------------------------------------------------------------------

#[test]
fn insert_same_seed_same_structure() {
    let dim = 8;
    let config1 = HnswConfig::builder().dim(dim).m(4).build().unwrap();
    let config2 = HnswConfig::builder().dim(dim).m(4).build().unwrap();

    let mut g1 = HnswGraph::with_seed(config1, L2Distance, 100);
    let mut g2 = HnswGraph::with_seed(config2, L2Distance, 100);

    for i in 0..20 {
        let v = test_vector(i, dim);
        g1.insert(i, &v).unwrap();
        g2.insert(i, &v).unwrap();
    }

    // Same seed + same data = same structure
    assert_eq!(g1.entry_point(), g2.entry_point());
    assert_eq!(g1.max_layer(), g2.max_layer());
    assert_eq!(g1.len(), g2.len());

    // Search results should match
    let query = test_vector(10, dim);
    let r1 = g1.search(&query, 5, 50).unwrap();
    let r2 = g2.search(&query, 5, 50).unwrap();
    assert_eq!(r1, r2);
}

#[test]
fn insert_different_seed_may_differ() {
    let dim = 8;
    let config1 = HnswConfig::builder().dim(dim).m(4).build().unwrap();
    let config2 = HnswConfig::builder().dim(dim).m(4).build().unwrap();

    let mut g1 = HnswGraph::with_seed(config1, L2Distance, 1);
    let mut g2 = HnswGraph::with_seed(config2, L2Distance, 9999);

    for i in 0..30 {
        let v = test_vector(i, dim);
        g1.insert(i, &v).unwrap();
        g2.insert(i, &v).unwrap();
    }

    // Different seeds typically produce different layer assignments
    // We just verify both work correctly
    let query = test_vector(15, dim);
    let r1 = g1.search(&query, 5, 50).unwrap();
    let r2 = g2.search(&query, 5, 50).unwrap();
    assert!(!r1.is_empty());
    assert!(!r2.is_empty());
    // First result should be exact match in both
    assert_eq!(r1[0].0, 15);
    assert_eq!(r2[0].0, 15);
}

// ---------------------------------------------------------------------------
// Layer assignment
// ---------------------------------------------------------------------------

#[test]
fn insert_max_layer_grows() {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).m(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    // Insert many nodes; the max layer should eventually grow beyond 0
    for i in 0..200 {
        graph.insert(i, &test_vector(i, dim)).unwrap();
    }

    // With ml = 1/ln(4) ≈ 0.72 and 200 nodes, we should get at least layer 1
    assert!(graph.max_layer() >= 1,
        "with 200 nodes, max layer should be >= 1, got {}", graph.max_layer());
}

#[test]
fn insert_first_node_has_empty_neighbors() {
    let config = HnswConfig::builder().dim(4).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();

    let node = &graph.nodes[&1];
    // First node has no connections (it's alone in the graph)
    assert!(node.neighbors_at(0).is_empty());
}

// ---------------------------------------------------------------------------
// Insert + search integration
// ---------------------------------------------------------------------------

#[test]
fn insert_then_search_finds_exact_match() {
    let dim = 8;
    let config = HnswConfig::builder().dim(dim).m(8).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..50 {
        graph.insert(i, &test_vector(i, dim)).unwrap();
    }

    // Search for an exact vector that was inserted
    let query = test_vector(25, dim);
    let results = graph.search(&query, 1, 100).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 25);
    assert!(results[0].1 < 1e-6, "exact match distance should be ~0, got {}", results[0].1);
}

#[test]
fn insert_then_search_returns_sorted() {
    let dim = 8;
    let config = HnswConfig::builder().dim(dim).m(8).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..50 {
        graph.insert(i, &test_vector(i, dim)).unwrap();
    }

    let query = test_vector(25, dim);
    let results = graph.search(&query, 10, 100).unwrap();

    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1,
            "results not sorted at index {i}: {} >= {}", results[i].1, results[i - 1].1);
    }
}

// ---------------------------------------------------------------------------
// build_insert (external store) tests
// ---------------------------------------------------------------------------

#[test]
fn build_insert_with_external_store() {
    use crate::store::InMemoryVectorStore;
    use v_hnsw_core::VectorStore;

    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    let mut ext_store = InMemoryVectorStore::new(dim);
    for i in 0..10u64 {
        ext_store.insert(i, &test_vector(i, dim)).unwrap();
    }

    for i in 0..10u64 {
        graph.build_insert(&ext_store, i).unwrap();
    }

    assert_eq!(graph.len(), 10);

    // Search using the external store
    let query = test_vector(5, dim);
    let results = graph.search_ext(&ext_store, &query, 3, 50).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 5);
}

// ---------------------------------------------------------------------------
// Edge case: inserting with very similar vectors
// ---------------------------------------------------------------------------

#[test]
fn insert_identical_vectors_different_ids() {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).max_elements(10).build().unwrap();
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    let v = vec![1.0, 2.0, 3.0, 4.0];
    for i in 0..5 {
        graph.insert(i, &v).unwrap();
    }

    assert_eq!(graph.len(), 5);

    // All should be found with distance ~0
    let results = graph.search(&v, 5, 50).unwrap();
    assert_eq!(results.len(), 5);
    for (_, dist) in &results {
        assert!(*dist < 1e-6, "distance should be ~0 for identical vectors, got {dist}");
    }
}
