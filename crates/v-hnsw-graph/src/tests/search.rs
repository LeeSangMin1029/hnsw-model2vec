//! Tests for the HNSW search module (search_layer, greedy_closest, search_with_store).

use std::collections::HashMap;

use v_hnsw_core::{DistanceMetric, PointId, VectorStore};

use crate::config::HnswConfig;
use crate::distance::l2::L2Distance;
use crate::node::Node;
use crate::search::{greedy_closest, search_layer, search_with_store, NodeGraph};
use crate::store::InMemoryVectorStore;

// ---------------------------------------------------------------------------
// Helper: build a small graph manually for unit-testing search internals
// ---------------------------------------------------------------------------

/// Create a store with vectors and a node map with specified connections.
fn make_test_graph(
    dim: usize,
    vectors: &[(PointId, Vec<f32>)],
    edges: &[(PointId, u8, Vec<PointId>)], // (node_id, max_layer, neighbors_at_layer_0)
) -> (HashMap<PointId, Node>, InMemoryVectorStore) {
    let mut store = InMemoryVectorStore::new(dim);
    let mut nodes = HashMap::new();

    for (id, vec) in vectors {
        store.insert(*id, vec).unwrap();
    }

    for (id, max_layer, neighbors) in edges {
        let mut node = Node::new(*id, *max_layer);
        node.set_neighbors(0, neighbors.clone());
        nodes.insert(*id, node);
    }

    (nodes, store)
}

// ---------------------------------------------------------------------------
// NodeGraph trait tests
// ---------------------------------------------------------------------------

#[test]
fn node_graph_neighbors_existing() {
    let mut nodes = HashMap::new();
    let mut node = Node::new(1, 0);
    node.set_neighbors(0, vec![2, 3, 4]);
    nodes.insert(1, node);

    let result = nodes.neighbors(1, 0);
    assert_eq!(result, Some(&[2, 3, 4][..]));
}

#[test]
fn node_graph_neighbors_missing_node() {
    let nodes: HashMap<PointId, Node> = HashMap::new();
    assert_eq!(nodes.neighbors(999, 0), None);
}

#[test]
fn node_graph_neighbors_layer_too_high() {
    let mut nodes = HashMap::new();
    let node = Node::new(1, 0); // max_layer = 0
    nodes.insert(1, node);

    // Layer 1 doesn't exist for this node, should return empty slice
    let result = nodes.neighbors(1, 1);
    assert_eq!(result, Some(&[][..]));
}

#[test]
fn node_graph_is_deleted_missing() {
    let nodes: HashMap<PointId, Node> = HashMap::new();
    assert!(nodes.is_deleted(999));
}

#[test]
fn node_graph_is_deleted_false() {
    let mut nodes = HashMap::new();
    nodes.insert(1, Node::new(1, 0));
    assert!(!nodes.is_deleted(1));
}

#[test]
fn node_graph_is_deleted_true() {
    let mut nodes = HashMap::new();
    let mut node = Node::new(1, 0);
    node.deleted = true;
    nodes.insert(1, node);
    assert!(nodes.is_deleted(1));
}

// ---------------------------------------------------------------------------
// greedy_closest tests
// ---------------------------------------------------------------------------

#[test]
fn greedy_closest_no_neighbors() {
    let dim = 3;
    let (nodes, store) = make_test_graph(
        dim,
        &[(1, vec![1.0, 0.0, 0.0])],
        &[(1, 0, vec![])],
    );

    let query = vec![0.0, 1.0, 0.0];
    let dist = L2Distance.distance(&query, store.get(1).unwrap());
    let (best_id, best_dist) = greedy_closest(&nodes, &store, &L2Distance, &query, 1, dist, 0).unwrap();

    assert_eq!(best_id, 1);
    assert!((best_dist - dist).abs() < 1e-6);
}

#[test]
fn greedy_closest_finds_closer_neighbor() {
    let dim = 2;
    let (nodes, store) = make_test_graph(
        dim,
        &[
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 0.0]),
            (3, vec![0.5, 0.1]), // closest to query
        ],
        &[
            (1, 0, vec![2, 3]),
            (2, 0, vec![1, 3]),
            (3, 0, vec![1, 2]),
        ],
    );

    let query = vec![0.5, 0.0];
    let start_dist = L2Distance.distance(&query, store.get(1).unwrap());
    let (best_id, _best_dist) = greedy_closest(&nodes, &store, &L2Distance, &query, 1, start_dist, 0).unwrap();

    // Should find node 2 (at [1.0, 0.0], dist=0.25) or node 3 (at [0.5, 0.1], dist=0.01)
    // Node 3 is closest
    assert_eq!(best_id, 3);
}

#[test]
fn greedy_closest_skips_deleted_nodes() {
    let dim = 2;
    let mut store = InMemoryVectorStore::new(dim);
    let mut nodes = HashMap::new();

    store.insert(1, &[0.0, 0.0]).unwrap();
    store.insert(2, &[0.9, 0.0]).unwrap(); // closest to query, but deleted
    store.insert(3, &[0.8, 0.0]).unwrap(); // second closest, not deleted

    let mut n1 = Node::new(1, 0);
    n1.set_neighbors(0, vec![2, 3]);
    nodes.insert(1, n1);

    let mut n2 = Node::new(2, 0);
    n2.deleted = true;
    n2.set_neighbors(0, vec![1, 3]);
    nodes.insert(2, n2);

    let mut n3 = Node::new(3, 0);
    n3.set_neighbors(0, vec![1, 2]);
    nodes.insert(3, n3);

    let query = vec![1.0, 0.0];
    let start_dist = L2Distance.distance(&query, store.get(1).unwrap()); // dist = 1.0
    let (best_id, _) = greedy_closest(&nodes, &store, &L2Distance, &query, 1, start_dist, 0).unwrap();

    // Node 2 at [0.9, 0.0] is closest but deleted, so should find node 3 at [0.8, 0.0]
    assert_eq!(best_id, 3);
}

// ---------------------------------------------------------------------------
// search_layer tests
// ---------------------------------------------------------------------------

#[test]
fn search_layer_single_node() {
    let dim = 3;
    let (nodes, store) = make_test_graph(
        dim,
        &[(1, vec![1.0, 0.0, 0.0])],
        &[(1, 0, vec![])],
    );

    let query = vec![1.0, 0.0, 0.0];
    let dist = L2Distance.distance(&query, store.get(1).unwrap());
    let entry_points = vec![(1, dist)];
    let results = search_layer(&nodes, &store, &L2Distance, &query, &entry_points, 10, 0).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 1e-6);
}

#[test]
fn search_layer_finds_nearest() {
    let dim = 2;
    let (nodes, store) = make_test_graph(
        dim,
        &[
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 0.0]),
            (3, vec![2.0, 0.0]),
            (4, vec![3.0, 0.0]),
        ],
        &[
            (1, 0, vec![2]),
            (2, 0, vec![1, 3]),
            (3, 0, vec![2, 4]),
            (4, 0, vec![3]),
        ],
    );

    let query = vec![2.1, 0.0];
    let dist = L2Distance.distance(&query, store.get(1).unwrap());
    let entry_points = vec![(1, dist)];
    let results = search_layer(&nodes, &store, &L2Distance, &query, &entry_points, 10, 0).unwrap();

    // Sort results by distance
    let mut sorted = results;
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Node 3 (at [2.0, 0.0]) should be closest to query [2.1, 0.0]
    assert_eq!(sorted[0].0, 3);
}

#[test]
fn search_layer_respects_ef() {
    let dim = 2;
    let (nodes, store) = make_test_graph(
        dim,
        &[
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 0.0]),
            (3, vec![2.0, 0.0]),
            (4, vec![3.0, 0.0]),
            (5, vec![4.0, 0.0]),
        ],
        &[
            (1, 0, vec![2]),
            (2, 0, vec![1, 3]),
            (3, 0, vec![2, 4]),
            (4, 0, vec![3, 5]),
            (5, 0, vec![4]),
        ],
    );

    let query = vec![2.5, 0.0];
    let dist = L2Distance.distance(&query, store.get(1).unwrap());
    let entry_points = vec![(1, dist)];

    // ef=2: should return at most 2 results
    let results = search_layer(&nodes, &store, &L2Distance, &query, &entry_points, 2, 0).unwrap();
    assert!(results.len() <= 2, "ef=2 should limit results to at most 2, got {}", results.len());
}

#[test]
fn search_layer_skips_deleted() {
    let dim = 2;
    let mut store = InMemoryVectorStore::new(dim);
    let mut nodes = HashMap::new();

    store.insert(1, &[0.0, 0.0]).unwrap();
    store.insert(2, &[1.0, 0.0]).unwrap();
    store.insert(3, &[2.0, 0.0]).unwrap();

    let mut n1 = Node::new(1, 0);
    n1.set_neighbors(0, vec![2]);
    nodes.insert(1, n1);

    let mut n2 = Node::new(2, 0);
    n2.deleted = true;
    n2.set_neighbors(0, vec![1, 3]);
    nodes.insert(2, n2);

    let mut n3 = Node::new(3, 0);
    n3.set_neighbors(0, vec![2]);
    nodes.insert(3, n3);

    let query = vec![1.0, 0.0];
    let dist = L2Distance.distance(&query, store.get(1).unwrap());
    let entry_points = vec![(1, dist)];
    let results = search_layer(&nodes, &store, &L2Distance, &query, &entry_points, 10, 0).unwrap();

    // Node 2 is deleted, should not appear in results
    for (id, _) in &results {
        assert_ne!(*id, 2, "deleted node should not appear in results");
    }
}

#[test]
fn search_layer_empty_entry_points() {
    let dim = 2;
    let (nodes, store) = make_test_graph(
        dim,
        &[(1, vec![0.0, 0.0])],
        &[(1, 0, vec![])],
    );

    let query = vec![0.0, 0.0];
    let entry_points: Vec<(PointId, f32)> = vec![];
    let results = search_layer(&nodes, &store, &L2Distance, &query, &entry_points, 10, 0).unwrap();
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// search_with_store tests
// ---------------------------------------------------------------------------

#[test]
fn search_with_store_dimension_mismatch() {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let store = InMemoryVectorStore::new(dim);
    let nodes: HashMap<PointId, Node> = HashMap::new();

    let query = vec![1.0, 2.0]; // wrong dimension
    let result = search_with_store(&nodes, &store, &L2Distance, &config, None, 0, &query, 5, 50);
    assert!(result.is_err());
}

#[test]
fn search_with_store_no_entry_point() {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let store = InMemoryVectorStore::new(dim);
    let nodes: HashMap<PointId, Node> = HashMap::new();

    let query = vec![1.0, 2.0, 3.0, 4.0];
    let result = search_with_store(&nodes, &store, &L2Distance, &config, None, 0, &query, 5, 50).unwrap();
    assert!(result.is_empty(), "no entry point should return empty results");
}

#[test]
fn search_with_store_single_node() {
    let dim = 3;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let (nodes, store) = make_test_graph(
        dim,
        &[(1, vec![1.0, 0.0, 0.0])],
        &[(1, 0, vec![])],
    );

    let query = vec![1.0, 0.0, 0.0];
    let results = search_with_store(&nodes, &store, &L2Distance, &config, Some(1), 0, &query, 5, 50).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 1e-6);
}

#[test]
fn search_with_store_k_zero() {
    let dim = 3;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let (nodes, store) = make_test_graph(
        dim,
        &[(1, vec![1.0, 0.0, 0.0])],
        &[(1, 0, vec![])],
    );

    let query = vec![1.0, 0.0, 0.0];
    let results = search_with_store(&nodes, &store, &L2Distance, &config, Some(1), 0, &query, 0, 50).unwrap();
    assert!(results.is_empty(), "k=0 should return empty results");
}

#[test]
fn search_with_store_results_sorted() {
    let dim = 2;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let (nodes, store) = make_test_graph(
        dim,
        &[
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 0.0]),
            (3, vec![2.0, 0.0]),
            (4, vec![3.0, 0.0]),
        ],
        &[
            (1, 0, vec![2]),
            (2, 0, vec![1, 3]),
            (3, 0, vec![2, 4]),
            (4, 0, vec![3]),
        ],
    );

    let query = vec![1.5, 0.0];
    let results = search_with_store(&nodes, &store, &L2Distance, &config, Some(1), 0, &query, 4, 50).unwrap();

    // Results should be sorted by ascending distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1,
            "results not sorted: {:?}", results);
    }
}

// ---------------------------------------------------------------------------
// search_two_stage tests (DistanceComputer-based)
// ---------------------------------------------------------------------------

/// A simple DistanceComputer wrapping an InMemoryVectorStore + L2Distance.
struct TestDc<'a> {
    store: &'a InMemoryVectorStore,
}

impl crate::search::DistanceComputer for TestDc<'_> {
    fn distance(&self, query: &[f32], id: PointId) -> v_hnsw_core::Result<f32> {
        let vec = self.store.get(id)?;
        Ok(L2Distance.distance(query, vec))
    }
}

/// A "noisy" DistanceComputer that adds small random noise to simulate quantized distance.
struct NoisyDc<'a> {
    store: &'a InMemoryVectorStore,
    noise: f32,
}

impl crate::search::DistanceComputer for NoisyDc<'_> {
    fn distance(&self, query: &[f32], id: PointId) -> v_hnsw_core::Result<f32> {
        let vec = self.store.get(id)?;
        let exact = L2Distance.distance(query, vec);
        // Deterministic noise based on id to simulate quantization error
        let noise = self.noise * ((id % 7) as f32 / 7.0 - 0.5);
        Ok((exact + noise).max(0.0))
    }
}

#[test]
fn two_stage_same_dc() {
    // When approx == exact, should match search_with_store
    let dim = 2;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let (nodes, store) = make_test_graph(
        dim,
        &[
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 0.0]),
            (3, vec![2.0, 0.0]),
            (4, vec![3.0, 0.0]),
        ],
        &[
            (1, 0, vec![2]),
            (2, 0, vec![1, 3]),
            (3, 0, vec![2, 4]),
            (4, 0, vec![3]),
        ],
    );

    let query = vec![0.8, 0.0]; // closest to node 2 at [1.0, 0.0]
    let dc = TestDc { store: &store };

    let results = crate::search::search_two_stage(
        &nodes, &dc, &dc, &config, Some(1), 0, &query, 4, 50,
    ).unwrap();

    assert!(!results.is_empty());
    // Results sorted by ascending distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }
    // Node 1 at [0.0, 0.0] dist=0.64, Node 2 at [1.0, 0.0] dist=0.04 → Node 2 closest
    assert_eq!(results[0].0, 2);
}

#[test]
fn two_stage_noisy_approx_rescores_correctly() {
    // Noisy approx may reorder candidates, but exact rescore should fix ranking
    let dim = 2;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let (nodes, store) = make_test_graph(
        dim,
        &[
            (1, vec![0.0, 0.0]),
            (2, vec![1.0, 0.0]),
            (3, vec![2.0, 0.0]),
            (4, vec![3.0, 0.0]),
        ],
        &[
            (1, 0, vec![2]),
            (2, 0, vec![1, 3]),
            (3, 0, vec![2, 4]),
            (4, 0, vec![3]),
        ],
    );

    let query = vec![0.8, 0.0];
    let approx = NoisyDc { store: &store, noise: 0.1 };
    let exact = TestDc { store: &store };

    let results = crate::search::search_two_stage(
        &nodes, &approx, &exact, &config, Some(1), 0, &query, 4, 50,
    ).unwrap();

    // After rescore, results should be sorted by exact distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1,
            "rescored results not sorted: {:?}", results);
    }
    // Node 2 at [1.0, 0.0] is closest by exact distance (0.04)
    assert_eq!(results[0].0, 2);
}

#[test]
fn two_stage_no_entry_point() {
    let dim = 2;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let store = InMemoryVectorStore::new(dim);
    let nodes: HashMap<PointId, Node> = HashMap::new();
    let dc = TestDc { store: &store };

    let results = crate::search::search_two_stage(
        &nodes, &dc, &dc, &config, None, 0, &[0.0, 0.0], 5, 50,
    ).unwrap();
    assert!(results.is_empty());
}

#[test]
fn two_stage_dimension_mismatch() {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let store = InMemoryVectorStore::new(dim);
    let nodes: HashMap<PointId, Node> = HashMap::new();
    let dc = TestDc { store: &store };

    let result = crate::search::search_two_stage(
        &nodes, &dc, &dc, &config, None, 0, &[1.0, 2.0], 5, 50,
    );
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Multi-layer search test
// ---------------------------------------------------------------------------

#[test]
fn search_with_store_multilayer() {
    let dim = 2;
    let config = HnswConfig::builder().dim(dim).build().unwrap();
    let mut store = InMemoryVectorStore::new(dim);
    let mut nodes = HashMap::new();

    // Node 1: exists at layer 0 and layer 1 (entry point)
    store.insert(1, &[0.0, 0.0]).unwrap();
    let mut n1 = Node::new(1, 1);
    n1.set_neighbors(0, vec![2, 3]);
    n1.set_neighbors(1, vec![4]); // layer 1 neighbor
    nodes.insert(1, n1);

    // Node 2: layer 0 only
    store.insert(2, &[1.0, 0.0]).unwrap();
    let mut n2 = Node::new(2, 0);
    n2.set_neighbors(0, vec![1, 3]);
    nodes.insert(2, n2);

    // Node 3: layer 0 only
    store.insert(3, &[0.5, 0.5]).unwrap();
    let mut n3 = Node::new(3, 0);
    n3.set_neighbors(0, vec![1, 2]);
    nodes.insert(3, n3);

    // Node 4: layer 0 and layer 1
    store.insert(4, &[2.0, 0.0]).unwrap();
    let mut n4 = Node::new(4, 1);
    n4.set_neighbors(0, vec![2]);
    n4.set_neighbors(1, vec![1]);
    nodes.insert(4, n4);

    let query = vec![0.9, 0.0];
    let results = search_with_store(
        &nodes, &store, &L2Distance, &config,
        Some(1), 1, // entry_point=1, max_layer=1
        &query, 3, 50,
    ).unwrap();

    assert!(!results.is_empty());
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }
    // Node 2 at [1.0, 0.0] is closest to query [0.9, 0.0]
    assert_eq!(results[0].0, 2);
}
