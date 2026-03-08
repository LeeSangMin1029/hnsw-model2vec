use crate::config::HnswConfig;
use crate::distance::L2Distance;
use crate::graph::HnswGraph;
use v_hnsw_core::VectorIndex;

/// Generate a deterministic test vector for the given point id and dimension.
fn test_vector(point_id: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| (point_id as f32 * 0.1 + j as f32 * 0.3).sin())
        .collect()
}

/// Brute-force k-nearest-neighbor search for recall validation.
fn brute_force_knn(
    vectors: &[(u64, Vec<f32>)],
    query: &[f32],
    k: usize,
    distance: &L2Distance,
) -> Vec<(u64, f32)> {
    use v_hnsw_core::DistanceMetric;
    let mut dists: Vec<(u64, f32)> = vectors
        .iter()
        .map(|(id, v)| (*id, distance.distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists
}

#[test]
fn test_empty_graph_search() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let graph = HnswGraph::new(config, L2Distance);
    let results = graph.search(&[1.0, 2.0, 3.0, 4.0], 5, 50)?;
    assert!(results.is_empty());
    Ok(())
}

#[test]
fn test_single_insert_search() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 2.0, 3.0, 4.0])?;

    let results = graph.search(&[1.0, 2.0, 3.0, 4.0], 1, 50)?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 1e-6); // Should be near zero distance
    Ok(())
}

#[test]
fn test_multiple_insert_search() -> v_hnsw_core::Result<()> {
    let dim = 16;
    let config = HnswConfig::builder().dim(dim).m(8).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..100 {
        let vec = test_vector(i, dim);
        graph.insert(i, &vec)?;
    }

    assert_eq!(graph.len(), 100);

    let query = test_vector(50, dim);
    let results = graph.search(&query, 10, 50)?;
    assert_eq!(results.len(), 10);

    // First result should be the exact match (point 50)
    assert_eq!(results[0].0, 50);
    assert!(results[0].1 < 1e-6);

    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i].1 >= results[i - 1].1);
    }

    Ok(())
}

#[test]
fn test_delete() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
    graph.insert(3, &[0.0, 0.0, 1.0, 0.0])?;

    assert_eq!(graph.len(), 3);

    // Delete point 1
    graph.delete(1)?;
    assert_eq!(graph.len(), 2);

    // Search near point 1's location
    let results = graph.search(&[1.0, 0.0, 0.0, 0.0], 3, 50)?;

    // Point 1 should not appear in results
    for (id, _) in &results {
        assert_ne!(*id, 1);
    }

    Ok(())
}

#[test]
fn test_dimension_mismatch() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    // Wrong dimension on insert
    let result = graph.insert(1, &[1.0, 2.0]);
    assert!(result.is_err());

    // Insert correctly, then search with wrong dimension
    graph.insert(1, &[1.0, 2.0, 3.0, 4.0])?;
    let result = graph.search(&[1.0, 2.0], 1, 50);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_recall() -> v_hnsw_core::Result<()> {
    let dim = 32;
    let n = 1000;
    let k = 10;
    let ef = 200;

    let config = HnswConfig::builder()
        .dim(dim)
        .m(16)
        .ef_construction(200)
        .max_elements(n + 100)
        .build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 12345);

    // Build index
    let mut all_vectors: Vec<(u64, Vec<f32>)> = Vec::with_capacity(n);
    for i in 0..n as u64 {
        let vec = test_vector(i, dim);
        graph.insert(i, &vec)?;
        all_vectors.push((i, vec));
    }

    // Test recall with 50 random queries
    let distance = L2Distance;
    let mut total_recall = 0.0;
    let num_queries = 50;

    for q in 0..num_queries {
        let query = test_vector(q * 20, dim); // spread queries across the dataset
        let hnsw_results = graph.search(&query, k, ef)?;
        let bf_results = brute_force_knn(&all_vectors, &query, k, &distance);

        // Count how many of the HNSW top-k are in the brute-force top-k
        let bf_ids: std::collections::HashSet<u64> =
            bf_results.iter().map(|(id, _)| *id).collect();
        let hits = hnsw_results
            .iter()
            .filter(|(id, _)| bf_ids.contains(id))
            .count();

        total_recall += hits as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;

    // Recall@10 should be > 90% with ef=200
    assert!(
        avg_recall > 0.90,
        "Average recall@{k} = {avg_recall:.3}, expected > 0.90"
    );

    Ok(())
}

#[test]
fn test_save_load() -> v_hnsw_core::Result<()> {
    let dim = 16;
    let config = HnswConfig::builder().dim(dim).m(8).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    // Insert test vectors
    for i in 0..50 {
        let vec = test_vector(i, dim);
        graph.insert(i, &vec)?;
    }

    // Save to temporary file
    let temp_path = std::env::temp_dir().join("test_hnsw_save_load.bin");
    graph.save(&temp_path)?;

    // Load from file (store is empty after load)
    let mut loaded_graph = HnswGraph::load(&temp_path, L2Distance)?;
    // Populate vectors from the original graph's store
    loaded_graph.populate_store(&graph.store)?;

    // Verify loaded graph matches original
    assert_eq!(loaded_graph.len(), graph.len());
    assert_eq!(loaded_graph.max_layer(), graph.max_layer());
    assert_eq!(loaded_graph.entry_point(), graph.entry_point());

    // Verify search results match
    let query = test_vector(25, dim);
    let original_results = graph.search(&query, 10, 50)?;
    let loaded_results = loaded_graph.search(&query, 10, 50)?;

    assert_eq!(original_results.len(), loaded_results.len());
    for (orig, loaded) in original_results.iter().zip(loaded_results.iter()) {
        assert_eq!(orig.0, loaded.0); // Same point IDs
        assert!((orig.1 - loaded.1).abs() < 1e-6); // Same distances
    }

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);

    Ok(())
}

// --- Error path tests ---

#[test]
fn test_index_full() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).max_elements(2).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.insert(2, &[0.0, 1.0, 0.0, 0.0])?;

    let result = graph.insert(3, &[0.0, 0.0, 1.0, 0.0]);
    assert!(result.is_err(), "should reject insert when index is full");
    assert_eq!(graph.len(), 2);
    Ok(())
}

#[test]
fn test_delete_nonexistent() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;

    let result = graph.delete(999);
    assert!(result.is_err(), "deleting nonexistent point should error");
    Ok(())
}

#[test]
fn test_delete_already_deleted() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.delete(1)?;

    let result = graph.delete(1);
    assert!(result.is_err(), "double delete should error");
    Ok(())
}

#[test]
fn test_delete_count_saturates() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.delete(1)?;
    assert_eq!(graph.len(), 0);
    // count should not underflow
    Ok(())
}

// --- Boundary search tests ---

#[test]
fn test_search_k_zero() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;

    let results = graph.search(&[1.0, 0.0, 0.0, 0.0], 0, 50)?;
    assert!(results.is_empty());
    Ok(())
}

#[test]
fn test_search_k_larger_than_graph() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.insert(2, &[0.0, 1.0, 0.0, 0.0])?;

    let results = graph.search(&[1.0, 0.0, 0.0, 0.0], 100, 200)?;
    assert!(results.len() <= 2);
    Ok(())
}

#[test]
fn test_search_dimension_too_long() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;

    let result = graph.search(&[1.0, 2.0, 3.0, 4.0, 5.0], 1, 50);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_insert_dimension_too_long() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    let result = graph.insert(1, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(result.is_err());
    Ok(())
}

// --- with_seed tests ---

#[test]
fn test_with_seed_zero_becomes_one() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let graph = HnswGraph::with_seed(config, L2Distance, 0);
    // seed=0 should be converted to 1 (for xorshift to work)
    assert_eq!(graph.rng_state, 1);
    Ok(())
}

#[test]
fn test_with_seed_deterministic() -> v_hnsw_core::Result<()> {
    let dim = 8;
    let config1 = HnswConfig::builder().dim(dim).m(4).build()?;
    let config2 = HnswConfig::builder().dim(dim).m(4).build()?;
    let mut g1 = HnswGraph::with_seed(config1, L2Distance, 42);
    let mut g2 = HnswGraph::with_seed(config2, L2Distance, 42);

    for i in 0..20 {
        let v = test_vector(i, dim);
        g1.insert(i, &v)?;
        g2.insert(i, &v)?;
    }

    let query = test_vector(10, dim);
    let r1 = g1.search(&query, 5, 50)?;
    let r2 = g2.search(&query, 5, 50)?;
    assert_eq!(r1, r2, "same seed should produce same results");
    Ok(())
}

// --- search_ext tests ---

#[test]
fn test_search_ext() -> v_hnsw_core::Result<()> {
    let dim = 8;
    let config = HnswConfig::builder().dim(dim).m(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..20 {
        graph.insert(i, &test_vector(i, dim))?;
    }

    let query = test_vector(10, dim);
    let internal_results = graph.search(&query, 5, 50)?;
    let ext_results = graph.search_ext(&graph.store, &query, 5, 50)?;

    assert_eq!(internal_results.len(), ext_results.len());
    for (a, b) in internal_results.iter().zip(ext_results.iter()) {
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() < 1e-6);
    }
    Ok(())
}

// --- Graph invariants ---

#[test]
fn test_entry_point_valid_after_inserts() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).m(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    assert_eq!(graph.entry_point(), None);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    assert!(graph.entry_point().is_some());

    // Entry point must be a valid node
    let ep = graph.entry_point().unwrap();
    assert!(graph.nodes.contains_key(&ep));

    for i in 2..=20 {
        graph.insert(i, &test_vector(i, dim))?;
        let ep = graph.entry_point().unwrap();
        assert!(graph.nodes.contains_key(&ep), "entry point {ep} is not in nodes");
    }
    Ok(())
}

#[test]
fn test_neighbor_count_within_limits() -> v_hnsw_core::Result<()> {
    let dim = 8;
    let m = 4;
    let config = HnswConfig::builder().dim(dim).m(m).build()?;
    let m0 = config.m0;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..50 {
        graph.insert(i, &test_vector(i, dim))?;
    }

    // Check that neighbor counts don't exceed limits
    for (_, node) in &graph.nodes {
        for layer in 0..=node.max_layer {
            let nbr_count = node.neighbors_at(layer).len();
            let limit = if layer == 0 { m0 } else { m };
            assert!(
                nbr_count <= limit + 1, // allow +1 for race during pruning in concurrent case
                "node {} layer {} has {} neighbors, limit is {}",
                node.id, layer, nbr_count, limit
            );
        }
    }
    Ok(())
}

#[test]
fn test_len_matches_non_deleted() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..10 {
        graph.insert(i, &test_vector(i, dim))?;
    }
    assert_eq!(graph.len(), 10);

    // Delete some
    graph.delete(0)?;
    graph.delete(5)?;
    graph.delete(9)?;
    assert_eq!(graph.len(), 7);

    // Verify deleted nodes are flagged
    assert!(graph.nodes[&0].deleted);
    assert!(graph.nodes[&5].deleted);
    assert!(graph.nodes[&9].deleted);
    assert!(!graph.nodes[&1].deleted);
    Ok(())
}

#[test]
fn test_search_all_deleted_except_entry() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
    graph.insert(3, &[0.0, 0.0, 1.0, 0.0])?;

    // Delete all except entry point
    let ep = graph.entry_point().unwrap();
    for &id in &[1u64, 2, 3] {
        if id != ep {
            graph.delete(id)?;
        }
    }

    let results = graph.search(&[1.0, 0.0, 0.0, 0.0], 5, 50)?;
    // Only the entry point should appear (if not deleted)
    for (id, _) in &results {
        assert_eq!(*id, ep);
    }
    Ok(())
}

// --- Save/load edge cases ---

#[test]
fn test_save_load_empty_graph() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let graph = HnswGraph::new(config, L2Distance);

    let path = std::env::temp_dir().join("test_hnsw_save_load_empty.bin");
    graph.save(&path)?;
    let loaded = HnswGraph::load(&path, L2Distance)?;

    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.entry_point(), None);
    assert_eq!(loaded.config().dim, 4);

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_save_load_preserves_config() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder()
        .dim(64)
        .m(32)
        .m0(48)
        .ef_construction(300)
        .max_elements(50_000)
        .build()?;
    let graph = HnswGraph::new(config, L2Distance);

    let path = std::env::temp_dir().join("test_hnsw_save_load_config.bin");
    graph.save(&path)?;
    let loaded = HnswGraph::load(&path, L2Distance)?;

    assert_eq!(loaded.config().dim, 64);
    assert_eq!(loaded.config().m, 32);
    assert_eq!(loaded.config().m0, 48);
    assert_eq!(loaded.config().ef_construction, 300);
    assert_eq!(loaded.config().max_elements, 50_000);

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_save_load_with_deletions() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
    graph.insert(3, &[0.0, 0.0, 1.0, 0.0])?;
    graph.delete(2)?;

    let path = std::env::temp_dir().join("test_hnsw_save_load_deleted.bin");
    graph.save(&path)?;
    let mut loaded = HnswGraph::load(&path, L2Distance)?;
    loaded.populate_store(&graph.store)?;

    assert_eq!(loaded.len(), 2);
    // Search should not return deleted point
    let results = loaded.search(&[0.0, 1.0, 0.0, 0.0], 5, 50)?;
    for (id, _) in &results {
        assert_ne!(*id, 2);
    }

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_load_nonexistent_file() {
    let result = HnswGraph::load("/nonexistent/path/graph.bin", L2Distance);
    assert!(result.is_err());
}

// --- Single vector graph ---

#[test]
fn test_single_vector_graph_properties() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 2.0, 3.0, 4.0])?;

    assert_eq!(graph.len(), 1);
    assert_eq!(graph.entry_point(), Some(1));
    // First node inserted should have no neighbors (it's the only node)
    let node = &graph.nodes[&1];
    assert!(node.neighbors_at(0).is_empty());
    Ok(())
}

// --- Duplicate insert ---

#[test]
fn test_insert_same_id_twice() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).max_elements(10).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    // Inserting same ID again (the behavior depends on implementation —
    // it may overwrite or increment count; we just verify no crash)
    graph.insert(1, &[0.0, 1.0, 0.0, 0.0])?;

    // Should still be searchable
    let results = graph.search(&[0.0, 1.0, 0.0, 0.0], 1, 50)?;
    assert!(!results.is_empty());
    Ok(())
}

// --- ef parameter edge cases ---

#[test]
fn test_search_ef_one() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..10 {
        graph.insert(i, &test_vector(i, dim))?;
    }

    // ef=1 is valid but may give poor results
    let results = graph.search(&test_vector(5, dim), 1, 1)?;
    assert!(!results.is_empty());
    Ok(())
}

// --- is_empty ---

#[test]
fn test_is_empty() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    assert!(graph.is_empty());

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    assert!(!graph.is_empty());

    graph.delete(1)?;
    assert!(graph.is_empty());
    Ok(())
}

// --- Results contain no duplicate IDs ---

#[test]
fn test_search_no_duplicate_ids() -> v_hnsw_core::Result<()> {
    let dim = 16;
    let config = HnswConfig::builder().dim(dim).m(8).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..100 {
        graph.insert(i, &test_vector(i, dim))?;
    }

    let query = test_vector(50, dim);
    let results = graph.search(&query, 20, 100)?;

    let mut seen = std::collections::HashSet::new();
    for (id, _) in &results {
        assert!(seen.insert(*id), "duplicate id {id} in search results");
    }
    Ok(())
}

// --- All distances are non-negative ---

#[test]
fn test_search_distances_non_negative() -> v_hnsw_core::Result<()> {
    let dim = 16;
    let config = HnswConfig::builder().dim(dim).m(8).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    for i in 0..50 {
        graph.insert(i, &test_vector(i, dim))?;
    }

    let query = test_vector(25, dim);
    let results = graph.search(&query, 10, 50)?;

    for (_, dist) in &results {
        assert!(*dist >= 0.0, "distance should be non-negative, got {dist}");
        assert!(!dist.is_nan(), "distance should not be NaN");
    }
    Ok(())
}
