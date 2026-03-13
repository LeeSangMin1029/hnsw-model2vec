use crate::config::HnswConfig;
use crate::distance::L2Distance;
use crate::graph::HnswGraph;
use crate::search::NodeGraph;
use crate::snapshot::HnswSnapshot;
use v_hnsw_core::VectorIndex;

use super::helpers::test_vector;

#[test]
fn test_snapshot_roundtrip() -> v_hnsw_core::Result<()> {
    let dim = 16;
    let config = HnswConfig::builder().dim(dim).m(8).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    for i in 0..100 {
        graph.insert(i, &test_vector(i, dim))?;
    }

    let path = std::env::temp_dir().join("test_hnsw_snapshot.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    assert_eq!(snap.len(), graph.len());
    assert_eq!(snap.entry_point(), graph.entry_point());
    assert_eq!(snap.config().dim, dim);

    // Verify neighbor access matches
    for (&id, node) in &graph.nodes {
        for layer in 0..=node.max_layer {
            let graph_nbrs = node.neighbors_at(layer);
            let snap_nbrs = snap.neighbors(id, layer);
            assert_eq!(
                snap_nbrs.map(|s| s.to_vec()),
                Some(graph_nbrs.to_vec()),
                "mismatch at node {} layer {}",
                id,
                layer
            );
        }
        assert_eq!(snap.is_deleted(id), node.deleted);
    }

    // Search comparison
    let query = test_vector(50, dim);
    let graph_results = graph.search(&query, 10, 50)?;
    let snap_results = snap.search_ext(&L2Distance, &graph.store, &query, 10, 50)?;

    assert_eq!(graph_results.len(), snap_results.len());
    for (g, s) in graph_results.iter().zip(snap_results.iter()) {
        assert_eq!(g.0, s.0);
        assert!((g.1 - s.1).abs() < 1e-6);
    }

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_empty_graph() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let graph = HnswGraph::new(config, L2Distance);

    let path = std::env::temp_dir().join("test_hnsw_snapshot_empty.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    assert_eq!(snap.len(), 0);
    assert_eq!(snap.entry_point(), None);

    let results = snap.search_ext(&L2Distance, &graph.store, &[1.0, 2.0, 3.0, 4.0], 5, 50)?;
    assert!(results.is_empty());

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_deleted_node() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
    graph.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
    graph.delete(1)?;

    let path = std::env::temp_dir().join("test_hnsw_snapshot_deleted.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    assert_eq!(snap.len(), 1); // only 1 live node
    assert!(snap.is_deleted(1));
    assert!(!snap.is_deleted(2));

    // Search near deleted point: should only return point 2
    let results = snap.search_ext(&L2Distance, &graph.store, &[1.0, 0.0, 0.0, 0.0], 5, 50)?;
    for (id, _) in &results {
        assert_ne!(*id, 1);
    }

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_preserves_config() -> v_hnsw_core::Result<()> {
    let dim = 32;
    let config = HnswConfig::builder()
        .dim(dim)
        .m(24)
        .m0(36)
        .ef_construction(150)
        .max_elements(5000)
        .build()?;
    let graph = HnswGraph::new(config, L2Distance);

    let path = std::env::temp_dir().join("test_hnsw_snap_config.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    assert_eq!(snap.config().dim, dim);
    assert_eq!(snap.config().m, 24);
    assert_eq!(snap.config().m0, 36);
    assert_eq!(snap.config().ef_construction, 150);
    assert_eq!(snap.config().max_elements, 5000);

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_nonexistent_node() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;

    let path = std::env::temp_dir().join("test_hnsw_snap_nonexistent.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    assert_eq!(snap.neighbors(999, 0), None);
    assert!(snap.is_deleted(999));

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_search_dimension_mismatch() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;

    let path = std::env::temp_dir().join("test_hnsw_snap_dim_mismatch.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    let result = snap.search_ext(&L2Distance, &graph.store, &[1.0, 2.0], 5, 50);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_single_node() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;

    let path = std::env::temp_dir().join("test_hnsw_snap_single.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    assert_eq!(snap.len(), 1);
    assert!(!snap.is_empty());
    assert_eq!(snap.entry_point(), Some(1));

    let results = snap.search_ext(&L2Distance, &graph.store, &[1.0, 0.0, 0.0, 0.0], 1, 50)?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 1e-6);

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_is_empty() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).build()?;
    let graph = HnswGraph::new(config, L2Distance);

    let path = std::env::temp_dir().join("test_hnsw_snap_is_empty.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    assert!(snap.is_empty());

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_layer_out_of_range() -> v_hnsw_core::Result<()> {
    let dim = 4;
    let config = HnswConfig::builder().dim(dim).build()?;
    let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
    graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;

    let path = std::env::temp_dir().join("test_hnsw_snap_layer_oor.snap");
    HnswSnapshot::save(&graph, &path)?;
    let snap = HnswSnapshot::open(&path)?;

    let nbrs = snap.neighbors(1, 200);
    assert_eq!(nbrs, Some(&[][..]));

    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_snapshot_open_invalid_file() {
    let path = std::env::temp_dir().join("test_hnsw_snap_invalid.snap");
    std::fs::write(&path, b"this is not a valid snapshot").unwrap();

    let result = HnswSnapshot::open(&path);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_snapshot_open_too_small() {
    let path = std::env::temp_dir().join("test_hnsw_snap_small.snap");
    std::fs::write(&path, b"tiny").unwrap();

    let result = HnswSnapshot::open(&path);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_snapshot_open_nonexistent() {
    let result = HnswSnapshot::open(std::path::Path::new("/nonexistent/path/snap.snap"));
    assert!(result.is_err());
}
