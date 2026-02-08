//! Main HNSW graph structure and `VectorIndex` implementation.

use std::collections::HashMap;
use std::path::Path;

use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorIndex, VhnswError};

use crate::config::HnswConfig;
use crate::node::Node;
use crate::store::InMemoryVectorStore;

/// Serializable representation of HnswGraph (excludes distance metric).
#[derive(bincode::Encode, bincode::Decode)]
struct HnswGraphSerialized {
    config: HnswConfig,
    nodes: HashMap<PointId, Node>,
    store: InMemoryVectorStore,
    entry_point: Option<PointId>,
    max_layer: LayerId,
    count: usize,
    rng_state: u64,
}

/// An in-memory Hierarchical Navigable Small World graph.
///
/// Generic over the distance metric `D`. Implements `VectorIndex` for
/// approximate nearest neighbor search.
///
/// # Example
///
/// ```ignore
/// use v_hnsw_graph::{HnswConfig, HnswGraph};
/// use v_hnsw_graph::L2Distance;
///
/// let config = HnswConfig::builder().dim(128).build()?;
/// let mut graph = HnswGraph::new(config, L2Distance);
/// graph.insert(1, &vec![0.0; 128])?;
/// let results = graph.search(&vec![0.0; 128], 10, 200)?;
/// ```
pub struct HnswGraph<D: DistanceMetric> {
    pub(crate) config: HnswConfig,
    pub(crate) nodes: HashMap<PointId, Node>,
    pub(crate) store: InMemoryVectorStore,
    pub(crate) entry_point: Option<PointId>,
    pub(crate) max_layer: LayerId,
    #[allow(dead_code)]
    pub(crate) distance: D,
    pub(crate) count: usize,
    /// Internal PRNG state for layer assignment.
    pub(crate) rng_state: u64,
}

impl<D: DistanceMetric> HnswGraph<D> {
    /// Create a new empty HNSW graph with the given configuration and distance metric.
    pub fn new(config: HnswConfig, distance: D) -> Self {
        let store = InMemoryVectorStore::with_capacity(config.dim, config.max_elements);
        Self {
            config,
            nodes: HashMap::new(),
            store,
            entry_point: None,
            max_layer: 0,
            distance,
            count: 0,
            rng_state: 0x5EED_CAFE_BABE_1234, // Arbitrary non-zero seed
        }
    }

    /// Create a new HNSW graph with a custom RNG seed.
    ///
    /// Useful for deterministic testing.
    pub fn with_seed(config: HnswConfig, distance: D, seed: u64) -> Self {
        let store = InMemoryVectorStore::with_capacity(config.dim, config.max_elements);
        // Ensure seed is non-zero for xorshift
        let seed = if seed == 0 { 1 } else { seed };
        Self {
            config,
            nodes: HashMap::new(),
            store,
            entry_point: None,
            max_layer: 0,
            distance,
            count: 0,
            rng_state: seed,
        }
    }

    /// Get a reference to the graph configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the current maximum layer in the graph.
    pub fn max_layer(&self) -> LayerId {
        self.max_layer
    }

    /// Get the current entry point, if any.
    pub fn entry_point(&self) -> Option<PointId> {
        self.entry_point
    }

    /// Save graph to file (bincode format).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written to.
    pub fn save(&self, path: impl AsRef<Path>) -> v_hnsw_core::Result<()> {
        use std::io::Write;

        let serialized = HnswGraphSerialized {
            config: self.config.clone(),
            nodes: self.nodes.clone(),
            store: self.store.clone(),
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            count: self.count,
            rng_state: self.rng_state,
        };

        let file = std::fs::File::create(path)
            .map_err(VhnswError::Storage)?;
        let mut writer = std::io::BufWriter::new(file);
        bincode::encode_into_std_write(&serialized, &mut writer, bincode::config::standard())
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("serialize failed: {e}"))))?;
        writer.flush().map_err(VhnswError::Storage)?;
        Ok(())
    }

    /// Load graph from file.
    ///
    /// The distance metric must be provided since it may not be serializable
    /// or may need to be reconstructed with specific runtime configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or deserialized.
    pub fn load(path: impl AsRef<Path>, distance: D) -> v_hnsw_core::Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(VhnswError::Storage)?;
        let mut reader = std::io::BufReader::new(file);
        let serialized = bincode::decode_from_std_read::<HnswGraphSerialized, _, _>(
            &mut reader,
            bincode::config::standard()
        ).map_err(|e| VhnswError::Storage(std::io::Error::other(format!("deserialize failed: {e}"))))?;

        Ok(Self {
            config: serialized.config,
            nodes: serialized.nodes,
            store: serialized.store,
            entry_point: serialized.entry_point,
            max_layer: serialized.max_layer,
            distance,
            count: serialized.count,
            rng_state: serialized.rng_state,
        })
    }
}

// VectorIndex requires Send + Sync. HnswGraph is Send+Sync if D is.
// D: DistanceMetric already requires Send + Sync + 'static.
// HashMap, InMemoryVectorStore, etc. are all Send+Sync.
// SAFETY: No interior mutability or raw pointers.
// Note: VectorIndex uses &mut self for insert/delete, so no concurrent mutation concern.

impl<D: DistanceMetric> VectorIndex for HnswGraph<D> {
    fn search(&self, query: &[f32], k: usize, ef: usize) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        crate::search::search(self, query, k, ef)
    }

    fn insert(&mut self, id: PointId, vector: &[f32]) -> v_hnsw_core::Result<()> {
        crate::insert::insert(self, id, vector)
    }

    fn delete(&mut self, id: PointId) -> v_hnsw_core::Result<()> {
        let node = self
            .nodes
            .get_mut(&id)
            .ok_or(VhnswError::PointNotFound(id))?;

        if node.deleted {
            return Err(VhnswError::PointNotFound(id));
        }

        node.deleted = true;
        self.count = self.count.saturating_sub(1);
        Ok(())
    }

    fn len(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::L2Distance;

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

        // Load from file
        let loaded_graph = HnswGraph::load(&temp_path, L2Distance)?;

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
}
