//! Main HNSW graph structure and `VectorIndex` implementation.

use std::collections::HashMap;
use std::path::Path;

use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorIndex, VectorStore, VhnswError};

use crate::config::HnswConfig;
use crate::node::{Node, NodeSerialized};
use crate::store::InMemoryVectorStore;

/// Serializable representation of HnswGraph (excludes distance metric and vectors).
///
/// Vectors are not stored in hnsw.bin; they are restored from StorageEngine
/// via `populate_store()` after loading.
#[derive(bincode::Encode, bincode::Decode)]
struct HnswGraphSerialized {
    config: HnswConfig,
    nodes: HashMap<PointId, NodeSerialized>,
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
    /// Only graph structure (nodes + config) is saved; vectors are NOT included.
    /// This reduces file size by ~50% compared to the previous format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written to.
    pub fn save(&self, path: impl AsRef<Path>) -> v_hnsw_core::Result<()> {
        use std::io::Write;

        let serialized = HnswGraphSerialized {
            config: self.config.clone(),
            nodes: self.nodes.iter()
                .map(|(&id, node)| (id, NodeSerialized::from(node)))
                .collect(),
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
    /// The loaded graph has an **empty vector store**. Call `populate_store()`
    /// with an external `VectorStore` (e.g. `StorageEngine::vector_store()`)
    /// before performing search or insert operations.
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

        let config = serialized.config;
        let store = InMemoryVectorStore::with_capacity(config.dim, config.max_elements);

        Ok(Self {
            nodes: serialized.nodes.into_iter()
                .map(|(id, ns)| (id, Node::from(ns)))
                .collect(),
            store,
            entry_point: serialized.entry_point,
            max_layer: serialized.max_layer,
            distance,
            count: serialized.count,
            rng_state: serialized.rng_state,
            config,
        })
    }

    /// Populate the internal vector store from an external source.
    ///
    /// Must be called after `load()` before any search or insert operations
    /// that use the internal store (i.e. `VectorIndex::search`/`insert`).
    ///
    /// **Prefer `search_ext()` / `build_insert()` instead** — they read
    /// vectors directly from an external store without this copy step.
    pub fn populate_store(&mut self, source: &dyn VectorStore) -> v_hnsw_core::Result<()> {
        for &id in self.nodes.keys() {
            if let Ok(vec) = source.get(id) {
                self.store.insert(id, vec)?;
            }
        }
        Ok(())
    }

    /// Insert a point reading its vector from an external store (no copy).
    ///
    /// The vector must already exist in `store` at the given `id`.
    /// Used by buildindex to read directly from mmap, avoiding a full
    /// vector copy into the internal `InMemoryVectorStore`.
    pub fn build_insert(&mut self, store: &dyn VectorStore, id: PointId) -> v_hnsw_core::Result<()> {
        crate::insert::insert_with_store(self, store, id)
    }

    /// Search using an external vector store (no `populate_store` needed).
    ///
    /// Reads vectors directly from `store` (e.g. mmap) during search,
    /// eliminating the startup cost and memory overhead of `populate_store()`.
    pub fn search_ext(
        &self,
        store: &dyn VectorStore,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        crate::search::search_ext(self, store, query, k, ef)
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
