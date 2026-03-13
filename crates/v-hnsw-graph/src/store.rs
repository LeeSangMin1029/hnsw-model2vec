//! In-memory vector storage backend.

use std::collections::HashMap;

use v_hnsw_core::{Dim, PointId, VectorStore, VhnswError};

/// A simple in-memory vector store backed by a `HashMap`.
#[derive(Clone, bincode::Encode, bincode::Decode)]
pub struct InMemoryVectorStore {
    dim: Dim,
    vectors: HashMap<PointId, Vec<f32>>,
}

impl InMemoryVectorStore {
    /// Create a new empty store for vectors of the given dimensionality.
    pub fn new(dim: Dim) -> Self {
        Self {
            dim,
            vectors: HashMap::new(),
        }
    }

    /// Create a new store with pre-allocated capacity.
    pub fn with_capacity(dim: Dim, capacity: usize) -> Self {
        Self {
            dim,
            vectors: HashMap::with_capacity(capacity),
        }
    }
}

impl VectorStore for InMemoryVectorStore {
    fn get(&self, id: PointId) -> v_hnsw_core::Result<&[f32]> {
        self.vectors
            .get(&id)
            .map(|v| v.as_slice())
            .ok_or(VhnswError::PointNotFound(id))
    }

    fn insert(&mut self, id: PointId, vector: &[f32]) -> v_hnsw_core::Result<()> {
        v_hnsw_core::check_dimension(self.dim, vector.len())?;
        self.vectors.insert(id, vector.to_vec());
        Ok(())
    }

    fn remove(&mut self, id: PointId) -> v_hnsw_core::Result<()> {
        self.vectors
            .remove(&id)
            .map(|_| ())
            .ok_or(VhnswError::PointNotFound(id))
    }

    fn dim(&self) -> Dim {
        self.dim
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}
