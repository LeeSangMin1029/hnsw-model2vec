//! In-memory vector storage backend.

use std::collections::HashMap;

use v_hnsw_core::{Dim, PointId, VectorStore, VhnswError};

/// A simple in-memory vector store backed by a `HashMap`.
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
        if vector.len() != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() -> v_hnsw_core::Result<()> {
        let mut store = InMemoryVectorStore::new(3);
        store.insert(1, &[1.0, 2.0, 3.0])?;
        let vec = store.get(1)?;
        assert_eq!(vec, &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut store = InMemoryVectorStore::new(3);
        let result = store.insert(1, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_missing() {
        let store = InMemoryVectorStore::new(3);
        let result = store.get(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove() -> v_hnsw_core::Result<()> {
        let mut store = InMemoryVectorStore::new(2);
        store.insert(1, &[1.0, 2.0])?;
        assert_eq!(store.len(), 1);
        store.remove(1)?;
        assert_eq!(store.len(), 0);
        assert!(store.get(1).is_err());
        Ok(())
    }

    #[test]
    fn test_remove_missing() {
        let mut store = InMemoryVectorStore::new(2);
        let result = store.remove(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_empty() -> v_hnsw_core::Result<()> {
        let mut store = InMemoryVectorStore::new(2);
        assert!(store.is_empty());
        store.insert(1, &[1.0, 2.0])?;
        assert!(!store.is_empty());
        Ok(())
    }
}
