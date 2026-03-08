//! Tests for core trait definitions and default method implementations.

use std::collections::HashMap;

use crate::types::{Dim, PointId};
use crate::{DistanceMetric, VectorIndex, VectorStore};

// ===========================================================================
// Mock implementations for testing trait default methods
// ===========================================================================

/// A trivial L2 distance metric for testing.
struct MockL2;
impl DistanceMetric for MockL2 {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }
    fn name(&self) -> &'static str {
        "mock_l2"
    }
}

/// In-memory VectorStore for testing the trait.
struct MemVectorStore {
    dim: Dim,
    data: HashMap<PointId, Vec<f32>>,
}

impl MemVectorStore {
    fn new(dim: Dim) -> Self {
        Self { dim, data: HashMap::new() }
    }
}

impl VectorStore for MemVectorStore {
    fn get(&self, id: PointId) -> crate::Result<&[f32]> {
        self.data
            .get(&id)
            .map(|v: &Vec<f32>| v.as_slice())
            .ok_or(crate::VhnswError::PointNotFound(id))
    }
    fn insert(&mut self, id: PointId, vector: &[f32]) -> crate::Result<()> {
        if vector.len() != self.dim {
            return Err(crate::VhnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.data.insert(id, vector.to_vec());
        Ok(())
    }
    fn remove(&mut self, id: PointId) -> crate::Result<()> {
        self.data
            .remove(&id)
            .map(|_| ())
            .ok_or(crate::VhnswError::PointNotFound(id))
    }
    fn dim(&self) -> Dim {
        self.dim
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// In-memory VectorIndex for testing the trait.
struct MemVectorIndex {
    vectors: Vec<(PointId, Vec<f32>)>,
}

impl MemVectorIndex {
    fn new() -> Self {
        Self { vectors: Vec::new() }
    }
}

impl VectorIndex for MemVectorIndex {
    fn search(&self, query: &[f32], k: usize, _ef: usize) -> crate::Result<Vec<(PointId, f32)>> {
        let metric = MockL2;
        let mut results: Vec<(PointId, f32)> = self
            .vectors
            .iter()
            .map(|(id, v)| (*id, metric.distance(query, v)))
            .collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        Ok(results)
    }
    fn insert(&mut self, id: PointId, vector: &[f32]) -> crate::Result<()> {
        self.vectors.push((id, vector.to_vec()));
        Ok(())
    }
    fn delete(&mut self, id: PointId) -> crate::Result<()> {
        self.vectors.retain(|(vid, _)| *vid != id);
        Ok(())
    }
    fn len(&self) -> usize {
        self.vectors.len()
    }
}

// ===========================================================================
// DistanceMetric tests
// ===========================================================================

#[test]
fn test_distance_metric_identity() {
    let m = MockL2;
    let v = [1.0, 2.0, 3.0];
    assert_eq!(m.distance(&v, &v), 0.0);
}

#[test]
fn test_distance_metric_symmetry() {
    let m = MockL2;
    let a = [1.0, 0.0];
    let b = [0.0, 1.0];
    assert!((m.distance(&a, &b) - m.distance(&b, &a)).abs() < 1e-6);
}

#[test]
fn test_distance_metric_non_negativity() {
    let m = MockL2;
    let a = [1.0, 2.0];
    let b = [-3.0, 4.0];
    assert!(m.distance(&a, &b) >= 0.0);
}

#[test]
fn test_distance_metric_name() {
    let m = MockL2;
    assert_eq!(m.name(), "mock_l2");
}

// ===========================================================================
// VectorStore trait — default is_empty()
// ===========================================================================

#[test]
fn test_vector_store_is_empty_when_new() {
    let store = MemVectorStore::new(4);
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

#[test]
fn test_vector_store_not_empty_after_insert() {
    let mut store = MemVectorStore::new(2);
    store.insert(1, &[1.0, 2.0]).unwrap();
    assert!(!store.is_empty());
    assert_eq!(store.len(), 1);
}

#[test]
fn test_vector_store_empty_after_remove_all() {
    let mut store = MemVectorStore::new(2);
    store.insert(1, &[1.0, 2.0]).unwrap();
    store.remove(1).unwrap();
    assert!(store.is_empty());
}

#[test]
fn test_vector_store_get_nonexistent() {
    let store = MemVectorStore::new(2);
    assert!(store.get(999).is_err());
}

#[test]
fn test_vector_store_dim_mismatch() {
    let mut store = MemVectorStore::new(3);
    let result = store.insert(1, &[1.0, 2.0]); // dim=2, expected 3
    assert!(result.is_err());
}

// ===========================================================================
// VectorIndex trait — default is_empty()
// ===========================================================================

#[test]
fn test_vector_index_is_empty_when_new() {
    let idx = MemVectorIndex::new();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
}

#[test]
fn test_vector_index_not_empty_after_insert() {
    let mut idx = MemVectorIndex::new();
    idx.insert(1, &[1.0, 0.0]).unwrap();
    assert!(!idx.is_empty());
}

#[test]
fn test_vector_index_empty_after_delete_all() {
    let mut idx = MemVectorIndex::new();
    idx.insert(1, &[1.0]).unwrap();
    idx.delete(1).unwrap();
    assert!(idx.is_empty());
}

#[test]
fn test_vector_index_search() {
    let mut idx = MemVectorIndex::new();
    idx.insert(1, &[1.0, 0.0]).unwrap();
    idx.insert(2, &[0.0, 1.0]).unwrap();
    idx.insert(3, &[0.5, 0.5]).unwrap();

    let results = idx.search(&[1.0, 0.0], 2, 10).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1); // closest
}
