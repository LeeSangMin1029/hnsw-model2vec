//! Core trait definitions for v-hnsw.

use crate::types::{Dim, Payload, PointId};
use crate::Result;

/// A distance metric for comparing vectors.
///
/// Implementations must be deterministic and satisfy:
/// - `distance(a, a) == 0` (identity)
/// - `distance(a, b) >= 0` (non-negativity)
/// - `distance(a, b) == distance(b, a)` (symmetry)
pub trait DistanceMetric: Send + Sync + 'static {
    /// Compute the distance between two vectors of equal length.
    ///
    /// # Panics
    ///
    /// May panic if `a.len() != b.len()` in debug builds.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Human-readable name of this metric (e.g., "l2", "cosine").
    fn name(&self) -> &'static str;
}

/// Storage backend for raw vector data.
pub trait VectorStore: Send + Sync {
    /// Retrieve the vector associated with `id`.
    fn get(&self, id: PointId) -> Result<&[f32]>;

    /// Insert or update a vector.
    fn insert(&mut self, id: PointId, vector: &[f32]) -> Result<()>;

    /// Remove a vector.
    fn remove(&mut self, id: PointId) -> Result<()>;

    /// The dimensionality of stored vectors.
    fn dim(&self) -> Dim;

    /// Number of vectors currently stored.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An approximate nearest neighbor index.
pub trait VectorIndex: Send + Sync {
    /// Search for the `k` nearest neighbors of `query`.
    ///
    /// `ef` controls the search quality (higher = more accurate but slower).
    /// Returns `(point_id, distance)` pairs sorted by ascending distance.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<(PointId, f32)>>;

    /// Insert a vector into the index.
    fn insert(&mut self, id: PointId, vector: &[f32]) -> Result<()>;

    /// Mark a vector as deleted (lazy deletion).
    fn delete(&mut self, id: PointId) -> Result<()>;

    /// Number of indexed vectors (excluding deleted).
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Storage backend for payload data (metadata + text) associated with vector points.
pub trait PayloadStore: Send + Sync {
    /// Retrieve metadata for a point.
    fn get_payload(&self, id: PointId) -> Result<Option<Payload>>;

    /// Set or replace the full payload for a point.
    fn set_payload(&mut self, id: PointId, payload: Payload) -> Result<()>;

    /// Remove the payload for a point.
    fn remove_payload(&mut self, id: PointId) -> Result<()>;

    /// Retrieve only the text chunk for a point.
    fn get_text(&self, id: PointId) -> Result<Option<String>>;
}
