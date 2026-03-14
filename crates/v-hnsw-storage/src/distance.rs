//! Distance computers for HNSW search.

use v_hnsw_core::{DistanceMetric, PointId, VectorStore};
use v_hnsw_graph::{DistanceComputer, NormalizedCosineDistance};

use crate::sq8::Sq8Params;
use crate::sq8_store::Sq8VectorStore;

/// f32 distance computer for exact rescore.
pub struct F32Dc<'a> {
    pub store: &'a dyn VectorStore,
}

impl DistanceComputer for F32Dc<'_> {
    fn distance(&self, query: &[f32], id: PointId) -> v_hnsw_core::Result<f32> {
        let vec = self.store.get(id)?;
        Ok(NormalizedCosineDistance.distance(query, vec))
    }
}

/// SQ8 distance computer for approximate traversal.
pub struct Sq8Dc<'a> {
    pub params: &'a Sq8Params,
    pub store: &'a Sq8VectorStore,
}

impl DistanceComputer for Sq8Dc<'_> {
    fn distance(&self, query: &[f32], id: PointId) -> v_hnsw_core::Result<f32> {
        let codes = self.store.get(id)?;
        Ok(self.params.asymmetric_distance(query, codes))
    }
}

/// SQ8 distance computer with pre-built query LUT for batch search.
///
/// ~2x faster than `Sq8Dc` when computing distance to many vectors with
/// the same query, since the per-element multiply is pre-computed.
pub struct Sq8LutDc<'a> {
    pub query_lut: Vec<f32>,
    pub store: &'a Sq8VectorStore,
}

impl<'a> Sq8LutDc<'a> {
    pub fn new(params: &Sq8Params, store: &'a Sq8VectorStore, query: &[f32]) -> Self {
        Self {
            query_lut: params.build_query_lut(query),
            store,
        }
    }
}

impl DistanceComputer for Sq8LutDc<'_> {
    fn distance(&self, _query: &[f32], id: PointId) -> v_hnsw_core::Result<f32> {
        let codes = self.store.get(id)?;
        Ok(Sq8Params::distance_with_lut(&self.query_lut, codes))
    }
}
