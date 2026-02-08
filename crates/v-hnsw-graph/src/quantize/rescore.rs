//! 2-stage search helper: quantized oversample followed by full-precision rescore.
//!
//! The typical workflow is:
//! 1. Run a fast approximate search (e.g. using quantized distances) to retrieve
//!    `oversample_factor * k` candidates.
//! 2. Call [`rescore`] to recompute exact distances for those candidates and
//!    return only the true top-`k`.

use v_hnsw_core::{DistanceMetric, PointId};

/// Rescore a set of pre-filtered candidates using full-precision distance.
///
/// Given `candidates` (typically obtained from a quantized search), each entry
/// is `(point_id, approximate_distance)`. The function retrieves the original
/// vector via `get_vector`, recomputes the distance with `distance`, sorts by
/// exact distance, and returns the top `k` results.
///
/// Candidates whose vectors cannot be retrieved (i.e. `get_vector` returns
/// `None`) are silently skipped.
pub fn rescore(
    candidates: &[(PointId, f32)],
    query: &[f32],
    get_vector: &dyn Fn(PointId) -> Option<Vec<f32>>,
    distance: &dyn DistanceMetric,
    k: usize,
) -> Vec<(PointId, f32)> {
    let mut rescored: Vec<(PointId, f32)> = candidates
        .iter()
        .filter_map(|(id, _approx_dist)| {
            let vec = get_vector(*id)?;
            let dist = distance.distance(query, &vec);
            Some((*id, dist))
        })
        .collect();

    rescored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    rescored.truncate(k);
    rescored
}
