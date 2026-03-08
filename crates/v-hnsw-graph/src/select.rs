//! Neighbor selection heuristic for HNSW graph construction.

use v_hnsw_core::PointId;

/// Select up to `m` nearest neighbors from a candidate list.
///
/// Uses simple selection: sort by distance ascending, take the first `m`.
/// A heuristic selection that considers graph diversity can be added later.
pub(crate) fn select_neighbors(candidates: &[(PointId, f32)], m: usize) -> Vec<PointId> {
    let mut sorted: Vec<(PointId, f32)> = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.into_iter().take(m).map(|(id, _)| id).collect()
}
