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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_fewer_than_m() {
        let candidates = vec![(1, 0.5), (2, 0.3)];
        let selected = select_neighbors(&candidates, 5);
        assert_eq!(selected, vec![2, 1]);
    }

    #[test]
    fn test_select_exact_m() {
        let candidates = vec![(1, 0.5), (2, 0.3), (3, 0.1)];
        let selected = select_neighbors(&candidates, 3);
        assert_eq!(selected, vec![3, 2, 1]);
    }

    #[test]
    fn test_select_more_than_m() {
        let candidates = vec![(1, 0.5), (2, 0.3), (3, 0.1), (4, 0.9)];
        let selected = select_neighbors(&candidates, 2);
        assert_eq!(selected, vec![3, 2]);
    }

    #[test]
    fn test_select_empty() {
        let candidates: Vec<(PointId, f32)> = vec![];
        let selected = select_neighbors(&candidates, 5);
        assert!(selected.is_empty());
    }
}
