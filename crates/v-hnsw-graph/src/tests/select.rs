use v_hnsw_core::PointId;
use crate::select::select_neighbors;

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

#[test]
fn test_select_m_zero() {
    let candidates = vec![(1, 0.5), (2, 0.3)];
    let selected = select_neighbors(&candidates, 0);
    assert!(selected.is_empty());
}

#[test]
fn test_select_m_one() {
    let candidates = vec![(1, 0.5), (2, 0.3), (3, 0.1)];
    let selected = select_neighbors(&candidates, 1);
    assert_eq!(selected, vec![3]); // closest only
}

#[test]
fn test_select_single_candidate() {
    let candidates = vec![(42, 1.0)];
    let selected = select_neighbors(&candidates, 5);
    assert_eq!(selected, vec![42]);
}

#[test]
fn test_select_equal_distances() {
    let candidates = vec![(1, 0.5), (2, 0.5), (3, 0.5)];
    let selected = select_neighbors(&candidates, 2);
    assert_eq!(selected.len(), 2);
}

#[test]
fn test_select_large_m() {
    // m much larger than candidates
    let candidates = vec![(1, 0.1)];
    let selected = select_neighbors(&candidates, 1000);
    assert_eq!(selected, vec![1]);
}

#[test]
fn test_select_preserves_distance_order() {
    let candidates: Vec<(PointId, f32)> = vec![
        (10, 1.0),
        (20, 0.5),
        (30, 0.1),
        (40, 0.9),
        (50, 0.2),
    ];
    let selected = select_neighbors(&candidates, 3);
    // Should be the 3 closest: 30 (0.1), 50 (0.2), 20 (0.5)
    assert_eq!(selected, vec![30, 50, 20]);
}
