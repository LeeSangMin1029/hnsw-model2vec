use crate::fusion::convex::{ConvexFusion, normalize};

#[test]
fn test_default_alpha() {
    let fusion = ConvexFusion::new();
    assert!((fusion.alpha() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_custom_alpha() {
    let fusion = ConvexFusion::with_alpha(0.7);
    assert!((fusion.alpha() - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_alpha_clamping() {
    let fusion = ConvexFusion::with_alpha(1.5);
    assert!((fusion.alpha() - 1.0).abs() < f32::EPSILON);

    let fusion = ConvexFusion::with_alpha(-0.3);
    assert!((fusion.alpha() - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_fuse_empty() {
    let fusion = ConvexFusion::new();
    let result = fusion.fuse(&[], &[], 10);
    assert!(result.is_empty());
}

#[test]
fn test_fuse_dense_only() {
    let fusion = ConvexFusion::with_alpha(1.0);
    let dense = vec![(1, 0.1), (2, 0.5), (3, 0.9)];
    let result = fusion.fuse(&dense, &[], 10);

    assert_eq!(result.len(), 3);
    // Closest (smallest distance) should be first
    assert_eq!(result[0].0, 1);
}

#[test]
fn test_fuse_sparse_only() {
    let fusion = ConvexFusion::with_alpha(0.0);
    let sparse = vec![(1, 5.0), (2, 3.0), (3, 1.0)];
    let result = fusion.fuse(&[], &sparse, 10);

    assert_eq!(result.len(), 3);
    // Highest BM25 score should be first
    assert_eq!(result[0].0, 1);
}

#[test]
fn test_fuse_balanced() {
    let fusion = ConvexFusion::with_alpha(0.5);
    // Doc 1: close in vector space (dist=0.1), low BM25 (1.0)
    // Doc 2: far in vector space (dist=0.9), high BM25 (5.0)
    let dense = vec![(1, 0.1), (2, 0.9)];
    let sparse = vec![(1, 1.0), (2, 5.0)];
    let result = fusion.fuse(&dense, &sparse, 10);

    assert_eq!(result.len(), 2);
    // With equal weights, both should contribute equally
    // Doc 1: 0.5 * 1.0 (dense normalized) + 0.5 * 0.0 (sparse normalized) = 0.5
    // Doc 2: 0.5 * 0.0 (dense normalized) + 0.5 * 1.0 (sparse normalized) = 0.5
    // They should be tied (or close)
    let score_diff = (result[0].1 - result[1].1).abs();
    assert!(score_diff < 0.01);
}

#[test]
fn test_fuse_with_limit() {
    let fusion = ConvexFusion::new();
    let dense = vec![(1, 0.1), (2, 0.2), (3, 0.3)];
    let sparse = vec![(4, 5.0), (5, 4.0), (6, 3.0)];
    let result = fusion.fuse(&dense, &sparse, 2);

    assert_eq!(result.len(), 2);
}

#[test]
fn test_fuse_overlapping_docs() {
    let fusion = ConvexFusion::with_alpha(0.5);
    // Doc 1 appears in both lists
    let dense = vec![(1, 0.1), (2, 0.5)];
    let sparse = vec![(1, 5.0), (3, 3.0)];
    let result = fusion.fuse(&dense, &sparse, 10);

    assert_eq!(result.len(), 3);
    // Doc 1 should be first (appears in both, good scores)
    assert_eq!(result[0].0, 1);
}

#[test]
fn test_normalize_same_values() {
    // When all distances are the same, all should get max score
    let results = vec![(1, 0.5), (2, 0.5), (3, 0.5)];
    let scores = normalize(&results, true);
    for (_, score) in &scores {
        assert!((*score - 1.0).abs() < f32::EPSILON);
    }
}

// ============================================================================
// Additional fusion tests: edge cases, boundary conditions
// ============================================================================

#[test]
fn test_fuse_single_dense_result() {
    let fusion = ConvexFusion::new();
    let dense = vec![(1, 0.5)];
    let result = fusion.fuse(&dense, &[], 10);
    assert_eq!(result.len(), 1);
    // Single value normalizes to 1.0, then multiplied by alpha=0.5
    assert!((result[0].1 - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_fuse_single_sparse_result() {
    let fusion = ConvexFusion::new();
    let sparse = vec![(1, 3.0)];
    let result = fusion.fuse(&[], &sparse, 10);
    assert_eq!(result.len(), 1);
    // Single value normalizes to 1.0, then multiplied by (1-alpha)=0.5
    assert!((result[0].1 - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_fuse_alpha_zero_ignores_dense() {
    let fusion = ConvexFusion::with_alpha(0.0);
    let dense = vec![(1, 0.0), (2, 1.0)]; // doc 1 is closer
    let sparse = vec![(3, 5.0)];
    let result = fusion.fuse(&dense, &sparse, 10);

    // With alpha=0, dense scores should be 0, only sparse matters
    // Doc 3 should be first with score 1.0 (normalized sparse)
    assert_eq!(result[0].0, 3);
    assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_fuse_alpha_one_ignores_sparse() {
    let fusion = ConvexFusion::with_alpha(1.0);
    let dense = vec![(1, 0.1)]; // closest
    let sparse = vec![(2, 100.0)]; // highest BM25
    let result = fusion.fuse(&dense, &sparse, 10);

    // With alpha=1.0, sparse scores should be 0, only dense matters
    // Doc 1 should have score 1.0 (normalized, inverted)
    assert_eq!(result[0].0, 1);
    assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_fuse_identical_docs_in_both() {
    let fusion = ConvexFusion::with_alpha(0.5);
    // Same doc appears in both with good scores
    let dense = vec![(1, 0.1)]; // close
    let sparse = vec![(1, 5.0)]; // high score
    let result = fusion.fuse(&dense, &sparse, 10);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, 1);
    // normalized dense = 1.0 (single value), normalized sparse = 1.0 (single value)
    // score = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_fuse_limit_zero() {
    let fusion = ConvexFusion::new();
    let dense = vec![(1, 0.1)];
    let sparse = vec![(1, 5.0)];
    let result = fusion.fuse(&dense, &sparse, 0);
    assert!(result.is_empty());
}

#[test]
fn test_fuse_many_results_with_limit() {
    let fusion = ConvexFusion::new();
    let dense: Vec<(u64, f32)> = (0..50).map(|i| (i, i as f32 * 0.01)).collect();
    let sparse: Vec<(u64, f32)> = (25..75).map(|i| (i, (100 - i) as f32)).collect();
    let result = fusion.fuse(&dense, &sparse, 5);
    assert_eq!(result.len(), 5);
    // Results should be sorted descending
    for w in result.windows(2) {
        assert!(w[0].1 >= w[1].1);
    }
}

#[test]
fn test_normalize_empty() {
    let scores = normalize(&[], false);
    assert!(scores.is_empty());
}

#[test]
fn test_normalize_single_value() {
    let results = vec![(1, 5.0)];
    let scores = normalize(&results, false);
    // Single value → range = 0 → all become 1.0
    assert!((scores[&1] - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_normalize_invert_direction() {
    // Dense: lower distance = better → invert
    let results = vec![(1, 0.1), (2, 0.5), (3, 1.0)];
    let scores = normalize(&results, true);
    // Doc 1 (closest) should have highest normalized score
    assert!(scores[&1] > scores[&2]);
    assert!(scores[&2] > scores[&3]);
    assert!((scores[&1] - 1.0).abs() < f32::EPSILON);
    assert!((scores[&3] - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_normalize_no_invert() {
    // Sparse: higher score = better → no invert
    let results = vec![(1, 1.0), (2, 3.0), (3, 5.0)];
    let scores = normalize(&results, false);
    // Doc 3 (highest score) should have highest normalized score
    assert!(scores[&3] > scores[&2]);
    assert!(scores[&2] > scores[&1]);
    assert!((scores[&3] - 1.0).abs() < f32::EPSILON);
    assert!((scores[&1] - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_normalize_negative_values() {
    // Some scoring systems can produce negative values
    let results = vec![(1, -2.0), (2, 0.0), (3, 2.0)];
    let scores = normalize(&results, false);
    // Should still normalize to [0, 1]
    assert!((scores[&3] - 1.0).abs() < f32::EPSILON);
    assert!((scores[&1] - 0.0).abs() < f32::EPSILON);
    assert!((scores[&2] - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_fuse_conflicting_scores() {
    // Doc 1: great dense, terrible sparse
    // Doc 2: terrible dense, great sparse
    let fusion = ConvexFusion::with_alpha(0.5);
    let dense = vec![(1, 0.0), (2, 1.0)]; // doc 1 closest
    let sparse = vec![(1, 0.0), (2, 10.0)]; // doc 2 highest BM25
    let result = fusion.fuse(&dense, &sparse, 10);

    assert_eq!(result.len(), 2);
    // With alpha=0.5, both should get equal weight
    // Doc 1: 0.5 * 1.0 (inverted dense) + 0.5 * 0.0 (normalized sparse) = 0.5
    // Doc 2: 0.5 * 0.0 (inverted dense) + 0.5 * 1.0 (normalized sparse) = 0.5
    let diff = (result[0].1 - result[1].1).abs();
    assert!(diff < 0.01, "Conflicting scores should balance: {:?}", result);
}

#[test]
fn test_fuse_dense_heavy_breaks_ties() {
    // When dense is weighted more, vector similarity should dominate
    let fusion = ConvexFusion::with_alpha(0.9);
    let dense = vec![(1, 0.0), (2, 0.5), (3, 1.0)]; // doc 1 closest
    let sparse = vec![(1, 1.0), (2, 2.0), (3, 5.0)]; // doc 3 highest BM25
    let result = fusion.fuse(&dense, &sparse, 3);

    // Doc 1 should win because dense weight is 0.9
    assert_eq!(result[0].0, 1, "Dense-heavy fusion should prefer closest vector");
}

// ── normalize tests ────────────────────────────────────────────────

#[test]
fn normalize_empty_returns_empty() {
    let result = normalize(&[], false);
    assert!(result.is_empty());
}

#[test]
fn normalize_single_element_returns_one() {
    let result = normalize(&[(42, 3.5)], false);
    assert_eq!(result.len(), 1);
    assert!((result[&42] - 1.0).abs() < f32::EPSILON, "single element should normalize to 1.0");
}

#[test]
fn normalize_two_elements_min_max() {
    let result = normalize(&[(1, 0.0), (2, 10.0)], false);
    assert!((result[&1] - 0.0).abs() < f32::EPSILON, "min should normalize to 0.0");
    assert!((result[&2] - 1.0).abs() < f32::EPSILON, "max should normalize to 1.0");
}

#[test]
fn normalize_invert_flips_scores() {
    let result = normalize(&[(1, 0.0), (2, 10.0)], true);
    assert!((result[&1] - 1.0).abs() < f32::EPSILON, "min distance should become 1.0 when inverted");
    assert!((result[&2] - 0.0).abs() < f32::EPSILON, "max distance should become 0.0 when inverted");
}

#[test]
fn normalize_equal_scores_all_one() {
    let result = normalize(&[(1, 5.0), (2, 5.0), (3, 5.0)], false);
    for &id in &[1, 2, 3] {
        assert!((result[&id] - 1.0).abs() < f32::EPSILON, "equal scores should all normalize to 1.0");
    }
}

#[test]
fn normalize_midpoint_is_half() {
    let result = normalize(&[(1, 0.0), (2, 5.0), (3, 10.0)], false);
    assert!((result[&2] - 0.5).abs() < f32::EPSILON, "midpoint should normalize to 0.5");
}
