use crate::config::*;

#[test]
fn test_default_config() {
    let config = HybridSearchConfig::default();
    assert_eq!(config.ef_search, 200);
    assert_eq!(config.dense_limit, 100);
    assert_eq!(config.sparse_limit, 100);
    assert!((config.fusion_alpha - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_builder() {
    let config = HybridSearchConfig::builder()
        .ef_search(300)
        .dense_limit(50)
        .sparse_limit(200)
        .build();

    assert_eq!(config.ef_search, 300);
    assert_eq!(config.dense_limit, 50);
    assert_eq!(config.sparse_limit, 200);
}

#[test]
fn test_builder_partial() {
    let config = HybridSearchConfig::builder().fusion_alpha(0.8).build();

    assert!((config.fusion_alpha - 0.8).abs() < f32::EPSILON);
    // Other values should be defaults
    assert_eq!(config.ef_search, 200);
}

// ============================================================================
// Additional config tests
// ============================================================================

#[test]
fn test_builder_all_fields() {
    let config = HybridSearchConfig::builder()
        .ef_search(500)
        .dense_limit(200)
        .sparse_limit(300)
        .fusion_alpha(0.3)
        .build();

    assert_eq!(config.ef_search, 500);
    assert_eq!(config.dense_limit, 200);
    assert_eq!(config.sparse_limit, 300);
    assert!((config.fusion_alpha - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_builder_default_same_as_default() {
    let from_builder = HybridSearchConfig::builder().build();
    let from_default = HybridSearchConfig::default();

    assert_eq!(from_builder.ef_search, from_default.ef_search);
    assert_eq!(from_builder.dense_limit, from_default.dense_limit);
    assert_eq!(from_builder.sparse_limit, from_default.sparse_limit);
    assert!((from_builder.fusion_alpha - from_default.fusion_alpha).abs() < f32::EPSILON);
}

#[test]
fn test_config_fusion_alpha_boundary() {
    // alpha = 0.0 (sparse only)
    let config = HybridSearchConfig::builder().fusion_alpha(0.0).build();
    assert!((config.fusion_alpha - 0.0).abs() < f32::EPSILON);

    // alpha = 1.0 (dense only)
    let config = HybridSearchConfig::builder().fusion_alpha(1.0).build();
    assert!((config.fusion_alpha - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_config_clone() {
    let config = HybridSearchConfig::builder()
        .ef_search(123)
        .fusion_alpha(0.42)
        .build();

    let cloned = config.clone();
    assert_eq!(config.ef_search, cloned.ef_search);
    assert_eq!(config.dense_limit, cloned.dense_limit);
    assert!((config.fusion_alpha - cloned.fusion_alpha).abs() < f32::EPSILON);
}

#[test]
fn test_config_debug() {
    let config = HybridSearchConfig::default();
    let debug = format!("{:?}", config);
    assert!(debug.contains("ef_search"));
    assert!(debug.contains("fusion_alpha"));
}
