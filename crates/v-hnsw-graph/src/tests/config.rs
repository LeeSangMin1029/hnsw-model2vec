use crate::config::HnswConfig;

#[test]
fn test_builder_defaults() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(128).build()?;
    assert_eq!(config.dim, 128);
    assert_eq!(config.m, 16);
    assert_eq!(config.m0, 32);
    assert_eq!(config.ef_construction, 200);
    assert_eq!(config.max_elements, 100_000);
    assert!((config.ml - 1.0 / (16.0_f64).ln()).abs() < 1e-10);
    Ok(())
}

#[test]
fn test_builder_custom() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder()
        .dim(384)
        .m(32)
        .m0(48)
        .ef_construction(400)
        .max_elements(500_000)
        .build()?;
    assert_eq!(config.dim, 384);
    assert_eq!(config.m, 32);
    assert_eq!(config.m0, 48);
    assert_eq!(config.ef_construction, 400);
    assert_eq!(config.max_elements, 500_000);
    Ok(())
}

#[test]
fn test_builder_no_dim() {
    let result = HnswConfig::builder().build();
    assert!(result.is_err());
}

#[test]
fn test_builder_zero_dim() {
    let result = HnswConfig::builder().dim(0).build();
    assert!(result.is_err());
}

#[test]
fn test_builder_m_too_small() {
    let result = HnswConfig::builder().dim(128).m(1).build();
    assert!(result.is_err());
}

#[test]
fn test_builder_zero_max_elements() {
    let result = HnswConfig::builder().dim(128).max_elements(0).build();
    assert!(result.is_err());
}

#[test]
fn test_builder_m_boundary_two() -> v_hnsw_core::Result<()> {
    // m=2 is the minimum valid value
    let config = HnswConfig::builder().dim(4).m(2).build()?;
    assert_eq!(config.m, 2);
    assert_eq!(config.m0, 4); // default m0 = 2*m
    Ok(())
}

#[test]
fn test_builder_m_zero() {
    let result = HnswConfig::builder().dim(4).m(0).build();
    assert!(result.is_err());
}

#[test]
fn test_builder_m0_not_overridden_defaults_to_2m() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).m(8).build()?;
    assert_eq!(config.m0, 16);
    Ok(())
}

#[test]
fn test_builder_m0_override_explicit() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).m(8).m0(10).build()?;
    assert_eq!(config.m0, 10);
    Ok(())
}

#[test]
fn test_builder_m0_zero_defaults_to_2m() -> v_hnsw_core::Result<()> {
    // m0(0) should behave like not set, defaulting to 2*m
    let config = HnswConfig::builder().dim(4).m(8).m0(0).build()?;
    assert_eq!(config.m0, 16);
    Ok(())
}

#[test]
fn test_builder_dim_one() -> v_hnsw_core::Result<()> {
    // dim=1 is valid (minimum non-zero)
    let config = HnswConfig::builder().dim(1).build()?;
    assert_eq!(config.dim, 1);
    Ok(())
}

#[test]
fn test_builder_max_elements_one() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).max_elements(1).build()?;
    assert_eq!(config.max_elements, 1);
    Ok(())
}

#[test]
fn test_builder_ml_computed_correctly() -> v_hnsw_core::Result<()> {
    for m in [2, 4, 8, 16, 32, 64] {
        let config = HnswConfig::builder().dim(4).m(m).build()?;
        let expected_ml = 1.0 / (m as f64).ln();
        assert!(
            (config.ml - expected_ml).abs() < 1e-10,
            "ml mismatch for m={m}: got {}, expected {expected_ml}",
            config.ml
        );
    }
    Ok(())
}

#[test]
fn test_builder_ef_construction_one() -> v_hnsw_core::Result<()> {
    let config = HnswConfig::builder().dim(4).ef_construction(1).build()?;
    assert_eq!(config.ef_construction, 1);
    Ok(())
}
