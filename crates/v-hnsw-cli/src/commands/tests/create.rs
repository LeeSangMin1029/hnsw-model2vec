//! Tests for the create module — DbConfig serialization / deserialization.
//!
//! The `run()` function is integration-only (creates StorageEngine); skipped.

use crate::commands::create::DbConfig;

#[test]
fn db_config_roundtrip_serde() {
    let config = DbConfig {
        dim: 128,
        korean: true,
        embed_model: Some("test-model".to_string()),
        code: true,
        ..DbConfig::default()
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: DbConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.version, config.version);
    assert_eq!(deserialized.dim, 128);
    assert_eq!(deserialized.metric, "cosine");
    assert_eq!(deserialized.m, 16);
    assert_eq!(deserialized.ef_construction, 200);
    assert!(deserialized.korean);
    assert_eq!(deserialized.embed_model.as_deref(), Some("test-model"));
    assert!(deserialized.code);
}

#[test]
fn db_config_defaults_for_optional_fields() {
    // Simulate JSON without embed_model and code
    let json = r#"{
        "version": 1,
        "dim": 64,
        "metric": "l2",
        "m": 8,
        "ef_construction": 100,
        "korean": false
    }"#;

    let config: DbConfig = serde_json::from_str(json).unwrap();
    assert!(config.embed_model.is_none(), "embed_model should default to None");
    assert!(!config.code, "code should default to false");
}

#[test]
fn db_config_legacy_content_type_migration() {
    // Simulate old config.json with content_type: "code"
    let json = r#"{
        "version": 1,
        "dim": 256,
        "metric": "cosine",
        "m": 16,
        "ef_construction": 200,
        "korean": false,
        "content_type": "code"
    }"#;

    let tmp_dir = tempfile::tempdir().unwrap();
    std::fs::write(tmp_dir.path().join("config.json"), json).unwrap();

    let config = DbConfig::load(tmp_dir.path()).unwrap();
    assert!(config.code, "content_type='code' should migrate to code=true");
}

#[test]
fn db_config_save_and_load_roundtrip() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let config = DbConfig {
        dim: 256,
        metric: "dot".to_string(),
        m: 32,
        ef_construction: 400,
        ..DbConfig::default()
    };

    config.save(tmp_dir.path()).unwrap();
    let loaded = DbConfig::load(tmp_dir.path()).unwrap();

    assert_eq!(loaded.dim, 256);
    assert_eq!(loaded.metric, "dot");
    assert_eq!(loaded.m, 32);
    assert_eq!(loaded.ef_construction, 400);
    assert!(!loaded.korean);
    assert!(!loaded.code);
}

#[test]
fn db_config_load_missing_file_errors() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let result = DbConfig::load(tmp_dir.path());
    assert!(result.is_err(), "loading from dir without config.json should fail");
}

#[test]
fn db_config_embed_model_skip_serializing_if_none() {
    let config = DbConfig {
        dim: 64,
        ..DbConfig::default()
    };

    let json = serde_json::to_string(&config).unwrap();
    assert!(!json.contains("embed_model"), "None embed_model should be omitted from JSON");
}
