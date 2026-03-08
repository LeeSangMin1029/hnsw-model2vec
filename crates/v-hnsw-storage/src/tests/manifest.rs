use crate::manifest::*;
use std::time::SystemTime;

fn make_info(name: &str, dim: usize) -> CollectionInfo {
    CollectionInfo {
        name: name.to_string(),
        dim,
        metric: "cosine".to_string(),
        created_at: 0,
        count: 0,
    }
}

#[test]
fn test_manifest_create_and_save() {
    let temp_dir = tempfile::tempdir().unwrap();
    let manifest_path = temp_dir.path().join("manifest.json");

    let mut manifest = Manifest::new();
    assert_eq!(manifest.version, MANIFEST_VERSION);
    assert!(manifest.collections.is_empty());

    let info = CollectionInfo {
        name: "test_collection".to_string(),
        dim: 384,
        metric: "cosine".to_string(),
        created_at: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        count: 0,
    };

    manifest.add_collection(info.clone()).unwrap();
    assert_eq!(manifest.collections.len(), 1);

    manifest.save(&manifest_path).unwrap();
    assert!(manifest_path.exists());

    let loaded = Manifest::load(&manifest_path).unwrap();
    assert_eq!(loaded.version, MANIFEST_VERSION);
    assert_eq!(loaded.collections.len(), 1);
    assert_eq!(loaded.get_collection("test_collection").unwrap().dim, 384);
}

#[test]
fn test_add_duplicate_collection() {
    let mut manifest = Manifest::new();
    let info = CollectionInfo {
        name: "dup".to_string(),
        dim: 128,
        metric: "l2".to_string(),
        created_at: 0,
        count: 0,
    };

    manifest.add_collection(info.clone()).unwrap();
    let result = manifest.add_collection(info);
    assert!(result.is_err());
}

#[test]
fn test_remove_collection() {
    let mut manifest = Manifest::new();
    let info = CollectionInfo {
        name: "to_remove".to_string(),
        dim: 256,
        metric: "dot".to_string(),
        created_at: 0,
        count: 42,
    };

    manifest.add_collection(info).unwrap();
    assert_eq!(manifest.collections.len(), 1);

    manifest.remove_collection("to_remove").unwrap();
    assert!(manifest.collections.is_empty());

    let result = manifest.remove_collection("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_update_count() {
    let mut manifest = Manifest::new();
    let info = CollectionInfo {
        name: "counter".to_string(),
        dim: 512,
        metric: "cosine".to_string(),
        created_at: 0,
        count: 0,
    };

    manifest.add_collection(info).unwrap();
    manifest.update_count("counter", 100).unwrap();
    assert_eq!(manifest.get_collection("counter").unwrap().count, 100);
}

// --- New tests ---

#[test]
fn test_manifest_load_nonexistent_returns_empty() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("does_not_exist.json");

    let manifest = Manifest::load(&path).unwrap();
    assert!(manifest.collections.is_empty());
    assert_eq!(manifest.version, MANIFEST_VERSION);
}

#[test]
fn test_manifest_load_invalid_json_returns_error() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("manifest.json");
    std::fs::write(&path, "{ not valid json !!!").unwrap();

    let result = Manifest::load(&path);
    assert!(result.is_err());
}

#[test]
fn test_manifest_version_mismatch() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("manifest.json");

    // Write a manifest with a wrong version
    let json = r#"{"version": 999, "collections": {}}"#;
    std::fs::write(&path, json).unwrap();

    let result = Manifest::load(&path);
    assert!(result.is_err());
}

#[test]
fn test_manifest_list_collections_multiple() {
    let mut manifest = Manifest::new();

    manifest.add_collection(make_info("alpha", 128)).unwrap();
    manifest.add_collection(make_info("beta", 256)).unwrap();
    manifest.add_collection(make_info("gamma", 512)).unwrap();

    let mut names = manifest.list_collections();
    names.sort();
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn test_manifest_get_nonexistent_collection() {
    let manifest = Manifest::new();
    assert!(manifest.get_collection("nope").is_none());
}

#[test]
fn test_manifest_update_count_nonexistent() {
    let mut manifest = Manifest::new();
    let result = manifest.update_count("ghost", 42);
    assert!(result.is_err());
}

#[test]
fn test_manifest_save_and_reload_preserves_all_fields() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("manifest.json");

    let mut manifest = Manifest::new();
    let info = CollectionInfo {
        name: "full".to_string(),
        dim: 768,
        metric: "dot".to_string(),
        created_at: 1700000000,
        count: 42,
    };
    manifest.add_collection(info).unwrap();
    manifest.save(&path).unwrap();

    let loaded = Manifest::load(&path).unwrap();
    let c = loaded.get_collection("full").unwrap();
    assert_eq!(c.dim, 768);
    assert_eq!(c.metric, "dot");
    assert_eq!(c.created_at, 1700000000);
    assert_eq!(c.count, 42);
}

#[test]
fn test_manifest_add_remove_add_same_name() {
    let mut manifest = Manifest::new();

    manifest.add_collection(make_info("recycle", 128)).unwrap();
    manifest.remove_collection("recycle").unwrap();

    // Should be able to re-add after removal
    manifest.add_collection(make_info("recycle", 256)).unwrap();
    assert_eq!(manifest.get_collection("recycle").unwrap().dim, 256);
}

#[test]
fn test_manifest_many_collections() {
    let mut manifest = Manifest::new();

    for i in 0..100 {
        manifest
            .add_collection(make_info(&format!("col_{i}"), 128))
            .unwrap();
    }

    assert_eq!(manifest.list_collections().len(), 100);

    // Remove half
    for i in 0..50 {
        manifest.remove_collection(&format!("col_{i}")).unwrap();
    }

    assert_eq!(manifest.list_collections().len(), 50);
}

#[test]
fn test_manifest_default_trait() {
    let m = Manifest::default();
    assert_eq!(m.version, MANIFEST_VERSION);
    assert!(m.collections.is_empty());
}
