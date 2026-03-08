//! Tests for Collection wrapper.

use crate::collection::Collection;
use crate::engine::StorageConfig;

use super::{cleanup, test_dir, test_payload, test_vector};

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

#[test]
fn test_create_collection() {
    let root = test_dir("col_create");
    std::fs::create_dir_all(&root).unwrap();

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 100,
        ..Default::default()
    };
    let col = Collection::create(&root, "test_col", config).unwrap();

    assert_eq!(col.name(), "test_col");
    assert!(col.path().ends_with("test_col"));
    assert!(col.path().exists());
    cleanup(&root);
}

#[test]
fn test_create_collection_directory_structure() {
    let root = test_dir("col_dirs");
    std::fs::create_dir_all(&root).unwrap();

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 50,
        ..Default::default()
    };
    let _col = Collection::create(&root, "my_col", config).unwrap();

    // The collection directory should be created
    assert!(root.join("my_col").exists());
    // Vector file should exist
    assert!(root.join("my_col").join("vectors.bin").exists());
    cleanup(&root);
}

// ---------------------------------------------------------------------------
// Open
// ---------------------------------------------------------------------------

#[test]
fn test_open_collection() {
    let root = test_dir("col_open");
    std::fs::create_dir_all(&root).unwrap();

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 100,
        ..Default::default()
    };

    // Create first
    {
        let _col = Collection::create(&root, "open_test", config).unwrap();
    }

    // Open existing
    let col = Collection::open(&root, "open_test").unwrap();
    assert_eq!(col.name(), "open_test");
    cleanup(&root);
}

#[test]
fn test_open_nonexistent_collection_fails() {
    let root = test_dir("col_open_fail");
    std::fs::create_dir_all(&root).unwrap();

    let result = Collection::open(&root, "does_not_exist");
    assert!(result.is_err());
    cleanup(&root);
}

// ---------------------------------------------------------------------------
// Engine accessors
// ---------------------------------------------------------------------------

#[test]
fn test_engine_accessor() {
    let root = test_dir("col_engine");
    std::fs::create_dir_all(&root).unwrap();

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 100,
        ..Default::default()
    };
    let col = Collection::create(&root, "engine_test", config).unwrap();

    // engine() returns immutable ref
    let engine = col.engine();
    assert_eq!(engine.len(), 0);
    cleanup(&root);
}

#[test]
fn test_engine_mut_accessor() {
    let root = test_dir("col_engine_mut");
    std::fs::create_dir_all(&root).unwrap();

    let config = StorageConfig {
        dim: 4,
        initial_capacity: 100,
        ..Default::default()
    };
    let mut col = Collection::create(&root, "mut_test", config).unwrap();

    // Insert a vector through the mutable engine
    let vec = test_vector(4, 1.0);
    let payload = test_payload("test.md", 0, 1);
    col.engine_mut().insert(1, &vec, payload, "hello world").unwrap();

    // Verify through immutable engine
    assert_eq!(col.engine().len(), 1);
    cleanup(&root);
}

// ---------------------------------------------------------------------------
// Name & Path
// ---------------------------------------------------------------------------

#[test]
fn test_name_returns_correct_name() {
    let root = test_dir("col_name");
    std::fs::create_dir_all(&root).unwrap();

    let config = StorageConfig::default();
    let col = Collection::create(&root, "fancy_name", config).unwrap();
    assert_eq!(col.name(), "fancy_name");
    cleanup(&root);
}

#[test]
fn test_path_is_under_root() {
    let root = test_dir("col_path");
    std::fs::create_dir_all(&root).unwrap();

    let config = StorageConfig::default();
    let col = Collection::create(&root, "sub", config).unwrap();
    assert!(col.path().starts_with(&root));
    assert!(col.path().ends_with("sub"));
    cleanup(&root);
}
