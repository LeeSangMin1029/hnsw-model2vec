use crate::collection_manager::*;
use crate::engine::StorageConfig;
use tempfile::TempDir;
use v_hnsw_core::VectorStore;

#[test]
fn test_create_and_open_collection() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    // Create a collection
    let config = StorageConfig {
        dim: 384,
        initial_capacity: 1000,
        checkpoint_threshold: 100,
    };
    manager.create_collection("test", config).unwrap();

    // List collections
    let collections = manager.list_collections();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0], "test");

    // Get collection
    let collection = manager.get_collection("test").unwrap();
    assert_eq!(collection.name(), "test");
}

#[test]
fn test_lazy_loading() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    // Create a collection
    let config = StorageConfig::default();
    manager.create_collection("lazy", config).unwrap();

    // Close manager (simulates restart)
    drop(manager);

    // Reopen manager
    let mut manager = CollectionManager::open_or_create(root).unwrap();

    // Collection should not be loaded yet
    assert!(manager.open_collections.is_empty());

    // Get collection triggers lazy load
    let collection = manager.get_collection("lazy").unwrap();
    assert_eq!(collection.name(), "lazy");

    // Now it should be in open_collections
    assert_eq!(manager.open_collections.len(), 1);
}

#[test]
fn test_delete_collection() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    // Create and delete
    let config = StorageConfig::default();
    manager.create_collection("to_delete", config).unwrap();
    assert_eq!(manager.list_collections().len(), 1);

    manager.delete_collection("to_delete").unwrap();
    assert!(manager.list_collections().is_empty());

    // Directory should be gone
    let collection_dir = root.join("collections").join("to_delete");
    assert!(!collection_dir.exists());
}

#[test]
fn test_rename_collection() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    // Create and rename
    let config = StorageConfig::default();
    manager.create_collection("old_name", config).unwrap();

    manager.rename_collection("old_name", "new_name").unwrap();

    let collections = manager.list_collections();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0], "new_name");

    // Old directory should be gone, new one should exist
    let old_dir = root.join("collections").join("old_name");
    let new_dir = root.join("collections").join("new_name");
    assert!(!old_dir.exists());
    assert!(new_dir.exists());
}

#[test]
fn test_default_collection() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    // Get default collection (should create it)
    let collection = manager.default_collection().unwrap();
    assert_eq!(collection.name(), "default");

    // Should be in manifest
    let collections = manager.list_collections();
    assert_eq!(collections.len(), 1);
    assert!(collections.contains(&"default".to_string()));
}

#[test]
fn test_duplicate_collection_error() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    let config = StorageConfig::default();
    manager.create_collection("dup", config).unwrap();

    // Try to create with same name
    let config2 = StorageConfig::default();
    let result = manager.create_collection("dup", config2);
    assert!(result.is_err());
}

// --- New tests ---

#[test]
fn test_rename_to_existing_name_fails() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();
    manager.create_collection("a", StorageConfig::default()).unwrap();
    manager.create_collection("b", StorageConfig::default()).unwrap();

    // Renaming a -> b should fail because b already exists
    let result = manager.rename_collection("a", "b");
    assert!(result.is_err());

    // Both collections should still exist
    assert_eq!(manager.list_collections().len(), 2);
}

#[test]
fn test_rename_nonexistent_fails() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();
    let result = manager.rename_collection("ghost", "new_ghost");
    assert!(result.is_err());
}

#[test]
fn test_delete_nonexistent_collection_fails() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();
    let result = manager.delete_collection("does_not_exist");
    assert!(result.is_err());
}

#[test]
fn test_get_nonexistent_collection_fails() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();
    let result = manager.get_collection("nope");
    assert!(result.is_err());
}

#[test]
fn test_get_collection_mut_nonexistent_fails() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();
    let result = manager.get_collection_mut("nope");
    assert!(result.is_err());
}

#[test]
fn test_open_after_delete_fails() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();
    manager.create_collection("ephemeral", StorageConfig::default()).unwrap();
    manager.delete_collection("ephemeral").unwrap();

    // Trying to get the deleted collection should fail
    let result = manager.get_collection("ephemeral");
    assert!(result.is_err());
}

#[test]
fn test_multiple_collections_independent() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    let config_a = StorageConfig { dim: 64, initial_capacity: 100, checkpoint_threshold: 100 };
    let config_b = StorageConfig { dim: 128, initial_capacity: 200, checkpoint_threshold: 100 };

    manager.create_collection("col_a", config_a).unwrap();
    manager.create_collection("col_b", config_b).unwrap();

    // Each collection should have its own dimension
    let a = manager.get_collection("col_a").unwrap();
    assert_eq!(a.engine().vector_store().dim(), 64);

    let b = manager.get_collection("col_b").unwrap();
    assert_eq!(b.engine().vector_store().dim(), 128);
}

#[test]
fn test_create_delete_recreate_same_name() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    manager.create_collection("cycle", StorageConfig::default()).unwrap();
    manager.delete_collection("cycle").unwrap();
    manager.create_collection("cycle", StorageConfig { dim: 64, ..StorageConfig::default() }).unwrap();

    let col = manager.get_collection("cycle").unwrap();
    assert_eq!(col.engine().vector_store().dim(), 64);
}

#[test]
fn test_rename_then_access_new_name() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();
    manager.create_collection("before", StorageConfig::default()).unwrap();
    manager.rename_collection("before", "after").unwrap();

    // Old name should fail
    assert!(manager.get_collection("before").is_err());

    // New name should work
    let col = manager.get_collection("after").unwrap();
    assert_eq!(col.name(), "after");
}

#[test]
fn test_manager_persists_across_reopen() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    {
        let mut manager = CollectionManager::open_or_create(root).unwrap();
        manager.create_collection("persistent", StorageConfig::default()).unwrap();
    }

    // Reopen
    let mut manager = CollectionManager::open_or_create(root).unwrap();
    let mut names = manager.list_collections();
    names.sort();
    assert!(names.contains(&"persistent".to_string()));

    // Should be lazy-loadable
    let col = manager.get_collection("persistent").unwrap();
    assert_eq!(col.name(), "persistent");
}

#[test]
fn test_default_collection_idempotent() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    // Calling default_collection multiple times should not create duplicates
    manager.default_collection().unwrap();
    manager.default_collection().unwrap();

    assert_eq!(manager.list_collections().len(), 1);
}

#[test]
fn test_many_collections() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();

    let mut manager = CollectionManager::open_or_create(root).unwrap();

    for i in 0..20 {
        let config = StorageConfig { dim: 8, initial_capacity: 10, checkpoint_threshold: 100 };
        manager.create_collection(&format!("col_{i}"), config).unwrap();
    }

    assert_eq!(manager.list_collections().len(), 20);

    // Delete every other one
    for i in (0..20).step_by(2) {
        manager.delete_collection(&format!("col_{i}")).unwrap();
    }

    assert_eq!(manager.list_collections().len(), 10);
}
