//! Collection-aware database operations.
//!
//! Provides a simplified API for managing multiple collections using
//! the storage layer's Collection abstraction.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use v_hnsw_core::{Result, VhnswError};
use v_hnsw_storage::{Collection, StorageConfig};

/// A multi-collection database wrapper.
///
/// Manages multiple named collections, each stored in its own subdirectory.
/// Directory structure: `db_root/collection_name/vectors.bin`, etc.
pub struct MultiCollectionDb {
    root: PathBuf,
    collections: HashMap<String, Collection>,
}

impl MultiCollectionDb {
    /// Open or create a multi-collection database at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - Root directory for the database
    ///
    /// # Errors
    ///
    /// Returns error if directory creation fails or cannot read existing collections.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();

        // Create root directory if it doesn't exist
        if !root.exists() {
            fs::create_dir_all(&root)?;
        }

        // Load existing collections
        let mut collections = HashMap::new();
        if let Ok(entries) = fs::read_dir(&root) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Ok(collection) = Collection::open(&root, &name) {
                        collections.insert(name, collection);
                    }
                }
            }
        }

        Ok(Self { root, collections })
    }

    /// Create a new collection.
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name (must be unique)
    /// * `dim` - Vector dimensionality
    ///
    /// # Errors
    ///
    /// Returns error if collection already exists or creation fails.
    pub fn create_collection(&mut self, name: &str, dim: usize) -> Result<()> {
        if self.collections.contains_key(name) {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Collection '{}' already exists", name),
            )));
        }

        let config = StorageConfig {
            dim,
            ..Default::default()
        };
        let collection = Collection::create(&self.root, name, config)?;
        self.collections.insert(name.to_string(), collection);
        Ok(())
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    /// Get a reference to a collection by name.
    pub fn get_collection(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    /// Get a mutable reference to a collection by name.
    pub fn get_collection_mut(&mut self, name: &str) -> Option<&mut Collection> {
        self.collections.get_mut(name)
    }

    /// Delete a collection.
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    ///
    /// # Errors
    ///
    /// Returns error if collection doesn't exist or deletion fails.
    pub fn delete_collection(&mut self, name: &str) -> Result<()> {
        if !self.collections.contains_key(name) {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Collection '{}' not found", name),
            )));
        }

        self.collections.remove(name);
        let collection_path = self.root.join(name);
        fs::remove_dir_all(&collection_path)?;

        Ok(())
    }

    /// Get the database root path.
    pub fn root_path(&self) -> &Path {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_multi_collection_db_create() {
        let temp = TempDir::new().unwrap();
        let mut db = MultiCollectionDb::open(temp.path()).unwrap();

        // Create collections
        assert!(db.create_collection("vectors1", 128).is_ok());
        assert!(db.create_collection("vectors2", 256).is_ok());

        // List collections
        let collections = db.list_collections();
        assert_eq!(collections.len(), 2);
        assert!(collections.contains(&"vectors1".to_string()));
        assert!(collections.contains(&"vectors2".to_string()));
    }

    #[test]
    fn test_multi_collection_db_duplicate() {
        let temp = TempDir::new().unwrap();
        let mut db = MultiCollectionDb::open(temp.path()).unwrap();

        db.create_collection("vectors", 128).unwrap();
        let result = db.create_collection("vectors", 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_collection_db_delete() {
        let temp = TempDir::new().unwrap();
        let mut db = MultiCollectionDb::open(temp.path()).unwrap();

        db.create_collection("vectors", 128).unwrap();
        assert_eq!(db.list_collections().len(), 1);

        db.delete_collection("vectors").unwrap();
        assert_eq!(db.list_collections().len(), 0);
    }

    #[test]
    fn test_multi_collection_db_get() {
        let temp = TempDir::new().unwrap();
        let mut db = MultiCollectionDb::open(temp.path()).unwrap();

        db.create_collection("vectors", 128).unwrap();

        let collection = db.get_collection("vectors");
        assert!(collection.is_some());
        assert_eq!(collection.unwrap().name(), "vectors");

        let missing = db.get_collection("missing");
        assert!(missing.is_none());
    }

    #[test]
    fn test_multi_collection_db_reopen() {
        let temp = TempDir::new().unwrap();

        // Create and close
        {
            let mut db = MultiCollectionDb::open(temp.path()).unwrap();
            db.create_collection("vectors", 128).unwrap();
        }

        // Reopen and verify
        {
            let db = MultiCollectionDb::open(temp.path()).unwrap();
            let collections = db.list_collections();
            assert_eq!(collections.len(), 1);
            assert!(collections.contains(&"vectors".to_string()));
        }
    }
}
