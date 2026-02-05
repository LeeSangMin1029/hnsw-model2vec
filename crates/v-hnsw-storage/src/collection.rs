//! Collection abstraction wrapping StorageEngine for single collection management.

use std::path::{Path, PathBuf};
use v_hnsw_core::Result;

use crate::engine::{StorageConfig, StorageEngine};

/// A single collection wrapping a StorageEngine.
///
/// Directory structure: `collections_root/name/vectors.bin`, etc.
pub struct Collection {
    name: String,
    root: PathBuf,
    engine: StorageEngine,
}

impl Collection {
    /// Create a new collection with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `collections_root` - Parent directory for all collections
    /// * `name` - Collection name (will be used as subdirectory)
    /// * `config` - Storage configuration for the collection
    ///
    /// # Errors
    ///
    /// Returns error if directory creation or storage initialization fails.
    pub fn create(
        collections_root: &Path,
        name: &str,
        config: StorageConfig,
    ) -> Result<Self> {
        let collection_dir = collections_root.join(name);
        let engine = StorageEngine::create(&collection_dir, config)?;

        Ok(Self {
            name: name.to_string(),
            root: collection_dir,
            engine,
        })
    }

    /// Open an existing collection.
    ///
    /// # Arguments
    ///
    /// * `collections_root` - Parent directory for all collections
    /// * `name` - Collection name (subdirectory)
    ///
    /// # Errors
    ///
    /// Returns error if directory doesn't exist, files are corrupt, or WAL replay fails.
    pub fn open(collections_root: &Path, name: &str) -> Result<Self> {
        let collection_dir = collections_root.join(name);
        let engine = StorageEngine::open(&collection_dir)?;

        Ok(Self {
            name: name.to_string(),
            root: collection_dir,
            engine,
        })
    }

    /// Get the collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the collection directory path.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Get a reference to the underlying storage engine.
    pub fn engine(&self) -> &StorageEngine {
        &self.engine
    }

    /// Get a mutable reference to the underlying storage engine.
    pub fn engine_mut(&mut self) -> &mut StorageEngine {
        &mut self.engine
    }
}
