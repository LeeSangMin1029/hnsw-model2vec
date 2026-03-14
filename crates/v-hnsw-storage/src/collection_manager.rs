//! Collection manager for multi-collection database.
//!
//! Manages multiple collections within a single database root:
//! - Lazy loading of collections (only open when requested)
//! - Manifest persistence (save after each mutation)
//! - Legacy detection (if vectors.bin exists at root without manifest.json)
//!
//! # Directory Structure
//!
//! ```text
//! <root>/
//!   manifest.json              - Tracks all collections
//!   collections/
//!     <name>/
//!       vectors.bin
//!       payload.dat
//!       text.dat
//!       payload.idx
//!       text.idx
//!       vectors.idx
//!       wal/
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use v_hnsw_core::{Result, VectorStore, VhnswError};

use crate::collection::Collection;
use crate::engine::StorageConfig;
use crate::manifest::{CollectionInfo, Manifest};

const DEFAULT_COLLECTION_NAME: &str = "default";

/// Manager for multiple collections within a single database root.
pub struct CollectionManager {
    /// Root directory for the database.
    root: PathBuf,
    /// Manifest tracking all collections.
    manifest: Manifest,
    /// Lazily-loaded open collections.
    pub(crate) open_collections: HashMap<String, Collection>,
}

impl CollectionManager {
    /// Validates that a collection either exists or does not exist in the manifest.
    ///
    /// When `should_exist` is `true`, returns an error if the collection is missing.
    /// When `should_exist` is `false`, returns an error if the collection already exists.
    fn check_collection(&self, name: &str, should_exist: bool) -> Result<()> {
        let exists = self.manifest.get_collection(name).is_some();
        if exists != should_exist {
            let msg = if should_exist {
                format!("Collection '{name}' not found")
            } else {
                format!("Collection '{name}' already exists")
            };
            return Err(VhnswError::Payload(msg));
        }
        Ok(())
    }

    /// Path to the `collections/` subdirectory under root.
    fn collections_dir(&self) -> PathBuf {
        self.root.join("collections")
    }

    /// Open or create a database at the given root directory.
    ///
    /// # Legacy Detection
    ///
    /// If `vectors.bin` exists at root without `manifest.json`, this indicates
    /// a legacy single-collection database. The manager will migrate to the new
    /// format by creating a "default" collection.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Manifest file exists but cannot be parsed
    /// - Directory creation fails
    /// - Legacy migration fails
    pub fn open_or_create(root: &Path) -> Result<Self> {
        let manifest_path = root.join("manifest.json");
        let legacy_vectors_path = root.join("vectors.bin");

        // Load or create manifest
        let mut manifest = Manifest::load(&manifest_path)?;

        // Detect legacy single-collection database
        if manifest.collections.is_empty() && legacy_vectors_path.exists() {
            // Migrate legacy database to "default" collection
            Self::migrate_legacy_to_default(root, &mut manifest)?;
            manifest.save(&manifest_path)?;
        }

        Ok(Self {
            root: root.to_path_buf(),
            manifest,
            open_collections: HashMap::new(),
        })
    }

    /// Migrate a legacy single-collection database to the "default" collection.
    ///
    /// Moves all legacy files (vectors.bin, payload.dat, etc.) into
    /// `collections/default/` subdirectory.
    fn migrate_legacy_to_default(root: &Path, manifest: &mut Manifest) -> Result<()> {
        let collections_dir = root.join("collections");
        let default_dir = collections_dir.join(DEFAULT_COLLECTION_NAME);

        // Create collections directory structure
        std::fs::create_dir_all(&default_dir)?;

        // Move legacy files
        let legacy_files = [
            "vectors.bin",
            "payload.dat",
            "text.dat",
            "payload.idx",
            "text.idx",
            "vectors.idx",
        ];

        for file in &legacy_files {
            let src = root.join(file);
            let dst = default_dir.join(file);
            if src.exists() {
                std::fs::rename(&src, &dst).map_err(|e| {
                    VhnswError::Storage(std::io::Error::other(
                        format!("Failed to move legacy file {file}: {e}"),
                    ))
                })?;
            }
        }

        // Move legacy WAL directory
        let legacy_wal = root.join("wal");
        let new_wal = default_dir.join("wal");
        if legacy_wal.exists() {
            std::fs::rename(&legacy_wal, &new_wal).map_err(|e| {
                VhnswError::Storage(std::io::Error::other(
                    format!("Failed to move legacy WAL directory: {e}"),
                ))
            })?;
        }

        // Add default collection to manifest
        // Note: dimension and metric are unknown during migration,
        // they will be read from the actual storage files when opened.
        let created_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| VhnswError::Payload(format!("System time error: {e}")))?
            .as_secs();

        let info = CollectionInfo {
            name: DEFAULT_COLLECTION_NAME.to_string(),
            dim: 384, // Placeholder, will be updated when collection is opened
            metric: "cosine".to_string(),
            created_at,
            count: 0, // Will be updated when collection is opened
        };

        manifest.add_collection(info)?;

        Ok(())
    }

    /// Create a new collection with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Collection name already exists
    /// - Directory creation fails
    /// - Storage initialization fails
    /// - Manifest save fails
    pub fn create_collection(&mut self, name: &str, config: StorageConfig) -> Result<&Collection> {
        // Check if collection already exists
        self.check_collection(name, false)?;

        // Create collection directory
        let collections_dir = self.collections_dir();
        std::fs::create_dir_all(&collections_dir)?;

        // Create collection
        let collection = Collection::create(&collections_dir, name, config)?;

        // Add to manifest
        let created_at = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| VhnswError::Payload(format!("System time error: {e}")))?
            .as_secs();

        let info = CollectionInfo {
            name: name.to_string(),
            dim: collection.engine().vector_store().dim(),
            metric: "cosine".to_string(), // TODO: Get from config or add metric method to StorageEngine
            created_at,
            count: 0,
        };

        self.manifest.add_collection(info)?;
        self.save_manifest()?;

        // Store in open collections
        self.open_collections.insert(name.to_string(), collection);

        self.open_collections
            .get(name)
            .ok_or_else(|| VhnswError::Payload(format!("Failed to retrieve collection '{name}'")))
    }

    /// Get a reference to an existing collection (lazy-loaded).
    pub fn get_collection(&mut self, name: &str) -> Result<&Collection> {
        self.ensure_loaded(name)?;
        self.open_collections
            .get(name)
            .ok_or_else(|| VhnswError::Payload(format!("Failed to retrieve collection '{name}'")))
    }

    /// Get a mutable reference to an existing collection (lazy-loaded).
    pub fn get_collection_mut(&mut self, name: &str) -> Result<&mut Collection> {
        self.ensure_loaded(name)?;
        self.open_collections
            .get_mut(name)
            .ok_or_else(|| VhnswError::Payload(format!("Failed to retrieve collection '{name}'")))
    }

    /// Check manifest and lazy-load collection if not already open.
    fn ensure_loaded(&mut self, name: &str) -> Result<()> {
        self.check_collection(name, true)?;
        if !self.open_collections.contains_key(name) {
            let collections_dir = self.collections_dir();
            let collection = Collection::open(&collections_dir, name)?;
            self.open_collections.insert(name.to_string(), collection);
        }
        Ok(())
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.manifest
            .list_collections()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Delete a collection and its data.
    ///
    /// Removes the collection from the manifest and deletes all associated files.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Collection does not exist
    /// - File deletion fails
    /// - Manifest save fails
    pub fn delete_collection(&mut self, name: &str) -> Result<()> {
        // Check if collection exists
        self.check_collection(name, true)?;

        // Close collection if open
        self.open_collections.remove(name);

        // Delete directory
        let collection_dir = self.collections_dir().join(name);
        if collection_dir.exists() {
            std::fs::remove_dir_all(&collection_dir)?;
        }

        // Remove from manifest
        self.manifest.remove_collection(name)?;
        self.save_manifest()?;

        Ok(())
    }

    /// Rename a collection.
    ///
    /// Updates the manifest and renames the collection directory.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Old collection does not exist
    /// - New name already exists
    /// - Directory rename fails
    /// - Manifest save fails
    pub fn rename_collection(&mut self, old_name: &str, new_name: &str) -> Result<()> {
        // Validate old collection exists
        self.check_collection(old_name, true)?;

        // Validate new name doesn't exist
        self.check_collection(new_name, false)?;

        // Close collection if open
        self.open_collections.remove(old_name);

        // Rename directory
        let collections_dir = self.collections_dir();
        let old_dir = collections_dir.join(old_name);
        let new_dir = collections_dir.join(new_name);

        if old_dir.exists() {
            std::fs::rename(&old_dir, &new_dir)?;
        }

        // Update manifest
        let mut info = self
            .manifest
            .get_collection(old_name)
            .ok_or_else(|| VhnswError::Payload(format!("Collection '{old_name}' not found in manifest")))?
            .clone();
        info.name = new_name.to_string();

        self.manifest.remove_collection(old_name)?;
        self.manifest.add_collection(info)?;
        self.save_manifest()?;

        Ok(())
    }

    /// Get a mutable reference to the default collection.
    ///
    /// Creates the default collection if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns error if collection creation or opening fails.
    pub fn default_collection(&mut self) -> Result<&mut Collection> {
        // Create default collection if it doesn't exist
        if self.manifest.get_collection(DEFAULT_COLLECTION_NAME).is_none() {
            let config = StorageConfig::default();
            self.create_collection(DEFAULT_COLLECTION_NAME, config)?;
        }

        self.get_collection_mut(DEFAULT_COLLECTION_NAME)
    }

    /// Save the manifest to disk.
    fn save_manifest(&self) -> Result<()> {
        let manifest_path = self.root.join("manifest.json");
        self.manifest.save(&manifest_path)
    }

    /// Get the root directory path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
}
