//! Collection manifest registry for multi-collection database.
//!
//! Tracks metadata for all collections in the database:
//! - Dimension and metric per collection
//! - Creation timestamp and point count
//! - Persistent across restarts (manifest.json)

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use serde::{Deserialize, Serialize};
use v_hnsw_core::{Dim, Result, VhnswError};

/// Current manifest format version (Phase 8).
pub const MANIFEST_VERSION: u32 = 2;

/// Metadata for a single collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// Collection name (unique identifier).
    pub name: String,
    /// Vector dimension.
    pub dim: Dim,
    /// Distance metric ("cosine", "l2", "dot").
    pub metric: String,
    /// Unix timestamp (seconds) when collection was created.
    pub created_at: u64,
    /// Number of points in this collection.
    pub count: usize,
}

/// Manifest tracking all collections in the database.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Format version (MANIFEST_VERSION).
    pub version: u32,
    /// Map of collection name -> metadata.
    pub collections: HashMap<String, CollectionInfo>,
}

impl Manifest {
    /// Creates a new empty manifest.
    pub fn new() -> Self {
        Self {
            version: MANIFEST_VERSION,
            collections: HashMap::new(),
        }
    }

    /// Loads manifest from disk, or creates empty if not found.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new());
        }

        let file = File::open(path).map_err(|e| {
            VhnswError::Storage(std::io::Error::other(
                format!("Failed to open manifest at {:?}: {}", path, e),
            ))
        })?;
        let reader = BufReader::new(file);
        let manifest: Manifest = serde_json::from_reader(reader).map_err(|e| {
            VhnswError::Payload(format!("Failed to parse manifest: {}", e))
        })?;

        // Validate version
        if manifest.version != MANIFEST_VERSION {
            return Err(VhnswError::Payload(format!(
                "Manifest version mismatch: expected {}, found {}",
                MANIFEST_VERSION, manifest.version
            )));
        }

        Ok(manifest)
    }

    /// Saves manifest to disk (atomic write via temp file).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let temp_path = path.with_extension("tmp");

        // Write to temp file first
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)
            .map_err(|e| {
                VhnswError::Storage(std::io::Error::other(
                    format!("Failed to create temp manifest at {:?}: {}", temp_path, e),
                ))
            })?;

        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self).map_err(|e| {
            VhnswError::Payload(format!("Failed to serialize manifest: {}", e))
        })?;

        // Atomic rename
        std::fs::rename(&temp_path, path).map_err(|e| {
            VhnswError::Storage(std::io::Error::other(
                format!("Failed to rename manifest: {}", e),
            ))
        })?;

        Ok(())
    }

    /// Adds a new collection to the manifest.
    ///
    /// # Errors
    ///
    /// Returns error if collection name already exists.
    pub fn add_collection(&mut self, info: CollectionInfo) -> Result<()> {
        if self.collections.contains_key(&info.name) {
            return Err(VhnswError::Payload(format!(
                "Collection '{}' already exists",
                info.name
            )));
        }
        self.collections.insert(info.name.clone(), info);
        Ok(())
    }

    /// Removes a collection from the manifest.
    ///
    /// # Errors
    ///
    /// Returns error if collection does not exist.
    pub fn remove_collection(&mut self, name: &str) -> Result<()> {
        if self.collections.remove(name).is_none() {
            return Err(VhnswError::Payload(format!(
                "Collection '{}' not found",
                name
            )));
        }
        Ok(())
    }

    /// Retrieves collection metadata.
    pub fn get_collection(&self, name: &str) -> Option<&CollectionInfo> {
        self.collections.get(name)
    }

    /// Updates the point count for a collection.
    pub fn update_count(&mut self, name: &str, count: usize) -> Result<()> {
        let info = self.collections.get_mut(name).ok_or_else(|| {
            VhnswError::Payload(format!("Collection '{}' not found", name))
        })?;
        info.count = count;
        Ok(())
    }

    /// Lists all collection names.
    pub fn list_collections(&self) -> Vec<&str> {
        self.collections.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

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
}
