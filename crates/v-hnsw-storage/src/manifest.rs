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
