//! File metadata tracking for incremental updates.
//!
//! Maintains a JSON index mapping source files to their modification time,
//! size, and associated chunk IDs for efficient incremental processing.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Metadata for a single source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    /// Source file path.
    pub path: String,
    /// Last modification time (Unix timestamp).
    pub mtime: u64,
    /// File size in bytes.
    pub size: u64,
    /// IDs of chunks generated from this file.
    pub chunk_ids: Vec<u64>,
}

/// File index structure stored as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileIndex {
    /// Index format version.
    pub version: u32,
    /// Map from file path to metadata.
    pub files: HashMap<String, FileMetadata>,
}

impl FileIndex {
    /// Current index version.
    pub const VERSION: u32 = 1;

    /// Create a new empty file index.
    pub fn new() -> Self {
        Self {
            version: Self::VERSION,
            files: HashMap::new(),
        }
    }

    /// Add or update file metadata.
    pub fn update_file(&mut self, path: String, mtime: u64, size: u64, chunk_ids: Vec<u64>) {
        self.files.insert(path.clone(), FileMetadata {
            path,
            mtime,
            size,
            chunk_ids,
        });
    }

    /// Get metadata for a file.
    pub fn get_file(&self, path: &str) -> Option<&FileMetadata> {
        self.files.get(path)
    }

    /// Check if a file has been modified since last index.
    pub fn is_modified(&self, path: &str, mtime: u64, size: u64) -> bool {
        match self.files.get(path) {
            Some(meta) => meta.mtime != mtime || meta.size != size,
            None => true, // New file
        }
    }
}

impl Default for FileIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Load file index from database directory.
pub fn load_file_index(db_path: &Path) -> Result<FileIndex> {
    let index_path = db_path.join("file_index.json");

    if !index_path.exists() {
        return Ok(FileIndex::new());
    }

    let data = std::fs::read_to_string(&index_path)
        .with_context(|| format!("Failed to read file index at {}", index_path.display()))?;

    let index: FileIndex = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse file index at {}", index_path.display()))?;

    Ok(index)
}

/// Save file index to database directory.
pub fn save_file_index(db_path: &Path, index: &FileIndex) -> Result<()> {
    let index_path = db_path.join("file_index.json");

    let data = serde_json::to_string_pretty(index)
        .with_context(|| "Failed to serialize file index")?;

    std::fs::write(&index_path, data)
        .with_context(|| format!("Failed to write file index to {}", index_path.display()))?;

    Ok(())
}

/// Get file size in bytes.
pub fn get_file_size(path: &Path) -> Result<u64> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Failed to read metadata for {}", path.display()))?;

    Ok(metadata.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_index_new() {
        let index = FileIndex::new();
        assert_eq!(index.version, FileIndex::VERSION);
        assert!(index.files.is_empty());
    }

    #[test]
    fn test_file_index_update() {
        let mut index = FileIndex::new();
        index.update_file("test.md".to_string(), 123456, 1024, vec![1, 2, 3]);

        let meta = index.get_file("test.md").unwrap();
        assert_eq!(meta.path, "test.md");
        assert_eq!(meta.mtime, 123456);
        assert_eq!(meta.size, 1024);
        assert_eq!(meta.chunk_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_file_index_is_modified() {
        let mut index = FileIndex::new();

        // New file is always modified
        assert!(index.is_modified("test.md", 123456, 1024));

        // Add file
        index.update_file("test.md".to_string(), 123456, 1024, vec![1]);

        // Same mtime and size = not modified
        assert!(!index.is_modified("test.md", 123456, 1024));

        // Different mtime = modified
        assert!(index.is_modified("test.md", 123457, 1024));

        // Different size = modified
        assert!(index.is_modified("test.md", 123456, 2048));
    }

    #[test]
    fn test_save_load_file_index() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("file_index_test");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir)?;

        let mut index = FileIndex::new();
        index.update_file("doc1.md".to_string(), 111111, 512, vec![1, 2]);
        index.update_file("doc2.md".to_string(), 222222, 1024, vec![3, 4, 5]);

        save_file_index(&temp_dir, &index)?;

        let loaded = load_file_index(&temp_dir)?;
        assert_eq!(loaded.version, FileIndex::VERSION);
        assert_eq!(loaded.files.len(), 2);

        let doc1 = loaded.get_file("doc1.md").unwrap();
        assert_eq!(doc1.mtime, 111111);
        assert_eq!(doc1.chunk_ids, vec![1, 2]);

        let _ = std::fs::remove_dir_all(&temp_dir);
        Ok(())
    }
}
