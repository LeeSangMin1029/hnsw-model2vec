//! File-based payload and text storage with memory buffering.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use v_hnsw_core::{Payload, PayloadStore, PointId, Result, VhnswError};

/// File-based storage for payload metadata and text chunks.
///
/// Uses two separate files for payloads and texts, with in-memory indices
/// to track offsets. Supports buffering for WAL-backed writes.
pub struct FilePayloadStore {
    #[allow(dead_code)]
    payload_path: PathBuf,
    #[allow(dead_code)]
    text_path: PathBuf,
    payload_file: File,
    text_file: File,
    /// PointId -> (offset, length) in payload.dat
    payload_index: HashMap<PointId, (u64, u32)>,
    /// PointId -> (offset, length) in text.dat
    text_index: HashMap<PointId, (u64, u32)>,
    /// Secondary index: source path -> set of PointIds
    source_index: HashMap<String, HashSet<PointId>>,
    /// In-memory payload cache for WAL buffer
    pending_payloads: HashMap<PointId, Payload>,
    /// In-memory text cache for WAL buffer
    pending_texts: HashMap<PointId, String>,
}

impl FilePayloadStore {
    /// Create a new payload store with fresh files.
    pub fn create(payload_path: impl AsRef<Path>, text_path: impl AsRef<Path>) -> Result<Self> {
        let payload_path = payload_path.as_ref().to_path_buf();
        let text_path = text_path.as_ref().to_path_buf();

        let payload_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(&payload_path)?;

        let text_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(&text_path)?;

        Ok(Self {
            payload_path,
            text_path,
            payload_file,
            text_file,
            payload_index: HashMap::new(),
            text_index: HashMap::new(),
            source_index: HashMap::new(),
            pending_payloads: HashMap::new(),
            pending_texts: HashMap::new(),
        })
    }

    /// Open an existing payload store.
    pub fn open(payload_path: impl AsRef<Path>, text_path: impl AsRef<Path>) -> Result<Self> {
        let payload_path = payload_path.as_ref().to_path_buf();
        let text_path = text_path.as_ref().to_path_buf();

        let payload_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .read(true)
            .open(&payload_path)?;

        let text_file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .read(true)
            .open(&text_path)?;

        Ok(Self {
            payload_path,
            text_path,
            payload_file,
            text_file,
            payload_index: HashMap::new(),
            text_index: HashMap::new(),
            source_index: HashMap::new(),
            pending_payloads: HashMap::new(),
            pending_texts: HashMap::new(),
        })
    }

    /// Write a payload to the file and update the index.
    pub fn write_payload(&mut self, id: PointId, payload: &Payload) -> Result<()> {
        let config = bincode::config::standard();
        let data = bincode::encode_to_vec(payload, config)
            .map_err(|e| VhnswError::Payload(format!("failed to encode payload: {e}")))?;

        let offset = self.payload_file.seek(SeekFrom::End(0))?;
        let length = data.len() as u32;

        self.payload_file.write_all(&data)?;

        self.payload_index.insert(id, (offset, length));

        // Update source index
        self.source_index
            .entry(payload.source.clone())
            .or_default()
            .insert(id);

        Ok(())
    }

    /// Write text to the file and update the index.
    pub fn write_text(&mut self, id: PointId, text: &str) -> Result<()> {
        let data = text.as_bytes();
        let offset = self.text_file.seek(SeekFrom::End(0))?;
        let length = data.len() as u32;

        self.text_file.write_all(data)?;

        self.text_index.insert(id, (offset, length));

        Ok(())
    }

    /// Buffer a payload in memory (for WAL-backed writes before flush).
    pub fn buffer_payload(&mut self, id: PointId, payload: Payload) {
        self.source_index
            .entry(payload.source.clone())
            .or_default()
            .insert(id);
        self.pending_payloads.insert(id, payload);
    }

    /// Buffer text in memory.
    pub fn buffer_text(&mut self, id: PointId, text: String) {
        self.pending_texts.insert(id, text);
    }

    /// Flush all buffered payloads and texts to disk.
    pub fn flush_buffers(&mut self) -> Result<()> {
        // Write buffered payloads
        let payloads: Vec<_> = self.pending_payloads.drain().collect();
        for (id, payload) in payloads {
            self.write_payload(id, &payload)?;
        }

        // Write buffered texts
        let texts: Vec<_> = self.pending_texts.drain().collect();
        for (id, text) in texts {
            self.write_text(id, &text)?;
        }

        self.payload_file.flush()?;
        self.text_file.flush()?;

        Ok(())
    }

    /// Get all point IDs associated with a given source path.
    pub fn points_by_source(&self, source: &str) -> Vec<PointId> {
        self.source_index
            .get(source)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Mark a point as removed (remove from all indices and buffers).
    pub fn mark_removed(&mut self, id: PointId) {
        self.payload_index.remove(&id);
        self.text_index.remove(&id);
        self.pending_payloads.remove(&id);
        self.pending_texts.remove(&id);

        // Remove from source index
        for points in self.source_index.values_mut() {
            points.remove(&id);
        }
    }

    /// Flush file handles to disk.
    pub fn flush(&self) -> Result<()> {
        self.payload_file.sync_all()?;
        self.text_file.sync_all()?;
        Ok(())
    }

    /// Save indices to disk as bincode-serialized files.
    pub fn save_indices(
        &self,
        payload_idx_path: impl AsRef<Path>,
        text_idx_path: impl AsRef<Path>,
    ) -> Result<()> {
        let config = bincode::config::standard();

        // Save payload index
        let payload_data = bincode::encode_to_vec(&self.payload_index, config)
            .map_err(|e| VhnswError::Payload(format!("failed to encode payload index: {e}")))?;
        std::fs::write(payload_idx_path, payload_data)?;

        // Save text index
        let text_data = bincode::encode_to_vec(&self.text_index, config)
            .map_err(|e| VhnswError::Payload(format!("failed to encode text index: {e}")))?;
        std::fs::write(text_idx_path, text_data)?;

        Ok(())
    }

    /// Load indices from disk.
    pub fn load_indices(
        &mut self,
        payload_idx_path: impl AsRef<Path>,
        text_idx_path: impl AsRef<Path>,
    ) -> Result<()> {
        let config = bincode::config::standard();

        // Load payload index
        let payload_data = std::fs::read(payload_idx_path)?;
        let (index, _): (HashMap<PointId, (u64, u32)>, usize) =
            bincode::decode_from_slice(&payload_data, config)
                .map_err(|e| VhnswError::Payload(format!("failed to decode payload index: {e}")))?;
        self.payload_index = index;

        // Load text index
        let text_data = std::fs::read(text_idx_path)?;
        let (index, _): (HashMap<PointId, (u64, u32)>, usize) =
            bincode::decode_from_slice(&text_data, config)
                .map_err(|e| VhnswError::Payload(format!("failed to decode text index: {e}")))?;
        self.text_index = index;

        // Rebuild source index from payload index
        self.source_index.clear();
        for (&id, &(offset, length)) in &self.payload_index {
            if let Ok(Some(payload)) = self.read_payload_at(offset, length) {
                self.source_index
                    .entry(payload.source.clone())
                    .or_default()
                    .insert(id);
            }
        }

        Ok(())
    }

    // --- Private helpers ---

    fn read_payload_at(&self, offset: u64, length: u32) -> Result<Option<Payload>> {
        let mut handle = self.payload_file.try_clone()?;
        handle.seek(SeekFrom::Start(offset))?;

        let mut data = vec![0u8; length as usize];
        handle.read_exact(&mut data)?;

        let config = bincode::config::standard();
        let (payload, _): (Payload, usize) = bincode::decode_from_slice(&data, config)
            .map_err(|e| VhnswError::Payload(format!("failed to decode payload: {e}")))?;

        Ok(Some(payload))
    }

    fn read_text_at(&self, offset: u64, length: u32) -> Result<Option<String>> {
        let mut handle = self.text_file.try_clone()?;
        handle.seek(SeekFrom::Start(offset))?;

        let mut data = vec![0u8; length as usize];
        handle.read_exact(&mut data)?;

        let text = String::from_utf8(data)
            .map_err(|e| VhnswError::Payload(format!("invalid UTF-8 in text: {e}")))?;

        Ok(Some(text))
    }
}

impl PayloadStore for FilePayloadStore {
    fn get_payload(&self, id: PointId) -> Result<Option<Payload>> {
        // Check pending buffer first
        if let Some(payload) = self.pending_payloads.get(&id) {
            return Ok(Some(payload.clone()));
        }

        // Read from file via index
        if let Some(&(offset, length)) = self.payload_index.get(&id) {
            return self.read_payload_at(offset, length);
        }

        Ok(None)
    }

    fn set_payload(&mut self, id: PointId, payload: Payload) -> Result<()> {
        // Buffer the payload (will be flushed during checkpoint)
        self.buffer_payload(id, payload);
        Ok(())
    }

    fn remove_payload(&mut self, id: PointId) -> Result<()> {
        self.mark_removed(id);
        Ok(())
    }

    fn get_text(&self, id: PointId) -> Result<Option<String>> {
        // Check pending buffer first
        if let Some(text) = self.pending_texts.get(&id) {
            return Ok(Some(text.clone()));
        }

        // Read from file via index
        if let Some(&(offset, length)) = self.text_index.get(&id) {
            return self.read_text_at(offset, length);
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_payload_store_create() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("payload_test_create");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir)?;

        let payload_path = temp_dir.join("payload.dat");
        let text_path = temp_dir.join("text.dat");

        let mut store = FilePayloadStore::create(&payload_path, &text_path)?;

        let payload = Payload {
            source: "test.md".to_string(),
            tags: vec!["tag1".to_string()],
            created_at: 123456,
            source_modified_at: 123456,
            chunk_index: 0,
            chunk_total: 1,
            custom: HashMap::new(),
        };

        store.buffer_payload(1, payload.clone());
        store.buffer_text(1, "test text".to_string());

        store.flush_buffers()?;

        // Read back
        let retrieved = store.get_payload(1)?;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.as_ref().map(|p| &p.source), Some(&"test.md".to_string()));

        let text = store.get_text(1)?;
        assert_eq!(text.as_deref(), Some("test text"));

        let _ = std::fs::remove_dir_all(&temp_dir);
        Ok(())
    }

    #[test]
    fn test_payload_store_buffering() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("payload_test_buffer");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir)?;

        let payload_path = temp_dir.join("payload.dat");
        let text_path = temp_dir.join("text.dat");

        let mut store = FilePayloadStore::create(&payload_path, &text_path)?;

        let payload = Payload {
            source: "test.md".to_string(),
            tags: vec![],
            created_at: 0,
            source_modified_at: 0,
            chunk_index: 0,
            chunk_total: 1,
            custom: HashMap::new(),
        };

        // Buffer without flushing
        store.buffer_payload(1, payload.clone());
        store.buffer_text(1, "buffered".to_string());

        // Should be readable from buffer
        assert!(store.get_payload(1)?.is_some());
        assert_eq!(store.get_text(1)?.as_deref(), Some("buffered"));

        // Flush and verify
        store.flush_buffers()?;
        assert!(store.get_payload(1)?.is_some());

        let _ = std::fs::remove_dir_all(&temp_dir);
        Ok(())
    }

    #[test]
    fn test_payload_store_source_index() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("payload_test_source");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir)?;

        let payload_path = temp_dir.join("payload.dat");
        let text_path = temp_dir.join("text.dat");

        let mut store = FilePayloadStore::create(&payload_path, &text_path)?;

        let payload1 = Payload {
            source: "doc1.md".to_string(),
            tags: vec![],
            created_at: 0,
            source_modified_at: 0,
            chunk_index: 0,
            chunk_total: 2,
            custom: HashMap::new(),
        };

        let payload2 = Payload {
            source: "doc1.md".to_string(),
            tags: vec![],
            created_at: 0,
            source_modified_at: 0,
            chunk_index: 1,
            chunk_total: 2,
            custom: HashMap::new(),
        };

        store.buffer_payload(1, payload1);
        store.buffer_payload(2, payload2);
        store.flush_buffers()?;

        let points = store.points_by_source("doc1.md");
        assert_eq!(points.len(), 2);
        assert!(points.contains(&1));
        assert!(points.contains(&2));

        let _ = std::fs::remove_dir_all(&temp_dir);
        Ok(())
    }
}
