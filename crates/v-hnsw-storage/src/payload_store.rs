//! File-based payload and text storage with memory buffering.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use roaring::RoaringTreemap;
use v_hnsw_core::{Payload, PayloadStore, PointId, Result, VhnswError};

use crate::fsst_text::CompressedTextReader;

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
    source_index: HashMap<String, Vec<PointId>>,
    /// Secondary index: tag -> Roaring Treemap of PointIds (u64)
    tag_index: HashMap<String, RoaringTreemap>,
    /// In-memory payload cache for WAL buffer
    pending_payloads: HashMap<PointId, Payload>,
    /// In-memory text cache for WAL buffer
    pending_texts: HashMap<PointId, String>,
    /// Zstd compressed text reader (loaded if text.zst exists)
    compressed_reader: Option<CompressedTextReader>,
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
            tag_index: HashMap::new(),  // RoaringBitmap
            pending_payloads: HashMap::new(),
            pending_texts: HashMap::new(),
            compressed_reader: None,
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
            tag_index: HashMap::new(),  // RoaringBitmap
            pending_payloads: HashMap::new(),
            pending_texts: HashMap::new(),
            compressed_reader: None,
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
        let source_points = self.source_index.entry(payload.source.clone()).or_default();
        if !source_points.contains(&id) {
            source_points.push(id);
        }

        // Update tag index (Roaring Treemap — u64 safe)
        for tag in &payload.tags {
            self.tag_index.entry(tag.clone()).or_default().insert(id);
        }

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
        let source_points = self.source_index.entry(payload.source.clone()).or_default();
        if !source_points.contains(&id) {
            source_points.push(id);
        }

        // Update tag index (Roaring Treemap — u64 safe)
        for tag in &payload.tags {
            self.tag_index.entry(tag.clone()).or_default().insert(id);
        }

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
            .cloned()
            .unwrap_or_default()
    }

    /// Get all point IDs with a given tag.
    pub fn points_by_tag(&self, tag: &str) -> Vec<PointId> {
        self.tag_index
            .get(tag)
            .map(|bm| bm.iter().collect())
            .unwrap_or_default()
    }

    /// Get all point IDs matching ALL given tags (AND logic).
    ///
    /// Uses Roaring Treemap intersection for efficient set operations.
    pub fn points_by_tags(&self, tags: &[String]) -> Vec<PointId> {
        if tags.is_empty() {
            return Vec::new();
        }

        let mut result: Option<RoaringTreemap> = None;

        for tag in tags {
            if let Some(bitmap) = self.tag_index.get(tag) {
                match &mut result {
                    None => {
                        result = Some(bitmap.clone());
                    }
                    Some(current) => {
                        *current &= bitmap;
                    }
                }
            } else {
                return Vec::new();
            }
        }

        result
            .map(|bm| bm.iter().collect())
            .unwrap_or_default()
    }

    /// Read all raw text bytes from text.dat.
    ///
    /// Used by FSST compression during build-index to train
    /// the compressor on the full corpus.
    pub fn all_text_bytes(&self) -> Result<Vec<(PointId, Vec<u8>)>> {
        let mut texts = Vec::with_capacity(self.text_index.len());
        for (&id, &(offset, length)) in &self.text_index {
            let mut handle = self.text_file.try_clone()?;
            handle.seek(SeekFrom::Start(offset))?;
            let mut data = vec![0u8; length as usize];
            handle.read_exact(&mut data)?;
            texts.push((id, data));
        }
        Ok(texts)
    }

    /// Mark a point as removed (remove from all indices and buffers).
    pub fn mark_removed(&mut self, id: PointId) {
        self.payload_index.remove(&id);
        self.text_index.remove(&id);
        self.pending_payloads.remove(&id);
        self.pending_texts.remove(&id);

        // Remove from source index
        for points in self.source_index.values_mut() {
            points.retain(|&pid| pid != id);
        }

        // Remove from tag index (Roaring Treemap)
        for bitmap in self.tag_index.values_mut() {
            bitmap.remove(id);
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

        // Save tag index
        self.save_tag_index()?;

        Ok(())
    }

    /// Save tag index to disk using Roaring Treemap serialization.
    fn save_tag_index(&self) -> Result<()> {
        let tag_idx_path = self.payload_path.with_file_name("tag_index.roaring");

        // Format v2: [magic: u32=0x524D4150] [num_tags: u32] [tag_name_len: u32, tag_name: bytes, bitmap_len: u32, bitmap: bytes]*
        let mut buf = Vec::new();
        buf.extend_from_slice(&TREEMAP_MAGIC.to_le_bytes());
        let num_tags = self.tag_index.len() as u32;
        buf.extend_from_slice(&num_tags.to_le_bytes());

        for (tag, bitmap) in &self.tag_index {
            let tag_bytes = tag.as_bytes();
            buf.extend_from_slice(&(tag_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(tag_bytes);

            let mut bm_buf = Vec::new();
            bitmap.serialize_into(&mut bm_buf)
                .map_err(|e| VhnswError::Payload(format!("failed to serialize treemap: {e}")))?;
            buf.extend_from_slice(&(bm_buf.len() as u32).to_le_bytes());
            buf.extend_from_slice(&bm_buf);
        }

        std::fs::write(tag_idx_path, buf)?;

        // Remove old formats
        let old_path = self.payload_path.with_file_name("tag_index.dat");
        let _ = std::fs::remove_file(old_path);

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

        // Load tag index
        self.load_tag_index()?;

        // Load zstd compressed text reader if available
        let dir = self.text_path.parent().unwrap_or(Path::new("."));
        self.compressed_reader = CompressedTextReader::load(dir)?;

        // Rebuild source index from payload index
        self.source_index.clear();
        for (&id, &(offset, length)) in &self.payload_index {
            if let Ok(Some(payload)) = self.read_payload_at(offset, length) {
                let source_points = self.source_index.entry(payload.source.clone()).or_default();
                if !source_points.contains(&id) {
                    source_points.push(id);
                }
            }
        }

        Ok(())
    }

    /// Load tag index from disk (Treemap format, with auto-migration from old formats).
    fn load_tag_index(&mut self) -> Result<()> {
        let roaring_path = self.payload_path.with_file_name("tag_index.roaring");

        if roaring_path.exists() {
            let data = std::fs::read(&roaring_path)?;
            if is_treemap_format(&data) {
                self.tag_index = deserialize_treemap_tag_index(&data)?;
            } else {
                // Old RoaringBitmap (u32) format — rebuild with correct u64 IDs
                tracing::info!("Migrating tag index from u32 to u64 format");
                self.rebuild_tag_index()?;
                self.save_tag_index()?;
            }
        } else {
            self.rebuild_tag_index()?;
        }

        Ok(())
    }

    /// Rebuild tag index from payload index.
    fn rebuild_tag_index(&mut self) -> Result<()> {
        self.tag_index.clear();
        for (&id, &(offset, length)) in &self.payload_index {
            if let Ok(Some(payload)) = self.read_payload_at(offset, length) {
                for tag in &payload.tags {
                    self.tag_index.entry(tag.clone()).or_default().insert(id);
                }
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

        // Must be in text_index to be a valid (non-removed) document
        let raw_entry = self.text_index.get(&id);
        if raw_entry.is_none() {
            return Ok(None);
        }

        // Try zstd compressed store first (less I/O)
        if let Some(reader) = &self.compressed_reader
            && let Some(text) = reader.get_text(id)? {
                return Ok(Some(text));
            }

        // Fall back to raw text.dat
        if let Some(&(offset, length)) = raw_entry {
            return self.read_text_at(offset, length);
        }

        Ok(None)
    }
}

/// Magic number for RoaringTreemap format (u64-safe).
const TREEMAP_MAGIC: u32 = 0x524D_4150; // "RMAP"

/// Check if data starts with treemap magic.
fn is_treemap_format(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    magic == TREEMAP_MAGIC
}

/// Deserialize Roaring Treemap tag index from bytes (v2 format with magic).
fn deserialize_treemap_tag_index(data: &[u8]) -> Result<HashMap<String, RoaringTreemap>> {
    use std::io::Cursor;

    let mut cursor = Cursor::new(data);
    let mut buf4 = [0u8; 4];

    // Skip magic
    std::io::Read::read_exact(&mut cursor, &mut buf4)?;

    let mut tag_index = HashMap::new();

    std::io::Read::read_exact(&mut cursor, &mut buf4)?;
    let num_tags = u32::from_le_bytes(buf4) as usize;

    for _ in 0..num_tags {
        // Read tag name
        std::io::Read::read_exact(&mut cursor, &mut buf4)?;
        let tag_len = u32::from_le_bytes(buf4) as usize;
        let mut tag_bytes = vec![0u8; tag_len];
        std::io::Read::read_exact(&mut cursor, &mut tag_bytes)?;
        let tag = String::from_utf8(tag_bytes)
            .map_err(|e| VhnswError::Payload(format!("invalid UTF-8 in tag: {e}")))?;

        // Read treemap
        std::io::Read::read_exact(&mut cursor, &mut buf4)?;
        let bm_len = u32::from_le_bytes(buf4) as usize;
        let mut bm_data = vec![0u8; bm_len];
        std::io::Read::read_exact(&mut cursor, &mut bm_data)?;
        let bitmap = RoaringTreemap::deserialize_from(&bm_data[..])
            .map_err(|e| VhnswError::Payload(format!("failed to deserialize treemap: {e}")))?;

        tag_index.insert(tag, bitmap);
    }

    Ok(tag_index)
}
