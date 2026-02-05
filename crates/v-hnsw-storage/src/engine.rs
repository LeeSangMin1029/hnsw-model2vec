//! Unified storage engine coordinating vectors, payloads, and WAL.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use v_hnsw_core::{Dim, Payload, PointId, Result, VhnswError, VectorStore};

use crate::mmap_store::MmapVectorStore;
use crate::payload_store::FilePayloadStore;
use crate::wal::{Wal, WalRecord};

/// Configuration for creating a new storage engine.
pub struct StorageConfig {
    pub dim: Dim,
    pub initial_capacity: u32,
    pub checkpoint_threshold: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            dim: 384,
            initial_capacity: 10_000,
            checkpoint_threshold: 1000,
        }
    }
}

/// The unified storage engine that coordinates vectors, payloads, and WAL.
///
/// # Directory Layout
///
/// ```text
/// <dir>/
///   vectors.bin     - mmap vector data
///   payload.dat     - payload metadata
///   text.dat        - text chunks
///   payload.idx     - payload index (bincode)
///   text.idx        - text index (bincode)
///   vectors.idx     - id-to-slot mapping (bincode)
///   wal/            - WAL segments
/// ```
pub struct StorageEngine {
    dir: PathBuf,
    vectors: MmapVectorStore,
    payloads: FilePayloadStore,
    wal: Wal,
    config: StorageConfig,
    checkpoint_seq: u64,
    ops_since_checkpoint: usize,
    next_batch_id: u64,
}

impl StorageEngine {
    /// Create a new storage directory with empty stores.
    ///
    /// # Errors
    ///
    /// Returns error if directory creation or file initialization fails.
    pub fn create(dir: impl AsRef<Path>, config: StorageConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let vectors_path = dir.join("vectors.bin");
        let payload_path = dir.join("payload.dat");
        let text_path = dir.join("text.dat");
        let wal_dir = dir.join("wal");

        let vectors = MmapVectorStore::create(&vectors_path, config.dim, config.initial_capacity)?;
        let payloads = FilePayloadStore::create(&payload_path, &text_path)?;
        let wal = Wal::create(&wal_dir)?;

        Ok(Self {
            dir,
            vectors,
            payloads,
            wal,
            config,
            checkpoint_seq: 0,
            ops_since_checkpoint: 0,
            next_batch_id: 1,
        })
    }

    /// Open an existing storage directory, replay WAL for recovery.
    ///
    /// # Errors
    ///
    /// Returns error if directory doesn't exist, files are corrupt, or WAL replay fails.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();

        if !dir.exists() {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("storage directory not found: {}", dir.display()),
            )));
        }

        let vectors_path = dir.join("vectors.bin");
        let payload_path = dir.join("payload.dat");
        let text_path = dir.join("text.dat");
        let wal_dir = dir.join("wal");

        // Open stores
        let mut vectors = MmapVectorStore::open(&vectors_path)?;
        let mut payloads = FilePayloadStore::open(&payload_path, &text_path)?;
        let wal = Wal::open(&wal_dir)?;

        // Determine config from existing store
        let config = StorageConfig {
            dim: vectors.dim(),
            initial_capacity: vectors.capacity(),
            checkpoint_threshold: 1000, // Default, not persisted
        };

        // Load indices if they exist
        let vectors_idx_path = dir.join("vectors.idx");
        let payload_idx_path = dir.join("payload.idx");
        let text_idx_path = dir.join("text.idx");

        if vectors_idx_path.exists() {
            Self::load_vectors_idx(&mut vectors, &vectors_idx_path)?;
        }

        if payload_idx_path.exists() && text_idx_path.exists() {
            payloads.load_indices(&payload_idx_path, &text_idx_path)?;
        }

        let mut engine = Self {
            dir,
            vectors,
            payloads,
            wal,
            config,
            checkpoint_seq: 0,
            ops_since_checkpoint: 0,
            next_batch_id: 1,
        };

        // Replay WAL to recover un-checkpointed operations
        engine.replay_wal()?;

        Ok(engine)
    }

    /// Insert a point with vector + payload + text.
    ///
    /// 1. Append WalRecord::Insert to WAL
    /// 2. Insert vector into MmapVectorStore
    /// 3. Buffer payload and text in FilePayloadStore
    /// 4. Auto-checkpoint if threshold reached
    pub fn insert(&mut self, id: PointId, vector: &[f32], payload: Payload, text: &str) -> Result<()> {
        // Write to WAL first for durability
        self.wal.append(&WalRecord::Insert {
            id,
            vector: vector.to_vec(),
            payload: payload.clone(),
            text: text.to_string(),
        })?;

        // Insert into vector store
        self.vectors.insert(id, vector)?;

        // Buffer payload and text (will be flushed at checkpoint)
        self.payloads.buffer_payload(id, payload);
        self.payloads.buffer_text(id, text.to_string());

        self.ops_since_checkpoint += 1;

        // Auto-checkpoint if threshold reached
        if self.ops_since_checkpoint >= self.config.checkpoint_threshold {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Insert a batch of points atomically (BatchBegin / inserts / BatchEnd).
    ///
    /// Uses `wal.append_batch()` and `vectors.insert_batch()` to reduce
    /// per-record overhead. Checkpoint check runs once per batch.
    pub fn insert_batch(
        &mut self,
        batch: &[(PointId, &[f32], Payload, &str)],
    ) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        // Build WAL records: BatchBegin + N Inserts + BatchEnd
        let mut wal_records = Vec::with_capacity(batch.len() + 2);
        wal_records.push(WalRecord::BatchBegin { batch_id });
        for &(id, vector, ref payload, text) in batch {
            wal_records.push(WalRecord::Insert {
                id,
                vector: vector.to_vec(),
                payload: payload.clone(),
                text: text.to_string(),
            });
        }
        wal_records.push(WalRecord::BatchEnd { batch_id });

        // Single WAL write
        self.wal.append_batch(&wal_records)?;

        // Batch vector insert (single header write)
        let vec_batch: Vec<(PointId, &[f32])> =
            batch.iter().map(|&(id, v, _, _)| (id, v)).collect();
        self.vectors.insert_batch(&vec_batch)?;

        // Buffer payloads and texts
        for &(id, _, ref payload, text) in batch {
            self.payloads.buffer_payload(id, payload.clone());
            self.payloads.buffer_text(id, text.to_string());
        }

        self.ops_since_checkpoint += batch.len();

        // Auto-checkpoint once per batch
        if self.ops_since_checkpoint >= self.config.checkpoint_threshold {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Remove a point and all associated data.
    pub fn remove(&mut self, id: PointId) -> Result<()> {
        // Write to WAL first
        self.wal.append(&WalRecord::Remove { id })?;

        // Remove from vector store
        self.vectors.remove(id)?;

        // Remove from payload store
        self.payloads.mark_removed(id);

        self.ops_since_checkpoint += 1;

        // Auto-checkpoint if threshold reached
        if self.ops_since_checkpoint >= self.config.checkpoint_threshold {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Replace all chunks from a source document.
    ///
    /// Uses BatchBegin/BatchEnd for atomic batch.
    pub fn replace_source(
        &mut self,
        source_path: &str,
        new_chunks: Vec<(PointId, Vec<f32>, Payload, String)>,
    ) -> Result<()> {
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        // Begin batch
        self.wal.append(&WalRecord::BatchBegin { batch_id })?;

        // Remove old chunks
        let old_ids = self.payloads.points_by_source(source_path);
        let old_count = old_ids.len();
        for id in old_ids {
            self.wal.append(&WalRecord::Remove { id })?;
            self.vectors.remove(id)?;
            self.payloads.mark_removed(id);
        }

        // Insert new chunks
        let new_count = new_chunks.len();
        for (id, vector, payload, text) in new_chunks {
            self.wal.append(&WalRecord::Insert {
                id,
                vector: vector.clone(),
                payload: payload.clone(),
                text: text.clone(),
            })?;

            self.vectors.insert(id, &vector)?;
            self.payloads.buffer_payload(id, payload);
            self.payloads.buffer_text(id, text);
        }

        // End batch
        self.wal.append(&WalRecord::BatchEnd { batch_id })?;

        self.ops_since_checkpoint += old_count + new_count + 2; // +2 for batch markers

        // Auto-checkpoint if threshold reached
        if self.ops_since_checkpoint >= self.config.checkpoint_threshold {
            self.checkpoint()?;
        }

        Ok(())
    }

    /// Flush all data to disk and create a checkpoint.
    pub fn checkpoint(&mut self) -> Result<()> {
        // Flush buffered payloads and texts to disk
        self.payloads.flush_buffers()?;
        self.payloads.flush()?;

        // Flush vector store mmap
        self.vectors.flush()?;

        // Save all index files
        self.save_indices()?;

        // Write checkpoint record to WAL
        self.checkpoint_seq += 1;
        let point_count = self.vectors.len();
        self.wal.checkpoint(self.checkpoint_seq, point_count)?;

        // Purge old WAL segments
        self.wal.purge_old_segments()?;

        // Reset counter
        self.ops_since_checkpoint = 0;

        Ok(())
    }

    /// Access the vector store (read-only, for passing to HnswGraph).
    pub fn vector_store(&self) -> &MmapVectorStore {
        &self.vectors
    }

    /// Access the payload store.
    pub fn payload_store(&self) -> &FilePayloadStore {
        &self.payloads
    }

    /// Number of live points.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.len() == 0
    }

    // --- Private methods ---

    /// Replay WAL records into stores.
    fn replay_wal(&mut self) -> Result<()> {
        let records = self.wal.replay()?;

        for record in records {
            match record {
                WalRecord::Insert { id, vector, payload, text } => {
                    // Insert vector
                    self.vectors.insert(id, &vector)?;

                    // Buffer payload and text
                    self.payloads.buffer_payload(id, payload);
                    self.payloads.buffer_text(id, text);
                }
                WalRecord::Remove { id } => {
                    // Remove from vector store (ignore errors if not found)
                    let _ = self.vectors.remove(id);

                    // Mark removed in payload store
                    self.payloads.mark_removed(id);
                }
                WalRecord::Checkpoint { seq, .. } => {
                    // Update checkpoint sequence
                    if seq > self.checkpoint_seq {
                        self.checkpoint_seq = seq;
                    }
                }
                WalRecord::BatchBegin { .. } | WalRecord::BatchEnd { .. } => {
                    // Batch markers are filtered during replay
                }
            }
        }

        // Flush buffers after replay to ensure consistency
        self.payloads.flush_buffers()?;

        Ok(())
    }

    /// Save all index files.
    fn save_indices(&self) -> Result<()> {
        let vectors_idx_path = self.dir.join("vectors.idx");
        let payload_idx_path = self.dir.join("payload.idx");
        let text_idx_path = self.dir.join("text.idx");

        // Save vector id_to_slot mapping
        Self::save_vectors_idx(&self.vectors, &vectors_idx_path)?;

        // Save payload and text indices
        self.payloads.save_indices(&payload_idx_path, &text_idx_path)?;

        Ok(())
    }

    /// Save vectors.idx (id_to_slot mapping).
    fn save_vectors_idx(vectors: &MmapVectorStore, path: &Path) -> Result<()> {
        let config = bincode::config::standard();
        let data = bincode::encode_to_vec(vectors.id_map(), config)
            .map_err(|e| VhnswError::Storage(std::io::Error::other(
                format!("failed to encode vectors index: {e}"),
            )))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load vectors.idx and restore id_to_slot mapping.
    fn load_vectors_idx(vectors: &mut MmapVectorStore, path: &Path) -> Result<()> {
        let data = std::fs::read(path)?;
        let config = bincode::config::standard();
        let (id_map, _): (HashMap<PointId, u32>, usize) = bincode::decode_from_slice(&data, config)
            .map_err(|e| VhnswError::Storage(std::io::Error::other(
                format!("failed to decode vectors index: {e}"),
            )))?;
        vectors.restore_id_map(id_map);
        Ok(())
    }
}
