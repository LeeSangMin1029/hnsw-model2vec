//! Memory-mapped SQ8 quantized vector storage.
//!
//! Stores u8 vectors (one byte per dimension) in a flat mmap file.
//! Used alongside the original f32 `MmapVectorStore` for 2-stage search:
//! SQ8 for candidate selection → f32 for final rescore.

#![allow(unsafe_code)]

use memmap2::MmapMut;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use v_hnsw_core::{Dim, PointId, Result, VhnswError};

/// File magic number: "VHNSW_Q8"
const MAGIC: u64 = 0x5648_4E53_575F_5138;
/// File format version
const FORMAT_VERSION: u32 = 1;
/// Header size in bytes
const HEADER_SIZE: usize = 64;

/// Memory-mapped SQ8 vector storage.
///
/// # File Layout
///
/// ```text
/// [Header: 64 bytes]
/// [Slot 0: dim bytes (u8)]
/// [Slot 1: dim bytes (u8)]
/// ...
/// ```
pub struct Sq8VectorStore {
    #[allow(dead_code)]
    path: PathBuf,
    file: File,
    mmap: MmapMut,
    dim: Dim,
    capacity: u32,
    id_to_slot: HashMap<PointId, u32>,
    next_slot: u32,
    live_count: usize,
}

impl Sq8VectorStore {
    /// Create a new SQ8 store.
    pub fn create(path: impl AsRef<Path>, dim: Dim, capacity: u32) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file_size = HEADER_SIZE + (capacity as usize) * dim;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.set_len(file_size as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let mut store = Self {
            path,
            file,
            mmap,
            dim,
            capacity,
            id_to_slot: HashMap::new(),
            next_slot: 0,
            live_count: 0,
        };

        store.write_header()?;
        store.mmap.flush()?;
        Ok(store)
    }

    /// Open an existing SQ8 store.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(VhnswError::InvalidArgument(
                "SQ8 store file too small".into(),
            ));
        }

        let magic = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        if magic != MAGIC {
            return Err(VhnswError::InvalidArgument(format!(
                "SQ8 store bad magic: {magic:#x}"
            )));
        }

        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(VhnswError::InvalidArgument(format!(
                "SQ8 store version mismatch: expected {FORMAT_VERSION}, got {version}"
            )));
        }

        let dim = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let live_count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let capacity = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as u32;

        Ok(Self {
            path,
            file,
            mmap,
            dim,
            capacity,
            id_to_slot: HashMap::with_capacity(live_count),
            next_slot: 0,
            live_count,
        })
    }

    /// Insert a quantized vector at a specific slot.
    ///
    /// Used during build to match the slot assignment of the main f32 store,
    /// so that `restore_id_map` works correctly after reopening.
    pub fn insert_at(&mut self, id: PointId, slot: u32, codes: &[u8]) -> Result<()> {
        if codes.len() != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: codes.len(),
            });
        }

        if slot >= self.capacity {
            self.grow(slot.saturating_mul(2).max(slot + 1))?;
        }
        let offset = HEADER_SIZE + (slot as usize) * self.dim;
        self.mmap[offset..offset + self.dim].copy_from_slice(codes);

        if !self.id_to_slot.contains_key(&id) {
            self.live_count += 1;
        }
        self.id_to_slot.insert(id, slot);
        if slot >= self.next_slot {
            self.next_slot = slot + 1;
        }
        Ok(())
    }

    /// Insert a quantized vector.
    pub fn insert(&mut self, id: PointId, codes: &[u8]) -> Result<()> {
        if codes.len() != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: codes.len(),
            });
        }

        if let Some(&slot) = self.id_to_slot.get(&id) {
            // Overwrite existing
            let offset = HEADER_SIZE + (slot as usize) * self.dim;
            self.mmap[offset..offset + self.dim].copy_from_slice(codes);
        } else {
            // Need new slot
            if self.next_slot >= self.capacity {
                self.grow(self.capacity.saturating_mul(2).max(self.next_slot + 1))?;
            }
            let slot = self.next_slot;
            self.next_slot += 1;
            let offset = HEADER_SIZE + (slot as usize) * self.dim;
            self.mmap[offset..offset + self.dim].copy_from_slice(codes);
            self.id_to_slot.insert(id, slot);
            self.live_count += 1;
        }

        Ok(())
    }

    /// Insert a batch of quantized vectors.
    pub fn insert_batch(&mut self, batch: &[(PointId, &[u8])]) -> Result<()> {
        // Pre-grow if needed
        let new_count = batch
            .iter()
            .filter(|(id, _)| !self.id_to_slot.contains_key(id))
            .count() as u32;
        let needed = self.next_slot + new_count;
        if needed > self.capacity {
            self.grow(needed.max(self.capacity.saturating_mul(2)))?;
        }

        for &(id, codes) in batch {
            self.insert(id, codes)?;
        }
        self.write_header()?;
        Ok(())
    }

    /// Get a quantized vector by point ID.
    #[inline]
    pub fn get(&self, id: PointId) -> Result<&[u8]> {
        let &slot = self
            .id_to_slot
            .get(&id)
            .ok_or(VhnswError::PointNotFound(id))?;
        let offset = HEADER_SIZE + (slot as usize) * self.dim;
        Ok(&self.mmap[offset..offset + self.dim])
    }

    /// Restore the id→slot mapping from an external source.
    ///
    /// Used after loading: the main `MmapVectorStore` maintains the authoritative
    /// id→slot map, and this store mirrors it (same slot layout).
    pub fn restore_id_map(&mut self, id_map: &HashMap<PointId, u32>) {
        self.id_to_slot = id_map.clone();
        self.live_count = id_map.len();
        if let Some(&max_slot) = id_map.values().max() {
            self.next_slot = max_slot + 1;
        }
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Dimension.
    pub fn dim(&self) -> Dim {
        self.dim
    }

    fn write_header(&mut self) -> Result<()> {
        self.mmap[0..8].copy_from_slice(&MAGIC.to_le_bytes());
        self.mmap[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        self.mmap[12..16].copy_from_slice(&(self.dim as u32).to_le_bytes());
        self.mmap[16..24].copy_from_slice(&(self.live_count as u64).to_le_bytes());
        self.mmap[24..32].copy_from_slice(&(self.capacity as u64).to_le_bytes());
        Ok(())
    }

    fn grow(&mut self, new_capacity: u32) -> Result<()> {
        if new_capacity <= self.capacity {
            return Ok(());
        }
        let new_size = HEADER_SIZE + (new_capacity as usize) * self.dim;
        self.file.set_len(new_size as u64)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        self.capacity = new_capacity;
        Ok(())
    }
}
