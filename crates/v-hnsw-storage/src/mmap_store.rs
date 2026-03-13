//! Memory-mapped vector storage with O(1) access and slot-based allocation.

#![allow(unsafe_code)]

use memmap2::MmapMut;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use v_hnsw_core::{Dim, PointId, Result, VhnswError, VectorStore};

/// File magic number for validation: "VHNSWVEC"
pub const MAGIC: u64 = 0x5648_4E53_5756_4543;
/// File format version
pub const FORMAT_VERSION: u32 = 1;
/// Size of the file header in bytes
pub const HEADER_SIZE: usize = 64;

/// Create a new mmap file with the given total size.
///
/// Opens (or creates) the file at `path`, truncates it to `file_size`,
/// and returns the writable memory-map together with the file handle.
pub(crate) fn create_mmap_file(path: &Path, file_size: usize) -> Result<(File, MmapMut)> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.set_len(file_size as u64)?;
    let mmap = unsafe { MmapMut::map_mut(&file)? };
    Ok((file, mmap))
}

/// Memory-mapped vector storage backend.
///
/// # File Layout
///
/// ```text
/// [Header: 64 bytes]
/// [Slot 0: dim*4 bytes]
/// [Slot 1: dim*4 bytes]
/// ...
/// ```
///
/// # Header Format (little-endian)
///
/// - Bytes 0-7:   magic (u64)
/// - Bytes 8-11:  version (u32)
/// - Bytes 12-15: dim (u32)
/// - Bytes 16-23: live count (u64)
/// - Bytes 24-31: capacity (u64)
/// - Bytes 32-35: slot_size (u32)
/// - Bytes 36-63: reserved zeros
///
/// # Design
///
/// - Fixed-size records: each vector slot = `dim * sizeof(f32)` bytes
/// - O(1) access via slot index: `offset = HEADER_SIZE + slot * slot_size`
/// - HashMap maps point IDs to slot numbers
/// - Free list for slot reuse after removals
pub struct MmapVectorStore {
    #[allow(dead_code)]
    path: PathBuf,
    file: File,
    mmap: MmapMut,
    dim: Dim,
    slot_size: usize,
    capacity: u32,
    id_to_slot: HashMap<PointId, u32>,
    free_slots: Vec<u32>,
    live_count: usize,
    next_slot: u32,
}

impl MmapVectorStore {
    /// Create a new store file with given dimension and initial capacity.
    ///
    /// # Errors
    ///
    /// Returns `VhnswError::Storage` if file creation or mapping fails.
    pub fn create(path: impl AsRef<Path>, dim: Dim, capacity: u32) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let slot_size = dim * std::mem::size_of::<f32>();
        let file_size = HEADER_SIZE + (capacity as usize) * slot_size;

        let (file, mmap) = create_mmap_file(&path, file_size)?;

        // Write header
        let mut store = Self {
            path,
            file,
            mmap,
            dim,
            slot_size,
            capacity,
            id_to_slot: HashMap::new(),
            free_slots: Vec::new(),
            live_count: 0,
            next_slot: 0,
        };

        store.write_header()?;
        store.flush()?;

        Ok(store)
    }

    /// Open an existing store file.
    ///
    /// # Errors
    ///
    /// Returns `VhnswError::Storage` if the file cannot be opened,
    /// is invalid, or has a version mismatch.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        let (dim, capacity, live_count_stored, slot_size_stored) = Self::read_header(&mmap)?;
        let slot_size = dim * std::mem::size_of::<f32>();

        // Validate slot_size matches computed value
        if slot_size_stored as usize != slot_size {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "slot_size mismatch: stored={}, computed={}",
                    slot_size_stored, slot_size
                ),
            )));
        }

        Ok(Self {
            path,
            file,
            mmap,
            dim,
            slot_size,
            capacity,
            id_to_slot: HashMap::new(),
            free_slots: Vec::new(),
            live_count: live_count_stored as usize,
            next_slot: 0,
        })
    }

    /// Get a vector slice from mmap by slot number.
    ///
    /// # Safety
    ///
    /// This method uses `from_raw_parts` internally. It is safe because:
    /// - Slot bounds are checked
    /// - Slot size is always a multiple of 4 (f32 alignment)
    /// - Mmap guarantees memory validity for the mapped region
    fn slot_slice(&self, slot: u32) -> Result<&[f32]> {
        let offset = HEADER_SIZE + (slot as usize) * self.slot_size;
        let end = offset + self.slot_size;

        if end > self.mmap.len() {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("slot {} out of mmap bounds", slot),
            )));
        }

        let bytes = &self.mmap[offset..end];
        // SAFETY: bytes are aligned to f32 (slot_size is always multiple of 4)
        // and mmap guarantees the memory is valid for the mapped region
        let ptr = bytes.as_ptr().cast::<f32>();
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.dim) })
    }

    /// Write a vector to a slot in mmap.
    fn write_slot(&mut self, slot: u32, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        let offset = HEADER_SIZE + (slot as usize) * self.slot_size;
        let end = offset + self.slot_size;

        if end > self.mmap.len() {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("slot {} out of mmap bounds", slot),
            )));
        }

        let bytes = &mut self.mmap[offset..end];
        let ptr = bytes.as_mut_ptr().cast::<f32>();
        let dest = unsafe { std::slice::from_raw_parts_mut(ptr, self.dim) };
        dest.copy_from_slice(vector);

        Ok(())
    }

    /// Allocate a slot (reuse from free list or use next_slot).
    ///
    /// # Errors
    ///
    /// Returns `VhnswError::IndexFull` if capacity is exhausted.
    fn allocate_slot(&mut self) -> Result<u32> {
        if let Some(slot) = self.free_slots.pop() {
            Ok(slot)
        } else if self.next_slot < self.capacity {
            let slot = self.next_slot;
            self.next_slot += 1;
            Ok(slot)
        } else {
            Err(VhnswError::IndexFull {
                capacity: self.capacity as usize,
            })
        }
    }

    /// Read and validate the file header.
    ///
    /// Returns `(dim, capacity, live_count, slot_size)`.
    fn read_header(mmap: &MmapMut) -> Result<(Dim, u32, u64, u32)> {
        if mmap.len() < HEADER_SIZE {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file too small for header",
            )));
        }

        // Helper to convert slice to array with error handling
        let to_array = |slice: &[u8], name: &str| -> Result<[u8; 8]> {
            slice.try_into().map_err(|_| {
                VhnswError::Storage(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid {} field in header", name),
                ))
            })
        };

        let magic = u64::from_le_bytes(
            mmap[0..8]
                .try_into()
                .map_err(|_| VhnswError::Storage(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid magic field",
                )))?,
        );
        if magic != MAGIC {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid magic: expected {:#x}, got {:#x}", MAGIC, magic),
            )));
        }

        let version = u32::from_le_bytes(
            mmap[8..12]
                .try_into()
                .map_err(|_| VhnswError::Storage(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid version field",
                )))?,
        );
        if version != FORMAT_VERSION {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported version: expected {}, got {}",
                    FORMAT_VERSION, version
                ),
            )));
        }

        let dim = u32::from_le_bytes(
            mmap[12..16]
                .try_into()
                .map_err(|_| VhnswError::Storage(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid dim field",
                )))?,
        ) as Dim;

        let live_count = u64::from_le_bytes(to_array(&mmap[16..24], "live_count")?);
        let capacity = u64::from_le_bytes(to_array(&mmap[24..32], "capacity")?) as u32;
        let slot_size = u32::from_le_bytes(
            mmap[32..36]
                .try_into()
                .map_err(|_| VhnswError::Storage(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid slot_size field",
                )))?,
        );

        Ok((dim, capacity, live_count, slot_size))
    }

    /// Write the file header to mmap.
    fn write_header(&mut self) -> Result<()> {
        if self.mmap.len() < HEADER_SIZE {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "mmap too small for header",
            )));
        }

        self.mmap[0..8].copy_from_slice(&MAGIC.to_le_bytes());
        self.mmap[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        self.mmap[12..16].copy_from_slice(&(self.dim as u32).to_le_bytes());
        self.mmap[16..24].copy_from_slice(&(self.live_count as u64).to_le_bytes());
        self.mmap[24..32].copy_from_slice(&(self.capacity as u64).to_le_bytes());
        self.mmap[32..36].copy_from_slice(&(self.slot_size as u32).to_le_bytes());
        // Bytes 36-63 remain zero (reserved)

        Ok(())
    }

    /// Flush mmap to disk.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Get the id_to_slot mapping (for persistence by engine).
    pub fn id_map(&self) -> &HashMap<PointId, u32> {
        &self.id_to_slot
    }

    /// Restore id_to_slot mapping from persisted data.
    pub fn restore_id_map(&mut self, map: HashMap<PointId, u32>) {
        // Update next_slot to be one past the max slot in use
        let max_slot = map.values().copied().max().unwrap_or(0);
        self.next_slot = max_slot.saturating_add(1);

        // Rebuild free_slots: O(n+m) using HashSet instead of O(n*m)
        let used_slots: std::collections::HashSet<u32> = map.values().copied().collect();
        self.free_slots.clear();
        for slot in 0..self.next_slot {
            if !used_slots.contains(&slot) {
                self.free_slots.push(slot);
            }
        }

        tracing::debug!(
            used = map.len(),
            free = self.free_slots.len(),
            "restored id_map"
        );

        self.live_count = map.len();
        self.id_to_slot = map;
    }

    /// Capacity of the store.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Insert multiple vectors in a batch, writing the header only once at the end.
    ///
    /// Auto-grows capacity if needed before inserting.
    pub fn insert_batch(&mut self, batch: &[(PointId, &[f32])]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        // Count how many genuinely new points need fresh slots
        let new_count = batch
            .iter()
            .filter(|(id, _)| !self.id_to_slot.contains_key(id))
            .count() as u32;

        // Pre-grow if needed
        let available = self.capacity.saturating_sub(self.next_slot) + self.free_slots.len() as u32;
        if new_count > available {
            let needed = (self.next_slot + new_count).saturating_sub(self.free_slots.len() as u32);
            let new_cap = needed.max(self.capacity.saturating_mul(2));
            self.grow(new_cap)?;
        }

        // Insert each vector
        for &(id, vector) in batch {
            if vector.len() != self.dim {
                return Err(VhnswError::DimensionMismatch {
                    expected: self.dim,
                    got: vector.len(),
                });
            }
            if let Some(&existing_slot) = self.id_to_slot.get(&id) {
                self.write_slot(existing_slot, vector)?;
            } else {
                let slot = self.allocate_slot()?;
                self.write_slot(slot, vector)?;
                self.id_to_slot.insert(id, slot);
                self.live_count += 1;
            }
        }

        // Write header once for the whole batch
        self.write_header()?;
        Ok(())
    }

    /// Grow capacity by remapping with a larger file.
    ///
    /// # Errors
    ///
    /// Returns `VhnswError::Storage` if the new capacity is smaller than current
    /// or if file operations fail.
    pub fn grow(&mut self, new_capacity: u32) -> Result<()> {
        if new_capacity <= self.capacity {
            return Err(VhnswError::Storage(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "new capacity must be greater than current capacity",
            )));
        }

        tracing::warn!(
            old_capacity = self.capacity,
            new_capacity,
            "growing mmap store"
        );

        let new_file_size = HEADER_SIZE + (new_capacity as usize) * self.slot_size;

        // Flush before dropping old mmap
        self.mmap.flush()?;

        // Replace with a tiny anonymous mmap so the old file-backed mmap is dropped
        let empty = MmapMut::map_anon(1).map_err(VhnswError::Storage)?;
        drop(std::mem::replace(&mut self.mmap, empty));

        // Resize the file (no mmap holds the file now)
        self.file.set_len(new_file_size as u64)?;

        // Create new mmap over the resized file
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        self.capacity = new_capacity;

        // Update header with new capacity
        self.write_header()?;
        self.flush()?;

        Ok(())
    }
}

impl VectorStore for MmapVectorStore {
    fn get(&self, id: PointId) -> Result<&[f32]> {
        let slot = self
            .id_to_slot
            .get(&id)
            .ok_or(VhnswError::PointNotFound(id))?;
        self.slot_slice(*slot)
    }

    fn insert(&mut self, id: PointId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VhnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        // If updating existing point, reuse its slot
        if let Some(&existing_slot) = self.id_to_slot.get(&id) {
            self.write_slot(existing_slot, vector)?;
        } else {
            // Allocate new slot, auto-growing if capacity is exhausted
            let slot = match self.allocate_slot() {
                Ok(s) => s,
                Err(VhnswError::IndexFull { .. }) => {
                    let new_cap = self.capacity.saturating_mul(2).max(self.capacity + 1);
                    self.grow(new_cap)?;
                    self.allocate_slot()?
                }
                Err(e) => return Err(e),
            };
            self.write_slot(slot, vector)?;
            self.id_to_slot.insert(id, slot);
            self.live_count += 1;

            // Update header to reflect new live_count
            self.write_header()?;
        }

        Ok(())
    }

    fn remove(&mut self, id: PointId) -> Result<()> {
        let slot = self
            .id_to_slot
            .remove(&id)
            .ok_or(VhnswError::PointNotFound(id))?;

        self.free_slots.push(slot);
        self.live_count = self.live_count.saturating_sub(1);

        // Update header to reflect new live_count
        self.write_header()?;

        Ok(())
    }

    fn dim(&self) -> Dim {
        self.dim
    }

    fn len(&self) -> usize {
        self.live_count
    }
}
