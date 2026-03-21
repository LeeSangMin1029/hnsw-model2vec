//! Shared memory ring buffer for zero-copy IPC.
//!
//! Layout (memory-mapped file):
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ Header (64 bytes)                                       │
//! │  [0..8]   write_offset: AtomicU64 (producer advances)   │
//! │  [8..16]  read_offset:  AtomicU64 (consumer advances)   │
//! │  [16..24] capacity:     u64                              │
//! │  [24..28] slot_count:   u32                              │
//! │  [28..32] _reserved                                      │
//! │  [32..64] _padding                                       │
//! ├─────────────────────────────────────────────────────────┤
//! │ Slot table (slot_count × 16 bytes)                      │
//! │  Each slot:                                              │
//! │    [0..4]  state:  AtomicU32 (Free=0, Writing=1,        │
//! │                               Ready=2, Reading=3)       │
//! │    [4..8]  length: u32 (message body length)             │
//! │    [8..16] offset: u64 (byte offset into data region)    │
//! ├─────────────────────────────────────────────────────────┤
//! │ Data region (remaining bytes)                            │
//! │  Variable-length message bodies, referenced by slots.    │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! Protocol:
//! - Producer: find Free slot → set Writing → write data → set Ready
//! - Consumer: find Ready slot → set Reading → read data → set Free

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use memmap2::MmapMut;

use crate::error::{LspError, Result};

// ── Constants ───────────────────────────────────────────────────────

const HEADER_SIZE: usize = 64;
const SLOT_SIZE: usize = 16;
const DEFAULT_SLOT_COUNT: u32 = 256;
const DEFAULT_DATA_SIZE: usize = 16 * 1024 * 1024; // 16 MB data region

/// Slot states.
const SLOT_FREE: u32 = 0;
const SLOT_WRITING: u32 = 1;
const SLOT_READY: u32 = 2;
const SLOT_READING: u32 = 3;

// ── Ring buffer ─────────────────────────────────────────────────────

/// A shared memory ring buffer backed by a memory-mapped file.
pub struct ShmRing {
    mmap: MmapMut,
    path: PathBuf,
    slot_count: u32,
    data_offset: usize, // byte offset where data region starts
    data_capacity: usize,
}

impl ShmRing {
    /// Create a new shared memory ring buffer file.
    pub fn create(path: &Path) -> Result<Self> {
        Self::create_with_params(path, DEFAULT_SLOT_COUNT, DEFAULT_DATA_SIZE)
    }

    /// Create with custom parameters.
    pub fn create_with_params(
        path: &Path,
        slot_count: u32,
        data_size: usize,
    ) -> Result<Self> {
        let slot_table_size = slot_count as usize * SLOT_SIZE;
        let total_size = HEADER_SIZE + slot_table_size + data_size;

        // Create the backing file.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(total_size as u64)?;

        #[expect(unsafe_code, reason = "mmap requires unsafe")]
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Initialize header.
        mmap[0..8].copy_from_slice(&0u64.to_le_bytes()); // write_offset
        mmap[8..16].copy_from_slice(&0u64.to_le_bytes()); // read_offset
        mmap[16..24].copy_from_slice(&(data_size as u64).to_le_bytes()); // capacity
        mmap[24..28].copy_from_slice(&slot_count.to_le_bytes()); // slot_count

        // Zero out slots (all Free).
        let slot_start = HEADER_SIZE;
        let slot_end = slot_start + slot_table_size;
        mmap[slot_start..slot_end].fill(0);

        mmap.flush()?;

        Ok(Self {
            mmap,
            path: path.to_path_buf(),
            slot_count,
            data_offset: HEADER_SIZE + slot_table_size,
            data_capacity: data_size,
        })
    }

    /// Open an existing shared memory ring buffer.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        #[expect(unsafe_code, reason = "mmap requires unsafe")]
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(LspError::Shm("file too small for header".into()));
        }

        let slot_count = u32::from_le_bytes(
            mmap[24..28].try_into().map_err(|_| LspError::Shm("bad header".into()))?
        );
        let data_capacity = u64::from_le_bytes(
            mmap[16..24].try_into().map_err(|_| LspError::Shm("bad header".into()))?
        ) as usize;

        let slot_table_size = slot_count as usize * SLOT_SIZE;
        let data_offset = HEADER_SIZE + slot_table_size;

        Ok(Self {
            mmap,
            path: path.to_path_buf(),
            slot_count,
            data_offset,
            data_capacity,
        })
    }

    /// Write a message into the ring buffer. Returns the slot index.
    pub fn write(&mut self, data: &[u8]) -> Result<u32> {
        if data.len() > self.data_capacity {
            return Err(LspError::Shm(format!(
                "message too large: {} > {}",
                data.len(),
                self.data_capacity
            )));
        }

        // Find a free slot.
        let slot_idx = self.find_free_slot()?;
        let slot_base = HEADER_SIZE + slot_idx as usize * SLOT_SIZE;

        // Claim the slot: Free → Writing.
        let state = self.slot_state(slot_base);
        match state.compare_exchange(SLOT_FREE, SLOT_WRITING, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {}
            Err(actual) => {
                return Err(LspError::Shm(format!(
                    "slot {slot_idx} race: expected Free, got {actual}"
                )));
            }
        }

        // Allocate data space using write_offset.
        let write_offset = self.write_offset();
        let current = write_offset.load(Ordering::Acquire);
        let new_offset = (current + data.len() as u64) % self.data_capacity as u64;
        write_offset.store(new_offset, Ordering::Release);

        // Write data.
        let data_start = self.data_offset + current as usize;
        let data_end = data_start + data.len();

        if data_end <= self.mmap.len() {
            self.mmap[data_start..data_end].copy_from_slice(data);
        } else {
            // Wrap around.
            let first_len = self.mmap.len() - data_start;
            self.mmap[data_start..].copy_from_slice(&data[..first_len]);
            let rest_start = self.data_offset;
            self.mmap[rest_start..rest_start + data.len() - first_len]
                .copy_from_slice(&data[first_len..]);
        }

        // Update slot metadata.
        let len_bytes = (data.len() as u32).to_le_bytes();
        self.mmap[slot_base + 4..slot_base + 8].copy_from_slice(&len_bytes);
        let offset_bytes = current.to_le_bytes();
        self.mmap[slot_base + 8..slot_base + 16].copy_from_slice(&offset_bytes);

        // Writing → Ready.
        self.slot_state(slot_base).store(SLOT_READY, Ordering::Release);

        Ok(slot_idx)
    }

    /// Try to read a message from the ring buffer. Returns `None` if no messages are ready.
    pub fn try_read(&mut self) -> Result<Option<Vec<u8>>> {
        self.try_read_filtered(None)
    }

    /// Try to read a message matching a specific direction byte (first byte of data).
    /// Use `Some(DIR_REQUEST)` or `Some(DIR_RESPONSE)` to filter.
    pub fn try_read_filtered(&mut self, direction: Option<u8>) -> Result<Option<Vec<u8>>> {
        // Scan for a Ready slot.
        for i in 0..self.slot_count {
            let slot_base = HEADER_SIZE + i as usize * SLOT_SIZE;
            let state = self.slot_state(slot_base);

            // Peek: is it Ready?
            if state.load(Ordering::Acquire) != SLOT_READY {
                continue;
            }

            // Peek at direction byte before claiming the slot.
            if let Some(expected_dir) = direction {
                let offset = u64::from_le_bytes(
                    self.mmap[slot_base + 8..slot_base + 16]
                        .try_into()
                        .map_err(|_| LspError::Shm("bad slot offset".into()))?,
                ) as usize;
                let data_start = self.data_offset + offset;
                if data_start < self.mmap.len() && self.mmap[data_start] != expected_dir {
                    continue; // Wrong direction, skip.
                }
            }

            if state
                .compare_exchange(SLOT_READY, SLOT_READING, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // Read data.
                let length = u32::from_le_bytes(
                    self.mmap[slot_base + 4..slot_base + 8]
                        .try_into()
                        .map_err(|_| LspError::Shm("bad slot length".into()))?,
                ) as usize;
                let offset = u64::from_le_bytes(
                    self.mmap[slot_base + 8..slot_base + 16]
                        .try_into()
                        .map_err(|_| LspError::Shm("bad slot offset".into()))?,
                ) as usize;

                let data_start = self.data_offset + offset;
                let mut data = vec![0u8; length];

                if data_start + length <= self.mmap.len() {
                    data.copy_from_slice(&self.mmap[data_start..data_start + length]);
                } else {
                    // Wrap around.
                    let first_len = self.mmap.len() - data_start;
                    data[..first_len].copy_from_slice(&self.mmap[data_start..]);
                    let rest_start = self.data_offset;
                    data[first_len..].copy_from_slice(
                        &self.mmap[rest_start..rest_start + length - first_len],
                    );
                }

                // Reading → Free.
                state.store(SLOT_FREE, Ordering::Release);

                return Ok(Some(data));
            }
        }
        Ok(None)
    }

    /// Path of the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    // ── Private helpers ─────────────────────────────────────────────

    fn find_free_slot(&self) -> Result<u32> {
        for i in 0..self.slot_count {
            let slot_base = HEADER_SIZE + i as usize * SLOT_SIZE;
            if self.slot_state(slot_base).load(Ordering::Acquire) == SLOT_FREE {
                return Ok(i);
            }
        }
        Err(LspError::Shm("all slots are busy".into()))
    }

    /// Get an atomic reference to the slot state field.
    ///
    /// # Safety
    /// `slot_base` must be within bounds and properly aligned (guaranteed by
    /// our layout: HEADER_SIZE=64, SLOT_SIZE=16, state is at offset 0 of each
    /// slot — all multiples of 4).
    fn slot_state(&self, slot_base: usize) -> &AtomicU32 {
        #[expect(unsafe_code, reason = "atomic view of mmap memory")]
        unsafe {
            &*(self.mmap.as_ptr().add(slot_base).cast::<AtomicU32>())
        }
    }

    fn write_offset(&self) -> &AtomicU64 {
        #[expect(unsafe_code, reason = "atomic view of mmap header")]
        unsafe {
            &*(self.mmap.as_ptr().cast::<AtomicU64>())
        }
    }
}
