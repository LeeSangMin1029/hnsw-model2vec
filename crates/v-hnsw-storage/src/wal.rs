//! Write-Ahead Log (WAL) for crash-safe persistence.

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use v_hnsw_core::{Payload, PointId, Result, VhnswError};

/// A record in the WAL.
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum WalRecord {
    /// Insert a new point.
    Insert {
        id: PointId,
        vector: Vec<f32>,
        payload: Payload,
        text: String,
    },
    /// Remove a point.
    Remove { id: PointId },
    /// Checkpoint marker.
    Checkpoint { seq: u64, point_count: usize },
    /// Begin a batch transaction.
    BatchBegin { batch_id: u64 },
    /// End a batch transaction.
    BatchEnd { batch_id: u64 },
}

/// Write-Ahead Log for crash recovery and durability.
///
/// Records are written in a simple format:
/// ```text
/// [crc32: u32][length: u32][bincode_data: length bytes]
/// ```
///
/// The WAL is split into segments (files) to allow purging old data after checkpoints.
pub struct Wal {
    dir: PathBuf,
    writer: BufWriter<File>,
    segment_number: u64,
    records_since_checkpoint: usize,
}

impl Wal {
    /// Create a new WAL directory and initialize the first segment.
    pub fn create(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let segment_number = 0;
        let segment_path = Self::segment_path(&dir, segment_number);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&segment_path)?;

        Ok(Self {
            dir,
            writer: BufWriter::new(file),
            segment_number,
            records_since_checkpoint: 0,
        })
    }

    /// Open an existing WAL directory and resume from the latest segment.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();

        if !dir.exists() {
            return Err(VhnswError::Wal(format!(
                "WAL directory does not exist: {}",
                dir.display()
            )));
        }

        // Find the latest segment number
        let segment_number = Self::find_latest_segment(&dir)?;

        let segment_path = Self::segment_path(&dir, segment_number);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&segment_path)?;

        Ok(Self {
            dir,
            writer: BufWriter::new(file),
            segment_number,
            records_since_checkpoint: 0,
        })
    }

    /// Append a record to the WAL.
    ///
    /// Format: `[crc32:4][length:4][bincode_data:length]`
    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        let config = bincode::config::standard();
        let data = bincode::encode_to_vec(record, config)
            .map_err(|e| VhnswError::Wal(format!("bincode encode failed: {e}")))?;

        let crc = crc32fast::hash(&data);
        let length = data.len() as u32;

        // Write CRC32, length, and data
        self.writer
            .write_all(&crc.to_le_bytes())
            .map_err(|e| VhnswError::Wal(format!("failed to write CRC: {e}")))?;
        self.writer
            .write_all(&length.to_le_bytes())
            .map_err(|e| VhnswError::Wal(format!("failed to write length: {e}")))?;
        self.writer
            .write_all(&data)
            .map_err(|e| VhnswError::Wal(format!("failed to write data: {e}")))?;

        self.records_since_checkpoint += 1;
        Ok(())
    }

    /// Append multiple records in a single buffered write.
    ///
    /// All records are encoded into one contiguous buffer and written with a
    /// single `write_all` call, reducing per-record syscall overhead.
    pub fn append_batch(&mut self, records: &[WalRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let config = bincode::config::standard();
        let mut buf = Vec::with_capacity(records.len() * 128);

        for record in records {
            let data = bincode::encode_to_vec(record, config)
                .map_err(|e| VhnswError::Wal(format!("bincode encode failed: {e}")))?;
            let crc = crc32fast::hash(&data);
            let length = data.len() as u32;

            buf.extend_from_slice(&crc.to_le_bytes());
            buf.extend_from_slice(&length.to_le_bytes());
            buf.extend_from_slice(&data);
        }

        self.writer
            .write_all(&buf)
            .map_err(|e| VhnswError::Wal(format!("failed to write batch: {e}")))?;

        self.records_since_checkpoint += records.len();
        Ok(())
    }

    /// Replay all records since the last checkpoint.
    ///
    /// Skips corrupt records and incomplete batches.
    pub fn replay(&self) -> Result<Vec<WalRecord>> {
        let mut all_records = Vec::new();

        // Read all segments in order
        for seg in 0..=self.segment_number {
            let path = Self::segment_path(&self.dir, seg);
            if !path.exists() {
                continue;
            }

            let records = Self::read_segment(&path)?;
            all_records.extend(records);
        }

        // Find the last checkpoint and discard everything before it
        let checkpoint_index = all_records
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, r)| matches!(r, WalRecord::Checkpoint { .. }).then_some(i));

        let records = if let Some(idx) = checkpoint_index {
            all_records.into_iter().skip(idx + 1).collect()
        } else {
            all_records
        };

        // Filter out incomplete batches
        let filtered = Self::filter_incomplete_batches(records)?;

        Ok(filtered)
    }

    /// Write a checkpoint record and reset the counter.
    pub fn checkpoint(&mut self, seq: u64, point_count: usize) -> Result<()> {
        self.append(&WalRecord::Checkpoint { seq, point_count })?;
        self.writer.flush()?;
        self.records_since_checkpoint = 0;
        Ok(())
    }

    /// Remove all segment files before the current one (after a checkpoint).
    pub fn purge_old_segments(&mut self) -> Result<()> {
        for seg in 0..self.segment_number {
            let path = Self::segment_path(&self.dir, seg);
            if path.exists() {
                fs::remove_file(&path)?;
            }
        }
        Ok(())
    }

    /// Truncate the WAL by removing all segment files and starting fresh.
    pub fn truncate(&mut self) -> Result<()> {
        // Flush current writer
        self.writer.flush()?;

        // Remove all existing segments
        for seg in 0..=self.segment_number {
            let path = Self::segment_path(&self.dir, seg);
            if path.exists() {
                fs::remove_file(&path)?;
            }
        }

        // Create a new first segment
        self.segment_number = 0;
        let segment_path = Self::segment_path(&self.dir, 0);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&segment_path)?;

        self.writer = BufWriter::new(file);
        self.records_since_checkpoint = 0;

        Ok(())
    }

    /// Number of records written since the last checkpoint.
    pub fn pending_count(&self) -> usize {
        self.records_since_checkpoint
    }

    // --- Private helpers ---

    fn segment_path(dir: &Path, segment_number: u64) -> PathBuf {
        dir.join(format!("wal-{segment_number:06}.log"))
    }

    fn find_latest_segment(dir: &Path) -> Result<u64> {
        let mut max_segment = 0u64;
        let entries = fs::read_dir(dir)?;

        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if let Some(rest) = name_str.strip_prefix("wal-")
                && let Some(num_str) = rest.strip_suffix(".log")
                && let Ok(num) = num_str.parse::<u64>()
                && num > max_segment
            {
                max_segment = num;
            }
        }

        Ok(max_segment)
    }

    fn read_segment(path: &Path) -> Result<Vec<WalRecord>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut records = Vec::new();

        let config = bincode::config::standard();

        loop {
            // Try to read CRC
            let mut crc_bytes = [0u8; 4];
            match reader.read_exact(&mut crc_bytes) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let expected_crc = u32::from_le_bytes(crc_bytes);

            // Read length
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(()) => {}
                Err(_) => break, // Incomplete record
            }
            let length = u32::from_le_bytes(len_bytes) as usize;

            // Read data
            let mut data = vec![0u8; length];
            match reader.read_exact(&mut data) {
                Ok(()) => {}
                Err(_) => break, // Incomplete record
            }

            // Verify CRC
            let actual_crc = crc32fast::hash(&data);
            if actual_crc != expected_crc {
                // Corrupt record, stop reading this segment
                break;
            }

            // Deserialize
            let (record, _): (WalRecord, usize) = bincode::decode_from_slice(&data, config)
                .map_err(|e| VhnswError::Wal(format!("bincode decode failed: {e}")))?;

            records.push(record);
        }

        Ok(records)
    }

    fn filter_incomplete_batches(records: Vec<WalRecord>) -> Result<Vec<WalRecord>> {
        use std::collections::HashMap;

        // Track which batches are complete
        let mut batch_begins: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut batch_ends: HashMap<u64, Vec<usize>> = HashMap::new();

        for (i, record) in records.iter().enumerate() {
            match record {
                WalRecord::BatchBegin { batch_id } => {
                    batch_begins.entry(*batch_id).or_default().push(i);
                }
                WalRecord::BatchEnd { batch_id } => {
                    batch_ends.entry(*batch_id).or_default().push(i);
                }
                _ => {}
            }
        }

        // Determine which batch ranges are valid (have matching begin+end)
        let mut valid_ranges = Vec::new();
        for (batch_id, begins) in &batch_begins {
            if let Some(ends) = batch_ends.get(batch_id) {
                // Pair up begins and ends
                for &begin_idx in begins {
                    if let Some(&end_idx) = ends.iter().find(|&&e| e > begin_idx) {
                        valid_ranges.push((begin_idx, end_idx));
                    }
                }
            }
        }

        // Filter: keep records not in any batch, or in a valid batch range
        let mut result = Vec::new();
        for (i, record) in records.into_iter().enumerate() {
            let in_batch = batch_begins.values().any(|v| v.contains(&i))
                || batch_ends.values().any(|v| v.contains(&i))
                || {
                    // Check if this record is between a BatchBegin and BatchEnd
                    valid_ranges
                        .iter()
                        .any(|&(begin, end)| i > begin && i < end)
                        || batch_begins.values().any(|begins| {
                            begins.iter().any(|&begin| {
                                i > begin
                                    && !valid_ranges.iter().any(|&(b, _)| b == begin)
                            })
                        })
                };

            let in_valid_batch = valid_ranges.iter().any(|&(begin, end)| i >= begin && i <= end);

            // Keep if not in a batch, or in a complete batch
            if !in_batch || in_valid_batch {
                result.push(record);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_wal_create_and_append() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("wal_test_create");
        let _ = fs::remove_dir_all(&temp_dir);

        let mut wal = Wal::create(&temp_dir)?;

        let payload = Payload {
            source: "test.md".to_string(),
            tags: vec![],
            created_at: 0,
            source_modified_at: 0,
            chunk_index: 0,
            chunk_total: 1,
            custom: HashMap::new(),
        };

        wal.append(&WalRecord::Insert {
            id: 1,
            vector: vec![1.0, 2.0],
            payload: payload.clone(),
            text: "test".to_string(),
        })?;

        wal.append(&WalRecord::Remove { id: 1 })?;

        assert_eq!(wal.pending_count(), 2);

        let _ = fs::remove_dir_all(&temp_dir);
        Ok(())
    }

    #[test]
    fn test_wal_replay() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("wal_test_replay");
        let _ = fs::remove_dir_all(&temp_dir);

        let mut wal = Wal::create(&temp_dir)?;

        let payload = Payload {
            source: "test.md".to_string(),
            tags: vec![],
            created_at: 0,
            source_modified_at: 0,
            chunk_index: 0,
            chunk_total: 1,
            custom: HashMap::new(),
        };

        wal.append(&WalRecord::Insert {
            id: 1,
            vector: vec![1.0, 2.0],
            payload: payload.clone(),
            text: "test1".to_string(),
        })?;

        wal.checkpoint(1, 1)?;

        wal.append(&WalRecord::Insert {
            id: 2,
            vector: vec![3.0, 4.0],
            payload: payload.clone(),
            text: "test2".to_string(),
        })?;

        drop(wal);

        // Reopen and replay
        let wal = Wal::open(&temp_dir)?;
        let records = wal.replay()?;

        // Should only have records after checkpoint
        assert_eq!(records.len(), 1);
        matches!(&records[0], WalRecord::Insert { id: 2, .. });

        let _ = fs::remove_dir_all(&temp_dir);
        Ok(())
    }

    #[test]
    fn test_wal_incomplete_batch() -> Result<()> {
        let temp_dir = std::env::temp_dir().join("wal_test_batch");
        let _ = fs::remove_dir_all(&temp_dir);

        let mut wal = Wal::create(&temp_dir)?;

        let payload = Payload {
            source: "test.md".to_string(),
            tags: vec![],
            created_at: 0,
            source_modified_at: 0,
            chunk_index: 0,
            chunk_total: 1,
            custom: HashMap::new(),
        };

        // Complete batch
        wal.append(&WalRecord::BatchBegin { batch_id: 1 })?;
        wal.append(&WalRecord::Insert {
            id: 1,
            vector: vec![1.0],
            payload: payload.clone(),
            text: "complete".to_string(),
        })?;
        wal.append(&WalRecord::BatchEnd { batch_id: 1 })?;

        // Incomplete batch (no end)
        wal.append(&WalRecord::BatchBegin { batch_id: 2 })?;
        wal.append(&WalRecord::Insert {
            id: 2,
            vector: vec![2.0],
            payload: payload.clone(),
            text: "incomplete".to_string(),
        })?;

        drop(wal);

        // Replay should only include the complete batch
        let wal = Wal::open(&temp_dir)?;
        let records = wal.replay()?;

        // Should have: BatchBegin(1), Insert(1), BatchEnd(1)
        assert_eq!(records.len(), 3);

        let _ = fs::remove_dir_all(&temp_dir);
        Ok(())
    }
}
