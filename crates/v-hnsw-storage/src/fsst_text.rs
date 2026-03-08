//! Zstd dictionary text compression for payload text storage.
//!
//! Provides offline compression (during build-index) and transparent
//! decompression during reads. Raw `text.dat` remains the write target;
//! `text.zst` is a compressed read-only snapshot created after indexing.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use v_hnsw_core::{PointId, Result, VhnswError};

/// Compressed text reader using zstd with a shared dictionary.
///
/// Loaded from disk during `StorageEngine::open()`. Individual documents
/// are decompressed on demand using the pre-trained dictionary.
pub(crate) struct CompressedTextReader {
    file: File,
    /// PointId -> (offset, compressed_len, original_len)
    index: HashMap<PointId, (u64, u32, u32)>,
    dict: zstd::dict::DecoderDictionary<'static>,
}

impl CompressedTextReader {
    /// Load compressed text store from a database directory.
    ///
    /// Returns `None` if any required file is missing.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let data_path = dir.join("text.zst");
        let dict_path = dir.join("text_dict.zst");
        let idx_path = dir.join("text_zst.idx");

        if !data_path.exists() || !dict_path.exists() || !idx_path.exists() {
            return Ok(None);
        }

        let dict_data = std::fs::read(&dict_path)?;
        let dict = zstd::dict::DecoderDictionary::copy(&dict_data);

        let idx_data = std::fs::read(&idx_path)?;
        let config = bincode::config::standard();
        let (index, _): (HashMap<PointId, (u64, u32, u32)>, usize) =
            bincode::decode_from_slice(&idx_data, config)
                .map_err(|e| VhnswError::Payload(format!("failed to decode zstd index: {e}")))?;

        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(&data_path)?;

        Ok(Some(Self { file, index, dict }))
    }

    /// Read and decompress text for a given point ID.
    pub fn get_text(&self, id: PointId) -> Result<Option<String>> {
        let Some(&(offset, compressed_len, original_len)) = self.index.get(&id) else {
            return Ok(None);
        };

        let mut handle = self.file.try_clone()?;
        handle.seek(SeekFrom::Start(offset))?;

        let mut compressed = vec![0u8; compressed_len as usize];
        handle.read_exact(&mut compressed)?;

        let mut decompressor = zstd::bulk::Decompressor::with_prepared_dictionary(&self.dict)
            .map_err(|e| VhnswError::Payload(format!("zstd decompressor init failed: {e}")))?;
        let decompressed = decompressor
            .decompress(&compressed, original_len as usize)
            .map_err(|e| VhnswError::Payload(format!("zstd decompress failed: {e}")))?;

        String::from_utf8(decompressed)
            .map(Some)
            .map_err(|e| VhnswError::Payload(format!("invalid UTF-8 in compressed text: {e}")))
    }
}

/// Compress raw text bytes using zstd with a trained dictionary.
///
/// Creates three files in `dir`:
/// - `text.zst` — compressed text data
/// - `text_dict.zst` — zstd dictionary (~112KB)
/// - `text_zst.idx` — PointId -> (offset, compressed_len, original_len)
pub fn compress_texts(texts: &[(PointId, Vec<u8>)], dir: &Path) -> Result<()> {
    if texts.is_empty() {
        return Ok(());
    }

    // Skip compression for very small datasets
    let total_bytes: usize = texts.iter().map(|(_, d)| d.len()).sum();
    if total_bytes < 4096 {
        return Ok(());
    }

    // Train dictionary from corpus (cap dict size to data size)
    let samples: Vec<&[u8]> = texts.iter().map(|(_, data)| data.as_slice()).collect();
    let max_dict_size = (512 * 1024).min(total_bytes);
    let dict_data = zstd::dict::from_samples(&samples, max_dict_size)
        .map_err(|e| VhnswError::Payload(format!("zstd dict training failed: {e}")))?;

    // Create compressor with dictionary (level 15: ~2x better ratio, same decompression speed)
    let enc_dict = zstd::dict::EncoderDictionary::copy(&dict_data, 15);
    let mut compressor = zstd::bulk::Compressor::with_prepared_dictionary(&enc_dict)
        .map_err(|e| VhnswError::Payload(format!("zstd compressor init failed: {e}")))?;

    // Compress and write
    let data_path = dir.join("text.zst");
    let mut data_file = File::create(&data_path)?;
    let mut index: HashMap<PointId, (u64, u32, u32)> = HashMap::with_capacity(texts.len());
    let mut raw_total: u64 = 0;

    for (id, data) in texts {
        raw_total += data.len() as u64;
        let compressed = compressor
            .compress(data)
            .map_err(|e| VhnswError::Payload(format!("zstd compress failed: {e}")))?;
        let offset = data_file.stream_position()?;
        data_file.write_all(&compressed)?;
        index.insert(*id, (offset, compressed.len() as u32, data.len() as u32));
    }
    data_file.flush()?;

    // Save dictionary
    std::fs::write(dir.join("text_dict.zst"), &dict_data)?;

    // Save index
    let config = bincode::config::standard();
    let idx_data = bincode::encode_to_vec(&index, config)
        .map_err(|e| VhnswError::Payload(format!("failed to encode zstd index: {e}")))?;
    std::fs::write(dir.join("text_zst.idx"), idx_data)?;

    // Clean up old FSST files if present
    let _ = std::fs::remove_file(dir.join("text.fsst"));
    let _ = std::fs::remove_file(dir.join("text_symbols.fsst"));
    let _ = std::fs::remove_file(dir.join("text_fsst.idx"));

    let compressed_size = std::fs::metadata(&data_path)?.len();
    let dict_size = dict_data.len();
    eprintln!(
        "  Zstd: {:.2}MB -> {:.2}MB ({:.1}x), dict {:.0}KB",
        raw_total as f64 / 1_000_000.0,
        compressed_size as f64 / 1_000_000.0,
        raw_total as f64 / compressed_size.max(1) as f64,
        dict_size as f64 / 1024.0,
    );

    Ok(())
}
