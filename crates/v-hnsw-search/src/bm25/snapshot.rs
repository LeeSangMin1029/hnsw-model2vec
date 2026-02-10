//! BM25 snapshot: mmap-ready flat file for zero-copy BM25 search.
//!
//! Layout (all sections 8-byte aligned):
//! - Header (64B = 8 × u64)
//! - Doc Lengths (N × 16B): sorted by doc_id for binary search
//! - Posting Offsets (T × 8B): byte offset per FST ordinal
//! - Posting Data (variable): per-term `[count: u64, entries: [PostingEntry]]`

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use v_hnsw_core::{PointId, VhnswError};

use super::index::PostingList;
use super::scorer::Bm25Params;
use crate::Tokenizer;

const MAGIC: u64 = 0x424D_3235_534E_4150; // "BM25SNAP"
const VERSION: u64 = 1;
const HEADER_SLOTS: usize = 8; // 8 × 8 = 64 bytes
const SNAP_FILE: &str = "bm25.snap";

/// Doc length entry: 16 bytes, sorted by doc_id for binary search.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DocLengthEntry {
    doc_id: u64,
    length: u64,
}

/// Posting entry: 16 bytes (doc_id + tf + padding).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PostingEntry {
    doc_id: u64,
    tf: u32,
    _pad: u32,
}

fn storage_err(msg: &str) -> VhnswError {
    VhnswError::Storage(std::io::Error::other(msg))
}

/// Write BM25 snapshot file from raw index data.
///
/// Called from `Bm25Index::save_snapshot()`.
pub(crate) fn write_bm25_snap(
    dir: &Path,
    postings: &HashMap<String, PostingList>,
    doc_lengths: &HashMap<PointId, u32>,
    total_docs: usize,
    total_length: u64,
    max_doc_id: u64,
    params: Bm25Params,
) -> Result<(), VhnswError> {
    let file = std::fs::File::create(dir.join(SNAP_FILE)).map_err(VhnswError::Storage)?;
    let mut w = BufWriter::new(file);

    // Sort doc_lengths by doc_id
    let mut doc_lens: Vec<DocLengthEntry> = doc_lengths
        .iter()
        .map(|(&id, &len)| DocLengthEntry { doc_id: id, length: len as u64 })
        .collect();
    doc_lens.sort_unstable_by_key(|e| e.doc_id);

    // Sort terms alphabetically (must match FST ordinal order)
    let mut sorted_terms: Vec<_> = postings.iter().collect();
    sorted_terms.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    // Build posting data blob + offsets
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(sorted_terms.len());

    for (_, pl) in &sorted_terms {
        offsets.push(data.len() as u64);
        // Sort postings by doc_id for binary search
        let mut sorted_pl: Vec<_> = pl.postings.iter().collect();
        sorted_pl.sort_unstable_by_key(|p| p.doc_id);

        data.extend_from_slice(&(sorted_pl.len() as u64).to_le_bytes());
        for p in sorted_pl {
            data.extend_from_slice(bytemuck::bytes_of(&PostingEntry {
                doc_id: p.doc_id,
                tf: p.tf,
                _pad: 0,
            }));
        }
    }

    // Pack k1 and b into one u64
    let k1_b = (params.k1.to_bits() as u64) | ((params.b.to_bits() as u64) << 32);

    let header: [u64; HEADER_SLOTS] = [
        MAGIC, VERSION,
        total_docs as u64, total_length, max_doc_id,
        sorted_terms.len() as u64, doc_lens.len() as u64, k1_b,
    ];

    w.write_all(bytemuck::cast_slice(&header)).map_err(VhnswError::Storage)?;
    w.write_all(bytemuck::cast_slice(&doc_lens)).map_err(VhnswError::Storage)?;
    w.write_all(bytemuck::cast_slice(&offsets)).map_err(VhnswError::Storage)?;
    w.write_all(&data).map_err(VhnswError::Storage)?;
    w.flush().map_err(VhnswError::Storage)?;
    Ok(())
}

/// BM25 index snapshot backed by memory-mapped file + FST term dictionary.
pub struct Bm25Snapshot {
    mmap: memmap2::Mmap,
    fst_map: fst::Map<Vec<u8>>,
    total_docs: usize,
    total_length: u64,
    max_doc_id: u64,
    num_terms: usize,
    num_doc_entries: usize,
    doc_lengths_offset: usize,
    posting_offsets_offset: usize,
    posting_data_offset: usize,
    params: Bm25Params,
}

impl Bm25Snapshot {
    /// Open a BM25 snapshot (reads `bm25.snap` + `bm25_terms.fst` from `dir`).
    pub fn open(dir: &Path) -> Result<Self, VhnswError> {
        let fst_bytes = std::fs::read(dir.join("bm25_terms.fst"))?;
        let fst_map = fst::Map::new(fst_bytes)
            .map_err(|e| storage_err(&format!("FST load: {e}")))?;

        let file = std::fs::File::open(dir.join(SNAP_FILE)).map_err(VhnswError::Storage)?;
        #[allow(unsafe_code)]
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(VhnswError::Storage)?;

        let header_bytes = HEADER_SLOTS * 8;
        if mmap.len() < header_bytes {
            return Err(storage_err("bm25 snapshot too small"));
        }

        let h: [u64; HEADER_SLOTS] = *bytemuck::try_from_bytes(&mmap[..header_bytes])
            .map_err(|_| storage_err("header alignment"))?;

        if h[0] != MAGIC { return Err(storage_err("invalid bm25 snapshot magic")); }
        if h[1] != VERSION { return Err(storage_err("unsupported bm25 snapshot version")); }

        let num_terms = h[5] as usize;
        let num_doc_entries = h[6] as usize;
        let doc_lengths_offset = header_bytes;
        let posting_offsets_offset = doc_lengths_offset + num_doc_entries * 16;
        let posting_data_offset = posting_offsets_offset + num_terms * 8;

        Ok(Self {
            mmap,
            fst_map,
            total_docs: h[2] as usize,
            total_length: h[3],
            max_doc_id: h[4],
            num_terms,
            num_doc_entries,
            doc_lengths_offset,
            posting_offsets_offset,
            posting_data_offset,
            params: Bm25Params::new(
                f32::from_bits((h[7] & 0xFFFF_FFFF) as u32),
                f32::from_bits((h[7] >> 32) as u32),
            ),
        })
    }

    /// Number of indexed documents.
    pub fn total_docs(&self) -> usize { self.total_docs }

    /// BM25 parameters.
    pub fn params(&self) -> &Bm25Params { &self.params }

    fn avg_doc_length(&self) -> f32 {
        if self.total_docs == 0 { 0.0 }
        else { self.total_length as f32 / self.total_docs as f32 }
    }

    /// Search for documents matching the query.
    pub fn search<T: Tokenizer>(
        &self, tokenizer: &T, query: &str, limit: usize,
    ) -> Vec<(PointId, f32)> {
        if self.total_docs == 0 { return Vec::new(); }
        let mut tokens = tokenizer.tokenize(query);
        let bigrams = super::bigram::generate(&tokens);
        tokens.extend(bigrams);
        let terms = self.resolve_terms(&tokens);
        if terms.is_empty() { return Vec::new(); }
        self.accumulate_and_rank(&terms, limit)
    }

    /// Score specific documents for hybrid search (Dense-Guided BM25).
    pub fn score_documents<T: Tokenizer>(
        &self, tokenizer: &T, query: &str, doc_ids: &[PointId],
    ) -> Vec<(PointId, f32)> {
        if self.total_docs == 0 || doc_ids.is_empty() { return Vec::new(); }
        let mut tokens = tokenizer.tokenize(query);
        let bigrams = super::bigram::generate(&tokens);
        tokens.extend(bigrams);
        let terms = self.resolve_terms(&tokens);
        if terms.is_empty() { return Vec::new(); }

        let avg = self.avg_doc_length();
        let mut results = Vec::with_capacity(doc_ids.len());
        for &doc_id in doc_ids {
            let doc_len = self.doc_length(doc_id);
            let mut score = 0.0f32;
            for &(entries, idf) in &terms {
                if let Ok(idx) = entries.binary_search_by_key(&doc_id, |e| e.doc_id) {
                    score += idf * self.params.tf_norm(entries[idx].tf, doc_len, avg);
                }
            }
            if score > 0.0 {
                results.push((doc_id, score));
            }
        }
        results
    }

    // -- Internal helpers --

    fn resolve_terms<'a>(&'a self, tokens: &[String]) -> Vec<(&'a [PostingEntry], f32)> {
        tokens
            .iter()
            .filter_map(|term| {
                let ordinal = self.fst_map.get(term.as_bytes())? as usize;
                let entries = self.posting_entries(ordinal)?;
                let idf = self.params.idf(entries.len() as u32, self.total_docs);
                Some((entries, idf))
            })
            .collect()
    }

    /// Vec accumulator ceiling: 256K entries = 1MB max allocation.
    const MAX_VEC_ACCUMULATOR_ID: u64 = 256_000;

    fn accumulate_and_rank(
        &self, terms: &[(&[PostingEntry], f32)], limit: usize,
    ) -> Vec<(PointId, f32)> {
        let avg = self.avg_doc_length();

        if self.max_doc_id <= Self::MAX_VEC_ACCUMULATOR_ID {
            let len = self.max_doc_id as usize + 1;
            let mut scores = vec![0.0f32; len];
            let mut touched: Vec<PointId> = Vec::with_capacity(256);

            for &(entries, idf) in terms {
                for e in entries {
                    let id = e.doc_id as usize;
                    if id < len {
                        if scores[id] == 0.0 {
                            touched.push(e.doc_id);
                        }
                        let doc_len = self.doc_length(e.doc_id);
                        scores[id] += idf * self.params.tf_norm(e.tf, doc_len, avg);
                    }
                }
            }

            let mut results: Vec<(PointId, f32)> = touched
                .into_iter()
                .map(|id| (id, scores[id as usize]))
                .collect();
            results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            results.truncate(limit);
            results
        } else {
            let mut score_map: HashMap<PointId, f32> = HashMap::new();
            for &(entries, idf) in terms {
                for e in entries {
                    let doc_len = self.doc_length(e.doc_id);
                    *score_map.entry(e.doc_id).or_insert(0.0) +=
                        idf * self.params.tf_norm(e.tf, doc_len, avg);
                }
            }
            let mut results: Vec<_> = score_map.into_iter().collect();
            results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            results.truncate(limit);
            results
        }
    }

    fn doc_length(&self, doc_id: PointId) -> u32 {
        let entries = self.doc_length_table();
        entries
            .binary_search_by_key(&doc_id, |e| e.doc_id)
            .ok()
            .map(|i| entries[i].length as u32)
            .unwrap_or(0)
    }

    fn doc_length_table(&self) -> &[DocLengthEntry] {
        let end = self.doc_lengths_offset + self.num_doc_entries * 16;
        bytemuck::try_cast_slice(&self.mmap[self.doc_lengths_offset..end]).unwrap_or(&[])
    }

    fn posting_entries(&self, ordinal: usize) -> Option<&[PostingEntry]> {
        if ordinal >= self.num_terms { return None; }
        let off_pos = self.posting_offsets_offset + ordinal * 8;
        let offset = read_u64(&self.mmap, off_pos)? as usize;
        let abs = self.posting_data_offset + offset;
        let count = read_u64(&self.mmap, abs)? as usize;
        if count == 0 { return Some(&[]); }
        let start = abs + 8;
        bytemuck::try_cast_slice(self.mmap.get(start..start + count * 16)?).ok()
    }
}

fn read_u64(data: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(data.get(offset..offset + 8)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bm25Index, WhitespaceTokenizer};

    fn make_temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_bm25_snapshot_roundtrip() {
        let dir = make_temp_dir("bm25_snapshot_test");

        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "the quick brown fox");
        index.add_document(2, "the lazy dog");
        index.add_document(3, "quick quick fox fox");

        // Save (creates bm25.bin + FST files)
        index.save(dir.join("bm25.bin")).expect("save failed");
        // Save snapshot
        index.save_snapshot(&dir).expect("snapshot save failed");

        // Open snapshot
        let snap = Bm25Snapshot::open(&dir).expect("snapshot open failed");
        assert_eq!(snap.total_docs(), 3);

        // Compare search results
        let tok = WhitespaceTokenizer;
        let snap_results = snap.search(&tok, "quick fox", 10);
        let index_results = index.search("quick fox", 10);

        assert_eq!(snap_results.len(), index_results.len());
        assert_eq!(snap_results[0].0, index_results[0].0); // same top doc
        // Scores should be close (both use same BM25 formula)
        for (s, i) in snap_results.iter().zip(index_results.iter()) {
            assert_eq!(s.0, i.0);
            assert!((s.1 - i.1).abs() < 0.01, "score mismatch: {} vs {}", s.1, i.1);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bm25_snapshot_score_documents() {
        let dir = make_temp_dir("bm25_snapshot_score_test");

        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "rust programming language");
        index.add_document(2, "python programming language");
        index.add_document(3, "rust systems programming");

        index.save(dir.join("bm25.bin")).expect("save");
        index.save_snapshot(&dir).expect("snapshot save");

        let snap = Bm25Snapshot::open(&dir).expect("open");
        let tok = WhitespaceTokenizer;

        // Score only docs 1 and 3
        let scores = snap.score_documents(&tok, "rust programming", &[1, 3]);
        assert_eq!(scores.len(), 2);
        // Both docs should have scores > 0
        for (_, score) in &scores {
            assert!(*score > 0.0);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
