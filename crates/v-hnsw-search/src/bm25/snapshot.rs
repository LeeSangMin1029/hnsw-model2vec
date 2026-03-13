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
use v_hnsw_core::{PointId, VhnswError, storage_err, read_le_u64};

use super::fieldnorm::FieldNormLut;
use super::index::PostingList;
use super::scorer::{Bm25Params, PostingView, ScoringCtx};
use crate::Tokenizer;

/// Resolved posting entries with pre-computed IDF weights.
type TermWeights<'a> = Vec<(&'a [PostingEntry], f32)>;

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
pub(crate) struct PostingEntry {
    pub doc_id: u64,
    pub tf: u32,
    _pad: u32,
}

impl PostingView for PostingEntry {
    #[inline]
    fn doc_id(&self) -> PointId { self.doc_id }
    #[inline]
    fn tf(&self) -> u32 { self.tf }
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
    posting_offsets_offset: usize,
    posting_data_offset: usize,
    params: Bm25Params,
    fieldnorm_codes: HashMap<PointId, u8>,
    fieldnorm_lut: FieldNormLut,
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

        let total_docs = h[2] as usize;
        let total_length = h[3];
        let params = Bm25Params::new(
            f32::from_bits((h[7] & 0xFFFF_FFFF) as u32),
            f32::from_bits((h[7] >> 32) as u32),
        );

        let avg_doc_len = if total_docs == 0 { 0.0 }
            else { total_length as f32 / total_docs as f32 };
        let fieldnorm_lut = FieldNormLut::build(params.b, avg_doc_len);

        // Build fieldnorm codes from the doc length table
        let dl_end = doc_lengths_offset + num_doc_entries * 16;
        let doc_entries: &[DocLengthEntry] =
            bytemuck::try_cast_slice(&mmap[doc_lengths_offset..dl_end]).unwrap_or(&[]);
        let fieldnorm_codes: HashMap<PointId, u8> = doc_entries
            .iter()
            .map(|e| (e.doc_id, super::fieldnorm::encode(e.length as u32)))
            .collect();

        Ok(Self {
            mmap,
            fst_map,
            total_docs,
            total_length,
            max_doc_id: h[4],
            num_terms,
            posting_offsets_offset,
            posting_data_offset,
            params,
            fieldnorm_codes,
            fieldnorm_lut,
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
        let (ctx, terms) = match self.prepare_query(tokenizer, query) {
            Some(v) => v,
            None => return Vec::new(),
        };
        super::scorer::accumulate_and_rank(&ctx, &terms, self.max_doc_id, limit)
    }

    /// Score specific documents for hybrid search (Dense-Guided BM25).
    pub fn score_documents<T: Tokenizer>(
        &self, tokenizer: &T, query: &str, doc_ids: &[PointId],
    ) -> Vec<(PointId, f32)> {
        if self.total_docs == 0 || doc_ids.is_empty() { return Vec::new(); }
        let (ctx, terms) = match self.prepare_query(tokenizer, query) {
            Some(v) => v,
            None => return Vec::new(),
        };
        super::scorer::score_documents_common(&ctx, &terms, doc_ids)
    }

    // -- Internal helpers --

    /// Tokenize query, generate bigrams, resolve terms, and build scoring context.
    fn prepare_query<T: Tokenizer>(
        &self, tokenizer: &T, query: &str,
    ) -> Option<(ScoringCtx<'_>, TermWeights<'_>)> {
        let mut tokens = tokenizer.tokenize(query);
        let bigrams = super::bigram::generate(&tokens);
        tokens.extend(bigrams);
        let terms = self.resolve_terms(&tokens);
        if terms.is_empty() { return None; }
        Some((self.scoring_ctx(), terms))
    }

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

    /// Build a `ScoringCtx` from current snapshot state.
    fn scoring_ctx(&self) -> ScoringCtx<'_> {
        ScoringCtx {
            params: &self.params,
            avg_doc_len: self.avg_doc_length(),
            fieldnorm_lut: Some(&self.fieldnorm_lut),
            fieldnorm_codes: &self.fieldnorm_codes,
            doc_lengths: None, // LUT is always available for snapshots
        }
    }

    fn posting_entries(&self, ordinal: usize) -> Option<&[PostingEntry]> {
        if ordinal >= self.num_terms { return None; }
        let off_pos = self.posting_offsets_offset + ordinal * 8;
        let offset = read_le_u64(&self.mmap, off_pos)? as usize;
        let abs = self.posting_data_offset + offset;
        let count = read_le_u64(&self.mmap, abs)? as usize;
        if count == 0 { return Some(&[]); }
        let start = abs + 8;
        bytemuck::try_cast_slice(self.mmap.get(start..start + count * 16)?).ok()
    }
}