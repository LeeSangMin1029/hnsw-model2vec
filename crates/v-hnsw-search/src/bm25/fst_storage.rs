//! FST-based BM25 storage for compact, read-only term dictionary.
//!
//! Saves term→ordinal mapping as FST, posting lists as bincode Vec.
//! FST compresses shared prefixes (especially effective for Korean morphemes).

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use v_hnsw_core::{PointId, VhnswError};

use super::index::PostingList;
use super::scorer::Bm25Params;
use crate::Tokenizer;

const FST_FILE: &str = "bm25_terms.fst";
const DATA_FILE: &str = "bm25_data.bin";

/// Metadata stored alongside FST (everything except the term dictionary).
#[derive(bincode::Encode, bincode::Decode)]
struct FstBm25Data<T> {
    tokenizer: T,
    postings: Vec<PostingList>,
    doc_lengths: Vec<(PointId, u32)>,
    total_length: u64,
    total_docs: usize,
    params: Bm25Params,
}

/// Check if FST index files exist in the given directory.
pub fn fst_exists(dir: &Path) -> bool {
    dir.join(FST_FILE).exists() && dir.join(DATA_FILE).exists()
}

/// Save BM25 index in FST format.
///
/// Writes two files to `dir`:
/// - `bm25_terms.fst` — FST map (term bytes → ordinal)
/// - `bm25_data.bin` — bincode (posting lists + metadata)
pub fn save_fst<T: Tokenizer>(
    dir: &Path,
    tokenizer: &T,
    postings: &HashMap<String, PostingList>,
    doc_lengths: &HashMap<PointId, u32>,
    total_length: u64,
    total_docs: usize,
    params: Bm25Params,
) -> Result<(), VhnswError> {
    // Sort terms alphabetically (required by FST builder)
    let mut sorted: Vec<(&String, &PostingList)> = postings.iter().collect();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    // Build FST map: term → ordinal
    let mut builder = fst::MapBuilder::memory();
    let mut posting_vec = Vec::with_capacity(sorted.len());

    for (i, (term, posting_list)) in sorted.iter().enumerate() {
        builder.insert(term.as_bytes(), i as u64).map_err(|e| {
            VhnswError::Storage(std::io::Error::other(format!("FST insert failed: {e}")))
        })?;
        let mut pl = (*posting_list).clone();
        // Sort postings by doc_id for binary search and MaxScore
        pl.postings.sort_unstable_by_key(|p| p.doc_id);
        posting_vec.push(pl);
    }

    let fst_bytes = builder.into_inner().map_err(|e| {
        VhnswError::Storage(std::io::Error::other(format!("FST build failed: {e}")))
    })?;

    // Write FST file
    let mut f = std::fs::File::create(dir.join(FST_FILE))?;
    f.write_all(&fst_bytes)?;
    f.sync_all()?;

    // Write data file
    let data = FstBm25Data {
        tokenizer: tokenizer.clone(),
        postings: posting_vec,
        doc_lengths: doc_lengths.iter().map(|(&k, &v)| (k, v)).collect(),
        total_length,
        total_docs,
        params,
    };

    let bytes = bincode::encode_to_vec(&data, bincode::config::standard()).map_err(|e| {
        VhnswError::Storage(std::io::Error::other(format!("FST data serialize: {e}")))
    })?;

    let mut f = std::fs::File::create(dir.join(DATA_FILE))?;
    f.write_all(&bytes)?;
    f.sync_all()?;

    let fst_size = std::fs::metadata(dir.join(FST_FILE))?.len();
    let data_size = std::fs::metadata(dir.join(DATA_FILE))?.len();
    eprintln!(
        "  FST saved: terms {:.2}MB + data {:.2}MB",
        fst_size as f64 / 1_000_000.0,
        data_size as f64 / 1_000_000.0,
    );

    Ok(())
}

/// Loaded FST storage: term dictionary + posting lists.
pub struct FstStorage<T: Tokenizer> {
    pub tokenizer: T,
    pub fst_map: fst::Map<Vec<u8>>,
    pub postings: Vec<PostingList>,
    pub doc_lengths: HashMap<PointId, u32>,
    pub total_length: u64,
    pub total_docs: usize,
    pub params: Bm25Params,
    /// Maximum document ID (computed at load time for Vec accumulator sizing).
    pub max_doc_id: u64,
}

/// Load BM25 index from FST format.
pub fn load_fst<T: Tokenizer>(dir: &Path) -> Result<FstStorage<T>, VhnswError> {
    use std::io::Read;

    // Read FST map
    let fst_bytes = std::fs::read(dir.join(FST_FILE))?;
    let fst_map = fst::Map::new(fst_bytes).map_err(|e| {
        VhnswError::Storage(std::io::Error::other(format!("FST load failed: {e}")))
    })?;

    // Read data
    let mut file = std::fs::File::open(dir.join(DATA_FILE))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let (data, _): (FstBm25Data<T>, usize) =
        bincode::decode_from_slice(&bytes, bincode::config::standard()).map_err(|e| {
            VhnswError::Storage(std::io::Error::other(format!("FST data deserialize: {e}")))
        })?;

    let doc_lengths: HashMap<PointId, u32> = data.doc_lengths.into_iter().collect();
    let max_doc_id = doc_lengths.keys().max().copied().unwrap_or(0);

    Ok(FstStorage {
        tokenizer: data.tokenizer,
        fst_map,
        postings: data.postings,
        doc_lengths,
        total_length: data.total_length,
        total_docs: data.total_docs,
        params: data.params,
        max_doc_id,
    })
}
