//! BM25 inverted index implementation.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use v_hnsw_core::{PointId, VhnswError};

use crate::bm25::scorer::{Bm25Params, PostingView, ScoringCtx};
use crate::Tokenizer;

impl PostingView for Posting {
    #[inline]
    fn doc_id(&self) -> PointId { self.doc_id }
    #[inline]
    fn tf(&self) -> u32 { self.tf }
}

/// A posting entry containing document ID and term frequency.
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct Posting {
    /// Document/point identifier.
    pub doc_id: PointId,
    /// Term frequency in this document.
    pub tf: u32,
}

/// Posting list for a single term.
#[derive(Debug, Clone, Default, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct PostingList {
    /// Documents containing this term.
    pub postings: Vec<Posting>,
}

impl PostingList {
    /// Create a new empty posting list.
    pub fn new() -> Self {
        Self {
            postings: Vec::new(),
        }
    }

    /// Add a document to the posting list.
    ///
    /// Caller must ensure no duplicate `doc_id` exists (e.g. by calling
    /// `remove_document` first). `Bm25Index::add_document` already does this.
    pub fn add(&mut self, doc_id: PointId, tf: u32) {
        self.postings.push(Posting { doc_id, tf });
    }

    /// Remove a document from the posting list.
    pub fn remove(&mut self, doc_id: PointId) -> bool {
        if let Some(pos) = self.postings.iter().position(|p| p.doc_id == doc_id) {
            self.postings.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Number of documents in this posting list (document frequency).
    pub fn df(&self) -> u32 {
        self.postings.len() as u32
    }
}

/// Serializable representation of BM25Index using Vec instead of HashMap.
/// bincode 2.x has issues with HashMap serialization, so we convert to Vec for storage.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
struct Bm25IndexData<T> {
    tokenizer: T,
    /// Postings stored as Vec<(term, posting_list)> for bincode compatibility.
    postings: Vec<(String, PostingList)>,
    /// Doc lengths stored as Vec<(doc_id, length)> for bincode compatibility.
    doc_lengths: Vec<(PointId, u32)>,
    total_length: u64,
    total_docs: usize,
    params: Bm25Params,
}

/// Internal term storage: mutable HashMap for building, or compact FST for search.
enum TermStorage {
    /// Mutable HashMap storage (used during index building).
    HashMap(HashMap<String, PostingList>),
    /// Compact FST storage (used for search after loading).
    Fst {
        map: fst::Map<Vec<u8>>,
        postings: Vec<PostingList>,
    },
}

impl std::fmt::Debug for TermStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HashMap(m) => write!(f, "HashMap({} terms)", m.len()),
            Self::Fst { postings, .. } => write!(f, "Fst({} terms)", postings.len()),
        }
    }
}

/// BM25 inverted index for sparse text search.
#[derive(Debug)]
pub struct Bm25Index<T: Tokenizer> {
    /// The tokenizer used for text processing.
    tokenizer: T,
    /// Term dictionary + posting lists.
    storage: TermStorage,
    /// Document lengths (number of tokens).
    doc_lengths: HashMap<PointId, u32>,
    /// Sum of all document lengths (for computing average).
    total_length: u64,
    /// Total number of documents.
    total_docs: usize,
    /// BM25 scoring parameters.
    params: Bm25Params,
    /// Cached maximum document ID (for Vec accumulator sizing).
    max_doc_id: u64,
    /// FieldNorm codes: doc_id → quantized length byte (lazy-built).
    fieldnorm_codes: HashMap<PointId, u8>,
    /// FieldNorm LUT: 256-entry cache for length normalization (lazy-built).
    fieldnorm_lut: Option<super::fieldnorm::FieldNormLut>,
}

impl<T: Tokenizer> Bm25Index<T> {
    /// Create a new BM25 index with the given tokenizer.
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            storage: TermStorage::HashMap(HashMap::new()),
            doc_lengths: HashMap::new(),
            total_length: 0,
            total_docs: 0,
            params: Bm25Params::default(),
            max_doc_id: 0,
            fieldnorm_codes: HashMap::new(),
            fieldnorm_lut: None,
        }
    }

    /// Create a new BM25 index with custom scoring parameters.
    pub fn with_params(tokenizer: T, k1: f32, b: f32) -> Self {
        Self {
            tokenizer,
            storage: TermStorage::HashMap(HashMap::new()),
            doc_lengths: HashMap::new(),
            total_length: 0,
            total_docs: 0,
            params: Bm25Params::new(k1, b),
            max_doc_id: 0,
            fieldnorm_codes: HashMap::new(),
            fieldnorm_lut: None,
        }
    }

    /// Get the BM25 parameters.
    pub fn params(&self) -> &Bm25Params {
        &self.params
    }

    /// Get the total number of indexed documents.
    pub fn len(&self) -> usize {
        self.total_docs
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }

    /// Get the average document length.
    pub fn avg_doc_length(&self) -> f32 {
        if self.total_docs == 0 {
            0.0
        } else {
            self.total_length as f32 / self.total_docs as f32
        }
    }

    /// Add a document to the index.
    ///
    /// If a document with the same ID already exists, it will be replaced.
    pub fn add_document(&mut self, doc_id: PointId, text: &str) {
        if !matches!(self.storage, TermStorage::HashMap(_)) {
            return; // FST mode is read-only
        }

        // Remove existing document if present
        if self.doc_lengths.contains_key(&doc_id) {
            self.remove_document(doc_id);
        }

        let TermStorage::HashMap(postings) = &mut self.storage else {
            return;
        };

        // Tokenize the text
        let tokens = self.tokenizer.tokenize(text);
        let doc_len = tokens.len() as u32; // unigram only (bigrams don't inflate length norm)

        // Generate bigram tokens for proximity scoring
        let bigrams = super::bigram::generate(&tokens);

        // Count term frequencies (unigrams + bigrams)
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0) += 1;
        }
        for bigram in bigrams {
            *term_freqs.entry(bigram).or_insert(0) += 1;
        }

        // Update posting lists
        for (term, tf) in term_freqs {
            postings.entry(term).or_default().add(doc_id, tf);
        }

        // Update document metadata
        self.doc_lengths.insert(doc_id, doc_len);
        self.fieldnorm_codes.insert(doc_id, super::fieldnorm::encode(doc_len));
        self.total_length += doc_len as u64;
        self.total_docs += 1;
        if doc_id > self.max_doc_id {
            self.max_doc_id = doc_id;
        }
        // Invalidate LUT (avg_doc_len changed)
        self.fieldnorm_lut = None;
    }

    /// Remove a document from the index.
    ///
    /// Returns `true` if the document was found and removed.
    pub fn remove_document(&mut self, doc_id: PointId) -> bool {
        let TermStorage::HashMap(postings) = &mut self.storage else {
            return false; // FST mode is read-only
        };

        let doc_len = match self.doc_lengths.remove(&doc_id) {
            Some(len) => len,
            None => return false,
        };
        self.fieldnorm_codes.remove(&doc_id);

        // Update statistics
        self.total_length = self.total_length.saturating_sub(doc_len as u64);
        self.total_docs = self.total_docs.saturating_sub(1);
        self.fieldnorm_lut = None;

        // Remove from all posting lists
        let mut empty_terms = Vec::new();
        for (term, posting_list) in postings.iter_mut() {
            posting_list.remove(doc_id);
            if posting_list.postings.is_empty() {
                empty_terms.push(term.clone());
            }
        }

        for term in empty_terms {
            postings.remove(&term);
        }

        true
    }

    /// Build the FieldNorm cache (codes + LUT) from current document lengths.
    ///
    /// Call after bulk document insertion and before searching. This is O(N)
    /// for codes + O(256) for the LUT. Subsequent searches use the cached values.
    pub fn build_fieldnorm_cache(&mut self) {
        let avg = self.avg_doc_length();
        self.fieldnorm_lut = Some(super::fieldnorm::FieldNormLut::build(self.params.b, avg));
        if self.fieldnorm_codes.len() != self.doc_lengths.len() {
            self.fieldnorm_codes = self
                .doc_lengths
                .iter()
                .map(|(&id, &len)| (id, super::fieldnorm::encode(len)))
                .collect();
        }
    }

    /// Search for documents matching the query.
    ///
    /// Returns documents sorted by BM25 score in descending order.
    /// Uses MaxScore pruning for 3+ term queries (FST mode),
    /// Vec accumulator (cache-friendly) otherwise.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(PointId, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }
        let mut query_tokens = self.tokenizer.tokenize(query);
        let bigrams = super::bigram::generate(&query_tokens);
        query_tokens.extend(bigrams);
        let terms = self.resolve_terms(&query_tokens);
        if terms.is_empty() {
            return Vec::new();
        }

        // MaxScore for 3+ terms on FST (sorted posting lists required)
        if terms.len() >= 3 && matches!(self.storage, TermStorage::Fst { .. }) {
            return super::maxscore::maxscore_search(
                &terms,
                limit,
                &self.params,
                &self.doc_lengths,
                self.avg_doc_length(),
                self.fieldnorm_lut.as_ref(),
                &self.fieldnorm_codes,
            );
        }

        let ctx = self.scoring_ctx();
        super::scorer::accumulate_and_rank(&ctx, &terms, self.max_doc_id, limit)
    }

    /// Get the document frequency for a term.
    pub fn document_frequency(&self, term: &str) -> u32 {
        self.get_posting_list(term).map(|pl| pl.df()).unwrap_or(0)
    }

    /// Get the posting list for a term.
    pub fn get_posting_list(&self, term: &str) -> Option<&PostingList> {
        match &self.storage {
            TermStorage::HashMap(map) => map.get(term),
            TermStorage::Fst { map, postings } => {
                map.get(term.as_bytes()).and_then(|ord| postings.get(ord as usize))
            }
        }
    }

    /// Score only the specified documents for the query (Dense-Guided shortcut).
    ///
    /// For hybrid search: HNSW provides candidate doc IDs, BM25 scores only those.
    /// O(|doc_ids| * |terms|) instead of O(N * |terms|). Thread-safe (&self).
    pub fn score_documents(&self, query: &str, doc_ids: &[PointId]) -> Vec<(PointId, f32)> {
        if self.total_docs == 0 || doc_ids.is_empty() {
            return Vec::new();
        }
        let mut query_tokens = self.tokenizer.tokenize(query);
        let bigrams = super::bigram::generate(&query_tokens);
        query_tokens.extend(bigrams);
        let terms = self.resolve_terms(&query_tokens);
        if terms.is_empty() {
            return Vec::new();
        }

        let ctx = self.scoring_ctx();
        super::scorer::score_documents_common(&ctx, &terms, doc_ids)
    }

    // -- Query resolution & scoring utilities --

    /// Resolve query tokens to posting slices with pre-computed IDF weights.
    ///
    /// Reusable by `search()`, `score_documents()`, and `MaxScore`.
    fn resolve_terms<'a>(&'a self, tokens: &[String]) -> Vec<(&'a [Posting], f32)> {
        tokens
            .iter()
            .filter_map(|term| {
                self.get_posting_list(term)
                    .map(|pl| (pl.postings.as_slice(), self.params.idf(pl.df(), self.total_docs)))
            })
            .collect()
    }

    /// Build a `ScoringCtx` from current index state.
    fn scoring_ctx(&self) -> ScoringCtx<'_> {
        ScoringCtx {
            params: &self.params,
            avg_doc_len: self.avg_doc_length(),
            fieldnorm_lut: self.fieldnorm_lut.as_ref(),
            fieldnorm_codes: &self.fieldnorm_codes,
            doc_lengths: Some(&self.doc_lengths),
        }
    }

    /// Save index to file (bincode + FST formats).
    ///
    /// Writes bincode to `path` for backward compatibility, and FST files
    /// to the parent directory for compact read-only loading.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VhnswError> {
        use std::io::Write;

        let TermStorage::HashMap(postings) = &self.storage else {
            return Err(VhnswError::Storage(std::io::Error::other(
                "cannot save FST-mode index as bincode",
            )));
        };

        // Write bincode format (backward compat)
        let data = Bm25IndexData {
            tokenizer: self.tokenizer.clone(),
            postings: postings.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            doc_lengths: self.doc_lengths.iter().map(|(k, v)| (*k, *v)).collect(),
            total_length: self.total_length,
            total_docs: self.total_docs,
            params: self.params,
        };

        let bytes = bincode::encode_to_vec(&data, bincode::config::standard()).map_err(|e| {
            VhnswError::Storage(std::io::Error::other(format!("serialize failed: {e}")))
        })?;

        let mut file = std::fs::File::create(path.as_ref())?;
        file.write_all(&bytes)?;
        file.sync_all()?;

        // Also write FST format
        if let Some(dir) = path.as_ref().parent() {
            super::fst_storage::save_fst(
                dir,
                &self.tokenizer,
                postings,
                &self.doc_lengths,
                self.total_length,
                self.total_docs,
                self.params,
            )?;
        }

        Ok(())
    }

    /// Save a BM25 snapshot file for mmap-based search.
    ///
    /// Requires HashMap mode (fresh build, not loaded from FST).
    /// The directory must already contain `bm25_terms.fst` (from `save()`).
    pub fn save_snapshot(&self, dir: &Path) -> Result<(), VhnswError> {
        let TermStorage::HashMap(postings) = &self.storage else {
            return Err(VhnswError::Storage(std::io::Error::other(
                "snapshot requires HashMap mode (not FST)",
            )));
        };
        super::snapshot::write_bm25_snap(
            dir,
            postings,
            &self.doc_lengths,
            self.total_docs,
            self.total_length,
            self.max_doc_id,
            self.params,
        )
    }

    /// Load index from file. Prefers FST format if available, falls back to bincode.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, VhnswError> {
        // Try FST format first (compact, faster load)
        if let Some(dir) = path.as_ref().parent()
            && super::fst_storage::fst_exists(dir) {
                let fst = super::fst_storage::load_fst::<T>(dir)?;
                let fieldnorm_codes: HashMap<PointId, u8> = fst.doc_lengths
                    .iter()
                    .map(|(&id, &len)| (id, super::fieldnorm::encode(len)))
                    .collect();
                let avg = if fst.total_docs == 0 { 0.0 }
                    else { fst.total_length as f32 / fst.total_docs as f32 };
                let lut = super::fieldnorm::FieldNormLut::build(fst.params.b, avg);
                return Ok(Self {
                    tokenizer: fst.tokenizer,
                    storage: TermStorage::Fst {
                        map: fst.fst_map,
                        postings: fst.postings,
                    },
                    doc_lengths: fst.doc_lengths,
                    total_length: fst.total_length,
                    total_docs: fst.total_docs,
                    params: fst.params,
                    max_doc_id: fst.max_doc_id,
                    fieldnorm_codes,
                    fieldnorm_lut: Some(lut),
                });
            }

        // Fall back to bincode (mutable HashMap mode)
        Self::load_mutable(path)
    }

    /// Load index in mutable (HashMap) mode from bincode format.
    ///
    /// Use this instead of `load()` when you need to mutate the index
    /// (e.g. `add_document` / `remove_document`, then `save`).
    /// `load()` may return a read-only FST index that silently ignores mutations.
    pub fn load_mutable(path: impl AsRef<Path>) -> Result<Self, VhnswError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path.as_ref())?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let (data, _): (Bm25IndexData<T>, usize) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).map_err(|e| {
                VhnswError::Storage(std::io::Error::other(format!("deserialize failed: {e}")))
            })?;

        let doc_lengths: HashMap<PointId, u32> = data.doc_lengths.into_iter().collect();
        let max_doc_id = doc_lengths.keys().max().copied().unwrap_or(0);

        let fieldnorm_codes: HashMap<PointId, u8> = doc_lengths
            .iter()
            .map(|(&id, &len)| (id, super::fieldnorm::encode(len)))
            .collect();
        let avg = if data.total_docs == 0 { 0.0 }
            else { data.total_length as f32 / data.total_docs as f32 };
        let lut = super::fieldnorm::FieldNormLut::build(data.params.b, avg);

        Ok(Self {
            tokenizer: data.tokenizer,
            storage: TermStorage::HashMap(data.postings.into_iter().collect()),
            doc_lengths,
            total_length: data.total_length,
            total_docs: data.total_docs,
            params: data.params,
            max_doc_id,
            fieldnorm_codes,
            fieldnorm_lut: Some(lut),
        })
    }
}

