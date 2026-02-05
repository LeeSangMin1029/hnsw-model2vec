//! BM25 inverted index implementation.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use v_hnsw_core::{PointId, VhnswError};

use crate::bm25::scorer::Bm25Params;
use crate::Tokenizer;

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
    pub fn add(&mut self, doc_id: PointId, tf: u32) {
        // Check if document already exists
        if let Some(posting) = self.postings.iter_mut().find(|p| p.doc_id == doc_id) {
            posting.tf = tf;
        } else {
            self.postings.push(Posting { doc_id, tf });
        }
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

/// BM25 inverted index for sparse text search.
#[derive(Debug)]
pub struct Bm25Index<T: Tokenizer> {
    /// The tokenizer used for text processing.
    tokenizer: T,
    /// Inverted index: term -> posting list.
    postings: HashMap<String, PostingList>,
    /// Document lengths (number of tokens).
    doc_lengths: HashMap<PointId, u32>,
    /// Sum of all document lengths (for computing average).
    total_length: u64,
    /// Total number of documents.
    total_docs: usize,
    /// BM25 scoring parameters.
    params: Bm25Params,
}

impl<T: Tokenizer> Bm25Index<T> {
    /// Create a new BM25 index with the given tokenizer.
    pub fn new(tokenizer: T) -> Self {
        Self {
            tokenizer,
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            total_docs: 0,
            params: Bm25Params::default(),
        }
    }

    /// Create a new BM25 index with custom scoring parameters.
    pub fn with_params(tokenizer: T, k1: f32, b: f32) -> Self {
        Self {
            tokenizer,
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            total_docs: 0,
            params: Bm25Params::new(k1, b),
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
        // Remove existing document if present
        if self.doc_lengths.contains_key(&doc_id) {
            self.remove_document(doc_id);
        }

        // Tokenize the text
        let tokens = self.tokenizer.tokenize(text);
        let doc_len = tokens.len() as u32;

        // Count term frequencies
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0) += 1;
        }

        // Update posting lists
        for (term, tf) in term_freqs {
            self.postings.entry(term).or_default().add(doc_id, tf);
        }

        // Update document metadata
        self.doc_lengths.insert(doc_id, doc_len);
        self.total_length += doc_len as u64;
        self.total_docs += 1;
    }

    /// Remove a document from the index.
    ///
    /// Returns `true` if the document was found and removed.
    pub fn remove_document(&mut self, doc_id: PointId) -> bool {
        let doc_len = match self.doc_lengths.remove(&doc_id) {
            Some(len) => len,
            None => return false,
        };

        // Update statistics
        self.total_length = self.total_length.saturating_sub(doc_len as u64);
        self.total_docs = self.total_docs.saturating_sub(1);

        // Remove from all posting lists
        // Keep track of empty terms to remove later
        let mut empty_terms = Vec::new();
        for (term, posting_list) in &mut self.postings {
            posting_list.remove(doc_id);
            if posting_list.postings.is_empty() {
                empty_terms.push(term.clone());
            }
        }

        // Remove empty posting lists
        for term in empty_terms {
            self.postings.remove(&term);
        }

        true
    }

    /// Search for documents matching the query.
    ///
    /// Returns documents sorted by BM25 score in descending order.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(PointId, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_tokens = self.tokenizer.tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let avg_doc_len = self.avg_doc_length();

        // Collect all matching documents with scores
        let mut scores: HashMap<PointId, f32> = HashMap::new();

        for term in &query_tokens {
            if let Some(posting_list) = self.postings.get(term) {
                let df = posting_list.df();
                for posting in &posting_list.postings {
                    let doc_len = self
                        .doc_lengths
                        .get(&posting.doc_id)
                        .copied()
                        .unwrap_or(0);
                    let score =
                        self.params
                            .score(posting.tf, df, doc_len, avg_doc_len, self.total_docs);
                    *scores.entry(posting.doc_id).or_insert(0.0) += score;
                }
            }
        }

        // Sort by score descending
        let mut results: Vec<(PointId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        results
    }

    /// Get the document frequency for a term.
    pub fn document_frequency(&self, term: &str) -> u32 {
        self.postings.get(term).map(|pl| pl.df()).unwrap_or(0)
    }

    /// Get the posting list for a term.
    pub fn get_posting_list(&self, term: &str) -> Option<&PostingList> {
        self.postings.get(term)
    }

    /// Save index to file (bincode format).
    ///
    /// # Errors
    ///
    /// Returns `VhnswError::Storage` if serialization or file I/O fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VhnswError> {
        use std::io::Write;

        // Convert HashMap to Vec for bincode 2.x compatibility
        let data = Bm25IndexData {
            tokenizer: self.tokenizer.clone(),
            postings: self.postings.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            doc_lengths: self.doc_lengths.iter().map(|(k, v)| (*k, *v)).collect(),
            total_length: self.total_length,
            total_docs: self.total_docs,
            params: self.params,
        };

        // Serialize to Vec first to avoid any buffering issues
        let bytes = bincode::encode_to_vec(&data, bincode::config::standard()).map_err(|e| {
            VhnswError::Storage(std::io::Error::other(format!("serialize failed: {e}")))
        })?;

        let mut file = std::fs::File::create(path.as_ref())?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(())
    }

    /// Load index from file.
    ///
    /// # Errors
    ///
    /// Returns `VhnswError::Storage` if deserialization or file I/O fails.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, VhnswError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path.as_ref())?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let (data, _): (Bm25IndexData<T>, usize) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).map_err(|e| {
                VhnswError::Storage(std::io::Error::other(format!("deserialize failed: {e}")))
            })?;

        // Convert Vec back to HashMap
        Ok(Self {
            tokenizer: data.tokenizer,
            postings: data.postings.into_iter().collect(),
            doc_lengths: data.doc_lengths.into_iter().collect(),
            total_length: data.total_length,
            total_docs: data.total_docs,
            params: data.params,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple whitespace tokenizer for testing.
    #[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
    struct WhitespaceTokenizer;

    impl Tokenizer for WhitespaceTokenizer {
        fn tokenize(&self, text: &str) -> Vec<String> {
            text.split_whitespace()
                .map(|s| s.to_lowercase())
                .collect()
        }
    }

    #[test]
    fn test_empty_index() {
        let index = Bm25Index::new(WhitespaceTokenizer);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!((index.avg_doc_length() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_add_document() {
        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "hello world");
        assert_eq!(index.len(), 1);
        assert_eq!(index.document_frequency("hello"), 1);
        assert_eq!(index.document_frequency("world"), 1);
    }

    #[test]
    fn test_remove_document() {
        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "hello world");
        index.add_document(2, "hello there");

        assert!(index.remove_document(1));
        assert_eq!(index.len(), 1);
        assert_eq!(index.document_frequency("hello"), 1);
        assert_eq!(index.document_frequency("world"), 0);
    }

    #[test]
    fn test_search_empty_index() {
        let index = Bm25Index::new(WhitespaceTokenizer);
        let results = index.search("hello", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_basic() {
        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "the quick brown fox");
        index.add_document(2, "the lazy dog");
        index.add_document(3, "quick quick fox fox");

        let results = index.search("quick fox", 10);
        assert!(!results.is_empty());
        // Document 3 should rank highest (more term occurrences)
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_search_no_match() {
        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "hello world");
        let results = index.search("goodbye", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_update_document() {
        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "hello world");
        index.add_document(1, "goodbye world"); // Replace

        assert_eq!(index.len(), 1);
        assert_eq!(index.document_frequency("hello"), 0);
        assert_eq!(index.document_frequency("goodbye"), 1);
    }

    #[test]
    fn test_posting_list() {
        let mut pl = PostingList::new();
        pl.add(1, 3);
        pl.add(2, 1);
        assert_eq!(pl.df(), 2);

        pl.remove(1);
        assert_eq!(pl.df(), 1);
    }

    #[test]
    fn test_save_load() {
        use std::env;

        let mut index = Bm25Index::new(WhitespaceTokenizer);
        index.add_document(1, "hello world");
        index.add_document(2, "hello rust");
        index.add_document(3, "rust programming");

        // Save to temp file
        let temp_path = env::temp_dir().join("test_bm25_index.bin");
        index.save(&temp_path).expect("Failed to save index");

        // Load from file
        let loaded_index: Bm25Index<WhitespaceTokenizer> =
            Bm25Index::load(&temp_path).expect("Failed to load index");

        // Verify index state
        assert_eq!(loaded_index.len(), 3);
        assert_eq!(loaded_index.document_frequency("hello"), 2);
        assert_eq!(loaded_index.document_frequency("rust"), 2);
        assert_eq!(loaded_index.document_frequency("world"), 1);
        assert_eq!(loaded_index.document_frequency("programming"), 1);

        // Verify search works
        let results = loaded_index.search("hello", 10);
        assert_eq!(results.len(), 2);

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }
}
