//! Hybrid search for v-hnsw.
//!
//! Combines dense vector search (HNSW) with sparse keyword search (BM25)
//! using Reciprocal Rank Fusion (RRF) for optimal recall.
//!
//! # Features
//!
//! - **BM25 Index**: Okapi BM25 ranking for keyword-based retrieval
//! - **RRF Fusion**: Combines multiple ranked lists into a single ranking
//! - **Hybrid Search**: Seamlessly combines dense and sparse results
//! - **Reranking**: Pluggable reranker interface for result refinement
//!
//! # Example
//!
//! ```ignore
//! use v_hnsw_search::{HybridSearcher, HybridSearchConfig, Bm25Index};
//! use v_hnsw_graph::{HnswGraph, HnswConfig};
//!
//! // Create dense and sparse indexes
//! let hnsw_config = HnswConfig::builder().dim(384).build()?;
//! let hnsw = HnswGraph::new(hnsw_config, distance);
//! let bm25 = Bm25Index::new(tokenizer);
//!
//! // Create hybrid searcher
//! let config = HybridSearchConfig::default();
//! let mut searcher = SimpleHybridSearcher::new(hnsw, bm25, config);
//!
//! // Add documents
//! searcher.add_document(1, &embedding, "document text")?;
//!
//! // Search
//! let results = searcher.search(&query_vector, "query text", 10)?;
//! ```

pub mod bm25;
pub mod config;
pub mod fusion;
pub mod hybrid;
#[cfg(feature = "korean")]
pub mod korean_tokenizer;
pub mod reranker;
#[cfg(feature = "korean")]
pub mod tokenizer;

#[cfg(test)]
mod tests;

// Re-exports
pub use bm25::{Bm25Index, Bm25Params, Posting, PostingList};
pub use config::{HybridSearchConfig, HybridSearchConfigBuilder};
pub use fusion::RrfFusion;
pub use hybrid::{HybridSearcher, SimpleHybridSearcher};
pub use reranker::{LengthBoostReranker, PassthroughReranker, Reranker};
#[cfg(feature = "korean")]
pub use korean_tokenizer::{KoreanBm25Tokenizer, init_korean_tokenizer};

/// Tokenizer trait for text processing.
///
/// Implementations convert text into a sequence of tokens for BM25 indexing.
pub trait Tokenizer:
    Clone
    + Send
    + Sync
    + serde::Serialize
    + for<'de> serde::Deserialize<'de>
    + bincode::Encode
    + bincode::Decode<()>
{
    /// Tokenize text into a list of tokens.
    ///
    /// Tokens should be normalized (e.g., lowercased, stemmed) for best results.
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// A simple whitespace tokenizer.
///
/// Splits on whitespace and lowercases tokens. Suitable for basic use cases.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    /// Create a new whitespace tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_lowercase())
            .collect()
    }
}

/// A simple tokenizer that splits on whitespace and punctuation.
///
/// More thorough than `WhitespaceTokenizer` as it also removes punctuation.
#[derive(
    Debug, Clone, Default, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
pub struct SimpleTokenizer;

impl SimpleTokenizer {
    /// Create a new simple tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }
}
