//! v-hnsw: Local vector database library.
//!
//! GPU-accelerated HNSW indexing with Korean language support,
//! hybrid search, and cross-platform compatibility.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use v_hnsw::{VectorDb, Metric};
//!
//! // Create an in-memory database
//! let db = VectorDb::builder()
//!     .dim(384)
//!     .metric(Metric::Cosine)
//!     .build()?;
//!
//! // Insert vectors
//! db.insert(1, &embedding)?;
//!
//! // Insert with text for hybrid search
//! db.insert_with_text(2, &embedding, "document text")?;
//!
//! // Dense search
//! let results = db.search(&query_vec, 10)?;
//!
//! // Hybrid search (dense + sparse BM25)
//! let results = db.hybrid_search(&query_vec, "search query", 10)?;
//! ```
//!
//! # Features
//!
//! - **HNSW Graph**: Approximate nearest neighbor search with configurable M, ef_construction, ef_search
//! - **Distance Metrics**: L2, Cosine, and Dot Product with SIMD optimization
//! - **Hybrid Search**: Combine dense vectors with BM25 sparse text search using RRF
//! - **Korean Support**: Morphological tokenization with Lindera ko-dic (optional `korean` feature)
//! - **Quantization**: SQ8 and PQ for memory-efficient storage (via v-hnsw-quantize)
//!
//! # Modules
//!
//! The facade crate re-exports key types from internal crates for convenience:
//!
//! - [`config`] - Configuration enums (Metric, Quantization)
//! - [`result`] - Search result types
//! - [`db`] - Main VectorDb facade and builder

mod config;
mod db;
mod result;

// Public API exports
pub use config::{Metric, Quantization};
pub use db::{VectorDb, VectorDbBuilder};
pub use result::SearchResult;

// Re-export core types for convenience
pub use v_hnsw_core::{Dim, Payload, PayloadValue, PointId, Result, VhnswError};

// Re-export graph types
pub use v_hnsw_graph::{HnswConfig, HnswConfigBuilder, HnswGraph};

// Re-export distance metrics
pub use v_hnsw_distance::{CosineDistance, DotProductDistance, L2Distance};

// Re-export search types
pub use v_hnsw_search::{
    Bm25Index, Bm25Params, HybridSearchConfig, HybridSearchConfigBuilder, Posting, PostingList,
    RrfFusion, SimpleHybridSearcher, SimpleTokenizer, WhitespaceTokenizer,
};

// Conditionally re-export tokenizer types
#[cfg(feature = "korean")]
pub use v_hnsw_tokenizer::{
    FilterChain, KoreanTokenizer, KoreanTokenizerConfig, LowercaseFilter, MinLengthFilter,
    StopwordFilter, Token, TokenFilter, TokenKind, TokenizerMode, UserDictionary,
};

// Re-export quantization types
pub use v_hnsw_quantize::{PqEncoded, PqQuantizer, Sq8Encoded, Sq8Quantizer};

// Re-export storage types
pub use v_hnsw_storage::{FilePayloadStore, MmapVectorStore, StorageConfig, StorageEngine, Wal};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        let db = VectorDb::builder().dim(4).build();
        assert!(db.is_ok());
        let db = db.ok();
        assert!(db.is_some());
        let db = db.map(|db| {
            // Insert vectors
            let _ = db.insert(1, &[1.0, 0.0, 0.0, 0.0]);
            let _ = db.insert(2, &[0.0, 1.0, 0.0, 0.0]);
            let _ = db.insert_with_text(3, &[0.0, 0.0, 1.0, 0.0], "hello world");

            assert_eq!(db.len(), 3);
            assert!(!db.is_empty());

            // Dense search
            let results = db.search(&[0.9, 0.1, 0.0, 0.0], 2);
            assert!(results.is_ok());
            let results = results.ok();
            assert!(results.is_some());
            results.map(|results| {
                assert_eq!(results.len(), 2);
                // Point 1 should be closest
                assert_eq!(results[0].id, 1);
            });

            // Hybrid search
            let results = db.hybrid_search(&[0.1, 0.1, 0.8, 0.0], "hello", 2);
            assert!(results.is_ok());

            // Get operations
            let vec = db.get(1);
            assert!(vec.is_ok());
            let text = db.get_text(3);
            assert!(text.is_ok());
            let text = text.ok();
            assert!(text.is_some());
            text.map(|t| assert_eq!(t, Some("hello world".to_string())));

            // Delete
            let result = db.delete(2);
            assert!(result.is_ok());
            assert_eq!(db.len(), 2);
        });
        assert!(db.is_some());
    }

    #[test]
    fn test_re_exports() {
        // Verify types are accessible through re-exports
        let _ = Metric::Cosine;
        let _ = Quantization::None;

        // Core types
        let _id: PointId = 42;
        let _dim: Dim = 128;

        // Distance metrics
        let _ = L2Distance;
        let _ = CosineDistance;
        let _ = DotProductDistance;

        // Search types
        let _ = SimpleTokenizer::new();
        let _ = WhitespaceTokenizer::new();
    }

    #[test]
    fn test_all_metrics() {
        for metric in [Metric::L2, Metric::Cosine, Metric::DotProduct] {
            let db = VectorDb::builder().dim(4).metric(metric).build();
            assert!(db.is_ok());
            let db = db.ok();
            assert!(db.is_some());
            db.map(|db| {
                let _ = db.insert(1, &[1.0, 0.0, 0.0, 0.0]);
                let results = db.search(&[1.0, 0.0, 0.0, 0.0], 1);
                assert!(results.is_ok());
                let results = results.ok();
                assert!(results.is_some());
                results.map(|results| {
                    assert_eq!(results.len(), 1);
                    assert_eq!(results[0].id, 1);
                });
            });
        }
    }
}
