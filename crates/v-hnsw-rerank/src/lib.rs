//! Cross-encoder reranking for v-hnsw.
//!
//! This crate provides cross-encoder models for reranking search results.
//! Cross-encoders process (query, document) pairs together, providing more
//! accurate relevance scores than bi-encoder retrieval alone.
//!
//! # Features
//!
//! - **Cross-Encoder Models**: MS-MARCO MiniLM and BGE reranker base
//! - **ONNX Runtime**: Fast inference with CPU/CUDA/DirectML support
//! - **Batched Scoring**: Efficient batch processing of candidates
//! - **Reranker Trait**: Implements `v_hnsw_search::Reranker` for integration
//!
//! # Example
//!
//! ```ignore
//! use v_hnsw_rerank::{CrossEncoderReranker, CrossEncoderConfig, RerankerModel};
//! use v_hnsw_search::Reranker;
//!
//! // Create cross-encoder reranker
//! let config = CrossEncoderConfig::new(RerankerModel::MsMiniLM);
//! let reranker = CrossEncoderReranker::new(config)?;
//!
//! // Rerank search results
//! let candidates = vec![
//!     (1, 0.8, "First document text".to_string()),
//!     (2, 0.7, "Second document text".to_string()),
//! ];
//! let reranked = reranker.rerank("query text", &candidates)?;
//! ```

pub mod cross_encoder;
pub mod model;

// Re-exports
pub use cross_encoder::{CrossEncoderConfig, CrossEncoderReranker};
pub use model::RerankerModel;
