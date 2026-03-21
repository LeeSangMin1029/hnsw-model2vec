//! Cross-encoder reranker using candle (pure Rust, no ONNX).
//!
//! Uses `cross-encoder/ms-marco-TinyBERT-L-2-v2` for fast CPU reranking.

mod model;

pub use model::{CrossEncoderReranker, RerankResult};

