//! Reranker model configurations.

/// Supported cross-encoder reranker models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum RerankerModel {
    /// MS MiniLM cross-encoder (FP32, 33M params).
    /// Model: microsoft/ms-marco-MiniLM-L-12-v2
    #[default]
    MsMiniLM,
    /// BGE reranker base (FP32, 110M params).
    /// Model: BAAI/bge-reranker-base
    BgeBase,
}

impl RerankerModel {
    /// Get the Hugging Face model repository ID.
    #[must_use]
    pub const fn model_id(self) -> &'static str {
        match self {
            Self::MsMiniLM => "Xenova/ms-marco-MiniLM-L-6-v2",
            Self::BgeBase => "Xenova/bge-reranker-base",
        }
    }

    /// Get the ONNX model file path within the repository.
    #[must_use]
    pub const fn model_file(self) -> &'static str {
        match self {
            Self::MsMiniLM => "onnx/model.onnx",
            Self::BgeBase => "onnx/model.onnx",
        }
    }

    /// Get the maximum sequence length (tokens) for this model.
    #[must_use]
    pub const fn max_length(self) -> usize {
        match self {
            Self::MsMiniLM => 512,
            Self::BgeBase => 512,
        }
    }
}

