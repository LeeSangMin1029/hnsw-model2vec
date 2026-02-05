//! Error types for embedding operations.

/// Errors that can occur during embedding operations.
#[derive(thiserror::Error, Debug)]
pub enum EmbedError {
    /// Failed to initialize or load the embedding model.
    #[error("model initialization failed: {0}")]
    ModelInit(String),

    /// Failed to generate embeddings from input text.
    #[error("embedding generation failed: {0}")]
    EmbeddingFailed(String),

    /// Input validation error (e.g., empty input).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Model download failed.
    #[error("model download failed: {0}")]
    Download(String),
}
