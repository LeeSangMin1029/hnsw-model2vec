//! Embedding model trait definition.

use crate::error::EmbedError;

/// Result type alias for embedding operations.
pub type Result<T> = std::result::Result<T, EmbedError>;

/// Trait for text embedding models.
///
/// Implementations of this trait can convert text into dense vector representations
/// suitable for similarity search in vector indices.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow concurrent embedding operations.
pub trait EmbeddingModel: Send + Sync {
    /// Embed multiple texts into vectors.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to embed
    ///
    /// # Returns
    ///
    /// A vector of embedding vectors, one per input text.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Embed a single query text.
    ///
    /// Some models use different processing for queries vs. documents.
    /// This method provides query-optimized embedding.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to embed
    ///
    /// # Returns
    ///
    /// The embedding vector for the query.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding generation fails.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;

    /// Returns the dimension of embeddings produced by this model.
    fn dim(&self) -> usize;

    /// Returns the model name/identifier.
    fn name(&self) -> &str;
}
