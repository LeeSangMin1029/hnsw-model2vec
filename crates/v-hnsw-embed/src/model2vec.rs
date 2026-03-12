//! Model2Vec embedding model — mmap zero-copy backend.
//!
//! Thin wrapper around `MmapStaticModel` that implements the public `Model2VecModel` API.
//! For f32 safetensors the embedding matrix is never copied to the heap;
//! the OS pages in only the vocabulary rows actually accessed.

use crate::error::EmbedError;
use crate::mmap_model::MmapStaticModel;
use crate::model::{EmbeddingModel, Result};

/// Default HuggingFace model ID.
const DEFAULT_MODEL: &str = "minishlab/potion-base-32M";

/// Model2Vec embedding model (mmap zero-copy backend).
///
/// Uses `minishlab/potion-multilingual-128M` by default:
/// - 256-dimensional embeddings
/// - Multilingual support (including Korean)
/// - Lightweight inference without neural networks
/// - Automatic download from HuggingFace Hub
pub struct Model2VecModel {
    inner: MmapStaticModel,
}

impl Model2VecModel {
    /// Create with the default multilingual model.
    ///
    /// # Errors
    ///
    /// Returns `EmbedError::ModelInit` if model loading fails.
    pub fn new() -> Result<Self> {
        Self::from_pretrained(DEFAULT_MODEL)
    }

    /// Create from a specific pretrained model.
    ///
    /// # Errors
    ///
    /// Returns `EmbedError::ModelInit` or `EmbedError::Download` on failure.
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let inner = MmapStaticModel::from_pretrained(model_name)?;
        Ok(Self { inner })
    }
}

impl EmbeddingModel for Model2VecModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(EmbedError::InvalidInput("empty text list".into()));
        }
        self.inner.encode_batch(texts)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(EmbedError::InvalidInput("empty query".into()));
        }
        self.inner.encode_single(query)
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}
