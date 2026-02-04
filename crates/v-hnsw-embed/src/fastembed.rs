//! FastEmbed wrapper implementing the EmbeddingModel trait.

use std::sync::Mutex;

use crate::error::EmbedError;
use crate::model::{EmbeddingModel, Result};
use fastembed::{EmbeddingModel as FastEmbedModelEnum, InitOptions, TextEmbedding};

/// Supported embedding model types.
///
/// These are re-exported from fastembed for convenience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// BGE Small English v1.5 (33M params, 384 dim)
    /// Good balance of quality and speed.
    BGESmallENV15,

    /// BGE Base English v1.5 (110M params, 768 dim)
    /// Higher quality, slower than Small.
    BGEBaseENV15,

    /// BGE Large English v1.5 (335M params, 1024 dim)
    /// Highest quality BGE model.
    BGELargeENV15,

    /// All-MiniLM-L6-v2 (22M params, 384 dim)
    /// Very fast, good for general use.
    AllMiniLML6V2,

    /// All-MiniLM-L12-v2 (33M params, 384 dim)
    /// Slightly better than L6, still fast.
    AllMiniLML12V2,

    /// Multilingual E5 Small (118M params, 384 dim)
    /// Supports 100+ languages including Korean.
    MultilingualE5Small,

    /// Multilingual E5 Base (278M params, 768 dim)
    /// Better multilingual quality.
    MultilingualE5Base,

    /// Multilingual E5 Large (560M params, 1024 dim)
    /// Best multilingual quality.
    MultilingualE5Large,
}

impl ModelType {
    /// Returns the embedding dimension for this model type.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self {
            Self::BGESmallENV15 | Self::AllMiniLML6V2 | Self::AllMiniLML12V2 | Self::MultilingualE5Small => 384,
            Self::BGEBaseENV15 | Self::MultilingualE5Base => 768,
            Self::BGELargeENV15 | Self::MultilingualE5Large => 1024,
        }
    }

    /// Returns a human-readable name for this model type.
    #[must_use]
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::BGESmallENV15 => "BAAI/bge-small-en-v1.5",
            Self::BGEBaseENV15 => "BAAI/bge-base-en-v1.5",
            Self::BGELargeENV15 => "BAAI/bge-large-en-v1.5",
            Self::AllMiniLML6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMiniLML12V2 => "sentence-transformers/all-MiniLM-L12-v2",
            Self::MultilingualE5Small => "intfloat/multilingual-e5-small",
            Self::MultilingualE5Base => "intfloat/multilingual-e5-base",
            Self::MultilingualE5Large => "intfloat/multilingual-e5-large",
        }
    }

    fn to_fastembed(self) -> FastEmbedModelEnum {
        match self {
            Self::BGESmallENV15 => FastEmbedModelEnum::BGESmallENV15,
            Self::BGEBaseENV15 => FastEmbedModelEnum::BGEBaseENV15,
            Self::BGELargeENV15 => FastEmbedModelEnum::BGELargeENV15,
            Self::AllMiniLML6V2 => FastEmbedModelEnum::AllMiniLML6V2,
            Self::AllMiniLML12V2 => FastEmbedModelEnum::AllMiniLML12V2,
            Self::MultilingualE5Small => FastEmbedModelEnum::MultilingualE5Small,
            Self::MultilingualE5Base => FastEmbedModelEnum::MultilingualE5Base,
            Self::MultilingualE5Large => FastEmbedModelEnum::MultilingualE5Large,
        }
    }
}

impl Default for ModelType {
    /// Returns the default model type: AllMiniLML6V2.
    ///
    /// This model provides a good balance of quality, speed, and model size,
    /// making it suitable for most use cases.
    fn default() -> Self {
        Self::AllMiniLML6V2
    }
}

/// FastEmbed-based embedding model.
///
/// This wraps the fastembed library to provide text embeddings using
/// transformer models that run locally.
///
/// # Model Download
///
/// On first use, the model will be downloaded from Hugging Face Hub.
/// This is a one-time operation (~100MB depending on model).
///
/// # Example
///
/// ```no_run
/// use v_hnsw_embed::{FastEmbedModel, EmbeddingModel, ModelType};
///
/// // Use default model (AllMiniLML6V2)
/// let model = FastEmbedModel::try_new().unwrap();
///
/// // Or specify a model
/// let model = FastEmbedModel::with_model(ModelType::BGESmallENV15).unwrap();
///
/// // Generate embeddings
/// let embeddings = model.embed(&["hello world", "test text"]).unwrap();
/// let query_vec = model.embed_query("search query").unwrap();
/// ```
pub struct FastEmbedModel {
    // Mutex needed because fastembed 5.x requires &mut self for embed
    inner: Mutex<TextEmbedding>,
    model_type: ModelType,
}

impl FastEmbedModel {
    /// Creates a new FastEmbedModel with the default model (AllMiniLML6V2).
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to initialize or download.
    pub fn try_new() -> Result<Self> {
        Self::with_model(ModelType::default())
    }

    /// Creates a new FastEmbedModel with the specified model type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of embedding model to use
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to initialize or download.
    pub fn with_model(model_type: ModelType) -> Result<Self> {
        let options = InitOptions::new(model_type.to_fastembed()).with_show_download_progress(true);

        let inner = TextEmbedding::try_new(options)?;

        Ok(Self {
            inner: Mutex::new(inner),
            model_type,
        })
    }

    /// Creates a new FastEmbedModel with custom initialization options.
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of embedding model to use
    /// * `cache_dir` - Optional custom cache directory for model files
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to initialize or download.
    pub fn with_cache_dir(model_type: ModelType, cache_dir: impl Into<std::path::PathBuf>) -> Result<Self> {
        let options = InitOptions::new(model_type.to_fastembed())
            .with_show_download_progress(true)
            .with_cache_dir(cache_dir.into());

        let inner = TextEmbedding::try_new(options)?;

        Ok(Self {
            inner: Mutex::new(inner),
            model_type,
        })
    }

    /// Returns the model type being used.
    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        self.model_type
    }
}

impl EmbeddingModel for FastEmbedModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(EmbedError::InvalidInput("empty input slice".to_string()));
        }

        // Convert &[&str] to Vec<String> for fastembed
        let texts_owned: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();

        let mut inner = self
            .inner
            .lock()
            .map_err(|e| EmbedError::EmbeddingFailed(format!("lock poisoned: {e}")))?;

        inner
            .embed(texts_owned, None)
            .map_err(|e| EmbedError::EmbeddingFailed(e.to_string()))
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(EmbedError::InvalidInput("empty query string".to_string()));
        }

        let embeddings = self.embed(&[query])?;

        // Safety: We checked that query is not empty and embed requires non-empty input,
        // so embeddings will have exactly one element
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::EmbeddingFailed("no embedding returned".to_string()))
    }

    fn dim(&self) -> usize {
        self.model_type.dimension()
    }

    fn name(&self) -> &str {
        self.model_type.model_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_dimensions() {
        assert_eq!(ModelType::BGESmallENV15.dimension(), 384);
        assert_eq!(ModelType::BGEBaseENV15.dimension(), 768);
        assert_eq!(ModelType::BGELargeENV15.dimension(), 1024);
        assert_eq!(ModelType::AllMiniLML6V2.dimension(), 384);
        assert_eq!(ModelType::AllMiniLML12V2.dimension(), 384);
        assert_eq!(ModelType::MultilingualE5Small.dimension(), 384);
        assert_eq!(ModelType::MultilingualE5Base.dimension(), 768);
        assert_eq!(ModelType::MultilingualE5Large.dimension(), 1024);
    }

    #[test]
    fn test_model_type_names() {
        assert_eq!(ModelType::AllMiniLML6V2.model_name(), "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(ModelType::BGESmallENV15.model_name(), "BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn test_default_model_type() {
        assert_eq!(ModelType::default(), ModelType::AllMiniLML6V2);
    }
}
