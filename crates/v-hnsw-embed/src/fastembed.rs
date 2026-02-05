//! FastEmbed wrapper implementing the EmbeddingModel trait.

use std::sync::Mutex;

use crate::error::EmbedError;
use crate::model::{EmbeddingModel, Result};
use fastembed::{EmbeddingModel as FastEmbedModelEnum, InitOptions, TextEmbedding};
use ort::execution_providers::ExecutionProviderDispatch;

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
    /// Returns the default model type: MultilingualE5Base.
    ///
    /// This model supports 100+ languages including Korean and English,
    /// with 768 dimensions for good quality. It's the recommended default
    /// for an all-in-one solution.
    fn default() -> Self {
        Self::MultilingualE5Base
    }
}

impl ModelType {
    /// Returns the short name for CLI display.
    #[must_use]
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::BGESmallENV15 => "bge-small-en-v1.5",
            Self::BGEBaseENV15 => "bge-base-en-v1.5",
            Self::BGELargeENV15 => "bge-large-en-v1.5",
            Self::AllMiniLML6V2 => "all-mini-lm-l6-v2",
            Self::AllMiniLML12V2 => "all-mini-lm-l12-v2",
            Self::MultilingualE5Small => "multilingual-e5-small",
            Self::MultilingualE5Base => "multilingual-e5-base",
            Self::MultilingualE5Large => "multilingual-e5-large",
        }
    }
}

/// Device selection for embedding model inference.
///
/// Controls which hardware accelerator (execution provider) is used
/// for ONNX Runtime inference. GPU options require the corresponding
/// cargo feature to be enabled.
///
/// # Features
///
/// - `cuda` feature: enables `Device::Cuda`
/// - `directml` feature: enables `Device::DirectML`
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Device {
    /// CPU inference (always available).
    #[default]
    Cpu,

    /// NVIDIA CUDA GPU inference.
    ///
    /// Requires the `cuda` cargo feature. The `i32` parameter is
    /// the CUDA device ID (0 for the first GPU).
    #[cfg(feature = "cuda")]
    Cuda(i32),

    /// DirectML GPU inference (Windows).
    ///
    /// Requires the `directml` cargo feature. The `i32` parameter is
    /// the DirectML device ID (0 for the first GPU).
    #[cfg(feature = "directml")]
    DirectML(i32),
}

impl Device {
    /// Automatically detect the best available device.
    ///
    /// Priority: CUDA > DirectML > CPU
    /// Uses device ID 0 for GPU devices.
    #[must_use]
    pub fn auto() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Try CUDA first
            return Self::Cuda(0);
        }

        #[cfg(feature = "directml")]
        {
            // Try DirectML on Windows
            return Self::DirectML(0);
        }

        #[allow(unreachable_code)]
        Self::Cpu
    }

    /// Returns a human-readable name for this device.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => "CUDA",
            #[cfg(feature = "directml")]
            Self::DirectML(_) => "DirectML",
        }
    }

    /// Convert the device selection into ONNX Runtime execution providers.
    ///
    /// GPU providers are configured with `fail_silently()` so that if the
    /// GPU is unavailable, inference falls back to CPU automatically.
    pub(crate) fn to_execution_providers(self) -> Vec<ExecutionProviderDispatch> {
        match self {
            Self::Cpu => vec![],
            #[cfg(feature = "cuda")]
            Self::Cuda(device_id) => {
                use ort::ep::{self, ArenaExtendStrategy};
                vec![ep::CUDA::default()
                    .with_device_id(device_id)
                    // Ampere (sm_86) TF32 tensor core acceleration
                    .with_tf32(true)
                    // Heuristic conv search avoids slow first-run exhaustive benchmark
                    .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
                    // Allow cuDNN to use maximum workspace for best algorithm selection
                    .with_conv_max_workspace(true)
                    // SameAsRequested avoids over-allocating VRAM on 6GB cards
                    .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                    .build()
                    .error_on_failure()]
            }
            #[cfg(feature = "directml")]
            Self::DirectML(device_id) => {
                use ort::execution_providers::DirectMLExecutionProvider;
                vec![DirectMLExecutionProvider::default()
                    .with_device_id(device_id)
                    .build()
                    .error_on_failure()]
            }
        }
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

    /// Creates a new FastEmbedModel with the specified device for hardware acceleration.
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of embedding model to use
    /// * `device` - The device to run inference on (CPU, CUDA, or DirectML)
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to initialize, download, or if
    /// the requested execution provider is not available.
    pub fn with_device(model_type: ModelType, device: Device) -> Result<Self> {
        let execution_providers = device.to_execution_providers();
        let options = InitOptions::new(model_type.to_fastembed())
            .with_show_download_progress(true)
            .with_execution_providers(execution_providers);

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

impl FastEmbedModel {
    /// Creates a new FastEmbedModel with automatic configuration.
    ///
    /// Uses the default model (MultilingualE5Base) and automatically
    /// detects the best available device (CUDA > DirectML > CPU).
    ///
    /// This is the recommended way to create a model for most use cases.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to initialize or download.
    pub fn auto() -> Result<Self> {
        Self::with_device(ModelType::default(), Device::auto())
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

        // Pass the full batch size so fastembed processes all texts
        // in a single ONNX session.run() call (no internal re-batching).
        // The caller is responsible for choosing an appropriate batch size.
        inner
            .embed(texts_owned, Some(texts.len()))
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
