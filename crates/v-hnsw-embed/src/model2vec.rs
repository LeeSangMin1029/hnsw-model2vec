//! Model2Vec embedding model implementation.
//!
//! This module provides integration with the official model2vec-rs library for lightweight,
//! static embedding models. Supports automatic download from HuggingFace Hub.

use crate::error::EmbedError;
use crate::model::{EmbeddingModel, Result};
use model2vec_rs::model::StaticModel;

/// Default HuggingFace model ID.
const DEFAULT_MODEL: &str = "minishlab/potion-multilingual-128M";

/// Local f16 model path (under ~/.v-hnsw/models/).
const F16_SUBDIR: &str = "potion-multilingual-128M-f16";

/// Model2Vec embedding model.
///
/// Uses the `minishlab/potion-multilingual-128M` model, which provides:
/// - 256-dimensional embeddings
/// - Multilingual support (including Korean)
/// - Lightweight inference without neural networks
/// - Automatic download from HuggingFace Hub
pub struct Model2VecModel {
    model: StaticModel,
    model_name: String,
    dim: usize,
}

impl Model2VecModel {
    /// Create a new Model2VecModel with the default multilingual model.
    ///
    /// Prefers a local f16 model at `~/.v-hnsw/models/potion-multilingual-128M-f16`
    /// for ~50% memory savings. Falls back to the HuggingFace f32 model if not found.
    ///
    /// # Errors
    ///
    /// Returns `EmbedError::ModelInit` if model loading fails.
    pub fn new() -> Result<Self> {
        Self::from_pretrained(DEFAULT_MODEL)
    }

    /// Create a Model2VecModel from a specific pretrained model.
    ///
    /// # Arguments
    ///
    /// * `model_name` - HuggingFace model identifier (e.g., "minishlab/potion-multilingual-128M")
    ///   or a local directory path.
    ///
    /// # Errors
    ///
    /// Returns `EmbedError::ModelInit` if model loading fails.
    /// Returns `EmbedError::Download` if model download fails.
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        // Prefer local f16 model for ~50% memory savings
        let load_path = if model_name == DEFAULT_MODEL {
            find_f16_model()
        } else {
            None
        };
        let (actual_path, display_name) = match load_path {
            Some(ref p) => (p.to_str().unwrap_or(model_name), "[f16] local"),
            None => (model_name, ""),
        };

        if !display_name.is_empty() {
            eprintln!("  Using f16 model: {}", actual_path);
        }

        let model = StaticModel::from_pretrained(actual_path, None, None, None)
            .map_err(|e| {
                let msg = e.to_string();
                if msg.contains("download") || msg.contains("network") || msg.contains("HTTP") {
                    EmbedError::Download(msg)
                } else {
                    EmbedError::ModelInit(msg)
                }
            })?;

        // Determine dimension by encoding a sample
        let sample = model.encode_single("dim");
        let dim = sample.len();

        Ok(Self {
            model,
            model_name: model_name.to_string(),
            dim,
        })
    }
}

/// Check if a local f16 model exists at ~/.v-hnsw/models/.
fn find_f16_model() -> Option<std::path::PathBuf> {
    let path = v_hnsw_core::data_dir().join("models").join(F16_SUBDIR);
    if path.join("model.safetensors").exists() && path.join("tokenizer.json").exists() {
        Some(path)
    } else {
        None
    }
}

impl EmbeddingModel for Model2VecModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(EmbedError::InvalidInput("empty text list".to_string()));
        }

        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        let embeddings = self.model.encode(&owned);

        Ok(embeddings)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(EmbedError::InvalidInput("empty query".to_string()));
        }

        Ok(self.model.encode_single(query))
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}
