//! Model2Vec embedding model implementation.
//!
//! This module provides integration with the official model2vec-rs library for lightweight,
//! static embedding models. Supports automatic download from HuggingFace Hub.

use crate::error::EmbedError;
use crate::model::{EmbeddingModel, Result};
use model2vec_rs::model::StaticModel;

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
    /// Uses `minishlab/potion-multilingual-128M` which produces 256-dimensional embeddings.
    /// Downloads from HuggingFace Hub on first use; cached locally afterwards.
    ///
    /// # Errors
    ///
    /// Returns `EmbedError::ModelInit` if model loading fails.
    pub fn new() -> Result<Self> {
        Self::from_pretrained("minishlab/potion-multilingual-128M")
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
        let model = StaticModel::from_pretrained(model_name, None, None, None)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model2vec_initialization() {
        let model = Model2VecModel::new();
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.dim(), 256);
        assert_eq!(model.name(), "minishlab/potion-multilingual-128M");
    }

    #[test]
    fn test_embed_single_text() {
        let model = Model2VecModel::new().unwrap();
        let result = model.embed(&["hello world"]);
        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 256);
    }

    #[test]
    fn test_embed_multiple_texts() {
        let model = Model2VecModel::new().unwrap();
        let texts = vec!["hello", "world", "안녕하세요"];
        let result = model.embed(&texts.iter().map(|s| *s).collect::<Vec<_>>());
        assert!(result.is_ok());
        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), 256);
        }
    }

    #[test]
    fn test_embed_query() {
        let model = Model2VecModel::new().unwrap();
        let result = model.embed_query("test query");
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 256);
    }

    #[test]
    fn test_empty_input_error() {
        let model = Model2VecModel::new().unwrap();
        let result = model.embed(&[]);
        assert!(result.is_err());
        matches!(result.unwrap_err(), EmbedError::InvalidInput(_));
    }

    #[test]
    fn test_empty_query_error() {
        let model = Model2VecModel::new().unwrap();
        let result = model.embed_query("");
        assert!(result.is_err());
        matches!(result.unwrap_err(), EmbedError::InvalidInput(_));
    }
}
