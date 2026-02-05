//! Cross-encoder reranker implementation.

use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex as ParkingLotMutex;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::debug;

use v_hnsw_core::{PointId, Result, VhnswError};
use v_hnsw_search::Reranker;

use crate::model::RerankerModel;

/// Configuration for cross-encoder reranker.
#[derive(Debug, Clone)]
pub struct CrossEncoderConfig {
    /// The reranker model to use.
    pub model: RerankerModel,
    /// Maximum sequence length (tokens). Defaults to model's max_length.
    pub max_length: Option<usize>,
    /// Batch size for reranking. Defaults to 32.
    pub batch_size: usize,
}

impl CrossEncoderConfig {
    /// Create a new config with the specified model.
    #[must_use]
    pub fn new(model: RerankerModel) -> Self {
        Self {
            model,
            max_length: None,
            batch_size: 32,
        }
    }

    /// Set the maximum sequence length.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set the batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Get the effective max length (config override or model default).
    #[must_use]
    pub fn effective_max_length(&self) -> usize {
        self.max_length.unwrap_or_else(|| self.model.max_length())
    }
}

impl Default for CrossEncoderConfig {
    fn default() -> Self {
        Self::new(RerankerModel::default())
    }
}

/// Cross-encoder reranker using ONNX Runtime.
///
/// Scores (query, document) pairs using a cross-encoder model that processes
/// both texts together, providing more accurate relevance scores than bi-encoders.
pub struct CrossEncoderReranker {
    session: Mutex<Session>,
    tokenizer: ParkingLotMutex<Tokenizer>,
    config: CrossEncoderConfig,
}

impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker with the given config.
    ///
    /// Downloads the model and tokenizer from Hugging Face Hub on first use.
    ///
    /// # Errors
    /// Returns error if model download or initialization fails.
    pub fn new(config: CrossEncoderConfig) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("hf-hub init failed: {e}"))))?;
        let repo = api.model(config.model.model_id().to_string());

        // Download model file
        debug!(
            "Downloading reranker model {} from Hugging Face Hub",
            config.model.model_id()
        );
        let model_path = repo
            .get(config.model.model_file())
            .map_err(|e| VhnswError::Storage(std::io::Error::new(std::io::ErrorKind::NotFound, format!("{}: {e}", config.model.model_file()))))?;

        // Download tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| VhnswError::Storage(std::io::Error::new(std::io::ErrorKind::NotFound, format!("tokenizer.json: {e}"))))?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| VhnswError::Tokenizer(format!("tokenizer load failed: {e}")))?;

        // Configure tokenizer
        let max_length = config.effective_max_length();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| VhnswError::Tokenizer(format!("truncation config failed: {e}")))?;

        // Build ort Session
        debug!("Initializing ONNX Runtime session for reranker");
        let session = Session::builder()
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("session builder failed: {e}"))))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("optimization level failed: {e}"))))?
            .commit_from_file(&model_path)
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("model load failed: {e}"))))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: ParkingLotMutex::new(tokenizer),
            config,
        })
    }

    /// Score a batch of (query, document) pairs.
    ///
    /// Returns relevance scores for each pair. Higher scores indicate better relevance.
    fn score_batch(&self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all pairs
        let tokenizer = self.tokenizer.lock();
        let encodings = tokenizer
            .encode_batch(
                pairs
                    .iter()
                    .map(|(q, d)| (q.as_str(), d.as_str()))
                    .collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| VhnswError::Tokenizer(format!("tokenization failed: {e}")))?;
        drop(tokenizer);

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        // Build input tensors
        let mut input_ids_data = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_data = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids_data = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            for id in enc.get_ids() {
                input_ids_data.push(*id as i64);
            }
            for m in enc.get_attention_mask() {
                attention_mask_data.push(*m as i64);
            }
            for t in enc.get_type_ids() {
                token_type_ids_data.push(*t as i64);
            }
        }

        let shape = vec![batch_size as i64, seq_len as i64];
        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids_data))
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("input_ids tensor error: {e}"))))?;
        let attention_mask_tensor = Tensor::from_array((shape.clone(), attention_mask_data))
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("attention_mask tensor error: {e}"))))?;
        let token_type_ids_tensor = Tensor::from_array((shape, token_type_ids_data))
            .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("token_type_ids tensor error: {e}"))))?;

        // Run inference
        let scores = {
            let mut session = self
                .session
                .lock()
                .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("lock poisoned: {e}"))))?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "token_type_ids" => token_type_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("inference failed: {e}"))))?;

            // Extract logits: [batch, num_labels] or [batch]
            let (logits_shape, logits_data) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| VhnswError::Storage(std::io::Error::other(format!("output extraction failed: {e}"))))?;

            // For cross-encoder, we typically get [batch, 1] or [batch]
            // Extract the score for each pair
            let shape_dims: Vec<i64> = logits_shape.iter().copied().collect();
            let scores: Vec<f32> = if shape_dims.len() == 2 {
                // [batch, num_labels] - take first column
                let num_labels = shape_dims[1] as usize;
                (0..batch_size)
                    .map(|i| logits_data[i * num_labels])
                    .collect()
            } else {
                // [batch] - already flat
                logits_data.to_vec()
            };

            scores
        };

        Ok(scores)
    }
}

impl Reranker for CrossEncoderReranker {
    fn rerank(
        &self,
        query: &str,
        candidates: &[(PointId, f32, String)],
    ) -> Result<Vec<(PointId, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Reranking {} candidates with cross-encoder", candidates.len());

        // Process in batches
        let mut all_scores = Vec::with_capacity(candidates.len());
        for chunk in candidates.chunks(self.config.batch_size) {
            // Create (query, document) pairs
            let pairs: Vec<(String, String)> = chunk
                .iter()
                .map(|(_, _, text)| (query.to_string(), text.clone()))
                .collect();

            // Score this batch
            let scores = self.score_batch(&pairs)?;
            all_scores.extend(scores);
        }

        // Combine with IDs and sort by score descending
        let mut results: Vec<(PointId, f32)> = candidates
            .iter()
            .zip(all_scores)
            .map(|((id, _, _), score)| (*id, score))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        debug!("Reranking complete, top score: {:.4}", results[0].1);

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_cross_encoder_rerank() {
        let config = CrossEncoderConfig::default();
        let reranker = CrossEncoderReranker::new(config).expect("failed to create reranker");

        let candidates = vec![
            (1, 0.5, "The cat sat on the mat".to_string()),
            (2, 0.6, "Python is a programming language".to_string()),
            (3, 0.7, "Cats are great pets".to_string()),
        ];

        let results = reranker
            .rerank("information about cats", &candidates)
            .expect("rerank failed");

        assert_eq!(results.len(), 3);
        // Expect cat-related documents to score higher
        assert!(results[0].0 == 1 || results[0].0 == 3);
    }
}
