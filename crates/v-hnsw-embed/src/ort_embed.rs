//! Direct ort Session embedding model with FP16 support and Rust mean pooling.

use std::sync::Mutex;

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::error::EmbedError;
use crate::fastembed::Device;
use crate::model::{EmbeddingModel, Result};

/// Configuration for a direct ort embedding model.
#[derive(Debug, Clone)]
pub struct OrtModelConfig {
    /// Hugging Face model repository ID.
    pub model_id: &'static str,
    /// Path to the ONNX model file within the repository.
    pub model_file: &'static str,
    /// Embedding vector dimension.
    pub dim: usize,
    /// Maximum sequence length in tokens.
    pub max_seq_length: usize,
    /// Whether to L2-normalize output embeddings.
    pub normalize: bool,
}

impl OrtModelConfig {
    /// MiniLM-L6-v2 FP16 preset (Xenova).
    #[must_use]
    pub const fn minilm_fp16() -> Self {
        Self {
            model_id: "Xenova/all-MiniLM-L6-v2",
            model_file: "onnx/model_fp16.onnx",
            dim: 384,
            max_seq_length: 256,
            normalize: true,
        }
    }

    /// MiniLM-L6-v2 INT8 quantized preset (Xenova).
    #[must_use]
    pub const fn minilm_q8() -> Self {
        Self {
            model_id: "Xenova/all-MiniLM-L6-v2",
            model_file: "onnx/model_quantized.onnx",
            dim: 384,
            max_seq_length: 256,
            normalize: true,
        }
    }

    /// BGE-small-en-v1.5 FP16 preset (Xenova).
    #[must_use]
    pub const fn bge_small_fp16() -> Self {
        Self {
            model_id: "Xenova/bge-small-en-v1.5",
            model_file: "onnx/model_fp16.onnx",
            dim: 384,
            max_seq_length: 512,
            normalize: true,
        }
    }

    /// Character hint for pre-truncation before tokenization.
    /// Approximately `max_seq_length * 4` characters covers the token budget.
    #[must_use]
    pub const fn max_char_hint(&self) -> usize {
        self.max_seq_length * 4
    }
}

/// Direct ort Session embedding model.
///
/// Bypasses fastembed to load Xenova FP16/quantized ONNX models directly,
/// with Rust-side mean pooling and L2 normalization.
pub struct OrtEmbedModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: OrtModelConfig,
}

impl OrtEmbedModel {
    /// Create a new `OrtEmbedModel` with the given config and device.
    ///
    /// Downloads the model and tokenizer from Hugging Face Hub on first use.
    pub fn with_device(config: OrtModelConfig, device: Device) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| EmbedError::ModelInit(format!("hf-hub init failed: {e}")))?;
        let repo = api.model(config.model_id.to_string());

        // Download model file
        let model_path = repo
            .get(config.model_file)
            .map_err(|e| EmbedError::Download(format!("{}: {e}", config.model_file)))?;

        // Download and configure tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| EmbedError::Download(format!("tokenizer.json: {e}")))?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::ModelInit(format!("tokenizer load failed: {e}")))?;

        // Configure padding (batch to longest in batch)
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            pad_token: "[PAD]".to_string(),
            ..Default::default()
        }));

        // Configure truncation to max_seq_length
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_seq_length,
                ..Default::default()
            }))
            .map_err(|e| EmbedError::ModelInit(format!("truncation config failed: {e}")))?;

        // Build ort Session with execution providers from Device
        let eps = device.to_execution_providers();
        let mut builder = Session::builder()
            .map_err(|e| EmbedError::ModelInit(format!("session builder failed: {e}")))?;
        builder = builder
            .with_execution_providers(eps)
            .map_err(|e| EmbedError::ModelInit(format!("execution provider failed: {e}")))?;
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbedError::ModelInit(format!("optimization level failed: {e}")))?;

        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| EmbedError::ModelInit(format!("model load failed: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
        })
    }

    /// Character hint for caller-side pre-truncation.
    #[must_use]
    pub const fn max_char_hint(&self) -> usize {
        self.config.max_char_hint()
    }

    /// Embed a batch of texts. Internal implementation shared by trait methods.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(EmbedError::InvalidInput("empty input slice".to_string()));
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::EmbeddingFailed(format!("tokenization failed: {e}")))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        // Build input_ids and attention_mask as flat Vec<i64> with shape [batch, seq_len]
        let mut input_ids_data = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_data = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            for &id in enc.get_ids() {
                input_ids_data.push(id as i64);
            }
            for &m in enc.get_attention_mask() {
                attention_mask_data.push(m as i64);
            }
        }

        let shape = vec![batch_size as i64, seq_len as i64];
        let input_ids_tensor = Tensor::from_array((shape.clone(), input_ids_data))
            .map_err(|e| EmbedError::EmbeddingFailed(format!("input_ids tensor error: {e}")))?;
        let attention_mask_tensor = Tensor::from_array((shape.clone(), attention_mask_data))
            .map_err(|e| EmbedError::EmbeddingFailed(format!("attention_mask tensor error: {e}")))?;
        // token_type_ids: all zeros for single-sentence embedding
        let token_type_ids_tensor = Tensor::from_array((shape, vec![0i64; batch_size * seq_len]))
            .map_err(|e| EmbedError::EmbeddingFailed(format!("token_type_ids tensor error: {e}")))?;

        // Run inference and extract tensor data while session lock is held
        let (hidden_shape, hidden_data) = {
            let mut session = self
                .session
                .lock()
                .map_err(|e| EmbedError::EmbeddingFailed(format!("lock poisoned: {e}")))?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "token_type_ids" => token_type_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| EmbedError::EmbeddingFailed(format!("inference failed: {e}")))?;

            // Extract last_hidden_state: [batch, seq_len, dim]
            let (shape, data) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbedError::EmbeddingFailed(format!("output extraction failed: {e}")))?;
            (shape.clone(), data.to_vec())
        };
        let dim = self.config.dim;

        // Mean pooling with attention mask + L2 normalization
        // hidden_data is flat: [batch * seq_len * dim]
        let _ = hidden_shape; // shape already known from batch_size, seq_len, dim
        let mut result = Vec::with_capacity(batch_size);
        for (i, encoding) in encodings.iter().enumerate().take(batch_size) {
            let mut emb = vec![0.0f32; dim];
            let mut count = 0.0f32;

            for j in 0..seq_len {
                let mask_val = encoding.get_attention_mask()[j] as f32;
                if mask_val > 0.0 {
                    count += mask_val;
                    let offset = (i * seq_len + j) * dim;
                    for d in 0..dim {
                        emb[d] += hidden_data[offset + d] * mask_val;
                    }
                }
            }

            // Average
            if count > 0.0 {
                for val in emb.iter_mut().take(dim) {
                    *val /= count;
                }
            }

            // L2 normalize
            if self.config.normalize {
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm = norm.max(1e-12);
                for val in emb.iter_mut().take(dim) {
                    *val /= norm;
                }
            }

            result.push(emb);
        }

        Ok(result)
    }
}

impl EmbeddingModel for OrtEmbedModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(EmbedError::InvalidInput("empty query string".to_string()));
        }
        let embeddings = self.embed_batch(&[query])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedError::EmbeddingFailed("no embedding returned".to_string()))
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn name(&self) -> &str {
        self.config.model_id
    }
}
