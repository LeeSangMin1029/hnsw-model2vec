//! Cross-encoder reranker model.

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const DEFAULT_MODEL: &str = "cross-encoder/ms-marco-TinyBERT-L-2-v2";

/// A single rerank result.
pub struct RerankResult {
    /// Original index in the input documents slice.
    pub index: usize,
    /// Relevance score (higher = more relevant).
    pub score: f32,
}

/// Cross-encoder reranker using a BERT-based model.
pub struct CrossEncoderReranker {
    model: BertModel,
    pooler: Linear,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
}

impl CrossEncoderReranker {
    /// Load the default cross-encoder model (`ms-marco-MiniLM-L-6-v2`).
    pub fn new() -> Result<Self> {
        Self::from_pretrained(DEFAULT_MODEL)
    }

    /// Load a cross-encoder model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        let device = Device::Cpu;

        let api = Api::new().context("Failed to create HF Hub API")?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        tracing::info!("Loading reranker: {model_id}");

        let config_path = repo.get("config.json").context("Failed to download config.json")?;
        let tokenizer_path = repo.get("tokenizer.json").context("Failed to download tokenizer.json")?;
        let weights_path = get_weights_path(&repo)?;

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Failed to read config.json")?,
        )
        .context("Failed to parse config.json")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("Failed to load model weights")?
        };

        let model = BertModel::load(vb.clone(), &config).context("Failed to build BERT model")?;

        // Load pooler: CLS token → dense + tanh
        let pooler = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("bert.pooler.dense"))
            .context("Failed to load pooler")?;

        // Load classifier head: pooled output → linear → single logit
        let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))
            .context("Failed to load classifier head")?;

        tracing::info!("Reranker loaded: {model_id}");

        Ok(Self { model, pooler, classifier, tokenizer, device })
    }

    /// Rerank documents by relevance to the query.
    ///
    /// Returns results sorted by score descending.
    pub fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<RerankResult> = Vec::with_capacity(documents.len());

        // Process each (query, document) pair
        // Cross-encoders need to see both texts together
        for (idx, doc) in documents.iter().enumerate() {
            let score = self.score_pair(query, doc)?;
            results.push(RerankResult { index: idx, score });
        }

        // Sort by score descending
        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Score a single (query, document) pair.
    fn score_pair(&self, query: &str, document: &str) -> Result<f32> {
        let encoding = self.tokenizer
            .encode((query, document), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        // Truncate to max 512 tokens (model's position embedding limit)
        let max_len = 512;
        let len = encoding.get_ids().len().min(max_len);
        let token_ids = &encoding.get_ids()[..len];
        let type_ids = &encoding.get_type_ids()[..len];
        tracing::debug!("tokens: {}", len);

        let token_ids_t = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let type_ids_t = Tensor::new(type_ids, &self.device)?.unsqueeze(0)?;


        let output = self.model.forward(&token_ids_t, &type_ids_t, None)
            .map_err(|e| anyhow::anyhow!("forward pass failed: {e}"))?;

        tracing::debug!("output shape: {:?}", output.shape());

        // CLS token → pooler (dense + tanh) → classifier (linear → logit)
        let cls_output = output.i((0, 0))?.unsqueeze(0)?; // [1, hidden_dim]
        let pooled = self.pooler.forward(&cls_output)?.tanh()?; // [1, hidden_dim]
        let logit = self.classifier.forward(&pooled)?.squeeze(0)?.squeeze(0)?; // scalar
        let score = logit.to_scalar::<f32>()?;
        tracing::debug!("score: {score}");
        Ok(score)
    }

    /// Model name.
    pub fn name(&self) -> &str {
        DEFAULT_MODEL
    }
}

/// Find the weights file (model.safetensors or split shards).
fn get_weights_path(repo: &hf_hub::api::sync::ApiRepo) -> Result<PathBuf> {
    // Try single file first
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(path);
    }
    anyhow::bail!("No model.safetensors found in repository")
}
