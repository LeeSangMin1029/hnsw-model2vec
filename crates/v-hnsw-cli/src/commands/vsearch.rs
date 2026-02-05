//! Vsearch command - Semantic vector search with auto-embedding.
//!
//! Takes a text query, embeds it automatically, and searches the HNSW index.
//! Automatically uses the same embedding model that was used to create the index.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use v_hnsw_core::VectorIndex;
use v_hnsw_distance::CosineDistance;
use v_hnsw_embed::{EmbeddingModel, FastEmbedModel, ModelType};
use v_hnsw_graph::HnswGraph;

use super::create::DbConfig;

/// Search result for JSON output.
#[derive(Debug, Serialize)]
struct VSearchOutput {
    results: Vec<VSearchResult>,
    query: String,
    model: String,
    elapsed_ms: f64,
}

#[derive(Debug, Serialize)]
struct VSearchResult {
    id: u64,
    score: f32,
    text: Option<String>,
}

/// Vsearch command parameters.
pub struct VSearchParams {
    pub path: PathBuf,
    pub query: String,
    pub k: usize,
    pub ef: usize,
    pub model: Option<String>,  // Optional - auto-detect from config
    pub show_text: bool,
}

/// Parse model name to ModelType.
fn parse_model_type(name: &str) -> Result<ModelType> {
    match name.to_lowercase().as_str() {
        "all-mini-lm-l6-v2" | "minilm" => Ok(ModelType::AllMiniLML6V2),
        "all-mini-lm-l12-v2" => Ok(ModelType::AllMiniLML12V2),
        "bge-small-en-v1.5" | "bge-small" => Ok(ModelType::BGESmallENV15),
        "bge-base-en-v1.5" | "bge-base" => Ok(ModelType::BGEBaseENV15),
        "bge-large-en-v1.5" | "bge-large" => Ok(ModelType::BGELargeENV15),
        "multilingual-e5-small" | "e5-small" => Ok(ModelType::MultilingualE5Small),
        "multilingual-e5-base" | "e5-base" => Ok(ModelType::MultilingualE5Base),
        "multilingual-e5-large" | "e5-large" => Ok(ModelType::MultilingualE5Large),
        other => anyhow::bail!(
            "Unknown model: '{}'. Available: all-mini-lm-l6-v2, bge-small-en-v1.5, bge-base-en-v1.5, multilingual-e5-small, etc.",
            other
        ),
    }
}

/// Run the vsearch command.
pub fn run(params: VSearchParams) -> Result<()> {
    let VSearchParams {
        path,
        query,
        k,
        ef,
        model,
        show_text,
    } = params;

    // Check database exists
    if !path.exists() {
        anyhow::bail!("Database not found at {}", path.display());
    }

    // Load config
    let config = DbConfig::load(&path)?;

    // Check for HNSW index
    let hnsw_path = path.join("hnsw.bin");
    if !hnsw_path.exists() {
        anyhow::bail!(
            "HNSW index not found. Run 'v-hnsw build-index {}' first.",
            path.display()
        );
    }

    // Determine model: user-specified or auto-detect from config
    let model_name = match model {
        Some(m) => m,
        None => {
            config.embed_model.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "No embedding model specified in database config.\n\
                     Either:\n\
                     1. Re-insert data with --embed to auto-save model info, or\n\
                     2. Specify --model manually"
                )
            })?
        }
    };

    let start = Instant::now();

    // Initialize embedding model
    let model_type = parse_model_type(&model_name)?;
    let embed_model = FastEmbedModel::with_model(model_type)
        .context("Failed to initialize embedding model")?;

    // Check dimension matches
    if embed_model.dim() != config.dim {
        anyhow::bail!(
            "Model dimension ({}) doesn't match database dimension ({}).\n\
             The database was likely created with a different model.",
            embed_model.dim(),
            config.dim
        );
    }

    // Embed the query
    let query_embedding = embed_model
        .embed(&[query.as_str()])
        .context("Failed to embed query")?
        .into_iter()
        .next()
        .context("No embedding returned")?;

    // Load HNSW index
    let hnsw: HnswGraph<CosineDistance> = HnswGraph::load(&hnsw_path, CosineDistance)
        .context("Failed to load HNSW index")?;

    // Search
    let results = hnsw
        .search(&query_embedding, k, ef)
        .context("Search failed")?;

    // Optionally load text
    let results_with_text: Vec<VSearchResult> = if show_text {
        let engine = v_hnsw_storage::StorageEngine::open(&path)
            .context("Failed to open storage")?;
        let payload_store = engine.payload_store();

        results
            .into_iter()
            .map(|(id, score)| {
                let text = v_hnsw_core::PayloadStore::get_text(&*payload_store, id)
                    .ok()
                    .flatten();
                VSearchResult { id, score, text }
            })
            .collect()
    } else {
        results
            .into_iter()
            .map(|(id, score)| VSearchResult {
                id,
                score,
                text: None,
            })
            .collect()
    };

    let elapsed = start.elapsed();

    // Output as JSON
    let output = VSearchOutput {
        results: results_with_text,
        query,
        model: model_name,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    };

    let json = serde_json::to_string_pretty(&output).context("Failed to serialize output")?;
    println!("{json}");

    Ok(())
}
