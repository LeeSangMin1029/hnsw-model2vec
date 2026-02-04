//! Embedding model integration for v-hnsw.
//!
//! This crate provides text embedding capabilities using fastembed-rs,
//! enabling semantic similarity search with the v-hnsw vector index.
//!
//! # Quick Start
//!
//! ```no_run
//! use v_hnsw_embed::{FastEmbedModel, EmbeddingModel, ModelType};
//!
//! // Create a model with defaults (AllMiniLML6V2)
//! let model = FastEmbedModel::try_new()?;
//!
//! // Or specify a model explicitly
//! let model = FastEmbedModel::with_model(ModelType::BGESmallENV15)?;
//!
//! // Embed documents
//! let docs = &["Document one", "Document two"];
//! let embeddings = model.embed(docs)?;
//!
//! // Embed a query (some models optimize query embeddings differently)
//! let query_embedding = model.embed_query("search query")?;
//!
//! // Use with v-hnsw index
//! println!("Model: {}", model.name());
//! println!("Dimension: {}", model.dim());
//! # Ok::<(), v_hnsw_embed::EmbedError>(())
//! ```
//!
//! # Available Models
//!
//! | Model | Dimensions | Size | Speed | Use Case |
//! |-------|------------|------|-------|----------|
//! | AllMiniLML6V2 | 384 | ~22MB | Fast | General purpose (default) |
//! | AllMiniLML12V2 | 384 | ~33MB | Fast | Slightly better quality |
//! | BGESmallENV15 | 384 | ~33MB | Fast | English, good quality |
//! | BGEBaseENV15 | 768 | ~110MB | Medium | English, better quality |
//! | BGELargeENV15 | 1024 | ~335MB | Slow | English, best quality |
//! | MultilingualE5Small | 384 | ~118MB | Medium | 100+ languages |
//! | MultilingualE5Base | 768 | ~278MB | Slow | 100+ languages, better |
//! | MultilingualE5Large | 1024 | ~560MB | Slowest | 100+ languages, best |
//!
//! # Model Download
//!
//! Models are downloaded from Hugging Face Hub on first use.
//! Download progress is displayed by default. Models are cached
//! locally for subsequent use.
//!
//! To customize the cache location:
//!
//! ```no_run
//! use v_hnsw_embed::{FastEmbedModel, ModelType};
//!
//! let model = FastEmbedModel::with_cache_dir(
//!     ModelType::BGESmallENV15,
//!     "/custom/cache/path"
//! )?;
//! # Ok::<(), v_hnsw_embed::EmbedError>(())
//! ```

mod error;
mod fastembed;
mod model;

#[cfg(test)]
mod tests;

pub use crate::error::EmbedError;
pub use crate::fastembed::{FastEmbedModel, ModelType};
pub use crate::model::{EmbeddingModel, Result};
