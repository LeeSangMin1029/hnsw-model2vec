//! Embedding model integration for v-hnsw.
//!
//! This crate provides text embedding capabilities using model2vec,
//! enabling semantic similarity search with the v-hnsw vector index.
//!
//! # Quick Start
//!
//! ```no_run
//! use v_hnsw_embed::{Model2VecModel, EmbeddingModel};
//!
//! // Create a model with the default model (potion-multilingual-128M)
//! let model = Model2VecModel::new()?;
//!
//! // Or specify a model explicitly
//! let model = Model2VecModel::from_pretrained("minishlab/potion-base-8M")?;
//!
//! // Embed documents
//! let docs = &["Document one", "Document two"];
//! let embeddings = model.embed(docs)?;
//!
//! // Embed a query
//! let query_embedding = model.embed_query("search query")?;
//!
//! // Use with v-hnsw index
//! println!("Model: {}", model.name());
//! println!("Dimension: {}", model.dim());
//! # Ok::<(), v_hnsw_embed::EmbedError>(())
//! ```
//!
//! # Model Download
//!
//! Models are downloaded from Hugging Face Hub on first use.
//! Models are cached locally for subsequent use.

mod error;
mod model;
mod model2vec;

pub use crate::error::EmbedError;
pub use crate::model::{EmbeddingModel, Result};
pub use crate::model2vec::Model2VecModel;
