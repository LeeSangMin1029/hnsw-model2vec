//! Tests for v-hnsw-embed crate.
//!
//! Note: Tests that require model download are marked with `#[ignore]`.
//! Run them with: `cargo test -p v-hnsw-embed -- --ignored`

use crate::{EmbedError, EmbeddingModel, FastEmbedModel, ModelType};

/// Test that model type dimensions are correct.
#[test]
fn model_type_dimensions() {
    assert_eq!(ModelType::BGESmallENV15.dimension(), 384);
    assert_eq!(ModelType::BGEBaseENV15.dimension(), 768);
    assert_eq!(ModelType::BGELargeENV15.dimension(), 1024);
    assert_eq!(ModelType::AllMiniLML6V2.dimension(), 384);
    assert_eq!(ModelType::AllMiniLML12V2.dimension(), 384);
    assert_eq!(ModelType::MultilingualE5Small.dimension(), 384);
    assert_eq!(ModelType::MultilingualE5Base.dimension(), 768);
    assert_eq!(ModelType::MultilingualE5Large.dimension(), 1024);
}

/// Test that model names are correct.
#[test]
fn model_type_names() {
    assert_eq!(
        ModelType::AllMiniLML6V2.model_name(),
        "sentence-transformers/all-MiniLM-L6-v2"
    );
    assert_eq!(ModelType::BGESmallENV15.model_name(), "BAAI/bge-small-en-v1.5");
    assert_eq!(ModelType::BGEBaseENV15.model_name(), "BAAI/bge-base-en-v1.5");
    assert_eq!(ModelType::BGELargeENV15.model_name(), "BAAI/bge-large-en-v1.5");
    assert_eq!(
        ModelType::MultilingualE5Small.model_name(),
        "intfloat/multilingual-e5-small"
    );
}

/// Test default model type.
#[test]
fn default_model_type() {
    assert_eq!(ModelType::default(), ModelType::AllMiniLML6V2);
}

/// Test model loading with default settings.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn model_loading_default() {
    let model = FastEmbedModel::try_new();
    assert!(model.is_ok(), "Failed to load default model: {:?}", model.err());

    let model = model.ok();
    assert!(model.is_some());
    let model = model.as_ref();
    assert!(model.is_some());
    let model = model.map(|m| m);
    assert!(model.is_some());
}

/// Test model loading with specific model type.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn model_loading_specific() {
    let model = FastEmbedModel::with_model(ModelType::AllMiniLML6V2);
    assert!(model.is_ok(), "Failed to load model: {:?}", model.err());
}

/// Test basic embedding generation.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn basic_embedding_generation() {
    let model = FastEmbedModel::try_new();
    assert!(model.is_ok());

    let model = match model {
        Ok(m) => m,
        Err(_) => return, // Skip if model fails to load
    };

    let texts = &["Hello, world!", "This is a test."];
    let embeddings = model.embed(texts);
    assert!(embeddings.is_ok(), "Embedding failed: {:?}", embeddings.err());

    let embeddings = match embeddings {
        Ok(e) => e,
        Err(_) => return,
    };

    // Check we got the right number of embeddings
    assert_eq!(embeddings.len(), 2);

    // Check embedding dimensions
    for emb in &embeddings {
        assert_eq!(emb.len(), model.dim());
    }
}

/// Test query embedding.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn query_embedding() {
    let model = FastEmbedModel::try_new();
    assert!(model.is_ok());

    let model = match model {
        Ok(m) => m,
        Err(_) => return,
    };

    let query_vec = model.embed_query("search query");
    assert!(query_vec.is_ok(), "Query embedding failed: {:?}", query_vec.err());

    let query_vec = match query_vec {
        Ok(v) => v,
        Err(_) => return,
    };

    assert_eq!(query_vec.len(), model.dim());
}

/// Test dimension check matches actual embedding.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn dimension_check() {
    let model = FastEmbedModel::with_model(ModelType::AllMiniLML6V2);
    assert!(model.is_ok());

    let model = match model {
        Ok(m) => m,
        Err(_) => return,
    };

    assert_eq!(model.dim(), 384);
    assert_eq!(model.model_type().dimension(), 384);

    let emb = model.embed(&["test"]);
    assert!(emb.is_ok());

    let emb = match emb {
        Ok(e) => e,
        Err(_) => return,
    };

    if let Some(first) = emb.first() {
        assert_eq!(first.len(), 384);
    }
}

/// Test empty input handling.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn empty_input_handling() {
    let model = FastEmbedModel::try_new();
    let model = match model {
        Ok(m) => m,
        Err(_) => return,
    };

    // Empty slice should return error
    let result = model.embed(&[]);
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(matches!(e, EmbedError::InvalidInput(_)));
    }

    // Empty query should return error
    let result = model.embed_query("");
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(matches!(e, EmbedError::InvalidInput(_)));
    }
}

/// Test model name getter.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn model_name_getter() {
    let model = FastEmbedModel::with_model(ModelType::BGESmallENV15);
    let model = match model {
        Ok(m) => m,
        Err(_) => return,
    };

    assert_eq!(model.name(), "BAAI/bge-small-en-v1.5");
    assert_eq!(model.model_type(), ModelType::BGESmallENV15);
}

/// Test embedding consistency - same input should produce same output.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn embedding_consistency() {
    let model = FastEmbedModel::try_new();
    let model = match model {
        Ok(m) => m,
        Err(_) => return,
    };

    let text = "consistent embedding test";
    let emb1 = model.embed(&[text]);
    let emb2 = model.embed(&[text]);

    let emb1 = match emb1 {
        Ok(e) => e,
        Err(_) => return,
    };

    let emb2 = match emb2 {
        Ok(e) => e,
        Err(_) => return,
    };

    // Embeddings should be identical for same input
    if let (Some(v1), Some(v2)) = (emb1.first(), emb2.first()) {
        assert_eq!(v1.len(), v2.len());
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6, "Embeddings differ: {} vs {}", a, b);
        }
    }
}

/// Test batch embedding produces same results as individual embeddings.
/// This test is ignored by default as it downloads the model.
#[test]
#[ignore]
fn batch_vs_individual_embedding() {
    let model = FastEmbedModel::try_new();
    let model = match model {
        Ok(m) => m,
        Err(_) => return,
    };

    let texts = &["first text", "second text"];

    // Batch embedding
    let batch_result = model.embed(texts);
    let batch_embs = match batch_result {
        Ok(e) => e,
        Err(_) => return,
    };

    // Individual embeddings
    let ind1 = model.embed(&[texts[0]]);
    let ind2 = model.embed(&[texts[1]]);

    let ind1 = match ind1 {
        Ok(e) => e,
        Err(_) => return,
    };

    let ind2 = match ind2 {
        Ok(e) => e,
        Err(_) => return,
    };

    // Compare batch vs individual results
    if let (Some(b0), Some(i0)) = (batch_embs.first(), ind1.first()) {
        for (a, b) in b0.iter().zip(i0.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    if let (Some(b1), Some(i1)) = (batch_embs.get(1), ind2.first()) {
        for (a, b) in b1.iter().zip(i1.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
