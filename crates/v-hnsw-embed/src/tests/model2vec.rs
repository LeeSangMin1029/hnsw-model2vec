use crate::error::EmbedError;
use crate::model::EmbeddingModel;
use crate::model2vec::Model2VecModel;

/// Cosine similarity helper.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

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

// ---------------------------------------------------------------------------
// Determinism — same input must produce identical embeddings
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_determinism() {
    let model = Model2VecModel::new().unwrap();
    let text = "deterministic embedding test";
    let emb1 = model.embed(&[text]).unwrap();
    let emb2 = model.embed(&[text]).unwrap();
    assert_eq!(emb1[0], emb2[0], "same input must produce identical embeddings");
}

#[test]
fn test_embed_query_determinism() {
    let model = Model2VecModel::new().unwrap();
    let query = "deterministic query";
    let emb1 = model.embed_query(query).unwrap();
    let emb2 = model.embed_query(query).unwrap();
    assert_eq!(emb1, emb2, "same query must produce identical embeddings");
}

// ---------------------------------------------------------------------------
// Similarity — related texts should be closer than unrelated
// ---------------------------------------------------------------------------

#[test]
fn test_related_texts_higher_similarity() {
    let model = Model2VecModel::new().unwrap();
    let texts = &["rust programming language", "cargo build system", "chocolate cake recipe"];
    let embeddings = model.embed(texts).unwrap();

    let sim_related = cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_unrelated = cosine_similarity(&embeddings[0], &embeddings[2]);

    assert!(
        sim_related > sim_unrelated,
        "related texts ({sim_related:.4}) should have higher similarity than unrelated ({sim_unrelated:.4})"
    );
}

// ---------------------------------------------------------------------------
// Long text embedding
// ---------------------------------------------------------------------------

#[test]
fn test_long_text_embedding() {
    let model = Model2VecModel::new().unwrap();
    let long_text = "word ".repeat(500);
    let result = model.embed(&[long_text.as_str()]);
    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 256);
}

// ---------------------------------------------------------------------------
// Special characters and Unicode
// ---------------------------------------------------------------------------

#[test]
fn test_special_characters() {
    let model = Model2VecModel::new().unwrap();
    let texts = &[
        "hello! @#$%^&*() special chars",
        "tabs\tand\nnewlines",
        "emoji: 🦀🚀",
        "math: ∑∫∂ε",
    ];
    let result = model.embed(texts);
    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 4);
    for emb in &embeddings {
        assert_eq!(emb.len(), 256);
    }
}

#[test]
fn test_unicode_multilingual() {
    let model = Model2VecModel::new().unwrap();
    let texts = &[
        "한국어 텍스트 테스트",
        "日本語テスト",
        "中文测试文本",
        "Ελληνικά κείμενο",
        "العربية نص",
    ];
    let result = model.embed(texts);
    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 5);
    for emb in &embeddings {
        assert_eq!(emb.len(), 256);
    }
}

// ---------------------------------------------------------------------------
// Embedding vectors should be non-zero
// ---------------------------------------------------------------------------

#[test]
fn test_embedding_is_nonzero() {
    let model = Model2VecModel::new().unwrap();
    let emb = model.embed_query("hello").unwrap();
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 0.0, "embedding vector should be non-zero");
}
