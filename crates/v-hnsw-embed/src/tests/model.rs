use crate::model::{EmbeddingModel, Result};
use crate::error::EmbedError;

/// Mock embedding model for testing the EmbeddingModel trait.
struct MockModel {
    dimension: usize,
    model_name: String,
}

impl MockModel {
    fn new(dim: usize, name: &str) -> Self {
        Self {
            dimension: dim,
            model_name: name.to_string(),
        }
    }
}

impl EmbeddingModel for MockModel {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(EmbedError::InvalidInput("empty input".into()));
        }
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32 / 10.0; self.dimension])
            .collect())
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(EmbedError::InvalidInput("empty query".into()));
        }
        Ok(vec![query.len() as f32 / 10.0; self.dimension])
    }

    fn dim(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}

#[test]
fn mock_dim_returns_configured_value() {
    let model = MockModel::new(128, "test-model");
    assert_eq!(model.dim(), 128);

    let model = MockModel::new(256, "big-model");
    assert_eq!(model.dim(), 256);
}

#[test]
fn mock_name_returns_configured_value() {
    let model = MockModel::new(64, "my-model");
    assert_eq!(model.name(), "my-model");
}

#[test]
fn embed_single_text() {
    let model = MockModel::new(4, "test");
    let result = model.embed(&["hello"]).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 4);
    // "hello" has 5 chars → 5/10 = 0.5
    assert!((result[0][0] - 0.5).abs() < 1e-6);
}

#[test]
fn embed_multiple_texts() {
    let model = MockModel::new(3, "test");
    let result = model.embed(&["hi", "hello", "world!"]).unwrap();
    assert_eq!(result.len(), 3);
    for emb in &result {
        assert_eq!(emb.len(), 3);
    }
    // "hi" = 2 chars, "hello" = 5, "world!" = 6
    assert!((result[0][0] - 0.2).abs() < 1e-6);
    assert!((result[1][0] - 0.5).abs() < 1e-6);
    assert!((result[2][0] - 0.6).abs() < 1e-6);
}

#[test]
fn embed_empty_input_returns_error() {
    let model = MockModel::new(4, "test");
    let result = model.embed(&[]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("empty input"));
}

#[test]
fn embed_query_returns_correct_dim() {
    let model = MockModel::new(8, "test");
    let result = model.embed_query("test query").unwrap();
    assert_eq!(result.len(), 8);
    // "test query" = 10 chars → 10/10 = 1.0
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn embed_query_empty_returns_error() {
    let model = MockModel::new(4, "test");
    let result = model.embed_query("");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("empty query"));
}

#[test]
fn result_type_ok_variant() {
    let ok: Result<i32> = Ok(42);
    assert_eq!(ok.unwrap(), 42);
}

#[test]
fn result_type_err_variant() {
    let err: Result<i32> = Err(EmbedError::ModelInit("test error".into()));
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("test error"));
}

#[test]
fn mock_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MockModel>();
}

#[test]
fn trait_object_works() {
    let model: Box<dyn EmbeddingModel> = Box::new(MockModel::new(16, "boxed"));
    assert_eq!(model.dim(), 16);
    assert_eq!(model.name(), "boxed");
    let emb = model.embed(&["x"]).unwrap();
    assert_eq!(emb.len(), 1);
    assert_eq!(emb[0].len(), 16);
}
