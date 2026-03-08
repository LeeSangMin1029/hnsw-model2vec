use crate::error::EmbedError;

// ---------------------------------------------------------------------------
// Display / Error message tests
// ---------------------------------------------------------------------------

#[test]
fn test_model_init_error_display() {
    let err = EmbedError::ModelInit("onnx runtime missing".into());
    let msg = err.to_string();
    assert!(msg.contains("model initialization failed"));
    assert!(msg.contains("onnx runtime missing"));
}

#[test]
fn test_embedding_failed_error_display() {
    let err = EmbedError::EmbeddingFailed("tensor shape mismatch".into());
    let msg = err.to_string();
    assert!(msg.contains("embedding generation failed"));
    assert!(msg.contains("tensor shape mismatch"));
}

#[test]
fn test_invalid_input_error_display() {
    let err = EmbedError::InvalidInput("empty input".into());
    let msg = err.to_string();
    assert!(msg.contains("invalid input"));
    assert!(msg.contains("empty input"));
}

#[test]
fn test_download_error_display() {
    let err = EmbedError::Download("404 not found".into());
    let msg = err.to_string();
    assert!(msg.contains("model download failed"));
    assert!(msg.contains("404 not found"));
}

// ---------------------------------------------------------------------------
// Debug trait tests
// ---------------------------------------------------------------------------

#[test]
fn test_model_init_error_debug() {
    let err = EmbedError::ModelInit("bad config".into());
    let debug = format!("{:?}", err);
    assert!(debug.contains("ModelInit"));
    assert!(debug.contains("bad config"));
}

#[test]
fn test_embedding_failed_error_debug() {
    let err = EmbedError::EmbeddingFailed("nan detected".into());
    let debug = format!("{:?}", err);
    assert!(debug.contains("EmbeddingFailed"));
    assert!(debug.contains("nan detected"));
}

#[test]
fn test_invalid_input_error_debug() {
    let err = EmbedError::InvalidInput("too long".into());
    let debug = format!("{:?}", err);
    assert!(debug.contains("InvalidInput"));
    assert!(debug.contains("too long"));
}

#[test]
fn test_download_error_debug() {
    let err = EmbedError::Download("connection refused".into());
    let debug = format!("{:?}", err);
    assert!(debug.contains("Download"));
    assert!(debug.contains("connection refused"));
}

// ---------------------------------------------------------------------------
// std::error::Error trait — source is None for string-wrapped variants
// ---------------------------------------------------------------------------

#[test]
fn test_error_trait_source_is_none() {
    use std::error::Error;
    let variants: Vec<EmbedError> = vec![
        EmbedError::ModelInit("a".into()),
        EmbedError::EmbeddingFailed("b".into()),
        EmbedError::InvalidInput("c".into()),
        EmbedError::Download("d".into()),
    ];
    for err in &variants {
        assert!(err.source().is_none(), "source should be None for {:?}", err);
    }
}

// ---------------------------------------------------------------------------
// Empty message edge case
// ---------------------------------------------------------------------------

#[test]
fn test_empty_message_variants() {
    let cases = vec![
        EmbedError::ModelInit(String::new()),
        EmbedError::EmbeddingFailed(String::new()),
        EmbedError::InvalidInput(String::new()),
        EmbedError::Download(String::new()),
    ];
    for err in &cases {
        // Should not panic, display should still work
        let _ = err.to_string();
        let _ = format!("{:?}", err);
    }
}
