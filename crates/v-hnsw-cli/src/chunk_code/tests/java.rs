//! Tests for Java code chunker.

use crate::chunk_code::{CodeChunkConfig, JavaCodeChunker};
use super::fixtures::SAMPLE_JAVA;


#[test]
fn java_extracts_class() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(has_chunk!(chunks, "PaymentService"), "should extract PaymentService class");
    assert!(has_chunk!(chunks, "PaymentResult"), "should extract PaymentResult class");
}

#[test]
fn java_extracts_methods() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(
        chunks.iter().any(|c| c.name.contains("PaymentService") && c.name.contains("processPayment")),
        "should extract methods with qualified name"
    );
}

#[test]
fn java_extracts_interface() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(has_chunk!(chunks, "PaymentGateway"), "should extract interface");
}

#[test]
fn java_extracts_enum() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(has_chunk!(chunks, "PaymentStatus"), "should extract enum");
}

#[test]
fn java_extracts_calls() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    let method = chunks
        .iter()
        .find(|c| c.name.contains("processPayment"))
        .expect("should find processPayment");

    assert!(
        method.calls.iter().any(|c| c.contains("validate")),
        "should detect validate call, got: {:?}",
        method.calls
    );
}

#[test]
fn java_extracts_doc_comments() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    let class = find_chunk!(chunks, "PaymentService");
    assert!(
        class.doc_comment.as_ref().is_some_and(|d| d.contains("payment transactions")),
        "should extract Javadoc comment, got: {:?}",
        class.doc_comment
    );
}

#[test]
fn java_embed_text_and_custom_fields() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    let class = find_chunk!(chunks, "PaymentService");

    let embed = class.to_embed_text("src/PaymentService.java", &[]);
    assert!(embed.contains("src/PaymentService.java"));

    let custom = class.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn java_empty_source_no_chunks() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::EMPTY_SOURCE);
    assert!(chunks.is_empty());
}

#[test]
fn java_abstract_class_extracted() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_JAVA_ABSTRACT);
    assert!(
        chunks.iter().any(|c| c.name == "AbstractProcessor"),
        "should extract abstract class, got: {:?}",
        chunks.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
}

#[test]
fn java_public_visibility_extracted() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);
    let class = find_chunk!(chunks, "PaymentService");
    assert_eq!(class.visibility, "public");
}

#[test]
fn java_private_method_visibility() {
    let chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);
    let method = chunks.iter().find(|c| c.name.contains("validate"));
    if let Some(m) = method {
        assert_eq!(m.visibility, "private", "validate should be private");
    }
}

