//! Tests for C code chunker.

use crate::chunk_code::{CodeChunkConfig, CodeNodeKind, CCodeChunker};
use super::fixtures::SAMPLE_C;

#[test]
fn c_extracts_function() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");
    assert_eq!(func.kind, CodeNodeKind::Function);
}

#[test]
fn c_extracts_struct() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    assert!(
        chunks.iter().any(|c| c.name == "Point" && c.kind == CodeNodeKind::Struct),
        "should extract typedef struct as named struct"
    );
}

#[test]
fn c_extracts_enum() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    assert!(has_chunk!(chunks, "Color"), "should extract enum Color");
}

#[test]
fn c_extracts_calls() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");
    assert!(
        func.calls.iter().any(|c| c.contains("validate_input")),
        "should detect validate_input call, got: {:?}",
        func.calls
    );
}

#[test]
fn c_extracts_doc_comments() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("data buffer")),
        "should extract C block comment as doc, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn c_embed_text_and_custom_fields() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");

    let embed = func.to_embed_text("src/buffer.c", &[]);
    assert!(embed.contains("[function]"));
    assert!(embed.contains("src/buffer.c"));

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn c_empty_source_no_chunks() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::EMPTY_SOURCE);
    assert!(chunks.is_empty());
}

#[test]
fn c_forward_declaration_handled() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_C_FORWARD_DECL);
    // Should not panic; forward declarations may or may not produce chunks
    // but the function should be extractable
    assert!(
        chunks.iter().any(|c| c.name == "use_forward"),
        "should extract use_forward function, got: {:?}",
        chunks.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
}

#[test]
fn c_static_function_extracted() {
    let chunker = CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);
    assert!(
        chunks.iter().any(|c| c.name == "helper_func"),
        "should extract static helper_func"
    );
}

