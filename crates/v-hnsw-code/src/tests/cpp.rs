//! Tests for C++ code chunker.

use crate::{CodeChunkConfig, CppCodeChunker};
use super::fixtures::SAMPLE_CPP;

#[test]
fn cpp_extracts_class() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "GraphNode"), "should extract class GraphNode");
}

#[test]
fn cpp_extracts_methods() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(
        chunks.iter().any(|c| c.name.contains("GraphNode") && c.name.contains("addNeighbor")),
        "should extract class methods with qualified name"
    );
}

#[test]
fn cpp_extracts_free_function() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "buildGraph"), "should extract free function");
}

#[test]
fn cpp_extracts_enum_class() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "NodeType"), "should extract enum class");
}

#[test]
fn cpp_extracts_struct() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "EdgeWeight"), "should extract struct");
}

#[test]
fn cpp_extracts_calls() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    let method = chunks
        .iter()
        .find(|c| c.name.contains("addNeighbor"))
        .expect("should find addNeighbor");

    assert!(
        method.calls.iter().any(|c| c.contains("updateIndex") || c.contains("push_back")),
        "should detect calls in method body, got: {:?}",
        method.calls
    );
}

#[test]
fn cpp_embed_text_and_custom_fields() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    let class = find_chunk!(chunks, "GraphNode");

    let embed = class.to_embed_text("src/graph.cpp", &[]);
    assert!(embed.contains("src/graph.cpp"));

    let custom = class.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn cpp_empty_source_no_chunks() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::EMPTY_SOURCE);
    assert!(chunks.is_empty());
}

#[test]
fn cpp_doc_comment_on_class() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);
    let class = find_chunk!(chunks, "GraphNode");
    assert!(
        class.doc_comment.as_ref().is_some_and(|d| d.contains("graph node")),
        "class should have doc comment, got: {:?}",
        class.doc_comment
    );
}

#[test]
fn cpp_doc_comment_on_free_function() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);
    let func = find_chunk!(chunks, "buildGraph");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("Build a graph")),
        "free function should have doc comment, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn cpp_struct_kind_is_struct() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);
    let s = find_chunk!(chunks, "EdgeWeight");
    assert_eq!(s.kind, crate::CodeNodeKind::Struct);
}

#[test]
fn cpp_enum_class_kind_is_enum() {
    let chunker = CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);
    let e = find_chunk!(chunks, "NodeType");
    assert_eq!(e.kind, crate::CodeNodeKind::Enum);
}

