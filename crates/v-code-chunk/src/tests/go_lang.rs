//! Tests for Go code chunker.

use crate::{CodeChunkConfig, CodeNodeKind, GoCodeChunker};
use super::fixtures::SAMPLE_GO;


#[test]
fn go_extracts_function() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert_eq!(func.kind, CodeNodeKind::Function);
}

#[test]
fn go_extracts_struct() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    assert!(has_chunk!(chunks, "RequestData"), "should extract RequestData struct");
}

#[test]
fn go_extracts_methods() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    assert!(
        chunks.iter().any(|c| c.name.contains("RequestData") && c.name.contains("String")),
        "should extract method with receiver-qualified name"
    );
}

#[test]
fn go_extracts_interface() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    assert!(has_chunk!(chunks, "Handler"), "should extract interface Handler");
}

#[test]
fn go_extracts_calls() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.calls.iter().any(|c| c.contains("ParseBody") || c.contains("Validate")),
        "should detect function calls, got: {:?}",
        func.calls
    );
}

#[test]
fn go_extracts_doc_comments() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("processes an incoming")),
        "should extract Go doc comment, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn go_embed_text_and_custom_fields() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");

    let embed = func.to_embed_text("pkg/handler.go", &[]);
    assert!(embed.contains("[function]"));
    assert!(embed.contains("pkg/handler.go"));

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}


#[test]
fn go_extracts_imports() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let has_imports = chunks.iter().any(|c| !c.imports.is_empty());
    assert!(has_imports, "Go chunks should include file-level imports");
}

#[test]
fn go_function_has_param_types() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.param_types.iter().any(|(n, _)| n == "w" || n == "r"),
        "should extract Go params, got: {:?}",
        func.param_types
    );
}

#[test]
fn go_function_has_return_type() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.return_type.as_ref().is_some_and(|r| r.contains("error")),
        "should extract error return type, got: {:?}",
        func.return_type
    );
}

#[test]
fn go_extracts_type_alias() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    // `type Status int` is a type declaration
    assert!(
        chunks.iter().any(|c| c.name == "Status"),
        "should extract type alias Status"
    );
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn go_empty_source_no_chunks() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::EMPTY_SOURCE);
    assert!(chunks.is_empty());
}

#[test]
fn go_empty_struct() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_GO_EMPTY_STRUCT);
    // Should not panic, and should at least find the function
    assert!(
        chunks.iter().any(|c| c.name == "NewEmpty"),
        "should extract NewEmpty function, got: {:?}",
        chunks.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
}

#[test]
fn go_exported_function_has_pub_visibility() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);
    let func = find_chunk!(chunks, "HandleRequest");
    assert_eq!(func.visibility, "pub", "exported Go func should have pub visibility");
}

#[test]
fn go_method_qualified_name_includes_receiver() {
    let chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);
    let method = chunks
        .iter()
        .find(|c| c.name.contains("Validate") && c.name.contains("RequestData"))
        .expect("should find receiver-qualified Validate");
    assert!(method.name.contains("RequestData"));
}
