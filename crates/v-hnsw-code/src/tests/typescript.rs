//! Tests for TypeScript code chunker.

use crate::{CodeChunkConfig, CodeNodeKind, TypeScriptCodeChunker};
use super::fixtures::SAMPLE_TS;


#[test]
fn ts_extracts_function() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert_eq!(func.kind, CodeNodeKind::Function);
    assert_eq!(func.visibility, "export");
}

#[test]
fn ts_extracts_interface() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    assert!(has_chunk!(chunks, "UserDTO"), "should extract interface UserDTO");
}

#[test]
fn ts_extracts_class_and_methods() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    assert!(has_chunk!(chunks, "UserController"), "should extract class");
    assert!(
        chunks.iter().any(|c| c.name.contains("UserController") && c.name.contains("getUser")),
        "should extract class methods with qualified name"
    );
}

#[test]
fn ts_extracts_enum() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    assert!(has_chunk!(chunks, "Role"), "should extract enum Role");
}

#[test]
fn ts_extracts_calls() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.calls.iter().any(|c| c.contains("validate")),
        "should detect validate call, got: {:?}",
        func.calls
    );
}

#[test]
fn ts_extracts_doc_comments() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("incoming HTTP request")),
        "should extract JSDoc comment, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn ts_embed_text_and_custom_fields() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");

    let embed = func.to_embed_text("src/controller.ts", &[]);
    assert!(embed.contains("[function]"), "embed text should include kind");
    assert!(embed.contains("src/controller.ts"), "embed text should include file path");

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

#[test]
fn ts_extracts_imports() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    // At least one chunk should have imports populated
    let has_imports = chunks.iter().any(|c| !c.imports.is_empty());
    assert!(has_imports, "TS chunks should include file-level imports");
}

#[test]
fn ts_function_has_param_types() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.param_types.iter().any(|(n, _)| n == "req"),
        "should extract req param, got: {:?}",
        func.param_types
    );
}

#[test]
fn ts_function_has_return_type() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.return_type.as_ref().is_some_and(|r| r.contains("Promise")),
        "should extract Promise return type, got: {:?}",
        func.return_type
    );
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn ts_empty_source_no_chunks() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::EMPTY_SOURCE);
    assert!(chunks.is_empty());
}

#[test]
fn ts_empty_class_extracted() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_TS_EMPTY_CLASS);
    // Empty class is a single line, may be excluded by min_lines
    // With default min_lines=2, "export class EmptyClass {}" is 1 line
    // This tests the boundary: empty class should be handled gracefully
    let _ = chunks; // should not panic
}

#[test]
fn ts_arrow_function_syntax_no_panic() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_TS_ARROW_FUNCTIONS);
    // Arrow functions may or may not be extracted depending on tree-sitter grammar
    // but should not panic
    let _ = chunks;
}

#[test]
fn ts_class_kind_is_class() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);
    let class = find_chunk!(chunks, "UserController");
    assert_eq!(class.kind, CodeNodeKind::Class);
}

#[test]
fn ts_interface_kind_is_interface() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);
    let iface = find_chunk!(chunks, "UserDTO");
    assert_eq!(iface.kind, CodeNodeKind::Interface);
}

#[test]
fn ts_enum_kind_is_enum() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);
    let e = find_chunk!(chunks, "Role");
    assert_eq!(e.kind, CodeNodeKind::Enum);
}

#[test]
fn ts_class_method_has_doc_comment() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let method = chunks
        .iter()
        .find(|c| c.name.contains("getUser"))
        .expect("should find getUser method");
    assert!(
        method.doc_comment.as_ref().is_some_and(|d| d.contains("Get a user")),
        "method should have JSDoc, got: {:?}",
        method.doc_comment
    );
}

#[test]
fn ts_class_visibility_is_export() {
    let chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);
    let class = find_chunk!(chunks, "UserController");
    assert_eq!(class.visibility, "export");
}

