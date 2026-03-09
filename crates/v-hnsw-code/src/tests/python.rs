//! Tests for Python code chunker.

use crate::{CodeChunkConfig, CodeNodeKind, PythonCodeChunker};
use super::fixtures::SAMPLE_PY;


#[test]
fn py_extracts_function() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert_eq!(func.kind, CodeNodeKind::Function);
}

#[test]
fn py_extracts_class() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    assert!(has_chunk!(chunks, "DataResult"), "should extract DataResult class");
    assert!(has_chunk!(chunks, "DataProcessor"), "should extract DataProcessor class");
}

#[test]
fn py_extracts_methods() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    assert!(
        chunks.iter().any(|c| c.name.contains("DataResult") && c.name.contains("is_complete")),
        "should extract methods with qualified name"
    );
}

#[test]
fn py_extracts_calls() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.calls.iter().any(|c| c.contains("validate_items")),
        "should detect validate_items call, got: {:?}",
        func.calls
    );
}

#[test]
fn py_extracts_docstrings() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("Process a list")),
        "should extract docstring, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn py_embed_text_and_custom_fields() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");

    let embed = func.to_embed_text("src/processor.py", &[]);
    assert!(embed.contains("[function]"));
    assert!(embed.contains("src/processor.py"));

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}


#[test]
fn py_extracts_private_function() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    assert!(
        has_chunk!(chunks, "_private_helper"),
        "should extract private helper function"
    );
}

#[test]
fn py_function_has_param_types() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.param_types.iter().any(|(n, _)| n == "items"),
        "should extract items param, got: {:?}",
        func.param_types
    );
}

#[test]
fn py_function_has_return_type() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.return_type.is_some(),
        "should extract return type annotation, got: {:?}",
        func.return_type
    );
}

#[test]
fn py_extracts_imports() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let has_imports = chunks.iter().any(|c| !c.imports.is_empty());
    assert!(has_imports, "Python chunks should include file-level imports");
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn py_empty_source_no_chunks() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::EMPTY_SOURCE);
    assert!(chunks.is_empty());
}

#[test]
fn py_pass_only_function() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_PY_EMPTY_BODY);
    // "def stub_function():\n    pass" is 2 lines, should be included with default min_lines=2
    assert!(
        chunks.iter().any(|c| c.name == "stub_function"),
        "pass-body function should be extracted, got: {:?}",
        chunks.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
}

#[test]
fn py_syntax_error_does_not_panic() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    // Should not panic; tree-sitter handles partial parses
    let chunks = chunker.chunk(super::fixtures::SAMPLE_PY_SYNTAX_ERROR);
    let _ = chunks;
}

#[test]
fn py_nested_class_outer_methods_extracted() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_PY_NESTED_CLASSES);
    let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();

    assert!(
        names.iter().any(|n| n.contains("Outer")),
        "should extract Outer class, got: {names:?}"
    );
    assert!(
        names.iter().any(|n| n.contains("outer_method")),
        "should extract outer_method, got: {names:?}"
    );
}

#[test]
fn py_class_kind_is_class() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);
    let class = find_chunk!(chunks, "DataResult");
    assert_eq!(class.kind, CodeNodeKind::Class);
}

#[test]
fn py_class_docstring_extracted() {
    let chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);
    let class = find_chunk!(chunks, "DataResult");
    assert!(
        class.doc_comment.as_ref().is_some_and(|d| d.contains("Result of data processing")),
        "class docstring should be extracted, got: {:?}",
        class.doc_comment
    );
}

