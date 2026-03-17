//! Cross-language validation, dispatcher, and extension registry tests.

use crate::{
    CodeChunkConfig, CodeChunk, is_supported_code_file, lang_for_extension,
    chunk_for_language,
    TypeScriptCodeChunker, PythonCodeChunker, GoCodeChunker, JavaCodeChunker,
    CCodeChunker, CppCodeChunker,
};
use super::fixtures::*;

#[test]
fn supported_extensions_include_all_languages() {
    let extensions = ["rs", "ts", "tsx", "py", "go", "java", "c", "cpp", "h", "hpp"];
    for ext in extensions {
        assert!(
            is_supported_code_file(ext),
            "extension {ext:?} should be supported"
        );
    }
}

#[test]
fn unsupported_extensions_rejected() {
    let extensions = ["txt", "md", "json", "toml", "yaml", "html", "css"];
    for ext in extensions {
        assert!(
            !is_supported_code_file(ext),
            "extension {ext:?} should NOT be supported"
        );
    }
}

#[test]
fn all_chunks_have_valid_line_ranges() {
    let ts_chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let py_chunker = PythonCodeChunker::new(CodeChunkConfig::default());
    let go_chunker = GoCodeChunker::new(CodeChunkConfig::default());
    let java_chunker = JavaCodeChunker::new(CodeChunkConfig::default());
    let c_chunker = CCodeChunker::new(CodeChunkConfig::default());
    let cpp_chunker = CppCodeChunker::new(CodeChunkConfig::default());

    let all_chunks: Vec<(&str, Vec<CodeChunk>)> = vec![
        ("ts", ts_chunker.chunk(SAMPLE_TS)),
        ("py", py_chunker.chunk(SAMPLE_PY)),
        ("go", go_chunker.chunk(SAMPLE_GO)),
        ("java", java_chunker.chunk(SAMPLE_JAVA)),
        ("c", c_chunker.chunk(SAMPLE_C)),
        ("cpp", cpp_chunker.chunk(SAMPLE_CPP)),
    ];

    for (lang, chunks) in &all_chunks {
        assert!(!chunks.is_empty(), "{lang} should produce chunks");
        for chunk in chunks {
            assert!(
                chunk.end_line >= chunk.start_line,
                "{lang}: {}: end_line ({}) < start_line ({})",
                chunk.name,
                chunk.end_line,
                chunk.start_line
            );
            assert!(
                chunk.end_byte > chunk.start_byte,
                "{lang}: {}: end_byte ({}) <= start_byte ({})",
                chunk.name,
                chunk.end_byte,
                chunk.start_byte
            );
            assert!(
                !chunk.name.is_empty(),
                "{lang}: chunk at line {} has empty name",
                chunk.start_line
            );
            assert!(
                !chunk.text.is_empty(),
                "{lang}: {}: chunk has empty text",
                chunk.name
            );
        }
    }
}

#[test]
fn all_chunkers_set_chunk_index_sequentially() {
    let ts_chunker = TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let py_chunker = PythonCodeChunker::new(CodeChunkConfig::default());

    for (label, chunks) in [
        ("ts", ts_chunker.chunk(SAMPLE_TS)),
        ("py", py_chunker.chunk(SAMPLE_PY)),
    ] {
        let indices: Vec<usize> = chunks.iter().map(|c| c.chunk_index).collect();
        for (i, idx) in indices.iter().enumerate() {
            assert_eq!(
                *idx, i,
                "{label}: chunk_index should be sequential, expected {i} at position {i}, got {idx}"
            );
        }
    }
}


#[test]
fn language_tag_for_extension() {
    // This tests the expected lang_for_extension helper that update.rs should use
    let cases = [
        ("rs", "rust"),
        ("ts", "typescript"),
        ("tsx", "typescript"),
        ("py", "python"),
        ("go", "go"),
        ("java", "java"),
        ("c", "c"),
        ("h", "c"),
        ("cpp", "cpp"),
        ("hpp", "cpp"),
    ];

    for (ext, expected_lang) in cases {
        let lang = lang_for_extension(ext);
        assert_eq!(
            lang,
            Some(expected_lang),
            "extension {ext:?} should map to lang {expected_lang:?}"
        );
    }
}

#[test]
fn lang_for_unknown_extension_returns_none() {
    assert_eq!(lang_for_extension("txt"), None);
    assert_eq!(lang_for_extension("md"), None);
    assert_eq!(lang_for_extension("json"), None);
}

#[test]
fn dispatcher_routes_to_correct_chunker() {
    // Rust should work (already implemented)
    let rs_chunks = chunk_for_language("rs", SAMPLE_RUST_MINI);
    assert!(rs_chunks.is_some(), "rs extension should be supported");
    let rs_chunks = rs_chunks.unwrap();
    assert!(!rs_chunks.is_empty(), "Rust dispatcher should produce chunks");
    assert!(
        rs_chunks.iter().any(|c| c.name == "hello"),
        "should find hello function via dispatcher"
    );
}


#[test]
fn dispatcher_returns_none_for_unsupported() {
    assert!(chunk_for_language("txt", "hello").is_none());
    assert!(chunk_for_language("md", "# heading").is_none());
    assert!(chunk_for_language("json", "{}").is_none());
}

#[test]
fn dispatcher_routes_ts_extension() {
    let chunks = chunk_for_language("ts", SAMPLE_TS);
    assert!(chunks.is_some(), "ts extension should be supported");
    let chunks = chunks.unwrap();
    assert!(!chunks.is_empty(), "TS dispatcher should produce chunks");
}

#[test]
fn dispatcher_routes_tsx_extension() {
    let chunks = chunk_for_language("tsx", SAMPLE_TS);
    assert!(chunks.is_some(), "tsx extension should be supported");
}

#[test]
fn dispatcher_routes_py_extension() {
    let chunks = chunk_for_language("py", SAMPLE_PY);
    assert!(chunks.is_some(), "py extension should be supported");
    let chunks = chunks.unwrap();
    assert!(!chunks.is_empty(), "Python dispatcher should produce chunks");
}

#[test]
fn dispatcher_routes_go_extension() {
    let chunks = chunk_for_language("go", SAMPLE_GO);
    assert!(chunks.is_some(), "go extension should be supported");
    let chunks = chunks.unwrap();
    assert!(!chunks.is_empty(), "Go dispatcher should produce chunks");
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn all_chunkers_handle_empty_source() {
    let empty = "";
    for ext in ["rs", "ts", "py", "go", "java", "c", "cpp"] {
        let chunks = chunk_for_language(ext, empty);
        assert!(
            chunks.is_some(),
            "{ext}: dispatcher should return Some for supported ext even with empty source"
        );
        assert!(
            chunks.unwrap().is_empty(),
            "{ext}: empty source should produce no chunks"
        );
    }
}

#[test]
fn all_chunkers_handle_whitespace_only() {
    let ws = "   \n\n\t  \n";
    for ext in ["rs", "ts", "py", "go", "java", "c", "cpp"] {
        let chunks = chunk_for_language(ext, ws).unwrap_or_default();
        assert!(
            chunks.is_empty(),
            "{ext}: whitespace-only source should produce no chunks"
        );
    }
}

#[test]
fn js_extensions_routed_via_typescript_chunker() {
    let js_src = "function hello() { return 42; }\n";
    for ext in ["js", "jsx", "mjs", "cjs"] {
        let chunks = chunk_for_language(ext, js_src);
        assert!(
            chunks.is_some(),
            "{ext}: should be supported"
        );
    }
}

#[test]
fn pyi_extension_supported() {
    assert!(is_supported_code_file("pyi"), "pyi should be supported");
    assert_eq!(lang_for_extension("pyi"), Some("python"));
}

#[test]
fn cc_cxx_hxx_hh_extensions_supported() {
    for ext in ["cc", "cxx", "hxx", "hh"] {
        assert!(
            is_supported_code_file(ext),
            "{ext} should be supported for C++"
        );
        assert_eq!(
            lang_for_extension(ext),
            Some("cpp"),
            "{ext} should map to cpp"
        );
    }
}

#[test]
fn h_extension_maps_to_c() {
    assert_eq!(lang_for_extension("h"), Some("c"));
}

#[test]
fn code_node_kind_as_str_exhaustive() {
    use crate::CodeNodeKind;
    let kinds = [
        (CodeNodeKind::Function, "function"),
        (CodeNodeKind::Struct, "struct"),
        (CodeNodeKind::Enum, "enum"),
        (CodeNodeKind::Impl, "impl"),
        (CodeNodeKind::Trait, "trait"),
        (CodeNodeKind::TypeAlias, "type_alias"),
        (CodeNodeKind::Const, "const"),
        (CodeNodeKind::Static, "static"),
        (CodeNodeKind::Module, "module"),
        (CodeNodeKind::MacroDefinition, "macro"),
        (CodeNodeKind::Class, "class"),
        (CodeNodeKind::Interface, "interface"),
    ];
    for (kind, expected) in kinds {
        assert_eq!(kind.as_str(), expected, "{kind:?} should map to {expected:?}");
    }
}

#[test]
fn embed_text_no_visibility_no_extra_space() {
    use crate::{CodeChunk, CodeNodeKind};
    let chunk = CodeChunk {
        text: "fn foo() {}".to_owned(),
        kind: CodeNodeKind::Function,
        name: "foo".to_owned(),
        signature: None,
        doc_comment: None,
        visibility: String::new(),
        start_line: 0,
        end_line: 0,
        start_byte: 0,
        end_byte: 11,
        chunk_index: 0,
        imports: vec![],
        calls: vec![],
        call_lines: vec![],
        type_refs: vec![],
        param_types: vec![],
        field_types: vec![],
        local_types: vec![],
        let_call_bindings: vec![],
        field_accesses: vec![],
        return_type: None, ast_hash: 0, body_hash: 0, sub_blocks: vec![], string_args: vec![], param_flows: vec![],
    };
    let embed = chunk.to_embed_text("test.rs", &[]);
    // "[function] foo" — no double space before name
    assert!(
        embed.starts_with("[function] foo"),
        "should not have extra space for empty visibility, got: {embed}"
    );
}

#[test]
fn embed_text_with_visibility_prefix() {
    use crate::{CodeChunk, CodeNodeKind};
    let chunk = CodeChunk {
        text: "pub fn bar() {}".to_owned(),
        kind: CodeNodeKind::Function,
        name: "bar".to_owned(),
        signature: None,
        doc_comment: None,
        visibility: "pub".to_owned(),
        start_line: 0,
        end_line: 0,
        start_byte: 0,
        end_byte: 15,
        chunk_index: 0,
        imports: vec![],
        calls: vec![],
        call_lines: vec![],
        type_refs: vec![],
        param_types: vec![],
        field_types: vec![],
        local_types: vec![],
        let_call_bindings: vec![],
        field_accesses: vec![],
        return_type: None, ast_hash: 0, body_hash: 0, sub_blocks: vec![], string_args: vec![], param_flows: vec![],
    };
    let embed = chunk.to_embed_text("test.rs", &[]);
    assert!(
        embed.starts_with("[function] pub bar"),
        "should include visibility in embed text, got: {embed}"
    );
}

#[test]
fn custom_fields_empty_collections_omitted() {
    use crate::{CodeChunk, CodeNodeKind};
    let chunk = CodeChunk {
        text: "fn foo() {}".to_owned(),
        kind: CodeNodeKind::Function,
        name: "foo".to_owned(),
        signature: None,
        doc_comment: None,
        visibility: String::new(),
        start_line: 0,
        end_line: 0,
        start_byte: 0,
        end_byte: 11,
        chunk_index: 0,
        imports: vec![],
        calls: vec![],
        call_lines: vec![],
        type_refs: vec![],
        param_types: vec![],
        field_types: vec![],
        local_types: vec![],
        let_call_bindings: vec![],
        field_accesses: vec![],
        return_type: None, ast_hash: 0, body_hash: 0, sub_blocks: vec![], string_args: vec![], param_flows: vec![],
    };
    let custom = chunk.to_custom_fields(&[]);
    assert!(!custom.contains_key("calls"), "empty calls should not be in custom fields");
    assert!(!custom.contains_key("imports"), "empty imports should not be in custom fields");
    assert!(!custom.contains_key("type_refs"), "empty type_refs should not be in custom fields");
    assert!(!custom.contains_key("called_by"), "empty called_by should not be in custom fields");
    assert!(!custom.contains_key("signature"), "None signature should not be in custom fields");
    assert!(!custom.contains_key("doc"), "None doc should not be in custom fields");
    assert!(!custom.contains_key("return_type"), "None return_type should not be in custom fields");
}

#[test]
fn sub_blocks_populated_for_function_with_control_flow() {
    let code = r#"
fn process(x: i32) -> bool {
    if x > 0 {
        let y = x * 2;
        println!("{}", y);
        return true;
    }
    for i in 0..x {
        println!("{}", i);
        if i > 5 {
            break;
        }
    }
    false
}
"#;
    let chunks = chunk_for_language("rs", code).unwrap();
    assert_eq!(chunks.len(), 1, "should extract one function");
    let sub = &chunks[0].sub_blocks;
    assert!(
        sub.len() >= 2,
        "function with if+for should have at least 2 sub-blocks, got {}",
        sub.len()
    );
    // All sub-blocks should have non-zero AST hashes
    for sb in sub {
        assert_ne!(sb.ast_hash, 0, "sub-block AST hash should not be zero");
    }
}

#[test]
fn sub_blocks_empty_for_simple_function() {
    let code = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
    let chunks = chunk_for_language("rs", code).unwrap();
    assert_eq!(chunks.len(), 1);
    assert!(
        chunks[0].sub_blocks.is_empty(),
        "simple function without control flow should have no sub-blocks"
    );
}

#[test]
fn sub_block_hashes_stored_in_custom_fields() {
    let code = r#"
fn example(x: i32) {
    if x > 0 {
        let y = x * 2;
        println!("{}", y);
    }
}
"#;
    let chunks = chunk_for_language("rs", code).unwrap();
    assert_eq!(chunks.len(), 1);
    let chunk = &chunks[0];
    assert!(!chunk.sub_blocks.is_empty(), "should have sub-blocks");

    let custom = chunk.to_custom_fields(&[]);
    assert!(
        custom.contains_key("sub_block_hashes"),
        "custom fields should contain sub_block_hashes"
    );
}
