use crate::extract;
use crate::types::CodeChunk;

use crate::{
    CCodeChunker, CppCodeChunker, GoCodeChunker, JavaCodeChunker, JavaScriptCodeChunker,
    PythonCodeChunker, RustCodeChunker, TypeScriptCodeChunker,
};
use crate::types::CodeChunkConfig;

/// Resolved language info for a file extension.
struct LangInfo {
    name: &'static str,
    ts_language: tree_sitter::Language,
    chunk_fn: fn(CodeChunkConfig, &str) -> Vec<CodeChunk>,
}

fn chunk_rust(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { RustCodeChunker::new(config).chunk(src) }
fn chunk_ts(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { TypeScriptCodeChunker::new(config).chunk(src) }
fn chunk_js(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { JavaScriptCodeChunker::new(config).chunk(src) }
fn chunk_py(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { PythonCodeChunker::new(config).chunk(src) }
fn chunk_go(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { GoCodeChunker::new(config).chunk(src) }
fn chunk_java(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { JavaCodeChunker::new(config).chunk(src) }
fn chunk_c(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { CCodeChunker::new(config).chunk(src) }
fn chunk_cpp(config: CodeChunkConfig, src: &str) -> Vec<CodeChunk> { CppCodeChunker::new(config).chunk(src) }

/// Single source of truth: extension → language info.
fn resolve_ext(ext: &str) -> Option<LangInfo> {
    match ext {
        "rs" => Some(LangInfo {
            name: "rust",
            ts_language: tree_sitter_rust::LANGUAGE.into(),
            chunk_fn: chunk_rust,
        }),
        "ts" | "tsx" => Some(LangInfo {
            name: "typescript",
            ts_language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            chunk_fn: chunk_ts,
        }),
        "js" | "jsx" | "mjs" | "cjs" => Some(LangInfo {
            name: "javascript",
            ts_language: tree_sitter_typescript::LANGUAGE_TSX.into(),
            chunk_fn: chunk_js,
        }),
        "py" | "pyi" => Some(LangInfo {
            name: "python",
            ts_language: tree_sitter_python::LANGUAGE.into(),
            chunk_fn: chunk_py,
        }),
        "go" => Some(LangInfo {
            name: "go",
            ts_language: tree_sitter_go::LANGUAGE.into(),
            chunk_fn: chunk_go,
        }),
        "java" => Some(LangInfo {
            name: "java",
            ts_language: tree_sitter_java::LANGUAGE.into(),
            chunk_fn: chunk_java,
        }),
        "c" | "h" => Some(LangInfo {
            name: "c",
            ts_language: tree_sitter_c::LANGUAGE.into(),
            chunk_fn: chunk_c,
        }),
        "cpp" | "hpp" | "cc" | "cxx" | "hxx" | "hh" => Some(LangInfo {
            name: "cpp",
            ts_language: tree_sitter_cpp::LANGUAGE.into(),
            chunk_fn: chunk_cpp,
        }),
        _ => None,
    }
}

/// Check if a file extension is a supported code file.
pub fn is_supported_code_file(ext: &str) -> bool {
    resolve_ext(ext).is_some()
}

/// Map a file extension to a language name for tagging.
pub fn lang_for_extension(ext: &str) -> Option<&'static str> {
    resolve_ext(ext).map(|info| info.name)
}

/// Dispatch to the appropriate language chunker based on file extension.
///
/// Returns `None` for unsupported extensions.
pub fn chunk_for_language(ext: &str, source: &str) -> Option<Vec<CodeChunk>> {
    let info = resolve_ext(ext)?;
    let config = CodeChunkConfig::default();
    let mut chunks = (info.chunk_fn)(config, source);

    // Post-process: compute AST structural hashes for clone detection.
    fill_ast_hashes(info.ts_language, source, &mut chunks);

    Some(chunks)
}

/// Parse the full source once and fill `ast_hash` for each chunk by byte range.
fn fill_ast_hashes(language: tree_sitter::Language, source: &str, chunks: &mut [CodeChunk]) {
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&language).is_err() {
        return;
    }
    let Some(tree) = parser.parse(source, None) else {
        return;
    };

    let src = source.as_bytes();
    for chunk in chunks {
        // Find the deepest node that spans this chunk's byte range
        let node = tree
            .root_node()
            .descendant_for_byte_range(chunk.start_byte, chunk.end_byte.saturating_sub(1));
        if let Some(node) = node {
            chunk.ast_hash = extract::ast_structure_hash(&node, src);
            // Split into sub-blocks for fine-grained clone detection
            chunk.sub_blocks = extract::split_into_blocks(&node, src);
        }
        chunk.body_hash = extract::body_hash(&chunk.text);
    }
}
