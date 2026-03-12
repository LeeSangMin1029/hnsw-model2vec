use crate::extract;
use crate::types::CodeChunk;

use crate::{
    CCodeChunker, CppCodeChunker, GoCodeChunker, JavaCodeChunker, JavaScriptCodeChunker,
    PythonCodeChunker, RustCodeChunker, TypeScriptCodeChunker,
};
use crate::types::CodeChunkConfig;

/// Check if a file extension is a supported code file.
pub fn is_supported_code_file(ext: &str) -> bool {
    lang_for_extension(ext).is_some()
}

/// Map a file extension to a language name for tagging.
pub fn lang_for_extension(ext: &str) -> Option<&'static str> {
    match ext {
        "rs" => Some("rust"),
        "ts" | "tsx" => Some("typescript"),
        "js" | "jsx" | "mjs" | "cjs" => Some("javascript"),
        "py" | "pyi" => Some("python"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "hpp" | "cc" | "cxx" | "hxx" | "hh" => Some("cpp"),
        _ => None,
    }
}

/// Dispatch to the appropriate language chunker based on file extension.
///
/// Returns `None` for unsupported extensions.
pub fn chunk_for_language(ext: &str, source: &str) -> Option<Vec<CodeChunk>> {
    let config = CodeChunkConfig::default();
    let mut chunks = match ext {
        "rs" => Some(RustCodeChunker::new(config).chunk(source)),
        "ts" | "tsx" => Some(TypeScriptCodeChunker::new(config).chunk(source)),
        "js" | "jsx" | "mjs" | "cjs" => Some(JavaScriptCodeChunker::new(config).chunk(source)),
        "py" | "pyi" => Some(PythonCodeChunker::new(config).chunk(source)),
        "go" => Some(GoCodeChunker::new(config).chunk(source)),
        "java" => Some(JavaCodeChunker::new(config).chunk(source)),
        "c" | "h" => Some(CCodeChunker::new(config).chunk(source)),
        "cpp" | "hpp" | "cc" | "cxx" | "hxx" | "hh" => {
            Some(CppCodeChunker::new(config).chunk(source))
        }
        _ => None,
    }?;

    // Post-process: compute AST structural hashes for clone detection
    fill_ast_hashes(ext, source, &mut chunks);

    Some(chunks)
}

/// Parse the full source once and fill `ast_hash` for each chunk by byte range.
fn fill_ast_hashes(ext: &str, source: &str, chunks: &mut [CodeChunk]) {
    let language: tree_sitter::Language = match ext {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "ts" | "tsx" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "js" | "jsx" | "mjs" | "cjs" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "py" | "pyi" => tree_sitter_python::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        "c" | "h" => tree_sitter_c::LANGUAGE.into(),
        "cpp" | "hpp" | "cc" | "cxx" | "hxx" | "hh" => tree_sitter_cpp::LANGUAGE.into(),
        _ => return,
    };

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
