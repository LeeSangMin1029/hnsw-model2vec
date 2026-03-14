//! Tree-sitter node extraction helpers.
//!
//! Split into submodules by role:
//! - [`hash`]: AST hashing, body hashing, sub-block splitting, MinHash
//! - [`common`]: Language-agnostic extractors (name, visibility, calls, etc.)
//! - [`lang`]: Language-specific extractors (C, Go, Java, Python, TypeScript)
//! - [`chunk`]: Chunk builders (`LangExtractors`, `build_chunk`, `chunk_standard`)

pub mod hash;
pub mod common;
pub mod lang;
pub mod chunk;

/// Result of parsing source code with tree-sitter.
pub struct ParsedSource {
    /// The parsed syntax tree.
    pub tree: tree_sitter::Tree,
    /// File-level import statements (empty when import extraction is disabled).
    pub imports: Vec<String>,
}

// Re-export everything for backwards compatibility (callers use `extract::*`).
pub use hash::{
    ast_structure_hash, body_hash, code_tokens, jaccard_from_minhash, minhash_from_hex,
    minhash_signature, minhash_to_hex, parse_source, split_into_blocks, MINHASH_K,
};
pub use common::{
    collect_sorted_unique, extract_doc_comment_before, extract_function_signature, extract_imports,
    extract_name, extract_param_types, extract_return_type, extract_struct_fields, extract_visibility,
    walk_for_calls, walk_for_type_ids,
};
pub use lang::{
    extract_block_doc_comment_before, extract_c_func_name, extract_c_params,
    extract_c_return_type, extract_c_visibility, extract_go_doc_comment_before,
    extract_go_params, extract_go_return_type, extract_go_visibility,
    extract_go_visibility_from_node, extract_imports_by_kind, extract_java_params,
    extract_java_visibility, extract_py_params, extract_python_doc_wrapper,
    extract_python_docstring, extract_python_return_type, extract_ts_params,
    extract_ts_return_type, no_visibility,
};
pub use chunk::{
    build_chunk, chunk_standard, extract_methods, simple_type_chunk, LangExtractors,
};
