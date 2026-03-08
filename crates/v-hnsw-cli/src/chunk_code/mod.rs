//! Tree-sitter based code chunker for Rust source files.
//!
//! Extracts function/struct/impl/trait/enum definitions as semantic chunks
//! with metadata (imports, calls, visibility) for vector database indexing.
//!
//! HOW TO EXTEND: Add new languages by implementing a new chunker struct
//! with the same `chunk()` -> `Vec<CodeChunk>` interface, using the
//! appropriate tree-sitter grammar crate.

pub(crate) mod extract;
mod rust;
mod typescript;
mod python;
mod go_lang;
mod java;
mod c_lang;
mod cpp;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use v_hnsw_core::PayloadValue;

/// Define a code chunker struct with a standard `new(config)` constructor.
macro_rules! define_chunker {
    ($name:ident) => {
        pub struct $name {
            config: super::CodeChunkConfig,
        }

        impl $name {
            pub fn new(config: super::CodeChunkConfig) -> Self {
                Self { config }
            }
        }
    };
}
pub(crate) use define_chunker;

/// Configuration for code chunking.
#[derive(Debug, Clone)]
pub struct CodeChunkConfig {
    /// Minimum lines for a chunk to be included.
    pub min_lines: usize,
    /// Extract file-level `use` statements.
    pub extract_imports: bool,
    /// Extract function calls from bodies.
    pub extract_calls: bool,
}

impl Default for CodeChunkConfig {
    fn default() -> Self {
        Self {
            min_lines: 2,
            extract_imports: true,
            extract_calls: true,
        }
    }
}

/// Kind of code node extracted by tree-sitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeNodeKind {
    Function,
    Struct,
    Enum,
    Impl,
    Trait,
    TypeAlias,
    Const,
    Static,
    Module,
    MacroDefinition,
    Class,
    Interface,
}

impl CodeNodeKind {
    /// String label for payload storage.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Impl => "impl",
            Self::Trait => "trait",
            Self::TypeAlias => "type_alias",
            Self::Const => "const",
            Self::Static => "static",
            Self::Module => "module",
            Self::MacroDefinition => "macro",
            Self::Class => "class",
            Self::Interface => "interface",
        }
    }
}

/// A semantic code chunk extracted via tree-sitter.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// The raw source code text.
    pub text: String,
    /// Kind of code node.
    pub kind: CodeNodeKind,
    /// Symbol name (e.g., `process_payment`, `PaymentIntent`).
    pub name: String,
    /// Function signature (params + return), if applicable.
    pub signature: Option<String>,
    /// Doc comment text, if any.
    pub doc_comment: Option<String>,
    /// Visibility: `"pub"`, `"pub(crate)"`, `""`.
    pub visibility: String,
    /// Start line (0-based).
    pub start_line: usize,
    /// End line (0-based).
    pub end_line: usize,
    /// Byte offsets in source.
    pub start_byte: usize,
    pub end_byte: usize,
    /// Sequential chunk index within the file.
    pub chunk_index: usize,
    /// File-level import statements.
    pub imports: Vec<String>,
    /// Function calls within the body.
    pub calls: Vec<String>,
    /// Type names referenced in signature and body.
    pub type_refs: Vec<String>,
    /// Parameter name-type pairs (e.g., `[("amount", "f64")]`).
    pub param_types: Vec<(String, String)>,
    /// Return type string (e.g., `"Result<Vec<Item>>"`).
    pub return_type: Option<String>,
    /// Structural AST hash for clone detection (0 = not computed).
    pub ast_hash: u64,
    /// Normalized body text hash for exact-logic clone detection (0 = not computed).
    pub body_hash: u64,
}

impl CodeChunk {
    /// Build text optimized for embedding (semantic search).
    ///
    /// Includes doc comment, signature, calls, called_by — not the full body.
    /// Use `called_by` to inject reverse-reference data from cross-file analysis.
    pub fn to_embed_text(&self, file_path: &str, called_by: &[String]) -> String {
        let mut parts = Vec::new();

        // [kind] visibility name
        let vis = if self.visibility.is_empty() {
            String::new()
        } else {
            format!("{} ", self.visibility)
        };
        parts.push(format!("[{}] {vis}{}", self.kind.as_str(), self.name));

        // File location
        parts.push(format!(
            "File: {file_path}:{}-{}",
            self.start_line + 1,
            self.end_line + 1
        ));

        // Doc comment
        if let Some(ref doc) = self.doc_comment {
            parts.push(doc.clone());
        }

        // Signature
        if let Some(ref sig) = self.signature {
            parts.push(format!("Signature: {sig}"));
        }

        // Parameter types
        if !self.param_types.is_empty() {
            let params: Vec<String> = self
                .param_types
                .iter()
                .map(|(n, t)| format!("{n}: {t}"))
                .collect();
            parts.push(format!("Params: {}", params.join(", ")));
        }

        // Type references
        if !self.type_refs.is_empty() {
            parts.push(format!("Types: {}", self.type_refs.join(", ")));
        }

        // Calls
        if !self.calls.is_empty() {
            parts.push(format!("Calls: {}", self.calls.join(", ")));
        }

        // Called by (reverse references)
        if !called_by.is_empty() {
            parts.push(format!("Called by: {}", called_by.join(", ")));
        }

        parts.join("\n")
    }

    /// Convert code metadata to `payload.custom` fields.
    ///
    /// Pass `called_by` from cross-file reverse-reference analysis.
    pub fn to_custom_fields(&self, called_by: &[String]) -> HashMap<String, PayloadValue> {
        let mut custom = HashMap::new();

        custom.insert(
            "kind".to_owned(),
            PayloadValue::String(self.kind.as_str().to_owned()),
        );
        custom.insert(
            "name".to_owned(),
            PayloadValue::String(self.name.clone()),
        );
        custom.insert(
            "visibility".to_owned(),
            PayloadValue::String(self.visibility.clone()),
        );
        custom.insert(
            "start_line".to_owned(),
            PayloadValue::Integer(i64::try_from(self.start_line).unwrap_or(0)),
        );
        custom.insert(
            "end_line".to_owned(),
            PayloadValue::Integer(i64::try_from(self.end_line).unwrap_or(0)),
        );

        if let Some(ref sig) = self.signature {
            custom.insert("signature".to_owned(), PayloadValue::String(sig.clone()));
        }
        if let Some(ref doc) = self.doc_comment {
            custom.insert("doc".to_owned(), PayloadValue::String(doc.clone()));
        }
        if !self.calls.is_empty() {
            custom.insert(
                "calls".to_owned(),
                PayloadValue::StringList(self.calls.clone()),
            );
        }
        if !self.imports.is_empty() {
            custom.insert(
                "imports".to_owned(),
                PayloadValue::StringList(self.imports.clone()),
            );
        }
        if !called_by.is_empty() {
            custom.insert(
                "called_by".to_owned(),
                PayloadValue::StringList(called_by.to_vec()),
            );
        }
        if !self.type_refs.is_empty() {
            custom.insert(
                "type_refs".to_owned(),
                PayloadValue::StringList(self.type_refs.clone()),
            );
        }
        if let Some(ref ret) = self.return_type {
            custom.insert("return_type".to_owned(), PayloadValue::String(ret.clone()));
        }
        if self.ast_hash != 0 {
            #[expect(clippy::cast_possible_wrap, reason = "hash bits reinterpreted")]
            custom.insert(
                "ast_hash".to_owned(),
                PayloadValue::Integer(self.ast_hash as i64),
            );
        }
        if self.body_hash != 0 {
            #[expect(clippy::cast_possible_wrap, reason = "hash bits reinterpreted")]
            custom.insert(
                "body_hash".to_owned(),
                PayloadValue::Integer(self.body_hash as i64),
            );
        }

        // MinHash token fingerprint for near-duplicate detection
        let tokens = extract::code_tokens(&self.text);
        if tokens.len() >= 10 {
            let sig = extract::minhash_signature(&tokens, extract::MINHASH_K);
            custom.insert(
                "minhash".to_owned(),
                PayloadValue::String(extract::minhash_to_hex(&sig)),
            );
        }

        custom
    }
}

// Re-export chunkers
pub use rust::RustCodeChunker;
pub use typescript::TypeScriptCodeChunker;
pub use python::PythonCodeChunker;
pub use go_lang::GoCodeChunker;
pub use java::JavaCodeChunker;
pub use c_lang::CCodeChunker;
pub use cpp::CppCodeChunker;

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

// ---------------------------------------------------------------------------
// Multi-language dispatch
// ---------------------------------------------------------------------------

/// Dispatch to the appropriate language chunker based on file extension.
///
/// Returns `None` for unsupported extensions.
pub fn chunk_for_language(ext: &str, source: &str) -> Option<Vec<CodeChunk>> {
    let config = CodeChunkConfig::default();
    let mut chunks = match ext {
        "rs" => Some(RustCodeChunker::new(config).chunk(source)),
        "ts" | "tsx" => Some(TypeScriptCodeChunker::new(config).chunk(source)),
        "js" | "jsx" | "mjs" | "cjs" => Some(TypeScriptCodeChunker::new(config).chunk_js(source)),
        "py" | "pyi" => Some(PythonCodeChunker::new(config).chunk(source)),
        "go" => Some(GoCodeChunker::new(config).chunk(source)),
        "java" => Some(JavaCodeChunker::new(config).chunk(source)),
        "c" | "h" => Some(CCodeChunker::new(config).chunk(source)),
        "cpp" | "hpp" | "cc" | "cxx" | "hxx" | "hh" => Some(CppCodeChunker::new(config).chunk(source)),
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
        }
        chunk.body_hash = extract::body_hash(&chunk.text);
    }
}

