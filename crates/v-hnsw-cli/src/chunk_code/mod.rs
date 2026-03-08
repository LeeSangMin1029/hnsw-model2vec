//! Tree-sitter based code chunker for Rust source files.
//!
//! Extracts function/struct/impl/trait/enum definitions as semantic chunks
//! with metadata (imports, calls, visibility) for vector database indexing.
//!
//! HOW TO EXTEND: Add new languages by implementing a new chunker struct
//! with the same `chunk()` -> `Vec<CodeChunk>` interface, using the
//! appropriate tree-sitter grammar crate.


mod extract;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_multilang;

use std::collections::HashMap;

use v_hnsw_core::PayloadValue;

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

        custom
    }
}

/// Tree-sitter based Rust code chunker.
pub struct RustCodeChunker {
    config: CodeChunkConfig,
}

impl RustCodeChunker {
    pub fn new(config: CodeChunkConfig) -> Self {
        Self { config }
    }

    /// Parse Rust source and extract semantic code chunks.
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
        if parser.set_language(&language).is_err() {
            return Vec::new();
        }

        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };

        let root = tree.root_node();
        let src = source.as_bytes();

        // File-level imports
        let imports = if self.config.extract_imports {
            extract::extract_imports(&root, src)
        } else {
            Vec::new()
        };

        let mut chunks = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            // Extract doc comments preceding this node
            let doc = extract::extract_doc_comment_before(&root, &child, src);

            if let Some(mut chunk) = self.node_to_chunk(&child, src, &imports, chunks.len()) {
                chunk.doc_comment = doc;
                chunks.push(chunk);
            }

            // For impl/trait blocks, also extract individual methods
            if child.kind() == "impl_item" || child.kind() == "trait_item" {
                let parent_name = extract::extract_name(&child, src);
                extract::extract_body_methods(
                    &self.config,
                    &child,
                    src,
                    &imports,
                    &parent_name,
                    &mut chunks,
                );
            }
        }

        chunks
    }

    /// Convert a tree-sitter node to a `CodeChunk`.
    fn node_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let kind = match node.kind() {
            "function_item" => CodeNodeKind::Function,
            "struct_item" => CodeNodeKind::Struct,
            "enum_item" => CodeNodeKind::Enum,
            "impl_item" => CodeNodeKind::Impl,
            "trait_item" => CodeNodeKind::Trait,
            "type_item" => CodeNodeKind::TypeAlias,
            "const_item" => CodeNodeKind::Const,
            "static_item" => CodeNodeKind::Static,
            "mod_item" => CodeNodeKind::Module,
            "macro_definition" => CodeNodeKind::MacroDefinition,
            _ => return None,
        };

        let text = node.utf8_text(src).ok()?.to_owned();
        let line_count = text.lines().count();
        if line_count < self.config.min_lines {
            return None;
        }

        let name = extract::extract_name(node, src);
        let visibility = extract::extract_visibility(node, src);

        let signature = if kind == CodeNodeKind::Function {
            Some(extract::extract_function_signature(node, src))
        } else {
            None
        };

        let calls = if self.config.extract_calls && kind == CodeNodeKind::Function {
            extract::extract_calls(node, src)
        } else {
            Vec::new()
        };

        let type_refs = extract::extract_type_refs(node, src);
        let param_types = extract::extract_param_types(node, src);
        let return_type = extract::extract_return_type(node, src);

        Some(CodeChunk {
            text,
            kind,
            name,
            signature,
            doc_comment: None, // filled by caller
            visibility,
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            chunk_index: index,
            imports: imports.to_vec(),
            calls,
            type_refs,
            param_types,
            return_type,
        })
    }
}

/// Check if a file extension is a supported code file.
pub fn is_supported_code_file(ext: &str) -> bool {
    lang_for_extension(ext).is_some()
}

/// Map a file extension to a language name for tagging.
pub fn lang_for_extension(ext: &str) -> Option<&'static str> {
    match ext {
        "rs" => Some("rust"),
        "ts" | "tsx" => Some("typescript"),
        "py" => Some("python"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "hpp" => Some("cpp"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Multi-language chunker stubs (T002 will provide real implementations)
// ---------------------------------------------------------------------------

macro_rules! define_lang_chunker {
    ($name:ident) => {
        /// Language-specific code chunker (stub — implementation in T002).
        pub struct $name {
            #[allow(dead_code)]
            config: CodeChunkConfig,
        }

        impl $name {
            pub fn new(config: CodeChunkConfig) -> Self {
                Self { config }
            }

            /// Parse source and extract semantic code chunks.
            pub fn chunk(&self, _source: &str) -> Vec<CodeChunk> {
                // Stub: returns empty until T002 implements the parser
                Vec::new()
            }
        }
    };
}

define_lang_chunker!(TypeScriptCodeChunker);
define_lang_chunker!(PythonCodeChunker);
define_lang_chunker!(GoCodeChunker);
define_lang_chunker!(JavaCodeChunker);
define_lang_chunker!(CCodeChunker);
define_lang_chunker!(CppCodeChunker);

/// Dispatch to the appropriate language chunker based on file extension.
///
/// Returns `None` for unsupported extensions.
pub fn chunk_for_language(ext: &str, source: &str) -> Option<Vec<CodeChunk>> {
    let config = CodeChunkConfig::default();
    match ext {
        "rs" => Some(RustCodeChunker::new(config).chunk(source)),
        "ts" | "tsx" => Some(TypeScriptCodeChunker::new(config).chunk(source)),
        "py" => Some(PythonCodeChunker::new(config).chunk(source)),
        "go" => Some(GoCodeChunker::new(config).chunk(source)),
        "java" => Some(JavaCodeChunker::new(config).chunk(source)),
        "c" | "h" => Some(CCodeChunker::new(config).chunk(source)),
        "cpp" | "hpp" => Some(CppCodeChunker::new(config).chunk(source)),
        _ => None,
    }
}
