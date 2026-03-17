//! Tree-sitter based multi-language code chunker library.
//!
//! Extracts function/struct/impl/trait/enum definitions as semantic chunks
//! with metadata (imports, calls, visibility) for vector database indexing.
//!
//! HOW TO EXTEND: Add new languages by implementing a new chunker struct
//! with the same `chunk()` -> `Vec<CodeChunk>` interface, using the
//! appropriate tree-sitter grammar crate.

pub mod extract;
pub mod extern_impl;
mod types;
mod dispatch;
mod rust;
mod typescript;
mod javascript;
mod python;
mod go_lang;
mod java;
mod c_lang;
mod cpp;

#[cfg(test)]
mod tests;

/// Define a code chunker struct with a standard `new(config)` constructor.
macro_rules! define_chunker {
    // Base form: struct + new() only (for Go, C with custom chunk logic)
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
    // Extended form: struct + new() + standard chunk() method
    ($name:ident, $language:expr, $import_kinds:expr, $extractors:expr) => {
        super::define_chunker!($name);

        impl $name {
            pub fn chunk(&self, source: &str) -> Vec<super::CodeChunk> {
                let Some(parsed) = super::extract::parse_source(
                    $language.into(),
                    source,
                    self.config.extract_imports,
                    $import_kinds,
                ) else {
                    return Vec::new();
                };
                super::extract::chunk_standard(&self.config, $extractors, &parsed, source)
            }
        }
    };
}
pub(crate) use define_chunker;

// Re-export types
pub use types::{CodeChunk, CodeChunkConfig, CodeNodeKind, SubBlock};

// Re-export chunkers
pub use rust::RustCodeChunker;
pub use typescript::TypeScriptCodeChunker;
pub use javascript::JavaScriptCodeChunker;
pub use python::PythonCodeChunker;
pub use go_lang::GoCodeChunker;
pub use java::JavaCodeChunker;
pub use c_lang::CCodeChunker;
pub use cpp::CppCodeChunker;

// Re-export dispatch
pub use dispatch::{chunk_for_language, is_supported_code_file, lang_for_extension};
