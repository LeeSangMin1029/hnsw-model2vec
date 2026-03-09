use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(RustCodeChunker);

const RUST_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_item", CodeNodeKind::Function),
        ("struct_item", CodeNodeKind::Struct),
        ("enum_item", CodeNodeKind::Enum),
        ("impl_item", CodeNodeKind::Impl),
        ("trait_item", CodeNodeKind::Trait),
        ("type_item", CodeNodeKind::TypeAlias),
        ("const_item", CodeNodeKind::Const),
        ("static_item", CodeNodeKind::Static),
        ("mod_item", CodeNodeKind::Module),
        ("macro_definition", CodeNodeKind::MacroDefinition),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::extract_visibility,
    extract_params_fn: extract::extract_param_types,
    extract_return_fn: extract::extract_return_type,
    extract_doc_fn: extract::extract_doc_comment_before,
    method_kinds: &["function_item"],
};

impl RustCodeChunker {
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

            if let Some(mut chunk) = extract::build_chunk(
                &self.config, &RUST_EXTRACTORS, &child, src, &imports, chunks.len(),
            ) {
                chunk.doc_comment = doc;
                chunks.push(chunk);
            }

            // For impl/trait blocks, also extract individual methods
            if child.kind() == "impl_item" || child.kind() == "trait_item" {
                let parent_name = extract::extract_name(&child, src);
                extract::extract_methods(
                    &self.config, &RUST_EXTRACTORS, &child, src, &imports, &parent_name, &mut chunks,
                );
            }
        }

        chunks
    }
}
