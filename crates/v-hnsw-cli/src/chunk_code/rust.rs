use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(RustCodeChunker);

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
            return_type, ast_hash: 0, body_hash: 0,
        })
    }
}
