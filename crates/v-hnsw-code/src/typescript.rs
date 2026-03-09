use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(TypeScriptCodeChunker);

const TS_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_declaration", CodeNodeKind::Function),
        ("class_declaration", CodeNodeKind::Class),
        ("interface_declaration", CodeNodeKind::Interface),
        ("enum_declaration", CodeNodeKind::Enum),
        ("type_alias_declaration", CodeNodeKind::TypeAlias),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::no_visibility,
    extract_params_fn: extract::extract_ts_params,
    extract_return_fn: extract::extract_ts_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["method_definition"],
};

impl TypeScriptCodeChunker {
    /// Chunk JavaScript source using the TSX parser (superset of JS + JSX).
    pub fn chunk_js(&self, source: &str) -> Vec<CodeChunk> {
        self.chunk_with_language(source, tree_sitter_typescript::LANGUAGE_TSX.into())
    }

    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        self.chunk_with_language(source, tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
    }

    fn chunk_with_language(&self, source: &str, language: tree_sitter::Language) -> Vec<CodeChunk> {
        let Some(parsed) = extract::parse_source(
            language,
            source,
            self.config.extract_imports,
            &["import_statement"],
        ) else {
            return Vec::new();
        };
        let root = parsed.tree.root_node();
        let src = source.as_bytes();
        let imports = parsed.imports;

        let mut chunks = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            // Handle export_statement wrapper
            let (actual_node, is_exported) = if child.kind() == "export_statement" {
                let mut inner_cursor = child.walk();
                let mut found = None;
                for inner in child.children(&mut inner_cursor) {
                    match inner.kind() {
                        "function_declaration" | "class_declaration"
                        | "interface_declaration" | "enum_declaration"
                        | "type_alias_declaration" => {
                            found = Some(inner);
                            break;
                        }
                        _ => {}
                    }
                }
                if let Some(n) = found { (n, true) } else { continue; }
            } else {
                (child, false)
            };

            let doc = extract::extract_block_doc_comment_before(&root, &child, src);

            if let Some(mut chunk) = extract::build_chunk(
                &self.config, &TS_EXTRACTORS, &actual_node, src, &imports, chunks.len(),
            ) {
                if is_exported {
                    chunk.visibility = "export".to_owned();
                }
                chunk.doc_comment = doc;
                chunks.push(chunk);
            }

            // Extract methods from classes
            if actual_node.kind() == "class_declaration" {
                let parent_name = extract::extract_name(&actual_node, src);
                extract::extract_methods(
                    &self.config, &TS_EXTRACTORS, &actual_node, src, &imports, &parent_name, &mut chunks,
                );
            }
        }

        chunks
    }
}
