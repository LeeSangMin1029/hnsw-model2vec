use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(GoCodeChunker);

const GO_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_declaration", CodeNodeKind::Function),
        ("method_declaration", CodeNodeKind::Function),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::extract_go_visibility_from_node,
    extract_params_fn: extract::extract_go_params,
    extract_return_fn: extract::extract_go_return_type,
    extract_doc_fn: extract::extract_go_doc_comment_before,
    method_kinds: &[],
};

impl GoCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        let language: tree_sitter::Language = tree_sitter_go::LANGUAGE.into();
        if parser.set_language(&language).is_err() {
            return Vec::new();
        }

        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };

        let root = tree.root_node();
        let src = source.as_bytes();

        let imports = if self.config.extract_imports {
            extract::extract_imports_by_kind(&root, src, &["import_declaration"])
        } else {
            Vec::new()
        };

        let mut chunks = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            let doc = extract::extract_go_doc_comment_before(&root, &child, src);

            match child.kind() {
                "function_declaration" => {
                    if let Some(mut chunk) = extract::build_chunk(
                        &self.config, &GO_EXTRACTORS, &child, src, &imports, chunks.len(),
                    ) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "method_declaration" => {
                    // Extract receiver type for qualified name (e.g., `Server::Start`)
                    let receiver_type = child.child_by_field_name("receiver")
                        .and_then(|r| {
                            let mut c = r.walk();
                            for param in r.children(&mut c) {
                                if param.kind() == "parameter_declaration" {
                                    return param.child_by_field_name("type")
                                        .and_then(|t| {
                                            let text = t.utf8_text(src).ok()?;
                                            Some(text.trim_start_matches('*').to_owned())
                                        });
                                }
                            }
                            None
                        });

                    if let Some(mut chunk) = extract::build_chunk(
                        &self.config, &GO_EXTRACTORS, &child, src, &imports, chunks.len(),
                    ) {
                        // Override name to include receiver type
                        if let Some(recv) = &receiver_type {
                            let raw_name = extract::extract_name(&child, src);
                            chunk.name = format!("{recv}::{raw_name}");
                        }
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "type_declaration" => {
                    self.extract_go_type_decl(&child, src, &imports, &mut chunks, doc);
                }
                _ => {}
            }
        }

        chunks
    }

    fn extract_go_type_decl(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        chunks: &mut Vec<CodeChunk>,
        doc: Option<String>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "type_spec" {
                continue;
            }

            let name = extract::extract_name(&child, src);

            let type_node = child.child_by_field_name("type");
            let kind = match type_node.map(|n| n.kind()) {
                Some("struct_type") => CodeNodeKind::Struct,
                Some("interface_type") => CodeNodeKind::Trait,
                _ => CodeNodeKind::TypeAlias,
            };

            let full_text = node.utf8_text(src).unwrap_or_default().to_owned();
            let line_count = full_text.lines().count();

            if line_count < self.config.min_lines && kind != CodeNodeKind::TypeAlias {
                continue;
            }

            let visibility = extract::extract_go_visibility(&name);

            chunks.push(CodeChunk {
                text: full_text,
                kind,
                name,
                signature: None,
                doc_comment: doc.clone(),
                visibility,
                start_line: node.start_position().row,
                end_line: node.end_position().row,
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                chunk_index: chunks.len(),
                imports: imports.to_vec(),
                calls: Vec::new(),
                type_refs: extract::extract_type_refs(&child, src),
                param_types: Vec::new(),
                return_type: None, ast_hash: 0, body_hash: 0,
            });
        }
    }
}
