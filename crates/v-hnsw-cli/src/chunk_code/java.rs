use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(JavaCodeChunker);

impl JavaCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        let language: tree_sitter::Language = tree_sitter_java::LANGUAGE.into();
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
        self.walk_java_declarations(&root, src, &imports, &mut chunks);

        chunks
    }

    fn walk_java_declarations(
        &self,
        parent: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        chunks: &mut Vec<CodeChunk>,
    ) {
        let mut cursor = parent.walk();
        for child in parent.children(&mut cursor) {
            let doc = extract::extract_block_doc_comment_before(parent, &child, src);

            match child.kind() {
                "class_declaration" => {
                    if let Some(mut chunk) = self.java_class_to_chunk(&child, src, imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                    let name = extract::extract_name(&child, src);
                    if let Some(body) = child.child_by_field_name("body") {
                        self.extract_java_methods(&body, src, imports, &name, chunks);
                    }
                }
                "interface_declaration" => {
                    if let Some(mut chunk) = self.java_type_decl_to_chunk(&child, src, imports, chunks.len(), CodeNodeKind::Interface) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "enum_declaration" => {
                    if let Some(mut chunk) = self.java_type_decl_to_chunk(&child, src, imports, chunks.len(), CodeNodeKind::Enum) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                _ => {}
            }
        }
    }

    fn java_class_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let name = extract::extract_name(node, src);
        let visibility = extract::extract_java_visibility(node, src);
        let type_refs = extract::extract_type_refs(node, src);

        Some(CodeChunk {
            text,
            kind: CodeNodeKind::Class,
            name,
            signature: None,
            doc_comment: None,
            visibility,
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            chunk_index: index,
            imports: imports.to_vec(),
            calls: Vec::new(),
            type_refs,
            param_types: Vec::new(),
            return_type: None, ast_hash: 0, body_hash: 0,
        })
    }

    fn java_type_decl_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
        kind: CodeNodeKind,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let name = extract::extract_name(node, src);
        let visibility = extract::extract_java_visibility(node, src);

        Some(CodeChunk {
            text,
            kind,
            name,
            signature: None,
            doc_comment: None,
            visibility,
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            chunk_index: index,
            imports: imports.to_vec(),
            calls: Vec::new(),
            type_refs: extract::extract_type_refs(node, src),
            param_types: Vec::new(),
            return_type: None, ast_hash: 0, body_hash: 0,
        })
    }

    fn extract_java_methods(
        &self,
        body: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        class_name: &str,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() != "method_declaration" && child.kind() != "constructor_declaration" {
                continue;
            }

            let text = match child.utf8_text(src) {
                Ok(t) => t.to_owned(),
                Err(_) => continue,
            };

            let line_count = text.lines().count();
            if line_count < self.config.min_lines {
                continue;
            }

            let method_name = extract::extract_name(&child, src);
            let full_name = format!("{class_name}::{method_name}");
            let visibility = extract::extract_java_visibility(&child, src);
            let signature = extract::extract_function_signature(&child, src);
            let calls = if self.config.extract_calls {
                extract::extract_calls(&child, src)
            } else {
                Vec::new()
            };
            let doc = extract::extract_block_doc_comment_before(body, &child, src);
            let type_refs = extract::extract_type_refs(&child, src);
            let param_types = self.extract_java_params(&child, src);
            let return_type = child.child_by_field_name("type")
                .and_then(|n| n.utf8_text(src).ok())
                .map(|s| s.to_owned());

            chunks.push(CodeChunk {
                text,
                kind: CodeNodeKind::Function,
                name: full_name,
                signature: Some(signature),
                doc_comment: doc,
                visibility,
                start_line: child.start_position().row,
                end_line: child.end_position().row,
                start_byte: child.start_byte(),
                end_byte: child.end_byte(),
                chunk_index: chunks.len(),
                imports: imports.to_vec(),
                calls,
                type_refs,
                param_types,
                return_type, ast_hash: 0, body_hash: 0,
            });
        }
    }

    fn extract_java_params(&self, node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
        let Some(params) = node.child_by_field_name("parameters") else {
            return Vec::new();
        };

        let mut result = Vec::new();
        let mut cursor = params.walk();

        for child in params.children(&mut cursor) {
            if child.kind() != "formal_parameter" && child.kind() != "spread_parameter" {
                continue;
            }

            let ty = child
                .child_by_field_name("type")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();
            let name = child
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();

            if !name.is_empty() && !ty.is_empty() {
                result.push((name.to_owned(), ty.to_owned()));
            }
        }

        result
    }
}
