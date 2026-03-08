use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(CCodeChunker);

impl CCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        let language: tree_sitter::Language = tree_sitter_c::LANGUAGE.into();
        if parser.set_language(&language).is_err() {
            return Vec::new();
        }

        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };

        let root = tree.root_node();
        let src = source.as_bytes();

        let imports = if self.config.extract_imports {
            extract::extract_imports_by_kind(&root, src, &["preproc_include"])
        } else {
            Vec::new()
        };

        let mut chunks = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            let doc = extract::extract_block_doc_comment_before(&root, &child, src);

            match child.kind() {
                "function_definition" => {
                    if let Some(mut chunk) = self.c_func_to_chunk(&child, src, &imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "struct_specifier" => {
                    if let Some(chunk) = extract::simple_type_chunk(
                        &child, src, CodeNodeKind::Struct, doc, &imports, chunks.len(), self.config.min_lines,
                    ) {
                        chunks.push(chunk);
                    }
                }
                "enum_specifier" => {
                    if let Some(chunk) = extract::simple_type_chunk(
                        &child, src, CodeNodeKind::Enum, doc, &imports, chunks.len(), self.config.min_lines,
                    ) {
                        chunks.push(chunk);
                    }
                }
                "type_definition" => {
                    if let Some(mut chunk) = self.c_typedef_to_chunk(&child, src, &imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                _ => {}
            }
        }

        chunks
    }

    fn c_func_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let line_count = text.lines().count();
        if line_count < self.config.min_lines {
            return None;
        }

        let name = extract::extract_c_func_name(node, src);
        let visibility = self.extract_c_visibility(node, src);
        let signature = extract::extract_function_signature(node, src);
        let calls = if self.config.extract_calls {
            extract::extract_calls(node, src)
        } else {
            Vec::new()
        };
        let type_refs = extract::extract_type_refs(node, src);
        let param_types = self.extract_c_params(node, src);
        let return_type = node.child_by_field_name("type")
            .and_then(|n| n.utf8_text(src).ok())
            .map(|s| s.to_owned());

        Some(CodeChunk {
            text,
            kind: CodeNodeKind::Function,
            name,
            signature: Some(signature),
            doc_comment: None,
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

    fn extract_c_visibility(&self, node: &tree_sitter::Node, src: &[u8]) -> String {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "storage_class_specifier" {
                if let Ok(text) = child.utf8_text(src) {
                    if text == "static" {
                        return "static".to_owned();
                    }
                }
            }
        }
        String::new()
    }

    fn c_typedef_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let line_count = text.lines().count();
        if line_count < self.config.min_lines {
            return None;
        }

        let mut name = String::new();
        let mut has_struct = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "struct_specifier" {
                has_struct = true;
            }
            if child.kind() == "type_identifier" {
                name = child.utf8_text(src).unwrap_or_default().to_owned();
            }
        }

        if name.is_empty() {
            return None;
        }

        let kind = if has_struct {
            CodeNodeKind::Struct
        } else {
            CodeNodeKind::TypeAlias
        };

        Some(CodeChunk {
            text,
            kind,
            name,
            signature: None,
            doc_comment: None,
            visibility: String::new(),
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

    fn extract_c_params(&self, node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
        let Some(declarator) = node.child_by_field_name("declarator") else {
            return Vec::new();
        };
        let func_decl = if declarator.kind() == "function_declarator" {
            declarator
        } else {
            let mut cursor = declarator.walk();
            match declarator.children(&mut cursor)
                .find(|c| c.kind() == "function_declarator") {
                Some(fd) => fd,
                None => return Vec::new(),
            }
        };

        let Some(params) = func_decl.child_by_field_name("parameters") else {
            return Vec::new();
        };

        let mut result = Vec::new();
        let mut cursor = params.walk();

        for child in params.children(&mut cursor) {
            if child.kind() != "parameter_declaration" {
                continue;
            }

            let ty = child.child_by_field_name("type")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();
            let name = child.child_by_field_name("declarator")
                .and_then(|n| {
                    if n.kind() == "identifier" {
                        n.utf8_text(src).ok()
                    } else {
                        let mut c = n.walk();
                        n.children(&mut c)
                            .find(|ch| ch.kind() == "identifier")
                            .and_then(|ch| ch.utf8_text(src).ok())
                    }
                })
                .unwrap_or_default();

            if !name.is_empty() && !ty.is_empty() {
                result.push((name.to_owned(), ty.to_owned()));
            }
        }

        result
    }
}
