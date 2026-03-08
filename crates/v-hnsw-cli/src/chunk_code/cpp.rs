use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(CppCodeChunker);

impl CppCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        let language: tree_sitter::Language = tree_sitter_cpp::LANGUAGE.into();
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
                    if let Some(mut chunk) = self.cpp_func_to_chunk(&child, src, &imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "class_specifier" => {
                    if let Some(mut chunk) = self.cpp_class_to_chunk(&child, src, &imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                    let name = extract::extract_name(&child, src);
                    self.extract_cpp_class_methods(&child, src, &imports, &name, &mut chunks);
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
                _ => {}
            }
        }

        chunks
    }

    fn cpp_func_to_chunk(
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
        let signature = extract::extract_function_signature(node, src);
        let calls = if self.config.extract_calls {
            extract::extract_calls(node, src)
        } else {
            Vec::new()
        };
        let type_refs = extract::extract_type_refs(node, src);

        Some(CodeChunk {
            text,
            kind: CodeNodeKind::Function,
            name,
            signature: Some(signature),
            doc_comment: None,
            visibility: String::new(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            chunk_index: index,
            imports: imports.to_vec(),
            calls,
            type_refs,
            param_types: Vec::new(),
            return_type: None, ast_hash: 0, body_hash: 0,
        })
    }

    fn cpp_class_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let name = extract::extract_name(node, src);
        if name.is_empty() {
            return None;
        }

        Some(CodeChunk {
            text,
            kind: CodeNodeKind::Class,
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

    fn extract_cpp_class_methods(
        &self,
        class_node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        class_name: &str,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let Some(body) = class_node.child_by_field_name("body") else {
            return;
        };

        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "function_definition" {
                let text = match child.utf8_text(src) {
                    Ok(t) => t.to_owned(),
                    Err(_) => continue,
                };

                let line_count = text.lines().count();
                if line_count < self.config.min_lines {
                    continue;
                }

                let method_name = extract::extract_c_func_name(&child, src);
                let full_name = format!("{class_name}::{method_name}");

                let signature = extract::extract_function_signature(&child, src);
                let calls = if self.config.extract_calls {
                    extract::extract_calls(&child, src)
                } else {
                    Vec::new()
                };
                let doc = extract::extract_block_doc_comment_before(&body, &child, src);
                let type_refs = extract::extract_type_refs(&child, src);

                chunks.push(CodeChunk {
                    text,
                    kind: CodeNodeKind::Function,
                    name: full_name,
                    signature: Some(signature),
                    doc_comment: doc,
                    visibility: String::new(),
                    start_line: child.start_position().row,
                    end_line: child.end_position().row,
                    start_byte: child.start_byte(),
                    end_byte: child.end_byte(),
                    chunk_index: chunks.len(),
                    imports: imports.to_vec(),
                    calls,
                    type_refs,
                    param_types: Vec::new(),
                    return_type: None, ast_hash: 0, body_hash: 0,
                });
            }
        }
    }
}
