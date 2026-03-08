use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(TypeScriptCodeChunker);

impl TypeScriptCodeChunker {
    /// Chunk JavaScript source using the TSX parser (superset of JS + JSX).
    pub fn chunk_js(&self, source: &str) -> Vec<CodeChunk> {
        self.chunk_with_language(source, tree_sitter_typescript::LANGUAGE_TSX.into())
    }

    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        self.chunk_with_language(source, tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
    }

    fn chunk_with_language(&self, source: &str, language: tree_sitter::Language) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        if parser.set_language(&language).is_err() {
            return Vec::new();
        }

        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };

        let root = tree.root_node();
        let src = source.as_bytes();

        let imports = if self.config.extract_imports {
            extract::extract_imports_by_kind(&root, src, &["import_statement"])
        } else {
            Vec::new()
        };

        let mut chunks = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            // Handle export_statement wrapper
            let actual_node = if child.kind() == "export_statement" {
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
                if let Some(n) = found { n } else { continue; }
            } else {
                child
            };

            let is_exported = child.kind() == "export_statement";
            let doc = extract::extract_block_doc_comment_before(&root, &child, src);

            if let Some(mut chunk) = self.ts_node_to_chunk(&actual_node, src, &imports, chunks.len(), is_exported) {
                chunk.doc_comment = doc;
                chunks.push(chunk);
            }

            // Extract methods from classes
            if actual_node.kind() == "class_declaration" {
                let parent_name = extract::extract_name(&actual_node, src);
                self.extract_ts_class_methods(&actual_node, src, &imports, &parent_name, &mut chunks);
            }
        }

        chunks
    }

    fn ts_node_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
        is_exported: bool,
    ) -> Option<CodeChunk> {
        let kind = match node.kind() {
            "function_declaration" => CodeNodeKind::Function,
            "class_declaration" => CodeNodeKind::Class,
            "interface_declaration" => CodeNodeKind::Interface,
            "enum_declaration" => CodeNodeKind::Enum,
            "type_alias_declaration" => CodeNodeKind::TypeAlias,
            _ => return None,
        };

        let text = node.utf8_text(src).ok()?.to_owned();
        let line_count = text.lines().count();
        if line_count < self.config.min_lines {
            return None;
        }

        let name = extract::extract_name(node, src);
        let visibility = if is_exported { "export".to_owned() } else { String::new() };

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
        let param_types = self.extract_ts_params(node, src);
        let return_type = extract::extract_ts_return_type(node, src);

        Some(CodeChunk {
            text,
            kind,
            name,
            signature,
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

    fn extract_ts_params(&self, node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
        let Some(params) = node.child_by_field_name("parameters") else {
            return Vec::new();
        };

        let mut result = Vec::new();
        let mut cursor = params.walk();

        for child in params.children(&mut cursor) {
            if child.kind() != "required_parameter" && child.kind() != "optional_parameter" {
                continue;
            }

            let name = child
                .child_by_field_name("pattern")
                .and_then(|n| n.utf8_text(src).ok())
                .unwrap_or_default();
            let ty = child
                .child_by_field_name("type")
                .and_then(|n| {
                    let mut c = n.walk();
                    for inner in n.children(&mut c) {
                        if inner.kind() != ":" {
                            return inner.utf8_text(src).ok();
                        }
                    }
                    n.utf8_text(src).ok()
                })
                .unwrap_or_default();

            if !name.is_empty() && !ty.is_empty() {
                result.push((name.to_owned(), ty.to_owned()));
            }
        }

        result
    }

    fn extract_ts_class_methods(
        &self,
        class_node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        parent_name: &str,
        chunks: &mut Vec<CodeChunk>,
    ) {
        let Some(body) = class_node.child_by_field_name("body") else {
            return;
        };

        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() != "method_definition" {
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
            let full_name = format!("{parent_name}::{method_name}");

            let visibility = String::new();
            let signature = extract::extract_function_signature(&child, src);
            let calls = if self.config.extract_calls {
                extract::extract_calls(&child, src)
            } else {
                Vec::new()
            };
            let doc = extract::extract_block_doc_comment_before(&body, &child, src);
            let type_refs = extract::extract_type_refs(&child, src);
            let param_types = self.extract_ts_params(&child, src);
            let return_type = extract::extract_ts_return_type(&child, src);

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
}
