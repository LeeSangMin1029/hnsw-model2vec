use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(PythonCodeChunker);

impl PythonCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        let language: tree_sitter::Language = tree_sitter_python::LANGUAGE.into();
        if parser.set_language(&language).is_err() {
            return Vec::new();
        }

        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };

        let root = tree.root_node();
        let src = source.as_bytes();

        let imports = if self.config.extract_imports {
            extract::extract_imports_by_kind(&root, src, &["import_statement", "import_from_statement"])
        } else {
            Vec::new()
        };

        let mut chunks = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            // Handle decorated_definition wrapper
            let actual_node = if child.kind() == "decorated_definition" {
                let mut inner_cursor = child.walk();
                let mut found = None;
                for inner in child.children(&mut inner_cursor) {
                    match inner.kind() {
                        "function_definition" | "class_definition" => {
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

            if let Some(chunk) = self.py_node_to_chunk(&actual_node, src, &imports, chunks.len()) {
                chunks.push(chunk);
            }

            // Extract methods from classes
            if actual_node.kind() == "class_definition" {
                let parent_name = extract::extract_name(&actual_node, src);
                self.extract_py_class_methods(&actual_node, src, &imports, &parent_name, &mut chunks);
            }
        }

        chunks
    }

    fn py_node_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let kind = match node.kind() {
            "function_definition" => CodeNodeKind::Function,
            "class_definition" => CodeNodeKind::Class,
            _ => return None,
        };

        let text = node.utf8_text(src).ok()?.to_owned();
        let line_count = text.lines().count();
        if line_count < self.config.min_lines {
            return None;
        }

        let name = extract::extract_name(node, src);
        let visibility = String::new();

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

        let doc_comment = extract::extract_python_docstring(node, src);

        let type_refs = extract::extract_type_refs(node, src);
        let param_types = self.extract_py_params(node, src);
        let return_type = extract::extract_python_return_type(node, src);

        Some(CodeChunk {
            text,
            kind,
            name,
            signature,
            doc_comment,
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

    fn extract_py_params(&self, node: &tree_sitter::Node, src: &[u8]) -> Vec<(String, String)> {
        let Some(params) = node.child_by_field_name("parameters") else {
            return Vec::new();
        };

        let mut result = Vec::new();
        let mut cursor = params.walk();

        for child in params.children(&mut cursor) {
            if child.kind() == "identifier" {
                continue; // Skip bare identifiers like self/cls
            }

            if child.kind() == "typed_parameter" || child.kind() == "typed_default_parameter" {
                let name = child
                    .child_by_field_name("name")
                    .or_else(|| {
                        let mut c = child.walk();
                        child.children(&mut c).find(|n| n.kind() == "identifier")
                    })
                    .and_then(|n| n.utf8_text(src).ok())
                    .unwrap_or_default();
                let ty = child
                    .child_by_field_name("type")
                    .and_then(|n| n.utf8_text(src).ok())
                    .unwrap_or_default();

                if !name.is_empty() && !ty.is_empty() && name != "self" && name != "cls" {
                    result.push((name.to_owned(), ty.to_owned()));
                }
            }
        }

        result
    }

    fn extract_py_class_methods(
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
            let method_node = if child.kind() == "decorated_definition" {
                let mut inner_cursor = child.walk();
                child.children(&mut inner_cursor).find(|n| n.kind() == "function_definition")
            } else if child.kind() == "function_definition" {
                Some(child)
            } else {
                None
            };

            let Some(method_node) = method_node else {
                continue;
            };

            let text = match method_node.utf8_text(src) {
                Ok(t) => t.to_owned(),
                Err(_) => continue,
            };

            let line_count = text.lines().count();
            if line_count < self.config.min_lines {
                continue;
            }

            let method_name = extract::extract_name(&method_node, src);
            let full_name = format!("{parent_name}::{method_name}");

            let signature = extract::extract_function_signature(&method_node, src);
            let calls = if self.config.extract_calls {
                extract::extract_calls(&method_node, src)
            } else {
                Vec::new()
            };
            let doc = extract::extract_python_docstring(&method_node, src);
            let type_refs = extract::extract_type_refs(&method_node, src);
            let param_types = self.extract_py_params(&method_node, src);
            let return_type = extract::extract_python_return_type(&method_node, src);

            chunks.push(CodeChunk {
                text,
                kind: CodeNodeKind::Function,
                name: full_name,
                signature: Some(signature),
                doc_comment: doc,
                visibility: String::new(),
                start_line: method_node.start_position().row,
                end_line: method_node.end_position().row,
                start_byte: method_node.start_byte(),
                end_byte: method_node.end_byte(),
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
