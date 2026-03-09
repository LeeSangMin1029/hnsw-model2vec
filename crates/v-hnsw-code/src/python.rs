use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(PythonCodeChunker);

const PY_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_definition", CodeNodeKind::Function),
        ("class_definition", CodeNodeKind::Class),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::no_visibility,
    extract_params_fn: extract::extract_py_params,
    extract_return_fn: extract::extract_python_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["function_definition"],
};

impl PythonCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let Some(parsed) = extract::parse_source(
            tree_sitter_python::LANGUAGE.into(),
            source,
            self.config.extract_imports,
            &["import_statement", "import_from_statement"],
        ) else {
            return Vec::new();
        };
        let root = parsed.tree.root_node();
        let src = source.as_bytes();
        let imports = parsed.imports;

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

            if let Some(mut chunk) = extract::build_chunk(
                &self.config, &PY_EXTRACTORS, &actual_node, src, &imports, chunks.len(),
            ) {
                // Python uses docstrings instead of preceding comments
                chunk.doc_comment = extract::extract_python_docstring(&actual_node, src);
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

    /// Python method extraction with decorated_definition handling.
    ///
    /// Cannot use generic `extract_methods` because Python has `decorated_definition`
    /// wrappers around methods and uses docstrings instead of preceding comments.
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

            if text.lines().count() < self.config.min_lines {
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
            let param_types = extract::extract_py_params(&method_node, src);
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
