use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(CppCodeChunker);

const CPP_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[("function_definition", CodeNodeKind::Function)],
    extract_name_fn: extract::extract_c_func_name,
    extract_vis_fn: extract::no_visibility,
    extract_params_fn: extract::extract_c_params,
    extract_return_fn: extract::extract_c_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["function_definition"],
};

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
                    if let Some(mut chunk) = extract::build_chunk(
                        &self.config, &CPP_EXTRACTORS, &child, src, &imports, chunks.len(),
                    ) {
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
                    extract::extract_methods(
                        &self.config, &CPP_EXTRACTORS, &child, src, &imports, &name, &mut chunks,
                    );
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
}
